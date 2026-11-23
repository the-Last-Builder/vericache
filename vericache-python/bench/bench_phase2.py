#!/usr/bin/env python3
"""Compare the Phase 1 transfer path with the Phase 2 connector on the same KV tensors."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vericache import (
    NativeVeriCacheEngine,
    VeriCacheConfig,
    compute_prefix_block_hashes,
    derive_layer_block_key,
)
from vericache._native import _splitmix64
from vericache.vllm_v1_connector import (
    BlockTransfer,
    VeriCacheConnectorMetadata,
    VeriCacheV1Connector,
)

_ROOT = Path(__file__).resolve().parents[1]


def _shard(key, count):
    # Match native shard routing so a DRAM-resident run fits on every shard.
    a, b, c, d = key
    return _splitmix64(a ^ _splitmix64(b ^ _splitmix64(c ^ _splitmix64(d)))) % count


def _latency(run):
    start = perf_counter()
    run()
    return perf_counter() - start


def _phase1(options, sources, outputs, blocks, keys):
    with (
        NativeVeriCacheEngine(VeriCacheConfig(**options)) as engine,
        engine.make_staging() as stage,
    ):
        # Scalar admissions wait for one spill at a time with eager writeback off.
        engine.set_writeback_policy(0)

        def save():
            for layer, tensor in sources.items():
                for block, key in zip(blocks, keys[layer]):
                    if not engine.contains(key):
                        engine.put_tensor(key, tensor[block.block_id], stage)

        def load():
            for layer, tensor in outputs.items():
                for block, key in zip(blocks, keys[layer]):
                    if not engine.get_tensor(key, tensor[block.block_id], stage):
                        raise RuntimeError(
                            "Phase 1 reference cache unexpectedly missed"
                        )

        timings = _latency(save), _latency(load)
        return timings, engine.extended_stats()


def _phase2(options, sources, outputs, blocks, _keys):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(
            engine_id=uuid4().hex,
            kv_connector_extra_config=options,
        ),
    )
    worker = VeriCacheV1Connector(
        config,
        KVConnectorRole.WORKER,
        SimpleNamespace(kv_cache_groups=()),
    )

    def save():
        for layer, tensor in sources.items():
            worker.save_kv_layer(layer, tensor, None)
        worker.wait_for_save()

    def load():
        worker.start_load_kv(None)
        for layer in outputs:
            worker.wait_for_layer_load(layer)
        worker.wait_for_save()
        if worker.get_block_ids_with_load_errors():
            raise RuntimeError("Phase 2 connector unexpectedly missed")

    try:
        worker.register_kv_caches(sources)
        worker.bind_connector_metadata(VeriCacheConnectorMetadata(save=blocks))
        save_time = _latency(save)
        worker.clear_connector_metadata()
        worker.register_kv_caches(outputs)
        worker.bind_connector_metadata(VeriCacheConnectorMetadata(load=blocks))
        load_time = _latency(load)
        return (save_time, load_time), worker.engine.extended_stats()
    finally:
        worker.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--block-bytes", type=int, default=64 * 1024)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--staging-pool-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--without-nvme", action="store_true")
    parser.add_argument(
        "--pressure", action="store_true", help="Size DRAM for about half the dataset"
    )
    args = parser.parse_args()
    if (
        min(
            args.layers,
            args.blocks,
            args.block_bytes,
            args.shards,
            args.repeats,
            args.staging_pool_size,
        )
        < 1
    ):
        parser.error("workload dimensions and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device unavailable")
    if args.without_nvme and args.pressure:
        parser.error("NVMe is required for the pressure run")
    torch.set_num_threads(1)
    layers = [f"layer.{i}" for i in range(args.layers)]
    hashes = compute_prefix_block_hashes(range(args.blocks * 16), 16, "benchmark")
    blocks = [BlockTransfer(digest, i) for i, digest in enumerate(hashes)]
    keys = {
        layer: [derive_layer_block_key(block.block_hash, layer) for block in blocks]
        for layer in layers
    }
    histogram = Counter(
        _shard(key, args.shards) for row in keys.values() for key in row
    )
    max_frames = max(histogram.values())
    host_frames = max(1, max_frames // 2) if args.pressure else max_frames + 1
    nvme_frames = 0 if args.without_nvme else max_frames
    sources = {
        layer: torch.full(
            (args.blocks, args.block_bytes),
            i % 255 + 1,
            dtype=torch.uint8,
            device=args.device,
        )
        for i, layer in enumerate(layers)
    }
    outputs = {layer: torch.empty_like(value) for layer, value in sources.items()}
    payload = args.layers * args.blocks * args.block_bytes
    print(
        f"{args.layers} layers × {args.blocks} blocks × {args.block_bytes:,} B "
        f"= {payload / 2**20:.1f} MiB on {args.device}, {args.shards} shards, "
        f"{host_frames} DRAM and {nvme_frames} NVMe frames/shard",
        flush=True,
    )
    print(
        f"Phase 2 uses {args.io_workers} I/O workers and {args.staging_pool_size} staging slots. "
        "The Phase 1 reference uses per-block staging and synchronous QD=1 spills.",
        flush=True,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    target = _ROOT / "target"
    with TemporaryDirectory(prefix="bench-phase2-", dir=target) as folder:
        times = [[], []]
        counters = [[], []]
        runners = (_phase1, _phase2)
        for repeat in range(args.repeats):
            for idx in (0, 1) if repeat % 2 == 0 else (1, 0):
                for tensor in outputs.values():
                    tensor.zero_()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                options = dict(
                    num_shards=args.shards,
                    host_frames_per_shard=host_frames,
                    nvme_frames_per_shard=nvme_frames,
                    frame_size=args.block_bytes,
                    nvme_path=(
                        Path(folder) / f"run-{repeat}-{idx}" if nvme_frames else None
                    ),
                    prefer_cuda_pinned=args.device == "cuda",
                    io_workers=args.io_workers,
                    staging_pool_size=args.staging_pool_size,
                    block_axis=0,
                )
                timings, stats = runners[idx](options, sources, outputs, blocks, keys)
                if not all(
                    torch.equal(sources[layer], value)
                    for layer, value in outputs.items()
                ):
                    raise AssertionError(f"Phase {idx + 1} KV restoration differed")
                times[idx].append(timings)
                counters[idx].append(stats)
                if options["nvme_path"]:
                    for file in Path(folder).glob(f"run-{repeat}-{idx}*"):
                        file.unlink()
    print(f"Median of {args.repeats} exact round trips. GB/s uses 10⁹ bytes.")
    print(
        f"{'operation':12} {'Phase 1 ms':>12} {'GB/s':>9} {'Phase 2 ms':>12} {'GB/s':>9} {'speedup':>9}"
    )
    operations = (
        ("save", lambda pair: pair[0], payload),
        ("load", lambda pair: pair[1], payload),
        ("round trip", sum, 2 * payload),
    )
    for label, measure, size in operations:
        old, new = (median(map(measure, trials)) for trials in times)
        print(
            f"{label:12} {old * 1000:12.2f} {size / old / 1e9:9.2f} {new * 1000:12.2f} {size / new / 1e9:9.2f} {old / new:8.2f}×"
        )
    for index, stats in enumerate(counters, 1):
        print(
            f"Phase {index} median completed writebacks / clean DRAM evictions / NVMe promotions "
            + " / ".join(
                str(median(row[name] for row in stats))
                for name in (
                    "async_writebacks",
                    "clean_dram_evictions",
                    "nvme_promotions",
                )
            )
        )


if __name__ == "__main__":
    main()
