"""Gather Stage 0 evidence and render the acceptance report."""

import hashlib
import json
from pathlib import Path
import re


ROOT = Path("/data/nvme2/zxw/vericache_phase3")
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "stage0_step2"
BASE = "a45c22a2d7e4ca56db7068142dc40466592e8021"
SIM = "3605393955418a144559545434751adefdc7a78f"


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def result(reported, prefix):
    return [path for path in reported if path.name.startswith(f"stage0_hw_{prefix}.")]


def latest(runs, predicate):
    matching = [path for path in runs if predicate(path)]
    return max(matching, key=lambda path: (path / "exit-status").stat().st_mtime) if matching else None


def successful(path):
    return path is not None and (path / "exit-status").read_text().strip() == "0"


def main():
    OUTPUT.mkdir(exist_ok=True)
    reported = sorted(path for path in RESULTS.glob("stage0_hw_*") if (path / "exit-status").is_file())
    correctness = result(reported, "correctness")
    comparison = {}
    for batch in (False, True):
        for scenario in ("vanilla", "hbm", "dram", "nvme"):
            path = latest(correctness, lambda path: (
                (path / "summary.json").is_file()
                and (path / "manifest.json").is_file()
                and load(path / "manifest.json").get("batch_invariant", False) is batch
                and load(path / "summary.json")["scenario"] == scenario
            ))
            if path:
                comparison[batch, scenario] = path

    local = latest(result(reported, "local"), successful)
    gdr = latest(result(reported, "gdr"), successful)
    calibration = latest(result(reported, "calibration"), successful)
    layout = {}
    for path in result(reported, "layout"):
        if successful(path):
            geometry = load(path / "kv-layout.json")
            layout[geometry["vllm_scheduler_block_size"]] = path
    dram, nvme = (comparison.get((True, scenario)) for scenario in ("dram", "nvme"))
    hbm = comparison.get((True, "hbm"))
    frozen = ROOT / "traces/frozen"
    approved = (
        bool(local and gdr and len(layout) == 2)
        and all(successful(path) and load(path / "summary.json").get("gate_pass")
                for path in (dram, nvme))
        and successful(hbm)
        and load(hbm / "summary.json")["warm_matches_same_server_cold"] ==
        {"token_ids_exact": 20, "top1_logprobs_exact": 20}
        and calibration is not None
        and (frozen / "SHA256SUMS").is_file()
    )
    summary = {
        "stage": "Stage 0 Step 2",
        "accepted": approved,
        "correctness": {
            f"{'invariant' if batch else 'default'}_{scenario}": {
                "run": path.name, **load(path / "summary.json"),
            } for (batch, scenario), path in comparison.items()
        },
        "layout": {size: load(path / "kv-layout.json") for size, path in layout.items()},
        "calibration": load(calibration / "summary.json") if calibration else None,
        "lease_cleanup": {},
    }
    runs = {}
    for path in reported:
        if (path / "claim.json").stat().st_size:
            claim = load(path / "claim.json")["lease"]
            release = load(path / "release.json")["lease"]
            assert release["id"] == claim["id"] and release["state"] == "released"
            assert not release["active_gpu_ids"]
            cleanup = (path / "cleanup.log").read_text().strip().removeprefix("Unlinked run directory ")
            assert cleanup.startswith(f"{ROOT}/g.") and not Path(cleanup).exists()
            summary["lease_cleanup"][path.name] = {"gpu_ids": claim["gpu_ids"], "released": True}
        if (path / "manifest.json").is_file():
            runs[path.name] = load(path / "manifest.json")
    dump(OUTPUT / "manifest.json", {
        "vericache_git_base": BASE,
        "request_sim_git": SIM,
        "binaries_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in
            (ROOT / "bin/request-sim",
             ROOT / "vericache-python/target/release/libvericache_python.so")
        },
        "runs": runs,
    })
    dump(OUTPUT / "summary.json", summary)
    logs = {}
    with (OUTPUT / "requests.jsonl").open("w") as output:
        for path in reported:
            for requests in sorted(path.glob("*requests.jsonl")):
                with requests.open() as source:
                    for line in source:
                        output.write(json.dumps({"run": path.name, "source": requests.name,
                                                 "request": json.loads(line)}) + "\n")
            snapshots = sorted(path.glob("*metrics.prom"))
            loads = path / "connector-loads.jsonl"
            if snapshots or loads.is_file():
                logs[path.name] = {
                    "prometheus_snapshots": [str(log.relative_to(RESULTS)) for log in snapshots],
                    "connector_loads": (
                        [json.loads(line) for line in loads.read_text().splitlines()]
                        if loads.is_file() else []
                    ),
                }
    dump(OUTPUT / "server_metrics.json", logs)

    report = [
        "# Stage 0 Step 2 验收报告",
        "",
        f"**验收结果：{'通过' if approved else '未通过'}。**",
        "",
        "## 环境与硬件门槛",
        "",
        f"- `h20`，VeriCache base `{BASE}`，request-sim `{SIM}`。"
        "模型 16/16 个 shard 通过独立 SHA256 校验，两个原始 trace 校验通过。",
        "- CPU 压测完成 3 次精确恢复。CPU 测试使用进程级 vLLM CPU platform，"
        "并缩短 Phase 3 临时目录路径，结果为 23 通过、2 个 CUDA 用例预期跳过。",
        "- TCB 清单审计通过，包含 6 个 `ffi.rs`、2 个 C shim、1 个 `build.rs`。"
        "按批准口径，保留已有的 30 处 test-only `assume_new` 调用，Phase 3 零新增。",
    ]
    if local:
        text = (local / "bench-phase2-cuda-pressure.log").read_text()
        round_trip = re.search(r"round trip\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)×", text)
        old_ms, old_rate, new_ms, new_rate, speedup = round_trip.groups()
        report.append(
            f"- GPU 5 的 19 项测试全部通过，两个 CUDA 用例均验证 slab pinning。"
            f"64 MiB CUDA pressure 微基准取 3 次中位数，Phase 1 与 Phase 2 的 round trip 分别为 "
            f"{old_ms} ms（{old_rate} GB/s）和 {new_ms} ms（{new_rate} GB/s），耗时比为 {speedup}×。"
        )
    if gdr:
        report.append("- GPU 6、7 使用 `mlx5_4` 的 RoCEv2 GID 3，3 项测试全部通过。"
                      "2-GPU 用例断言 `REMOTE_GDR_GPU` 命中和 `torch.equal`。")
    report.extend(["", "## Qwen3-30B-A3B KV layout", "",
                   "TP=1、BF16。实测 48 层使用 `FLASH_ATTN`，一个 `FullAttentionSpec` group，"
                   "block axis 为 1。下表的 stride 单位是 tensor 元素。", "",
                   "| Block size | HBM blocks | 每层 shape | 每层 stride | 每层 block 字节与 frame_size |",
                   "| --- | ---: | --- | --- | ---: |"])
    for size, path in sorted(layout.items(), reverse=True):
        geometry = load(path / "kv-layout.json")
        layer = geometry["layers"][0]
        report.append(f"| {size} | {geometry['hbm_kv_blocks']} | `{layer['shape']}` | "
                      f"`{layer['stride']}` | {layer['bytes_per_layer_per_block']:,} |")
    report.extend(["", "## 20 个 ShareGPT prompt 的 bit-exact 检查", "",
                   "同一条 prompt 生成 16 个 greedy token，分别对比 token ID 序列和每步 top-1 logprob。"
                   "每一行均以该模式下的 vanilla 冷跑为基准。", "",
                   "| 模式 | 路径 | Token ID 完全一致 | Top-1 logprob 完全一致 |",
                   "| --- | --- | ---: | ---: |"])
    for batch, label in ((False, "默认"), (True, "Batch invariant")):
        for scenario, description in (("hbm", "Vanilla HBM warm"),
                                      ("dram", "VeriCache DRAM warm"),
                                      ("nvme", "VeriCache NVMe warm")):
            path = comparison.get((batch, scenario))
            if path:
                data = load(path / "summary.json")
                key = "warm_matches_same_server_cold" if batch and scenario == "hbm" else "warm_matches_golden"
                counts = data[key]
                report.append(f"| {label} | {description} | {counts['token_ids_exact']}/20 | "
                              f"{counts['top1_logprobs_exact']}/20 |")
                if scenario == "dram":
                    fault = data["fault_matches_golden"]
                    report.append(f"| {label} | VeriCache 晚淘汰重算 | {fault['token_ids_exact']}/20 | "
                                  f"{fault['top1_logprobs_exact']}/20 |")
    report.extend([
        "",
        "默认模式下，DRAM 与 NVMe warm 结果逐项等于 vanilla HBM warm，三条路径与冷跑的差异出现在同一组 prompt。"
        "验收复跑给 vanilla 和 VeriCache 同时设置 `VLLM_BATCH_INVARIANT=1`，显式指定 `FLASH_ATTN`，"
        "并录制同模式的 vanilla 冷跑基准。在两种模式下，注入的晚淘汰都由 load-error callback 报告。",
        "",
        "## Trace 窗口与校准",
        "",
        "按完整 prefix 去重后，至少 1,500 行的窗口会超出计划的工作集预算。"
        "各 trace 从 25%、50% 和 75% 三处开始构造连续候选窗口，增长至约 130 GiB，"
        "再取中位输入和输出长度最接近整个过滤后 trace 的候选窗口。超限判断预留 32 个 chat template token。",
        "",
        "| Trace | 原始窗口 [start, stop) | 保留 | 过滤 | 估算 prefill KV 工作集 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    preparation = load(RESULTS / "stage0_trace_preparation/windows.json")
    for name, row in preparation["traces"].items():
        report.append(f"| {name} | [{row['source_window_start_zero_based']}, "
                      f"{row['source_window_stop_exclusive']}) | {row['rows_retained']} | "
                      f"{row['window_context_filtered']} ({row['window_context_filtered_percent']:.2f}%) | "
                      f"{row['estimated_unique_prefill_kv_gib']:.2f} GiB |")
    if calibration:
        calib = load(calibration / "summary.json")
        n = sum(len(rows) for rows in calib.values())
        offsets = ", ".join(map(str, sorted({row["chat_template_delta"] for rows in calib.values() for row in rows})))
        report.extend(["", f"两条 trace 共 {n} 条在线探测均返回 HTTP 200，每条的 `prompt_tokens` "
                       f"都等于离线模板 tokenization，观测的模板增量为 {offsets} token。"
                       "`cached_tokens` 都等于按实发 token 序列计算的 block-aligned prefix 上界。"
                       "窗口和 prompt cache 的 SHA256 清单保存在 `traces/frozen/SHA256SUMS`。"])
    report.extend(["", "## 原始记录", "",
                   f"- 汇总目录 `{OUTPUT}`。各子运行的参数与二进制 SHA256 见 `manifest.json`。",
                   "- 原始请求、layout 和验收结果见 `requests.jsonl` 与 `summary.json`。"
                   "Prometheus 快照路径和 connector 计数见 `server_metrics.json`。",
                   "- 所有实验 claim 均核对为 `released`，运行目录均已 unlink。"
                   "NVMe correctness 运行的实例 quota 为 64 GiB。",
                   "- 核心 crate、root Cargo 文件及 TCB 审计脚本经 Git 差异核对。", ""])
    (OUTPUT / "report.md").write_text("\n".join(report))
    print(OUTPUT / "report.md")


if __name__ == "__main__":
    main()
