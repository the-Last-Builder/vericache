use crate::schema::{bytes_per_sec_to_gbps, JobSpec, LatencyHist, MetricRecord, ProcSampler};
use std::fs;
use std::time::Instant;
use transfer::context::*;
use transfer::cuda::{self, CudaEvent, CudaSlice, CudaStream};
use transfer::domain::DeviceId;
use transfer::host::{self, HostContext};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::engine::LocalEngine;
use vericache_core::kernel::CacheCore;
use vericache_core::spec::{ObjectKey, Ticket};
use vericache_core::timeline::TimelineKind;
use vstd::prelude::{Ghost, Map, Tracked};

pub fn run(job: &JobSpec) -> Vec<MetricRecord> {
    match job.case.as_str() {
        "1a_gpu_to_host_scatter" => vec![run_1a_gpu_to_host_scatter(job)],
        "1a_host_to_gpu_scatter" => vec![run_1a_host_to_gpu_scatter(job)],
        "1b_nvme_write_spill" => vec![run_1b_nvme_write_spill(job)],
        "1b_nvme_read_cold" => vec![run_1b_nvme_read_cold(job)],
        other => panic!("unknown s1_local_tiers case '{other}'"),
    }
}

fn fill_pattern(len: usize, seed: u8) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for (i, b) in v.iter_mut().enumerate() {
        *b = seed.wrapping_add(((i.wrapping_mul(13)) & 0xff) as u8);
    }
    v
}

/// 1A-D2H: 16 non-contiguous GPU Paged KV blocks -> `CudaBackend::put_gpu_scatter`
/// -> `copy_hop_cuda` into Pinned Host Frame + `CudaEvent` / `synchronize_cuda`.
fn run_1a_gpu_to_host_scatter(job: &JobSpec) -> MetricRecord {
    let dev: DeviceId = job.donor_gpu;
    let frame_size = job.chunk_bytes;
    let num_blocks = job.num_blocks.max(1);
    assert!(
        frame_size % num_blocks == 0,
        "chunk_bytes ({frame_size}) must be divisible by num_blocks ({num_blocks})"
    );
    let block_len = frame_size / num_blocks;

    let (instance, mut alloc, mut events, mut shares) = NodeContext::bootstrap(0).into_sources();
    let mut core = CacheCore::new(0, frame_size, Vec::new(), 64, Tracked::assume_new());
    let host_src: Tracked<EventSourceToken> = Tracked::assume_new();
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        host_src,
        4,
        4096,
    )
    .expect("HostBackend::new");

    let mut cuda_backend = CudaBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        dev,
        4,
        64,
        false,
    )
    .expect("CudaBackend::new");
    cuda_backend
        .attach_pinned_host(&core, &host)
        .expect("attach_pinned_host");

    let caller_tl = core
        .register_timeline(TimelineKind::CudaStream { device: dev })
        .expect("register caller timeline");
    let mut caller_stream =
        CudaStream::new(dev, Tracked::assume_new()).expect("CudaStream::new");
    let mut done_event =
        CudaEvent::new(dev, Ghost::assume_new(), Tracked::assume_new()).expect("CudaEvent::new");

    let eng_ctx = HostContext::new(Tracked::assume_new());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, Some(cuda_backend), None);

    // Allocate 16 genuinely non-contiguous GPU Paged KV blocks (interleaved with spacer
    // allocations so blocks are non-adjacent in HBM, mimicking vLLM PagedAttention).
    let mut src_gpu_bufs = Vec::with_capacity(num_blocks);
    let mut _spacers = Vec::with_capacity(num_blocks);
    let mut prep_ctx = HostContext::new(Tracked::assume_new());
    let mut last_ticket = Ticket::NONE;

    for i in 0..num_blocks {
        let data = fill_pattern(block_len, (i as u8).wrapping_mul(17).wrapping_add(3));
        let (h_buf, _) =
            cuda::alloc_host(Tracked::assume_new(), block_len).expect("alloc_host stage");
        let h_slice = h_buf.as_host_slice();
        let _w_comp = host::write_slice(
            &mut prep_ctx,
            Tracked::assume_new(),
            &h_slice,
            &data,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Ghost::assume_new(),
        );
        let (g_buf, _) =
            cuda::alloc_device(Tracked::assume_new(), dev, block_len).expect("alloc_device block");
        let g_slice = g_buf.slot_slice(0);
        let h_cuda = h_buf.as_cuda_slice();
        let (_, receipt) = cuda::memcpy_async(
            &mut caller_stream,
            Tracked::assume_new(),
            &g_slice,
            &h_cuda,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Ghost::assume_new(),
        )
        .expect("upload paged block");
        last_ticket = engine
            .core
            .issue_ticket(caller_tl, receipt)
            .expect("issue prep ticket");
        src_gpu_bufs.push(g_buf);
        // Allocate a small 4KB spacer to force non-contiguous device addresses.
        let (sp, _) = cuda::alloc_device(Tracked::assume_new(), dev, 4096).expect("spacer");
        _spacers.push(sp);
    }
    let k_prep = cuda::stream_synchronize(&caller_stream, Tracked::assume_new())
        .expect("sync prep stream");
    if last_ticket != Ticket::NONE {
        engine
            .core
            .mark_timeline_complete(caller_tl, last_ticket.seq(), k_prep);
    }

    let src_slices: Vec<CudaSlice<'_>> = src_gpu_bufs.iter().map(|b| b.slot_slice(0)).collect();
    let host_tier = engine.host_tier();
    let gpu_tier = engine.gpu_tier().unwrap();
    let total_iters = job.warmup_iters + job.iters;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();

    for it in 0..total_iters {
        if it == job.warmup_iters {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let key = ObjectKey([1, it as u64, frame_size as u64, 0]);
        let mut src_perms: Tracked<Map<vstd::prelude::int, ReadPerm>> = Tracked::assume_new();

        let t0 = Instant::now();
        let (_ver, gpu_f, _t_sc, _sp) = engine
            .put_gpu_scatter(
                Tracked::assume_new(),
                key,
                &src_slices,
                Tracked::assume_new(),
                Ghost::assume_new(),
                Ghost::assume_new(),
                Ghost::assume_new(),
            )
            .expect("put_gpu_scatter");
        engine
            .synchronize_cuda(Tracked::assume_new())
            .expect("sync after put_gpu_scatter");

        // D2H hop from gathered GPU VRAM frame -> Pinned Host Frame + CudaEvent sync.
        let (host_f, _t_hop, _sp_hop) = {
            let cuda_b = engine.cuda.as_mut().unwrap();
            let res = cuda_b
                .copy_hop_cuda(
                    &mut engine.core,
                    Tracked::assume_new(),
                    Tracked::assume_new(),
                    gpu_f,
                    host_tier,
                )
                .expect("copy_hop_cuda GPU->PinnedHost");
            cuda::event_record(&mut done_event, &cuda_b.timeline.stream)
                .expect("event_record");
            res
        };
        engine
            .synchronize_cuda(Tracked::assume_new())
            .expect("sync after D2H copy_hop_cuda");
        let dt_ns = t0.elapsed().as_nanos() as u64;

        let _ = &mut src_perms;
        engine.evict_frame(gpu_f).expect("evict gpu_f");
        engine.evict_frame(host_f).expect("evict host_f");
        let r1 = engine.reclaim_draining(gpu_tier, 8);
        let r2 = engine.reclaim_draining(host_tier, 8);
        assert_eq!(r1 + r2, 2, "reclaim both gpu and pinned host frames");

        if it >= job.warmup_iters {
            hist.record_ns(dt_ns);
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = (job.iters as u64) * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);

    let _ = done_event.destroy();
    let _ = caller_stream.destroy();
    let _ = (&instance, &mut alloc, &mut events, &mut shares);

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = format!(
        "CudaBackend::put_gpu_scatter ({num_blocks} non-contiguous GPU blocks -> GPU frame) + copy_hop_cuda D2H -> Pinned Host Frame + CudaEvent sync"
    );
    rec
}

/// 1A-H2D: Pinned Host Frame -> 16 non-contiguous GPU Paged KV blocks via
/// 1-hop `LocalEngine::get_into_gpu_scatter` + `CudaEvent`.
fn run_1a_host_to_gpu_scatter(job: &JobSpec) -> MetricRecord {
    let dev: DeviceId = job.donor_gpu;
    let frame_size = job.chunk_bytes;
    let num_blocks = job.num_blocks.max(1);
    assert!(
        frame_size % num_blocks == 0,
        "chunk_bytes ({frame_size}) must be divisible by num_blocks ({num_blocks})"
    );
    let block_len = frame_size / num_blocks;

    let (instance, mut alloc, mut events, mut shares) = NodeContext::bootstrap(0).into_sources();
    let mut core = CacheCore::new(0, frame_size, Vec::new(), 64, Tracked::assume_new());
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        4,
        4096,
    )
    .expect("HostBackend::new");

    let mut cuda_backend = CudaBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        dev,
        2,
        64,
        false,
    )
    .expect("CudaBackend::new");
    cuda_backend
        .attach_pinned_host(&core, &host)
        .expect("attach_pinned_host");

    let caller_tl = core
        .register_timeline(TimelineKind::CudaStream { device: dev })
        .expect("register caller timeline");
    let mut caller_stream =
        CudaStream::new(dev, Tracked::assume_new()).expect("CudaStream::new");
    let mut done_event =
        CudaEvent::new(dev, Ghost::assume_new(), Tracked::assume_new()).expect("CudaEvent::new");

    let eng_ctx = HostContext::new(Tracked::assume_new());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, Some(cuda_backend), None);

    // Publish a frame in Pinned Host DRAM (NOT in GPU VRAM, so get_into_gpu_scatter executes
    // a direct 1-hop Pinned Host Frame -> 16 non-contiguous GPU blocks H2D scatter!).
    let payload = fill_pattern(frame_size, 0x5A);
    let key = ObjectKey([2, frame_size as u64, 0, 0]);
    let (_ver, _host_f) = engine
        .put_host(Tracked::assume_new(), key, &payload)
        .expect("put_host pinned frame");

    // Allocate 16 non-contiguous destination GPU blocks.
    let mut dst_gpu_bufs = Vec::with_capacity(num_blocks);
    let mut _spacers = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let (g_buf, _) =
            cuda::alloc_device(Tracked::assume_new(), dev, block_len).expect("alloc_device dst");
        dst_gpu_bufs.push(g_buf);
        let (sp, _) = cuda::alloc_device(Tracked::assume_new(), dev, 4096).expect("spacer");
        _spacers.push(sp);
    }
    let dst_slices: Vec<CudaSlice<'_>> = dst_gpu_bufs.iter().map(|b| b.slot_slice(0)).collect();

    let total_iters = job.warmup_iters + job.iters;
    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();

    for it in 0..total_iters {
        if it == job.warmup_iters {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let (_v, t_out) = engine
            .get_into_gpu_scatter(
                Tracked::assume_new(),
                Tracked::assume_new(),
                key,
                &mut caller_stream,
                caller_tl,
                &dst_slices,
                Tracked::assume_new(),
                Ghost::assume_new(),
            )
            .expect("get_into_gpu_scatter")
            .expect("pinned host hit");
        cuda::event_record(&mut done_event, &caller_stream).expect("event_record");
        let k_out = cuda::stream_synchronize(&caller_stream, Tracked::assume_new())
            .expect("stream_synchronize");
        engine
            .core
            .mark_timeline_complete(caller_tl, t_out.seq(), k_out);
        let dt_ns = t0.elapsed().as_nanos() as u64;

        if it >= job.warmup_iters {
            hist.record_ns(dt_ns);
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = (job.iters as u64) * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);

    let _ = done_event.destroy();
    let _ = caller_stream.destroy();
    let _ = (&instance, &mut alloc, &mut events, &mut shares);

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = format!(
        "LocalEngine::get_into_gpu_scatter 1-hop Pinned Host Frame -> {num_blocks} non-contiguous GPU blocks + CudaEvent sync"
    );
    rec
}

/// 1B-Write: `NvmeBackend` (`O_DIRECT` + `io_uring` in `/workspace/target/bench-nvme`) write spill.
/// Supports both single-flight (`concurrency == 1`) and queued batch (`concurrency > 1`).
fn run_1b_nvme_write_spill(job: &JobSpec) -> MetricRecord {
    fs::create_dir_all(&job.nvme_dir).expect("create nvme_dir");
    let frame_size = job.chunk_bytes;
    let qd = job.concurrency.max(1).min(32);
    let pool_frames = (qd + 2) as u32;

    let (instance, mut alloc, mut events, mut shares) = NodeContext::bootstrap(0).into_sources();
    let mut core = CacheCore::new(0, frame_size, Vec::new(), 128, Tracked::assume_new());
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        pool_frames,
        4096,
    )
    .expect("HostBackend::new");

    let ring_entries = (qd * 4).next_power_of_two().max(16) as u32;
    let nvme = NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        &job.nvme_dir,
        pool_frames,
        ring_entries,
    )
    .expect("NvmeBackend::new");
    let eng_ctx = HostContext::new(Tracked::assume_new());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, Some(nvme));
    let nvme_tier = engine.nvme_tier().unwrap();

    // Pre-populate `qd` Host DRAM frames with deterministic payloads.
    let mut host_frames = Vec::with_capacity(qd);
    let mut keys = Vec::with_capacity(qd);
    for i in 0..qd {
        let key = ObjectKey([10, i as u64, frame_size as u64, 0]);
        let data = fill_pattern(frame_size, (i as u8).wrapping_mul(31).wrapping_add(7));
        let (_v, hf) = engine
            .put_host(Tracked::assume_new(), key, &data)
            .expect("put_host seed");
        keys.push(key);
        host_frames.push(hf);
    }

    let warmup_batches = (job.warmup_iters + qd - 1) / qd;
    let meas_batches = (job.iters + qd - 1) / qd;
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();
    let mut total_ops = 0u64;

    for b in 0..total_batches {
        if b == warmup_batches {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let mut nvme_frames = Vec::with_capacity(qd);
        if qd == 1 {
            let (nf, ticket) = engine
                .tier_dram_to_nvme(Tracked::assume_new(), Tracked::assume_new(), keys[0])
                .expect("tier_dram_to_nvme");
            let prog = engine
                .wait_nvme_ticket(Tracked::assume_new(), ticket)
                .expect("wait_nvme_ticket");
            assert_eq!(prog.failed, 0);
            nvme_frames.push(nf);
        } else {
            let mut nvme_b = engine.nvme.take().unwrap();
            let mut last_ticket = Ticket::NONE;
            for &hf in &host_frames {
                let (nf, ticket) = nvme_b
                    .stage_dram_to_nvme(
                        engine.host,
                        &mut engine.core,
                        Tracked::assume_new(),
                        Tracked::assume_new(),
                        hf,
                    )
                    .expect("stage_dram_to_nvme");
                nvme_frames.push(nf);
                last_ticket = ticket;
            }
            nvme_b
                .submit(Ghost::assume_new())
                .expect("nvme batch submit");
            engine.nvme = Some(nvme_b);
            let prog = engine
                .wait_nvme_ticket(Tracked::assume_new(), last_ticket)
                .expect("wait_nvme_ticket batch");
            assert_eq!(prog.failed, 0);
        }
        let dt_ns = t0.elapsed().as_nanos() as u64;

        for nf in nvme_frames {
            engine.evict_frame(nf).expect("evict nvme frame");
        }
        let reclaimed = engine.reclaim_draining(nvme_tier, qd + 4);
        assert_eq!(reclaimed, qd, "reclaim all nvme spill frames");

        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += qd as u64;
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);
    let _ = (&instance, &mut alloc, &mut events, &mut shares);

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = format!("NvmeBackend O_DIRECT + io_uring write spill (QD={qd}, dir={})", job.nvme_dir);
    rec
}

/// 1B-Read: `NvmeBackend` (`O_DIRECT` + `io_uring` in `/workspace/target/bench-nvme`) cold read
/// (DRAM copy evicted prior to read so every transfer hits NVMe `O_DIRECT`).
fn run_1b_nvme_read_cold(job: &JobSpec) -> MetricRecord {
    fs::create_dir_all(&job.nvme_dir).expect("create nvme_dir");
    let frame_size = job.chunk_bytes;
    let qd = job.concurrency.max(1).min(32);
    let pool_frames = (qd + 2) as u32;

    let (instance, mut alloc, mut events, mut shares) = NodeContext::bootstrap(0).into_sources();
    let mut core = CacheCore::new(0, frame_size, Vec::new(), 128, Tracked::assume_new());
    let host = HostBackend::new(
        &mut core,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        pool_frames,
        4096,
    )
    .expect("HostBackend::new");

    let ring_entries = (qd * 4).next_power_of_two().max(16) as u32;
    let nvme = NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked::assume_new(),
        Tracked::assume_new(),
        Tracked::assume_new(),
        &job.nvme_dir,
        pool_frames,
        ring_entries,
    )
    .expect("NvmeBackend::new");
    let eng_ctx = HostContext::new(Tracked::assume_new());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, Some(nvme));
    let host_tier = engine.host_tier();

    // Seed `qd` frames onto NVMe, then evict their DRAM copies so they live ONLY on NVMe.
    let mut nvme_frames = Vec::with_capacity(qd);
    let mut keys = Vec::with_capacity(qd);
    for i in 0..qd {
        let key = ObjectKey([11, i as u64, frame_size as u64, 0]);
        let data = fill_pattern(frame_size, (i as u8).wrapping_mul(19).wrapping_add(11));
        let (_v, hf) = engine
            .put_host(Tracked::assume_new(), key, &data)
            .expect("put_host seed");
        let (nf, ticket) = engine
            .tier_dram_to_nvme(Tracked::assume_new(), Tracked::assume_new(), key)
            .expect("tier_dram_to_nvme seed");
        let prog = engine
            .wait_nvme_ticket(Tracked::assume_new(), ticket)
            .expect("wait seed spill");
        assert_eq!(prog.failed, 0);
        engine.evict_frame(hf).expect("evict seed dram");
        assert_eq!(engine.reclaim_draining(host_tier, 4), 1);
        keys.push(key);
        nvme_frames.push(nf);
    }

    let warmup_batches = (job.warmup_iters + qd - 1) / qd;
    let meas_batches = (job.iters + qd - 1) / qd;
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();
    let mut total_ops = 0u64;

    for b in 0..total_batches {
        if b == warmup_batches {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let mut dram_frames = Vec::with_capacity(qd);
        if qd == 1 {
            let (df, ticket) = engine
                .warm_host_from_nvme(Tracked::assume_new(), Tracked::assume_new(), keys[0])
                .expect("warm_host_from_nvme");
            let prog = engine
                .wait_nvme_ticket(Tracked::assume_new(), ticket)
                .expect("wait_nvme_ticket read");
            assert_eq!(prog.failed, 0);
            dram_frames.push(df);
        } else {
            let mut nvme_b = engine.nvme.take().unwrap();
            let mut last_ticket = Ticket::NONE;
            for &nf in &nvme_frames {
                let (df, ticket) = nvme_b
                    .stage_nvme_to_dram(
                        engine.host,
                        &mut engine.core,
                        Tracked::assume_new(),
                        Tracked::assume_new(),
                        nf,
                    )
                    .expect("stage_nvme_to_dram");
                dram_frames.push(df);
                last_ticket = ticket;
            }
            nvme_b
                .submit(Ghost::assume_new())
                .expect("nvme batch read submit");
            engine.nvme = Some(nvme_b);
            let prog = engine
                .wait_nvme_ticket(Tracked::assume_new(), last_ticket)
                .expect("wait_nvme_ticket batch read");
            assert_eq!(prog.failed, 0);
        }
        let dt_ns = t0.elapsed().as_nanos() as u64;

        for df in dram_frames {
            engine.evict_frame(df).expect("evict warmed dram");
        }
        let reclaimed = engine.reclaim_draining(host_tier, qd + 4);
        assert_eq!(reclaimed, qd, "reclaim all warmed dram frames");

        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += qd as u64;
        }
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);
    let _ = (&instance, &mut alloc, &mut events, &mut shares);

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = format!("NvmeBackend O_DIRECT + io_uring cold read (QD={qd}, dir={})", job.nvme_dir);
    rec
}
