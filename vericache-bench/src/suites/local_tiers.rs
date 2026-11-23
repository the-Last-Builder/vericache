use crate::schema::{JobSpec, LatencyHist, MetricRecord, ProcSampler, bytes_per_sec_to_gbps};
use crate::shard::ShardCapabilityBundle;
use std::fs;
use std::time::Instant;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::context::*;
use transfer::cuda::{self, CudaEvent, CudaSlice, CudaStream};
use transfer::domain::DeviceId;
use transfer::host::{self, HostContext};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::engine::LocalEngine;
#[cfg(verus_only)]
use vericache_core::provenance::{ValueCertificate, ValueProof};
use vericache_core::spec::{ObjectKey, Ticket};
use vericache_core::timeline::TimelineKind;
#[cfg(verus_only)]
use vstd::prelude::Seq;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use vstd::prelude::{Ghost, Map, Tracked, verus};

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

fn finish_local_metric(
    job: &JobSpec,
    elapsed_s: f64,
    sampler: ProcSampler,
    hist: &LatencyHist,
    total_ops: u64,
    frame_size: usize,
    notes: String,
) -> MetricRecord {
    let (cpu_pct, rss_mb) = sampler.finish();
    let total_bytes = total_ops * (frame_size as u64);
    let gbps = bytes_per_sec_to_gbps(total_bytes, elapsed_s);

    let mut rec = MetricRecord::from_job("vericache", job);
    rec.throughput_gbps = gbps;
    rec.goodput_gbps = gbps;
    rec.p50_us = hist.p50_us();
    rec.p95_us = hist.p95_us();
    rec.p99_us = hist.p99_us();
    rec.hit_rate = 1.0;
    rec.cpu_pct = cpu_pct;
    rec.rss_mb = rss_mb;
    rec.notes = notes;
    rec
}

verus! {

/// 1A-D2H: 16 non-contiguous GPU Paged KV blocks -> `CudaBackend::put_gpu_scatter`
/// -> `copy_hop_cuda` into Pinned Host Frame + `CudaEvent` / `synchronize_cuda`.
fn run_1a_gpu_to_host_scatter_inner(
    job: &JobSpec,
    num_blocks: usize,
    block_len: usize,
) -> (f64, ProcSampler, LatencyHist) {
    let dev: DeviceId = job.donor_gpu;
    let frame_size = job.chunk_bytes;

    let mut caps = ShardCapabilityBundle::take(0);
    let mut core = caps.make_local_core(frame_size, 64);
    let host_src = caps.mint_host();
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        host_src,
        4,
        4096,
    )
    .expect("HostBackend::new");

    let cuda_src = caps.mint_cuda();
    let mut cuda_backend = CudaBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        cuda_src,
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
    let mut caller_stream = CudaStream::new(dev, caps.mint_cuda()).expect("CudaStream::new");
    let mut done_event =
        CudaEvent::new(dev, Ghost(0), Tracked(caps.instance.borrow())).expect("CudaEvent::new");

    let eng_ctx = HostContext::new(caps.mint_host());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, Some(cuda_backend), None);

    // Allocate 16 genuinely non-contiguous GPU Paged KV blocks (interleaved with spacer
    // allocations so blocks are non-adjacent in HBM, mimicking vLLM PagedAttention).
    let mut src_gpu_bufs = Vec::with_capacity(num_blocks);
    let mut _spacers = Vec::with_capacity(num_blocks);
    let mut prep_ctx = HostContext::new(caps.mint_host());
    let mut last_ticket = Ticket::NONE;
    let tracked mut src_perms: Map<int, ReadPerm> = Map::tracked_empty();
    let ghost mut src_certs: Seq<ValueCertificate> = Seq::empty();
    let ghost mut offsets: Seq<usize> = Seq::empty();
    let ghost mut gathered_bytes: Seq<u8> = Seq::empty();

    let mut i = 0usize;
    while i < num_blocks {
        let data = fill_pattern(block_len, (i as u8).wrapping_mul(17).wrapping_add(3));
        let (h_buf, Tracked(mut h_tok)) =
            cuda::alloc_host(Tracked(caps.alloc.borrow_mut()), block_len).expect("alloc_host stage");
        let h_slice = h_buf.as_host_slice();
        let Tracked(w_comp) = host::write_slice(
            &mut prep_ctx,
            Tracked(caps.instance.borrow()),
            &h_slice,
            &data,
            Tracked(&mut h_tok),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        let ghost w_ev = w_comp.submitted.key();
        let t_w = engine
            .core
            .issue_ticket(engine.host.timeline_id(), Tracked(w_comp.submitted))
            .expect("issue host prep ticket");
        engine
            .core
            .mark_timeline_complete(engine.host.timeline_id(), t_w.seq(), Tracked(w_comp.knows));

        let (g_buf, Tracked(mut g_tok)) =
            cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), dev, block_len)
                .expect("alloc_device block");
        let g_slice = g_buf.slot_slice(0);
        let h_cuda = h_buf.as_cuda_slice();
        let tracked mut h_perm = ReadPerm::Owned(h_tok);
        let ghost citations = engine.core.citations();
        let (Ghost(copy_ev), receipt) = {
            let tracked ledger = engine.core.ledger.borrow_mut();
            cuda::memcpy_async(
                &mut caller_stream,
                Tracked(caps.instance.borrow()),
                &g_slice,
                &h_cuda,
                Tracked(&mut h_perm),
                Tracked(&mut g_tok),
                Tracked(&mut ledger.submitted),
                Tracked(&ledger.knows),
                Ghost(citations),
            )
            .expect("upload paged block")
        };
        last_ticket = engine
            .core
            .issue_ticket(caller_tl, receipt)
            .expect("issue prep ticket");
        proof {
            offsets = offsets.push(i * block_len);
            gathered_bytes = gathered_bytes.add(data@);
            let h_cert = ValueCertificate {
                slot: h_cuda.slot_ref(),
                writer: w_ev,
                bytes: data@,
                proof: ValueProof::HostWrite,
            };
            let g_cert = ValueCertificate {
                slot: g_slice.slot_ref(),
                writer: copy_ev,
                bytes: data@,
                proof: ValueProof::ReplicaFrom {
                    source: Box::new(h_cert),
                    source_slot: h_cuda.slot_ref(),
                },
            };
            src_certs = src_certs.push(g_cert);
            src_perms.tracked_insert(i as int, ReadPerm::Owned(g_tok));
        }
        src_gpu_bufs.push(g_buf);
        // Allocate a small 4KB spacer to force non-contiguous device addresses.
        let (sp, _sp_tok) =
            cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), dev, 4096).expect("spacer");
        _spacers.push(sp);
        i += 1;
    }
    proof {
        offsets = offsets.push(frame_size);
    }
    let Tracked(k_prep) = cuda::stream_synchronize(&caller_stream, Tracked(caps.instance.borrow()))
        .expect("sync prep stream");
    if last_ticket != Ticket::NONE {
        engine
            .core
            .mark_timeline_complete(caller_tl, last_ticket.seq(), Tracked(k_prep));
    }

    let mut src_slices: Vec<CudaSlice<'_>> = Vec::with_capacity(num_blocks);
    let mut s_idx = 0usize;
    while s_idx < src_gpu_bufs.len() {
        src_slices.push(src_gpu_bufs[s_idx].slot_slice(0));
        s_idx += 1;
    }

    let host_tier = engine.host_tier();
    let gpu_tier = engine.gpu_tier().unwrap();
    let total_iters = job.warmup_iters + job.iters;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();

    let mut it = 0usize;
    while it < total_iters {
        if it == job.warmup_iters {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let key = ObjectKey([1, it as u64, frame_size as u64, 0]);

        let t0 = Instant::now();
        let (_ver, gpu_f, _t_sc, _sp) = engine
            .put_gpu_scatter(
                Tracked(caps.instance.borrow()),
                key,
                &src_slices,
                Tracked(&mut src_perms),
                Ghost(src_certs),
                Ghost(offsets),
                Ghost(gathered_bytes),
            )
            .expect("put_gpu_scatter");
        engine
            .synchronize_cuda(Tracked(caps.instance.borrow()))
            .expect("sync after put_gpu_scatter");

        // D2H hop from gathered GPU VRAM frame -> Pinned Host Frame + CudaEvent sync.
        let (host_f, _t_hop, _sp_hop) = {
            let cuda_b = engine.cuda.as_mut().unwrap();
            let res = cuda_b
                .copy_hop_cuda(
                    &mut engine.core,
                    Tracked(caps.instance.borrow()),
                    Tracked(caps.shares.borrow_mut()),
                    gpu_f,
                    host_tier,
                )
                .expect("copy_hop_cuda GPU->PinnedHost");
            cuda::event_record(&mut done_event, &cuda_b.timeline.stream).expect("event_record");
            res
        };
        engine
            .synchronize_cuda(Tracked(caps.instance.borrow()))
            .expect("sync after D2H copy_hop_cuda");
        let dt_ns = t0.elapsed().as_nanos() as u64;

        engine.evict_frame(gpu_f).expect("evict gpu_f");
        engine.evict_frame(host_f).expect("evict host_f");
        let r1 = engine.reclaim_draining(gpu_tier, 8);
        let r2 = engine.reclaim_draining(host_tier, 8);
        assert!(r1 + r2 == 2, "reclaim both gpu and pinned host frames");

        if it >= job.warmup_iters {
            hist.record_ns(dt_ns);
        }
        it += 1;
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let _ = done_event.destroy();
    let _ = caller_stream.destroy();
    (elapsed_s, sampler, hist)
}

/// 1A-H2D: Pinned Host Frame -> 16 non-contiguous GPU Paged KV blocks via
/// 1-hop `LocalEngine::get_into_gpu_scatter` + `CudaEvent`.
fn run_1a_host_to_gpu_scatter_inner(
    job: &JobSpec,
    num_blocks: usize,
    block_len: usize,
) -> (f64, ProcSampler, LatencyHist) {
    let dev: DeviceId = job.donor_gpu;
    let frame_size = job.chunk_bytes;

    let mut caps = ShardCapabilityBundle::take(0);
    let mut core = caps.make_local_core(frame_size, 64);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        4,
        4096,
    )
    .expect("HostBackend::new");

    let mut cuda_backend = CudaBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_cuda(),
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
    let mut caller_stream = CudaStream::new(dev, caps.mint_cuda()).expect("CudaStream::new");
    let mut done_event =
        CudaEvent::new(dev, Ghost(0), Tracked(caps.instance.borrow())).expect("CudaEvent::new");

    let eng_ctx = HostContext::new(caps.mint_host());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, Some(cuda_backend), None);

    // Publish a frame in Pinned Host DRAM (NOT in GPU VRAM, so get_into_gpu_scatter executes
    // a direct 1-hop Pinned Host Frame -> 16 non-contiguous GPU blocks H2D scatter!).
    let payload = fill_pattern(frame_size, 0x5A);
    let key = ObjectKey([2, frame_size as u64, 0, 0]);
    let (_ver, _host_f) = engine
        .put_host(Tracked(caps.instance.borrow()), key, &payload)
        .expect("put_host pinned frame");

    // Allocate 16 non-contiguous destination GPU blocks.
    let mut dst_gpu_bufs = Vec::with_capacity(num_blocks);
    let mut _spacers = Vec::with_capacity(num_blocks);
    let tracked mut dst_toks: Map<int, SlotToken> = Map::tracked_empty();
    let ghost mut offsets: Seq<usize> = Seq::empty();

    let mut i = 0usize;
    while i < num_blocks {
        let (g_buf, Tracked(g_tok)) =
            cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), dev, block_len)
                .expect("alloc_device dst");
        proof {
            offsets = offsets.push(i * block_len);
            dst_toks.tracked_insert(i as int, g_tok);
        }
        dst_gpu_bufs.push(g_buf);
        let (sp, _sp_tok) =
            cuda::alloc_device(Tracked(caps.alloc.borrow_mut()), dev, 4096).expect("spacer");
        _spacers.push(sp);
        i += 1;
    }
    proof {
        offsets = offsets.push(frame_size);
    }

    let mut dst_slices: Vec<CudaSlice<'_>> = Vec::with_capacity(num_blocks);
    let mut d_idx = 0usize;
    while d_idx < dst_gpu_bufs.len() {
        dst_slices.push(dst_gpu_bufs[d_idx].slot_slice(0));
        d_idx += 1;
    }

    let total_iters = job.warmup_iters + job.iters;
    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();

    let mut it = 0usize;
    while it < total_iters {
        if it == job.warmup_iters {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let (_v, t_out) = engine
            .get_into_gpu_scatter(
                Tracked(caps.instance.borrow()),
                Tracked(caps.shares.borrow_mut()),
                key,
                &mut caller_stream,
                caller_tl,
                &dst_slices,
                Tracked(&mut dst_toks),
                Ghost(offsets),
            )
            .expect("get_into_gpu_scatter")
            .expect("pinned host hit");
        cuda::event_record(&mut done_event, &caller_stream).expect("event_record");
        let Tracked(k_out) = cuda::stream_synchronize(&caller_stream, Tracked(caps.instance.borrow()))
            .expect("stream_synchronize");
        engine
            .core
            .mark_timeline_complete(caller_tl, t_out.seq(), Tracked(k_out));
        let dt_ns = t0.elapsed().as_nanos() as u64;

        if it >= job.warmup_iters {
            hist.record_ns(dt_ns);
        }
        it += 1;
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    let _ = done_event.destroy();
    let _ = caller_stream.destroy();
    (elapsed_s, sampler, hist)
}

/// 1B-Write: `NvmeBackend` (`O_DIRECT` + `io_uring` in `/workspace/target/bench-nvme`) write spill.
fn run_1b_nvme_write_spill_inner(
    job: &JobSpec,
    qd: usize,
    ring_entries: u32,
) -> (f64, ProcSampler, LatencyHist, u64) {
    let frame_size = job.chunk_bytes;
    let pool_frames = (qd + 2) as u32;

    let mut caps = ShardCapabilityBundle::take(0);
    let mut core = caps.make_local_core(frame_size, 128);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        pool_frames,
        4096,
    )
    .expect("HostBackend::new");

    let nvme = NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_nvme(),
        &job.nvme_dir,
        pool_frames,
        ring_entries,
    )
    .expect("NvmeBackend::new");
    let eng_ctx = HostContext::new(caps.mint_host());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, Some(nvme));
    let nvme_tier = engine.nvme_tier().unwrap();

    // Pre-populate `qd` Host DRAM frames with deterministic payloads.
    let mut host_frames = Vec::with_capacity(qd);
    let mut keys = Vec::with_capacity(qd);
    let mut i = 0usize;
    while i < qd {
        let key = ObjectKey([10, i as u64, frame_size as u64, 0]);
        let data = fill_pattern(frame_size, (i as u8).wrapping_mul(31).wrapping_add(7));
        let (_v, hf) = engine
            .put_host(Tracked(caps.instance.borrow()), key, &data)
            .expect("put_host seed");
        keys.push(key);
        host_frames.push(hf);
        i += 1;
    }

    let warmup_batches = (job.warmup_iters + qd - 1) / qd;
    let meas_batches = (job.iters + qd - 1) / qd;
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();
    let mut total_ops = 0u64;

    let mut b = 0usize;
    while b < total_batches {
        if b == warmup_batches {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let mut nvme_frames = Vec::with_capacity(qd);
        if qd == 1 {
            let (nf, ticket) = engine
                .tier_dram_to_nvme(
                    Tracked(caps.instance.borrow()),
                    Tracked(caps.shares.borrow_mut()),
                    keys[0],
                )
                .expect("tier_dram_to_nvme");
            let prog = engine
                .wait_nvme_ticket(Tracked(caps.instance.borrow()), ticket)
                .expect("wait_nvme_ticket");
            assert!(prog.failed == 0);
            nvme_frames.push(nf);
        } else {
            let mut nvme_b = engine.nvme.take().unwrap();
            let mut last_ticket = Ticket::NONE;
            let mut h_idx = 0usize;
            while h_idx < host_frames.len() {
                let hf = host_frames[h_idx];
                let (nf, ticket) = nvme_b
                    .stage_dram_to_nvme(
                        engine.host,
                        &mut engine.core,
                        Tracked(caps.instance.borrow()),
                        Tracked(caps.shares.borrow_mut()),
                        hf,
                    )
                    .expect("stage_dram_to_nvme");
                nvme_frames.push(nf);
                last_ticket = ticket;
                h_idx += 1;
            }
            nvme_b
                .submit(Ghost(&engine.core))
                .expect("nvme batch submit");
            engine.nvme = Some(nvme_b);
            let prog = engine
                .wait_nvme_ticket(Tracked(caps.instance.borrow()), last_ticket)
                .expect("wait_nvme_ticket batch");
            assert!(prog.failed == 0);
        }
        let dt_ns = t0.elapsed().as_nanos() as u64;

        let mut f_idx = 0usize;
        while f_idx < nvme_frames.len() {
            engine.evict_frame(nvme_frames[f_idx]).expect("evict nvme frame");
            f_idx += 1;
        }
        let reclaimed = engine.reclaim_draining(nvme_tier, qd + 4);
        assert!(reclaimed == qd, "reclaim all nvme spill frames");

        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += qd as u64;
        }
        b += 1;
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    (elapsed_s, sampler, hist, total_ops)
}

/// 1B-Read: `NvmeBackend` (`O_DIRECT` + `io_uring` in `/workspace/target/bench-nvme`) cold read.
fn run_1b_nvme_read_cold_inner(
    job: &JobSpec,
    qd: usize,
    ring_entries: u32,
) -> (f64, ProcSampler, LatencyHist, u64) {
    let frame_size = job.chunk_bytes;
    let pool_frames = (qd + 2) as u32;

    let mut caps = ShardCapabilityBundle::take(0);
    let mut core = caps.make_local_core(frame_size, 128);
    let host = HostBackend::new(
        &mut core,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_host(),
        pool_frames,
        4096,
    )
    .expect("HostBackend::new");

    let nvme = NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked(caps.instance.borrow()),
        Tracked(caps.alloc.borrow_mut()),
        caps.mint_nvme(),
        &job.nvme_dir,
        pool_frames,
        ring_entries,
    )
    .expect("NvmeBackend::new");
    let eng_ctx = HostContext::new(caps.mint_host());
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, Some(nvme));
    let host_tier = engine.host_tier();

    // Seed `qd` frames onto NVMe, then evict their DRAM copies so they live ONLY on NVMe.
    let mut nvme_frames = Vec::with_capacity(qd);
    let mut keys = Vec::with_capacity(qd);
    let mut i = 0usize;
    while i < qd {
        let key = ObjectKey([11, i as u64, frame_size as u64, 0]);
        let data = fill_pattern(frame_size, (i as u8).wrapping_mul(19).wrapping_add(11));
        let (_v, hf) = engine
            .put_host(Tracked(caps.instance.borrow()), key, &data)
            .expect("put_host seed");
        let (nf, ticket) = engine
            .tier_dram_to_nvme(
                Tracked(caps.instance.borrow()),
                Tracked(caps.shares.borrow_mut()),
                key,
            )
            .expect("tier_dram_to_nvme seed");
        let prog = engine
            .wait_nvme_ticket(Tracked(caps.instance.borrow()), ticket)
            .expect("wait seed spill");
        assert!(prog.failed == 0);
        engine.evict_frame(hf).expect("evict seed dram");
        assert!(engine.reclaim_draining(host_tier, 4) == 1);
        keys.push(key);
        nvme_frames.push(nf);
        i += 1;
    }

    let warmup_batches = (job.warmup_iters + qd - 1) / qd;
    let meas_batches = (job.iters + qd - 1) / qd;
    let total_batches = warmup_batches + meas_batches;

    let mut hist = LatencyHist::new();
    let mut sampler = ProcSampler::start();
    let mut measure_start = Instant::now();
    let mut total_ops = 0u64;

    let mut b = 0usize;
    while b < total_batches {
        if b == warmup_batches {
            sampler = ProcSampler::start();
            measure_start = Instant::now();
        }
        let t0 = Instant::now();
        let mut dram_frames = Vec::with_capacity(qd);
        if qd == 1 {
            let (df, ticket) = engine
                .warm_host_from_nvme(
                    Tracked(caps.instance.borrow()),
                    Tracked(caps.shares.borrow_mut()),
                    keys[0],
                )
                .expect("warm_host_from_nvme");
            let prog = engine
                .wait_nvme_ticket(Tracked(caps.instance.borrow()), ticket)
                .expect("wait_nvme_ticket read");
            assert!(prog.failed == 0);
            dram_frames.push(df);
        } else {
            let mut nvme_b = engine.nvme.take().unwrap();
            let mut last_ticket = Ticket::NONE;
            let mut n_idx = 0usize;
            while n_idx < nvme_frames.len() {
                let nf = nvme_frames[n_idx];
                let (df, ticket) = nvme_b
                    .stage_nvme_to_dram(
                        engine.host,
                        &mut engine.core,
                        Tracked(caps.instance.borrow()),
                        Tracked(caps.shares.borrow_mut()),
                        nf,
                    )
                    .expect("stage_nvme_to_dram");
                dram_frames.push(df);
                last_ticket = ticket;
                n_idx += 1;
            }
            nvme_b
                .submit(Ghost(&engine.core))
                .expect("nvme batch read submit");
            engine.nvme = Some(nvme_b);
            let prog = engine
                .wait_nvme_ticket(Tracked(caps.instance.borrow()), last_ticket)
                .expect("wait_nvme_ticket batch read");
            assert!(prog.failed == 0);
        }
        let dt_ns = t0.elapsed().as_nanos() as u64;

        let mut d_idx = 0usize;
        while d_idx < dram_frames.len() {
            engine.evict_frame(dram_frames[d_idx]).expect("evict warmed dram");
            d_idx += 1;
        }
        let reclaimed = engine.reclaim_draining(host_tier, qd + 4);
        assert!(reclaimed == qd, "reclaim all warmed dram frames");

        if b >= warmup_batches {
            hist.record_ns(dt_ns);
            total_ops += qd as u64;
        }
        b += 1;
    }

    let elapsed_s = measure_start.elapsed().as_secs_f64();
    (elapsed_s, sampler, hist, total_ops)
}

} // verus!

fn run_1a_gpu_to_host_scatter(job: &JobSpec) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let num_blocks = job.num_blocks.max(1);
    assert!(
        frame_size % num_blocks == 0,
        "chunk_bytes ({frame_size}) must be divisible by num_blocks ({num_blocks})"
    );
    let block_len = frame_size / num_blocks;
    let (elapsed_s, sampler, hist) = run_1a_gpu_to_host_scatter_inner(job, num_blocks, block_len);
    finish_local_metric(
        job,
        elapsed_s,
        sampler,
        &hist,
        job.iters as u64,
        frame_size,
        format!(
            "CudaBackend::put_gpu_scatter ({num_blocks} non-contiguous GPU blocks -> GPU frame) + copy_hop_cuda D2H -> Pinned Host Frame + CudaEvent sync"
        ),
    )
}

fn run_1a_host_to_gpu_scatter(job: &JobSpec) -> MetricRecord {
    let frame_size = job.chunk_bytes;
    let num_blocks = job.num_blocks.max(1);
    assert!(
        frame_size % num_blocks == 0,
        "chunk_bytes ({frame_size}) must be divisible by num_blocks ({num_blocks})"
    );
    let block_len = frame_size / num_blocks;
    let (elapsed_s, sampler, hist) = run_1a_host_to_gpu_scatter_inner(job, num_blocks, block_len);
    finish_local_metric(
        job,
        elapsed_s,
        sampler,
        &hist,
        job.iters as u64,
        frame_size,
        format!(
            "LocalEngine::get_into_gpu_scatter 1-hop Pinned Host Frame -> {num_blocks} non-contiguous GPU blocks + CudaEvent sync"
        ),
    )
}

fn run_1b_nvme_write_spill(job: &JobSpec) -> MetricRecord {
    fs::create_dir_all(&job.nvme_dir).expect("create nvme_dir");
    let qd = job.concurrency.max(1).min(32);
    let ring_entries = (qd * 4).next_power_of_two().max(16) as u32;
    let (elapsed_s, sampler, hist, total_ops) =
        run_1b_nvme_write_spill_inner(job, qd, ring_entries);
    finish_local_metric(
        job,
        elapsed_s,
        sampler,
        &hist,
        total_ops,
        job.chunk_bytes,
        format!(
            "NvmeBackend O_DIRECT + io_uring write spill (QD={qd}, dir={})",
            job.nvme_dir
        ),
    )
}

fn run_1b_nvme_read_cold(job: &JobSpec) -> MetricRecord {
    fs::create_dir_all(&job.nvme_dir).expect("create nvme_dir");
    let qd = job.concurrency.max(1).min(32);
    let ring_entries = (qd * 4).next_power_of_two().max(16) as u32;
    let (elapsed_s, sampler, hist, total_ops) = run_1b_nvme_read_cold_inner(job, qd, ring_entries);
    finish_local_metric(
        job,
        elapsed_s,
        sampler,
        &hist,
        total_ops,
        job.chunk_bytes,
        format!(
            "NvmeBackend O_DIRECT + io_uring cold read (QD={qd}, dir={})",
            job.nvme_dir
        ),
    )
}
