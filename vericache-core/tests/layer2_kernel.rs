use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use transfer::domain::DeviceId;
#[cfg(verus_only)]
use transfer::domain::*;
use transfer::{context::*, cuda, host, rdma, transport::*};
use vericache_core::backends::cuda::{CudaBackend, CudaTimeline};
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::backends::rdma::{PeerCompletion, RdmaBackend};
use vericache_core::frame::{FLAG_READ_OVERFLOW, STATE_DRAINING, STATE_EMPTY, STATE_PUBLISHED};
use vericache_core::kernel::CacheCore;
#[cfg(verus_only)]
use vericache_core::provenance::*;
use vericache_core::spec::*;
use vericache_core::timeline::*;
use vstd::prelude::*;

const FRAME_SIZE: usize = 4096;

#[cfg_attr(not(verus_only), allow(dead_code))]
struct Sources {
    instance: Tracked<TransferInstance>,
    allocations: Tracked<AllocSource>,
    events: Tracked<EventSourceAllocator>,
    shares: Tracked<ShareSource>,
}

fn sources() -> std::sync::MutexGuard<'static, Sources> {
    static SOURCES: OnceLock<Mutex<Sources>> = OnceLock::new();
    SOURCES
        .get_or_init(|| {
            let (instance, allocations, events, shares) = NodeContext::bootstrap(0).into_sources();
            Mutex::new(Sources {
                instance,
                allocations,
                events,
                shares,
            })
        })
        .lock()
        .unwrap()
}

fn require_hw() -> bool {
    std::env::var_os("VERICACHE_REQUIRE_HW").is_some()
}

fn pattern(seed: u8) -> Vec<u8> {
    (0..FRAME_SIZE)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

fn has_cuda_device() -> bool {
    match cuda::device_count() {
        Ok(n) if n > 0 => true,
        _ => false,
    }
}

verus! {

fn run_custom_tier_and_frame_table_lifecycle(_src: &mut Sources) {
    let custom_tier = TierId(0);
    let custom_desc = TierDescriptor {
        id: custom_tier,
        kind: TierKind::Custom(77),
        device_id: None,
        host_accessible: true,
        rdma_capable: false,
    };
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_source = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host_ctx = host::HostContext::new(Tracked(host_source));
    let tl = core.register_timeline(TimelineKind::Custom(77)).unwrap();

    let (slab, tokens) = host::alloc_slab(
        &host_ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        2,
        FRAME_SIZE,
        4096,
    ).unwrap();
    let ghost slots = Seq::new(2nat, |i: int| slab.slot_ref(i as nat));
    let (tier_id, first) = core.register_tier(custom_desc, Ghost(slots), 2, tokens).unwrap();
    assert!(tier_id == TierId(0) && first == FrameId(0));
    assert!(core.frames.ready_queue[0].len() == 2);

    // Reserve then abort returns the frame directly to the ready queue.
    let k_abort = ObjectKey([10, 11, 12, 13]);
    let (h_abort, tok_abort) = core.begin_write(tier_id, k_abort).unwrap();
    assert!(core.frames.ready_queue[0].len() == 1);
    core.abort_write(h_abort, tok_abort);
    assert!(core.frames.ready_queue[0].len() == 2);

    // Reserve, write, publish, lookup, evict, and reclaim.
    let key = ObjectKey([1, 2, 3, 4]);
    let (handle, Tracked(mut tok)) = core.begin_write(tier_id, key).unwrap();
    let slice = slab.slot_slice(handle.frame.0 as usize);
    let data = pattern(0x33);
    let Tracked(comp) = host::write_slice(
        &mut host_ctx,
        Tracked(_src.instance.borrow()),
        &slice,
        &data,
        Tracked(&mut tok),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    let ticket = core.issue_ticket(tl, Tracked(comp.submitted)).unwrap();
    core.mark_timeline_complete(tl, ticket.seq(), Tracked(comp.knows));
    let frame = handle.frame;
    let ver = handle.version;
    core.finish_write(
        handle,
        Tracked(tok),
        ticket,
        Ghost(
            ValueCertificate {
                slot: slice.slot_ref(),
                writer: core.timelines.events@[(tl, ticket.sequence())],
                bytes: data@,
                proof: ValueProof::HostWrite,
            },
        ),
    );
    assert!(core.frames.metas[frame.0 as usize].state == STATE_PUBLISHED);
    assert!(core.lookup_tier(key, tier_id) == Some((frame, ver)));

    core.evict_frame(frame).unwrap();
    assert!(core.frames.metas[frame.0 as usize].state == STATE_DRAINING);
    assert!(core.lookup_tier(key, tier_id).is_none());
    let reclaimed = core.reclaim_draining(tier_id, 4);
    assert!(reclaimed == 1);
    assert!(core.frames.metas[frame.0 as usize].state == STATE_EMPTY);
    assert!(core.frames.ready_queue[0].len() == 2);
}

fn run_inline_read_compaction_and_overflow_fallback(_src: &mut Sources) {
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 8, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();

    // Register 5 additional host timelines to saturate the 4 inline read slots and trigger overflow.
    let mut tls = Vec::new();
    let mut ctxs = Vec::new();
    for _ in 0..5 {
        let tl = core.register_timeline(TimelineKind::Host).unwrap();
        let tracked ev_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
        tls.push(tl);
        ctxs.push(host::HostContext::new(Tracked(ev_src)));
    }

    let key = ObjectKey([20, 21, 22, 23]);
    let data = pattern(0x41);
    let (_ver, frame) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        key,
        &data,
    ).unwrap();

    // Perform 5 reads on 5 distinct timelines WITHOUT marking them complete first.
    let mut tickets = Vec::new();
    let mut knows_list = Vec::new();
    for idx in 0..5usize {
        let (lease, _v, _cert, Tracked(cap)) = core.acquire_local_lease(
            frame,
            false,
            Tracked(_src.shares.borrow_mut()),
            Tracked(_src.instance.borrow()),
        ).unwrap();
        let tracked mut perm = ReadPerm::Shared(cap);
        let slice = host.frame_slice(frame);
        let mut out = vec![0u8; FRAME_SIZE];
        let Tracked(comp) = host::read_slice(
            &mut ctxs[idx],
            Tracked(_src.instance.borrow()),
            &slice,
            &mut out,
            Tracked(&mut perm),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Ghost(core.citations()),
        );
        assert!(out == data);
        let tracked cap_back = perm.into_shared();
        let t = core.issue_ticket(tls[idx], Tracked(comp.submitted)).unwrap();
        core.release_local_lease(
            lease,
            Tracked(cap_back),
            t,
            Tracked(_src.instance.borrow()),
        ).unwrap();
        tickets.push(t);
        knows_list.push(Tracked(comp.knows));
    }

    // 5 distinct active timelines must have triggered FLAG_READ_OVERFLOW.
    assert!((core.frames.metas[frame.0 as usize].flags & FLAG_READ_OVERFLOW) != 0);

    core.evict_frame(frame).unwrap();
    // Cannot reclaim while any read timeline is still unconfirmed.
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);
    assert!(core.frames.metas[frame.0 as usize].state == STATE_DRAINING);

    // Confirm first 4 timelines; the 5th is still pending so reclamation remains blocked.
    for idx in 0..4usize {
        let Tracked(k) = knows_list[idx];
        core.mark_timeline_complete(tls[idx], tickets[idx].seq(), Tracked(k));
    }
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);

    // Confirm the 5th timeline; now draining frame reclaims cleanly and resets overflow flag.
    let Tracked(k4) = knows_list[4];
    core.mark_timeline_complete(tls[4], tickets[4].seq(), Tracked(k4));
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);
    assert!(core.frames.metas[frame.0 as usize].state == STATE_EMPTY);
    assert!((core.frames.metas[frame.0 as usize].flags & FLAG_READ_OVERFLOW) == 0);
}

fn run_concurrent_local_and_remote_leases_overflow_and_eviction(_src: &mut Sources) {
    // Cap max_local_leases at 2 so we test lease table overflow explicitly.
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 2, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();

    let key = ObjectKey([30, 31, 32, 33]);
    let data = pattern(0x77);
    let (ver, frame) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        key,
        &data,
    ).unwrap();

    // Acquire 2 concurrent local leases + 1 concurrent remote lease on the same frame.
    let (l1, v1, _c1, cap1) = core.acquire_local_lease(
        frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    let (l2, v2, _c2, cap2) = core.acquire_local_lease(
        frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(v1 == ver && v2 == ver);

    // 3rd local lease hits lease table overflow.
    let extra_lease = core.acquire_local_lease(
        frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    );
    assert!(extra_lease.is_err());

    // Remote lease still succeeds independently of the local lease table.
    let (rv, _rc, Tracked(rcap)) = core.pin_remote_lease(
        frame,
        0,
        1,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(rv == ver);
    assert!(core.frames.metas[frame.0 as usize].active_local_reads == 2);
    assert!(core.frames.metas[frame.0 as usize].active_remote_leases == 1);

    // Evicting while leased transitions to Draining, hides from catalog, and blocks reclamation.
    core.evict_frame(frame).unwrap();
    assert!(core.lookup_tier(key, host.tier_id()).is_none());
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);

    // Release local lease #2 first (out of order), still blocked by local #1 and remote lease.
    core.release_local_lease(l2, cap2, Ticket::NONE, Tracked(_src.instance.borrow())).unwrap();
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);

    // Release remote lease cooperatively; still blocked by local #1.
    let tracked returned = rcap.return_share();
    core.unpin_remote_lease(
        frame,
        0,
        1,
        Tracked(returned),
        Ticket::NONE,
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(core.frames.metas[frame.0 as usize].active_remote_leases == 0);
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);

    // Release local lease #1; now frame reclaims cleanly and is reusable.
    core.release_local_lease(l1, cap1, Ticket::NONE, Tracked(_src.instance.borrow())).unwrap();
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);
    assert!(core.frames.metas[frame.0 as usize].state == STATE_EMPTY);
}

fn run_host_backend_put_read_and_copy_tier(_src: &mut Sources) {
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked src0 = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let tracked src1 = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let tracked src_shared = _src.events.borrow_mut().mint(EventSourceKind::Host);

    let mut host0 = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(src0),
        2,
        4096,
    ).unwrap();
    let host1 = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(src1),
        2,
        4096,
    ).unwrap();
    let mut shared_ctx = host::HostContext::new(Tracked(src_shared));

    let k0 = ObjectKey([40, 41, 42, 43]);
    let k1 = ObjectKey([44, 45, 46, 47]);
    let p0 = pattern(0x12);
    let p1 = pattern(0x9E);

    let (v0, f0) = host0.put_host(&mut core, Tracked(_src.instance.borrow()), k0, &p0).unwrap();
    let (v1, f1) = host0.put_host_with(
        &mut shared_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        k1,
        &p1,
    ).unwrap();

    let mut buf0 = vec![0u8; FRAME_SIZE];
    let mut buf1 = vec![0u8; FRAME_SIZE];
    let rv0 = host0.read_frame_into(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f0,
        &mut buf0,
    ).unwrap();
    let rv1 = host0.read_frame_into_with(
        &mut shared_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f1,
        &mut buf1,
    ).unwrap();
    assert!(rv0 == v0 && buf0 == p0);
    assert!(rv1 == v1 && buf1 == p1);

    // Copy k0 from host0 tier -> host1 tier and verify exact bytes from host1.
    let f_dst = host0.copy_frame_to_host_tier(
        &host1,
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f0,
    ).unwrap();
    let mut buf_copy = vec![0u8; FRAME_SIZE];
    let rvc = host1.read_frame_into_with(
        &mut shared_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f_dst,
        &mut buf_copy,
    ).unwrap();
    assert!(rvc == v0 && buf_copy == p0);
}

fn run_nvme_backend_io_uring_round_trip(_src: &mut Sources) {
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();

    let tracked ring_src = _src.events.borrow_mut().mint(EventSourceKind::IoUring);
    let mut nvme = match NvmeBackend::new(
        &mut core,
        &host.ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(ring_src),
        "/tmp",
        2,
        16,
    ) {
        Ok(n) => n,
        Err(CoreError::Backend { kind: BACKEND_NVME, code: errno }) => {
            if require_hw() {
                panic!("VERICACHE_REQUIRE_HW set, but NvmeBackend::new failed with errno {errno}");
            }
            return;
        },
        Err(e) => panic!("unexpected NvmeBackend::new error: {e:?}"),
    };

    let key = ObjectKey([50, 51, 52, 53]);
    let data = pattern(0xC5);
    let (ver, dram_f) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        key,
        &data,
    ).unwrap();

    // Stage and submit DRAM -> NVMe.
    let (nvme_f, t_down) = nvme.stage_dram_to_nvme(
        &host,
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        dram_f,
    ).unwrap();
    assert!(nvme.staged_len() == 1);
    let submitted = nvme.submit(Ghost(&core)).unwrap();
    assert!(submitted == 1);

    let prog = nvme.wait_one_completion(
        &mut core,
        Tracked(_src.instance.borrow()),
        vericache_core::backends::nvme::NvmeProgress::empty(),
    ).unwrap();
    assert!(prog.completed == 1 && prog.failed == 0);
    assert!(core.timelines.is_retired(t_down));
    assert!(core.lookup_tier(key, nvme.tier_id()) == Some((nvme_f, ver)));

    // Evict DRAM replica, reclaim it, overwrite DRAM with different key, then evict again.
    core.evict_frame(dram_f).unwrap();
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);
    let noise_key = ObjectKey([54, 55, 56, 57]);
    let noise = pattern(0x11);
    let (_, noise_f) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        noise_key,
        &noise,
    ).unwrap();
    core.evict_frame(noise_f).unwrap();
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);

    // Stage and submit NVMe -> DRAM warm-up.
    let (warm_f, t_up) = nvme.stage_nvme_to_dram(
        &host,
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        nvme_f,
    ).unwrap();
    let sub_up = nvme.submit(Ghost(&core)).unwrap();
    assert!(sub_up == 1);
    let mut total_completed = 0usize;
    while !core.timelines.is_retired(t_up) {
        let p = nvme.poll_completions(&mut core, Tracked(_src.instance.borrow())).unwrap();
        total_completed += p.completed;
        assert!(p.failed == 0);
        if !core.timelines.is_retired(t_up) {
            std::thread::yield_now();
        }
    }
    assert!(total_completed == 1);
    assert!(core.lookup_tier(key, host.tier_id()) == Some((warm_f, ver)));

    let mut out = vec![0u8; FRAME_SIZE];
    let rv = host.read_frame_into(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        warm_f,
        &mut out,
    ).unwrap();
    assert!(rv == ver && out == data);
}

fn run_cuda_wait_event_not_ready_before_poll(_src: &mut Sources) {
    if !has_cuda_device() {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but no usable CUDA device or driver");
        }
        return;
    }
    let dev: DeviceId = 0;
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked s_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut tl = CudaTimeline::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        dev,
        Tracked(s_src),
        8,
    ).unwrap();

    // Record a CUDA event on an idle stream. Even though the GPU may complete it
    // almost instantaneously, M2 forbids `is_sync_point_ready` from returning `true`
    // before `poll_events` explicitly observes `EventStatus::Ready` and retires the slot.
    let sp = tl.record_event(&core).unwrap();
    assert!(!tl.is_sync_point_ready(&core, sp));

    // Synchronize stream and poll events; now the slot is retired and ready.
    tl.synchronize(&mut core, Tracked(_src.instance.borrow())).unwrap();
    let completed = tl.poll_events(&mut core, Tracked(_src.instance.borrow())).unwrap();
    assert!(completed >= 1);
    assert!(tl.is_sync_point_ready(&core, sp));
}

fn run_cuda_backend_ordered_handoff_and_section_5_3_hazard(_src: &mut Sources) {
    if !has_cuda_device() {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but no usable CUDA device or driver");
        }
        return;
    }
    let dev: DeviceId = 0;
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();
    let tracked host_op_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host_op_ctx = host::HostContext::new(Tracked(host_op_src));
    let tracked gpu_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut gpu = CudaBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(gpu_src),
        dev,
        2,
        16,
        false,
    ).unwrap();
    gpu.attach_pinned_host(&core, &host).unwrap();

    // Create a second independent caller CUDA stream + timeline for Ordered Handoff.
    let tracked caller_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut caller_stream = cuda::CudaStream::new(dev, Tracked(caller_src)).unwrap();
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: dev }).unwrap();

    // Put data into pinned host DRAM (using shared `&host` + disjoint `host_op_ctx`).
    let key = ObjectKey([60, 61, 62, 63]);
    let data = pattern(0xA5);
    let (ver, dram_f) = host.put_host_with(
        &mut host_op_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        key,
        &data,
    ).unwrap();

    // Ordered Handoff H2D from pinned host frame into a caller GPU buffer on caller_stream.
    let (dev_buf, Tracked(mut dev_tok)) = cuda::alloc_device(
        Tracked(_src.allocations.borrow_mut()),
        dev,
        FRAME_SIZE,
    ).unwrap();
    let dev_slice = dev_buf.slot_slice(0);
    let (v_got, t_read, _ev) = gpu.get_into_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        dram_f,
        &mut caller_stream,
        caller_tl,
        &dev_slice,
        Tracked(&mut dev_tok),
    ).unwrap();
    assert!(v_got == ver);

    // §5.3 hazard: immediately evict `dram_f` BEFORE confirming `caller_stream`'s read ticket!
    // The frame MUST remain Draining and MUST NOT be reclaimed until caller_tl confirms `t_read`.
    core.evict_frame(dram_f).unwrap();
    assert!(core.frames.metas[dram_f.0 as usize].state == STATE_DRAINING);
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);

    // Synchronize caller stream and advance caller_tl confirmation watermark.
    let Tracked(knows) = cuda::stream_synchronize(
        &caller_stream,
        Tracked(_src.instance.borrow()),
    ).unwrap();
    core.mark_timeline_complete(caller_tl, t_read.seq(), Tracked(knows));

    // Now reclamation succeeds safely!
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);
    assert!(core.frames.metas[dram_f.0 as usize].state == STATE_EMPTY);

    // Copy dev_buf back into a fresh host slab via put_gpu -> pinned_host + synchronize + read
    // to verify exact end-to-end byte integrity on real GPU hardware.
    let key2 = ObjectKey([64, 65, 66, 67]);
    let tracked mut src_perm = ReadPerm::Owned(dev_tok);
    let ghost src_cert = ValueCertificate {
        slot: dev_slice.slot_ref(),
        writer: _ev@,
        bytes: data@,
        proof: ValueProof::HostWrite,
    };
    let (ver2, pinned_f, _t_put, _sp) = gpu.put_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        host.tier_id(),
        key2,
        &dev_slice,
        Tracked(&mut src_perm),
        Ghost(src_cert),
    ).unwrap();
    assert!(core.lookup_tier(key2, host.tier_id()).is_none());
    let poll_rep = gpu.synchronize(&mut core, Tracked(_src.instance.borrow())).unwrap();
    assert!(poll_rep.writes_published == 1);
    assert!(core.lookup_tier(key2, host.tier_id()) == Some((pinned_f, ver2)));

    let mut verify_buf = vec![0u8; FRAME_SIZE];
    let rv = host.read_frame_into_with(
        &mut host_op_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        pinned_f,
        &mut verify_buf,
    ).unwrap();
    assert!(rv == ver2 && verify_buf == data);
    let _ = caller_stream.destroy();
}

fn run_remote_lease_and_fenced_revocation_contract(_src: &mut Sources) {
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();

    let key = ObjectKey([70, 71, 72, 73]);
    let data = pattern(0x55);
    let (ver, frame) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        key,
        &data,
    ).unwrap();

    let (rv, _rc, Tracked(_rcap)) = core.pin_remote_lease(
        frame,
        3,
        9,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(rv == ver);
    assert!(core.frames.metas[frame.0 as usize].active_remote_leases == 1);

    // Evict while remotely pinned; reclamation is blocked until fenced revocation completes.
    core.evict_frame(frame).unwrap();
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);

    // Wrong (mw_slot, generation) is rejected cleanly.
    let ghost share = _rcap.share();
    let bad_rev = core.begin_fenced_revocation(frame, 3, 10, Ghost(share));
    assert!(bad_rev.is_err());

    // Begin fenced revocation with matching (mw_slot, generation), returning the lent SlotToken.
    // The frame stays pinned (`active_remote_leases == 1`) until `finish_fenced_revocation` runs
    // after the hardware fence completes.
    let pending = core.begin_fenced_revocation(frame, 3, 9, Ghost(share)).unwrap();
    assert!(pending.frame == frame && pending.slot == 3 && pending.generation == 9);
    assert!(core.frames.metas[frame.0 as usize].active_remote_leases == 1);
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);
}

fn run_cuda_one_slot_ring_readiness_bidirectional(_src: &mut Sources) {
    if !has_cuda_device() {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but no usable CUDA device or driver");
        }
        return;
    }
    let dev: DeviceId = 0;
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();
    let tracked host_op_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host_op_ctx = host::HostContext::new(Tracked(host_op_src));
    let tracked gpu_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    // 2 resident GPU frames sharing a 1-entry CUDA event ring!
    let mut gpu = CudaBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(gpu_src),
        dev,
        2,
        1,
        false,
    ).unwrap();

    let tracked caller_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut caller_stream = cuda::CudaStream::new(dev, Tracked(caller_src)).unwrap();
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: dev }).unwrap();

    let d1 = pattern(0x11);
    let d2 = pattern(0x22);
    let (_hv1, hf1) = host.put_host_with(
        &mut host_op_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        ObjectKey([80, 0, 0, 1]),
        &d1,
    ).unwrap();
    let (_hv2, hf2) = host.put_host_with(
        &mut host_op_ctx,
        &mut core,
        Tracked(_src.instance.borrow()),
        ObjectKey([80, 0, 0, 2]),
        &d2,
    ).unwrap();

    let (dev1, Tracked(mut tok1)) = cuda::alloc_device(
        Tracked(_src.allocations.borrow_mut()),
        dev,
        FRAME_SIZE,
    ).unwrap();
    let (dev2, Tracked(mut tok2)) = cuda::alloc_device(
        Tracked(_src.allocations.borrow_mut()),
        dev,
        FRAME_SIZE,
    ).unwrap();
    let (out_buf, Tracked(mut out_tok)) = cuda::alloc_device(
        Tracked(_src.allocations.borrow_mut()),
        dev,
        FRAME_SIZE,
    ).unwrap();
    let sl1 = dev1.slot_slice(0);
    let sl2 = dev2.slot_slice(0);
    let out_sl = out_buf.slot_slice(0);

    gpu.attach_pinned_host(&core, &host).unwrap();
    let (_, _, _ev1) = gpu.get_into_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        hf1,
        &mut caller_stream,
        caller_tl,
        &sl1,
        Tracked(&mut tok1),
    ).unwrap();
    let (_, _, _ev2) = gpu.get_into_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        hf2,
        &mut caller_stream,
        caller_tl,
        &sl2,
        Tracked(&mut tok2),
    ).unwrap();
    let _ = cuda::stream_synchronize(&caller_stream, Tracked(_src.instance.borrow())).unwrap();

    // (a) Put frame 1 on 1-entry event ring: must NOT be ready before confirmation.
    let tracked mut p1 = ReadPerm::Owned(tok1);
    let ghost c1 = ValueCertificate {
        slot: sl1.slot_ref(),
        writer: _ev1@,
        bytes: d1@,
        proof: ValueProof::HostWrite,
    };
    let (v1, f1, _, _) = gpu.put_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        gpu.gpu_tier_id(),
        ObjectKey([81, 0, 0, 1]),
        &sl1,
        Tracked(&mut p1),
        Ghost(c1),
    ).unwrap();
    assert!(!gpu.is_frame_ready(&core, f1));

    let rep1 = gpu.synchronize(&mut core, Tracked(_src.instance.borrow())).unwrap();
    assert!(rep1.writes_published == 1);
    assert!(gpu.is_frame_ready(&core, f1));

    // (b) Recycle the 1-entry event ring for frame 2; frame 1 MUST remain readable
    // via `get_into_gpu` without hitting `WouldConflict` on the recycled ring slot!
    let tracked mut p2 = ReadPerm::Owned(tok2);
    let ghost c2 = ValueCertificate {
        slot: sl2.slot_ref(),
        writer: _ev2@,
        bytes: d2@,
        proof: ValueProof::HostWrite,
    };
    let (_v2, f2, _, _) = gpu.put_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        gpu.gpu_tier_id(),
        ObjectKey([81, 0, 0, 2]),
        &sl2,
        Tracked(&mut p2),
        Ghost(c2),
    ).unwrap();
    assert!(f1 != f2);
    assert!(!gpu.is_frame_ready(&core, f2));

    let (got_v1, _, _) = gpu.get_into_gpu(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f1,
        &mut caller_stream,
        caller_tl,
        &out_sl,
        Tracked(&mut out_tok),
    ).unwrap();
    assert!(got_v1 == v1);
    let _ = cuda::stream_synchronize(&caller_stream, Tracked(_src.instance.borrow())).unwrap();
    let rep2 = gpu.synchronize(&mut core, Tracked(_src.instance.borrow())).unwrap();
    assert!(rep2.writes_published == 1);
    assert!(gpu.is_frame_ready(&core, f1) && gpu.is_frame_ready(&core, f2));
    let _ = caller_stream.destroy();
}

fn run_rdma_lease_handle_swap_remove_regression(
    _src: &mut Sources,
    dev_idx: u32,
    gid_idx: u8,
    t: TcpTransport,
) {
    let dev0 = rdma::DeviceContext::open(dev_idx, 0).unwrap();
    let tracked pd0_tok = _src.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair);
    let pd0 = rdma::ProtectionDomain::new(&dev0, Tracked(pd0_tok)).unwrap();
    let tracked cq0_tok = _src.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair);
    let cq0 = rdma::CompletionQueue::new(&dev0, 32, Tracked(cq0_tok)).unwrap();
    let tracked qp0_tok = _src.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair);
    let uqp0 = rdma::UnconnectedQueuePair::new(
        &pd0,
        &cq0,
        32,
        1,
        gid_idx,
        1,
        Tracked(qp0_tok),
    ).unwrap();
    let mut channel = Channel::new(t, 0, 1, Tracked(_src.instance.borrow()));
    let ep1 = channel.exchange_queue_pair_endpoint(
        &uqp0.endpoint(),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    let qp0 = match rdma::connect_rc(uqp0, ep1, 1) {
        Ok(q) => q,
        Err(e) => {
            panic!("connect_rc failed for qp0: {e:?}");
        },
    };

    let tracked mw0_tok = _src.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair);
    let tracked mw1_tok = _src.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair);
    let w0 = rdma::alloc_window(&pd0, Tracked(mw0_tok)).unwrap();
    let w1 = rdma::alloc_window(&pd0, Tracked(mw1_tok)).unwrap();
    let mut windows = Vec::new();
    windows.push(w0);
    windows.push(w1);

    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();
    let d0 = pattern(0x41);
    let d1 = pattern(0x42);
    let (_, f0) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        ObjectKey([21, 0, 0, 0]),
        &d0,
    ).unwrap();
    let (_, f1) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        ObjectKey([22, 0, 0, 0]),
        &d1,
    ).unwrap();

    let mut rdma = match RdmaBackend::new(&mut core, 1, qp0, cq0, windows) {
        Ok(b) => b,
        Err((e, _, _, _)) => panic!("RdmaBackend::new failed: {e:?}"),
    };
    rdma.register_host_slab(
        &core,
        &pd0,
        &host.buffer,
        host.tier,
        host.first_frame,
        host.num_frames,
        rdma::AccessFlags { local_write: true, mw_bind: true },
    ).unwrap();

    let (h0, _, _, wr0) = rdma.bind_remote_read_lease(
        &mut core,
        f0,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    let (h1, _, _, wr1) = rdma.bind_remote_read_lease(
        &mut core,
        f1,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(h0 != h1 && h0.mw_slot != h1.mw_slot);

    let mut bound0 = false;
    let mut bound1 = false;
    let mut polls: usize = 0;
    while (!bound0 || !bound1) && polls < 100_000 {
        match rdma.poll_cq(&mut core, Tracked(_src.instance.borrow())).unwrap() {
            Some(PeerCompletion::Control(rdma::Completed::WindowBound { wr_id, witness })) => {
                if wr_id == wr0 && !bound0 {
                    let _ = rdma.complete_bind(&core, h0, witness).unwrap();
                    bound0 = true;
                } else if wr_id == wr1 && !bound1 {
                    let _ = rdma.complete_bind(&core, h1, witness).unwrap();
                    bound1 = true;
                } else {
                    assert!(false);
                }
            },
            Some(_) => assert!(false),
            None => {},
        }
        polls = polls + 1;
    }
    assert!(bound0 && bound1);

    // Revoke `h0` (slot index 0 in `active_leases`), which `swap_remove`s index 0
    // and shifts `h1` down to vector index 0.
    let inv0 = rdma.begin_fenced_invalidate(&core, h0).unwrap();
    let mut inv0_done = false;
    polls = 0;
    while !inv0_done && polls < 100_000 {
        match rdma.poll_cq(&mut core, Tracked(_src.instance.borrow())).unwrap() {
            Some(
                PeerCompletion::Control(rdma::Completed::WindowInvalidated { wr_id, witness }),
            ) if wr_id == inv0 => {
                let _ = rdma.finish_fenced_revoke_with_invalidate(
                    &mut core,
                    &mut host.ctx,
                    h0,
                    witness,
                    Tracked(_src.instance.borrow()),
                ).unwrap();
                inv0_done = true;
            },
            Some(_) => assert!(false),
            None => {},
        }
        polls = polls + 1;
    }
    assert!(inv0_done);

    // Stale `h0` must be rejected cleanly and must NOT affect `h1`.
    match rdma.begin_fenced_invalidate(&core, h0) {
        Err(CoreError::InvalidArgument) => {},
        _ => assert!(false),
    }

    // Acknowledge quarantine and replenish the vacant slot (`h0.mw_slot`), bumping its generation.
    rdma.acknowledge_quarantined_window(&core, h0.mw_slot as u32).unwrap();
    let tracked replenish_tok = _src.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair);
    rdma.replenish_window(&core, h0.mw_slot as u32, &pd0, Tracked(replenish_tok)).unwrap();

    // Re-bind `f0`, reusing `h0.mw_slot` with bumped `generation`; stale `h0` must still fail.
    let (h2, _, _, wr2) = rdma.bind_remote_read_lease(
        &mut core,
        f0,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(h2.mw_slot == h0.mw_slot && h2.generation != h0.generation);
    let mut bound2 = false;
    polls = 0;
    while !bound2 && polls < 100_000 {
        match rdma.poll_cq(&mut core, Tracked(_src.instance.borrow())).unwrap() {
            Some(PeerCompletion::Control(rdma::Completed::WindowBound { wr_id, witness })) if wr_id
                == wr2 => {
                let _ = rdma.complete_bind(&core, h2, witness).unwrap();
                bound2 = true;
            },
            Some(_) => assert!(false),
            None => {},
        }
        polls = polls + 1;
    }
    assert!(bound2);
    match rdma.begin_fenced_invalidate(&core, h0) {
        Err(CoreError::InvalidArgument) => {},
        _ => assert!(false),
    }

    // Both live handles `h1` and `h2` remain independently valid and revocable.
    let inv1 = rdma.begin_fenced_invalidate(&core, h1).unwrap();
    let mut inv1_done = false;
    polls = 0;
    while !inv1_done && polls < 100_000 {
        match rdma.poll_cq(&mut core, Tracked(_src.instance.borrow())).unwrap() {
            Some(
                PeerCompletion::Control(rdma::Completed::WindowInvalidated { wr_id, witness }),
            ) if wr_id == inv1 => {
                let _ = rdma.finish_fenced_revoke_with_invalidate(
                    &mut core,
                    &mut host.ctx,
                    h1,
                    witness,
                    Tracked(_src.instance.borrow()),
                ).unwrap();
                inv1_done = true;
            },
            Some(_) => assert!(false),
            None => {},
        }
        polls = polls + 1;
    }
    assert!(inv1_done);

    let inv2 = rdma.begin_fenced_invalidate(&core, h2).unwrap();
    let mut inv2_done = false;
    polls = 0;
    while !inv2_done && polls < 100_000 {
        match rdma.poll_cq(&mut core, Tracked(_src.instance.borrow())).unwrap() {
            Some(
                PeerCompletion::Control(rdma::Completed::WindowInvalidated { wr_id, witness }),
            ) if wr_id == inv2 => {
                let _ = rdma.finish_fenced_revoke_with_invalidate(
                    &mut core,
                    &mut host.ctx,
                    h2,
                    witness,
                    Tracked(_src.instance.borrow()),
                ).unwrap();
                inv2_done = true;
            },
            Some(_) => assert!(false),
            None => {},
        }
        polls = polls + 1;
    }
    assert!(inv2_done);
    let done_stmt = Tracked(Statement::empty(core.instance_id()));
    channel.send_completion_notice(b"done", done_stmt).unwrap();
}

fn run_rdma_lease_peer_inner(dev_idx: u32, gid_idx: u8, t: TcpTransport) {
    let (Tracked(instance), _, Tracked(mut events1), _) = NodeContext::bootstrap(1).into_sources();
    let dev1 = rdma::DeviceContext::open(dev_idx, 1).unwrap();
    let tracked pd1_tok = events1.mint(EventSourceKind::RdmaQueuePair);
    let pd1 = rdma::ProtectionDomain::new(&dev1, Tracked(pd1_tok)).unwrap();
    let tracked cq1_tok = events1.mint(EventSourceKind::RdmaQueuePair);
    let cq1 = rdma::CompletionQueue::new(&dev1, 32, Tracked(cq1_tok)).unwrap();
    let tracked qp1_tok = events1.mint(EventSourceKind::RdmaQueuePair);
    let uqp1 = rdma::UnconnectedQueuePair::new(
        &pd1,
        &cq1,
        32,
        1,
        gid_idx,
        2,
        Tracked(qp1_tok),
    ).unwrap();
    let mut channel = Channel::new(t, 1, 0, Tracked(&instance));
    let ep0 = channel.exchange_queue_pair_endpoint(&uqp1.endpoint(), Tracked(&instance)).unwrap();
    let qp1 = rdma::connect_rc(uqp1, ep0, 0).expect("connect_rc failed for peer qp1");
    let notice = channel.recv_completion_notice(Tracked(&instance)).unwrap();
    assert!(notice.user_bytes == b"done");
    qp1.destroy(&cq1).unwrap();
    cq1.destroy().unwrap();
    pd1.destroy().unwrap();
    dev1.close().unwrap();
}

fn perform_host_read_on_tl(
    core: &mut CacheCore,
    host: &HostBackend,
    shares: Tracked<&mut ShareSource>,
    instance: Tracked<&TransferInstance>,
    ctx: &mut host::HostContext,
    tl: u8,
    frame: FrameId,
    expected: &Vec<u8>,
) -> (Ticket, Tracked<Knows>) {
    let (lease, _v, _cert, Tracked(cap)) = core.acquire_local_lease(
        frame,
        false,
        shares,
        instance,
    ).unwrap();
    let tracked mut perm = ReadPerm::Shared(cap);
    let slice = host.frame_slice(frame);
    let mut out = vec![0u8; expected.len()];
    let Tracked(comp) = host::read_slice(
        ctx,
        instance,
        &slice,
        &mut out,
        Tracked(&mut perm),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    assert!(out == *expected);
    let tracked cap_back = perm.into_shared();
    let ticket = core.issue_ticket(tl, Tracked(comp.submitted)).unwrap();
    core.release_local_lease(lease, Tracked(cap_back), ticket, instance).unwrap();
    (ticket, Tracked(comp.knows))
}

fn run_reclamation_liveness_under_continuous_hot_overflow(_src: &mut Sources) {
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 8, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        2,
        4096,
    ).unwrap();

    let mut tls: Vec<u8> = Vec::new();
    let mut ctxs: Vec<host::HostContext> = Vec::new();
    let mut idx = 0usize;
    while idx < 5 {
        let tl = core.register_timeline(TimelineKind::Host).unwrap();
        let tracked ev_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
        tls.push(tl);
        ctxs.push(host::HostContext::new(Tracked(ev_src)));
        idx += 1;
    }

    let data_a = pattern(0x31);
    let data_b = pattern(0x62);
    let (_va, frame_a) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        ObjectKey([101, 0, 0, 1]),
        &data_a,
    ).unwrap();
    let (_vb, frame_b) = host.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        ObjectKey([101, 0, 0, 2]),
        &data_b,
    ).unwrap();

    // Overflow frame_a across all 5 read timelines, then evict frame_a into draining.
    let mut tickets_a: Vec<Ticket> = Vec::new();
    let mut knows_a: Vec<Tracked<Knows>> = Vec::new();
    let mut ctx_idx = 0usize;
    while ctx_idx < 5 {
        let tl = tls[ctx_idx];
        let (ticket, knows) = perform_host_read_on_tl(
            &mut core,
            &host,
            Tracked(_src.shares.borrow_mut()),
            Tracked(_src.instance.borrow()),
            &mut ctxs[ctx_idx],
            tl,
            frame_a,
            &data_a,
        );
        tickets_a.push(ticket);
        knows_a.push(knows);
        ctx_idx += 1;
    }
    assert!((core.frames.metas[frame_a.0 as usize].flags & FLAG_READ_OVERFLOW) != 0);
    core.evict_frame(frame_a).unwrap();
    assert!(core.frames.metas[frame_a.0 as usize].state == STATE_DRAINING);

    // Hot frame_b continues overflowing across ALL 5 timelines with newer unconfirmed reads.
    let mut tickets_b: Vec<Ticket> = Vec::new();
    let mut knows_b: Vec<Tracked<Knows>> = Vec::new();
    let mut idx_b = 0usize;
    while idx_b < 5 {
        let tl = tls[idx_b];
        let (ticket, knows) = perform_host_read_on_tl(
            &mut core,
            &host,
            Tracked(_src.shares.borrow_mut()),
            Tracked(_src.instance.borrow()),
            &mut ctxs[idx_b],
            tl,
            frame_b,
            &data_b,
        );
        assert!(ticket.seq() > tickets_a[idx_b].seq());
        tickets_b.push(ticket);
        knows_b.push(knows);
        idx_b += 1;
    }
    assert!((core.frames.metas[frame_b.0 as usize].flags & FLAG_READ_OVERFLOW) != 0);

    // Confirm only frame_a's earlier reads; frame_a MUST reclaim immediately while frame_b's
    // newer reads on all 5 timelines remain unconfirmed!
    let mut c_idx = 0usize;
    while c_idx < 5 {
        let Tracked(k) = knows_a.remove(0);
        core.mark_timeline_complete(tls[c_idx], tickets_a[c_idx].seq(), Tracked(k));
        assert!(core.timelines.confirmed_seq[tls[c_idx] as usize] < tickets_b[c_idx].seq());
        c_idx += 1;
    }
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);
    assert!(core.frames.metas[frame_a.0 as usize].state == STATE_EMPTY);
    assert!((core.frames.metas[frame_a.0 as usize].flags & FLAG_READ_OVERFLOW) == 0);

    // Evicting frame_b now stays blocked while its newer reads are unconfirmed, then succeeds once confirmed.
    core.evict_frame(frame_b).unwrap();
    assert!(core.reclaim_draining(host.tier_id(), 4) == 0);
    assert!(core.frames.metas[frame_b.0 as usize].state == STATE_DRAINING);

    let mut d_idx = 0usize;
    while d_idx < 5 {
        let Tracked(k) = knows_b.remove(0);
        core.mark_timeline_complete(tls[d_idx], tickets_b[d_idx].seq(), Tracked(k));
        d_idx += 1;
    }
    assert!(core.reclaim_draining(host.tier_id(), 4) == 1);
    assert!(core.frames.metas[frame_b.0 as usize].state == STATE_EMPTY);
}

fn run_multi_size_class_host_tiers_and_copy_hop_rejection(_src: &mut Sources) {
    let mut core = CacheCore::new(0, 4096, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked s0 = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host0 = HostBackend::new_with_frame_size(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(s0),
        2,
        4096,
        4096,
    ).unwrap();
    let tracked s1 = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host1 = HostBackend::new_with_frame_size(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(s1),
        2,
        4096,
        16384,
    ).unwrap();
    let tracked s2 = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let host2 = HostBackend::new_with_frame_size(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(s2),
        1,
        4096,
        4096,
    ).unwrap();

    assert!(core.tier_frame_size(host0.tier_id()) == 4096);
    assert!(core.tier_frame_size(host1.tier_id()) == 16384);
    assert!(core.tier_frame_size(host2.tier_id()) == 4096);

    let data_4k = vec![0xA7u8; 4096];
    let data_16k = vec![0x5Eu8; 16384];
    let key_4k = ObjectKey([201, 0, 0, 1]);
    let key_16k = ObjectKey([201, 0, 0, 2]);

    // Wrong-size writes to each tier are rejected cleanly.
    match host0.put_host(&mut core, Tracked(_src.instance.borrow()), key_4k, &data_16k) {
        Err(CoreError::InvalidArgument) => {},
        _ => assert!(false),
    }
    match host1.put_host(&mut core, Tracked(_src.instance.borrow()), key_16k, &data_4k) {
        Err(CoreError::InvalidArgument) => {},
        _ => assert!(false),
    }

    let (v0, f0) = host0.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        key_4k,
        &data_4k,
    ).unwrap();
    let (v1, f1) = host1.put_host(
        &mut core,
        Tracked(_src.instance.borrow()),
        key_16k,
        &data_16k,
    ).unwrap();
    assert!(core.lookup_tier(key_4k, host0.tier_id()) == Some((f0, v0)));
    assert!(core.lookup_tier(key_16k, host1.tier_id()) == Some((f1, v1)));

    let mut out_4k = vec![0u8; 4096];
    let got_v0 = host0.read_frame_into(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f0,
        &mut out_4k,
    ).unwrap();
    assert!(got_v0 == v0 && out_4k == data_4k);

    let mut out_16k = vec![0u8; 16384];
    let got_v1 = host1.read_frame_into(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        f1,
        &mut out_16k,
    ).unwrap();
    assert!(got_v1 == v1 && out_16k == data_16k);

    // Cross-size copy hop MUST be rejected without pinning or leaking leases.
    let leases_before = core.free_lease_slots.len();
    match core.begin_copy_hop(
        f0,
        host1.tier_id(),
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ) {
        Err(CoreError::InvalidArgument) => {},
        _ => assert!(false),
    }
    assert!(core.free_lease_slots.len() == leases_before);

    // Matching-size copy hop succeeds and aborts cleanly.
    let (hop, Tracked(cap), Tracked(tok)) = core.begin_copy_hop(
        f0,
        host2.tier_id(),
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    core.abort_copy_hop(
        hop,
        Tracked(cap),
        Ticket::NONE,
        Tracked(tok),
        Tracked(_src.instance.borrow()),
    );
    assert!(core.free_lease_slots.len() == leases_before);

    core.evict_frame(f0).unwrap();
    core.evict_frame(f1).unwrap();
    assert!(core.reclaim_draining(host0.tier_id(), 4) == 1);
    assert!(core.reclaim_draining(host1.tier_id(), 4) == 1);
}

fn run_nvme_backend_explicit_path_and_size(_src: &mut Sources, path: &str) -> Result<
    usize,
    CoreError,
> {
    let mut core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 4, Tracked(_src.instance.borrow()));
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        1,
        4096,
    ).unwrap();
    let tracked ring_src = _src.events.borrow_mut().mint(EventSourceKind::IoUring);
    let nvme = NvmeBackend::new_with_path(
        &mut core,
        &host.ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(ring_src),
        path,
        2,
        8,
        true,
        8192,
    )?;
    Ok(core.tier_frame_size(nvme.tier_id()))
}

} // verus!
const RDMA_SWAP_ROLE: &str = "VERICACHE_CORE_RDMA_SWAP_ROLE";
const RDMA_SWAP_ADDR: &str = "VERICACHE_CORE_RDMA_SWAP_ADDR";
const RDMA_SWAP_SESSION: &str = "VERICACHE_CORE_RDMA_SWAP_SESSION";

struct TcpTransport(TcpStream);

impl TcpTransport {
    fn new(stream: TcpStream) -> Self {
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        Self(stream)
    }
}

impl ByteTransport for TcpTransport {
    type Error = io::Error;
    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        self.0.write_all(&n.to_be_bytes())?;
        self.0.write_all(frame)
    }
    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        let mut prefix = [0; 4];
        self.0.read_exact(&mut prefix)?;
        let n = u32::from_be_bytes(prefix) as usize;
        if n > 1024 * 1024 {
            return Err(io::Error::other("frame exceeds test limit"));
        }
        let mut frame = vec![0; n];
        self.0.read_exact(&mut frame)?;
        Ok(frame)
    }
}

struct ChildGuard(Option<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn rdma_gid_index() -> u8 {
    std::env::var("VERICACHE_RDMA_GID_INDEX")
        .map(|s| s.parse().expect("invalid VERICACHE_RDMA_GID_INDEX"))
        .unwrap_or(0)
}

fn select_type2_rdma_device_index() -> Option<u32> {
    let requested = std::env::var("VERICACHE_RDMA_DEV").ok();
    let count = match rdma::DeviceContext::count() {
        Ok(n) => n,
        Err(_) => {
            if require_hw() {
                panic!("VERICACHE_REQUIRE_HW set, but RDMA device enumeration failed");
            }
            return None;
        }
    };
    for idx in 0..count {
        if let Ok(d) = rdma::DeviceContext::open(idx, 0) {
            let name_ok = requested.as_deref().is_none_or(|name| name == d.name());
            let type2 = d.supports_type_2().unwrap_or(false);
            let _ = d.close();
            if name_ok && type2 {
                return Some(idx);
            }
        }
    }
    if require_hw() {
        panic!("VERICACHE_REQUIRE_HW set, but no RDMA Type-2 device available");
    }
    None
}

fn start_rdma_swap_peer() -> (TcpTransport, ChildGuard) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut secret = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut secret)
        .unwrap();
    let session: String = secret.iter().map(|b| format!("{b:02x}")).collect();
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "rdma_lease_handle_swap_remove_regression",
            "--nocapture",
        ])
        .env(RDMA_SWAP_ROLE, "peer")
        .env(RDMA_SWAP_SESSION, &session)
        .env(RDMA_SWAP_ADDR, listener.local_addr().unwrap().to_string())
        .spawn()
        .unwrap();
    let guard = ChildGuard(Some(child));
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        match listener.accept() {
            Ok((s, _)) => {
                let mut t = TcpTransport::new(s);
                assert!(t.recv_frame().unwrap() == session.as_bytes()[..32]);
                t.send_frame(&session.as_bytes()[32..]).unwrap();
                return (t, guard);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= until {
                    panic!("rdma swap_remove peer connect timed out");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("accept: {e}"),
        }
    }
}

fn connect_rdma_swap_peer() -> TcpTransport {
    let mut t =
        TcpTransport::new(TcpStream::connect(std::env::var(RDMA_SWAP_ADDR).unwrap()).unwrap());
    let session = std::env::var(RDMA_SWAP_SESSION).unwrap();
    t.send_frame(&session.as_bytes()[..32]).unwrap();
    assert!(t.recv_frame().unwrap() == session.as_bytes()[32..]);
    t
}

#[test]
fn custom_tier_and_frame_table_lifecycle() {
    let mut src = sources();
    run_custom_tier_and_frame_table_lifecycle(&mut src);
}

#[test]
fn inline_read_compaction_and_overflow_fallback() {
    let mut src = sources();
    run_inline_read_compaction_and_overflow_fallback(&mut src);
}

#[test]
fn concurrent_local_and_remote_leases_overflow_and_eviction() {
    let mut src = sources();
    run_concurrent_local_and_remote_leases_overflow_and_eviction(&mut src);
}

#[test]
fn host_backend_put_read_and_copy_tier() {
    let mut src = sources();
    run_host_backend_put_read_and_copy_tier(&mut src);
}

#[test]
fn nvme_backend_io_uring_round_trip() {
    let mut src = sources();
    run_nvme_backend_io_uring_round_trip(&mut src);
}

#[test]
fn cuda_wait_event_not_ready_before_poll() {
    let mut src = sources();
    run_cuda_wait_event_not_ready_before_poll(&mut src);
}

#[test]
fn cuda_backend_ordered_handoff_and_section_5_3_hazard() {
    let mut src = sources();
    run_cuda_backend_ordered_handoff_and_section_5_3_hazard(&mut src);
}

#[test]
fn remote_lease_and_fenced_revocation_contract() {
    let mut src = sources();
    run_remote_lease_and_fenced_revocation_contract(&mut src);
}

#[test]
fn cuda_one_slot_ring_readiness_bidirectional() {
    let mut src = sources();
    run_cuda_one_slot_ring_readiness_bidirectional(&mut src);
}

#[test]
fn rdma_lease_handle_swap_remove_regression() {
    let dev_idx = match select_type2_rdma_device_index() {
        Some(idx) => idx,
        None => return,
    };
    let gid_idx = rdma_gid_index();
    if std::env::var(RDMA_SWAP_ROLE).as_deref() == Ok("peer") {
        let t = connect_rdma_swap_peer();
        run_rdma_lease_peer_inner(dev_idx, gid_idx, t);
        return;
    }
    let mut src = sources();
    let (t, mut child) = start_rdma_swap_peer();
    run_rdma_lease_handle_swap_remove_regression(&mut src, dev_idx, gid_idx, t);
    let status = child.0.take().unwrap().wait().unwrap();
    assert!(status.success(), "rdma peer child failed: {status}");
}

#[test]
fn reclamation_liveness_under_continuous_hot_overflow() {
    let mut src = sources();
    run_reclamation_liveness_under_continuous_hot_overflow(&mut src);
}

#[test]
fn multi_size_class_host_tiers_and_copy_hop_rejection() {
    let mut src = sources();
    run_multi_size_class_host_tiers_and_copy_hop_rejection(&mut src);
}

#[test]
fn nvme_backend_explicit_path_and_size() {
    let mut src = sources();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path_buf = std::env::temp_dir().join(format!(
        "vericache_w2b_{}_{}.slab",
        std::process::id(),
        unique
    ));
    let path_str = path_buf.to_str().unwrap().to_string();
    let res = run_nvme_backend_explicit_path_and_size(&mut src, &path_str);
    let _ = std::fs::remove_file(&path_buf);
    match res {
        Ok(sz) => assert_eq!(sz, 8192),
        Err(CoreError::Backend {
            kind: BACKEND_NVME,
            code: errno,
        }) => {
            if require_hw() {
                panic!("VERICACHE_REQUIRE_HW set, but NvmeBackend::new_with_path failed with errno {errno}");
            }
        }
        Err(e) => panic!("unexpected NvmeBackend::new_with_path error: {e:?}"),
    }
}
