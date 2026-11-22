use std::sync::{Mutex, OnceLock};
use transfer::domain::DeviceId;
#[cfg(verus_only)]
use transfer::domain::*;
use transfer::{context::*, cuda, host};
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::host::HostBackend;
use vericache_core::backends::nvme::NvmeBackend;
use vericache_core::engine::LocalEngine;
use vericache_core::frame::{STATE_DRAINING, STATE_EMPTY};
use vericache_core::kernel::CacheCore;
use vericache_core::provenance::ValueCertificate;
#[cfg(verus_only)]
use vericache_core::provenance::*;
use vericache_core::spec::*;
use vericache_core::timeline::TimelineKind;
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

fn has_cuda_device() -> bool {
    match cuda::device_count() {
        Ok(n) if n > 0 => true,
        _ => false,
    }
}

fn pattern(seed: u8) -> Vec<u8> {
    (0..FRAME_SIZE)
        .map(|i| seed.wrapping_add(((i * 7) % 251) as u8))
        .collect()
}

verus! {

fn run_host_put_get_concurrent_lease_evict_reclaim(_src: &mut Sources) {
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
    let tracked eng_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let eng_ctx = host::HostContext::new(Tracked(eng_src));
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, None);

    let k1 = ObjectKey([1, 2, 3, 4]);
    let data1 = pattern(0x21);
    let (v1, f1) = engine.put_host(Tracked(_src.instance.borrow()), k1, &data1).unwrap();

    // 1-hop get_into_host hit.
    let mut out = vec![0u8; FRAME_SIZE];
    let hit = engine.get_into_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k1,
        &mut out,
    ).unwrap();
    assert!(hit == Some(v1));
    assert!(out == data1);

    // Two concurrent HostLease<'a> handles coexist with &mut engine operations!
    let lease_a = engine.get_lease_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k1,
    ).unwrap().unwrap();
    let lease_b = engine.get_lease_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k1,
    ).unwrap().unwrap();
    assert!(lease_a.version == v1 && lease_b.version == v1);

    // Verify zero-copy read through lease_a.slice using a separate HostContext.
    let tracked read_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut read_ctx = host::HostContext::new(Tracked(read_src));
    let tracked mut perm_a = ReadPerm::Shared(lease_a.cap.get());
    let mut lease_buf = vec![0u8; FRAME_SIZE];
    let _comp = host::read_slice(
        &mut read_ctx,
        Tracked(_src.instance.borrow()),
        &lease_a.slice,
        &mut lease_buf,
        Tracked(&mut perm_a),
        Tracked(&engine.core.ledger.borrow().submitted),
        Tracked(&engine.core.ledger.borrow().knows),
        Ghost(engine.core.citations()),
    );
    assert!(lease_buf == data1);
    let tracked cap_a = perm_a.into_shared();

    // Evict while both leases are active: hidden from catalog, cannot reclaim yet.
    engine.evict_frame(f1).unwrap();
    let miss_after_evict = engine.get_into_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k1,
        &mut out,
    ).unwrap();
    assert!(miss_after_evict.is_none());
    assert!(engine.reclaim_draining(engine.host_tier(), 4) == 0);

    // Release lease_a; still blocked by lease_b.
    engine.release_lease(
        Tracked(_src.instance.borrow()),
        lease_a.handle,
        Tracked(cap_a),
        Ticket::NONE,
    ).unwrap();
    assert!(engine.reclaim_draining(engine.host_tier(), 4) == 0);

    // Release lease_b; poll_progress reclaims f1 back to ready_queue!
    engine.release_lease(
        Tracked(_src.instance.borrow()),
        lease_b.handle,
        lease_b.cap,
        Ticket::NONE,
    ).unwrap();
    let rep = engine.poll_progress(Tracked(_src.instance.borrow())).unwrap();
    assert!(rep.reclaimed_frames == 1);
    assert!(engine.core.frames.metas[f1.0 as usize].state == STATE_EMPTY);

    // Reuse reclaimed frame for a new key.
    let k2 = ObjectKey([5, 6, 7, 8]);
    let data2 = pattern(0x99);
    let (v2, _f2) = engine.put_host(Tracked(_src.instance.borrow()), k2, &data2).unwrap();
    let mut out2 = vec![0u8; FRAME_SIZE];
    let hit2 = engine.get_into_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k2,
        &mut out2,
    ).unwrap();
    assert!(hit2 == Some(v2));
    assert!(out2 == data2);
}

fn run_version_range_exhaustion(_src: &mut Sources) {
    // Disjoint shard range [100, 102): exactly 2 versions available.
    let mut core = CacheCore::new_with_version_range(
        0,
        FRAME_SIZE,
        Vec::new(),
        4,
        100,
        102,
        Tracked(_src.instance.borrow()),
    );
    let tracked host_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let host = HostBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(host_src),
        4,
        4096,
    ).unwrap();
    let tracked eng_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let eng_ctx = host::HostContext::new(Tracked(eng_src));
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, None);

    let p = pattern(0x42);
    let (v0, _) = engine.put_host(
        Tracked(_src.instance.borrow()),
        ObjectKey([1, 0, 0, 0]),
        &p,
    ).unwrap();
    let (v1, _) = engine.put_host(
        Tracked(_src.instance.borrow()),
        ObjectKey([2, 0, 0, 0]),
        &p,
    ).unwrap();
    assert!(v0.seq == 100 && v1.seq == 101);

    let err = engine.put_host(Tracked(_src.instance.borrow()), ObjectKey([3, 0, 0, 0]), &p);
    assert!(err == Err(CoreError::SequenceExhausted));
    // Verify no frame was leaked by the rejected write (2 of 4 frames still ready).
    assert!(engine.core.frames.ready_queue[engine.host_tier().as_index()].len() == 2);
}

fn run_dram_nvme_two_hop_tier_and_reload(_src: &mut Sources) {
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
    let tracked ring_src = _src.events.borrow_mut().mint(EventSourceKind::IoUring);
    let nvme = match NvmeBackend::new(
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
                panic!("VERICACHE_REQUIRE_HW set, but io_uring_setup returned errno {errno}");
            }
            return;
        },
        Err(e) => panic!("unexpected NvmeBackend::new error: {e:?}"),
    };
    let tracked eng_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let eng_ctx = host::HostContext::new(Tracked(eng_src));
    let mut engine = LocalEngine::new(core, &host, eng_ctx, None, Some(nvme));

    let key = ObjectKey([10, 20, 30, 40]);
    let data = pattern(0xB3);
    let (ver, dram_f) = engine.put_host(Tracked(_src.instance.borrow()), key, &data).unwrap();

    // Tier Host DRAM -> Local NVMe and wait for io_uring ticket retirement.
    let (_nvme_f, t_down) = engine.tier_dram_to_nvme(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        key,
    ).unwrap();
    let prog = engine.wait_nvme_ticket(Tracked(_src.instance.borrow()), t_down).unwrap();
    assert!(prog.completed == 1 && prog.failed == 0);

    // Evict DRAM, overwrite all DRAM frames with noise, then evict noise so only NVMe holds `key`.
    engine.evict_frame(dram_f).unwrap();
    assert!(engine.reclaim_draining(engine.host_tier(), 4) == 1);
    let (_, nf1) = engine.put_host(
        Tracked(_src.instance.borrow()),
        ObjectKey([91, 0, 0, 0]),
        &pattern(0x01),
    ).unwrap();
    let (_, nf2) = engine.put_host(
        Tracked(_src.instance.borrow()),
        ObjectKey([92, 0, 0, 0]),
        &pattern(0x02),
    ).unwrap();
    engine.evict_frame(nf1).unwrap();
    engine.evict_frame(nf2).unwrap();
    assert!(engine.reclaim_draining(engine.host_tier(), 4) == 2);
    assert!(engine.core.lookup_tier(key, engine.host_tier()).is_none());

    // 2-hop get_into_host: automatically warms Host DRAM from Local NVMe and returns exact bytes!
    let mut out = vec![0u8; FRAME_SIZE];
    let got = engine.get_into_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        key,
        &mut out,
    ).unwrap();
    assert!(got == Some(ver));
    assert!(out == data);
    assert!(engine.core.lookup_tier(key, engine.host_tier()).is_some());
}

fn run_gpu_ordered_handoff_put_gpu_and_section_5_3_hazard(_src: &mut Sources) {
    if !has_cuda_device() {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but CUDA unavailable");
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
    let tracked gpu_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut cuda = CudaBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(gpu_src),
        dev,
        2,
        16,
        false,
    ).unwrap();
    cuda.attach_pinned_host(&core, &host).unwrap();

    let tracked caller_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut caller_stream = cuda::CudaStream::new(dev, Tracked(caller_src)).unwrap();
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: dev }).unwrap();

    let tracked eng_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let eng_ctx = host::HostContext::new(Tracked(eng_src));
    let mut engine = LocalEngine::new(core, &host, eng_ctx, Some(cuda), None);

    // 1. put_host -> get_into_gpu (Ordered Handoff H2D).
    let k_host = ObjectKey([100, 101, 102, 103]);
    let data = pattern(0x6D);
    let (v_host, dram_f) = engine.put_host(Tracked(_src.instance.borrow()), k_host, &data).unwrap();

    let (dev_buf, Tracked(mut dev_tok)) = cuda::alloc_device(
        Tracked(_src.allocations.borrow_mut()),
        dev,
        FRAME_SIZE,
    ).unwrap();
    let dev_slice = dev_buf.slot_slice(0);
    let (v_h2d, t_h2d, _ev_h2d) = engine.get_into_gpu(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k_host,
        &mut caller_stream,
        caller_tl,
        &dev_slice,
        Tracked(&mut dev_tok),
    ).unwrap().unwrap();
    assert!(v_h2d == v_host);

    // Synchronize caller stream so dev_buf is ready as a put_gpu source.
    let Tracked(k_h2d) = cuda::stream_synchronize(
        &caller_stream,
        Tracked(_src.instance.borrow()),
    ).unwrap();
    engine.core.mark_timeline_complete(caller_tl, t_h2d.seq(), Tracked(k_h2d));

    // 2. put_gpu into GPU VRAM -> verify not yet published before poll_progress/synchronize_cuda.
    let k_gpu = ObjectKey([104, 105, 106, 107]);
    let gpu_tier = engine.gpu_tier().unwrap();
    let tracked mut dev_perm = ReadPerm::Owned(dev_tok);
    let ghost dev_cert = ValueCertificate {
        slot: dev_slice.slot_ref(),
        writer: _ev_h2d@,
        bytes: data@,
        proof: ValueProof::HostWrite,
    };
    let (v_gpu, gpu_f, _t_put, _sp) = engine.put_gpu(
        Tracked(_src.instance.borrow()),
        gpu_tier,
        k_gpu,
        &dev_slice,
        Tracked(&mut dev_perm),
        Ghost(dev_cert),
    ).unwrap();
    assert!(engine.core.lookup_tier(k_gpu, gpu_tier).is_none());

    let sync_rep = engine.synchronize_cuda(Tracked(_src.instance.borrow())).unwrap();
    assert!(sync_rep.writes_published == 1);
    assert!(engine.core.lookup_tier(k_gpu, gpu_tier) == Some((gpu_f, v_gpu)));

    // 3. Evict Host DRAM copy so get_into_host must execute 2-hop GPU -> Pinned DRAM -> Host!
    engine.evict_frame(dram_f).unwrap();
    assert!(engine.reclaim_draining(engine.host_tier(), 4) == 1);

    let mut host_out = vec![0u8; FRAME_SIZE];
    let got_2hop = engine.get_into_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k_gpu,
        &mut host_out,
    ).unwrap();
    assert!(got_2hop == Some(v_gpu));
    assert!(host_out == data);

    // 4. §5.3 hazard on GPU VRAM: acquire zero-copy GpuLease, enqueue D2D read on caller_stream,
    //    release_lease with caller ticket, immediately evict_frame -> reclamation MUST stay blocked
    //    until caller_stream confirms its ticket, then succeeds cleanly.
    let gpu_lease = engine.get_lease_gpu(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        k_gpu,
    ).unwrap().unwrap();
    assert!(gpu_lease.frame == gpu_f && gpu_lease.version == v_gpu);

    let (dst2_buf, Tracked(mut dst2_tok)) = cuda::alloc_device(
        Tracked(_src.allocations.borrow_mut()),
        dev,
        FRAME_SIZE,
    ).unwrap();
    let dst2_slice = dst2_buf.slot_slice(0);
    let src_gpu_slice = engine.gpu_frame_slice(gpu_lease.frame).unwrap();
    let tracked mut lease_perm = ReadPerm::Shared(gpu_lease.cap.get());
    let (_ev_d2d, receipt_d2d) = {
        let tracked ledger_ref = engine.core.ledger.borrow_mut();
        cuda::memcpy_async(
            &mut caller_stream,
            Tracked(_src.instance.borrow()),
            &dst2_slice,
            &src_gpu_slice,
            Tracked(&mut lease_perm),
            Tracked(&mut dst2_tok),
            Tracked(&mut ledger_ref.submitted),
            Tracked(&ledger_ref.knows),
            Ghost(engine.core.citations()),
        ).unwrap()
    };
    let t_d2d = engine.core.issue_ticket(caller_tl, receipt_d2d).unwrap();
    let tracked cap_back = lease_perm.into_shared();
    engine.release_lease(
        Tracked(_src.instance.borrow()),
        gpu_lease.handle,
        Tracked(cap_back),
        t_d2d,
    ).unwrap();

    // Evict GPU frame while t_d2d is unconfirmed: MUST stay Draining and NOT reclaim!
    engine.evict_frame(gpu_f).unwrap();
    assert!(engine.core.frames.metas[gpu_f.0 as usize].state == STATE_DRAINING);
    let rep_before = engine.poll_progress(Tracked(_src.instance.borrow())).unwrap();
    assert!(rep_before.reclaimed_frames == 0);

    // Confirm caller_stream; now poll_progress safely reclaims gpu_f!
    let Tracked(k_d2d) = cuda::stream_synchronize(
        &caller_stream,
        Tracked(_src.instance.borrow()),
    ).unwrap();
    engine.core.mark_timeline_complete(caller_tl, t_d2d.seq(), Tracked(k_d2d));
    let rep_after = engine.poll_progress(Tracked(_src.instance.borrow())).unwrap();
    assert!(rep_after.reclaimed_frames == 1);
    assert!(engine.core.frames.metas[gpu_f.0 as usize].state == STATE_EMPTY);

    let _ = caller_stream.destroy();
}

fn run_gpu_dram_nvme_full_multitier_4slice_scatter_gather(_src: &mut Sources) {
    if !has_cuda_device() {
        if require_hw() {
            panic!("VERICACHE_REQUIRE_HW set, but CUDA unavailable");
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
    let tracked gpu_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut cuda = CudaBackend::new(
        &mut core,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        Tracked(gpu_src),
        dev,
        2,
        16,
        false,
    ).unwrap();
    cuda.attach_pinned_host(&core, &host).unwrap();

    let tracked ring_src = _src.events.borrow_mut().mint(EventSourceKind::IoUring);
    let nvme = match NvmeBackend::new(
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
                panic!("VERICACHE_REQUIRE_HW set, but io_uring_setup returned errno {errno}");
            }
            return;
        },
        Err(e) => panic!("unexpected NvmeBackend::new error: {e:?}"),
    };

    let tracked caller_src = _src.events.borrow_mut().mint(EventSourceKind::CudaStream);
    let mut caller_stream = cuda::CudaStream::new(dev, Tracked(caller_src)).unwrap();
    let caller_tl = core.register_timeline(TimelineKind::CudaStream { device: dev }).unwrap();

    let tracked eng_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let eng_ctx = host::HostContext::new(Tracked(eng_src));
    let mut engine = LocalEngine::new(core, &host, eng_ctx, Some(cuda), Some(nvme));

    // Prepare 4 distinct 1024-byte slices (total 4096 == FRAME_SIZE) in pinned host memory,
    // then upload each into its own independent caller GPU buffer.
    let slice_len = FRAME_SIZE / 4;
    let tracked prep_src = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut prep_ctx = host::HostContext::new(Tracked(prep_src));

    let mut expected = Vec::with_capacity(FRAME_SIZE);
    let mut src_gpu_bufs = Vec::new();
    let tracked mut src_perms: Map<int, ReadPerm> = Map::tracked_empty();
    let mut ghost_certs: Ghost<Seq<ValueCertificate>> = Ghost(Seq::empty());
    let mut last_prep_ticket = Ticket::NONE;

    for i in 0..4usize {
        let chunk: Vec<u8> = (0..slice_len).map(
            |j| ((i as u8).wrapping_mul(53)).wrapping_add((j % 241) as u8),
        ).collect();
        expected.extend_from_slice(&chunk);

        let (h_buf, Tracked(mut h_tok)) = cuda::alloc_host(
            Tracked(_src.allocations.borrow_mut()),
            slice_len,
        ).unwrap();
        let h_slice = h_buf.as_host_slice();
        let Tracked(w_comp) = host::write_slice(
            &mut prep_ctx,
            Tracked(_src.instance.borrow()),
            &h_slice,
            &chunk,
            Tracked(&mut h_tok),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        let t_w = engine.core.issue_ticket(
            engine.host.timeline_id(),
            Tracked(w_comp.submitted),
        ).unwrap();
        engine.core.mark_timeline_complete(
            engine.host.timeline_id(),
            t_w.seq(),
            Tracked(w_comp.knows),
        );

        let (g_buf, Tracked(mut g_tok)) = cuda::alloc_device(
            Tracked(_src.allocations.borrow_mut()),
            dev,
            slice_len,
        ).unwrap();
        let g_slice = g_buf.slot_slice(0);
        let h_cuda = h_buf.as_cuda_slice();
        let tracked mut h_perm = ReadPerm::Owned(h_tok);
        let (_ev_g, receipt_g) = {
            let tracked ledger_ref = engine.core.ledger.borrow_mut();
            cuda::memcpy_async(
                &mut caller_stream,
                Tracked(_src.instance.borrow()),
                &g_slice,
                &h_cuda,
                Tracked(&mut h_perm),
                Tracked(&mut g_tok),
                Tracked(&mut ledger_ref.submitted),
                Tracked(&ledger_ref.knows),
                Ghost(engine.core.citations()),
            ).unwrap()
        };
        last_prep_ticket = engine.core.issue_ticket(caller_tl, receipt_g).unwrap();
        let ghost cert_i = ValueCertificate {
            slot: g_slice.slot_ref(),
            writer: _ev_g@,
            bytes: chunk@,
            proof: ValueProof::HostWrite,
        };
        ghost_certs = Ghost(ghost_certs@.push(cert_i));
        proof {
            src_perms.tracked_insert(i as int, ReadPerm::Owned(g_tok));
        }
        src_gpu_bufs.push(g_buf);
    }
    let Tracked(k_prep) = cuda::stream_synchronize(
        &caller_stream,
        Tracked(_src.instance.borrow()),
    ).unwrap();
    engine.core.mark_timeline_complete(caller_tl, last_prep_ticket.seq(), Tracked(k_prep));

    // 1. 4-slice GPU gather (`put_gpu_scatter`) into ONE GPU VRAM frame!
    let src_slices = [
        src_gpu_bufs[0].slot_slice(0),
        src_gpu_bufs[1].slot_slice(0),
        src_gpu_bufs[2].slot_slice(0),
        src_gpu_bufs[3].slot_slice(0),
    ];
    let ghost offsets: Seq<usize> = seq![0usize, 1024usize, 2048usize, 3072usize, 4096usize];
    let key = ObjectKey([200, 201, 202, 203]);
    let gpu_tier = engine.gpu_tier().unwrap();
    let (ver, gpu_f, _t_scatter, _sp) = engine.put_gpu_scatter(
        Tracked(_src.instance.borrow()),
        key,
        &src_slices,
        Tracked(&mut src_perms),
        ghost_certs,
        Ghost(offsets),
        Ghost(expected@),
    ).unwrap();
    // Not visible in GPU tier before synchronize_cuda / poll_progress!
    assert!(engine.core.lookup_tier(key, gpu_tier).is_none());

    let pub_rep = engine.synchronize_cuda(Tracked(_src.instance.borrow())).unwrap();
    assert!(pub_rep.writes_published == 1);
    assert!(engine.core.lookup_tier(key, gpu_tier) == Some((gpu_f, ver)));

    // 2. 2-hop get_into_host (GPU VRAM -> Pinned Host DRAM -> caller slice) verifies exact bytes!
    let mut host_check = vec![0u8; FRAME_SIZE];
    let got_host = engine.get_into_host(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        key,
        &mut host_check,
    ).unwrap();
    assert!(got_host == Some(ver));
    assert!(host_check == expected);
    let (dram_f, _) = engine.core.lookup_tier(key, engine.host_tier()).unwrap();

    // 3. Tier Host DRAM -> Local NVMe (`io_uring`), wait for completion.
    let (nvme_f, t_nvme) = engine.tier_dram_to_nvme(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        key,
    ).unwrap();
    let nvme_prog = engine.wait_nvme_ticket(Tracked(_src.instance.borrow()), t_nvme).unwrap();
    assert!(nvme_prog.completed == 1 && nvme_prog.failed == 0);
    assert!(engine.core.lookup_tier(key, engine.nvme_tier().unwrap()) == Some((nvme_f, ver)));

    // 4. Evict & reclaim BOTH GPU VRAM and Host DRAM replicas; only Local NVMe remains!
    engine.evict_frame(gpu_f).unwrap();
    engine.evict_frame(dram_f).unwrap();
    let clean_rep = engine.poll_progress(Tracked(_src.instance.borrow())).unwrap();
    assert!(clean_rep.reclaimed_frames == 2);
    assert!(engine.core.lookup_tier(key, gpu_tier).is_none());
    assert!(engine.core.lookup_tier(key, engine.host_tier()).is_none());

    // 5. 2-hop `get_into_gpu_scatter`: Local NVMe -> Pinned Host DRAM -> 4 caller GPU slices!
    let mut dst_gpu_bufs = Vec::new();
    let tracked mut dst_toks: Map<int, SlotToken> = Map::tracked_empty();
    for _i in 0..4usize {
        let (g_buf, Tracked(g_tok)) = cuda::alloc_device(
            Tracked(_src.allocations.borrow_mut()),
            dev,
            slice_len,
        ).unwrap();
        proof {
            dst_toks.tracked_insert(_i as int, g_tok);
        }
        dst_gpu_bufs.push(g_buf);
    }
    let dst_slices = [
        dst_gpu_bufs[0].slot_slice(0),
        dst_gpu_bufs[1].slot_slice(0),
        dst_gpu_bufs[2].slot_slice(0),
        dst_gpu_bufs[3].slot_slice(0),
    ];
    let (v_out, t_out) = engine.get_into_gpu_scatter(
        Tracked(_src.instance.borrow()),
        Tracked(_src.shares.borrow_mut()),
        key,
        &mut caller_stream,
        caller_tl,
        &dst_slices,
        Tracked(&mut dst_toks),
        Ghost(offsets),
    ).unwrap().unwrap();
    assert!(v_out == ver);

    // Synchronize caller stream and read all 4 GPU destination slices back to host to check bytes!
    let Tracked(k_out) = cuda::stream_synchronize(
        &caller_stream,
        Tracked(_src.instance.borrow()),
    ).unwrap();
    engine.core.mark_timeline_complete(caller_tl, t_out.seq(), Tracked(k_out));

    let mut reassembled = Vec::with_capacity(FRAME_SIZE);
    for i in 0..4usize {
        let (rb_buf, Tracked(mut rb_tok)) = cuda::alloc_host(
            Tracked(_src.allocations.borrow_mut()),
            slice_len,
        ).unwrap();
        let rb_cuda = rb_buf.as_cuda_slice();
        let tracked g_tok = dst_toks.tracked_remove(i as int);
        let tracked mut g_perm = ReadPerm::Owned(g_tok);
        let (_ev_rb, receipt_rb) = {
            let tracked ledger_ref = engine.core.ledger.borrow_mut();
            cuda::memcpy_async(
                &mut caller_stream,
                Tracked(_src.instance.borrow()),
                &rb_cuda,
                &dst_slices[i],
                Tracked(&mut g_perm),
                Tracked(&mut rb_tok),
                Tracked(&mut ledger_ref.submitted),
                Tracked(&ledger_ref.knows),
                Ghost(engine.core.citations()),
            ).unwrap()
        };
        let t_rb = engine.core.issue_ticket(caller_tl, receipt_rb).unwrap();
        let Tracked(k_rb) = cuda::stream_synchronize(
            &caller_stream,
            Tracked(_src.instance.borrow()),
        ).unwrap();
        engine.core.mark_timeline_complete(caller_tl, t_rb.seq(), Tracked(k_rb));

        let mut chunk_out = vec![0u8; slice_len];
        let rb_host = rb_buf.as_host_slice();
        let tracked mut rb_perm = ReadPerm::Owned(rb_tok);
        let _r_comp = host::read_slice(
            &mut prep_ctx,
            Tracked(_src.instance.borrow()),
            &rb_host,
            &mut chunk_out,
            Tracked(&mut rb_perm),
            Tracked(&engine.core.ledger.borrow().submitted),
            Tracked(&engine.core.ledger.borrow().knows),
            Ghost(engine.core.citations()),
        );
        reassembled.extend_from_slice(&chunk_out);
    }
    assert!(reassembled == expected);
    let _ = caller_stream.destroy();
}

} // verus!
#[test]
fn host_put_get_concurrent_lease_evict_reclaim() {
    let mut src = sources();
    run_host_put_get_concurrent_lease_evict_reclaim(&mut src);
}

#[test]
fn version_range_exhaustion_returns_sequence_exhausted() {
    let mut src = sources();
    run_version_range_exhaustion(&mut src);
}

#[test]
fn dram_nvme_two_hop_tier_and_reload() {
    let mut src = sources();
    run_dram_nvme_two_hop_tier_and_reload(&mut src);
}

#[test]
fn gpu_ordered_handoff_put_gpu_and_section_5_3_hazard() {
    let mut src = sources();
    run_gpu_ordered_handoff_put_gpu_and_section_5_3_hazard(&mut src);
}

#[test]
fn gpu_dram_nvme_full_multitier_4slice_scatter_gather() {
    let mut src = sources();
    run_gpu_dram_nvme_full_multitier_4slice_scatter_gather(&mut src);
}
