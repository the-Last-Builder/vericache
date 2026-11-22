use std::sync::{Mutex, OnceLock};
#[cfg(verus_only)]
use transfer::domain::*;
use transfer::{context::*, host};
#[cfg(verus_only)]
use vericache_core::provenance::*;
use vericache_core::{catalog::*, frame::*, kernel::*, spec::*, timeline::*};
use vstd::prelude::*;

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

verus! {

fn run_custom_tier_lifecycle(_src: &mut Sources) {
    let num_tiers = 16usize;
    let frame_size = 4096usize;
    let custom_tier = TierId(12);
    let descriptor = TierDescriptor {
        id: custom_tier,
        kind: TierKind::Custom(12),
        device_id: None,
        host_accessible: true,
        rdma_capable: false,
    };
    assert!(descriptor.id == custom_tier);
    assert!(descriptor.kind == TierKind::Custom(12));

    let tracked host_source = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host_ctx = host::HostContext::new(Tracked(host_source));
    let mut table = TimelineTable::new();
    let host_tl = table.register_timeline(TimelineKind::Custom(12)).unwrap();
    let tracked mut ledger = KnowledgeLedger::empty(_src.instance@.id());
    let mut frames = FrameTable::new(num_tiers, frame_size);
    let mut catalog = Catalog::new(num_tiers);

    let (buf, tokens) = host::alloc_slab(
        &host_ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        2,
        frame_size,
        4096,
    ).unwrap();
    let ghost slots = Seq::new(2nat, |i: int| buf.slot_ref(i as nat));
    frames.register_slab(custom_tier, Ghost(slots), tokens, 2, &table, Tracked(&ledger), Ghost(0));
    assert!(frames.ready_queue[custom_tier.as_index()].len() == 2);

    let key = ObjectKey([1, 2, 3, 4]);
    let version = VersionId { origin_node: 7, seq: 42 };
    let (f, Tracked(mut dst_tok)) = frames.reserve_ready_with_token(
        custom_tier,
        key,
        version,
        &table,
        Ghost(KernelView::of(&table, &ledger, 0)),
    ).unwrap();

    let slice = buf.slot_slice(f.0 as usize);
    let payload = vec![0xABu8; frame_size];
    let Tracked(comp) = host::write_slice(
        &mut host_ctx,
        Tracked(_src.instance.borrow()),
        &slice,
        &payload,
        Tracked(&mut dst_tok),
        Tracked(&ledger.submitted),
        Tracked(&ledger.knows),
        Ghost(ledger.citations(0)),
    );
    let ticket = table.issue_ticket(
        host_tl,
        Ghost(comp.submitted.key()),
        Tracked(comp.submitted),
        Tracked(&mut ledger),
    ).unwrap();
    table.advance_confirmed(
        host_tl,
        ticket.seq(),
        Tracked(comp.knows),
        Tracked(&mut ledger),
        Ghost(0),
    );
    catalog.publish_frame_with_token(
        &mut frames,
        f,
        Tracked(dst_tok),
        ticket,
        Ghost(
            ValueCertificate {
                slot: slice.slot_ref(),
                writer: table.events@[(host_tl, ticket.sequence())],
                bytes: payload@,
                proof: ValueProof::HostWrite,
            },
        ),
        Ghost(KernelView::of(&table, &ledger, 0)),
    );

    assert!(catalog.lookup_tier(key, custom_tier) == Some((version, f)));
    assert!(
        catalog.best_replica(key, &[TierId(0), TierId(99), custom_tier, TierId(1)])
            == Some((f, custom_tier, version))
    );
    assert!(catalog.best_replica(key, &[TierId(0), TierId(1)]).is_none());

    catalog.evict_frame(&mut frames, f, Ghost(KernelView::of(&table, &ledger, 0))).unwrap();
    assert!(catalog.lookup(key).is_none());
    let reclaimed = frames.reclaim_draining(
        &table,
        custom_tier,
        8,
        Ghost(KernelView::of(&table, &ledger, 0)),
    );
    assert!(reclaimed == 1);
    assert!(frames.ready_queue[custom_tier.as_index()].len() == 2);
}

fn run_cache_core_end_to_end(_src: &mut Sources) {
    let frame_size = 4096usize;
    let desc0 = TierDescriptor {
        id: TierId(0),
        kind: TierKind::Custom(100),
        device_id: None,
        host_accessible: true,
        rdma_capable: false,
    };
    let desc1 = TierDescriptor {
        id: TierId(1),
        kind: TierKind::Custom(200),
        device_id: None,
        host_accessible: true,
        rdma_capable: false,
    };

    // 1. Version range exhaustion (`SequenceExhausted`) & non-wasting capacity check.
    let mut bounded = CacheCore::new_with_version_range(
        0,
        frame_size,
        vec![desc0, desc1],
        2,
        10,
        12,
        Tracked(_src.instance.borrow()),
    );
    assert!(
        bounded.begin_write(TierId(0), ObjectKey([9, 9, 9, 9])).is_err()
    );
    assert!(bounded.next_version_seq == 10);
    assert!(bounded.allocate_version() == Ok(VersionId { origin_node: 0, seq: 10 }));
    assert!(bounded.allocate_version() == Ok(VersionId { origin_node: 0, seq: 11 }));
    assert!(bounded.allocate_version() == Err(CoreError::SequenceExhausted));

    // 2. Two-custom-tier CacheCore lifecycle starting empty and calling `register_tier`.
    let mut core = CacheCore::new(0, frame_size, Vec::new(), 2, Tracked(_src.instance.borrow()));
    let tracked host_source = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host_ctx = host::HostContext::new(Tracked(host_source));
    let tl = core.register_timeline(TimelineKind::Custom(100)).unwrap();

    let (slab0, tokens0) = host::alloc_slab(
        &host_ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        2,
        frame_size,
        4096,
    ).unwrap();
    let (slab1, tokens1) = host::alloc_slab(
        &host_ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        1,
        frame_size,
        4096,
    ).unwrap();
    let (_slab_bad, tokens_bad) = host::alloc_slab(
        &host_ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        1,
        frame_size,
        4096,
    ).unwrap();
    let ghost bad_slots = Seq::new(1nat, |i: int| _slab_bad.slot_ref(i as nat));
    let bad_slab_res = core.register_tier_slab(TierId(2), Ghost(bad_slots), 1, tokens_bad);
    assert!(bad_slab_res == Err(CoreError::UnsupportedTier));

    let ghost slots0 = Seq::new(2nat, |i: int| slab0.slot_ref(i as nat));
    let ghost slots1 = Seq::new(1nat, |i: int| slab1.slot_ref(i as nat));
    let (tier0, first0) = core.register_tier(desc0, Ghost(slots0), 2, tokens0).unwrap();
    let (tier1, first1) = core.register_tier(desc1, Ghost(slots1), 1, tokens1).unwrap();
    assert!(tier0 == TierId(0) && first0 == FrameId(0));
    assert!(tier1 == TierId(1) && first1 == FrameId(2));
    assert!(core.frames.ready_queue[0].len() == 2);
    assert!(core.frames.ready_queue[1].len() == 1);

    // 3. `begin_write` / `abort_write` returns frame to ready_queue.
    let key_abort = ObjectKey([10, 20, 30, 40]);
    let (handle_abort, tok_abort) = core.begin_write(tier0, key_abort).unwrap();
    assert!(core.frames.ready_queue[0].len() == 1);
    core.abort_write(handle_abort, tok_abort);
    assert!(core.frames.ready_queue[0].len() == 2);

    // 4. Write and publish into tier0 using ergonomic single-key `CacheCore` transactions.
    let key = ObjectKey([1, 2, 3, 4]);
    let (handle, Tracked(mut dst_tok)) = core.begin_write(tier0, key).unwrap();
    let slice0 = slab0.slot_slice(handle.frame.0 as usize);
    let payload = vec![0x5Au8; frame_size];
    let Tracked(comp) = host::write_slice(
        &mut host_ctx,
        Tracked(_src.instance.borrow()),
        &slice0,
        &payload,
        Tracked(&mut dst_tok),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    let ticket = core.issue_ticket(tl, Tracked(comp.submitted)).unwrap();
    core.mark_timeline_complete(tl, ticket.seq(), Tracked(comp.knows));
    let write_frame = handle.frame;
    let write_version = handle.version;
    core.finish_write(
        handle,
        Tracked(dst_tok),
        ticket,
        Ghost(
            ValueCertificate {
                slot: slice0.slot_ref(),
                writer: core.timelines.events@[(tl, ticket.sequence())],
                bytes: payload@,
                proof: ValueProof::HostWrite,
            },
        ),
    );
    assert!(core.lookup_tier(key, tier0) == Some((write_frame, write_version)));
    assert!(core.lookup_tier(key, tier1).is_none());
    assert!(core.lookup_best(key, &[tier1, tier0]) == Some((write_frame, tier0, write_version)));

    // 5. `acquire_local_lease` / `release_local_lease` with single-key `ShareCap` & O(1) slot reuse.
    let (h1, v1, _c1, cap1) = core.acquire_local_lease(
        write_frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    let (h2, v2, _c2, cap2) = core.acquire_local_lease(
        write_frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(v1 == write_version && v2 == write_version);
    assert!(h1.slot != h2.slot);
    assert!(core.free_lease_slots.is_empty());
    let extra_lease = core.acquire_local_lease(
        write_frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    );
    assert!(extra_lease.is_err());
    core.release_local_lease(h1, cap1, Ticket::NONE, Tracked(_src.instance.borrow())).unwrap();
    assert!(core.free_lease_slots.len() == 1);
    let (h3, _v3, _c3, cap3) = core.acquire_local_lease(
        write_frame,
        false,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(h3.slot == h1.slot);
    assert!(h3.generation > h1.generation);
    core.release_local_lease(h2, cap2, Ticket::NONE, Tracked(_src.instance.borrow())).unwrap();
    core.release_local_lease(h3, cap3, Ticket::NONE, Tracked(_src.instance.borrow())).unwrap();
    assert!(core.free_lease_slots.len() == 2);

    // 6. `begin_copy_hop` -> `abort_copy_hop`, then `begin_copy_hop` -> `finish_copy_hop`,
    //    then atomic rollback when `tier1` is exhausted.
    let (hop_abort, src_cap_abort, dst_tok_abort) = core.begin_copy_hop(
        write_frame,
        tier1,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(core.frames.ready_queue[1].len() == 0);
    core.abort_copy_hop(
        hop_abort,
        src_cap_abort,
        Ticket::NONE,
        dst_tok_abort,
        Tracked(_src.instance.borrow()),
    );
    assert!(core.frames.ready_queue[1].len() == 1);
    assert!(core.free_lease_slots.len() == 2);

    let (hop, Tracked(src_cap), Tracked(mut copy_dst_tok)) = core.begin_copy_hop(
        write_frame,
        tier1,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    ).unwrap();
    assert!(hop.dst_frame == first1);
    let dst_slice = slab1.slot_slice(0);
    let tracked mut src_perm = ReadPerm::Shared(src_cap);
    let Tracked(copy_comp) = host::copy(
        &mut host_ctx,
        Tracked(_src.instance.borrow()),
        &dst_slice,
        &slice0,
        Tracked(&mut src_perm),
        Tracked(&mut copy_dst_tok),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    let tracked returned_cap = src_perm.into_shared();
    let copy_ticket = core.issue_ticket(tl, Tracked(copy_comp.submitted)).unwrap();
    core.mark_timeline_complete(tl, copy_ticket.seq(), Tracked(copy_comp.knows));
    let ghost dst_cert = copied_certificate(
        hop.src_cert@,
        core.timelines.events@[(tl, copy_ticket.sequence())],
        dst_slice.slot_ref(),
    );
    core.finish_copy_hop(
        hop,
        Tracked(returned_cap),
        copy_ticket,
        Tracked(copy_dst_tok),
        copy_ticket,
        Ghost(dst_cert),
        Tracked(_src.instance.borrow()),
    );
    assert!(core.free_lease_slots.len() == 2);
    assert!(core.lookup_tier(key, tier1) == Some((first1, write_version)));
    assert!(core.lookup_best(key, &[tier1, tier0]) == Some((first1, tier1, write_version)));

    // Now tier1 has 0 ready frames: `begin_copy_hop` must fail with `CapacityExhausted`
    // AND atomically roll back the internal read lease on `write_frame`.
    let full_hop = core.begin_copy_hop(
        write_frame,
        tier1,
        Tracked(_src.shares.borrow_mut()),
        Tracked(_src.instance.borrow()),
    );
    assert!(full_hop.is_err());
    assert!(core.free_lease_slots.len() == 2);
    assert!(core.frames.metas[write_frame.0 as usize].active_local_reads == 0);

    // 7. `evict_frame` on both tiers + `reclaim_all_draining`.
    core.evict_frame(write_frame).unwrap();
    core.evict_frame(first1).unwrap();
    assert!(core.lookup_best(key, &[tier0, tier1]).is_none());
    let reclaimed = core.reclaim_all_draining();
    assert!(reclaimed == 2);
    assert!(core.frames.ready_queue[0].len() == 2);
    assert!(core.frames.ready_queue[1].len() == 1);
}

fn run_sibling_shard_version_range_import(_src: &mut Sources) {
    let frame_size = 4096usize;
    let mut core = CacheCore::new_with_version_range(
        0,
        frame_size,
        Vec::new(),
        2,
        1,
        100,
        Tracked(_src.instance.borrow()),
    );
    let tracked host_source = _src.events.borrow_mut().mint(EventSourceKind::Host);
    let mut host_ctx = host::HostContext::new(Tracked(host_source));
    let tl = core.register_timeline(TimelineKind::Custom(100)).unwrap();

    let (slab0, tokens0) = host::alloc_slab(
        &host_ctx,
        Tracked(_src.instance.borrow()),
        Tracked(_src.allocations.borrow_mut()),
        2,
        frame_size,
        4096,
    ).unwrap();
    let desc0 = TierDescriptor {
        id: TierId(0),
        kind: TierKind::HostDram,
        device_id: None,
        host_accessible: true,
        rdma_capable: false,
    };
    let ghost slots0 = Seq::new(2nat, |i: int| slab0.slot_ref(i as nat));
    let (tier0, _first0) = core.register_tier(desc0, Ghost(slots0), 2, tokens0).unwrap();

    // Unminted own-range versions on the same node MUST still be rejected.
    let own_unminted = VersionId { origin_node: 0, seq: 50 };
    match core.begin_write_version(tier0, ObjectKey([10, 0, 0, 1]), own_unminted) {
        Err(CoreError::InvalidArgument) => {},
        _ => assert!(false),
    }

    // Sibling-shard higher-range versions on the same node MUST succeed cleanly.
    let sibling_ver = VersionId { origin_node: 0, seq: 150 };
    let key_sib = ObjectKey([10, 0, 0, 2]);
    let (handle, Tracked(mut dst_tok)) = core.begin_write_version(
        tier0,
        key_sib,
        sibling_ver,
    ).unwrap();
    let slice0 = slab0.slot_slice(handle.frame.0 as usize);
    let payload = vec![0x3Cu8; frame_size];
    let Tracked(comp) = host::write_slice(
        &mut host_ctx,
        Tracked(_src.instance.borrow()),
        &slice0,
        &payload,
        Tracked(&mut dst_tok),
        Tracked(&core.ledger.borrow().submitted),
        Tracked(&core.ledger.borrow().knows),
        Ghost(core.citations()),
    );
    let ticket = core.issue_ticket(tl, Tracked(comp.submitted)).unwrap();
    core.mark_timeline_complete(tl, ticket.seq(), Tracked(comp.knows));
    let written_frame = handle.frame;
    core.finish_write(
        handle,
        Tracked(dst_tok),
        ticket,
        Ghost(
            ValueCertificate {
                slot: slice0.slot_ref(),
                writer: core.timelines.events@[(tl, ticket.sequence())],
                bytes: payload@,
                proof: ValueProof::HostWrite,
            },
        ),
    );
    assert!(core.lookup_tier(key_sib, tier0) == Some((written_frame, sibling_ver)));
    assert!(core.lookup_best(key_sib, &[tier0]) == Some((written_frame, tier0, sibling_ver)));

    // Local minting continues independently in [1, 100) without colliding with 150.
    let (h_local, tok_local) = core.begin_write(tier0, ObjectKey([10, 0, 0, 3])).unwrap();
    assert!(h_local.version == VersionId { origin_node: 0, seq: 1 });
    core.abort_write(h_local, tok_local);
}

} // verus!
#[test]
fn frame_sync_meta_layout_64_bytes() {
    assert_eq!(size_of::<FrameSyncMeta>(), 64);
    assert_eq!(align_of::<FrameSyncMeta>(), 64);
    let storage = 1u64 << 40;
    let metadata = storage / (256 << 10) * size_of::<FrameSyncMeta>() as u64;
    assert_eq!(metadata, 256 << 20);
    assert_eq!(
        format!("{:.3}%", 100.0 * metadata as f64 / storage as f64),
        "0.024%"
    );
}

#[test]
fn version_id_and_ticket_packing_round_trip() {
    for origin_node in [0, 1, 0xaaaa, u16::MAX] {
        for seq in [0, 1, 1 << 47, MAX_VERSION_SEQ] {
            let version = VersionId { origin_node, seq };
            assert_eq!(VersionId::unpack(version.pack()), version);
        }
    }
    for raw in [0, u64::MAX, 0x1234_abcd_5678_ef90] {
        assert_eq!(VersionId::unpack(raw).pack(), raw);
    }
    assert!(Ticket::NONE.is_none());
    assert_eq!(VersionId::NONE.pack(), 0);
    for timeline in 0..MAX_TIMELINES as u8 {
        for seq in [1, 255, 1 << 55, MAX_TICKET_SEQ] {
            let ticket = Ticket::pack(timeline, seq);
            assert_eq!((ticket.timeline(), ticket.seq()), (timeline, seq));
            assert!(!ticket.is_none());
        }
    }
}

#[test]
fn sliding_bitmap_out_of_order_stress() {
    use std::collections::BTreeSet;
    let mut pending = BTreeSet::new();
    let mut bitmap = SlidingBitmap::new();
    let mut prefix = 0;
    let mut expected = 0;
    for window in 0..10 {
        let mut offsets: Vec<u64> = (1..=512).collect();
        match window % 3 {
            0 => offsets.reverse(),
            1 => offsets.sort_by_key(|n| (n % 2, *n)),
            _ => offsets.sort_by_key(|n| (n * 193) % 512),
        }
        for n in offsets {
            let completed = window * 512 + n;
            pending.insert(completed);
            let before = expected;
            while pending.remove(&(expected + 1)) {
                expected += 1;
            }
            let advanced = bitmap.mark_and_advance(&mut prefix, completed);
            assert_eq!((prefix, advanced), (expected, expected - before));
        }
    }
    assert_eq!(prefix, 5120);
}

#[test]
fn dynamic_custom_tier_id_registration_and_catalog_best_replica() {
    let mut src = sources();
    run_custom_tier_lifecycle(&mut src);
}

#[test]
fn cache_core_two_custom_tiers_end_to_end() {
    let mut src = sources();
    run_cache_core_end_to_end(&mut src);
}

#[test]
fn sibling_shard_version_range_import() {
    let mut src = sources();
    run_sibling_shard_version_range_import(&mut src);
}
