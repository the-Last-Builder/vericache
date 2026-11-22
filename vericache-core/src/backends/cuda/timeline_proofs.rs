use super::timeline::{event_ring_valid, ring_latest_covers};
use super::*;
use foundation::tracked::{closure_from, knowledge_of, EventInfo};
use transfer::cuda::{self, CudaStream};
use vstd::prelude::*;

verus! {

pub(super) proof fn lemma_empty_imap_submap(m: IMap<TransferEvent, EventInfo<TransferDomain>>)
    ensures
        IMap::<TransferEvent, EventInfo<TransferDomain>>::empty().submap_of(m),
{
    broadcast use vstd::imap::group_imap_lemmas;

}

/// After `cuda::memcpy_async` temporarily borrows and restores `core.ledger.submitted`
/// with identical `map()` and `instance_id()`, `core.well_formed()` still holds.
pub(super) proof fn lemma_core_wf_same_submitted_map(before: &CacheCore, after: &CacheCore)
    requires
        before.well_formed(),
        after.node_id == before.node_id,
        after.tiers == before.tiers,
        after.timelines == before.timelines,
        after.frames == before.frames,
        after.catalog == before.catalog,
        after.version_seq_start == before.version_seq_start,
        after.next_version_seq == before.next_version_seq,
        after.version_seq_limit == before.version_seq_limit,
        after.active_leases == before.active_leases,
        after.free_lease_slots == before.free_lease_slots,
        after.lease_generation == before.lease_generation,
        after.spec == before.spec,
        after.version_registry == before.version_registry,
        after.ledger@.submitted.instance_id() == before.ledger@.submitted.instance_id(),
        after.ledger@.submitted.map() == before.ledger@.submitted.map(),
        after.ledger@.knows == before.ledger@.knows,
        after.ledger@.succeeded == before.ledger@.succeeded,
        after.ledger@.revoked == before.ledger@.revoked,
    ensures
        after.well_formed(),
        after.citations() == before.citations(),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    assert(after.citations() =~= before.citations());
    assert(after.ledger@.valid(after.instance_id(), after.node()));
    assert(complete_receipts(after.ledger@));
    assert(after.timelines.well_formed(after.instance_id(), after.node(), &after.ledger@));
    assert(after.kernel_view().extends(before.kernel_view()));
    after.frames.lemma_timeline_progress(before.kernel_view(), after.kernel_view());
    assert forall|f: FrameId|
        f.0 < after.frames.metas.len() && (#[trigger] after.frames.meta(f)).state
            == STATE_PUBLISHED implies {
        let g = after.frames.frame(f);
        g.cert matches Some(cert) && {
            &&& replica_valid(after.ledger@, after.node(), cert)
            &&& event_succeeded(after.ledger@, after.node(), cert.writer)
            &&& after.ledger@.citations(after.node()).contains(cert.writer)
            &&& g.bound_version matches Some(ver)
            &&& after.frames.versions@.contains_key(ver)
            &&& after.frames.versions@[ver] == cert.bytes
        }
    } by {
        assert(before.frames.valid_frame(f, before.kernel_view()));
        let cert = before.frames.frame(f).cert.unwrap();
        replica_survives_disjoint_writes(before.ledger@, after.ledger@, after.node(), cert);
    }
}

pub(super) proof fn lemma_cuda_timeline_init_wf(
    ring: &Seq<EventSlot>,
    table: &TimelineTable,
    t: u8,
    device: DeviceId,
    node: NodeId,
    ledger: &KnowledgeLedger,
    newest: int,
)
    requires
        t < table.count <= MAX_TIMELINES,
        table.next_seq[t as int] == 1,
        table.confirmed_seq[t as int] == 0,
        0 < ring.len() <= u32::MAX,
        0 <= newest < ring.len(),
        forall|j: int|
            0 <= j < ring.len() ==> {
                let s = #[trigger] ring[j];
                &&& s.event.well_formed()
                &&& s.event.device() == device
                &&& s.event.observer() == node
                &&& s.event.infos().instance_id() == ledger.submitted.instance_id()
                &&& s.event.recorded() == ISet::empty()
                &&& s.event.infos().map() == IMap::empty()
                &&& !s.in_use
                &&& s.recorded_seq == 0
            },
    ensures
        event_ring_valid(*ring, table, t, device, node, ledger),
        ring_latest_covers(*ring, newest),
{
    broadcast use vstd::imap::group_imap_lemmas;

    reveal(event_ring_valid);
    reveal(ring_latest_covers);
}

pub(super) proof fn lemma_event_ring_index(
    ring: Seq<EventSlot>,
    table: &TimelineTable,
    t: u8,
    device: DeviceId,
    node: NodeId,
    ledger: &KnowledgeLedger,
    idx: int,
)
    requires
        event_ring_valid(ring, table, t, device, node, ledger),
        0 <= idx < ring.len(),
    ensures
        ring[idx].valid(table, t, device, node, ledger),
{
    reveal(event_ring_valid);
}

pub(super) proof fn lemma_event_ring_after_record(
    old_ring: Seq<EventSlot>,
    new_ring: Seq<EventSlot>,
    table: &TimelineTable,
    t: u8,
    device: DeviceId,
    node: NodeId,
    ledger: &KnowledgeLedger,
    idx: int,
)
    requires
        event_ring_valid(old_ring, table, t, device, node, ledger),
        0 <= idx < old_ring.len(),
        new_ring.len() == old_ring.len(),
        forall|j: int| 0 <= j < old_ring.len() && j != idx ==> new_ring[j] == old_ring[j],
        new_ring[idx].valid(table, t, device, node, ledger),
        new_ring[idx].recorded_seq == table.next_seq[t as int] - 1,
    ensures
        event_ring_valid(new_ring, table, t, device, node, ledger),
        ring_latest_covers(new_ring, idx),
{
    reveal(event_ring_valid);
    reveal(ring_latest_covers);
    assert forall|j: int| 0 <= j < new_ring.len() implies (#[trigger] new_ring[j]).recorded_seq
        <= new_ring[idx].recorded_seq by {
        if j != idx {
            assert(old_ring[j].valid(table, t, device, node, ledger));
        }
    }
}

pub(super) proof fn lemma_stream_wait_preserves_submap(
    a: IMap<TransferEvent, EventInfo<TransferDomain>>,
    b: IMap<TransferEvent, EventInfo<TransferDomain>>,
    sup: IMap<TransferEvent, EventInfo<TransferDomain>>,
    ta: ISet<TransferEvent>,
    tb: ISet<TransferEvent>,
)
    requires
        a.submap_of(sup),
        b.submap_of(sup),
        ta.subset_of(a.dom()),
        tb.subset_of(b.dom()),
    ensures
        a.union_prefer_right(b).submap_of(sup),
        ta.union(tb).subset_of(a.union_prefer_right(b).dom()),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    let u = a.union_prefer_right(b);
    assert(u.submap_of(sup)) by {
        assert forall|k: TransferEvent| #[trigger]
            u.dom().contains(k) implies #[trigger] sup.dom().contains(k) && u[k] == sup[k] by {
            if b.dom().contains(k) {
                assert(sup.dom().contains(k));
                assert(b[k] == sup[k]);
            } else {
                assert(a.dom().contains(k));
                assert(sup.dom().contains(k));
                assert(a[k] == sup[k]);
            }
        }
    }
}

pub(super) proof fn lemma_event_ready_observes_prefix(
    slot: &EventSlot,
    table: &TimelineTable,
    t: u8,
    device: DeviceId,
    node: NodeId,
    ledger: &KnowledgeLedger,
    k: Knows,
    prefix: u64,
)
    requires
        table.bounds(),
        slot.valid(table, t, device, node, ledger),
        t < table.count <= MAX_TIMELINES,
        k.set() == knowledge_of(
            node,
            closure_from(slot.event.infos().map(), slot.event.recorded()),
        ),
        prefix == if slot.recorded_seq > table.confirmed_seq[t as int] {
            slot.recorded_seq
        } else {
            table.confirmed_seq[t as int]
        },
    ensures
        table.confirmed_seq[t as int] <= prefix < table.next_seq[t as int],
        forall|s: u64|
            table.confirmed_seq[t as int] < s <= prefix ==> #[trigger] table.observed(t, s, node, k)
                || table.observed(t, s, node, ledger.knows),
{
    broadcast use vstd::iset::group_iset_lemmas;

    assert forall|s: u64|
        table.confirmed_seq[t as int] < s <= prefix implies #[trigger] table.observed(
        t,
        s,
        node,
        k,
    ) by {
        assert(closure_from(slot.event.infos().map(), slot.event.recorded()).contains(
            table.events@[(t, s)],
        ));
    }
}

pub(super) proof fn lemma_event_ring_after_poll(
    old_ring: Seq<EventSlot>,
    new_ring: Seq<EventSlot>,
    newest: int,
    old_table: &TimelineTable,
    new_table: &TimelineTable,
    t: u8,
    device: DeviceId,
    node: NodeId,
    old_ledger: &KnowledgeLedger,
    new_ledger: &KnowledgeLedger,
    idx: int,
)
    requires
        event_ring_valid(old_ring, old_table, t, device, node, old_ledger),
        ring_latest_covers(old_ring, newest),
        0 <= idx < old_ring.len(),
        new_ring.len() == old_ring.len(),
        forall|j: int| 0 <= j < old_ring.len() && j != idx ==> new_ring[j] == old_ring[j],
        new_ring[idx].event == old_ring[idx].event,
        new_ring[idx].generation == old_ring[idx].generation,
        new_ring[idx].recorded_seq == old_ring[idx].recorded_seq,
        !new_ring[idx].in_use,
        new_ring[idx].recorded_seq <= new_table.confirmed_seq[t as int],
        new_table.count == old_table.count,
        new_table.next_seq == old_table.next_seq,
        new_table.events == old_table.events,
        new_table.confirmed_seq[t as int] >= old_table.confirmed_seq[t as int],
        new_ledger.submitted == old_ledger.submitted,
    ensures
        event_ring_valid(new_ring, new_table, t, device, node, new_ledger),
        ring_latest_covers(new_ring, newest),
{
    reveal(event_ring_valid);
    reveal(ring_latest_covers);
    assert forall|j: int| 0 <= j < new_ring.len() implies #[trigger] new_ring[j].valid(
        new_table,
        t,
        device,
        node,
        new_ledger,
    ) by {
        assert(old_ring[j].valid(old_table, t, device, node, old_ledger));
    }
}

pub(super) proof fn lemma_cuda_extends_refl(core: &CacheCore)
    requires
        core.well_formed(),
    ensures
        core.kernel_view().extends(core.kernel_view()),
        core.ledger@.knows.set().subset_of(core.ledger@.knows.set()),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

}

pub(super) proof fn lemma_cuda_extends_trans(
    a: KernelView,
    b: KernelView,
    c: KernelView,
    ka: ISet<(NodeId, TransferEvent)>,
    kb: ISet<(NodeId, TransferEvent)>,
    kc: ISet<(NodeId, TransferEvent)>,
)
    requires
        b.extends(a),
        c.extends(b),
        ka.subset_of(kb),
        kb.subset_of(kc),
    ensures
        c.extends(a),
        ka.subset_of(kc),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

}

pub(super) proof fn lemma_cuda_memcpy_admitted(
    stream: CudaStream,
    src_perm: ReadPerm,
    dst_tok: SlotToken,
    core: &CacheCore,
)
    requires
        core.well_formed(),
        stream.well_formed(),
        stream.source().id() == core.instance_id(),
        stream.source().node() == core.node(),
        stream.infos().map().submap_of(core.ledger@.submitted.map()),
        stream.tail().subset_of(stream.infos().dom()),
        read_admitted(
            src_perm.frontier(),
            closure_from(core.ledger@.submitted.map(), stream.tail().union(core.citations())),
        ),
        write_admitted(
            slot_frontier(dst_tok),
            closure_from(core.ledger@.submitted.map(), stream.tail().union(core.citations())),
        ),
    ensures
        stream.infos().map().union_prefer_right(core.ledger@.submitted.map())
            == core.ledger@.submitted.map(),
        cuda::copy_admitted(
            stream,
            src_perm,
            dst_tok,
            core.ledger@.submitted,
            core.ledger@.knows,
            core.citations(),
        ),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    lemma_core_wf_facts(core);
    let merged = stream.infos().map().union_prefer_right(core.ledger@.submitted.map());
    assert(merged =~= core.ledger@.submitted.map()) by {
        assert forall|k: TransferEvent| #[trigger]
            merged.dom().contains(k) implies core.ledger@.submitted.dom().contains(k) && merged[k]
            == core.ledger@.submitted.map()[k] by {
            if !core.ledger@.submitted.dom().contains(k) {
                assert(stream.infos().dom().contains(k));
            }
        }
    }
    assert(core.citations().subset_of(core.ledger@.submitted.dom()));
    assert(knowledge_of(core.node(), core.citations()).subset_of(core.ledger@.knows.set()));
}

pub(super) proof fn lemma_cuda_memcpy_receipt_wf(
    before_stream: CudaStream,
    after_stream: CudaStream,
    before_core: &CacheCore,
    dst_slot: SlotRef,
    src_slot: SlotRef,
    ev: TransferEvent,
    receipt: Submitted,
)
    requires
        before_core.well_formed(),
        before_stream.well_formed(),
        before_stream.source().id() == before_core.instance_id(),
        before_stream.source().node() == before_core.node(),
        before_stream.infos().map().submap_of(before_core.ledger@.submitted.map()),
        before_stream.tail().subset_of(before_stream.infos().dom()),
        source_fresh_for_ledger(before_stream.source(), before_core.ledger@),
        ev == before_stream.source().next_event(),
        receipt.key() == ev,
        receipt.instance_id() == before_core.instance_id(),
        receipt.value().access == copy_access(src_slot, dst_slot),
        receipt.value().preds == closure_from(
            before_stream.infos().map().union_prefer_right(before_core.ledger@.submitted.map()),
            before_stream.tail().union(before_core.citations()),
        ),
        before_core.disjoint_from_published(ISet::empty().insert(dst_slot.loc)),
        after_stream.tail() == ISet::empty().insert(ev),
        after_stream.infos().map() == IMap::empty().insert(ev, receipt.value()),
    ensures
        !before_core.ledger@.submitted.dom().contains(ev),
        receipt.key().observer == before_core.node(),
        receipt.value().preds == closure_from(
            before_core.ledger@.submitted.map(),
            before_stream.tail().union(before_core.citations()),
        ),
        receipt.value().preds.subset_of(before_core.ledger@.submitted.dom()),
        before_core.disjoint_from_published(receipt.value().access.writes),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    lemma_core_wf_facts(before_core);
    lemma_fresh_source_event_absent(before_stream.source(), before_core.ledger@);
    let merged = before_stream.infos().map().union_prefer_right(
        before_core.ledger@.submitted.map(),
    );
    assert(merged =~= before_core.ledger@.submitted.map());
    let waits = before_stream.tail().union(before_core.citations());
    assert(waits.subset_of(before_core.ledger@.submitted.dom()));
    assert forall|e: TransferEvent| #[trigger]
        receipt.value().preds.contains(e) implies before_core.ledger@.submitted.dom().contains(
        e,
    ) by {
        if !waits.contains(e) {
            let s = choose|s: TransferEvent|
                waits.contains(s)
                    && #[trigger] before_core.ledger@.submitted.map()[s].preds.contains(e);
            assert(before_core.ledger@.submitted.dom().contains(s));
            assert(before_core.ledger@.submitted.map()[s].preds.subset_of(
                before_core.ledger@.submitted.dom(),
            ));
        }
    }
    assert(receipt.value().access.writes =~= ISet::empty().insert(dst_slot.loc));
}

pub(super) proof fn lemma_cuda_timeline_after_memcpy(
    before_stream: CudaStream,
    after_stream: CudaStream,
    ring: Seq<EventSlot>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    t: u8,
    device: DeviceId,
    ev: TransferEvent,
    receipt: Submitted,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        before_stream.well_formed(),
        after_stream.well_formed(),
        after_stream.device() == device,
        after_stream.source().source() == before_stream.source().source(),
        after_stream.source().node() == before_stream.source().node(),
        after_stream.source().id() == before_core.instance_id(),
        after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
        source_fresh_for_ledger(before_stream.source(), before_core.ledger@),
        before_stream.source().node() == before_core.node(),
        before_stream.infos().map().submap_of(before_core.ledger@.submitted.map()),
        before_stream.tail().subset_of(before_stream.infos().dom()),
        before_core.timelines.prefix_in(
            t,
            (before_core.timelines.next_seq[t as int] - 1) as u64,
            closure_from(before_stream.infos().map(), before_stream.tail()),
        ),
        event_ring_valid(
            ring,
            &before_core.timelines,
            t,
            device,
            before_core.node(),
            &before_core.ledger@,
        ),
        t < before_core.timelines.count <= MAX_TIMELINES,
        before_core.timelines.next_seq[t as int] <= MAX_TICKET_SEQ,
        ev == before_stream.source().next_event(),
        receipt.key() == ev,
        receipt.value().preds == closure_from(
            before_core.ledger@.submitted.map(),
            before_stream.tail().union(before_core.citations()),
        ),
        after_stream.tail() == ISet::empty().insert(ev),
        after_stream.infos().map() == IMap::empty().insert(ev, receipt.value()),
        after_core.node_id == before_core.node_id,
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.kinds == before_core.timelines.kinds,
        after_core.timelines.confirmed_seq == before_core.timelines.confirmed_seq,
        after_core.timelines.next_seq@ == before_core.timelines.next_seq@.update(
            t as int,
            (before_core.timelines.next_seq[t as int] + 1) as u64,
        ),
        after_core.timelines.events@ == before_core.timelines.events@.insert(
            (t, before_core.timelines.next_seq[t as int]),
            ev,
        ),
        after_core.ledger@.submitted.instance_id() == before_core.ledger@.submitted.instance_id(),
        after_core.ledger@.submitted.map() == before_core.ledger@.submitted.map().insert(
            ev,
            receipt.value(),
        ),
        before_core.ledger@.submitted.map().submap_of(after_core.ledger@.submitted.map()),
    ensures
        source_fresh_for_ledger(after_stream.source(), after_core.ledger@),
        after_stream.infos().map().submap_of(after_core.ledger@.submitted.map()),
        after_stream.tail().subset_of(after_stream.infos().dom()),
        after_core.timelines.prefix_in(
            t,
            (after_core.timelines.next_seq[t as int] - 1) as u64,
            closure_from(after_stream.infos().map(), after_stream.tail()),
        ),
        event_ring_valid(
            ring,
            &after_core.timelines,
            t,
            device,
            after_core.node(),
            &after_core.ledger@,
        ),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    lemma_source_fresh_stepped(
        before_stream.source(),
        after_stream.source(),
        before_core.ledger@,
        after_core.ledger@,
        ev,
    );
    let old_closure = closure_from(before_stream.infos().map(), before_stream.tail());
    let new_closure = closure_from(after_stream.infos().map(), after_stream.tail());
    assert(old_closure.subset_of(receipt.value().preds)) by {
        assert forall|s: TransferEvent|
            before_stream.tail().contains(
                s,
            ) implies #[trigger] before_core.ledger@.submitted.map()[s]
            == before_stream.infos().map()[s] by {
            assert(before_stream.infos().map().dom().contains(s));
            assert(before_core.ledger@.submitted.map().dom().contains(s));
        }
        crate::provenance::closure_monotonic(
            before_stream.infos().map(),
            before_stream.tail(),
            before_core.ledger@.submitted.map(),
            before_stream.tail().union(before_core.citations()),
        );
    }
    assert(receipt.value().preds.subset_of(new_closure)) by {
        assert(after_stream.tail().contains(ev));
        assert(after_stream.infos().map()[ev] == receipt.value());
    }
    let s_new = before_core.timelines.next_seq[t as int];
    assert forall|q: u64| 1 <= q <= s_new implies #[trigger] new_closure.contains(
        after_core.timelines.events@[(t, q)],
    ) by {
        if q == s_new {
            assert(after_core.timelines.events@[(t, q)] == ev);
            assert(after_stream.tail().contains(ev));
        } else {
            assert(q <= s_new - 1);
            assert(after_core.timelines.events@[(t, q)] == before_core.timelines.events@[(t, q)]);
            assert(old_closure.contains(before_core.timelines.events@[(t, q)]));
        }
    }
    reveal(event_ring_valid);
    assert forall|j: int| 0 <= j < ring.len() implies #[trigger] ring[j].valid(
        &after_core.timelines,
        t,
        device,
        after_core.node(),
        &after_core.ledger@,
    ) by {
        assert(ring[j].valid(
            &before_core.timelines,
            t,
            device,
            before_core.node(),
            &before_core.ledger@,
        ));
        vstd::imap_lib::lemma_submap_of_trans(
            ring[j].event.infos().map(),
            before_core.ledger@.submitted.map(),
            after_core.ledger@.submitted.map(),
        );
        assert forall|q: u64| 1 <= q <= ring[j].recorded_seq implies closure_from(
            ring[j].event.infos().map(),
            ring[j].event.recorded(),
        ).contains(#[trigger] after_core.timelines.events@[(t, q)]) by {
            assert(q < s_new);
            assert(after_core.timelines.events@[(t, q)] == before_core.timelines.events@[(t, q)]);
            assert(closure_from(ring[j].event.infos().map(), ring[j].event.recorded()).contains(
                before_core.timelines.events@[(t, q)],
            ));
        }
        assert(after_core.timelines.prefix_in(
            t,
            ring[j].recorded_seq,
            closure_from(ring[j].event.infos().map(), ring[j].event.recorded()),
        ));
    }
}

/// Pure ledger monotonicity: citations grow when `submitted` and `knows` grow.
pub(super) proof fn lemma_ledger_citations_monotone(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
)
    requires
        before.submitted.map().submap_of(after.submitted.map()),
        before.knows.set().subset_of(after.knows.set()),
    ensures
        before.citations(node).subset_of(after.citations(node)),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

}

/// Pure ledger preservation for polling (only `knows` grows).
pub(super) proof fn lemma_ledger_preserves_dest_poll(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    loc: SlotLoc,
)
    requires
        after.submitted == before.submitted,
        after.succeeded == before.succeeded,
        revoked_submap(before.revoked, after.revoked),
        before.knows.set().subset_of(after.knows.set()),
    ensures
        before.citations(node).subset_of(after.citations(node)),
        preserves_destination(before, after, node, loc),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

}

/// Pure ledger preservation for a disjoint step.
pub(super) proof fn lemma_ledger_preserves_dest_disjoint(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    loc: SlotLoc,
    disjoint_write_loc: Option<SlotLoc>,
)
    requires
        before.submitted.instance_id() == after.submitted.instance_id(),
        before.submitted.map().submap_of(after.submitted.map()),
        revoked_submap(before.revoked, after.revoked),
        before.knows.set().subset_of(after.knows.set()),
        before.succeeded.dom().subset_of(after.succeeded.dom()),
        forall|e: TransferEvent| #[trigger]
            after.submitted.dom().contains(e) && !before.submitted.dom().contains(e)
                ==> match disjoint_write_loc {
                None => false,
                Some(wloc) => after.submitted.map()[e].access.writes == ISet::empty().insert(wloc),
            },
        disjoint_write_loc != Some(loc),
    ensures
        before.citations(node).subset_of(after.citations(node)),
        preserves_destination(before, after, node, loc),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    assert forall|e: TransferEvent| #[trigger]
        after.submitted.dom().contains(e) && !before.submitted.dom().contains(
            e,
        ) implies !after.submitted.map()[e].access.writes.contains(loc) by {
        if let Some(wloc) = disjoint_write_loc {
            assert(after.submitted.map()[e].access.writes == ISet::empty().insert(wloc));
        }
    }
}

/// Pure `ISet` / `closure_from` monotonicity helper (no `CacheCore` in scope).
pub(super) proof fn lemma_iset_subset_closure_union(
    infos: IMap<TransferEvent, EventInfo<TransferDomain>>,
    a: ISet<TransferEvent>,
    b: ISet<TransferEvent>,
    sub: ISet<TransferEvent>,
    x: TransferEvent,
)
    requires
        sub.subset_of(b),
    ensures
        sub.subset_of(closure_from(infos, a.union(b))),
        a.contains(x) ==> closure_from(infos, a.union(b)).contains(x),
{
    broadcast use vstd::iset::group_iset_lemmas;

}

/// Reflexivity of `IMap::submap_of` without dragging `CacheCore` into `imap` broadcasts.
pub(super) proof fn lemma_imap_submap_refl(m: IMap<TransferEvent, EventInfo<TransferDomain>>)
    ensures
        m.submap_of(m),
{
    broadcast use vstd::imap::group_imap_lemmas;

}

} // verus!
