use super::super::cuda::{
    caller_stream_valid, cuda_put_events_covered, cuda_puts_bindings_covered, cuda_puts_valid,
    lemma_cuda_puts_bindings_extract,
    timeline::{event_ring_valid, CudaTimeline},
    CudaBackend, PendingGpuAction, PendingGpuWrite,
};
use super::super::host::HostBackend;
use super::super::nvme::{
    nvme_copy_retained, nvme_pending_valid, nvme_window_wf, InFlightNvmeOp, NvmeBackend,
};
use super::super::rdma::{
    peer::PeerSession, rdma_data_events, rdma_leases_valid, rdma_window_wf, ActiveRemoteLease,
    RdmaBackend,
};
use super::*;
use transfer::cuda::CudaStream;
use transfer::rdma;
use transfer::transport::ByteTransport;
use vstd::prelude::*;

verus! {

pub proof fn lemma_core_transition_preserves_source(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    source: EventSourceToken,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        source_fresh_for_ledger(source, before.ledger@),
        allowed_source != Some(source.source()),
    ensures
        source_fresh_for_ledger(source, after.ledger@),
{
    reveal(core_transition);
    assert forall|e: TransferEvent| #[trigger]
        after.ledger@.submitted.dom().contains(e) && e.source == source.source() && e.observer
            == source.node() implies e.seq < source.next_seq() by {
        assert(before.ledger@.submitted.dom().contains(e));
    }
}

pub proof fn lemma_core_transition_preserves_host(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    host: &HostBackend,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        host.well_formed(before),
        touched_timeline != Some(host.timeline),
        allowed_source != Some(host.ctx.source@.source()),
    ensures
        host.well_formed(after),
{
    reveal(core_transition);
    assert(after.timelines.next_seq[host.timeline as int]
        == before.timelines.next_seq[host.timeline as int]);
    assert(after.timelines.confirmed_seq[host.timeline as int]
        == before.timelines.confirmed_seq[host.timeline as int]);
    lemma_core_transition_preserves_source(
        before,
        after,
        touched_timeline,
        allowed_source,
        mode,
        host.ctx.source@,
    );
    lemma_slab_bound_same_layout(
        before,
        after,
        host.tier,
        host.first_frame,
        host.num_frames,
        host.buffer.space(),
    );
}

proof fn lemma_core_transition_event_eq(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    t: u8,
    s: u64,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        t < before.timelines.count,
        touched_timeline != Some(t),
        1 <= s < before.timelines.next_seq[t as int],
    ensures
        after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)],
{
    reveal(core_transition);
}

proof fn lemma_core_transition_cuda_put_facts(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    tl: u8,
    p: PendingGpuWrite,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        touched_timeline != Some(tl),
        mode is PreserveAllInit || mode is PreserveParkedInit,
        p.valid(before, tl),
    ensures
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.next_version_seq >= before.next_version_seq,
        after.version_seq_limit == before.version_seq_limit,
        after.frames.metas.len() == before.frames.metas.len(),
        after.frames.frame_size == before.frames.frame_size,
        after.frames.tier_frame_sizes == before.frames.tier_frame_sizes,
        after.frames.meta(p.frame) == before.frames.meta(p.frame),
        after.frames.frame(p.frame) == before.frames.frame(p.frame),
        after.timelines.count == before.timelines.count,
        after.timelines.next_seq[tl as int] == before.timelines.next_seq[tl as int],
        after.timelines.events@[(tl, p.ticket.sequence())] == before.timelines.events@[(
            tl,
            p.ticket.sequence(),
        )],
        before.citations().subset_of(after.citations()),
        before.frames.versions@.submap_of(after.frames.versions@),
        preserves_destination(before.ledger@, after.ledger@, after.node(), p.slot@.loc),
        p.action matches PendingGpuAction::Publish { version, .. } ==> (
        !before.frames.versions@.contains_key(version) ==> !after.frames.versions@.contains_key(
            version,
        )),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    reveal(core_transition);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    assert(after.timelines.next_seq[tl as int] == before.timelines.next_seq[tl as int]);
    assert(before.citations().subset_of(after.citations())) by {
        assert forall|e: TransferEvent|
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger@.knows.set().contains((before.node(), e)));
        }
    }
    assert(preserves_destination(before.ledger@, after.ledger@, after.node(), p.slot@.loc)) by {
        assert forall|e: TransferEvent| #[trigger]
            after.ledger@.submitted.dom().contains(e) && !before.ledger@.submitted.dom().contains(
                e,
            ) implies !after.ledger@.submitted.map()[e].access.writes.contains(p.slot@.loc) by {
            assert(before.frames.meta(p.frame).state == STATE_INITIALIZING);
            assert(before.frames.frame(p.frame).slot.loc == p.slot@.loc);
        }
    }
}

proof fn lemma_core_transition_preserves_cuda_timeline(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    tl: &CudaTimeline,
)
    requires
        before.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        tl.well_formed(&before.timelines, before.node(), &before.ledger@),
        touched_timeline != Some(tl.timeline_id),
        allowed_source != Some(tl.stream.source().source()),
    ensures
        tl.well_formed(&after.timelines, after.node(), &after.ledger@),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    lemma_core_wf_facts(before);
    reveal(core_transition);
    let t = tl.timeline_id;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    assert(before.timelines.next_seq[t as int] >= 1);
    lemma_core_transition_preserves_source(
        before,
        after,
        touched_timeline,
        allowed_source,
        mode,
        tl.stream.source(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before.ledger@.submitted.map(),
        after.ledger@.submitted.map(),
    );
    assert(after.timelines.prefix_in(
        t,
        (after.timelines.next_seq[t as int] - 1) as u64,
        tl.closure(),
    )) by {
        assert forall|s: u64|
            1 <= s <= (after.timelines.next_seq[t as int] - 1) as u64 implies tl.closure().contains(
            #[trigger] after.timelines.events@[(t, s)],
        ) by {
            assert(1 <= s < before.timelines.next_seq[t as int]);
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(event_ring_valid);
    assert forall|i: int| 0 <= i < tl.event_ring@.len() implies #[trigger] tl.event_ring@[i].valid(
        &after.timelines,
        t,
        tl.device,
        after.node(),
        &after.ledger@,
    ) by {
        let slot = tl.event_ring@[i];
        assert(slot.valid(&before.timelines, t, tl.device, before.node(), &before.ledger@));
        vstd::imap_lib::lemma_submap_of_trans(
            slot.event.infos().map(),
            before.ledger@.submitted.map(),
            after.ledger@.submitted.map(),
        );
        let c = foundation::tracked::closure_from(slot.event.infos().map(), slot.event.recorded());
        assert(slot.recorded_seq < before.timelines.next_seq[t as int]);
        assert forall|s: u64| 1 <= s <= slot.recorded_seq implies c.contains(
            #[trigger] after.timelines.events@[(t, s)],
        ) by {
            lemma_core_transition_event_eq(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                t,
                s,
            );
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
}

proof fn lemma_core_transition_preserves_cuda_put_events(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    tl: u8,
    s: u64,
    events: Seq<TransferEvent>,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        tl < before.timelines.count,
        touched_timeline != Some(tl),
        s < before.timelines.next_seq[tl as int],
        cuda_put_events_covered(events, before.timelines.events@, tl, s),
    ensures
        cuda_put_events_covered(events, after.timelines.events@, tl, s),
{
    reveal(core_transition);
    reveal(cuda_put_events_covered);
    assert forall|k: int| 0 <= k < events.len() implies super::super::cuda::cuda_event_at_prefix(
        #[trigger] events[k],
        after.timelines.events@,
        tl,
        s,
    ) by {
        assert(super::super::cuda::cuda_event_at_prefix(
            events[k],
            before.timelines.events@,
            tl,
            s,
        ));
        let q = choose|q: u64|
            1 <= q <= s && (#[trigger] before.timelines.events@[(tl, q)]) == events[k]
                && events[k].source.succeeds_on_observe();
        assert(1 <= q < before.timelines.next_seq[tl as int]);
        assert(after.timelines.events@[(tl, q)] == events[k]);
    }
}

proof fn lemma_iset_subset_trans_local<T>(a: ISet<T>, b: ISet<T>, c: ISet<T>)
    requires
        a.subset_of(b),
        b.subset_of(c),
    ensures
        a.subset_of(c),
{
    broadcast use vstd::iset::group_iset_lemmas;

}

proof fn lemma_map_submap_get_local<K, V>(a: Map<K, V>, b: Map<K, V>, k: K)
    requires
        a.submap_of(b),
        a.contains_key(k),
    ensures
        b.contains_key(k),
        b[k] == a[k],
{
    broadcast use vstd::map::group_map_lemmas;

    assert(a.dom().contains(k));
    assert(b.dom().contains(k));
    assert(a.index(k) == b.index(k));
}

proof fn lemma_core_transition_abort_ticket_covers(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    tl: u8,
    ticket: Ticket,
    w: TransferEvent,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        tl < before.timelines.count,
        touched_timeline != Some(tl),
        ticket != Ticket::NONE,
        ticket.timeline_id() == tl,
        before.timelines.issued(tl, ticket.sequence()),
        before.timelines.ticket_covers(ticket, w),
    ensures
        after.timelines.ticket_covers(ticket, w),
{
    assert(exists|s: u64|
        1 <= s <= ticket.sequence() && #[trigger] before.timelines.events@[(
            ticket.timeline_id(),
            s,
        )] == w);
    let q = choose|q: u64|
        1 <= q <= ticket.sequence() && (#[trigger] before.timelines.events@[(
            ticket.timeline_id(),
            q,
        )]) == w;
    lemma_core_transition_event_eq(before, after, touched_timeline, allowed_source, mode, tl, q);
    assert(1 <= q <= ticket.sequence() && after.timelines.events@[(ticket.timeline_id(), q)] == w);
    assert(exists|s: u64|
        1 <= s <= ticket.sequence() && #[trigger] after.timelines.events@[(ticket.timeline_id(), s)]
            == w);
}

proof fn lemma_core_transition_preserves_one_cuda_put(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    tl: u8,
    p: PendingGpuWrite,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        touched_timeline != Some(tl),
        mode is PreserveAllInit || mode is PreserveParkedInit,
        p.valid(before, tl),
    ensures
        p.valid(after, tl),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(TimelineTable::ticket_covers);
    lemma_core_transition_cuda_put_facts(
        before,
        after,
        touched_timeline,
        allowed_source,
        mode,
        tl,
        p,
    );
    let s = p.ticket.sequence();
    let tok = before.frames.frame(p.frame).token.unwrap();
    lemma_iset_subset_trans_local(
        slot_frontier(tok).readers,
        before.citations(),
        after.citations(),
    );
    assert(after.timelines.issued(tl, s));
    match p.action {
        PendingGpuAction::Abort => {
            if let Some(w) = slot_frontier(tok).last_writer {
                if !before.citations().contains(w) {
                    lemma_core_transition_abort_ticket_covers(
                        before,
                        after,
                        touched_timeline,
                        allowed_source,
                        mode,
                        tl,
                        p.ticket,
                        w,
                    );
                }
            }
        },
        PendingGpuAction::Publish { version, proof: put_proof, .. } => {
            if before.frames.versions@.contains_key(version) {
                lemma_map_submap_get_local(
                    before.frames.versions@,
                    after.frames.versions@,
                    version,
                );
            }
            put_proof@.preserve(before.ledger@, after.ledger@, after.node());
            lemma_core_transition_preserves_cuda_put_events(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                tl,
                s,
                put_proof@.events,
            );
        },
    }
}

pub proof fn lemma_core_transition_preserves_cuda<'a>(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    cuda: &CudaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        cuda.well_formed(before),
        touched_timeline != Some(cuda.timeline.timeline_id),
        allowed_source != Some(cuda.timeline.stream.source().source()),
        mode is PreserveAllInit || mode is PreserveParkedInit,
    ensures
        cuda.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    assert(after.node_id == before.node_id && after.instance_id() == before.instance_id()
        && after.tiers == before.tiers && after.frames.same_layout(&before.frames)) by {
        reveal(core_transition);
    }
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_slab_bound_same_layout(
        before,
        after,
        cuda.gpu_tier,
        cuda.gpu_first_frame,
        cuda.gpu_num_frames,
        cuda.gpu_buffer.space(),
    );
    if let Some(ref h) = cuda.pinned_host {
        lemma_slab_bound_same_layout(
            before,
            after,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
    lemma_core_transition_preserves_cuda_timeline(
        before,
        after,
        touched_timeline,
        allowed_source,
        mode,
        &cuda.timeline,
    );
    let tl = cuda.timeline.timeline_id;
    reveal(cuda_puts_valid);
    assert forall|i: int| 0 <= i < cuda.active_gpu_puts@.len() implies (
    #[trigger] cuda.active_gpu_puts@[i]).valid(after, tl) by {
        lemma_core_transition_preserves_one_cuda_put(
            before,
            after,
            touched_timeline,
            allowed_source,
            mode,
            tl,
            cuda.active_gpu_puts@[i],
        );
    }
}

pub proof fn lemma_core_transition_preserves_nvme(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    nvme: &NvmeBackend,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        nvme.well_formed(before),
        touched_timeline != Some(nvme.timeline),
        allowed_source != Some(nvme.ring.source().source()),
        mode is PreserveAllInit || mode is PreserveUnparkedInit,
    ensures
        nvme.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_slab_bound_same_layout(
        before,
        after,
        nvme.tier,
        nvme.first_frame,
        nvme.num_frames,
        nvme.file.space(),
    );
    let t = nvme.timeline;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    reveal(nvme_window_wf);
    assert(nvme.bitmap.observations(&after.timelines, t, after.node(), &after.ledger@)) by {
        assert forall|i: int| #[trigger] nvme.bitmap.bit_set(i) implies {
            let s = (after.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            after.timelines.issued(t, s) && after.timelines.observed(
                t,
                s,
                after.node(),
                after.ledger@.knows,
            )
        } by {
            let s = (before.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            assert(before.timelines.issued(t, s));
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
            assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
        }
    }
    assert(window_matches(
        nvme.in_flight_ops,
        nvme.bitmap,
        &after.timelines,
        t,
        super::super::nvme::ring_event_map(nvme.ring),
    )) by {
        let pending = super::super::nvme::ring_event_map(nvme.ring);
        assert forall|s: u64| #[trigger]
            pending.contains_key(s) implies after.timelines.confirmed_seq[t as int] < s
            < after.timelines.next_seq[t as int] && pending[s] == after.timelines.events@[(t, s)]
            && (nvme.in_flight_ops[(s % BITMAP_CAPACITY) as int] matches Some(op) && op.sequence
            == s) by {
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(nvme_pending_valid);
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (
    #[trigger] nvme.pending_transfers@[i] matches Some(op) ==> {
        &&& op.valid(after, t, i)
        &&& nvme.ring.pending().contains_key(op.ticket.sequence())
        &&& nvme.ring.pending()[op.ticket.sequence()].event == op.copy@.writer
    }) by {
        if let Some(op) = nvme.pending_transfers@[i] {
            assert(op.valid(before, t, i));
            let s = op.ticket.sequence();
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
            assert(after.frames.meta(op.hop.dst_frame) == before.frames.meta(op.hop.dst_frame));
            assert(after.frames.frame(op.hop.dst_frame) == before.frames.frame(op.hop.dst_frame));
            assert(after.active_leases[op.hop.src_lease.slot as int]
                == before.active_leases[op.hop.src_lease.slot as int]);
            assert(before.frames.versions@.dom().contains(op.hop.version));
            assert(after.frames.versions@.dom().contains(op.hop.version));
            assert(before.citations().subset_of(after.citations())) by {
                assert forall|e: TransferEvent|
                    before.citations().contains(e) implies after.citations().contains(e) by {
                    assert(before.ledger@.knows.set().contains((before.node(), e)));
                }
            }
            vstd::imap_lib::lemma_submap_of_trans(
                op.copy@.before.submitted.map(),
                before.ledger@.submitted.map(),
                after.ledger@.submitted.map(),
            );
            assert(after.ledger@.submitted.dom().contains(op.copy@.writer));
            assert(after.ledger@.submitted.map()[op.copy@.writer]
                == before.ledger@.submitted.map()[op.copy@.writer]);
            assert(nvme_copy_retained(op.copy@, after.ledger@, after.node()));
        }
    }
}

pub proof fn lemma_core_transition_preserves_caller_stream(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    stream: CudaStream,
    dst_timeline: u8,
    backend_stream: CudaStream,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        caller_stream_valid(stream, before, dst_timeline, backend_stream),
        allowed_source != Some(stream.source().source()),
    ensures
        caller_stream_valid(stream, after, dst_timeline, backend_stream),
{
    reveal(core_transition);
    lemma_core_transition_preserves_source(
        before,
        after,
        touched_timeline,
        allowed_source,
        mode,
        stream.source(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        stream.infos().map(),
        before.ledger@.submitted.map(),
        after.ledger@.submitted.map(),
    );
}

pub proof fn lemma_core_transition_preserves_slot_disjoint_and_admitted(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    tok: SlotToken,
    loc: SlotLoc,
)
    requires
        core_transition(before, after, touched_timeline, allowed_source, mode),
        write_admitted(slot_frontier(tok), before.citations()),
        forall|f: FrameId|
            f.0 < before.frames.metas.len() ==> loc != (#[trigger] before.frames.frame(f)).slot.loc,
    ensures
        write_admitted(slot_frontier(tok), after.citations()),
        after.disjoint_from_published(ISet::empty().insert(loc)),
        forall|f: FrameId|
            f.0 < after.frames.metas.len() ==> loc != (#[trigger] after.frames.frame(f)).slot.loc,
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    reveal(core_transition);
    assert(before.citations().subset_of(after.citations())) by {
        assert forall|e: TransferEvent|
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger@.knows.set().contains((before.node(), e)));
        }
    }
    assert forall|f: FrameId| f.0 < after.frames.metas.len() implies loc != (
    #[trigger] after.frames.frame(f)).slot.loc by {
        assert(after.frames.frame(f).slot == before.frames.frame(f).slot);
    }
}

/// Prove that any sibling Host/CUDA/NVMe `core_transition` on a disjoint
/// timeline preserves `RdmaBackend::well_formed` across Host/GPU MR slabs,
/// `rdma_window_wf`, `Quarantined` / `WindowPool` slots, and `active_leases`.
pub proof fn lemma_core_transition_preserves_rdma<'a>(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    rdma_backend: &RdmaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        rdma_backend.well_formed(before),
        touched_timeline != Some(rdma_backend.timeline_id),
    ensures
        rdma_backend.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_core_timelines_wf(after);
    if let Some(ref s) = rdma_backend.host_mr {
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    if let Some(ref s) = rdma_backend.gpu_mr {
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    let pd_id = rdma_backend.window_pool.pd();
    if let Some(s) = rdma_backend.shared_host_mr {
        s.lemma_shared_wf_iff(before, pd_id);
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
        s.lemma_shared_wf_iff(after, pd_id);
    }
    if let Some(s) = rdma_backend.shared_gpu_mr {
        s.lemma_shared_wf_iff(before, pd_id);
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
        s.lemma_shared_wf_iff(after, pd_id);
    }
    let t = rdma_backend.timeline_id;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    reveal(rdma_window_wf);
    assert(rdma_backend.bitmap.observations(&after.timelines, t, after.node(), &after.ledger@)) by {
        assert forall|i: int| #[trigger] rdma_backend.bitmap.bit_set(i) implies {
            let s = (after.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            after.timelines.issued(t, s) && after.timelines.observed(
                t,
                s,
                after.node(),
                after.ledger@.knows,
            )
        } by {
            let s = (before.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            assert(before.timelines.issued(t, s));
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
            assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
        }
    }
    assert(window_matches(
        rdma_backend.in_flight_ops,
        rdma_backend.bitmap,
        &after.timelines,
        t,
        rdma_data_events(rdma::CqSnapshot::of(rdma_backend.cq)),
    )) by {
        let pending = rdma_data_events(rdma::CqSnapshot::of(rdma_backend.cq));
        assert forall|s: u64| #[trigger]
            pending.contains_key(s) implies after.timelines.confirmed_seq[t as int] < s
            < after.timelines.next_seq[t as int] && pending[s] == after.timelines.events@[(t, s)]
            && (rdma_backend.in_flight_ops[(s % BITMAP_CAPACITY) as int] matches Some(op)
            && op.sequence == s) by {
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < rdma_backend.active_leases@.len() implies (
    #[trigger] rdma_backend.active_leases@[i]).valid(
        &rdma_backend.window_pool,
        &rdma_backend.qp,
        after,
    ) by {
        let lease = rdma_backend.active_leases@[i];
        assert(lease.valid(&rdma_backend.window_pool, &rdma_backend.qp, before));
        let f = lease.frame;
        assert(after.frames.frame(f).slot == before.frames.frame(f).slot);
        assert(after.frames.frame(f).remote_shares == before.frames.frame(f).remote_shares);
        assert(after.frames.meta(f).active_remote_leases == before.frames.meta(
            f,
        ).active_remote_leases);
    }
}

/// Prove that any sibling Host/CUDA/NVMe `core_transition` on a disjoint
/// timeline preserves `PeerSession::well_formed`.
pub proof fn lemma_core_transition_preserves_peer_session<'a, T: ByteTransport>(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
    session: &PeerSession<'a, T>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, touched_timeline, allowed_source, mode),
        session.well_formed(before),
        touched_timeline != Some(session.rdma.timeline_id),
    ensures
        session.well_formed(after),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_core_transition_preserves_rdma(
        before,
        after,
        touched_timeline,
        allowed_source,
        mode,
        &session.rdma,
    );
    reveal(core_transition);
    reveal(FrameTable::same_layout);
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    assert forall|i: int| 0 <= i < session.active_grants@.len() implies {
        let g = #[trigger] session.active_grants@[i];
        &&& g.grant_tag < session.next_grant_tag
        &&& g.frame.0 < after.frames.metas.len()
        &&& session.rdma.lease_index_spec(g.handle) matches Some(idx) && {
            let lease = session.rdma.active_leases@[idx];
            &&& lease.frame == g.frame
            &&& lease.state == super::super::rdma::RemoteLeaseState::Bound
            &&& session.channel.sent().contains_key(g.grant_tag)
            &&& session.channel.sent()[g.grant_tag] == (transfer::transport::GrantKind::ReadShare {
                slot: after.frames.frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = session.active_grants@[i];
        assert(after.frames.frame(g.frame).slot == before.frames.frame(g.frame).slot);
    }
    assert forall|i: int| 0 <= i < session.pending_outbound_grants@.len() implies (
    #[trigger] session.pending_outbound_grants@[i]).valid(&session.rdma, after) by {
        let p = session.pending_outbound_grants@[i];
        assert(p.valid(&session.rdma, before));
        assert(after.frames.frame(p.frame).slot == before.frames.frame(p.frame).slot
            && after.frames.frame(p.frame).tier == before.frames.frame(p.frame).tier);
        assert(after.frames.tier_frame_size(after.frames.frame(p.frame).tier)
            == before.frames.tier_frame_size(before.frames.frame(p.frame).tier));
        vstd::imap_lib::lemma_submap_of_trans(
            p.bind_sub@.map(),
            before.ledger@.submitted.map(),
            after.ledger@.submitted.map(),
        );
        assert forall|e: TransferEvent| #[trigger] p.bind_facts@.contains(e) implies {
            &&& after.ledger@.submitted.map().contains_key(e)
            &&& after.ledger@.knows.set().contains((after.node(), e))
        } by {
            assert(before.ledger@.submitted.map().contains_key(e));
            assert(before.ledger@.knows.set().contains((before.node(), e)));
        }
    }
    assert forall|i: int| 0 <= i < session.pending_inbound_grants@.len() implies (
    #[trigger] session.pending_inbound_grants@[i]).valid(
        after,
        &session.channel,
        session.rdma.peer_node,
    ) by {
        let g = session.pending_inbound_grants@[i];
        assert(g.valid(before, &session.channel, session.rdma.peer_node));
    }
}

/// Prove that an RDMA / `PeerSession` step (`peer_core_step`) preserves
/// `HostBackend::well_formed` once the host source freshness is re-established.
pub proof fn lemma_peer_core_step_preserves_host(
    before: &CacheCore,
    after: &CacheCore,
    rdma_tl: u8,
    host: &HostBackend,
)
    requires
        peer_core_step(before, after, rdma_tl),
        host.well_formed(before),
        host.timeline != rdma_tl,
        source_fresh_for_ledger(host.ctx.source@, after.ledger@),
    ensures
        host.well_formed(after),
{
    reveal(peer_core_step_except);
    assert(after.timelines.next_seq[host.timeline as int]
        == before.timelines.next_seq[host.timeline as int]);
    assert(after.timelines.confirmed_seq[host.timeline as int]
        == before.timelines.confirmed_seq[host.timeline as int]);
    lemma_slab_bound_same_layout(
        before,
        after,
        host.tier,
        host.first_frame,
        host.num_frames,
        host.buffer.space(),
    );
}

/// Prove that an RDMA / `PeerSession` step (`peer_core_step`) preserves an idle
/// `CudaBackend::well_formed` (`active_gpu_puts@.len() == 0`) once stream
/// source freshness is re-established.
pub proof fn lemma_peer_core_step_preserves_idle_cuda<'a>(
    before: &CacheCore,
    after: &CacheCore,
    rdma_tl: u8,
    cuda: &CudaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        peer_core_step(before, after, rdma_tl),
        cuda.well_formed(before),
        cuda.timeline.timeline_id != rdma_tl,
        cuda.active_gpu_puts@.len() == 0,
        source_fresh_for_ledger(cuda.timeline.stream.source(), after.ledger@),
    ensures
        cuda.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_slab_bound_same_layout(
        before,
        after,
        cuda.gpu_tier,
        cuda.gpu_first_frame,
        cuda.gpu_num_frames,
        cuda.gpu_buffer.space(),
    );
    if let Some(ref h) = cuda.pinned_host {
        lemma_slab_bound_same_layout(
            before,
            after,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
    let tl = &cuda.timeline;
    let t = tl.timeline_id;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    assert(before.timelines.next_seq[t as int] >= 1);
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before.ledger@.submitted.map(),
        after.ledger@.submitted.map(),
    );
    assert(after.timelines.prefix_in(
        t,
        (after.timelines.next_seq[t as int] - 1) as u64,
        tl.closure(),
    )) by {
        assert forall|s: u64|
            1 <= s <= (after.timelines.next_seq[t as int] - 1) as u64 implies tl.closure().contains(
            #[trigger] after.timelines.events@[(t, s)],
        ) by {
            assert(1 <= s < before.timelines.next_seq[t as int]);
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(event_ring_valid);
    assert forall|i: int| 0 <= i < tl.event_ring@.len() implies #[trigger] tl.event_ring@[i].valid(
        &after.timelines,
        t,
        tl.device,
        after.node(),
        &after.ledger@,
    ) by {
        let slot = tl.event_ring@[i];
        assert(slot.valid(&before.timelines, t, tl.device, before.node(), &before.ledger@));
        vstd::imap_lib::lemma_submap_of_trans(
            slot.event.infos().map(),
            before.ledger@.submitted.map(),
            after.ledger@.submitted.map(),
        );
        let c = foundation::tracked::closure_from(slot.event.infos().map(), slot.event.recorded());
        assert(slot.recorded_seq < before.timelines.next_seq[t as int]);
        assert forall|s: u64| 1 <= s <= slot.recorded_seq implies c.contains(
            #[trigger] after.timelines.events@[(t, s)],
        ) by {
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(cuda_puts_valid);
}

/// Prove that an RDMA / `PeerSession` step (`peer_core_step`) on a disjoint
/// timeline preserves `NvmeBackend::well_formed`, including arbitrary in-flight
/// `io_uring` transfers (`QD >= 1`).
pub proof fn lemma_peer_core_step_preserves_nvme(
    before: &CacheCore,
    after: &CacheCore,
    rdma_tl: u8,
    nvme: &NvmeBackend,
)
    requires
        before.well_formed(),
        after.well_formed(),
        peer_core_step(before, after, rdma_tl),
        nvme.well_formed(before),
        nvme.timeline != rdma_tl,
    ensures
        nvme.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::preserves_initializing);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_slab_bound_same_layout(
        before,
        after,
        nvme.tier,
        nvme.first_frame,
        nvme.num_frames,
        nvme.file.space(),
    );
    let t = nvme.timeline;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    reveal(nvme_window_wf);
    assert(nvme.bitmap.observations(&after.timelines, t, after.node(), &after.ledger@)) by {
        assert forall|i: int| #[trigger] nvme.bitmap.bit_set(i) implies {
            let s = (after.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            after.timelines.issued(t, s) && after.timelines.observed(
                t,
                s,
                after.node(),
                after.ledger@.knows,
            )
        } by {
            let s = (before.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            assert(before.timelines.issued(t, s));
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
            assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
        }
    }
    assert(window_matches(
        nvme.in_flight_ops,
        nvme.bitmap,
        &after.timelines,
        t,
        super::super::nvme::ring_event_map(nvme.ring),
    )) by {
        let pending = super::super::nvme::ring_event_map(nvme.ring);
        assert forall|s: u64| #[trigger]
            pending.contains_key(s) implies after.timelines.confirmed_seq[t as int] < s
            < after.timelines.next_seq[t as int] && pending[s] == after.timelines.events@[(t, s)]
            && (nvme.in_flight_ops[(s % BITMAP_CAPACITY) as int] matches Some(op) && op.sequence
            == s) by {
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(nvme_pending_valid);
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (
    #[trigger] nvme.pending_transfers@[i] matches Some(op) ==> {
        &&& op.valid(after, t, i)
        &&& nvme.ring.pending().contains_key(op.ticket.sequence())
        &&& nvme.ring.pending()[op.ticket.sequence()].event == op.copy@.writer
    }) by {
        if let Some(op) = nvme.pending_transfers@[i] {
            assert(op.valid(before, t, i));
            let s = op.ticket.sequence();
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
            assert(after.frames.meta(op.hop.dst_frame) == before.frames.meta(op.hop.dst_frame));
            assert(after.frames.frame(op.hop.dst_frame) == before.frames.frame(op.hop.dst_frame));
            assert(after.active_leases[op.hop.src_lease.slot as int]
                == before.active_leases[op.hop.src_lease.slot as int]);
            assert(before.frames.versions@.dom().contains(op.hop.version));
            assert(after.frames.versions@.dom().contains(op.hop.version));
            assert(before.citations().subset_of(after.citations())) by {
                assert forall|e: TransferEvent|
                    before.citations().contains(e) implies after.citations().contains(e) by {
                    assert(before.ledger@.knows.set().contains((before.node(), e)));
                }
            }
            vstd::imap_lib::lemma_submap_of_trans(
                op.copy@.before.submitted.map(),
                before.ledger@.submitted.map(),
                after.ledger@.submitted.map(),
            );
            vstd::imap_lib::lemma_submap_of_trans(
                op.copy@.before.revoked.map(),
                before.ledger@.revoked.map(),
                after.ledger@.revoked.map(),
            );
            assert(after.ledger@.submitted.dom().contains(op.copy@.writer));
            assert(after.ledger@.submitted.map()[op.copy@.writer]
                == before.ledger@.submitted.map()[op.copy@.writer]);
            assert(nvme_copy_retained(op.copy@, after.ledger@, after.node()));
        }
    }
}

/// Tracked helper proving `preserves_destination` for every parked `PendingGpuAction::Publish`
/// in `puts[0..n]` across a `peer_core_step` by temporarily borrowing each frame's `SlotToken`
/// from `ghosts` and invoking `TransferInstance::slot_writer_is_latest`.
pub proof fn lemma_prove_cuda_puts_dest_preserved(
    before: &CacheCore,
    tracked ghosts: &mut Map<nat, FrameGhost>,
    tracked after_ledger: &KnowledgeLedger,
    tracked instance: &TransferInstance,
    puts: Seq<PendingGpuWrite>,
    tl: u8,
    n: nat,
)
    requires
        before.well_formed(),
        instance.id() == before.instance_id(),
        after_ledger.valid(before.instance_id(), before.node()),
        before.ledger@.submitted.map().submap_of(after_ledger.submitted.map()),
        revoked_submap(before.ledger@.revoked, after_ledger.revoked),
        before.ledger@.knows.set().subset_of(after_ledger.knows.set()),
        before.ledger@.succeeded.dom().subset_of(after_ledger.succeeded.dom()),
        tl < before.timelines.count,
        cuda_puts_valid(puts, before, tl),
        n <= puts.len(),
        forall|f: FrameId|
            f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                == STATE_INITIALIZING ==> old(ghosts).contains_key(f.0 as nat) && old(
                ghosts,
            )[f.0 as nat] == before.frames.frame(f),
    ensures
        *final(ghosts) == *old(ghosts),
        forall|i: int|
            0 <= i < n && ((#[trigger] puts[i]).action is Publish) ==> preserves_destination(
                before.ledger@,
                *after_ledger,
                before.node(),
                puts[i].slot@.loc,
            ),
    decreases n,
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    if n == 0 {
        return;
    }
    lemma_prove_cuda_puts_dest_preserved(
        before,
        ghosts,
        after_ledger,
        instance,
        puts,
        tl,
        (n - 1) as nat,
    );
    let idx = (n - 1) as int;
    let p = puts[idx];
    assert(p.valid(before, tl)) by {
        reveal(cuda_puts_valid);
    }
    lemma_core_wf_facts(before);
    if let PendingGpuAction::Publish { proof: put_proof, .. } = p.action {
        let cert = put_proof@.certificate;
        let w = cert.writer;
        let loc = p.slot@.loc;
        let s = p.ticket.sequence();
        assert(before.timelines.well_formed(before.instance_id(), before.node(), &before.ledger@))
            by {
            lemma_core_timelines_wf(before);
        }
        assert(before.timelines.issued(tl, s));
        assert(before.timelines.events@.contains_key((tl, s)));
        assert(before.ledger@.submitted.dom().contains(w));
        assert(complete_receipts(before.ledger@)) by {
            reveal(CacheCore::well_formed);
        }
        assert(before.ledger@.submitted.map()[w].preds.subset_of(before.ledger@.submitted.dom()))
            by {
            reveal(complete_receipts);
        }
        assert(before.citations().subset_of(after_ledger.citations(before.node()))) by {
            assert forall|e: TransferEvent| #[trigger]
                before.citations().contains(e) implies after_ledger.citations(
                before.node(),
            ).contains(e) by {
                assert(before.ledger@.knows.set().contains((before.node(), e)));
            }
        }
        assert(before.frames.meta(p.frame).state == STATE_INITIALIZING);
        assert(ghosts.contains_key(p.frame.0 as nat));
        assert(ghosts[p.frame.0 as nat] == before.frames.frame(p.frame));
        if exists|e: TransferEvent|
            after_ledger.submitted.dom().contains(e) && !before.ledger@.submitted.dom().contains(e)
                && (#[trigger] after_ledger.submitted.map()[e]).access.writes.contains(loc) {
            let e = choose|e: TransferEvent|
                after_ledger.submitted.dom().contains(e)
                    && !before.ledger@.submitted.dom().contains(e) && (
                #[trigger] after_ledger.submitted.map()[e]).access.writes.contains(loc);
            assert(e != w);
            assert(after_ledger.submitted.dom().contains(w));
            assert(after_ledger.submitted.map()[w] == before.ledger@.submitted.map()[w]);
            let ghost prev_ghosts = *ghosts;
            let tracked mut fg = ghosts.tracked_remove(p.frame.0 as nat);
            let tracked mut tok_opt: Option<SlotToken> = None;
            vstd::modes::tracked_swap(&mut fg.token, &mut tok_opt);
            let tracked mut tok = tok_opt.tracked_unwrap();
            instance.slot_writer_is_latest(w, e, &mut tok, &after_ledger.submitted);
            fg.token = Some(tok);
            ghosts.tracked_insert(p.frame.0 as nat, fg);
            assert(*ghosts =~= prev_ghosts);
            assert(before.ledger@.submitted.dom().contains(e));
            assert(false);
        }
        assert(preserves_destination(before.ledger@, *after_ledger, before.node(), loc));
    }
}

proof fn prove_cuda_puts_valid_after_peer_step(
    before: &CacheCore,
    after: &CacheCore,
    tracked reg_opt: &Option<VersionRegistry>,
    tracked pending_bindings: &Map<VersionId, VersionBinding>,
    rdma_tl: u8,
    puts: Seq<PendingGpuWrite>,
    tl: u8,
    n: nat,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.all_versions_bound(),
        *reg_opt == after.version_registry@,
        peer_core_step(before, after, rdma_tl),
        tl < before.timelines.count,
        tl != rdma_tl,
        cuda_puts_valid(puts, before, tl),
        cuda_puts_bindings_covered(puts, *pending_bindings),
        n <= puts.len(),
        forall|i: int|
            0 <= i < n && ((#[trigger] puts[i]).action is Publish) ==> preserves_destination(
                before.ledger@,
                after.ledger@,
                after.node(),
                puts[i].slot@.loc,
            ),
    ensures
        forall|i: int| 0 <= i < n ==> (#[trigger] puts[i]).valid(after, tl),
    decreases n,
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(TimelineTable::ticket_covers);
    if n == 0 {
        return;
    }
    prove_cuda_puts_valid_after_peer_step(
        before,
        after,
        reg_opt,
        pending_bindings,
        rdma_tl,
        puts,
        tl,
        (n - 1) as nat,
    );
    let idx = (n - 1) as int;
    let p = puts[idx];
    assert(p.valid(before, tl)) by {
        reveal(cuda_puts_valid);
    }
    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    assert(before.frames.meta(p.frame).state == STATE_INITIALIZING);
    assert(after.frames.meta(p.frame) == before.frames.meta(p.frame));
    assert(after.frames.frame(p.frame) == before.frames.frame(p.frame));
    assert(after.timelines.next_seq[tl as int] == before.timelines.next_seq[tl as int]);
    let s = p.ticket.sequence();
    assert(1 <= s < before.timelines.next_seq[tl as int]);
    assert(after.timelines.events@[(tl, s)] == before.timelines.events@[(tl, s)]);
    assert(after.timelines.issued(tl, s));
    assert(before.citations().subset_of(after.citations())) by {
        assert forall|e: TransferEvent|
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger@.knows.set().contains((before.node(), e)));
        }
    }
    let tok_spec = before.frames.frame(p.frame).token.unwrap();
    lemma_iset_subset_trans_local(
        slot_frontier(tok_spec).readers,
        before.citations(),
        after.citations(),
    );
    match p.action {
        PendingGpuAction::Abort => {
            if let Some(w) = slot_frontier(tok_spec).last_writer {
                if !before.citations().contains(w) {
                    reveal(TimelineTable::ticket_covers);
                    let q = choose|q: u64|
                        1 <= q <= s && (#[trigger] before.timelines.events@[(tl, q)]) == w;
                    assert(1 <= q < before.timelines.next_seq[tl as int]);
                    assert(after.timelines.events@[(tl, q)] == w);
                }
            }
        },
        PendingGpuAction::Publish { version, proof: put_proof, .. } => {
            assert(p.action is Publish);
            put_proof@.preserve(before.ledger@, after.ledger@, after.node());
            assert(cuda_put_events_covered(put_proof@.events, after.timelines.events@, tl, s)) by {
                reveal(cuda_put_events_covered);
                assert forall|k: int|
                    0 <= k
                        < put_proof@.events.len() implies super::super::cuda::cuda_event_at_prefix(
                    #[trigger] put_proof@.events[k],
                    after.timelines.events@,
                    tl,
                    s,
                ) by {
                    assert(super::super::cuda::cuda_event_at_prefix(
                        put_proof@.events[k],
                        before.timelines.events@,
                        tl,
                        s,
                    ));
                    let q = choose|q: u64|
                        1 <= q <= s && (#[trigger] before.timelines.events@[(tl, q)])
                            == put_proof@.events[k]
                            && put_proof@.events[k].source.succeeds_on_observe();
                    assert(1 <= q < before.timelines.next_seq[tl as int]);
                    assert(after.timelines.events@[(tl, q)] == put_proof@.events[k]);
                }
            }
            if before.frames.versions@.contains_key(version) {
                lemma_map_submap_get_local(
                    before.frames.versions@,
                    after.frames.versions@,
                    version,
                );
            } else if after.frames.versions@.contains_key(version) {
                lemma_cuda_puts_bindings_extract(puts, *pending_bindings, idx);
                assert(after.has_version_binding(version));
                assert(after.version_registry_valid()) by {
                    reveal(CacheCore::well_formed);
                }
                match reg_opt {
                    Some(reg) => {
                        version_binding_agree(
                            reg.bindings.tracked_borrow(version),
                            pending_bindings.tracked_borrow(version),
                        );
                    },
                    None => {},
                }
            }
        },
    }
}

/// Prove that an RDMA / `PeerSession` step (`peer_core_step`) preserves an
/// arbitrary `CudaBackend::well_formed` (including any in-flight parked async
/// GPU writes covered by `pending_bindings`) once stream source freshness and
/// `lemma_prove_cuda_puts_dest_preserved` have been established.
pub proof fn lemma_peer_core_step_preserves_cuda<'a>(
    before: &CacheCore,
    after: &CacheCore,
    tracked reg_opt: &Option<VersionRegistry>,
    tracked pending_bindings: &Map<VersionId, VersionBinding>,
    rdma_tl: u8,
    cuda: &CudaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.all_versions_bound(),
        *reg_opt == after.version_registry@,
        peer_core_step(before, after, rdma_tl),
        cuda.well_formed(before),
        cuda_puts_bindings_covered(cuda.active_gpu_puts@, *pending_bindings),
        cuda.timeline.timeline_id != rdma_tl,
        source_fresh_for_ledger(cuda.timeline.stream.source(), after.ledger@),
        forall|i: int|
            0 <= i < cuda.active_gpu_puts@.len() && ((
            #[trigger] cuda.active_gpu_puts@[i]).action is Publish) ==> preserves_destination(
                before.ledger@,
                after.ledger@,
                after.node(),
                cuda.active_gpu_puts@[i].slot@.loc,
            ),
    ensures
        cuda.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_slab_bound_same_layout(
        before,
        after,
        cuda.gpu_tier,
        cuda.gpu_first_frame,
        cuda.gpu_num_frames,
        cuda.gpu_buffer.space(),
    );
    if let Some(ref h) = cuda.pinned_host {
        lemma_slab_bound_same_layout(
            before,
            after,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
    let tl = &cuda.timeline;
    let t = tl.timeline_id;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    assert(before.timelines.next_seq[t as int] >= 1);
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before.ledger@.submitted.map(),
        after.ledger@.submitted.map(),
    );
    assert(after.timelines.prefix_in(
        t,
        (after.timelines.next_seq[t as int] - 1) as u64,
        tl.closure(),
    )) by {
        assert forall|s: u64|
            1 <= s <= (after.timelines.next_seq[t as int] - 1) as u64 implies tl.closure().contains(
            #[trigger] after.timelines.events@[(t, s)],
        ) by {
            assert(1 <= s < before.timelines.next_seq[t as int]);
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    reveal(event_ring_valid);
    assert forall|i: int| 0 <= i < tl.event_ring@.len() implies #[trigger] tl.event_ring@[i].valid(
        &after.timelines,
        t,
        tl.device,
        after.node(),
        &after.ledger@,
    ) by {
        let slot = tl.event_ring@[i];
        assert(slot.valid(&before.timelines, t, tl.device, before.node(), &before.ledger@));
        vstd::imap_lib::lemma_submap_of_trans(
            slot.event.infos().map(),
            before.ledger@.submitted.map(),
            after.ledger@.submitted.map(),
        );
        let c = foundation::tracked::closure_from(slot.event.infos().map(), slot.event.recorded());
        assert(slot.recorded_seq < before.timelines.next_seq[t as int]);
        assert forall|s: u64| 1 <= s <= slot.recorded_seq implies c.contains(
            #[trigger] after.timelines.events@[(t, s)],
        ) by {
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    prove_cuda_puts_valid_after_peer_step(
        before,
        after,
        reg_opt,
        pending_bindings,
        rdma_tl,
        cuda.active_gpu_puts@,
        t,
        cuda.active_gpu_puts@.len(),
    );
    reveal(cuda_puts_valid);
}

/// Prove that an active `PeerSession` step (`peer_core_step` +
/// `peer_foreign_shares_preserved`) on `active_tl` preserves a sibling
/// `PeerSession` on a distinct timeline `sibling.rdma.timeline_id != active_tl`,
/// along with pairwise pinned-share disjointness.
pub proof fn lemma_peer_step_preserves_sibling_session<'a, T: ByteTransport>(
    before: &CacheCore,
    after: &CacheCore,
    active_tl: u8,
    active_before_leases: Seq<ActiveRemoteLease>,
    active_after_leases: Seq<ActiveRemoteLease>,
    sibling: &PeerSession<'a, T>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        peer_core_step(before, after, active_tl),
        peer_foreign_shares_preserved(before, after, active_before_leases, active_after_leases),
        sibling.well_formed(before),
        sibling.rdma.timeline_id != active_tl,
        peer_leases_disjoint(sibling.rdma.active_leases@, active_before_leases),
    ensures
        sibling.well_formed(after),
        peer_leases_disjoint(sibling.rdma.active_leases@, active_after_leases),
        peer_leases_disjoint(active_after_leases, sibling.rdma.active_leases@),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_core_timelines_wf(after);
    let rdma_backend = &sibling.rdma;
    if let Some(ref s) = rdma_backend.host_mr {
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    if let Some(ref s) = rdma_backend.gpu_mr {
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
    }
    let pd_id = rdma_backend.window_pool.pd();
    if let Some(s) = rdma_backend.shared_host_mr {
        s.lemma_shared_wf_iff(before, pd_id);
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
        s.lemma_shared_wf_iff(after, pd_id);
    }
    if let Some(s) = rdma_backend.shared_gpu_mr {
        s.lemma_shared_wf_iff(before, pd_id);
        lemma_slab_bound_same_layout(
            before,
            after,
            s.tier,
            s.first_frame,
            s.num_frames,
            s.mr.space(),
        );
        s.lemma_shared_wf_iff(after, pd_id);
    }
    let t = rdma_backend.timeline_id;
    assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    assert(after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    reveal(rdma_window_wf);
    assert(rdma_backend.bitmap.observations(&after.timelines, t, after.node(), &after.ledger@)) by {
        assert forall|i: int| #[trigger] rdma_backend.bitmap.bit_set(i) implies {
            let s = (after.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            after.timelines.issued(t, s) && after.timelines.observed(
                t,
                s,
                after.node(),
                after.ledger@.knows,
            )
        } by {
            let s = (before.timelines.confirmed_seq[t as int] + 1 + i) as u64;
            assert(before.timelines.issued(t, s));
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
            assert(before.timelines.observed(t, s, before.node(), before.ledger@.knows));
        }
    }
    assert(window_matches(
        rdma_backend.in_flight_ops,
        rdma_backend.bitmap,
        &after.timelines,
        t,
        rdma_data_events(rdma::CqSnapshot::of(rdma_backend.cq)),
    )) by {
        let pending = rdma_data_events(rdma::CqSnapshot::of(rdma_backend.cq));
        assert forall|s: u64| #[trigger]
            pending.contains_key(s) implies after.timelines.confirmed_seq[t as int] < s
            < after.timelines.next_seq[t as int] && pending[s] == after.timelines.events@[(t, s)]
            && (rdma_backend.in_flight_ops[(s % BITMAP_CAPACITY) as int] matches Some(op)
            && op.sequence == s) by {
            assert(after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]);
        }
    }
    assert(super::super::rdma::completed_pulls_wf(
        rdma_backend.completed_pulls,
        &after.timelines,
        t,
        rdma_backend.cq.observer(),
        &after.ledger@,
    )) by {
        assert forall|i: int|
            0 <= i < BITMAP_SLOTS implies match #[trigger] rdma_backend.completed_pulls[i] {
            None => true,
            Some((seq, st)) => {
                &&& ((seq as int) % (BITMAP_CAPACITY as int)) == i
                &&& after.timelines.issued(t, seq)
                &&& after.ledger@.knows.set().contains(
                    (rdma_backend.cq.observer(), after.timelines.events@[(t, seq)]),
                )
                &&& (st.is_ok() ==> after.ledger@.succeeded.dom().contains(
                    after.timelines.events@[(t, seq)],
                ))
            },
        } by {
            if let Some((seq, _st)) = rdma_backend.completed_pulls[i] {
                assert(before.timelines.issued(t, seq));
                assert(after.timelines.events@[(t, seq)] == before.timelines.events@[(t, seq)]);
            }
        }
    }
    reveal(rdma_leases_valid);
    assert forall|i: int| 0 <= i < rdma_backend.active_leases@.len() implies {
        &&& (#[trigger] rdma_backend.active_leases@[i]).valid(
            &rdma_backend.window_pool,
            &rdma_backend.qp,
            after,
        )
        &&& (!rdma_backend.active_leases@[i].is_unpinned() ==> leases_free_of_share(
            active_after_leases,
            rdma_backend.active_leases@[i].frame,
            rdma_backend.active_leases@[i].share@,
        ))
    } by {
        let lease = rdma_backend.active_leases@[i];
        assert(lease.valid(&rdma_backend.window_pool, &rdma_backend.qp, before));
        let f = lease.frame;
        assert(after.frames.frame(f).slot == before.frames.frame(f).slot);
        if !lease.is_unpinned() {
            assert(leases_free_of_share(active_before_leases, f, lease.share@));
            lemma_foreign_shares_lookup(
                before,
                after,
                active_before_leases,
                active_after_leases,
                f,
                lease.share@,
            );
            super::super::rdma::proofs::lemma_frame_remote_share_implies_nonzero(
                after,
                f,
                lease.share@,
            );
        }
    }
    assert forall|i: int|
        0 <= i < rdma_backend.active_leases@.len() && !(
        #[trigger] rdma_backend.active_leases@[i]).is_unpinned() implies leases_free_of_share(
        active_after_leases,
        rdma_backend.active_leases@[i].frame,
        rdma_backend.active_leases@[i].share@,
    ) by {
        assert(rdma_backend.active_leases@[i].valid(
            &rdma_backend.window_pool,
            &rdma_backend.qp,
            after,
        ));
    }
    lemma_peer_leases_disjoint_sym(rdma_backend.active_leases@, active_after_leases);
    assert forall|i: int| 0 <= i < sibling.active_grants@.len() implies {
        let g = #[trigger] sibling.active_grants@[i];
        &&& g.grant_tag < sibling.next_grant_tag
        &&& g.frame.0 < after.frames.metas.len()
        &&& sibling.rdma.lease_index_spec(g.handle) matches Some(idx) && {
            let lease = sibling.rdma.active_leases@[idx];
            &&& lease.frame == g.frame
            &&& lease.state == super::super::rdma::RemoteLeaseState::Bound
            &&& sibling.channel.sent().contains_key(g.grant_tag)
            &&& sibling.channel.sent()[g.grant_tag] == (transfer::transport::GrantKind::ReadShare {
                slot: after.frames.frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = sibling.active_grants@[i];
        assert(after.frames.frame(g.frame).slot == before.frames.frame(g.frame).slot);
    }
    assert forall|i: int| 0 <= i < sibling.pending_outbound_grants@.len() implies (
    #[trigger] sibling.pending_outbound_grants@[i]).valid(&sibling.rdma, after) by {
        let p = sibling.pending_outbound_grants@[i];
        assert(p.valid(&sibling.rdma, before));
        assert(after.frames.frame(p.frame).slot == before.frames.frame(p.frame).slot
            && after.frames.frame(p.frame).tier == before.frames.frame(p.frame).tier);
        assert(after.frames.tier_frame_size(after.frames.frame(p.frame).tier)
            == before.frames.tier_frame_size(before.frames.frame(p.frame).tier));
        vstd::imap_lib::lemma_submap_of_trans(
            p.bind_sub@.map(),
            before.ledger@.submitted.map(),
            after.ledger@.submitted.map(),
        );
        assert forall|e: TransferEvent| #[trigger] p.bind_facts@.contains(e) implies {
            &&& after.ledger@.submitted.map().contains_key(e)
            &&& after.ledger@.knows.set().contains((after.node(), e))
        } by {
            assert(before.ledger@.submitted.map().contains_key(e));
            assert(before.ledger@.knows.set().contains((before.node(), e)));
        }
    }
    assert forall|i: int| 0 <= i < sibling.pending_inbound_grants@.len() implies (
    #[trigger] sibling.pending_inbound_grants@[i]).valid(
        after,
        &sibling.channel,
        sibling.rdma.peer_node,
    ) by {
        let g = sibling.pending_inbound_grants@[i];
        assert(g.valid(before, &sibling.channel, sibling.rdma.peer_node));
    }
}

} // verus!
