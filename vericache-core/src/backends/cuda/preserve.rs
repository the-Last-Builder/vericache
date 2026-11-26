use super::*;
use timeline::{event_ring_valid, CudaTimeline};
use transfer::cuda::CudaStream;

verus! {

broadcast use vstd::iset::group_iset_lemmas;

impl<'a> CudaBackend<'a> {
    pub open(crate) spec fn poll_footprint(&self) -> CoreFootprint {
        let puts = self.active_gpu_puts@;
        CoreFootprint::completion(
            self.timeline.timeline_id,
            ISet::new(
                |f: FrameId| exists|i: int| 0 <= i < puts.len() && (#[trigger] puts[i]).frame == f,
            ),
            ISet::empty(),
            VersionFootprint::only(
                ISet::new(
                    |v: VersionId|
                        exists|i: int|
                            0 <= i < puts.len() && (#[trigger] puts[i]).action.publishes_version(v),
                ),
            ),
        )
    }

    /// A pending write retains its frame and the exclusive right to publish its version.
    pub open(crate) spec fn pending_untouched(
        &self,
        before: &CacheCore,
        fp: CoreFootprint,
    ) -> bool {
        let puts = self.active_gpu_puts@;
        &&& fp.allowed_sources is Some
        &&& self.put_frames_untouched(before, fp.touched_init_frames)
        &&& self.put_frames_untouched(before, fp.written_init_frames)
        &&& fp.versions.specific matches Some(versions) && (versions.is_empty() || forall|
            i: int,
            v: VersionId,
        |
            0 <= i < puts.len() && (#[trigger] puts[i].action.publishes_version(v))
                ==> !versions.contains(v))
    }

    pub open(crate) spec fn put_frames_untouched(
        &self,
        before: &CacheCore,
        frames: ISet<FrameId>,
    ) -> bool {
        frames.is_empty() || (forall|f: FrameId| #[trigger]
            frames.contains(f) && f.0 < before.frames_spec().metas_spec().len()
                ==> before.frames_spec().frame(f).token is None) || (forall|i: int|
            0 <= i < self.active_gpu_puts@.len() ==> !frames.contains(
                (#[trigger] self.active_gpu_puts@[i]).frame,
            ))
    }
}

pub(super) proof fn lemma_cuda_poll_footprint_frames(backend: &CudaBackend<'_>, before: &CacheCore)
    requires
        backend.well_formed(before),
    ensures
        forall|f: FrameId| #[trigger]
            backend.poll_footprint().touched_init_frames.contains(f) ==> before.frames_spec().frame(
                f,
            ).token is Some,
{
    broadcast use vstd::iset::group_iset_lemmas;

    reveal(cuda_puts_valid);
    let puts = backend.active_gpu_puts@;
    assert forall|f: FrameId| #[trigger]
        backend.poll_footprint().touched_init_frames.contains(f) implies before.frames_spec().frame(
        f,
    ).token is Some by {
        let i = choose|i: int| 0 <= i < puts.len() && (#[trigger] puts[i]).frame == f;
        assert(puts[i].valid(before, backend.timeline.timeline_id));
    }
}

pub(super) proof fn lemma_cuda_poll_keeps_scope(
    initial: &CudaBackend<'_>,
    prior: Seq<PendingGpuWrite>,
    remaining: Seq<PendingGpuWrite>,
    removed: PendingGpuWrite,
    idx: int,
)
    requires
        0 <= idx < prior.len(),
        removed == prior[idx],
        remaining.len() + 1 == prior.len(),
        forall|i: int|
            0 <= i < remaining.len() ==> #[trigger] remaining[i] == if i == idx {
                prior[prior.len() - 1]
            } else {
                prior[i]
            },
        forall|i: int|
            0 <= i < prior.len() ==> initial.active_gpu_puts@.contains(#[trigger] prior[i]),
    ensures
        forall|i: int|
            0 <= i < remaining.len() ==> initial.active_gpu_puts@.contains(#[trigger] remaining[i]),
        initial.poll_footprint().touched_init_frames.contains(removed.frame),
        forall|v: VersionId| #[trigger]
            removed.action.publishes_version(v)
                ==> initial.poll_footprint().versions.specific.unwrap().contains(v),
{
    broadcast use vstd::iset::group_iset_lemmas;

    let puts = initial.active_gpu_puts@;
    assert(puts.contains(removed));
    let original_idx = choose|i: int| 0 <= i < puts.len() && #[trigger] puts[i] == removed;
    assert(puts[original_idx].frame == removed.frame);
    assert forall|i: int| 0 <= i < remaining.len() implies puts.contains(
        #[trigger] remaining[i],
    ) by {
        let j = if i == idx {
            prior.len() - 1
        } else {
            i
        };
        assert(prior[j] == remaining[i]);
    }
    assert forall|v: VersionId| #[trigger]
        removed.action.publishes_version(
            v,
        ) implies initial.poll_footprint().versions.specific.unwrap().contains(v) by {
        assert(puts[original_idx].action.publishes_version(v));
    }
}

/// Completion publishes the retired write's existing bound version, if any.
pub(super) proof fn lemma_cuda_retired_version(
    before: &CacheCore,
    after: &CacheCore,
    entry: PendingGpuWrite,
    fp: CoreFootprint,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.spec_spec()@.versions == before.spec_spec()@.versions || (
        entry.action matches PendingGpuAction::Publish { version, proof, .. }
            && after.spec_spec()@.versions == before.spec_spec()@.versions.insert(
            version,
            proof@.certificate.bytes,
        )),
        forall|v: VersionId| #[trigger]
            entry.action.publishes_version(v) ==> fp.versions.allows(before, v),
    ensures
        forall|v: VersionId| #[trigger]
            after.frames_spec().versions_spec()@.contains_key(v)
                && !before.frames_spec().versions_spec()@.contains_key(v) ==> fp.versions.allows(
                before,
                v,
            ),
{
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    broadcast use vstd::map::group_map_lemmas;

    assert forall|v: VersionId| #[trigger]
        after.frames_spec().versions_spec()@.contains_key(v)
            && !before.frames_spec().versions_spec()@.contains_key(v) implies fp.versions.allows(
        before,
        v,
    ) by {
        if let PendingGpuAction::Publish { version, proof, .. } = entry.action {
            assert(before.spec_spec()@.versions.insert(
                version,
                proof@.certificate.bytes,
            ).contains_key(v));
            assert(v == version);
            assert(entry.action.publishes_version(v));
        }
    }
}

proof fn lemma_core_transition_cuda_put_facts(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    tl: TimelineId,
    p: PendingGpuWrite,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        !fp.touched_timelines.contains(tl),
        p.valid(before, tl),
        !fp.touched_init_frames.contains(p.frame),
        !fp.written_init_frames.contains(p.frame),
        fp.allowed_sources is Some,
        p.action matches PendingGpuAction::Publish { version, .. } ==> !fp.versions.allows(
            before,
            version,
        ),
    ensures
        after.node_id_spec() == before.node_id_spec(),
        after.instance_id() == before.instance_id(),
        after.next_version_seq_spec() >= before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
        after.frames_spec().frame_size_spec() == before.frames_spec().frame_size_spec(),
        after.frames_spec().tier_frame_sizes_spec() == before.frames_spec().tier_frame_sizes_spec(),
        after.frames_spec().meta(p.frame) == before.frames_spec().meta(p.frame),
        after.frames_spec().frame(p.frame) == before.frames_spec().frame(p.frame),
        after.timelines_spec().count_spec() >= before.timelines_spec().count_spec(),
        after.timelines_spec().next_seq_spec()[tl as int]
            == before.timelines_spec().next_seq_spec()[tl as int],
        after.timelines_spec().events_spec()@[(tl, p.ticket.sequence())]
            == before.timelines_spec().events_spec()@[(tl, p.ticket.sequence())],
        before.citations().subset_of(after.citations()),
        before.frames_spec().versions_spec()@.submap_of(after.frames_spec().versions_spec()@),
        preserves_destination(
            before.ledger_spec()@,
            after.ledger_spec()@,
            after.node(),
            p.slot@.loc,
        ),
        p.action matches PendingGpuAction::Publish { version, .. } ==> (
        !before.frames_spec().versions_spec()@.contains_key(version)
            ==> !after.frames_spec().versions_spec()@.contains_key(version)),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    reveal(core_transition);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    assert(after.timelines_spec().next_seq_spec()[tl as int]
        == before.timelines_spec().next_seq_spec()[tl as int]);
    assert(before.citations().subset_of(after.citations())) by {
        assert forall|e: TransferEvent|
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
        }
    }
    assert(preserves_destination(
        before.ledger_spec()@,
        after.ledger_spec()@,
        after.node(),
        p.slot@.loc,
    )) by {
        assert forall|e: TransferEvent| #[trigger]
            after.ledger_spec()@.submitted.dom().contains(e)
                && !before.ledger_spec()@.submitted.dom().contains(
                e,
            ) implies !after.ledger_spec()@.submitted.map()[e].access.writes.contains(
            p.slot@.loc,
        ) by {
            assert(before.frames_spec().meta(p.frame).state == STATE_INITIALIZING);
            assert(before.frames_spec().frame(p.frame).slot.loc == p.slot@.loc);
        }
    }
}

proof fn lemma_core_transition_preserves_cuda_timeline(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    tl: &CudaTimeline,
)
    requires
        before.well_formed(),
        core_transition(before, after, fp),
        tl.well_formed(&before.timelines_spec(), before.node(), &before.ledger_spec()@),
        !fp.touched_timelines.contains(tl.timeline_id),
        fp.excludes_source(tl.stream.source().source()) || source_fresh_for_ledger(
            tl.stream.source(),
            after.ledger_spec()@,
        ),
    ensures
        tl.well_formed(&after.timelines_spec(), after.node(), &after.ledger_spec()@),
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
    assert(after.timelines_spec().next_seq_spec()[t as int]
        == before.timelines_spec().next_seq_spec()[t as int]);
    assert(after.timelines_spec().confirmed_seq_spec()[t as int]
        == before.timelines_spec().confirmed_seq_spec()[t as int]);
    assert(before.timelines_spec().next_seq_spec()[t as int] >= 1);
    if fp.excludes_source(tl.stream.source().source()) {
        lemma_core_transition_preserves_source(before, after, fp, tl.stream.source());
    }
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before.ledger_spec()@.submitted.map(),
        after.ledger_spec()@.submitted.map(),
    );
    assert(after.timelines_spec().prefix_in(
        t,
        (after.timelines_spec().next_seq_spec()[t as int] - 1) as u64,
        tl.closure(),
    )) by {
        assert forall|s: u64|
            1 <= s <= (after.timelines_spec().next_seq_spec()[t as int]
                - 1) as u64 implies tl.closure().contains(
            #[trigger] after.timelines_spec().events_spec()@[(t, s)],
        ) by {
            assert(1 <= s < before.timelines_spec().next_seq_spec()[t as int]);
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
    reveal(event_ring_valid);
    assert forall|i: int| 0 <= i < tl.event_ring@.len() implies #[trigger] tl.event_ring@[i].valid(
        &after.timelines_spec(),
        t,
        tl.device,
        after.node(),
        &after.ledger_spec()@,
    ) by {
        let slot = tl.event_ring@[i];
        assert(slot.valid(
            &before.timelines_spec(),
            t,
            tl.device,
            before.node(),
            &before.ledger_spec()@,
        ));
        vstd::imap_lib::lemma_submap_of_trans(
            slot.event.infos().map(),
            before.ledger_spec()@.submitted.map(),
            after.ledger_spec()@.submitted.map(),
        );
        let c = foundation::tracked::closure_from(slot.event.infos().map(), slot.event.recorded());
        assert(slot.recorded_seq < before.timelines_spec().next_seq_spec()[t as int]);
        assert forall|s: u64| 1 <= s <= slot.recorded_seq implies c.contains(
            #[trigger] after.timelines_spec().events_spec()@[(t, s)],
        ) by {
            lemma_core_transition_event_eq(before, after, fp, t, s);
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
}

proof fn lemma_core_transition_preserves_cuda_put_events(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    tl: TimelineId,
    s: u64,
    events: Seq<TransferEvent>,
)
    requires
        core_transition(before, after, fp),
        tl < before.timelines_spec().count_spec(),
        !fp.touched_timelines.contains(tl),
        s < before.timelines_spec().next_seq_spec()[tl as int],
        cuda_put_events_covered(events, before.timelines_spec().events_spec()@, tl, s),
    ensures
        cuda_put_events_covered(events, after.timelines_spec().events_spec()@, tl, s),
{
    reveal(core_transition);
    reveal(cuda_put_events_covered);
    assert forall|k: int| 0 <= k < events.len() implies cuda_event_at_prefix(
        #[trigger] events[k],
        after.timelines_spec().events_spec()@,
        tl,
        s,
    ) by {
        assert(cuda_event_at_prefix(events[k], before.timelines_spec().events_spec()@, tl, s));
        let q = choose|q: u64|
            1 <= q <= s && (#[trigger] before.timelines_spec().events_spec()@[(tl, q)]) == events[k]
                && events[k].source.succeeds_on_observe();
        assert(1 <= q < before.timelines_spec().next_seq_spec()[tl as int]);
        assert(after.timelines_spec().events_spec()@[(tl, q)] == events[k]);
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
    fp: CoreFootprint,
    tl: TimelineId,
    ticket: Ticket,
    w: TransferEvent,
)
    requires
        core_transition(before, after, fp),
        tl < before.timelines_spec().count_spec(),
        !fp.touched_timelines.contains(tl),
        ticket != Ticket::NONE,
        ticket.timeline_id() == tl,
        before.timelines_spec().issued(tl, ticket.sequence()),
        before.timelines_spec().ticket_covers(ticket, w),
    ensures
        after.timelines_spec().ticket_covers(ticket, w),
{
    assert(exists|s: u64|
        1 <= s <= ticket.sequence() && #[trigger] before.timelines_spec().events_spec()@[(
            ticket.timeline_id(),
            s,
        )] == w);
    let q = choose|q: u64|
        1 <= q <= ticket.sequence() && (#[trigger] before.timelines_spec().events_spec()@[(
            ticket.timeline_id(),
            q,
        )]) == w;
    lemma_core_transition_event_eq(before, after, fp, tl, q);
    assert(1 <= q <= ticket.sequence() && after.timelines_spec().events_spec()@[(
        ticket.timeline_id(),
        q,
    )] == w);
    assert(exists|s: u64|
        1 <= s <= ticket.sequence() && #[trigger] after.timelines_spec().events_spec()@[(
            ticket.timeline_id(),
            s,
        )] == w);
}

proof fn lemma_core_transition_preserves_one_cuda_put(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    tl: TimelineId,
    p: PendingGpuWrite,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        !fp.touched_timelines.contains(tl),
        p.valid(before, tl),
        !fp.touched_init_frames.contains(p.frame),
        !fp.written_init_frames.contains(p.frame),
        fp.allowed_sources is Some,
        p.action matches PendingGpuAction::Publish { version, .. } ==> !fp.versions.allows(
            before,
            version,
        ),
    ensures
        p.valid(after, tl),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(TimelineTable::ticket_covers);
    lemma_core_transition_cuda_put_facts(before, after, fp, tl, p);
    let s = p.ticket.sequence();
    let tok = before.frames_spec().frame(p.frame).token.unwrap();
    lemma_iset_subset_trans_local(
        slot_frontier(tok).readers,
        before.citations(),
        after.citations(),
    );
    assert(after.timelines_spec().issued(tl, s));
    match p.action {
        PendingGpuAction::Abort => {
            if let Some(w) = slot_frontier(tok).last_writer {
                if !before.citations().contains(w) {
                    lemma_core_transition_abort_ticket_covers(before, after, fp, tl, p.ticket, w);
                }
            }
        },
        PendingGpuAction::Publish { version, proof: put_proof, .. } => {
            if before.frames_spec().versions_spec()@.contains_key(version) {
                lemma_map_submap_get_local(
                    before.frames_spec().versions_spec()@,
                    after.frames_spec().versions_spec()@,
                    version,
                );
            }
            put_proof@.preserve(before.ledger_spec()@, after.ledger_spec()@, after.node());
            lemma_core_transition_preserves_cuda_put_events(
                before,
                after,
                fp,
                tl,
                s,
                put_proof@.events,
            );
        },
    }
}

pub(crate) proof fn lemma_core_transition_preserves_cuda<'a>(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    cuda: &CudaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        cuda.well_formed(before),
        !fp.touched_timelines.contains(cuda.timeline.timeline_id),
        fp.excludes_source(cuda.timeline.stream.source().source()) || source_fresh_for_ledger(
            cuda.timeline.stream.source(),
            after.ledger_spec()@,
        ),
        cuda.pending_untouched(before, fp),
    ensures
        cuda.well_formed(after),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    assert(after.node_id_spec() == before.node_id_spec() && after.instance_id()
        == before.instance_id() && after.tiers_spec() == before.tiers_spec()
        && after.frames_spec().same_layout(&before.frames_spec())) by {
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
    lemma_core_transition_preserves_cuda_timeline(before, after, fp, &cuda.timeline);
    let tl = cuda.timeline.timeline_id;
    reveal(cuda_puts_valid);
    assert forall|i: int| 0 <= i < cuda.active_gpu_puts@.len() implies (
    #[trigger] cuda.active_gpu_puts@[i]).valid(after, tl) by {
        let p = cuda.active_gpu_puts@[i];
        assert(p.valid(before, tl));
        assert(!fp.touched_init_frames.contains(p.frame));
        assert(!fp.written_init_frames.contains(p.frame));
        if let PendingGpuAction::Publish { version, .. } = p.action {
            assert((cuda.active_gpu_puts@[i]).action.publishes_version(version));
            assert(!fp.versions.specific.unwrap().contains(version));
            assert(!fp.versions.allows(before, version));
        }
        lemma_core_transition_preserves_one_cuda_put(
            before,
            after,
            fp,
            tl,
            cuda.active_gpu_puts@[i],
        );
    }
}

pub proof fn lemma_core_transition_preserves_caller_stream(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    stream: CudaStream,
    dst_timeline: TimelineId,
    backend_stream: CudaStream,
)
    requires
        core_transition(before, after, fp),
        caller_stream_valid(stream, before, dst_timeline, backend_stream),
        fp.excludes_source(stream.source().source()),
    ensures
        caller_stream_valid(stream, after, dst_timeline, backend_stream),
{
    reveal(core_transition);
    lemma_core_transition_preserves_source(before, after, fp, stream.source());
    vstd::imap_lib::lemma_submap_of_trans(
        stream.infos().map(),
        before.ledger_spec()@.submitted.map(),
        after.ledger_spec()@.submitted.map(),
    );
}

pub(crate) proof fn lemma_core_transition_preserves_idle_cuda<'a>(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    cuda: &CudaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        cuda.well_formed(before),
        !fp.touched_timelines.contains(cuda.timeline.timeline_id),
        cuda.active_gpu_puts@.len() == 0,
        source_fresh_for_ledger(cuda.timeline.stream.source(), after.ledger_spec()@),
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

    reveal(core_transition);
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
    assert(after.timelines_spec().next_seq_spec()[t as int]
        == before.timelines_spec().next_seq_spec()[t as int]);
    assert(after.timelines_spec().confirmed_seq_spec()[t as int]
        == before.timelines_spec().confirmed_seq_spec()[t as int]);
    assert(before.timelines_spec().next_seq_spec()[t as int] >= 1);
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before.ledger_spec()@.submitted.map(),
        after.ledger_spec()@.submitted.map(),
    );
    assert(after.timelines_spec().prefix_in(
        t,
        (after.timelines_spec().next_seq_spec()[t as int] - 1) as u64,
        tl.closure(),
    )) by {
        assert forall|s: u64|
            1 <= s <= (after.timelines_spec().next_seq_spec()[t as int]
                - 1) as u64 implies tl.closure().contains(
            #[trigger] after.timelines_spec().events_spec()@[(t, s)],
        ) by {
            assert(1 <= s < before.timelines_spec().next_seq_spec()[t as int]);
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
    reveal(event_ring_valid);
    assert forall|i: int| 0 <= i < tl.event_ring@.len() implies #[trigger] tl.event_ring@[i].valid(
        &after.timelines_spec(),
        t,
        tl.device,
        after.node(),
        &after.ledger_spec()@,
    ) by {
        let slot = tl.event_ring@[i];
        assert(slot.valid(
            &before.timelines_spec(),
            t,
            tl.device,
            before.node(),
            &before.ledger_spec()@,
        ));
        vstd::imap_lib::lemma_submap_of_trans(
            slot.event.infos().map(),
            before.ledger_spec()@.submitted.map(),
            after.ledger_spec()@.submitted.map(),
        );
        let c = foundation::tracked::closure_from(slot.event.infos().map(), slot.event.recorded());
        assert(slot.recorded_seq < before.timelines_spec().next_seq_spec()[t as int]);
        assert forall|s: u64| 1 <= s <= slot.recorded_seq implies c.contains(
            #[trigger] after.timelines_spec().events_spec()@[(t, s)],
        ) by {
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
    reveal(cuda_puts_valid);
}

/// Tracked helper proving `preserves_destination` for every parked `PendingGpuAction::Publish`
/// in `puts[0..n]` across external event imports by temporarily borrowing each frame's `SlotToken`
/// from `ghosts` and invoking `TransferInstance::slot_writer_is_latest`.
pub(crate) proof fn lemma_prove_cuda_puts_dest_preserved(
    before: &CacheCore,
    tracked ghosts: &mut Map<nat, FrameGhost>,
    tracked after_ledger: &KnowledgeLedger,
    tracked instance: &TransferInstance,
    puts: Seq<PendingGpuWrite>,
    tl: TimelineId,
    n: nat,
)
    requires
        before.well_formed(),
        instance.id() == before.instance_id(),
        after_ledger.valid(before.instance_id(), before.node()),
        before.ledger_spec()@.submitted.map().submap_of(after_ledger.submitted.map()),
        revoked_submap(before.ledger_spec()@.revoked, after_ledger.revoked),
        before.ledger_spec()@.knows.set().subset_of(after_ledger.knows.set()),
        before.ledger_spec()@.succeeded.dom().subset_of(after_ledger.succeeded.dom()),
        tl < before.timelines_spec().count_spec(),
        cuda_puts_valid(puts, before, tl),
        n <= puts.len(),
        forall|f: FrameId|
            f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().meta(
                f,
            )).state == STATE_INITIALIZING ==> old(ghosts).contains_key(f.0 as nat) && old(
                ghosts,
            )[f.0 as nat] == before.frames_spec().frame(f),
    ensures
        *final(ghosts) == *old(ghosts),
        forall|i: int|
            0 <= i < n && ((#[trigger] puts[i]).action is Publish) ==> preserves_destination(
                before.ledger_spec()@,
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
        assert(before.timelines_spec().well_formed(
            before.instance_id(),
            before.node(),
            &before.ledger_spec()@,
        )) by {
            lemma_core_timelines_wf(before);
        }
        assert(before.timelines_spec().issued(tl, s));
        assert(before.timelines_spec().events_spec()@.contains_key((tl, s)));
        assert(before.ledger_spec()@.submitted.dom().contains(w));
        assert(complete_receipts(before.ledger_spec()@)) by {
            reveal(CacheCore::well_formed);
        }
        assert(before.ledger_spec()@.submitted.map()[w].preds.subset_of(
            before.ledger_spec()@.submitted.dom(),
        )) by {
            reveal(complete_receipts);
        }
        assert(before.citations().subset_of(after_ledger.citations(before.node()))) by {
            assert forall|e: TransferEvent| #[trigger]
                before.citations().contains(e) implies after_ledger.citations(
                before.node(),
            ).contains(e) by {
                assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
            }
        }
        assert(before.frames_spec().meta(p.frame).state == STATE_INITIALIZING);
        assert(ghosts.contains_key(p.frame.0 as nat));
        assert(ghosts[p.frame.0 as nat] == before.frames_spec().frame(p.frame));
        if exists|e: TransferEvent|
            after_ledger.submitted.dom().contains(e)
                && !before.ledger_spec()@.submitted.dom().contains(e) && (
            #[trigger] after_ledger.submitted.map()[e]).access.writes.contains(loc) {
            let e = choose|e: TransferEvent|
                after_ledger.submitted.dom().contains(e)
                    && !before.ledger_spec()@.submitted.dom().contains(e) && (
                #[trigger] after_ledger.submitted.map()[e]).access.writes.contains(loc);
            assert(e != w);
            assert(after_ledger.submitted.dom().contains(w));
            assert(after_ledger.submitted.map()[w] == before.ledger_spec()@.submitted.map()[w]);
            let ghost prev_ghosts = *ghosts;
            let tracked mut fg = ghosts.tracked_remove(p.frame.0 as nat);
            let tracked mut tok_opt: Option<SlotToken> = None;
            vstd::modes::tracked_swap(&mut fg.token, &mut tok_opt);
            let tracked mut tok = tok_opt.tracked_unwrap();
            instance.slot_writer_is_latest(w, e, &mut tok, &after_ledger.submitted);
            fg.token = Some(tok);
            ghosts.tracked_insert(p.frame.0 as nat, fg);
            assert(*ghosts =~= prev_ghosts);
            assert(before.ledger_spec()@.submitted.dom().contains(e));
            assert(false);
        }
        assert(preserves_destination(before.ledger_spec()@, *after_ledger, before.node(), loc));
    }
}

proof fn prove_cuda_puts_valid_after_import(
    before: &CacheCore,
    after: &CacheCore,
    tracked reg_opt: &Option<VersionRegistry>,
    tracked pending_bindings: &Map<VersionId, VersionBinding>,
    fp: CoreFootprint,
    puts: Seq<PendingGpuWrite>,
    tl: TimelineId,
    n: nat,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.all_versions_bound(),
        *reg_opt == after.version_registry_spec()@,
        core_transition(before, after, fp),
        tl < before.timelines_spec().count_spec(),
        !fp.touched_timelines.contains(tl),
        fp.touched_init_frames.is_empty(),
        cuda_puts_valid(puts, before, tl),
        cuda_puts_bindings_covered(puts, *pending_bindings),
        n <= puts.len(),
        forall|i: int|
            0 <= i < n && ((#[trigger] puts[i]).action is Publish) ==> preserves_destination(
                before.ledger_spec()@,
                after.ledger_spec()@,
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
    prove_cuda_puts_valid_after_import(
        before,
        after,
        reg_opt,
        pending_bindings,
        fp,
        puts,
        tl,
        (n - 1) as nat,
    );
    let idx = (n - 1) as int;
    let p = puts[idx];
    assert(p.valid(before, tl)) by {
        reveal(cuda_puts_valid);
    }
    reveal(core_transition);
    reveal(FrameTable::same_layout);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    assert(before.frames_spec().meta(p.frame).state == STATE_INITIALIZING);
    assert(after.frames_spec().meta(p.frame) == before.frames_spec().meta(p.frame));
    assert(after.frames_spec().frame(p.frame) == before.frames_spec().frame(p.frame));
    assert(after.timelines_spec().next_seq_spec()[tl as int]
        == before.timelines_spec().next_seq_spec()[tl as int]);
    let s = p.ticket.sequence();
    assert(1 <= s < before.timelines_spec().next_seq_spec()[tl as int]);
    assert(after.timelines_spec().events_spec()@[(tl, s)]
        == before.timelines_spec().events_spec()@[(tl, s)]);
    assert(after.timelines_spec().issued(tl, s));
    assert(before.citations().subset_of(after.citations())) by {
        assert forall|e: TransferEvent|
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
        }
    }
    let tok_spec = before.frames_spec().frame(p.frame).token.unwrap();
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
                        1 <= q <= s && (#[trigger] before.timelines_spec().events_spec()@[(tl, q)])
                            == w;
                    assert(1 <= q < before.timelines_spec().next_seq_spec()[tl as int]);
                    assert(after.timelines_spec().events_spec()@[(tl, q)] == w);
                }
            }
        },
        PendingGpuAction::Publish { version, proof: put_proof, .. } => {
            assert(p.action is Publish);
            put_proof@.preserve(before.ledger_spec()@, after.ledger_spec()@, after.node());
            assert(cuda_put_events_covered(
                put_proof@.events,
                after.timelines_spec().events_spec()@,
                tl,
                s,
            )) by {
                reveal(cuda_put_events_covered);
                assert forall|k: int| 0 <= k < put_proof@.events.len() implies cuda_event_at_prefix(
                    #[trigger] put_proof@.events[k],
                    after.timelines_spec().events_spec()@,
                    tl,
                    s,
                ) by {
                    assert(cuda_event_at_prefix(
                        put_proof@.events[k],
                        before.timelines_spec().events_spec()@,
                        tl,
                        s,
                    ));
                    let q = choose|q: u64|
                        1 <= q <= s && (#[trigger] before.timelines_spec().events_spec()@[(tl, q)])
                            == put_proof@.events[k]
                            && put_proof@.events[k].source.succeeds_on_observe();
                    assert(1 <= q < before.timelines_spec().next_seq_spec()[tl as int]);
                    assert(after.timelines_spec().events_spec()@[(tl, q)] == put_proof@.events[k]);
                }
            }
            if before.frames_spec().versions_spec()@.contains_key(version) {
                lemma_map_submap_get_local(
                    before.frames_spec().versions_spec()@,
                    after.frames_spec().versions_spec()@,
                    version,
                );
            } else if after.frames_spec().versions_spec()@.contains_key(version) {
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

/// Reestablish the CUDA invariants after a step that can import external events.
pub(crate) proof fn lemma_core_transition_preserves_cuda_after_import<'a>(
    before: &CacheCore,
    after: &CacheCore,
    tracked reg_opt: &Option<VersionRegistry>,
    tracked pending_bindings: &Map<VersionId, VersionBinding>,
    fp: CoreFootprint,
    cuda: &CudaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.all_versions_bound(),
        *reg_opt == after.version_registry_spec()@,
        core_transition(before, after, fp),
        cuda.well_formed(before),
        cuda_puts_bindings_covered(cuda.active_gpu_puts@, *pending_bindings),
        !fp.touched_timelines.contains(cuda.timeline.timeline_id),
        fp.touched_init_frames.is_empty(),
        source_fresh_for_ledger(cuda.timeline.stream.source(), after.ledger_spec()@),
        forall|i: int|
            0 <= i < cuda.active_gpu_puts@.len() && ((
            #[trigger] cuda.active_gpu_puts@[i]).action is Publish) ==> preserves_destination(
                before.ledger_spec()@,
                after.ledger_spec()@,
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

    reveal(core_transition);
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
    assert(after.timelines_spec().next_seq_spec()[t as int]
        == before.timelines_spec().next_seq_spec()[t as int]);
    assert(after.timelines_spec().confirmed_seq_spec()[t as int]
        == before.timelines_spec().confirmed_seq_spec()[t as int]);
    assert(before.timelines_spec().next_seq_spec()[t as int] >= 1);
    vstd::imap_lib::lemma_submap_of_trans(
        tl.stream.infos().map(),
        before.ledger_spec()@.submitted.map(),
        after.ledger_spec()@.submitted.map(),
    );
    assert(after.timelines_spec().prefix_in(
        t,
        (after.timelines_spec().next_seq_spec()[t as int] - 1) as u64,
        tl.closure(),
    )) by {
        assert forall|s: u64|
            1 <= s <= (after.timelines_spec().next_seq_spec()[t as int]
                - 1) as u64 implies tl.closure().contains(
            #[trigger] after.timelines_spec().events_spec()@[(t, s)],
        ) by {
            assert(1 <= s < before.timelines_spec().next_seq_spec()[t as int]);
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
    reveal(event_ring_valid);
    assert forall|i: int| 0 <= i < tl.event_ring@.len() implies #[trigger] tl.event_ring@[i].valid(
        &after.timelines_spec(),
        t,
        tl.device,
        after.node(),
        &after.ledger_spec()@,
    ) by {
        let slot = tl.event_ring@[i];
        assert(slot.valid(
            &before.timelines_spec(),
            t,
            tl.device,
            before.node(),
            &before.ledger_spec()@,
        ));
        vstd::imap_lib::lemma_submap_of_trans(
            slot.event.infos().map(),
            before.ledger_spec()@.submitted.map(),
            after.ledger_spec()@.submitted.map(),
        );
        let c = foundation::tracked::closure_from(slot.event.infos().map(), slot.event.recorded());
        assert(slot.recorded_seq < before.timelines_spec().next_seq_spec()[t as int]);
        assert forall|s: u64| 1 <= s <= slot.recorded_seq implies c.contains(
            #[trigger] after.timelines_spec().events_spec()@[(t, s)],
        ) by {
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
    prove_cuda_puts_valid_after_import(
        before,
        after,
        reg_opt,
        pending_bindings,
        fp,
        cuda.active_gpu_puts@,
        t,
        cuda.active_gpu_puts@.len(),
    );
    reveal(cuda_puts_valid);
}

} // verus!
