use super::*;

verus! {

broadcast use vstd::iset::group_iset_lemmas;

pub open(crate) spec fn nvme_poll_footprint(
    timeline: TimelineId,
    pending: Seq<Option<InFlightNvmeOp>>,
) -> CoreFootprint {
    CoreFootprint::completion(
        timeline,
        ISet::new(
            |f: FrameId|
                exists|i: int|
                    0 <= i < pending.len() && ((#[trigger] pending[i]) matches Some(op)
                        && op.hop.dst_frame == f),
        ),
        ISet::new(
            |s: u32|
                exists|i: int|
                    0 <= i < pending.len() && ((#[trigger] pending[i]) matches Some(op)
                        && op.hop.src_lease.slot_spec() == s),
        ),
        VersionFootprint::unchanged(),
    )
}

impl NvmeBackend {
    pub open(crate) spec fn poll_footprint(&self) -> CoreFootprint {
        nvme_poll_footprint(self.timeline, self.pending_transfers@)
    }

    pub open(crate) spec fn pending_transfers_spec(&self) -> Seq<Option<InFlightNvmeOp>> {
        self.pending_transfers@
    }

    pub open(crate) spec fn pending_from(&self, before: &Self) -> bool {
        &&& self.pending_transfers@.len() == before.pending_transfers@.len()
        &&& forall|i: int|
            0 <= i < self.pending_transfers@.len() && (
            #[trigger] self.pending_transfers@[i]).is_some() ==> self.pending_transfers@[i]
                == before.pending_transfers@[i]
    }

    pub open(crate) spec fn pending_untouched(
        &self,
        before: &CacheCore,
        fp: CoreFootprint,
    ) -> bool {
        let pending = self.pending_transfers@;
        &&& (fp.touched_init_frames.is_empty() || (forall|f: FrameId| #[trigger]
            fp.touched_init_frames.contains(f) && f.0 < before.frames_spec().metas_spec().len()
                ==> before.frames_spec().frame(f).token is Some || (before.frames_spec().frame(
                f,
            ).bound_version matches Some(v) && !before.frames_spec().versions_spec()@.contains_key(
                v,
            ))) || (forall|i: int|
            0 <= i < pending.len() ==> ((#[trigger] pending[i]) matches Some(op)
                ==> !fp.touched_init_frames.contains(op.hop.dst_frame))))
        &&& (fp.touched_internal_leases.is_empty() || forall|i: int|
            0 <= i < pending.len() ==> ((#[trigger] pending[i]) matches Some(op)
                ==> !fp.touched_internal_leases.contains(op.hop.src_lease.slot_spec())))
    }
}

pub(super) proof fn lemma_nvme_poll_footprint_frames(backend: &NvmeBackend, before: &CacheCore)
    requires
        backend.well_formed(before),
    ensures
        forall|f: FrameId| #[trigger]
            backend.poll_footprint().touched_init_frames.contains(f) ==> before.frames_spec().frame(
                f,
            ).token is None,
{
    reveal(nvme_pending_valid);
    broadcast use vstd::iset::group_iset_lemmas;

    let pending = backend.pending_transfers@;
    assert forall|f: FrameId| #[trigger]
        backend.poll_footprint().touched_init_frames.contains(f) implies before.frames_spec().frame(
        f,
    ).token is None by {
        let i = choose|i: int|
            0 <= i < pending.len() && ((#[trigger] pending[i]) matches Some(op) && op.hop.dst_frame
                == f);
        assert(pending[i]->Some_0.valid(before, backend.timeline, i));
    }
}

pub(super) proof fn lemma_nvme_poll_contains(
    pending: Seq<Option<InFlightNvmeOp>>,
    timeline: TimelineId,
    i: int,
    op: InFlightNvmeOp,
)
    requires
        0 <= i < pending.len(),
        pending[i] == Some(op),
    ensures
        nvme_poll_footprint(timeline, pending).touched_init_frames.contains(op.hop.dst_frame),
        nvme_poll_footprint(timeline, pending).touched_internal_leases.contains(
            op.hop.src_lease.slot_spec(),
        ),
{
    broadcast use vstd::iset::group_iset_lemmas;

}

pub(crate) proof fn lemma_nvme_pending_from_trans(a: &NvmeBackend, b: &NvmeBackend, c: &NvmeBackend)
    requires
        c.pending_from(b),
        b.pending_from(a),
    ensures
        c.pending_from(a),
{
    assert forall|i: int|
        0 <= i < c.pending_transfers@.len() && (
        #[trigger] c.pending_transfers@[i]).is_some() implies c.pending_transfers@[i]
        == a.pending_transfers@[i] by {
        assert(c.pending_transfers@[i] == b.pending_transfers@[i]);
    }
}

/// All operations still pending during a drain belonged to its initial queue.
pub(crate) proof fn lemma_nvme_widen_poll_footprint(
    before: &CacheCore,
    after: &CacheCore,
    current: &NvmeBackend,
    initial: &NvmeBackend,
)
    requires
        core_transition(before, after, current.poll_footprint()),
        initial.timeline == current.timeline,
        current.pending_from(initial),
    ensures
        core_transition(before, after, initial.poll_footprint()),
{
    let current_ops = current.pending_transfers@;
    let initial_ops = initial.pending_transfers@;
    let smaller = current.poll_footprint();
    let larger = initial.poll_footprint();
    assert forall|f: FrameId| #[trigger]
        smaller.touched_init_frames.contains(f) implies larger.touched_init_frames.contains(f) by {
        let i = choose|i: int|
            0 <= i < current_ops.len() && ((#[trigger] current_ops[i]) matches Some(op)
                && op.hop.dst_frame == f);
        assert(initial_ops[i] == current_ops[i]);
    }
    assert forall|s: u32| #[trigger]
        smaller.touched_internal_leases.contains(s) implies larger.touched_internal_leases.contains(
        s,
    ) by {
        let i = choose|i: int|
            0 <= i < current_ops.len() && ((#[trigger] current_ops[i]) matches Some(op)
                && op.hop.src_lease.slot_spec() == s);
        assert(initial_ops[i] == current_ops[i]);
    }
    reveal(core_transition);
}

pub(crate) proof fn lemma_core_transition_preserves_nvme(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    nvme: &NvmeBackend,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        nvme.well_formed(before),
        !fp.touched_timelines.contains(nvme.timeline),
        nvme.pending_untouched(before, fp),
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
    assert(after.timelines_spec().next_seq_spec()[t as int]
        == before.timelines_spec().next_seq_spec()[t as int]);
    assert(after.timelines_spec().confirmed_seq_spec()[t as int]
        == before.timelines_spec().confirmed_seq_spec()[t as int]);
    reveal(nvme_window_wf);
    assert(nvme.bitmap.observations(
        &after.timelines_spec(),
        t,
        after.node(),
        &after.ledger_spec()@,
    )) by {
        assert forall|i: int| #[trigger] nvme.bitmap.bit_set(i) implies {
            let s = (after.timelines_spec().confirmed_seq_spec()[t as int] + 1 + i) as u64;
            after.timelines_spec().issued(t, s) && after.timelines_spec().observed(
                t,
                s,
                after.node(),
                after.ledger_spec()@.knows,
            )
        } by {
            let s = (before.timelines_spec().confirmed_seq_spec()[t as int] + 1 + i) as u64;
            assert(before.timelines_spec().issued(t, s));
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
            assert(before.timelines_spec().observed(
                t,
                s,
                before.node(),
                before.ledger_spec()@.knows,
            ));
        }
    }
    assert(window_matches(
        nvme.in_flight_ops,
        nvme.bitmap,
        &after.timelines_spec(),
        t,
        ring_event_map(nvme.ring),
    )) by {
        let pending = ring_event_map(nvme.ring);
        assert forall|s: u64| #[trigger]
            pending.contains_key(s) implies after.timelines_spec().confirmed_seq_spec()[t as int]
            < s < after.timelines_spec().next_seq_spec()[t as int] && pending[s]
            == after.timelines_spec().events_spec()@[(t, s)] && (nvme.in_flight_ops[(s
            % BITMAP_CAPACITY) as int] matches Some(op) && op.sequence == s) by {
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
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
            assert(!fp.touched_init_frames.contains(op.hop.dst_frame));
            assert(!fp.touched_internal_leases.contains(op.hop.src_lease.slot_spec()));
            let s = op.ticket.sequence();
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
            assert(after.frames_spec().meta(op.hop.dst_frame) == before.frames_spec().meta(
                op.hop.dst_frame,
            ));
            assert(after.frames_spec().frame(op.hop.dst_frame) == before.frames_spec().frame(
                op.hop.dst_frame,
            ));
            assert(after.active_leases_spec()[op.hop.src_lease.slot_spec() as int]
                == before.active_leases_spec()[op.hop.src_lease.slot_spec() as int]);
            assert(before.frames_spec().versions_spec()@.dom().contains(op.hop.version));
            assert(after.frames_spec().versions_spec()@.dom().contains(op.hop.version));
            assert(before.citations().subset_of(after.citations())) by {
                assert forall|e: TransferEvent|
                    before.citations().contains(e) implies after.citations().contains(e) by {
                    assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
                }
            }
            vstd::imap_lib::lemma_submap_of_trans(
                op.copy@.before.submitted.map(),
                before.ledger_spec()@.submitted.map(),
                after.ledger_spec()@.submitted.map(),
            );
            vstd::imap_lib::lemma_submap_of_trans(
                op.copy@.before.revoked.map(),
                before.ledger_spec()@.revoked.map(),
                after.ledger_spec()@.revoked.map(),
            );
            assert(after.ledger_spec()@.submitted.dom().contains(op.copy@.writer));
            assert(after.ledger_spec()@.submitted.map()[op.copy@.writer]
                == before.ledger_spec()@.submitted.map()[op.copy@.writer]);
            assert(nvme_copy_retained(op.copy@, after.ledger_spec()@, after.node()));
        }
    }
}

} // verus!
