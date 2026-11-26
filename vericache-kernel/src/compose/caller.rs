use super::{core_transition, CoreFootprint};
use crate::frame::FrameTable;
use crate::kernel::CacheCore;
use crate::spec::{FrameId, TimelineId};
use crate::spi::lemma_core_wf_facts;
use transfer_core::domain::{SlotLoc, TransferEvent};
use vstd::prelude::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    pub open spec fn writes_disjoint_from_frames(&self, writes: ISet<SlotLoc>) -> bool {
        forall|f: FrameId|
            f.0 < self.frames_spec().metas_spec().len() ==> !writes.contains(
                #[trigger] self.frames_spec().frame(f).slot.loc,
            )
    }
}

pub proof fn lemma_core_transition_after_timeline_append(before: &CacheCore, after: &CacheCore)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id_spec() == before.node_id_spec(),
        after.ledger_spec() == before.ledger_spec(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec() == before.frames_spec(),
        after.active_leases_spec() == before.active_leases_spec(),
        after.next_version_seq_spec() == before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.timelines_spec().count_spec() >= before.timelines_spec().count_spec(),
        after.timelines_spec().events_spec() == before.timelines_spec().events_spec(),
        forall|t: TimelineId|
            t < before.timelines_spec().count_spec() ==> {
                &&& (#[trigger] after.timelines_spec().kinds_spec()[t as int])
                    == before.timelines_spec().kinds_spec()[t as int]
                &&& after.timelines_spec().next_seq_spec()[t as int]
                    == before.timelines_spec().next_seq_spec()[t as int]
                &&& after.timelines_spec().confirmed_seq_spec()[t as int]
                    == before.timelines_spec().confirmed_seq_spec()[t as int]
            },
    ensures
        core_transition(before, after, CoreFootprint::preserve_all(None, None)),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    reveal(core_transition);
    reveal(FrameTable::same_layout);
    assert forall|t: TimelineId| t < before.timelines_spec().count_spec() implies {
        &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
            == before.timelines_spec().next_seq_spec()[t as int]
        &&& after.timelines_spec().confirmed_seq_spec()[t as int]
            == before.timelines_spec().confirmed_seq_spec()[t as int]
    } by {
        assert(after.timelines_spec().kinds_spec()[t as int]
            == before.timelines_spec().kinds_spec()[t as int]);
    }

}

pub proof fn lemma_core_transition_external_event(
    before: &CacheCore,
    after: &CacheCore,
    timeline: Option<TimelineId>,
    event: TransferEvent,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec() == before.frames_spec(),
        after.active_leases_spec() == before.active_leases_spec(),
        after.next_version_seq_spec() == before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.timelines_spec().count_spec() == before.timelines_spec().count_spec(),
        after.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
        after.timelines_spec().confirmed_seq_spec() == before.timelines_spec().confirmed_seq_spec(),
        match timeline {
            None => after.timelines_spec() == before.timelines_spec(),
            Some(t) => {
                &&& after.timelines_spec().next_seq_spec()@
                    == before.timelines_spec().next_seq_spec()@.update(
                    t as int,
                    (before.timelines_spec().next_seq_spec()[t as int] + 1) as u64,
                )
                &&& after.timelines_spec().events_spec()@
                    == before.timelines_spec().events_spec()@.insert(
                    (t, before.timelines_spec().next_seq_spec()[t as int]),
                    event,
                )
            },
        },
        before.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map()),
        after.ledger_spec()@.submitted.dom() =~= before.ledger_spec()@.submitted.dom().insert(
            event,
        ),
        after.ledger_spec()@.revoked == before.ledger_spec()@.revoked,
        before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
        after.ledger_spec()@.succeeded == before.ledger_spec()@.succeeded,
        before.writes_disjoint_from_frames(
            after.ledger_spec()@.submitted.map()[event].access.writes,
        ),
    ensures
        core_transition(before, after, CoreFootprint::preserve_all(timeline, Some(event.source))),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    reveal(core_transition);
    reveal(FrameTable::same_layout);
}

pub proof fn lemma_core_transition_after_external_completion(
    before: &CacheCore,
    after: &CacheCore,
    timeline: TimelineId,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id_spec() == before.node_id_spec(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec() == before.frames_spec(),
        after.active_leases_spec() == before.active_leases_spec(),
        after.next_version_seq_spec() == before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.timelines_spec().polled(&before.timelines_spec(), timeline),
        before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
        after.ledger_spec()@.submitted == before.ledger_spec()@.submitted,
        after.ledger_spec()@.succeeded == before.ledger_spec()@.succeeded,
        after.ledger_spec()@.revoked == before.ledger_spec()@.revoked,
    ensures
        core_transition(before, after, CoreFootprint::preserve_all(Some(timeline), None)),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    reveal(core_transition);
    reveal(FrameTable::same_layout);
}

} // verus!
