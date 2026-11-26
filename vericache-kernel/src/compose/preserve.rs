use super::{core_transition, CoreFootprint};
use crate::kernel::CacheCore;
use crate::spec::{FrameId, TimelineId};
use crate::spi::source_fresh_for_ledger;
use transfer_core::context::{slot_frontier, write_admitted, EventSourceToken, SlotToken};
use transfer_core::domain::{SlotLoc, TransferEvent};
use vstd::prelude::*;

verus! {

broadcast use vstd::iset::group_iset_lemmas;

pub proof fn lemma_core_transition_preserves_source(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    source: EventSourceToken,
)
    requires
        core_transition(before, after, fp),
        source_fresh_for_ledger(source, before.ledger_spec()@),
        fp.excludes_source(source.source()),
    ensures
        source_fresh_for_ledger(source, after.ledger_spec()@),
{
    reveal(core_transition);
    assert forall|e: TransferEvent| #[trigger]
        after.ledger_spec()@.submitted.dom().contains(e) && e.source == source.source()
            && e.observer == source.node() implies e.seq < source.next_seq() by {
        assert(before.ledger_spec()@.submitted.dom().contains(e));
    }
}

pub proof fn lemma_core_transition_event_eq(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    t: TimelineId,
    s: u64,
)
    requires
        core_transition(before, after, fp),
        t < before.timelines_spec().count_spec(),
        !fp.touched_timelines.contains(t),
        1 <= s < before.timelines_spec().next_seq_spec()[t as int],
    ensures
        after.timelines_spec().events_spec()@[(t, s)] == before.timelines_spec().events_spec()@[(
            t,
            s,
        )],
{
    reveal(core_transition);
}

pub proof fn lemma_core_transition_preserves_slot_disjoint_and_admitted(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    tok: SlotToken,
    loc: SlotLoc,
)
    requires
        core_transition(before, after, fp),
        write_admitted(slot_frontier(tok), before.citations()),
        forall|f: FrameId|
            f.0 < before.frames_spec().metas_spec().len() ==> loc != (
            #[trigger] before.frames_spec().frame(f)).slot.loc,
    ensures
        write_admitted(slot_frontier(tok), after.citations()),
        after.disjoint_from_published(ISet::empty().insert(loc)),
        forall|f: FrameId|
            f.0 < after.frames_spec().metas_spec().len() ==> loc != (
            #[trigger] after.frames_spec().frame(f)).slot.loc,
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    reveal(core_transition);
    assert(before.citations().subset_of(after.citations())) by {
        assert forall|e: TransferEvent|
            before.citations().contains(e) implies after.citations().contains(e) by {
            assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
        }
    }
    assert forall|f: FrameId| f.0 < after.frames_spec().metas_spec().len() implies loc != (
    #[trigger] after.frames_spec().frame(f)).slot.loc by {
        assert(after.frames_spec().frame(f).slot == before.frames_spec().frame(f).slot);
    }
}

} // verus!
