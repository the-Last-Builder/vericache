use super::*;

verus! {

broadcast use vstd::iset::group_iset_lemmas;

pub(crate) proof fn lemma_core_transition_preserves_host(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    host: &HostBackend,
)
    requires
        core_transition(before, after, fp),
        host.well_formed(before),
        !fp.touched_timelines.contains(host.timeline),
        fp.excludes_source(host.ctx.source@.source()) || source_fresh_for_ledger(
            host.ctx.source@,
            after.ledger_spec()@,
        ),
    ensures
        host.well_formed(after),
{
    reveal(core_transition);
    assert(after.timelines_spec().next_seq_spec()[host.timeline as int]
        == before.timelines_spec().next_seq_spec()[host.timeline as int]);
    assert(after.timelines_spec().confirmed_seq_spec()[host.timeline as int]
        == before.timelines_spec().confirmed_seq_spec()[host.timeline as int]);
    if fp.excludes_source(host.ctx.source@.source()) {
        lemma_core_transition_preserves_source(before, after, fp, host.ctx.source@);
    }
    lemma_slab_bound_same_layout(
        before,
        after,
        host.tier,
        host.first_frame,
        host.num_frames,
        host.buffer.space(),
    );
}

} // verus!
