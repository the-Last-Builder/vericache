use super::*;
use vstd::prelude::*;

mod preserve;
pub use preserve::*;

verus! {

/// Classification of which initializing frames and internal leases a verified
/// backend step may modify on `CacheCore`.
pub ghost enum FrameTouchMode {
    /// No pre-existing `STATE_INITIALIZING` frame modified, no pre-existing
    /// `internal` active lease modified, and any newly added version was freshly
    /// minted at `origin_node == before.node_id && before.next_version_seq <= seq < before.version_seq_limit`.
    PreserveAllInit,
    /// Only parked (`token.is_some()`) initializing frames modified (CUDA poll);
    /// all unparked (`token.is_none()`) initializing frames and internal leases
    /// are preserved.
    PreserveUnparkedInit,
    /// Only unparked (`token.is_none()`) initializing frames modified (NVMe poll);
    /// all parked (`token.is_some()`) initializing frames are preserved and no
    /// new versions are added.
    PreserveParkedInit,
}

/// Backend-agnostic opaque step summary describing how one backend mutated
/// `CacheCore` so sibling backends and caller streams can preserve their
/// `well_formed` invariants without unfolding `CacheCore::well_formed`.
#[verifier::opaque]
pub open spec fn core_transition(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
) -> bool {
    &&& after.node_id == before.node_id
    &&& after.instance_id() == before.instance_id()
    &&& after.tiers == before.tiers
    &&& after.frames.same_layout(&before.frames)
    &&& after.frames.preserves_remote_leases(&before.frames)
    &&& after.next_version_seq >= before.next_version_seq
    &&& after.version_seq_limit == before.version_seq_limit
    &&& before.frames.versions@.submap_of(after.frames.versions@)
    &&& after.timelines.count == before.timelines.count
    &&& after.timelines.kinds == before.timelines.kinds
    &&& forall|t: u8|
        t < before.timelines.count && touched_timeline != Some(t) ==> {
            &&& (#[trigger] after.timelines.next_seq[t as int])
                == before.timelines.next_seq[t as int]
            &&& after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]
        }
    &&& forall|t: u8, s: u64|
        t < before.timelines.count && touched_timeline != Some(t) && 1 <= s
            < before.timelines.next_seq[t as int] ==> #[trigger] after.timelines.events@[(t, s)]
            == before.timelines.events@[(t, s)]
    &&& before.ledger@.submitted.map().submap_of(after.ledger@.submitted.map())
    &&& revoked_submap(before.ledger@.revoked, after.ledger@.revoked)
    &&& before.ledger@.knows.set().subset_of(after.ledger@.knows.set())
    &&& before.ledger@.succeeded.dom().subset_of(after.ledger@.succeeded.dom())
    &&& forall|e: TransferEvent| #[trigger]
        after.ledger@.submitted.dom().contains(e) && !before.ledger@.submitted.dom().contains(e)
            ==> {
            &&& allowed_source == Some(e.source)
            &&& forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING
                    ==> !after.ledger@.submitted.map()[e].access.writes.contains(
                    before.frames.frame(f).slot.loc,
                )
        }
    &&& match mode {
        FrameTouchMode::PreserveAllInit => {
            &&& after.frames.preserves_initializing(&before.frames, None)
            &&& after.active_leases.len() == before.active_leases.len()
            &&& forall|s: int|
                0 <= s < before.active_leases.len() && (
                #[trigger] before.active_leases[s] matches Some(l) && l.internal)
                    ==> after.active_leases[s] == before.active_leases[s]
            &&& forall|v: VersionId| #[trigger]
                after.frames.versions@.contains_key(v) && !before.frames.versions@.contains_key(v)
                    ==> v.origin_node == before.node_id && before.next_version_seq <= v.seq && v.seq
                    < before.version_seq_limit
        },
        FrameTouchMode::PreserveUnparkedInit => {
            &&& forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING && before.frames.frame(f).token is None ==> {
                    &&& after.frames.meta(f) == before.frames.meta(f)
                    &&& after.frames.frame(f) == before.frames.frame(f)
                }
            &&& after.active_leases.len() == before.active_leases.len()
            &&& forall|s: int|
                0 <= s < before.active_leases.len() && (
                #[trigger] before.active_leases[s] matches Some(l) && l.internal)
                    ==> after.active_leases[s] == before.active_leases[s]
        },
        FrameTouchMode::PreserveParkedInit => {
            &&& forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING && before.frames.frame(f).token is Some ==> {
                    &&& after.frames.meta(f) == before.frames.meta(f)
                    &&& after.frames.frame(f) == before.frames.frame(f)
                }
            &&& after.frames.versions@ == before.frames.versions@
        },
    }
}

pub proof fn lemma_core_transition_refl(
    core: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
)
    ensures
        core_transition(core, core, touched_timeline, allowed_source, mode),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
}

pub proof fn lemma_core_transition_no_event(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
)
    requires
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.tiers == before.tiers,
        after.frames.same_directory(&before.frames),
        after.frames.preserves_initializing(&before.frames, None),
        after.frames.preserves_remote_leases(&before.frames),
        after.next_version_seq >= before.next_version_seq,
        after.version_seq_limit == before.version_seq_limit,
        after.timelines == before.timelines || after.timelines.only_watermarks(&before.timelines),
        after.ledger == before.ledger,
        after.active_leases.len() == before.active_leases.len(),
        forall|s: int|
            0 <= s < before.active_leases.len() && (#[trigger] before.active_leases[s] matches Some(
                l,
            ) && l.internal) ==> after.active_leases[s] == before.active_leases[s],
    ensures
        core_transition(
            before,
            after,
            touched_timeline,
            allowed_source,
            FrameTouchMode::PreserveAllInit,
        ),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    reveal(FrameTable::same_directory);
    reveal(FrameTable::same_layout);
}

pub proof fn lemma_core_transition_single_event(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    source: EventSource,
    ev: TransferEvent,
    write_loc: Option<SlotLoc>,
    new_version: Option<VersionId>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.tiers == before.tiers,
        after.frames.same_layout(&before.frames),
        after.frames.preserves_initializing(&before.frames, None),
        after.frames.preserves_remote_leases(&before.frames),
        after.next_version_seq >= before.next_version_seq,
        after.version_seq_limit == before.version_seq_limit,
        before.frames.versions@.submap_of(after.frames.versions@),
        match new_version {
            None => after.frames.versions@ == before.frames.versions@,
            Some(nv) => nv.origin_node == before.node_id && before.next_version_seq <= nv.seq
                && nv.seq < before.version_seq_limit && after.frames.versions@.dom()
                == before.frames.versions@.dom().insert(nv),
        },
        after.timelines.count == before.timelines.count,
        after.timelines.kinds == before.timelines.kinds,
        forall|t: u8|
            t < before.timelines.count && touched_timeline != Some(t) ==> {
                &&& (#[trigger] after.timelines.next_seq[t as int])
                    == before.timelines.next_seq[t as int]
                &&& after.timelines.confirmed_seq[t as int]
                    == before.timelines.confirmed_seq[t as int]
                &&& forall|s: u64|
                    1 <= s < before.timelines.next_seq[t as int]
                        ==> #[trigger] after.timelines.events@[(t, s)] == before.timelines.events@[(
                        t,
                        s,
                    )]
            },
        before.ledger@.submitted.map().submap_of(after.ledger@.submitted.map()),
        revoked_submap(before.ledger@.revoked, after.ledger@.revoked),
        before.ledger@.knows.set().subset_of(after.ledger@.knows.set()),
        before.ledger@.succeeded.dom().subset_of(after.ledger@.succeeded.dom()),
        after.ledger@.submitted.dom() == before.ledger@.submitted.dom().insert(ev),
        ev.source == source,
        match write_loc {
            None => after.ledger@.submitted.map()[ev].access.writes.is_empty(),
            Some(loc) => {
                &&& after.ledger@.submitted.map()[ev].access.writes == ISet::empty().insert(loc)
                &&& forall|f: FrameId|
                    f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                        == STATE_INITIALIZING ==> loc != before.frames.frame(f).slot.loc
            },
        },
        after.active_leases.len() == before.active_leases.len(),
        forall|s: int|
            0 <= s < before.active_leases.len() && (#[trigger] before.active_leases[s] matches Some(
                l,
            ) && l.internal) ==> after.active_leases[s] == before.active_leases[s],
    ensures
        core_transition(
            before,
            after,
            touched_timeline,
            Some(source),
            FrameTouchMode::PreserveAllInit,
        ),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    assert forall|t: u8, s: u64|
        t < before.timelines.count && touched_timeline != Some(t) && 1 <= s
            < before.timelines.next_seq[t as int] implies #[trigger] after.timelines.events@[(t, s)]
        == before.timelines.events@[(t, s)] by {
        assert(after.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
    }
    assert forall|e: TransferEvent| #[trigger]
        after.ledger@.submitted.dom().contains(e) && !before.ledger@.submitted.dom().contains(
            e,
        ) implies Some(source) == Some(e.source) && forall|f: FrameId|
        f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
            == STATE_INITIALIZING ==> !after.ledger@.submitted.map()[e].access.writes.contains(
            before.frames.frame(f).slot.loc,
        ) by {
        assert(e == ev);
    }
}

pub proof fn lemma_core_transition_trans(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    mode: FrameTouchMode,
)
    requires
        core_transition(a, b, touched_timeline, allowed_source, mode),
        core_transition(b, c, touched_timeline, allowed_source, mode),
        (mode is PreserveUnparkedInit || mode is PreserveParkedInit) ==> allowed_source is None,
    ensures
        core_transition(a, c, touched_timeline, allowed_source, mode),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    vstd::map_lib::lemma_submap_of_trans(
        a.frames.versions@,
        b.frames.versions@,
        c.frames.versions@,
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.submitted.map(),
        b.ledger@.submitted.map(),
        c.ledger@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.revoked.map(),
        b.ledger@.revoked.map(),
        c.ledger@.revoked.map(),
    );
    assert(c.frames.preserves_remote_leases(&a.frames)) by {
        assert forall|f: FrameId| f.0 < a.frames.metas.len() implies {
            &&& (#[trigger] c.frames.frame(f)).remote_shares == a.frames.frame(f).remote_shares
            &&& c.frames.meta(f).active_remote_leases == a.frames.meta(f).active_remote_leases
        } by {
            assert(b.frames.frame(f).remote_shares == a.frames.frame(f).remote_shares);
            assert(b.frames.meta(f).active_remote_leases == a.frames.meta(f).active_remote_leases);
        }
    }
    assert forall|e: TransferEvent| #[trigger]
        c.ledger@.submitted.dom().contains(e) && !a.ledger@.submitted.dom().contains(
            e,
        ) implies allowed_source == Some(e.source) && forall|f: FrameId|
        f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state == STATE_INITIALIZING
            ==> !c.ledger@.submitted.map()[e].access.writes.contains(
            a.frames.frame(f).slot.loc,
        ) by {
        if b.ledger@.submitted.dom().contains(e) {
            assert(c.ledger@.submitted.map()[e] == b.ledger@.submitted.map()[e]);
        } else {
            assert forall|f: FrameId|
                f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state
                    == STATE_INITIALIZING implies !c.ledger@.submitted.map()[e].access.writes.contains(
            a.frames.frame(f).slot.loc) by {
                match mode {
                    FrameTouchMode::PreserveAllInit => {
                        assert(b.frames.meta(f) == a.frames.meta(f));
                        assert(b.frames.frame(f) == a.frames.frame(f));
                    },
                    FrameTouchMode::PreserveUnparkedInit => {
                        assert(allowed_source == Some(e.source));
                    },
                    FrameTouchMode::PreserveParkedInit => {
                        assert(allowed_source == Some(e.source));
                    },
                }
            }
        }
    }
    match mode {
        FrameTouchMode::PreserveAllInit => {
            assert forall|f: FrameId|
                f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state
                    == STATE_INITIALIZING implies c.frames.meta(f) == a.frames.meta(f)
                && c.frames.frame(f) == a.frames.frame(f) by {
                assert(b.frames.meta(f) == a.frames.meta(f));
                assert(b.frames.frame(f) == a.frames.frame(f));
            }
            assert forall|s: int|
                0 <= s < a.active_leases.len() && (#[trigger] a.active_leases[s] matches Some(l)
                    && l.internal) implies c.active_leases[s] == a.active_leases[s] by {
                assert(b.active_leases[s] == a.active_leases[s]);
            }
        },
        FrameTouchMode::PreserveUnparkedInit => {
            assert forall|f: FrameId|
                f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state
                    == STATE_INITIALIZING && a.frames.frame(f).token is None implies c.frames.meta(
                f,
            ) == a.frames.meta(f) && c.frames.frame(f) == a.frames.frame(f) by {
                assert(b.frames.meta(f) == a.frames.meta(f));
                assert(b.frames.frame(f) == a.frames.frame(f));
            }
            assert forall|s: int|
                0 <= s < a.active_leases.len() && (#[trigger] a.active_leases[s] matches Some(l)
                    && l.internal) implies c.active_leases[s] == a.active_leases[s] by {
                assert(b.active_leases[s] == a.active_leases[s]);
            }
        },
        FrameTouchMode::PreserveParkedInit => {
            assert forall|f: FrameId|
                f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state
                    == STATE_INITIALIZING && a.frames.frame(f).token is Some implies c.frames.meta(
                f,
            ) == a.frames.meta(f) && c.frames.frame(f) == a.frames.frame(f) by {
                assert(b.frames.meta(f) == a.frames.meta(f));
                assert(b.frames.frame(f) == a.frames.frame(f));
            }
        },
    }
}

/// Construct `core_transition` for one `CudaBackend::poll_completions` step
/// (event polling + optional single parked-write publish/abort).
pub proof fn lemma_core_transition_cuda_poll_step(
    before: &CacheCore,
    after: &CacheCore,
    tl: u8,
    retired_frame: Option<FrameId>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.tiers == before.tiers,
        after.frames.same_layout(&before.frames),
        after.frames.preserves_initializing(&before.frames, retired_frame),
        after.frames.preserves_remote_leases(&before.frames),
        retired_frame matches Some(rf) ==> rf.0 < before.frames.metas.len() && before.frames.frame(
            rf,
        ).token is Some,
        after.next_version_seq == before.next_version_seq,
        after.version_seq_limit == before.version_seq_limit,
        before.frames.versions@.submap_of(after.frames.versions@),
        after.active_leases == before.active_leases,
        after.timelines.count == before.timelines.count,
        after.timelines.kinds == before.timelines.kinds,
        after.timelines.next_seq == before.timelines.next_seq,
        after.timelines.events == before.timelines.events,
        forall|t: u8|
            t < before.timelines.count && t != tl
                ==> #[trigger] after.timelines.confirmed_seq[t as int]
                == before.timelines.confirmed_seq[t as int],
        after.ledger@.submitted == before.ledger@.submitted,
        after.ledger@.revoked == before.ledger@.revoked,
        before.ledger@.knows.set().subset_of(after.ledger@.knows.set()),
        after.ledger@.succeeded == before.ledger@.succeeded,
    ensures
        core_transition(before, after, Some(tl), None, FrameTouchMode::PreserveUnparkedInit),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
}

/// Construct `core_transition` for one `NvmeBackend` CQE completion step.
pub proof fn lemma_core_transition_nvme_poll_step(
    before: &CacheCore,
    after: &CacheCore,
    tl: u8,
    reaped_frame: FrameId,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id == before.node_id,
        after.instance_id() == before.instance_id(),
        after.tiers == before.tiers,
        after.frames.same_layout(&before.frames),
        after.frames.preserves_initializing(&before.frames, Some(reaped_frame)),
        after.frames.preserves_remote_leases(&before.frames),
        reaped_frame.0 < before.frames.metas.len(),
        before.frames.frame(reaped_frame).token is None,
        after.next_version_seq == before.next_version_seq,
        after.version_seq_limit == before.version_seq_limit,
        after.frames.versions@ == before.frames.versions@,
        after.timelines.count == before.timelines.count,
        after.timelines.kinds == before.timelines.kinds,
        after.timelines.next_seq == before.timelines.next_seq,
        after.timelines.events == before.timelines.events,
        forall|t: u8|
            t < before.timelines.count && t != tl
                ==> #[trigger] after.timelines.confirmed_seq[t as int]
                == before.timelines.confirmed_seq[t as int],
        after.ledger@.submitted == before.ledger@.submitted,
        after.ledger@.revoked == before.ledger@.revoked,
        before.ledger@.knows.set().subset_of(after.ledger@.knows.set()),
        before.ledger@.succeeded.dom().subset_of(after.ledger@.succeeded.dom()),
    ensures
        core_transition(before, after, Some(tl), None, FrameTouchMode::PreserveParkedInit),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
}

/// Extend `core_transition(before, mid, ...)` across a `mid -> after` step that
/// emits no ledger events and no new versions, modifying at most `target` (which
/// was not `STATE_INITIALIZING` in `before`) and `released_lease_slot` (which
/// was `None` in `before`).
pub proof fn lemma_core_transition_step_no_event(
    before: &CacheCore,
    mid: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<u8>,
    allowed_source: Option<EventSource>,
    target: Option<FrameId>,
    released_lease_slot: Option<u32>,
)
    requires
        core_transition(
            before,
            mid,
            touched_timeline,
            allowed_source,
            FrameTouchMode::PreserveAllInit,
        ),
        before.well_formed(),
        after.node_id == mid.node_id,
        after.instance_id() == mid.instance_id(),
        after.tiers == mid.tiers,
        after.frames.same_directory(&mid.frames),
        after.frames.preserves_initializing(&mid.frames, target),
        after.frames.preserves_remote_leases(&mid.frames),
        target matches Some(tf) ==> tf.0 < before.frames.metas.len() && before.frames.meta(tf).state
            != STATE_INITIALIZING,
        after.next_version_seq >= mid.next_version_seq,
        after.version_seq_limit == mid.version_seq_limit,
        after.timelines == mid.timelines || after.timelines.only_watermarks(&mid.timelines),
        after.ledger == mid.ledger,
        after.active_leases.len() == mid.active_leases.len(),
        released_lease_slot matches Some(sl) ==> (sl as usize) < before.active_leases.len()
            && before.active_leases[sl as int].is_none(),
        forall|s: int|
            0 <= s < mid.active_leases.len() && released_lease_slot != Some(s as u32)
                ==> #[trigger] after.active_leases[s] == mid.active_leases[s],
    ensures
        core_transition(
            before,
            after,
            touched_timeline,
            allowed_source,
            FrameTouchMode::PreserveAllInit,
        ),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    reveal(core_transition);
    reveal(FrameTable::same_directory);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    assert(after.frames.metas.len() == before.frames.metas.len());
    assert forall|f: FrameId| f.0 < before.frames.metas.len() implies {
        &&& (#[trigger] after.frames.frame(f)).slot == before.frames.frame(f).slot
        &&& after.frames.frame(f).tier == before.frames.frame(f).tier
    } by {
        assert(mid.frames.frame(f).slot == before.frames.frame(f).slot);
        assert(mid.frames.frame(f).tier == before.frames.frame(f).tier);
    }
    assert(after.frames.same_layout(&before.frames));
    assert(after.frames.preserves_remote_leases(&before.frames)) by {
        assert forall|f: FrameId| f.0 < before.frames.metas.len() implies {
            &&& (#[trigger] after.frames.frame(f)).remote_shares == before.frames.frame(
                f,
            ).remote_shares
            &&& after.frames.meta(f).active_remote_leases == before.frames.meta(
                f,
            ).active_remote_leases
        } by {
            assert(mid.frames.frame(f).remote_shares == before.frames.frame(f).remote_shares);
            assert(mid.frames.meta(f).active_remote_leases == before.frames.meta(
                f,
            ).active_remote_leases);
        }
    }
    assert forall|t: u8| t < before.timelines.count && touched_timeline != Some(t) implies {
        &&& (#[trigger] after.timelines.next_seq[t as int]) == before.timelines.next_seq[t as int]
        &&& after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]
        &&& forall|s: u64|
            1 <= s < before.timelines.next_seq[t as int] ==> #[trigger] after.timelines.events@[(
                t,
                s,
            )] == before.timelines.events@[(t, s)]
    } by {
        assert(mid.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
        assert(mid.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]);
    }
    assert forall|f: FrameId|
        f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
            == STATE_INITIALIZING implies after.frames.meta(f) == before.frames.meta(f)
        && after.frames.frame(f) == before.frames.frame(f) by {
        assert(target != Some(f));
        assert(mid.frames.meta(f) == before.frames.meta(f));
        assert(mid.frames.frame(f) == before.frames.frame(f));
    }
    assert forall|s: int|
        0 <= s < before.active_leases.len() && (#[trigger] before.active_leases[s] matches Some(l)
            && l.internal) implies after.active_leases[s] == before.active_leases[s] by {
        if let Some(sl) = released_lease_slot {
            assert(sl as int != s);
        }
        assert(released_lease_slot != Some(s as u32));
        assert(mid.active_leases[s] == before.active_leases[s]);
    }
}

/// Extend `core_transition(before, mid, ...)` across a `mid -> after` step that
/// submits one event `ev` from `source` on `tl`, writing only to `write_loc`
/// (which is disjoint from every `STATE_INITIALIZING` frame in `before`), without
/// modifying `frames`, `active_leases`, or `versions`.
pub proof fn lemma_core_transition_step_single_event(
    before: &CacheCore,
    mid: &CacheCore,
    after: &CacheCore,
    tl: u8,
    source: EventSource,
    ev: TransferEvent,
    write_loc: SlotLoc,
)
    requires
        core_transition(before, mid, Some(tl), Some(source), FrameTouchMode::PreserveAllInit),
        after.node_id == mid.node_id,
        after.instance_id() == mid.instance_id(),
        after.tiers == mid.tiers,
        after.frames == mid.frames,
        after.active_leases == mid.active_leases,
        after.next_version_seq == mid.next_version_seq,
        after.version_seq_limit == mid.version_seq_limit,
        after.timelines.count == mid.timelines.count,
        after.timelines.kinds == mid.timelines.kinds,
        after.timelines.confirmed_seq == mid.timelines.confirmed_seq,
        forall|t: u8|
            t < mid.timelines.count && t != tl ==> {
                &&& (#[trigger] after.timelines.next_seq[t as int])
                    == mid.timelines.next_seq[t as int]
                &&& forall|s: u64|
                    1 <= s < mid.timelines.next_seq[t as int]
                        ==> #[trigger] after.timelines.events@[(t, s)] == mid.timelines.events@[(
                        t,
                        s,
                    )]
            },
        mid.ledger@.submitted.map().submap_of(after.ledger@.submitted.map()),
        revoked_submap(mid.ledger@.revoked, after.ledger@.revoked),
        mid.ledger@.knows.set().subset_of(after.ledger@.knows.set()),
        mid.ledger@.succeeded.dom().subset_of(after.ledger@.succeeded.dom()),
        after.ledger@.submitted.dom() == mid.ledger@.submitted.dom().insert(ev),
        ev.source == source,
        after.ledger@.submitted.map()[ev].access.writes == ISet::empty().insert(write_loc),
        forall|f: FrameId|
            f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                == STATE_INITIALIZING ==> write_loc != before.frames.frame(f).slot.loc,
    ensures
        core_transition(before, after, Some(tl), Some(source), FrameTouchMode::PreserveAllInit),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    vstd::imap_lib::lemma_submap_of_trans(
        before.ledger@.submitted.map(),
        mid.ledger@.submitted.map(),
        after.ledger@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        before.ledger@.revoked.map(),
        mid.ledger@.revoked.map(),
        after.ledger@.revoked.map(),
    );
    assert forall|t: u8, s: u64|
        t < before.timelines.count && Some(tl) != Some(t) && 1 <= s
            < before.timelines.next_seq[t as int] implies #[trigger] after.timelines.events@[(t, s)]
        == before.timelines.events@[(t, s)] by {
        assert(mid.timelines.next_seq[t as int] == before.timelines.next_seq[t as int]);
        assert(after.timelines.next_seq[t as int] == mid.timelines.next_seq[t as int]);
    }
    assert forall|e: TransferEvent| #[trigger]
        after.ledger@.submitted.dom().contains(e) && !before.ledger@.submitted.dom().contains(
            e,
        ) implies Some(source) == Some(e.source) && forall|f: FrameId|
        f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
            == STATE_INITIALIZING ==> !after.ledger@.submitted.map()[e].access.writes.contains(
            before.frames.frame(f).slot.loc,
        ) by {
        if mid.ledger@.submitted.dom().contains(e) {
            assert(after.ledger@.submitted.map()[e] == mid.ledger@.submitted.map()[e]);
        } else {
            assert(e == ev);
        }
    }
}

/// Prove that a slot belonging to a `STATE_EMPTY` (or non-initializing) frame
/// `dst_frame` is disjoint from every `STATE_INITIALIZING` frame's slot.
pub proof fn lemma_non_init_frame_disjoint_from_init(core: &CacheCore, dst_frame: FrameId)
    requires
        core.well_formed(),
        dst_frame.0 < core.frames.metas.len(),
        core.frames.meta(dst_frame).state != STATE_INITIALIZING,
    ensures
        forall|f: FrameId|
            f.0 < core.frames.metas.len() && (#[trigger] core.frames.meta(f)).state
                == STATE_INITIALIZING ==> core.frames.frame(dst_frame).slot.loc
                != core.frames.frame(f).slot.loc,
{
    hide(CacheCore::well_formed);
    lemma_core_frames_separated(core);
    assert forall|f: FrameId|
        f.0 < core.frames.metas.len() && (#[trigger] core.frames.meta(f)).state
            == STATE_INITIALIZING implies core.frames.frame(dst_frame).slot.loc
        != core.frames.frame(f).slot.loc by {
        assert(dst_frame != f);
        if dst_frame.0 < f.0 {
            assert(core.frames.ghosts@[dst_frame.0 as nat].slot.loc
                != core.frames.ghosts@[f.0 as nat].slot.loc);
        } else {
            assert(core.frames.ghosts@[f.0 as nat].slot.loc
                != core.frames.ghosts@[dst_frame.0 as nat].slot.loc);
        }
    }
}

/// Compact one-call step helper for CUDA `enqueue_*` into a frame that was not
/// `STATE_INITIALIZING` in `before`, isolating all `ISet`/`IMap` extensional
/// equalities from the caller's SMT context.
pub proof fn lemma_core_transition_after_single_enqueue(
    before: &CacheCore,
    mid: &CacheCore,
    after: &CacheCore,
    tl: u8,
    source: EventSource,
    dst_frame: FrameId,
    dst_slot: SlotRef,
    ev: TransferEvent,
    seq: u64,
)
    requires
        before.well_formed(),
        core_transition(before, mid, Some(tl), Some(source), FrameTouchMode::PreserveAllInit),
        dst_frame.0 < before.frames.metas.len(),
        before.frames.meta(dst_frame).state != STATE_INITIALIZING,
        before.frames.frame(dst_frame).slot == dst_slot,
        after.node_id == mid.node_id,
        after.instance_id() == mid.instance_id(),
        after.tiers == mid.tiers,
        after.frames == mid.frames,
        after.next_version_seq == mid.next_version_seq,
        after.version_seq_limit == mid.version_seq_limit,
        after.active_leases == mid.active_leases,
        after.timelines.count == mid.timelines.count,
        after.timelines.kinds == mid.timelines.kinds,
        after.timelines.confirmed_seq == mid.timelines.confirmed_seq,
        after.timelines.next_seq@ == mid.timelines.next_seq@.update(tl as int, (seq + 1) as u64),
        after.timelines.events@ == mid.timelines.events@.insert((tl, seq), ev),
        mid.ledger@.submitted.map().submap_of(after.ledger@.submitted.map()),
        after.ledger@.submitted.map() == mid.ledger@.submitted.map().insert(
            ev,
            after.ledger@.submitted.map()[ev],
        ),
        after.ledger@.knows == mid.ledger@.knows,
        after.ledger@.succeeded == mid.ledger@.succeeded,
        after.ledger@.revoked == mid.ledger@.revoked,
        ev.source == source,
        after.ledger@.submitted.map()[ev].access.writes == ISet::empty().insert(dst_slot.loc),
    ensures
        core_transition(before, after, Some(tl), Some(source), FrameTouchMode::PreserveAllInit),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    lemma_non_init_frame_disjoint_from_init(before, dst_frame);
    assert(after.ledger@.submitted.dom() =~= mid.ledger@.submitted.dom().insert(ev));
    assert(after.ledger@.submitted.map()[ev].access.writes =~= ISet::empty().insert(dst_slot.loc));
    lemma_core_transition_step_single_event(before, mid, after, tl, source, ev, dst_slot.loc);
}

/// Opaque step summary describing how an RDMA / `PeerSession` operation mutated
/// `CacheCore`, optionally retiring a single initializing frame `retired_init`
/// (used by `finish_pull_read` / `fetch_complete_pull`).
#[verifier::opaque]
pub open spec fn peer_core_step_except(
    before: &CacheCore,
    after: &CacheCore,
    rdma_tl: u8,
    retired_init: Option<FrameId>,
) -> bool {
    &&& after.node_id == before.node_id
    &&& after.instance_id() == before.instance_id()
    &&& after.tiers == before.tiers
    &&& after.frames.same_layout(&before.frames)
    &&& after.frames.preserves_initializing(&before.frames, retired_init)
    &&& after.next_version_seq >= before.next_version_seq
    &&& after.version_seq_limit == before.version_seq_limit
    &&& before.frames.versions@.submap_of(after.frames.versions@)
    &&& after.active_leases == before.active_leases
    &&& after.timelines.count == before.timelines.count
    &&& after.timelines.kinds == before.timelines.kinds
    &&& forall|t: u8|
        t < before.timelines.count && t != rdma_tl ==> {
            &&& (#[trigger] after.timelines.next_seq[t as int])
                == before.timelines.next_seq[t as int]
            &&& after.timelines.confirmed_seq[t as int] == before.timelines.confirmed_seq[t as int]
        }
    &&& forall|t: u8, s: u64|
        t < before.timelines.count && t != rdma_tl && 1 <= s < before.timelines.next_seq[t as int]
            ==> #[trigger] after.timelines.events@[(t, s)] == before.timelines.events@[(t, s)]
    &&& before.ledger@.submitted.map().submap_of(after.ledger@.submitted.map())
    &&& revoked_submap(before.ledger@.revoked, after.ledger@.revoked)
    &&& before.ledger@.knows.set().subset_of(after.ledger@.knows.set())
    &&& before.ledger@.succeeded.dom().subset_of(after.ledger@.succeeded.dom())
}

/// Opaque step summary describing how an RDMA / `PeerSession` operation mutated
/// `CacheCore`, sufficient to preserve sibling `HostBackend`, `HostContext`,
/// `NvmeBackend` (including arbitrary in-flight `io_uring` operations), and
/// idle `CudaBackend` (`active_gpu_puts@.len() == 0`) invariants.
pub open spec fn peer_core_step(before: &CacheCore, after: &CacheCore, rdma_tl: u8) -> bool {
    peer_core_step_except(before, after, rdma_tl, None)
}

pub proof fn lemma_peer_core_step_except_refl(
    core: &CacheCore,
    rdma_tl: u8,
    retired_init: Option<FrameId>,
)
    ensures
        peer_core_step_except(core, core, rdma_tl, retired_init),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
}

pub proof fn lemma_peer_core_step_refl(core: &CacheCore, rdma_tl: u8)
    ensures
        peer_core_step(core, core, rdma_tl),
{
    lemma_peer_core_step_except_refl(core, rdma_tl, None);
}

pub proof fn lemma_peer_core_step_trans(a: &CacheCore, b: &CacheCore, c: &CacheCore, rdma_tl: u8)
    requires
        peer_core_step(a, b, rdma_tl),
        peer_core_step(b, c, rdma_tl),
    ensures
        peer_core_step(a, c, rdma_tl),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    vstd::map_lib::lemma_submap_of_trans(
        a.frames.versions@,
        b.frames.versions@,
        c.frames.versions@,
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.submitted.map(),
        b.ledger@.submitted.map(),
        c.ledger@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.revoked.map(),
        b.ledger@.revoked.map(),
        c.ledger@.revoked.map(),
    );
    assert forall|f: FrameId| f.0 < c.frames.metas.len() implies {
        &&& (#[trigger] c.frames.frame(f)).slot == a.frames.frame(f).slot
        &&& c.frames.frame(f).tier == a.frames.frame(f).tier
    } by {
        assert(b.frames.frame(f).slot == a.frames.frame(f).slot);
        assert(b.frames.frame(f).tier == a.frames.frame(f).tier);
    }
    assert forall|f: FrameId|
        f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state
            == STATE_INITIALIZING implies c.frames.meta(f) == a.frames.meta(f) && c.frames.frame(f)
        == a.frames.frame(f) by {
        assert(b.frames.meta(f) == a.frames.meta(f));
        assert(b.frames.frame(f) == a.frames.frame(f));
    }
    assert forall|t: u8| t < a.timelines.count && t != rdma_tl implies {
        &&& (#[trigger] c.timelines.next_seq[t as int]) == a.timelines.next_seq[t as int]
        &&& c.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]
    } by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]);
    }
    assert forall|t: u8, s: u64|
        t < a.timelines.count && t != rdma_tl && 1 <= s
            < a.timelines.next_seq[t as int] implies #[trigger] c.timelines.events@[(t, s)]
        == a.timelines.events@[(t, s)] by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.events@[(t, s)] == a.timelines.events@[(t, s)]);
    }
}

/// Compose a `peer_core_step(a, b, rdma_tl)` with a completion step
/// `peer_core_step_except(b, c, rdma_tl, Some(retired))` when `retired` was not
/// `STATE_INITIALIZING` in `a` (e.g., freshly reserved during `a -> b`).
pub proof fn lemma_peer_core_step_trans_retire_new_init(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    rdma_tl: u8,
    retired: FrameId,
)
    requires
        peer_core_step(a, b, rdma_tl),
        peer_core_step_except(b, c, rdma_tl, Some(retired)),
        retired.0 < a.frames.metas.len(),
        a.frames.meta(retired).state != STATE_INITIALIZING,
    ensures
        peer_core_step(a, c, rdma_tl),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    vstd::map_lib::lemma_submap_of_trans(
        a.frames.versions@,
        b.frames.versions@,
        c.frames.versions@,
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.submitted.map(),
        b.ledger@.submitted.map(),
        c.ledger@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.revoked.map(),
        b.ledger@.revoked.map(),
        c.ledger@.revoked.map(),
    );
    assert forall|f: FrameId| f.0 < c.frames.metas.len() implies {
        &&& (#[trigger] c.frames.frame(f)).slot == a.frames.frame(f).slot
        &&& c.frames.frame(f).tier == a.frames.frame(f).tier
    } by {
        assert(b.frames.frame(f).slot == a.frames.frame(f).slot);
        assert(b.frames.frame(f).tier == a.frames.frame(f).tier);
    }
    assert forall|f: FrameId|
        f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state
            == STATE_INITIALIZING implies c.frames.meta(f) == a.frames.meta(f) && c.frames.frame(f)
        == a.frames.frame(f) by {
        assert(f != retired);
        assert(b.frames.meta(f) == a.frames.meta(f));
        assert(b.frames.frame(f) == a.frames.frame(f));
    }
    assert forall|t: u8| t < a.timelines.count && t != rdma_tl implies {
        &&& (#[trigger] c.timelines.next_seq[t as int]) == a.timelines.next_seq[t as int]
        &&& c.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]
    } by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]);
    }
    assert forall|t: u8, s: u64|
        t < a.timelines.count && t != rdma_tl && 1 <= s
            < a.timelines.next_seq[t as int] implies #[trigger] c.timelines.events@[(t, s)]
        == a.timelines.events@[(t, s)] by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.events@[(t, s)] == a.timelines.events@[(t, s)]);
    }
}

/// Compose `peer_core_step(a, b, rdma_tl)` followed by
/// `peer_core_step_except(b, c, rdma_tl, retired)` (where `retired` may be
/// `STATE_INITIALIZING` in `a`, e.g., `fetch_await_pull` followed by `finish_pull_read`).
pub proof fn lemma_peer_core_step_then_except(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    rdma_tl: u8,
    retired: Option<FrameId>,
)
    requires
        peer_core_step(a, b, rdma_tl),
        peer_core_step_except(b, c, rdma_tl, retired),
    ensures
        peer_core_step_except(a, c, rdma_tl, retired),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    vstd::map_lib::lemma_submap_of_trans(
        a.frames.versions@,
        b.frames.versions@,
        c.frames.versions@,
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.submitted.map(),
        b.ledger@.submitted.map(),
        c.ledger@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.revoked.map(),
        b.ledger@.revoked.map(),
        c.ledger@.revoked.map(),
    );
    assert forall|f: FrameId| f.0 < c.frames.metas.len() implies {
        &&& (#[trigger] c.frames.frame(f)).slot == a.frames.frame(f).slot
        &&& c.frames.frame(f).tier == a.frames.frame(f).tier
    } by {
        assert(b.frames.frame(f).slot == a.frames.frame(f).slot);
        assert(b.frames.frame(f).tier == a.frames.frame(f).tier);
    }
    assert forall|f: FrameId|
        f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state == STATE_INITIALIZING
            && retired != Some(f) implies c.frames.meta(f) == a.frames.meta(f) && c.frames.frame(f)
        == a.frames.frame(f) by {
        assert(b.frames.meta(f) == a.frames.meta(f));
        assert(b.frames.frame(f) == a.frames.frame(f));
    }
    assert forall|t: u8| t < a.timelines.count && t != rdma_tl implies {
        &&& (#[trigger] c.timelines.next_seq[t as int]) == a.timelines.next_seq[t as int]
        &&& c.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]
    } by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]);
    }
    assert forall|t: u8, s: u64|
        t < a.timelines.count && t != rdma_tl && 1 <= s
            < a.timelines.next_seq[t as int] implies #[trigger] c.timelines.events@[(t, s)]
        == a.timelines.events@[(t, s)] by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.events@[(t, s)] == a.timelines.events@[(t, s)]);
    }
}

/// Compose `peer_core_step_except(a, b, rdma_tl, retired)` followed by
/// `peer_core_step(b, c, rdma_tl)` (e.g., `finish_pull_read` followed by
/// `try_register_version_binding`).
pub proof fn lemma_peer_core_step_except_then(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    rdma_tl: u8,
    retired: Option<FrameId>,
)
    requires
        peer_core_step_except(a, b, rdma_tl, retired),
        peer_core_step(b, c, rdma_tl),
    ensures
        peer_core_step_except(a, c, rdma_tl, retired),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(peer_core_step_except);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    vstd::map_lib::lemma_submap_of_trans(
        a.frames.versions@,
        b.frames.versions@,
        c.frames.versions@,
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.submitted.map(),
        b.ledger@.submitted.map(),
        c.ledger@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger@.revoked.map(),
        b.ledger@.revoked.map(),
        c.ledger@.revoked.map(),
    );
    assert forall|f: FrameId| f.0 < c.frames.metas.len() implies {
        &&& (#[trigger] c.frames.frame(f)).slot == a.frames.frame(f).slot
        &&& c.frames.frame(f).tier == a.frames.frame(f).tier
    } by {
        assert(b.frames.frame(f).slot == a.frames.frame(f).slot);
        assert(b.frames.frame(f).tier == a.frames.frame(f).tier);
    }
    assert forall|f: FrameId|
        f.0 < a.frames.metas.len() && (#[trigger] a.frames.meta(f)).state == STATE_INITIALIZING
            && retired != Some(f) implies c.frames.meta(f) == a.frames.meta(f) && c.frames.frame(f)
        == a.frames.frame(f) by {
        assert(b.frames.meta(f) == a.frames.meta(f));
        assert(b.frames.frame(f) == a.frames.frame(f));
    }
    assert forall|t: u8| t < a.timelines.count && t != rdma_tl implies {
        &&& (#[trigger] c.timelines.next_seq[t as int]) == a.timelines.next_seq[t as int]
        &&& c.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]
    } by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.confirmed_seq[t as int] == a.timelines.confirmed_seq[t as int]);
    }
    assert forall|t: u8, s: u64|
        t < a.timelines.count && t != rdma_tl && 1 <= s
            < a.timelines.next_seq[t as int] implies #[trigger] c.timelines.events@[(t, s)]
        == a.timelines.events@[(t, s)] by {
        assert(b.timelines.next_seq[t as int] == a.timelines.next_seq[t as int]);
        assert(b.timelines.events@[(t, s)] == a.timelines.events@[(t, s)]);
    }
}

} // verus!
