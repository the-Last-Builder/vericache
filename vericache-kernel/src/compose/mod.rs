use crate::frame::{FrameTable, STATE_INITIALIZING};
use crate::kernel::CacheCore;
use crate::provenance::revoked_submap;
use crate::spec::{FrameId, TimelineId, VersionId, MAX_TIMELINES};
use crate::spi::{lemma_core_frames_separated, lemma_core_wf_facts};
use transfer_core::domain::{EventSource, SlotLoc, SlotRef, TransferEvent, TransferShare};
use vstd::prelude::*;

mod caller;
pub use caller::*;
mod preserve;
pub use preserve::*;

verus! {

broadcast use vstd::iset::group_iset_lemmas;

/// Newly introduced version IDs may be fresh local IDs, explicitly named IDs, or both.
pub ghost struct VersionFootprint {
    pub fresh_local: bool,
    /// `None` permits arbitrary IDs. An empty set names no additional IDs.
    pub specific: Option<ISet<VersionId>>,
}

impl VersionFootprint {
    pub open spec fn unchanged() -> Self {
        Self::only(ISet::empty())
    }

    pub open spec fn only(versions: ISet<VersionId>) -> Self {
        Self { fresh_local: false, specific: Some(versions) }
    }

    pub open spec fn fresh_local_only() -> Self {
        Self { fresh_local: true, specific: Some(ISet::empty()) }
    }

    pub open spec fn any() -> Self {
        Self { fresh_local: false, specific: None }
    }

    pub open spec fn allows(self, before: &CacheCore, v: VersionId) -> bool {
        self.specific is None || self.specific.unwrap().contains(v) || (self.fresh_local
            && v.origin_node == before.node_id_spec() && before.next_version_seq_spec() <= v.seq
            && v.seq < before.version_seq_limit_spec())
    }

    pub open spec fn union(self, next: Self) -> Self {
        Self {
            fresh_local: self.fresh_local || next.fresh_local,
            specific: union_limits(self.specific, next.specific),
        }
    }
}

pub open spec fn optional_set<T>(value: Option<T>) -> ISet<T> {
    match value {
        None => ISet::empty(),
        Some(x) => ISet::empty().insert(x),
    }
}

pub open spec fn union_limits<T>(a: Option<ISet<T>>, b: Option<ISet<T>>) -> Option<ISet<T>> {
    if a is Some && b is Some {
        Some(a.unwrap().union(b.unwrap()))
    } else {
        None
    }
}

/// The exceptions to the state preserved by a core transition.
pub ghost struct CoreFootprint {
    /// Existing timelines whose cursors and events may change.
    pub touched_timelines: ISet<TimelineId>,
    /// Sources for new events. `None` allows unrestricted imports.
    pub allowed_sources: Option<ISet<EventSource>>,
    /// Existing initializing frames whose metadata or contents may change.
    pub touched_init_frames: ISet<FrameId>,
    /// Existing initializing slots that new events may write when sources are restricted.
    pub written_init_frames: ISet<FrameId>,
    /// Existing internal leases that may change or be released.
    pub touched_internal_leases: ISet<u32>,
    /// Existing fenced pins that may be removed.
    pub touched_fenced_shares: ISet<(FrameId, TransferShare)>,
    pub versions: VersionFootprint,
}

impl CoreFootprint {
    pub open spec fn preserve_all(
        timeline: Option<TimelineId>,
        source: Option<EventSource>,
    ) -> Self {
        Self {
            touched_timelines: optional_set(timeline),
            allowed_sources: Some(optional_set(source)),
            touched_init_frames: ISet::empty(),
            written_init_frames: ISet::empty(),
            touched_internal_leases: ISet::empty(),
            touched_fenced_shares: ISet::empty(),
            versions: VersionFootprint::fresh_local_only(),
        }
    }

    pub open spec fn finish_write(
        timeline: TimelineId,
        source: EventSource,
        frame: FrameId,
        version: VersionId,
    ) -> Self {
        Self {
            touched_init_frames: optional_set(Some(frame)),
            written_init_frames: optional_set(Some(frame)),
            versions: VersionFootprint::only(optional_set(Some(version))),
            ..Self::preserve_all(Some(timeline), Some(source))
        }
    }

    pub open spec fn completion(
        timeline: TimelineId,
        frames: ISet<FrameId>,
        released_leases: ISet<u32>,
        versions: VersionFootprint,
    ) -> Self {
        Self {
            touched_init_frames: frames,
            touched_internal_leases: released_leases,
            versions,
            ..Self::preserve_all(Some(timeline), None)
        }
    }

    pub open spec fn excludes_source(self, source: EventSource) -> bool {
        self.allowed_sources is Some && !self.allowed_sources.unwrap().contains(source)
    }

    /// The second step can write an initializing slot retired by the first.
    /// Include those frames in the composed write allowance when needed.
    pub open spec fn union(self, next: Self) -> Self {
        let written = self.written_init_frames.union(next.written_init_frames);
        Self {
            touched_timelines: self.touched_timelines.union(next.touched_timelines),
            allowed_sources: union_limits(self.allowed_sources, next.allowed_sources),
            touched_init_frames: self.touched_init_frames.union(next.touched_init_frames),
            written_init_frames: if next.allowed_sources == Some(ISet::empty()) {
                written
            } else {
                written.union(self.touched_init_frames)
            },
            touched_internal_leases: self.touched_internal_leases.union(
                next.touched_internal_leases,
            ),
            touched_fenced_shares: self.touched_fenced_shares.union(next.touched_fenced_shares),
            versions: self.versions.union(next.versions),
        }
    }
}

pub open spec fn fenced_shares_preserved(
    before: &CacheCore,
    after: &CacheCore,
    exceptions: ISet<(FrameId, TransferShare)>,
) -> bool {
    forall|f: FrameId, s: TransferShare|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().frame(
            f,
        ).has_fenced_share(s)) && !exceptions.contains((f, s)) ==> after.frames_spec().frame(
            f,
        ).has_fenced_share(s)
}

/// State and ledger guarantees for one core transition.
#[verifier::opaque]
pub open spec fn core_transition(before: &CacheCore, after: &CacheCore, fp: CoreFootprint) -> bool {
    &&& after.node_id_spec() == before.node_id_spec()
    &&& after.instance_id() == before.instance_id()
    &&& after.tiers_spec() == before.tiers_spec()
    &&& after.frames_spec().same_layout(&before.frames_spec())
    &&& fenced_shares_preserved(before, after, fp.touched_fenced_shares)
    &&& after.next_version_seq_spec() >= before.next_version_seq_spec()
    &&& after.version_seq_limit_spec() == before.version_seq_limit_spec()
    &&& before.frames_spec().versions_spec()@.submap_of(after.frames_spec().versions_spec()@)
    &&& forall|v: VersionId| #[trigger]
        after.frames_spec().versions_spec()@.contains_key(v)
            && !before.frames_spec().versions_spec()@.contains_key(v) ==> fp.versions.allows(
            before,
            v,
        )
    &&& after.timelines_spec().count_spec() >= before.timelines_spec().count_spec()
    &&& (before.timelines_spec().count_spec() <= MAX_TIMELINES
        ==> after.timelines_spec().count_spec() <= MAX_TIMELINES)
    &&& forall|t: TimelineId|
        t < before.timelines_spec().count_spec() ==> (
        #[trigger] after.timelines_spec().kinds_spec()[t as int])
            == before.timelines_spec().kinds_spec()[t as int]
    &&& forall|t: TimelineId|
        t < before.timelines_spec().count_spec() && !fp.touched_timelines.contains(t) ==> {
            &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
                == before.timelines_spec().next_seq_spec()[t as int]
            &&& after.timelines_spec().confirmed_seq_spec()[t as int]
                == before.timelines_spec().confirmed_seq_spec()[t as int]
        }
    &&& forall|t: TimelineId, s: u64|
        t < before.timelines_spec().count_spec() && !fp.touched_timelines.contains(t) && 1 <= s
            < before.timelines_spec().next_seq_spec()[t as int]
            ==> #[trigger] after.timelines_spec().events_spec()@[(t, s)]
            == before.timelines_spec().events_spec()@[(t, s)]
    &&& before.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map())
    &&& revoked_submap(before.ledger_spec()@.revoked, after.ledger_spec()@.revoked)
    &&& before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set())
    &&& before.ledger_spec()@.succeeded.dom().subset_of(after.ledger_spec()@.succeeded.dom())
    &&& (fp.allowed_sources matches Some(sources) ==> forall|e: TransferEvent| #[trigger]
        after.ledger_spec()@.submitted.dom().contains(e)
            && !before.ledger_spec()@.submitted.dom().contains(e) ==> {
            &&& sources.contains(e.source)
            &&& forall|f: FrameId|
                f.0 < before.frames_spec().metas_spec().len() && (
                #[trigger] before.frames_spec().meta(f)).state == STATE_INITIALIZING
                    && !fp.written_init_frames.contains(f)
                    ==> !after.ledger_spec()@.submitted.map()[e].access.writes.contains(
                    before.frames_spec().frame(f).slot.loc,
                )
        })
    &&& forall|f: FrameId|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().meta(
            f,
        )).state == STATE_INITIALIZING && !fp.touched_init_frames.contains(f) ==> {
            &&& after.frames_spec().meta(f) == before.frames_spec().meta(f)
            &&& after.frames_spec().ovf(f) == before.frames_spec().ovf(f)
            &&& after.frames_spec().frame(f) == before.frames_spec().frame(f)
        }
    &&& after.active_leases_spec().len() == before.active_leases_spec().len()
    &&& forall|s: int|
        0 <= s < before.active_leases_spec().len() && (
        #[trigger] before.active_leases_spec()[s] matches Some(l) && l.internal)
            && !fp.touched_internal_leases.contains(s as u32) ==> after.active_leases_spec()[s]
            == before.active_leases_spec()[s]
}

pub proof fn lemma_core_transition_refl(core: &CacheCore, fp: CoreFootprint)
    ensures
        core_transition(core, core, fp),
{
    broadcast use {vstd::map::group_map_lemmas, vstd::imap::group_imap_lemmas};

    reveal(core_transition);
    reveal(FrameTable::same_layout);
}

pub proof fn lemma_core_transition_trans(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    ab: CoreFootprint,
    bc: CoreFootprint,
)
    requires
        core_transition(a, b, ab),
        core_transition(b, c, bc),
    ensures
        core_transition(a, c, ab.union(bc)),
{
    broadcast use {vstd::map::group_map_lemmas, vstd::imap::group_imap_lemmas};

    reveal(core_transition);
    let ac = ab.union(bc);
    vstd::map_lib::lemma_submap_of_trans(
        a.frames_spec().versions_spec()@,
        b.frames_spec().versions_spec()@,
        c.frames_spec().versions_spec()@,
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger_spec()@.submitted.map(),
        b.ledger_spec()@.submitted.map(),
        c.ledger_spec()@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        a.ledger_spec()@.revoked.map(),
        b.ledger_spec()@.revoked.map(),
        c.ledger_spec()@.revoked.map(),
    );
    assert forall|v: VersionId| #[trigger]
        c.frames_spec().versions_spec()@.contains_key(v)
            && !a.frames_spec().versions_spec()@.contains_key(v) implies ac.versions.allows(
        a,
        v,
    ) by {
        if b.frames_spec().versions_spec()@.contains_key(v) {
            assert(ab.versions.allows(a, v));
        } else {
            assert(bc.versions.allows(b, v));
        }
    }
    assert forall|f: FrameId| f.0 < c.frames_spec().metas_spec().len() implies (
    #[trigger] c.frames_spec().frame(f)).slot == a.frames_spec().frame(f).slot
        && c.frames_spec().frame(f).tier == a.frames_spec().frame(f).tier by {
        assert(b.frames_spec().frame(f).slot == a.frames_spec().frame(f).slot);
    }
    assert forall|t: TimelineId|
        t < a.timelines_spec().count_spec() && !ac.touched_timelines.contains(t) implies {
        &&& (#[trigger] c.timelines_spec().next_seq_spec()[t as int])
            == a.timelines_spec().next_seq_spec()[t as int]
        &&& c.timelines_spec().confirmed_seq_spec()[t as int]
            == a.timelines_spec().confirmed_seq_spec()[t as int]
    } by {
        assert(b.timelines_spec().next_seq_spec()[t as int]
            == a.timelines_spec().next_seq_spec()[t as int]);
        assert(b.timelines_spec().confirmed_seq_spec()[t as int]
            == a.timelines_spec().confirmed_seq_spec()[t as int]);
    }
    assert forall|t: TimelineId, s: u64|
        t < a.timelines_spec().count_spec() && !ac.touched_timelines.contains(t) && 1 <= s
            < a.timelines_spec().next_seq_spec()[t as int] implies #[trigger] c.timelines_spec().events_spec()@[(
    t, s)] == a.timelines_spec().events_spec()@[(t, s)] by {
        assert(b.timelines_spec().next_seq_spec()[t as int]
            == a.timelines_spec().next_seq_spec()[t as int]);
        assert(b.timelines_spec().events_spec()@[(t, s)] == a.timelines_spec().events_spec()@[(
            t,
            s,
        )]);
    }
    assert forall|f: FrameId|
        f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().meta(f)).state
            == STATE_INITIALIZING && !ac.touched_init_frames.contains(f) implies {
        &&& c.frames_spec().meta(f) == a.frames_spec().meta(f)
        &&& c.frames_spec().ovf(f) == a.frames_spec().ovf(f)
        &&& c.frames_spec().frame(f) == a.frames_spec().frame(f)
    } by {
        assert(b.frames_spec().meta(f) == a.frames_spec().meta(f));
    }
    assert forall|f: FrameId, s: TransferShare|
        f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().frame(
            f,
        ).has_fenced_share(s)) && !ac.touched_fenced_shares.contains(
            (f, s),
        ) implies c.frames_spec().frame(f).has_fenced_share(s) by {
        assert(b.frames_spec().frame(f).has_fenced_share(s));
    }
    assert forall|s: int|
        0 <= s < a.active_leases_spec().len() && (#[trigger] a.active_leases_spec()[s] matches Some(
            l,
        ) && l.internal) && !ac.touched_internal_leases.contains(
            s as u32,
        ) implies c.active_leases_spec()[s] == a.active_leases_spec()[s] by {
        assert(b.active_leases_spec()[s] == a.active_leases_spec()[s]);
    }
    if let Some(sources) = ac.allowed_sources {
        assert forall|e: TransferEvent| #[trigger]
            c.ledger_spec()@.submitted.dom().contains(e)
                && !a.ledger_spec()@.submitted.dom().contains(e) implies {
            &&& sources.contains(e.source)
            &&& forall|f: FrameId|
                f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().meta(
                    f,
                )).state == STATE_INITIALIZING && !ac.written_init_frames.contains(f)
                    ==> !c.ledger_spec()@.submitted.map()[e].access.writes.contains(
                    a.frames_spec().frame(f).slot.loc,
                )
        } by {
            if b.ledger_spec()@.submitted.dom().contains(e) {
                assert(ab.allowed_sources.unwrap().contains(e.source));
                assert(c.ledger_spec()@.submitted.map()[e] == b.ledger_spec()@.submitted.map()[e]);
            } else {
                assert(bc.allowed_sources.unwrap().contains(e.source));
                assert(bc.allowed_sources != Some(ISet::empty()));
                assert forall|f: FrameId|
                    f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().meta(
                        f,
                    )).state == STATE_INITIALIZING && !ac.written_init_frames.contains(
                        f,
                    ) implies !c.ledger_spec()@.submitted.map()[e].access.writes.contains(
                    a.frames_spec().frame(f).slot.loc,
                ) by {
                    assert(!ab.touched_init_frames.contains(f));
                    assert(b.frames_spec().meta(f) == a.frames_spec().meta(f));
                    assert(b.frames_spec().frame(f) == a.frames_spec().frame(f));
                }
            }
        }
    }
}

pub proof fn lemma_core_transition_trans_same(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    fp: CoreFootprint,
)
    requires
        core_transition(a, b, fp),
        core_transition(b, c, fp),
        fp.allowed_sources is None || fp.allowed_sources == Some(ISet::empty())
            || fp.touched_init_frames.subset_of(fp.written_init_frames),
    ensures
        core_transition(a, c, fp),
{
    lemma_core_transition_trans(a, b, c, fp, fp);
    let union = fp.union(fp);
    assert(union.touched_timelines =~= fp.touched_timelines);
    assert(union.touched_init_frames =~= fp.touched_init_frames);
    assert(union.touched_internal_leases =~= fp.touched_internal_leases);
    assert(union.touched_fenced_shares =~= fp.touched_fenced_shares);
    if fp.allowed_sources is Some {
        assert(union.allowed_sources.unwrap() =~= fp.allowed_sources.unwrap());
        assert(union.written_init_frames =~= fp.written_init_frames);
    }
    if fp.versions.specific is Some {
        assert(union.versions.specific.unwrap() =~= fp.versions.specific.unwrap());
    }
    if fp.allowed_sources is None {
        reveal(core_transition);
    } else {
        assert(union == fp);
    }
}

/// Frame an observation step that may retire one initializing frame and
/// release the internal lease associated with that operation.
pub proof fn lemma_core_transition_after_retirement(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    retired: Option<FrameId>,
    released: Option<u32>,
)
    requires
        after.node_id_spec() == before.node_id_spec(),
        after.instance_id() == before.instance_id(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec().same_layout(&before.frames_spec()),
        after.frames_spec().preserves_initializing(&before.frames_spec(), retired),
        after.frames_spec().preserves_fenced_leases(&before.frames_spec()),
        retired matches Some(f) ==> fp.touched_init_frames.contains(f),
        released matches Some(s) ==> fp.touched_internal_leases.contains(s),
        after.next_version_seq_spec() == before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        before.frames_spec().versions_spec()@.submap_of(after.frames_spec().versions_spec()@),
        forall|v: VersionId| #[trigger]
            after.frames_spec().versions_spec()@.contains_key(v)
                && !before.frames_spec().versions_spec()@.contains_key(v) ==> fp.versions.allows(
                before,
                v,
            ),
        after.active_leases_spec().len() == before.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before.active_leases_spec().len() && released != Some(s as u32)
                ==> #[trigger] after.active_leases_spec()[s] == before.active_leases_spec()[s],
        after.timelines_spec().count_spec() == before.timelines_spec().count_spec(),
        after.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
        after.timelines_spec().next_seq_spec() == before.timelines_spec().next_seq_spec(),
        after.timelines_spec().events_spec() == before.timelines_spec().events_spec(),
        forall|t: TimelineId|
            t < before.timelines_spec().count_spec() && !fp.touched_timelines.contains(t)
                ==> #[trigger] after.timelines_spec().confirmed_seq_spec()[t as int]
                == before.timelines_spec().confirmed_seq_spec()[t as int],
        after.ledger_spec()@.submitted == before.ledger_spec()@.submitted,
        after.ledger_spec()@.revoked == before.ledger_spec()@.revoked,
        before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
        before.ledger_spec()@.succeeded.dom().subset_of(after.ledger_spec()@.succeeded.dom()),
    ensures
        core_transition(before, after, fp),
{
    broadcast use {vstd::imap::group_imap_lemmas, vstd::map::group_map_lemmas};

    reveal(core_transition);
}

pub proof fn lemma_core_transition_no_event(
    before: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<TimelineId>,
    allowed_source: Option<EventSource>,
)
    requires
        after.node_id_spec() == before.node_id_spec(),
        after.instance_id() == before.instance_id(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec().same_directory(&before.frames_spec()),
        after.frames_spec().preserves_initializing(&before.frames_spec(), None),
        after.frames_spec().preserves_fenced_leases(&before.frames_spec()),
        after.next_version_seq_spec() >= before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        after.timelines_spec() == before.timelines_spec() || after.timelines_spec().only_watermarks(
            &before.timelines_spec(),
        ),
        after.ledger_spec() == before.ledger_spec(),
        after.active_leases_spec().len() == before.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before.active_leases_spec().len() && (
            #[trigger] before.active_leases_spec()[s] matches Some(l) && l.internal)
                ==> after.active_leases_spec()[s] == before.active_leases_spec()[s],
    ensures
        core_transition(
            before,
            after,
            CoreFootprint::preserve_all(touched_timeline, allowed_source),
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
    touched_timeline: Option<TimelineId>,
    source: EventSource,
    ev: TransferEvent,
    write_loc: Option<SlotLoc>,
    new_version: Option<VersionId>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        after.node_id_spec() == before.node_id_spec(),
        after.instance_id() == before.instance_id(),
        after.tiers_spec() == before.tiers_spec(),
        after.frames_spec().same_layout(&before.frames_spec()),
        after.frames_spec().preserves_initializing(&before.frames_spec(), None),
        after.frames_spec().preserves_fenced_leases(&before.frames_spec()),
        after.next_version_seq_spec() >= before.next_version_seq_spec(),
        after.version_seq_limit_spec() == before.version_seq_limit_spec(),
        before.frames_spec().versions_spec()@.submap_of(after.frames_spec().versions_spec()@),
        match new_version {
            None => after.frames_spec().versions_spec()@ == before.frames_spec().versions_spec()@,
            Some(nv) => nv.origin_node == before.node_id_spec() && before.next_version_seq_spec()
                <= nv.seq && nv.seq < before.version_seq_limit_spec()
                && after.frames_spec().versions_spec()@.dom()
                == before.frames_spec().versions_spec()@.dom().insert(nv),
        },
        after.timelines_spec().count_spec() == before.timelines_spec().count_spec(),
        after.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
        forall|t: TimelineId|
            t < before.timelines_spec().count_spec() && touched_timeline != Some(t) ==> {
                &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
                    == before.timelines_spec().next_seq_spec()[t as int]
                &&& after.timelines_spec().confirmed_seq_spec()[t as int]
                    == before.timelines_spec().confirmed_seq_spec()[t as int]
                &&& forall|s: u64|
                    1 <= s < before.timelines_spec().next_seq_spec()[t as int]
                        ==> #[trigger] after.timelines_spec().events_spec()@[(t, s)]
                        == before.timelines_spec().events_spec()@[(t, s)]
            },
        before.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map()),
        revoked_submap(before.ledger_spec()@.revoked, after.ledger_spec()@.revoked),
        before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
        before.ledger_spec()@.succeeded.dom().subset_of(after.ledger_spec()@.succeeded.dom()),
        after.ledger_spec()@.submitted.dom() == before.ledger_spec()@.submitted.dom().insert(ev),
        ev.source == source,
        match write_loc {
            None => after.ledger_spec()@.submitted.map()[ev].access.writes.is_empty(),
            Some(loc) => {
                &&& after.ledger_spec()@.submitted.map()[ev].access.writes == ISet::empty().insert(
                    loc,
                )
                &&& forall|f: FrameId|
                    f.0 < before.frames_spec().metas_spec().len() && (
                    #[trigger] before.frames_spec().meta(f)).state == STATE_INITIALIZING ==> loc
                        != before.frames_spec().frame(f).slot.loc
            },
        },
        after.active_leases_spec().len() == before.active_leases_spec().len(),
        forall|s: int|
            0 <= s < before.active_leases_spec().len() && (
            #[trigger] before.active_leases_spec()[s] matches Some(l) && l.internal)
                ==> after.active_leases_spec()[s] == before.active_leases_spec()[s],
    ensures
        core_transition(before, after, CoreFootprint::preserve_all(touched_timeline, Some(source))),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    assert forall|t: TimelineId, s: u64|
        t < before.timelines_spec().count_spec() && touched_timeline != Some(t) && 1 <= s
            < before.timelines_spec().next_seq_spec()[t as int] implies #[trigger] after.timelines_spec().events_spec()@[(
    t, s)] == before.timelines_spec().events_spec()@[(t, s)] by {
        assert(after.timelines_spec().next_seq_spec()[t as int]
            == before.timelines_spec().next_seq_spec()[t as int]);
    }
    let fp = CoreFootprint::preserve_all(touched_timeline, Some(source));
    assert forall|v: VersionId| #[trigger]
        after.frames_spec().versions_spec()@.contains_key(v)
            && !before.frames_spec().versions_spec()@.contains_key(v) implies fp.versions.allows(
        before,
        v,
    ) by {
        if let Some(nv) = new_version {
            assert(before.frames_spec().versions_spec()@.dom().insert(nv).contains(v));
            assert(v == nv);
        }
    }
    assert forall|t: TimelineId|
        t < before.timelines_spec().count_spec() && !fp.touched_timelines.contains(t) implies {
        &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
            == before.timelines_spec().next_seq_spec()[t as int]
        &&& after.timelines_spec().confirmed_seq_spec()[t as int]
            == before.timelines_spec().confirmed_seq_spec()[t as int]
    } by {
        assert(touched_timeline != Some(t));
    }
    assert forall|e: TransferEvent| #[trigger]
        after.ledger_spec()@.submitted.dom().contains(e)
            && !before.ledger_spec()@.submitted.dom().contains(e) implies optional_set(
        Some(source),
    ).contains(e.source) && forall|f: FrameId|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().meta(
            f,
        )).state == STATE_INITIALIZING
            ==> !after.ledger_spec()@.submitted.map()[e].access.writes.contains(
            before.frames_spec().frame(f).slot.loc,
        ) by {
        assert(e == ev);
    }
}

/// Extend `core_transition(before, mid, ...)` across a `mid -> after` step that
/// emits no ledger events and no new versions, modifying at most `target` (which
/// was not `STATE_INITIALIZING` in `before`) and `released_lease_slot` (which
/// was `None` in `before`).
pub proof fn lemma_core_transition_step_no_event(
    before: &CacheCore,
    mid: &CacheCore,
    after: &CacheCore,
    touched_timeline: Option<TimelineId>,
    allowed_source: Option<EventSource>,
    target: Option<FrameId>,
    released_lease_slot: Option<u32>,
)
    requires
        core_transition(before, mid, CoreFootprint::preserve_all(touched_timeline, allowed_source)),
        before.well_formed(),
        after.node_id_spec() == mid.node_id_spec(),
        after.instance_id() == mid.instance_id(),
        after.tiers_spec() == mid.tiers_spec(),
        after.frames_spec().same_directory(&mid.frames_spec()),
        after.frames_spec().preserves_initializing(&mid.frames_spec(), target),
        after.frames_spec().preserves_fenced_leases(&mid.frames_spec()),
        target matches Some(tf) ==> tf.0 < before.frames_spec().metas_spec().len()
            && before.frames_spec().meta(tf).state != STATE_INITIALIZING,
        after.next_version_seq_spec() >= mid.next_version_seq_spec(),
        after.version_seq_limit_spec() == mid.version_seq_limit_spec(),
        after.timelines_spec() == mid.timelines_spec() || after.timelines_spec().only_watermarks(
            &mid.timelines_spec(),
        ),
        after.ledger_spec() == mid.ledger_spec(),
        after.active_leases_spec().len() == mid.active_leases_spec().len(),
        released_lease_slot matches Some(sl) ==> (sl as usize) < before.active_leases_spec().len()
            && before.active_leases_spec()[sl as int].is_none(),
        forall|s: int|
            0 <= s < mid.active_leases_spec().len() && released_lease_slot != Some(s as u32)
                ==> #[trigger] after.active_leases_spec()[s] == mid.active_leases_spec()[s],
    ensures
        core_transition(
            before,
            after,
            CoreFootprint::preserve_all(touched_timeline, allowed_source),
        ),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    reveal(core_transition);
    reveal(FrameTable::same_directory);
    reveal(FrameTable::same_layout);
    reveal(FrameTable::preserves_initializing);
    assert(after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len());
    assert forall|f: FrameId| f.0 < before.frames_spec().metas_spec().len() implies {
        &&& (#[trigger] after.frames_spec().frame(f)).slot == before.frames_spec().frame(f).slot
        &&& after.frames_spec().frame(f).tier == before.frames_spec().frame(f).tier
    } by {
        assert(mid.frames_spec().frame(f).slot == before.frames_spec().frame(f).slot);
        assert(mid.frames_spec().frame(f).tier == before.frames_spec().frame(f).tier);
    }
    assert(after.frames_spec().same_layout(&before.frames_spec()));
    assert forall|f: FrameId, share: TransferShare|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().frame(
            f,
        ).has_fenced_share(share)) implies after.frames_spec().frame(f).has_fenced_share(share) by {
        assert(mid.frames_spec().frame(f).has_fenced_share(share));
        assert(after.frames_spec().frame(f).fenced_shares == mid.frames_spec().frame(
            f,
        ).fenced_shares);
    }
    assert forall|t: TimelineId|
        t < before.timelines_spec().count_spec() && touched_timeline != Some(t) implies {
        &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
            == before.timelines_spec().next_seq_spec()[t as int]
        &&& after.timelines_spec().confirmed_seq_spec()[t as int]
            == before.timelines_spec().confirmed_seq_spec()[t as int]
        &&& forall|s: u64|
            1 <= s < before.timelines_spec().next_seq_spec()[t as int]
                ==> #[trigger] after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]
    } by {
        assert(mid.timelines_spec().next_seq_spec()[t as int]
            == before.timelines_spec().next_seq_spec()[t as int]);
        assert(mid.timelines_spec().confirmed_seq_spec()[t as int]
            == before.timelines_spec().confirmed_seq_spec()[t as int]);
    }
    assert forall|f: FrameId|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().meta(
            f,
        )).state == STATE_INITIALIZING implies after.frames_spec().meta(f)
        == before.frames_spec().meta(f) && after.frames_spec().ovf(f) == before.frames_spec().ovf(f)
        && after.frames_spec().frame(f) == before.frames_spec().frame(f) by {
        assert(target != Some(f));
        assert(mid.frames_spec().meta(f) == before.frames_spec().meta(f));
        assert(mid.frames_spec().ovf(f) == before.frames_spec().ovf(f));
        assert(mid.frames_spec().frame(f) == before.frames_spec().frame(f));
    }
    assert forall|s: int|
        0 <= s < before.active_leases_spec().len() && (
        #[trigger] before.active_leases_spec()[s] matches Some(l)
            && l.internal) implies after.active_leases_spec()[s]
        == before.active_leases_spec()[s] by {
        if let Some(sl) = released_lease_slot {
            assert(sl as int != s);
        }
        assert(released_lease_slot != Some(s as u32));
        assert(mid.active_leases_spec()[s] == before.active_leases_spec()[s]);
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
    tl: TimelineId,
    source: EventSource,
    ev: TransferEvent,
    write_loc: SlotLoc,
)
    requires
        core_transition(before, mid, CoreFootprint::preserve_all(Some(tl), Some(source))),
        after.node_id_spec() == mid.node_id_spec(),
        after.instance_id() == mid.instance_id(),
        after.tiers_spec() == mid.tiers_spec(),
        after.frames_spec() == mid.frames_spec(),
        after.active_leases_spec() == mid.active_leases_spec(),
        after.next_version_seq_spec() == mid.next_version_seq_spec(),
        after.version_seq_limit_spec() == mid.version_seq_limit_spec(),
        after.timelines_spec().count_spec() == mid.timelines_spec().count_spec(),
        after.timelines_spec().kinds_spec() == mid.timelines_spec().kinds_spec(),
        after.timelines_spec().confirmed_seq_spec() == mid.timelines_spec().confirmed_seq_spec(),
        forall|t: TimelineId|
            t < mid.timelines_spec().count_spec() && t != tl ==> {
                &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
                    == mid.timelines_spec().next_seq_spec()[t as int]
                &&& forall|s: u64|
                    1 <= s < mid.timelines_spec().next_seq_spec()[t as int]
                        ==> #[trigger] after.timelines_spec().events_spec()@[(t, s)]
                        == mid.timelines_spec().events_spec()@[(t, s)]
            },
        mid.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map()),
        revoked_submap(mid.ledger_spec()@.revoked, after.ledger_spec()@.revoked),
        mid.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
        mid.ledger_spec()@.succeeded.dom().subset_of(after.ledger_spec()@.succeeded.dom()),
        after.ledger_spec()@.submitted.dom() == mid.ledger_spec()@.submitted.dom().insert(ev),
        ev.source == source,
        after.ledger_spec()@.submitted.map()[ev].access.writes == ISet::empty().insert(write_loc),
        forall|f: FrameId|
            f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().meta(
                f,
            )).state == STATE_INITIALIZING ==> write_loc != before.frames_spec().frame(f).slot.loc,
    ensures
        core_transition(before, after, CoreFootprint::preserve_all(Some(tl), Some(source))),
{
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    reveal(core_transition);
    vstd::imap_lib::lemma_submap_of_trans(
        before.ledger_spec()@.submitted.map(),
        mid.ledger_spec()@.submitted.map(),
        after.ledger_spec()@.submitted.map(),
    );
    vstd::imap_lib::lemma_submap_of_trans(
        before.ledger_spec()@.revoked.map(),
        mid.ledger_spec()@.revoked.map(),
        after.ledger_spec()@.revoked.map(),
    );
    assert forall|t: TimelineId| t < before.timelines_spec().count_spec() && t != tl implies {
        &&& (#[trigger] after.timelines_spec().next_seq_spec()[t as int])
            == before.timelines_spec().next_seq_spec()[t as int]
        &&& after.timelines_spec().confirmed_seq_spec()[t as int]
            == before.timelines_spec().confirmed_seq_spec()[t as int]
    } by {
        assert(mid.timelines_spec().next_seq_spec()[t as int]
            == before.timelines_spec().next_seq_spec()[t as int]);
    }
    assert forall|t: TimelineId, s: u64|
        t < before.timelines_spec().count_spec() && Some(tl) != Some(t) && 1 <= s
            < before.timelines_spec().next_seq_spec()[t as int] implies #[trigger] after.timelines_spec().events_spec()@[(
    t, s)] == before.timelines_spec().events_spec()@[(t, s)] by {
        assert(mid.timelines_spec().next_seq_spec()[t as int]
            == before.timelines_spec().next_seq_spec()[t as int]);
        assert(after.timelines_spec().next_seq_spec()[t as int]
            == mid.timelines_spec().next_seq_spec()[t as int]);
    }
    assert forall|e: TransferEvent| #[trigger]
        after.ledger_spec()@.submitted.dom().contains(e)
            && !before.ledger_spec()@.submitted.dom().contains(e) implies optional_set(
        Some(source),
    ).contains(e.source) && forall|f: FrameId|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().meta(
            f,
        )).state == STATE_INITIALIZING
            ==> !after.ledger_spec()@.submitted.map()[e].access.writes.contains(
            before.frames_spec().frame(f).slot.loc,
        ) by {
        if mid.ledger_spec()@.submitted.dom().contains(e) {
            assert(after.ledger_spec()@.submitted.map()[e]
                == mid.ledger_spec()@.submitted.map()[e]);
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
        dst_frame.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(dst_frame).state != STATE_INITIALIZING,
    ensures
        forall|f: FrameId|
            f.0 < core.frames_spec().metas_spec().len() && (#[trigger] core.frames_spec().meta(
                f,
            )).state == STATE_INITIALIZING ==> core.frames_spec().frame(dst_frame).slot.loc
                != core.frames_spec().frame(f).slot.loc,
{
    hide(CacheCore::well_formed);
    lemma_core_frames_separated(core);
    assert forall|f: FrameId|
        f.0 < core.frames_spec().metas_spec().len() && (#[trigger] core.frames_spec().meta(f)).state
            == STATE_INITIALIZING implies core.frames_spec().frame(dst_frame).slot.loc
        != core.frames_spec().frame(f).slot.loc by {
        assert(dst_frame != f);
        if dst_frame.0 < f.0 {
            assert(core.frames_spec().ghosts_spec()@[dst_frame.0 as nat].slot.loc
                != core.frames_spec().ghosts_spec()@[f.0 as nat].slot.loc);
        } else {
            assert(core.frames_spec().ghosts_spec()@[f.0 as nat].slot.loc
                != core.frames_spec().ghosts_spec()@[dst_frame.0 as nat].slot.loc);
        }
    }
}

/// Compact one-call step helper for submitting to a frame that was not
/// `STATE_INITIALIZING` in `before`, isolating all `ISet`/`IMap` extensional
/// equalities from the caller's SMT context.
pub proof fn lemma_core_transition_after_single_enqueue(
    before: &CacheCore,
    mid: &CacheCore,
    after: &CacheCore,
    tl: TimelineId,
    source: EventSource,
    dst_frame: FrameId,
    dst_slot: SlotRef,
    ev: TransferEvent,
    seq: u64,
)
    requires
        before.well_formed(),
        core_transition(before, mid, CoreFootprint::preserve_all(Some(tl), Some(source))),
        dst_frame.0 < before.frames_spec().metas_spec().len(),
        before.frames_spec().meta(dst_frame).state != STATE_INITIALIZING,
        before.frames_spec().frame(dst_frame).slot == dst_slot,
        after.node_id_spec() == mid.node_id_spec(),
        after.instance_id() == mid.instance_id(),
        after.tiers_spec() == mid.tiers_spec(),
        after.frames_spec() == mid.frames_spec(),
        after.next_version_seq_spec() == mid.next_version_seq_spec(),
        after.version_seq_limit_spec() == mid.version_seq_limit_spec(),
        after.active_leases_spec() == mid.active_leases_spec(),
        after.timelines_spec().count_spec() == mid.timelines_spec().count_spec(),
        after.timelines_spec().kinds_spec() == mid.timelines_spec().kinds_spec(),
        after.timelines_spec().confirmed_seq_spec() == mid.timelines_spec().confirmed_seq_spec(),
        after.timelines_spec().next_seq_spec()@ == mid.timelines_spec().next_seq_spec()@.update(
            tl as int,
            (seq + 1) as u64,
        ),
        after.timelines_spec().events_spec()@ == mid.timelines_spec().events_spec()@.insert(
            (tl, seq),
            ev,
        ),
        mid.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map()),
        after.ledger_spec()@.submitted.map() == mid.ledger_spec()@.submitted.map().insert(
            ev,
            after.ledger_spec()@.submitted.map()[ev],
        ),
        after.ledger_spec()@.knows == mid.ledger_spec()@.knows,
        after.ledger_spec()@.succeeded == mid.ledger_spec()@.succeeded,
        after.ledger_spec()@.revoked == mid.ledger_spec()@.revoked,
        ev.source == source,
        after.ledger_spec()@.submitted.map()[ev].access.writes == ISet::empty().insert(
            dst_slot.loc,
        ),
    ensures
        core_transition(before, after, CoreFootprint::preserve_all(Some(tl), Some(source))),
{
    hide(CacheCore::well_formed);
    broadcast use {
        vstd::map::group_map_lemmas,
        vstd::imap::group_imap_lemmas,
        vstd::iset::group_iset_lemmas,
    };

    lemma_non_init_frame_disjoint_from_init(before, dst_frame);
    assert(after.ledger_spec()@.submitted.dom() =~= mid.ledger_spec()@.submitted.dom().insert(ev));
    assert(after.ledger_spec()@.submitted.map()[ev].access.writes =~= ISet::empty().insert(
        dst_slot.loc,
    ));
    lemma_core_transition_step_single_event(before, mid, after, tl, source, ev, dst_slot.loc);
}

} // verus!
