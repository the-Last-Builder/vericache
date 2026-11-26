use super::peer::PeerSession;
use super::*;
use transfer::transport::ByteTransport;

verus! {

broadcast use vstd::iset::group_iset_lemmas;

/// Remote operations may merge external events and version IDs.
pub open spec fn peer_step_footprint(
    timeline: TimelineId,
    retired: Option<FrameId>,
) -> CoreFootprint {
    CoreFootprint {
        allowed_sources: None,
        touched_init_frames: optional_set(retired),
        touched_fenced_shares: ISet::full(),
        versions: VersionFootprint::any(),
        ..CoreFootprint::preserve_all(Some(timeline), None)
    }
}

pub open spec fn peer_step_with_pins(
    timeline: TimelineId,
    retired: Option<FrameId>,
    leases: Seq<ActiveRemoteLease>,
) -> CoreFootprint {
    CoreFootprint {
        touched_fenced_shares: ISet::new(
            |pin: (FrameId, TransferShare)| !leases_free_of_share(leases, pin.0, pin.1),
        ),
        ..peer_step_footprint(timeline, retired)
    }
}

impl<'a> RdmaBackend<'a> {
    pub open(crate) spec fn pins_untouched(&self, fp: CoreFootprint) -> bool {
        fp.touched_fenced_shares.is_empty() || forall|i: int|
            0 <= i < self.active_leases@.len() && !(#[trigger] self.active_leases@[i]).is_unpinned()
                ==> !fp.touched_fenced_shares.contains(
                (self.active_leases@[i].frame, self.active_leases@[i].share@),
            )
    }
}

/// In addition to the core footprint, an operation on a peer's existing
/// timeline leaves timeline registration and the local lease table intact.
#[verifier::opaque]
pub open(crate) spec fn peer_core_step_except(
    before: &CacheCore,
    after: &CacheCore,
    timeline: TimelineId,
    retired: Option<FrameId>,
) -> bool {
    &&& core_transition(before, after, peer_step_footprint(timeline, retired))
    &&& after.active_leases_spec() == before.active_leases_spec()
    &&& after.timelines_spec().count_spec() == before.timelines_spec().count_spec()
    &&& after.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec()
}

pub open spec fn peer_core_step(
    before: &CacheCore,
    after: &CacheCore,
    timeline: TimelineId,
) -> bool {
    peer_core_step_except(before, after, timeline, None)
}

pub proof fn lemma_peer_core_step_except_refl(
    core: &CacheCore,
    timeline: TimelineId,
    retired: Option<FrameId>,
)
    ensures
        peer_core_step_except(core, core, timeline, retired),
{
    reveal(peer_core_step_except);
    lemma_core_transition_refl(core, peer_step_footprint(timeline, retired));
}

pub proof fn lemma_peer_core_step_refl(core: &CacheCore, timeline: TimelineId)
    ensures
        peer_core_step(core, core, timeline),
{
    lemma_peer_core_step_except_refl(core, timeline, None);
}

pub proof fn lemma_peer_core_step_trans(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    timeline: TimelineId,
)
    requires
        peer_core_step(a, b, timeline),
        peer_core_step(b, c, timeline),
    ensures
        peer_core_step(a, c, timeline),
{
    reveal(peer_core_step_except);
    lemma_core_transition_trans_same(a, b, c, peer_step_footprint(timeline, None));
}

proof fn lemma_peer_core_step_chain(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    timeline: TimelineId,
    first: Option<FrameId>,
    second: Option<FrameId>,
    retired: Option<FrameId>,
)
    requires
        peer_core_step_except(a, b, timeline, first),
        peer_core_step_except(b, c, timeline, second),
        forall|f: FrameId|
            f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().meta(f)).state
                == STATE_INITIALIZING && (first == Some(f) || second == Some(f)) ==> retired
                == Some(f),
    ensures
        peer_core_step_except(a, c, timeline, retired),
{
    broadcast use vstd::iset::group_iset_lemmas;

    reveal(peer_core_step_except);
    let ab = peer_step_footprint(timeline, first);
    let bc = peer_step_footprint(timeline, second);
    let ac = peer_step_footprint(timeline, retired);
    lemma_core_transition_trans(a, b, c, ab, bc);
    reveal(core_transition);
    assert(ab.union(bc).touched_timelines =~= ac.touched_timelines);
    assert forall|f: FrameId|
        f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().meta(f)).state
            == STATE_INITIALIZING && !ac.touched_init_frames.contains(f) implies {
        &&& c.frames_spec().meta(f) == a.frames_spec().meta(f)
        &&& c.frames_spec().ovf(f) == a.frames_spec().ovf(f)
        &&& c.frames_spec().frame(f) == a.frames_spec().frame(f)
    } by {
        assert(!ab.union(bc).touched_init_frames.contains(f));
    }
}

pub(crate) proof fn lemma_peer_core_step_trans_retire_new_init(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    timeline: TimelineId,
    retired: FrameId,
)
    requires
        peer_core_step(a, b, timeline),
        peer_core_step_except(b, c, timeline, Some(retired)),
        retired.0 < a.frames_spec().metas_spec().len(),
        a.frames_spec().meta(retired).state != STATE_INITIALIZING,
    ensures
        peer_core_step(a, c, timeline),
{
    lemma_peer_core_step_chain(a, b, c, timeline, None, Some(retired), None);
}

pub proof fn lemma_peer_core_step_then_except(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    timeline: TimelineId,
    retired: Option<FrameId>,
)
    requires
        peer_core_step(a, b, timeline),
        peer_core_step_except(b, c, timeline, retired),
    ensures
        peer_core_step_except(a, c, timeline, retired),
{
    lemma_peer_core_step_chain(a, b, c, timeline, None, retired, retired);
}

pub proof fn lemma_peer_core_step_except_then(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    timeline: TimelineId,
    retired: Option<FrameId>,
)
    requires
        peer_core_step_except(a, b, timeline, retired),
        peer_core_step(b, c, timeline),
    ensures
        peer_core_step_except(a, c, timeline, retired),
{
    lemma_peer_core_step_chain(a, b, c, timeline, retired, None, retired);
}

pub(crate) proof fn lemma_peer_step_footprint_with_pins(
    before: &CacheCore,
    after: &CacheCore,
    timeline: TimelineId,
    retired: Option<FrameId>,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
)
    requires
        peer_core_step_except(before, after, timeline, retired),
        peer_foreign_shares_preserved(before, after, before_leases, after_leases),
    ensures
        core_transition(before, after, peer_step_with_pins(timeline, retired, before_leases)),
{
    reveal(peer_core_step_except);
    reveal(core_transition);
    reveal(peer_foreign_shares_preserved);
    broadcast use vstd::iset::group_iset_lemmas;

}

pub(crate) proof fn lemma_core_transition_preserves_rdma<'a>(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    rdma_backend: &RdmaBackend<'a>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        rdma_backend.well_formed(before),
        !fp.touched_timelines.contains(rdma_backend.timeline_id),
        rdma_backend.pins_untouched(fp),
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
    assert(after.timelines_spec().next_seq_spec()[t as int]
        == before.timelines_spec().next_seq_spec()[t as int]);
    assert(after.timelines_spec().confirmed_seq_spec()[t as int]
        == before.timelines_spec().confirmed_seq_spec()[t as int]);
    reveal(rdma_window_wf);
    assert(rdma_backend.bitmap.observations(
        &after.timelines_spec(),
        t,
        after.node(),
        &after.ledger_spec()@,
    )) by {
        assert forall|i: int| #[trigger] rdma_backend.bitmap.bit_set(i) implies {
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
        rdma_backend.in_flight_ops,
        rdma_backend.bitmap,
        &after.timelines_spec(),
        t,
        rdma_data_events(rdma::CqSnapshot::of(rdma_backend.cq)),
    )) by {
        let pending = rdma_data_events(rdma::CqSnapshot::of(rdma_backend.cq));
        assert forall|s: u64| #[trigger]
            pending.contains_key(s) implies after.timelines_spec().confirmed_seq_spec()[t as int]
            < s < after.timelines_spec().next_seq_spec()[t as int] && pending[s]
            == after.timelines_spec().events_spec()@[(t, s)] && (rdma_backend.in_flight_ops[(s
            % BITMAP_CAPACITY) as int] matches Some(op) && op.sequence == s) by {
            assert(after.timelines_spec().events_spec()@[(t, s)]
                == before.timelines_spec().events_spec()@[(t, s)]);
        }
    }
    assert(completed_pulls_wf(
        rdma_backend.completed_pulls,
        &after.timelines_spec(),
        t,
        rdma_backend.cq.observer(),
        &after.ledger_spec()@,
    )) by {
        assert forall|i: int|
            0 <= i < BITMAP_SLOTS implies match #[trigger] rdma_backend.completed_pulls[i] {
            None => true,
            Some((seq, st)) => {
                &&& ((seq as int) % (BITMAP_CAPACITY as int)) == i
                &&& after.timelines_spec().issued(t, seq)
                &&& after.ledger_spec()@.knows.set().contains(
                    (rdma_backend.cq.observer(), after.timelines_spec().events_spec()@[(t, seq)]),
                )
                &&& (st.is_ok() ==> after.ledger_spec()@.succeeded.dom().contains(
                    after.timelines_spec().events_spec()@[(t, seq)],
                ))
            },
        } by {
            if let Some((seq, _st)) = rdma_backend.completed_pulls[i] {
                assert(before.timelines_spec().issued(t, seq));
                assert(after.timelines_spec().events_spec()@[(t, seq)]
                    == before.timelines_spec().events_spec()@[(t, seq)]);
            }
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
        assert(after.frames_spec().frame(f).slot == before.frames_spec().frame(f).slot);
        if !lease.is_unpinned() {
            assert(!fp.touched_fenced_shares.contains((f, lease.share@)));
            assert(after.frames_spec().frame(f).has_fenced_share(lease.share@));
            proofs::lemma_frame_fenced_share_implies_nonzero(after, f, lease.share@);
        }
    }
}

/// Preserve a peer session across a step on other timelines.
pub(crate) proof fn lemma_core_transition_preserves_peer_session<'a, T: ByteTransport>(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    session: &PeerSession<'a, T>,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        session.well_formed(before),
        !fp.touched_timelines.contains(session.rdma.timeline_id),
        session.rdma.pins_untouched(fp),
    ensures
        session.well_formed(after),
{
    hide(CacheCore::well_formed);
    lemma_core_wf_facts(before);
    lemma_core_wf_facts(after);
    lemma_core_transition_preserves_rdma(before, after, fp, &session.rdma);
    reveal(core_transition);
    reveal(FrameTable::same_layout);
    broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

    assert forall|i: int| 0 <= i < session.active_grants@.len() implies {
        let g = #[trigger] session.active_grants@[i];
        &&& g.grant_tag < session.next_grant_tag
        &&& g.frame.0 < after.frames_spec().metas_spec().len()
        &&& session.rdma.lease_index_spec(g.handle) matches Some(idx) && {
            let lease = session.rdma.active_leases@[idx];
            &&& lease.frame == g.frame
            &&& lease.state == RemoteLeaseState::Bound
            &&& session.channel.sent().contains_key(g.grant_tag)
            &&& session.channel.sent()[g.grant_tag] == (transfer::transport::GrantKind::ReadShare {
                slot: after.frames_spec().frame(g.frame).slot,
                share_id: lease.share@,
            })
        }
    } by {
        let g = session.active_grants@[i];
        assert(after.frames_spec().frame(g.frame).slot == before.frames_spec().frame(g.frame).slot);
    }
    assert forall|i: int| 0 <= i < session.pending_outbound_grants@.len() implies (
    #[trigger] session.pending_outbound_grants@[i]).valid(&session.rdma, after) by {
        let p = session.pending_outbound_grants@[i];
        assert(p.valid(&session.rdma, before));
        assert(after.frames_spec().frame(p.frame).slot == before.frames_spec().frame(p.frame).slot
            && after.frames_spec().frame(p.frame).tier == before.frames_spec().frame(p.frame).tier);
        assert(after.frames_spec().tier_frame_size(after.frames_spec().frame(p.frame).tier)
            == before.frames_spec().tier_frame_size(before.frames_spec().frame(p.frame).tier));
        vstd::imap_lib::lemma_submap_of_trans(
            p.bind_sub@.map(),
            before.ledger_spec()@.submitted.map(),
            after.ledger_spec()@.submitted.map(),
        );
        assert forall|e: TransferEvent| #[trigger] p.bind_facts@.contains(e) implies {
            &&& after.ledger_spec()@.submitted.map().contains_key(e)
            &&& after.ledger_spec()@.knows.set().contains((after.node(), e))
        } by {
            assert(before.ledger_spec()@.submitted.map().contains_key(e));
            assert(before.ledger_spec()@.knows.set().contains((before.node(), e)));
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

/// Prove that an active `PeerSession` step (`peer_core_step` +
/// `peer_foreign_shares_preserved`) on `active_tl` preserves a sibling
/// `PeerSession` on a distinct timeline `sibling.rdma.timeline_id != active_tl`,
/// along with pairwise pinned-share disjointness.
pub(crate) proof fn lemma_peer_step_preserves_sibling_session<'a, T: ByteTransport>(
    before: &CacheCore,
    after: &CacheCore,
    active_tl: TimelineId,
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
    let fp = peer_step_with_pins(active_tl, None, active_before_leases);
    lemma_peer_step_footprint_with_pins(
        before,
        after,
        active_tl,
        None,
        active_before_leases,
        active_after_leases,
    );
    broadcast use vstd::iset::group_iset_lemmas;

    lemma_core_transition_preserves_peer_session(before, after, fp, sibling);
    let leases = sibling.rdma.active_leases@;
    reveal(rdma_leases_valid);
    assert forall|i: int|
        0 <= i < leases.len() && !(#[trigger] leases[i]).is_unpinned() implies leases_free_of_share(
        active_after_leases,
        leases[i].frame,
        leases[i].share@,
    ) by {
        let lease = leases[i];
        assert(lease.valid(&sibling.rdma.window_pool, &sibling.rdma.qp, before));
        lemma_foreign_shares_lookup(
            before,
            after,
            active_before_leases,
            active_after_leases,
            lease.frame,
            lease.share@,
        );
    }
    lemma_peer_leases_disjoint_sym(leases, active_after_leases);
}

/// True when no pinned (`!is_unpinned()`) entry in `leases` owns `(f, s)`.
pub open spec fn leases_free_of_share(
    leases: Seq<ActiveRemoteLease>,
    f: FrameId,
    s: TransferShare,
) -> bool {
    forall|k: int|
        0 <= k < leases.len() && !(#[trigger] leases[k]).is_unpinned() ==> !(leases[k].frame == f
            && leases[k].share@ == s)
}

/// Pairwise disjointness of pinned `(frame, share)` ownership between two peer sessions' RDMA lease tables.
pub open spec fn peer_leases_disjoint(
    a: Seq<ActiveRemoteLease>,
    b: Seq<ActiveRemoteLease>,
) -> bool {
    forall|i: int|
        0 <= i < a.len() && !(#[trigger] a[i]).is_unpinned() ==> leases_free_of_share(
            b,
            a[i].frame,
            a[i].share@,
        )
}

pub proof fn lemma_peer_leases_disjoint_sym(a: Seq<ActiveRemoteLease>, b: Seq<ActiveRemoteLease>)
    requires
        peer_leases_disjoint(a, b),
    ensures
        peer_leases_disjoint(b, a),
{
    assert forall|j: int|
        0 <= j < b.len() && !(#[trigger] b[j]).is_unpinned() implies leases_free_of_share(
        a,
        b[j].frame,
        b[j].share@,
    ) by {
        assert forall|i: int| 0 <= i < a.len() && !(#[trigger] a[i]).is_unpinned() implies !(
        a[i].frame == b[j].frame && a[i].share@ == b[j].share@) by {
            assert(leases_free_of_share(b, a[i].frame, a[i].share@));
        }
    }
}

pub(crate) proof fn lemma_foreign_shares_lookup(
    before_core: &CacheCore,
    after_core: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    f: FrameId,
    s: TransferShare,
)
    requires
        peer_foreign_shares_preserved(before_core, after_core, before_leases, after_leases),
        f.0 < before_core.frames_spec().metas_spec().len(),
        before_core.frames_spec().frame(f).has_fenced_share(s),
        leases_free_of_share(before_leases, f, s),
    ensures
        f.0 < after_core.frames_spec().metas_spec().len(),
        after_core.frames_spec().frame(f).has_fenced_share(s),
        leases_free_of_share(after_leases, f, s),
{
    reveal(peer_foreign_shares_preserved);
}

/// Each fenced share outside this peer's pinned leases remains present and
/// outside the peer's pinned leases after the step.
#[verifier::opaque]
pub open(crate) spec fn peer_foreign_shares_preserved(
    before_core: &CacheCore,
    after_core: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
) -> bool {
    forall|f: FrameId, s: TransferShare|
        f.0 < before_core.frames_spec().metas_spec().len() && (
        #[trigger] before_core.frames_spec().frame(f).has_fenced_share(s)) && leases_free_of_share(
            before_leases,
            f,
            s,
        ) ==> {
            &&& f.0 < after_core.frames_spec().metas_spec().len()
            &&& after_core.frames_spec().frame(f).has_fenced_share(s)
            &&& leases_free_of_share(after_leases, f, s)
        }
}

pub proof fn lemma_foreign_shares_refl(core: &CacheCore, leases: Seq<ActiveRemoteLease>)
    ensures
        peer_foreign_shares_preserved(core, core, leases, leases),
{
    reveal(peer_foreign_shares_preserved);
}

pub proof fn lemma_foreign_shares_trans(
    a: &CacheCore,
    b: &CacheCore,
    c: &CacheCore,
    la: Seq<ActiveRemoteLease>,
    lb: Seq<ActiveRemoteLease>,
    lc: Seq<ActiveRemoteLease>,
)
    requires
        peer_foreign_shares_preserved(a, b, la, lb),
        peer_foreign_shares_preserved(b, c, lb, lc),
    ensures
        peer_foreign_shares_preserved(a, c, la, lc),
{
    reveal(peer_foreign_shares_preserved);
    assert forall|f: FrameId, s: TransferShare|
        f.0 < a.frames_spec().metas_spec().len() && (#[trigger] a.frames_spec().frame(
            f,
        ).has_fenced_share(s)) && leases_free_of_share(la, f, s) implies {
        &&& f.0 < c.frames_spec().metas_spec().len()
        &&& c.frames_spec().frame(f).has_fenced_share(s)
        &&& leases_free_of_share(lc, f, s)
    } by {
        assert(b.frames_spec().frame(f).has_fenced_share(s));
        assert(leases_free_of_share(lb, f, s));
    }
}

pub(crate) proof fn lemma_foreign_shares_same_pins(
    before: &CacheCore,
    after: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
)
    requires
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
        after.frames_spec().preserves_fenced_leases(&before.frames_spec()),
        forall|j: int|
            0 <= j < after_leases.len() && !(#[trigger] after_leases[j]).is_unpinned() ==> exists|
                i: int,
            |
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == after_leases[j].frame && before_leases[i].share@
                    == after_leases[j].share@,
    ensures
        peer_foreign_shares_preserved(before, after, before_leases, after_leases),
{
    reveal(peer_foreign_shares_preserved);
    assert forall|f: FrameId, s: TransferShare|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().frame(
            f,
        ).has_fenced_share(s)) && leases_free_of_share(before_leases, f, s) implies {
        &&& f.0 < after.frames_spec().metas_spec().len()
        &&& after.frames_spec().frame(f).has_fenced_share(s)
        &&& leases_free_of_share(after_leases, f, s)
    } by {
        assert(after.frames_spec().frame(f).fenced_shares == before.frames_spec().frame(
            f,
        ).fenced_shares);
        assert forall|k: int|
            0 <= k < after_leases.len() && !(#[trigger] after_leases[k]).is_unpinned() implies !(
        after_leases[k].frame == f && after_leases[k].share@ == s) by {
            let i = choose|i: int|
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == after_leases[k].frame && before_leases[i].share@
                    == after_leases[k].share@;
            assert(!before_leases[i].is_unpinned());
        }
    }
}

pub(crate) proof fn lemma_foreign_shares_after_pin(
    before: &CacheCore,
    after: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    frame: FrameId,
    new_share: TransferShare,
)
    requires
        frame.0 < before.frames_spec().metas_spec().len(),
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
        !before.frames_spec().frame(frame).has_fenced_share(new_share),
        after.frames_spec().same_other_pins(&before.frames_spec(), Some(frame)),
        after.frames_spec().frame(frame).fenced_shares == before.frames_spec().frame(
            frame,
        ).fenced_shares.insert(new_share),
        forall|j: int|
            0 <= j < after_leases.len() && !(#[trigger] after_leases[j]).is_unpinned() ==> (
            after_leases[j].frame == frame && after_leases[j].share@ == new_share) || (exists|
                i: int,
            |
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == after_leases[j].frame && before_leases[i].share@
                    == after_leases[j].share@),
    ensures
        peer_foreign_shares_preserved(before, after, before_leases, after_leases),
{
    broadcast use vstd::set::group_set_lemmas;

    reveal(peer_foreign_shares_preserved);
    assert forall|f: FrameId, s: TransferShare|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().frame(
            f,
        ).has_fenced_share(s)) && leases_free_of_share(before_leases, f, s) implies {
        &&& f.0 < after.frames_spec().metas_spec().len()
        &&& after.frames_spec().frame(f).has_fenced_share(s)
        &&& leases_free_of_share(after_leases, f, s)
    } by {
        if f != frame {
            assert(after.frames_spec().frame(f).fenced_shares == before.frames_spec().frame(
                f,
            ).fenced_shares);
        } else {
            assert(s != new_share);
        }
        assert forall|k: int|
            0 <= k < after_leases.len() && !(#[trigger] after_leases[k]).is_unpinned() implies !(
        after_leases[k].frame == f && after_leases[k].share@ == s) by {
            if !(after_leases[k].frame == frame && after_leases[k].share@ == new_share) {
                let i = choose|i: int|
                    0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                        && before_leases[i].frame == after_leases[k].frame
                        && before_leases[i].share@ == after_leases[k].share@;
                assert(!before_leases[i].is_unpinned());
            }
        }
    }
}

pub(crate) proof fn lemma_foreign_shares_after_unpin(
    before: &CacheCore,
    after: &CacheCore,
    before_leases: Seq<ActiveRemoteLease>,
    after_leases: Seq<ActiveRemoteLease>,
    idx: int,
)
    requires
        0 <= idx < before_leases.len(),
        !before_leases[idx].is_unpinned(),
        before_leases[idx].frame.0 < before.frames_spec().metas_spec().len(),
        after.frames_spec().metas_spec().len() == before.frames_spec().metas_spec().len(),
        after.frames_spec().same_other_pins(&before.frames_spec(), Some(before_leases[idx].frame)),
        after.frames_spec().frame(before_leases[idx].frame).fenced_shares
            == before.frames_spec().frame(before_leases[idx].frame).fenced_shares.remove(
            before_leases[idx].share@,
        ),
        forall|j: int|
            0 <= j < after_leases.len() && !(#[trigger] after_leases[j]).is_unpinned() ==> exists|
                i: int,
            |
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == after_leases[j].frame && before_leases[i].share@
                    == after_leases[j].share@,
    ensures
        peer_foreign_shares_preserved(before, after, before_leases, after_leases),
{
    broadcast use vstd::set::group_set_lemmas;

    reveal(peer_foreign_shares_preserved);
    let uf = before_leases[idx].frame;
    let us = before_leases[idx].share@;
    assert forall|f: FrameId, s: TransferShare|
        f.0 < before.frames_spec().metas_spec().len() && (#[trigger] before.frames_spec().frame(
            f,
        ).has_fenced_share(s)) && leases_free_of_share(before_leases, f, s) implies {
        &&& f.0 < after.frames_spec().metas_spec().len()
        &&& after.frames_spec().frame(f).has_fenced_share(s)
        &&& leases_free_of_share(after_leases, f, s)
    } by {
        assert(!before_leases[idx].is_unpinned());
        assert(!(uf == f && us == s));
        if f != uf {
            assert(after.frames_spec().frame(f).fenced_shares == before.frames_spec().frame(
                f,
            ).fenced_shares);
        } else {
            assert(s != us);
        }
        assert forall|k: int|
            0 <= k < after_leases.len() && !(#[trigger] after_leases[k]).is_unpinned() implies !(
        after_leases[k].frame == f && after_leases[k].share@ == s) by {
            let i = choose|i: int|
                0 <= i < before_leases.len() && !(#[trigger] before_leases[i]).is_unpinned()
                    && before_leases[i].frame == after_leases[k].frame && before_leases[i].share@
                    == after_leases[k].share@;
            assert(!before_leases[i].is_unpinned());
        }
    }
}

} // verus!
