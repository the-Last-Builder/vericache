use super::*;
use foundation::tracked::closure_from;
use vstd::prelude::*;

verus! {

pub(super) proof fn lemma_cuda_puts_empty(
    puts: Seq<PendingGpuWrite>,
    core: &CacheCore,
    timeline: TimelineId,
)
    requires
        puts.len() == 0,
    ensures
        cuda_puts_valid(puts, core, timeline),
{
    reveal(cuda_puts_disjoint);
    reveal(cuda_puts_valid);
}

pub(super) proof fn lemma_cuda_put_extract_valid(
    puts: Seq<PendingGpuWrite>,
    core: &CacheCore,
    timeline: TimelineId,
    idx: int,
)
    requires
        cuda_puts_valid(puts, core, timeline),
        0 <= idx < puts.len(),
    ensures
        puts[idx].valid(core, timeline),
{
    reveal(cuda_puts_valid);
}

proof fn lemma_cuda_put_valid_after_poll(
    entry: &PendingGpuWrite,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
)
    requires
        entry.valid(before_core, timeline),
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.frames_spec() == before_core.frames_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
        after_core.timelines_spec().next_seq_spec() == before_core.timelines_spec().next_seq_spec(),
        after_core.timelines_spec().events_spec() == before_core.timelines_spec().events_spec(),
        after_core.ledger_spec()@.submitted == before_core.ledger_spec()@.submitted,
        after_core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
        after_core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
        before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
    ensures
        entry.valid(after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(after_core);
    lemma_ledger_preserves_dest_poll(
        before_core.ledger_spec()@,
        after_core.ledger_spec()@,
        after_core.node(),
        entry.slot@.loc,
    );
    match entry.action {
        PendingGpuAction::Abort => {},
        PendingGpuAction::Publish { proof, .. } => {
            proof@.preserve(
                before_core.ledger_spec()@,
                after_core.ledger_spec()@,
                after_core.node(),
            );
        },
    }
}

pub(super) proof fn lemma_cuda_backend_after_poll_events<'a>(
    puts: Seq<PendingGpuWrite>,
    before_core: &CacheCore,
    after_core: &CacheCore,
    gpu_tier: TierId,
    gpu_first_frame: FrameId,
    gpu_num_frames: u32,
    gpu_space: MemorySpace,
    pinned_host: &Option<PinnedHostSlab<'a>>,
    timeline: TimelineId,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        slab_bound_to_core(before_core, gpu_tier, gpu_first_frame, gpu_num_frames, gpu_space),
        pinned_host matches Some(h) ==> h.well_formed(before_core),
        cuda_puts_valid(puts, before_core, timeline),
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.tiers_spec() == before_core.tiers_spec(),
        after_core.frames_spec() == before_core.frames_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
        after_core.timelines_spec().next_seq_spec() == before_core.timelines_spec().next_seq_spec(),
        after_core.timelines_spec().events_spec() == before_core.timelines_spec().events_spec(),
        after_core.ledger_spec()@.submitted == before_core.ledger_spec()@.submitted,
        after_core.ledger_spec()@.succeeded == before_core.ledger_spec()@.succeeded,
        after_core.ledger_spec()@.revoked == before_core.ledger_spec()@.revoked,
        before_core.ledger_spec()@.knows.set().subset_of(after_core.ledger_spec()@.knows.set()),
    ensures
        slab_bound_to_core(after_core, gpu_tier, gpu_first_frame, gpu_num_frames, gpu_space),
        pinned_host matches Some(h) ==> h.well_formed(after_core),
        cuda_puts_valid(puts, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(PendingGpuWrite::valid);
    lemma_slab_bound_same_frames(
        before_core,
        after_core,
        gpu_tier,
        gpu_first_frame,
        gpu_num_frames,
        gpu_space,
    );
    if let Some(ref h) = *pinned_host {
        lemma_slab_bound_same_frames(
            before_core,
            after_core,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
    reveal(cuda_puts_valid);
    assert forall|i: int| 0 <= i < puts.len() implies (#[trigger] puts[i]).valid(
        after_core,
        timeline,
    ) by {
        lemma_cuda_put_valid_after_poll(&puts[i], before_core, after_core, timeline);
    }
}

pub(super) proof fn lemma_cuda_certify_ready_put(
    entry: &PendingGpuWrite,
    proof_obj: PendingPutProof,
    core: &CacheCore,
    timeline: TimelineId,
) -> (cert: ValueCertificate)
    requires
        core.well_formed(),
        entry.valid(core, timeline),
        entry.ticket.sequence() <= core.timelines_spec().confirmed_seq_spec()[timeline as int],
        entry.action matches PendingGpuAction::Publish { proof, .. } && proof@ == proof_obj,
    ensures
        cert == proof_obj.certificate,
        replica_valid(core.ledger_spec()@, core.node(), cert),
        core.citations().contains(cert.writer),
        core.frames_spec().frame(entry.frame).completed_writes(cert),
{
    lemma_core_wf_facts(core);
    let s = entry.ticket.sequence();
    assert(core.timelines_spec().issued(timeline, s));
    assert(core.timelines_spec().events_spec()@.contains_key((timeline, s)));
    assert(core.ledger_spec()@.submitted.dom().contains(proof_obj.certificate.writer));
    assert(core.timelines_spec().observed(timeline, s, core.node(), core.ledger_spec()@.knows));
    assert(core.citations().contains(proof_obj.certificate.writer));
    reveal(cuda_put_events_covered);
    assert forall|p: int| 0 <= p < proof_obj.events.len() implies core.citations().contains(
        #[trigger] proof_obj.events[p],
    ) && event_succeeded(core.ledger_spec()@, core.node(), proof_obj.events[p]) by {
        assert(cuda_event_at_prefix(
            proof_obj.events[p],
            core.timelines_spec().events_spec()@,
            timeline,
            s,
        ));
        let q = choose|q: u64|
            1 <= q <= s && (#[trigger] core.timelines_spec().events_spec()@[(timeline, q)])
                == proof_obj.events[p] && proof_obj.events[p].source.succeeds_on_observe();
        assert(q <= core.timelines_spec().confirmed_seq_spec()[timeline as int]);
        assert(core.timelines_spec().issued(timeline, q));
        assert(core.timelines_spec().events_spec()@.contains_key((timeline, q)));
        assert(core.ledger_spec()@.submitted.dom().contains(proof_obj.events[p]));
        assert(core.timelines_spec().observed(timeline, q, core.node(), core.ledger_spec()@.knows));
    }
    assert(proof_obj.observed(core.ledger_spec()@, core.node()));
    let cert = proof_obj.certify(core.ledger_spec()@, core.node());
    let tok = core.frames_spec().frame(entry.frame).token.unwrap();
    assert(slot_frontier(tok).last_writer == Some(cert.writer));
    assert(core.frames_spec().frame(entry.frame).completed_writes(cert));
    cert
}

proof fn lemma_confirmed_seq_in_citations(core: &CacheCore, timeline: TimelineId, s: u64)
    requires
        core.timelines_spec().well_formed(core.instance_id(), core.node(), &core.ledger_spec()@),
        timeline < core.timelines_spec().count_spec(),
        1 <= s <= core.timelines_spec().confirmed_seq_spec()[timeline as int],
    ensures
        core.timelines_spec().issued(timeline, s),
        core.timelines_spec().events_spec()@.contains_key((timeline, s)),
        core.citations().contains(core.timelines_spec().events_spec()@[(timeline, s)]),
{
    let w = core.timelines_spec().events_spec()@[(timeline, s)];
    assert(core.timelines_spec().issued(timeline, s));
    assert(core.timelines_spec().events_spec()@.contains_key((timeline, s)));
    assert(core.ledger_spec()@.submitted.dom().contains(w));
    assert(core.timelines_spec().observed(timeline, s, core.node(), core.ledger_spec()@.knows));
    assert(core.citations().contains(w));
}

pub(super) proof fn lemma_cuda_ready_put_writable(
    entry: &PendingGpuWrite,
    core: &CacheCore,
    timeline: TimelineId,
)
    requires
        core.well_formed(),
        entry.valid(core, timeline),
        entry.ticket.sequence() <= core.timelines_spec().confirmed_seq_spec()[timeline as int],
    ensures
        core.frames_spec().frame(entry.frame).writable(
            core.instance_id(),
            core.node(),
            &core.ledger_spec()@,
            ISet::empty(),
        ),
{
    hide(CacheCore::well_formed);
    hide(TimelineTable::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(core);
    lemma_core_timelines_wf(core);
    let tok = core.frames_spec().frame(entry.frame).token.unwrap();
    let s = entry.ticket.sequence();
    if let Some(w) = slot_frontier(tok).last_writer {
        if !core.citations().contains(w) {
            match entry.action {
                PendingGpuAction::Publish { .. } => {
                    assert(w == core.timelines_spec().events_spec()@[(timeline, s)]);
                    lemma_confirmed_seq_in_citations(core, timeline, s);
                    assert(core.citations().contains(w));
                    assert(false);
                },
                PendingGpuAction::Abort => {
                    assert(core.timelines_spec().ticket_covers(entry.ticket, w));
                    assert(exists|q: u64|
                        1 <= q <= s && #[trigger] core.timelines_spec().events_spec()@[(
                            timeline,
                            q,
                        )] == w);
                    let q = choose|q: u64|
                        1 <= q <= s && #[trigger] core.timelines_spec().events_spec()@[(
                            timeline,
                            q,
                        )] == w;
                    assert(core.timelines_spec().events_spec()@[(timeline, q)] == w);
                    lemma_confirmed_seq_in_citations(core, timeline, q);
                    assert(core.citations().contains(w));
                    assert(false);
                },
            }
        }
    }
    assert(core.citations().union(ISet::empty()) =~= core.citations());
    assert(write_admitted(slot_frontier(tok), core.citations()));
    assert(write_admitted(slot_frontier(tok), core.citations().union(ISet::empty())));
}

proof fn lemma_cuda_puts_disjoint_swap_remove(
    old_puts: Seq<PendingGpuWrite>,
    new_puts: Seq<PendingGpuWrite>,
    idx: int,
)
    requires
        cuda_puts_disjoint(old_puts),
        0 <= idx < old_puts.len(),
        new_puts.len() == old_puts.len() - 1,
        forall|i: int|
            0 <= i < new_puts.len() ==> new_puts[i] == if i == idx {
                old_puts[old_puts.len() - 1]
            } else {
                old_puts[i]
            },
    ensures
        cuda_puts_disjoint(new_puts),
{
    reveal(cuda_puts_disjoint);
    assert forall|i: int, j: int|
        0 <= i < new_puts.len() && 0 <= j < new_puts.len() && i != j implies {
        &&& (#[trigger] new_puts[i]).frame != (#[trigger] new_puts[j]).frame
        &&& new_puts[i].slot@.loc != new_puts[j].slot@.loc
        &&& match (new_puts[i].action, new_puts[j].action) {
            (
                PendingGpuAction::Publish { version: v1, only_if_current: false, .. },
                PendingGpuAction::Publish { version: v2, only_if_current: false, .. },
            ) => v1 != v2,
            _ => true,
        }
    } by {
        let oi = if i == idx {
            old_puts.len() - 1
        } else {
            i
        };
        let oj = if j == idx {
            old_puts.len() - 1
        } else {
            j
        };
        assert(oi != oj);
        assert(new_puts[i] == old_puts[oi]);
        assert(new_puts[j] == old_puts[oj]);
    }
}

proof fn lemma_cuda_disjoint_pair(puts: Seq<PendingGpuWrite>, i: int, j: int)
    requires
        cuda_puts_disjoint(puts),
        0 <= i < puts.len(),
        0 <= j < puts.len(),
        i != j,
    ensures
        puts[i].frame != puts[j].frame,
        puts[i].slot@.loc != puts[j].slot@.loc,
        match (puts[i].action, puts[j].action) {
            (
                PendingGpuAction::Publish { version: v1, only_if_current: false, .. },
                PendingGpuAction::Publish { version: v2, only_if_current: false, .. },
            ) => v1 != v2,
            _ => true,
        },
{
    reveal(cuda_puts_disjoint);
}

proof fn lemma_cuda_put_valid_after_retire_other(
    entry: &PendingGpuWrite,
    retired: &PendingGpuWrite,
    before_core: &CacheCore,
    after_core: &CacheCore,
    timeline: TimelineId,
)
    requires
        entry.valid(before_core, timeline),
        retired.valid(before_core, timeline),
        entry.frame != retired.frame,
        match (entry.action, retired.action) {
            (
                PendingGpuAction::Publish { version: v1, only_if_current: false, .. },
                PendingGpuAction::Publish { version: v2, only_if_current: false, .. },
            ) => v1 != v2,
            _ => true,
        },
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.timelines_spec() == before_core.timelines_spec(),
        after_core.ledger_spec() == before_core.ledger_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        after_core.frames_spec().preserves_initializing(
            &before_core.frames_spec(),
            Some(retired.frame),
        ),
        before_core.frames_spec().versions_spec()@.submap_of(
            after_core.frames_spec().versions_spec()@,
        ),
        after_core.frames_spec().versions_spec()@ == before_core.frames_spec().versions_spec()@ || (
        retired.action matches PendingGpuAction::Publish { version, proof, .. }
            && after_core.frames_spec().versions_spec()@
            == before_core.frames_spec().versions_spec()@.insert(version, proof@.certificate.bytes)
            && (before_core.frames_spec().versions_spec()@.contains_key(version)
            ==> before_core.frames_spec().versions_spec()@[version] == proof@.certificate.bytes)),
    ensures
        entry.valid(after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    assert(after_core.frames_spec().meta(entry.frame) == before_core.frames_spec().meta(
        entry.frame,
    ));
    assert(after_core.frames_spec().frame(entry.frame) == before_core.frames_spec().frame(
        entry.frame,
    ));
    match entry.action {
        PendingGpuAction::Abort => {},
        PendingGpuAction::Publish { version, only_if_current, .. } => {
            broadcast use vstd::map::group_map_lemmas;

            assert(retired.action matches PendingGpuAction::Publish {
                version: rv,
                only_if_current: true,
                ..
            } ==> before_core.frames_spec().versions_spec()@.contains_key(rv));
            if only_if_current {
                assert(before_core.frames_spec().versions_spec()@.dom().contains(version));
                assert(after_core.frames_spec().versions_spec()@.dom().contains(version));
            }
        },
    }
}

pub(super) proof fn lemma_cuda_backend_after_retire_put<'a>(
    old_puts: Seq<PendingGpuWrite>,
    new_puts: Seq<PendingGpuWrite>,
    idx: int,
    retired: &PendingGpuWrite,
    before_core: &CacheCore,
    after_core: &CacheCore,
    gpu_tier: TierId,
    gpu_first_frame: FrameId,
    gpu_num_frames: u32,
    gpu_space: MemorySpace,
    pinned_host: &Option<PinnedHostSlab<'a>>,
    timeline: TimelineId,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        slab_bound_to_core(before_core, gpu_tier, gpu_first_frame, gpu_num_frames, gpu_space),
        pinned_host matches Some(h) ==> h.well_formed(before_core),
        cuda_puts_valid(old_puts, before_core, timeline),
        0 <= idx < old_puts.len(),
        old_puts[idx] == *retired,
        new_puts.len() == old_puts.len() - 1,
        forall|i: int|
            0 <= i < new_puts.len() ==> new_puts[i] == if i == idx {
                old_puts[old_puts.len() - 1]
            } else {
                old_puts[i]
            },
        after_core.node_id_spec() == before_core.node_id_spec(),
        after_core.tiers_spec() == before_core.tiers_spec(),
        after_core.timelines_spec() == before_core.timelines_spec(),
        after_core.ledger_spec() == before_core.ledger_spec(),
        after_core.next_version_seq_spec() == before_core.next_version_seq_spec(),
        after_core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        after_core.frames_spec().same_layout(&before_core.frames_spec()),
        after_core.frames_spec().preserves_initializing(
            &before_core.frames_spec(),
            Some(retired.frame),
        ),
        before_core.spec_spec()@.versions_monotonic(after_core.spec_spec()@),
        after_core.spec_spec()@.versions == before_core.spec_spec()@.versions || (
        retired.action matches PendingGpuAction::Publish { version, proof, .. }
            && after_core.spec_spec()@.versions == before_core.spec_spec()@.versions.insert(
            version,
            proof@.certificate.bytes,
        ) && (before_core.spec_spec()@.versions.contains_key(version)
            ==> before_core.spec_spec()@.versions[version] == proof@.certificate.bytes)),
    ensures
        slab_bound_to_core(after_core, gpu_tier, gpu_first_frame, gpu_num_frames, gpu_space),
        pinned_host matches Some(h) ==> h.well_formed(after_core),
        cuda_puts_valid(new_puts, after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(PendingGpuWrite::valid);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(after_core);
    lemma_slab_bound_same_layout(
        before_core,
        after_core,
        gpu_tier,
        gpu_first_frame,
        gpu_num_frames,
        gpu_space,
    );
    if let Some(ref h) = *pinned_host {
        lemma_slab_bound_same_layout(
            before_core,
            after_core,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
        );
    }
    reveal(cuda_puts_valid);
    lemma_cuda_puts_disjoint_swap_remove(old_puts, new_puts, idx);
    assert(old_puts[idx].valid(before_core, timeline));
    assert forall|i: int| 0 <= i < new_puts.len() implies (#[trigger] new_puts[i]).valid(
        after_core,
        timeline,
    ) by {
        let oi = if i == idx {
            old_puts.len() - 1
        } else {
            i
        };
        assert(oi != idx);
        assert(new_puts[i] == old_puts[oi]);
        lemma_cuda_disjoint_pair(old_puts, oi, idx);
        assert(old_puts[oi].valid(before_core, timeline));
        lemma_cuda_put_valid_after_retire_other(
            &old_puts[oi],
            retired,
            before_core,
            after_core,
            timeline,
        );
    }
}

/// Consolidate the post-`begin_copy_hop` slot/slab/admission proofs into one
/// branch-free call so `copy_hop_cuda` stays small for CVC5.
pub(super) proof fn lemma_copy_hop_slots_ready<'a>(
    backend: &CudaBackend<'a>,
    before_core: &CacheCore,
    core: &CacheCore,
    src_frame: FrameId,
    dst_frame: FrameId,
    src_is_gpu: bool,
    dst_is_gpu: bool,
    src_frontier: foundation::tracked::Frontier<TransferDomain>,
    dst_frontier: foundation::tracked::Frontier<TransferDomain>,
)
    requires
        before_core.well_formed(),
        core.well_formed(),
        backend.well_formed(before_core),
        core.node_id_spec() == before_core.node_id_spec(),
        core.instance_id() == before_core.instance_id(),
        core.tiers_spec() == before_core.tiers_spec(),
        core.timelines_spec().only_watermarks(&before_core.timelines_spec()),
        core.ledger_spec() == before_core.ledger_spec(),
        core.frames_spec().same_directory(&before_core.frames_spec()),
        src_frame.0 < core.frames_spec().metas_spec().len(),
        dst_frame.0 < core.frames_spec().metas_spec().len(),
        core.frames_spec().meta(src_frame).state == STATE_PUBLISHED,
        core.frames_spec().meta(dst_frame).state == STATE_INITIALIZING,
        src_is_gpu ==> backend.owns_gpu(src_frame),
        !src_is_gpu ==> backend.owns_pinned_host(src_frame),
        dst_is_gpu ==> core.frames_spec().frame(dst_frame).tier == backend.gpu_tier,
        !dst_is_gpu ==> (backend.pinned_host matches Some(h) && core.frames_spec().frame(
            dst_frame,
        ).tier == h.tier),
        read_admitted(src_frontier, core.citations()),
        write_admitted(dst_frontier, core.citations()),
    ensures
        slab_bound_to_core(
            core,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
        ),
        backend.pinned_host matches Some(h) ==> h.well_formed(core) && h.tier != backend.gpu_tier
            && core.frames_spec().tier_frame_size(h.tier) == core.frames_spec().tier_frame_size(
            backend.gpu_tier,
        ),
        backend.timeline.well_formed(&core.timelines_spec(), core.node(), &core.ledger_spec()@),
        core.frames_spec().frame_size_spec() == before_core.frames_spec().frame_size_spec(),
        core.frames_spec().tier_frame_sizes_spec()
            == before_core.frames_spec().tier_frame_sizes_spec(),
        core.frames_spec().metas_spec().len() == before_core.frames_spec().metas_spec().len(),
        core.frames_spec().frame(src_frame).slot.is_full(),
        core.frames_spec().frame(src_frame).slot.well_formed(),
        core.frames_spec().frame(src_frame).slot.slot_len == core.frames_spec().tier_frame_size(
            backend.gpu_tier,
        ),
        core.frames_spec().frame(src_frame).slot.loc.node == core.node(),
        core.frames_spec().frame(dst_frame).slot.is_full(),
        core.frames_spec().frame(dst_frame).slot.well_formed(),
        core.frames_spec().frame(dst_frame).slot.slot_len == core.frames_spec().tier_frame_size(
            backend.gpu_tier,
        ),
        core.frames_spec().frame(dst_frame).slot.loc.node == core.node(),
        core.frames_spec().frame(src_frame).bound_version is Some,
        core.frames_spec().frame(src_frame).bound_version.unwrap().well_formed(),
        core.frames_spec().frame(src_frame).bound_version.unwrap() != VersionId::NONE,
        core.frames_spec().frame(src_frame).bound_version.unwrap().origin_node
            != core.node_id_spec() || core.frames_spec().frame(src_frame).bound_version.unwrap().seq
            < core.next_version_seq_spec() || core.frames_spec().frame(
            src_frame,
        ).bound_version.unwrap().seq >= core.version_seq_limit_spec(),
        core.frames_spec().frame(src_frame).cert is Some,
        core.frames_spec().frame(src_frame).cert.unwrap().slot == core.frames_spec().frame(
            src_frame,
        ).slot,
        core.frames_spec().frame(src_frame).cert.unwrap().bytes.len() == core.frames_spec().frame(
            src_frame,
        ).slot.sub_len,
        src_is_gpu ==> backend.gpu_buffer.slot_ref((src_frame.0 - backend.gpu_first_frame.0) as nat)
            == core.frames_spec().frame(src_frame).slot,
        !src_is_gpu ==> (backend.pinned_host matches Some(h) && h.owns(src_frame) && h.reg.slot_ref(
            (src_frame.0 - h.first_frame.0) as nat,
        ) == core.frames_spec().frame(src_frame).slot),
        dst_is_gpu ==> backend.owns_gpu(dst_frame) && backend.gpu_buffer.slot_ref(
            (dst_frame.0 - backend.gpu_first_frame.0) as nat,
        ) == core.frames_spec().frame(dst_frame).slot,
        !dst_is_gpu ==> (backend.pinned_host matches Some(h) && h.owns(dst_frame) && h.reg.slot_ref(
            (dst_frame.0 - h.first_frame.0) as nat,
        ) == core.frames_spec().frame(dst_frame).slot),
        read_admitted(
            src_frontier,
            foundation::tracked::closure_from(
                core.ledger_spec()@.submitted.map(),
                backend.timeline.stream.tail().union(core.citations()),
            ),
        ),
        write_admitted(
            dst_frontier,
            foundation::tracked::closure_from(
                core.ledger_spec()@.submitted.map(),
                backend.timeline.stream.tail().union(core.citations()),
            ),
        ),
        core.disjoint_from_published(
            ISet::empty().insert(core.frames_spec().frame(dst_frame).slot.loc),
        ),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(core);
    lemma_published_frame_slot_valid(core, src_frame);
    lemma_initializing_frame_slot_valid(core, src_frame);
    lemma_initializing_frame_slot_valid(core, dst_frame);
    lemma_cuda_slabs_same_directory(backend, before_core, core);
    if src_is_gpu {
        lemma_slab_frame_slot(
            core,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
            src_frame,
        );
    } else {
        let h = backend.pinned_host.unwrap();
        lemma_slab_frame_slot(core, h.tier, h.first_frame, h.num_frames, h.reg.space(), src_frame);
    }
    if dst_is_gpu {
        lemma_slab_owns_tier_frame(
            core,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
            dst_frame,
        );
        lemma_slab_frame_slot(
            core,
            backend.gpu_tier,
            backend.gpu_first_frame,
            backend.gpu_num_frames,
            backend.gpu_buffer.space(),
            dst_frame,
        );
    } else {
        let h = backend.pinned_host.unwrap();
        lemma_slab_owns_tier_frame(
            core,
            h.tier,
            h.first_frame,
            h.num_frames,
            h.reg.space(),
            dst_frame,
        );
        lemma_slab_frame_slot(core, h.tier, h.first_frame, h.num_frames, h.reg.space(), dst_frame);
    }
    lemma_cuda_timeline_preserved(&backend.timeline, before_core, core);
    lemma_citations_admit_stream_closure(core, backend.timeline.stream, src_frontier);
    lemma_citations_admit_stream_closure(core, backend.timeline.stream, dst_frontier);
    lemma_write_disjoint_from_published(
        core,
        dst_frame,
        ISet::empty().insert(core.frames_spec().frame(dst_frame).slot.loc),
    );
}

pub(super) proof fn lemma_finish_put_gpu_core_step(
    before_core: &CacheCore,
    after_begin_core: &CacheCore,
    core: &CacheCore,
    timeline_id: TimelineId,
    source: EventSource,
    h_frame: FrameId,
    h_slot: SlotRef,
    src_slot: SlotRef,
    ev: TransferEvent,
    ticket: Ticket,
    src_cert: ValueCertificate,
    old_tail: ISet<TransferEvent>,
) -> (put_proof: PendingPutProof)
    requires
        before_core.well_formed(),
        after_begin_core.well_formed(),
        core.well_formed(),
        core.node_id_spec() == before_core.node_id_spec() && before_core.node_id_spec()
            == after_begin_core.node_id_spec(),
        core.instance_id() == before_core.instance_id() && before_core.instance_id()
            == after_begin_core.instance_id(),
        core.tiers_spec() == before_core.tiers_spec() && before_core.tiers_spec()
            == after_begin_core.tiers_spec(),
        core.spec_spec() == before_core.spec_spec() && before_core.spec_spec()
            == after_begin_core.spec_spec(),
        core.next_version_seq_spec() == before_core.next_version_seq_spec() + 1,
        core.next_version_seq_spec() == after_begin_core.next_version_seq_spec(),
        core.version_seq_limit_spec() == before_core.version_seq_limit_spec(),
        core.version_seq_limit_spec() == after_begin_core.version_seq_limit_spec(),
        core.frames_spec() == after_begin_core.frames_spec(),
        core.active_leases_spec() == after_begin_core.active_leases_spec(),
        after_begin_core.frames_spec().same_directory(&before_core.frames_spec()),
        after_begin_core.frames_spec().preserves_initializing(&before_core.frames_spec(), None),
        after_begin_core.timelines_spec() == before_core.timelines_spec(),
        after_begin_core.ledger_spec() == before_core.ledger_spec(),
        h_frame.0 < before_core.frames_spec().metas_spec().len(),
        before_core.frames_spec().meta(h_frame).state != STATE_INITIALIZING,
        after_begin_core.frames_spec().frame(h_frame).slot == h_slot,
        core.timelines_spec().count_spec() == before_core.timelines_spec().count_spec(),
        timeline_id < before_core.timelines_spec().count_spec(),
        (timeline_id as int) < before_core.timelines_spec().next_seq_spec()@.len(),
        core.timelines_spec().kinds_spec() == after_begin_core.timelines_spec().kinds_spec(),
        core.timelines_spec().confirmed_seq_spec()
            == after_begin_core.timelines_spec().confirmed_seq_spec(),
        core.timelines_spec().next_seq_spec()@
            == before_core.timelines_spec().next_seq_spec()@.update(
            timeline_id as int,
            (ticket.sequence() + 1) as u64,
        ),
        core.timelines_spec().events_spec()@ == before_core.timelines_spec().events_spec()@.insert(
            (timeline_id, ticket.sequence()),
            ev,
        ),
        core.timelines_spec().events_spec()@[(timeline_id, ticket.sequence())] == ev,
        core.timelines_spec().issued(timeline_id, ticket.sequence()),
        ticket != Ticket::NONE,
        ticket.timeline_id() == timeline_id,
        ticket.sequence() == before_core.timelines_spec().next_seq_spec()[timeline_id as int],
        !after_begin_core.ledger_spec()@.submitted.dom().contains(ev),
        ev.source == source,
        ev.source.succeeds_on_observe(),
        core.ledger_spec()@.knows == after_begin_core.ledger_spec()@.knows,
        core.ledger_spec()@.succeeded == after_begin_core.ledger_spec()@.succeeded,
        core.ledger_spec()@.revoked == after_begin_core.ledger_spec()@.revoked,
        core.ledger_spec()@.submitted.instance_id()
            == after_begin_core.ledger_spec()@.submitted.instance_id(),
        core.ledger_spec()@.submitted.dom().contains(ev),
        core.ledger_spec()@.submitted.map()
            == after_begin_core.ledger_spec()@.submitted.map().insert(
            ev,
            foundation::tracked::EventInfo {
                access: copy_access(src_slot, h_slot),
                read_guards: core.ledger_spec()@.submitted.map()[ev].read_guards,
                effect: copy_effect(src_slot, h_slot),
                preds: closure_from(
                    after_begin_core.ledger_spec()@.submitted.map(),
                    old_tail.union(after_begin_core.citations()),
                ),
            },
        ),
        core.ledger_spec()@.submitted.map()[ev].preds.subset_of(
            after_begin_core.ledger_spec()@.submitted.dom(),
        ),
        after_begin_core.ledger_spec()@.submitted.map().submap_of(
            core.ledger_spec()@.submitted.map(),
        ),
        replica_valid(after_begin_core.ledger_spec()@, after_begin_core.node(), src_cert),
        source_slice_valid(src_cert, src_slot),
        src_cert.slot == src_slot,
        src_cert.bytes.len() == src_slot.sub_len,
        src_slot.sub_len == h_slot.slot_len,
        src_slot.loc != h_slot.loc,
        h_slot.is_full(),
        h_slot.well_formed(),
        h_slot.sub_len == h_slot.slot_len,
        h_slot.loc.node == core.node(),
        core_transition(
            before_core,
            after_begin_core,
            CoreFootprint::preserve_all(Some(timeline_id), Some(source)),
        ),
    ensures
        cuda_timeline_events_preserved(
            before_core.timelines_spec().events_spec()@,
            core.timelines_spec().events_spec()@,
            timeline_id,
            before_core.timelines_spec().next_seq_spec()[timeline_id as int],
        ),
        cuda_ledger_writes_only(before_core.ledger_spec()@, core.ledger_spec()@, Some(h_slot.loc)),
        core.frames_spec().preserves_initializing(&before_core.frames_spec(), Some(h_frame)),
        core.frames_spec().versions_spec()@ == before_core.frames_spec().versions_spec()@,
        core_transition(
            before_core,
            core,
            CoreFootprint::preserve_all(Some(timeline_id), Some(source)),
        ),
        core.timelines_spec().ticket_covers(ticket, put_proof.certificate.writer),
        put_proof.certificate.slot == h_slot,
        put_proof.certificate.bytes == src_cert.bytes,
        put_proof.certificate.bytes.len() == h_slot.slot_len,
        put_proof.certificate.writer == core.timelines_spec().events_spec()@[(
            timeline_id,
            ticket.sequence(),
        )],
        put_proof.pending(core.ledger_spec()@, core.node()),
        cuda_put_events_covered(
            put_proof.events,
            core.timelines_spec().events_spec()@,
            timeline_id,
            ticket.sequence(),
        ),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    hide(copy_effect);
    hide(effect_implies);
    hide(FrameTable::same_directory);
    hide(FrameTable::preserves_initializing);
    hide(TimelineTable::ticket_covers);
    lemma_core_wf_facts(after_begin_core);
    lemma_core_wf_facts(core);
    let val = core.ledger_spec()@.submitted.map()[ev];
    lemma_iset_subset_closure_union(
        after_begin_core.ledger_spec()@.submitted.map(),
        old_tail,
        after_begin_core.citations(),
        after_begin_core.citations(),
        ev,
    );
    let put_proof = lemma_cuda_single_put_proof(
        after_begin_core.ledger_spec()@,
        core.ledger_spec()@,
        core.node(),
        src_cert,
        src_slot,
        h_slot,
        ev,
        val,
    );
    lemma_cuda_single_event_covered(
        put_proof.events,
        core.timelines_spec().events_spec()@,
        timeline_id,
        ticket.sequence(),
        ev,
    );
    lemma_cuda_events_preserved_insert(
        before_core.timelines_spec().events_spec()@,
        core.timelines_spec().events_spec()@,
        timeline_id,
        before_core.timelines_spec().next_seq_spec()[timeline_id as int],
        timeline_id,
        ticket.sequence(),
        ev,
    );
    lemma_cuda_ledger_writes_only_insert(
        before_core.ledger_spec()@,
        core.ledger_spec()@,
        ev,
        val,
        h_slot.loc,
    );
    assert(core.frames_spec().preserves_initializing(&before_core.frames_spec(), Some(h_frame)))
        by {
        reveal(FrameTable::preserves_initializing);
    }
    assert(core.frames_spec().versions_spec()@ == before_core.frames_spec().versions_spec()@) by {
        reveal(FrameTable::same_directory);
    }
    assert(before_core.frames_spec().frame(h_frame).slot == h_slot) by {
        reveal(FrameTable::same_directory);
    }
    lemma_core_transition_after_single_enqueue(
        before_core,
        after_begin_core,
        core,
        timeline_id,
        source,
        h_frame,
        h_slot,
        ev,
        ticket.sequence(),
    );
    lemma_cuda_ticket_covers_exact(&core.timelines_spec(), ticket, ev);
    put_proof
}

pub proof fn lemma_cuda_puts_bindings_empty(
    puts: Seq<PendingGpuWrite>,
    bindings: Map<VersionId, VersionBinding>,
)
    requires
        puts.len() == 0,
    ensures
        cuda_puts_bindings_covered(puts, bindings),
{
    reveal(cuda_puts_bindings_covered);
}

pub proof fn lemma_cuda_puts_bindings_push_abort(
    old_puts: Seq<PendingGpuWrite>,
    new_puts: Seq<PendingGpuWrite>,
    entry: PendingGpuWrite,
    bindings: Map<VersionId, VersionBinding>,
)
    requires
        cuda_puts_bindings_covered(old_puts, bindings),
        new_puts == old_puts.push(entry),
        !(entry.action matches PendingGpuAction::Publish { only_if_current: false, .. }),
    ensures
        cuda_puts_bindings_covered(new_puts, bindings),
{
    reveal(cuda_puts_bindings_covered);
    assert forall|i: int| 0 <= i < new_puts.len() implies match (#[trigger] new_puts[i]).action {
        PendingGpuAction::Publish { version, proof, only_if_current: false, .. } => {
            &&& bindings.contains_key(version)
            &&& bindings[version].instance_id() == cluster_version_instance_id()
            &&& bindings[version].key() == (version.origin_node as NodeId, version.seq)
            &&& bindings[version].value() == proof@.certificate.bytes
        },
        _ => true,
    } by {
        if i < old_puts.len() {
            assert(new_puts[i] == old_puts[i]);
        } else {
            assert(new_puts[i] == entry);
        }
    }
}

pub proof fn lemma_cuda_puts_bindings_push_publish(
    old_puts: Seq<PendingGpuWrite>,
    new_puts: Seq<PendingGpuWrite>,
    entry: PendingGpuWrite,
    old_bindings: Map<VersionId, VersionBinding>,
    new_bindings: Map<VersionId, VersionBinding>,
    version: VersionId,
    binding: VersionBinding,
    after_core: &CacheCore,
    timeline: TimelineId,
)
    requires
        cuda_puts_bindings_covered(old_puts, old_bindings),
        cuda_puts_valid(new_puts, after_core, timeline),
        new_puts == old_puts.push(entry),
        entry.action matches PendingGpuAction::Publish {
            version: ev,
            proof: ep,
            only_if_current: false,
            ..
        } && ev == version && binding.value() == ep@.certificate.bytes,
        binding.instance_id() == cluster_version_instance_id(),
        binding.key() == (version.origin_node as NodeId, version.seq),
        new_bindings == old_bindings.insert(version, binding),
    ensures
        cuda_puts_bindings_covered(new_puts, new_bindings),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(cuda_puts_valid);
    reveal(cuda_puts_disjoint);
    reveal(cuda_puts_bindings_covered);
    let last = old_puts.len() as int;
    assert(new_puts[last] == entry);
    assert forall|i: int| 0 <= i < new_puts.len() implies match (#[trigger] new_puts[i]).action {
        PendingGpuAction::Publish { version: v, proof, only_if_current: false, .. } => {
            &&& new_bindings.contains_key(v)
            &&& new_bindings[v].instance_id() == cluster_version_instance_id()
            &&& new_bindings[v].key() == (v.origin_node as NodeId, v.seq)
            &&& new_bindings[v].value() == proof@.certificate.bytes
        },
        _ => true,
    } by {
        if i < last {
            assert(new_puts[i] == old_puts[i]);
            assert(i != last);
            if let PendingGpuAction::Publish { version: v, only_if_current: false, .. } =
                new_puts[i].action {
                assert(v != version);
                assert(old_bindings.contains_key(v));
                assert(new_bindings[v] == old_bindings[v]);
            }
        }
    }
}

pub proof fn lemma_cuda_puts_bindings_extract(
    puts: Seq<PendingGpuWrite>,
    bindings: Map<VersionId, VersionBinding>,
    idx: int,
)
    requires
        cuda_puts_bindings_covered(puts, bindings),
        0 <= idx < puts.len(),
    ensures
        match puts[idx].action {
            PendingGpuAction::Publish { version, proof, only_if_current: false, .. } => {
                &&& bindings.contains_key(version)
                &&& bindings[version].instance_id() == cluster_version_instance_id()
                &&& bindings[version].key() == (version.origin_node as NodeId, version.seq)
                &&& bindings[version].value() == proof@.certificate.bytes
            },
            _ => true,
        },
{
    reveal(cuda_puts_bindings_covered);
}

pub(super) proof fn lemma_cuda_puts_bindings_swap_remove(
    old_puts: Seq<PendingGpuWrite>,
    new_puts: Seq<PendingGpuWrite>,
    old_bindings: Map<VersionId, VersionBinding>,
    new_bindings: Map<VersionId, VersionBinding>,
    idx: int,
    removed_ver: Option<VersionId>,
    before_core: &CacheCore,
    timeline: TimelineId,
)
    requires
        cuda_puts_valid(old_puts, before_core, timeline),
        cuda_puts_bindings_covered(old_puts, old_bindings),
        0 <= idx < old_puts.len(),
        new_puts.len() == old_puts.len() - 1,
        forall|i: int|
            0 <= i < new_puts.len() ==> new_puts[i] == if i == idx {
                old_puts[old_puts.len() - 1]
            } else {
                old_puts[i]
            },
        match removed_ver {
            None => new_bindings == old_bindings,
            Some(rv) => (old_puts[idx].action matches PendingGpuAction::Publish {
                version,
                only_if_current: false,
                ..
            } && version == rv) && new_bindings == old_bindings.remove(rv),
        },
    ensures
        cuda_puts_bindings_covered(new_puts, new_bindings),
{
    broadcast use vstd::map::group_map_lemmas;

    reveal(cuda_puts_valid);
    reveal(cuda_puts_bindings_covered);
    assert forall|i: int| 0 <= i < new_puts.len() implies match (#[trigger] new_puts[i]).action {
        PendingGpuAction::Publish { version, proof, only_if_current: false, .. } => {
            &&& new_bindings.contains_key(version)
            &&& new_bindings[version].instance_id() == cluster_version_instance_id()
            &&& new_bindings[version].key() == (version.origin_node as NodeId, version.seq)
            &&& new_bindings[version].value() == proof@.certificate.bytes
        },
        _ => true,
    } by {
        let oi = if i == idx {
            old_puts.len() - 1
        } else {
            i
        };
        assert(oi != idx);
        assert(new_puts[i] == old_puts[oi]);
        lemma_cuda_disjoint_pair(old_puts, oi, idx);
        if let PendingGpuAction::Publish { version, only_if_current: false, .. } =
            old_puts[oi].action {
            assert(old_bindings.contains_key(version));
            if let Some(rv) = removed_ver {
                assert(version != rv);
                assert(new_bindings[version] == old_bindings[version]);
            }
        }
    }
}

} // verus!
