use super::*;
use foundation::tracked::closure_from;
use vstd::prelude::*;

verus! {

pub(super) proof fn lemma_cuda_puts_empty(
    puts: Seq<PendingGpuWrite>,
    core: &CacheCore,
    timeline: u8,
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
    timeline: u8,
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
    timeline: u8,
)
    requires
        entry.valid(before_core, timeline),
        before_core.well_formed(),
        after_core.well_formed(),
        after_core.node_id == before_core.node_id,
        after_core.frames == before_core.frames,
        after_core.next_version_seq == before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.next_seq == before_core.timelines.next_seq,
        after_core.timelines.events == before_core.timelines.events,
        after_core.ledger@.submitted == before_core.ledger@.submitted,
        after_core.ledger@.succeeded == before_core.ledger@.succeeded,
        after_core.ledger@.revoked == before_core.ledger@.revoked,
        before_core.ledger@.knows.set().subset_of(after_core.ledger@.knows.set()),
    ensures
        entry.valid(after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(before_core);
    lemma_core_wf_facts(after_core);
    lemma_ledger_preserves_dest_poll(
        before_core.ledger@,
        after_core.ledger@,
        after_core.node(),
        entry.slot@.loc,
    );
    match entry.action {
        PendingGpuAction::Abort => {},
        PendingGpuAction::Publish { proof, .. } => {
            proof@.preserve(before_core.ledger@, after_core.ledger@, after_core.node());
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
    timeline: u8,
)
    requires
        before_core.well_formed(),
        after_core.well_formed(),
        slab_bound_to_core(before_core, gpu_tier, gpu_first_frame, gpu_num_frames, gpu_space),
        pinned_host matches Some(h) ==> h.well_formed(before_core),
        cuda_puts_valid(puts, before_core, timeline),
        after_core.node_id == before_core.node_id,
        after_core.tiers == before_core.tiers,
        after_core.frames == before_core.frames,
        after_core.next_version_seq == before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.timelines.count == before_core.timelines.count,
        after_core.timelines.next_seq == before_core.timelines.next_seq,
        after_core.timelines.events == before_core.timelines.events,
        after_core.ledger@.submitted == before_core.ledger@.submitted,
        after_core.ledger@.succeeded == before_core.ledger@.succeeded,
        after_core.ledger@.revoked == before_core.ledger@.revoked,
        before_core.ledger@.knows.set().subset_of(after_core.ledger@.knows.set()),
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
    timeline: u8,
) -> (cert: ValueCertificate)
    requires
        core.well_formed(),
        entry.valid(core, timeline),
        entry.ticket.sequence() <= core.timelines.confirmed_seq[timeline as int],
        entry.action matches PendingGpuAction::Publish { proof, .. } && proof@ == proof_obj,
    ensures
        cert == proof_obj.certificate,
        replica_valid(core.ledger@, core.node(), cert),
        core.citations().contains(cert.writer),
        core.frames.frame(entry.frame).completed_writes(cert),
{
    lemma_core_wf_facts(core);
    let s = entry.ticket.sequence();
    assert(core.timelines.issued(timeline, s));
    assert(core.timelines.events@.contains_key((timeline, s)));
    assert(core.ledger@.submitted.dom().contains(proof_obj.certificate.writer));
    assert(core.timelines.observed(timeline, s, core.node(), core.ledger@.knows));
    assert(core.citations().contains(proof_obj.certificate.writer));
    reveal(cuda_put_events_covered);
    assert forall|p: int| 0 <= p < proof_obj.events.len() implies core.citations().contains(
        #[trigger] proof_obj.events[p],
    ) && event_succeeded(core.ledger@, core.node(), proof_obj.events[p]) by {
        assert(cuda_event_at_prefix(proof_obj.events[p], core.timelines.events@, timeline, s));
        let q = choose|q: u64|
            1 <= q <= s && (#[trigger] core.timelines.events@[(timeline, q)]) == proof_obj.events[p]
                && proof_obj.events[p].source.succeeds_on_observe();
        assert(q <= core.timelines.confirmed_seq[timeline as int]);
        assert(core.timelines.issued(timeline, q));
        assert(core.timelines.events@.contains_key((timeline, q)));
        assert(core.ledger@.submitted.dom().contains(proof_obj.events[p]));
        assert(core.timelines.observed(timeline, q, core.node(), core.ledger@.knows));
    }
    assert(proof_obj.observed(core.ledger@, core.node()));
    let cert = proof_obj.certify(core.ledger@, core.node());
    let tok = core.frames.frame(entry.frame).token.unwrap();
    assert(slot_frontier(tok).last_writer == Some(cert.writer));
    assert(core.frames.frame(entry.frame).completed_writes(cert));
    cert
}

proof fn lemma_confirmed_seq_in_citations(core: &CacheCore, timeline: u8, s: u64)
    requires
        core.timelines.well_formed(core.instance_id(), core.node(), &core.ledger@),
        timeline < core.timelines.count,
        1 <= s <= core.timelines.confirmed_seq[timeline as int],
    ensures
        core.timelines.issued(timeline, s),
        core.timelines.events@.contains_key((timeline, s)),
        core.citations().contains(core.timelines.events@[(timeline, s)]),
{
    let w = core.timelines.events@[(timeline, s)];
    assert(core.timelines.issued(timeline, s));
    assert(core.timelines.events@.contains_key((timeline, s)));
    assert(core.ledger@.submitted.dom().contains(w));
    assert(core.timelines.observed(timeline, s, core.node(), core.ledger@.knows));
    assert(core.citations().contains(w));
}

pub(super) proof fn lemma_cuda_ready_put_writable(
    entry: &PendingGpuWrite,
    core: &CacheCore,
    timeline: u8,
)
    requires
        core.well_formed(),
        entry.valid(core, timeline),
        entry.ticket.sequence() <= core.timelines.confirmed_seq[timeline as int],
    ensures
        core.frames.frame(entry.frame).writable(
            core.instance_id(),
            core.node(),
            &core.ledger@,
            ISet::empty(),
        ),
{
    hide(CacheCore::well_formed);
    hide(TimelineTable::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    lemma_core_wf_facts(core);
    lemma_core_timelines_wf(core);
    let tok = core.frames.frame(entry.frame).token.unwrap();
    let s = entry.ticket.sequence();
    if let Some(w) = slot_frontier(tok).last_writer {
        if !core.citations().contains(w) {
            match entry.action {
                PendingGpuAction::Publish { .. } => {
                    assert(w == core.timelines.events@[(timeline, s)]);
                    lemma_confirmed_seq_in_citations(core, timeline, s);
                    assert(core.citations().contains(w));
                    assert(false);
                },
                PendingGpuAction::Abort => {
                    assert(core.timelines.ticket_covers(entry.ticket, w));
                    assert(exists|q: u64|
                        1 <= q <= s && #[trigger] core.timelines.events@[(timeline, q)] == w);
                    let q = choose|q: u64|
                        1 <= q <= s && #[trigger] core.timelines.events@[(timeline, q)] == w;
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
    timeline: u8,
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
        after_core.node_id == before_core.node_id,
        after_core.timelines == before_core.timelines,
        after_core.ledger == before_core.ledger,
        after_core.next_version_seq == before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.frames.same_layout(&before_core.frames),
        after_core.frames.preserves_initializing(&before_core.frames, Some(retired.frame)),
        before_core.frames.versions@.submap_of(after_core.frames.versions@),
        after_core.frames.versions@ == before_core.frames.versions@ || (
        retired.action matches PendingGpuAction::Publish { version, proof, .. }
            && after_core.frames.versions@ == before_core.frames.versions@.insert(
            version,
            proof@.certificate.bytes,
        ) && (before_core.frames.versions@.contains_key(version)
            ==> before_core.frames.versions@[version] == proof@.certificate.bytes)),
    ensures
        entry.valid(after_core, timeline),
{
    hide(CacheCore::well_formed);
    hide(replica_valid);
    hide(complete_receipts);
    assert(after_core.frames.meta(entry.frame) == before_core.frames.meta(entry.frame));
    assert(after_core.frames.frame(entry.frame) == before_core.frames.frame(entry.frame));
    match entry.action {
        PendingGpuAction::Abort => {},
        PendingGpuAction::Publish { version, only_if_current, .. } => {
            broadcast use vstd::map::group_map_lemmas;

            assert(retired.action matches PendingGpuAction::Publish {
                version: rv,
                only_if_current: true,
                ..
            } ==> before_core.frames.versions@.contains_key(rv));
            if only_if_current {
                assert(before_core.frames.versions@.dom().contains(version));
                assert(after_core.frames.versions@.dom().contains(version));
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
    timeline: u8,
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
        after_core.node_id == before_core.node_id,
        after_core.tiers == before_core.tiers,
        after_core.timelines == before_core.timelines,
        after_core.ledger == before_core.ledger,
        after_core.next_version_seq == before_core.next_version_seq,
        after_core.version_seq_limit == before_core.version_seq_limit,
        after_core.frames.same_layout(&before_core.frames),
        after_core.frames.preserves_initializing(&before_core.frames, Some(retired.frame)),
        before_core.spec@.versions_monotonic(after_core.spec@),
        after_core.spec@.versions == before_core.spec@.versions || (
        retired.action matches PendingGpuAction::Publish { version, proof, .. }
            && after_core.spec@.versions == before_core.spec@.versions.insert(
            version,
            proof@.certificate.bytes,
        ) && (before_core.spec@.versions.contains_key(version)
            ==> before_core.spec@.versions[version] == proof@.certificate.bytes)),
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
        core.node_id == before_core.node_id,
        core.instance_id() == before_core.instance_id(),
        core.tiers == before_core.tiers,
        core.timelines.only_watermarks(&before_core.timelines),
        core.ledger == before_core.ledger,
        core.frames.same_directory(&before_core.frames),
        src_frame.0 < core.frames.metas.len(),
        dst_frame.0 < core.frames.metas.len(),
        core.frames.meta(src_frame).state == STATE_PUBLISHED,
        core.frames.meta(dst_frame).state == STATE_INITIALIZING,
        src_is_gpu ==> backend.owns_gpu(src_frame),
        !src_is_gpu ==> backend.owns_pinned_host(src_frame),
        dst_is_gpu ==> core.frames.frame(dst_frame).tier == backend.gpu_tier,
        !dst_is_gpu ==> (backend.pinned_host matches Some(h) && core.frames.frame(dst_frame).tier
            == h.tier),
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
            && core.frames.tier_frame_size(h.tier) == core.frames.tier_frame_size(backend.gpu_tier),
        backend.timeline.well_formed(&core.timelines, core.node(), &core.ledger@),
        core.frames.frame_size == before_core.frames.frame_size,
        core.frames.tier_frame_sizes == before_core.frames.tier_frame_sizes,
        core.frames.metas.len() == before_core.frames.metas.len(),
        core.frames.frame(src_frame).slot.is_full(),
        core.frames.frame(src_frame).slot.well_formed(),
        core.frames.frame(src_frame).slot.slot_len == core.frames.tier_frame_size(backend.gpu_tier),
        core.frames.frame(src_frame).slot.loc.node == core.node(),
        core.frames.frame(dst_frame).slot.is_full(),
        core.frames.frame(dst_frame).slot.well_formed(),
        core.frames.frame(dst_frame).slot.slot_len == core.frames.tier_frame_size(backend.gpu_tier),
        core.frames.frame(dst_frame).slot.loc.node == core.node(),
        core.frames.frame(src_frame).bound_version is Some,
        core.frames.frame(src_frame).bound_version.unwrap().well_formed(),
        core.frames.frame(src_frame).bound_version.unwrap() != VersionId::NONE,
        core.frames.frame(src_frame).bound_version.unwrap().origin_node != core.node_id
            || core.frames.frame(src_frame).bound_version.unwrap().seq < core.next_version_seq
            || core.frames.frame(src_frame).bound_version.unwrap().seq >= core.version_seq_limit,
        core.frames.frame(src_frame).cert is Some,
        core.frames.frame(src_frame).cert.unwrap().slot == core.frames.frame(src_frame).slot,
        core.frames.frame(src_frame).cert.unwrap().bytes.len() == core.frames.frame(
            src_frame,
        ).slot.sub_len,
        src_is_gpu ==> backend.gpu_buffer.slot_ref((src_frame.0 - backend.gpu_first_frame.0) as nat)
            == core.frames.frame(src_frame).slot,
        !src_is_gpu ==> (backend.pinned_host matches Some(h) && h.owns(src_frame) && h.reg.slot_ref(
            (src_frame.0 - h.first_frame.0) as nat,
        ) == core.frames.frame(src_frame).slot),
        dst_is_gpu ==> backend.owns_gpu(dst_frame) && backend.gpu_buffer.slot_ref(
            (dst_frame.0 - backend.gpu_first_frame.0) as nat,
        ) == core.frames.frame(dst_frame).slot,
        !dst_is_gpu ==> (backend.pinned_host matches Some(h) && h.owns(dst_frame) && h.reg.slot_ref(
            (dst_frame.0 - h.first_frame.0) as nat,
        ) == core.frames.frame(dst_frame).slot),
        read_admitted(
            src_frontier,
            foundation::tracked::closure_from(
                core.ledger@.submitted.map(),
                backend.timeline.stream.tail().union(core.citations()),
            ),
        ),
        write_admitted(
            dst_frontier,
            foundation::tracked::closure_from(
                core.ledger@.submitted.map(),
                backend.timeline.stream.tail().union(core.citations()),
            ),
        ),
        core.disjoint_from_published(ISet::empty().insert(core.frames.frame(dst_frame).slot.loc)),
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
        ISet::empty().insert(core.frames.frame(dst_frame).slot.loc),
    );
}

pub(super) proof fn lemma_finish_put_gpu_core_step(
    before_core: &CacheCore,
    after_begin_core: &CacheCore,
    core: &CacheCore,
    timeline_id: u8,
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
        core.node_id == before_core.node_id && before_core.node_id == after_begin_core.node_id,
        core.instance_id() == before_core.instance_id() && before_core.instance_id()
            == after_begin_core.instance_id(),
        core.tiers == before_core.tiers && before_core.tiers == after_begin_core.tiers,
        core.spec == before_core.spec && before_core.spec == after_begin_core.spec,
        core.next_version_seq == before_core.next_version_seq + 1,
        core.next_version_seq == after_begin_core.next_version_seq,
        core.version_seq_limit == before_core.version_seq_limit,
        core.version_seq_limit == after_begin_core.version_seq_limit,
        core.frames == after_begin_core.frames,
        core.active_leases == after_begin_core.active_leases,
        after_begin_core.frames.same_directory(&before_core.frames),
        after_begin_core.frames.preserves_initializing(&before_core.frames, None),
        after_begin_core.timelines == before_core.timelines,
        after_begin_core.ledger == before_core.ledger,
        h_frame.0 < before_core.frames.metas.len(),
        before_core.frames.meta(h_frame).state != STATE_INITIALIZING,
        after_begin_core.frames.frame(h_frame).slot == h_slot,
        core.timelines.count == before_core.timelines.count,
        timeline_id < before_core.timelines.count,
        (timeline_id as int) < before_core.timelines.next_seq@.len(),
        core.timelines.kinds == after_begin_core.timelines.kinds,
        core.timelines.confirmed_seq == after_begin_core.timelines.confirmed_seq,
        core.timelines.next_seq@ == before_core.timelines.next_seq@.update(
            timeline_id as int,
            (ticket.sequence() + 1) as u64,
        ),
        core.timelines.events@ == before_core.timelines.events@.insert(
            (timeline_id, ticket.sequence()),
            ev,
        ),
        core.timelines.events@[(timeline_id, ticket.sequence())] == ev,
        core.timelines.issued(timeline_id, ticket.sequence()),
        ticket != Ticket::NONE,
        ticket.timeline_id() == timeline_id,
        ticket.sequence() == before_core.timelines.next_seq[timeline_id as int],
        !after_begin_core.ledger@.submitted.dom().contains(ev),
        ev.source == source,
        ev.source.succeeds_on_observe(),
        core.ledger@.knows == after_begin_core.ledger@.knows,
        core.ledger@.succeeded == after_begin_core.ledger@.succeeded,
        core.ledger@.revoked == after_begin_core.ledger@.revoked,
        core.ledger@.submitted.instance_id() == after_begin_core.ledger@.submitted.instance_id(),
        core.ledger@.submitted.dom().contains(ev),
        core.ledger@.submitted.map() == after_begin_core.ledger@.submitted.map().insert(
            ev,
            foundation::tracked::EventInfo {
                access: copy_access(src_slot, h_slot),
                read_guards: core.ledger@.submitted.map()[ev].read_guards,
                effect: copy_effect(src_slot, h_slot),
                preds: closure_from(
                    after_begin_core.ledger@.submitted.map(),
                    old_tail.union(after_begin_core.citations()),
                ),
            },
        ),
        core.ledger@.submitted.map()[ev].preds.subset_of(after_begin_core.ledger@.submitted.dom()),
        after_begin_core.ledger@.submitted.map().submap_of(core.ledger@.submitted.map()),
        replica_valid(after_begin_core.ledger@, after_begin_core.node(), src_cert),
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
            Some(timeline_id),
            Some(source),
            FrameTouchMode::PreserveAllInit,
        ),
    ensures
        cuda_timeline_events_preserved(
            before_core.timelines.events@,
            core.timelines.events@,
            timeline_id,
            before_core.timelines.next_seq[timeline_id as int],
        ),
        cuda_ledger_writes_only(before_core.ledger@, core.ledger@, Some(h_slot.loc)),
        core.frames.preserves_initializing(&before_core.frames, Some(h_frame)),
        core.frames.versions@ == before_core.frames.versions@,
        core_transition(
            before_core,
            core,
            Some(timeline_id),
            Some(source),
            FrameTouchMode::PreserveAllInit,
        ),
        core.timelines.ticket_covers(ticket, put_proof.certificate.writer),
        put_proof.certificate.slot == h_slot,
        put_proof.certificate.bytes == src_cert.bytes,
        put_proof.certificate.bytes.len() == h_slot.slot_len,
        put_proof.certificate.writer == core.timelines.events@[(timeline_id, ticket.sequence())],
        put_proof.pending(core.ledger@, core.node()),
        cuda_put_events_covered(
            put_proof.events,
            core.timelines.events@,
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
    let val = core.ledger@.submitted.map()[ev];
    lemma_iset_subset_closure_union(
        after_begin_core.ledger@.submitted.map(),
        old_tail,
        after_begin_core.citations(),
        after_begin_core.citations(),
        ev,
    );
    let put_proof = lemma_cuda_single_put_proof(
        after_begin_core.ledger@,
        core.ledger@,
        core.node(),
        src_cert,
        src_slot,
        h_slot,
        ev,
        val,
    );
    lemma_cuda_single_event_covered(
        put_proof.events,
        core.timelines.events@,
        timeline_id,
        ticket.sequence(),
        ev,
    );
    lemma_cuda_events_preserved_insert(
        before_core.timelines.events@,
        core.timelines.events@,
        timeline_id,
        before_core.timelines.next_seq[timeline_id as int],
        timeline_id,
        ticket.sequence(),
        ev,
    );
    lemma_cuda_ledger_writes_only_insert(before_core.ledger@, core.ledger@, ev, val, h_slot.loc);
    assert(core.frames.preserves_initializing(&before_core.frames, Some(h_frame))) by {
        reveal(FrameTable::preserves_initializing);
    }
    assert(core.frames.versions@ == before_core.frames.versions@) by {
        reveal(FrameTable::same_directory);
    }
    assert(before_core.frames.frame(h_frame).slot == h_slot) by {
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
    lemma_cuda_ticket_covers_exact(&core.timelines, ticket, ev);
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
    timeline: u8,
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
    timeline: u8,
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
