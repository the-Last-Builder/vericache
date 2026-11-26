use super::*;
#[cfg(verus_only)]
use foundation::tracked::{closure_from, Frontier};

verus! {

#[verifier::opaque]
pub(super) open spec fn put_gpu_enqueue_ready<'a>(
    b: &CudaBackend<'a>,
    core: &CacheCore,
    before_core: &CacheCore,
    instance: &TransferInstance,
    handle: WriteHandle,
    is_gpu: bool,
    src: &CudaSlice,
    src_perm: ReadPerm,
    dst_tok: SlotToken,
    src_cert: ValueCertificate,
) -> bool {
    &&& instance.id() == core.instance_id()
    &&& core.node_id_spec() == before_core.node_id_spec()
    &&& core.instance_id() == before_core.instance_id()
    &&& core.tiers_spec() == before_core.tiers_spec()
    &&& core.spec_spec() == before_core.spec_spec()
    &&& core.next_version_seq_spec() == before_core.next_version_seq_spec() + 1
    &&& core.version_seq_limit_spec() == before_core.version_seq_limit_spec()
    &&& core.version_registry_spec() == before_core.version_registry_spec()
    &&& core.timelines_spec() == before_core.timelines_spec()
    &&& core.ledger_spec() == before_core.ledger_spec()
    &&& core.frames_spec().same_directory(&before_core.frames_spec())
    &&& core.frames_spec().preserves_initializing(&before_core.frames_spec(), None)
    &&& handle.frame.0 < before_core.frames_spec().metas_spec().len()
    &&& before_core.frames_spec().meta(handle.frame).state != STATE_INITIALIZING
    &&& handle.frame.0 < core.frames_spec().metas_spec().len()
    &&& core.frames_spec().meta(handle.frame).state == STATE_INITIALIZING
    &&& !core.frames_spec().meta(handle.frame).queued()
    &&& core.frames_spec().frame(handle.frame).token is None
    &&& core.frames_spec().frame(handle.frame).tier == handle.tier
    &&& core.frames_spec().frame(handle.frame).slot == handle.slot@
    &&& core.frames_spec().frame(handle.frame).bound_key == Some(handle.key)
    &&& core.frames_spec().frame(handle.frame).bound_version == Some(handle.version)
    &&& handle.version.well_formed() && handle.version != VersionId::NONE
    &&& handle.version.origin_node == before_core.node_id_spec()
    &&& handle.version.seq == before_core.next_version_seq_spec()
    &&& !before_core.frames_spec().versions_spec()@.contains_key(handle.version)
    &&& handle.slot@.is_full()
    &&& handle.slot@.well_formed()
    &&& handle.slot@.slot_len == core.frames_spec().tier_frame_size(b.gpu_tier)
    &&& handle.slot@.loc.node == core.node()
    &&& src.well_formed()
    &&& src.instance_id() == core.instance_id()
    &&& src.slot_ref().loc.node == core.node()
    &&& src.slot_ref().sub_len == core.frames_spec().tier_frame_size(b.gpu_tier)
    &&& src.slot_ref().loc != handle.slot@.loc
    &&& src_perm.valid_cooperative(core.instance_id(), src.slot_ref().loc)
    &&& (src.slot_ref().loc.space.is_gpu() && src.slot_ref().loc.space.gpu_device() != b.device
        ==> (src_perm.frontier().last_writer matches Some(w) ==> !(w.source.is_rdma_qp())))
    &&& slot_is_owned(dst_tok, core.instance_id(), handle.slot@.loc)
    &&& write_admitted(slot_frontier(dst_tok), core.citations())
    &&& read_admitted(
        src_perm.frontier(),
        closure_from(
            core.ledger_spec()@.submitted.map(),
            b.timeline.stream.tail().union(core.citations()),
        ),
    )
    &&& write_admitted(
        slot_frontier(dst_tok),
        closure_from(
            core.ledger_spec()@.submitted.map(),
            b.timeline.stream.tail().union(core.citations()),
        ),
    )
    &&& replica_valid(core.ledger_spec()@, core.node(), src_cert)
    &&& source_slice_valid(src_cert, src.slot_ref())
    &&& src_cert.slot == src.slot_ref()
    &&& src_cert.bytes.len() == src.slot_ref().sub_len
    &&& core.disjoint_from_published(ISet::empty().insert(handle.slot@.loc))
    &&& b.gpu_buffer.space().is_gpu()
    &&& b.gpu_buffer.space().gpu_device() == b.device
    &&& (is_gpu ==> b.gpu_tier == handle.tier && b.owns_gpu(handle.frame) && b.gpu_buffer.slot_ref(
        (handle.frame.0 - b.gpu_first_frame.0) as nat,
    ) == handle.slot@)
    &&& (!is_gpu ==> (b.pinned_host matches Some(h) && handle.tier == h.tier && h.owns(handle.frame)
        && h.reg.slot_ref((handle.frame.0 - h.first_frame.0) as nat) == handle.slot@))
}

impl<'a> CudaBackend<'a> {
    /// Submit DMA for `put_gpu` + construct `PendingPutProof`, or cleanly abort
    /// the reserved initializing frame if DMA submission fails.
    pub(super) fn enqueue_or_abort_put_gpu(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        handle: WriteHandle,
        is_gpu: bool,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Ghost(src_cert): Ghost<ValueCertificate>,
        Ghost(before_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(Ticket, WriteHandle, Tracked<SlotToken>, Ghost<PendingPutProof>)>)
        requires
            old(core).well_formed(),
            before_core.well_formed(),
            cuda_backend_pre_park_wf(old(self), old(core), &before_core),
            core_transition(
                &before_core,
                old(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline.timeline_id),
                    Some(old(self).timeline.stream.source().source()),
                ),
            ),
            put_gpu_enqueue_ready(
                old(self),
                old(core),
                &before_core,
                instance,
                handle,
                is_gpu,
                src,
                *old(src_perm),
                dst_tok,
                src_cert,
            ),
        ensures
            match r {
                Err(_) => {
                    &&& final(core).well_formed()
                    &&& final(self).well_formed(final(core))
                    &&& final(self).device == old(self).device
                    &&& final(self).gpu_tier == old(self).gpu_tier
                    &&& final(self).gpu_first_frame == old(self).gpu_first_frame
                    &&& final(self).gpu_num_frames == old(self).gpu_num_frames
                    &&& final(self).same_pinned_layout(old(self))
                    &&& final(self).active_gpu_puts == old(self).active_gpu_puts
                    &&& final(self).timeline.timeline_id == old(self).timeline.timeline_id
                    &&& final(core).node_id_spec() == old(core).node_id_spec()
                    &&& final(core).instance_id() == old(core).instance_id()
                    &&& final(core).tiers_spec() == old(core).tiers_spec()
                    &&& final(core).spec_spec() == old(core).spec_spec()
                    &&& final(core).version_registry_spec() == before_core.version_registry_spec()
                    &&& core_transition(
                        &before_core,
                        final(core),
                        CoreFootprint::preserve_all(
                            Some(old(self).timeline.timeline_id),
                            Some(old(self).timeline.stream.source().source()),
                        ),
                    )
                },
                Ok((ticket, h_out, tok, put_proof)) => {
                    &&& h_out == handle
                    &&& put_proof@.certificate.bytes == src_cert.bytes
                    &&& final(core).well_formed()
                    &&& final(self).timeline.well_formed(
                        &final(core).timelines_spec(),
                        final(core).node(),
                        &final(core).ledger_spec()@,
                    )
                    &&& core_transition(
                        &before_core,
                        final(core),
                        CoreFootprint::preserve_all(
                            Some(old(self).timeline.timeline_id),
                            Some(old(self).timeline.stream.source().source()),
                        ),
                    )
                    &&& put_gpu_committed_ready(
                        final(self),
                        old(self),
                        final(core),
                        old(core),
                        &before_core,
                        h_out,
                        tok@,
                        ticket,
                        put_proof@,
                    )
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(CacheCore::disjoint_from_published);
        hide(TimelineTable::ticket_covers);
        hide(CudaBackend::well_formed);
        hide(PinnedHostSlab::well_formed);
        hide(cuda_backend_pre_park_wf);
        hide(closure_from);
        hide(source_slice_valid);
        let ghost after_begin_self = *self;
        let ghost after_begin_core = *core;
        let tracked mut dst_tok = dst_tok;
        proof {
            reveal(put_gpu_enqueue_ready);
            assert(self.gpu_buffer.well_formed() && self.gpu_buffer.instance_id()
                == core.instance_id() && self.gpu_buffer.node() == core.node()
                && self.gpu_buffer.num_slots() == self.gpu_num_frames as usize
                && self.gpu_buffer.slot_len() == core.frames_spec().tier_frame_size(self.gpu_tier)
                && (self.pinned_host matches Some(h) ==> h.well_formed(core)
                && core.frames_spec().tier_frame_size(h.tier) == core.frames_spec().tier_frame_size(
                self.gpu_tier,
            )) && self.timeline.well_formed(
                &core.timelines_spec(),
                core.node(),
                &core.ledger_spec()@,
            )) by {
                reveal(cuda_backend_pre_park_wf);
            }
            lemma_core_wf_facts(core);
            assert(!core.frames_spec().versions_spec()@.contains_key(handle.version)) by {
                reveal(FrameTable::same_directory);
            }
        }
        let (ticket, Ghost(put_proof)) = match self.enqueue_put_memcpy_proved(
            core,
            Tracked(instance),
            handle.frame,
            is_gpu,
            handle.slot,
            src,
            Tracked(src_perm),
            Tracked(&mut dst_tok),
            Ghost(src_cert),
            Ghost(before_core),
        ) {
            Ok(pair) => pair,
            Err(e) => {
                self.abort_unsubmitted_gpu_put(core, handle, Tracked(dst_tok), Ghost(before_core));
                return Err(e);
            },
        };
        proof {
            reveal(put_gpu_committed_ready);
            assert(put_gpu_committed_ready(
                self,
                &after_begin_self,
                core,
                &after_begin_core,
                &before_core,
                handle,
                dst_tok,
                ticket,
                put_proof,
            ));
        }
        Ok((ticket, handle, Tracked(dst_tok), Ghost(put_proof)))
    }
}

} // verus!
