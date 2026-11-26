use super::*;
#[cfg(verus_only)]
use crate::backends::traits::host_backend_layout;
#[cfg(verus_only)]
use transfer::nvme::DIRECT_IO_ALIGNMENT;
#[cfg(verus_only)]
use transfer_core::memory::HostDmaSlab;

verus! {

impl NvmeBackend {
    pub(super) fn stage_ring_io<H: HostSlabBackend>(
        &mut self,
        host_backend: &H,
        core: &CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        kind: NvmeOpKind,
        seq: u64,
        hop: &CopyHopHandle,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: Result<Tracked<Submitted>, nvme::NvmeError>)
        requires
            core.well_formed(),
            old(self).file.well_formed(),
            old(self).file.instance_id() == core.instance_id(),
            old(self).file.node() == core.node(),
            old(self).file.num_slots() == old(self).num_frames as usize,
            old(self).file.slot_len() == core.frames_spec().tier_frame_size(old(self).tier) as u64,
            0 < old(self).file.slot_len() <= MAX_IO_LEN,
            old(self).file.slot_len() % DIRECT_IO_ALIGNMENT == 0,
            old(self).ring.well_formed(),
            old(self).ring.instance_id() == core.instance_id(),
            old(self).ring.node() == core.node(),
            old(self).ring.staged_count() < old(self).ring.sq_capacity(),
            old(self).ring.pending_count() < old(self).ring.max_in_flight(),
            !old(self).ring.pending().contains_key(seq),
            host_backend_layout(host_backend, core),
            host_backend.host_alignment_spec() % (DIRECT_IO_ALIGNMENT as usize) == 0,
            instance.id() == core.instance_id(),
            slab_bound_to_core(
                core,
                old(self).tier,
                old(self).first_frame,
                old(self).num_frames,
                old(self).file.space(),
            ),
            match kind {
                NvmeOpKind::DramToNvme => host_backend.host_owns(hop.src_frame) && old(self).owns(
                    hop.dst_frame,
                ),
                NvmeOpKind::NvmeToDram => old(self).owns(hop.src_frame) && host_backend.host_owns(
                    hop.dst_frame,
                ),
            },
            hop.src_frame.0 < core.frames_spec().metas_spec().len(),
            hop.dst_frame.0 < core.frames_spec().metas_spec().len(),
            hop.src_slot@ == core.frames_spec().frame(hop.src_frame).slot,
            hop.dst_slot@ == core.frames_spec().frame(hop.dst_frame).slot,
            hop.src_slot@.slot_len == hop.dst_slot@.slot_len,
            hop.src_slot@.loc != hop.dst_slot@.loc,
            old(src_perm).valid_cooperative(core.instance_id(), hop.src_slot@.loc),
            slot_is_owned(*old(dst_tok), core.instance_id(), hop.dst_slot@.loc),
            read_admitted(old(src_perm).frontier(), core.citations()),
            write_admitted(slot_frontier(*old(dst_tok)), core.citations()),
        ensures
            final(self).file == old(self).file,
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(self).bitmap == old(self).bitmap,
            final(self).in_flight_ops == old(self).in_flight_ops,
            final(self).pending_transfers == old(self).pending_transfers,
            final(self).ring.stage_result(
                old(self).ring,
                hop.src_slot@,
                hop.dst_slot@,
                seq,
                *old(src_perm),
                *final(src_perm),
                *old(dst_tok),
                *final(dst_tok),
                core.ledger_spec()@.submitted,
                core.citations(),
                r,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let _ = core;
        proof {
            lemma_core_wf_facts(core);
            match kind {
                NvmeOpKind::DramToNvme => {
                    lemma_slab_frame_slot(
                        core,
                        host_backend.host_tier_spec(),
                        host_backend.host_first_frame_spec(),
                        host_backend.host_num_frames_spec(),
                        host_backend.host_buffer_spec().slab_space(),
                        hop.src_frame,
                    );
                    lemma_slab_frame_slot(
                        core,
                        self.tier,
                        self.first_frame,
                        self.num_frames,
                        self.file.space(),
                        hop.dst_frame,
                    );
                },
                NvmeOpKind::NvmeToDram => {
                    lemma_slab_frame_slot(
                        core,
                        self.tier,
                        self.first_frame,
                        self.num_frames,
                        self.file.space(),
                        hop.src_frame,
                    );
                    lemma_slab_frame_slot(
                        core,
                        host_backend.host_tier_spec(),
                        host_backend.host_first_frame_spec(),
                        host_backend.host_num_frames_spec(),
                        host_backend.host_buffer_spec().slab_space(),
                        hop.dst_frame,
                    );
                },
            }
            lemma_citations_closure(core);
            core.lemma_citations_admit_closure(src_perm.frontier(), ISet::empty());
            core.lemma_citations_admit_closure(slot_frontier(*dst_tok), ISet::empty());
            assert(ISet::empty().union(core.citations()) =~= core.citations());
        }
        match kind {
            NvmeOpKind::DramToNvme => {
                let src_slice = host_backend.host_frame_slice(hop.src_frame);
                proof {
                    assert(host_backend.host_buffer_spec().slab_slot_len() as u64
                        == self.file.slot_len());
                    lemma_aligned_host_slot(
                        host_backend.host_buffer_spec().slab_address(),
                        host_backend.host_alignment_spec(),
                        (hop.src_frame.0 - host_backend.host_first_frame_spec().0) as usize,
                        host_backend.host_buffer_spec().slab_slot_len(),
                    );
                }
                let dst_slice = self.file.slot_slice(
                    (hop.dst_frame.0 - self.first_frame.0) as usize,
                );
                self.ring.stage_write(
                    Tracked(instance),
                    &dst_slice,
                    &src_slice,
                    seq,
                    Tracked(src_perm),
                    Tracked(dst_tok),
                    Tracked(&core.borrow_ledger().submitted),
                    Tracked(&core.borrow_ledger().knows),
                    Ghost(core.citations()),
                )
            },
            NvmeOpKind::NvmeToDram => {
                let src_slice = self.file.slot_slice(
                    (hop.src_frame.0 - self.first_frame.0) as usize,
                );
                let dst_slice = host_backend.host_frame_slice(hop.dst_frame);
                proof {
                    assert(host_backend.host_buffer_spec().slab_slot_len() as u64
                        == self.file.slot_len());
                    lemma_aligned_host_slot(
                        host_backend.host_buffer_spec().slab_address(),
                        host_backend.host_alignment_spec(),
                        (hop.dst_frame.0 - host_backend.host_first_frame_spec().0) as usize,
                        host_backend.host_buffer_spec().slab_slot_len(),
                    );
                }
                self.ring.stage_read(
                    Tracked(instance),
                    &dst_slice,
                    &src_slice,
                    seq,
                    Tracked(src_perm),
                    Tracked(dst_tok),
                    Tracked(&core.borrow_ledger().submitted),
                    Tracked(&core.borrow_ledger().knows),
                    Ghost(core.citations()),
                )
            },
        }
    }

    pub(super) fn record_staged_nvme_op(
        &mut self,
        core: &mut CacheCore,
        kind: NvmeOpKind,
        seq: u64,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Tracked(receipt): Tracked<Submitted>,
        Ghost(before_ring): Ghost<nvme::IoUring>,
        Ghost(before_hop_core): Ghost<CacheCore>,
    ) -> (r: (FrameId, Ticket))
        requires
            before_hop_core.well_formed(),
            old(core).well_formed(),
            old(self).file.well_formed(),
            old(self).file.instance_id() == old(core).instance_id(),
            old(self).file.node() == old(core).node(),
            old(self).file.num_slots() == old(self).num_frames as usize,
            old(self).file.slot_len() == old(core).frames_spec().tier_frame_size(
                old(self).tier,
            ) as u64,
            0 < old(self).file.slot_len() <= MAX_IO_LEN,
            old(self).file.slot_len() % DIRECT_IO_ALIGNMENT == 0,
            before_ring.well_formed(),
            before_ring.source().source().is_io_uring(),
            old(self).ring.well_formed(),
            old(self).ring.same_ring(before_ring),
            old(self).ring.instance_id() == old(core).instance_id(),
            old(self).ring.node() == old(core).node(),
            old(self).ring.source().source() == before_ring.source().source(),
            old(self).ring.source().node() == before_ring.source().node(),
            old(self).ring.source().next_seq() == before_ring.source().next_seq() + 1,
            old(self).ring.pending() == before_ring.pending().insert(
                seq,
                old(self).ring.pending()[seq],
            ),
            old(self).ring.pending()[seq].event == before_ring.source().next_event(),
            old(self).tier.valid(old(core).tiers_spec().len()),
            old(core).tiers_spec()[old(self).tier.index() as int].kind == TierKind::LocalNvme,
            old(self).timeline < old(core).timelines_spec().count_spec() <= MAX_TIMELINES,
            old(core).timelines_spec().kinds_spec()[old(self).timeline as int]
                == TimelineKind::IoUring,
            slab_bound_to_core(
                old(core),
                old(self).tier,
                old(self).first_frame,
                old(self).num_frames,
                old(self).file.space(),
            ),
            old(self).pending_transfers.len() == BITMAP_CAPACITY,
            nvme_window_wf(
                old(self).bitmap,
                old(self).in_flight_ops,
                before_ring,
                &before_hop_core,
                old(self).timeline,
            ),
            nvme_pending_valid(
                old(self).pending_transfers@,
                old(self).in_flight_ops,
                before_ring,
                &before_hop_core,
                old(self).timeline,
            ),
            seq == old(core).timelines_spec().next_seq_spec()[old(self).timeline as int],
            seq <= MAX_TICKET_SEQ,
            seq <= old(core).timelines_spec().confirmed_seq_spec()[old(self).timeline as int]
                + BITMAP_CAPACITY,
            old(core).node_id_spec() == before_hop_core.node_id_spec(),
            old(core).tiers_spec() == before_hop_core.tiers_spec(),
            old(core).ledger_spec() == before_hop_core.ledger_spec(),
            old(core).spec_spec() == before_hop_core.spec_spec(),
            old(core).version_registry_spec() == before_hop_core.version_registry_spec(),
            old(core).next_version_seq_spec() == before_hop_core.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before_hop_core.version_seq_limit_spec(),
            old(core).timelines_spec().only_watermarks(&before_hop_core.timelines_spec()),
            old(core).frames_spec().same_directory(&before_hop_core.frames_spec()),
            old(core).frames_spec().preserves_initializing(&before_hop_core.frames_spec(), None),
            old(core).frames_spec().preserves_fenced_leases(&before_hop_core.frames_spec()),
            before_hop_core.frames_spec().meta(hop.dst_frame).state != STATE_INITIALIZING,
            (hop.src_lease.slot_spec() as usize) < before_hop_core.active_leases_spec().len(),
            before_hop_core.active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
            old(core).active_leases_spec().len() == before_hop_core.active_leases_spec().len(),
            forall|s: int|
                0 <= s < before_hop_core.active_leases_spec().len() && s
                    != hop.src_lease.slot_spec() as int ==> #[trigger] old(
                    core,
                ).active_leases_spec()[s] == before_hop_core.active_leases_spec()[s],
            hop.src_frame != hop.dst_frame,
            hop.src_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.src_slot@ == old(core).frames_spec().frame(hop.src_frame).slot,
            hop.dst_slot@ == old(core).frames_spec().frame(hop.dst_frame).slot,
            hop.src_slot@.slot_len == hop.dst_slot@.slot_len,
            hop.src_slot@.loc != hop.dst_slot@.loc,
            old(core).frames_spec().meta(hop.src_frame).state == STATE_PUBLISHED,
            old(core).frames_spec().frame(hop.src_frame).cert == Some(hop.src_cert@),
            replica_valid(old(core).ledger_spec()@, old(core).node(), hop.src_cert@),
            old(core).citations().contains(hop.src_cert@.writer),
            old(core).frames_spec().versions_spec()@.contains_key(hop.version),
            old(core).frames_spec().versions_spec()@[hop.version] == hop.src_cert@.bytes,
            hop.version.origin_node != old(core).node_id_spec() || hop.version.seq < old(
                core,
            ).next_version_seq_spec() || hop.version.seq >= old(core).version_seq_limit_spec(),
            hop.src_lease.frame_spec() == hop.src_frame,
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.internal,
            src_cap.share() == old(
                core,
            ).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@,
            old(core).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(hop.dst_frame).queued(),
            old(core).frames_spec().frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames_spec().frame(hop.dst_frame).bound_key == Some(hop.key),
            old(core).frames_spec().frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames_spec().frame(hop.dst_frame).token is None,
            src_cap.is_cooperative(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == hop.src_slot@.loc,
            src_cap.frontier().readers == ISet::empty().insert(before_ring.source().next_event()),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                hop.dst_slot@.loc,
                Frontier {
                    last_writer: Some(before_ring.source().next_event()),
                    readers: ISet::empty(),
                },
            ),
            receipt.instance_id() == old(core).instance_id(),
            receipt.key() == before_ring.source().next_event(),
            receipt.value().access == copy_access(hop.src_slot@, hop.dst_slot@),
            receipt.value().effect == copy_effect(hop.src_slot@, hop.dst_slot@),
            receipt.value().preds == foundation::tracked::closure_from(
                old(core).ledger_spec()@.submitted.map(),
                old(core).citations(),
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == before_hop_core.version_registry_spec(),
            core_transition(
                &before_hop_core,
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline),
                    Some(before_ring.source().source()),
                ),
            ),
            r.0 == hop.dst_frame,
            r.1 != Ticket::NONE,
            r.1.timeline_id() == old(self).timeline,
            r.1.sequence() == seq,
            final(core).timelines_spec().valid_ticket(r.1),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost writer = before_ring.source().next_event();
        let ghost before_ops = self.in_flight_ops;
        let ghost before_transfers = self.pending_transfers@;
        let ghost sub_val = receipt.value();
        let data_op = match kind {
            NvmeOpKind::DramToNvme => DataOperation::Write,
            NvmeOpKind::NvmeToDram => DataOperation::Read,
        };
        let dst_frame = hop.dst_frame;
        let ghost before_submit_core = *core;
        let _ = core;
        proof {
            lemma_core_wf_facts(core);
            lemma_published_frame_slot_valid(core, hop.src_frame);
            lemma_initializing_frame_slot_valid(core, hop.dst_frame);
            lemma_nvme_stage_unfold_window(
                self.bitmap,
                self.in_flight_ops,
                self.pending_transfers@,
                before_ring,
                &before_hop_core,
                self.timeline,
                seq,
            );
            lemma_nvme_ring_pending_insert(before_ring, self.ring, seq, writer);
            lemma_core_timelines_wf(core);
        }
        let ticket = core.record_submission(
            &mut self.in_flight_ops,
            Ghost(self.bitmap),
            self.timeline,
            dst_frame,
            data_op,
            Ghost(ring_event_map(self.ring)),
            Tracked(receipt),
        );
        let ghost copy;
        proof {
            lemma_citations_closure(&before_submit_core);
            lemma_nvme_preserve_core_after_submission(
                &before_submit_core,
                core,
                dst_frame,
                self.timeline,
                seq,
                writer,
                sub_val,
            );
            lemma_slab_bound_same_frames(
                &before_submit_core,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
            );
            copy =
            lemma_nvme_stage_copy_retained(
                before_submit_core.ledger_spec()@,
                core.ledger_spec()@,
                core.node(),
                hop.src_cert@,
                hop.src_slot@,
                hop.dst_slot@,
                writer,
                sub_val,
            );
        }
        let idx = (seq % BITMAP_CAPACITY) as usize;
        let op = InFlightNvmeOp {
            kind,
            hop,
            ticket,
            src_cap: Tracked(src_cap),
            dst_tok: Tracked(dst_tok),
            copy: Ghost(copy),
        };
        self.pending_transfers.set(idx, Some(op));
        proof {
            assert(self.pending_transfers[idx as int]->Some_0.valid(
                core,
                self.timeline,
                idx as int,
            ));
            lemma_nvme_pack_window(self.bitmap, self.in_flight_ops, self.ring, core, self.timeline);
            lemma_nvme_pending_preserved_after_stage(
                before_transfers,
                self.pending_transfers@,
                before_ops,
                self.in_flight_ops,
                before_ring,
                self.ring,
                &before_hop_core,
                &before_submit_core,
                core,
                self.timeline,
                idx as int,
                self.pending_transfers[idx as int]->Some_0,
                writer,
                sub_val,
            );
            lemma_core_frames_separated(&before_hop_core);
            assert(core.frames_spec().same_layout(&before_hop_core.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            assert(core.frames_spec().preserves_initializing(&before_hop_core.frames_spec(), None))
                by {
                reveal(FrameTable::preserves_initializing);
            }
            assert forall|f: FrameId|
                f.0 < before_hop_core.frames_spec().metas_spec().len() && (
                #[trigger] before_hop_core.frames_spec().meta(f)).state
                    == STATE_INITIALIZING implies hop.dst_slot@.loc
                != before_hop_core.frames_spec().frame(f).slot.loc by {
                assert(dst_frame != f);
                if dst_frame.0 < f.0 {
                    assert(before_hop_core.frames_spec().ghosts_spec()@[dst_frame.0 as nat].slot.loc
                        != before_hop_core.frames_spec().ghosts_spec()@[f.0 as nat].slot.loc);
                } else {
                    assert(before_hop_core.frames_spec().ghosts_spec()@[f.0 as nat].slot.loc
                        != before_hop_core.frames_spec().ghosts_spec()@[dst_frame.0 as nat].slot.loc);
                }
            }
            assert(core.ledger_spec()@.submitted.dom()
                =~= before_hop_core.ledger_spec()@.submitted.dom().insert(writer));
            assert(core.ledger_spec()@.submitted.map()[writer].access.writes
                =~= ISet::empty().insert(hop.dst_slot@.loc));
            lemma_core_transition_single_event(
                &before_hop_core,
                core,
                Some(self.timeline),
                before_ring.source().source(),
                writer,
                Some(hop.dst_slot@.loc),
                None,
            );
        }
        (dst_frame, ticket)
    }
}

} // verus!
