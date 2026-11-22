use super::super::host::HostBackend;
use super::*;
#[cfg(verus_only)]
use transfer::nvme::DIRECT_IO_ALIGNMENT;

verus! {

impl NvmeBackend {
    fn commit_staged_nvme_op(
        &mut self,
        host_backend: &HostBackend,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        kind: NvmeOpKind,
        seq: u64,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Ghost(before_hop_core): Ghost<CacheCore>,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
        requires
            before_hop_core.well_formed(),
            old(core).well_formed(),
            old(self).well_formed(&before_hop_core),
            host_backend.well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            host_backend.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0,
            old(self).ring.staged_count() < old(self).ring.sq_capacity(),
            old(self).ring.pending_count() < old(self).ring.max_in_flight(),
            seq == old(core).timelines.next_seq[old(self).timeline as int],
            seq <= MAX_TICKET_SEQ,
            seq <= old(core).timelines.confirmed_seq[old(self).timeline as int] + BITMAP_CAPACITY,
            old(core).node_id == before_hop_core.node_id,
            old(core).tiers == before_hop_core.tiers,
            old(core).ledger == before_hop_core.ledger,
            old(core).spec == before_hop_core.spec,
            old(core).version_registry == before_hop_core.version_registry,
            old(core).next_version_seq == before_hop_core.next_version_seq,
            old(core).version_seq_limit == before_hop_core.version_seq_limit,
            old(core).timelines.only_watermarks(&before_hop_core.timelines),
            old(core).frames.same_directory(&before_hop_core.frames),
            old(core).frames.preserves_initializing(&before_hop_core.frames, None),
            old(core).frames.preserves_remote_leases(&before_hop_core.frames),
            before_hop_core.frames.meta(hop.dst_frame).state != STATE_INITIALIZING,
            (hop.src_lease.slot as usize) < before_hop_core.active_leases.len(),
            before_hop_core.active_leases[hop.src_lease.slot as int].is_none(),
            old(core).active_leases.len() == before_hop_core.active_leases.len(),
            forall|s: int|
                0 <= s < before_hop_core.active_leases.len() && s != hop.src_lease.slot as int
                    ==> #[trigger] old(core).active_leases[s] == before_hop_core.active_leases[s],
            match kind {
                NvmeOpKind::DramToNvme => host_backend.owns(hop.src_frame) && old(self).owns(
                    hop.dst_frame,
                ) && hop.dst_tier == old(self).tier,
                NvmeOpKind::NvmeToDram => old(self).owns(hop.src_frame) && host_backend.owns(
                    hop.dst_frame,
                ) && hop.dst_tier == host_backend.tier,
            },
            hop.src_frame != hop.dst_frame,
            hop.src_frame.0 < old(core).frames.metas.len(),
            hop.dst_frame.0 < old(core).frames.metas.len(),
            hop.src_slot@ == old(core).frames.frame(hop.src_frame).slot,
            hop.dst_slot@ == old(core).frames.frame(hop.dst_frame).slot,
            hop.src_slot@.loc != hop.dst_slot@.loc,
            old(core).frames.meta(hop.src_frame).state == STATE_PUBLISHED,
            old(core).frames.frame(hop.src_frame).cert == Some(hop.src_cert@),
            replica_valid(old(core).ledger@, old(core).node(), hop.src_cert@),
            old(core).citations().contains(hop.src_cert@.writer),
            old(core).frames.versions@.contains_key(hop.version),
            old(core).frames.versions@[hop.version] == hop.src_cert@.bytes,
            hop.version.origin_node != old(core).node_id || hop.version.seq < old(
                core,
            ).next_version_seq || hop.version.seq >= old(core).version_seq_limit,
            hop.src_lease.frame == hop.src_frame,
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases[hop.src_lease.slot as int]->Some_0.internal,
            src_cap.share() == old(core).active_leases[hop.src_lease.slot as int]->Some_0.share@,
            old(core).frames.meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(hop.dst_frame).queued(),
            old(core).frames.frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames.frame(hop.dst_frame).bound_key == Some(hop.key),
            old(core).frames.frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames.frame(hop.dst_frame).token is None,
            src_cap.is_local(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == hop.src_slot@.loc,
            read_admitted(src_cap.frontier(), old(core).citations()),
            src_cap.frontier().readers.is_empty(),
            slot_is_owned(dst_tok, old(core).instance_id(), hop.dst_slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(core).citations()),
            old(core).frames.returned_reads(
                hop.src_frame,
                src_cap,
                Ticket::NONE,
                old(core).kernel_view(),
            ),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == before_hop_core.version_registry,
            core_transition(
                &before_hop_core,
                final(core),
                Some(old(self).timeline),
                Some(old(self).ring.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((dst_frame, ticket)) ==> dst_frame == hop.dst_frame && ticket
                != Ticket::NONE && ticket.timeline_id() == old(self).timeline && ticket.sequence()
                == seq && final(core).timelines.valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_ring = self.ring;
        let ghost before_ops = self.in_flight_ops;
        let ghost before_transfers = self.pending_transfers@;
        let ghost writer = self.ring.source().next_event();
        let tracked mut src_perm = ReadPerm::Shared(src_cap);
        let tracked mut dst_tok = dst_tok;
        proof {
            lemma_core_wf_facts(core);
            lemma_published_frame_slot_valid(core, hop.src_frame);
            lemma_initializing_frame_slot_valid(core, hop.dst_frame);
            lemma_slab_bound_same_directory(
                &before_hop_core,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
            );
            lemma_nvme_seq_not_pending(
                self.bitmap,
                self.in_flight_ops,
                self.ring,
                &before_hop_core,
                self.timeline,
                seq,
            );
        }
        let stage_res = self.stage_ring_io(
            host_backend,
            core,
            Tracked(instance),
            kind,
            seq,
            &hop,
            Tracked(&mut src_perm),
            Tracked(&mut dst_tok),
        );
        let tracked src_cap = src_perm.into_shared();
        let Tracked(receipt) = match stage_res {
            Ok(r) => r,
            Err(e) => {
                core.abort_copy_hop(
                    hop,
                    Tracked(src_cap),
                    Ticket::NONE,
                    Tracked(dst_tok),
                    Tracked(instance),
                );
                proof {
                    lemma_slab_bound_same_directory(
                        &before_hop_core,
                        core,
                        self.tier,
                        self.first_frame,
                        self.num_frames,
                        self.file.space(),
                    );
                    lemma_nvme_state_after_hop_rollback(
                        self.bitmap,
                        self.in_flight_ops,
                        self.pending_transfers@,
                        self.ring,
                        &before_hop_core,
                        core,
                        self.timeline,
                        Some(hop.dst_frame),
                        Some(hop.src_lease.slot),
                    );
                    assert(core.frames.same_layout(&before_hop_core.frames)) by {
                        reveal(FrameTable::same_layout);
                    }
                    assert(core.frames.preserves_initializing(&before_hop_core.frames, None)) by {
                        reveal(FrameTable::preserves_initializing);
                    }
                    assert(core.frames.preserves_remote_leases(&before_hop_core.frames)) by {
                        assert forall|f: FrameId| f.0 < before_hop_core.frames.metas.len() implies {
                            &&& (#[trigger] core.frames.frame(f)).remote_shares
                                == before_hop_core.frames.frame(f).remote_shares
                            &&& core.frames.meta(f).active_remote_leases
                                == before_hop_core.frames.meta(f).active_remote_leases
                        } by {
                            assert(old(core).frames.frame(f).remote_shares
                                == before_hop_core.frames.frame(f).remote_shares);
                            assert(old(core).frames.meta(f).active_remote_leases
                                == before_hop_core.frames.meta(f).active_remote_leases);
                        }
                    }
                    lemma_core_transition_no_event(
                        &before_hop_core,
                        core,
                        Some(self.timeline),
                        Some(before_ring.source().source()),
                    );
                }
                return Err(nvme_error(e));
            },
        };
        proof {
            assert(src_cap.frontier().readers =~= ISet::empty().insert(writer));
        }
        Ok(
            self.record_staged_nvme_op(
                core,
                kind,
                seq,
                hop,
                Tracked(src_cap),
                Tracked(dst_tok),
                Tracked(receipt),
                Ghost(before_ring),
                Ghost(before_hop_core),
            ),
        )
    }

    /// Stage an asynchronous DRAM -> NVMe tier-down copy hop via `io_uring`.
    pub fn stage_dram_to_nvme(
        &mut self,
        host_backend: &HostBackend,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            host_backend.well_formed(old(core)),
            host_backend.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline),
                Some(old(self).ring.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((dst_frame, ticket)) ==> final(self).owns(dst_frame) && ticket
                != Ticket::NONE && ticket.timeline_id() == old(self).timeline
                && final(core).timelines.valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if !host_backend.owns_frame(src_frame) {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline),
                    Some(self.ring.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        if self.ring.staged_len() >= self.ring.sq_capacity_val() || self.ring.pending_len()
            >= self.ring.max_in_flight_val() {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline),
                    Some(self.ring.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::CapacityExhausted);
        }
        proof {
            lemma_core_wf_facts(core);
        }
        let seq = match core.timelines.next_data_id(self.timeline) {
            Ok(s) => s,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(self.timeline),
                        Some(self.ring.source().source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        let ghost before_hop_core = *core;
        let (hop, src_cap, dst_tok) = match core.begin_copy_hop(
            src_frame,
            self.tier,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_slab_bound_same_directory(
                        old(core),
                        core,
                        self.tier,
                        self.first_frame,
                        self.num_frames,
                        self.file.space(),
                    );
                    lemma_nvme_state_after_hop_rollback(
                        self.bitmap,
                        self.in_flight_ops,
                        self.pending_transfers@,
                        self.ring,
                        old(core),
                        core,
                        self.timeline,
                        None,
                        None,
                    );
                    assert(core.frames.same_layout(&old(core).frames)) by {
                        reveal(FrameTable::same_layout);
                    }
                    lemma_core_transition_no_event(
                        old(core),
                        core,
                        Some(self.timeline),
                        Some(self.ring.source().source()),
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_slab_bound_same_directory(
                old(core),
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
            );
            lemma_slab_bound_same_directory(
                old(core),
                core,
                host_backend.tier,
                host_backend.first_frame,
                host_backend.num_frames,
                host_backend.buffer.space(),
            );
            lemma_slab_owns_tier_frame(
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
                hop.dst_frame,
            );
        }
        self.commit_staged_nvme_op(
            host_backend,
            core,
            Tracked(instance),
            NvmeOpKind::DramToNvme,
            seq,
            hop,
            src_cap,
            dst_tok,
            Ghost(before_hop_core),
        )
    }

    /// Stage an asynchronous NVMe -> DRAM warm-up / read copy hop via `io_uring`.
    pub fn stage_nvme_to_dram(
        &mut self,
        host_backend: &HostBackend,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            host_backend.well_formed(old(core)),
            host_backend.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            core_transition(
                old(core),
                final(core),
                Some(old(self).timeline),
                Some(old(self).ring.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((dst_frame, ticket)) ==> host_backend.owns(dst_frame) && ticket
                != Ticket::NONE && ticket.timeline_id() == old(self).timeline
                && final(core).timelines.valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if !self.owns_frame(src_frame) {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline),
                    Some(self.ring.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        if self.ring.staged_len() >= self.ring.sq_capacity_val() || self.ring.pending_len()
            >= self.ring.max_in_flight_val() {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline),
                    Some(self.ring.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::CapacityExhausted);
        }
        proof {
            lemma_core_wf_facts(core);
        }
        let seq = match core.timelines.next_data_id(self.timeline) {
            Ok(s) => s,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(self.timeline),
                        Some(self.ring.source().source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        let ghost before_hop_core = *core;
        let (hop, src_cap, dst_tok) = match core.begin_copy_hop(
            src_frame,
            host_backend.tier,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_slab_bound_same_directory(
                        old(core),
                        core,
                        self.tier,
                        self.first_frame,
                        self.num_frames,
                        self.file.space(),
                    );
                    lemma_nvme_state_after_hop_rollback(
                        self.bitmap,
                        self.in_flight_ops,
                        self.pending_transfers@,
                        self.ring,
                        old(core),
                        core,
                        self.timeline,
                        None,
                        None,
                    );
                    assert(core.frames.same_layout(&old(core).frames)) by {
                        reveal(FrameTable::same_layout);
                    }
                    lemma_core_transition_no_event(
                        old(core),
                        core,
                        Some(self.timeline),
                        Some(self.ring.source().source()),
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_slab_bound_same_directory(
                old(core),
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.file.space(),
            );
            lemma_slab_bound_same_directory(
                old(core),
                core,
                host_backend.tier,
                host_backend.first_frame,
                host_backend.num_frames,
                host_backend.buffer.space(),
            );
            lemma_slab_owns_tier_frame(
                core,
                host_backend.tier,
                host_backend.first_frame,
                host_backend.num_frames,
                host_backend.buffer.space(),
                hop.dst_frame,
            );
        }
        self.commit_staged_nvme_op(
            host_backend,
            core,
            Tracked(instance),
            NvmeOpKind::NvmeToDram,
            seq,
            hop,
            src_cap,
            dst_tok,
            Ghost(before_hop_core),
        )
    }
}

} // verus!
