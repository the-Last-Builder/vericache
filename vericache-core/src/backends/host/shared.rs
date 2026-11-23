use super::*;
use transfer::host::{self, HostContext};

verus! {

impl HostBackend {
    fn write_and_certify_host_slot_with(
        &self,
        ctx: &mut HostContext,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        frame: FrameId,
        Ghost(slot): Ghost<SlotRef>,
        Tracked(tok): Tracked<&mut SlotToken>,
        data: &[u8],
    ) -> (r: (Ticket, Ghost<ValueCertificate>))
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            old(ctx).ready(old(core).instance_id(), old(core).node()),
            source_fresh_for_ledger(old(ctx).source@, old(core).ledger@),
            old(ctx).source@.source() != self.ctx.source@.source(),
            instance.id() == old(core).instance_id(),
            old(core).timelines.next_seq[self.timeline as int] <= MAX_TICKET_SEQ,
            data@.len() == old(core).frames.tier_frame_size(self.tier),
            self.owns(frame),
            frame.0 < old(core).frames.metas.len(),
            old(core).frames.meta(frame).state == STATE_INITIALIZING,
            old(core).frames.frame(frame).slot == slot,
            slot_is_owned(*old(tok), old(core).instance_id(), slot.loc),
            write_admitted(slot_frontier(*old(tok)), old(core).citations()),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            final(ctx).ready(final(core).instance_id(), final(core).node()),
            final(ctx).source@.source() == old(ctx).source@.source(),
            source_fresh_for_ledger(final(ctx).source@, final(core).ledger@),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            forall|t: u8|
                t < old(core).timelines.count && t != self.timeline ==> {
                    &&& (#[trigger] final(core).timelines.next_seq[t as int]) == old(
                        core,
                    ).timelines.next_seq[t as int]
                    &&& final(core).timelines.confirmed_seq[t as int] == old(
                        core,
                    ).timelines.confirmed_seq[t as int]
                    &&& forall|s: u64|
                        1 <= s < old(core).timelines.next_seq[t as int]
                            ==> #[trigger] final(core).timelines.events@[(t, s)] == old(
                            core,
                        ).timelines.events@[(t, s)]
                },
            old(core).ledger@.submitted.map().submap_of(final(core).ledger@.submitted.map()),
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            old(core).ledger@.succeeded.dom().subset_of(final(core).ledger@.succeeded.dom()),
            final(core).ledger@.submitted.dom() == old(core).ledger@.submitted.dom().insert(
                r.1@.writer,
            ),
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            r.1@.writer.source == old(ctx).source@.source(),
            final(core).ledger@.submitted.map()[r.1@.writer].access.writes == ISet::empty().insert(
                slot.loc,
            ),
            slot_is_owned(*final(tok), final(core).instance_id(), slot.loc),
            slot_frontier(*final(tok)).readers.is_empty(),
            slot_frontier(*final(tok)).last_writer == Some(r.1@.writer),
            r.1@.slot == slot,
            r.1@.bytes == data@,
            r.1@.bytes.len() == slot.slot_len,
            final(core).timelines.issued(r.0.timeline_id(), r.0.sequence()),
            r.0 != Ticket::NONE,
            r.1@.writer == final(core).timelines.events@[(r.0.timeline_id(), r.0.sequence())],
            replica_valid(final(core).ledger@, final(core).node(), r.1@),
            final(core).citations().contains(r.1@.writer),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_write = *core;
        let ghost before_ctx = *ctx;
        let dst = self.buffer.slot_slice((frame.0 - self.first_frame.0) as usize);
        let ghost writer = ctx.source@.next_event();
        proof {
            lemma_core_wf_facts(core);
            lemma_initializing_frame_slot_valid(core, frame);
            lemma_slab_frame_slot(
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
                frame,
            );
            assert(dst.slot_ref() == slot);
            lemma_citations_closure(core);
            lemma_fresh_source_event_absent(ctx.source@, core.ledger@);
            core.lemma_citations_admit_closure(slot_frontier(*tok), ISet::empty());
            assert(ISet::empty().union(core.citations()) =~= core.citations());
        }
        let Tracked(completion) = host::write_slice(
            ctx,
            Tracked(instance),
            &dst,
            data,
            Tracked(tok),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Ghost(core.citations()),
        );
        proof {
            lemma_write_disjoint_from_published(
                core,
                frame,
                completion.submitted.value().access.writes,
            );
        }
        let ghost sub_val = completion.submitted.value();
        let ghost knows_val = completion.knows;
        let ghost write_preds = sub_val.preds;
        let ticket = core.issue_ticket(self.timeline, Tracked(completion.submitted)).unwrap();
        proof {
            assert(core.timelines.events@[(self.timeline, ticket.sequence())] == writer);
            assert(core.timelines.observed(
                self.timeline,
                ticket.sequence(),
                core.node(),
                completion.knows,
            ));
        }
        core.mark_timeline_complete(self.timeline, ticket.seq(), Tracked(completion.knows));
        let ghost cert;
        proof {
            lemma_observed_event_in_citations(
                before_write.ledger@,
                core.ledger@,
                core.node(),
                writer,
                sub_val,
                knows_val,
                write_preds,
            );
            lemma_source_fresh_stepped(
                before_ctx.source@,
                ctx.source@,
                before_write.ledger@,
                core.ledger@,
                writer,
            );
            assert(source_fresh_for_ledger(self.ctx.source@, core.ledger@)) by {
                assert forall|e: TransferEvent| #[trigger]
                    core.ledger@.submitted.dom().contains(e) && e.source
                        == self.ctx.source@.source() && e.observer
                        == self.ctx.source@.node() implies e.seq < self.ctx.source@.next_seq() by {
                    assert(e != writer);
                    assert(before_write.ledger@.submitted.dom().contains(e));
                }
            }
            lemma_slab_bound_same_frames(
                old(core),
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            assert(core.ledger@.submitted.map()[writer].access.writes =~= ISet::empty().insert(
                slot.loc,
            ));
            cert =
            lemma_certify_host_write(
                before_write.ledger@,
                core.ledger@,
                core.node(),
                writer,
                slot,
                data@,
            );
        }
        (ticket, Ghost(cert))
    }

    /// Synchronously write `data` into a fresh Host-DRAM frame using an external
    /// `HostContext` (allowing `&self` to stay shared-borrowed by `CudaBackend`).
    pub fn put_host_with(
        &self,
        ctx: &mut HostContext,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        data: &[u8],
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            old(ctx).ready(old(core).instance_id(), old(core).node()),
            source_fresh_for_ledger(old(ctx).source@, old(core).ledger@),
            old(ctx).source@.source() != self.ctx.source@.source(),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            final(ctx).ready(final(core).instance_id(), final(core).node()),
            final(ctx).source@.source() == old(ctx).source@.source(),
            source_fresh_for_ledger(final(ctx).source@, final(core).ledger@),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(self.timeline),
                Some(old(ctx).source@.source()),
                FrameTouchMode::PreserveAllInit,
            ),
            match r {
                Err(_) => final(core).spec == old(core).spec,
                Ok((ver, frame)) => {
                    &&& self.owns(frame)
                    &&& ver.origin_node == old(core).node_id
                    &&& ver.seq == old(core).next_version_seq
                    &&& final(core).next_version_seq == old(core).next_version_seq + 1
                    &&& final(core).spec@.versions == old(core).spec@.versions.insert(ver, data@)
                    &&& final(core).spec@.catalog == old(core).spec@.catalog.insert(key, ver)
                    &&& final(core).catalog.slot(key, self.tier.index() as int) == Some(frame)
                    &&& final(core).frames.meta(frame).state == STATE_PUBLISHED
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if data.len() != core.tier_frame_size(self.tier) {
            proof {
                lemma_core_transition_refl(
                    core,
                    Some(self.timeline),
                    Some(ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::InvalidArgument);
        }
        let (handle, Tracked(mut tok)) = match core.begin_write(self.tier, key) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(self.timeline),
                        Some(ctx.source@.source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        proof {
            lemma_core_wf_facts(old(core));
            lemma_core_wf_facts(core);
            lemma_core_frames_separated(old(core));
            lemma_slab_bound_same_directory(
                old(core),
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            lemma_slab_owns_tier_frame(
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
                handle.frame,
            );
        }
        if core.timelines.next_seq[self.timeline as usize] > MAX_TICKET_SEQ {
            let ghost before_abort = *core;
            core.abort_write(handle, Tracked(tok));
            proof {
                lemma_slab_bound_same_directory(
                    old(core),
                    core,
                    self.tier,
                    self.first_frame,
                    self.num_frames,
                    self.buffer.space(),
                );
                assert(core.frames.preserves_remote_leases(&old(core).frames)) by {
                    assert forall|f: FrameId| f.0 < old(core).frames.metas.len() implies {
                        &&& (#[trigger] core.frames.frame(f)).remote_shares == old(
                            core,
                        ).frames.frame(f).remote_shares
                        &&& core.frames.meta(f).active_remote_leases == old(core).frames.meta(
                            f,
                        ).active_remote_leases
                    } by {
                        assert(before_abort.frames.frame(f).remote_shares == old(core).frames.frame(
                            f,
                        ).remote_shares);
                        assert(before_abort.frames.meta(f).active_remote_leases == old(
                            core,
                        ).frames.meta(f).active_remote_leases);
                    }
                }
                lemma_core_transition_no_event(
                    old(core),
                    core,
                    Some(self.timeline),
                    Some(ctx.source@.source()),
                );
            }
            return Err(CoreError::SequenceExhausted);
        }
        let ver = handle.version;
        let frame = handle.frame;
        let ghost slot = handle.slot@;
        let (ticket, cert) = self.write_and_certify_host_slot_with(
            ctx,
            core,
            Tracked(instance),
            frame,
            handle.slot,
            Tracked(&mut tok),
            data,
        );
        let ghost before_finish = *core;
        core.finish_write(handle, Tracked(tok), ticket, cert);
        proof {
            lemma_core_wf_facts(&before_finish);
            lemma_core_wf_facts(core);
            lemma_slab_bound_same_layout(
                &before_finish,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            assert(core.frames.same_layout(&old(core).frames)) by {
                reveal(FrameTable::same_layout);
            }
            assert(core.frames.preserves_initializing(&old(core).frames, None)) by {
                reveal(FrameTable::preserves_initializing);
            }
            assert(core.frames.preserves_remote_leases(&old(core).frames)) by {
                assert forall|f: FrameId| f.0 < old(core).frames.metas.len() implies {
                    &&& (#[trigger] core.frames.frame(f)).remote_shares == old(core).frames.frame(
                        f,
                    ).remote_shares
                    &&& core.frames.meta(f).active_remote_leases == old(core).frames.meta(
                        f,
                    ).active_remote_leases
                } by {
                    assert(before_finish.frames.frame(f).remote_shares == old(core).frames.frame(
                        f,
                    ).remote_shares);
                    assert(before_finish.frames.meta(f).active_remote_leases == old(
                        core,
                    ).frames.meta(f).active_remote_leases);
                }
            }
            assert forall|f: FrameId|
                f.0 < old(core).frames.metas.len() && (#[trigger] old(core).frames.meta(f)).state
                    == STATE_INITIALIZING implies slot.loc != old(core).frames.frame(
                f,
            ).slot.loc by {
                assert(old(core).frames.meta(frame).state == STATE_EMPTY);
                assert(frame != f);
                if frame.0 < f.0 {
                    assert(old(core).frames.ghosts@[frame.0 as nat].slot.loc != old(
                        core,
                    ).frames.ghosts@[f.0 as nat].slot.loc);
                } else {
                    assert(old(core).frames.ghosts@[f.0 as nat].slot.loc != old(
                        core,
                    ).frames.ghosts@[frame.0 as nat].slot.loc);
                }
            }
            lemma_core_wf_facts(old(core));
            lemma_core_wf_facts(core);
            lemma_published_frame_replica_valid(core, frame);
            assert(core.frames.frame(frame).cert == Some(cert@)) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames.frame(frame).bound_version == Some(ver)) by {
                reveal(FrameTable::same_directory);
            }
            lemma_core_transition_single_event(
                old(core),
                core,
                Some(self.timeline),
                old(ctx).source@.source(),
                cert@.writer,
                Some(slot.loc),
                Some(ver),
            );
        }
        Ok((ver, frame))
    }

    /// Synchronously read a published Host-DRAM frame into `dst` using an external
    /// `HostContext` (allowing `&self` to stay shared-borrowed by `CudaBackend`).
    pub fn read_frame_into_with(
        &self,
        ctx: &mut HostContext,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst: &mut [u8],
    ) -> (r: BackendResult<VersionId>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            old(ctx).ready(old(core).instance_id(), old(core).node()),
            source_fresh_for_ledger(old(ctx).source@, old(core).ledger@),
            old(ctx).source@.source() != self.ctx.source@.source(),
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            final(ctx).ready(final(core).instance_id(), final(core).node()),
            final(ctx).source@.source() == old(ctx).source@.source(),
            source_fresh_for_ledger(final(ctx).source@, final(core).ledger@),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).version_registry == old(core).version_registry,
            final(core).frames.same_directory(&old(core).frames),
            core_transition(
                old(core),
                final(core),
                None,
                Some(old(ctx).source@.source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok(ver) ==> self.owns(frame) && old(core).frames.meta(frame).state
                == STATE_PUBLISHED && old(core).frames.frame(frame).bound_version == Some(ver)
                && old(core).frames.frame(frame).cert is Some && host_read_witness(
                final(core),
                old(ctx).source@.next_event(),
                old(core).frames.frame(frame).cert.unwrap(),
                ver,
                final(dst)@,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if dst.len() != core.tier_frame_size(self.tier) {
            proof {
                lemma_core_transition_refl(
                    core,
                    None,
                    Some(ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::InvalidArgument);
        }
        if !self.owns_frame(frame) {
            proof {
                lemma_core_transition_refl(
                    core,
                    None,
                    Some(ctx.source@.source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        let (lease, version, Ghost(cert), Tracked(cap)) = match core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        None,
                        Some(ctx.source@.source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        let ghost before_read_core = *core;
        let ghost before_ctx = *ctx;
        let src = self.buffer.slot_slice((frame.0 - self.first_frame.0) as usize);
        let ghost reader = ctx.source@.next_event();
        let tracked mut perm = ReadPerm::Shared(cap);
        proof {
            lemma_published_frame_slot_valid(old(core), frame);
            assert(before_read_core.frames.published(frame) == old(core).frames.published(frame));
            lemma_core_wf_facts(core);
            lemma_slab_bound_same_directory(
                old(core),
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            lemma_slab_frame_slot(
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
                frame,
            );
            assert(src.slot_ref() == core.frames.frame(frame).slot);
            lemma_citations_closure(core);
            lemma_fresh_source_event_absent(ctx.source@, core.ledger@);
            core.lemma_citations_admit_closure(perm.frontier(), ISet::empty());
            assert(ISet::empty().union(core.citations()) =~= core.citations());
        }
        let Tracked(completion) = host::read_slice(
            ctx,
            Tracked(instance),
            &src,
            dst,
            Tracked(&mut perm),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Ghost(core.citations()),
        );
        let tracked cap = perm.into_shared();
        proof {
            assert(core.disjoint_from_published(completion.submitted.value().access.writes));
        }
        let ghost sub_val = completion.submitted.value();
        let ghost knows_val = completion.knows;
        let ghost read_preds = sub_val.preds;
        core.record_disjoint_event(
            Tracked(completion.submitted),
            Tracked(Some(completion.knows)),
            Tracked(None),
        );
        proof {
            lemma_observed_event_in_citations(
                before_read_core.ledger@,
                core.ledger@,
                core.node(),
                reader,
                sub_val,
                knows_val,
                read_preds,
            );
            lemma_source_fresh_stepped(
                before_ctx.source@,
                ctx.source@,
                before_read_core.ledger@,
                core.ledger@,
                reader,
            );
            assert(source_fresh_for_ledger(self.ctx.source@, core.ledger@)) by {
                assert forall|e: TransferEvent| #[trigger]
                    core.ledger@.submitted.dom().contains(e) && e.source
                        == self.ctx.source@.source() && e.observer
                        == self.ctx.source@.node() implies e.seq < self.ctx.source@.next_seq() by {
                    assert(e != reader);
                    assert(before_read_core.ledger@.submitted.dom().contains(e));
                }
            }
            lemma_returned_reads_observed(core, frame, cap, reader);
        }
        let ghost before_release = *core;
        core.release_local_lease(lease, Tracked(cap), Ticket::NONE, Tracked(instance)).unwrap();
        proof {
            lemma_core_wf_facts(old(core));
            lemma_core_wf_facts(core);
            lemma_slab_bound_same_frames(
                &before_read_core,
                &before_release,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            lemma_slab_bound_same_directory(
                &before_release,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            assert(core.frames.same_directory(&old(core).frames));
            assert(core.frames.same_layout(&old(core).frames)) by {
                reveal(FrameTable::same_layout);
            }
            assert(core.frames.preserves_initializing(&old(core).frames, None)) by {
                reveal(FrameTable::preserves_initializing);
            }
            assert(core.frames.preserves_remote_leases(&old(core).frames)) by {
                assert forall|f: FrameId| f.0 < old(core).frames.metas.len() implies {
                    &&& (#[trigger] core.frames.frame(f)).remote_shares == old(core).frames.frame(
                        f,
                    ).remote_shares
                    &&& core.frames.meta(f).active_remote_leases == old(core).frames.meta(
                        f,
                    ).active_remote_leases
                } by {
                    assert(before_release.frames.frame(f).remote_shares == old(core).frames.frame(
                        f,
                    ).remote_shares);
                    assert(before_release.frames.meta(f).active_remote_leases == old(
                        core,
                    ).frames.meta(f).active_remote_leases);
                }
            }
            assert forall|s: int|
                0 <= s < old(core).active_leases.len() && (#[trigger] old(
                    core,
                ).active_leases[s] matches Some(l) && l.internal) implies core.active_leases[s]
                == old(core).active_leases[s] by {
                assert(s != lease.slot as int);
            }
            lemma_core_wf_facts(old(core));
            lemma_core_wf_facts(core);
            lemma_published_frame_replica_valid(core, frame);
            assert(core.frames.frame(frame).cert == Some(cert)) by {
                reveal(FrameTable::same_directory);
            }
            assert(core.frames.frame(frame).bound_version == Some(version)) by {
                reveal(FrameTable::same_directory);
            }
            assert(reader.source.succeeds_on_observe()) by {
                assert(reader.source == before_ctx.source@.source());
            }
            assert(core.ledger@.citations(core.node()).contains(reader));
            assert(event_succeeded(core.ledger@, core.node(), reader));
            assert(core.ledger@.submitted.map()[reader] == sub_val);
            assert(sub_val.access.reads.contains(cert.slot.loc));
            assert(effect_implies(sub_val.effect, read_bytes_effect(cert.slot, dst@)));
            lemma_core_transition_single_event(
                old(core),
                core,
                None,
                old(ctx).source@.source(),
                reader,
                None,
                None,
            );
        }
        Ok(version)
    }
}

} // verus!
