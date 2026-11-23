use super::*;
use transfer::host;

verus! {

impl HostBackend {
    fn write_and_certify_host_slot(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        frame: FrameId,
        Ghost(slot): Ghost<SlotRef>,
        Tracked(tok): Tracked<&mut SlotToken>,
        data: &[u8],
    ) -> (r: (Ticket, Ghost<ValueCertificate>))
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(core).timelines.next_seq[old(self).timeline as int] <= MAX_TICKET_SEQ,
            data@.len() == old(core).frames.tier_frame_size(old(self).tier),
            old(self).owns(frame),
            frame.0 < old(core).frames.metas.len(),
            old(core).frames.meta(frame).state == STATE_INITIALIZING,
            old(core).frames.frame(frame).slot == slot,
            slot_is_owned(*old(tok), old(core).instance_id(), slot.loc),
            write_admitted(slot_frontier(*old(tok)), old(core).citations()),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).buffer == old(self).buffer,
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).spec == old(core).spec,
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
        let ghost before_ctx = self.ctx;
        let dst = self.buffer.slot_slice((frame.0 - self.first_frame.0) as usize);
        let ghost writer = self.ctx.source@.next_event();
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
            lemma_fresh_source_event_absent(self.ctx.source@, core.ledger@);
            core.lemma_citations_admit_closure(slot_frontier(*tok), ISet::empty());
            assert(ISet::empty().union(core.citations()) =~= core.citations());
        }
        let Tracked(completion) = host::write_slice(
            &mut self.ctx,
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
                self.ctx.source@,
                before_write.ledger@,
                core.ledger@,
                writer,
            );
            lemma_slab_bound_same_frames(
                old(core),
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
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

    /// Synchronously write `data` into a fresh Host-DRAM frame and publish it
    /// into `core` with a `ValueProof::HostWrite` certificate.
    pub fn put_host(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        data: &[u8],
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).buffer == old(self).buffer,
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            old(core).spec@.versions_monotonic(final(core).spec@),
            match r {
                Err(_) => final(core).spec == old(core).spec,
                Ok((ver, frame)) => {
                    &&& final(self).owns(frame)
                    &&& final(core).spec@.versions == old(core).spec@.versions.insert(ver, data@)
                    &&& final(core).spec@.catalog == old(core).spec@.catalog.insert(key, ver)
                    &&& final(core).catalog.slot(key, old(self).tier.index() as int) == Some(frame)
                    &&& final(core).frames.meta(frame).state == STATE_PUBLISHED
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if data.len() != core.tier_frame_size(self.tier) {
            return Err(CoreError::InvalidArgument);
        }
        let (handle, Tracked(mut tok)) = core.begin_write(self.tier, key)?;
        proof {
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
            }
            return Err(CoreError::SequenceExhausted);
        }
        let ver = handle.version;
        let frame = handle.frame;
        let (ticket, cert) = self.write_and_certify_host_slot(
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
            lemma_slab_bound_same_layout(
                &before_finish,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
        }
        Ok((ver, frame))
    }

    /// Synchronously read a published Host-DRAM frame into `dst` using a local
    /// read lease (`acquire_local_lease` -> `host::read_slice` -> `release_local_lease`).
    pub fn read_frame_into(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst: &mut [u8],
    ) -> (r: BackendResult<VersionId>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
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
            final(dst)@.len() == old(dst)@.len(),
            final(self).buffer == old(self).buffer,
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).spec == old(core).spec,
            final(core).frames.same_directory(&old(core).frames),
            r matches Ok(ver) ==> final(self).owns(frame) && old(core).frames.meta(frame).state
                == STATE_PUBLISHED && old(core).frames.frame(frame).bound_version == Some(ver),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if dst.len() != core.tier_frame_size(self.tier) {
            return Err(CoreError::InvalidArgument);
        }
        if !self.owns_frame(frame) {
            return Err(CoreError::UnsupportedTier);
        }
        let (lease, version, _cert, Tracked(cap)) = core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        )?;
        let ghost before_read_core = *core;
        let ghost before_ctx = self.ctx;
        let src = self.buffer.slot_slice((frame.0 - self.first_frame.0) as usize);
        let ghost reader = self.ctx.source@.next_event();
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
            core.lemma_citations_admit_closure(perm.frontier(), ISet::empty());
            assert(ISet::empty().union(core.citations()) =~= core.citations());
        }
        let Tracked(completion) = host::read_slice(
            &mut self.ctx,
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
        core.record_disjoint_event(
            Tracked(completion.submitted),
            Tracked(Some(completion.knows)),
            Tracked(None),
        );
        proof {
            lemma_source_fresh_stepped(
                before_ctx.source@,
                self.ctx.source@,
                before_read_core.ledger@,
                core.ledger@,
                reader,
            );
            assert(core.ledger@.knows.set().contains((core.node(), reader))) by {
                assert(completion.knows.set().contains((core.node(), reader)));
            }
            assert(core.citations().contains(reader));
            lemma_returned_reads_observed(core, frame, cap, reader);
        }
        let ghost before_release = *core;
        core.release_local_lease(lease, Tracked(cap), Ticket::NONE, Tracked(instance)).unwrap();
        proof {
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
            assert(core.frames.same_directory(&old(core).frames)) by {
                assert(before_read_core.frames.same_directory(&old(core).frames));
                assert(before_release.frames == before_read_core.frames);
                assert(core.frames.same_directory(&before_release.frames));
            }
        }
        Ok(version)
    }

    fn commit_host_copy(
        &mut self,
        dst_backend: &HostBackend,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
    )
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            dst_backend.well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(core).timelines.next_seq[old(self).timeline as int] <= MAX_TICKET_SEQ,
            old(self).owns(hop.src_frame),
            dst_backend.owns(hop.dst_frame),
            hop.src_frame != hop.dst_frame,
            hop.src_frame.0 < old(core).frames.metas.len(),
            hop.dst_frame.0 < old(core).frames.metas.len(),
            hop.src_slot@ == old(core).frames.frame(hop.src_frame).slot,
            hop.dst_slot@ == old(core).frames.frame(hop.dst_frame).slot,
            hop.src_slot@.loc != hop.dst_slot@.loc,
            old(core).frames.meta(hop.src_frame).state == STATE_PUBLISHED,
            old(core).frames.frame(hop.src_frame).tier == old(self).tier,
            old(core).frames.frame(hop.src_frame).bound_version == Some(hop.version),
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
            src_cap.share() == old(core).active_leases[hop.src_lease.slot as int]->Some_0.share@,
            old(core).frames.meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(hop.dst_frame).queued(),
            old(core).frames.frame(hop.dst_frame).tier == dst_backend.tier,
            old(core).frames.frame(hop.dst_frame).bound_key == Some(hop.key),
            old(core).frames.frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames.frame(hop.dst_frame).token is None,
            hop.src_slot@.slot_len == hop.dst_slot@.slot_len,
            src_cap.is_local(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == hop.src_slot@.loc,
            read_admitted(src_cap.frontier(), old(core).citations()),
            src_cap.frontier().readers.is_empty(),
            slot_is_owned(dst_tok, old(core).instance_id(), hop.dst_slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(core).citations()),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).buffer == old(self).buffer,
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            old(core).spec@.versions_monotonic(final(core).spec@),
            final(core).frames.meta(hop.dst_frame).state == STATE_PUBLISHED,
            final(core).frames.frame(hop.dst_frame).tier == dst_backend.tier,
            final(core).catalog.slot(hop.key, dst_backend.tier.index() as int) == Some(
                hop.dst_frame,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_copy = *core;
        let ghost before_ctx = self.ctx;
        let tracked mut src_perm = ReadPerm::Shared(src_cap);
        let tracked mut dst_tok = dst_tok;
        let src_slice = self.buffer.slot_slice((hop.src_frame.0 - self.first_frame.0) as usize);
        let dst_slice = dst_backend.frame_slice(hop.dst_frame);
        let ghost writer = self.ctx.source@.next_event();
        proof {
            lemma_core_wf_facts(core);
            lemma_published_frame_slot_valid(core, hop.src_frame);
            lemma_initializing_frame_slot_valid(core, hop.dst_frame);
            lemma_slab_frame_slot(
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
                hop.src_frame,
            );
            lemma_slab_frame_slot(
                core,
                dst_backend.tier,
                dst_backend.first_frame,
                dst_backend.num_frames,
                dst_backend.buffer.space(),
                hop.dst_frame,
            );
            assert(src_slice.slot_ref() == hop.src_slot@);
            assert(dst_slice.slot_ref() == hop.dst_slot@);
            lemma_citations_closure(core);
            lemma_fresh_source_event_absent(self.ctx.source@, core.ledger@);
            core.lemma_citations_admit_closure(src_perm.frontier(), ISet::empty());
            core.lemma_citations_admit_closure(slot_frontier(dst_tok), ISet::empty());
            assert(ISet::empty().union(core.citations()) =~= core.citations());
        }
        let Tracked(completion) = host::copy(
            &mut self.ctx,
            Tracked(instance),
            &dst_slice,
            &src_slice,
            Tracked(&mut src_perm),
            Tracked(&mut dst_tok),
            Tracked(&core.ledger.borrow().submitted),
            Tracked(&core.ledger.borrow().knows),
            Ghost(core.citations()),
        );
        let tracked src_cap = src_perm.into_shared();
        proof {
            lemma_write_disjoint_from_published(
                core,
                hop.dst_frame,
                completion.submitted.value().access.writes,
            );
        }
        let ghost sub_val = completion.submitted.value();
        let ghost knows_val = completion.knows;
        let ghost copy_preds = sub_val.preds;
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
        let ghost dst_cert;
        proof {
            lemma_observed_event_in_citations(
                before_copy.ledger@,
                core.ledger@,
                core.node(),
                writer,
                sub_val,
                knows_val,
                copy_preds,
            );
            lemma_source_fresh_stepped(
                before_ctx.source@,
                self.ctx.source@,
                before_copy.ledger@,
                core.ledger@,
                writer,
            );
            assert(writer.source.succeeds_on_observe());
            assert(event_succeeded(core.ledger@, core.node(), writer));
            dst_cert =
            lemma_certify_single_copy(
                before_copy.ledger@,
                core.ledger@,
                core.node(),
                hop.src_cert@,
                hop.src_slot@,
                hop.dst_slot@,
                writer,
            );
            lemma_returned_reads_observed(core, hop.src_frame, src_cap, writer);
        }
        let ghost before_finish = *core;
        core.finish_copy_hop(
            hop,
            Tracked(src_cap),
            Ticket::NONE,
            Tracked(dst_tok),
            ticket,
            Ghost(dst_cert),
            Tracked(instance),
        );
        proof {
            lemma_slab_bound_same_frames(
                old(core),
                &before_finish,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
            lemma_slab_bound_same_layout(
                &before_finish,
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
            );
        }
    }

    /// Synchronously copy a published frame from `self` into a destination
    /// `HostBackend` tier via `begin_copy_hop` -> `host::copy` -> `finish_copy_hop`.
    pub fn copy_frame_to_host_tier(
        &mut self,
        dst_backend: &HostBackend,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<FrameId>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            dst_backend.well_formed(old(core)),
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
            final(self).buffer == old(self).buffer,
            final(self).tier == old(self).tier,
            final(self).timeline == old(self).timeline,
            final(self).first_frame == old(self).first_frame,
            final(self).num_frames == old(self).num_frames,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            old(core).spec@.versions_monotonic(final(core).spec@),
            r matches Ok(dst_frame) ==> {
                &&& dst_backend.owns(dst_frame)
                &&& final(core).frames.meta(dst_frame).state == STATE_PUBLISHED
                &&& final(core).frames.frame(dst_frame).tier == dst_backend.tier
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if !self.owns_frame(src_frame) {
            return Err(CoreError::UnsupportedTier);
        }
        if core.timelines.next_seq[self.timeline as usize] > MAX_TICKET_SEQ {
            return Err(CoreError::SequenceExhausted);
        }
        let (hop, src_cap, dst_tok) = match core.begin_copy_hop(
            src_frame,
            dst_backend.tier,
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
                        self.buffer.space(),
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
                self.buffer.space(),
            );
            lemma_slab_bound_same_directory(
                old(core),
                core,
                dst_backend.tier,
                dst_backend.first_frame,
                dst_backend.num_frames,
                dst_backend.buffer.space(),
            );
            lemma_slab_frame_slot(
                core,
                self.tier,
                self.first_frame,
                self.num_frames,
                self.buffer.space(),
                src_frame,
            );
            lemma_slab_owns_tier_frame(
                core,
                dst_backend.tier,
                dst_backend.first_frame,
                dst_backend.num_frames,
                dst_backend.buffer.space(),
                hop.dst_frame,
            );
        }
        let dst_frame = hop.dst_frame;
        self.commit_host_copy(dst_backend, core, Tracked(instance), hop, src_cap, dst_tok);
        Ok(dst_frame)
    }
}

} // verus!
