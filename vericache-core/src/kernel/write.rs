#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    pub fn allocate_version(&mut self) -> (r: Result<VersionId, CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).frames == old(self).frames,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            match r {
                Err(e) => e == CoreError::SequenceExhausted && old(self).next_version_seq == old(
                    self,
                ).version_seq_limit && *final(self) == *old(self),
                Ok(v) => v == (VersionId {
                    origin_node: old(self).node_id,
                    seq: old(self).next_version_seq,
                }) && v.well_formed() && v != VersionId::NONE && !old(
                    self,
                ).frames.versions@.contains_key(v) && final(self).next_version_seq == old(
                    self,
                ).next_version_seq + 1,
            },
    {
        if self.next_version_seq >= self.version_seq_limit {
            return Err(CoreError::SequenceExhausted);
        }
        let seq = self.next_version_seq;
        self.next_version_seq = seq + 1;
        Ok(VersionId { origin_node: self.node_id, seq })
    }

    /// Reserve a ready frame on `tier` with a freshly allocated version and check
    /// out its exclusive `SlotToken` while keeping `CacheCore::well_formed()` true.
    pub fn begin_write(&mut self, tier: TierId, key: ObjectKey) -> (r: Result<
        (WriteHandle, Tracked<SlotToken>),
        CoreError,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((handle, tok)) => {
                    let f = handle.frame;
                    let g = final(self).frames.frame(f);
                    let m = final(self).frames.meta(f);
                    &&& tier.valid(old(self).tiers.len())
                    &&& handle.tier == tier
                    &&& handle.key == key
                    &&& handle.slot@ == g.slot
                    &&& handle.version == (VersionId {
                        origin_node: old(self).node_id,
                        seq: old(self).next_version_seq,
                    })
                    &&& handle.version.well_formed() && handle.version != VersionId::NONE
                    &&& !old(self).frames.versions@.contains_key(handle.version)
                    &&& final(self).next_version_seq == old(self).next_version_seq + 1
                    &&& old(self).frames.meta(f).state == STATE_EMPTY
                    &&& f.0 < final(self).frames.metas.len()
                    &&& m.state == STATE_INITIALIZING && !m.queued()
                    &&& g.tier == tier && g.bound_key == Some(key)
                    &&& g.bound_version == Some(handle.version)
                    &&& g.token is None
                    &&& slot_is_owned(tok@, final(self).instance_id(), g.slot.loc)
                    &&& write_admitted(slot_frontier(tok@), final(self).citations())
                },
            },
    {
        if tier.as_index() >= self.tiers.len() {
            return Err(CoreError::UnsupportedTier);
        }
        if self.frames.ready_queue[tier.as_index()].is_empty() {
            return Err(CoreError::CapacityExhausted);
        }
        if self.next_version_seq >= self.version_seq_limit {
            return Err(CoreError::SequenceExhausted);
        }
        let seq = self.next_version_seq;
        let version = VersionId { origin_node: self.node_id, seq };
        let ghost v = self.kernel_view();
        let (f, tok) = self.frames.reserve_ready_with_token(
            tier,
            key,
            version,
            &self.timelines,
            Ghost(v),
        )?;
        let ghost slot = self.frames.frame(f).slot;
        self.next_version_seq = seq + 1;
        proof {
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
        Ok((WriteHandle { frame: f, tier, key, version, slot: Ghost(slot) }, tok))
    }

    /// Reserve a ready frame on `tier` for an existing `version` and check out its
    /// exclusive `SlotToken` while keeping `CacheCore::well_formed()` true.
    pub fn begin_write_version(&mut self, tier: TierId, key: ObjectKey, version: VersionId) -> (r:
        Result<(WriteHandle, Tracked<SlotToken>), CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((handle, tok)) => {
                    let f = handle.frame;
                    let g = final(self).frames.frame(f);
                    let m = final(self).frames.meta(f);
                    &&& tier.valid(old(self).tiers.len())
                    &&& handle.tier == tier
                    &&& handle.key == key
                    &&& handle.slot@ == g.slot
                    &&& handle.version == version
                    &&& version.well_formed() && version != VersionId::NONE
                    &&& (version.origin_node != old(self).node_id || version.seq < old(
                        self,
                    ).next_version_seq || version.seq >= old(self).version_seq_limit)
                    &&& old(self).frames.meta(f).state == STATE_EMPTY
                    &&& f.0 < final(self).frames.metas.len()
                    &&& m.state == STATE_INITIALIZING && !m.queued()
                    &&& g.tier == tier && g.bound_key == Some(key)
                    &&& g.bound_version == Some(version)
                    &&& g.token is None
                    &&& slot_is_owned(tok@, final(self).instance_id(), g.slot.loc)
                    &&& write_admitted(slot_frontier(tok@), final(self).citations())
                },
            },
    {
        if tier.as_index() >= self.tiers.len() {
            return Err(CoreError::UnsupportedTier);
        }
        if !version.well_formed() || (version.origin_node == 0 && version.seq == 0) {
            return Err(CoreError::InvalidArgument);
        }
        if version.origin_node == self.node_id && version.seq >= self.next_version_seq
            && version.seq < self.version_seq_limit {
            return Err(CoreError::InvalidArgument);
        }
        let ghost v = self.kernel_view();
        let (f, tok) = self.frames.reserve_ready_with_token(
            tier,
            key,
            version,
            &self.timelines,
            Ghost(v),
        )?;
        let ghost slot = self.frames.frame(f).slot;
        proof {
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
        Ok((WriteHandle { frame: f, tier, key, version, slot: Ghost(slot) }, tok))
    }

    /// Re-install the checked-out `SlotToken` and publish `handle.frame` into the
    /// catalog in one atomic transaction.
    pub fn finish_write(
        &mut self,
        handle: WriteHandle,
        Tracked(tok): Tracked<SlotToken>,
        writer_ticket: Ticket,
        Ghost(cert): Ghost<ValueCertificate>,
    )
        requires
            old(self).well_formed(),
            handle.frame.0 < old(self).frames.metas.len(),
            old(self).frames.meta(handle.frame).state == STATE_INITIALIZING,
            !old(self).frames.meta(handle.frame).queued(),
            old(self).frames.frame(handle.frame).token is None,
            old(self).frames.frame(handle.frame).slot == handle.slot@,
            slot_is_owned(tok, old(self).instance_id(), handle.slot@.loc),
            slot_frontier(tok).readers.is_empty(),
            slot_frontier(tok).last_writer == Some(cert.writer),
            old(self).frames.versions@.contains_key(
                old(self).frames.frame(handle.frame).bound_version.unwrap(),
            ) ==> old(self).frames.versions@[old(self).frames.frame(
                handle.frame,
            ).bound_version.unwrap()] == cert.bytes,
            old(self).frames.frame(handle.frame).bound_version.unwrap().origin_node != old(
                self,
            ).node_id || old(self).frames.frame(handle.frame).bound_version.unwrap().seq < old(
                self,
            ).next_version_seq || old(self).frames.frame(handle.frame).bound_version.unwrap().seq
                >= old(self).version_seq_limit,
            cert.slot == handle.slot@,
            cert.bytes.len() == cert.slot.slot_len,
            old(self).timelines.issued(writer_ticket.timeline_id(), writer_ticket.sequence()),
            writer_ticket != Ticket::NONE,
            cert.writer == old(self).timelines.events@[(
                writer_ticket.timeline_id(),
                writer_ticket.sequence(),
            )],
            replica_valid(old(self).ledger@, old(self).node(), cert),
            old(self).citations().contains(cert.writer),
        ensures
            final(self).well_formed(),
            old(self).spec@.versions_monotonic(final(self).spec@),
            final(self).spec@.versions == old(self).spec@.versions.insert(
                old(self).frames.frame(handle.frame).bound_version.unwrap(),
                cert.bytes,
            ),
            final(self).spec@.catalog == old(self).spec@.catalog.insert(
                old(self).frames.frame(handle.frame).bound_key.unwrap(),
                old(self).frames.frame(handle.frame).bound_version.unwrap(),
            ),
            final(self).frames.meta(handle.frame).state == STATE_PUBLISHED,
            final(self).frames.frame(handle.frame).bound_version == old(self).frames.frame(
                handle.frame,
            ).bound_version,
            final(self).frames.frame(handle.frame).cert == Some(cert),
            final(self).frames.same_layout(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, Some(handle.frame)),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            forall|g: FrameId|
                g != handle.frame && g.0 < old(self).frames.metas.len() ==> {
                    &&& (#[trigger] final(self).frames.frame(g)) == old(self).frames.frame(g)
                    &&& final(self).frames.meta(g).active_remote_leases == old(self).frames.meta(
                        g,
                    ).active_remote_leases
                },
            final(self).catalog.slot(
                old(self).frames.frame(handle.frame).bound_key.unwrap(),
                old(self).frames.frame(handle.frame).tier.index() as int,
            ) == Some(handle.frame),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).version_registry == old(self).version_registry,
    {
        let f = handle.frame;
        let ghost v = self.kernel_view();
        let ghost bound_key = self.frames.frame(f).bound_key.unwrap();
        let ghost bound_ver = self.frames.frame(f).bound_version.unwrap();
        proof {
            assert(self.frames.valid_frame(f, v));
        }
        self.catalog.publish_frame_with_token(
            &mut self.frames,
            f,
            Tracked(tok),
            writer_ticket,
            Ghost(cert),
            Ghost(v),
        );
        proof {
            self.spec = Ghost(Self::service(&self.frames, &self.catalog));
            self.preserve_after_publish(*old(self), f, bound_key, bound_ver, cert, v);
            assert(self.frames.valid_frame(f, v));
            assert(self.frames.meta(f).active_remote_leases == old(self).frames.meta(
                f,
            ).active_remote_leases);
        }
    }

    /// Return a checked-out `SlotToken` on an aborted/failed write and move the
    /// frame back to `ready_queue`.
    pub fn abort_write(&mut self, handle: WriteHandle, Tracked(tok): Tracked<SlotToken>)
        requires
            old(self).well_formed(),
            handle.frame.0 < old(self).frames.metas.len(),
            old(self).frames.meta(handle.frame).state == STATE_INITIALIZING,
            old(self).frames.frame(handle.frame).tier == handle.tier,
            old(self).frames.frame(handle.frame).slot == handle.slot@,
            !old(self).frames.meta(handle.frame).queued(),
            old(self).frames.frame(handle.frame).token is None,
            slot_is_owned(tok, old(self).instance_id(), handle.slot@.loc),
            write_admitted(slot_frontier(tok), old(self).citations()),
        ensures
            final(self).well_formed(),
            final(self).spec == old(self).spec,
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.same_layout(&old(self).frames),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, Some(handle.frame)),
            final(self).frames.meta(handle.frame).state == STATE_EMPTY && final(self).frames.meta(
                handle.frame,
            ).queued(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).version_registry == old(self).version_registry,
    {
        let ghost v = self.kernel_view();
        self.frames.abort_checked_out_write(handle.frame, handle.tier, Tracked(tok), Ghost(v));
        proof {
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
    }

    /// Park a checked-out initializing frame back into `FrameTable` with its
    /// write ticket so async completion can later call `publish_parked_write` or
    /// `abort_parked_write`.
    pub fn park_write(
        &mut self,
        handle: WriteHandle,
        Tracked(tok): Tracked<SlotToken>,
        writer_ticket: Ticket,
    )
        requires
            old(self).well_formed(),
            handle.frame.0 < old(self).frames.metas.len(),
            old(self).frames.meta(handle.frame).state == STATE_INITIALIZING,
            !old(self).frames.meta(handle.frame).queued(),
            old(self).frames.frame(handle.frame).token is None,
            old(self).frames.frame(handle.frame).slot == handle.slot@,
            old(self).timelines.issued(writer_ticket.timeline_id(), writer_ticket.sequence()),
            writer_ticket != Ticket::NONE,
            slot_is_owned(tok, old(self).instance_id(), handle.slot@.loc),
            slot_frontier(tok).readers.subset_of(old(self).citations()),
            slot_frontier(tok).last_writer matches Some(w) ==> old(self).citations().contains(w)
                || old(self).timelines.ticket_covers(writer_ticket, w),
        ensures
            final(self).well_formed(),
            final(self).frames.preserves_initializing(&old(self).frames, Some(handle.frame)),
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            final(self).frames.meta(handle.frame).state == STATE_INITIALIZING
                && !final(self).frames.meta(handle.frame).queued(),
            final(self).frames.meta(handle.frame).last_write_ticket == writer_ticket,
            final(self).frames.frame(handle.frame).tier == old(self).frames.frame(
                handle.frame,
            ).tier,
            final(self).frames.frame(handle.frame).slot == old(self).frames.frame(
                handle.frame,
            ).slot,
            final(self).frames.frame(handle.frame).bound_key == old(self).frames.frame(
                handle.frame,
            ).bound_key,
            final(self).frames.frame(handle.frame).bound_version == old(self).frames.frame(
                handle.frame,
            ).bound_version,
            final(self).frames.frame(handle.frame).token == Some(tok),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
    {
        let ghost v = self.kernel_view();
        self.frames.park_initializing_write(handle.frame, writer_ticket, Tracked(tok), Ghost(v));
        proof {
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
    }

    pub fn publish_parked_write(
        &mut self,
        f: FrameId,
        writer_ticket: Ticket,
        Ghost(cert): Ghost<ValueCertificate>,
    )
        requires
            old(self).well_formed(),
            f.0 < old(self).frames.metas.len(),
            old(self).frames.meta(f).state == STATE_INITIALIZING,
            !old(self).frames.meta(f).queued(),
            old(self).frames.frame(f).completed_writes(cert),
            old(self).frames.versions@.contains_key(
                old(self).frames.frame(f).bound_version.unwrap(),
            ) ==> old(self).frames.versions@[old(self).frames.frame(f).bound_version.unwrap()]
                == cert.bytes,
            old(self).frames.frame(f).bound_version.unwrap().origin_node != old(self).node_id
                || old(self).frames.frame(f).bound_version.unwrap().seq < old(self).next_version_seq
                || old(self).frames.frame(f).bound_version.unwrap().seq >= old(
                self,
            ).version_seq_limit,
            cert.slot == old(self).frames.frame(f).slot,
            cert.bytes.len() == cert.slot.slot_len,
            old(self).timelines.issued(writer_ticket.timeline_id(), writer_ticket.sequence()),
            writer_ticket != Ticket::NONE,
            cert.writer == old(self).timelines.events@[(
                writer_ticket.timeline_id(),
                writer_ticket.sequence(),
            )],
            replica_valid(old(self).ledger@, old(self).node(), cert),
            old(self).citations().contains(cert.writer),
        ensures
            final(self).well_formed(),
            old(self).spec@.versions_monotonic(final(self).spec@),
            final(self).spec@.versions == old(self).spec@.versions.insert(
                old(self).frames.frame(f).bound_version.unwrap(),
                cert.bytes,
            ),
            final(self).spec@.catalog == old(self).spec@.catalog.insert(
                old(self).frames.frame(f).bound_key.unwrap(),
                old(self).frames.frame(f).bound_version.unwrap(),
            ),
            final(self).frames.meta(f).state == STATE_PUBLISHED,
            final(self).frames.frame(f).cert == Some(cert),
            final(self).frames.same_layout(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, Some(f)),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            final(self).catalog.slot(
                old(self).frames.frame(f).bound_key.unwrap(),
                old(self).frames.frame(f).tier.index() as int,
            ) == Some(f),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).version_registry == old(self).version_registry,
    {
        let ghost v = self.kernel_view();
        let ghost bound_key = self.frames.frame(f).bound_key.unwrap();
        let ghost bound_ver = self.frames.frame(f).bound_version.unwrap();
        proof {
            assert(self.frames.valid_frame(f, v));
        }
        self.catalog.publish_frame(&mut self.frames, f, writer_ticket, Ghost(cert), Ghost(v));
        proof {
            self.spec = Ghost(Self::service(&self.frames, &self.catalog));
            self.preserve_after_publish(*old(self), f, bound_key, bound_ver, cert, v);
            assert(self.frames.valid_frame(f, v));
            assert(self.frames.meta(f).active_remote_leases == old(self).frames.meta(
                f,
            ).active_remote_leases);
        }
    }

    pub fn abort_parked_write(&mut self, f: FrameId, tier: TierId)
        requires
            old(self).well_formed(),
            f.0 < old(self).frames.metas.len(),
            old(self).frames.meta(f).state == STATE_INITIALIZING,
            old(self).frames.frame(f).tier == tier,
            !old(self).frames.meta(f).queued(),
            old(self).frames.frame(f).writable(
                old(self).instance_id(),
                old(self).node(),
                &old(self).ledger@,
                ISet::empty(),
            ),
        ensures
            final(self).well_formed(),
            final(self).spec == old(self).spec,
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, Some(f)),
            final(self).frames.meta(f).state == STATE_EMPTY && final(self).frames.meta(f).queued(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).timelines == old(self).timelines,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).active_leases == old(self).active_leases,
            final(self).free_lease_slots == old(self).free_lease_slots,
            final(self).lease_generation == old(self).lease_generation,
            final(self).version_registry == old(self).version_registry,
    {
        let ghost v = self.kernel_view();
        self.frames.discard_observed_write(f, tier, Ghost(v));
        proof {
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
    }

    proof fn preserve_after_publish(
        &self,
        before: Self,
        f: FrameId,
        bound_key: ObjectKey,
        bound_ver: VersionId,
        cert: ValueCertificate,
        v: KernelView,
    )
        requires
            before.well_formed(),
            v == before.kernel_view(),
            f.0 < before.frames.metas.len(),
            before.frames.meta(f).state == STATE_INITIALIZING,
            before.frames.frame(f).bound_key == Some(bound_key),
            before.frames.frame(f).bound_version == Some(bound_ver),
            bound_ver.origin_node != before.node_id || bound_ver.seq < before.next_version_seq
                || bound_ver.seq >= before.version_seq_limit,
            before.frames.versions@.contains_key(bound_ver) ==> before.frames.versions@[bound_ver]
                == cert.bytes,
            cert.bytes.len() == before.frames.tier_frame_size(before.frames.frame(f).tier),
            replica_valid(before.ledger@, before.node(), cert),
            before.citations().contains(cert.writer),
            self.node_id == before.node_id,
            self.tiers == before.tiers,
            self.timelines == before.timelines,
            self.ledger == before.ledger,
            self.version_seq_start == before.version_seq_start,
            self.next_version_seq == before.next_version_seq,
            self.version_seq_limit == before.version_seq_limit,
            self.active_leases == before.active_leases,
            self.free_lease_slots == before.free_lease_slots,
            self.lease_generation == before.lease_generation,
            self.version_registry == before.version_registry,
            self.spec@ == Self::service(&self.frames, &self.catalog),
            self.frames.consistent(v),
            self.catalog.well_formed(&self.frames),
            self.frames.num_tiers == before.frames.num_tiers,
            self.frames.frame_size == before.frames.frame_size,
            self.frames.tier_frame_sizes == before.frames.tier_frame_sizes,
            self.frames.metas.len() == before.frames.metas.len(),
            self.frames.versions@ == before.frames.versions@.insert(bound_ver, cert.bytes),
            self.frames.meta(f).state == STATE_PUBLISHED,
            self.frames.frame(f).cert == Some(cert),
            self.frames.frame(f).bound_version == Some(bound_ver),
            self.catalog.view().contains_key(bound_key),
            self.catalog.view()[bound_key].version == bound_ver,
            forall|other: ObjectKey|
                other != bound_key ==> #[trigger] self.catalog.view().contains_key(other)
                    == before.catalog.view().contains_key(other) && (
                before.catalog.view().contains_key(other) ==> self.catalog.view()[other]
                    == before.catalog.view()[other]),
            forall|g: FrameId|
                g != f && g.0 < before.frames.metas.len() ==> (#[trigger] self.frames.frame(g))
                    == before.frames.frame(g) && (self.frames.published(g).is_some()
                    ==> self.frames.published(g) == before.frames.published(g)),
        ensures
            self.well_formed(),
            before.spec@.versions_monotonic(self.spec@),
            self.spec@.versions == before.spec@.versions.insert(bound_ver, cert.bytes),
            self.spec@.catalog == before.spec@.catalog.insert(bound_key, bound_ver),
    {
        assert(before.frames.valid_frame(f, v));
        ServiceSpec::insert_version(
            self.frames.frame_size as nat,
            before.frames.versions@,
            bound_ver,
            cert.bytes,
        );
        Self::lemma_service_well_formed(&self.frames, &self.catalog, v);
        assert(self.spec@.catalog =~= before.spec@.catalog.insert(bound_key, bound_ver));
        assert(before.spec@.versions.submap_of(self.spec@.versions));
        assert forall|g: FrameId|
            g.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(g)).state
                == STATE_PUBLISHED implies {
            let fg = self.frames.frame(g);
            fg.cert matches Some(c) && {
                &&& replica_valid(v.ledger, v.node, c)
                &&& event_succeeded(v.ledger, v.node, c.writer)
                &&& v.ledger.citations(v.node).contains(c.writer)
                &&& fg.bound_version matches Some(ver)
                &&& self.frames.versions@.contains_key(ver)
                &&& self.frames.versions@[ver] == c.bytes
            }
        } by {
            if g != f {
                assert(self.frames.published(g) == before.frames.published(g));
                assert(before.frames.meta(g).state == STATE_PUBLISHED);
                assert(before.frames.valid_frame(g, v));
                let ver = self.frames.frame(g).bound_version.unwrap();
                assert(before.frames.versions@.contains_key(ver));
                assert(self.frames.versions@[ver] == before.frames.versions@[ver]);
            }
        }
        assert forall|s: int|
            0 <= s < self.active_leases.len() implies match #[trigger] self.active_leases[s] {
            None => self.free_lease_slots@.contains(s as u32),
            Some(lease) => {
                &&& !self.free_lease_slots@.contains(s as u32)
                &&& lease.handle.slot as int == s
                &&& 1 <= lease.handle.generation <= self.lease_generation
                &&& lease.handle.frame.0 < self.frames.metas.len()
                &&& self.frames.frame(lease.handle.frame).local_shares.contains_key(lease.share@)
            },
        } by {
            if let Some(lease) = self.active_leases[s] {
                if lease.handle.frame == f {
                    assert(before.frames.frame(f).local_shares.len() == 0);
                    assert(!before.frames.frame(f).local_shares.contains_key(lease.share@));
                }
            }
        }
    }
}

} // verus!
