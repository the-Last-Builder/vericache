#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    proof fn lemma_published_binding(&self, f: FrameId)
        requires
            self.well_formed(),
            f.0 < self.frames.metas.len(),
            self.frames.meta(f).state == STATE_PUBLISHED,
        ensures
            self.frames.frame(f).bound_key == Some(self.frames.bindings[f.0 as int].0),
            self.frames.frame(f).tier == self.frames.bindings[f.0 as int].1,
    {
        assert(self.frames.valid_frame(f, self.kernel_view()));
    }

    proof fn lemma_copy_hop_reserved(
        &self,
        before: Self,
        after_lease: Self,
        src_frame: FrameId,
        dst_frame: FrameId,
        key: ObjectKey,
        version: VersionId,
        src_cert: ValueCertificate,
        src_cap: ShareCap,
    )
        requires
            before.well_formed(),
            after_lease.well_formed(),
            self.well_formed(),
            self.node_id == after_lease.node_id,
            self.timelines == after_lease.timelines,
            after_lease.timelines == before.timelines,
            self.ledger == after_lease.ledger,
            after_lease.frames.same_directory(&before.frames),
            self.frames.same_directory(&after_lease.frames),
            after_lease.frames.preserves_initializing(&before.frames, None),
            self.frames.preserves_initializing(&after_lease.frames, None),
            after_lease.frames.preserves_fenced_leases(&before.frames),
            self.frames.preserves_fenced_leases(&after_lease.frames),
            src_frame.0 < after_lease.frames.metas.len(),
            dst_frame.0 < self.frames.metas.len(),
            after_lease.frames.meta(src_frame).state == STATE_PUBLISHED,
            after_lease.frames.meta(dst_frame).state == STATE_EMPTY,
            after_lease.frames.frame(src_frame).bound_key == Some(key),
            after_lease.frames.frame(src_frame).bound_version == Some(version),
            after_lease.frames.frame(src_frame).cert == Some(src_cert),
            after_lease.frames.tier_frame_size(after_lease.frames.frame(src_frame).tier)
                == after_lease.frames.tier_frame_size(self.frames.frame(dst_frame).tier),
            after_lease.frames.returned_reads(
                src_frame,
                src_cap,
                Ticket::NONE,
                after_lease.kernel_view(),
            ),
        ensures
            self.timelines.only_watermarks(&before.timelines),
            self.frames.same_directory(&before.frames),
            self.frames.preserves_initializing(&before.frames, None),
            self.frames.preserves_fenced_leases(&before.frames),
            before.frames.meta(dst_frame).state != STATE_INITIALIZING,
            src_frame != dst_frame,
            self.frames.frame(src_frame).slot.loc != self.frames.frame(dst_frame).slot.loc,
            self.frames.frame(src_frame).slot.slot_len == self.frames.frame(
                dst_frame,
            ).slot.slot_len,
            self.frames.tier_frame_size(self.frames.frame(src_frame).tier)
                == self.frames.tier_frame_size(self.frames.frame(dst_frame).tier),
            self.frames.meta(src_frame).state == STATE_PUBLISHED,
            self.frames.frame(src_frame).slot == after_lease.frames.frame(src_frame).slot,
            self.frames.frame(src_frame).tier == after_lease.frames.frame(src_frame).tier,
            self.frames.frame(src_frame).bound_key == Some(key),
            self.frames.frame(src_frame).bound_version == Some(version),
            self.frames.frame(src_frame).cert == Some(src_cert),
            self.frames.returned_reads(src_frame, src_cap, Ticket::NONE, self.kernel_view()),
    {
        assert(dst_frame != src_frame);
        if src_frame.0 < dst_frame.0 {
            assert(self.frames.ghosts_spec()@[src_frame.0 as nat].slot.loc
                != self.frames.ghosts_spec()@[dst_frame.0 as nat].slot.loc);
        } else {
            assert(self.frames.ghosts_spec()@[dst_frame.0 as nat].slot.loc
                != self.frames.ghosts_spec()@[src_frame.0 as nat].slot.loc);
        }
        assert(self.frames.published(src_frame) == after_lease.frames.published(src_frame));
        assert(self.frames.valid_frame(src_frame, self.kernel_view()));
        assert(self.frames.valid_frame(dst_frame, self.kernel_view()));
        assert forall|g: FrameId| g.0 < self.frames.metas.len() implies {
            &&& (#[trigger] self.frames.frame(g)).slot == before.frames.frame(g).slot
            &&& self.frames.frame(g).tier == before.frames.frame(g).tier
            &&& self.frames.published(g) == before.frames.published(g)
        } by {
            assert(after_lease.frames.frame(g).slot == before.frames.frame(g).slot);
        }
        assert forall|g: FrameId|
            g.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(g)).state
                == STATE_INITIALIZING implies self.frames.meta(g) == before.frames.meta(g)
            && self.frames.frame(g) == before.frames.frame(g) by {
            assert(after_lease.frames.meta(g) == before.frames.meta(g));
        }
        assert forall|g: FrameId| g.0 < before.frames.metas.len() implies {
            &&& (#[trigger] self.frames.frame(g)).fenced_shares == before.frames.frame(
                g,
            ).fenced_shares
            &&& self.frames.meta(g).active_fenced_leases == before.frames.meta(
                g,
            ).active_fenced_leases
        } by {
            assert(after_lease.frames.frame(g).fenced_shares == before.frames.frame(
                g,
            ).fenced_shares);
            assert(after_lease.frames.meta(g).active_fenced_leases == before.frames.meta(
                g,
            ).active_fenced_leases);
        }
    }

    proof fn lemma_copy_hop_rolled_back(
        &self,
        before: Self,
        after_lease: Self,
        src_frame: FrameId,
        lease_slot: usize,
        share: TransferShare,
    )
        requires
            before.well_formed(),
            after_lease.well_formed(),
            self.well_formed(),
            after_lease.timelines == before.timelines,
            self.timelines.only_watermarks(&after_lease.timelines),
            after_lease.frames.same_directory(&before.frames),
            self.frames.same_directory(&after_lease.frames),
            after_lease.frames.preserves_initializing(&before.frames, None),
            self.frames.preserves_initializing(&after_lease.frames, None),
            after_lease.frames.preserves_fenced_leases(&before.frames),
            self.frames.preserves_fenced_leases(&after_lease.frames),
            src_frame.0 < before.frames.metas.len(),
            after_lease.frames.metas.len() == before.frames.metas.len(),
            self.frames.metas.len() == before.frames.metas.len(),
            after_lease.frames.same_other_pins(&before.frames, Some(src_frame)),
            self.frames.same_other_pins(&after_lease.frames, Some(src_frame)),
            !before.frames.frame(src_frame).local_shares.contains_key(share),
            after_lease.frames.frame(src_frame).local_shares == before.frames.frame(
                src_frame,
            ).local_shares.insert(share, LocalShareEntry { share }),
            self.frames.frame(src_frame).local_shares == after_lease.frames.frame(
                src_frame,
            ).local_shares.remove(share),
            after_lease.frames.frame(src_frame).fenced_shares == before.frames.frame(
                src_frame,
            ).fenced_shares,
            self.frames.frame(src_frame).fenced_shares == after_lease.frames.frame(
                src_frame,
            ).fenced_shares,
            after_lease.frames.meta(src_frame).active_fenced_leases == before.frames.meta(
                src_frame,
            ).active_fenced_leases,
            self.frames.meta(src_frame).active_fenced_leases == after_lease.frames.meta(
                src_frame,
            ).active_fenced_leases,
            after_lease.active_leases.len() == before.active_leases.len(),
            self.active_leases.len() == after_lease.active_leases.len(),
            lease_slot < before.active_leases.len(),
            before.active_leases[lease_slot as int].is_none(),
            self.active_leases[lease_slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases.len() && s != lease_slot as int
                    ==> #[trigger] after_lease.active_leases[s] == before.active_leases[s],
            forall|s: int|
                0 <= s < after_lease.active_leases.len() && s != lease_slot as int
                    ==> #[trigger] self.active_leases[s] == after_lease.active_leases[s],
        ensures
            self.timelines.only_watermarks(&before.timelines),
            self.frames.same_directory(&before.frames),
            self.frames.preserves_initializing(&before.frames, None),
            self.frames.preserves_fenced_leases(&before.frames),
            self.frames.same_other_pins(&before.frames, None),
            self.active_leases@ == before.active_leases@,
    {
        assert(self.active_leases@ =~= before.active_leases@) by {
            assert forall|s: int| 0 <= s < before.active_leases.len() implies self.active_leases[s]
                == before.active_leases[s] by {
                if s != lease_slot as int {
                    assert(after_lease.active_leases[s] == before.active_leases[s]);
                    assert(self.active_leases[s] == after_lease.active_leases[s]);
                }
            }
        }
        assert(self.frames.frame(src_frame).local_shares =~= before.frames.frame(
            src_frame,
        ).local_shares);
        assert forall|g: FrameId| g.0 < self.frames.metas.len() implies {
            &&& (#[trigger] self.frames.frame(g)).slot == before.frames.frame(g).slot
            &&& self.frames.frame(g).tier == before.frames.frame(g).tier
            &&& self.frames.published(g) == before.frames.published(g)
            &&& self.frames.frame(g).local_shares == before.frames.frame(g).local_shares
            &&& self.frames.frame(g).fenced_shares == before.frames.frame(g).fenced_shares
            &&& self.frames.meta(g).active_fenced_leases == before.frames.meta(
                g,
            ).active_fenced_leases
        } by {
            assert(after_lease.frames.frame(g).slot == before.frames.frame(g).slot);
            if g != src_frame {
                assert(after_lease.frames.frame(g).local_shares == before.frames.frame(
                    g,
                ).local_shares);
                assert(self.frames.frame(g).local_shares == after_lease.frames.frame(
                    g,
                ).local_shares);
                assert(after_lease.frames.frame(g).fenced_shares == before.frames.frame(
                    g,
                ).fenced_shares);
                assert(self.frames.frame(g).fenced_shares == after_lease.frames.frame(
                    g,
                ).fenced_shares);
                assert(after_lease.frames.meta(g).active_fenced_leases == before.frames.meta(
                    g,
                ).active_fenced_leases);
                assert(self.frames.meta(g).active_fenced_leases == after_lease.frames.meta(
                    g,
                ).active_fenced_leases);
            }
        }
        assert forall|g: FrameId|
            g.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(g)).state
                == STATE_INITIALIZING implies self.frames.meta(g) == before.frames.meta(g)
            && self.frames.frame(g) == before.frames.frame(g) by {
            assert(after_lease.frames.meta(g) == before.frames.meta(g));
        }
        assert forall|g: FrameId| g.0 < before.frames.metas.len() implies {
            &&& (#[trigger] self.frames.frame(g)).fenced_shares == before.frames.frame(
                g,
            ).fenced_shares
            &&& self.frames.meta(g).active_fenced_leases == before.frames.meta(
                g,
            ).active_fenced_leases
        } by {
            assert(after_lease.frames.frame(g).fenced_shares == before.frames.frame(
                g,
            ).fenced_shares);
            assert(after_lease.frames.meta(g).active_fenced_leases == before.frames.meta(
                g,
            ).active_fenced_leases);
        }
    }

    fn try_reserve_copy_dst(
        &mut self,
        src_frame: FrameId,
        src_tier: TierId,
        src_lease: ReadLeaseHandle,
        dst_tier: TierId,
        key: ObjectKey,
        version: VersionId,
        Ghost(src_cert): Ghost<ValueCertificate>,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(instance): Tracked<&TransferInstance>,
        Ghost(before): Ghost<Self>,
    ) -> (r: Result<(CopyHopHandle, Tracked<ShareCap>, Tracked<SlotToken>), CoreError>)
        requires
            before.well_formed(),
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(self).node_id == before.node_id,
            old(self).tiers == before.tiers,
            old(self).timelines == before.timelines,
            old(self).catalog == before.catalog,
            old(self).ledger == before.ledger,
            old(self).version_seq_start == before.version_seq_start,
            old(self).next_version_seq == before.next_version_seq,
            old(self).version_seq_limit == before.version_seq_limit,
            old(self).spec == before.spec,
            old(self).version_registry == before.version_registry,
            old(self).active_leases.len() == before.active_leases.len(),
            (src_lease.slot as usize) < before.active_leases.len(),
            before.active_leases[src_lease.slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases.len() && s != src_lease.slot as int
                    ==> #[trigger] old(self).active_leases[s] == before.active_leases[s],
            old(self).free_lease_slots.len() + 1 == before.free_lease_slots.len(),
            old(self).frames.same_directory(&before.frames),
            old(self).frames.preserves_initializing(&before.frames, None),
            old(self).frames.same_other_pins(&before.frames, Some(src_frame)),
            old(self).frames.preserves_fenced_leases(&before.frames),
            src_frame.0 < before.frames.metas.len(),
            old(self).frames.meta(src_frame).state == STATE_PUBLISHED,
            old(self).frames.frame(src_frame).tier == src_tier,
            old(self).frames.tier_frame_size(src_tier) == old(self).frames.tier_frame_size(
                dst_tier,
            ),
            old(self).frames.frame(src_frame).bound_key == Some(key),
            old(self).frames.frame(src_frame).bound_version == Some(version),
            old(self).frames.frame(src_frame).cert == Some(src_cert),
            replica_valid(old(self).ledger@, old(self).node(), src_cert),
            old(self).citations().contains(src_cert.writer),
            old(self).frames.versions_spec()@.contains_key(version),
            old(self).frames.versions_spec()@[version] == src_cert.bytes,
            version.well_formed() && version != VersionId::NONE,
            version.origin_node != old(self).node_id || version.seq < old(self).next_version_seq
                || version.seq >= old(self).version_seq_limit,
            src_lease.frame == src_frame,
            old(self).active_lease_matches(src_lease),
            old(self).active_leases[src_lease.slot as int] == Some(
                ActiveLocalLease {
                    internal: true,
                    handle: src_lease,
                    share: Ghost(src_cap.share()),
                },
            ),
            !before.frames.frame(src_frame).local_shares.contains_key(src_cap.share()),
            old(self).frames.frame(src_frame).local_shares == before.frames.frame(
                src_frame,
            ).local_shares.insert(src_cap.share(), LocalShareEntry { share: src_cap.share() }),
            old(self).frames.frame(src_frame).fenced_shares == before.frames.frame(
                src_frame,
            ).fenced_shares,
            old(self).frames.meta(src_frame).active_fenced_leases == before.frames.meta(
                src_frame,
            ).active_fenced_leases,
            src_cap.is_cooperative(),
            src_cap.instance_id() == old(self).instance_id(),
            src_cap.loc() == old(self).frames.frame(src_frame).slot.loc,
            src_cap.frontier() == (Frontier {
                last_writer: Some(src_cert.writer),
                readers: ISet::empty(),
            }),
            old(self).frames.returned_reads(
                src_frame,
                src_cap,
                Ticket::NONE,
                old(self).kernel_view(),
            ),
        ensures
            final(self).well_formed(),
            final(self).node_id == before.node_id,
            final(self).tiers == before.tiers,
            final(self).catalog == before.catalog,
            final(self).ledger == before.ledger,
            final(self).version_seq_start == before.version_seq_start,
            final(self).next_version_seq == before.next_version_seq,
            final(self).version_seq_limit == before.version_seq_limit,
            final(self).spec == before.spec,
            final(self).version_registry == before.version_registry,
            final(self).timelines.only_watermarks(&before.timelines),
            final(self).frames.same_directory(&before.frames),
            final(self).frames.preserves_initializing(&before.frames, None),
            final(self).frames.preserves_fenced_leases(&before.frames),
            r matches Err(_) ==> {
                &&& final(self).active_leases@ == before.active_leases@
                &&& final(self).free_lease_slots.len() == before.free_lease_slots.len()
                &&& final(self).frames.same_other_pins(&before.frames, None)
            },
            r matches Ok((hop, out_cap, dst_tok)) ==> {
                &&& hop.src_frame == src_frame
                &&& hop.dst_tier == dst_tier
                &&& hop.src_frame != hop.dst_frame
                &&& hop.src_frame.0 < final(self).frames.metas.len()
                &&& hop.dst_frame.0 < final(self).frames.metas.len()
                &&& before.frames.meta(hop.dst_frame).state != STATE_INITIALIZING
                &&& hop.src_slot@ == final(self).frames.frame(hop.src_frame).slot
                &&& hop.dst_slot@ == final(self).frames.frame(hop.dst_frame).slot
                &&& hop.src_slot@.slot_len == hop.dst_slot@.slot_len
                &&& hop.src_slot@.loc != hop.dst_slot@.loc
                &&& final(self).frames.meta(hop.src_frame).state == STATE_PUBLISHED
                &&& final(self).frames.frame(hop.src_frame).tier == hop.src_tier
                &&& final(self).frames.tier_frame_size(hop.src_tier)
                    == final(self).frames.tier_frame_size(hop.dst_tier)
                &&& final(self).frames.frame(hop.src_frame).bound_key == Some(hop.key)
                &&& final(self).frames.frame(hop.src_frame).bound_version == Some(hop.version)
                &&& final(self).frames.frame(hop.src_frame).cert == Some(hop.src_cert@)
                &&& replica_valid(final(self).ledger@, final(self).node(), hop.src_cert@)
                &&& final(self).citations().contains(hop.src_cert@.writer)
                &&& final(self).frames.versions_spec()@.contains_key(hop.version)
                &&& final(self).frames.versions_spec()@[hop.version] == hop.src_cert@.bytes
                &&& (hop.version.origin_node != final(self).node_id || hop.version.seq
                    < final(self).next_version_seq || hop.version.seq
                    >= final(self).version_seq_limit)
                &&& hop.src_lease.frame == src_frame
                &&& final(self).active_leases.len() == before.active_leases.len()
                &&& before.active_leases[hop.src_lease.slot as int].is_none()
                &&& forall|s: int|
                    0 <= s < before.active_leases.len() && s != hop.src_lease.slot as int
                        ==> #[trigger] final(self).active_leases[s] == before.active_leases[s]
                &&& final(self).active_lease_matches(hop.src_lease)
                &&& final(self).active_leases[hop.src_lease.slot as int] == Some(
                    ActiveLocalLease {
                        internal: true,
                        handle: hop.src_lease,
                        share: Ghost(out_cap@.share()),
                    },
                )
                &&& final(self).free_lease_slots.len() + 1 == before.free_lease_slots.len()
                &&& final(self).frames.meta(hop.dst_frame).state == STATE_INITIALIZING
                &&& !final(self).frames.meta(hop.dst_frame).queued()
                &&& final(self).frames.frame(hop.dst_frame).tier == dst_tier
                &&& final(self).frames.frame(hop.dst_frame).bound_key == Some(hop.key)
                &&& final(self).frames.frame(hop.dst_frame).bound_version == Some(hop.version)
                &&& final(self).frames.frame(hop.dst_frame).token is None
                &&& out_cap@.is_cooperative()
                &&& out_cap@.instance_id() == final(self).instance_id()
                &&& out_cap@.loc() == hop.src_slot@.loc
                &&& out_cap@.frontier() == (Frontier {
                    last_writer: Some(hop.src_cert@.writer),
                    readers: ISet::empty(),
                })
                &&& read_admitted(out_cap@.frontier(), final(self).citations())
                &&& slot_is_owned(dst_tok@, final(self).instance_id(), hop.dst_slot@.loc)
                &&& write_admitted(slot_frontier(dst_tok@), final(self).citations())
                &&& final(self).frames.returned_reads(
                    src_frame,
                    out_cap@,
                    Ticket::NONE,
                    final(self).kernel_view(),
                )
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(FrameTable::same_directory);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::same_other_pins);
        hide(FrameTable::preserves_fenced_leases);
        let ghost after_lease = *self;
        proof {
            assert(after_lease.frames.metas.len() == before.frames.metas.len()
                && after_lease.frames.tier_frame_sizes == before.frames.tier_frame_sizes) by {
                reveal(FrameTable::same_directory);
            };
        }
        let ghost src_slot = self.frames.frame(src_frame).slot;
        match self.begin_write_version(dst_tier, key, version) {
            Ok((dst_handle, dst_tok)) => {
                proof {
                    assert(self.frames.metas.len() == after_lease.frames.metas.len()
                        && self.frames.versions == after_lease.frames.versions
                        && self.frames.tier_frame_sizes == after_lease.frames.tier_frame_sizes) by {
                        reveal(FrameTable::same_directory);
                    };
                    self.lemma_copy_hop_reserved(
                        before,
                        after_lease,
                        src_frame,
                        dst_handle.frame,
                        key,
                        version,
                        src_cert,
                        src_cap,
                    );
                }
                Ok(
                    (
                        CopyHopHandle {
                            src_frame,
                            src_tier,
                            src_slot: Ghost(src_slot),
                            src_lease,
                            dst_frame: dst_handle.frame,
                            dst_tier,
                            dst_slot: dst_handle.slot,
                            key,
                            version,
                            src_cert: Ghost(src_cert),
                        },
                        Tracked(src_cap),
                        dst_tok,
                    ),
                )
            },
            Err(err) => {
                let ghost share = src_cap.share();
                self.release_local_lease(
                    src_lease,
                    Tracked(src_cap),
                    Ticket::NONE,
                    Tracked(instance),
                ).unwrap();
                proof {
                    assert(self.frames.metas.len() == after_lease.frames.metas.len()) by {
                        reveal(FrameTable::same_directory);
                    };
                    self.lemma_copy_hop_rolled_back(
                        before,
                        after_lease,
                        src_frame,
                        src_lease.slot as usize,
                        share,
                    );
                }
                Err(err)
            },
        }
    }

    /// Atomically acquire a local `ShareCap` on `src_frame` and reserve a ready
    /// destination frame on `dst_tier` with its exclusive `SlotToken` checked out.
    /// Rolls back the source lease atomically if destination allocation fails.
    pub fn begin_copy_hop(
        &mut self,
        src_frame: FrameId,
        dst_tier: TierId,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(CopyHopHandle, Tracked<ShareCap>, Tracked<SlotToken>), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).instance_id(),
            old(shares).node() == old(self).node(),
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).timelines_spec().only_watermarks(&old(self).timelines_spec()),
            final(self).frames_spec().same_directory(&old(self).frames_spec()),
            final(self).frames_spec().preserves_initializing(&old(self).frames_spec(), None),
            final(self).frames_spec().preserves_fenced_leases(&old(self).frames_spec()),
            final(shares).well_formed(),
            final(shares).id() == old(self).instance_id(),
            final(shares).node() == old(self).node(),
            r matches Err(_) ==> {
                &&& final(self).active_leases_spec()@ == old(self).active_leases_spec()@
                &&& final(self).free_lease_slots_spec().len() == old(
                    self,
                ).free_lease_slots_spec().len()
                &&& final(self).frames_spec().same_other_pins(&old(self).frames_spec(), None)
            },
            r matches Ok((hop, src_cap, dst_tok)) ==> {
                &&& hop.src_frame == src_frame
                &&& hop.dst_tier == dst_tier
                &&& hop.src_frame != hop.dst_frame
                &&& hop.src_frame.0 < final(self).frames_spec().metas_spec().len()
                &&& hop.dst_frame.0 < final(self).frames_spec().metas_spec().len()
                &&& old(self).frames_spec().meta(hop.dst_frame).state != STATE_INITIALIZING
                &&& hop.src_slot@ == final(self).frames_spec().frame(hop.src_frame).slot
                &&& hop.dst_slot@ == final(self).frames_spec().frame(hop.dst_frame).slot
                &&& hop.src_slot@.slot_len == hop.dst_slot@.slot_len
                &&& hop.src_slot@.loc != hop.dst_slot@.loc
                &&& final(self).frames_spec().meta(hop.src_frame).state == STATE_PUBLISHED
                &&& final(self).frames_spec().frame(hop.src_frame).tier == hop.src_tier
                &&& final(self).frames_spec().tier_frame_size(hop.src_tier)
                    == final(self).frames_spec().tier_frame_size(hop.dst_tier)
                &&& final(self).frames_spec().frame(hop.src_frame).bound_key == Some(hop.key)
                &&& final(self).frames_spec().frame(hop.src_frame).bound_version == Some(
                    hop.version,
                )
                &&& final(self).frames_spec().frame(hop.src_frame).cert == Some(hop.src_cert@)
                &&& replica_valid(final(self).ledger_spec()@, final(self).node(), hop.src_cert@)
                &&& final(self).citations().contains(hop.src_cert@.writer)
                &&& final(self).frames_spec().versions_spec()@.contains_key(hop.version)
                &&& final(self).frames_spec().versions_spec()@[hop.version] == hop.src_cert@.bytes
                &&& (hop.version.origin_node != final(self).node_id_spec() || hop.version.seq
                    < final(self).next_version_seq_spec() || hop.version.seq
                    >= final(self).version_seq_limit_spec())
                &&& hop.src_lease.frame_spec() == src_frame
                &&& final(self).active_leases_spec().len() == old(self).active_leases_spec().len()
                &&& old(self).active_leases_spec()[hop.src_lease.slot_spec() as int].is_none()
                &&& forall|s: int|
                    0 <= s < old(self).active_leases_spec().len() && s
                        != hop.src_lease.slot_spec() as int
                        ==> #[trigger] final(self).active_leases_spec()[s] == old(
                        self,
                    ).active_leases_spec()[s]
                &&& final(self).active_lease_matches(hop.src_lease)
                &&& final(self).active_leases_spec()[hop.src_lease.slot_spec() as int] == Some(
                    ActiveLocalLease {
                        internal: true,
                        handle: hop.src_lease,
                        share: Ghost(src_cap@.share()),
                    },
                )
                &&& final(self).free_lease_slots_spec().len() + 1 == old(
                    self,
                ).free_lease_slots_spec().len()
                &&& final(self).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING
                &&& !final(self).frames_spec().meta(hop.dst_frame).queued()
                &&& final(self).frames_spec().frame(hop.dst_frame).tier == dst_tier
                &&& final(self).frames_spec().frame(hop.dst_frame).bound_key == Some(hop.key)
                &&& final(self).frames_spec().frame(hop.dst_frame).bound_version == Some(
                    hop.version,
                )
                &&& final(self).frames_spec().frame(hop.dst_frame).token is None
                &&& src_cap@.is_cooperative()
                &&& src_cap@.instance_id() == final(self).instance_id()
                &&& src_cap@.loc() == hop.src_slot@.loc
                &&& src_cap@.frontier() == (Frontier {
                    last_writer: Some(hop.src_cert@.writer),
                    readers: ISet::empty(),
                })
                &&& read_admitted(src_cap@.frontier(), final(self).citations())
                &&& slot_is_owned(dst_tok@, final(self).instance_id(), hop.dst_slot@.loc)
                &&& write_admitted(slot_frontier(dst_tok@), final(self).citations())
                &&& final(self).frames_spec().returned_reads(
                    src_frame,
                    src_cap@,
                    Ticket::NONE,
                    final(self).kernel_view(),
                )
            },
    {
        if dst_tier.as_index() >= self.tiers.len() {
            return Err(CoreError::UnsupportedTier);
        }
        if src_frame.0 as usize >= self.frames.metas.len()
            || self.frames.metas[src_frame.0 as usize].state != STATE_PUBLISHED {
            return Err(CoreError::StaleReplica);
        }
        let (_, src_tier) = self.frames.bindings[src_frame.0 as usize];
        proof {
            assert(self.frames.valid_frame(src_frame, self.kernel_view()));
        }
        if self.frames.tier_size(src_tier) != self.frames.tier_size(dst_tier) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = *self;
        let (src_lease, version, Ghost(src_cert), Tracked(src_cap)) = self.acquire_local_lease(
            src_frame,
            true,
            Tracked(shares),
            Tracked(instance),
        )?;
        let (key, src_tier) = self.frames.bindings[src_frame.0 as usize];
        proof {
            self.lemma_published_binding(src_frame);
        }
        self.try_reserve_copy_dst(
            src_frame,
            src_tier,
            src_lease,
            dst_tier,
            key,
            version,
            Ghost(src_cert),
            Tracked(src_cap),
            Tracked(instance),
            Ghost(before),
        )
    }

    /// Release the source local `ShareCap`, re-install the destination `SlotToken`,
    /// and publish `hop.dst_frame` into the catalog atomically.
    pub fn finish_copy_hop(
        &mut self,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        src_read_ticket: Ticket,
        Tracked(dst_tok): Tracked<SlotToken>,
        dst_write_ticket: Ticket,
        Ghost(dst_cert): Ghost<ValueCertificate>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(self).active_lease_matches(hop.src_lease),
            src_cap.share() == old(
                self,
            ).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@,
            old(self).timelines_spec().valid_ticket(src_read_ticket),
            old(self).frames_spec().returned_reads(
                hop.src_lease.frame_spec(),
                src_cap,
                src_read_ticket,
                old(self).kernel_view(),
            ),
            hop.dst_frame.0 < old(self).frames_spec().metas_spec().len(),
            old(self).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(self).frames_spec().meta(hop.dst_frame).queued(),
            old(self).frames_spec().frame(hop.dst_frame).token is None,
            old(self).frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            slot_is_owned(dst_tok, old(self).instance_id(), hop.dst_slot@.loc),
            slot_frontier(dst_tok).readers.is_empty(),
            slot_frontier(dst_tok).last_writer == Some(dst_cert.writer),
            old(self).frames_spec().versions_spec()@.contains_key(
                old(self).frames_spec().frame(hop.dst_frame).bound_version.unwrap(),
            ) ==> old(self).frames_spec().versions_spec()@[old(self).frames_spec().frame(
                hop.dst_frame,
            ).bound_version.unwrap()] == dst_cert.bytes,
            old(self).frames_spec().frame(hop.dst_frame).bound_version.unwrap().origin_node != old(
                self,
            ).node_id_spec() || old(self).frames_spec().frame(
                hop.dst_frame,
            ).bound_version.unwrap().seq < old(self).next_version_seq_spec() || old(
                self,
            ).frames_spec().frame(hop.dst_frame).bound_version.unwrap().seq >= old(
                self,
            ).version_seq_limit_spec(),
            dst_cert.slot == hop.dst_slot@,
            dst_cert.bytes.len() == dst_cert.slot.slot_len,
            old(self).timelines_spec().issued(
                dst_write_ticket.timeline_id(),
                dst_write_ticket.sequence(),
            ),
            dst_write_ticket != Ticket::NONE,
            dst_cert.writer == old(self).timelines_spec().events_spec()@[(
                dst_write_ticket.timeline_id(),
                dst_write_ticket.sequence(),
            )],
            replica_valid(old(self).ledger_spec()@, old(self).node(), dst_cert),
            old(self).citations().contains(dst_cert.writer),
        ensures
            final(self).well_formed(),
            old(self).spec_spec()@.versions_monotonic(final(self).spec_spec()@),
            final(self).spec_spec()@.versions == old(self).spec_spec()@.versions.insert(
                old(self).frames_spec().frame(hop.dst_frame).bound_version.unwrap(),
                dst_cert.bytes,
            ),
            final(self).spec_spec()@.catalog == old(self).spec_spec()@.catalog.insert(
                old(self).frames_spec().frame(hop.dst_frame).bound_key.unwrap(),
                old(self).frames_spec().frame(hop.dst_frame).bound_version.unwrap(),
            ),
            final(self).frames_spec().meta(hop.dst_frame).state == STATE_PUBLISHED,
            final(self).frames_spec().frame(hop.dst_frame).cert == Some(dst_cert),
            final(self).frames_spec().same_layout(&old(self).frames_spec()),
            final(self).frames_spec().preserves_initializing(
                &old(self).frames_spec(),
                Some(hop.dst_frame),
            ),
            final(self).frames_spec().preserves_fenced_leases(&old(self).frames_spec()),
            final(self).catalog_spec().slot(
                old(self).frames_spec().frame(hop.dst_frame).bound_key.unwrap(),
                old(self).frames_spec().frame(hop.dst_frame).tier.index() as int,
            ) == Some(hop.dst_frame),
            final(self).active_leases_spec().len() == old(self).active_leases_spec().len(),
            final(self).active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
            forall|s: int|
                0 <= s < old(self).active_leases_spec().len() && s
                    != hop.src_lease.slot_spec() as int
                    ==> #[trigger] final(self).active_leases_spec()[s] == old(
                    self,
                ).active_leases_spec()[s],
            final(self).free_lease_slots_spec().len() == old(self).free_lease_slots_spec().len()
                + 1,
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).timelines_spec().only_watermarks(&old(self).timelines_spec()),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
    {
        self.release_local_lease(
            hop.src_lease,
            Tracked(src_cap),
            src_read_ticket,
            Tracked(instance),
        ).unwrap();
        let write_handle = WriteHandle {
            frame: hop.dst_frame,
            tier: hop.dst_tier,
            key: hop.key,
            version: hop.version,
            slot: hop.dst_slot,
        };
        self.finish_write(write_handle, Tracked(dst_tok), dst_write_ticket, Ghost(dst_cert));
    }

    /// Release the source local `ShareCap` and abort the destination frame back
    /// to `ready_queue` atomically.
    pub fn abort_copy_hop(
        &mut self,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        src_read_ticket: Ticket,
        Tracked(dst_tok): Tracked<SlotToken>,
        Tracked(instance): Tracked<&TransferInstance>,
    )
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(self).active_lease_matches(hop.src_lease),
            src_cap.share() == old(
                self,
            ).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@,
            old(self).timelines_spec().valid_ticket(src_read_ticket),
            old(self).frames_spec().returned_reads(
                hop.src_lease.frame_spec(),
                src_cap,
                src_read_ticket,
                old(self).kernel_view(),
            ),
            hop.dst_frame.0 < old(self).frames_spec().metas_spec().len(),
            old(self).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(self).frames_spec().meta(hop.dst_frame).queued(),
            old(self).frames_spec().frame(hop.dst_frame).tier == hop.dst_tier,
            old(self).frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            old(self).frames_spec().frame(hop.dst_frame).token is None,
            slot_is_owned(dst_tok, old(self).instance_id(), hop.dst_slot@.loc),
            write_admitted(slot_frontier(dst_tok), old(self).citations()),
        ensures
            final(self).well_formed(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).frames_spec().same_directory(&old(self).frames_spec()),
            final(self).frames_spec().preserves_initializing(
                &old(self).frames_spec(),
                Some(hop.dst_frame),
            ),
            final(self).frames_spec().preserves_fenced_leases(&old(self).frames_spec()),
            final(self).frames_spec().meta(hop.dst_frame).state == STATE_EMPTY
                && final(self).frames_spec().meta(hop.dst_frame).queued(),
            final(self).active_leases_spec().len() == old(self).active_leases_spec().len(),
            final(self).active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
            forall|s: int|
                0 <= s < old(self).active_leases_spec().len() && s
                    != hop.src_lease.slot_spec() as int
                    ==> #[trigger] final(self).active_leases_spec()[s] == old(
                    self,
                ).active_leases_spec()[s],
            final(self).free_lease_slots_spec().len() == old(self).free_lease_slots_spec().len()
                + 1,
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).timelines_spec().only_watermarks(&old(self).timelines_spec()),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
    {
        self.release_local_lease(
            hop.src_lease,
            Tracked(src_cap),
            src_read_ticket,
            Tracked(instance),
        ).unwrap();
        let write_handle = WriteHandle {
            frame: hop.dst_frame,
            tier: hop.dst_tier,
            key: hop.key,
            version: hop.version,
            slot: hop.dst_slot,
        };
        self.abort_write(write_handle, Tracked(dst_tok));
    }
}

} // verus!
