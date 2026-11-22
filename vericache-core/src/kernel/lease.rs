#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    proof fn lemma_leases_after_acquire(
        old_core: &CacheCore,
        new_core: &CacheCore,
        f: FrameId,
        slot: u32,
        handle: ReadLeaseHandle,
        internal: bool,
        share: TransferShare,
    )
        requires
            old_core.leases_valid(),
            old_core.free_lease_slots.len() > 0,
            old_core.lease_generation < u64::MAX,
            slot == old_core.free_lease_slots@[old_core.free_lease_slots.len() - 1],
            new_core.free_lease_slots@ == old_core.free_lease_slots@.drop_last(),
            new_core.lease_generation == old_core.lease_generation + 1,
            handle == (ReadLeaseHandle { slot, generation: new_core.lease_generation, frame: f }),
            new_core.active_leases@ == old_core.active_leases@.update(
                slot as int,
                Some(ActiveLocalLease { internal, handle, share: Ghost(share) }),
            ),
            f.0 < old_core.frames.metas.len(),
            new_core.frames.metas.len() == old_core.frames.metas.len(),
            !old_core.frames.frame(f).local_shares.contains_key(share),
            new_core.frames.frame(f).local_shares == old_core.frames.frame(f).local_shares.insert(
                share,
                LocalShareEntry { share },
            ),
            forall|g: FrameId|
                g != f && g.0 < old_core.frames.metas.len() ==> (#[trigger] new_core.frames.frame(
                    g,
                )).local_shares == old_core.frames.frame(g).local_shares,
        ensures
            new_core.leases_valid(),
    {
        assert(!new_core.free_lease_slots@.contains(slot)) by {
            if new_core.free_lease_slots@.contains(slot) {
                let i = choose|i: int|
                    0 <= i < new_core.free_lease_slots.len() && new_core.free_lease_slots[i]
                        == slot;
                assert(old_core.free_lease_slots[i] == slot);
                assert(i != old_core.free_lease_slots.len() - 1);
                assert(!old_core.free_lease_slots@.no_duplicates());
            }
        }
        assert(new_core.free_lease_slots@.no_duplicates()) by {
            assert forall|i: int, j: int|
                0 <= i < new_core.free_lease_slots.len() && 0 <= j < new_core.free_lease_slots.len()
                    && i != j implies new_core.free_lease_slots[i]
                != new_core.free_lease_slots[j] by {
                assert(new_core.free_lease_slots[i] == old_core.free_lease_slots[i]);
                assert(new_core.free_lease_slots[j] == old_core.free_lease_slots[j]);
            }
        }
        assert forall|i: int| 0 <= i < new_core.free_lease_slots.len() implies {
            let s = #[trigger] new_core.free_lease_slots[i];
            (s as usize) < new_core.active_leases.len()
                && new_core.active_leases[s as int].is_none()
        } by {
            let s = new_core.free_lease_slots[i];
            assert(s == old_core.free_lease_slots[i]);
            assert(s != slot);
            assert(old_core.active_leases[s as int].is_none());
        }
        assert forall|s: int|
            0 <= s
                < new_core.active_leases.len() implies match #[trigger] new_core.active_leases[s] {
            None => new_core.free_lease_slots@.contains(s as u32),
            Some(lease) => {
                &&& !new_core.free_lease_slots@.contains(s as u32)
                &&& lease.handle.slot as int == s
                &&& 1 <= lease.handle.generation <= new_core.lease_generation
                &&& lease.handle.frame.0 < new_core.frames.metas.len()
                &&& new_core.frames.frame(lease.handle.frame).local_shares.contains_key(
                    lease.share@,
                )
            },
        } by {
            if s != slot as int {
                assert(new_core.active_leases[s] == old_core.active_leases[s]);
                if new_core.active_leases[s].is_none() {
                    let idx = choose|idx: int|
                        0 <= idx < old_core.free_lease_slots.len() && old_core.free_lease_slots[idx]
                            == s as u32;
                    assert(idx < new_core.free_lease_slots.len());
                    assert(new_core.free_lease_slots[idx] == s as u32);
                } else {
                    let lease = new_core.active_leases[s].unwrap();
                    if new_core.free_lease_slots@.contains(s as u32) {
                        let idx = choose|idx: int|
                            0 <= idx < new_core.free_lease_slots.len()
                                && new_core.free_lease_slots[idx] == s as u32;
                        assert(old_core.free_lease_slots[idx] == s as u32);
                    }
                    assert(old_core.frames.frame(lease.handle.frame).local_shares.contains_key(
                        lease.share@,
                    ));
                }
            }
        }
        assert forall|s1: int, s2: int|
            0 <= s1 < new_core.active_leases.len() && 0 <= s2 < new_core.active_leases.len() && s1
                != s2 implies match (#[trigger]
            new_core.active_leases[s1],
            #[trigger] new_core.active_leases[s2],
        ) {
            (Some(l1), Some(l2)) => l1.handle.frame == l2.handle.frame ==> l1.share@ != l2.share@,
            _ => true,
        } by {
            match (new_core.active_leases[s1], new_core.active_leases[s2]) {
                (Some(l1), Some(l2)) => {
                    if s1 == slot as int && l1.handle.frame == l2.handle.frame {
                        assert(old_core.frames.frame(f).local_shares.contains_key(l2.share@));
                    } else if s2 == slot as int && l1.handle.frame == l2.handle.frame {
                        assert(old_core.frames.frame(f).local_shares.contains_key(l1.share@));
                    }
                },
                _ => {},
            }
        }
    }

    pub fn acquire_local_lease(
        &mut self,
        f: FrameId,
        internal: bool,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<
        (ReadLeaseHandle, VersionId, Ghost<ValueCertificate>, Tracked<ShareCap>),
        CoreError,
    >)
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).instance_id(),
            old(shares).node() == old(self).node(),
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
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).active_leases.len() == old(self).active_leases.len(),
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, Some(f)),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            final(shares).well_formed(),
            final(shares).id() == old(self).instance_id(),
            final(shares).node() == old(self).node(),
            match r {
                Err(_) => *final(self) == *old(self) && *final(shares) == *old(shares),
                Ok((handle, version, cert, cap)) => {
                    &&& f.0 < old(self).frames.metas.len()
                    &&& old(self).frames.meta(f).state == STATE_PUBLISHED
                    &&& final(self).frames.meta(f).state == STATE_PUBLISHED
                    &&& handle.frame == f
                    &&& (handle.slot as usize) < final(self).active_leases.len()
                    &&& old(self).active_leases[handle.slot as int].is_none()
                    &&& final(self).active_lease_matches(handle)
                    &&& final(self).active_leases[handle.slot as int] == Some(
                        ActiveLocalLease { internal, handle, share: Ghost(cap@.share()) },
                    )
                    &&& forall|s: int|
                        0 <= s < old(self).active_leases.len() && s != handle.slot as int
                            ==> #[trigger] final(self).active_leases[s] == old(
                            self,
                        ).active_leases[s]
                    &&& final(self).free_lease_slots.len() + 1 == old(self).free_lease_slots.len()
                    &&& final(self).frames.frame(f).bound_version == Some(version)
                    &&& final(self).frames.frame(f).cert == Some(cert@)
                    &&& version.well_formed() && version != VersionId::NONE
                    &&& replica_valid(final(self).ledger@, final(self).node(), cert@)
                    &&& final(self).citations().contains(cert@.writer)
                    &&& final(self).frames.versions@.contains_key(version)
                    &&& final(self).frames.versions@[version] == cert@.bytes
                    &&& cap@.is_local()
                    &&& cap@.instance_id() == final(self).instance_id()
                    &&& cap@.loc() == final(self).frames.frame(f).slot.loc
                    &&& cap@.frontier() == (Frontier {
                        last_writer: Some(cert@.writer),
                        readers: ISet::empty(),
                    })
                    &&& read_admitted(cap@.frontier(), final(self).citations())
                    &&& !old(self).frames.frame(f).local_shares.contains_key(cap@.share())
                    &&& final(self).frames.frame(f).local_shares == old(self).frames.frame(
                        f,
                    ).local_shares.insert(cap@.share(), LocalShareEntry { share: cap@.share() })
                    &&& final(self).frames.frame(f).remote_shares == old(self).frames.frame(
                        f,
                    ).remote_shares
                    &&& final(self).frames.meta(f).active_remote_leases == old(self).frames.meta(
                        f,
                    ).active_remote_leases
                    &&& final(self).frames.returned_reads(
                        f,
                        cap@,
                        Ticket::NONE,
                        final(self).kernel_view(),
                    )
                },
            },
    {
        if self.free_lease_slots.is_empty() {
            return Err(CoreError::CapacityExhausted);
        }
        if self.lease_generation == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let ghost v = self.kernel_view();
        let Tracked(cap) = self.frames.pin_local_share(
            f,
            Tracked(instance),
            Tracked(shares),
            Ghost(v),
        )?;
        let version = VersionId::unpack(self.frames.metas[f.0 as usize].version_packed);
        let ghost cert = self.frames.frame(f).cert.unwrap();
        let slot = self.free_lease_slots.pop().unwrap();
        self.lease_generation = self.lease_generation + 1;
        let handle = ReadLeaseHandle { slot, generation: self.lease_generation, frame: f };
        self.active_leases.set(
            slot as usize,
            Some(ActiveLocalLease { internal, handle, share: Ghost(cap.share()) }),
        );
        proof {
            assert(old(self).frames.valid_frame(f, v));
            assert(self.frames.valid_frame(f, v));
            lemma_version_pack_unpack(
                self.frames.frame(f).bound_version.unwrap(),
                self.frames.meta(f).version_packed,
            );
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
            assert(!old(self).frames.frame(f).local_shares.contains_key(cap.share())) by {
                if old(self).frames.frame(f).local_shares.contains_key(cap.share()) {
                    assert(old(self).frames.frame(f).live_shares().contains(cap.share()));
                }
            }
            Self::lemma_leases_after_acquire(
                old(self),
                self,
                f,
                slot,
                handle,
                internal,
                cap.share(),
            );
            assert(self.frames.returned_reads(f, cap, Ticket::NONE, v));
        }
        Ok((handle, version, Ghost(cert), Tracked(cap)))
    }

    pub fn release_local_lease(
        &mut self,
        handle: ReadLeaseHandle,
        Tracked(cap): Tracked<ShareCap>,
        read_ticket: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(self).timelines.valid_ticket(read_ticket),
            old(self).active_lease_matches(handle) ==> {
                let share = old(self).active_leases[handle.slot as int]->Some_0.share@;
                &&& cap.share() == share
                &&& old(self).frames.returned_reads(
                    handle.frame,
                    cap,
                    read_ticket,
                    old(self).kernel_view(),
                )
            },
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
            final(self).catalog == old(self).catalog,
            final(self).ledger == old(self).ledger,
            final(self).version_seq_start == old(self).version_seq_start,
            final(self).next_version_seq == old(self).next_version_seq,
            final(self).version_seq_limit == old(self).version_seq_limit,
            final(self).lease_generation == old(self).lease_generation,
            final(self).spec == old(self).spec,
            final(self).version_registry == old(self).version_registry,
            final(self).timelines.only_watermarks(&old(self).timelines),
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, Some(handle.frame)),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            r.is_ok() == old(self).active_lease_matches(handle),
            r.is_err() ==> *final(self) == *old(self),
            r.is_ok() ==> {
                &&& final(self).active_leases.len() == old(self).active_leases.len()
                &&& final(self).active_leases[handle.slot as int].is_none()
                &&& final(self).free_lease_slots.len() == old(self).free_lease_slots.len() + 1
                &&& final(self).free_lease_slots@.last() == handle.slot
                &&& final(self).frames.frame(handle.frame).local_shares == old(self).frames.frame(
                    handle.frame,
                ).local_shares.remove(old(self).active_leases[handle.slot as int]->Some_0.share@)
                &&& final(self).frames.frame(handle.frame).remote_shares == old(self).frames.frame(
                    handle.frame,
                ).remote_shares
                &&& final(self).frames.meta(handle.frame).active_remote_leases == old(
                    self,
                ).frames.meta(handle.frame).active_remote_leases
                &&& forall|s: int|
                    0 <= s < old(self).active_leases.len() && s != handle.slot as int
                        ==> #[trigger] final(self).active_leases[s] == old(self).active_leases[s]
            },
    {
        let idx = handle.slot as usize;
        if idx >= self.active_leases.len() {
            return Err(CoreError::InvalidArgument);
        }
        let matches = match &self.active_leases[idx] {
            Some(lease) => {
                lease.handle.slot == handle.slot && lease.handle.generation == handle.generation
                    && lease.handle.frame.0 == handle.frame.0
            },
            None => false,
        };
        proof {
            if matches {
                let lease = self.active_leases[idx as int]->Some_0;
                assert(lease.handle.frame == handle.frame);
                assert(lease.handle == handle);
            }
            assert(matches == self.active_lease_matches(handle));
        }
        if !matches {
            return Err(CoreError::StaleReplica);
        }
        let ghost v_before = self.kernel_view();
        let _lease = self.active_leases[idx].take().unwrap();
        self.free_lease_slots.push(handle.slot);
        self.frames.unpin_local_share(
            &mut self.timelines,
            handle.frame,
            Tracked(cap),
            read_ticket,
            Tracked(instance),
            Ghost(v_before),
        );
        proof {
            let v_after = self.kernel_view();
            assert(old(self).frames.valid_frame(handle.frame, v_before));
            assert(self.frames.valid_frame(handle.frame, v_after));
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v_before,
                v_after,
            );
            assert(!old(self).free_lease_slots@.contains(handle.slot));
            assert(self.free_lease_slots@.no_duplicates()) by {
                let n = old(self).free_lease_slots.len();
                assert forall|i: int, j: int|
                    0 <= i < self.free_lease_slots.len() && 0 <= j < self.free_lease_slots.len()
                        && i != j implies self.free_lease_slots[i] != self.free_lease_slots[j] by {
                    if i < n && j < n {
                        assert(old(self).free_lease_slots[i] != old(self).free_lease_slots[j]);
                    } else if i == n {
                        assert(self.free_lease_slots[i] == handle.slot);
                        assert(self.free_lease_slots[j] == old(self).free_lease_slots[j]);
                    } else {
                        assert(self.free_lease_slots[j] == handle.slot);
                        assert(self.free_lease_slots[i] == old(self).free_lease_slots[i]);
                    }
                }
            }
            assert forall|i: int| 0 <= i < self.free_lease_slots.len() implies {
                let s = #[trigger] self.free_lease_slots[i];
                (s as usize) < self.active_leases.len() && self.active_leases[s as int].is_none()
            } by {
                if i < old(self).free_lease_slots.len() {
                    let s = self.free_lease_slots[i];
                    assert(s == old(self).free_lease_slots[i]);
                    assert(s != handle.slot);
                }
            }
            assert forall|s: int|
                0 <= s < self.active_leases.len() implies match #[trigger] self.active_leases[s] {
                None => self.free_lease_slots@.contains(s as u32),
                Some(other) => {
                    &&& !self.free_lease_slots@.contains(s as u32)
                    &&& other.handle.slot as int == s
                    &&& 1 <= other.handle.generation <= self.lease_generation
                    &&& other.handle.frame.0 < self.frames.metas.len()
                    &&& self.frames.frame(other.handle.frame).local_shares.contains_key(
                        other.share@,
                    )
                },
            } by {
                if s == idx as int {
                    assert(self.free_lease_slots[old(self).free_lease_slots.len() as int]
                        == handle.slot);
                } else {
                    assert(self.active_leases[s] == old(self).active_leases[s]);
                    if self.active_leases[s].is_none() {
                        assert(old(self).free_lease_slots@.contains(s as u32));
                        let pos = choose|pos: int|
                            0 <= pos < old(self).free_lease_slots.len() && old(
                                self,
                            ).free_lease_slots[pos] == s as u32;
                        assert(self.free_lease_slots[pos] == s as u32);
                    } else {
                        let other = self.active_leases[s].unwrap();
                        if self.free_lease_slots@.contains(s as u32) {
                            let pos = choose|pos: int|
                                0 <= pos < self.free_lease_slots.len() && self.free_lease_slots[pos]
                                    == s as u32;
                            assert(pos < old(self).free_lease_slots.len());
                            assert(old(self).free_lease_slots[pos] == s as u32);
                        }
                        if other.handle.frame == handle.frame {
                            assert(old(self).active_leases[s] == Some(other));
                            assert(old(self).active_leases[idx as int] == Some(_lease));
                            assert(other.share@ != _lease.share@);
                            assert(old(self).frames.frame(handle.frame).local_shares.contains_key(
                                other.share@,
                            ));
                        } else {
                            assert(old(self).frames.frame(
                                other.handle.frame,
                            ).local_shares.contains_key(other.share@));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn pin_remote_lease(
        &mut self,
        f: FrameId,
        slot: u32,
        generation: u32,
        Tracked(shares): Tracked<&mut ShareSource>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(VersionId, Ghost<ValueCertificate>, Tracked<ShareCap>), CoreError>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).instance_id(),
            old(shares).node() == old(self).node(),
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
            final(self).active_leases.len() == old(self).active_leases.len(),
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, Some(f)),
            final(shares).well_formed(),
            final(shares).id() == old(self).instance_id(),
            final(shares).node() == old(self).node(),
            match r {
                Err(_) => *final(self) == *old(self) && *final(shares) == *old(shares),
                Ok((version, cert, cap)) => {
                    &&& f.0 < old(self).frames.metas.len()
                    &&& old(self).frames.meta(f).state == STATE_PUBLISHED
                    &&& final(self).frames.meta(f).state == STATE_PUBLISHED
                    &&& final(self).frames.frame(f).bound_version == Some(version)
                    &&& final(self).frames.frame(f).cert == Some(cert@)
                    &&& replica_valid(final(self).ledger@, final(self).node(), cert@)
                    &&& final(self).citations().contains(cert@.writer)
                    &&& final(self).frames.versions@[version] == cert@.bytes
                    &&& !old(self).frames.frame(f).remote_shares.contains_key(cap@.share())
                    &&& final(self).frames.frame(f).remote_shares == old(self).frames.frame(
                        f,
                    ).remote_shares.insert(
                        cap@.share(),
                        RemoteShareEntry { mw_slot: slot, mw_gen: generation },
                    )
                    &&& final(self).frames.frame(f).matches_lease(cap@.share(), slot, generation)
                    &&& cap@.is_unbound()
                    &&& cap@.instance_id() == final(self).instance_id()
                    &&& cap@.loc() == final(self).frames.frame(f).slot.loc
                    &&& cap@.frontier() == (Frontier {
                        last_writer: Some(cert@.writer),
                        readers: ISet::empty(),
                    })
                    &&& read_admitted(cap@.frontier(), final(self).citations())
                },
            },
    {
        let ghost v = self.kernel_view();
        let Tracked(cap) = self.frames.pin_remote_lease(
            f,
            slot,
            generation,
            Tracked(instance),
            Tracked(shares),
            Ghost(v),
        )?;
        let version = VersionId::unpack(self.frames.metas[f.0 as usize].version_packed);
        let ghost cert = self.frames.frame(f).cert.unwrap();
        proof {
            assert(old(self).frames.valid_frame(f, v));
            assert(self.frames.valid_frame(f, v));
            lemma_version_pack_unpack(
                self.frames.frame(f).bound_version.unwrap(),
                self.frames.meta(f).version_packed,
            );
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
        Ok((version, Ghost(cert), Tracked(cap)))
    }

    pub fn unpin_remote_lease(
        &mut self,
        f: FrameId,
        slot: u32,
        generation: u32,
        Tracked(cap): Tracked<ShareCap>,
        k: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Tracked<ReturnedShare>, (CoreError, Tracked<ShareCap>)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).instance_id(),
            f.0 < old(self).frames.metas.len() && old(self).frames.meta(f).active_remote_leases > 0
                ==> old(self).frames.frame(f).matches_lease(cap.share(), slot, generation),
            f.0 < old(self).frames.metas.len() && old(self).frames.meta(f).active_remote_leases > 0
                ==> old(self).frames.returned_reads(f, cap, k, old(self).kernel_view()) && old(
                self,
            ).timelines.valid_ticket(k),
        ensures
            final(self).well_formed(),
            final(self).node_id == old(self).node_id,
            final(self).tiers == old(self).tiers,
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
            final(self).timelines.only_watermarks(&old(self).timelines),
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, Some(f)),
            f.0 < old(self).frames.metas.len() && old(self).frames.meta(f).active_remote_leases > 0
                ==> r.is_ok(),
            match r {
                Ok(receipt) => final(self).frames.frame(f).remote_shares == old(self).frames.frame(
                    f,
                ).remote_shares.remove(cap.share()) && receipt@.share() == cap.share()
                    && receipt@.instance_id() == old(self).instance_id() && receipt@.loc() == old(
                    self,
                ).frames.frame(f).slot.loc,
                Err((_, back)) => back@ == cap && *final(self) == *old(self),
            },
    {
        let ghost v_before = self.kernel_view();
        let res = self.frames.unpin_remote_lease(
            &mut self.timelines,
            f,
            slot,
            generation,
            Tracked(cap),
            k,
            Tracked(instance),
            Ghost(v_before),
        );
        proof {
            let v_after = self.kernel_view();
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v_before,
                v_after,
            );
        }
        res
    }
}

} // verus!
