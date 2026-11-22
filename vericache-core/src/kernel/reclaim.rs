#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    pub fn lookup_best(&self, key: ObjectKey, priority: &[TierId]) -> (r: Option<
        (FrameId, TierId, VersionId),
    >)
        requires
            self.well_formed(),
        ensures
            r matches Some((f, tier, version)) ==> {
                &&& tier.valid(self.tiers.len())
                &&& f.0 < self.frames.metas.len()
                &&& self.frames.meta(f).state == STATE_PUBLISHED
                &&& self.frames.frame(f).tier == tier
                &&& self.frames.frame(f).bound_key == Some(key)
                &&& self.frames.frame(f).bound_version == Some(version)
                &&& self.catalog.slot(key, tier.index() as int) == Some(f)
                &&& self.spec@.catalog.contains_key(key) && self.spec@.catalog[key] == version
                &&& exists|idx: int|
                    0 <= idx < priority.len() && #[trigger] priority[idx] == tier && forall|
                        prev: int,
                    |
                        0 <= prev < idx && (#[trigger] priority[prev]).valid(self.tiers.len())
                            ==> self.catalog.slot(key, priority[prev].index() as int).is_none()
            },
            r.is_none() ==> forall|idx: int|
                0 <= idx < priority.len() && (#[trigger] priority[idx]).valid(self.tiers.len())
                    ==> self.catalog.slot(key, priority[idx].index() as int).is_none(),
    {
        let res = self.catalog.best_replica(key, priority);
        proof {
            if let Some((f, tier, _version)) = res {
                assert(self.catalog.slot(key, tier.index() as int) == Some(f));
                assert(self.frames.frame(f).tier.index() == tier.index());
                assert(self.frames.frame(f).tier == tier);
            }
        }
        res
    }

    pub fn lookup_tier(&self, key: ObjectKey, tier: TierId) -> (r: Option<(FrameId, VersionId)>)
        requires
            self.well_formed(),
        ensures
            r matches Some((f, version)) ==> {
                &&& tier.valid(self.tiers.len())
                &&& f.0 < self.frames.metas.len()
                &&& self.frames.meta(f).state == STATE_PUBLISHED
                &&& self.frames.frame(f).tier == tier
                &&& self.frames.frame(f).bound_key == Some(key)
                &&& self.frames.frame(f).bound_version == Some(version)
                &&& self.catalog.slot(key, tier.index() as int) == Some(f)
                &&& self.spec@.catalog.contains_key(key) && self.spec@.catalog[key] == version
            },
            r.is_none() == (!tier.valid(self.tiers.len()) || self.catalog.slot(
                key,
                tier.index() as int,
            ).is_none()),
    {
        if tier.as_index() >= self.tiers.len() {
            return None;
        }
        match self.catalog.lookup_tier(key, tier) {
            Some((version, f)) => {
                proof {
                    assert(self.catalog.slot(key, tier.index() as int) == Some(f));
                    assert(self.frames.frame(f).tier.index() == tier.index());
                    assert(self.frames.frame(f).tier == tier);
                }
                Some((f, version))
            },
            None => None,
        }
    }

    pub fn evict_frame(&mut self, f: FrameId) -> (r: Result<(), CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            old(self).spec@.versions_monotonic(final(self).spec@),
            final(self).spec@.versions == old(self).spec@.versions,
            final(self).spec@.catalog.submap_of(old(self).spec@.catalog),
            final(self).frames.same_layout(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
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
            r.is_ok() == (f.0 < old(self).frames.metas.len() && old(self).frames.meta(f).state
                == STATE_PUBLISHED),
            r.is_ok() ==> final(self).frames.meta(f).state == STATE_DRAINING,
            r.is_err() ==> r == Err(CoreError::StaleReplica) && *final(self) == *old(self),
    {
        let ghost v = self.kernel_view();
        self.catalog.evict_frame(&mut self.frames, f, Ghost(v))?;
        proof {
            self.spec = Ghost(Self::service(&self.frames, &self.catalog));
            assert(self.spec@.catalog.submap_of(old(self).spec@.catalog)) by {
                assert forall|k: ObjectKey| #[trigger]
                    self.spec@.catalog.contains_key(k) implies old(self).spec@.catalog.contains_key(
                    k,
                ) && self.spec@.catalog[k] == old(self).spec@.catalog[k] by {
                    assert(self.catalog.view().contains_key(k));
                    assert(old(self).catalog.view().contains_key(k));
                }
            }
            self.spec@.restrict_directory(old(self).spec@);
            assert forall|g: FrameId|
                g.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(g)).state
                    == STATE_PUBLISHED implies {
                let fg = self.frames.frame(g);
                fg.cert matches Some(cert) && {
                    &&& replica_valid(v.ledger, v.node, cert)
                    &&& event_succeeded(v.ledger, v.node, cert.writer)
                    &&& v.ledger.citations(v.node).contains(cert.writer)
                    &&& fg.bound_version matches Some(ver)
                    &&& self.frames.versions@.contains_key(ver)
                    &&& self.frames.versions@[ver] == cert.bytes
                }
            } by {
                assert(g != f);
                assert(old(self).frames.meta(g).state == STATE_PUBLISHED);
            }
        }
        Ok(())
    }

    pub fn reclaim_draining(&mut self, tier: TierId, budget: usize) -> (reclaimed: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).spec == old(self).spec,
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
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
            reclaimed <= budget,
            !tier.valid(old(self).tiers.len()) ==> reclaimed == 0 && *final(self) == *old(self),
            tier.valid(old(self).tiers.len()) ==> {
                &&& final(self).frames.ready_queue[tier.index() as int].len() == old(
                    self,
                ).frames.ready_queue[tier.index() as int].len() + reclaimed
                &&& final(self).frames.draining_queue[tier.index() as int].len() + reclaimed == old(
                    self,
                ).frames.draining_queue[tier.index() as int].len()
            },
    {
        if tier.as_index() >= self.tiers.len() {
            return 0;
        }
        let ghost v = self.kernel_view();
        let reclaimed = self.frames.reclaim_draining(&self.timelines, tier, budget, Ghost(v));
        proof {
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &old(self).frames,
                &self.frames,
                v,
                v,
            );
        }
        reclaimed
    }

    pub fn reclaim_all_draining(&mut self) -> (total: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).spec == old(self).spec,
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
            final(self).frames.same_directory(&old(self).frames),
            final(self).frames.preserves_initializing(&old(self).frames, None),
            final(self).frames.same_other_pins(&old(self).frames, None),
            final(self).frames.preserves_remote_leases(&old(self).frames),
    {
        let num_tiers = self.tiers.len();
        let mut total: usize = 0;
        let mut t: usize = 0;
        while t < num_tiers
            invariant
                t <= num_tiers,
                num_tiers == self.tiers.len(),
                self.well_formed(),
                self.spec == old(self).spec,
                self.node_id == old(self).node_id,
                self.tiers == old(self).tiers,
                self.timelines == old(self).timelines,
                self.catalog == old(self).catalog,
                self.ledger == old(self).ledger,
                self.version_seq_start == old(self).version_seq_start,
                self.next_version_seq == old(self).next_version_seq,
                self.version_seq_limit == old(self).version_seq_limit,
                self.active_leases == old(self).active_leases,
                self.free_lease_slots == old(self).free_lease_slots,
                self.lease_generation == old(self).lease_generation,
                self.version_registry == old(self).version_registry,
                self.frames.same_directory(&old(self).frames),
                self.frames.preserves_initializing(&old(self).frames, None),
                self.frames.same_other_pins(&old(self).frames, None),
                self.frames.preserves_remote_leases(&old(self).frames),
            decreases num_tiers - t,
        {
            let tier = self.tiers[t].id;
            let budget = self.frames.draining_queue[t].len();
            let reclaimed = self.reclaim_draining(tier, budget);
            total = total.saturating_add(reclaimed);
            t += 1;
        }
        total
    }
}

} // verus!
