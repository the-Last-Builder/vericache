#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use super::*;
use crate::provenance::*;
#[cfg(verus_only)]
use foundation::{history::reachable, value::valid_run};
use transfer::context::*;

verus! {

impl Catalog {
    proof fn lemma_unlink_covers(
        before: &Catalog,
        after: &Catalog,
        before_frames: &FrameTable,
        after_frames: &FrameTable,
        f: FrameId,
        key: ObjectKey,
        t: int,
        entry: CatalogEntry,
        some: bool,
        publishing: Option<FrameId>,
    )
        requires
            before.covers(before_frames, publishing),
            after.indexed(),
            after.num_tiers() == before.num_tiers(),
            after_frames.num_tiers == before_frames.num_tiers,
            after_frames.metas.len() == before_frames.metas.len(),
            after_frames.ghosts == before_frames.ghosts,
            f.0 < before_frames.metas.len(),
            before_frames.meta(f).state == STATE_PUBLISHED,
            after_frames.meta(f).state == STATE_DRAINING,
            Some(f) != publishing,
            before_frames.frame(f).bound_key == Some(key),
            before_frames.frame(f).tier.index() == t,
            0 <= t < before.num_tiers(),
            forall|g: FrameId|
                g != f && g.0 < before_frames.metas.len() ==> #[trigger] after_frames.meta(g)
                    == before_frames.meta(g),
            before.view().contains_key(key),
            entry.version == before.view()[key].version,
            entry.slots[t].is_none(),
            forall|u: int|
                0 <= u < MAX_TIERS && u != t ==> (#[trigger] entry.slots[u])
                    == before.view()[key].slots[u],
            some == (exists|j: int|
                0 <= j < before.num_tiers() && (#[trigger] entry.slots[j]).is_some()),
            after.view() == if some {
                before.view().insert(key, entry)
            } else {
                before.view().remove(key)
            },
        ensures
            after.covers(after_frames, publishing),
            forall|k: ObjectKey, u: int|
                0 <= u < before.num_tiers() ==> #[trigger] after.slot(k, u) == (if before.slot(k, u)
                    == Some(f) {
                    None
                } else {
                    before.slot(k, u)
                }),
    {
        assert forall|k: ObjectKey, u: int|
            0 <= u < after.num_tiers() implies #[trigger] after.slot(k, u) == (if before.slot(k, u)
            == Some(f) {
            None
        } else {
            before.slot(k, u)
        }) by {
            if before.slot(k, u) == Some(f) {
                assert(before_frames.frame(f).bound_key == Some(k));
                assert(before_frames.frame(f).tier.index() == u);
            }
            if k == key && u != t {
                assert(entry.slots[u] == before.slot(k, u));
                if entry.slots[u].is_some() {
                    assert(some);
                }
            }
        }
        assert forall|k: ObjectKey, u: int| after.num_tiers() <= u < MAX_TIERS implies (
        #[trigger] after.slot(k, u)).is_none() by {
            assert(before.slot(k, u).is_none());
        }
        assert forall|k: ObjectKey, u: int| 0 <= u < after.num_tiers() implies (
        #[trigger] after.slot(k, u) matches Some(g) ==> {
            &&& Some(g) != publishing
            &&& g.0 < after_frames.metas.len()
            &&& after_frames.meta(g).state == STATE_PUBLISHED
            &&& after_frames.frame(g).tier.index() == u
            &&& after_frames.frame(g).bound_key == Some(k)
            &&& after_frames.frame(g).bound_version == Some(after.view()[k].version)
        }) by {
            if let Some(g) = after.slot(k, u) {
                assert(before.slot(k, u) == Some(g));
                assert(g != f);
                assert(after_frames.meta(g) == before_frames.meta(g));
                assert(after.view()[k].version == before.view()[k].version);
            }
        }
        assert forall|g: FrameId|
            Some(g) != publishing && g.0 < after_frames.metas.len() && (
            #[trigger] after_frames.meta(g)).state == STATE_PUBLISHED implies {
            let info = after_frames.frame(g);
            &&& info.bound_key.is_some()
            &&& info.tier.valid(after.num_tiers())
            &&& after.slot(info.bound_key.unwrap(), info.tier.index() as int) == Some(g)
        } by {
            assert(g != f);
            assert(before_frames.meta(g).state == STATE_PUBLISHED);
            let info = before_frames.frame(g);
            assert(before.slot(info.bound_key.unwrap(), info.tier.index() as int) == Some(g));
        }
        assert forall|k| after.view().contains_key(k) implies exists|u: int|
            0 <= u < after.num_tiers() && #[trigger] after.slot(k, u).is_some() by {
            if k != key {
                let u = choose|u: int|
                    0 <= u < after.num_tiers() && #[trigger] before.slot(k, u).is_some();
                assert(after.slot(k, u) == before.slot(k, u));
            } else {
                let u = choose|u: int|
                    0 <= u < after.num_tiers() && (#[trigger] entry.slots[u]).is_some();
                assert(after.slot(k, u).is_some());
            }
        }
    }

    fn unlink(
        &mut self,
        frames: &mut FrameTable,
        f: FrameId,
        Ghost(publishing): Ghost<Option<FrameId>>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).covers(old(frames), publishing),
            old(frames).consistent(v),
            f.0 < old(frames).metas.len(),
            old(frames).meta(f).state == STATE_PUBLISHED,
            Some(f) != publishing,
        ensures
            final(frames).num_tiers == old(frames).num_tiers,
            final(frames).versions == old(frames).versions,
            final(frames).frame_size == old(frames).frame_size,
            final(frames).tier_frame_sizes == old(frames).tier_frame_sizes,
            final(frames).overflow_seqs == old(frames).overflow_seqs,
            forall|key: ObjectKey|
                key != old(frames).frame(f).bound_key.unwrap()
                    ==> #[trigger] final(self).view().contains_key(key) == old(
                    self,
                ).view().contains_key(key) && (old(self).view().contains_key(key)
                    ==> final(self).view()[key] == old(self).view()[key]),
            final(self).covers(final(frames), publishing),
            final(frames).consistent(v),
            final(frames).bindings == old(frames).bindings,
            final(frames).metas.len() == old(frames).metas.len(),
            final(frames).ghosts == old(frames).ghosts,
            final(frames).meta(f).state == STATE_DRAINING,
            final(frames).meta(f).active_remote_leases == old(frames).meta(f).active_remote_leases,
            forall|g: FrameId|
                g != f && g.0 < old(frames).metas.len() ==> #[trigger] final(frames).meta(g) == old(
                    frames,
                ).meta(g),
            forall|k: ObjectKey, t: int|
                0 <= t < old(self).num_tiers() ==> #[trigger] final(self).slot(k, t) == (if old(
                    self,
                ).slot(k, t) == Some(f) {
                    None
                } else {
                    old(self).slot(k, t)
                }),
            forall|k: ObjectKey, t: int|
                0 <= t < old(self).num_tiers() ==> #[trigger] final(self).slot(k, t) == (if k
                    == old(frames).frame(f).bound_key.unwrap() && t == old(frames).frame(
                    f,
                ).tier.index() {
                    None
                } else {
                    old(self).slot(k, t)
                }),
            forall|k| #[trigger]
                final(self).view().contains_key(k) ==> old(self).view().contains_key(k)
                    && final(self).view()[k].version == old(self).view()[k].version,
    {
        let (key, tier) = frames.bindings[f.0 as usize];
        assert(frames.meta(f).state == STATE_PUBLISHED);
        let t = tier.as_index();
        assert(self.slot(key, t as int) == Some(f));
        let mut entry = *self.lookup(key).unwrap();
        entry.slots[t] = None;
        let num_tiers = self.tier_count();
        let mut some = false;
        for i in 0..num_tiers
            invariant
                num_tiers == self.num_tiers(),
                some == (exists|j: int| 0 <= j < i && (#[trigger] entry.slots[j]).is_some()),
        {
            if entry.slots[i].is_some() {
                some = true;
            }
        }
        self.set_entry(
            key,
            if some {
                Some(entry)
            } else {
                None
            },
        );
        let _ = frames.evict_frame(f, tier, Ghost(v));
        proof {
            Self::lemma_unlink_covers(
                old(self),
                self,
                old(frames),
                frames,
                f,
                key,
                t as int,
                entry,
                some,
                publishing,
            );
            assert forall|k: ObjectKey, u: int|
                0 <= u < self.num_tiers() implies #[trigger] self.slot(k, u) == (if k == key && u
                == t {
                None
            } else {
                old(self).slot(k, u)
            }) by {
                if k == key && u == t {
                    assert(old(self).slot(k, u) == Some(f));
                } else if old(self).slot(k, u) == Some(f) {
                    assert(old(frames).frame(f).bound_key == Some(k));
                    assert(old(frames).frame(f).tier.index() == u);
                }
            }
        }
    }

    pub fn evict_frame(
        &mut self,
        frames: &mut FrameTable,
        f: FrameId,
        Ghost(v): Ghost<KernelView>,
    ) -> (result: Result<(), CoreError>)
        requires
            old(self).well_formed(old(frames)),
            old(frames).consistent(v),
        ensures
            forall|k| #[trigger]
                final(self).view().contains_key(k) ==> old(self).view().contains_key(k)
                    && final(self).view()[k].version == old(self).view()[k].version,
            final(frames).num_tiers == old(frames).num_tiers,
            final(frames).versions == old(frames).versions,
            final(frames).frame_size == old(frames).frame_size,
            final(frames).tier_frame_sizes == old(frames).tier_frame_sizes,
            final(frames).overflow_seqs == old(frames).overflow_seqs,
            final(frames).metas.len() == old(frames).metas.len(),
            final(frames).ghosts == old(frames).ghosts,
            result.is_ok() ==> final(frames).meta(f).active_remote_leases == old(frames).meta(
                f,
            ).active_remote_leases,
            forall|g: FrameId|
                g != f && g.0 < old(frames).metas.len() ==> #[trigger] final(frames).meta(g) == old(
                    frames,
                ).meta(g),
            final(self).well_formed(final(frames)),
            final(frames).consistent(v),
            result.is_ok() == (f.0 < old(frames).metas.len() && old(frames).meta(f).state
                == STATE_PUBLISHED),
            result.is_ok() ==> final(frames).meta(f).state == STATE_DRAINING,
            result.is_err() ==> result == Err(CoreError::StaleReplica) && *final(self) == *old(self)
                && *final(frames) == *old(frames),
            forall|k: ObjectKey, t: int|
                0 <= t < old(self).num_tiers() ==> #[trigger] final(self).slot(k, t) == (if old(
                    self,
                ).slot(k, t) == Some(f) {
                    None
                } else {
                    old(self).slot(k, t)
                }),
    {
        if f.0 as usize >= frames.metas.len() || frames.metas[f.0 as usize].state
            != STATE_PUBLISHED {
            return Err(CoreError::StaleReplica);
        }
        self.unlink(frames, f, Ghost(None), Ghost(v));
        Ok(())
    }

    #[cfg_attr(verus_only, verifier::loop_isolation(true))]
    fn clear_prior_slots(
        &mut self,
        frames: &mut FrameTable,
        Ghost(f): Ghost<FrameId>,
        Ghost(key): Ghost<ObjectKey>,
        t: usize,
        previous: CatalogEntry,
        replace: bool,
        entry: &mut CatalogEntry,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).covers(old(frames), Some(f)),
            old(frames).consistent(v),
            f.0 < old(frames).metas.len(),
            old(frames).meta(f).state == STATE_PUBLISHED,
            t < old(self).num_tiers(),
            old(self).view().contains_key(key),
            old(self).view()[key] == previous,
            forall|u: int| 0 <= u < MAX_TIERS ==> (#[trigger] old(entry).slots[u]).is_none(),
        ensures
            final(frames).num_tiers == old(frames).num_tiers && final(frames).frame_size == old(
                frames,
            ).frame_size && final(frames).tier_frame_sizes == old(frames).tier_frame_sizes
                && final(frames).overflow_seqs == old(frames).overflow_seqs
                && final(frames).versions == old(frames).versions && final(frames).ghosts == old(
                frames,
            ).ghosts && final(frames).metas.len() == old(frames).metas.len(),
            final(frames).meta(f) == old(frames).meta(f) && final(frames).consistent(v),
            final(self).covers(final(frames), Some(f)) && final(entry).version == old(
                entry,
            ).version,
            forall|other: ObjectKey|
                other != key ==> #[trigger] final(self).view().contains_key(other) == old(
                    self,
                ).view().contains_key(other) && (old(self).view().contains_key(other)
                    ==> final(self).view()[other] == old(self).view()[other]),
            forall|g: FrameId|
                g.0 < old(frames).metas.len() && ((#[trigger] final(frames).meta(g)).state
                    == STATE_PUBLISHED || old(frames).meta(g).state == STATE_INITIALIZING)
                    ==> final(frames).meta(g) == old(frames).meta(g),
            forall|g: FrameId|
                g.0 < old(frames).metas.len() ==> (#[trigger] final(frames).meta(
                    g,
                )).active_remote_leases == old(frames).meta(g).active_remote_leases,
            forall|k: ObjectKey, u: int|
                0 <= u < old(self).num_tiers() && k != key ==> #[trigger] final(self).slot(k, u)
                    == old(self).slot(k, u),
            forall|u: int|
                0 <= u < old(self).num_tiers() ==> #[trigger] final(self).slot(key, u) == (
                if replace || u == t {
                    None
                } else {
                    previous.slots[u]
                }),
            final(self).view().contains_key(key) ==> final(self).view()[key].version
                == previous.version,
            forall|u: int|
                0 <= u < old(self).num_tiers() ==> (#[trigger] final(entry).slots[u]) == (
                if !replace && u != t {
                    previous.slots[u]
                } else {
                    None
                }),
            forall|u: int|
                old(self).num_tiers() <= u < MAX_TIERS ==> (
                #[trigger] final(entry).slots[u]).is_none(),
    {
        hide(FrameTable::consistent);
        let num_tiers = self.tier_count();
        let ghost (committed, orig_self, orig_ver) = (*frames, *self, entry.version);
        for i in 0..num_tiers
            invariant
                num_tiers == self.num_tiers() && num_tiers == orig_self.num_tiers() && t < num_tiers
                    && num_tiers <= MAX_TIERS && self.covers(frames, Some(f)) && frames.consistent(
                    v,
                ) && f.0 < committed.metas.len() && committed.meta(f).state == STATE_PUBLISHED,
                frames.num_tiers == committed.num_tiers && frames.frame_size == committed.frame_size
                    && frames.tier_frame_sizes == committed.tier_frame_sizes && frames.overflow_seqs
                    == committed.overflow_seqs && frames.versions == committed.versions
                    && frames.ghosts == committed.ghosts && frames.metas.len()
                    == committed.metas.len(),
                frames.meta(f) == committed.meta(f) && entry.version == orig_ver,
                forall|other: ObjectKey|
                    other != key ==> #[trigger] self.view().contains_key(other)
                        == orig_self.view().contains_key(other) && (orig_self.view().contains_key(
                        other,
                    ) ==> self.view()[other] == orig_self.view()[other]),
                forall|g: FrameId|
                    g.0 < committed.metas.len() && ((#[trigger] frames.meta(g)).state
                        == STATE_PUBLISHED || committed.meta(g).state == STATE_INITIALIZING)
                        ==> frames.meta(g) == committed.meta(g),
                forall|g: FrameId|
                    g.0 < committed.metas.len() ==> (#[trigger] frames.meta(g)).active_remote_leases
                        == committed.meta(g).active_remote_leases,
                forall|k: ObjectKey, u: int|
                    0 <= u < num_tiers && k != key ==> #[trigger] self.slot(k, u) == orig_self.slot(
                        k,
                        u,
                    ),
                forall|u: int|
                    0 <= u < num_tiers ==> #[trigger] self.slot(key, u) == (if u < i && (replace
                        || u == t) {
                        None
                    } else {
                        previous.slots[u]
                    }),
                self.view().contains_key(key) ==> self.view()[key].version == previous.version,
                forall|u: int|
                    0 <= u < num_tiers ==> (#[trigger] entry.slots[u]) == (if u < i && !replace && u
                        != t {
                        previous.slots[u]
                    } else {
                        None
                    }),
                forall|u: int| num_tiers <= u < MAX_TIERS ==> (#[trigger] entry.slots[u]).is_none(),
        {
            if replace || i == t {
                if let Some(prior) = previous.slots[i] {
                    assert(self.slot(key, i as int) == Some(prior));
                    assert(frames.frame(prior).bound_key == Some(key));
                    assert(frames.frame(prior).tier.index() == i);
                    self.unlink(frames, prior, Ghost(Some(f)), Ghost(v));
                }
            } else {
                entry.slots[i] = previous.slots[i];
            }
        }
    }

    fn link_published_frame(
        &mut self,
        frames: &mut FrameTable,
        f: FrameId,
        key: ObjectKey,
        tier: TierId,
        version: VersionId,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).covers(old(frames), Some(f)),
            old(frames).consistent(v),
            f.0 < old(frames).metas.len(),
            old(frames).meta(f).state == STATE_PUBLISHED,
            old(frames).frame(f).tier == tier,
            tier.valid(old(self).num_tiers()),
            old(frames).frame(f).bound_key == Some(key),
            old(frames).frame(f).bound_version == Some(version),
        ensures
            final(frames).num_tiers == old(frames).num_tiers,
            final(frames).frame_size == old(frames).frame_size && final(frames).tier_frame_sizes
                == old(frames).tier_frame_sizes && final(frames).metas.len() == old(
                frames,
            ).metas.len(),
            final(frames).versions == old(frames).versions,
            final(frames).ghosts == old(frames).ghosts,
            final(frames).meta(f) == old(frames).meta(f),
            final(frames).preserves_initializing(old(frames), Some(f)),
            forall|g: FrameId|
                g != f && g.0 < old(frames).metas.len() ==> (#[trigger] final(frames).frame(g))
                    == old(frames).frame(g) && final(frames).meta(g).active_remote_leases == old(
                    frames,
                ).meta(g).active_remote_leases && (final(frames).published(g).is_some()
                    ==> final(frames).published(g) == old(frames).published(g)),
            forall|other: ObjectKey|
                other != key ==> #[trigger] final(self).view().contains_key(other) == old(
                    self,
                ).view().contains_key(other) && (old(self).view().contains_key(other)
                    ==> final(self).view()[other] == old(self).view()[other]),
            final(self).well_formed(final(frames)),
            final(frames).consistent(v),
            final(self).slot(key, tier.index() as int) == Some(f),
            final(self).view().contains_key(key),
            final(self).view()[key].version == version,
            forall|u: int|
                #![trigger final(self).slot(key, u)]
                0 <= u < old(self).num_tiers() && u != tier.index() ==> final(self).slot(key, u)
                    == if old(self).view().contains_key(key) && old(self).view()[key].version
                    == version {
                    old(self).slot(key, u)
                } else {
                    None
                },
            forall|k: ObjectKey, t: int|
                0 <= t < old(self).num_tiers() && k != key ==> #[trigger] final(self).slot(k, t)
                    == old(self).slot(k, t),
    {
        let t = tier.as_index();
        let ghost num_tiers = self.num_tiers();
        let mut entry = CatalogEntry { version, slots: [None;MAX_TIERS] };
        if let Some(previous) = self.lookup(key) {
            let previous = *previous;
            let replace = previous.version.origin_node != version.origin_node
                || previous.version.seq != version.seq;
            self.clear_prior_slots(
                frames,
                Ghost(f),
                Ghost(key),
                t,
                previous,
                replace,
                &mut entry,
                Ghost(v),
            );
        }
        assert forall|u: int| 0 <= u < num_tiers implies #[trigger] entry.slots[u] == self.slot(
            key,
            u,
        ) by {}
        assert(self.slot(key, t as int).is_none());
        assert(self.view().contains_key(key) ==> self.view()[key].version == version) by {
            if self.view().contains_key(key) {
                let u = choose|u: int| 0 <= u < num_tiers && #[trigger] self.slot(key, u).is_some();
                assert(self.slot(key, u).is_some());
            }
        }
        let ghost mid_self = *self;
        self.attach(Ghost(frames), key, t, f, entry);
        assert forall|other: ObjectKey| other != key implies #[trigger] self.view().contains_key(
            other,
        ) == old(self).view().contains_key(other) && (old(self).view().contains_key(other)
            ==> self.view()[other] == old(self).view()[other]) by {
            assert(mid_self.view().contains_key(other) == old(self).view().contains_key(other));
        }
    }

    pub fn publish_frame(
        &mut self,
        frames: &mut FrameTable,
        f: FrameId,
        ticket: Ticket,
        Ghost(cert): Ghost<ValueCertificate>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).well_formed(old(frames)),
            old(frames).except_frame(f, v),
            old(frames).meta(f).state == STATE_INITIALIZING,
            !old(frames).meta(f).queued(),
            old(frames).frame(f).binding(old(frames).meta(f), &v.timelines),
            old(frames).frame(f).authority(old(frames).meta(f), v.id()),
            old(frames).frame(f).completed_writes(cert),
            old(frames).versions@.contains_key(old(frames).frame(f).bound_version.unwrap()) ==> old(
                frames,
            ).versions@[old(frames).frame(f).bound_version.unwrap()] == cert.bytes,
            cert.slot == old(frames).frame(f).slot,
            cert.bytes.len() == cert.slot.slot_len,
            v.timelines.issued(ticket.timeline_id(), ticket.sequence()),
            ticket != Ticket::NONE,
            cert.writer == v.timelines.events@[(ticket.timeline_id(), ticket.sequence())],
            event_succeeded(v.ledger, v.node, cert.writer),
            forall|h, run|
                reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                    v.ledger,
                    v.node,
                    h,
                    run,
                ) ==> #[trigger] certificate_valid_in(cert, h, run),
            old(frames).frame(f).tier.valid(old(frames).num_tiers),
            cert.slot.loc.node == v.node,
            cert.slot.slot_len == old(frames).tier_frame_size(old(frames).frame(f).tier),
            cert.slot.is_full(),
        ensures
            final(frames).num_tiers == old(frames).num_tiers,
            final(frames).frame_size == old(frames).frame_size && final(frames).tier_frame_sizes
                == old(frames).tier_frame_sizes && final(frames).metas.len() == old(
                frames,
            ).metas.len(),
            final(frames).versions@ == old(frames).versions@.insert(
                old(frames).frame(f).bound_version.unwrap(),
                cert.bytes,
            ),
            final(frames).frame(f) == (FrameGhost { cert: Some(cert), ..old(frames).frame(f) }),
            final(frames).preserves_initializing(old(frames), Some(f)),
            forall|g: FrameId|
                g != f && g.0 < old(frames).metas.len() ==> (#[trigger] final(frames).frame(g))
                    == old(frames).frame(g) && final(frames).meta(g).active_remote_leases == old(
                    frames,
                ).meta(g).active_remote_leases && (final(frames).published(g).is_some()
                    ==> final(frames).published(g) == old(frames).published(g)),
            forall|other: ObjectKey|
                other != old(frames).frame(f).bound_key.unwrap()
                    ==> #[trigger] final(self).view().contains_key(other) == old(
                    self,
                ).view().contains_key(other) && (old(self).view().contains_key(other)
                    ==> final(self).view()[other] == old(self).view()[other]),
            final(self).well_formed(final(frames)),
            final(frames).consistent(v),
            final(frames).meta(f).state == STATE_PUBLISHED,
            final(frames).frame(f).cert == Some(cert),
            final(self).slot(
                old(frames).frame(f).bound_key.unwrap(),
                old(frames).frame(f).tier.index() as int,
            ) == Some(f),
            final(self).view().contains_key(old(frames).frame(f).bound_key.unwrap()),
            final(self).view()[old(frames).frame(f).bound_key.unwrap()].version == old(
                frames,
            ).frame(f).bound_version.unwrap(),
            forall|u: int|
                #![trigger final(self).slot(old(frames).frame(f).bound_key.unwrap(), u)]
                0 <= u < old(self).num_tiers() && u != old(frames).frame(f).tier.index() ==> {
                    let key = old(frames).frame(f).bound_key.unwrap();
                    final(self).slot(key, u) == if old(self).view().contains_key(key) && old(
                        self,
                    ).view()[key].version == old(frames).frame(f).bound_version.unwrap() {
                        old(self).slot(key, u)
                    } else {
                        None
                    }
                },
            forall|k: ObjectKey, t: int|
                0 <= t < old(self).num_tiers() && k != old(frames).frame(f).bound_key.unwrap()
                    ==> #[trigger] final(self).slot(k, t) == old(self).slot(k, t),
    {
        let (key, tier) = frames.bindings[f.0 as usize];
        let version = VersionId::unpack(frames.metas[f.0 as usize].version_packed);
        proof {
            crate::spec::lemma_version_pack_unpack(
                frames.frame(f).bound_version.unwrap(),
                frames.meta(f).version_packed,
            );
        }
        frames.publish_frame(f, ticket, Ghost(cert), Ghost(v));
        proof {
            self.begin_publication(*old(frames), *frames, f);
        }
        self.link_published_frame(frames, f, key, tier, version, Ghost(v));
    }

    pub fn publish_frame_with_token(
        &mut self,
        frames: &mut FrameTable,
        f: FrameId,
        Tracked(tok): Tracked<SlotToken>,
        ticket: Ticket,
        Ghost(cert): Ghost<ValueCertificate>,
        Ghost(v): Ghost<KernelView>,
    )
        requires
            old(self).well_formed(old(frames)),
            old(frames).consistent(v),
            f.0 < old(frames).metas.len(),
            old(frames).meta(f).state == STATE_INITIALIZING,
            !old(frames).meta(f).queued(),
            old(frames).frame(f).token is None,
            old(frames).versions@.contains_key(old(frames).frame(f).bound_version.unwrap()) ==> old(
                frames,
            ).versions@[old(frames).frame(f).bound_version.unwrap()] == cert.bytes,
            slot_is_owned(tok, v.id(), old(frames).frame(f).slot.loc),
            slot_frontier(tok).readers.is_empty(),
            slot_frontier(tok).last_writer == Some(cert.writer),
            cert.slot == old(frames).frame(f).slot,
            cert.bytes.len() == cert.slot.slot_len,
            v.timelines.issued(ticket.timeline_id(), ticket.sequence()),
            ticket != Ticket::NONE,
            cert.writer == v.timelines.events@[(ticket.timeline_id(), ticket.sequence())],
            event_succeeded(v.ledger, v.node, cert.writer),
            forall|h, run|
                reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                    v.ledger,
                    v.node,
                    h,
                    run,
                ) ==> #[trigger] certificate_valid_in(cert, h, run),
        ensures
            final(frames).num_tiers == old(frames).num_tiers,
            final(frames).frame_size == old(frames).frame_size && final(frames).tier_frame_sizes
                == old(frames).tier_frame_sizes && final(frames).metas.len() == old(
                frames,
            ).metas.len(),
            final(frames).versions@ == old(frames).versions@.insert(
                old(frames).frame(f).bound_version.unwrap(),
                cert.bytes,
            ),
            final(frames).frame(f) == (FrameGhost {
                token: Some(tok),
                cert: Some(cert),
                ..old(frames).frame(f)
            }),
            final(frames).preserves_initializing(old(frames), Some(f)),
            forall|g: FrameId|
                g != f && g.0 < old(frames).metas.len() ==> (#[trigger] final(frames).frame(g))
                    == old(frames).frame(g) && final(frames).meta(g).active_remote_leases == old(
                    frames,
                ).meta(g).active_remote_leases && (final(frames).published(g).is_some()
                    ==> final(frames).published(g) == old(frames).published(g)),
            forall|other: ObjectKey|
                other != old(frames).frame(f).bound_key.unwrap()
                    ==> #[trigger] final(self).view().contains_key(other) == old(
                    self,
                ).view().contains_key(other) && (old(self).view().contains_key(other)
                    ==> final(self).view()[other] == old(self).view()[other]),
            final(self).well_formed(final(frames)),
            final(frames).consistent(v),
            final(frames).meta(f).state == STATE_PUBLISHED,
            final(frames).frame(f).cert == Some(cert),
            final(self).slot(
                old(frames).frame(f).bound_key.unwrap(),
                old(frames).frame(f).tier.index() as int,
            ) == Some(f),
            final(self).view().contains_key(old(frames).frame(f).bound_key.unwrap()),
            final(self).view()[old(frames).frame(f).bound_key.unwrap()].version == old(
                frames,
            ).frame(f).bound_version.unwrap(),
            forall|u: int|
                #![trigger final(self).slot(old(frames).frame(f).bound_key.unwrap(), u)]
                0 <= u < old(self).num_tiers() && u != old(frames).frame(f).tier.index() ==> {
                    let key = old(frames).frame(f).bound_key.unwrap();
                    final(self).slot(key, u) == if old(self).view().contains_key(key) && old(
                        self,
                    ).view()[key].version == old(frames).frame(f).bound_version.unwrap() {
                        old(self).slot(key, u)
                    } else {
                        None
                    }
                },
            forall|k: ObjectKey, t: int|
                0 <= t < old(self).num_tiers() && k != old(frames).frame(f).bound_key.unwrap()
                    ==> #[trigger] final(self).slot(k, t) == old(self).slot(k, t),
    {
        let (key, tier) = frames.bindings[f.0 as usize];
        let version = VersionId::unpack(frames.metas[f.0 as usize].version_packed);
        proof {
            assert(frames.valid_frame(f, v));
            crate::spec::lemma_version_pack_unpack(
                frames.frame(f).bound_version.unwrap(),
                frames.meta(f).version_packed,
            );
        }
        frames.publish_with_token(f, Tracked(tok), ticket, Ghost(cert), Ghost(v));
        proof {
            self.begin_publication(*old(frames), *frames, f);
        }
        self.link_published_frame(frames, f, key, tier, version, Ghost(v));
    }

    proof fn begin_publication(&self, before: FrameTable, after: FrameTable, f: FrameId)
        requires
            self.well_formed(&before),
            before.num_tiers == after.num_tiers,
            before.meta(f).state == STATE_INITIALIZING,
            before.metas.len() == after.metas.len(),
            forall|g: FrameId|
                g != f && g.0 < before.metas.len() ==> #[trigger] after.meta(g) == before.meta(g)
                    && after.frame(g) == before.frame(g),
        ensures
            self.covers(&after, Some(f)),
    {
        assert forall|k: ObjectKey, u: int| 0 <= u < self.num_tiers() implies (#[trigger] self.slot(
            k,
            u,
        ) matches Some(g) ==> {
            &&& g != f
            &&& g.0 < after.metas.len()
            &&& after.meta(g).state == STATE_PUBLISHED
            &&& after.frame(g).tier.index() == u
            &&& after.frame(g).bound_key == Some(k)
            &&& after.frame(g).bound_version == Some(self.view()[k].version)
        }) by {
            if let Some(g) = self.slot(k, u) {
                assert(before.meta(g).state == STATE_PUBLISHED);
                assert(after.meta(g) == before.meta(g));
            }
        }
        assert forall|g: FrameId|
            g != f && g.0 < after.metas.len() && (#[trigger] after.meta(g)).state
                == STATE_PUBLISHED implies {
            let info = after.frame(g);
            &&& info.bound_key.is_some()
            &&& info.tier.valid(self.num_tiers())
            &&& self.slot(info.bound_key.unwrap(), info.tier.index() as int) == Some(g)
        } by {
            assert(before.meta(g).state == STATE_PUBLISHED);
        }
    }

    fn attach(
        &mut self,
        Ghost(frames): Ghost<&FrameTable>,
        key: ObjectKey,
        t: usize,
        f: FrameId,
        mut entry: CatalogEntry,
    )
        requires
            old(self).covers(frames, Some(f)),
            t < old(self).num_tiers(),
            f.0 < frames.metas.len(),
            frames.meta(f).state == STATE_PUBLISHED,
            frames.frame(f).tier.valid(old(self).num_tiers()),
            frames.frame(f).tier.index() == t,
            frames.frame(f).bound_key == Some(key),
            frames.frame(f).bound_version == Some(entry.version),
            forall|u: int|
                0 <= u < old(self).num_tiers() ==> #[trigger] entry.slots[u] == old(self).slot(
                    key,
                    u,
                ),
            forall|u: int|
                old(self).num_tiers() <= u < MAX_TIERS ==> (#[trigger] entry.slots[u]).is_none(),
            old(self).slot(key, t as int).is_none(),
            old(self).view().contains_key(key) ==> old(self).view()[key].version == entry.version,
        ensures
            forall|other: ObjectKey|
                other != key ==> #[trigger] final(self).view().contains_key(other) == old(
                    self,
                ).view().contains_key(other) && (old(self).view().contains_key(other)
                    ==> final(self).view()[other] == old(self).view()[other]),
            final(self).well_formed(frames),
            final(self).slot(key, t as int) == Some(f),
            final(self).view().contains_key(key),
            final(self).view()[key].version == entry.version,
            forall|u: int|
                0 <= u < old(self).num_tiers() && u != t ==> #[trigger] final(self).slot(key, u)
                    == old(self).slot(key, u),
            forall|k: ObjectKey, u: int|
                0 <= u < old(self).num_tiers() && k != key ==> #[trigger] final(self).slot(k, u)
                    == old(self).slot(k, u),
    {
        entry.slots[t] = Some(f);
        let ghost before_insert = *self;
        self.set_entry(key, Some(entry));
        assert forall|k: ObjectKey, u: int| 0 <= u < self.num_tiers() implies #[trigger] self.slot(
            k,
            u,
        ) == (if k == key && u == t {
            Some(f)
        } else {
            before_insert.slot(k, u)
        }) by {
            if k == key {
                assert(self.view().contains_key(key));
                assert(self.view()[key] == entry);
                if u == t {
                    assert(entry.slots[t as int] == Some(f));
                } else {
                    assert(entry.slots[u] == before_insert.slot(key, u));
                }
            }
        }
        assert forall|k: ObjectKey, u: int| self.num_tiers() <= u < MAX_TIERS implies (
        #[trigger] self.slot(k, u)).is_none() by {
            if k != key {
                assert(before_insert.slot(k, u).is_none());
            }
        }
        assert forall|k: ObjectKey, u: int| 0 <= u < self.num_tiers() implies (#[trigger] self.slot(
            k,
            u,
        ) matches Some(g) ==> {
            &&& g.0 < frames.metas.len()
            &&& frames.meta(g).state == STATE_PUBLISHED
            &&& frames.frame(g).tier.index() == u
            &&& frames.frame(g).bound_key == Some(k)
            &&& frames.frame(g).bound_version == Some(self.view()[k].version)
        }) by {
            if let Some(g) = self.slot(k, u) {
                if k != key || u != t {
                    assert(before_insert.slot(k, u) == Some(g));
                    assert(before_insert.view().contains_key(k));
                    assert(self.view()[k].version == before_insert.view()[k].version);
                }
            }
        }
        assert forall|g: FrameId|
            g.0 < frames.metas.len() && (#[trigger] frames.meta(g)).state
                == STATE_PUBLISHED implies {
            let info = frames.frame(g);
            &&& info.bound_key.is_some()
            &&& info.tier.valid(self.num_tiers())
            &&& self.slot(info.bound_key.unwrap(), info.tier.index() as int) == Some(g)
        } by {
            if g != f {
                let info = frames.frame(g);
                let bk = info.bound_key.unwrap();
                let bu = info.tier.index() as int;
                assert(before_insert.slot(bk, bu) == Some(g));
                assert(!(bk == key && bu == t));
            }
        }
        assert forall|k| self.view().contains_key(k) implies exists|u: int|
            0 <= u < self.num_tiers() && #[trigger] self.slot(k, u).is_some() by {
            if k == key {
                assert(self.slot(k, t as int).is_some());
            } else {
                let u = choose|u: int|
                    0 <= u < self.num_tiers() && #[trigger] before_insert.slot(k, u).is_some();
                assert(self.slot(k, u).is_some());
            }
        }
    }
}

} // verus!
