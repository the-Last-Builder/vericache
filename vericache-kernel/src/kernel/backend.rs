//! Transactions and tracked borrows used by verified backend implementations.
use super::*;
#[cfg(verus_only)]
use vstd::tokens::ElementToken;

verus! {

impl CacheCore {
    /// The state shared by all backends outside the timeline, ledger, and
    /// cluster-wide version registry.
    pub open spec fn same_cache_state(&self, before: &Self) -> bool {
        &&& self.same_runtime_setup(before)
        &&& self.node_id_spec() == before.node_id_spec()
        &&& self.tiers_spec() == before.tiers_spec()
        &&& self.frames_spec() == before.frames_spec()
        &&& self.catalog_spec() == before.catalog_spec()
        &&& self.version_seq_start_spec() == before.version_seq_start_spec()
        &&& self.next_version_seq_spec() == before.next_version_seq_spec()
        &&& self.version_seq_limit_spec() == before.version_seq_limit_spec()
        &&& self.active_leases_spec() == before.active_leases_spec()
        &&& self.free_lease_slots_spec() == before.free_lease_slots_spec()
        &&& self.lease_generation_spec() == before.lease_generation_spec()
        &&& self.spec_spec() == before.spec_spec()
    }

    /// Reconstitute equality from the public projections of an opaque core.
    pub proof fn lemma_same_state(&self, before: &Self)
        requires
            self.same_cache_state(before),
            self.timelines_spec() == before.timelines_spec(),
            self.ledger_spec() == before.ledger_spec(),
            self.version_registry_spec() == before.version_registry_spec(),
        ensures
            *self == *before,
    {
    }

    pub proof fn borrow_ledger(tracked &self) -> (tracked r: &KnowledgeLedger)
        ensures
            *r == self.ledger_spec()@,
    {
        self.ledger.borrow()
    }

    pub proof fn borrow_version_registry(tracked &self) -> (tracked r: &Option<VersionRegistry>)
        ensures
            *r == self.version_registry_spec()@,
    {
        self.version_registry.borrow()
    }

    /// Borrow ghost knowledge for a driver submission that uses tracked receipts.
    pub proof fn borrow_ledger_mut(tracked &mut self) -> (tracked r: &mut KnowledgeLedger)
        ensures
            *r == old(self).ledger_spec()@,
            *final(r) == final(self).ledger_spec()@,
            final(self).same_cache_state(old(self)),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            *final(r) == *r ==> *final(self) == *old(self),
    {
        self.ledger.borrow_mut()
    }

    /// Temporarily access frame tokens while retaining the ledger needed to prove
    /// their preservation across backend transitions.
    pub proof fn borrow_frame_ghosts_and_ledger(tracked &mut self) -> (tracked r: (
        &mut Map<nat, FrameGhost>,
        &KnowledgeLedger,
    ))
        ensures
            *r.0 == old(self).frames_spec().ghosts_spec()@,
            *r.1 == old(self).ledger_spec()@,
            *final(r.0) == final(self).frames_spec().ghosts_spec()@,
            *final(r.0) == *r.0 ==> *final(self) == *old(self),
    {
        (self.frames.ghosts.borrow_mut(), self.ledger.borrow())
    }

    /// Reserve the naming authority for an in-flight local version.
    pub proof fn reserve_version_name(tracked &mut self, version: VersionId) -> (tracked r:
        VersionUnused)
        requires
            old(self).well_formed(),
            old(self).has_version_registry(),
            version.origin_node == old(self).node_id_spec(),
            version.seq as nat >= old(self).version_registry_spec()@->Some_0.source.min_seq(),
            version.seq < old(self).next_version_seq_spec(),
        ensures
            final(self).well_formed(),
            final(self).has_version_registry(),
            final(self).same_cache_state(old(self)),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_registry_spec()@->Some_0.bindings == old(
                self,
            ).version_registry_spec()@->Some_0.bindings,
            r.instance_id() == cluster_version_instance_id(),
            r.element() == (version.origin_node as NodeId, version.seq),
    {
        let tracked mut registry = None;
        vstd::modes::tracked_swap(self.version_registry.borrow_mut(), &mut registry);
        let tracked mut reg = registry.tracked_unwrap();
        let tracked name = reg.source.take_seq(version.seq);
        self.version_registry = Tracked(Some(reg));
        name
    }

    pub fn record_submission(
        &mut self,
        ops: &mut [Option<PendingOperation>; BITMAP_SLOTS],
        Ghost(bitmap): Ghost<SlidingBitmap>,
        t: TimelineId,
        frame: FrameId,
        operation: DataOperation,
        Ghost(pending): Ghost<Map<u64, TransferEvent>>,
        Tracked(receipt): Tracked<Submitted>,
    ) -> (ticket: Ticket)
        requires
            old(self).timelines_spec().well_formed(
                receipt.instance_id(),
                receipt.key().observer,
                &old(self).ledger_spec()@,
            ),
            t < old(self).timelines_spec().count_spec(),
            bitmap.well_formed(),
            bitmap.observations(
                &old(self).timelines_spec(),
                t,
                receipt.key().observer,
                &old(self).ledger_spec()@,
            ),
            ({
                let s = old(self).timelines_spec().next_seq_spec()[t as int];
                &&& s <= MAX_TICKET_SEQ && s <= old(
                    self,
                ).timelines_spec().confirmed_seq_spec()[t as int] + BITMAP_CAPACITY
                &&& pending.contains_key(s) && pending[s] == receipt.key()
                &&& window_matches(
                    *old(ops),
                    bitmap,
                    &old(self).timelines_spec(),
                    t,
                    pending.remove(s),
                )
            }),
        ensures
            final(self).same_cache_state(old(self)),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).timelines_spec().well_formed(
                receipt.instance_id(),
                receipt.key().observer,
                &final(self).ledger_spec()@,
            ),
            bitmap.observations(
                &final(self).timelines_spec(),
                t,
                receipt.key().observer,
                &final(self).ledger_spec()@,
            ),
            window_matches(*final(ops), bitmap, &final(self).timelines_spec(), t, pending),
            final(self).ledger_spec()@.submitted.map() == old(
                self,
            ).ledger_spec()@.submitted.map().insert(receipt.key(), receipt.value()),
            final(self).ledger_spec()@.knows == old(self).ledger_spec()@.knows,
            final(self).ledger_spec()@.succeeded == old(self).ledger_spec()@.succeeded,
            final(self).ledger_spec()@.revoked == old(self).ledger_spec()@.revoked,
            old(self).ledger_spec()@.submitted.map().submap_of(
                final(self).ledger_spec()@.submitted.map(),
            ),
            final(ops)@ == old(ops)@.update(
                (ticket.sequence() % BITMAP_CAPACITY) as int,
                Some(PendingOperation { sequence: ticket.sequence(), frame, operation }),
            ),
            ticket.sequence() == old(self).timelines_spec().next_seq_spec()[t as int]
                && ticket.timeline_id() == t && ticket != Ticket::NONE,
            final(self).timelines_spec().events_spec()@ == old(
                self,
            ).timelines_spec().events_spec()@.insert(
                (ticket.timeline_id(), ticket.sequence()),
                receipt.key(),
            ),
            final(self).timelines_spec().next_seq_spec()@ == old(
                self,
            ).timelines_spec().next_seq_spec()@.update(
                ticket.timeline_id() as int,
                (ticket.sequence() + 1) as u64,
            ),
            final(self).timelines_spec().kinds_spec() == old(self).timelines_spec().kinds_spec(),
            final(self).timelines_spec().count_spec() == old(self).timelines_spec().count_spec(),
            final(self).timelines_spec().confirmed_seq_spec() == old(
                self,
            ).timelines_spec().confirmed_seq_spec(),
    {
        crate::timeline::record_submission(
            ops,
            Ghost(bitmap),
            &mut self.timelines,
            t,
            frame,
            operation,
            Ghost(pending),
            Tracked(receipt),
            Tracked(self.ledger.borrow_mut()),
        )
    }

    pub fn complete_data(
        &mut self,
        ops: &mut [Option<PendingOperation>; BITMAP_SLOTS],
        bitmap: &mut SlidingBitmap,
        t: TimelineId,
        s: u64,
        Ghost(pending): Ghost<Map<u64, TransferEvent>>,
        Ghost(node): Ghost<NodeId>,
        Tracked(observation): Tracked<Knows>,
        Tracked(success): Tracked<Option<transfer_core::context::Succeeded>>,
    ) -> (op: PendingOperation)
        requires
            old(self).timelines_spec().well_formed(
                observation.instance_id(),
                node,
                &old(self).ledger_spec()@,
            ),
            t < old(self).timelines_spec().count_spec(),
            old(bitmap).well_formed(),
            old(bitmap).observations(
                &old(self).timelines_spec(),
                t,
                node,
                &old(self).ledger_spec()@,
            ),
            window_matches(*old(ops), *old(bitmap), &old(self).timelines_spec(), t, pending),
            pending.contains_key(s),
            old(self).timelines_spec().observed(t, s, node, observation),
            success matches Some(w) ==> w.event() == pending[s] && w.instance_id()
                == observation.instance_id(),
        ensures
            final(self).same_cache_state(old(self)),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).timelines_spec().well_formed(
                observation.instance_id(),
                node,
                &final(self).ledger_spec()@,
            ),
            final(bitmap).well_formed(),
            final(bitmap).observations(
                &final(self).timelines_spec(),
                t,
                node,
                &final(self).ledger_spec()@,
            ),
            window_matches(
                *final(ops),
                *final(bitmap),
                &final(self).timelines_spec(),
                t,
                pending.remove(s),
            ),
            Some(op) == old(ops)[(s % BITMAP_CAPACITY) as int],
            final(ops)@ == old(ops)@.update((s % BITMAP_CAPACITY) as int, None),
            op.sequence == s,
            final(self).timelines_spec().polled(&old(self).timelines_spec(), t),
            final(self).timelines_spec().kinds_spec() == old(self).timelines_spec().kinds_spec(),
            final(self).timelines_spec().count_spec() == old(self).timelines_spec().count_spec(),
            final(self).timelines_spec().next_seq_spec() == old(
                self,
            ).timelines_spec().next_seq_spec(),
            final(self).timelines_spec().events_spec() == old(self).timelines_spec().events_spec(),
            final(self).ledger_spec()@.submitted == old(self).ledger_spec()@.submitted,
            final(self).ledger_spec()@.revoked == old(self).ledger_spec()@.revoked,
            final(self).ledger_spec()@.knows.set() == old(self).ledger_spec()@.knows.set().union(
                observation.set(),
            ),
            final(self).ledger_spec()@.succeeded.witnesses == match success {
                None => old(self).ledger_spec()@.succeeded.witnesses,
                Some(w) => old(self).ledger_spec()@.succeeded.witnesses.insert(pending[s], w),
            },
    {
        crate::timeline::complete_data(
            ops,
            bitmap,
            &mut self.timelines,
            t,
            s,
            Ghost(pending),
            Ghost(node),
            Tracked(observation),
            Tracked(success),
            Tracked(self.ledger.borrow_mut()),
        )
    }
}

} // verus!
