#[cfg(verus_only)]
mod proofs;
#[cfg(verus_only)]
pub use proofs::*;
#[cfg(verus_only)]
mod replica;
#[cfg(verus_only)]
pub use replica::*;

#[cfg(verus_only)]
use foundation::space::{Effect, Snapshot};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, EventInfo};
pub use transfer_core::context::SucceededMap;
use transfer_core::context::*;
use transfer_core::domain::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

pub tracked struct KnowledgeLedger {
    pub tracked submitted: SubmittedMap,
    pub tracked knows: Knows,
    pub tracked succeeded: SucceededMap,
    pub tracked revoked: RevokedMap,
}

pub open spec fn revoked_submap(a: RevokedMap, b: RevokedMap) -> bool {
    forall|k: (SlotLoc, TransferShare)| #[trigger]
        a.dom().contains(k) ==> b.dom().contains(k) && b.map()[k] == a.map()[k]
}

impl KnowledgeLedger {
    pub open spec fn valid(self, id: InstanceId, node: NodeId) -> bool {
        &&& self.submitted.instance_id() == id && self.knows.instance_id() == id
        &&& self.revoked.instance_id() == id
        &&& self.succeeded.instance_id == id && self.succeeded.well_formed()
        &&& forall|e| #[trigger] self.succeeded.contains(e) ==> self.submitted.dom().contains(e)
    }

    pub open spec fn has_submitted(self, e: TransferEvent) -> bool {
        self.submitted.dom().contains(e)
    }

    pub open spec fn has_succeeded(self, e: TransferEvent) -> bool {
        self.succeeded.contains(e)
    }

    pub open spec fn cites(self, node: NodeId, e: TransferEvent) -> bool {
        self.citations(node).contains(e)
    }

    pub proof fn empty(id: InstanceId) -> (tracked r: Self)
        ensures
            forall|node| #[trigger] r.valid(id, node),
            r.submitted.map() == IMap::empty(),
            r.knows.set() == ISet::empty(),
            r.succeeded.witnesses == Map::empty(),
            r.revoked.map() == IMap::empty(),
    {
        Self {
            submitted: SubmittedMap::empty(id),
            knows: Knows::empty(id),
            succeeded: SucceededMap::empty(id),
            revoked: RevokedMap::empty(id),
        }
    }

    pub proof fn record_submitted(tracked &mut self, tracked receipt: Submitted)
        requires
            old(self).submitted.instance_id() == receipt.instance_id(),
        ensures
            final(self).submitted.instance_id() == receipt.instance_id(),
            final(self).submitted.map() == old(self).submitted.map().insert(
                receipt.key(),
                receipt.value(),
            ),
            old(self).submitted.map().submap_of(final(self).submitted.map()),
            final(self).succeeded == old(self).succeeded,
            final(self).knows == old(self).knows,
            final(self).revoked == old(self).revoked,
    {
        if self.submitted.dom().contains(receipt.key()) {
            let tracked previous = self.submitted.remove(receipt.key());
            receipt.agree(&previous);
        }
        self.submitted.insert(receipt);
        assert(self.submitted.map() =~= old(self).submitted.map().insert(
            receipt.key(),
            receipt.value(),
        ));
    }

    pub proof fn record_submitted_map(tracked &mut self, tracked more: SubmittedMap)
        requires
            old(self).submitted.instance_id() == more.instance_id(),
        ensures
            final(self).submitted.instance_id() == more.instance_id(),
            final(self).submitted.map() == old(self).submitted.map().union_prefer_right(more.map()),
            old(self).submitted.map().submap_of(final(self).submitted.map()),
            more.map().submap_of(final(self).submitted.map()),
            final(self).succeeded == old(self).succeeded,
            final(self).knows == old(self).knows,
            final(self).revoked == old(self).revoked,
    {
        let id = more.instance_id();
        let tracked mut previous = SubmittedMap::empty(id);
        vstd::modes::tracked_swap(&mut previous, &mut self.submitted);
        let tracked mut prev_map = previous.into_map();
        let tracked more_map = more.into_map();
        if exists|k: TransferEvent| #[trigger]
            prev_map.dom().contains(k) && more_map.dom().contains(k) && prev_map[k].value()
                != more_map[k].value() {
            let k = choose|k: TransferEvent| #[trigger]
                prev_map.dom().contains(k) && more_map.dom().contains(k) && prev_map[k].value()
                    != more_map[k].value();
            let tracked a = prev_map.tracked_borrow(k);
            let tracked b = more_map.tracked_borrow(k);
            a.agree(b);
            assert(false);
        }
        prev_map.tracked_union_prefer_right(more_map);
        self.submitted = SubmittedMap::from_map(id, prev_map);
        assert(self.submitted.map() =~= old(self).submitted.map().union_prefer_right(more.map()));
        assert(old(self).submitted.map().submap_of(self.submitted.map()));
        assert(more.map().submap_of(self.submitted.map()));
    }

    pub proof fn record_knows(tracked &mut self, tracked observation: Knows)
        requires
            old(self).knows.instance_id() == observation.instance_id(),
        ensures
            final(self).knows.instance_id() == observation.instance_id(),
            final(self).knows.set() == old(self).knows.set().union(observation.set()),
            final(self).submitted == old(self).submitted,
            final(self).succeeded == old(self).succeeded,
            final(self).revoked == old(self).revoked,
    {
        let id = observation.instance_id();
        let tracked mut previous = Knows::empty(id);
        vstd::modes::tracked_swap(&mut previous, &mut self.knows);
        let tracked mut entries = previous.into_map();
        entries.tracked_union_prefer_right(observation.into_map());
        self.knows = Knows::from_map(id, entries);
        assert(self.knows.set() =~= old(self).knows.set().union(observation.set()));
    }

    pub proof fn record_succeeded(tracked &mut self, tracked witness: Succeeded)
        requires
            old(self).succeeded.well_formed(),
            witness.instance_id() == old(self).succeeded.instance_id,
            old(self).submitted.dom().contains(witness.event()),
        ensures
            final(self).submitted == old(self).submitted,
            final(self).knows == old(self).knows,
            final(self).revoked == old(self).revoked,
            final(self).succeeded.well_formed(),
            final(self).succeeded.instance_id == old(self).succeeded.instance_id,
            final(self).succeeded.witnesses == old(self).succeeded.witnesses.insert(
                witness.event(),
                witness,
            ),
    {
        self.succeeded.insert(witness);
    }

    pub proof fn record_succeeded_map(tracked &mut self, tracked more: SucceededMap)
        requires
            old(self).succeeded.well_formed(),
            more.well_formed(),
            more.instance_id == old(self).succeeded.instance_id,
            forall|e| #[trigger] more.contains(e) ==> old(self).submitted.dom().contains(e),
        ensures
            final(self).submitted == old(self).submitted,
            final(self).knows == old(self).knows,
            final(self).revoked == old(self).revoked,
            final(self).succeeded.well_formed(),
            final(self).succeeded.instance_id == old(self).succeeded.instance_id,
            final(self).succeeded.dom() == old(self).succeeded.dom().union(more.dom()),
            old(self).succeeded.dom().subset_of(final(self).succeeded.dom()),
            more.dom().subset_of(final(self).succeeded.dom()),
    {
        self.succeeded.witnesses.tracked_union_prefer_right(more.witnesses);
        assert(self.succeeded.dom() =~= old(self).succeeded.dom().union(more.dom()));
    }

    pub proof fn record_revoked(tracked &mut self, tracked tok: RevokedToken)
        requires
            old(self).revoked.instance_id() == tok.instance_id(),
        ensures
            final(self).revoked.instance_id() == tok.instance_id(),
            final(self).revoked.map() == old(self).revoked.map().insert(tok.key(), tok.value()),
            old(self).revoked.map().submap_of(final(self).revoked.map()),
            revoked_submap(old(self).revoked, final(self).revoked),
            final(self).submitted == old(self).submitted,
            final(self).knows == old(self).knows,
            final(self).succeeded == old(self).succeeded,
    {
        if self.revoked.dom().contains(tok.key()) {
            let tracked previous = self.revoked.remove(tok.key());
            tok.agree(&previous);
        }
        self.revoked.insert(tok);
        assert(self.revoked.map() =~= old(self).revoked.map().insert(tok.key(), tok.value()));
    }

    pub proof fn record_revoked_map(tracked &mut self, tracked more: RevokedMap)
        requires
            old(self).revoked.instance_id() == more.instance_id(),
        ensures
            final(self).revoked.instance_id() == more.instance_id(),
            final(self).revoked.map() == old(self).revoked.map().union_prefer_right(more.map()),
            old(self).revoked.map().submap_of(final(self).revoked.map()),
            revoked_submap(old(self).revoked, final(self).revoked),
            more.map().submap_of(final(self).revoked.map()),
            revoked_submap(more, final(self).revoked),
            final(self).submitted == old(self).submitted,
            final(self).knows == old(self).knows,
            final(self).succeeded == old(self).succeeded,
    {
        let id = more.instance_id();
        let tracked mut previous = RevokedMap::empty(id);
        vstd::modes::tracked_swap(&mut previous, &mut self.revoked);
        let tracked mut prev_map = previous.into_map();
        let tracked more_map = more.into_map();
        if exists|k: (SlotLoc, TransferShare)| #[trigger]
            prev_map.dom().contains(k) && more_map.dom().contains(k) && prev_map[k].value()
                != more_map[k].value() {
            let k = choose|k: (SlotLoc, TransferShare)| #[trigger]
                prev_map.dom().contains(k) && more_map.dom().contains(k) && prev_map[k].value()
                    != more_map[k].value();
            let tracked a = prev_map.tracked_borrow(k);
            let tracked b = more_map.tracked_borrow(k);
            a.agree(b);
            assert(false);
        }
        prev_map.tracked_union_prefer_right(more_map);
        self.revoked = RevokedMap::from_map(id, prev_map);
        assert(self.revoked.map() =~= old(self).revoked.map().union_prefer_right(more.map()));
        assert(old(self).revoked.map().submap_of(self.revoked.map()));
        assert(more.map().submap_of(self.revoked.map()));
    }

    /// Record a submitted event together with an observation of its completion,
    /// uniformly across any backend.
    pub proof fn record_observed_event(
        tracked &mut self,
        tracked receipt: Submitted,
        tracked observation: Knows,
    )
        requires
            receipt.instance_id() == old(self).submitted.instance_id(),
            observation.instance_id() == old(self).knows.instance_id(),
        ensures
            final(self).submitted.instance_id() == old(self).submitted.instance_id(),
            final(self).knows.instance_id() == old(self).knows.instance_id(),
            final(self).submitted.map() == old(self).submitted.map().insert(
                receipt.key(),
                receipt.value(),
            ),
            old(self).submitted.map().submap_of(final(self).submitted.map()),
            forall|node: NodeId|
                old(self).valid(receipt.instance_id(), node) ==> #[trigger] final(self).valid(
                    receipt.instance_id(),
                    node,
                ),
            final(self).knows.set() == old(self).knows.set().union(observation.set()),
            final(self).succeeded == old(self).succeeded,
            final(self).revoked == old(self).revoked,
    {
        self.record_submitted(receipt);
        self.record_knows(observation);
    }

    pub open spec fn citations(self, node: NodeId) -> ISet<TransferEvent> {
        ISet::new(|e| self.submitted.dom().contains(e) && self.knows.set().contains((node, e)))
    }
}

pub open spec fn event_succeeded(
    ledger: KnowledgeLedger,
    node: NodeId,
    event: TransferEvent,
) -> bool {
    if event.source.succeeds_on_observe() {
        ledger.citations(node).contains(event)
    } else {
        ledger.succeeded.contains(event)
    }
}

pub open spec fn effect_implies(
    actual: Effect<TransferDomain>,
    expected: Effect<TransferDomain>,
) -> bool {
    forall|input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>| #[trigger]
        actual(input, output) ==> expected(input, output)
}

/// Subrange-only copy relation implied by both `copy_effect` and single-event
/// multi-SGE scatter-gather effects.
pub open spec fn subrange_copy_effect(src: SlotRef, dst: SlotRef) -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>|
        input[src.loc].len() >= src.sub_offset + src.sub_len && output[dst.loc].len()
            == dst.slot_len && output[dst.loc].subrange(
            dst.sub_offset as int,
            (dst.sub_offset + dst.sub_len) as int,
        ) == input[src.loc].subrange(src.sub_offset as int, (src.sub_offset + src.sub_len) as int)
}

pub proof fn closure_monotonic(
    infos: IMap<TransferEvent, EventInfo<TransferDomain>>,
    seeds: ISet<TransferEvent>,
    more_infos: IMap<TransferEvent, EventInfo<TransferDomain>>,
    more_seeds: ISet<TransferEvent>,
)
    requires
        seeds.subset_of(more_seeds),
        seeds.subset_of(infos.dom()),
        forall|e| seeds.contains(e) ==> #[trigger] more_infos[e] == infos[e],
    ensures
        closure_from(infos, seeds).subset_of(closure_from(more_infos, more_seeds)),
{
    assert forall|e| closure_from(infos, seeds).contains(e) implies closure_from(
        more_infos,
        more_seeds,
    ).contains(e) by {
        if !seeds.contains(e) {
            let seed = choose|s| seeds.contains(s) && #[trigger] infos[s].preds.contains(e);
            assert(more_infos[seed].preds.contains(e));
        }
    }
}

/// One piece of a multi-piece gather into a single destination slot.
pub ghost struct GatherPiece {
    pub source: ValueCertificate,
    pub source_slot: SlotRef,
    pub dst_offset: u64,
    pub writer: TransferEvent,
}

/// Backend-agnostic inductive value proof over slot-level transfers.
pub ghost enum ValueProof {
    HostWrite,
    CopyHop { source: Box<ValueCertificate>, source_slot: SlotRef },
    GatherHop { pieces: Seq<GatherPiece> },
}

pub ghost struct ValueCertificate {
    pub slot: SlotRef,
    pub writer: TransferEvent,
    pub bytes: Seq<u8>,
    pub proof: ValueProof,
}

/// Erased snapshot of the source version at submission. Concurrent key eviction
/// does not discard the receipts on which an outstanding copy depends.
pub ghost struct ReplicaCopy {
    pub before: KnowledgeLedger,
    pub source: ValueCertificate,
    pub source_slot: SlotRef,
    pub writer: TransferEvent,
    pub destination: SlotRef,
}

/// Erased snapshot for a guarded remote read copy (RDMA READ via a `ShareCap`).
/// Unlike `ReplicaCopy`, the remote source slot may have been revoked and
/// overwritten concurrently by the donor; `observe_share_bound` + `revoked`
/// ensures any succeeded RDMA READ still observed `source.writer`.
pub ghost struct GuardedReplicaCopy {
    pub before: KnowledgeLedger,
    pub source: ValueCertificate,
    pub source_slot: SlotRef,
    pub share: TransferShare,
    pub writer: TransferEvent,
    pub destination: SlotRef,
}

/// Candidate certificate and the receipts covered by the publication ticket.
pub ghost struct PendingPutProof {
    pub certificate: ValueCertificate,
    pub events: Seq<TransferEvent>,
}

} // verus!
