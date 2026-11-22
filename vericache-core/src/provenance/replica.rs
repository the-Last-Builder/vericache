use super::*;
use foundation::{
    history::{reachable, reachable_satisfies_invariant, History},
    order::is_last,
    space::conflict_at,
    value::*,
};

verus! {

/// Every predecessor of a recorded transfer has a receipt in `submitted`.
pub open spec fn complete_receipts_map(submitted: SubmittedMap) -> bool {
    forall|e| #[trigger]
        submitted.dom().contains(e) ==> submitted.map()[e].preds.subset_of(submitted.dom())
}

/// Every predecessor of a locally submitted transfer has a receipt in this directory.
pub open spec fn complete_receipts(ledger: KnowledgeLedger) -> bool {
    complete_receipts_map(ledger.submitted)
}

/// The slot's certified writer is still the latest recorded writer on this shard.
pub open spec fn recorded_writer_certified(
    cert: ValueCertificate,
    ledger: KnowledgeLedger,
    node: NodeId,
    h: History<TransferDomain>,
) -> bool {
    let loc = cert.slot.loc;
    let w = cert.writer;
    &&& ledger.citations(node).contains(w)
    &&& (h.access)(w).writes.contains(loc)
    &&& forall|e|
        ledger.submitted.dom().contains(e) && (#[trigger] (h.access)(e)).writes.contains(loc) && e
            != w ==> (h.happens_before)(e, w)
}

pub open spec fn replica_valid(
    ledger: KnowledgeLedger,
    node: NodeId,
    cert: ValueCertificate,
) -> bool {
    &&& event_succeeded(ledger, node, cert.writer)
    &&& forall|h, run|
        reachable(h) && valid_run(h, run) && ledger_consistent_with_history(ledger, node, h, run)
            ==> #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
            cert,
            ledger,
            node,
            h,
        )
}

pub proof fn replica_survives_disjoint_writes(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    cert: ValueCertificate,
)
    requires
        before.valid(before.submitted.instance_id(), node),
        replica_valid(before, node, cert),
        before.submitted.map().submap_of(after.submitted.map()),
        revoked_submap(before.revoked, after.revoked),
        forall|e| #[trigger]
            after.submitted.dom().contains(e) && !before.submitted.dom().contains(e)
                ==> !after.submitted.map()[e].access.writes.contains(cert.slot.loc),
        before.knows.set().subset_of(after.knows.set()),
        before.succeeded.dom().subset_of(after.succeeded.dom()),
    ensures
        replica_valid(after, node, cert),
{
    assert(before.citations(node).subset_of(after.citations(node)));
    assert(event_succeeded(after, node, cert.writer));
    assert forall|h, run|
        reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
            after,
            node,
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
        cert,
        after,
        node,
        h,
    ) by {
        ledger_history_restrict(before, after, node, h, run);
        assert(certificate_valid_in(cert, h, run));
        assert(recorded_writer_certified(cert, before, node, h));
        let loc = cert.slot.loc;
        let w = cert.writer;
        assert(after.submitted.dom().contains(w));
        assert forall|e|
            after.submitted.dom().contains(e) && (#[trigger] (h.access)(e)).writes.contains(loc)
                && e != w implies (h.happens_before)(e, w) by {
            if !before.submitted.dom().contains(e) {
                assert(!after.submitted.map()[e].access.writes.contains(loc));
            }
        }
    }
}

/// Citing the source frontier makes `cert.writer` the last writer for `reader`.
pub proof fn local_copy_last_writer(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    cert: ValueCertificate,
    reader: TransferEvent,
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
)
    requires
        ledger_consistent_with_history(after, node, h, run),
        recorded_writer_certified(cert, before, node, h),
        before.submitted.map().submap_of(after.submitted.map()),
        after.submitted.dom().contains(reader),
        after.submitted.map()[reader].preds.subset_of(after.submitted.dom()),
        forall|e| #[trigger]
            after.submitted.map()[reader].preds.contains(e) && !before.submitted.dom().contains(e)
                ==> !after.submitted.map()[e].access.writes.contains(cert.slot.loc),
        before.citations(node).subset_of(after.submitted.map()[reader].preds),
    ensures
        last_writer_certified(h, reader, cert),
{
    let loc = cert.slot.loc;
    let w = cert.writer;
    assert(after.submitted.map()[reader].preds.contains(w));
    assert(after.submitted.dom().contains(w));
    assert forall|e| writers_before(h, reader, loc).contains(e) && e != w implies #[trigger] (
    h.happens_before)(e, w) by {
        assert(after.submitted.map()[reader].preds.contains(e));
        assert(after.submitted.dom().contains(e));
        assert(after.submitted.map()[e].access == (h.access)(e));
        assert(before.submitted.dom().contains(e));
        assert((h.access)(e).writes.contains(loc));
    }
}

/// A newly submitted slot write follows every earlier recorded writer of its slot.
pub proof fn latest_recorded_writer(
    before: KnowledgeLedger,
    l: KnowledgeLedger,
    node: NodeId,
    cert: ValueCertificate,
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
)
    requires
        reachable(h),
        ledger_consistent_with_history(l, node, h, run),
        complete_receipts(before),
        before.submitted.map().submap_of(l.submitted.map()),
        !before.submitted.dom().contains(cert.writer),
        l.submitted.dom().contains(cert.writer),
        l.citations(node).contains(cert.writer),
        l.submitted.map()[cert.writer].access.writes.contains(cert.slot.loc),
        forall|e| #[trigger]
            l.submitted.dom().contains(e) && !before.submitted.dom().contains(e) && e != cert.writer
                ==> !l.submitted.map()[e].access.writes.contains(cert.slot.loc),
    ensures
        recorded_writer_certified(cert, l, node, h),
{
    let w = cert.writer;
    let loc = cert.slot.loc;
    reachable_satisfies_invariant(h);
    assert((h.access)(w).writes.contains(loc));
    assert forall|e|
        l.submitted.dom().contains(e) && (#[trigger] (h.access)(e)).writes.contains(loc) && e
            != w implies (h.happens_before)(e, w) by {
        assert(l.submitted.map()[e].access == (h.access)(e));
        assert(before.submitted.dom().contains(e));
        assert(!before.submitted.map()[e].preds.contains(w));
        assert(!(h.happens_before)(w, e));
        assert(conflict_at((h.access)(e), (h.access)(w), loc));
        assert(!(h.read_guards)(e).dom().contains(loc));
        assert(!(h.read_guards)(w).dom().contains(loc));
    }
}

pub open spec fn copied_slice_certificate(
    source: ValueCertificate,
    source_slot: SlotRef,
    writer: TransferEvent,
    dst: SlotRef,
) -> ValueCertificate {
    ValueCertificate {
        slot: dst,
        bytes: source_slice_bytes(source, source_slot),
        writer,
        proof: ValueProof::CopyHop { source: Box::new(source), source_slot },
    }
}

pub open spec fn copied_certificate(
    source: ValueCertificate,
    writer: TransferEvent,
    dst: SlotRef,
) -> ValueCertificate {
    copied_slice_certificate(source, source.slot, writer, dst)
}

/// A successful concrete byte write provides a history-independent replica certificate.
pub proof fn certify_host_replica(
    before: KnowledgeLedger,
    ledger: KnowledgeLedger,
    node: NodeId,
    writer: TransferEvent,
    slot: SlotRef,
    bytes: Seq<u8>,
) -> (cert: ValueCertificate)
    requires
        complete_receipts(before),
        !before.submitted.dom().contains(writer),
        before.submitted.map().submap_of(ledger.submitted.map()),
        ledger.submitted.dom() == before.submitted.dom().insert(writer),
        slot.well_formed(),
        bytes.len() == slot.sub_len,
        ledger.citations(node).contains(writer),
        event_succeeded(ledger, node, writer),
        ledger.submitted.map()[writer].access.writes.contains(slot.loc),
        effect_implies(ledger.submitted.map()[writer].effect, splice_bytes(slot, bytes)),
    ensures
        replica_valid(ledger, node, cert),
        cert.writer == writer,
        cert.slot == slot,
        cert.bytes == bytes,
        cert.proof == ValueProof::HostWrite,
{
    let cert = ValueCertificate { writer, slot, bytes, proof: ValueProof::HostWrite };
    assert forall|h, run|
        reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
            ledger,
            node,
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
        cert,
        ledger,
        node,
        h,
    ) by {
        let established = lemma_host_write_certifies(ledger, node, h, run, writer, slot, bytes);
        assert(established == cert);
        latest_recorded_writer(before, ledger, node, cert, h, run);
    }
    cert
}

/// A transfer keeps its value proof even when the source replica has since been evicted.
pub proof fn lemma_completed_replica_copy(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    source: ValueCertificate,
    source_slot: SlotRef,
    writer: TransferEvent,
    dst: SlotRef,
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
)
    requires
        before.valid(before.submitted.instance_id(), node),
        before.submitted.map().submap_of(after.submitted.map()),
        revoked_submap(before.revoked, after.revoked),
        before.succeeded.dom().subset_of(after.succeeded.dom()),
        before.citations(node).subset_of(after.citations(node)),
        replica_valid(before, node, source),
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(after, node, h, run),
        event_succeeded(after, node, writer),
        after.submitted.dom().contains(writer),
        source_slice_valid(source, source_slot),
        dst.well_formed(),
        source_slot.sub_len == dst.sub_len,
        after.submitted.map()[writer].access.reads.contains(source_slot.loc),
        after.submitted.map()[writer].access.writes.contains(dst.loc),
        effect_implies(after.submitted.map()[writer].effect, copy_effect(source_slot, dst)),
        after.submitted.map()[writer].preds.subset_of(before.submitted.dom()),
        before.citations(node).subset_of(after.submitted.map()[writer].preds),
    ensures
        certificate_valid_in(copied_slice_certificate(source, source_slot, writer, dst), h, run),
{
    ledger_history_restrict(before, after, node, h, run);
    assert(certificate_valid_in(source, h, run));
    local_copy_last_writer(before, after, node, source, writer, h, run);
    let result = lemma_copy_hop_preserves_value(
        after,
        node,
        h,
        run,
        source,
        source_slot,
        writer,
        dst,
    );
    assert(result == copied_slice_certificate(source, source_slot, writer, dst));
}

impl ReplicaCopy {
    pub open spec fn retained(self, l: KnowledgeLedger, node: NodeId) -> bool {
        let d = self.destination;
        let info = l.submitted.map()[self.writer];
        &&& self.before.valid(l.submitted.instance_id(), node)
        &&& complete_receipts(self.before) && !self.before.submitted.dom().contains(self.writer)
        &&& self.before.submitted.map().submap_of(l.submitted.map()) && revoked_submap(
            self.before.revoked,
            l.revoked,
        ) && self.before.succeeded.dom().subset_of(l.succeeded.dom())
        &&& self.before.citations(node).subset_of(l.citations(node))
        &&& replica_valid(self.before, node, self.source)
        &&& l.submitted.dom().contains(self.writer)
        &&& source_slice_valid(self.source, self.source_slot)
        &&& d.well_formed() && self.source_slot.sub_len == d.sub_len
        &&& info.access.reads.contains(self.source_slot.loc) && info.access.writes.contains(d.loc)
        &&& effect_implies(info.effect, copy_effect(self.source_slot, d))
        &&& info.preds.subset_of(self.before.submitted.dom())
        &&& self.before.citations(node).subset_of(info.preds)
        &&& forall|e| #[trigger]
            l.submitted.dom().contains(e) && !self.before.submitted.dom().contains(e) && e
                != self.writer ==> !l.submitted.map()[e].access.writes.contains(d.loc)
    }

    pub proof fn certify(self, l: KnowledgeLedger, node: NodeId) -> (cert: ValueCertificate)
        requires
            self.retained(l, node),
            event_succeeded(l, node, self.writer),
            l.citations(node).contains(self.writer),
        ensures
            cert == copied_slice_certificate(
                self.source,
                self.source_slot,
                self.writer,
                self.destination,
            ),
            replica_valid(l, node, cert),
    {
        let w = self.writer;
        let cert = copied_slice_certificate(self.source, self.source_slot, w, self.destination);
        assert forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                l,
                node,
                h,
                run,
            ) implies #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
            cert,
            l,
            node,
            h,
        ) by {
            lemma_completed_replica_copy(
                self.before,
                l,
                node,
                self.source,
                self.source_slot,
                w,
                self.destination,
                h,
                run,
            );
            latest_recorded_writer(self.before, l, node, cert, h, run);
        }
        cert
    }
}

impl GuardedReplicaCopy {
    pub open spec fn retained(self, l: KnowledgeLedger, node: NodeId) -> bool {
        let d = self.destination;
        let info = l.submitted.map()[self.writer];
        &&& self.before.valid(l.submitted.instance_id(), node)
        &&& self.before.submitted.map().submap_of(l.submitted.map())
        &&& revoked_submap(self.before.revoked, l.revoked)
        &&& self.before.succeeded.dom().subset_of(l.succeeded.dom())
        &&& self.before.citations(node).subset_of(l.citations(node))
        &&& forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                self.before,
                node,
                h,
                run,
            ) ==> #[trigger] certificate_valid_in(self.source, h, run)
        &&& l.submitted.dom().contains(self.writer)
        &&& l.submitted.dom().contains(self.source.writer)
        &&& source_slice_valid(self.source, self.source_slot)
        &&& d.well_formed() && self.source_slot.sub_len == d.sub_len
        &&& info.access.reads.contains(self.source_slot.loc) && info.access.writes.contains(d.loc)
        &&& effect_implies(info.effect, copy_effect(self.source_slot, d))
        &&& info.preds.contains(self.source.writer)
        &&& info.preds.subset_of(l.submitted.dom())
        &&& info.read_guards.dom().contains(self.source_slot.loc)
        &&& info.read_guards[self.source_slot.loc] == self.share
    }

    pub proof fn certify(self, l: KnowledgeLedger, node: NodeId) -> (cert: ValueCertificate)
        requires
            self.retained(l, node),
            event_succeeded(l, node, self.writer),
            l.citations(node).contains(self.writer),
            forall|e: TransferEvent| #[trigger]
                l.submitted.dom().contains(e) && l.submitted.map()[e].access.writes.contains(
                    self.source_slot.loc,
                ) && e != self.source.writer ==> (
                l.submitted.map()[self.source.writer].preds.contains(e) || (
                l.revoked.dom().contains((self.source_slot.loc, self.share))
                    && l.submitted.map()[e].preds.contains(
                    l.revoked.map()[(self.source_slot.loc, self.share)],
                ))),
            forall|e: TransferEvent| #[trigger]
                l.submitted.dom().contains(e) && l.submitted.map()[e].access.writes.contains(
                    self.destination.loc,
                ) && e != self.writer ==> l.submitted.map()[self.writer].preds.contains(e),
        ensures
            cert == copied_slice_certificate(
                self.source,
                self.source_slot,
                self.writer,
                self.destination,
            ),
            replica_valid(l, node, cert),
    {
        let w = self.writer;
        let cert = copied_slice_certificate(self.source, self.source_slot, w, self.destination);
        assert forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                l,
                node,
                h,
                run,
            ) implies #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
            cert,
            l,
            node,
            h,
        ) by {
            ledger_history_restrict(self.before, l, node, h, run);
            assert(certificate_valid_in(self.source, h, run));
            let res = lemma_guarded_copy_hop_preserves_value(
                l,
                node,
                h,
                run,
                self.source,
                self.source_slot,
                self.share,
                w,
                self.destination,
            );
            assert(res == cert);
            assert forall|e|
                l.submitted.dom().contains(e) && (#[trigger] (h.access)(e)).writes.contains(
                    cert.slot.loc,
                ) && e != w implies (h.happens_before)(e, w) by {
                assert(l.submitted.map()[e].access == (h.access)(e));
                assert(l.submitted.map()[w].preds.contains(e));
            }
        }
        cert
    }
}

/// Ledger growth that leaves the destination slot unwritten.
pub open spec fn preserves_destination(
    before: KnowledgeLedger,
    after: KnowledgeLedger,
    node: NodeId,
    dst: SlotLoc,
) -> bool {
    &&& before.submitted.instance_id() == after.submitted.instance_id()
    &&& before.submitted.map().submap_of(after.submitted.map())
    &&& revoked_submap(before.revoked, after.revoked)
    &&& before.succeeded.dom().subset_of(after.succeeded.dom())
    &&& before.citations(node).subset_of(after.citations(node))
    &&& forall|e| #[trigger]
        after.submitted.dom().contains(e) && !before.submitted.dom().contains(e)
            ==> !after.submitted.map()[e].access.writes.contains(dst)
}

impl PendingPutProof {
    pub open spec fn observed(self, l: KnowledgeLedger, node: NodeId) -> bool {
        forall|p: int|
            0 <= p < self.events.len() ==> l.citations(node).contains(#[trigger] self.events[p])
                && event_succeeded(l, node, self.events[p])
    }

    #[verifier::opaque]
    pub open spec fn pending(self, l: KnowledgeLedger, node: NodeId) -> bool {
        forall|after: KnowledgeLedger|
            preserves_destination(l, after, node, self.certificate.slot.loc)
                && #[trigger] self.observed(after, node) ==> replica_valid(
                after,
                node,
                self.certificate,
            )
    }

    pub proof fn preserve(self, before: KnowledgeLedger, l: KnowledgeLedger, node: NodeId)
        requires
            self.pending(before, node),
            preserves_destination(before, l, node, self.certificate.slot.loc),
        ensures
            self.pending(l, node),
    {
        reveal(PendingPutProof::pending);
        assert forall|after: KnowledgeLedger|
            preserves_destination(l, after, node, self.certificate.slot.loc)
                && #[trigger] self.observed(after, node) implies replica_valid(
            after,
            node,
            self.certificate,
        ) by {
            vstd::imap_lib::lemma_submap_of_trans(
                before.submitted.map(),
                l.submitted.map(),
                after.submitted.map(),
            );
            vstd::imap_lib::lemma_submap_of_trans(
                before.revoked.map(),
                l.revoked.map(),
                after.revoked.map(),
            );
            assert forall|e| #[trigger]
                after.submitted.dom().contains(e) && !before.submitted.dom().contains(
                    e,
                ) implies !after.submitted.map()[e].access.writes.contains(
                self.certificate.slot.loc,
            ) by {
                if l.submitted.dom().contains(e) {
                    assert(l.submitted.map()[e] == after.submitted.map()[e]);
                }
            }
            assert(preserves_destination(before, after, node, self.certificate.slot.loc));
        }
    }

    pub proof fn certify(self, l: KnowledgeLedger, node: NodeId) -> (cert: ValueCertificate)
        requires
            self.pending(l, node),
            self.observed(l, node),
        ensures
            cert == self.certificate,
            replica_valid(l, node, cert),
    {
        reveal(PendingPutProof::pending);
        assert(preserves_destination(l, l, node, self.certificate.slot.loc));
        self.certificate
    }

    pub proof fn single(copy: ReplicaCopy, l: KnowledgeLedger, node: NodeId) -> (c: Self)
        requires
            copy.retained(l, node),
        ensures
            c.certificate == copied_slice_certificate(
                copy.source,
                copy.source_slot,
                copy.writer,
                copy.destination,
            ),
            c.events == seq![copy.writer],
            c.pending(l, node),
    {
        let c = Self {
            certificate: copied_slice_certificate(
                copy.source,
                copy.source_slot,
                copy.writer,
                copy.destination,
            ),
            events: seq![copy.writer],
        };
        reveal(PendingPutProof::pending);
        assert forall|after: KnowledgeLedger|
            preserves_destination(l, after, node, copy.destination.loc) && #[trigger] c.observed(
                after,
                node,
            ) implies replica_valid(after, node, c.certificate) by {
            vstd::imap_lib::lemma_submap_of_trans(
                copy.before.submitted.map(),
                l.submitted.map(),
                after.submitted.map(),
            );
            vstd::imap_lib::lemma_submap_of_trans(
                copy.before.revoked.map(),
                l.revoked.map(),
                after.revoked.map(),
            );
            assert forall|e| #[trigger]
                after.submitted.dom().contains(e) && !copy.before.submitted.dom().contains(e) && e
                    != copy.writer implies !after.submitted.map()[e].access.writes.contains(
                copy.destination.loc,
            ) by {
                if l.submitted.dom().contains(e) {
                    assert(l.submitted.map()[e] == after.submitted.map()[e]);
                }
            }
            assert(copy.retained(after, node));
            assert(c.events[0] == copy.writer);
            copy.certify(after, node);
        }
        c
    }
}

pub open spec fn gather_certificate(
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
) -> ValueCertificate {
    ValueCertificate {
        slot: dst,
        writer: pieces.last().writer,
        bytes,
        proof: ValueProof::GatherHop { pieces },
    }
}

pub open spec fn gather_put_proof(
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
) -> PendingPutProof {
    PendingPutProof {
        certificate: gather_certificate(pieces, dst, bytes),
        events: pieces.map(|_i: int, p: GatherPiece| p.writer),
    }
}

/// Backend-agnostic predicate describing a staged multi-piece gather (covers both
/// sequential multi-event gathers like Host/CUDA/NVMe and single-event multi-SGE RDMA).
pub open spec fn submitted_gather(
    before: KnowledgeLedger,
    staged: KnowledgeLedger,
    node: NodeId,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
) -> bool {
    let single = gather_is_single_event(pieces, pieces.last().writer);
    &&& before.valid(before.submitted.instance_id(), node)
    &&& complete_receipts(before)
    &&& complete_receipts(staged)
    &&& before.submitted.map().submap_of(staged.submitted.map())
    &&& revoked_submap(before.revoked, staged.revoked)
    &&& before.citations(node).subset_of(staged.citations(node))
    &&& before.succeeded.dom().subset_of(staged.succeeded.dom())
    &&& dst.is_full() && dst.slot_len == bytes.len()
    &&& gather_pieces_contiguous(dst, pieces)
    &&& (!single ==> forall|p: int, q: int|
        0 <= p && q == p + 1 && q < pieces.len() ==> {
            let prev = #[trigger] pieces[p];
            let curr = #[trigger] pieces[q];
            &&& staged.submitted.map()[curr.writer].preds.contains(prev.writer)
            &&& forall|x| #[trigger]
                staged.submitted.map()[curr.writer].preds.contains(x)
                    && !before.submitted.dom().contains(x) && x != prev.writer
                    ==> !staged.submitted.map()[x].access.writes.contains(dst.loc)
                    || staged.submitted.map()[prev.writer].preds.contains(x)
        })
    &&& forall|p: int|
        0 <= p < pieces.len() ==> {
            let piece = #[trigger] pieces[p];
            let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
            &&& source_slice_valid(piece.source, piece.source_slot)
            &&& piece.source.slot.well_formed()
            &&& piece.source.bytes.len() == piece.source.slot.sub_len
            &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
                piece.dst_offset as int,
                (piece.dst_offset + piece.source_slot.sub_len) as int,
            )
            &&& piece.source_slot.loc != dst.loc
            &&& replica_valid(before, node, piece.source)
            &&& staged.submitted.dom().contains(piece.writer)
            &&& !before.submitted.dom().contains(piece.writer)
            &&& staged.submitted.map()[piece.writer].access.reads.contains(piece.source_slot.loc)
            &&& staged.submitted.map()[piece.writer].access.writes.contains(dst.loc)
            &&& before.citations(node).subset_of(staged.submitted.map()[piece.writer].preds)
            &&& if single {
                effect_implies(
                    staged.submitted.map()[piece.writer].effect,
                    subrange_copy_effect(piece.source_slot, dst_sub),
                )
            } else {
                &&& effect_implies(
                    staged.submitted.map()[piece.writer].effect,
                    copy_effect(piece.source_slot, dst_sub),
                )
                &&& (p > 0 && !dst_sub.is_full()
                    ==> staged.submitted.map()[piece.writer].access.reads.contains(dst.loc))
            }
        }
    &&& forall|x| #[trigger]
        staged.submitted.dom().contains(x) && !before.submitted.dom().contains(x)
            ==> staged.submitted.map()[x].access.writes.subset_of(ISet::empty().insert(dst.loc))
            && (x != pieces.last().writer ==> (!staged.submitted.map()[x].access.writes.contains(
            dst.loc,
        ) || staged.submitted.map()[pieces.last().writer].preds.contains(x)))
}

proof fn stage_gather_piece_chain(
    before: KnowledgeLedger,
    staged: KnowledgeLedger,
    l: KnowledgeLedger,
    node: NodeId,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
)
    requires
        submitted_gather(before, staged, node, pieces, dst, bytes),
        preserves_destination(staged, l, node, dst.loc),
        gather_put_proof(pieces, dst, bytes).observed(l, node),
        before.submitted.map().submap_of(l.submitted.map()),
        revoked_submap(before.revoked, l.revoked),
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(l, node, h, run),
        !gather_is_single_event(pieces, pieces.last().writer),
    ensures
        gather_sequential_chain(pieces, h),
        gather_sequential_last_writers(pieces, dst, h),
{
    let proof_obj = gather_put_proof(pieces, dst, bytes);
    reachable_satisfies_invariant(h);
    assert forall|p: int, q: int| 0 <= p && q == p + 1 && q < pieces.len() implies (
    h.happens_before)((#[trigger] pieces[p]).writer, (#[trigger] pieces[q]).writer) && is_last(
        h.happens_before,
        writers_before(h, pieces[q].writer, dst.loc),
        pieces[p].writer,
    ) by {
        let prev = pieces[p];
        let curr = pieces[q];
        assert(proof_obj.events[q] == curr.writer);
        assert(proof_obj.events[p] == prev.writer);
        assert(staged.submitted.dom().contains(curr.writer));
        assert(staged.submitted.dom().contains(prev.writer));
        assert(l.submitted.dom().contains(curr.writer));
        assert(l.submitted.dom().contains(prev.writer));
        assert(staged.submitted.map()[curr.writer] == l.submitted.map()[curr.writer]);
        assert(staged.submitted.map()[prev.writer] == l.submitted.map()[prev.writer]);
        assert(l.submitted.map()[curr.writer].preds.contains(prev.writer));
        assert((h.happens_before)(prev.writer, curr.writer));
        assert forall|x|
            writers_before(h, curr.writer, dst.loc).contains(x) && x
                != prev.writer implies #[trigger] (h.happens_before)(x, prev.writer) by {
            assert(staged.submitted.map()[curr.writer].preds.contains(x));
            assert(staged.submitted.dom().contains(x));
            assert(l.submitted.dom().contains(x));
            assert(staged.submitted.map()[x] == l.submitted.map()[x]);
            if before.submitted.dom().contains(x) {
                assert(!before.submitted.map()[x].preds.contains(prev.writer));
                assert(!(h.happens_before)(prev.writer, x));
                assert(conflict_at((h.access)(x), (h.access)(prev.writer), dst.loc));
                assert(!(h.read_guards)(x).dom().contains(dst.loc));
                assert(!(h.read_guards)(prev.writer).dom().contains(dst.loc));
            } else {
                assert(staged.submitted.map()[prev.writer].preds.contains(x));
            }
        }
        assert(is_last(h.happens_before, writers_before(h, curr.writer, dst.loc), prev.writer));
    }
}

proof fn stage_gather_valid_in_run(
    before: KnowledgeLedger,
    staged: KnowledgeLedger,
    l: KnowledgeLedger,
    node: NodeId,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
    h: History<TransferDomain>,
    run: Run<TransferDomain>,
)
    requires
        submitted_gather(before, staged, node, pieces, dst, bytes),
        preserves_destination(staged, l, node, dst.loc),
        gather_put_proof(pieces, dst, bytes).observed(l, node),
        before.submitted.map().submap_of(l.submitted.map()),
        revoked_submap(before.revoked, l.revoked),
        reachable(h),
        valid_run(h, run),
        ledger_consistent_with_history(l, node, h, run),
    ensures
        certificate_valid_in(gather_certificate(pieces, dst, bytes), h, run),
        recorded_writer_certified(gather_certificate(pieces, dst, bytes), l, node, h),
{
    let proof_obj = gather_put_proof(pieces, dst, bytes);
    let cert = proof_obj.certificate;
    let single = gather_is_single_event(pieces, pieces.last().writer);
    ledger_history_restrict(before, l, node, h, run);
    reachable_satisfies_invariant(h);
    if !single {
        stage_gather_piece_chain(before, staged, l, node, pieces, dst, bytes, h, run);
    }
    assert forall|p: int| 0 <= p < pieces.len() implies {
        let piece = #[trigger] pieces[p];
        let dst_sub = dst.sub_slice(piece.dst_offset, piece.source_slot.sub_len);
        &&& source_slice_valid(piece.source, piece.source_slot)
        &&& certificate_valid_in(piece.source, h, run)
        &&& source_slice_bytes(piece.source, piece.source_slot) == bytes.subrange(
            piece.dst_offset as int,
            (piece.dst_offset + piece.source_slot.sub_len) as int,
        )
        &&& event_succeeded(l, node, piece.writer)
        &&& l.submitted.dom().contains(piece.writer)
        &&& l.submitted.map()[piece.writer].access.reads.contains(piece.source_slot.loc)
        &&& l.submitted.map()[piece.writer].access.writes.contains(dst.loc)
        &&& last_writer_certified(h, piece.writer, piece.source)
        &&& if single {
            effect_implies(
                l.submitted.map()[piece.writer].effect,
                subrange_copy_effect(piece.source_slot, dst_sub),
            )
        } else {
            &&& effect_implies(
                l.submitted.map()[piece.writer].effect,
                copy_effect(piece.source_slot, dst_sub),
            )
            &&& (p > 0 && !dst_sub.is_full()
                ==> l.submitted.map()[piece.writer].access.reads.contains(dst.loc))
        }
    } by {
        let piece = pieces[p];
        assert(proof_obj.events[p] == piece.writer);
        assert(staged.submitted.dom().contains(piece.writer));
        assert(l.submitted.dom().contains(piece.writer));
        assert(staged.submitted.map()[piece.writer] == l.submitted.map()[piece.writer]);
        assert(l.submitted.map()[piece.writer].preds.subset_of(l.submitted.dom()));
        assert forall|x| #[trigger]
            l.submitted.map()[piece.writer].preds.contains(x) && !before.submitted.dom().contains(
                x,
            ) implies !l.submitted.map()[x].access.writes.contains(piece.source.slot.loc) by {
            assert(staged.submitted.dom().contains(x));
            assert(l.submitted.dom().contains(x));
            assert(staged.submitted.map()[x] == l.submitted.map()[x]);
        }
        assert(certificate_valid_in(piece.source, h, run));
        assert(recorded_writer_certified(piece.source, before, node, h));
        local_copy_last_writer(before, l, node, piece.source, piece.writer, h, run);
    }
    let established = lemma_gather_hop_preserves_value(l, node, h, run, pieces, dst, bytes);
    assert(established == cert);
    let last = pieces.len() - 1;
    let w = cert.writer;
    assert(proof_obj.events[last] == w);
    assert(l.citations(node).contains(w));
    assert((h.access)(w).writes.contains(dst.loc));
    assert forall|x|
        l.submitted.dom().contains(x) && (#[trigger] (h.access)(x)).writes.contains(dst.loc) && x
            != w implies (h.happens_before)(x, w) by {
        assert(l.submitted.map()[x].access == (h.access)(x));
        assert(staged.submitted.dom().contains(x));
        assert(staged.submitted.map()[x] == l.submitted.map()[x]);
        if before.submitted.dom().contains(x) {
            assert(!before.submitted.map()[x].preds.contains(w));
            assert(!(h.happens_before)(w, x));
            assert(conflict_at((h.access)(x), (h.access)(w), dst.loc));
            assert(!(h.read_guards)(x).dom().contains(dst.loc));
            assert(!(h.read_guards)(w).dom().contains(dst.loc));
        } else {
            assert(staged.submitted.map()[w].preds.contains(x));
        }
    }
}

/// Freeze submitted gather receipts into a `PendingPutProof` that survives later
/// disjoint writes (including overwriting the caller's source buffers).
pub proof fn stage_gather_proof(
    before: KnowledgeLedger,
    staged: KnowledgeLedger,
    node: NodeId,
    pieces: Seq<GatherPiece>,
    dst: SlotRef,
    bytes: Seq<u8>,
)
    requires
        submitted_gather(before, staged, node, pieces, dst, bytes),
    ensures
        gather_put_proof(pieces, dst, bytes).pending(staged, node),
{
    let proof_obj = gather_put_proof(pieces, dst, bytes);
    let cert = proof_obj.certificate;
    reveal(PendingPutProof::pending);
    assert forall|l: KnowledgeLedger|
        preserves_destination(staged, l, node, dst.loc) && #[trigger] proof_obj.observed(
            l,
            node,
        ) implies replica_valid(l, node, cert) by {
        vstd::imap_lib::lemma_submap_of_trans(
            before.submitted.map(),
            staged.submitted.map(),
            l.submitted.map(),
        );
        vstd::imap_lib::lemma_submap_of_trans(
            before.revoked.map(),
            staged.revoked.map(),
            l.revoked.map(),
        );
        let last = pieces.len() - 1;
        assert(proof_obj.events[last] == pieces[last].writer);
        assert(cert.writer == pieces[last].writer);
        assert(event_succeeded(l, node, cert.writer));
        assert forall|h, run|
            reachable(h) && valid_run(h, run) && ledger_consistent_with_history(
                l,
                node,
                h,
                run,
            ) implies #[trigger] certificate_valid_in(cert, h, run) && recorded_writer_certified(
            cert,
            l,
            node,
            h,
        ) by {
            stage_gather_valid_in_run(before, staged, l, node, pieces, dst, bytes, h, run);
        }
    }
}

} // verus!
