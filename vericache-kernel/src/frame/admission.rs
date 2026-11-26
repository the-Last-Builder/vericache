use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl FrameTable {
    pub(super) fn frame_overflow_retired(&self, table: &TimelineTable, f: FrameId) -> (r: bool)
        requires
            f.0 < self.metas.len(),
            self.overflow_seqs.len() == self.metas.len(),
            table.bounds(),
            self.ovf(f).len() <= table.count,
        ensures
            r == FrameSyncMeta::overflow_retired(self.ovf(f), table),
    {
        let ovf = &self.overflow_seqs[f.0 as usize];
        let mut i = 0;
        while i < ovf.len()
            invariant
                table.bounds(),
                i <= ovf.len() <= table.count,
                forall|j: int| 0 <= j < i ==> #[trigger] ovf@[j] <= table.confirmed_seq[j],
            decreases ovf.len() - i,
        {
            if table.confirmed_seq[i] < ovf[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    pub(crate) fn is_reusable_observed(&self, table: &TimelineTable, f: FrameId) -> (r: bool)
        requires
            f.0 < self.metas.len(),
            self.overflow_seqs.len() == self.metas.len(),
            table.bounds(),
            self.ovf(f).len() <= table.count,
            self.meta(f).valid_tickets(table),
        ensures
            r == self.meta(f).reusable(self.ovf(f), table),
    {
        let m = &self.metas[f.0 as usize];
        if !(m.state == STATE_EMPTY || m.state == STATE_DRAINING) || m.active_local_reads != 0
            || m.active_fenced_leases != 0 {
            return false;
        }
        assert(m.ticket(4) == m.last_write_ticket);
        if !table.is_retired(m.last_write_ticket) {
            return false;
        }
        if m.flags & FLAG_READ_OVERFLOW != 0 {
            return self.frame_overflow_retired(table, f);
        }
        for i in 0..4
            invariant
                forall|j: int| 0 <= j < i ==> table.retired(#[trigger] m.inline_read_tickets[j]),
        {
            assert(m.ticket(i as int) == m.inline_read_tickets[i as int]);
            if !table.is_retired(m.inline_read_tickets[i]) {
                return false;
            }
        }
        true
    }

    pub(crate) fn can_ordered_handoff(&self, f: FrameId) -> (r: bool)
        requires
            f.0 < self.metas.len(),
        ensures
            r == self.meta(f).ordered_eligible(),
    {
        let m = &self.metas[f.0 as usize];
        (m.state == STATE_EMPTY || m.state == STATE_DRAINING) && m.active_local_reads == 0
            && m.active_fenced_leases == 0 && (m.flags & FLAG_READ_OVERFLOW == 0)
    }

    pub(crate) fn frame_dependencies(&self, f: FrameId) -> (tickets: [Ticket; 5])
        requires
            f.0 < self.metas.len(),
        ensures
            forall|i: int| 0 <= i < 5 ==> #[trigger] tickets[i] == self.meta(f).ticket(i),
    {
        let m = &self.metas[f.0 as usize];
        [
            m.inline_read_tickets[0],
            m.inline_read_tickets[1],
            m.inline_read_tickets[2],
            m.inline_read_tickets[3],
            m.last_write_ticket,
        ]
    }

    /// Mode A: an observed-reusable frame admits any subsequent write with empty `after`.
    pub(crate) proof fn lemma_observed_reusable_admits_any_write(
        &self,
        f: FrameId,
        table: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
    )
        requires
            self.well_formed(ledger.submitted.instance_id(), node, table, ledger),
            f.0 < self.metas.len(),
            self.meta(f).reusable(self.ovf(f), table),
        ensures
            self.frame(f).writable(ledger.submitted.instance_id(), node, ledger, ISet::empty()),
            self.frame(f).write_admitted_in(node, ledger, ISet::empty()),
    {
        let m = self.meta(f);
        let ovf = self.ovf(f);
        assert(self.frame(f).valid(
            m,
            ovf,
            self.num_tiers,
            self.tier_frame_size(self.frame(f).tier),
            ledger.submitted.instance_id(),
            node,
            table,
            ledger,
        ));
        self.frame(f).prove_frontier_ordered(m, ovf, table, node, ledger, ISet::empty());
        self.frame(f).writable_admission(node, ledger, ISet::empty(), ISet::empty());
    }

    /// Mode B: a backend-neutral ordered handoff admits a writer whose predecessor
    /// closure `covered` (generated from `after` and local citations) covers every
    /// unretired dependency ticket of the frame.
    pub(crate) proof fn lemma_ordered_handoff_admits_writer(
        &self,
        f: FrameId,
        table: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
        after: ISet<TransferEvent>,
        covered: ISet<TransferEvent>,
    )
        requires
            self.well_formed(ledger.submitted.instance_id(), node, table, ledger),
            f.0 < self.metas.len(),
            self.meta(f).ordered_eligible(),
            self.meta(f).ordered(table, covered),
            after.subset_of(ledger.submitted.dom()),
            covered.subset_of(
                foundation::tracked::closure_from(
                    ledger.submitted.map(),
                    after.union(ledger.citations(node)),
                ),
            ),
        ensures
            self.frame(f).writable(ledger.submitted.instance_id(), node, ledger, covered),
            self.frame(f).write_admitted_in(node, ledger, after),
    {
        let m = self.meta(f);
        let ovf = self.ovf(f);
        assert(self.frame(f).valid(
            m,
            ovf,
            self.num_tiers,
            self.tier_frame_size(self.frame(f).tier),
            ledger.submitted.instance_id(),
            node,
            table,
            ledger,
        ));
        self.frame(f).prove_frontier_ordered(m, ovf, table, node, ledger, covered);
        self.frame(f).writable_admission(node, ledger, after, covered);
    }
}

} // verus!
