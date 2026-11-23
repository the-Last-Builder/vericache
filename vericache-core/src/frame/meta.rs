use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

pub const STATE_EMPTY: u8 = 0;

pub const STATE_INITIALIZING: u8 = 1;

pub const STATE_PUBLISHED: u8 = 2;

pub const STATE_DRAINING: u8 = 3;

pub const FLAG_READ_OVERFLOW: u8 = 1;

pub const FLAG_IN_READY_QUEUE: u8 = 2;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct FrameSyncMeta {
    pub state: u8,
    pub flags: u8,
    pub active_remote_leases: u16,
    pub active_local_reads: u32,
    pub version_packed: u64,
    pub last_write_ticket: Ticket,
    pub inline_read_tickets: [Ticket; 4],
    pub mw_slot: u32,
    pub mw_gen: u32,
}

pub ghost struct LocalShareEntry {
    pub share: TransferShare,
}

pub ghost struct RemoteShareEntry {
    pub mw_slot: u32,
    pub mw_gen: u32,
}

pub tracked struct FrameGhost {
    pub ghost slot: SlotRef,
    pub ghost tier: TierId,
    pub tracked token: Option<SlotToken>,
    pub ghost local_shares: Map<TransferShare, LocalShareEntry>,
    pub ghost remote_shares: Map<TransferShare, RemoteShareEntry>,
    pub ghost cert: Option<ValueCertificate>,
    pub ghost bound_key: Option<ObjectKey>,
    pub ghost bound_version: Option<VersionId>,
}

impl FrameSyncMeta {
    pub open spec fn overflow(self) -> bool {
        self.flags & FLAG_READ_OVERFLOW != 0
    }

    pub open spec fn queued(self) -> bool {
        self.flags & FLAG_IN_READY_QUEUE != 0
    }

    pub open spec fn available(self) -> bool {
        (self.state == STATE_EMPTY || self.state == STATE_DRAINING) && self.active_local_reads == 0
            && self.active_remote_leases == 0
    }

    pub open spec fn ordered_eligible(self) -> bool {
        self.available() && !self.overflow()
    }

    pub open spec fn ticket(self, i: int) -> Ticket {
        if i == 4 {
            self.last_write_ticket
        } else {
            self.inline_read_tickets[i]
        }
    }

    pub open spec fn valid_tickets(self, table: &TimelineTable) -> bool {
        forall|i: int| 0 <= i < 5 ==> table.valid_ticket(#[trigger] self.ticket(i))
    }

    pub open spec fn ovf_bounded(ovf: [u64; MAX_TIMELINES], table: &TimelineTable) -> bool {
        &&& forall|t: int| 0 <= t < table.count ==> #[trigger] ovf[t] < table.next_seq[t]
        &&& forall|t: int| table.count <= t < MAX_TIMELINES ==> #[trigger] ovf[t] == 0
    }

    pub open spec fn overflow_retired(ovf: [u64; MAX_TIMELINES], table: &TimelineTable) -> bool {
        forall|i: int| 0 <= i < table.count ==> ovf[i] <= #[trigger] table.confirmed_seq[i]
    }

    pub open spec fn reusable(self, ovf: [u64; MAX_TIMELINES], table: &TimelineTable) -> bool {
        &&& self.available() && table.retired(self.last_write_ticket)
        &&& if self.overflow() {
            Self::overflow_retired(ovf, table)
        } else {
            forall|i: int| 0 <= i < 4 ==> table.retired(#[trigger] self.inline_read_tickets[i])
        }
    }

    pub open spec fn accounts(
        self,
        ovf: [u64; MAX_TIMELINES],
        table: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
        r: TransferEvent,
    ) -> bool {
        ledger.citations(node).contains(r) || exists|t: u8, s: u64|
            table.issued(t, s) && #[trigger] table.events@[(t, s)] == r && self.bounds_reader(
                ovf,
                t,
                s,
            )
    }

    pub open spec fn bounds_reader(self, ovf: [u64; MAX_TIMELINES], t: u8, s: u64) -> bool {
        Self::read_bounds(self.overflow(), ovf, self.inline_read_tickets, t, s)
    }

    pub open spec fn read_bounds(
        overflow: bool,
        ovf: [u64; MAX_TIMELINES],
        slots: [Ticket; 4],
        t: u8,
        s: u64,
    ) -> bool {
        if overflow {
            s <= ovf[t as int]
        } else {
            exists|i: int|
                0 <= i < 4 && {
                    let k = #[trigger] slots[i];
                    k != Ticket::NONE && k.timeline_id() == t && s <= k.sequence()
                }
        }
    }

    pub open spec fn ordered(self, table: &TimelineTable, after: ISet<TransferEvent>) -> bool {
        forall|i: int|
            0 <= i < 5 && (!self.overflow() || i == 4) ==> {
                let k = #[trigger] self.ticket(i);
                table.retired(k) || table.prefix_in(k.timeline_id(), k.sequence(), after)
            }
    }

    pub fn dependency(&self, i: usize) -> (k: Ticket)
        requires
            i < 5,
        ensures
            k == self.ticket(i as int),
    {
        if i == 4 {
            self.last_write_ticket
        } else {
            self.inline_read_tickets[i]
        }
    }

    pub(super) fn empty() -> (m: Self)
        ensures
            m.state == STATE_EMPTY,
            m.flags == FLAG_IN_READY_QUEUE,
            m.active_local_reads == 0,
            m.active_remote_leases == 0,
            m.version_packed == 0,
            m.last_write_ticket == Ticket::NONE,
            forall|i: int| 0 <= i < 4 ==> #[trigger] m.inline_read_tickets[i] == Ticket::NONE,
            m.mw_slot == u32::MAX,
            m.mw_gen == 0,
    {
        Self {
            state: STATE_EMPTY,
            flags: FLAG_IN_READY_QUEUE,
            active_remote_leases: 0,
            active_local_reads: 0,
            version_packed: 0,
            last_write_ticket: Ticket::NONE,
            inline_read_tickets: [Ticket::NONE;4],
            mw_slot: u32::MAX,
            mw_gen: 0,
        }
    }

    fn read_slot(&self, table: &TimelineTable, t: u8) -> (index: usize)
        requires
            table.bounds(),
            self.valid_tickets(table),
        ensures
            index <= 4,
            index < 4 ==> self.inline_read_tickets[index as int] == Ticket::NONE || table.retired(
                self.inline_read_tickets[index as int],
            ) || self.inline_read_tickets[index as int].timeline_id() == t,
    {
        let mut candidate = 4;
        for i in 0..4
            invariant
                candidate == 4 || (candidate < i && table.retired(
                    self.inline_read_tickets[candidate as int],
                )),
        {
            let k = self.dependency(i);
            if !k.is_none() && k.timeline() == t {
                return i;
            }
            if candidate == 4 && table.is_retired(k) {
                candidate = i;
            }
        }
        candidate
    }

    fn record_ovf(ovf: &mut [u64; MAX_TIMELINES], table: &TimelineTable, k: Ticket)
        requires
            table.bounds(),
            table.valid_ticket(k),
            Self::ovf_bounded(*old(ovf), table),
        ensures
            Self::ovf_bounded(*final(ovf), table),
            forall|i: int|
                0 <= i < MAX_TIMELINES ==> #[trigger] final(ovf)[i] == if k != Ticket::NONE && i
                    == k.timeline_id() {
                    vstd::math::max(old(ovf)[i] as int, k.sequence() as int)
                } else {
                    old(ovf)[i] as int
                },
    {
        if !k.is_none() {
            let i = k.timeline() as usize;
            let s = k.seq();
            if ovf[i] < s {
                ovf[i] = s;
            }
        }
    }

    pub(super) fn note_read(
        &mut self,
        ovf: &mut [u64; MAX_TIMELINES],
        table: &TimelineTable,
        k: Ticket,
    )
        requires
            table.bounds(),
            Self::ovf_bounded(*old(ovf), table),
            old(self).valid_tickets(table),
            table.valid_ticket(k),
            k != Ticket::NONE,
        ensures
            Self::ovf_bounded(*final(ovf), table),
            final(self).valid_tickets(table),
            final(self).bounds_reader(*final(ovf), k.timeline_id(), k.sequence()),
            !old(self).overflow() && !final(self).overflow() ==> *final(ovf) == *old(ovf),
            final(self).state == old(self).state,
            final(self).last_write_ticket == old(self).last_write_ticket,
            final(self).queued() == old(self).queued(),
            final(self).active_local_reads == old(self).active_local_reads,
            final(self).active_remote_leases == old(self).active_remote_leases,
            final(self).mw_slot == old(self).mw_slot,
            final(self).mw_gen == old(self).mw_gen,
            final(self).version_packed == old(self).version_packed,
            forall|t: u8, s: u64|
                table.issued(t, s) && old(self).bounds_reader(*old(ovf), t, s)
                    ==> #[trigger] final(self).bounds_reader(*final(ovf), t, s) || s
                    <= table.confirmed_seq[t as int],
    {
        if self.flags & FLAG_READ_OVERFLOW != 0 {
            Self::record_ovf(ovf, table, k);
            return;
        }
        let index = self.read_slot(table, k.timeline());
        if index < 4 {
            let prev = self.inline_read_tickets[index];
            if prev.is_none() || prev.timeline() != k.timeline() || prev.seq() < k.seq() {
                self.inline_read_tickets[index] = k;
            }
            assert(self.bounds_reader(*ovf, k.timeline_id(), k.sequence())) by {
                assert(self.inline_read_tickets[index as int].sequence() >= k.sequence());
            }
            assert forall|t: u8, s: u64|
                table.issued(t, s) && old(self).bounds_reader(
                    *old(ovf),
                    t,
                    s,
                ) implies #[trigger] self.bounds_reader(*ovf, t, s) || s
                <= table.confirmed_seq[t as int] by {
                let i = choose|i: int|
                    0 <= i < 4 && {
                        let prev = #[trigger] old(self).inline_read_tickets[i];
                        prev != Ticket::NONE && prev.timeline_id() == t && s <= prev.sequence()
                    };
                if s > table.confirmed_seq[t as int] {
                    assert(self.inline_read_tickets[i].timeline_id() == t && s
                        <= self.inline_read_tickets[i].sequence());
                }
            }
            assert forall|j: int| 0 <= j < 5 implies table.valid_ticket(
                #[trigger] self.ticket(j),
            ) by {
                assert(table.valid_ticket(old(self).ticket(j)));
            }
            return;
        }
        for i in 0..4
            invariant
                table.bounds(),
                Self::ovf_bounded(*ovf, table),
                self.valid_tickets(table),
                forall|t: int| 0 <= t < MAX_TIMELINES ==> #[trigger] ovf[t] >= old(ovf)[t],
                forall|j: int|
                    0 <= j < i ==> {
                        let p = #[trigger] self.inline_read_tickets[j];
                        p == Ticket::NONE || p.sequence() <= ovf[p.timeline_id() as int]
                    },
        {
            let p = self.dependency(i);
            let ghost before_ovf = *ovf;
            Self::record_ovf(ovf, table, p);
            assert forall|j: int| 0 <= j <= i implies {
                let q = #[trigger] self.inline_read_tickets[j];
                q == Ticket::NONE || q.sequence() <= ovf[q.timeline_id() as int]
            } by {
                let q = self.ticket(j);
                assert(table.valid_ticket(q));
                if j < i && q != Ticket::NONE {
                    assert(q.sequence() <= before_ovf[q.timeline_id() as int]);
                }
            }
        }
        Self::record_ovf(ovf, table, k);
        let flags = self.flags;
        self.flags = flags | FLAG_READ_OVERFLOW;
        self.inline_read_tickets = [Ticket::NONE;4];
        assert(table.valid_ticket(old(self).ticket(4)));
        assert(((flags | 1u8) & 1u8) != 0u8 && ((flags | 1u8) & 2u8) == (flags & 2u8))
            by (bit_vector);
        assert forall|t: u8, s: u64|
            table.issued(t, s) && old(self).bounds_reader(
                *old(ovf),
                t,
                s,
            ) implies #[trigger] self.bounds_reader(*ovf, t, s) by {
            let i = choose|i: int|
                0 <= i < 4 && {
                    let p = #[trigger] old(self).inline_read_tickets[i];
                    p != Ticket::NONE && p.timeline_id() == t && s <= p.sequence()
                };
            assert(old(self).inline_read_tickets[i].sequence() <= ovf[t as int]);
        }
    }
}

impl FrameGhost {
    pub open spec fn has_token(self) -> bool {
        self.token is Some
    }

    pub open spec fn slot_tok(self) -> SlotToken
        recommends
            self.token is Some,
    {
        self.token->Some_0
    }

    pub open spec fn frontier(self) -> Frontier<TransferDomain>
        recommends
            self.token is Some,
    {
        match self.slot_tok().value() {
            Slot::Owned(f) => f,
            Slot::Lent { base, .. } => base,
        }
    }

    pub open spec fn live_shares(self) -> ISet<TransferShare> {
        ISet::new(|s| self.local_shares.contains_key(s) || self.remote_shares.contains_key(s))
    }

    pub open spec fn has_pin(self, share: TransferShare, remote: bool) -> bool {
        if remote {
            self.remote_shares.contains_key(share)
        } else {
            self.local_shares.contains_key(share)
        }
    }

    pub open spec fn matches_lease(self, share: TransferShare, slot: u32, generation: u32) -> bool {
        self.remote_shares.contains_key(share) && self.remote_shares[share] == (RemoteShareEntry {
            mw_slot: slot,
            mw_gen: generation,
        })
    }

    pub open spec fn version_agrees(self, versions: Map<VersionId, Seq<u8>>) -> bool {
        self.cert matches Some(c) ==> self.bound_version matches Some(v) && versions.contains_key(v)
            && versions[v] == c.bytes
    }

    pub open spec fn authority(self, m: FrameSyncMeta, id: InstanceId) -> bool {
        &&& self.local_shares.len() == m.active_local_reads && self.remote_shares.len()
            == m.active_remote_leases
        &&& self.local_shares.dom().disjoint(self.remote_shares.dom())
        &&& match self.token {
            None => m.state == STATE_INITIALIZING && m.active_local_reads == 0
                && m.active_remote_leases == 0,
            Some(tok) => {
                &&& tok.well_formed() && tok.instance_id() == id && tok.key() == self.slot.loc
                &&& match tok.value() {
                    Slot::Owned(_) => m.active_local_reads == 0 && m.active_remote_leases == 0,
                    Slot::Lent { shares, .. } => (m.active_local_reads != 0
                        || m.active_remote_leases != 0) && shares == self.live_shares(),
                }
            },
        }
        &&& m.mw_slot == u32::MAX ==> m.mw_gen == 0
        &&& m.mw_slot != u32::MAX ==> m.active_remote_leases == 1 && (exists|s|
            self.remote_shares.contains_key(s) && #[trigger] self.remote_shares[s].mw_slot
                == m.mw_slot && self.remote_shares[s].mw_gen == m.mw_gen)
    }

    /// Single-slot completed-write check: the frame holds its owned token at
    /// `cert.writer` with no pending readers.
    pub open spec fn completed_writes(self, cert: ValueCertificate) -> bool {
        self.token matches Some(tok) && tok.value() is Owned && {
            let fr = slot_frontier(tok);
            fr.readers.is_empty() && fr.last_writer == Some(cert.writer)
        }
    }

    pub open spec fn binding(self, m: FrameSyncMeta, table: &TimelineTable) -> bool {
        &&& m.state <= STATE_DRAINING
        &&& (m.state == STATE_EMPTY) == self.bound_version.is_none()
        &&& self.bound_key.is_some() == self.bound_version.is_some()
        &&& match self.bound_version {
            Some(v) => v.well_formed() && v != VersionId::NONE && v.packed() == m.version_packed,
            None => m.version_packed == 0,
        }
        &&& (m.state >= STATE_PUBLISHED) == self.cert.is_some()
        &&& self.cert matches Some(c) ==> {
            &&& self.token is Some
            &&& self.frontier().last_writer == Some(c.writer)
            &&& c.slot == self.slot && c.bytes.len() == self.slot.slot_len
            &&& m.last_write_ticket != Ticket::NONE && c.writer == table.events@[(
                m.last_write_ticket.timeline_id(),
                m.last_write_ticket.sequence(),
            )]
        }
        &&& m.state < STATE_PUBLISHED ==> m.active_local_reads == 0 && m.active_remote_leases == 0
    }

    pub open spec fn covered(
        self,
        m: FrameSyncMeta,
        ovf: [u64; MAX_TIMELINES],
        table: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
    ) -> bool {
        &&& m.valid_tickets(table)
        &&& FrameSyncMeta::ovf_bounded(ovf, table)
        &&& self.token is Some ==> {
            let f = self.frontier();
            &&& f.last_writer matches Some(w) ==> ledger.citations(node).contains(w) || (
            m.last_write_ticket != Ticket::NONE && w == table.events@[(
                m.last_write_ticket.timeline_id(),
                m.last_write_ticket.sequence(),
            )]) || (self.cert.is_none() && table.ticket_covers(m.last_write_ticket, w))
            &&& forall|r| #[trigger]
                f.readers.contains(r) ==> m.accounts(ovf, table, node, ledger, r)
        }
    }

    pub open spec fn valid(
        self,
        m: FrameSyncMeta,
        ovf: [u64; MAX_TIMELINES],
        num_tiers: usize,
        size: usize,
        id: InstanceId,
        node: NodeId,
        table: &TimelineTable,
        ledger: &KnowledgeLedger,
    ) -> bool {
        &&& self.tier.valid(num_tiers) && self.slot.loc.node == node && self.slot.is_full()
            && self.slot.slot_len == size
        &&& self.binding(m, table) && self.authority(m, id) && self.covered(
            m,
            ovf,
            table,
            node,
            ledger,
        )
    }

    pub open spec fn writable(
        self,
        id: InstanceId,
        node: NodeId,
        ledger: &KnowledgeLedger,
        after: ISet<TransferEvent>,
    ) -> bool {
        let permitted = ledger.citations(node).union(after);
        &&& self.token matches Some(tok)
        &&& slot_is_owned(tok, id, self.slot.loc)
        &&& write_admitted(slot_frontier(tok), permitted)
    }

    pub open spec fn write_admitted_in(
        self,
        node: NodeId,
        ledger: &KnowledgeLedger,
        after: ISet<TransferEvent>,
    ) -> bool {
        &&& self.token matches Some(tok)
        &&& slot_is_owned(tok, ledger.submitted.instance_id(), self.slot.loc)
        &&& write_admitted(
            slot_frontier(tok),
            foundation::tracked::closure_from(
                ledger.submitted.map(),
                after.union(ledger.citations(node)),
            ),
        )
    }

    pub proof fn writable_admission(
        self,
        node: NodeId,
        ledger: &KnowledgeLedger,
        after: ISet<TransferEvent>,
        covered: ISet<TransferEvent>,
    )
        requires
            ledger.valid(ledger.submitted.instance_id(), node),
            self.writable(ledger.submitted.instance_id(), node, ledger, covered),
            after.subset_of(ledger.submitted.dom()),
            covered.subset_of(
                foundation::tracked::closure_from(
                    ledger.submitted.map(),
                    after.union(ledger.citations(node)),
                ),
            ),
        ensures
            self.write_admitted_in(node, ledger, after),
    {
        let permitted = ledger.citations(node).union(covered);
        let ordered = foundation::tracked::closure_from(
            ledger.submitted.map(),
            after.union(ledger.citations(node)),
        );
        assert(permitted.subset_of(ordered));
    }

    pub(super) proof fn prove_frontier_ordered(
        self,
        m: FrameSyncMeta,
        ovf: [u64; MAX_TIMELINES],
        table: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
        after: ISet<TransferEvent>,
    )
        requires
            table.well_formed(ledger.submitted.instance_id(), node, ledger),
            self.token is Some,
            self.authority(m, ledger.submitted.instance_id()),
            self.covered(m, ovf, table, node, ledger),
            m.active_local_reads == 0,
            m.active_remote_leases == 0,
            m.ordered(table, after),
            m.overflow() ==> FrameSyncMeta::overflow_retired(ovf, table),
        ensures
            self.writable(ledger.submitted.instance_id(), node, ledger, after),
    {
        let id = ledger.submitted.instance_id();
        let f = self.frontier();
        let permitted = ledger.citations(node).union(after);
        if let Some(w) = f.last_writer {
            let k = m.ticket(4);
            if !ledger.citations(node).contains(w) {
                let s = choose|s: u64|
                    1 <= s <= k.sequence() && #[trigger] table.events@[(k.timeline_id(), s)] == w;
                assert(table.issued(k.timeline_id(), s));
                assert(table.events@.contains_key((k.timeline_id(), s)));
                if table.retired(k) {
                    assert(table.observed(k.timeline_id(), s, node, ledger.knows));
                } else {
                    assert(table.prefix_in(k.timeline_id(), k.sequence(), after));
                }
            }
        }
        assert forall|e| #[trigger] f.readers.contains(e) implies permitted.contains(e) by {
            if !ledger.citations(node).contains(e) {
                let (t, s) = choose|t: u8, s: u64|
                    table.issued(t, s) && #[trigger] table.events@[(t, s)] == e && m.bounds_reader(
                        ovf,
                        t,
                        s,
                    );
                assert(table.events@.contains_key((t, s)));
                if m.overflow() {
                    assert(s <= table.confirmed_seq[t as int]);
                    assert(table.observed(t, s, node, ledger.knows));
                } else {
                    let i = choose|i: int|
                        0 <= i < 4 && {
                            let k = #[trigger] m.inline_read_tickets[i];
                            k != Ticket::NONE && k.timeline_id() == t && s <= k.sequence()
                        };
                    let k = m.ticket(i);
                    if table.retired(k) {
                        assert(table.observed(t, s, node, ledger.knows));
                    } else {
                        assert(table.prefix_in(t, k.sequence(), after));
                    }
                }
            }
        }
    }

    pub(super) proof fn preserve_readers(
        self,
        prior: Self,
        m: FrameSyncMeta,
        ovf: [u64; MAX_TIMELINES],
        old_meta: FrameSyncMeta,
        old_ovf: [u64; MAX_TIMELINES],
        table: &TimelineTable,
        before: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
    )
        requires
            self.cert.is_some() ==> prior.cert.is_some(),
            table.only_watermarks(before),
            table.well_formed(ledger.submitted.instance_id(), node, ledger),
            prior.covered(old_meta, old_ovf, before, node, ledger),
            m.valid_tickets(table),
            FrameSyncMeta::ovf_bounded(ovf, table),
            m.last_write_ticket == old_meta.last_write_ticket,
            self.slot == prior.slot,
            self.token is Some ==> prior.token is Some && self.frontier() == prior.frontier(),
            forall|t: u8, s: u64|
                before.issued(t, s) && old_meta.bounds_reader(old_ovf, t, s)
                    ==> #[trigger] m.bounds_reader(ovf, t, s) || s <= table.confirmed_seq[t as int],
        ensures
            self.covered(m, ovf, table, node, ledger),
    {
        if self.token is Some {
            assert(prior.frontier() == self.frontier());
            assert forall|r| #[trigger]
                self.frontier().readers.contains(r) && !ledger.citations(node).contains(
                    r,
                ) implies exists|t: u8, s: u64|
                table.issued(t, s) && #[trigger] table.events@[(t, s)] == r && m.bounds_reader(
                    ovf,
                    t,
                    s,
                ) by {
                let (t, s) = choose|t: u8, s: u64|
                    before.issued(t, s) && #[trigger] before.events@[(t, s)] == r
                        && old_meta.bounds_reader(old_ovf, t, s);
                if !m.bounds_reader(ovf, t, s) {
                    assert(table.observed(t, s, node, ledger.knows));
                    assert(table.events@.contains_key((t, s)));
                }
            }
        }
    }
}

} // verus!
const _: () = assert!(
    core::mem::size_of::<FrameSyncMeta>() == 64 && core::mem::align_of::<FrameSyncMeta>() == 64
);
