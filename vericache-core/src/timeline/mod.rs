use crate::provenance::KnowledgeLedger;
use crate::spec::*;
use transfer::context::{Knows, Submitted};
use transfer::domain::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

mod bitmap;
pub use bitmap::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::imap::group_imap_lemmas,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineKind {
    Host,
    CudaStream { device: DeviceId },
    IoUring,
    RdmaQueuePair { peer_node: NodeId },
    Custom(u16),
}

pub struct TimelineTable {
    pub confirmed_seq: [u64; MAX_TIMELINES],
    pub next_seq: [u64; MAX_TIMELINES],
    pub overflow_watermark: [u64; MAX_TIMELINES],
    pub kinds: [TimelineKind; MAX_TIMELINES],
    pub count: usize,
    pub events: Ghost<Map<(u8, u64), TransferEvent>>,
}

impl TimelineTable {
    pub open spec fn polled(&self, prior: &Self, timeline: u8) -> bool {
        &&& self.events == prior.events && self.next_seq == prior.next_seq
        &&& self.count == prior.count && self.kinds == prior.kinds && self.overflow_watermark
            == prior.overflow_watermark
        &&& forall|i: int|
            0 <= i < MAX_TIMELINES ==> (#[trigger] self.confirmed_seq[i]) >= prior.confirmed_seq[i]
                && (i != timeline ==> self.confirmed_seq[i] == prior.confirmed_seq[i])
    }

    pub open spec fn only_watermarks(&self, before: &Self) -> bool {
        &&& self.confirmed_seq == before.confirmed_seq && self.next_seq == before.next_seq
            && self.count == before.count && self.kinds == before.kinds && self.events
            == before.events
        &&& self.bounds()
        &&& forall|i: int|
            0 <= i < MAX_TIMELINES ==> #[trigger] self.overflow_watermark[i]
                >= before.overflow_watermark[i]
    }

    pub open spec fn issued(&self, t: u8, s: u64) -> bool {
        t < self.count && 1 <= s < self.next_seq[t as int]
    }

    /// A FIFO tail protects each earlier sequence on the same timeline.
    pub open spec fn ticket_covers(&self, k: Ticket, e: TransferEvent) -> bool {
        k != Ticket::NONE && exists|s: u64|
            1 <= s <= k.sequence() && #[trigger] self.events@[(k.timeline_id(), s)] == e
    }

    pub open spec fn valid_ticket(&self, ticket: Ticket) -> bool {
        ticket == Ticket::NONE || self.issued(ticket.timeline_id(), ticket.sequence())
    }

    pub open spec fn retired(&self, ticket: Ticket) -> bool {
        ticket == Ticket::NONE || ticket.sequence()
            <= self.confirmed_seq[ticket.timeline_id() as int]
    }

    pub open spec fn observed(&self, t: u8, s: u64, node: NodeId, k: Knows) -> bool {
        k.set().contains((node, self.events@[(t, s)]))
    }

    pub open spec fn bounds(&self) -> bool {
        &&& self.count <= MAX_TIMELINES
        &&& forall|i: int|
            0 <= i < self.count ==> 1 <= #[trigger] self.next_seq[i] <= MAX_TICKET_SEQ + 1
                && self.confirmed_seq[i] < self.next_seq[i] && self.overflow_watermark[i]
                < self.next_seq[i]
    }

    pub open spec fn well_formed(
        &self,
        id: InstanceId,
        node: NodeId,
        ledger: &KnowledgeLedger,
    ) -> bool {
        &&& self.bounds() && ledger.valid(id, node)
        &&& forall|t: u8, s: u64|
            self.issued(t, s) ==> #[trigger] self.events@.contains_key((t, s))
                && ledger.submitted.dom().contains(self.events@[(t, s)])
        &&& forall|t: u8, s: u64|
            t < self.count && 1 <= s <= self.confirmed_seq[t as int] ==> #[trigger] self.observed(
                t,
                s,
                node,
                ledger.knows,
            )
    }

    pub open spec fn overflow_retired(&self) -> bool {
        forall|i: int|
            0 <= i < self.count ==> self.overflow_watermark[i] <= #[trigger] self.confirmed_seq[i]
    }

    pub proof fn lemma_well_formed_ledger_growth(
        &self,
        id: InstanceId,
        node: NodeId,
        before: &KnowledgeLedger,
        after: &KnowledgeLedger,
    )
        requires
            self.well_formed(id, node, before),
            after.valid(id, node),
            before.submitted.map().submap_of(after.submitted.map()),
            before.knows.set().subset_of(after.knows.set()),
        ensures
            self.well_formed(id, node, after),
    {
        assert forall|t: u8, s: u64| self.issued(t, s) implies #[trigger] self.events@.contains_key(
            (t, s),
        ) by {
            assert(self.events@.contains_key((t, s)));
        }
        assert forall|t: u8, s: u64| self.issued(t, s) implies after.submitted.dom().contains(
            #[trigger] self.events@[(t, s)],
        ) by {
            assert(self.events@.contains_key((t, s)));
            let e = self.events@[(t, s)];
            assert(before.submitted.dom().contains(e));
            assert(after.submitted.dom().contains(e));
        }
        assert forall|t: u8, s: u64|
            t < self.count && 1 <= s
                <= self.confirmed_seq[t as int] implies #[trigger] self.observed(
            t,
            s,
            node,
            after.knows,
        ) by {
            assert(self.observed(t, s, node, before.knows));
        }
        assert forall|t: u8, s: u64|
            t < self.count && 1 <= s
                <= self.confirmed_seq[t as int] implies after.knows.set().contains(
            (node, #[trigger] self.events@[(t, s)]),
        ) by {
            assert(self.observed(t, s, node, before.knows));
        }
    }

    pub open spec fn prefix_in(&self, t: u8, through: u64, events: ISet<TransferEvent>) -> bool {
        forall|s: u64| 1 <= s <= through ==> events.contains(#[trigger] self.events@[(t, s)])
    }

    pub open spec fn window_has_room(&self, t: u8) -> bool {
        self.next_seq[t as int] <= MAX_TICKET_SEQ && self.next_seq[t as int]
            <= self.confirmed_seq[t as int] + BITMAP_CAPACITY
    }

    pub fn next_data_id(&self, timeline: u8) -> (r: Result<u64, CoreError>)
        requires
            self.bounds(),
            timeline < self.count,
        ensures
            r matches Ok(s) ==> s == self.next_seq[timeline as int] && s <= MAX_TICKET_SEQ && s
                <= self.confirmed_seq[timeline as int] + BITMAP_CAPACITY,
    {
        let t = timeline as usize;
        let next = self.next_seq[t];
        if next > MAX_TICKET_SEQ {
            return Err(CoreError::SequenceExhausted);
        }
        if next - self.confirmed_seq[t] > BITMAP_CAPACITY {
            return Err(CoreError::CapacityExhausted);
        }
        Ok(next)
    }

    pub fn next_data_ids(&self, timeline: u8, count: usize) -> (r: Result<u64, CoreError>)
        requires
            self.bounds(),
            timeline < self.count,
        ensures
            r.is_ok() == (count > 0 && self.next_seq[timeline as int] + count - 1 <= MAX_TICKET_SEQ
                && self.next_seq[timeline as int] + count - 1 <= self.confirmed_seq[timeline as int]
                + BITMAP_CAPACITY),
            r matches Ok(s) ==> s == self.next_seq[timeline as int] && count > 0 && s + count - 1
                <= MAX_TICKET_SEQ && s + count - 1 <= self.confirmed_seq[timeline as int]
                + BITMAP_CAPACITY,
    {
        if count == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let t = timeline as usize;
        let next = self.next_seq[t];
        let count_u64 = count as u64;
        if count_u64 > BITMAP_CAPACITY {
            return Err(CoreError::CapacityExhausted);
        }
        if next > MAX_TICKET_SEQ || count_u64 - 1 > MAX_TICKET_SEQ - next {
            return Err(CoreError::SequenceExhausted);
        }
        let last = next + (count_u64 - 1);
        if last - self.confirmed_seq[t] > BITMAP_CAPACITY {
            return Err(CoreError::CapacityExhausted);
        }
        Ok(next)
    }

    pub fn new() -> (r: Self)
        ensures
            r.count == 0,
            r.events@ == Map::empty(),
            r.bounds(),
    {
        Self {
            confirmed_seq: [0;MAX_TIMELINES],
            next_seq: [1;MAX_TIMELINES],
            overflow_watermark: [0;MAX_TIMELINES],
            kinds: [TimelineKind::Host;MAX_TIMELINES],
            count: 0,
            events: Ghost(Map::empty()),
        }
    }

    pub fn register_timeline(&mut self, kind: TimelineKind) -> (r: Result<u8, CoreError>)
        requires
            old(self).bounds(),
        ensures
            match r {
                Err(_) => *final(self) == *old(self) && old(self).count == MAX_TIMELINES,
                Ok(t) => t == old(self).count && final(self).count == t + 1
                    && final(self).kinds[t as int] == kind && final(self).next_seq[t as int] == 1
                    && final(self).confirmed_seq[t as int] == 0
                    && final(self).overflow_watermark[t as int] == 0 && final(self).events == old(
                    self,
                ).events && final(self).bounds() && forall|i: int|
                    0 <= i < t ==> #[trigger] final(self).kinds[i] == old(self).kinds[i]
                        && final(self).confirmed_seq[i] == old(self).confirmed_seq[i]
                        && final(self).next_seq[i] == old(self).next_seq[i]
                        && final(self).overflow_watermark[i] == old(self).overflow_watermark[i],
            },
    {
        let i = self.count;
        if i == MAX_TIMELINES {
            return Err(CoreError::CapacityExhausted);
        }
        self.kinds[i] = kind;
        self.confirmed_seq[i] = 0;
        self.next_seq[i] = 1;
        self.overflow_watermark[i] = 0;
        self.count += 1;
        Ok(i as u8)
    }

    pub fn issue_ticket(
        &mut self,
        t: u8,
        Ghost(event): Ghost<TransferEvent>,
        Tracked(receipt): Tracked<Submitted>,
        Tracked(ledger): Tracked<&mut KnowledgeLedger>,
    ) -> (r: Result<Ticket, CoreError>)
        requires
            old(self).well_formed(receipt.instance_id(), event.observer, old(ledger)),
            receipt.key() == event,
        ensures
            final(self).well_formed(receipt.instance_id(), event.observer, final(ledger)),
            match r {
                Err(_) => *final(self) == *old(self) && *final(ledger) == *old(ledger) && (t >= old(
                    self,
                ).count || old(self).next_seq[t as int] > MAX_TICKET_SEQ),
                Ok(ticket) => {
                    let s = old(self).next_seq[t as int];
                    &&& t < old(self).count && ticket.timeline_id() == t && ticket.sequence() == s
                        && ticket != Ticket::NONE
                    &&& final(self).next_seq@ == old(self).next_seq@.update(
                        t as int,
                        (s + 1) as u64,
                    )
                    &&& final(self).events@ == old(self).events@.insert((t, s), event)
                    &&& final(self).count == old(self).count && final(self).kinds == old(self).kinds
                        && final(self).confirmed_seq == old(self).confirmed_seq
                        && final(self).overflow_watermark == old(self).overflow_watermark
                    &&& old(ledger).submitted.map().submap_of(final(ledger).submitted.map())
                    &&& final(ledger).submitted.map() == old(ledger).submitted.map().insert(
                        event,
                        receipt.value(),
                    ) && final(ledger).knows == old(ledger).knows && final(ledger).succeeded == old(
                        ledger,
                    ).succeeded && final(ledger).revoked == old(ledger).revoked
                },
            },
    {
        if t as usize >= self.count {
            return Err(CoreError::InvalidArgument);
        }
        let s = self.next_seq[t as usize];
        if s > MAX_TICKET_SEQ {
            return Err(CoreError::SequenceExhausted);
        }
        proof {
            ledger.record_submitted(receipt);
            self.events = Ghost(self.events@.insert((t, s), event));
        }
        self.next_seq[t as usize] = s + 1;
        assert forall|u: u8, q: u64| self.issued(u, q) implies #[trigger] self.events@.contains_key(
            (u, q),
        ) && ledger.submitted.dom().contains(self.events@[(u, q)]) by {
            if (u, q) != (t, s) {
                assert(old(self).events@.contains_key((u, q)));
            }
        }
        assert forall|u: u8, q: u64|
            u < self.count && 1 <= q
                <= self.confirmed_seq[u as int] implies #[trigger] self.observed(
            u,
            q,
            event.observer,
            ledger.knows,
        ) by {
            assert(old(self).observed(u, q, event.observer, old(ledger).knows));
        }
        Ok(Ticket::pack(t, s))
    }

    pub fn advance_confirmed(
        &mut self,
        t: u8,
        new_confirmed: u64,
        Tracked(knows): Tracked<Knows>,
        Tracked(ledger): Tracked<&mut KnowledgeLedger>,
        Ghost(node): Ghost<NodeId>,
    )
        requires
            old(self).well_formed(knows.instance_id(), node, old(ledger)),
            t < old(self).count,
            old(self).confirmed_seq[t as int] <= new_confirmed < old(self).next_seq[t as int],
            forall|s: u64|
                old(self).confirmed_seq[t as int] < s <= new_confirmed ==> #[trigger] old(
                    self,
                ).observed(t, s, node, knows) || old(self).observed(t, s, node, old(ledger).knows),
        ensures
            final(self).well_formed(knows.instance_id(), node, final(ledger)),
            final(self).confirmed_seq@ == old(self).confirmed_seq@.update(t as int, new_confirmed),
            final(self).next_seq == old(self).next_seq,
            final(self).kinds == old(self).kinds,
            final(self).count == old(self).count,
            final(self).overflow_watermark == old(self).overflow_watermark,
            final(self).events == old(self).events,
            final(ledger).knows.set() == old(ledger).knows.set().union(knows.set()),
            final(ledger).submitted == old(ledger).submitted,
            final(ledger).succeeded == old(ledger).succeeded,
            final(ledger).revoked == old(ledger).revoked,
    {
        proof {
            ledger.record_knows(knows);
        }
        self.confirmed_seq[t as usize] = new_confirmed;
        assert forall|u: u8, s: u64|
            u < self.count && 1 <= s
                <= self.confirmed_seq[u as int] implies #[trigger] self.observed(
            u,
            s,
            node,
            ledger.knows,
        ) by {
            if s <= old(self).confirmed_seq[u as int] {
                assert(old(self).observed(u, s, node, old(ledger).knows));
            } else {
                assert(old(self).observed(u, s, node, knows) || old(self).observed(
                    u,
                    s,
                    node,
                    old(ledger).knows,
                ));
            }
        }
    }

    pub fn is_retired(&self, ticket: Ticket) -> (r: bool)
        requires
            self.bounds(),
            self.valid_ticket(ticket),
        ensures
            r == self.retired(ticket),
    {
        ticket.is_none() || ticket.seq() <= self.confirmed_seq[ticket.timeline() as usize]
    }

    pub fn record_overflow(&mut self, ticket: Ticket)
        requires
            old(self).bounds(),
            old(self).valid_ticket(ticket),
        ensures
            final(self).bounds(),
            final(self).events == old(self).events,
            final(self).kinds == old(self).kinds,
            final(self).next_seq == old(self).next_seq,
            final(self).confirmed_seq == old(self).confirmed_seq,
            final(self).count == old(self).count,
            forall|i: int|
                0 <= i < MAX_TIMELINES ==> #[trigger] final(self).overflow_watermark[i] == if ticket
                    != Ticket::NONE && i == ticket.timeline_id() {
                    vstd::math::max(
                        old(self).overflow_watermark[i] as int,
                        ticket.sequence() as int,
                    )
                } else {
                    old(self).overflow_watermark[i] as int
                },
    {
        if !ticket.is_none() {
            let i = ticket.timeline() as usize;
            let s = ticket.seq();
            if self.overflow_watermark[i] < s {
                self.overflow_watermark[i] = s;
            }
        }
    }

    pub fn overflow_all_retired(&self) -> (r: bool)
        requires
            self.bounds(),
        ensures
            r == self.overflow_retired(),
    {
        let mut i = 0;
        while i < self.count
            invariant
                i <= self.count <= MAX_TIMELINES,
                forall|j: int|
                    0 <= j < i ==> self.overflow_watermark[j] <= #[trigger] self.confirmed_seq[j],
            decreases self.count - i,
        {
            if self.confirmed_seq[i] < self.overflow_watermark[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    pub proof fn lemma_retired_ticket_in_frontier(
        &self,
        id: InstanceId,
        node: NodeId,
        ledger: &KnowledgeLedger,
        ticket: Ticket,
    )
        requires
            self.well_formed(id, node, ledger),
            self.valid_ticket(ticket),
            self.retired(ticket),
            ticket != Ticket::NONE,
        ensures
            self.observed(ticket.timeline_id(), ticket.sequence(), node, ledger.knows),
    {
    }

    pub proof fn lemma_overflow_all_retired(
        &self,
        id: InstanceId,
        node: NodeId,
        ledger: &KnowledgeLedger,
        t: u8,
        s: u64,
    )
        requires
            self.well_formed(id, node, ledger),
            self.overflow_retired(),
            self.issued(t, s),
            s <= self.overflow_watermark[t as int],
        ensures
            self.observed(t, s, node, ledger.knows),
    {
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataOperation {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingOperation {
    pub frame: FrameId,
    pub operation: DataOperation,
    pub sequence: u64,
}

pub open spec fn remove_range(m: Map<u64, TransferEvent>, start: u64, count: int) -> Map<
    u64,
    TransferEvent,
> {
    Map::new(m.dom().filter(|s: u64| !(start <= s < start + count)), |s: u64| m[s])
}

pub open spec fn window_matches(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    pending: Map<u64, TransferEvent>,
) -> bool {
    let base = table.confirmed_seq[t as int];
    &&& table.next_seq[t as int] <= base + BITMAP_CAPACITY + 1
    &&& forall|s: u64| #[trigger]
        pending.contains_key(s) ==> base < s < table.next_seq[t as int] && pending[s]
            == table.events@[(t, s)] && (ops[(s % BITMAP_CAPACITY) as int] matches Some(op)
            && op.sequence == s)
    &&& forall|i: int|
        0 <= i < BITMAP_CAPACITY ==> (#[trigger] ops[i] matches Some(op) ==> op.sequence
            % BITMAP_CAPACITY == i && pending.contains_key(op.sequence))
    &&& forall|s: u64|
        base < s < table.next_seq[t as int] ==> #[trigger] pending.contains_key(s)
            == !bitmap.bit_set(s - base - 1)
}

pub proof fn window_after_completion(
    prior_ops: [Option<PendingOperation>; 512],
    ops: [Option<PendingOperation>; 512],
    prior_bitmap: SlidingBitmap,
    bitmap: SlidingBitmap,
    before: &TimelineTable,
    table: &TimelineTable,
    t: u8,
    s: u64,
    pending: Map<u64, TransferEvent>,
    advanced: u64,
)
    requires
        window_matches(prior_ops, prior_bitmap, before, t, pending),
        pending.contains_key(s),
        table.next_seq == before.next_seq,
        table.events == before.events,
        ops@ == prior_ops@.update((s % BITMAP_CAPACITY) as int, None),
        table.confirmed_seq[t as int] == before.confirmed_seq[t as int] + advanced,
        forall|i: int|
            0 <= i < advanced ==> #[trigger] prior_bitmap.bit_set(i)
                || before.confirmed_seq[t as int] + 1 + i == s,
        forall|i: int|
            0 <= i ==> #[trigger] bitmap.bit_set(i) == (prior_bitmap.bit_set(i + advanced)
                || table.confirmed_seq[t as int] + 1 + i == s),
    ensures
        window_matches(ops, bitmap, table, t, pending.remove(s)),
{
    assert forall|q: u64| #[trigger] pending.remove(s).contains_key(q) implies {
        &&& table.confirmed_seq[t as int] < q < table.next_seq[t as int]
        &&& pending.remove(s)[q] == table.events@[(t, q)]
        &&& ops[(q % BITMAP_CAPACITY) as int] matches Some(p) && p.sequence == q
    } by {
        assert(pending.contains_key(q));
        assert(!prior_bitmap.bit_set(q - before.confirmed_seq[t as int] - 1));
        assert(q % BITMAP_CAPACITY != s % BITMAP_CAPACITY) by {
            assert(prior_ops[(q % BITMAP_CAPACITY) as int]->Some_0.sequence == q);
        }
    }
    let base = table.confirmed_seq[t as int];
    let prior = before.confirmed_seq[t as int];
    assert forall|q: u64| base < q < table.next_seq[t as int] implies #[trigger] pending.remove(
        s,
    ).contains_key(q) == !bitmap.bit_set(q - base - 1) by {
        assert(bitmap.bit_set(q - base - 1) == (prior_bitmap.bit_set(q - prior - 1) || q == s));
        assert(pending.contains_key(q) == !prior_bitmap.bit_set(q - prior - 1));
    }
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (#[trigger] ops[i] matches Some(p)
        ==> p.sequence % BITMAP_CAPACITY == i && pending.remove(s).contains_key(p.sequence)) by {
        if let Some(p) = ops[i] {
            assert(prior_ops[i] == Some(p));
            assert(p.sequence != s);
        }
    }
}

pub fn record_submission(
    ops: &mut [Option<PendingOperation>; 512],
    Ghost(bitmap): Ghost<SlidingBitmap>,
    table: &mut TimelineTable,
    t: u8,
    frame: FrameId,
    operation: DataOperation,
    Ghost(pending): Ghost<Map<u64, TransferEvent>>,
    Tracked(receipt): Tracked<Submitted>,
    Tracked(ledger): Tracked<&mut KnowledgeLedger>,
) -> (ticket: Ticket)
    requires
        old(table).well_formed(receipt.instance_id(), receipt.key().observer, old(ledger)),
        t < old(table).count,
        bitmap.well_formed(),
        bitmap.observations(old(table), t, receipt.key().observer, old(ledger)),
        ({
            let s = old(table).next_seq[t as int];
            &&& s <= MAX_TICKET_SEQ && s <= old(table).confirmed_seq[t as int] + BITMAP_CAPACITY
            &&& pending.contains_key(s) && pending[s] == receipt.key()
            &&& window_matches(*old(ops), bitmap, old(table), t, pending.remove(s))
        }),
    ensures
        final(table).well_formed(receipt.instance_id(), receipt.key().observer, final(ledger)),
        bitmap.observations(final(table), t, receipt.key().observer, final(ledger)),
        window_matches(*final(ops), bitmap, final(table), t, pending),
        final(ledger).submitted.map() == old(ledger).submitted.map().insert(
            receipt.key(),
            receipt.value(),
        ),
        final(ledger).knows == old(ledger).knows,
        final(ledger).succeeded == old(ledger).succeeded,
        final(ledger).revoked == old(ledger).revoked,
        old(ledger).submitted.map().submap_of(final(ledger).submitted.map()),
        final(ops)@ == old(ops)@.update(
            (ticket.sequence() % BITMAP_CAPACITY) as int,
            Some(PendingOperation { sequence: ticket.sequence(), frame, operation }),
        ),
        ticket.sequence() == old(table).next_seq[t as int] && ticket.timeline_id() == t && ticket
            != Ticket::NONE,
        final(table).events@ == old(table).events@.insert(
            (ticket.timeline_id(), ticket.sequence()),
            receipt.key(),
        ),
        final(table).next_seq@ == old(table).next_seq@.update(
            ticket.timeline_id() as int,
            (ticket.sequence() + 1) as u64,
        ),
        final(table).kinds == old(table).kinds,
        final(table).count == old(table).count,
        final(table).confirmed_seq == old(table).confirmed_seq,
        final(table).overflow_watermark == old(table).overflow_watermark,
{
    let s = table.next_seq[t as usize];
    let ticket = table.issue_ticket(
        t,
        Ghost(receipt.key()),
        Tracked(receipt),
        Tracked(ledger),
    ).unwrap();
    let idx = (s % BITMAP_CAPACITY) as usize;
    ops[idx] = Some(PendingOperation { sequence: s, frame, operation });
    let ghost base = table.confirmed_seq[t as int];
    assert forall|q: u64| #[trigger] pending.contains_key(q) implies base < q
        < table.next_seq[t as int] && pending[q] == table.events@[(t, q)] && (ops[(q
        % BITMAP_CAPACITY) as int] matches Some(p) && p.sequence == q) by {
        if q != s {
            assert(pending.remove(s).contains_key(q));
            assert(q % 512 != s % 512) by {
                assert(q == 512 * (q / 512) + q % 512);
                assert(s == 512 * (s / 512) + s % 512);
            }
        }
    }
    assert forall|i: int| #[trigger] bitmap.bit_set(i) implies {
        let q = (base + 1 + i) as u64;
        table.issued(t, q) && table.observed(t, q, receipt.key().observer, ledger.knows)
    } by {
        assert(bitmap.bit_set(i));
    }
    assert(!bitmap.bit_set(s - base - 1)) by {
        if bitmap.bit_set(s - base - 1) {
            assert(old(table).issued(t, s));
        }
    }
    assert forall|q: u64|
        base < q < table.next_seq[t as int] implies #[trigger] pending.contains_key(q)
        == !bitmap.bit_set(q - base - 1) by {
        if q != s {
            assert(pending.remove(s).contains_key(q) == !bitmap.bit_set(q - base - 1));
        }
    }
    assert forall|i: int| 0 <= i < BITMAP_CAPACITY implies (#[trigger] ops[i] matches Some(p)
        ==> p.sequence % BITMAP_CAPACITY == i && pending.contains_key(p.sequence)) by {
        if i != idx {
            if let Some(p) = ops[i] {
                assert(old(ops)[i] == Some(p));
                assert(pending.remove(s).contains_key(p.sequence));
            }
        }
    }
    ticket
}

/// Dispatch a reaped data completion and retain its success witness only on success.
pub fn complete_data(
    ops: &mut [Option<PendingOperation>; 512],
    bitmap: &mut SlidingBitmap,
    table: &mut TimelineTable,
    t: u8,
    s: u64,
    Ghost(pending): Ghost<Map<u64, TransferEvent>>,
    Ghost(node): Ghost<NodeId>,
    Tracked(observation): Tracked<Knows>,
    Tracked(success): Tracked<Option<transfer::context::Succeeded>>,
    Tracked(ledger): Tracked<&mut KnowledgeLedger>,
) -> (op: PendingOperation)
    requires
        old(table).well_formed(observation.instance_id(), node, old(ledger)),
        t < old(table).count,
        old(bitmap).well_formed(),
        old(bitmap).observations(old(table), t, node, old(ledger)),
        window_matches(*old(ops), *old(bitmap), old(table), t, pending),
        pending.contains_key(s),
        old(table).observed(t, s, node, observation),
        success matches Some(w) ==> w.event() == pending[s] && w.instance_id()
            == observation.instance_id(),
    ensures
        final(table).well_formed(observation.instance_id(), node, final(ledger)),
        final(bitmap).well_formed(),
        final(bitmap).observations(final(table), t, node, final(ledger)),
        window_matches(*final(ops), *final(bitmap), final(table), t, pending.remove(s)),
        Some(op) == old(ops)[(s % BITMAP_CAPACITY) as int],
        final(ops)@ == old(ops)@.update((s % BITMAP_CAPACITY) as int, None),
        op.sequence == s,
        final(table).polled(old(table), t),
        final(table).kinds == old(table).kinds,
        final(table).count == old(table).count,
        final(table).next_seq == old(table).next_seq,
        final(table).events == old(table).events,
        final(table).overflow_watermark == old(table).overflow_watermark,
        final(ledger).submitted == old(ledger).submitted,
        final(ledger).revoked == old(ledger).revoked,
        final(ledger).knows.set() == old(ledger).knows.set().union(observation.set()),
        final(ledger).succeeded.witnesses == match success {
            None => old(ledger).succeeded.witnesses,
            Some(w) => old(ledger).succeeded.witnesses.insert(pending[s], w),
        },
{
    let idx = (s % BITMAP_CAPACITY) as usize;
    let op = ops[idx].take().unwrap();
    let _advanced = apply_bitmap_completion(
        bitmap,
        table,
        t,
        s,
        Tracked(observation),
        Tracked(ledger),
        Ghost(node),
    );
    let ghost after_bitmap = *ledger;
    proof {
        if let Some(w) = success {
            assert(table.events@.contains_key((t, s)));
            ledger.record_succeeded(w);
        }
        assert forall|u: u8, q: u64|
            u < table.count && 1 <= q
                <= table.confirmed_seq[u as int] implies #[trigger] table.observed(
            u,
            q,
            node,
            ledger.knows,
        ) by {
            assert(table.observed(u, q, node, after_bitmap.knows));
        }
        assert forall|e| #[trigger]
            ledger.succeeded.contains(e) implies ledger.submitted.dom().contains(e) by {
            if e != pending[s] || success.is_none() {
                assert(after_bitmap.succeeded.contains(e));
            }
        }
        assert(ledger.valid(observation.instance_id(), node));
        window_after_completion(
            *old(ops),
            *ops,
            *old(bitmap),
            *bitmap,
            old(table),
            table,
            t,
            s,
            pending,
            _advanced,
        );
    }
    op
}

} // verus!
