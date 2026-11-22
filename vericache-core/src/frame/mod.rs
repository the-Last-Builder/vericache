#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use crate::{provenance::*, spec::*, timeline::*};
#[cfg(verus_only)]
use foundation::tracked::{fold_share, untouched, Frontier, Slot};
use transfer::{context::*, domain::*};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::ElementToken;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

mod admission;
mod lease;
mod lifecycle;
mod meta;
mod share;
mod slab;
mod write;

pub use lease::*;
pub use meta::*;
#[cfg(verus_only)]
pub use slab::lemma_slab_slots_valid;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

pub struct FrameTable {
    pub num_tiers: usize,
    pub bindings: Vec<(ObjectKey, TierId)>,
    pub metas: Vec<FrameSyncMeta>,
    pub ghosts: Tracked<Map<nat, FrameGhost>>,
    pub ready_queue: Vec<Vec<FrameId>>,
    pub draining_queue: Vec<Vec<FrameId>>,
    pub frame_size: usize,
    pub versions: Ghost<Map<VersionId, Seq<u8>>>,
    pub draining_cursor: Vec<usize>,
}

/// Erased observation context used by frame-only state transitions.
pub ghost struct KernelView {
    pub timelines: TimelineTable,
    pub ledger: KnowledgeLedger,
    pub node: NodeId,
}

impl KernelView {
    pub open spec fn of(t: &TimelineTable, l: &KnowledgeLedger, node: NodeId) -> Self {
        Self { timelines: *t, ledger: *l, node }
    }

    pub open spec fn id(self) -> InstanceId {
        self.ledger.submitted.instance_id()
    }

    pub open spec fn extends(self, before: Self) -> bool {
        let t = self.timelines;
        let prior = before.timelines;
        &&& self.node == before.node && self.id() == before.id() && t.well_formed(
            self.id(),
            self.node,
            &self.ledger,
        )
        &&& before.ledger.citations(self.node).subset_of(self.ledger.citations(self.node))
        &&& t.count >= prior.count
        &&& forall|i: int|
            0 <= i < prior.count ==> #[trigger] t.confirmed_seq[i] >= prior.confirmed_seq[i]
                && t.next_seq[i] >= prior.next_seq[i] && t.overflow_watermark[i]
                >= prior.overflow_watermark[i] && t.kinds[i] == prior.kinds[i]
        &&& forall|u: u8, s: u64|
            prior.issued(u, s) ==> #[trigger] t.events@[(u, s)] == prior.events@[(u, s)]
    }
}

impl FrameTable {
    pub open spec fn frame(&self, f: FrameId) -> FrameGhost {
        self.ghosts@[f.0 as nat]
    }

    pub open spec fn meta(&self, f: FrameId) -> FrameSyncMeta {
        self.metas[f.0 as int]
    }

    pub open spec fn bindings_match(&self) -> bool {
        &&& self.bindings.len() == self.metas.len()
        &&& forall|i: nat|
            i < self.metas.len() ==> {
                let entry = #[trigger] self.bindings[i as int];
                &&& entry.1 == self.ghosts@[i].tier
                &&& self.ghosts@[i].bound_key matches Some(key) ==> key == entry.0
            }
    }

    pub open spec fn shape(&self) -> bool {
        &&& self.bindings_match()
        &&& self.num_tiers <= MAX_TIERS
        &&& 0 < self.frame_size
        &&& self.metas.len() <= u32::MAX
        &&& self.ready_queue.len() == self.num_tiers
        &&& self.draining_queue.len() == self.num_tiers
        &&& self.draining_cursor.len() == self.num_tiers
        &&& forall|i: nat| #[trigger] self.ghosts@.contains_key(i) == (i < self.metas.len())
    }

    pub open spec fn separated(&self) -> bool {
        forall|i: nat, j: nat|
            i < j < self.metas.len() ==> (#[trigger] self.ghosts@[i].slot).loc != (
            #[trigger] self.ghosts@[j].slot).loc
    }

    pub open spec fn queues(&self, table: &TimelineTable) -> bool {
        forall|t: int| 0 <= t < self.num_tiers ==> #[trigger] self.tier_queues(t, table)
    }

    pub open spec fn tier_queues(&self, t: int, table: &TimelineTable) -> bool {
        let ready = self.ready_queue[t]@;
        let draining = self.draining_queue[t]@;
        &&& ready.no_duplicates() && draining.no_duplicates()
        &&& forall|i: int|
            0 <= i < ready.len() ==> {
                let f = #[trigger] ready[i];
                f.0 < self.metas.len() && self.frame(f).tier.index() == t && self.meta(f).state
                    == STATE_EMPTY && self.meta(f).queued() && !self.meta(f).overflow()
                    && self.meta(f).reusable(table)
            }
        &&& forall|i: int|
            0 <= i < draining.len() ==> {
                let f = #[trigger] draining[i];
                f.0 < self.metas.len() && self.frame(f).tier.index() == t && self.meta(f).state
                    == STATE_DRAINING && !self.meta(f).queued()
            }
    }

    pub open spec fn values_agree(&self) -> bool {
        forall|i: nat|
            i < self.metas.len() ==> #[trigger] self.ghosts@[i].version_agrees(self.versions@)
    }

    pub open spec fn well_formed(
        &self,
        id: InstanceId,
        node: NodeId,
        table: &TimelineTable,
        ledger: &KnowledgeLedger,
    ) -> bool {
        &&& self.values_agree() && table.well_formed(id, node, ledger) && self.shape()
            && self.separated() && self.queues(table)
        &&& forall|i: nat|
            i < self.metas.len() ==> #[trigger] self.ghosts@[i].valid(
                self.metas[i as int],
                self.num_tiers,
                self.frame_size,
                id,
                node,
                table,
                ledger,
            )
    }

    pub open spec fn empty_queues(&self) -> bool {
        forall|t: int|
            0 <= t < self.num_tiers ==> (#[trigger] self.ready_queue[t])@.len() == 0
                && self.draining_queue[t]@.len() == 0
    }

    pub open spec fn consistent(&self, v: KernelView) -> bool {
        self.well_formed(v.id(), v.node, &v.timelines, &v.ledger)
    }

    pub open spec fn valid_frame(&self, f: FrameId, v: KernelView) -> bool {
        self.frame(f).version_agrees(self.versions@) && self.frame(f).valid(
            self.meta(f),
            self.num_tiers,
            self.frame_size,
            v.id(),
            v.node,
            &v.timelines,
            &v.ledger,
        )
    }

    pub open spec fn except_frame(&self, f: FrameId, v: KernelView) -> bool {
        &&& self.values_agree()
        &&& v.timelines.well_formed(v.id(), v.node, &v.ledger) && self.shape() && self.separated()
            && self.queues(&v.timelines)
        &&& f.0 < self.metas.len()
        &&& forall|i: nat|
            i < self.metas.len() && i != f.0 ==> #[trigger] self.ghosts@[i].valid(
                self.metas[i as int],
                self.num_tiers,
                self.frame_size,
                v.id(),
                v.node,
                &v.timelines,
                &v.ledger,
            )
    }

    pub open spec fn returned_reads(
        &self,
        f: FrameId,
        cap: ShareCap,
        k: Ticket,
        v: KernelView,
    ) -> bool {
        &&& cap.well_formed() && cap.instance_id() == v.id() && cap.loc() == self.frame(f).slot.loc
        &&& forall|e| #[trigger]
            cap.frontier().readers.contains(e) ==> v.ledger.citations(v.node).contains(e) || exists|
                s: u64,
            |
                1 <= s <= k.sequence() && #[trigger] v.timelines.events@[(k.timeline_id(), s)] == e
    }

    fn new_queues(num_tiers: usize) -> (queues: Vec<Vec<FrameId>>)
        requires
            num_tiers <= MAX_TIERS,
        ensures
            queues.len() == num_tiers,
            forall|i: int| 0 <= i < num_tiers ==> (#[trigger] queues[i])@.len() == 0,
    {
        let mut queues = Vec::<Vec<FrameId>>::with_capacity(num_tiers);
        while queues.len() < num_tiers
            invariant
                queues.len() <= num_tiers,
                forall|j: int| 0 <= j < queues.len() ==> (#[trigger] queues[j])@.len() == 0,
            decreases num_tiers - queues.len(),
        {
            queues.push(Vec::new());
        }
        queues
    }

    fn new_cursors(num_tiers: usize) -> (cursors: Vec<usize>)
        requires
            num_tiers <= MAX_TIERS,
        ensures
            cursors.len() == num_tiers,
    {
        let mut cursors = Vec::<usize>::with_capacity(num_tiers);
        while cursors.len() < num_tiers
            invariant
                cursors.len() <= num_tiers,
            decreases num_tiers - cursors.len(),
        {
            cursors.push(0);
        }
        cursors
    }

    pub fn new(num_tiers: usize, frame_size: usize) -> (r: Self)
        requires
            num_tiers <= MAX_TIERS,
            0 < frame_size,
        ensures
            r.shape(),
            r.num_tiers == num_tiers,
            r.versions@ == Map::empty(),
            r.separated(),
            r.frame_size == frame_size,
            r.metas.len() == 0,
            r.empty_queues(),
    {
        let ready = Self::new_queues(num_tiers);
        let draining = Self::new_queues(num_tiers);
        let cursors = Self::new_cursors(num_tiers);
        let r = Self {
            num_tiers,
            frame_size,
            bindings: Vec::new(),
            metas: Vec::new(),
            versions: Ghost(Map::empty()),
            ghosts: Tracked(Map::tracked_empty()),
            ready_queue: ready,
            draining_queue: draining,
            draining_cursor: cursors,
        };
        assert forall|t: int| 0 <= t < num_tiers implies (#[trigger] r.ready_queue[t])@.len() == 0
            && r.draining_queue[t]@.len() == 0 by {
            assert(ready[t]@.len() == 0);
        }
        assert(r.empty_queues());
        r
    }

    pub fn add_tier(&mut self, Ghost(v): Ghost<KernelView>) -> (tier: TierId)
        requires
            old(self).consistent(v),
            old(self).num_tiers < MAX_TIERS,
        ensures
            final(self).consistent(v),
            tier == TierId(old(self).num_tiers as u16),
            final(self).num_tiers == old(self).num_tiers + 1,
            final(self).frame_size == old(self).frame_size,
            final(self).versions == old(self).versions,
            final(self).bindings == old(self).bindings,
            final(self).metas == old(self).metas,
            final(self).ghosts == old(self).ghosts,
            final(self).ready_queue[old(self).num_tiers as int]@.len() == 0,
            final(self).draining_queue[old(self).num_tiers as int]@.len() == 0,
            forall|t: int|
                0 <= t < old(self).num_tiers ==> (#[trigger] final(self).ready_queue[t]) == old(
                    self,
                ).ready_queue[t] && final(self).draining_queue[t] == old(self).draining_queue[t],
    {
        let idx = self.num_tiers;
        self.num_tiers = idx + 1;
        self.ready_queue.push(Vec::new());
        self.draining_queue.push(Vec::new());
        self.draining_cursor.push(0);
        proof {
            assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts@[i].valid(
                self.metas[i as int],
                self.num_tiers,
                self.frame_size,
                v.id(),
                v.node,
                &v.timelines,
                &v.ledger,
            ) by {
                assert(old(self).ghosts@[i].valid(
                    old(self).metas[i as int],
                    old(self).num_tiers,
                    self.frame_size,
                    v.id(),
                    v.node,
                    &v.timelines,
                    &v.ledger,
                ));
            }
            assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
                u,
                &v.timelines,
            ) by {
                if u < idx {
                    assert(old(self).tier_queues(u, &v.timelines));
                }
            }
        }
        TierId(idx as u16)
    }

    pub(super) proof fn restore_frame(&self, before: Self, f: FrameId, v: KernelView)
        requires
            before.except_frame(f, v),
            self.bindings_match(),
            before.versions@.submap_of(self.versions@),
            self.num_tiers == before.num_tiers,
            self.metas.len() == before.metas.len(),
            self.frame_size == before.frame_size,
            self.ghosts@.dom() == before.ghosts@.dom(),
            self.ready_queue.len() == self.num_tiers,
            self.draining_queue.len() == self.num_tiers,
            self.draining_cursor.len() == self.num_tiers,
            self.frame(f).slot == before.frame(f).slot,
            self.frame(f).tier == before.frame(f).tier,
            self.valid_frame(f, v),
            self.queues(&v.timelines),
            forall|i: nat|
                i < self.metas.len() && i != f.0 ==> #[trigger] self.ghosts@[i] == before.ghosts@[i]
                    && self.metas[i as int] == before.metas[i as int],
        ensures
            self.consistent(v),
    {
        assert forall|i: nat|
            i < self.metas.len() implies #[trigger] self.ghosts@[i].version_agrees(
            self.versions@,
        ) by {
            if i != f.0 {
                let g = before.ghosts@[i];
                assert(g.version_agrees(before.versions@));
                if let Some(_cert) = g.cert {
                    let version = g.bound_version->Some_0;
                    assert(self.versions@.contains_key(version));
                }
            }
        }
        assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts@[i].valid(
            self.metas[i as int],
            self.num_tiers,
            self.frame_size,
            v.id(),
            v.node,
            &v.timelines,
            &v.ledger,
        ) by {
            if i != f.0 {
                assert(before.ghosts@[i].valid(
                    before.metas[i as int],
                    self.num_tiers,
                    self.frame_size,
                    v.id(),
                    v.node,
                    &v.timelines,
                    &v.ledger,
                ));
            }
        }
    }

    pub(super) proof fn queues_unchanged_off_frame(&self, before: Self, f: FrameId, v: KernelView)
        requires
            before.queues(&v.timelines),
            self.num_tiers == before.num_tiers,
            !before.meta(f).queued(),
            before.meta(f).state != STATE_DRAINING || !before.draining_queue[before.frame(
                f,
            ).tier.index() as int]@.contains(f),
            self.ready_queue == before.ready_queue,
            self.draining_queue == before.draining_queue,
            self.metas.len() == before.metas.len(),
            forall|j: FrameId|
                j != f && j.0 < self.metas.len() ==> #[trigger] self.meta(j) == before.meta(j)
                    && self.frame(j).tier == before.frame(j).tier,
        ensures
            self.queues(&v.timelines),
    {
        assert forall|t: int| 0 <= t < self.num_tiers implies #[trigger] self.tier_queues(
            t,
            &v.timelines,
        ) by {
            assert(before.tier_queues(t, &v.timelines));
            assert forall|i: int| 0 <= i < self.ready_queue[t].len() implies self.meta(
                #[trigger] self.ready_queue[t][i],
            ) == before.meta(self.ready_queue[t][i]) by {
                assert(before.ready_queue[t][i] != f);
            }
            assert forall|i: int| 0 <= i < self.draining_queue[t].len() implies self.meta(
                #[trigger] self.draining_queue[t][i],
            ) == before.meta(self.draining_queue[t][i]) by {
                assert(before.draining_queue[t][i] != f);
            }
        }
    }

    pub(super) proof fn refresh_lease(&self, before: Self, f: FrameId, v: KernelView)
        requires
            before.consistent(v),
            self.bindings_match(),
            before.versions == self.versions,
            self.num_tiers == before.num_tiers,
            f.0 < before.metas.len(),
            self.metas.len() == before.metas.len(),
            self.frame_size == before.frame_size,
            self.ghosts@.dom() == before.ghosts@.dom(),
            self.ready_queue == before.ready_queue,
            self.draining_queue == before.draining_queue,
            self.draining_cursor == before.draining_cursor,
            self.frame(f).slot == before.frame(f).slot,
            self.frame(f).tier == before.frame(f).tier,
            self.valid_frame(f, v),
            self.meta(f).state == before.meta(f).state,
            self.meta(f).state >= STATE_PUBLISHED,
            self.meta(f).flags == before.meta(f).flags,
            forall|i: nat|
                i < self.metas.len() && i != f.0 ==> #[trigger] self.ghosts@[i] == before.ghosts@[i]
                    && self.metas[i as int] == before.metas[i as int],
        ensures
            self.consistent(v),
    {
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &v.timelines,
        ) by {
            assert(before.tier_queues(u, &v.timelines));
        }
        self.restore_frame(before, f, v);
    }

    proof fn lemma_frame_timeline_progress(
        g: FrameGhost,
        m: FrameSyncMeta,
        num_tiers: usize,
        size: usize,
        before: KernelView,
        v: KernelView,
    )
        requires
            before.timelines.well_formed(before.id(), before.node, &before.ledger),
            g.valid(
                m,
                num_tiers,
                size,
                before.id(),
                before.node,
                &before.timelines,
                &before.ledger,
            ),
            v.extends(before),
        ensures
            g.valid(m, num_tiers, size, v.id(), v.node, &v.timelines, &v.ledger),
    {
        let a = before.timelines;
        let b = v.timelines;
        assert forall|j: int| 0 <= j < 5 implies b.valid_ticket(#[trigger] m.ticket(j)) by {
            let k = m.ticket(j);
            assert(a.valid_ticket(k));
        }
        assert(a.valid_ticket(m.ticket(4)));
        if let Some(c) = g.cert {
            let k = m.last_write_ticket;
            assert(a.issued(k.timeline_id(), k.sequence()));
            assert(b.events@[(k.timeline_id(), k.sequence())] == c.writer);
        }
        assert(g.binding(m, &b));
        assert(g.authority(m, v.id()));
        if g.token is Some {
            if let Some(w) = g.frontier().last_writer {
                if !v.ledger.citations(v.node).contains(w) {
                    assert(!before.ledger.citations(before.node).contains(w));
                    let k = m.last_write_ticket;
                    if a.ticket_covers(k, w) {
                        let s = choose|s: u64|
                            1 <= s <= k.sequence() && #[trigger] a.events@[(k.timeline_id(), s)]
                                == w;
                        assert(a.issued(k.timeline_id(), s));
                        assert(b.events@[(k.timeline_id(), s)] == w);
                        assert(b.ticket_covers(k, w));
                    } else {
                        assert(a.issued(k.timeline_id(), k.sequence()));
                        assert(b.events@[(k.timeline_id(), k.sequence())] == w);
                    }
                }
            }
            assert forall|r| #[trigger]
                g.frontier().readers.contains(r) && !v.ledger.citations(v.node).contains(
                    r,
                ) implies exists|t: u8, s: u64|
                b.issued(t, s) && #[trigger] b.events@[(t, s)] == r && m.bounds_reader(
                    &b,
                    t,
                    s,
                ) by {
                assert(!before.ledger.citations(before.node).contains(r));
                let (t, s) = choose|t: u8, s: u64|
                    a.issued(t, s) && #[trigger] a.events@[(t, s)] == r && m.bounds_reader(
                        &a,
                        t,
                        s,
                    );
                assert(b.confirmed_seq[t as int] >= a.confirmed_seq[t as int]);
                assert(b.overflow_watermark[t as int] >= a.overflow_watermark[t as int]);
                assert(b.events@[(t, s)] == r);
                assert(m.bounds_reader(&b, t, s));
            }
        }
        assert(g.covered(m, &b, v.node, &v.ledger));
    }

    pub proof fn lemma_timeline_progress(&self, before: KernelView, v: KernelView)
        requires
            self.consistent(before),
            v.extends(before),
        ensures
            self.consistent(v),
    {
        let a = before.timelines;
        let b = v.timelines;
        assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts@[i].valid(
            self.metas[i as int],
            self.num_tiers,
            self.frame_size,
            v.id(),
            v.node,
            &b,
            &v.ledger,
        ) by {
            Self::lemma_frame_timeline_progress(
                self.ghosts@[i],
                self.metas[i as int],
                self.num_tiers,
                self.frame_size,
                before,
                v,
            );
        }
        assert forall|u: int| 0 <= u < self.num_tiers implies #[trigger] self.tier_queues(
            u,
            &b,
        ) by {
            assert(self.tier_queues(u, &a));
            assert forall|i: int| 0 <= i < self.ready_queue[u].len() implies self.meta(
                #[trigger] self.ready_queue[u][i],
            ).reusable(&b) by {
                let f = self.ready_queue[u][i];
                let m = self.meta(f);
                assert(m.reusable(&a));
                assert(self.valid_frame(f, before));
                assert forall|j: int| 0 <= j < 5 implies b.retired(#[trigger] m.ticket(j)) by {
                    let k = m.ticket(j);
                    assert(a.valid_ticket(k));
                    assert(a.retired(k));
                }
                assert(b.retired(m.ticket(4)));
                assert forall|j: int| 0 <= j < 4 implies b.retired(
                    #[trigger] m.inline_read_tickets[j],
                ) by {
                    assert(b.retired(m.ticket(j)));
                }
            }
        }
    }

    pub(super) proof fn preserve_coverage(&self, before: Self, f: FrameId, v: KernelView)
        requires
            v.timelines.well_formed(v.id(), v.node, &v.ledger),
            self.frame(f).cert.is_some() ==> before.frame(f).cert.is_some(),
            before.frame(f).covered(before.meta(f), &v.timelines, v.node, &v.ledger),
            self.frame(f).slot == before.frame(f).slot,
            self.frame(f).token == before.frame(f).token,
            self.meta(f).last_write_ticket == before.meta(f).last_write_ticket,
            self.meta(f).inline_read_tickets == before.meta(f).inline_read_tickets,
            self.meta(f).overflow() == before.meta(f).overflow(),
        ensures
            self.frame(f).covered(self.meta(f), &v.timelines, v.node, &v.ledger),
    {
        assert forall|j: int| 0 <= j < 5 implies v.timelines.valid_ticket(
            #[trigger] self.meta(f).ticket(j),
        ) by {
            assert(before.meta(f).ticket(j) == self.meta(f).ticket(j));
        }
        self.frame(f).preserve_readers(
            before.frame(f),
            self.meta(f),
            before.meta(f),
            &v.timelines,
            &v.timelines,
            v.node,
            &v.ledger,
        );
    }

    pub open spec fn preserves_initializing(&self, before: &Self, except: Option<FrameId>) -> bool {
        forall|g: FrameId|
            Some(g) != except && g.0 < before.metas.len() && (#[trigger] before.meta(g)).state
                == STATE_INITIALIZING ==> self.meta(g) == before.meta(g) && self.frame(g)
                == before.frame(g)
    }

    pub open spec fn published(&self, f: FrameId) -> Option<
        (ObjectKey, VersionId, ValueCertificate, Ticket),
    > {
        let g = self.frame(f);
        if self.meta(f).state == STATE_PUBLISHED {
            Some(
                (
                    g.bound_key.unwrap(),
                    g.bound_version.unwrap(),
                    g.cert.unwrap(),
                    self.meta(f).last_write_ticket,
                ),
            )
        } else {
            None
        }
    }

    pub open spec fn same_layout(&self, before: &Self) -> bool {
        &&& self.num_tiers == before.num_tiers && self.frame_size == before.frame_size
            && self.metas.len() == before.metas.len()
        &&& forall|f: FrameId|
            f.0 < self.metas.len() ==> (#[trigger] self.frame(f)).slot == before.frame(f).slot
                && self.frame(f).tier == before.frame(f).tier
    }

    pub open spec fn same_directory(&self, before: &Self) -> bool {
        &&& self.num_tiers == before.num_tiers && self.versions == before.versions
            && self.frame_size == before.frame_size && self.metas.len() == before.metas.len()
        &&& forall|f: FrameId|
            f.0 < self.metas.len() ==> {
                &&& (#[trigger] self.frame(f)).slot == before.frame(f).slot
                &&& self.frame(f).tier == before.frame(f).tier
                &&& self.published(f) == before.published(f)
            }
    }

    pub open spec fn same_other_pins(&self, before: &Self, changed: Option<FrameId>) -> bool {
        forall|f: FrameId|
            f.0 < self.metas.len() && Some(f) != changed ==> {
                &&& (#[trigger] self.frame(f)).local_shares == before.frame(f).local_shares
                &&& self.frame(f).remote_shares == before.frame(f).remote_shares
                &&& self.meta(f).active_remote_leases == before.meta(f).active_remote_leases
            }
    }

    pub open spec fn preserves_remote_leases(&self, before: &Self) -> bool {
        forall|f: FrameId|
            f.0 < before.metas.len() ==> {
                &&& (#[trigger] self.frame(f)).remote_shares == before.frame(f).remote_shares
                &&& self.meta(f).active_remote_leases == before.meta(f).active_remote_leases
            }
    }
}

} // verus!
