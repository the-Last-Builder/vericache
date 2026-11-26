#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use crate::{provenance::*, spec::*, timeline::*};
#[cfg(verus_only)]
use foundation::tracked::{fold_share, untouched, Frontier, Slot};
use transfer_core::{context::*, domain::*};
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
pub use slab::{lemma_slab_slots_valid, slab_slots_ready};

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

pub struct FrameTable {
    pub(crate) num_tiers: usize,
    pub(crate) bindings: Vec<(ObjectKey, TierId)>,
    pub(crate) metas: Vec<FrameSyncMeta>,
    /// Empty for inline readers. Overflowing frames store a prefix indexed by timeline ID.
    pub(crate) overflow_seqs: Vec<Vec<u64>>,
    pub(crate) ghosts: Tracked<Map<nat, FrameGhost>>,
    pub(crate) ready_queue: Vec<Vec<FrameId>>,
    pub(crate) draining_queue: Vec<Vec<FrameId>>,
    pub(crate) frame_size: usize,
    pub(crate) tier_frame_sizes: Vec<usize>,
    pub(crate) versions: Ghost<Map<VersionId, Seq<u8>>>,
    pub(crate) draining_cursor: Vec<usize>,
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
        &&& t.count_spec() >= prior.count_spec()
        &&& forall|i: int|
            0 <= i < prior.count_spec() ==> #[trigger] t.confirmed_seq_spec()[i]
                >= prior.confirmed_seq_spec()[i] && t.next_seq_spec()[i] >= prior.next_seq_spec()[i]
                && t.kinds_spec()[i] == prior.kinds_spec()[i]
        &&& forall|u: TimelineId, s: u64|
            prior.issued(u, s) ==> #[trigger] t.events_spec()@[(u, s)] == prior.events_spec()@[(
                u,
                s,
            )]
    }
}

impl FrameTable {
    pub open(crate) spec fn draining_cursor_spec(&self) -> Vec<usize> {
        self.draining_cursor
    }

    pub open(crate) spec fn draining_queue_spec(&self) -> Vec<Vec<FrameId>> {
        self.draining_queue
    }

    pub open(crate) spec fn tier_frame_sizes_spec(&self) -> Vec<usize> {
        self.tier_frame_sizes
    }

    pub open(crate) spec fn overflow_seqs_spec(&self) -> Vec<Vec<u64>> {
        self.overflow_seqs
    }

    pub open(crate) spec fn ghosts_spec(&self) -> Tracked<Map<nat, FrameGhost>> {
        self.ghosts
    }

    pub open(crate) spec fn ready_queue_spec(&self) -> Vec<Vec<FrameId>> {
        self.ready_queue
    }

    pub open(crate) spec fn bindings_spec(&self) -> Vec<(ObjectKey, TierId)> {
        self.bindings
    }

    pub open(crate) spec fn metas_spec(&self) -> Vec<FrameSyncMeta> {
        self.metas
    }

    pub open(crate) spec fn frame_size_spec(&self) -> usize {
        self.frame_size
    }

    pub open(crate) spec fn num_tiers_spec(&self) -> usize {
        self.num_tiers
    }

    pub open(crate) spec fn versions_spec(&self) -> Ghost<Map<VersionId, Seq<u8>>> {
        self.versions
    }

    pub open spec fn tier_draining_count_spec(&self, tier: TierId) -> Option<usize> {
        if tier.index() < self.draining_queue_spec().len() {
            Some(self.draining_queue_spec()[tier.index() as int].len())
        } else {
            None
        }
    }

    pub fn tier_draining_count(&self, tier: TierId) -> (r: Option<usize>)
        ensures
            r == self.tier_draining_count_spec(tier),
    {
        let idx = tier.as_index();
        if idx < self.draining_queue.len() {
            Some(self.draining_queue[idx].len())
        } else {
            None
        }
    }

    pub open spec fn tier_ready_count_spec(&self, tier: TierId) -> Option<usize> {
        if tier.index() < self.ready_queue_spec().len() {
            Some(self.ready_queue_spec()[tier.index() as int].len())
        } else {
            None
        }
    }

    pub fn tier_ready_count(&self, tier: TierId) -> (r: Option<usize>)
        ensures
            r == self.tier_ready_count_spec(tier),
    {
        let idx = tier.as_index();
        if idx < self.ready_queue.len() {
            Some(self.ready_queue[idx].len())
        } else {
            None
        }
    }

    pub open spec fn frame(&self, f: FrameId) -> FrameGhost {
        self.ghosts_spec()@[f.0 as nat]
    }

    pub open spec fn meta(&self, f: FrameId) -> FrameSyncMeta {
        self.metas_spec()[f.0 as int]
    }

    pub open spec fn ovf(&self, f: FrameId) -> Seq<u64> {
        self.overflow_seqs_spec()[f.0 as int]@
    }

    pub open spec fn tier_frame_size(&self, t: TierId) -> usize {
        self.tier_frame_sizes_spec()[t.index() as int]
    }

    #[verifier::when_used_as_spec(tier_frame_size)]
    pub(crate) fn tier_size(&self, t: TierId) -> (s: usize)
        requires
            self.shape(),
            t.valid(self.num_tiers),
        ensures
            s == self.tier_frame_size(t),
            0 < s <= u64::MAX,
    {
        self.tier_frame_sizes[t.as_index()]
    }

    pub open spec fn bindings_match(&self) -> bool {
        &&& self.bindings_spec().len() == self.metas_spec().len()
        &&& forall|i: nat|
            i < self.metas_spec().len() ==> {
                let entry = #[trigger] self.bindings_spec()[i as int];
                &&& entry.1 == self.ghosts_spec()@[i].tier
                &&& self.ghosts_spec()@[i].bound_key matches Some(key) ==> key == entry.0
            }
    }

    pub open spec fn shape(&self) -> bool {
        &&& self.bindings_match()
        &&& self.overflow_seqs_spec().len() == self.metas_spec().len()
        &&& self.num_tiers_spec() <= MAX_TIERS
        &&& 0 < self.frame_size_spec() <= u64::MAX
        &&& self.tier_frame_sizes_spec().len() == self.num_tiers_spec()
        &&& forall|t: int|
            0 <= t < self.num_tiers_spec() ==> 0 < #[trigger] self.tier_frame_sizes_spec()[t]
                <= u64::MAX
        &&& self.metas_spec().len() <= u32::MAX
        &&& self.ready_queue_spec().len() == self.num_tiers_spec()
        &&& self.draining_queue_spec().len() == self.num_tiers_spec()
        &&& self.draining_cursor_spec().len() == self.num_tiers_spec()
        &&& forall|i: nat| #[trigger]
            self.ghosts_spec()@.contains_key(i) == (i < self.metas_spec().len())
    }

    pub open spec fn separated(&self) -> bool {
        forall|i: nat, j: nat|
            i < j < self.metas_spec().len() ==> (#[trigger] self.ghosts_spec()@[i].slot).loc != (
            #[trigger] self.ghosts_spec()@[j].slot).loc
    }

    pub open spec fn queues(&self, table: &TimelineTable) -> bool {
        forall|t: int| 0 <= t < self.num_tiers_spec() ==> #[trigger] self.tier_queues(t, table)
    }

    pub open spec fn tier_queues(&self, t: int, table: &TimelineTable) -> bool {
        let ready = self.ready_queue_spec()[t]@;
        let draining = self.draining_queue_spec()[t]@;
        &&& ready.no_duplicates() && draining.no_duplicates()
        &&& forall|i: int|
            0 <= i < ready.len() ==> {
                let f = #[trigger] ready[i];
                f.0 < self.metas_spec().len() && self.frame(f).tier.index() == t && self.meta(
                    f,
                ).state == STATE_EMPTY && self.meta(f).queued() && !self.meta(f).overflow()
                    && self.meta(f).reusable(self.ovf(f), table)
            }
        &&& forall|i: int|
            0 <= i < draining.len() ==> {
                let f = #[trigger] draining[i];
                f.0 < self.metas_spec().len() && self.frame(f).tier.index() == t && self.meta(
                    f,
                ).state == STATE_DRAINING && !self.meta(f).queued()
            }
    }

    pub open spec fn values_agree(&self) -> bool {
        forall|i: nat|
            i < self.metas_spec().len() ==> #[trigger] self.ghosts_spec()@[i].version_agrees(
                self.versions_spec()@,
            )
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
            i < self.metas_spec().len() ==> #[trigger] self.ghosts_spec()@[i].valid(
                self.metas_spec()[i as int],
                self.overflow_seqs_spec()[i as int]@,
                self.num_tiers_spec(),
                self.tier_frame_size(self.ghosts_spec()@[i].tier),
                id,
                node,
                table,
                ledger,
            )
    }

    pub open spec fn empty_queues(&self) -> bool {
        forall|t: int|
            0 <= t < self.num_tiers_spec() ==> (#[trigger] self.ready_queue_spec()[t])@.len() == 0
                && self.draining_queue_spec()[t]@.len() == 0
    }

    pub open spec fn consistent(&self, v: KernelView) -> bool {
        self.well_formed(v.id(), v.node, &v.timelines, &v.ledger)
    }

    pub open spec fn valid_frame(&self, f: FrameId, v: KernelView) -> bool {
        self.frame(f).version_agrees(self.versions_spec()@) && self.frame(f).valid(
            self.meta(f),
            self.ovf(f),
            self.num_tiers_spec(),
            self.tier_frame_size(self.frame(f).tier),
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
        &&& f.0 < self.metas_spec().len()
        &&& forall|i: nat|
            i < self.metas_spec().len() && i != f.0 ==> #[trigger] self.ghosts_spec()@[i].valid(
                self.metas_spec()[i as int],
                self.overflow_seqs_spec()[i as int]@,
                self.num_tiers_spec(),
                self.tier_frame_size(self.ghosts_spec()@[i].tier),
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
                1 <= s <= k.sequence() && #[trigger] v.timelines.events_spec()@[(
                    k.timeline_id(),
                    s,
                )] == e
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

    fn new_tier_sizes(num_tiers: usize, frame_size: usize) -> (r: Vec<usize>)
        requires
            0 < frame_size,
        ensures
            r.len() == num_tiers,
            forall|t: int| 0 <= t < num_tiers ==> #[trigger] r[t] == frame_size,
    {
        let mut sizes: Vec<usize> = Vec::new();
        for _t in 0..num_tiers
            invariant
                sizes.len() == _t,
                forall|j: int| 0 <= j < _t ==> #[trigger] sizes[j] == frame_size,
        {
            sizes.push(frame_size);
        }
        sizes
    }

    pub fn new(num_tiers: usize, frame_size: usize) -> (r: Self)
        requires
            num_tiers <= MAX_TIERS,
            0 < frame_size,
        ensures
            r.shape(),
            r.num_tiers_spec() == num_tiers,
            r.versions_spec()@ == Map::empty(),
            r.separated(),
            r.frame_size_spec() == frame_size,
            r.tier_frame_sizes_spec().len() == num_tiers,
            forall|t: int|
                0 <= t < num_tiers ==> #[trigger] r.tier_frame_sizes_spec()[t] == frame_size,
            r.metas_spec().len() == 0,
            r.empty_queues(),
    {
        let ready = Self::new_queues(num_tiers);
        let draining = Self::new_queues(num_tiers);
        let cursors = Self::new_cursors(num_tiers);
        let tier_frame_sizes = Self::new_tier_sizes(num_tiers, frame_size);
        let r = Self {
            num_tiers,
            frame_size,
            tier_frame_sizes,
            bindings: Vec::new(),
            metas: Vec::new(),
            overflow_seqs: Vec::new(),
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

    pub(crate) fn add_tier_with_size(&mut self, size: usize, Ghost(v): Ghost<KernelView>) -> (tier:
        TierId)
        requires
            old(self).consistent(v),
            old(self).num_tiers < MAX_TIERS,
            0 < size,
        ensures
            final(self).consistent(v),
            tier == TierId(old(self).num_tiers as u16),
            final(self).num_tiers == old(self).num_tiers + 1,
            final(self).frame_size == old(self).frame_size,
            final(self).tier_frame_sizes@ == old(self).tier_frame_sizes@.push(size),
            final(self).versions == old(self).versions,
            final(self).bindings == old(self).bindings,
            final(self).metas == old(self).metas,
            final(self).overflow_seqs == old(self).overflow_seqs,
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
        self.tier_frame_sizes.push(size);
        self.ready_queue.push(Vec::new());
        self.draining_queue.push(Vec::new());
        self.draining_cursor.push(0);
        proof {
            assert forall|i: nat|
                i < self.metas_spec().len() implies #[trigger] self.ghosts_spec()@[i].valid(
                self.metas_spec()[i as int],
                self.overflow_seqs_spec()[i as int]@,
                self.num_tiers_spec(),
                self.tier_frame_size(self.ghosts_spec()@[i].tier),
                v.id(),
                v.node,
                &v.timelines,
                &v.ledger,
            ) by {
                assert(old(self).ghosts_spec()@[i].valid(
                    old(self).metas_spec()[i as int],
                    old(self).overflow_seqs_spec()[i as int]@,
                    old(self).num_tiers_spec(),
                    old(self).tier_frame_size(old(self).ghosts_spec()@[i].tier),
                    v.id(),
                    v.node,
                    &v.timelines,
                    &v.ledger,
                ));
            }
            assert forall|u: int|
                0 <= u < self.num_tiers_spec() implies #[trigger] self.tier_queues(
                u,
                &v.timelines,
            ) by {
                if u < idx {
                    assert(old(self).tier_queues(u, &v.timelines));
                }
            }
            assert(self.shape());
            assert forall|i: nat|
                i
                    < self.metas_spec().len() implies #[trigger] self.ghosts_spec()@[i].version_agrees(
            self.versions_spec()@) by {
                assert(old(self).ghosts_spec()@[i].version_agrees(old(self).versions_spec()@));
            }
            assert(self.values_agree());
            assert(self.separated());
            assert(self.consistent(v));
        }
        TierId(idx as u16)
    }

    pub(crate) fn add_tier(&mut self, Ghost(v): Ghost<KernelView>) -> (tier: TierId)
        requires
            old(self).consistent(v),
            old(self).num_tiers < MAX_TIERS,
        ensures
            final(self).consistent(v),
            tier == TierId(old(self).num_tiers as u16),
            final(self).num_tiers == old(self).num_tiers + 1,
            final(self).frame_size == old(self).frame_size,
            final(self).tier_frame_sizes@ == old(self).tier_frame_sizes@.push(old(self).frame_size),
            final(self).versions == old(self).versions,
            final(self).bindings == old(self).bindings,
            final(self).metas == old(self).metas,
            final(self).overflow_seqs == old(self).overflow_seqs,
            final(self).ghosts == old(self).ghosts,
            final(self).ready_queue[old(self).num_tiers as int]@.len() == 0,
            final(self).draining_queue[old(self).num_tiers as int]@.len() == 0,
            forall|t: int|
                0 <= t < old(self).num_tiers ==> (#[trigger] final(self).ready_queue[t]) == old(
                    self,
                ).ready_queue[t] && final(self).draining_queue[t] == old(self).draining_queue[t],
    {
        self.add_tier_with_size(self.frame_size, Ghost(v))
    }

    pub(super) proof fn prove_valid_frames_unchanged(
        &self,
        before: Self,
        except: Option<nat>,
        id: InstanceId,
        node: NodeId,
        table: &TimelineTable,
        ledger: &KnowledgeLedger,
    )
        requires
            before.well_formed(id, node, table, ledger),
            self.num_tiers == before.num_tiers,
            self.tier_frame_sizes == before.tier_frame_sizes,
            before.metas.len() <= self.metas.len(),
            self.overflow_seqs.len() == self.metas.len(),
            forall|i: nat|
                i < before.metas.len() && except != Some(i) ==> #[trigger] self.ghosts_spec()@[i]
                    == before.ghosts_spec()@[i] && self.metas[i as int] == before.metas[i as int]
                    && self.overflow_seqs[i as int] == before.overflow_seqs[i as int],
        ensures
            forall|i: nat|
                i < before.metas.len() && except != Some(i)
                    ==> #[trigger] self.ghosts_spec()@[i].valid(
                    self.metas[i as int],
                    self.overflow_seqs[i as int]@,
                    self.num_tiers,
                    self.tier_frame_size(self.ghosts_spec()@[i].tier),
                    id,
                    node,
                    table,
                    ledger,
                ),
    {
        assert forall|i: nat|
            i < before.metas.len() && except != Some(
                i,
            ) implies #[trigger] self.ghosts_spec()@[i].valid(
            self.metas[i as int],
            self.overflow_seqs[i as int]@,
            self.num_tiers,
            self.tier_frame_size(self.ghosts_spec()@[i].tier),
            id,
            node,
            table,
            ledger,
        ) by {
            assert(before.ghosts_spec()@[i].valid(
                before.metas[i as int],
                before.overflow_seqs[i as int]@,
                before.num_tiers,
                before.tier_frame_size(before.ghosts_spec()@[i].tier),
                id,
                node,
                table,
                ledger,
            ));
        }
    }

    pub(super) proof fn restore_frame(&self, before: Self, f: FrameId, v: KernelView)
        requires
            before.except_frame(f, v),
            self.bindings_match(),
            before.versions_spec()@.submap_of(self.versions_spec()@),
            self.num_tiers == before.num_tiers,
            self.metas.len() == before.metas.len(),
            self.overflow_seqs.len() == self.metas.len(),
            self.frame_size == before.frame_size,
            self.tier_frame_sizes == before.tier_frame_sizes,
            self.ghosts_spec()@.dom() == before.ghosts_spec()@.dom(),
            self.ready_queue.len() == self.num_tiers,
            self.draining_queue.len() == self.num_tiers,
            self.draining_cursor.len() == self.num_tiers,
            self.frame(f).slot == before.frame(f).slot,
            self.frame(f).tier == before.frame(f).tier,
            self.valid_frame(f, v),
            self.queues(&v.timelines),
            forall|i: nat|
                i < self.metas.len() && i != f.0 ==> #[trigger] self.ghosts_spec()@[i]
                    == before.ghosts_spec()@[i] && self.metas[i as int] == before.metas[i as int]
                    && self.overflow_seqs[i as int] == before.overflow_seqs[i as int],
        ensures
            self.consistent(v),
    {
        assert forall|i: nat|
            i < self.metas.len() implies #[trigger] self.ghosts_spec()@[i].version_agrees(
            self.versions_spec()@,
        ) by {
            if i != f.0 {
                let g = before.ghosts_spec()@[i];
                assert(g.version_agrees(before.versions_spec()@));
                if let Some(_cert) = g.cert {
                    let version = g.bound_version->Some_0;
                    assert(self.versions_spec()@.contains_key(version));
                }
            }
        }
        assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts_spec()@[i].valid(
            self.metas[i as int],
            self.overflow_seqs[i as int]@,
            self.num_tiers,
            self.tier_frame_size(self.ghosts_spec()@[i].tier),
            v.id(),
            v.node,
            &v.timelines,
            &v.ledger,
        ) by {
            if i != f.0 {
                assert(before.ghosts_spec()@[i].valid(
                    before.metas[i as int],
                    before.overflow_seqs[i as int]@,
                    before.num_tiers,
                    before.tier_frame_size(before.ghosts_spec()@[i].tier),
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
            before.shape(),
            before.queues(&v.timelines),
            self.num_tiers == before.num_tiers,
            !before.meta(f).queued(),
            before.meta(f).state != STATE_DRAINING || !before.draining_queue[before.frame(
                f,
            ).tier.index() as int]@.contains(f),
            self.ready_queue == before.ready_queue,
            self.draining_queue == before.draining_queue,
            self.metas.len() == before.metas.len(),
            self.overflow_seqs.len() == self.metas.len(),
            forall|j: FrameId|
                j != f && j.0 < self.metas.len() ==> #[trigger] self.meta(j) == before.meta(j)
                    && self.ovf(j) == before.ovf(j) && self.frame(j).tier == before.frame(j).tier,
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
            ) == before.meta(self.ready_queue[t][i]) && self.ovf(self.ready_queue[t][i])
                == before.ovf(self.ready_queue[t][i]) by {
                let q = self.ready_queue[t][i];
                assert(q == before.ready_queue[t][i]);
                assert(q != f);
                assert(q.0 < before.metas.len());
                assert(self.meta(q) == before.meta(q));
                assert(self.ovf(q) == before.ovf(q));
            }
            assert forall|i: int| 0 <= i < self.draining_queue[t].len() implies self.meta(
                #[trigger] self.draining_queue[t][i],
            ) == before.meta(self.draining_queue[t][i]) by {
                let q = before.draining_queue[t][i];
                assert(q != f);
                assert(q.0 < before.metas.len());
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
            self.overflow_seqs.len() == self.metas.len(),
            self.frame_size == before.frame_size,
            self.tier_frame_sizes == before.tier_frame_sizes,
            self.ghosts_spec()@.dom() == before.ghosts_spec()@.dom(),
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
                i < self.metas.len() && i != f.0 ==> #[trigger] self.ghosts_spec()@[i]
                    == before.ghosts_spec()@[i] && self.metas[i as int] == before.metas[i as int]
                    && self.overflow_seqs[i as int] == before.overflow_seqs[i as int],
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
        ovf: Seq<u64>,
        num_tiers: usize,
        size: usize,
        before: KernelView,
        v: KernelView,
    )
        requires
            before.timelines.well_formed(before.id(), before.node, &before.ledger),
            g.valid(
                m,
                ovf,
                num_tiers,
                size,
                before.id(),
                before.node,
                &before.timelines,
                &before.ledger,
            ),
            v.extends(before),
        ensures
            g.valid(m, ovf, num_tiers, size, v.id(), v.node, &v.timelines, &v.ledger),
    {
        hide(TimelineTable::well_formed);
        hide(KnowledgeLedger::citations);
        hide(FrameGhost::authority);
        hide(FrameSyncMeta::read_bounds);
        let a = before.timelines;
        let b = v.timelines;
        assert forall|j: int| 0 <= j < 5 implies b.valid_ticket(#[trigger] m.ticket(j)) by {
            let k = m.ticket(j);
            assert(a.valid_ticket(k));
            if k != Ticket::NONE {
                let t = k.timeline_id();
                assert(a.issued(t, k.sequence()));
                assert(b.confirmed_seq_spec()[t as int] >= a.confirmed_seq_spec()[t as int]);
                assert(b.next_seq_spec()[t as int] >= a.next_seq_spec()[t as int]);
            }
        }
        assert(a.valid_ticket(m.ticket(4)));
        if let Some(c) = g.cert {
            let k = m.last_write_ticket;
            assert(a.issued(k.timeline_id(), k.sequence()));
            assert(b.events_spec()@[(k.timeline_id(), k.sequence())] == c.writer);
        }
        assert(g.binding(m, &b));
        assert(g.authority(m, v.id()));
        assert(FrameSyncMeta::ovf_bounded(ovf, &b)) by {
            assert forall|t: int| 0 <= t < ovf.len() implies #[trigger] ovf[t]
                < b.next_seq_spec()[t] by {
                assert(ovf[t] < a.next_seq_spec()[t]);
                assert(b.confirmed_seq_spec()[t] >= a.confirmed_seq_spec()[t]);
            }
        }
        if g.token is Some {
            if let Some(w) = g.frontier().last_writer {
                if !v.ledger.citations(v.node).contains(w) {
                    assert(!before.ledger.citations(before.node).contains(w));
                    let k = m.last_write_ticket;
                    if a.ticket_covers(k, w) {
                        let s = choose|s: u64|
                            1 <= s <= k.sequence() && #[trigger] a.events_spec()@[(
                                k.timeline_id(),
                                s,
                            )] == w;
                        assert(a.issued(k.timeline_id(), s));
                        assert(a.events_spec()@[(k.timeline_id(), s)] == w);
                        assert(b.events_spec()@[(k.timeline_id(), s)] == w);
                        assert(b.ticket_covers(k, w));
                    } else {
                        assert(a.issued(k.timeline_id(), k.sequence()));
                        assert(b.events_spec()@[(k.timeline_id(), k.sequence())] == w);
                    }
                }
            }
            assert forall|r| #[trigger]
                g.frontier().readers.contains(r) && !v.ledger.citations(v.node).contains(
                    r,
                ) implies exists|t: TimelineId, s: u64|
                b.issued(t, s) && #[trigger] b.events_spec()@[(t, s)] == r && m.bounds_reader(
                    ovf,
                    t,
                    s,
                ) by {
                assert(!before.ledger.citations(before.node).contains(r));
                assert(m.accounts(ovf, &a, before.node, &before.ledger, r));
                let (t, s) = choose|t: TimelineId, s: u64|
                    a.issued(t, s) && #[trigger] a.events_spec()@[(t, s)] == r && m.bounds_reader(
                        ovf,
                        t,
                        s,
                    );
                assert(b.confirmed_seq_spec()[t as int] >= a.confirmed_seq_spec()[t as int]);
                assert(a.events_spec()@[(t, s)] == r);
                assert(b.issued(t, s));
                assert(b.events_spec()@[(t, s)] == r);
                assert(m.bounds_reader(ovf, t, s));
            }
        }
        assert(g.covered(m, ovf, &b, v.node, &v.ledger));
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
        assert forall|i: nat| i < self.metas.len() implies #[trigger] self.ghosts_spec()@[i].valid(
            self.metas[i as int],
            self.overflow_seqs[i as int]@,
            self.num_tiers,
            self.tier_frame_size(self.ghosts_spec()@[i].tier),
            v.id(),
            v.node,
            &b,
            &v.ledger,
        ) by {
            Self::lemma_frame_timeline_progress(
                self.ghosts_spec()@[i],
                self.metas[i as int],
                self.overflow_seqs[i as int]@,
                self.num_tiers,
                self.tier_frame_size(self.ghosts_spec()@[i].tier),
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
            ).reusable(self.ovf(self.ready_queue[u][i]), &b) by {
                let f = self.ready_queue[u][i];
                let m = self.meta(f);
                assert(m.reusable(self.ovf(f), &a));
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
            before.frame(f).covered(before.meta(f), before.ovf(f), &v.timelines, v.node, &v.ledger),
            self.frame(f).slot == before.frame(f).slot,
            self.frame(f).token == before.frame(f).token,
            self.meta(f).last_write_ticket == before.meta(f).last_write_ticket,
            self.meta(f).inline_read_tickets == before.meta(f).inline_read_tickets,
            self.meta(f).overflow() == before.meta(f).overflow(),
            self.ovf(f) == before.ovf(f),
        ensures
            self.frame(f).covered(self.meta(f), self.ovf(f), &v.timelines, v.node, &v.ledger),
    {
        assert forall|j: int| 0 <= j < 5 implies v.timelines.valid_ticket(
            #[trigger] self.meta(f).ticket(j),
        ) by {
            assert(before.meta(f).ticket(j) == self.meta(f).ticket(j));
        }
        self.frame(f).preserve_readers(
            before.frame(f),
            self.meta(f),
            self.ovf(f),
            before.meta(f),
            before.ovf(f),
            &v.timelines,
            &v.timelines,
            v.node,
            &v.ledger,
        );
    }

    pub open spec fn preserves_initializing(&self, before: &Self, except: Option<FrameId>) -> bool {
        forall|g: FrameId|
            Some(g) != except && g.0 < before.metas_spec().len() && (#[trigger] before.meta(
                g,
            )).state == STATE_INITIALIZING ==> self.meta(g) == before.meta(g) && self.ovf(g)
                == before.ovf(g) && self.frame(g) == before.frame(g)
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
        &&& self.num_tiers_spec() == before.num_tiers_spec() && self.frame_size_spec()
            == before.frame_size_spec() && self.tier_frame_sizes_spec()
            == before.tier_frame_sizes_spec() && self.metas_spec().len()
            == before.metas_spec().len()
        &&& forall|f: FrameId|
            f.0 < self.metas_spec().len() ==> (#[trigger] self.frame(f)).slot == before.frame(
                f,
            ).slot && self.frame(f).tier == before.frame(f).tier
    }

    pub open spec fn same_directory(&self, before: &Self) -> bool {
        &&& self.num_tiers_spec() == before.num_tiers_spec() && self.versions_spec()
            == before.versions_spec() && self.frame_size_spec() == before.frame_size_spec()
            && self.tier_frame_sizes_spec() == before.tier_frame_sizes_spec()
            && self.metas_spec().len() == before.metas_spec().len()
        &&& forall|f: FrameId|
            f.0 < self.metas_spec().len() ==> {
                &&& (#[trigger] self.frame(f)).slot == before.frame(f).slot
                &&& self.frame(f).tier == before.frame(f).tier
                &&& self.published(f) == before.published(f)
            }
    }

    pub open spec fn same_other_pins(&self, before: &Self, changed: Option<FrameId>) -> bool {
        forall|f: FrameId|
            f.0 < self.metas_spec().len() && Some(f) != changed ==> {
                &&& (#[trigger] self.frame(f)).local_shares == before.frame(f).local_shares
                &&& self.frame(f).fenced_shares == before.frame(f).fenced_shares
                &&& self.meta(f).active_fenced_leases == before.meta(f).active_fenced_leases
            }
    }

    pub open spec fn preserves_fenced_leases(&self, before: &Self) -> bool {
        forall|f: FrameId|
            f.0 < before.metas_spec().len() ==> {
                &&& (#[trigger] self.frame(f)).fenced_shares == before.frame(f).fenced_shares
                &&& self.meta(f).active_fenced_leases == before.meta(f).active_fenced_leases
            }
    }
}

} // verus!
