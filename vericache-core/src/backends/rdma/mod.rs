use super::*;
use crate::timeline::{complete_data, PendingOperation, SlidingBitmap};
#[cfg(verus_only)]
use crate::timeline::{
    record_submission, window_matches, DataOperation, TimelineTable, BITMAP_CAPACITY,
};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, EventInfo};
#[cfg(verus_only)]
use transfer::context::SlotToken;
use transfer::context::{EventSourceToken, ReturnedShare, ShareCap, ShareSource, TransferInstance};
use transfer::domain::{DeviceId, NodeId, TransferShare};
#[cfg(verus_only)]
use transfer::domain::{TransferDomain, TransferEvent};
use transfer::host::HostContext;
use transfer::rdma;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

pub mod batch;
pub mod ops;
pub mod peer;
pub mod poll;
#[cfg(verus_only)]
pub mod proofs;
pub mod protocol;
pub mod pull;
pub mod revoke;
pub mod unbind;
pub mod window_lifecycle;
pub mod window_pool;

#[cfg(verus_only)]
use proofs::*;
#[cfg(verus_only)]
pub use window_pool::window_pool_slots_wf;
pub use window_pool::{MwPoolSlot, MwSlotState, WindowPool};

verus! {

/// Control WR IDs (`BindWindow`, `LocalInvalidate`) live in the high half of
/// `u64`, strictly disjoint from FIFO data sequence numbers `1..=MAX_TICKET_SEQ`.
pub const CONTROL_WR_ID_BASE: u64 = 0x8000_0000_0000_0000u64;

/// Stable `(mw_slot, generation)` handle identifying an active remote RDMA read
/// lease across `swap_remove`s in `active_leases` (M5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteLeaseHandle {
    pub mw_slot: usize,
    pub generation: u32,
}

/// Predicate stating every lease except `removed` stays present across a `swap_remove`.
pub open spec fn rdma_leases_preserved_except(
    before: Seq<ActiveRemoteLease>,
    after: Seq<ActiveRemoteLease>,
    removed: RemoteLeaseHandle,
) -> bool {
    &&& forall|k: int|
        0 <= k < before.len() && !((#[trigger] before[k]).mw_slot as usize == removed.mw_slot
            && before[k].generation == removed.generation) ==> exists|j: int|
            0 <= j < after.len() && (#[trigger] after[j]) == before[k]
    &&& forall|j: int|
        0 <= j < after.len() ==> !((#[trigger] after[j]).mw_slot as usize == removed.mw_slot
            && after[j].generation == removed.generation)
}

/// Result returned by `RdmaBackend::poll_cq` / `poll_cq_batch`.
pub enum PeerCompletion {
    Data(PendingOperation, Result<(), i32>),
    BindFailed { wr_id: u64, status: i32, witness: Tracked<rdma::BindFailedWitness> },
    Control(rdma::Completed),
}

/// Lifecycle state of an active remote RDMA read lease managed by `RdmaBackend`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RemoteLeaseState {
    /// MW bind WR is posted; `share_cap` is retained locally until bind completes or fails.
    Binding,
    /// MW is bound; `RemoteCapability` + `ShareCap` were handed off to the protocol layer.
    Bound,
    /// Cooperatively unpinned in `CacheCore`; `ReturnedShare` awaits `WindowInvalidated`.
    Unbinding,
    /// Unilateral fenced revocation in progress; still pinned in `CacheCore`, awaiting `WindowInvalidated`.
    Revoking,
}

/// One active remote read lease tracked by [`RdmaBackend`].
pub struct ActiveRemoteLease {
    pub frame: FrameId,
    pub mw_slot: u32,
    pub generation: u32,
    pub wr_id: u64,
    pub state: RemoteLeaseState,
    pub share: Ghost<TransferShare>,
    pub share_cap: Option<Tracked<ShareCap>>,
    pub returned_share: Option<Tracked<ReturnedShare>>,
}

/// Registered RDMA `MemoryRegion` bound to a contiguous slab of `CacheCore` frames.
pub struct RegisteredSlab<'a> {
    pub mr: rdma::MemoryRegion<'a>,
    pub tier: TierId,
    pub first_frame: FrameId,
    pub num_frames: u32,
}

impl<'a> RegisteredSlab<'a> {
    pub open spec fn well_formed(&self, core: &CacheCore, pd_id: nat) -> bool {
        &&& self.mr.well_formed()
        &&& self.mr.instance_id() == core.instance_id()
        &&& self.mr.node() == core.node()
        &&& self.mr.pd_id() == pd_id
        &&& self.mr.access().mw_bind
        &&& self.mr.num_slots() == self.num_frames as usize
        &&& self.mr.slot_len() == core.frames.frame_size
        &&& self.tier.valid(core.tiers.len())
        &&& slab_bound_to_core(core, self.tier, self.first_frame, self.num_frames, self.mr.space())
    }

    pub open spec fn owns(&self, frame: FrameId) -> bool {
        self.first_frame.0 <= frame.0 && frame.0 < self.first_frame.0 + self.num_frames
    }

    pub fn owns_frame(&self, frame: FrameId) -> (r: bool)
        ensures
            r == self.owns(frame),
    {
        frame.0 >= self.first_frame.0 && frame.0 - self.first_frame.0 < self.num_frames
    }

    /// Query the GPUDirect RDMA write ordering mode (returns `None` for host DRAM MRs).
    pub fn gdr_ordering(&self) -> Option<rdma::RdmaGdrWritesOrdering> {
        self.mr.gdr_ordering()
    }

    /// Check whether GPUDirect RDMA writes to this slab are natively visible
    /// to `consumer_device` without a host-side flush.
    pub fn gdr_visible_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            self.mr.well_formed(),
        ensures
            r == self.mr.gdr_visible_to_device_spec(consumer_device),
    {
        self.mr.gdr_visible_to_device(consumer_device)
    }

    /// Check whether GPUDirect RDMA writes to this slab are visible to
    /// `consumer_device` after `self.flush_gdr_writes()`.
    pub fn gdr_visible_after_flush_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            self.mr.well_formed(),
        ensures
            r == self.mr.gdr_visible_after_flush_to_device_spec(consumer_device),
    {
        self.mr.gdr_visible_after_flush_to_device(consumer_device)
    }

    /// Issue a GPUDirect RDMA write flush targeting the slab's owning GPU derived
    /// from `self.mr.gpu_device()` when `RdmaGdrWritesOrdering::None`.
    pub fn flush_gdr_writes(&self) -> BackendResult<()>
        requires
            self.mr.well_formed(),
    {
        match self.mr.flush_gdr_writes() {
            Ok(()) => Ok(()),
            Err(e) => Err(cuda_error(e)),
        }
    }
}

/// Filter a CQ snapshot's pending map down to data-transfer work requests.
pub open spec fn rdma_data_events(cq: rdma::CqSnapshot) -> Map<u64, TransferEvent> {
    Map::new(
        cq.pending.dom().filter(|s| cq.pending[s].kind is DataTransfer),
        |s: u64| cq.pending[s].kind->DataTransfer_event,
    )
}

/// Remove a contiguous sequence range `[start, start + count)` from a pending data map.
pub open spec fn rdma_remove_range(m: Map<u64, TransferEvent>, start: u64, count: int) -> Map<
    u64,
    TransferEvent,
> {
    Map::new(m.dom().filter(|s: u64| !(start <= s < start + count)), |s: u64| m[s])
}

/// Opaque data-window and control-WR-ID invariant for `RdmaBackend`.
#[verifier::opaque]
pub open spec fn rdma_window_wf(
    ops: [Option<PendingOperation>; 512],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: u8,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
) -> bool {
    &&& t < table.count
    &&& table.well_formed(id, cq.observer, ledger)
    &&& bitmap.well_formed()
    &&& bitmap.observations(table, t, cq.observer, ledger)
    &&& window_matches(ops, bitmap, table, t, rdma_data_events(cq))
    &&& CONTROL_WR_ID_BASE <= next_control_wr_id
    &&& forall|s: u64|
        cq.pending.contains_key(s) && !((#[trigger] cq.pending[s]).kind is DataTransfer)
            ==> CONTROL_WR_ID_BASE <= s < next_control_wr_id
}

impl ActiveRemoteLease {
    pub open spec fn valid<'a>(
        &self,
        pool: &WindowPool<'a>,
        qp: &rdma::QueuePair<'a>,
        core: &CacheCore,
    ) -> bool {
        &&& self.frame.0 < core.frames.metas.len()
        &&& (self.mw_slot as int) < pool.slots.len()
        &&& pool.slots[self.mw_slot as int].generation == self.generation
        &&& match self.state {
            RemoteLeaseState::Binding => {
                &&& core.frames.meta(self.frame).active_remote_leases > 0
                &&& core.frames.frame(self.frame).matches_lease(
                    self.share@,
                    self.mw_slot,
                    self.generation,
                )
                &&& pool.slots[self.mw_slot as int].state is Binding
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Binding
                &&& w.state()->Binding_qp_id == qp.id()
                &&& w.state()->Binding_instance_id == core.instance_id()
                &&& w.state()->Binding_share == Some(self.share@)
                &&& w.state()->Binding_wr_id == self.wr_id
                &&& w.state()->Binding_target_slot == core.frames.frame(self.frame).slot
                &&& w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                &&& self.share_cap matches Some(cap)
                &&& cap@.is_rdma_bound()
                &&& cap@.instance_id() == core.instance_id()
                &&& cap@.loc() == core.frames.frame(self.frame).slot.loc
                &&& cap@.share() == self.share@
                &&& cap@.frontier().readers.is_empty()
                &&& self.returned_share is None
            },
            RemoteLeaseState::Bound => {
                &&& core.frames.meta(self.frame).active_remote_leases > 0
                &&& core.frames.frame(self.frame).matches_lease(
                    self.share@,
                    self.mw_slot,
                    self.generation,
                )
                &&& pool.slots[self.mw_slot as int].state is Bound
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Bound
                &&& w.state()->Bound_qp_id == qp.id()
                &&& w.state()->Bound_instance_id == core.instance_id()
                &&& w.state()->Bound_share == Some(self.share@)
                &&& w.state()->Bound_slot == core.frames.frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share is None
            },
            RemoteLeaseState::Unbinding => {
                &&& pool.slots[self.mw_slot as int].state is Invalidating
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Invalidating
                &&& w.state()->Invalidating_qp_id == qp.id()
                &&& w.state()->Invalidating_instance_id == core.instance_id()
                &&& w.state()->Invalidating_share == Some(self.share@)
                &&& w.state()->Invalidating_wr_id == self.wr_id
                &&& w.state()->Invalidating_slot == core.frames.frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share matches Some(r)
                &&& r@.share() == self.share@
                &&& r@.instance_id() == core.instance_id()
                &&& r@.loc() == core.frames.frame(self.frame).slot.loc
            },
            RemoteLeaseState::Revoking => {
                &&& core.frames.meta(self.frame).active_remote_leases > 0
                &&& core.frames.frame(self.frame).matches_lease(
                    self.share@,
                    self.mw_slot,
                    self.generation,
                )
                &&& pool.slots[self.mw_slot as int].state is Invalidating
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Invalidating
                &&& w.state()->Invalidating_qp_id == qp.id()
                &&& w.state()->Invalidating_instance_id == core.instance_id()
                &&& w.state()->Invalidating_share == Some(self.share@)
                &&& w.state()->Invalidating_wr_id == self.wr_id
                &&& w.state()->Invalidating_slot == core.frames.frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share is None
            },
        }
    }
}

/// Pairwise MW-slot disjointness of all active remote leases (split out for CVC5).
#[verifier::opaque]
pub open spec fn rdma_leases_disjoint(leases: Seq<ActiveRemoteLease>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < leases.len() ==> (#[trigger] leases[i]).mw_slot != (
        #[trigger] leases[j]).mw_slot
}

/// Opaque validity of all active remote leases.
#[verifier::opaque]
pub open spec fn rdma_leases_valid<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    qp: &rdma::QueuePair<'a>,
    core: &CacheCore,
) -> bool {
    &&& rdma_leases_disjoint(leases)
    &&& forall|i: int| 0 <= i < leases.len() ==> (#[trigger] leases[i]).valid(pool, qp, core)
}

pub open spec fn rdma_lease_index_spec(
    leases: Seq<ActiveRemoteLease>,
    handle: RemoteLeaseHandle,
) -> Option<int> {
    if exists|i: int|
        0 <= i < leases.len() && (#[trigger] leases[i]).mw_slot as usize == handle.mw_slot
            && leases[i].generation == handle.generation {
        Some(
            choose|i: int|
                0 <= i < leases.len() && (#[trigger] leases[i]).mw_slot as usize == handle.mw_slot
                    && leases[i].generation == handle.generation,
        )
    } else {
        None
    }
}

pub(super) proof fn lemma_rdma_leases_push_index(
    leases: Seq<ActiveRemoteLease>,
    handle: RemoteLeaseHandle,
)
    requires
        rdma_leases_disjoint(leases),
        leases.len() > 0,
        leases.last().mw_slot as usize == handle.mw_slot,
        leases.last().generation == handle.generation,
    ensures
        rdma_lease_index_spec(leases, handle) == Some((leases.len() - 1) as int),
{
    reveal(rdma_leases_disjoint);
    let last_idx = (leases.len() - 1) as int;
    assert(leases[last_idx].mw_slot as usize == handle.mw_slot && leases[last_idx].generation
        == handle.generation);
    assert forall|k: int|
        0 <= k < leases.len() && (#[trigger] leases[k]).mw_slot as usize == handle.mw_slot
            && leases[k].generation == handle.generation implies k == last_idx by {
        if k < last_idx {
            assert(leases[k].mw_slot != leases[last_idx].mw_slot);
        }
    }
}

/// Pluggable RDMA peer backend built strictly over `CacheCore`'s verified
/// transaction API and `vericache-transfer`'s Type-2 MW / single-key revocation API.
pub struct RdmaBackend<'a> {
    pub timeline_id: u8,
    pub peer_node: NodeId,
    pub qp: rdma::QueuePair<'a>,
    pub cq: rdma::CompletionQueue<'a>,
    pub bitmap: SlidingBitmap,
    pub in_flight_ops: [Option<PendingOperation>; 512],
    pub window_pool: WindowPool<'a>,
    pub host_mr: Option<RegisteredSlab<'a>>,
    pub gpu_mr: Option<RegisteredSlab<'a>>,
    pub active_leases: Vec<ActiveRemoteLease>,
    pub next_control_wr_id: u64,
}

impl<'a> RdmaBackend<'a> {
    pub open spec fn pending(&self) -> Map<u64, TransferEvent> {
        rdma_data_events(rdma::CqSnapshot::of(self.cq))
    }

    pub open spec fn well_formed(&self, core: &CacheCore) -> bool {
        let t = self.timeline_id;
        &&& t < core.timelines.count
        &&& core.timelines.kinds[t as int] == (TimelineKind::RdmaQueuePair {
            peer_node: self.peer_node,
        })
        &&& self.qp.cq_id() == self.cq.id()
        &&& (self.qp.state() is ReadyToSend ==> self.qp.peer() == self.peer_node && self.qp.pd_id()
            == self.window_pool.pd())
        &&& self.qp.source().id() == core.instance_id()
        &&& self.qp.source().node() == core.node()
        &&& self.cq.well_formed()
        &&& self.cq.observer() == core.node()
        &&& self.cq.infos().instance_id() == core.instance_id()
        &&& self.window_pool.well_formed()
        &&& self.window_pool.node() == core.node()
        &&& match self.host_mr {
            None => true,
            Some(ref s) => s.well_formed(core, self.window_pool.pd()),
        }
        &&& match self.gpu_mr {
            None => true,
            Some(ref s) => s.well_formed(core, self.window_pool.pd()),
        }
        &&& rdma_window_wf(
            self.in_flight_ops,
            self.bitmap,
            &core.timelines,
            t,
            &core.ledger@,
            rdma::CqSnapshot::of(self.cq),
            core.instance_id(),
            self.next_control_wr_id,
        )
        &&& rdma_leases_valid(self.active_leases@, &self.window_pool, &self.qp, core)
    }

    pub fn timeline_id(&self) -> (t: u8)
        ensures
            t == self.timeline_id,
    {
        self.timeline_id
    }

    pub fn peer_node(&self) -> (p: NodeId)
        ensures
            p == self.peer_node,
    {
        self.peer_node
    }

    pub fn active_lease_count(&self) -> (n: usize)
        ensures
            n == self.active_leases@.len(),
    {
        self.active_leases.len()
    }

    /// Spec-level index of the active lease matching `(handle.mw_slot, handle.generation)`.
    pub open spec fn lease_index_spec(&self, handle: RemoteLeaseHandle) -> Option<int> {
        rdma_lease_index_spec(self.active_leases@, handle)
    }

    /// Spec helper returning the `MemoryWindow` referenced by `handle` when its
    /// lease is currently in `state`.
    pub open spec fn lease_matches_window_state(
        &self,
        handle: RemoteLeaseHandle,
        state: RemoteLeaseState,
    ) -> Option<rdma::MemoryWindow<'a>> {
        match self.lease_index_spec(handle) {
            Some(idx) => if self.active_leases@[idx].state == state {
                self.window_pool.slots[self.active_leases@[idx].mw_slot as int].window
            } else {
                None
            },
            None => None,
        }
    }

    /// Look up an active remote lease by stable `(mw_slot, generation)` handle.
    fn find_lease_index(&self, handle: RemoteLeaseHandle) -> (r: Option<usize>)
        requires
            rdma_leases_disjoint(self.active_leases@),
        ensures
            match r {
                None => self.lease_index_spec(handle) is None,
                Some(idx) => {
                    &&& idx < self.active_leases@.len()
                    &&& self.lease_index_spec(handle) == Some(idx as int)
                    &&& self.active_leases@[idx as int].mw_slot as usize == handle.mw_slot
                    &&& self.active_leases@[idx as int].generation == handle.generation
                },
            },
    {
        proof {
            reveal(rdma_leases_disjoint);
        }
        let mut i: usize = 0;
        while i < self.active_leases.len()
            invariant
                i <= self.active_leases@.len(),
                rdma_leases_disjoint(self.active_leases@),
                forall|k: int|
                    0 <= k < i ==> !((#[trigger] self.active_leases@[k]).mw_slot as usize
                        == handle.mw_slot && self.active_leases@[k].generation
                        == handle.generation),
            decreases self.active_leases.len() - i,
        {
            let entry = &self.active_leases[i];
            if entry.mw_slot as usize == handle.mw_slot && entry.generation == handle.generation {
                proof {
                    reveal(rdma_leases_disjoint);
                    assert forall|k: int|
                        0 <= k < self.active_leases@.len() && (
                        #[trigger] self.active_leases@[k]).mw_slot as usize == handle.mw_slot
                            && self.active_leases@[k].generation == handle.generation implies k
                        == i as int by {
                        if k < i as int {
                        } else if k > i as int {
                            assert(self.active_leases@[i as int].mw_slot
                                != self.active_leases@[k].mw_slot);
                        }
                    }
                    assert(self.active_leases@[i as int].mw_slot as usize == handle.mw_slot
                        && self.active_leases@[i as int].generation == handle.generation);
                }
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Create an `RdmaBackend`, registering its `TimelineKind::RdmaQueuePair`
    /// timeline in `core` and wrapping pre-allocated Type-2 MWs in a `WindowPool`.
    pub fn new(
        core: &mut CacheCore,
        peer_node: NodeId,
        qp: rdma::QueuePair<'a>,
        cq: rdma::CompletionQueue<'a>,
        windows: Vec<rdma::MemoryWindow<'a>>,
    ) -> (r: Result<
        Self,
        (CoreError, rdma::QueuePair<'a>, rdma::CompletionQueue<'a>, Vec<rdma::MemoryWindow<'a>>),
    >)
        requires
            old(core).well_formed(),
            qp.ready(),
            qp.cq_id() == cq.id(),
            qp.peer() == peer_node,
            qp.source().id() == old(core).instance_id(),
            qp.source().node() == old(core).node(),
            cq.well_formed(),
            cq.observer() == old(core).node(),
            cq.infos().instance_id() == old(core).instance_id(),
            cq.in_flight().is_empty(),
            windows.len() <= u32::MAX,
            forall|j: int|
                0 <= j < windows.len() ==> (#[trigger] windows[j]).state()
                    == rdma::WindowState::Unbound && windows[j].pd_id() == qp.pd_id()
                    && windows[j].node() == old(core).node() && windows[j].consistent(),
        ensures
            final(core).well_formed(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).ledger == old(core).ledger,
            final(core).spec == old(core).spec,
            match r {
                Err((_, ret_qp, ret_cq, ret_windows)) => *final(core) == *old(core) && ret_qp == qp
                    && ret_cq == cq && ret_windows == windows,
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.timeline_id == old(core).timelines.count
                    &&& b.peer_node == peer_node
                    &&& b.qp == qp
                    &&& b.cq == cq
                    &&& b.host_mr is None
                    &&& b.gpu_mr is None
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let timeline_id = match core.register_timeline(TimelineKind::RdmaQueuePair { peer_node }) {
            Ok(t) => t,
            Err(e) => return Err((e, qp, cq, windows)),
        };
        let ghost pd_id = qp.pd_id();
        let ghost node = core.node();
        let window_pool = WindowPool::new(windows, Ghost(pd_id), Ghost(node));
        let backend = Self {
            timeline_id,
            peer_node,
            qp,
            cq,
            bitmap: SlidingBitmap::new(),
            in_flight_ops: [None;512],
            window_pool,
            host_mr: None,
            gpu_mr: None,
            active_leases: Vec::new(),
            next_control_wr_id: CONTROL_WR_ID_BASE,
        };
        proof {
            lemma_core_wf_facts(core);
            lemma_rdma_init_window_wf(
                backend.in_flight_ops,
                backend.bitmap,
                &core.timelines,
                timeline_id,
                &core.ledger@,
                rdma::CqSnapshot::of(backend.cq),
                core.instance_id(),
            );
            lemma_rdma_leases_empty(
                backend.active_leases@,
                &backend.window_pool,
                &backend.qp,
                core,
            );
        }
        Ok(backend)
    }

    /// Attach a registered host DRAM slab MR (aligned to a `CacheCore` tier).
    pub fn register_host_slab(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a transfer::host::HostBuffer,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).host_mr is None,
            pd.id() == old(self).window_pool.pd(),
            pd.node() == core.node(),
            flags.mw_bind,
            buffer.well_formed(),
            buffer.instance_id() == core.instance_id(),
            buffer.node() == core.node(),
            buffer.num_slots() == num_frames as usize,
            buffer.slot_len() == core.frames.frame_size,
            tier.valid(core.tiers.len()),
            slab_bound_to_core(core, tier, first_frame, num_frames, buffer.space()),
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> final(self).host_mr is Some,
    {
        if self.host_mr.is_some() {
            return Err(CoreError::InvalidArgument);
        }
        let _ = core;
        let mr = match rdma::register_memory(pd, rdma::RegistrationBuffer::Host(buffer), flags) {
            Ok(mr) => mr,
            Err(e) => return Err(rdma_error(e)),
        };
        self.host_mr = Some(RegisteredSlab { mr, tier, first_frame, num_frames });
        Ok(())
    }

    /// Attach a registered GPU VRAM slab MR (with GPUDirect RDMA ordering queried).
    pub fn register_gpu_slab(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a transfer::cuda::CudaBuffer,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).gpu_mr is None,
            pd.id() == old(self).window_pool.pd(),
            pd.node() == core.node(),
            flags.mw_bind,
            buffer.well_formed(),
            buffer.instance_id() == core.instance_id(),
            buffer.node() == core.node(),
            buffer.num_slots() == num_frames as usize,
            buffer.slot_len() == core.frames.frame_size,
            tier.valid(core.tiers.len()),
            slab_bound_to_core(core, tier, first_frame, num_frames, buffer.space()),
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> final(self).gpu_mr is Some,
    {
        if self.gpu_mr.is_some() {
            return Err(CoreError::InvalidArgument);
        }
        let _ = core;
        let mr = match rdma::register_memory(pd, rdma::RegistrationBuffer::Cuda(buffer), flags) {
            Ok(mr) => mr,
            Err(e) => return Err(rdma_error(e)),
        };
        self.gpu_mr = Some(RegisteredSlab { mr, tier, first_frame, num_frames });
        Ok(())
    }

    /// Query GPUDirect RDMA write ordering on the attached GPU MR, if present.
    pub fn gpu_gdr_ordering(&self) -> Option<rdma::RdmaGdrWritesOrdering> {
        match &self.gpu_mr {
            Some(slab) => slab.gdr_ordering(),
            None => None,
        }
    }

    /// Check if GPUDirect RDMA writes to the attached GPU MR are natively visible
    /// to `consumer_device` without a host-side flush.
    pub fn gdr_visible_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            self.gpu_mr matches Some(ref s) ==> s.mr.well_formed(),
        ensures
            self.gpu_mr matches Some(ref s) ==> r == s.mr.gdr_visible_to_device_spec(
                consumer_device,
            ),
            self.gpu_mr is None ==> r,
    {
        match &self.gpu_mr {
            Some(slab) => slab.gdr_visible_to_device(consumer_device),
            None => true,
        }
    }

    /// Check if GPUDirect RDMA writes to the attached GPU MR are visible to
    /// `consumer_device` after `self.flush_gdr_writes()`.
    pub fn gdr_visible_after_flush_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            self.gpu_mr matches Some(ref s) ==> s.mr.well_formed(),
        ensures
            self.gpu_mr matches Some(ref s) ==> r == s.mr.gdr_visible_after_flush_to_device_spec(
                consumer_device,
            ),
            self.gpu_mr is None ==> r,
    {
        match &self.gpu_mr {
            Some(slab) => slab.gdr_visible_after_flush_to_device(consumer_device),
            None => true,
        }
    }

    /// Flush GPUDirect RDMA writes to the owning GPU derived from `gpu_mr` when required.
    pub fn flush_gdr_writes(&self) -> BackendResult<()>
        requires
            self.gpu_mr matches Some(ref s) ==> s.mr.well_formed(),
    {
        match &self.gpu_mr {
            Some(slab) => slab.flush_gdr_writes(),
            None => Ok(()),
        }
    }
}

} // verus!
