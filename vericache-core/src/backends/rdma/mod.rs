use super::*;
#[cfg(verus_only)]
use crate::timeline::{window_matches, DataOperation, TimelineTable, BITMAP_CAPACITY};
use crate::timeline::{PendingOperation, SlidingBitmap};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, EventInfo};
#[cfg(verus_only)]
use transfer::context::SlotToken;
use transfer::context::{
    EventSourceToken, ReturnedShare, ShareCap, ShareSource, SyncContext, TransferInstance,
};
use transfer::domain::{DeviceId, NodeId, TransferShare};
#[cfg(verus_only)]
use transfer::domain::{TransferDomain, TransferEvent};
use transfer::rdma::{self, RdmaError};
use transfer_core::memory::RdmaRegisterableSlab;
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
pub use window_lifecycle::WindowInvalidation;
#[cfg(verus_only)]
pub use window_pool::window_pool_slots_wf;
pub use window_pool::{MwPoolSlot, MwSlotState, WindowPool};

#[cfg(verus_only)]
mod preserve;
#[cfg(verus_only)]
pub use preserve::*;

verus! {

pub fn rdma_error(e: RdmaError) -> (r: CoreError)
    ensures
        r == (CoreError::Backend { kind: BACKEND_RDMA, code: e.0 }),
{
    CoreError::Backend { kind: BACKEND_RDMA, code: e.0 }
}

/// Control WR IDs (`BindWindow`, `LocalInvalidate`) live in `[1 << 60, 1 << 62)`,
/// strictly disjoint from FIFO data sequence numbers `1..=MAX_TICKET_SEQ` and
/// always fitting within the 62-bit inline `wr_id <= UINT64_MAX >> 2` range
/// (zero C heap `malloc` in `verbs.c`).
pub const CONTROL_WR_ID_BASE: u64 = 0x1000_0000_0000_0000u64;

/// Exclusive upper bound (`1 << 62`) for inline 62-bit control WR IDs.
pub const CONTROL_WR_ID_LIMIT: u64 = 0x4000_0000_0000_0000u64;

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
    /// `WindowBound` CQE has been reaped by `pump_cq`; `bind_witness` is staged for `complete_bind`.
    BindReady,
    /// `BindFailed` CQE has been reaped by `pump_cq`; `bind_failed_witness` + `staged_error` are staged.
    BindFailedReady,
    /// MW is bound; `RemoteCapability` + `ShareCap` were handed off to the protocol layer.
    Bound,
    /// Cooperatively unpinned in `CacheCore`; `ReturnedShare` awaits `WindowInvalidated`.
    Unbinding,
    /// Unilateral fenced revocation in progress; still pinned in `CacheCore`, awaiting `WindowInvalidated`.
    Revoking,
    /// `WindowInvalidated` CQE for a `Revoking` lease was reaped by `pump_cq`; `inv_witness` is staged.
    RevokeInvReady,
    /// `ControlFailed` CQE was reaped by `pump_cq`; `staged_error` records the failure status.
    ControlFailedState,
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
    pub bind_witness: Option<Tracked<rdma::BindWitness>>,
    pub bind_failed_witness: Option<Tracked<rdma::BindFailedWitness>>,
    pub inv_witness: Option<Tracked<rdma::InvalidateWitness>>,
    pub staged_error: Option<i32>,
}

/// Transparent layout predicate matching [`slab_bound_to_core`] (used for shared slabs
/// so sibling `compose` preservation proofs work without revealing `slab_bound_to_core`).
pub open(crate) spec fn slab_bound_open(
    core: &CacheCore,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
    space: MemorySpace,
) -> bool {
    &&& tier.valid(core.tiers_spec().len())
    &&& num_frames > 0
    &&& (first_frame.0 as nat) + (num_frames as nat) <= core.frames_spec().metas_spec().len()
    &&& forall|j: nat|
        j < num_frames ==> {
            &&& (#[trigger] core.frames_spec().frame(FrameId((first_frame.0 + j) as u32))).tier
                == tier
            &&& core.frames_spec().frame(FrameId((first_frame.0 + j) as u32)).slot == SlotRef::full(
                SlotLoc { node: core.node(), space, slot_index: j },
                core.frames_spec().tier_frame_size(tier) as u64,
            )
        }
    &&& forall|f: FrameId|
        f.0 < core.frames_spec().metas_spec().len() && (#[trigger] core.frames_spec().frame(f)).tier
            == tier ==> first_frame.0 <= f.0 && f.0 < first_frame.0 + num_frames
}

/// Registered RDMA `MemoryRegion` bound to a contiguous slab of `CacheCore` frames.
pub struct RegisteredSlab<'a> {
    pub(crate) core_id: Option<u64>,
    pub(crate) mr: rdma::MemoryRegion<'a>,
    pub(crate) tier: TierId,
    pub(crate) first_frame: FrameId,
    pub(crate) num_frames: u32,
}

impl<'a> RegisteredSlab<'a> {
    pub open(crate) spec fn num_frames_spec(&self) -> u32 {
        self.num_frames
    }

    pub open(crate) spec fn first_frame_spec(&self) -> FrameId {
        self.first_frame
    }

    pub open(crate) spec fn tier_spec(&self) -> TierId {
        self.tier
    }

    pub open(crate) spec fn mr_spec(&self) -> rdma::MemoryRegion<'a> {
        self.mr
    }

    pub open(crate) spec fn well_formed(&self, core: &CacheCore, pd_id: nat) -> bool {
        &&& self.mr.well_formed()
        &&& self.mr.instance_id() == core.instance_id()
        &&& self.mr.node() == core.node()
        &&& self.mr.pd_id() == pd_id
        &&& self.mr.access().mw_bind
        &&& self.mr.num_slots() == self.num_frames as usize
        &&& self.mr.slot_len() == core.frames_spec().tier_frame_size(self.tier)
        &&& self.tier.valid(core.tiers_spec().len())
        &&& slab_bound_to_core(core, self.tier, self.first_frame, self.num_frames, self.mr.space())
    }

    pub open(crate) spec fn well_formed_shared(&self, core: &CacheCore, pd_id: nat) -> bool {
        &&& self.mr.well_formed()
        &&& self.mr.instance_id() == core.instance_id()
        &&& self.mr.node() == core.node()
        &&& self.mr.pd_id() == pd_id
        &&& self.mr.access().mw_bind
        &&& self.mr.num_slots() == self.num_frames as usize
        &&& self.mr.slot_len() == core.frames_spec().tier_frame_size(self.tier)
        &&& self.tier.valid(core.tiers_spec().len())
        &&& slab_bound_open(core, self.tier, self.first_frame, self.num_frames, self.mr.space())
    }

    pub proof fn lemma_shared_wf_iff(&self, core: &CacheCore, pd_id: nat)
        ensures
            self.well_formed_shared(core, pd_id) <==> self.well_formed(core, pd_id),
    {
        reveal(slab_bound_to_core);
    }

    pub open spec fn registration_valid<B: RdmaRegisterableSlab>(
        core: &CacheCore,
        pd: &rdma::ProtectionDomain<'_>,
        buffer: &B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> bool {
        &&& core.well_formed()
        &&& pd.node() == core.node()
        &&& flags.mw_bind
        &&& buffer.reg_well_formed()
        &&& buffer.reg_instance_id() == core.instance_id()
        &&& buffer.reg_node() == core.node()
        &&& buffer.reg_num_slots() == num_frames as usize
        &&& buffer.reg_slot_len() == core.frames_spec().tier_frame_size(tier)
        &&& tier.valid(core.tiers_spec().len())
        &&& slab_bound_to_core(core, tier, first_frame, num_frames, buffer.reg_space())
    }

    /// Register a slab for sharing across multiple QPs on `pd`.
    pub fn register_slab<B: RdmaRegisterableSlab>(
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<Self>)
        requires
            Self::registration_valid(core, pd, buffer, tier, first_frame, num_frames, flags),
        ensures
            r matches Ok(s) ==> {
                &&& s.well_formed(core, pd.id())
                &&& s.well_formed_shared(core, pd.id())
                &&& s.tier_spec() == tier
                &&& s.first_frame_spec() == first_frame
                &&& s.num_frames_spec() == num_frames
                &&& s.mr_spec().access() == flags
                &&& s.mr_spec().gpu_device_spec() == buffer.reg_gpu_device()
            },
    {
        if !flags.mw_bind || buffer.reg_num_slots_exec() != num_frames as usize
            || !core.matches_slab(
            tier,
            first_frame,
            num_frames,
            buffer.reg_slot_len_exec() as u64,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let mr = match rdma::register_memory(pd, buffer, flags) {
            Ok(mr) => mr,
            Err(e) => return Err(rdma_error(e)),
        };
        let slab = RegisteredSlab { core_id: core.runtime_id(), mr, tier, first_frame, num_frames };
        proof {
            slab.lemma_shared_wf_iff(core, pd.id());
        }
        Ok(slab)
    }

    /// Register a host slab for sharing across multiple QPs on `pd`.
    pub fn register_host<B: RdmaRegisterableSlab>(
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<Self>)
        requires
            Self::registration_valid(core, pd, buffer, tier, first_frame, num_frames, flags),
            buffer.reg_gpu_device() is None,
        ensures
            r matches Ok(s) ==> {
                &&& s.well_formed(core, pd.id())
                &&& s.well_formed_shared(core, pd.id())
                &&& s.tier_spec() == tier
                &&& s.first_frame_spec() == first_frame
                &&& s.num_frames_spec() == num_frames
                &&& s.mr_spec().access() == flags
                &&& s.mr_spec().gpu_device_spec() is None
            },
    {
        Self::register_slab(core, pd, buffer, tier, first_frame, num_frames, flags)
    }

    /// Register a gpu slab for sharing across multiple QPs on `pd`.
    pub fn register_gpu<B: RdmaRegisterableSlab>(
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<Self>)
        requires
            Self::registration_valid(core, pd, buffer, tier, first_frame, num_frames, flags),
            buffer.reg_gpu_device() is Some,
        ensures
            r matches Ok(s) ==> {
                &&& s.well_formed(core, pd.id())
                &&& s.well_formed_shared(core, pd.id())
                &&& s.tier_spec() == tier
                &&& s.first_frame_spec() == first_frame
                &&& s.num_frames_spec() == num_frames
                &&& s.mr_spec().access() == flags
                &&& s.mr_spec().gpu_device_spec() is Some
            },
    {
        Self::register_slab(core, pd, buffer, tier, first_frame, num_frames, flags)
    }

    pub open(crate) spec fn owns(&self, frame: FrameId) -> bool {
        self.first_frame.0 <= frame.0 && frame.0 < self.first_frame.0 + self.num_frames
    }

    pub fn owns_frame(&self, frame: FrameId) -> (r: bool)
        ensures
            r == self.owns(frame),
    {
        frame.0 >= self.first_frame.0 && frame.0 - self.first_frame.0 < self.num_frames
    }

    /// Query the GPUDirect RDMA write ordering mode (returns `None` for host DRAM MRs).
    pub(crate) fn gdr_ordering(&self) -> Option<rdma::RdmaGdrWritesOrdering> {
        self.mr.gdr_ordering()
    }

    /// Check whether GPUDirect RDMA writes to this slab are natively visible
    /// to `consumer_device` without a host-side flush.
    pub(crate) fn gdr_visible_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            self.mr.well_formed(),
        ensures
            r == self.mr.gdr_visible_to_device_spec(consumer_device),
    {
        self.mr.gdr_visible_to_device(consumer_device)
    }

    /// Check whether GPUDirect RDMA writes to this slab are visible to
    /// `consumer_device` after `self.flush_gdr_writes()`.
    pub(crate) fn gdr_visible_after_flush_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            self.mr.well_formed(),
        ensures
            r == self.mr.gdr_visible_after_flush_to_device_spec(consumer_device),
    {
        self.mr.gdr_visible_after_flush_to_device(consumer_device)
    }

    /// Issue a GPUDirect RDMA write flush targeting the slab's owning GPU derived
    /// from `self.mr.gpu_device()` when `RdmaGdrWritesOrdering::None`.
    pub(crate) fn flush_gdr_writes(&self) -> BackendResult<()>
        requires
            self.mr.well_formed(),
    {
        match self.mr.flush_gdr_writes() {
            Ok(()) => Ok(()),
            Err(e) => Err(CoreError::Backend { kind: BACKEND_CUDA, code: e.0 }),
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
pub open(crate) spec fn rdma_window_wf(
    ops: [Option<PendingOperation>; BITMAP_SLOTS],
    bitmap: SlidingBitmap,
    table: &TimelineTable,
    t: TimelineId,
    ledger: &KnowledgeLedger,
    cq: rdma::CqSnapshot,
    id: InstanceId,
    next_control_wr_id: u64,
) -> bool {
    &&& t < table.count_spec()
    &&& table.well_formed(id, cq.observer, ledger)
    &&& bitmap.well_formed()
    &&& bitmap.observations(table, t, cq.observer, ledger)
    &&& window_matches(ops, bitmap, table, t, rdma_data_events(cq))
    &&& CONTROL_WR_ID_BASE <= next_control_wr_id <= CONTROL_WR_ID_LIMIT
    &&& forall|s: u64|
        cq.pending.contains_key(s) && !((#[trigger] cq.pending[s]).kind is DataTransfer)
            ==> CONTROL_WR_ID_BASE <= s < next_control_wr_id
}

/// Staging invariant for reaped data CQ completions stored in `RdmaBackend::completed_pulls`.
pub open(crate) spec fn completed_pulls_wf(
    completed_pulls: [Option<(u64, Result<(), i32>)>; BITMAP_SLOTS],
    table: &TimelineTable,
    t: TimelineId,
    observer: NodeId,
    ledger: &KnowledgeLedger,
) -> bool {
    forall|i: int|
        0 <= i < BITMAP_SLOTS ==> match #[trigger] completed_pulls[i] {
            None => true,
            Some((seq, st)) => {
                &&& ((seq as int) % (BITMAP_CAPACITY as int)) == i
                &&& table.issued(t, seq)
                &&& ledger.knows.set().contains((observer, table.events_spec()@[(t, seq)]))
                &&& (st.is_ok() ==> ledger.succeeded.dom().contains(table.events_spec()@[(t, seq)]))
            },
        }
}

/// Invariant tying active control-WR leases (`Binding`, `Unbinding`, `Revoking`) to
/// pending entries in `cq` and guaranteeing distinct `wr_id`s across active control WRs.
#[verifier::opaque]
pub open(crate) spec fn rdma_control_leases_match_cq<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
    cq: rdma::CqSnapshot,
    next_control_wr_id: u64,
) -> bool {
    &&& forall|i: int|
        0 <= i < leases.len() && (#[trigger] leases[i]).has_active_wr() ==> {
            let l = leases[i];
            &&& (l.mw_slot as int) < pool.slots.len()
            &&& {
                let w = pool.slots[l.mw_slot as int].window;
                &&& CONTROL_WR_ID_BASE <= l.wr_id < next_control_wr_id
                &&& (l.state == RemoteLeaseState::Binding && cq.pending.contains_key(l.wr_id)) ==> {
                    &&& cq.pending[l.wr_id].kind is BindWindow
                    &&& (w matches Some(win) ==> cq.pending[l.wr_id].kind->BindWindow_window_id
                        == win.id())
                }
                &&& ((l.state == RemoteLeaseState::Unbinding || l.state
                    == RemoteLeaseState::Revoking) && cq.pending.contains_key(l.wr_id)) ==> {
                    &&& cq.pending[l.wr_id].kind is LocalInvalidate
                    &&& (w matches Some(win) ==> cq.pending[l.wr_id].kind->LocalInvalidate_window_id
                        == win.id())
                }
            }
        }
    &&& forall|i: int, j: int|
        0 <= i < j < leases.len() && (#[trigger] leases[i]).has_active_wr() && (
        #[trigger] leases[j]).has_active_wr() ==> leases[i].wr_id != leases[j].wr_id
}

/// Every checked-out MW slot (`Binding`, `Bound`, `Invalidating`) in `pool` is owned
/// by an entry in `leases` (so draining `leases` guarantees `pool` can `dealloc_all`).
#[verifier::opaque]
pub open(crate) spec fn rdma_pool_slots_owned_by_leases<'a>(
    leases: Seq<ActiveRemoteLease>,
    pool: &WindowPool<'a>,
) -> bool {
    forall|s: int|
        0 <= s < pool.slots.len() && ((#[trigger] pool.slots[s]).state is Binding
            || pool.slots[s].state is Bound || pool.slots[s].state is Invalidating) ==> exists|
            i: int,
        |
            0 <= i < leases.len() && (#[trigger] leases[i]).mw_slot as int == s
}

impl ActiveRemoteLease {
    /// True when the lease has already unpinned its frame in `CacheCore`.
    pub open spec fn is_unpinned(&self) -> bool {
        self.state == RemoteLeaseState::Unbinding || (self.state
            == RemoteLeaseState::ControlFailedState && self.returned_share is Some)
    }

    /// True when the lease currently owns an active control WR ID (`Binding`, `Unbinding`, `Revoking`).
    pub open spec fn has_active_wr(&self) -> bool {
        self.state == RemoteLeaseState::Binding || self.state == RemoteLeaseState::Unbinding
            || self.state == RemoteLeaseState::Revoking
    }

    pub open(crate) spec fn valid<'a>(
        &self,
        pool: &WindowPool<'a>,
        qp: &rdma::QueuePair<'a>,
        core: &CacheCore,
    ) -> bool {
        &&& self.frame.0 < core.frames_spec().metas_spec().len()
        &&& (self.mw_slot as int) < pool.slots.len()
        &&& pool.slots[self.mw_slot as int].generation == self.generation
        &&& !self.is_unpinned() ==> {
            &&& core.frames_spec().meta(self.frame).active_fenced_leases > 0
            &&& core.frames_spec().frame(self.frame).has_fenced_share(self.share@)
        }
        &&& match self.state {
            RemoteLeaseState::Binding => {
                &&& pool.slots[self.mw_slot as int].state is Binding
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Binding
                &&& w.state()->Binding_qp_id == qp.id()
                &&& w.state()->Binding_instance_id == core.instance_id()
                &&& w.state()->Binding_share == Some(self.share@)
                &&& w.state()->Binding_wr_id == self.wr_id
                &&& w.state()->Binding_target_slot == core.frames_spec().frame(self.frame).slot
                &&& w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                &&& self.share_cap matches Some(cap)
                &&& cap@.is_fenced()
                &&& cap@.instance_id() == core.instance_id()
                &&& cap@.loc() == core.frames_spec().frame(self.frame).slot.loc
                &&& cap@.share() == self.share@
                &&& cap@.frontier().readers.is_empty()
                &&& self.returned_share is None
                &&& self.bind_witness is None
                &&& self.bind_failed_witness is None
                &&& self.inv_witness is None
                &&& self.staged_error is None
            },
            RemoteLeaseState::BindReady => {
                &&& pool.slots[self.mw_slot as int].state is Binding
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Binding
                &&& w.state()->Binding_qp_id == qp.id()
                &&& w.state()->Binding_instance_id == core.instance_id()
                &&& w.state()->Binding_share == Some(self.share@)
                &&& w.state()->Binding_wr_id == self.wr_id
                &&& w.state()->Binding_target_slot == core.frames_spec().frame(self.frame).slot
                &&& w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                &&& self.share_cap matches Some(cap)
                &&& cap@.is_fenced()
                &&& cap@.instance_id() == core.instance_id()
                &&& cap@.loc() == core.frames_spec().frame(self.frame).slot.loc
                &&& cap@.share() == self.share@
                &&& cap@.frontier().readers.is_empty()
                &&& self.returned_share is None
                &&& self.bind_witness matches Some(bw) && bw@.matches(w)
                &&& self.bind_failed_witness is None
                &&& self.inv_witness is None
                &&& self.staged_error is None
            },
            RemoteLeaseState::BindFailedReady => {
                &&& pool.slots[self.mw_slot as int].state is Binding
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Binding
                &&& w.state()->Binding_qp_id == qp.id()
                &&& w.state()->Binding_instance_id == core.instance_id()
                &&& w.state()->Binding_share == Some(self.share@)
                &&& w.state()->Binding_wr_id == self.wr_id
                &&& w.state()->Binding_target_slot == core.frames_spec().frame(self.frame).slot
                &&& w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                &&& self.share_cap matches Some(cap)
                &&& cap@.is_fenced()
                &&& cap@.instance_id() == core.instance_id()
                &&& cap@.loc() == core.frames_spec().frame(self.frame).slot.loc
                &&& cap@.share() == self.share@
                &&& cap@.frontier().readers.is_empty()
                &&& self.returned_share is None
                &&& self.bind_witness is None
                &&& self.bind_failed_witness matches Some(bfw) && bfw@.matches(w)
                &&& self.inv_witness is None
                &&& self.staged_error is Some
            },
            RemoteLeaseState::Bound => {
                &&& pool.slots[self.mw_slot as int].state is Bound
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Bound
                &&& w.state()->Bound_qp_id == qp.id()
                &&& w.state()->Bound_instance_id == core.instance_id()
                &&& w.state()->Bound_share == Some(self.share@)
                &&& w.state()->Bound_slot == core.frames_spec().frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share is None
                &&& self.bind_witness is None
                &&& self.bind_failed_witness is None
                &&& self.inv_witness is None
                &&& self.staged_error is None
            },
            RemoteLeaseState::Unbinding => {
                &&& pool.slots[self.mw_slot as int].state is Invalidating
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Invalidating
                &&& w.state()->Invalidating_qp_id == qp.id()
                &&& w.state()->Invalidating_instance_id == core.instance_id()
                &&& w.state()->Invalidating_share == Some(self.share@)
                &&& w.state()->Invalidating_wr_id == self.wr_id
                &&& w.state()->Invalidating_slot == core.frames_spec().frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share matches Some(r)
                &&& r@.share() == self.share@
                &&& r@.instance_id() == core.instance_id()
                &&& r@.loc() == core.frames_spec().frame(self.frame).slot.loc
                &&& self.bind_witness is None
                &&& self.bind_failed_witness is None
                &&& self.inv_witness is None
                &&& self.staged_error is None
            },
            RemoteLeaseState::Revoking => {
                &&& pool.slots[self.mw_slot as int].state is Invalidating
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Invalidating
                &&& w.state()->Invalidating_qp_id == qp.id()
                &&& w.state()->Invalidating_instance_id == core.instance_id()
                &&& w.state()->Invalidating_share == Some(self.share@)
                &&& w.state()->Invalidating_wr_id == self.wr_id
                &&& w.state()->Invalidating_slot == core.frames_spec().frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share is None
                &&& self.bind_witness is None
                &&& self.bind_failed_witness is None
                &&& self.inv_witness is None
                &&& self.staged_error is None
            },
            RemoteLeaseState::RevokeInvReady => {
                &&& pool.slots[self.mw_slot as int].state is Invalidating
                &&& pool.slots[self.mw_slot as int].window matches Some(w)
                &&& w.state() is Invalidating
                &&& w.state()->Invalidating_qp_id == qp.id()
                &&& w.state()->Invalidating_instance_id == core.instance_id()
                &&& w.state()->Invalidating_share == Some(self.share@)
                &&& w.state()->Invalidating_wr_id == self.wr_id
                &&& w.state()->Invalidating_slot == core.frames_spec().frame(self.frame).slot
                &&& self.share_cap is None
                &&& self.returned_share is None
                &&& self.bind_witness is None
                &&& self.bind_failed_witness is None
                &&& self.inv_witness matches Some(iw) && iw@.matches(w)
                &&& self.staged_error is None
            },
            RemoteLeaseState::ControlFailedState => {
                &&& self.bind_witness is None
                &&& self.bind_failed_witness is None
                &&& self.inv_witness is None
                &&& self.staged_error is Some
                &&& match (self.share_cap, self.returned_share) {
                    (Some(cap), None) => {
                        &&& pool.slots[self.mw_slot as int].state is Binding
                        &&& pool.slots[self.mw_slot as int].window matches Some(w)
                        &&& w.state() is Binding
                        &&& w.state()->Binding_qp_id == qp.id()
                        &&& w.state()->Binding_instance_id == core.instance_id()
                        &&& w.state()->Binding_share == Some(self.share@)
                        &&& w.state()->Binding_wr_id == self.wr_id
                        &&& w.state()->Binding_target_slot == core.frames_spec().frame(
                            self.frame,
                        ).slot
                        &&& w.state()->Binding_access == rdma::RemoteAccess::ReadOnly
                        &&& cap@.is_fenced()
                        &&& cap@.instance_id() == core.instance_id()
                        &&& cap@.loc() == core.frames_spec().frame(self.frame).slot.loc
                        &&& cap@.share() == self.share@
                        &&& cap@.frontier().readers.is_empty()
                    },
                    (None, Some(r)) => {
                        &&& pool.slots[self.mw_slot as int].state is Invalidating
                        &&& pool.slots[self.mw_slot as int].window matches Some(w)
                        &&& w.state() is Invalidating
                        &&& w.state()->Invalidating_qp_id == qp.id()
                        &&& w.state()->Invalidating_instance_id == core.instance_id()
                        &&& w.state()->Invalidating_share == Some(self.share@)
                        &&& w.state()->Invalidating_wr_id == self.wr_id
                        &&& w.state()->Invalidating_slot == core.frames_spec().frame(
                            self.frame,
                        ).slot
                        &&& r@.share() == self.share@
                        &&& r@.instance_id() == core.instance_id()
                        &&& r@.loc() == core.frames_spec().frame(self.frame).slot.loc
                    },
                    (None, None) => {
                        &&& pool.slots[self.mw_slot as int].state is Invalidating
                        &&& pool.slots[self.mw_slot as int].window matches Some(w)
                        &&& w.state() is Invalidating
                        &&& w.state()->Invalidating_qp_id == qp.id()
                        &&& w.state()->Invalidating_instance_id == core.instance_id()
                        &&& w.state()->Invalidating_share == Some(self.share@)
                        &&& w.state()->Invalidating_wr_id == self.wr_id
                        &&& w.state()->Invalidating_slot == core.frames_spec().frame(
                            self.frame,
                        ).slot
                    },
                    (Some(_), Some(_)) => false,
                }
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

/// A fenced share on a frame has at most one pinned lease in this peer.
#[verifier::opaque]
pub open spec fn rdma_pinned_shares_disjoint(leases: Seq<ActiveRemoteLease>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < leases.len() && !(#[trigger] leases[i]).is_unpinned() && !(
        #[trigger] leases[j]).is_unpinned() && leases[i].frame == leases[j].frame
            ==> leases[i].share@ != leases[j].share@
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
    &&& rdma_pinned_shares_disjoint(leases)
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
    pub(crate) core_id: Option<u64>,
    pub(crate) timeline_id: TimelineId,
    pub(crate) peer_node: NodeId,
    pub(crate) qp: rdma::QueuePair<'a>,
    pub(crate) cq: rdma::CompletionQueue<'a>,
    pub(crate) bitmap: SlidingBitmap,
    pub(crate) in_flight_ops: [Option<PendingOperation>; BITMAP_SLOTS],
    pub(crate) window_pool: WindowPool<'a>,
    pub(crate) host_mr: Option<RegisteredSlab<'a>>,
    pub(crate) gpu_mr: Option<RegisteredSlab<'a>>,
    pub(crate) shared_host_mr: Option<&'a RegisteredSlab<'a>>,
    pub(crate) shared_gpu_mr: Option<&'a RegisteredSlab<'a>>,
    pub(crate) active_leases: Vec<ActiveRemoteLease>,
    pub(crate) next_control_wr_id: u64,
    pub(crate) completed_pulls: [Option<(u64, Result<(), i32>)>; BITMAP_SLOTS],
}

impl<'a> RdmaBackend<'a> {
    pub open(crate) spec fn completed_pulls_spec(&self) -> [Option<
        (u64, Result<(), i32>),
    >; BITMAP_SLOTS] {
        self.completed_pulls
    }

    pub open(crate) spec fn next_control_wr_id_spec(&self) -> u64 {
        self.next_control_wr_id
    }

    pub open(crate) spec fn in_flight_ops_spec(&self) -> [Option<PendingOperation>; BITMAP_SLOTS] {
        self.in_flight_ops
    }

    pub open(crate) spec fn bitmap_spec(&self) -> SlidingBitmap {
        self.bitmap
    }

    pub open(crate) spec fn shared_host_mr_spec(&self) -> Option<&'a RegisteredSlab<'a>> {
        self.shared_host_mr
    }

    pub open(crate) spec fn shared_gpu_mr_spec(&self) -> Option<&'a RegisteredSlab<'a>> {
        self.shared_gpu_mr
    }

    pub open(crate) spec fn timeline_id_spec(&self) -> TimelineId {
        self.timeline_id
    }

    pub open(crate) spec fn qp_spec(&self) -> rdma::QueuePair<'a> {
        self.qp
    }

    pub open(crate) spec fn active_leases_spec(&self) -> Vec<ActiveRemoteLease> {
        self.active_leases
    }

    pub open(crate) spec fn peer_node_spec(&self) -> NodeId {
        self.peer_node
    }

    pub open(crate) spec fn window_pool_spec(&self) -> WindowPool<'a> {
        self.window_pool
    }

    pub open(crate) spec fn cq_spec(&self) -> rdma::CompletionQueue<'a> {
        self.cq
    }

    pub open(crate) spec fn host_mr_spec(&self) -> Option<RegisteredSlab<'a>> {
        self.host_mr
    }

    pub open(crate) spec fn gpu_mr_spec(&self) -> Option<RegisteredSlab<'a>> {
        self.gpu_mr
    }

    pub open(crate) spec fn pending(&self) -> Map<u64, TransferEvent> {
        rdma_data_events(rdma::CqSnapshot::of(self.cq))
    }

    pub open(crate) spec fn effective_host_mr_spec(&self) -> Option<RegisteredSlab<'a>> {
        match self.host_mr {
            Some(s) => Some(s),
            None => match self.shared_host_mr {
                Some(s) => Some(*s),
                None => None,
            },
        }
    }

    pub open(crate) spec fn effective_gpu_mr_spec(&self) -> Option<RegisteredSlab<'a>> {
        match self.gpu_mr {
            Some(s) => Some(s),
            None => match self.shared_gpu_mr {
                Some(s) => Some(*s),
                None => None,
            },
        }
    }

    pub(crate) fn effective_host_mr(&self) -> (r: Option<&RegisteredSlab<'a>>)
        requires
            !(self.host_mr is Some && self.shared_host_mr is Some),
        ensures
            match r {
                None => self.effective_host_mr_spec() is None,
                Some(s) => self.effective_host_mr_spec() == Some(*s),
            },
    {
        match &self.host_mr {
            Some(s) => Some(s),
            None => match self.shared_host_mr {
                Some(s) => Some(s),
                None => None,
            },
        }
    }

    pub(crate) fn effective_gpu_mr(&self) -> (r: Option<&RegisteredSlab<'a>>)
        requires
            !(self.gpu_mr is Some && self.shared_gpu_mr is Some),
        ensures
            match r {
                None => self.effective_gpu_mr_spec() is None,
                Some(s) => self.effective_gpu_mr_spec() == Some(*s),
            },
    {
        match &self.gpu_mr {
            Some(s) => Some(s),
            None => match self.shared_gpu_mr {
                Some(s) => Some(s),
                None => None,
            },
        }
    }

    pub open(crate) spec fn well_formed(&self, core: &CacheCore) -> bool {
        let t = self.timeline_id;
        &&& t < core.timelines_spec().count_spec()
        &&& core.timelines_spec().kinds_spec()[t as int] == (TimelineKind::RdmaQueuePair {
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
        &&& !(self.host_mr is Some && self.shared_host_mr is Some)
        &&& !(self.gpu_mr is Some && self.shared_gpu_mr is Some)
        &&& match self.host_mr {
            None => true,
            Some(ref s) => s.well_formed(core, self.window_pool.pd()),
        }
        &&& match self.gpu_mr {
            None => true,
            Some(ref s) => s.well_formed(core, self.window_pool.pd()),
        }
        &&& match self.shared_host_mr {
            None => true,
            Some(s) => s.well_formed_shared(core, self.window_pool.pd()),
        }
        &&& match self.shared_gpu_mr {
            None => true,
            Some(s) => s.well_formed_shared(core, self.window_pool.pd()),
        }
        &&& rdma_window_wf(
            self.in_flight_ops,
            self.bitmap,
            &core.timelines_spec(),
            t,
            &core.ledger_spec()@,
            rdma::CqSnapshot::of(self.cq),
            core.instance_id(),
            self.next_control_wr_id,
        )
        &&& completed_pulls_wf(
            self.completed_pulls,
            &core.timelines_spec(),
            t,
            self.cq.observer(),
            &core.ledger_spec()@,
        )
        &&& rdma_leases_disjoint(self.active_leases@)
        &&& rdma_leases_valid(self.active_leases@, &self.window_pool, &self.qp, core)
        &&& rdma_control_leases_match_cq(
            self.active_leases@,
            &self.window_pool,
            rdma::CqSnapshot::of(self.cq),
            self.next_control_wr_id,
        )
        &&& rdma_pool_slots_owned_by_leases(self.active_leases@, &self.window_pool)
    }

    pub(crate) proof fn lemma_effective_slabs_wf(&self, core: &CacheCore)
        requires
            self.well_formed(core),
        ensures
            self.effective_host_mr_spec() matches Some(s) ==> s.well_formed(
                core,
                self.window_pool.pd(),
            ),
            self.effective_gpu_mr_spec() matches Some(s) ==> s.well_formed(
                core,
                self.window_pool.pd(),
            ),
    {
        if let Some(s) = self.shared_host_mr {
            s.lemma_shared_wf_iff(core, self.window_pool.pd());
        }
        if let Some(s) = self.shared_gpu_mr {
            s.lemma_shared_wf_iff(core, self.window_pool.pd());
        }
    }

    pub open(crate) spec fn completed_pull_spec(&self, seq: u64) -> Option<Result<(), i32>> {
        let slot = ((seq as int) % (BITMAP_CAPACITY as int));
        match self.completed_pulls[slot] {
            Some((s, st)) => if s == seq {
                Some(st)
            } else {
                None
            },
            None => None,
        }
    }

    pub fn has_completed_pull(&self, seq: u64) -> (r: bool)
        ensures
            r == self.completed_pull_spec(seq) is Some,
    {
        let slot = (seq % BITMAP_CAPACITY) as usize;
        match self.completed_pulls[slot] {
            Some((s, _)) => s == seq,
            None => false,
        }
    }

    pub(crate) fn take_completed_pull(&mut self, core: &CacheCore, seq: u64) -> (r: Option<
        Result<(), i32>,
    >)
        requires
            core.well_formed(),
            old(self).well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).timeline_id == old(self).timeline_id,
            final(self).peer_node == old(self).peer_node,
            final(self).qp == old(self).qp,
            final(self).cq == old(self).cq,
            final(self).bitmap == old(self).bitmap,
            final(self).in_flight_ops == old(self).in_flight_ops,
            final(self).window_pool == old(self).window_pool,
            final(self).host_mr == old(self).host_mr,
            final(self).gpu_mr == old(self).gpu_mr,
            final(self).shared_host_mr == old(self).shared_host_mr,
            final(self).shared_gpu_mr == old(self).shared_gpu_mr,
            final(self).active_leases == old(self).active_leases,
            final(self).next_control_wr_id == old(self).next_control_wr_id,
            r == old(self).completed_pull_spec(seq),
            final(self).completed_pull_spec(seq) is None,
            forall|w: u64|
                w != seq ==> (#[trigger] final(self).completed_pull_spec(w)) == old(
                    self,
                ).completed_pull_spec(w),
            r matches Some(st) ==> {
                &&& core.timelines_spec().issued(old(self).timeline_id, seq)
                &&& core.ledger_spec()@.knows.set().contains(
                    (
                        core.node(),
                        core.timelines_spec().events_spec()@[(old(self).timeline_id, seq)],
                    ),
                )
                &&& (st.is_ok() ==> core.ledger_spec()@.succeeded.dom().contains(
                    core.timelines_spec().events_spec()@[(old(self).timeline_id, seq)],
                ))
            },
    {
        let _ = core;
        let slot = (seq % BITMAP_CAPACITY) as usize;
        let entry = self.completed_pulls[slot];
        match entry {
            Some((s, st)) => {
                if s == seq {
                    self.completed_pulls[slot] = None;
                    Some(st)
                } else {
                    None
                }
            },
            None => None,
        }
    }

    pub fn window_free_count(&self) -> (n: usize)
        ensures
            n == self.window_pool_spec().free_spec()@.len(),
    {
        self.window_pool.free_count()
    }

    pub fn protection_domain(&self) -> (pd: &'a rdma::ProtectionDomain<'a>)
        ensures
            pd.id() == self.qp_spec().pd_id(),
    {
        self.qp.protection_domain()
    }

    /// Recover transport resources from a detached backend with an empty CQ.
    pub fn into_idle_parts(self) -> Option<
        (rdma::QueuePair<'a>, rdma::CompletionQueue<'a>, WindowPool<'a>),
    >
        requires
            self.cq_spec().well_formed(),
    {
        if !self.active_leases.is_empty() || self.cq.in_flight_count() != 0 {
            return None;
        }
        Some((self.qp, self.cq, self.window_pool))
    }

    pub fn timeline_id(&self) -> (t: TimelineId)
        ensures
            t == self.timeline_id_spec(),
    {
        self.timeline_id
    }

    pub fn peer_node(&self) -> (p: NodeId)
        ensures
            p == self.peer_node_spec(),
    {
        self.peer_node
    }

    pub fn active_lease_count(&self) -> (n: usize)
        ensures
            n == self.active_leases_spec()@.len(),
    {
        self.active_leases.len()
    }

    /// Spec-level index of the active lease matching `(handle.mw_slot, handle.generation)`.
    pub open(crate) spec fn lease_index_spec(&self, handle: RemoteLeaseHandle) -> Option<int> {
        rdma_lease_index_spec(self.active_leases@, handle)
    }

    /// Spec helper returning the `MemoryWindow` referenced by `handle` when its
    /// lease is currently in `state`.
    pub open(crate) spec fn lease_matches_window_state(
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
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec() == old(core).frames_spec(),
            final(core).catalog_spec() == old(core).catalog_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            match r {
                Err((_, ret_qp, ret_cq, ret_windows)) => *final(core) == *old(core) && ret_qp == qp
                    && ret_cq == cq && ret_windows == windows,
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.timeline_id_spec() == old(core).timelines_spec().count_spec()
                    &&& b.peer_node_spec() == peer_node
                    &&& b.qp_spec() == qp
                    &&& b.cq_spec() == cq
                    &&& b.host_mr_spec() is None
                    &&& b.gpu_mr_spec() is None
                    &&& b.shared_host_mr_spec() is None
                    &&& b.shared_gpu_mr_spec() is None
                    &&& forall|seq: u64| #[trigger] b.completed_pull_spec(seq) is None
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        if peer_node != qp.peer_node() || !qp.is_ready_to_send() || !qp.uses_completion_queue(&cq)
            || cq.in_flight_count() != 0 || windows.len() > u32::MAX as usize {
            return Err((CoreError::InvalidArgument, qp, cq, windows));
        }
        let domain = qp.protection_domain().runtime_id();
        let mut i = 0;
        while i < windows.len()
            invariant
                i <= windows.len(),
                *core == *old(core),
                core.well_formed(),
            decreases windows.len() - i,
        {
            if windows[i].protection_domain().runtime_id() != domain {
                return Err((CoreError::InvalidArgument, qp, cq, windows));
            }
            i += 1;
        }
        let timeline_id = match core.register_timeline(TimelineKind::RdmaQueuePair { peer_node }) {
            Ok(t) => t,
            Err(e) => return Err((e, qp, cq, windows)),
        };
        let ghost pd_id = qp.pd_id();
        let ghost node = core.node();
        let window_pool = WindowPool::new(windows, Ghost(pd_id), Ghost(node));
        let backend = Self {
            core_id: core.runtime_id(),
            timeline_id,
            peer_node,
            qp,
            cq,
            bitmap: SlidingBitmap::new(),
            in_flight_ops: [None;BITMAP_SLOTS],
            window_pool,
            host_mr: None,
            gpu_mr: None,
            shared_host_mr: None,
            shared_gpu_mr: None,
            active_leases: Vec::new(),
            next_control_wr_id: CONTROL_WR_ID_BASE,
            completed_pulls: [None;BITMAP_SLOTS],
        };
        proof {
            lemma_core_wf_facts(core);
            lemma_rdma_init_window_wf(
                backend.in_flight_ops,
                backend.bitmap,
                &core.timelines_spec(),
                timeline_id,
                &core.ledger_spec()@,
                rdma::CqSnapshot::of(backend.cq),
                core.instance_id(),
            );
            lemma_rdma_leases_empty(
                backend.active_leases@,
                &backend.window_pool,
                &backend.qp,
                core,
            );
            assert(rdma_leases_disjoint(backend.active_leases@)) by {
                reveal(rdma_leases_disjoint);
            }
            assert(completed_pulls_wf(
                backend.completed_pulls,
                &core.timelines_spec(),
                timeline_id,
                backend.cq.observer(),
                &core.ledger_spec()@,
            ));
            assert forall|seq: u64| #[trigger] backend.completed_pull_spec(seq) is None by {
                let slot = ((seq as int) % (BITMAP_CAPACITY as int));
                assert(0 <= slot < BITMAP_SLOTS);
                assert(backend.completed_pulls[slot] is None);
            }
            assert(rdma_control_leases_match_cq(
                backend.active_leases@,
                &backend.window_pool,
                rdma::CqSnapshot::of(backend.cq),
                backend.next_control_wr_id,
            )) by {
                reveal(rdma_control_leases_match_cq);
            }
            assert(rdma_pool_slots_owned_by_leases(backend.active_leases@, &backend.window_pool))
                by {
                reveal(rdma_pool_slots_owned_by_leases);
            }
        }
        Ok(backend)
    }

    /// Register and attach an owned MR in the host or device slot, according to the slab.
    pub fn attach_slab<B: RdmaRegisterableSlab>(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(core),
            pd.id() == old(self).window_pool_spec().pd(),
            RegisteredSlab::registration_valid(
                core,
                pd,
                buffer,
                tier,
                first_frame,
                num_frames,
                flags,
            ),
            if buffer.reg_gpu_device().is_none() {
                old(self).host_mr_spec() is None
            } else {
                old(self).gpu_mr_spec() is None
            },
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> if buffer.reg_gpu_device().is_none() {
                &&& final(self).host_mr_spec() matches Some(s)
                    && s.mr_spec().gpu_device_spec() is None
                &&& final(self).shared_host_mr_spec() is None
            } else {
                &&& final(self).gpu_mr_spec() matches Some(s) && s.mr_spec().gpu_device_spec()
                    == buffer.reg_gpu_device()
                &&& final(self).shared_gpu_mr_spec() is None
            },
            r.is_err() ==> *final(self) == *old(self),
    {
        let is_gpu = buffer.reg_gpu_device_exec().is_some();
        let already_attached = if is_gpu {
            self.gpu_mr.is_some() || self.shared_gpu_mr.is_some()
        } else {
            self.host_mr.is_some() || self.shared_host_mr.is_some()
        };
        if already_attached {
            return Err(CoreError::InvalidArgument);
        }
        if self.core_id != core.runtime_id() || self.qp.protection_domain().runtime_id()
            != pd.runtime_id() {
            return Err(CoreError::InvalidArgument);
        }
        let slab = RegisteredSlab::register_slab(
            core,
            pd,
            buffer,
            tier,
            first_frame,
            num_frames,
            flags,
        )?;
        if is_gpu {
            self.gpu_mr = Some(slab);
        } else {
            self.host_mr = Some(slab);
        }
        Ok(())
    }

    /// Attach an owned host slab MR aligned to a `CacheCore` tier.
    pub fn attach_host_slab<B: RdmaRegisterableSlab>(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(core),
            pd.id() == old(self).window_pool_spec().pd(),
            RegisteredSlab::registration_valid(
                core,
                pd,
                buffer,
                tier,
                first_frame,
                num_frames,
                flags,
            ),
            old(self).host_mr_spec() is None,
            buffer.reg_gpu_device() is None,
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> {
                &&& final(self).host_mr_spec() matches Some(s)
                    && s.mr_spec().gpu_device_spec() is None
                &&& final(self).shared_host_mr_spec() is None
            },
            r.is_err() ==> *final(self) == *old(self),
    {
        self.attach_slab(core, pd, buffer, tier, first_frame, num_frames, flags)
    }

    /// Register and attach an owned host slab MR.
    pub fn register_host_slab<B: RdmaRegisterableSlab>(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(core),
            pd.id() == old(self).window_pool_spec().pd(),
            RegisteredSlab::registration_valid(
                core,
                pd,
                buffer,
                tier,
                first_frame,
                num_frames,
                flags,
            ),
            old(self).host_mr_spec() is None,
            buffer.reg_gpu_device() is None,
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> {
                &&& final(self).host_mr_spec() matches Some(s)
                    && s.mr_spec().gpu_device_spec() is None
                &&& final(self).shared_host_mr_spec() is None
            },
            r.is_err() ==> *final(self) == *old(self),
    {
        self.attach_host_slab(core, pd, buffer, tier, first_frame, num_frames, flags)
    }

    /// Attach an owned gpu slab MR aligned to a `CacheCore` tier.
    pub fn attach_gpu_slab<B: RdmaRegisterableSlab>(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(core),
            pd.id() == old(self).window_pool_spec().pd(),
            RegisteredSlab::registration_valid(
                core,
                pd,
                buffer,
                tier,
                first_frame,
                num_frames,
                flags,
            ),
            old(self).gpu_mr_spec() is None,
            buffer.reg_gpu_device() is Some,
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> {
                &&& final(self).gpu_mr_spec() matches Some(s)
                    && s.mr_spec().gpu_device_spec() is Some
                &&& final(self).shared_gpu_mr_spec() is None
            },
            r.is_err() ==> *final(self) == *old(self),
    {
        self.attach_slab(core, pd, buffer, tier, first_frame, num_frames, flags)
    }

    /// Register and attach an owned gpu slab MR.
    pub fn register_gpu_slab<B: RdmaRegisterableSlab>(
        &mut self,
        core: &CacheCore,
        pd: &'a rdma::ProtectionDomain<'a>,
        buffer: &'a B,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
        flags: rdma::AccessFlags,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(core),
            pd.id() == old(self).window_pool_spec().pd(),
            RegisteredSlab::registration_valid(
                core,
                pd,
                buffer,
                tier,
                first_frame,
                num_frames,
                flags,
            ),
            old(self).gpu_mr_spec() is None,
            buffer.reg_gpu_device() is Some,
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> {
                &&& final(self).gpu_mr_spec() matches Some(s)
                    && s.mr_spec().gpu_device_spec() is Some
                &&& final(self).shared_gpu_mr_spec() is None
            },
            r.is_err() ==> *final(self) == *old(self),
    {
        self.attach_gpu_slab(core, pd, buffer, tier, first_frame, num_frames, flags)
    }

    /// Attach a shared read-only-reference host DRAM slab MR registered on the shard's PD.
    pub fn attach_shared_host_slab(&mut self, core: &CacheCore, slab: &'a RegisteredSlab<'a>) -> (r:
        BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).host_mr_spec() is None,
            old(self).shared_host_mr_spec() is None,
            slab.well_formed(core, old(self).window_pool_spec().pd()),
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> final(self).shared_host_mr_spec() == Some(slab)
                && final(self).effective_host_mr_spec() == Some(*slab),
    {
        if self.host_mr.is_some() || self.shared_host_mr.is_some() {
            return Err(CoreError::InvalidArgument);
        }
        if self.core_id != core.runtime_id() || slab.core_id != core.runtime_id()
            || self.qp.protection_domain().runtime_id() != slab.mr.protection_domain().runtime_id()
            || slab.mr.gpu_device().is_some() {
            return Err(CoreError::InvalidArgument);
        }
        proof {
            slab.lemma_shared_wf_iff(core, self.window_pool.pd());
        }
        self.shared_host_mr = Some(slab);
        Ok(())
    }

    /// Attach a shared GPU VRAM slab MR registered on the shard's PD.
    pub fn attach_shared_gpu_slab(&mut self, core: &CacheCore, slab: &'a RegisteredSlab<'a>) -> (r:
        BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).gpu_mr_spec() is None,
            old(self).shared_gpu_mr_spec() is None,
            slab.well_formed(core, old(self).window_pool_spec().pd()),
        ensures
            final(self).well_formed(core),
            r.is_ok() ==> final(self).shared_gpu_mr_spec() == Some(slab)
                && final(self).effective_gpu_mr_spec() == Some(*slab),
    {
        if self.gpu_mr.is_some() || self.shared_gpu_mr.is_some() {
            return Err(CoreError::InvalidArgument);
        }
        if self.core_id != core.runtime_id() || slab.core_id != core.runtime_id()
            || self.qp.protection_domain().runtime_id() != slab.mr.protection_domain().runtime_id()
            || slab.mr.gpu_device().is_none() {
            return Err(CoreError::InvalidArgument);
        }
        proof {
            slab.lemma_shared_wf_iff(core, self.window_pool.pd());
        }
        self.shared_gpu_mr = Some(slab);
        Ok(())
    }

    pub(crate) fn has_lease(&self, handle: RemoteLeaseHandle) -> (r: bool)
        requires
            rdma_leases_disjoint(self.active_leases@),
        ensures
            r == self.lease_index_spec(handle) is Some,
    {
        self.find_lease_index(handle).is_some()
    }

    pub(crate) fn lease_state(&self, handle: RemoteLeaseHandle) -> (r: Option<RemoteLeaseState>)
        requires
            rdma_leases_disjoint(self.active_leases@),
        ensures
            match self.lease_index_spec(handle) {
                None => r is None,
                Some(idx) => r == Some(self.active_leases@[idx].state),
            },
    {
        match self.find_lease_index(handle) {
            Some(idx) => Some(self.active_leases[idx].state),
            None => None,
        }
    }

    pub(crate) fn lease_staged_error(&self, handle: RemoteLeaseHandle) -> (r: Option<i32>)
        requires
            rdma_leases_disjoint(self.active_leases@),
        ensures
            match self.lease_index_spec(handle) {
                None => r is None,
                Some(idx) => r == self.active_leases@[idx].staged_error,
            },
    {
        match self.find_lease_index(handle) {
            Some(idx) => self.active_leases[idx].staged_error,
            None => None,
        }
    }

    /// Query GPUDirect RDMA write ordering on the attached GPU MR, if present.
    pub(crate) fn gpu_gdr_ordering(&self) -> Option<rdma::RdmaGdrWritesOrdering>
        requires
            !(self.gpu_mr is Some && self.shared_gpu_mr is Some),
    {
        match self.effective_gpu_mr() {
            Some(slab) => slab.gdr_ordering(),
            None => None,
        }
    }

    /// Check if GPUDirect RDMA writes to the attached GPU MR are natively visible
    /// to `consumer_device` without a host-side flush.
    pub(crate) fn gdr_visible_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            !(self.gpu_mr is Some && self.shared_gpu_mr is Some),
            self.effective_gpu_mr_spec() matches Some(s) ==> s.mr.well_formed(),
        ensures
            self.effective_gpu_mr_spec() matches Some(s) ==> r == s.mr.gdr_visible_to_device_spec(
                consumer_device,
            ),
            self.effective_gpu_mr_spec() is None ==> r,
    {
        match self.effective_gpu_mr() {
            Some(slab) => slab.gdr_visible_to_device(consumer_device),
            None => true,
        }
    }

    /// Check if GPUDirect RDMA writes to the attached GPU MR are visible to
    /// `consumer_device` after `self.flush_gdr_writes()`.
    pub(crate) fn gdr_visible_after_flush_to_device(&self, consumer_device: DeviceId) -> (r: bool)
        requires
            !(self.gpu_mr is Some && self.shared_gpu_mr is Some),
            self.effective_gpu_mr_spec() matches Some(s) ==> s.mr.well_formed(),
        ensures
            self.effective_gpu_mr_spec() matches Some(s) ==> r
                == s.mr.gdr_visible_after_flush_to_device_spec(consumer_device),
            self.effective_gpu_mr_spec() is None ==> r,
    {
        match self.effective_gpu_mr() {
            Some(slab) => slab.gdr_visible_after_flush_to_device(consumer_device),
            None => true,
        }
    }

    /// Flush GPUDirect RDMA writes to the owning GPU derived from `gpu_mr` when required.
    pub(crate) fn flush_gdr_writes(&self) -> BackendResult<()>
        requires
            !(self.gpu_mr is Some && self.shared_gpu_mr is Some),
            self.effective_gpu_mr_spec() matches Some(s) ==> s.mr.well_formed(),
    {
        match self.effective_gpu_mr() {
            Some(slab) => slab.flush_gdr_writes(),
            None => Ok(()),
        }
    }
}

} // verus!
