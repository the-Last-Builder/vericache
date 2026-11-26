//! Cluster provisioning supplies the local partition of one shared history.
use super::*;
use crate::memory::DmaAllocation;
use foundation::space::{Access, Effect};
use std::sync::atomic::{AtomicBool, Ordering};

static STARTED: AtomicBool = AtomicBool::new(false);

verus! {

impl DmaAllocation {
    /// Record the range returned by the allocation shim. The driver owns the
    /// allocation and must keep it live for the lifetime of any borrow.
    ///
    /// # Safety
    /// `ptr` must cover `len` live bytes in the indicated CPU or device address
    /// space. Only the owner may create this record, and reclaim must consume it.
    pub unsafe fn from_live_allocation(ptr: *mut u8, len: usize, device: Option<DeviceId>) -> (a:
        Self)
        requires
            ptr.addr() != 0,
            ptr.addr() + len <= usize::MAX,
        ensures
            a.well_formed(),
            a.address() == ptr.addr(),
            a.total_len() == len,
            a.device_spec() == device,
    {
        Self::new(ptr, len, device)
    }
}

impl CompletedAccess {
    /// # Safety
    /// The operation in `access` must have finished with the stated effect.
    pub unsafe fn from_completed_access(
        access: Ghost<Access<TransferDomain>>,
        effect: Ghost<Effect<TransferDomain>>,
    ) -> (r: Tracked<Self>)
        ensures
            r@.access() == access@,
            r@.effect() == effect@,
    {
        Self::new(access, effect)
    }
}

impl ObservedEvents {
    /// # Safety
    /// The driver must have observed completion of every event in `completed`
    /// and confirmed success of every event in `successful`.
    pub unsafe fn from_completion(
        id: Ghost<InstanceId>,
        observer: Ghost<NodeId>,
        completed: Ghost<ISet<TransferEvent>>,
        successful: Ghost<ISet<TransferEvent>>,
    ) -> (r: Tracked<Self>)
        ensures
            r@.observed(id@, observer@, completed@),
            forall|e: TransferEvent| #[trigger] r@.succeeded(id@, e) == successful@.contains(e),
    {
        Self::new(id, observer, completed, successful)
    }
}

impl CompletedFence {
    /// # Safety
    /// The window backing this read share must be fenced and deallocated.
    pub unsafe fn from_fence(
        id: Ghost<InstanceId>,
        loc: Ghost<SlotLoc>,
        share: Ghost<TransferShare>,
    ) -> (r: Tracked<Self>)
        ensures
            r@.matches(id@, loc@, share@),
    {
        Self::new(id, loc, share)
    }
}

impl NodeContext {
    /// Non-panicking variant of `bootstrap`: returns `None` if this process has
    /// already been bootstrapped.
    #[verifier::external_body]
    pub fn try_bootstrap(node: NodeId) -> (r: Option<Self>)
        ensures
            r matches Some(context) ==> context.node == node && context.initial_partition(),
    {
        if STARTED.swap(true, Ordering::Relaxed) {
            return None;
        }
        Some(
            Self {
                node,
                instance: Tracked(arbitrary()),
                unused: Tracked(arbitrary()),
                unused_shares: Tracked(arbitrary()),
                local_regions: Tracked(arbitrary()),
                next_alloc_id: Ghost(0nat),
                next_source_id: Ghost(0nat),
                version_instance: Tracked(arbitrary()),
                unused_versions: Tracked(arbitrary()),
            },
        )
    }

    /// The deployment must assign this process a cluster-unique node identifier.
    /// The process-wide guard enforces one bootstrap across all caller threads.
    #[verifier::external_body]
    pub fn bootstrap(node: NodeId) -> (context: Self)
        ensures
            context.node == node,
            context.initial_partition(),
    {
        assert!(!STARTED.swap(true, Ordering::Relaxed), "node already bootstrapped");
        Self {
            node,
            instance: Tracked(arbitrary()),
            unused: Tracked(arbitrary()),
            unused_shares: Tracked(arbitrary()),
            local_regions: Tracked(arbitrary()),
            next_alloc_id: Ghost(0nat),
            next_source_id: Ghost(0nat),
            version_instance: Tracked(arbitrary()),
            unused_versions: Tracked(arbitrary()),
        }
    }
}

} // verus!
