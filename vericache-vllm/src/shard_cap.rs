use std::sync::{Mutex, OnceLock};
use transfer::context::{
    AllocSource, EventSourceAllocator, EventSourceToken, NodeContext, ShardAllocator, ShardSources,
    ShareSource, TransferInstance,
};
use vericache_core::kernel::CacheCore;
use vstd::prelude::{Tracked, verus};

#[cfg(verus_only)]
use transfer::context::EventSourceKind;

const SHARD_VERSION_QUOTA: u64 = 1 << 32;
static ALLOCATOR: OnceLock<Option<Mutex<ShardAllocator>>> = OnceLock::new();

fn take_sources() -> Option<ShardSources> {
    ALLOCATOR
        .get_or_init(|| {
            NodeContext::try_bootstrap(0)
                .map(ShardAllocator::from_node_context)
                .map(Mutex::new)
        })
        .as_ref()?
        .lock()
        .ok()?
        .split_shard(SHARD_VERSION_QUOTA)
}

verus! {

/// The local-engine subset of the bench bundle, carved from the process-wide allocator.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub(crate) struct ShardCapabilityBundle {
    node: u64,
    pub instance: Tracked<TransferInstance>,
    pub alloc: Tracked<AllocSource>,
    pub shares: Tracked<ShareSource>,
    events: Tracked<EventSourceAllocator>,
    version_start: u64,
    version_limit: u64,
}

impl ShardCapabilityBundle {
    pub fn take() -> Option<Self> {
        let shard = take_sources()?;
        Some(Self {
            node: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            version_start: shard.version_start,
            version_limit: shard.version_limit,
        })
    }

    pub fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::Host))
    }

    pub fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::IoUring))
    }

    pub fn make_local_core(&self, frame_size: usize) -> CacheCore {
        CacheCore::new_with_version_range(
            self.node as u16,
            frame_size,
            Vec::new(),
            64,
            self.version_start,
            self.version_limit,
            Tracked(self.instance.borrow()),
        )
    }
}

} // verus!
