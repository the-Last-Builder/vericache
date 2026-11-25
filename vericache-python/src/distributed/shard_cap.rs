//! Per-shard linear capability allocation for `DistributedVllmEngine`.
//!
//! Carves per-shard `ShardSources` from the one-shot `NodeContext::try_bootstrap(node_id)`
//! with zero `assume_new` and `max_leases = 16_384` per shard.

use std::sync::{Mutex, OnceLock};
#[cfg(verus_only)]
use transfer::context::EventSourceKind;
use transfer::context::{
    AllocSource, EventSourceAllocator, EventSourceToken, NodeContext, ShardAllocator, ShardSources,
    ShareSource, TransferInstance, VersionInstance, VersionSource,
};
use vericache_core::kernel::CacheCore;
use vstd::prelude::{Tracked, verus};

const SHARD_VERSION_QUOTA: u64 = 1u64 << 40;
const DIST_MAX_LEASES_PER_SHARD: usize = 16_384;

static DIST_ALLOCATOR: OnceLock<Mutex<(u64, ShardAllocator)>> = OnceLock::new();

fn take_shard_sources(node_id: u64) -> Result<ShardSources, String> {
    let lock = DIST_ALLOCATOR.get_or_init(|| {
        let ctx = NodeContext::try_bootstrap(node_id)
            .expect("NodeContext::try_bootstrap once per OS process");
        Mutex::new((node_id, ShardAllocator::from_node_context(ctx)))
    });
    let mut guard = lock
        .lock()
        .map_err(|_| "distributed ShardAllocator mutex poisoned".to_string())?;
    if guard.0 != node_id {
        return Err(format!(
            "process already bootstrapped for node_id {}, cannot switch to {}",
            guard.0, node_id
        ));
    }
    guard
        .1
        .split_shard(SHARD_VERSION_QUOTA)
        .ok_or_else(|| "version quota exhausted in distributed ShardAllocator".to_string())
}

fn take_current_shard_sources() -> Option<ShardSources> {
    DIST_ALLOCATOR
        .get()?
        .lock()
        .ok()?
        .1
        .split_shard(SHARD_VERSION_QUOTA)
}

verus! {

#[cfg_attr(not(verus_only), allow(dead_code))]
pub(crate) struct DistShardCaps {
    pub(crate) node: u64,
    pub(crate) instance: Tracked<TransferInstance>,
    pub(crate) alloc: Tracked<AllocSource>,
    pub(crate) events: Tracked<EventSourceAllocator>,
    pub(crate) shares: Tracked<ShareSource>,
    pub(crate) version_instance: Tracked<VersionInstance>,
    pub(crate) versions: Option<Tracked<VersionSource>>,
    pub(crate) version_start: u64,
    pub(crate) version_limit: u64,
}

impl DistShardCaps {
    pub(crate) fn take(node_id: u64) -> Result<Self, String> {
        let shard = take_shard_sources(node_id)?;
        Ok(Self {
            node: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            version_instance: shard.version_instance,
            versions: Some(shard.versions),
            version_start: shard.version_start,
            version_limit: shard.version_limit,
        })
    }

    pub(crate) fn take_current() -> Option<Self> {
        let shard = take_current_shard_sources()?;
        Some(Self {
            node: shard.node,
            instance: shard.instance,
            alloc: shard.alloc,
            events: shard.events,
            shares: shard.shares,
            version_instance: shard.version_instance,
            versions: Some(shard.versions),
            version_start: shard.version_start,
            version_limit: shard.version_limit,
        })
    }

    pub(crate) fn mint_host(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::Host))
    }

    pub(crate) fn mint_nvme(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::IoUring))
    }

    pub(crate) fn mint_rdma(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.events.borrow_mut().mint(EventSourceKind::RdmaQueuePair))
    }

    pub(crate) fn make_core(&mut self, frame_size: usize) -> Result<CacheCore, String> {
        let ver_src = self
            .versions
            .take()
            .ok_or_else(|| "shard VersionSource already consumed".to_string())?;
        Ok(CacheCore::from_shard_versions(
            self.node as u16,
            frame_size,
            Vec::new(),
            DIST_MAX_LEASES_PER_SHARD,
            self.version_start,
            self.version_limit,
            Tracked(self.instance.borrow()),
            Tracked(self.version_instance.borrow()),
            ver_src,
        ))
    }
}

} // verus!
