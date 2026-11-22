//! Cluster provisioning supplies the local partition of one shared history.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

static STARTED: AtomicBool = AtomicBool::new(false);

verus! {

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
