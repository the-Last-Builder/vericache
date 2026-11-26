#[cfg(feature = "rdma")]
use super::DistributedEngine;
use super::LocalEngine;
use crate::backends::host::HostBackend;
#[cfg(all(feature = "rdma", verus_only))]
use crate::backends::lemma_core_wf_facts;
#[cfg(feature = "rdma")]
use crate::backends::DeviceAttachment;
#[cfg(verus_only)]
use crate::backends::{alloc_fresh_for_core, source_fresh_for_ledger};
use crate::backends::{BackendResult, NoDevice};
#[cfg(verus_only)]
use crate::frame::STATE_DRAINING;
use crate::frame::STATE_EMPTY;
use crate::kernel::CacheCore;
use crate::spec::{CoreError, ObjectKey, Ticket};
use transfer_core::context::*;
#[cfg(verus_only)]
use transfer_core::domain::EventSourceKind;
#[cfg(feature = "rdma")]
use transfer_core::transport::ByteTransport;
use vstd::prelude::*;

verus! {

const FRAME_SIZE: usize = 4096;

/// The distributed engine starts with an empty peer set.
#[cfg(feature = "rdma")]
struct ClosedTransport;

#[cfg(feature = "rdma")]
impl ByteTransport for ClosedTransport {
    type Error = ();

    fn send_frame(&mut self, _frame: &[u8]) -> Result<(), Self::Error> {
        Err(())
    }

    fn recv_frame(&mut self) -> Result<Vec<u8>, Self::Error> {
        Err(())
    }
}

fn attach_host(
    core: &mut CacheCore,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(alloc): Tracked<&mut AllocSource>,
    Tracked(events): Tracked<&mut EventSourceAllocator>,
) -> (r: BackendResult<(HostBackend, SyncContext)>)
    requires
        old(core).well_formed(),
        instance.id() == old(core).instance_id(),
        old(alloc).well_formed(),
        old(alloc).id() == instance.id(),
        old(alloc).node() == old(core).node(),
        alloc_fresh_for_core(*old(alloc), old(core)),
        old(events).well_formed(),
        old(events).id() == instance.id(),
        old(events).node() == old(core).node(),
    ensures
        final(core).well_formed(),
        final(core).instance_id() == old(core).instance_id(),
        final(core).node() == old(core).node(),
        final(core).spec_spec() == old(core).spec_spec(),
        final(core).version_registry_spec() == old(core).version_registry_spec(),
        final(alloc).well_formed(),
        final(alloc).id() == old(alloc).id(),
        final(alloc).node() == old(alloc).node(),
        final(events).well_formed(),
        final(events).id() == old(events).id(),
        final(events).node() == old(events).node(),
        r matches Ok((host, ctx)) ==> {
            &&& host.well_formed(final(core))
            &&& ctx.ready(final(core).instance_id(), final(core).node())
            &&& source_fresh_for_ledger(ctx.source@, final(core).ledger_spec()@)
            &&& ctx.source@.source() != host.ctx_spec().source@.source()
        },
{
    let tracked host_source = events.mint(EventSourceKind::host());
    let ghost host_source_id = host_source.source().id();
    proof {
        host_source.lemma_all_future_unsubmitted(instance, &core.borrow_ledger().submitted);
    }
    let host = HostBackend::new(
        core,
        Tracked(instance),
        Tracked(alloc),
        Tracked(host_source),
        1,
        FRAME_SIZE,
    )?;
    let tracked engine_source = events.mint(EventSourceKind::host());
    proof {
        assert(engine_source.source().id() != host_source_id);
        engine_source.lemma_all_future_unsubmitted(instance, &core.borrow_ledger().submitted);
    }
    Ok((host, SyncContext::new(Tracked(engine_source))))
}

#[cfg_attr(not(verus_only), allow(unused_mut, unused_variables))]
fn host_only_roundtrips() -> BackendResult<(Vec<u8>, Option<Vec<u8>>)> {
    let Some(ctx) = NodeContext::try_bootstrap(0) else {
        return Err(CoreError::WouldConflict);
    };
    let (instance, mut alloc, mut events, mut shares, _version_instance, _version_source) =
        ctx.into_sources_with_versions();
    let key = ObjectKey([1, 2, 3, 4]);

    let mut local_core = CacheCore::new(0, FRAME_SIZE, Vec::new(), 1, Tracked(instance.borrow()));
    let (local_host, local_ctx) = attach_host(
        &mut local_core,
        Tracked(instance.borrow()),
        Tracked(alloc.borrow_mut()),
        Tracked(events.borrow_mut()),
    )?;
    let mut local = LocalEngine::<NoDevice>::new(local_core, &local_host, local_ctx, None, None)?;
    assert(local.device_spec() is None);
    let local_data = vec![0x21u8; FRAME_SIZE];
    let (local_version, local_frame) = local.put_host(
        Tracked(instance.borrow()),
        key,
        &local_data,
    )?;
    assert(local.core_spec().spec_spec()@.versions[local_version] == local_data@);
    let mut local_out = vec![0u8; FRAME_SIZE];
    let Some(hit) = local.get_into_host(
        Tracked(instance.borrow()),
        Tracked(shares.borrow_mut()),
        key,
        &mut local_out,
    )? else {
        return Err(CoreError::Miss);
    };
    if hit.origin_node != local_version.origin_node || hit.seq != local_version.seq {
        return Err(CoreError::StaleReplica);
    }
    let Some(lease) = local.get_lease_host(
        Tracked(instance.borrow()),
        Tracked(shares.borrow_mut()),
        key,
    )? else {
        return Err(CoreError::Miss);
    };
    assert(local.core_spec().active_lease_matches(lease.handle));
    if lease.handle.frame().0 != local_frame.0 {
        return Err(CoreError::StaleReplica);
    }
    local.release_lease(Tracked(instance.borrow()), lease.handle, lease.cap, Ticket::NONE)?;
    local.evict_frame(local_frame)?;
    assert(local.core_spec().frame_state_spec(local_frame) == STATE_DRAINING);
    let local_tier = local.host_tier();
    if local.reclaim_draining(local_tier, 1)? != 1
        || !matches!(local.core().frame_state(local_frame), Some(STATE_EMPTY)) {
        return Err(CoreError::WouldConflict);
    }
    #[cfg(feature = "rdma")]
    let dist_out = {
        let mut dist_core = CacheCore::from_shard_versions(
            0,
            FRAME_SIZE,
            Vec::new(),
            1,
            1,
            4,
            Tracked(instance.borrow()),
            Tracked(_version_instance.borrow()),
            _version_source,
        );
        let (dist_host, dist_ctx) = attach_host(
            &mut dist_core,
            Tracked(instance.borrow()),
            Tracked(alloc.borrow_mut()),
            Tracked(events.borrow_mut()),
        )?;
        proof {
            lemma_core_wf_facts(&dist_core);
            assert(dist_core.all_versions_bound());
        }
        let mut dist = DistributedEngine::<ClosedTransport, NoDevice>::new_multi_peer(
            dist_core,
            &dist_host,
            dist_ctx,
            DeviceAttachment::None,
            None,
            Vec::new(),
        )?;
        assert(dist.device_spec() is None);
        let dist_data = vec![0x42u8; FRAME_SIZE];
        let (dist_version, dist_frame) = dist.put_host(
            Tracked(instance.borrow()),
            key,
            &dist_data,
        )?;
        proof {
            lemma_core_wf_facts(&dist.core_spec());
            assert(dist.core_spec().has_version_binding(dist_version));
            assert(dist.core_spec().spec_spec()@.versions[dist_version] == dist_data@);
        }
        let mut dist_out = vec![0u8; FRAME_SIZE];
        let Some(hit) = dist.get_into_host(
            Tracked(instance.borrow()),
            Tracked(shares.borrow_mut()),
            key,
            &mut dist_out,
        )? else {
            return Err(CoreError::Miss);
        };
        if hit.origin_node != dist_version.origin_node || hit.seq != dist_version.seq {
            return Err(CoreError::StaleReplica);
        }
        let Some(lease) = dist.get_lease_host(
            Tracked(instance.borrow()),
            Tracked(shares.borrow_mut()),
            key,
        )? else {
            return Err(CoreError::Miss);
        };
        assert(dist.core_spec().active_lease_matches(lease.handle));
        if lease.handle.frame().0 != dist_frame.0 {
            return Err(CoreError::StaleReplica);
        }
        dist.release_lease(Tracked(instance.borrow()), lease.handle, lease.cap, Ticket::NONE)?;
        dist.evict_frame(dist_frame)?;
        assert(dist.core_spec().frame_state_spec(dist_frame) == STATE_DRAINING);
        let dist_tier = dist.host_tier();
        if dist.reclaim_draining(dist_tier, 1)? != 1
            || !matches!(dist.core().frame_state(dist_frame), Some(STATE_EMPTY)) {
            return Err(CoreError::WouldConflict);
        }
        Some(dist_out)
    };
    #[cfg(not(feature = "rdma"))]
    let dist_out = None;
    Ok((local_out, dist_out))
}

} // verus!
#[cfg(test)]
mod runtime {
    #[test]
    fn no_device_host_lifecycles() {
        let distributed = cfg!(feature = "rdma").then(|| vec![0x42; super::FRAME_SIZE]);
        assert_eq!(
            super::host_only_roundtrips(),
            Ok((vec![0x21; super::FRAME_SIZE], distributed)),
        );
    }
}
