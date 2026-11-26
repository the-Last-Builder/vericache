#[cfg(verus_only)]
use super::nvme_storage::nvme_well_formed;
#[cfg(feature = "rdma")]
use super::DistributedEngine;
use super::NvmeStorage;
use crate::backends::host::HostBackend;
#[cfg(feature = "rdma")]
use crate::backends::rdma::peer::PeerSession;
#[cfg(feature = "rdma")]
use crate::backends::rdma::pull::PullTarget;
use crate::backends::{DeviceBackend, DeviceSlice, DeviceStream};
use crate::kernel::CacheCore;
use crate::spec::TimelineId;
#[cfg(feature = "rdma")]
use crate::spec::{CoreError, TierId};
use crate::timeline::TimelineKind;
#[cfg(feature = "rdma")]
use transfer::{rdma, transport::ByteTransport};
use vstd::prelude::*;

verus! {

/// Retain the issuing stream for each used caller timeline until the engine is dropped.
pub(super) fn bind_caller_device_stream<S: DeviceStream>(
    streams: &mut Vec<Option<u64>>,
    timeline: TimelineId,
    stream: &S,
) -> bool {
    let index = timeline as usize;
    while streams.len() <= index
        decreases index + 1 - streams.len(),
    {
        streams.push(None);
    }
    let id = stream.runtime_id();
    if let Some(bound) = streams[index] {
        return bound == id;
    }
    streams.set(index, Some(id));
    true
}

pub(super) fn owns_caller_device_timeline<S: DeviceStream>(
    streams: &[Option<u64>],
    timeline: TimelineId,
    stream: &S,
) -> bool {
    let index = timeline as usize;
    index < streams.len() && streams[index] == Some(stream.runtime_id())
}

pub(super) fn device_slices_cover_frame<S: DeviceSlice>(
    slices: &[S],
    frame_len: Option<usize>,
) -> bool
    requires
        forall|i: int| 0 <= i < slices.len() ==> (#[trigger] slices[i]).well_formed(),
{
    let Some(mut remaining) = frame_len else {
        return false;
    };
    let mut i = 0;
    while i < slices.len()
        invariant
            i <= slices.len(),
            forall|j: int| 0 <= j < slices.len() ==> (#[trigger] slices[j]).well_formed(),
        decreases slices.len() - i,
    {
        let len = slices[i].len();
        if len == 0 || len > remaining {
            return false;
        }
        remaining -= len;
        i += 1;
    }
    i != 0 && remaining == 0
}

pub(crate) fn matches_backends<'a, D: DeviceBackend<'a>>(
    core: &CacheCore,
    host: &HostBackend,
    device: Option<&D>,
    nvme: &NvmeStorage,
) -> bool
    requires
        core.well_formed(),
        host.well_formed(core),
        device matches Some(c) ==> c.well_formed(core),
        nvme_well_formed(nvme, core),
{
    let kinds = core.timelines().kinds();
    let ht = host.timeline as usize;
    if core.runtime_id().is_none() || host.core_id != core.runtime_id() || ht >= kinds.len()
        || !matches!(kinds[ht], TimelineKind::Host) || !core.matches_slab(
        host.tier,
        host.first_frame,
        host.num_frames,
        host.buffer.slot_len_val() as u64,
    ) {
        return false;
    }
    if let Some(c) = device {
        if !c.matches_core_host(core, host) {
            return false;
        }
    }
    #[cfg(feature = "nvme")]
    if let Some(nvme) = nvme {
        let nt = nvme.timeline as usize;
        if nvme.core_id != core.runtime_id() || nt >= kinds.len()
            || !matches!(kinds[nt], TimelineKind::IoUring) || host.buffer.alignment() % (
        transfer::nvme::DIRECT_IO_ALIGNMENT as usize) != 0 || !core.matches_slab(
            nvme.tier,
            nvme.first_frame,
            nvme.num_frames,
            nvme.file.slot_len_val(),
        ) {
            return false;
        }
    }
    #[cfg(not(feature = "nvme"))]
    if nvme.is_some() {
        return false;
    }
    true
}

pub(crate) fn is_caller_device_timeline<'a, D: DeviceBackend<'a>>(
    core: &CacheCore,
    timeline: TimelineId,
    stream: &D::Stream,
) -> (r: bool)
    ensures
        r ==> timeline < core.timelines_spec().kinds_spec().len()
            && core.timelines_spec().kinds_spec()[timeline as int] == stream.timeline_kind_spec()
            && !(stream.timeline_kind_spec() is Host) && !(stream.timeline_kind_spec() is IoUring)
            && !(stream.timeline_kind_spec() is RdmaQueuePair),
{
    D::is_caller_timeline(core, timeline, stream)
}

#[cfg(feature = "rdma")]
impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    pub(crate) fn check_peer_ready(&self, peer_idx: usize) -> (r: Result<(), CoreError>)
        ensures
            peer_idx >= self.peers_spec().len() ==> r == Err(CoreError::InvalidArgument),
            r.is_ok() ==> peer_idx < self.peers_spec()@.len()
                && self.peers_spec()@[peer_idx as int].rdma_spec().qp_spec().state() is ReadyToSend,
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        if !self.peers[peer_idx].rdma.qp.is_ready_to_send() {
            return Err(CoreError::WouldConflict);
        }
        Ok(())
    }

    pub(crate) fn check_pull_target(&self, peer_idx: usize, tier: TierId, target: PullTarget) -> (r:
        Result<(), CoreError>)
        requires
            self.well_formed(),
            peer_idx < self.peers_spec().len(),
        ensures
            r.is_ok() ==> {
                &&& tier.valid(self.core_spec().tiers_spec().len())
                &&& self.core_spec().tier_frame_size_spec(tier) as u64 <= rdma::MAX_RDMA_MSG_LEN
                &&& self.peers_spec()[peer_idx as int].pull_tier_valid(tier, target)
            },
    {
        if tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        if self.core.registered_tier_frame_size(tier) as u64 > rdma::MAX_RDMA_MSG_LEN {
            return Err(CoreError::InvalidArgument);
        }
        let rdma = &self.peers[peer_idx].rdma;
        let slab = match target {
            PullTarget::Host => rdma.effective_host_mr(),
            PullTarget::GpuGdr => rdma.effective_gpu_mr(),
        };
        let Some(slab) = slab else {
            return Err(CoreError::UnsupportedTier);
        };
        if slab.tier.0 != tier.0 || !slab.mr.access_flags().local_write {
            return Err(CoreError::UnsupportedTier);
        }
        let is_gpu = slab.mr.gpu_device().is_some();
        match target {
            PullTarget::Host if is_gpu => Err(CoreError::UnsupportedTier),
            PullTarget::GpuGdr if !is_gpu => Err(CoreError::UnsupportedTier),
            _ => Ok(()),
        }
    }

    pub(crate) fn check_peer_pd(&self, peer_idx: usize, pd: &rdma::ProtectionDomain<'_>) -> (r:
        Result<(), CoreError>)
        requires
            peer_idx < self.peers_spec().len(),
        ensures
            r.is_err() ==> r == Err(CoreError::InvalidArgument),
    {
        if self.peers[peer_idx].rdma.qp.protection_domain().runtime_id() != pd.runtime_id() {
            return Err(CoreError::InvalidArgument);
        }
        Ok(())
    }

    pub(crate) fn check_all_peers_ready(&self) -> Result<(), CoreError> {
        let mut i = 0;
        while i < self.peers.len()
            invariant
                i <= self.peers@.len(),
            decreases self.peers@.len() - i,
        {
            self.check_peer_ready(i)?;
            i += 1;
        }
        Ok(())
    }

    pub(crate) fn valid_configuration(&self) -> bool
        requires
            self.well_formed(),
    {
        if !self.core.version_registry_enabled() || !matches_backends(
            &self.core,
            self.host,
            self.device.as_ref(),
            &self.nvme,
        ) {
            return false;
        }
        let mut i = 0;
        while i < self.peers.len()
            invariant
                i <= self.peers@.len(),
            decreases self.peers@.len() - i,
        {
            let peer = &self.peers[i];
            if !self.matches_peer(peer) || peer.rdma.active_lease_count() != 0 {
                return false;
            }
            let mut j = 0;
            while j < i
                invariant
                    j <= i < self.peers@.len(),
                decreases i - j,
            {
                let existing = &self.peers[j];
                if peer.rdma.timeline_id == existing.rdma.timeline_id || peer.rdma.peer_node
                    == existing.rdma.peer_node {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }
        true
    }

    pub(crate) fn matches_peer(&self, peer: &PeerSession<'a, T>) -> bool {
        let p = &peer.rdma;
        let tl = p.timeline_id as usize;
        let kinds = self.core.timelines().kinds();
        if tl >= kinds.len() {
            return false;
        }
        let kind = kinds[tl];
        p.core_id == self.core.runtime_id() && p.peer_node != self.core.node_id() as u64
            && p.peer_node == p.qp.peer_node() && peer.channel.local_node()
            == self.core.node_id() as u64 && peer.channel.peer_node() == p.peer_node
            && matches!(kind, TimelineKind::RdmaQueuePair { peer_node } if peer_node == p.peer_node)
    }
}

} // verus!
