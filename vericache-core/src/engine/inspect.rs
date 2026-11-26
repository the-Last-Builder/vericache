#[cfg(feature = "rdma")]
use super::DistributedEngine;
use super::LocalEngine;
use crate::backends::host::HostBackend;
#[cfg(feature = "nvme")]
use crate::backends::nvme::NvmeBackend;
#[cfg(feature = "rdma")]
use crate::backends::rdma::peer::PeerSession;
use crate::backends::DeviceBackend;
use crate::kernel::CacheCore;
#[cfg(feature = "rdma")]
use transfer::{rdma::ProtectionDomain, transport::ByteTransport};
use vstd::prelude::*;

verus! {

impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    pub fn core(&self) -> (core: &CacheCore)
        ensures
            *core == self.core_spec(),
    {
        &self.core
    }

    pub fn host(&self) -> (host: &'a HostBackend)
        ensures
            host == self.host_spec(),
    {
        self.host
    }

    pub fn device(&self) -> (r: Option<&D>)
        ensures
            match r {
                Some(device) => self.device_spec() == Some(*device),
                None => self.device_spec() is None,
            },
    {
        self.device.as_ref()
    }

    pub fn cuda(&self) -> (r: Option<&D>)
        ensures
            match r {
                Some(device) => self.device_spec() == Some(*device),
                None => self.device_spec() is None,
            },
    {
        self.device()
    }

    #[cfg(feature = "nvme")]
    pub fn nvme(&self) -> (r: Option<&NvmeBackend>)
        ensures
            match r {
                Some(nvme) => self.nvme_spec() == Some(*nvme),
                None => self.nvme_spec() is None,
            },
    {
        self.nvme.as_ref()
    }
}

#[cfg(feature = "rdma")]
impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    pub fn core(&self) -> (core: &CacheCore)
        ensures
            *core == self.core_spec(),
    {
        &self.core
    }

    pub fn host(&self) -> (host: &'a HostBackend)
        ensures
            host == self.host_spec(),
    {
        self.host
    }

    pub fn device(&self) -> (r: Option<&D>)
        ensures
            match r {
                Some(device) => self.device_spec().spec_ref() == Some(*device),
                None => self.device_spec().spec_ref() is None,
            },
    {
        self.device.as_ref()
    }

    pub fn cuda(&self) -> (r: Option<&D>)
        ensures
            match r {
                Some(device) => self.device_spec().spec_ref() == Some(*device),
                None => self.device_spec().spec_ref() is None,
            },
    {
        self.device()
    }

    #[cfg(feature = "nvme")]
    pub fn nvme(&self) -> (r: Option<&NvmeBackend>)
        ensures
            match r {
                Some(nvme) => self.nvme_spec() == Some(*nvme),
                None => self.nvme_spec() is None,
            },
    {
        self.nvme.as_ref()
    }

    pub fn peer(&self, peer_idx: usize) -> (r: Option<&PeerSession<'a, T>>)
        ensures
            match r {
                Some(peer) => peer_idx < self.peers_spec().len() && *peer
                    == self.peers_spec()[peer_idx as int],
                None => peer_idx >= self.peers_spec().len(),
            },
    {
        if peer_idx < self.peers.len() {
            Some(&self.peers[peer_idx])
        } else {
            None
        }
    }

    pub fn peer_node(&self, peer_idx: usize) -> Option<u64> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.peer_node())
    }

    pub fn peer_window_free_count(&self, peer_idx: usize) -> Option<usize> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.window_pool.free_count())
    }

    pub fn peer_window_capacity(&self, peer_idx: usize) -> Option<usize> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.window_pool.capacity())
    }

    pub fn peer_active_lease_count(&self, peer_idx: usize) -> Option<usize> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.active_lease_count())
    }

    pub fn peer_max_rd_atomic(&self, peer_idx: usize) -> Option<u8> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.qp.effective_max_rd_atomic_exec())
    }

    pub fn peer_pd_id(&self, peer_idx: usize) -> Option<usize> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.qp.protection_domain().runtime_id())
    }

    pub fn peer_protection_domain(&self, peer_idx: usize) -> Option<&'a ProtectionDomain<'a>> {
        let peer = self.peer(peer_idx)?;
        Some(peer.rdma.qp.protection_domain())
    }

    pub fn peer_pending_grant_count(&self, peer_idx: usize) -> Option<usize> {
        let peer = self.peer(peer_idx)?;
        Some(peer.pending_outbound_grants.len())
    }

    pub fn peer_next_grant_tag(&self, peer_idx: usize) -> Option<u64> {
        let peer = self.peer(peer_idx)?;
        Some(peer.next_grant_tag)
    }
}

} // verus!
