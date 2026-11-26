use super::{DistributedEngine, PendingReadGrant};
use crate::backends::{BackendResult, DeviceBackend};
use crate::spec::CoreError;
use transfer_core::context::TransferInstance;
use transfer_core::transport::ByteTransport;
use vstd::prelude::*;

verus! {

impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    pub fn take_inbound_grant(&mut self, peer_idx: usize, request_id: u64) -> (r: BackendResult<
        Option<PendingReadGrant>,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            r matches Ok(Some(g)) ==> g.valid_for_peer(final(self), peer_idx as int)
                && g.meta_spec().request_id == request_id,
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        let grant = self.peers[peer_idx].take_inbound_grant(&self.core, request_id);
        match grant {
            Some(g) => Ok(
                Some(
                    PendingReadGrant {
                        core_id: self.peers[peer_idx].rdma.core_id,
                        peer_timeline: self.peers[peer_idx].rdma.timeline_id,
                        connection_epoch: self.peers[peer_idx].reconnect_epoch,
                        tag: g.tag,
                        meta: g.meta,
                        cap: g.cap,
                        share: g.share,
                        statement: g.statement,
                        cert: g.cert,
                        revoked: g.revoked,
                        version_binding: g.version_binding,
                    },
                ),
            ),
            None => Ok(None),
        }
    }

    /// Return an unused grant to its donor.
    pub fn discard_peer_grant(
        &mut self,
        peer_idx: usize,
        grant: PendingReadGrant,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            grant.valid_for_peer(old(self), peer_idx as int),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
    {
        if peer_idx >= self.peers.len() || !grant.matches_connection(&self.peers[peer_idx]) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost before_peers = self.peers@;
        self.peers[peer_idx].discard_read_grant(
            &mut self.core,
            grant.into_pending(),
            Tracked(instance),
        );
        self.restore_after_peer_step(
            peer_idx,
            Ghost(before),
            Ghost(before_peers),
            Tracked(instance),
        );
        Ok(())
    }

    pub fn take_inbound_miss(&mut self, peer_idx: usize, request_id: u64) -> (r: BackendResult<
        bool,
    >)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
    {
        if peer_idx >= self.peers.len() {
            return Err(CoreError::InvalidArgument);
        }
        Ok(self.peers[peer_idx].take_inbound_miss(request_id))
    }

    pub fn has_inbound_grant(&self, peer_idx: usize, request_id: u64) -> bool {
        let Some(peer) = self.peer(peer_idx) else {
            return false;
        };
        let mut i = 0;
        while i < peer.pending_inbound_grants.len()
            invariant
                i <= peer.pending_inbound_grants@.len(),
            decreases peer.pending_inbound_grants@.len() - i,
        {
            if peer.pending_inbound_grants[i].meta.request_id == request_id {
                return true;
            }
            i += 1;
        }
        false
    }
}

} // verus!
