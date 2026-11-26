//! Remote memory and queue-pair descriptors carried by the control channel.
use super::*;

verus! {

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RemoteAccess {
    ReadOnly,
    ReadWrite,
}

pub struct RemoteCapability {
    share: Ghost<Option<TransferShare>>,
    addr: u64,
    len: u64,
    rkey: u32,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    slot: Ghost<SlotRef>,
    access: Ghost<RemoteAccess>,
}

impl RemoteCapability {
    pub(super) fn new(
        addr: u64,
        len: u64,
        rkey: u32,
        slot: Ghost<SlotRef>,
        access: Ghost<RemoteAccess>,
        share: Ghost<Option<TransferShare>>,
    ) -> (cap: Self)
        requires
            addr + len <= u64::MAX,
            len > 0,
            slot@.well_formed(),
            len == slot@.sub_len,
        ensures
            cap.well_formed(),
            cap.descriptor() == (addr, len, rkey),
            cap.slot_ref() == slot@,
            cap.access() == access@,
            cap.share() == share@,
    {
        Self { addr, len, rkey, slot, access, share }
    }

    pub closed spec fn share(&self) -> Option<TransferShare> {
        self.share@
    }

    pub closed spec fn slot_ref(&self) -> SlotRef {
        self.slot@
    }

    pub closed spec fn access(&self) -> RemoteAccess {
        self.access@
    }

    pub closed spec fn descriptor(&self) -> (u64, u64, u32) {
        (self.addr, self.len, self.rkey)
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.len == self.slot@.sub_len && self.slot@.well_formed() && self.addr + self.len
            <= u64::MAX && self.len > 0
    }

    pub fn len(&self) -> (n: u64)
        requires
            self.well_formed(),
        ensures
            n == self.slot_ref().sub_len,
    {
        self.len
    }

    pub proof fn lemma_nonempty(&self)
        requires
            self.well_formed(),
        ensures
            self.slot_ref().sub_len > 0,
    {
    }

    pub(super) fn from_wire(
        addr: u64,
        len: u64,
        rkey: u32,
        Ghost(sent): Ghost<RemoteCapability>,
    ) -> (r: Self)
        requires
            sent.descriptor() == (addr, len, rkey),
        ensures
            r == sent,
    {
        Self {
            addr,
            len,
            rkey,
            share: Ghost(sent.share()),
            slot: Ghost(sent.slot_ref()),
            access: Ghost(sent.access()),
        }
    }

    /// A narrower descriptor keeps the original window key, slot identity, and access rights.
    pub fn sub_slice(&self, offset: u64, len: u64) -> (s: Self)
        requires
            self.well_formed(),
            0 < len,
            offset + len <= self.slot_ref().sub_len,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref().sub_slice(offset, len),
            s.share() == self.share(),
            s.access() == self.access(),
            s.descriptor() == ((self.descriptor().0 + offset) as u64, len, self.descriptor().2),
    {
        assert!(len > 0 && offset <= self.len && len <= self.len - offset);
        Self {
            addr: self.addr + offset,
            len,
            rkey: self.rkey,
            share: self.share,
            access: self.access,
            slot: Ghost(self.slot_ref().sub_slice(offset, len)),
        }
    }

    pub fn wire_fields(&self) -> (r: (u64, u64, u32))
        ensures
            r == self.descriptor(),
    {
        (self.addr, self.len, self.rkey)
    }

    pub fn descriptor_copy(&self) -> (cap: Self)
        ensures
            cap == *self,
    {
        Self {
            addr: self.addr,
            len: self.len,
            rkey: self.rkey,
            share: self.share,
            access: self.access,
            slot: self.slot,
        }
    }
}

#[derive(Clone, Copy)]
pub struct QueuePairEndpoint {
    number: u32,
    psn: u32,
    lid: u16,
    gid: [u8; 16],
    mtu: u8,
    responder_resources: u8,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    origin: Ghost<NodeId>,
}

impl QueuePairEndpoint {
    pub fn new(
        number: u32,
        psn: u32,
        lid: u16,
        gid: [u8; 16],
        mtu: u8,
        responder_resources: u8,
        origin: Ghost<NodeId>,
    ) -> (r: Self)
        ensures
            r.node() == origin@,
            r.fields() == (number, psn, lid, gid@, mtu, responder_resources),
            r.responder_resources_spec() == responder_resources,
    {
        Self { number, psn, lid, gid, mtu, responder_resources, origin }
    }

    pub fn with_resources(&self, responder_resources: u8, origin: Ghost<NodeId>) -> (r: Self)
        ensures
            r.node() == origin@,
            r.responder_resources_spec() == responder_resources,
    {
        Self { responder_resources, origin, ..*self }
    }

    pub fn with_psn(&self, psn: u32) -> (r: Self)
        ensures
            r.node() == self.node(),
            r.responder_resources_spec() == self.responder_resources_spec(),
    {
        Self { psn, ..*self }
    }

    pub fn responder_resources(&self) -> (n: u8)
        ensures
            n == self.responder_resources_spec(),
    {
        self.responder_resources
    }

    pub closed spec fn node(&self) -> NodeId {
        self.origin@
    }

    pub closed spec fn responder_resources_spec(&self) -> u8 {
        self.responder_resources
    }

    pub closed spec fn fields(&self) -> (u32, u32, u16, Seq<u8>, u8, u8) {
        (self.number, self.psn, self.lid, self.gid@, self.mtu, self.responder_resources)
    }

    pub fn components(&self) -> (r: (u32, u32, u16, [u8; 16], u8, u8))
        ensures
            self.fields() == (r.0, r.1, r.2, r.3@, r.4, r.5),
    {
        (self.number, self.psn, self.lid, self.gid, self.mtu, self.responder_resources)
    }

    pub(super) fn restore(
        number: u32,
        psn: u32,
        lid: u16,
        gid: [u8; 16],
        mtu: u8,
        responder_resources: u8,
        Ghost(sent): Ghost<Self>,
    ) -> (r: Self)
        requires
            sent.fields() == (number, psn, lid, gid@, mtu, responder_resources),
        ensures
            r == sent,
    {
        assert(gid == sent.gid);
        Self { number, psn, lid, gid, mtu, responder_resources, origin: Ghost(sent.node()) }
    }
}

} // verus!
