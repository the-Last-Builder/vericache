//! Framed, authenticated, at-most-once transport between two provisioned nodes.
mod channel;
mod ffi;
mod wire;

use crate::context::*;
use crate::domain::*;
#[cfg(verus_only)]
use crate::rdma::RemoteAccess;
use crate::rdma::{QueuePairEndpoint, RemoteCapability};
#[cfg(verus_only)]
use foundation::tracked::*;
use vstd::bytes::*;
use vstd::prelude::*;
use vstd::slice::{slice_subrange, slice_to_vec};
use vstd::tokens::InstanceId;

pub use channel::Channel;
#[cfg(verus_only)]
pub use wire::endpoint_matches;
use wire::*;

verus! {

/// Each implementation supplies a fresh session with the designated peer.
/// A received frame must be exactly one authentic send on that session.
pub trait ByteTransport {
    type Error;

    fn send_frame(&mut self, frame: &[u8]) -> Result<(), Self::Error>;

    fn recv_frame(&mut self) -> Result<Vec<u8>, Self::Error>;

    fn try_recv_frame(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        match self.recv_frame() {
            Ok(frame) => Ok(Some(frame)),
            Err(e) => Err(e),
        }
    }
}

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub type GrantTag = u64;

pub ghost enum GrantKind {
    ReadShare { slot: SlotRef, share_id: TransferShare },
    OwnedSlot { slot: SlotRef },
}

pub tracked struct Statement {
    pub tracked succeeded: SucceededMap,
    pub tracked submitted: SubmittedMap,
    pub tracked knows: Knows,
    pub ghost facts: ISet<TransferEvent>,
}

impl Statement {
    pub proof fn empty(id: InstanceId) -> (tracked statement: Self)
        ensures
            forall|sender| #[trigger] statement.valid(id, sender),
            statement.facts == ISet::empty(),
            statement.submitted.map() == IMap::empty(),
            statement.knows.set() == ISet::empty(),
            statement.succeeded.dom() == Set::empty(),
    {
        Self {
            submitted: SubmittedMap::empty(id),
            knows: Knows::empty(id),
            succeeded: SucceededMap::empty(id),
            facts: ISet::empty(),
        }
    }

    pub open spec fn valid(self, id: InstanceId, sender: NodeId) -> bool {
        &&& self.submitted.instance_id() == id && self.knows.instance_id() == id
        &&& self.facts.subset_of(self.submitted.dom())
        &&& knowledge_of(sender, self.facts).subset_of(self.knows.set())
        &&& self.succeeded.instance_id() == id && self.succeeded.well_formed()
        &&& forall|e| #[trigger] self.succeeded.contains(e) ==> self.submitted.dom().contains(e)
    }

    pub open spec fn closure(self) -> ISet<TransferEvent> {
        closure_from(self.submitted.map(), self.facts)
    }
}

pub open spec fn covers(front: Frontier<TransferDomain>, events: ISet<TransferEvent>) -> bool {
    ordered_or_absent(front.last_writer, events) && front.readers.subset_of(events)
}

pub open spec fn share_matches(
    share: ShareCap,
    id: InstanceId,
    slot: SlotRef,
    share_id: TransferShare,
    events: ISet<TransferEvent>,
) -> bool {
    share.is_rdma_bound() && share.instance_id() == id && share.loc() == slot.loc && share.share()
        == share_id && covers(share.frontier(), events)
}

pub open spec fn slot_matches(
    tok: SlotToken,
    id: InstanceId,
    slot: SlotRef,
    events: ISet<TransferEvent>,
) -> bool {
    slot_is_owned(tok, id, slot.loc) && covers(slot_frontier(tok), events)
}

#[derive(Debug)]
pub enum ChannelError<E> {
    Transport(E),
    InvalidFrame,
    Closed,
}

/// Constructive protocol-layer attachment carried alongside wire frames.
///
/// Protocols with non-zero `protocol_magic()` prefix every wire frame with
/// `[0xff, le_u32(magic)]` and append a 4-byte little-endian CRC32 trailer over
/// the prefixed header + payload bytes, isolating and integrity-checking
/// protocol sessions against legacy (`()`) frames and bit corruption.
pub trait ChannelAttachment: Sized {
    spec fn protocol_magic() -> u32;

    fn magic_exec() -> (m: u32)
        ensures
            m == Self::protocol_magic(),
    ;

    /// `frontier` is the ghost frontier of the share carried with a READ grant
    /// or release (`None` for frames without a share).
    open spec fn valid(
        &self,
        id: InstanceId,
        sender: NodeId,
        op: u8,
        tag: GrantTag,
        user: Seq<u8>,
        kind: Option<GrantKind>,
        cap: Option<RemoteCapability>,
        frontier: Option<Frontier<TransferDomain>>,
        statement: Statement,
    ) -> bool {
        true
    }
}

/// Ghost frontier of an optional share, as passed to `ChannelAttachment::valid`.
pub open spec fn share_frontier(share: Option<ShareCap>) -> Option<Frontier<TransferDomain>> {
    match share {
        Some(sh) => Some(sh.frontier()),
        None => None,
    }
}

impl ChannelAttachment for () {
    open spec fn protocol_magic() -> u32 {
        0
    }

    fn magic_exec() -> (m: u32) {
        0
    }
}

pub struct Received<M = ()> {
    pub endpoint: Ghost<Option<QueuePairEndpoint>>,
    pub tag: GrantTag,
    pub user_bytes: Vec<u8>,
    pub capability: Option<RemoteCapability>,
    pub kind: Ghost<Option<GrantKind>>,
    pub slot: Tracked<Option<SlotToken>>,
    pub share: Tracked<Option<ShareCap>>,
    pub statement: Tracked<Statement>,
    pub attachment: Tracked<M>,
}

impl<M> Received<M> {
    pub open spec fn learned(&self, id: InstanceId, receiver: NodeId) -> bool {
        self.statement@.valid(id, receiver) && self.statement@.knows.set() == knowledge_of(
            receiver,
            self.statement@.closure(),
        )
    }
}

pub enum ReceivedFrame<M = ()> {
    ReadGrant(Received<M>),
    ReadRelease(Received<M>),
    WriteGrant(Received<M>),
    WriteRelease(Received<M>),
    Notice(Received<M>),
}

pub enum EndpointOrFrame<M = ()> {
    Endpoint(QueuePairEndpoint),
    Frame(ReceivedFrame<M>),
}

} // verus!
