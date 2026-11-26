//! The session delivers exactly the peer's bytes and the linear attachment
//! consumed by its corresponding send. Knowledge is still the sender's here.
use super::*;
use std::collections::HashMap;
verus! {

impl RemoteCapability {
    /// Record a descriptor obtained from a successfully bound window.
    ///
    /// # Safety
    /// The address, length and key must denote the live window with the stated
    /// slot, access rights and share.
    pub unsafe fn from_bound_window(
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
        Self::new(addr, len, rkey, slot, access, share)
    }
}

#[verifier::external_body]
fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[verifier::external_body]
fn decode_and_validate<M: ChannelAttachment>(
    bytes: &[u8],
    magic: u32,
    exec_sent: &HashMap<GrantTag, u8>,
    exec_received: &HashMap<GrantTag, u8>,
) -> RecvPoll<M> {
    let decoded = if magic == 0 {
        decode_with_magic(0, bytes)
    } else {
        if bytes.len() < 4 {
            return RecvPoll::Invalid;
        }
        let split = bytes.len() - 4;
        let payload = &bytes[..split];
        let trailer = &bytes[split..];
        let expected = crc32_ieee(payload);
        let actual = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
        if expected != actual {
            return RecvPoll::Invalid;
        }
        decode_with_magic(magic, payload)
    };
    let Some((h, user_bytes)) = decoded else {
        return RecvPoll::Invalid;
    };
    if h.op == READ_GRANT || h.op == WRITE_GRANT {
        if exec_received.contains_key(&h.tag) {
            return RecvPoll::Invalid;
        }
    } else if h.op == READ_RELEASE {
        match exec_sent.get(&h.tag) {
            Some(op) if *op == READ_GRANT => {},
            _ => return RecvPoll::Invalid,
        }
    } else if h.op == WRITE_RELEASE {
        match exec_sent.get(&h.tag) {
            Some(op) if *op == WRITE_GRANT => {},
            _ => return RecvPoll::Invalid,
        }
    } else if h.op == ENDPOINT {
        if user_bytes.len() != 28 {
            return RecvPoll::Invalid;
        }
    }
    RecvPoll::Frame(h, user_bytes, Tracked(arbitrary()))
}

#[verifier::external_body]
pub(super) fn send<T: ByteTransport, M: ChannelAttachment>(
    transport: &mut T,
    bytes: &[u8],
    Tracked(a): Tracked<Attachment<M>>,
    Ghost(id): Ghost<InstanceId>,
    _sender: NodeId,
    _receiver: NodeId,
) -> Result<(), T::Error>
    requires
        a.valid(id, _sender),
        header_matches_with_magic(M::protocol_magic(), a.header, bytes@, a.user),
{
    let magic = M::magic_exec();
    if magic == 0 {
        transport.send_frame(bytes)
    } else {
        let crc = crc32_ieee(bytes);
        let mut framed = Vec::with_capacity(bytes.len() + 4);
        framed.extend_from_slice(bytes);
        framed.extend_from_slice(&crc.to_le_bytes());
        transport.send_frame(&framed)
    }
}

#[verifier::external_body]
pub(super) fn recv<T: ByteTransport, M: ChannelAttachment>(
    transport: &mut T,
    magic: u32,
    Ghost(id): Ghost<InstanceId>,
    _receiver: NodeId,
    _sender: NodeId,
    exec_sent: &HashMap<GrantTag, u8>,
    exec_received: &HashMap<GrantTag, u8>,
    Ghost(sent): Ghost<Map<GrantTag, GrantKind>>,
    Ghost(received): Ghost<Map<GrantTag, GrantKind>>,
) -> (r: Result<RecvPoll<M>, T::Error>)
    requires
        magic == M::protocol_magic(),
        tag_map_matches(exec_sent@, sent),
        tag_map_matches(exec_received@, received),
    ensures
        r matches Ok(poll) ==> !(poll is Empty),
        r matches Ok(RecvPoll::Frame(h, user_bytes, t)) ==> {
            let a = t@;
            &&& a.valid(id, _sender)
            &&& a.header == h
            &&& a.user == user_bytes@
            &&& h.grant() ==> !received.contains_key(h.tag)
            &&& !h.grant() && h.op < NOTICE ==> sent.contains_key(h.tag) && a.kind == Some(
                sent[h.tag],
            )
        },
{
    let bytes = transport.recv_frame()?;
    Ok(decode_and_validate(&bytes, magic, exec_sent, exec_received))
}

#[verifier::external_body]
pub(super) fn try_recv<T: ByteTransport, M: ChannelAttachment>(
    transport: &mut T,
    magic: u32,
    Ghost(id): Ghost<InstanceId>,
    _receiver: NodeId,
    _sender: NodeId,
    exec_sent: &HashMap<GrantTag, u8>,
    exec_received: &HashMap<GrantTag, u8>,
    Ghost(sent): Ghost<Map<GrantTag, GrantKind>>,
    Ghost(received): Ghost<Map<GrantTag, GrantKind>>,
) -> (r: Result<RecvPoll<M>, T::Error>)
    requires
        magic == M::protocol_magic(),
        tag_map_matches(exec_sent@, sent),
        tag_map_matches(exec_received@, received),
    ensures
        r matches Ok(RecvPoll::Frame(h, user_bytes, t)) ==> {
            let a = t@;
            &&& a.valid(id, _sender)
            &&& a.header == h
            &&& a.user == user_bytes@
            &&& h.grant() ==> !received.contains_key(h.tag)
            &&& !h.grant() && h.op < NOTICE ==> sent.contains_key(h.tag) && a.kind == Some(
                sent[h.tag],
            )
        },
{
    match transport.try_recv_frame()? {
        Some(bytes) => Ok(decode_and_validate(&bytes, magic, exec_sent, exec_received)),
        None => Ok(RecvPoll::Empty),
    }
}

} // verus!
