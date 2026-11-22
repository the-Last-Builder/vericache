//! The session delivers exactly the peer's bytes and the linear attachment
//! consumed by its corresponding send. Knowledge is still the sender's here.
use super::*;
verus! {

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
    transport.send_frame(bytes)
}

#[verifier::external_body]
pub(super) fn recv<T: ByteTransport, M: ChannelAttachment>(
    transport: &mut T,
    magic: u32,
    Ghost(id): Ghost<InstanceId>,
    _receiver: NodeId,
    _sender: NodeId,
    Ghost(sent): Ghost<Map<GrantTag, GrantKind>>,
    Ghost(received): Ghost<Map<GrantTag, GrantKind>>,
) -> (r: Result<Option<(Header, Vec<u8>, Tracked<Attachment<M>>)>, T::Error>)
    requires
        magic == M::protocol_magic(),
    ensures
        r matches Ok(Some((h, user_bytes, t))) ==> {
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
    match decode_with_magic(magic, &bytes) {
        Some((h, user_bytes)) => Ok(Some((h, user_bytes, Tracked(arbitrary())))),
        None => Ok(None),
    }
}

} // verus!
