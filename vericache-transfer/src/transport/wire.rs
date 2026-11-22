use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub(super) const READ_GRANT: u8 = 1;

pub(super) const READ_RELEASE: u8 = 2;

pub(super) const WRITE_GRANT: u8 = 3;

pub(super) const WRITE_RELEASE: u8 = 4;

pub(super) const NOTICE: u8 = 5;

// Setup frames certify which node owns an RC endpoint before the first grant.
pub(super) const ENDPOINT: u8 = 6;

pub(super) const PROTOCOL_MAGIC_TAG: u8 = 0xff;

#[derive(Clone, Copy)]
pub(super) struct Header {
    pub(super) op: u8,
    pub(super) tag: u64,
    pub(super) addr: u64,
    pub(super) len: u64,
    pub(super) rkey: u32,
}

impl Header {
    pub(super) open spec fn grant(self) -> bool {
        self.op == READ_GRANT || self.op == WRITE_GRANT
    }

    pub(super) open spec fn size(self) -> int {
        if self.grant() {
            29
        } else if self.op >= NOTICE {
            1
        } else {
            9
        }
    }

    pub(super) open spec fn valid(self) -> bool {
        1 <= self.op <= ENDPOINT && (!self.grant() ==> self.addr == 0 && self.len == 0 && self.rkey
            == 0) && (self.op >= NOTICE ==> self.tag == 0)
    }

    pub(super) open spec fn matches(self, b: Seq<u8>, payload: Seq<u8>) -> bool {
        &&& self.valid() && self.size() <= b.len()
        &&& self.op == b[0]
        &&& self.op < NOTICE ==> self.tag == spec_u64_from_le_bytes(b.subrange(1, 9))
        &&& self.grant() ==> self.addr == spec_u64_from_le_bytes(b.subrange(9, 17)) && self.len
            == spec_u64_from_le_bytes(b.subrange(17, 25)) && self.rkey == spec_u32_from_le_bytes(
            b.subrange(25, 29),
        )
        &&& payload == b.skip(self.size())
    }
}

fn append(bytes: &mut Vec<u8>, s: &[u8])
    ensures
        final(bytes)@ == old(bytes)@ + s@,
{
    bytes.extend_from_slice(s);
    assert(bytes@ =~= old(bytes)@ + s@);
}

pub(super) fn encode(h: Header, user: &[u8]) -> (b: Vec<u8>)
    requires
        h.valid(),
    ensures
        h.matches(b@, user@),
{
    proof {
        lemma_auto_spec_u64_to_from_le_bytes();
        lemma_auto_spec_u32_to_from_le_bytes();
    }
    let mut b = Vec::new();
    b.push(h.op);
    if h.op < NOTICE {
        append(&mut b, &u64_to_le_bytes(h.tag));
    }
    if h.op == READ_GRANT || h.op == WRITE_GRANT {
        append(&mut b, &u64_to_le_bytes(h.addr));
        append(&mut b, &u64_to_le_bytes(h.len));
        append(&mut b, &u32_to_le_bytes(h.rkey));
    }
    append(&mut b, user);
    proof {
        if h.op < NOTICE {
            assert(b@.subrange(1, 9) =~= spec_u64_to_le_bytes(h.tag));
        }
        if h.grant() {
            assert(b@.subrange(9, 17) =~= spec_u64_to_le_bytes(h.addr));
            assert(b@.subrange(17, 25) =~= spec_u64_to_le_bytes(h.len));
            assert(b@.subrange(25, 29) =~= spec_u32_to_le_bytes(h.rkey));
        }
        assert(b@.skip(h.size()) =~= user@);
    }
    b
}

pub(super) fn decode(b: &[u8]) -> (r: Option<(Header, Vec<u8>)>)
    ensures
        r matches Some((h, p)) ==> h.matches(b@, p@),
        forall|h: Header, p: Seq<u8>| #[trigger]
            h.matches(b@, p) ==> (r matches Some((d, out)) && d == h && out@ == p),
{
    if b.len() == 0 {
        return None;
    }
    let op = b[0];
    let grant = op == READ_GRANT || op == WRITE_GRANT;
    let size = if grant {
        29
    } else if op == NOTICE || op == ENDPOINT {
        1
    } else if op == READ_RELEASE || op == WRITE_RELEASE {
        9
    } else {
        return None;
    };
    if b.len() < size {
        return None;
    }
    let tag = if op < NOTICE {
        u64_from_le_bytes(slice_subrange(b, 1, 9))
    } else {
        0
    };
    let (addr, len, rkey) = if grant {
        (
            u64_from_le_bytes(slice_subrange(b, 9, 17)),
            u64_from_le_bytes(slice_subrange(b, 17, 25)),
            u32_from_le_bytes(slice_subrange(b, 25, 29)),
        )
    } else {
        (0, 0, 0)
    };
    let payload = slice_to_vec(slice_subrange(b, size, b.len()));
    let h = Header { op, tag, addr, len, rkey };
    assert(payload@ =~= b@.skip(h.size()));
    Some((h, payload))
}

pub(super) open spec fn header_matches_with_magic(
    magic: u32,
    h: Header,
    b: Seq<u8>,
    payload: Seq<u8>,
) -> bool {
    if magic == 0 {
        h.matches(b, payload)
    } else {
        b.len() >= 5 && b[0] == PROTOCOL_MAGIC_TAG && spec_u32_from_le_bytes(b.subrange(1, 5))
            == magic && h.matches(b.skip(5), payload)
    }
}

pub(super) fn encode_with_magic(magic: u32, h: Header, user: &[u8]) -> (b: Vec<u8>)
    requires
        h.valid(),
    ensures
        header_matches_with_magic(magic, h, b@, user@),
{
    if magic == 0 {
        encode(h, user)
    } else {
        proof {
            lemma_auto_spec_u32_to_from_le_bytes();
        }
        let inner = encode(h, user);
        let mut b = Vec::new();
        b.push(PROTOCOL_MAGIC_TAG);
        append(&mut b, &u32_to_le_bytes(magic));
        append(&mut b, &inner);
        proof {
            assert(b@.subrange(1, 5) =~= spec_u32_to_le_bytes(magic));
            assert(b@.skip(5) =~= inner@);
        }
        b
    }
}

pub(super) fn decode_with_magic(magic: u32, b: &[u8]) -> (r: Option<(Header, Vec<u8>)>)
    ensures
        r matches Some((h, p)) ==> header_matches_with_magic(magic, h, b@, p@),
        forall|h: Header, p: Seq<u8>| #[trigger]
            header_matches_with_magic(magic, h, b@, p) ==> (r matches Some((d, out)) && d == h
                && out@ == p),
{
    if magic == 0 {
        decode(b)
    } else {
        if b.len() < 5 || b[0] != PROTOCOL_MAGIC_TAG {
            return None;
        }
        let got = u32_from_le_bytes(slice_subrange(b, 1, 5));
        if got != magic {
            return None;
        }
        let rest = slice_subrange(b, 5, b.len());
        proof {
            assert(rest@ =~= b@.skip(5));
        }
        decode(rest)
    }
}

// Endpoint fields share the same little-endian encoding as grant fields.
pub closed spec fn endpoint_matches(e: QueuePairEndpoint, b: Seq<u8>) -> bool {
    &&& b.len() == 28
    &&& e.fields() == (
        spec_u32_from_le_bytes(b.subrange(0, 4)),
        spec_u32_from_le_bytes(b.subrange(4, 8)),
        spec_u16_from_le_bytes(b.subrange(8, 10)),
        b.subrange(10, 26),
        b[26],
        b[27],
    )
}

pub(super) fn encode_endpoint(e: &QueuePairEndpoint) -> (bytes: Vec<u8>)
    ensures
        endpoint_matches(*e, bytes@),
{
    let (number, psn, lid, gid, mtu, responder_resources) = e.components();
    proof {
        lemma_auto_spec_u32_to_from_le_bytes();
        lemma_auto_spec_u16_to_from_le_bytes();
    }
    let mut bytes = u32_to_le_bytes(number);
    append(&mut bytes, &u32_to_le_bytes(psn));
    append(&mut bytes, &u16_to_le_bytes(lid));
    append(&mut bytes, &gid);
    bytes.push(mtu);
    bytes.push(responder_resources);
    assert(bytes@.subrange(0, 4) =~= spec_u32_to_le_bytes(number));
    assert(bytes@.subrange(4, 8) =~= spec_u32_to_le_bytes(psn));
    assert(bytes@.subrange(8, 10) =~= spec_u16_to_le_bytes(lid));
    assert(bytes@.subrange(10, 26) =~= gid@);
    bytes
}

pub(super) fn decode_endpoint(bytes: &[u8], Ghost(sent): Ghost<QueuePairEndpoint>) -> (r: Option<
    QueuePairEndpoint,
>)
    requires
        endpoint_matches(sent, bytes@),
    ensures
        r == Some(sent),
{
    if bytes.len() != 28 {
        return None;
    }
    let number = u32_from_le_bytes(slice_subrange(bytes, 0, 4));
    let psn = u32_from_le_bytes(slice_subrange(bytes, 4, 8));
    let lid = u16_from_le_bytes(slice_subrange(bytes, 8, 10));
    let mut gid = [0u8;16];
    let mut i = 0usize;
    while i < 16
        invariant
            i <= 16,
            bytes.len() == 28,
            forall|j: int| 0 <= j < i ==> #[trigger] gid@[j] == bytes@[10 + j],
        decreases 16 - i,
    {
        gid[i] = bytes[10 + i];
        i += 1;
    }
    assert(gid@ =~= bytes@.subrange(10, 26));
    Some(QueuePairEndpoint::restore(number, psn, lid, gid, bytes[26], bytes[27], Ghost(sent)))
}

// This package exists only in proofs. On the wire only Header and user bytes exist.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub(super) tracked struct Attachment<M: ChannelAttachment = ()> {
    pub(super) ghost endpoint: Option<QueuePairEndpoint>,
    pub(super) ghost header: Header,
    pub(super) ghost user: Seq<u8>,
    pub(super) ghost kind: Option<GrantKind>,
    pub(super) ghost cap: Option<RemoteCapability>,
    pub(super) tracked statement: Statement,
    pub(super) tracked slot: Option<SlotToken>,
    pub(super) tracked share: Option<ShareCap>,
    pub(super) tracked extra: Option<M>,
}

impl<M: ChannelAttachment> Attachment<M> {
    pub(super) open spec fn valid(self, id: InstanceId, sender: NodeId) -> bool {
        let h = self.header;
        &&& if h.op == ENDPOINT {
            self.endpoint matches Some(e) && e.node() == sender && endpoint_matches(e, self.user)
        } else {
            self.endpoint is None
        }
        &&& if h.op == ENDPOINT {
            self.extra is None
        } else {
            self.extra matches Some(m) && m.valid(
                id,
                sender,
                h.op,
                h.tag,
                self.user,
                self.kind,
                self.cap,
                share_frontier(self.share),
                self.statement,
            )
        }
        &&& h.valid() && self.statement.valid(id, sender)
        &&& if h.op >= NOTICE {
            self.kind is None && self.slot is None && self.share is None
        } else if h.op == READ_GRANT || h.op == READ_RELEASE {
            self.kind matches Some(GrantKind::ReadShare { slot, share_id }) && self.slot is None
                && (self.share matches Some(sh) && share_matches(
                sh,
                id,
                slot,
                share_id,
                self.statement.closure(),
            ))
        } else {
            self.kind matches Some(GrantKind::OwnedSlot { slot }) && self.share is None && (
            self.slot matches Some(tok) && slot_matches(tok, id, slot, self.statement.closure()))
        }
        &&& if h.grant() {
            self.cap matches Some(c) && c.well_formed() && c.slot_ref().loc.node == sender
                && c.descriptor() == (h.addr, h.len, h.rkey) && (if h.op == READ_GRANT {
                c.share() == Some(self.kind->Some_0->ReadShare_share_id) && c.access()
                    == RemoteAccess::ReadOnly && self.kind->Some_0->ReadShare_slot == c.slot_ref()
            } else {
                c.access() == RemoteAccess::ReadWrite && self.kind->Some_0->OwnedSlot_slot
                    == c.slot_ref()
            })
        } else {
            self.cap is None
        }
    }
}

} // verus!
