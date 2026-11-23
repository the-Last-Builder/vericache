//! Verified Milestone 3 RDMA peer protocol: LE binary codecs and constructive
//! `VeriCacheAttachment` implementing `ChannelAttachment`.
use super::*;
#[cfg(verus_only)]
use crate::provenance::{
    certificate_valid_in, ledger_consistent_with_history, replica_valid, revoked_submap,
    KnowledgeLedger, TransferHistory, TransferRun, ValueCertificate,
};
#[cfg(verus_only)]
use foundation::tracked::Frontier;
#[cfg(verus_only)]
use transfer::context::{cluster_version_instance_id, SubmittedMap, SucceededMap};
use transfer::context::{RevokedMap, VersionBinding};
#[cfg(verus_only)]
use transfer::domain::{NodeId, SlotLoc, TransferDomain, TransferEvent, TransferShare};
#[cfg(verus_only)]
use transfer::rdma::RemoteCapability;
use transfer::transport::ChannelAttachment;
#[cfg(verus_only)]
use transfer::transport::{GrantKind, GrantTag, Statement};
#[cfg(verus_only)]
use vstd::bytes::{
    lemma_auto_spec_u16_to_from_le_bytes, lemma_auto_spec_u64_to_from_le_bytes,
    spec_u16_to_le_bytes, spec_u64_to_le_bytes,
};
use vstd::bytes::{u16_from_le_bytes, u16_to_le_bytes, u64_from_le_bytes, u64_to_le_bytes};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::seq::group_seq_axioms,
    vstd::seq_lib::group_seq_properties,
};

/// Wire protocol magic isolating Milestone 3 `VeriCacheAttachment` sessions (`b"VCM3"`).
pub const PROTOCOL_MAGIC: u32 = 0x5643_4d33;

/// Control-plane notice payloads carried inside `Channel` `NOTICE` (`op == 5`) frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlPayload {
    FetchRequest { request_id: u64, key: ObjectKey },
    FetchMiss { request_id: u64 },
    RevokeNotice { grant_tag: u64 },
    RevokeAck { grant_tag: u64 },
    ReconnectBarrier { epoch: u64 },
}

/// Wire metadata carried inside `READ_GRANT` (`op == 1`) `user_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadGrantMeta {
    pub request_id: u64,
    pub key: ObjectKey,
    pub version: VersionId,
    pub frame_size: u64,
}

/// Wire metadata carried inside `READ_RELEASE` (`op == 2`) `user_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadReleaseMeta {
    pub request_id: u64,
}

pub const CONTROL_TAG_FETCH_REQUEST: u8 = 1;

pub const CONTROL_TAG_FETCH_MISS: u8 = 2;

pub const CONTROL_TAG_REVOKE_NOTICE: u8 = 3;

pub const CONTROL_TAG_REVOKE_ACK: u8 = 4;

pub const CONTROL_TAG_RECONNECT_BARRIER: u8 = 5;

pub const FETCH_REQUEST_LEN: usize = 41;

pub const RECONNECT_BARRIER_LEN: usize = 9;

pub const READ_GRANT_META_LEN: usize = 58;

pub const READ_RELEASE_META_LEN: usize = 8;

pub open spec fn encode_key_spec(k: ObjectKey) -> Seq<u8> {
    spec_u64_to_le_bytes(k.0[0]) + spec_u64_to_le_bytes(k.0[1]) + spec_u64_to_le_bytes(k.0[2])
        + spec_u64_to_le_bytes(k.0[3])
}

pub open spec fn encode_control_payload_spec(p: ControlPayload) -> Seq<u8> {
    match p {
        ControlPayload::FetchRequest { request_id, key } => seq![CONTROL_TAG_FETCH_REQUEST]
            + spec_u64_to_le_bytes(request_id) + encode_key_spec(key),
        ControlPayload::FetchMiss { request_id } => seq![CONTROL_TAG_FETCH_MISS]
            + spec_u64_to_le_bytes(request_id),
        ControlPayload::RevokeNotice { grant_tag } => seq![CONTROL_TAG_REVOKE_NOTICE]
            + spec_u64_to_le_bytes(grant_tag),
        ControlPayload::RevokeAck { grant_tag } => seq![CONTROL_TAG_REVOKE_ACK]
            + spec_u64_to_le_bytes(grant_tag),
        ControlPayload::ReconnectBarrier { epoch } => seq![CONTROL_TAG_RECONNECT_BARRIER]
            + spec_u64_to_le_bytes(epoch),
    }
}

pub open spec fn encode_read_grant_meta_spec(m: ReadGrantMeta) -> Seq<u8> {
    spec_u64_to_le_bytes(m.request_id) + encode_key_spec(m.key) + spec_u16_to_le_bytes(
        m.version.origin_node,
    ) + spec_u64_to_le_bytes(m.version.seq) + spec_u64_to_le_bytes(m.frame_size)
}

pub open spec fn encode_read_release_meta_spec(m: ReadReleaseMeta) -> Seq<u8> {
    spec_u64_to_le_bytes(m.request_id)
}

fn push_u16_le(buf: &mut Vec<u8>, v: u16)
    ensures
        final(buf)@ == old(buf)@ + spec_u16_to_le_bytes(v),
{
    let arr = u16_to_le_bytes(v);
    let ghost start = buf@;
    buf.push(arr[0]);
    buf.push(arr[1]);
    proof {
        assert(buf@ =~= start + arr@);
    }
}

fn push_u64_le(buf: &mut Vec<u8>, v: u64)
    ensures
        final(buf)@ == old(buf)@ + spec_u64_to_le_bytes(v),
{
    let arr = u64_to_le_bytes(v);
    let ghost start = buf@;
    buf.push(arr[0]);
    buf.push(arr[1]);
    buf.push(arr[2]);
    buf.push(arr[3]);
    buf.push(arr[4]);
    buf.push(arr[5]);
    buf.push(arr[6]);
    buf.push(arr[7]);
    proof {
        assert(buf@ =~= start + arr@);
    }
}

fn push_key_le(buf: &mut Vec<u8>, key: ObjectKey)
    ensures
        final(buf)@ == old(buf)@ + encode_key_spec(key),
{
    let ghost start = buf@;
    push_u64_le(buf, key.0[0]);
    push_u64_le(buf, key.0[1]);
    push_u64_le(buf, key.0[2]);
    push_u64_le(buf, key.0[3]);
    proof {
        assert(buf@ =~= start + encode_key_spec(key));
    }
}

fn read_u16_le(bytes: &[u8], off: usize) -> (v: u16)
    requires
        off + 2 <= bytes@.len(),
    ensures
        spec_u16_to_le_bytes(v) == bytes@.subrange(off as int, (off + 2) as int),
{
    proof {
        lemma_auto_spec_u16_to_from_le_bytes();
    }
    let arr: [u8; 2] = [bytes[off], bytes[off + 1]];
    let v = u16_from_le_bytes(&arr);
    proof {
        assert(arr@ =~= bytes@.subrange(off as int, (off + 2) as int));
    }
    v
}

fn read_u64_le(bytes: &[u8], off: usize) -> (v: u64)
    requires
        off + 8 <= bytes@.len(),
    ensures
        spec_u64_to_le_bytes(v) == bytes@.subrange(off as int, (off + 8) as int),
{
    proof {
        lemma_auto_spec_u64_to_from_le_bytes();
    }
    let arr: [u8; 8] = [
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
        bytes[off + 4],
        bytes[off + 5],
        bytes[off + 6],
        bytes[off + 7],
    ];
    let v = u64_from_le_bytes(&arr);
    proof {
        assert(arr@ =~= bytes@.subrange(off as int, (off + 8) as int));
    }
    v
}

fn read_key_le(bytes: &[u8], off: usize) -> (k: ObjectKey)
    requires
        off <= usize::MAX - 32,
        off + 32 <= bytes@.len(),
    ensures
        encode_key_spec(k) == bytes@.subrange(off as int, (off + 32) as int),
{
    let w0 = read_u64_le(bytes, off);
    let w1 = read_u64_le(bytes, off + 8);
    let w2 = read_u64_le(bytes, off + 16);
    let w3 = read_u64_le(bytes, off + 24);
    let k = ObjectKey([w0, w1, w2, w3]);
    proof {
        assert(encode_key_spec(k) =~= bytes@.subrange(off as int, (off + 32) as int));
    }
    k
}

pub proof fn lemma_encode_key_injective(a: ObjectKey, b: ObjectKey)
    requires
        encode_key_spec(a) == encode_key_spec(b),
    ensures
        a == b,
{
    lemma_auto_spec_u64_to_from_le_bytes();
    assert(encode_key_spec(a).subrange(0, 8) =~= spec_u64_to_le_bytes(a.0[0]));
    assert(encode_key_spec(b).subrange(0, 8) =~= spec_u64_to_le_bytes(b.0[0]));
    assert(encode_key_spec(a).subrange(8, 16) =~= spec_u64_to_le_bytes(a.0[1]));
    assert(encode_key_spec(b).subrange(8, 16) =~= spec_u64_to_le_bytes(b.0[1]));
    assert(encode_key_spec(a).subrange(16, 24) =~= spec_u64_to_le_bytes(a.0[2]));
    assert(encode_key_spec(b).subrange(16, 24) =~= spec_u64_to_le_bytes(b.0[2]));
    assert(encode_key_spec(a).subrange(24, 32) =~= spec_u64_to_le_bytes(a.0[3]));
    assert(encode_key_spec(b).subrange(24, 32) =~= spec_u64_to_le_bytes(b.0[3]));
    assert(a.0 =~= b.0);
}

pub fn encode_control_payload(payload: ControlPayload) -> (out: Vec<u8>)
    ensures
        out@ == encode_control_payload_spec(payload),
{
    let mut out: Vec<u8> = Vec::new();
    match payload {
        ControlPayload::FetchRequest { request_id, key } => {
            out.push(CONTROL_TAG_FETCH_REQUEST);
            push_u64_le(&mut out, request_id);
            push_key_le(&mut out, key);
        },
        ControlPayload::FetchMiss { request_id } => {
            out.push(CONTROL_TAG_FETCH_MISS);
            push_u64_le(&mut out, request_id);
        },
        ControlPayload::RevokeNotice { grant_tag } => {
            out.push(CONTROL_TAG_REVOKE_NOTICE);
            push_u64_le(&mut out, grant_tag);
        },
        ControlPayload::RevokeAck { grant_tag } => {
            out.push(CONTROL_TAG_REVOKE_ACK);
            push_u64_le(&mut out, grant_tag);
        },
        ControlPayload::ReconnectBarrier { epoch } => {
            out.push(CONTROL_TAG_RECONNECT_BARRIER);
            push_u64_le(&mut out, epoch);
        },
    }
    proof {
        assert(out@ =~= encode_control_payload_spec(payload));
    }
    out
}

pub proof fn lemma_encode_control_payload_injective(a: ControlPayload, b: ControlPayload)
    requires
        encode_control_payload_spec(a) == encode_control_payload_spec(b),
    ensures
        a == b,
{
    lemma_auto_spec_u64_to_from_le_bytes();
    assert(encode_control_payload_spec(a)[0] == encode_control_payload_spec(b)[0]);
    match (a, b) {
        (
            ControlPayload::FetchRequest { request_id: r1, key: k1 },
            ControlPayload::FetchRequest { request_id: r2, key: k2 },
        ) => {
            assert(encode_control_payload_spec(a).subrange(1, 9) =~= spec_u64_to_le_bytes(r1));
            assert(encode_control_payload_spec(b).subrange(1, 9) =~= spec_u64_to_le_bytes(r2));
            assert(encode_control_payload_spec(a).subrange(9, 41) =~= encode_key_spec(k1));
            assert(encode_control_payload_spec(b).subrange(9, 41) =~= encode_key_spec(k2));
            lemma_encode_key_injective(k1, k2);
        },
        (
            ControlPayload::FetchMiss { request_id: r1 },
            ControlPayload::FetchMiss { request_id: r2 },
        ) => {
            assert(encode_control_payload_spec(a).subrange(1, 9) =~= spec_u64_to_le_bytes(r1));
            assert(encode_control_payload_spec(b).subrange(1, 9) =~= spec_u64_to_le_bytes(r2));
        },
        (
            ControlPayload::RevokeNotice { grant_tag: t1 },
            ControlPayload::RevokeNotice { grant_tag: t2 },
        ) => {
            assert(encode_control_payload_spec(a).subrange(1, 9) =~= spec_u64_to_le_bytes(t1));
            assert(encode_control_payload_spec(b).subrange(1, 9) =~= spec_u64_to_le_bytes(t2));
        },
        (
            ControlPayload::RevokeAck { grant_tag: t1 },
            ControlPayload::RevokeAck { grant_tag: t2 },
        ) => {
            assert(encode_control_payload_spec(a).subrange(1, 9) =~= spec_u64_to_le_bytes(t1));
            assert(encode_control_payload_spec(b).subrange(1, 9) =~= spec_u64_to_le_bytes(t2));
        },
        (
            ControlPayload::ReconnectBarrier { epoch: e1 },
            ControlPayload::ReconnectBarrier { epoch: e2 },
        ) => {
            assert(encode_control_payload_spec(a).subrange(1, 9) =~= spec_u64_to_le_bytes(e1));
            assert(encode_control_payload_spec(b).subrange(1, 9) =~= spec_u64_to_le_bytes(e2));
        },
        _ => {
            assert(encode_control_payload_spec(a)[0] == match a {
                ControlPayload::FetchRequest { .. } => CONTROL_TAG_FETCH_REQUEST,
                ControlPayload::FetchMiss { .. } => CONTROL_TAG_FETCH_MISS,
                ControlPayload::RevokeNotice { .. } => CONTROL_TAG_REVOKE_NOTICE,
                ControlPayload::RevokeAck { .. } => CONTROL_TAG_REVOKE_ACK,
                ControlPayload::ReconnectBarrier { .. } => CONTROL_TAG_RECONNECT_BARRIER,
            });
            assert(encode_control_payload_spec(b)[0] == match b {
                ControlPayload::FetchRequest { .. } => CONTROL_TAG_FETCH_REQUEST,
                ControlPayload::FetchMiss { .. } => CONTROL_TAG_FETCH_MISS,
                ControlPayload::RevokeNotice { .. } => CONTROL_TAG_REVOKE_NOTICE,
                ControlPayload::RevokeAck { .. } => CONTROL_TAG_REVOKE_ACK,
                ControlPayload::ReconnectBarrier { .. } => CONTROL_TAG_RECONNECT_BARRIER,
            });
        },
    }
}

pub fn decode_control_payload(bytes: &[u8]) -> (r: Option<ControlPayload>)
    ensures
        r matches Some(p) ==> bytes@ == encode_control_payload_spec(p),
        forall|p: ControlPayload|
            bytes@ == #[trigger] encode_control_payload_spec(p) ==> r == Some(p),
{
    proof {
        lemma_auto_spec_u64_to_from_le_bytes();
    }
    if bytes.len() == 0 {
        proof {
            assert forall|p: ControlPayload|
                bytes@ == #[trigger] encode_control_payload_spec(p) implies false by {
                assert(encode_control_payload_spec(p).len() > 0);
            }
        }
        return None;
    }
    let tag = bytes[0];
    let res = if tag == CONTROL_TAG_FETCH_REQUEST && bytes.len() == FETCH_REQUEST_LEN {
        let request_id = read_u64_le(bytes, 1);
        let key = read_key_le(bytes, 9);
        Some(ControlPayload::FetchRequest { request_id, key })
    } else if tag == CONTROL_TAG_FETCH_MISS && bytes.len() == 9 {
        let request_id = read_u64_le(bytes, 1);
        Some(ControlPayload::FetchMiss { request_id })
    } else if tag == CONTROL_TAG_REVOKE_NOTICE && bytes.len() == 9 {
        let grant_tag = read_u64_le(bytes, 1);
        Some(ControlPayload::RevokeNotice { grant_tag })
    } else if tag == CONTROL_TAG_REVOKE_ACK && bytes.len() == 9 {
        let grant_tag = read_u64_le(bytes, 1);
        Some(ControlPayload::RevokeAck { grant_tag })
    } else if tag == CONTROL_TAG_RECONNECT_BARRIER && bytes.len() == RECONNECT_BARRIER_LEN {
        let epoch = read_u64_le(bytes, 1);
        Some(ControlPayload::ReconnectBarrier { epoch })
    } else {
        None
    };
    proof {
        if let Some(p) = res {
            assert(bytes@ =~= encode_control_payload_spec(p));
        }
        assert forall|p: ControlPayload|
            bytes@ == #[trigger] encode_control_payload_spec(p) implies res == Some(p) by {
            match res {
                Some(q) => {
                    lemma_encode_control_payload_injective(p, q);
                },
                None => {
                    assert(bytes@[0] == encode_control_payload_spec(p)[0]);
                    assert(bytes@.len() == encode_control_payload_spec(p).len());
                },
            }
        }
    }
    res
}

pub fn encode_read_grant_meta(meta: ReadGrantMeta) -> (out: Vec<u8>)
    ensures
        out@ == encode_read_grant_meta_spec(meta),
        out@.len() == READ_GRANT_META_LEN,
{
    let mut out: Vec<u8> = Vec::with_capacity(READ_GRANT_META_LEN);
    push_u64_le(&mut out, meta.request_id);
    push_key_le(&mut out, meta.key);
    push_u16_le(&mut out, meta.version.origin_node);
    push_u64_le(&mut out, meta.version.seq);
    push_u64_le(&mut out, meta.frame_size);
    proof {
        lemma_auto_spec_u16_to_from_le_bytes();
        lemma_auto_spec_u64_to_from_le_bytes();
        assert(out@ =~= encode_read_grant_meta_spec(meta));
    }
    out
}

pub proof fn lemma_encode_read_grant_meta_injective(a: ReadGrantMeta, b: ReadGrantMeta)
    requires
        encode_read_grant_meta_spec(a) == encode_read_grant_meta_spec(b),
    ensures
        a == b,
{
    lemma_auto_spec_u16_to_from_le_bytes();
    lemma_auto_spec_u64_to_from_le_bytes();
    assert(encode_read_grant_meta_spec(a).subrange(0, 8) =~= spec_u64_to_le_bytes(a.request_id));
    assert(encode_read_grant_meta_spec(b).subrange(0, 8) =~= spec_u64_to_le_bytes(b.request_id));
    assert(encode_read_grant_meta_spec(a).subrange(8, 40) =~= encode_key_spec(a.key));
    assert(encode_read_grant_meta_spec(b).subrange(8, 40) =~= encode_key_spec(b.key));
    lemma_encode_key_injective(a.key, b.key);
    assert(encode_read_grant_meta_spec(a).subrange(40, 42) =~= spec_u16_to_le_bytes(
        a.version.origin_node,
    ));
    assert(encode_read_grant_meta_spec(b).subrange(40, 42) =~= spec_u16_to_le_bytes(
        b.version.origin_node,
    ));
    assert(encode_read_grant_meta_spec(a).subrange(42, 50) =~= spec_u64_to_le_bytes(a.version.seq));
    assert(encode_read_grant_meta_spec(b).subrange(42, 50) =~= spec_u64_to_le_bytes(b.version.seq));
    assert(encode_read_grant_meta_spec(a).subrange(50, 58) =~= spec_u64_to_le_bytes(a.frame_size));
    assert(encode_read_grant_meta_spec(b).subrange(50, 58) =~= spec_u64_to_le_bytes(b.frame_size));
}

pub fn decode_read_grant_meta(bytes: &[u8]) -> (r: Option<ReadGrantMeta>)
    ensures
        r matches Some(m) ==> bytes@ == encode_read_grant_meta_spec(m),
        forall|m: ReadGrantMeta|
            bytes@ == #[trigger] encode_read_grant_meta_spec(m) ==> r == Some(m),
{
    proof {
        lemma_auto_spec_u16_to_from_le_bytes();
        lemma_auto_spec_u64_to_from_le_bytes();
    }
    if bytes.len() != READ_GRANT_META_LEN {
        proof {
            assert forall|m: ReadGrantMeta|
                bytes@ == #[trigger] encode_read_grant_meta_spec(m) implies false by {
                assert(encode_read_grant_meta_spec(m).len() == READ_GRANT_META_LEN);
            }
        }
        return None;
    }
    let request_id = read_u64_le(bytes, 0);
    let key = read_key_le(bytes, 8);
    let origin_node = read_u16_le(bytes, 40);
    let seq = read_u64_le(bytes, 42);
    let frame_size = read_u64_le(bytes, 50);
    let m = ReadGrantMeta { request_id, key, version: VersionId { origin_node, seq }, frame_size };
    proof {
        assert(bytes@ =~= encode_read_grant_meta_spec(m));
        assert forall|other: ReadGrantMeta|
            bytes@ == #[trigger] encode_read_grant_meta_spec(other) implies Some(m) == Some(
            other,
        ) by {
            lemma_encode_read_grant_meta_injective(m, other);
        }
    }
    Some(m)
}

pub fn encode_read_release_meta(meta: ReadReleaseMeta) -> (out: Vec<u8>)
    ensures
        out@ == encode_read_release_meta_spec(meta),
        out@.len() == READ_RELEASE_META_LEN,
{
    let mut out: Vec<u8> = Vec::with_capacity(READ_RELEASE_META_LEN);
    push_u64_le(&mut out, meta.request_id);
    proof {
        lemma_auto_spec_u64_to_from_le_bytes();
        assert(out@ =~= encode_read_release_meta_spec(meta));
    }
    out
}

pub proof fn lemma_encode_read_release_meta_injective(a: ReadReleaseMeta, b: ReadReleaseMeta)
    requires
        encode_read_release_meta_spec(a) == encode_read_release_meta_spec(b),
    ensures
        a == b,
{
    lemma_auto_spec_u64_to_from_le_bytes();
}

pub fn decode_read_release_meta(bytes: &[u8]) -> (r: Option<ReadReleaseMeta>)
    ensures
        r matches Some(m) ==> bytes@ == encode_read_release_meta_spec(m),
        forall|m: ReadReleaseMeta|
            bytes@ == #[trigger] encode_read_release_meta_spec(m) ==> r == Some(m),
{
    proof {
        lemma_auto_spec_u64_to_from_le_bytes();
    }
    if bytes.len() != READ_RELEASE_META_LEN {
        proof {
            assert forall|m: ReadReleaseMeta|
                bytes@ == #[trigger] encode_read_release_meta_spec(m) implies false by {
                assert(encode_read_release_meta_spec(m).len() == READ_RELEASE_META_LEN);
            }
        }
        return None;
    }
    let request_id = read_u64_le(bytes, 0);
    let m = ReadReleaseMeta { request_id };
    proof {
        assert(bytes@ =~= encode_read_release_meta_spec(m));
        assert forall|other: ReadReleaseMeta|
            bytes@ == #[trigger] encode_read_release_meta_spec(other) implies Some(m) == Some(
            other,
        ) by {
            lemma_encode_read_release_meta_injective(m, other);
        }
    }
    Some(m)
}

/// Sender-independent consistency of a transferred `Statement` + `RevokedMap`
/// with a concrete `(h, run)`. Matches `ledger_consistent_with_history` when
/// `facts == ledger.citations(node)`.
pub open spec fn statement_consistent_with_history(
    submitted: SubmittedMap,
    succeeded: SucceededMap,
    revoked: RevokedMap,
    facts: ISet<TransferEvent>,
    h: TransferHistory,
    run: TransferRun,
) -> bool {
    &&& forall|e: TransferEvent| #[trigger]
        submitted.dom().contains(e) ==> {
            let info = submitted.map()[e];
            &&& h.events.contains(e)
            &&& (h.access)(e) == info.access
            &&& (h.effect)(e) == info.effect
            &&& (h.read_guards)(e) == info.read_guards
            &&& forall|p| #[trigger] info.preds.contains(p) == (h.happens_before)(p, e)
        }
    &&& forall|k: (SlotLoc, TransferShare)| #[trigger]
        revoked.dom().contains(k) ==> h.revoked.dom().contains(k) && h.revoked[k]
            == revoked.map()[k]
    &&& forall|e: TransferEvent|
        e.source.succeeds_on_observe() && #[trigger] facts.contains(e) ==> submitted.dom().contains(
            e,
        ) && (run.succeeded)(e) && (run.end)(e).is_some()
    &&& forall|e: TransferEvent|
        !e.source.succeeds_on_observe() && #[trigger] succeeded.contains(e)
            ==> submitted.dom().contains(e) && (run.succeeded)(e) && (run.end)(e).is_some()
}

/// Sender-independent value-certificate validity carried in `VeriCacheAttachment::ReadGrant`.
pub open spec fn statement_replica_valid(
    submitted: SubmittedMap,
    succeeded: SucceededMap,
    revoked: RevokedMap,
    facts: ISet<TransferEvent>,
    cert: ValueCertificate,
) -> bool {
    &&& cert.slot.well_formed()
    &&& cert.bytes.len() == cert.slot.sub_len
    &&& submitted.dom().contains(cert.writer)
    &&& facts.contains(cert.writer)
    &&& ((cert.writer.source.succeeds_on_observe() && facts.contains(cert.writer))
        || succeeded.contains(cert.writer))
    &&& facts.subset_of(submitted.dom())
    &&& (forall|e: TransferEvent| #[trigger] succeeded.contains(e) ==> submitted.dom().contains(e))
    &&& forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && statement_consistent_with_history(submitted, succeeded, revoked, facts, h, run)
            ==> #[trigger] certificate_valid_in(cert, h, run)
}

/// Donor-side bridge: a locally valid `ValueCertificate` on `ledger` yields
/// `statement_replica_valid` for `(ledger.submitted, ledger.succeeded, ledger.revoked, ledger.citations(node))`.
pub proof fn lemma_replica_valid_to_statement(
    ledger: KnowledgeLedger,
    node: NodeId,
    cert: ValueCertificate,
)
    requires
        ledger.valid(ledger.submitted.instance_id(), node),
        replica_valid(ledger, node, cert),
        cert.slot.well_formed(),
        cert.bytes.len() == cert.slot.sub_len,
        ledger.citations(node).contains(cert.writer),
    ensures
        statement_replica_valid(
            ledger.submitted,
            ledger.succeeded,
            ledger.revoked,
            ledger.citations(node),
            cert,
        ),
{
    assert(ledger.citations(node).subset_of(ledger.submitted.dom()));
    assert forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && statement_consistent_with_history(
            ledger.submitted,
            ledger.succeeded,
            ledger.revoked,
            ledger.citations(node),
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) by {
        assert forall|e: TransferEvent| #[trigger]
            crate::provenance::event_succeeded(
                ledger,
                node,
                e,
            ) implies ledger.submitted.dom().contains(e) && (run.succeeded)(e) && (run.end)(
            e,
        ).is_some() by {
            if e.source.succeeds_on_observe() {
                assert(ledger.citations(node).contains(e));
            } else {
                assert(ledger.succeeded.contains(e));
            }
        }
        assert(ledger_consistent_with_history(ledger, node, h, run));
    }
}

/// Extensionality: `statement_replica_valid` depends only on `submitted.map()`.
pub proof fn lemma_statement_replica_valid_ext(
    sub1: SubmittedMap,
    sub2: SubmittedMap,
    succeeded: SucceededMap,
    revoked: RevokedMap,
    facts: ISet<TransferEvent>,
    cert: ValueCertificate,
)
    requires
        statement_replica_valid(sub1, succeeded, revoked, facts, cert),
        sub2.map() == sub1.map(),
    ensures
        statement_replica_valid(sub2, succeeded, revoked, facts, cert),
{
    assert forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && statement_consistent_with_history(
            sub2,
            succeeded,
            revoked,
            facts,
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) by {
        assert(statement_consistent_with_history(sub1, succeeded, revoked, facts, h, run));
    }
}

/// Extensionality: `statement_replica_valid` depends only on `revoked.map()`.
pub proof fn lemma_statement_replica_valid_revoked_ext(
    submitted: SubmittedMap,
    succeeded: SucceededMap,
    rev1: RevokedMap,
    rev2: RevokedMap,
    facts: ISet<TransferEvent>,
    cert: ValueCertificate,
)
    requires
        statement_replica_valid(submitted, succeeded, rev1, facts, cert),
        rev2.map() == rev1.map(),
    ensures
        statement_replica_valid(submitted, succeeded, rev2, facts, cert),
{
    assert forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && statement_consistent_with_history(
            submitted,
            succeeded,
            rev2,
            facts,
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) by {
        assert(statement_consistent_with_history(submitted, succeeded, rev1, facts, h, run));
    }
}

/// Receiver-side bridge: after merging the peer's `submitted`, `knows`, `succeeded`,
/// and `revoked` into `receiver_ledger`, `certificate_valid_in(cert, h, run)` holds
/// for every `(h, run)` consistent with `receiver_ledger`.
pub proof fn lemma_statement_valid_in_receiver_ledger(
    receiver_ledger: KnowledgeLedger,
    receiver_node: NodeId,
    submitted: SubmittedMap,
    succeeded: SucceededMap,
    revoked: RevokedMap,
    facts: ISet<TransferEvent>,
    cert: ValueCertificate,
)
    requires
        statement_replica_valid(submitted, succeeded, revoked, facts, cert),
        submitted.map().submap_of(receiver_ledger.submitted.map()),
        revoked_submap(revoked, receiver_ledger.revoked),
        succeeded.dom().subset_of(receiver_ledger.succeeded.dom()),
        facts.subset_of(receiver_ledger.citations(receiver_node)),
    ensures
        forall|h, run|
            foundation::history::reachable(h) && foundation::value::valid_run(h, run)
                && ledger_consistent_with_history(receiver_ledger, receiver_node, h, run)
                ==> #[trigger] certificate_valid_in(cert, h, run),
{
    assert forall|h, run|
        foundation::history::reachable(h) && foundation::value::valid_run(h, run)
            && ledger_consistent_with_history(
            receiver_ledger,
            receiver_node,
            h,
            run,
        ) implies #[trigger] certificate_valid_in(cert, h, run) by {
        assert forall|e: TransferEvent| #[trigger] submitted.dom().contains(e) implies {
            let info = submitted.map()[e];
            &&& h.events.contains(e)
            &&& (h.access)(e) == info.access
            &&& (h.effect)(e) == info.effect
            &&& (h.read_guards)(e) == info.read_guards
            &&& forall|p| #[trigger] info.preds.contains(p) == (h.happens_before)(p, e)
        } by {
            assert(receiver_ledger.submitted.dom().contains(e));
            assert(receiver_ledger.submitted.map()[e] == submitted.map()[e]);
        }
        assert forall|k: (SlotLoc, TransferShare)| #[trigger]
            revoked.dom().contains(k) implies h.revoked.dom().contains(k) && h.revoked[k]
            == revoked.map()[k] by {
            assert(receiver_ledger.revoked.dom().contains(k));
            assert(receiver_ledger.revoked.map()[k] == revoked.map()[k]);
        }
        assert forall|e: TransferEvent|
            e.source.succeeds_on_observe() && #[trigger] facts.contains(
                e,
            ) implies submitted.dom().contains(e) && (run.succeeded)(e) && (run.end)(
            e,
        ).is_some() by {
            assert(submitted.dom().contains(e));
            assert(receiver_ledger.submitted.dom().contains(e));
            assert(receiver_ledger.citations(receiver_node).contains(e));
            assert(crate::provenance::event_succeeded(receiver_ledger, receiver_node, e));
        }
        assert forall|e: TransferEvent|
            !e.source.succeeds_on_observe() && #[trigger] succeeded.contains(
                e,
            ) implies submitted.dom().contains(e) && (run.succeeded)(e) && (run.end)(
            e,
        ).is_some() by {
            assert(submitted.dom().contains(e));
            assert(receiver_ledger.submitted.dom().contains(e));
            assert(receiver_ledger.succeeded.contains(e));
            assert(crate::provenance::event_succeeded(receiver_ledger, receiver_node, e));
        }
        assert(statement_consistent_with_history(submitted, succeeded, revoked, facts, h, run));
    }
}

/// Constructive Milestone 3 `ChannelAttachment` carried alongside wire frames.
#[allow(inconsistent_fields)]
pub tracked enum VeriCacheAttachment {
    ReadGrant {
        ghost meta: ReadGrantMeta,
        ghost cert: ValueCertificate,
        tracked revoked: RevokedMap,
        tracked version_binding: Option<VersionBinding>,
    },
    ReadRelease { ghost meta: ReadReleaseMeta },
    Control { ghost payload: ControlPayload },
}

impl ChannelAttachment for VeriCacheAttachment {
    open spec fn protocol_magic() -> u32 {
        PROTOCOL_MAGIC
    }

    fn magic_exec() -> (m: u32) {
        PROTOCOL_MAGIC
    }

    open spec fn valid(
        &self,
        id: InstanceId,
        sender: NodeId,
        op: u8,
        _tag: GrantTag,
        user: Seq<u8>,
        kind: Option<GrantKind>,
        cap: Option<RemoteCapability>,
        frontier: Option<Frontier<TransferDomain>>,
        statement: Statement,
    ) -> bool {
        match self {
            VeriCacheAttachment::ReadGrant { meta, cert, revoked, version_binding } => {
                &&& op == 1
                &&& user == encode_read_grant_meta_spec(*meta)
                &&& revoked.instance_id() == id
                &&& kind matches Some(GrantKind::ReadShare { slot, .. }) && cert.slot == slot
                    && slot.is_full() && slot.loc.node == sender
                &&& cap matches Some(c) && c.slot_ref() == cert.slot
                &&& frontier matches Some(fr) && fr.last_writer == Some(cert.writer)
                &&& cert.bytes.len() == meta.frame_size
                &&& crate::provenance::complete_receipts_map(statement.submitted)
                &&& statement_replica_valid(
                    statement.submitted,
                    statement.succeeded,
                    *revoked,
                    statement.facts,
                    *cert,
                )
                &&& (version_binding matches Some(b) && b.instance_id()
                    == cluster_version_instance_id() && b.key() == (
                    meta.version.origin_node as NodeId,
                    meta.version.seq,
                ) && b.value() == cert.bytes)
            },
            VeriCacheAttachment::ReadRelease { meta } => {
                &&& op == 2
                &&& user == encode_read_release_meta_spec(*meta)
                &&& crate::provenance::complete_receipts_map(statement.submitted)
            },
            VeriCacheAttachment::Control { payload } => {
                &&& op == 5
                &&& user == encode_control_payload_spec(*payload)
            },
        }
    }
}

} // verus!
