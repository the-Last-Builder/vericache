//! Integration tests for `ShardAllocator` / `ShardSources` multi-shard
//! partitioning and non-blocking / integrity-enforcing `Channel` paths.
use std::cell::RefCell;
use std::collections::VecDeque;
use std::process::Command;
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};
use transfer::context::*;
#[cfg(verus_only)]
use transfer::domain::SPACE_HOST;
#[cfg_attr(not(verus_only), allow(unused_imports))]
use transfer::transport::Statement;
use transfer::transport::{
    ByteTransport, Channel, ChannelAttachment, ChannelError, EndpointOrFrame, GrantTag,
    QueuePairEndpoint, ReceivedFrame, RemoteCapability,
};
use vstd::prelude::*;

static NODE_ALLOCATOR: OnceLock<Mutex<ShardAllocator>> = OnceLock::new();

const CHILD_ENV: &str = "VERICACHE_SHARD_ALLOC_CHILD";
const TEST_MAGIC: u32 = 0x5643_5431;

#[derive(Clone)]
struct SharedQueue(Rc<RefCell<VecDeque<Vec<u8>>>>);

impl SharedQueue {
    fn new() -> Self {
        Self(Rc::new(RefCell::new(VecDeque::new())))
    }

    fn push(&self, frame: Vec<u8>) {
        self.0.borrow_mut().push_back(frame);
    }

    fn pop(&self) -> Option<Vec<u8>> {
        self.0.borrow_mut().pop_front()
    }
}

struct NonBlockingLoopback {
    inbound: SharedQueue,
    outbound: SharedQueue,
}

impl NonBlockingLoopback {
    fn pair() -> (Self, Self, SharedQueue, SharedQueue) {
        let a_to_b = SharedQueue::new();
        let b_to_a = SharedQueue::new();
        let a = Self {
            inbound: b_to_a.clone(),
            outbound: a_to_b.clone(),
        };
        let b = Self {
            inbound: a_to_b.clone(),
            outbound: b_to_a.clone(),
        };
        (a, b, b_to_a, a_to_b)
    }

    fn with_inbound(frames: Vec<Vec<u8>>) -> Self {
        let inbound = SharedQueue::new();
        for f in frames {
            inbound.push(f);
        }
        Self {
            inbound,
            outbound: SharedQueue::new(),
        }
    }
}

impl ByteTransport for NonBlockingLoopback {
    type Error = &'static str;

    fn send_frame(&mut self, frame: &[u8]) -> Result<(), Self::Error> {
        self.outbound.push(frame.to_vec());
        Ok(())
    }

    fn recv_frame(&mut self) -> Result<Vec<u8>, Self::Error> {
        self.inbound.pop().ok_or("would block")
    }

    fn try_recv_frame(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(self.inbound.pop())
    }
}

fn take_shard(quota: u64) -> ShardSources {
    let lock = NODE_ALLOCATOR
        .get_or_init(|| Mutex::new(ShardAllocator::from_node_context(NodeContext::bootstrap(0))));
    let mut guard = lock.lock().unwrap();
    guard.split_shard(quota).expect("shard quota available")
}

fn raw_legacy_read_grant(tag: u64, user: &[u8]) -> Vec<u8> {
    let mut frame = vec![1u8];
    frame.extend_from_slice(&tag.to_le_bytes());
    frame.extend_from_slice(&0x1020_3040_5060_7080u64.to_le_bytes());
    frame.extend_from_slice(&4096u64.to_le_bytes());
    frame.extend_from_slice(&0xfeed_beefu32.to_le_bytes());
    frame.extend_from_slice(user);
    frame
}

fn raw_legacy_endpoint(qp_num: u32) -> Vec<u8> {
    let mut frame = vec![6u8];
    frame.extend_from_slice(&qp_num.to_le_bytes());
    frame.extend_from_slice(&456u32.to_le_bytes());
    frame.extend_from_slice(&7u16.to_le_bytes());
    frame.extend_from_slice(&[0xab; 16]);
    frame.push(4);
    frame.push(1);
    assert_eq!(frame.len(), 29);
    frame
}

verus! {

struct TestProto;

impl ChannelAttachment for TestProto {
    open spec fn protocol_magic() -> u32 {
        TEST_MAGIC
    }

    fn magic_exec() -> (m: u32) {
        TEST_MAGIC
    }
}

fn split_four_from_node_context(ctx: NodeContext) -> (
    ShardSources,
    ShardSources,
    ShardSources,
    ShardSources,
) {
    let mut alloc = ShardAllocator::from_node_context(ctx);
    assert!(alloc.split_shard(0).is_none());
    let too_many = alloc.remaining_version_budget() + 1;
    assert!(alloc.split_shard(too_many).is_none());
    let s0 = alloc.split_shard(100).unwrap();
    let s1 = alloc.split_shard(200).unwrap();
    let s2 = alloc.split_shard(300).unwrap();
    let s3 = alloc.into_shard().unwrap();
    (s0, s1, s2, s3)
}

fn exercise_shard_tokens(shard: &mut ShardSources, seq: u64) -> Tracked<VersionBinding> {
    let tracked alloc = shard.alloc.borrow_mut();
    let tracked _slot = alloc.allocate_slot_in(alloc.space_for(SPACE_HOST, 0));
    let tracked _ev_src = shard.events.borrow_mut().mint(EventSourceKind::host());
    let tracked _share = shard.shares.borrow_mut().allocate();
    let tracked ver_tok = shard.versions.borrow_mut().take_seq(seq);
    let _key = (shard.node, seq);
    Tracked(shard.version_instance.borrow().bind(_key, Seq::empty(), ver_tok))
}

fn capture_read_grant(
    Tracked(instance): Tracked<&TransferInstance>,
    local_node: u64,
    peer_node: u64,
    tag: u64,
) -> (RemoteCapability, Tracked<ShareCap>) {
    let t = NonBlockingLoopback::with_inbound(vec![raw_legacy_read_grant(tag, b"cap")]);
    let mut ch = Channel::new(t, local_node, peer_node, Tracked(instance));
    let rec = ch.recv_remote_read_grant(Tracked(instance)).unwrap();
    let cap = rec.capability.unwrap();
    let tracked share = rec.share.get().tracked_unwrap();
    (cap, Tracked(share))
}

fn capture_endpoint(
    Tracked(instance): Tracked<&TransferInstance>,
    local_node: u64,
    peer_node: u64,
    qp_num: u32,
) -> QueuePairEndpoint {
    let t = NonBlockingLoopback::with_inbound(vec![raw_legacy_endpoint(qp_num)]);
    let mut ch = Channel::new(t, local_node, peer_node, Tracked(instance));
    let EndpointOrFrame::Endpoint(ep) = ch.recv_endpoint_or_any(Tracked(instance)).unwrap() else {
        panic!("expected endpoint");
    };
    ep
}

fn send_test_notice<T: ByteTransport>(
    ch: &mut Channel<T, TestProto>,
    user: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
) -> Result<(), ChannelError<T::Error>> {
    let tracked s = Statement::empty(instance.id());
    match ch.send_notice_with(
        user,
        Tracked(instance),
        Tracked(&s.submitted),
        Tracked(&s.knows),
        Tracked(&s.succeeded),
        Ghost(s.facts),
        Tracked(()),
    ) {
        Ok(()) => Ok(()),
        Err((e, _)) => Err(e),
    }
}

fn send_test_read_grant<T: ByteTransport>(
    ch: &mut Channel<T, TestProto>,
    tag: GrantTag,
    cap: &RemoteCapability,
    user: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(share): Tracked<ShareCap>,
) -> Result<(), ChannelError<T::Error>> {
    let tracked s = Statement::empty(instance.id());
    match ch.send_remote_read_grant_with(
        tag,
        cap,
        user,
        Tracked(instance),
        Tracked(&s.submitted),
        Tracked(&s.knows),
        Tracked(&s.succeeded),
        Ghost(s.facts),
        Tracked(share),
        Tracked(()),
    ) {
        Ok(()) => Ok(()),
        Err((e, _)) => Err(e),
    }
}

fn send_test_read_release<T: ByteTransport>(
    ch: &mut Channel<T, TestProto>,
    tag: GrantTag,
    user: &[u8],
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(share): Tracked<ShareCap>,
) -> Result<(), ChannelError<T::Error>> {
    let tracked s = Statement::empty(instance.id());
    match ch.send_remote_read_release_with(
        tag,
        user,
        Tracked(instance),
        Tracked(&s.submitted),
        Tracked(&s.knows),
        Tracked(&s.succeeded),
        Ghost(s.facts),
        Tracked(share),
        Tracked(()),
    ) {
        Ok(()) => Ok(()),
        Err((e, _)) => Err(e),
    }
}

fn run_shard_from_shard(big: ShardSources) {
    let base = big.version_start;
    let mut sub_alloc = ShardAllocator::from_shard(big);
    assert!(sub_alloc.split_shard(0).is_none());
    assert!(sub_alloc.split_shard(41).is_none());
    let mut a = sub_alloc.split_shard(10).unwrap();
    let mut b = sub_alloc.split_shard(10).unwrap();
    let mut c = sub_alloc.split_shard(10).unwrap();
    let mut d = sub_alloc.split_shard(10).unwrap();
    assert_eq!(sub_alloc.remaining_version_budget(), 0);
    assert!(sub_alloc.split_shard(1).is_none());
    assert!(sub_alloc.into_shard().is_none());
    assert_eq!((a.version_start, a.version_limit), (base, base + 10));
    assert_eq!((b.version_start, b.version_limit), (base + 10, base + 20));
    assert_eq!((c.version_start, c.version_limit), (base + 20, base + 30));
    assert_eq!((d.version_start, d.version_limit), (base + 30, base + 40));
    let sa = a.version_start;
    let sb = b.version_start;
    let sc = c.version_start;
    let sd = d.version_start;
    let _ = exercise_shard_tokens(&mut a, sa);
    let _ = exercise_shard_tokens(&mut b, sb);
    let _ = exercise_shard_tokens(&mut c, sc);
    let _ = exercise_shard_tokens(&mut d, sd);
}

fn run_try_recv_any_empty(
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    shard: ShardSources,
) {
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    let inst = clone_transfer_instance(Tracked(shard.instance.borrow()));
    let (local_t, peer_t, _, _) = NonBlockingLoopback::pair();
    let mut local = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        local_t,
        0,
        1,
        Tracked(inst.borrow()),
    );
    let mut peer = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        peer_t,
        1,
        0,
        Tracked(inst.borrow()),
    );

    let r0 = local.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r0, Ok(None)));
    assert!(!local.is_closed());

    send_test_notice(&mut peer, b"after-empty", Tracked(inst.borrow())).expect("send notice");
    match local.try_recv_any(Tracked(inst.borrow())).expect("try_recv_any succeeds") {
        Some(ReceivedFrame::Notice(msg)) => assert_eq!(msg.user_bytes, b"after-empty"),
        _ => panic!("expected Notice frame"),
    }
    assert!(!local.is_closed());
    let r1 = local.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r1, Ok(None)));
    assert!(!local.is_closed());
}

fn run_corrupted_checksum(
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    shard: ShardSources,
) {
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    let inst = clone_transfer_instance(Tracked(shard.instance.borrow()));
    let (_local_t, peer_t, peer_to_local, _) = NonBlockingLoopback::pair();
    let mut peer = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        peer_t,
        1,
        0,
        Tracked(inst.borrow()),
    );

    send_test_notice(&mut peer, b"integrity-check", Tracked(inst.borrow())).expect("send notice");
    let valid_wire = peer_to_local.pop().expect("wire bytes");

    // 1. Corrupt trailing CRC32 byte.
    let mut bad_crc = valid_wire.clone();
    let last = bad_crc.len() - 1;
    bad_crc[last] ^= 0x01;
    let mut ch_crc = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        NonBlockingLoopback::with_inbound(vec![bad_crc]),
        0,
        1,
        Tracked(inst.borrow()),
    );
    let r_crc = ch_crc.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_crc, Err(ChannelError::InvalidFrame)));
    assert!(ch_crc.is_closed());
    let r_crc_closed = ch_crc.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_crc_closed, Err(ChannelError::Closed)));

    // 2. Corrupt payload byte.
    let mut bad_payload = valid_wire.clone();
    bad_payload[6] ^= 0x80;
    let mut ch_payload = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        NonBlockingLoopback::with_inbound(vec![bad_payload]),
        0,
        1,
        Tracked(inst.borrow()),
    );
    let r_payload = ch_payload.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_payload, Err(ChannelError::InvalidFrame)));
    assert!(ch_payload.is_closed());

    // 3. Truncated frame shorter than 4-byte CRC32 trailer.
    let mut ch_short = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        NonBlockingLoopback::with_inbound(vec![valid_wire[..3].to_vec()]),
        0,
        1,
        Tracked(inst.borrow()),
    );
    let r_short = ch_short.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_short, Err(ChannelError::InvalidFrame)));
    assert!(ch_short.is_closed());
}

fn run_duplicate_read_grant(
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    shard: ShardSources,
) {
    #[cfg_attr(not(verus_only), allow(unused_variables))]
    let inst = clone_transfer_instance(Tracked(shard.instance.borrow()));
    let (local_t, peer_t, peer_to_local, _) = NonBlockingLoopback::pair();
    let mut local = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        local_t,
        0,
        1,
        Tracked(inst.borrow()),
    );
    let mut peer = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        peer_t,
        1,
        0,
        Tracked(inst.borrow()),
    );

    let (cap, share) = capture_read_grant(Tracked(inst.borrow()), 0, 1, 42);
    send_test_read_grant(&mut peer, 42, &cap, b"grant-42", Tracked(inst.borrow()), share).expect(
        "send grant",
    );
    let wire = peer_to_local.pop().expect("grant wire frame");
    peer_to_local.push(wire.clone());
    peer_to_local.push(wire);

    match local.try_recv_any(Tracked(inst.borrow())).expect("first READ_GRANT accepted") {
        Some(ReceivedFrame::ReadGrant(g)) => {
            assert_eq!(g.tag, 42);
            assert_eq!(g.user_bytes, b"grant-42");
        },
        _ => panic!("expected ReadGrant"),
    }
    assert!(!local.is_closed());

    let r_dup = local.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_dup, Err(ChannelError::InvalidFrame)));
    assert!(local.is_closed());
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn run_forged_and_replayed_release(shard: ShardSources) {
    let inst = clone_transfer_instance(Tracked(shard.instance.borrow()));

    // 1. Unknown/forged READ_RELEASE tag (never sent by `local`) is rejected.
    let (donor_t, peer_t, peer_to_donor, _) = NonBlockingLoopback::pair();
    let mut helper_donor = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        donor_t,
        0,
        1,
        Tracked(inst.borrow()),
    );
    let mut helper_peer = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        peer_t,
        1,
        0,
        Tracked(inst.borrow()),
    );
    let (cap99, share99) = capture_read_grant(Tracked(inst.borrow()), 1, 0, 99);
    send_test_read_grant(
        &mut helper_donor,
        99,
        &cap99,
        b"grant-99",
        Tracked(inst.borrow()),
        share99,
    ).expect("helper donor sends grant");
    let ReceivedFrame::ReadGrant(rec99) = helper_peer.recv_any(Tracked(inst.borrow())).expect(
        "helper peer receives grant",
    ) else {
        panic!("expected ReadGrant");
    };
    send_test_read_release(
        &mut helper_peer,
        99,
        b"forged-rel",
        Tracked(inst.borrow()),
        Tracked(rec99.share.get().tracked_unwrap()),
    ).expect("helper peer builds READ_RELEASE");
    let forged_release_wire = peer_to_donor.pop().expect("forged release wire");

    let mut fresh_local = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        NonBlockingLoopback::with_inbound(vec![forged_release_wire]),
        0,
        1,
        Tracked(inst.borrow()),
    );
    let r_forged = fresh_local.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_forged, Err(ChannelError::InvalidFrame)));
    assert!(fresh_local.is_closed());

    // 2. Replayed second READ_RELEASE for the same tag is also rejected.
    let (local_t, peer2_t, peer_to_local, _) = NonBlockingLoopback::pair();
    let mut local = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        local_t,
        0,
        1,
        Tracked(inst.borrow()),
    );
    let mut peer = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        peer2_t,
        1,
        0,
        Tracked(inst.borrow()),
    );
    let (cap88, share88) = capture_read_grant(Tracked(inst.borrow()), 1, 0, 88);
    send_test_read_grant(
        &mut local,
        88,
        &cap88,
        b"grant-88",
        Tracked(inst.borrow()),
        share88,
    ).expect("local sends grant");
    let ReceivedFrame::ReadGrant(rec88) = peer.recv_any(Tracked(inst.borrow())).expect(
        "peer receives grant",
    ) else {
        panic!("expected ReadGrant");
    };
    send_test_read_release(
        &mut peer,
        88,
        b"rel-88",
        Tracked(inst.borrow()),
        Tracked(rec88.share.get().tracked_unwrap()),
    ).expect("peer sends release");
    let rel_wire = peer_to_local.pop().expect("release wire");
    peer_to_local.push(rel_wire.clone());
    peer_to_local.push(rel_wire);

    match local.try_recv_any(Tracked(inst.borrow())).expect("first READ_RELEASE accepted") {
        Some(ReceivedFrame::ReadRelease(rel)) => {
            assert_eq!(rel.tag, 88);
            assert_eq!(rel.user_bytes, b"rel-88");
        },
        _ => panic!("expected ReadRelease"),
    }
    assert!(!local.is_closed());
    let r_replay = local.try_recv_any(Tracked(inst.borrow()));
    assert!(matches!(r_replay, Err(ChannelError::InvalidFrame)));
    assert!(local.is_closed());
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn run_interleaved_endpoint_or_any(shard: ShardSources) {
    let inst = clone_transfer_instance(Tracked(shard.instance.borrow()));
    let (local_t, peer_t, _, _) = NonBlockingLoopback::pair();
    let mut local = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        local_t,
        0,
        1,
        Tracked(inst.borrow()),
    );
    let mut peer = Channel::<NonBlockingLoopback, TestProto>::new_with_protocol(
        peer_t,
        1,
        0,
        Tracked(inst.borrow()),
    );

    let (cap55, share55) = capture_read_grant(Tracked(inst.borrow()), 1, 0, 55);
    let local_ep = capture_endpoint(Tracked(inst.borrow()), 1, 0, 111);
    let peer_ep = capture_endpoint(Tracked(inst.borrow()), 0, 1, 222);

    send_test_read_grant(
        &mut local,
        55,
        &cap55,
        b"grant-55",
        Tracked(inst.borrow()),
        share55,
    ).expect("send grant 55");
    let Some(ReceivedFrame::ReadGrant(rec55)) = peer.try_recv_any(Tracked(inst.borrow())).expect(
        "recv grant 55",
    ) else {
        panic!("expected ReadGrant");
    };

    // Local advertises its new endpoint first without blocking for peer's endpoint.
    local.send_endpoint(&local_ep).expect("send_endpoint succeeds");
    let r_ep_empty = local.try_recv_endpoint_or_any(Tracked(inst.borrow()));
    assert!(matches!(r_ep_empty, Ok(None)));
    assert!(!local.is_closed());

    // Peer emits pre-barrier Notice and ReadRelease before advertising its Endpoint.
    send_test_notice(&mut peer, b"pre-barrier-notice", Tracked(inst.borrow())).expect(
        "send notice",
    );
    send_test_read_release(
        &mut peer,
        55,
        b"pre-barrier-release",
        Tracked(inst.borrow()),
        Tracked(rec55.share.get().tracked_unwrap()),
    ).expect("send read release");
    peer.send_endpoint(&peer_ep).expect("peer send_endpoint succeeds");

    match local.try_recv_endpoint_or_any(Tracked(inst.borrow())).expect("drain notice") {
        Some(EndpointOrFrame::Frame(ReceivedFrame::Notice(n))) => {
            assert_eq!(n.user_bytes, b"pre-barrier-notice");
        },
        _ => panic!("expected Notice before Endpoint"),
    }
    match local.try_recv_endpoint_or_any(Tracked(inst.borrow())).expect("drain release") {
        Some(EndpointOrFrame::Frame(ReceivedFrame::ReadRelease(r))) => {
            assert_eq!(r.tag, 55);
            assert_eq!(r.user_bytes, b"pre-barrier-release");
        },
        _ => panic!("expected ReadRelease before Endpoint"),
    }
    match local.try_recv_endpoint_or_any(Tracked(inst.borrow())).expect("receive peer endpoint") {
        Some(EndpointOrFrame::Endpoint(_ep)) => {},
        _ => panic!("expected Endpoint"),
    }
    let r_ep_done = local.try_recv_endpoint_or_any(Tracked(inst.borrow()));
    assert!(matches!(r_ep_done, Ok(None)));
    assert!(!local.is_closed());

    // Peer also sees local's advertised endpoint cleanly via blocking recv_endpoint_or_any.
    match peer.recv_endpoint_or_any(Tracked(inst.borrow())).expect("peer receives local endpoint") {
        EndpointOrFrame::Endpoint(_ep) => {},
        _ => panic!("expected Endpoint on peer"),
    }
}

} // verus!
#[test]
fn shard_allocator_splits_four_disjoint_shards() {
    if std::env::var(CHILD_ENV).as_deref() == Ok("1") {
        let (mut s0, mut s1, mut s2, mut s3) =
            split_four_from_node_context(NodeContext::bootstrap(0));
        assert_eq!((s0.version_start, s0.version_limit), (1, 101));
        assert_eq!((s1.version_start, s1.version_limit), (101, 301));
        assert_eq!((s2.version_start, s2.version_limit), (301, 601));
        assert_eq!(
            (s3.version_start, s3.version_limit),
            (601, VERSION_SEQ_LIMIT)
        );
        let (v0, v1, v2, v3) = (
            s0.version_start,
            s1.version_start,
            s2.version_start,
            s3.version_start,
        );
        let _ = exercise_shard_tokens(&mut s0, v0);
        let _ = exercise_shard_tokens(&mut s1, v1);
        let _ = exercise_shard_tokens(&mut s2, v2);
        let _ = exercise_shard_tokens(&mut s3, v3);
        return;
    }

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "shard_allocator_splits_four_disjoint_shards",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .status()
        .expect("spawn child process");
    assert!(status.success());

    run_shard_from_shard(take_shard(40));
}

#[test]
fn try_recv_any_empty_does_not_close_channel() {
    run_try_recv_any_empty(take_shard(8));
}

#[test]
fn corrupted_checksum_rejected_as_invalid_frame() {
    run_corrupted_checksum(take_shard(8));
}

#[test]
fn duplicate_read_grant_tag_rejected_at_runtime() {
    run_duplicate_read_grant(take_shard(8));
}

#[test]
fn forged_and_replayed_read_release_rejected_at_runtime() {
    run_forged_and_replayed_release(take_shard(8));
}

#[test]
fn try_recv_endpoint_or_any_interleaves_notice_and_release_before_endpoint() {
    run_interleaved_endpoint_or_any(take_shard(8));
}
