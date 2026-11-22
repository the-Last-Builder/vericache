//! Protocol interoperability and physical RC transfers across process boundaries.
use std::io;
#[cfg(verus_only)]
use transfer::context::*;
#[cfg(verus_only)]
use transfer::host::Completion;
use transfer::rdma::*;
use transfer::transport::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

#[path = "common/rdma.rs"]
mod common;
use common::*;
#[path = "common/data_plane.rs"]
mod data_plane;

verus! {

fn protocol_initiator(node: &Node, t: TcpTransport) {
    let mut channel = node.channel(t);
    let i = Tracked(node.instance.borrow());
    let ReceivedFrame::ReadGrant(read) = channel.recv_any(i).unwrap() else {
        panic!("expected ReadGrant");
    };
    assert_eq!(read.tag, 0xfedcba9876543210);
    assert_eq!(read.user_bytes, b"fixture\0read");
    assert_eq!(
        read.capability.unwrap().wire_fields(),
        (0x0102030405060708, 0x12345678, 0x89abcdef)
    );
    channel.send_remote_read_release(
        read.tag,
        &read.user_bytes,
        read.statement,
        Tracked(read.share.get().tracked_unwrap()),
    ).map_err(|e| e.0).unwrap();
    let ReceivedFrame::WriteGrant(write) = channel.recv_any(i).unwrap() else {
        panic!("expected WriteGrant");
    };
    assert_eq!(write.tag, 42);
    assert_eq!(write.user_bytes, b"fixture\0write");
    assert_eq!(
        write.capability.unwrap().wire_fields(),
        (0x0102030405060708, 0x12345678, 0x89abcdef)
    );
    channel.send_remote_write_release(
        write.tag,
        &write.user_bytes,
        write.statement,
        Tracked(write.slot.get().tracked_unwrap()),
    ).map_err(|e| e.0).unwrap();
    let ReceivedFrame::Notice(notice) = channel.recv_any(i).unwrap() else {
        panic!("expected Notice");
    };
    assert_eq!(notice.user_bytes, b"\0done\xff");
    channel.send_completion_notice(&notice.user_bytes, notice.statement).unwrap();
}

} // verus!
// A one-shot capture stands in for the donor HCA during the protocol-only
// test. It describes an untouched slot, so its completion statement is empty.
// Restore the descriptor through the same receive path used for a real grant.
// Only the donor fixture uses this transport. Every tested message uses TCP.
struct RecordedFrame(Option<Vec<u8>>);
impl ByteTransport for RecordedFrame {
    type Error = io::Error;
    fn send_frame(&mut self, _: &[u8]) -> io::Result<()> {
        Err(io::Error::other("read-only fixture"))
    }
    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        self.0
            .take()
            .ok_or_else(|| io::Error::other("fixture exhausted"))
    }
}

fn recorded_grant(op: u8, tag: u64, payload: &[u8]) -> RecordedFrame {
    let mut frame = vec![op];
    frame.extend(tag.to_le_bytes());
    frame.extend(0x0102030405060708u64.to_le_bytes());
    frame.extend(0x12345678u64.to_le_bytes());
    frame.extend(0x89abcdefu32.to_le_bytes());
    frame.extend(payload);
    RecordedFrame(Some(frame))
}

verus! {

fn protocol_donor(node: &Node, t: TcpTransport) {
    let mut channel = node.channel(t);
    let i = Tracked(node.instance.borrow());
    // Setup frames, unknown operations, and short grant headers end the data session.
    for bytes in [vec![6; 29], vec![7], vec![1; 28]] {
        let mut invalid = Channel::new(RecordedFrame(Some(bytes)), 1, 0, i);
        assert!(!invalid.is_closed());
        assert!(matches!(invalid.recv_any(i), Err(ChannelError::InvalidFrame)));
        assert!(invalid.is_closed());
        assert!(matches!(invalid.recv_any(i), Err(ChannelError::Closed)));
    }
    let read_tag = 0xfedcba9876543210u64;
    let mut read_capture = Channel::new(recorded_grant(1, read_tag, b"fixture\0read"), 1, 0, i);
    let captured = read_capture.recv_remote_read_grant(i).unwrap();
    let read_cap = captured.capability.unwrap();
    // Sending on an already-closed channel preserves the ShareCap instead of dropping it.
    let mut closed = Channel::new(RecordedFrame(Some(vec![7])), 1, 0, i);
    assert!(matches!(closed.recv_any(i), Err(ChannelError::InvalidFrame)));
    assert!(closed.is_closed());
    let Err((ChannelError::Closed, _recovered)) = closed.send_remote_read_grant(
        read_tag,
        &read_cap,
        &captured.user_bytes,
        node.empty(),
        Tracked(captured.share.get().tracked_unwrap()),
    ) else {
        panic!("expected ChannelError::Closed with recovered ShareCap");
    };
    let tracked share_cap = _recovered.get().tracked_unwrap();
    channel.send_remote_read_grant(
        read_tag,
        &read_cap,
        &captured.user_bytes,
        node.empty(),
        Tracked(share_cap),
    ).map_err(|e| e.0).unwrap();
    let ReceivedFrame::ReadRelease(returned) = channel.recv_any(i).unwrap() else {
        panic!("expected ReadRelease");
    };
    assert_eq!(returned.tag, read_tag);
    assert_eq!(returned.user_bytes, b"fixture\0read");

    let mut write_capture = Channel::new(recorded_grant(3, 42, b"fixture\0write"), 1, 0, i);
    let captured = write_capture.recv_remote_write_grant(i).unwrap();
    let write_cap = captured.capability.unwrap();
    let Err((ChannelError::Closed, _recovered)) = closed.send_remote_write_grant(
        42,
        &write_cap,
        &captured.user_bytes,
        node.empty(),
        Tracked(captured.slot.get().tracked_unwrap()),
    ) else {
        panic!("expected ChannelError::Closed with recovered SlotToken");
    };
    let tracked slot_tok = _recovered.get().tracked_unwrap();
    channel.send_remote_write_grant(
        42,
        &write_cap,
        &captured.user_bytes,
        node.empty(),
        Tracked(slot_tok),
    ).map_err(|e| e.0).unwrap();
    let ReceivedFrame::WriteRelease(returned) = channel.recv_any(i).unwrap() else {
        panic!("expected WriteRelease");
    };
    assert_eq!(returned.tag, 42);
    assert_eq!(returned.user_bytes, b"fixture\0write");

    channel.send_completion_notice(b"\0done\xff", node.empty()).unwrap();
    let echo = channel.recv_completion_notice(i).unwrap();
    assert_eq!(echo.user_bytes, b"\0done\xff");
}

} // verus!
#[test]
fn framed_tcp_protocol() {
    let test = "framed_tcp_protocol";
    match std::env::var(ROLE).as_deref() {
        Ok("donor") => {
            let node = Node::new(0);
            let (t, mut p) = start_peer(test);
            protocol_donor(&node, t);
            assert!(p.wait().unwrap().success());
        }
        Ok("initiator") => protocol_initiator(&Node::new(1), connect()),
        _ => assert!(child(test, "donor").status().unwrap().success()),
    }
}

fn pattern(seed: usize) -> Vec<u8> {
    (0..16387)
        .map(|i| ((i * 131 + i / 251 + seed) % 256) as u8)
        .collect()
}

verus! {

fn donor(
    node: &mut Node,
    pd: &ProtectionDomain,
    cq: &mut CompletionQueue,
    qp: &QueuePair,
    channel: &mut Channel<TcpTransport>,
    fenced: bool,
) {
    let initial = pattern(17);
    let n = initial.len();
    let (buf, slot) = node.buffer(n);
    let empty = node.empty();
    let (writer, slot) = node.write(&buf, &initial, slot, empty);
    let flags = AccessFlags { local_write: true, mw_bind: true };
    let mr = register_memory(pd, RegistrationBuffer::Host(&buf), flags).unwrap();
    let mut mw = alloc_window(pd, node.name()).unwrap();
    if fenced {
        let replacement = pattern(109);
        let (done, slot) = fenced_revocation_donor(
            channel,
            qp,
            cq,
            &mr,
            mw,
            &buf.as_slice(),
            &mut node.host,
            &replacement,
            Tracked(node.instance.borrow()),
            slot,
            writer,
            Tracked(node.shares.borrow_mut()),
            91,
        ).expect("invalidate, retire, destroy, overwrite and notify");
        let mut out = vec![0; n];
        let (done, slot) = node.read(&buf, &mut out, slot, statement(done));
        assert_eq!(out, replacement);
        // Keep the QP and backing MR live until the delayed WR has a CQE.
        let recovered = recv_notice(node, channel);
        assert_eq!(recovered.user_bytes, b"recovered");
        let (done, slot) = node.deregister(mr, slot, statement(done));
        node.free(buf, slot, statement(done));
        return;
    }
    let (done, slot) = read_lifecycle_donor(
        channel,
        qp,
        cq,
        mr,
        &mut mw,
        &mut node.host,
        Tracked(node.instance.borrow()),
        slot,
        writer,
        Tracked(node.shares.borrow_mut()),
        91,
        b"read",
    ).expect("read grant, return, local invalidation and deregistration");
    proof {
        assert(mw.state() == WindowState::Unbound);
    }
    let mr = register_memory(pd, RegistrationBuffer::Host(&buf), flags).unwrap();
    let key = mw.next_rkey();
    post_bind_window(
        qp,
        cq,
        &mr.as_slice(),
        &mut mw,
        key,
        RemoteAccess::ReadWrite,
        Tracked(WindowBindGrant::ReadWrite),
        u64::MAX,
    ).map_err(|e| e.0).unwrap();
    let Completed::WindowBound { witness: bound, wr_id: u64::MAX } = complete(
        cq,
        Tracked(node.instance.borrow()),
    ) else {
        panic!("expected second bind");
    };
    mw.finish_bind(bound);
    channel.send_remote_write_grant(
        92,
        &mw.remote_capability(),
        b"write",
        statement(done),
        slot,
    ).map_err(|e| e.0).unwrap();
    let returned = channel.recv_remote_write_release(Tracked(node.instance.borrow())).unwrap();
    assert_eq!(returned.tag, 92);
    assert_eq!(returned.user_bytes, b"written");
    let mut out = vec![0; n];
    let (done, slot) = node.read(
        &buf,
        &mut out,
        Tracked(returned.slot.get().tracked_unwrap()),
        returned.statement,
    );
    assert_eq!(out, pattern(243));
    post_local_invalidate(qp, cq, &mut mw, 4).unwrap();
    let Completed::WindowInvalidated { witness: invalidated, wr_id: 4 } = complete(
        cq,
        Tracked(node.instance.borrow()),
    ) else {
        panic!("expected local invalidation");
    };
    mw.finish_invalidate(invalidated, Tracked(None));
    proof {
        assert(mw.state() == WindowState::Unbound);
    }
    dealloc_window(mw).map_err(|e| e.0).unwrap();
    let (done, slot) = node.deregister(mr, slot, statement(done));
    node.free(buf, slot, statement(done));
    channel.send_completion_notice(b"verified write", node.empty()).unwrap();
}

fn initiator(
    node: &mut Node,
    pd: &ProtectionDomain,
    cq: &mut CompletionQueue,
    qp: &mut QueuePair,
    channel: &mut Channel<TcpTransport>,
    fenced: bool,
    device: &DeviceContext,
) {
    let expected = pattern(17);
    let n = expected.len();
    let (buf, slot) = node.buffer(n);
    let mr = register_memory(
        pd,
        RegistrationBuffer::Host(&buf),
        AccessFlags { local_write: true, mw_bind: false },
    ).unwrap();
    let mut out = vec![0; n];
    if fenced {
        let sentinel = vec![0x5a; n];
        let (status, slot, read) = fenced_revocation_initiator(
            channel,
            qp,
            cq,
            &mr.as_slice(),
            &buf.as_slice(),
            &mut node.host,
            &mut out,
            &sentinel,
            Tracked(node.instance.borrow()),
            slot,
        ).expect("observe failed RDMA_READ and recover local destination");
        println!("delayed RDMA_READ CQE status {status}, verified {n} sentinel bytes");
        assert_ne!(status, 0);
        if device.name() == "mlx5_0" {
            const IBV_WC_REM_ACCESS_ERR: i32 = 10;
            assert_eq!(status, IBV_WC_REM_ACCESS_ERR);
        }
        assert_eq!(out, sentinel);
        let (done, slot) = node.deregister(mr, slot, statement(read));
        node.free(buf, slot, statement(done));
        channel.send_completion_notice(b"recovered", node.empty()).unwrap();
        return;
    }
    let (slot, read) = read_lifecycle_initiator(
        channel,
        qp,
        cq,
        &mr.as_slice(),
        &buf.as_slice(),
        &mut node.host,
        &mut out,
        &expected,
        Tracked(node.instance.borrow()),
        slot,
    ).expect("read CQE, check bytes, return share");
    assert_eq!(out, expected);
    let (written, Tracked(local_tok)) = node.write(&buf, &pattern(243), slot, statement(read));
    let Tracked(local) = statement(written);
    let grant = channel.recv_remote_write_grant(Tracked(node.instance.borrow())).unwrap();
    assert_eq!(grant.tag, 92);
    assert_eq!(grant.user_bytes, b"write");
    let cap = grant.capability.unwrap();
    let Tracked(remote) = grant.statement;
    let tracked remote_tok = grant.slot.get().tracked_unwrap();
    let tracked mut local_perm = ReadPerm::Owned(local_tok);
    let tracked mut infos = local.submitted.into_map();
    proof {
        infos.tracked_union_prefer_right(remote.submitted.into_map());
    }
    let tracked mut infos = SubmittedMap::from_map(node.instance@.id(), infos);
    let tracked mut knows = local.knows.into_map();
    proof {
        knows.tracked_union_prefer_right(remote.knows.into_map());
    }
    let tracked knows = Knows::from_map(node.instance@.id(), knows);
    let (Ghost(_write_ev), _, Tracked(mut pending_write)) = post_rdma_write(
        qp,
        cq,
        Tracked(node.instance.borrow()),
        &mr.as_slice(),
        &cap,
        2,
        true,
        Tracked(&mut local_perm),
        Tracked(remote_tok),
        Tracked(&mut infos),
        Tracked(&knows),
        Ghost(local.facts.union(remote.facts)),
    ).map_err(|e| e.0).unwrap();
    let Completed::Data { wr_id: 2, submitted: _wsub, knows: _wknows, .. } = complete(
        cq,
        Tracked(node.instance.borrow()),
    ) else {
        panic!("expected write CQE");
    };
    let tracked mut local_tok = local_perm.into_owned();
    let tracked mut flush_infos = SubmittedMap::empty(node.instance@.id());
    let tracked flush_knows = _wknows.get();
    proof {
        flush_infos.insert(_wsub.get());
    }
    let _ = post_rdma_write_flush_read(
        qp,
        cq,
        Tracked(node.instance.borrow()),
        &mr.as_slice().sub_slice(0, 1),
        &cap.sub_slice(0, 1),
        3,
        true,
        Tracked(&mut pending_write),
        Tracked(&mut local_tok),
        Tracked(&mut flush_infos),
        Tracked(&flush_knows),
        Ghost(ISet::empty().insert(_write_ev)),
    ).unwrap();
    let Completed::Data {
        wr_id: 3,
        event: flush_ev,
        submitted: _submitted,
        knows: _knows,
        succeeded: _succ,
    } = complete(cq, Tracked(node.instance.borrow())) else {
        panic!("expected post-write flush read CQE");
    };
    let tracked c = Completion { submitted: _submitted.get(), knows: _knows.get() };
    let tracked mut vis_infos = SubmittedMap::empty(node.instance@.id());
    proof {
        vis_infos.insert(c.submitted);
    }
    let tracked remote_tok = complete_remote_write_visibility(
        pending_write,
        flush_ev,
        &vis_infos,
        _succ.borrow(),
    );
    // Keep the observed event locally as well as sending it to the donor.
    let tracked mut known = c.knows;
    let tracked k = known.remove((node.id, c.submitted.key()));
    let tracked mut remote_known = Knows::empty(node.instance@.id());
    proof {
        remote_known.insert(k);
        known.insert(k);
    }
    let tracked remote_done = Completion { submitted: c.submitted, knows: remote_known };
    channel.send_remote_write_release(
        grant.tag,
        b"written",
        statement(Tracked(remote_done)),
        Tracked(remote_tok),
    ).map_err(|e| e.0).unwrap();
    let local_done = statement(Tracked(Completion { submitted: c.submitted, knows: known }));
    let (done, slot) = node.deregister(mr, Tracked(local_tok), local_done);
    node.free(buf, slot, statement(done));
    let notice = recv_notice(node, channel);
    assert_eq!(notice.user_bytes, b"verified write");
}

fn hardware(node: &mut Node, device: &DeviceContext, t: TcpTransport, fenced: bool) {
    let pd = ProtectionDomain::new(device, node.name()).unwrap();
    let mut cq = CompletionQueue::new(device, 32, node.name()).unwrap();
    let reset = UnconnectedQueuePair::new(
        &pd,
        &cq,
        16,
        1,
        gid_index(),
        0x420 + node.id as u32,
        node.name(),
    ).expect("create RC QP and query port 1");
    let mut channel = node.channel(t);
    let peer = channel.exchange_queue_pair_endpoint(
        &reset.endpoint(),
        Tracked(node.instance.borrow()),
    ).unwrap();
    let mut qp = connect_rc(reset, peer, 1 - node.id).expect(
        "transition RESET -> INIT -> RTR -> RTS",
    );
    if node.id == 0 {
        donor(node, &pd, &mut cq, &qp, &mut channel, fenced);
    } else {
        initiator(node, &pd, &mut cq, &mut qp, &mut channel, fenced, device);
    }
    qp.destroy(&cq).unwrap();
    cq.destroy().unwrap();
    pd.destroy().unwrap();
}

} // verus!
#[test]
fn physical_rc_read_write() {
    run_hardware_test("physical_rc_read_write", false);
}

#[test]
fn physical_rdma_fenced_revocation_and_initiator_recovery() {
    run_hardware_test(
        "physical_rdma_fenced_revocation_and_initiator_recovery",
        true,
    );
}

fn run_hardware_test(test: &str, fenced: bool) {
    match std::env::var(ROLE).as_deref() {
        Ok("donor") => {
            let mut node = Node::new(0);
            if let Some(device) = device(&mut node, false) {
                let (t, mut p) = start_peer(test);
                hardware(&mut node, &device, t, fenced);
                device.close().unwrap();
                assert!(p.wait().unwrap().success());
                if fenced {
                    println!(
                        "donor verified 16387 replacement bytes after MW retirement and destruction"
                    );
                } else {
                    println!(
                        "physical Type-2 MW: RDMA_READ and RDMA_WRITE verified, 16387 bytes each"
                    );
                }
            }
        }
        Ok("initiator") => {
            let mut node = Node::new(1);
            let device = device(&mut node, false).unwrap();
            hardware(&mut node, &device, connect(), fenced);
            device.close().unwrap();
        }
        _ => assert!(child(test, "donor").status().unwrap().success()),
    }
}

#[test]
fn physical_rdma_multi_sge_scatter_gather() {
    data_plane::run(
        "physical_rdma_multi_sge_scatter_gather",
        data_plane::Mode::ScatterGather,
    );
}

#[test]
fn physical_rdma_batched_doorbell_and_poll() {
    data_plane::run(
        "physical_rdma_batched_doorbell_and_poll",
        data_plane::Mode::Batch,
    );
}

#[test]
fn physical_rdma_pipeline_throughput_smoke() {
    data_plane::run(
        "physical_rdma_pipeline_throughput_smoke",
        data_plane::Mode::Throughput,
    );
}
