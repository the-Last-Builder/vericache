//! Cross-process data-plane exercises shared by the host and GPUDirect tests.
use super::common::*;
use std::time::{Duration, Instant};
use transfer::context::*;
#[cfg(feature = "cuda")]
use transfer::cuda::{self, CudaBuffer, CudaHostBuffer, CudaSlice, CudaStream};
use transfer::domain::{MemorySpace, NodeId, SlotLoc, TransferEvent};
use transfer::host::{self, Completion, HostBuffer, HostSlice};
use transfer::rdma::*;
use transfer::transport::{Channel, Received, ReceivedFrame, Statement};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Each integration binary runs its own subset.
pub enum Mode {
    ScatterGather,
    Batch,
    Throughput,
    GdrSingle,
    GdrBatch,
}

impl Mode {
    fn gpu(self) -> bool {
        matches!(self, Self::GdrSingle | Self::GdrBatch)
    }
    fn blocks(self) -> usize {
        if matches!(self, Self::ScatterGather | Self::GdrSingle) {
            1
        } else {
            16
        }
    }
    fn block_len(self) -> usize {
        if self == Self::Throughput {
            512 * 1024
        } else {
            4 * 4099
        }
    }
    fn segments(self, i: usize, serial: bool) -> usize {
        if serial
            || self == Self::Throughput
            || self == Self::GdrSingle
            || (self == Self::Batch && i % 2 == 0)
        {
            1
        } else {
            4
        }
    }
}

#[cfg(feature = "cuda")]
fn gpu_index(node: u64) -> u32 {
    let key = if node == 0 {
        "VERICACHE_GDR_DONOR_GPU"
    } else {
        "VERICACHE_GDR_INITIATOR_GPU"
    };
    std::env::var(key)
        .map(|s| s.parse().expect("invalid GPU ordinal"))
        .unwrap_or(node as u32)
}

fn pattern(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

// Scatter reverses the order of chunks within each multi-SGE block. This
// permutation is its own inverse and makes gather independently checkable.
fn permuted(data: &[u8], mode: Mode, serial: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for (i, block) in data.chunks_exact(mode.block_len()).enumerate() {
        for chunk in block
            .chunks_exact(block.len() / mode.segments(i, serial))
            .rev()
        {
            out.extend_from_slice(chunk);
        }
    }
    out
}

verus! {

#[cfg_attr(not(verus_only), allow(dead_code))]
struct Ledger {
    owned: Tracked<Map<SlotLoc, SlotToken>>,
    shares: Tracked<Map<SlotLoc, ShareCap>>,
    citations: Tracked<Statement>,
    latest: Ghost<Option<TransferEvent>>,
}

impl Ledger {
    fn new(node: &Node) -> Self {
        Self {
            owned: Tracked(Map::tracked_empty()),
            shares: Tracked(Map::tracked_empty()),
            citations: node.empty(),
            latest: Ghost(None),
        }
    }

    fn add_slot(&mut self, Tracked(tok): Tracked<SlotToken>) {
        proof {
            self.owned.borrow_mut().tracked_insert(tok.key(), tok);
        }
    }

    fn add_slab(&mut self, Tracked(mut tokens): Tracked<Map<usize, SlotToken>>, count: usize) {
        let mut i = 0usize;
        while i < count {
            proof {
                let tracked tok = tokens.tracked_remove(i);
                self.owned.borrow_mut().tracked_insert(tok.key(), tok);
            }
            i += 1;
        }
    }

    fn take_slot(&mut self, Ghost(loc): Ghost<SlotLoc>) -> Tracked<SlotToken> {
        Tracked(self.owned.borrow_mut().tracked_remove(loc))
    }

    fn take_slab(
        &mut self,
        Ghost(node): Ghost<NodeId>,
        Ghost(space): Ghost<MemorySpace>,
        count: usize,
    ) -> Tracked<Map<usize, SlotToken>> {
        let tracked mut out = Map::tracked_empty();
        let mut i = 0usize;
        while i < count {
            proof {
                let loc = SlotLoc { node, space, slot_index: i as nat };
                let tracked tok = self.owned.borrow_mut().tracked_remove(loc);
                out.tracked_insert(i, tok);
            }
            i += 1;
        }
        Tracked(out)
    }

    fn record(&mut self, Tracked(c): Tracked<Completion>) {
        proof {
            let tracked s = self.citations.borrow_mut();
            self.latest = Ghost(Some(c.submitted.key()));
            s.submitted.insert(c.submitted);
            s.facts = s.facts.insert(c.submitted.key());
            let tracked mut known = Knows::empty(c.knows.instance_id());
            vstd::modes::tracked_swap(&mut known, &mut s.knows);
            let id = known.instance_id();
            let tracked mut entries = known.into_map();
            entries.tracked_union_prefer_right(c.knows.into_map());
            s.knows = Knows::from_map(id, entries);
        }
    }

    fn receive(&mut self, msg: Received) -> Option<RemoteCapability> {
        let Tracked(slot_opt) = msg.slot;
        let Tracked(share_opt) = msg.share;
        proof {
            if let Some(tok) = slot_opt {
                self.owned.borrow_mut().tracked_insert(tok.key(), tok);
            }
            if let Some(cap) = share_opt {
                self.shares.borrow_mut().tracked_insert(cap.loc(), cap);
            }
        }
        self.citations = msg.statement;
        msg.capability
    }

    #[cfg(feature = "cuda")]
    fn copy(&mut self, _node: &mut Node, gpu: u32, dst: &CudaSlice, src: &CudaSlice) {
        let tracked mut src_perm = ReadPerm::Owned(
            self.owned.borrow_mut().tracked_remove(src.slot_ref().loc),
        );
        let tracked mut dst_tok = self.owned.borrow_mut().tracked_remove(dst.slot_ref().loc);
        let tracked s = self.citations.borrow_mut();
        let mut stream = CudaStream::new(
            gpu,
            Tracked(_node.sources.borrow_mut().mint(EventSourceKind::cuda_stream())),
        ).unwrap();
        let (_, _submitted) = cuda::memcpy_async(
            &mut stream,
            Tracked(_node.instance.borrow()),
            dst,
            src,
            Tracked(&mut src_perm),
            Tracked(&mut dst_tok),
            Tracked(&mut s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        ).unwrap();
        let _knows = cuda::stream_synchronize(&stream, Tracked(_node.instance.borrow())).unwrap();
        stream.destroy().unwrap();
        proof {
            let tracked src_tok = src_perm.into_owned();
            self.owned.borrow_mut().tracked_insert(src_tok.key(), src_tok);
            self.owned.borrow_mut().tracked_insert(dst_tok.key(), dst_tok);
        }
        self.record(Tracked(Completion { submitted: _submitted.get(), knows: _knows.get() }));
    }

    fn observe(&mut self, c: Completed) -> u64 {
        match c {
            Completed::Data {
                wr_id,
                submitted: _submitted,
                knows: _knows,
                succeeded: _succeeded,
                ..
            } => {
                self.record(
                    Tracked(Completion { submitted: _submitted.get(), knows: _knows.get() }),
                );
                proof {
                    self.citations.borrow_mut().succeeded.insert(_succeeded.get());
                }
                wr_id
            },
            Completed::DataFailed {
                wr_id,
                status,
                ..
            } => panic!("data WR {wr_id} failed with CQ status {status}"),
            _ => panic!("unexpected control CQE"),
        }
    }

    fn deregister(&mut self, node: &mut Node, mr: MemoryRegion) {
        let count = mr.num_slots_val();
        let ghost mr_node = mr.node();
        let ghost mr_space = mr.space();
        let tracked s = self.citations.borrow();
        if count == 1 {
            let tok = self.take_slot(Ghost(mr.slot_loc(0)));
            let (done, tok) = deregister_memory(
                &mut node.host,
                Tracked(node.instance.borrow()),
                mr,
                tok,
                Tracked(&s.submitted),
                Tracked(&s.knows),
                Ghost(s.facts),
            ).map_err(|e| e.0).unwrap();
            self.add_slot(tok);
            self.record(done);
        } else {
            let tokens = self.take_slab(Ghost(mr_node), Ghost(mr_space), count);
            let (done, tokens) = deregister_memory_slab(
                &mut node.host,
                Tracked(node.instance.borrow()),
                mr,
                tokens,
                Tracked(&s.submitted),
                Tracked(&s.knows),
                Ghost(s.facts),
            ).map_err(|e| e.0).unwrap();
            self.add_slab(tokens, count);
            self.record(done);
        }
    }
}

enum Buffer {
    Host(HostBuffer),
    #[cfg(feature = "cuda")]
    Gpu { device: CudaBuffer, staging: CudaHostBuffer, gpu: u32 },
}

impl Buffer {
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn new(
        node: &mut Node,
        ledger: &mut Ledger,
        mode: Mode,
        num_slots: usize,
        slot_len: usize,
    ) -> Self {
        #[cfg(feature = "cuda")]
        if mode.gpu() {
            let gpu = gpu_index(node.id);
            return if num_slots == 1 {
                let (device, tok) = cuda::alloc_device(
                    Tracked(node.alloc.borrow_mut()),
                    gpu,
                    slot_len,
                ).unwrap();
                ledger.add_slot(tok);
                let (staging, tok) = cuda::alloc_host(
                    Tracked(node.alloc.borrow_mut()),
                    slot_len,
                ).unwrap();
                ledger.add_slot(tok);
                Self::Gpu { device, staging, gpu }
            } else {
                let (device, tokens) = cuda::alloc_device_slab(
                    Tracked(node.alloc.borrow_mut()),
                    gpu,
                    num_slots,
                    slot_len,
                ).unwrap();
                ledger.add_slab(tokens, num_slots);
                let (staging, tokens) = cuda::alloc_host_slab(
                    Tracked(node.alloc.borrow_mut()),
                    num_slots,
                    slot_len,
                ).unwrap();
                ledger.add_slab(tokens, num_slots);
                Self::Gpu { device, staging, gpu }
            };
        }
        if num_slots == 1 {
            let (host, tok) = node.buffer(slot_len);
            ledger.add_slot(tok);
            Self::Host(host)
        } else {
            let (host, tokens) = node.buffer_slab(num_slots, slot_len);
            ledger.add_slab(tokens, num_slots);
            Self::Host(host)
        }
    }

    fn num_slots(&self) -> usize {
        match self {
            Self::Host(b) => b.num_slots_val(),
            #[cfg(feature = "cuda")]
            Self::Gpu { device, .. } => device.num_slots_val(),
        }
    }

    fn slot_len(&self) -> usize {
        match self {
            Self::Host(b) => b.slot_len_val(),
            #[cfg(feature = "cuda")]
            Self::Gpu { device, .. } => device.slot_len_val(),
        }
    }

    fn register<'a>(&'a self, pd: &'a ProtectionDomain<'a>, flags: AccessFlags) -> Result<
        MemoryRegion<'a>,
        RdmaError,
    > {
        match self {
            Self::Host(b) => register_memory(pd, b, flags),
            #[cfg(feature = "cuda")]
            Self::Gpu { device, .. } => register_memory(pd, device, flags),
        }
    }

    fn host_slot(&self, idx: usize) -> HostSlice<'_> {
        match self {
            Self::Host(b) => b.slot_slice(idx),
            #[cfg(feature = "cuda")]
            Self::Gpu { staging, .. } => staging.host_slot_slice(idx),
        }
    }

    fn write(&self, node: &mut Node, ledger: &mut Ledger, data: &[u8]) {
        let count = self.num_slots();
        let stride = self.slot_len();
        for idx in 0..count {
            let slice = self.host_slot(idx);
            let tracked mut tok = ledger.owned.borrow_mut().tracked_remove(slice.slot_ref().loc);
            let tracked s = ledger.citations.borrow();
            let done = host::write_slice(
                &mut node.host,
                Tracked(node.instance.borrow()),
                &slice,
                &data[idx * stride..(idx + 1) * stride],
                Tracked(&mut tok),
                Tracked(&s.submitted),
                Tracked(&s.knows),
                Ghost(s.facts),
            );
            ledger.add_slot(Tracked(tok));
            ledger.record(done);
        }
    }

    #[cfg(feature = "cuda")]
    fn upload(&self, node: &mut Node, ledger: &mut Ledger) {
        if let Self::Gpu { device, staging, gpu } = self {
            for idx in 0..device.num_slots_val() {
                ledger.copy(node, *gpu, &device.slot_slice(idx), &staging.slot_slice(idx));
            }
        }
    }

    fn read_all(&self, node: &mut Node, ledger: &mut Ledger) -> Vec<u8> {
        let count = self.num_slots();
        let stride = self.slot_len();
        #[cfg(feature = "cuda")]
        if let Self::Gpu { device, staging, gpu } = self {
            for idx in 0..count {
                ledger.copy(node, *gpu, &staging.slot_slice(idx), &device.slot_slice(idx));
            }
        }
        let mut out = vec![0u8; count * stride];
        for idx in 0..count {
            let slice = self.host_slot(idx);
            let tracked mut perm = ReadPerm::Owned(
                ledger.owned.borrow_mut().tracked_remove(slice.slot_ref().loc),
            );
            let tracked s = ledger.citations.borrow();
            let done = host::read_slice(
                &mut node.host,
                Tracked(node.instance.borrow()),
                &slice,
                &mut out[idx * stride..(idx + 1) * stride],
                Tracked(&mut perm),
                Tracked(&s.submitted),
                Tracked(&s.knows),
                Ghost(s.facts),
            );
            ledger.add_slot(Tracked(perm.into_owned()));
            ledger.record(done);
        }
        out
    }

    fn check(&self, node: &mut Node, ledger: &mut Ledger, expected: &[u8]) {
        let out = self.read_all(node, ledger);
        assert!(out == expected, "incorrect payload after DMA");
    }

    fn free(self, node: &mut Node, ledger: &mut Ledger) {
        let count = self.num_slots();
        let tracked s = ledger.citations.borrow();
        match self {
            Self::Host(b) => {
                if count == 1 {
                    let tok = ledger.take_slot(Ghost(b.slot_loc(0)));
                    let done = host::free(
                        &mut node.host,
                        Tracked(node.instance.borrow()),
                        b,
                        tok,
                        Tracked(&s.submitted),
                        Tracked(&s.knows),
                        Ghost(s.facts),
                    );
                    ledger.record(done);
                } else {
                    let tokens = ledger.take_slab(Ghost(b.node()), Ghost(b.space()), count);
                    let done = host::free_slab(
                        &mut node.host,
                        Tracked(node.instance.borrow()),
                        b,
                        tokens,
                        Tracked(&s.submitted),
                        Tracked(&s.knows),
                        Ghost(s.facts),
                    );
                    ledger.record(done);
                }
            },
            #[cfg(feature = "cuda")]
            Self::Gpu { device, staging, .. } => {
                if count == 1 {
                    let tok = ledger.take_slot(Ghost(device.slot_loc(0)));
                    let done = cuda::free_device(
                        &mut node.host,
                        Tracked(node.instance.borrow()),
                        device,
                        tok,
                        Tracked(&s.submitted),
                        Tracked(&s.knows),
                        Ghost(s.facts),
                    ).map_err(|e| e.0).unwrap();
                    ledger.record(done);
                    let tracked s = ledger.citations.borrow();
                    let tok = ledger.take_slot(Ghost(staging.slot_loc(0)));
                    let done = cuda::free_host(
                        &mut node.host,
                        Tracked(node.instance.borrow()),
                        staging,
                        tok,
                        Tracked(&s.submitted),
                        Tracked(&s.knows),
                        Ghost(s.facts),
                    ).map_err(|e| e.0).unwrap();
                    ledger.record(done);
                } else {
                    let tokens = ledger.take_slab(
                        Ghost(device.node()),
                        Ghost(device.space()),
                        count,
                    );
                    let done = cuda::free_device_slab(
                        &mut node.host,
                        Tracked(node.instance.borrow()),
                        device,
                        tokens,
                        Tracked(&s.submitted),
                        Tracked(&s.knows),
                        Ghost(s.facts),
                    ).map_err(|e| e.0).unwrap();
                    ledger.record(done);
                    let tracked s = ledger.citations.borrow();
                    let tokens = ledger.take_slab(
                        Ghost(staging.node()),
                        Ghost(staging.space()),
                        count,
                    );
                    let done = cuda::free_host_slab(
                        &mut node.host,
                        Tracked(node.instance.borrow()),
                        staging,
                        tokens,
                        Tracked(&s.submitted),
                        Tracked(&s.knows),
                        Ghost(s.facts),
                    ).map_err(|e| e.0).unwrap();
                    ledger.record(done);
                }
            },
        }
    }
}

fn donor(
    node: &mut Node,
    pd: &ProtectionDomain,
    qp: &QueuePair,
    cq: &mut CompletionQueue,
    channel: &mut Channel<TcpTransport>,
    mode: Mode,
) {
    let n = mode.blocks() * mode.block_len();
    let mut ledger = Ledger::new(node);
    let buf = Buffer::new(node, &mut ledger, mode, 1, n);
    buf.write(node, &mut ledger, &pattern(n, 17));
    #[cfg(feature = "cuda")]
    buf.upload(node, &mut ledger);
    let mr = buf.register(pd, AccessFlags { local_write: true, mw_bind: true }).expect(
        "register donor allocation",
    );
    let mut mw = alloc_window(pd, node.name()).unwrap();
    if mode != Mode::ScatterGather {
        // Only the gather test needs write permission on donor memory.
        let Tracked(mut facts) = ledger.citations;
        let tracked writer = Completion {
            submitted: facts.submitted.remove(ledger.latest@.unwrap()),
            knows: facts.knows,
        };
        ledger.citations = node.empty();
        let tok = ledger.take_slot(Ghost(mr.slot_loc(0)));
        let (done, tok) = read_lifecycle_donor(
            channel,
            qp,
            cq,
            mr,
            &mut mw,
            &mut node.host,
            Tracked(node.instance.borrow()),
            tok,
            Tracked(writer),
            Tracked(node.shares.borrow_mut()),
            1,
            b"blocks",
        ).unwrap();
        ledger.add_slot(tok);
        ledger.record(done);
        dealloc_window(mw).map_err(|e| e.0).unwrap();
        buf.free(node, &mut ledger);
        return;
    }
    let key = mw.next_rkey();
    post_bind_window(
        qp,
        cq,
        &mr.as_slice(),
        &mut mw,
        key,
        RemoteAccess::ReadWrite,
        Tracked(WindowBindGrant::ReadWrite),
        1,
    ).map_err(|e| e.0).unwrap();
    let Completed::WindowBound { witness: bound, .. } = complete(
        cq,
        Tracked(node.instance.borrow()),
    ) else { panic!("missing bind CQE") };
    mw.finish_bind(bound);
    let cap = mw.remote_capability();
    let remote = ledger.take_slot(Ghost(cap.slot_ref().loc));
    channel.send_remote_write_grant(1, &cap, b"blocks", ledger.citations, remote).map_err(
        |e| e.0,
    ).unwrap();
    ledger.citations = node.empty();
    let returned = channel.recv_remote_write_release(Tracked(node.instance.borrow())).unwrap();
    ledger.receive(returned);
    post_local_invalidate(qp, cq, &mut mw, 2).unwrap();
    let Completed::WindowInvalidated { witness: invalidated, .. } = complete(
        cq,
        Tracked(node.instance.borrow()),
    ) else { panic!("missing invalidate CQE") };
    mw.finish_invalidate(invalidated, Tracked(None));
    dealloc_window(mw).map_err(|e| e.0).unwrap();
    ledger.deregister(node, mr);
    buf.check(node, &mut ledger, &permuted(&pattern(n, 53), mode, false));
    buf.free(node, &mut ledger);
}

fn transfer_blocks(
    _node: &Node,
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    mr: &MemoryRegion,
    even_mr: Option<&MemoryRegion>,
    cap: &RemoteCapability,
    ledger: &mut Ledger,
    mode: Mode,
    serial: bool,
    read: bool,
) {
    let block = mode.block_len();
    let count = mode.blocks();
    let mut lists = Vec::with_capacity(count);
    let mut odd_idx = 0usize;
    let mut even_idx = 0usize;
    for i in 0..count {
        let pieces = mode.segments(i, serial);
        let mut list = Vec::with_capacity(pieces);
        if pieces == 1 {
            if let Some(emr) = even_mr {
                list.push(emr.slot_slice(even_idx));
                even_idx += 1;
            } else {
                list.push(mr.slot_slice(i));
            }
        } else {
            let base = odd_idx * pieces;
            for j in (0..pieces).rev() {
                list.push(mr.slot_slice(base + j));
            }
            odd_idx += 1;
        }
        lists.push(list);
    }
    let mut requests = Vec::with_capacity(count);
    let first_id = if mode == Mode::Batch {
        u64::MAX
    } else {
        count as u64
    };
    for (i, list) in lists.iter().enumerate() {
        requests.push(
            TransferRequest {
                locals: list,
                remote: cap.sub_slice((i * block) as u64, block as u64),
                wr_id: first_id - i as u64,
                read,
                fence: false,
            },
        );
    }
    let start = Instant::now();
    if serial || count == 1 {
        for r in &requests {
            let i = Tracked(_node.instance.borrow());
            if mode == Mode::GdrSingle {
                let ghost rloc = r.remote.slot_ref().loc;
                let ghost lloc = r.locals[0].slot_ref().loc;
                let tracked mut remote_perm = if r.remote.share() is Some {
                    ReadPerm::Shared(ledger.shares.borrow_mut().tracked_remove(rloc))
                } else {
                    ReadPerm::Owned(ledger.owned.borrow_mut().tracked_remove(rloc))
                };
                let tracked local_tok = ledger.owned.borrow_mut().tracked_remove(lloc);
                let tracked s = ledger.citations.borrow_mut();
                let (_, _, pending_gdr) = post_rdma_read_gdr(
                    qp,
                    cq,
                    i,
                    &r.locals[0],
                    &r.remote,
                    r.wr_id,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(local_tok),
                    Tracked(&mut s.submitted),
                    Tracked(&s.knows),
                    Ghost(s.facts),
                ).map_err(|e| e.0).unwrap();
                proof {
                    match remote_perm {
                        ReadPerm::Shared(c) => {
                            ledger.shares.borrow_mut().tracked_insert(rloc, c);
                        },
                        ReadPerm::Owned(t) => {
                            ledger.owned.borrow_mut().tracked_insert(rloc, t);
                        },
                    }
                }
                assert_eq!(ledger.observe(complete(cq, i)), r.wr_id);
                let tracked s = ledger.citations.borrow();
                let tok = mr.complete_gdr_slot(i, pending_gdr, Tracked(&s.knows)).map_err(
                    |e| e.0,
                ).unwrap();
                ledger.add_slot(tok);
                continue;
            } else if serial {
                let ghost rloc = r.remote.slot_ref().loc;
                let ghost lloc = r.locals[0].slot_ref().loc;
                let tracked mut remote_perm = if r.remote.share() is Some {
                    ReadPerm::Shared(ledger.shares.borrow_mut().tracked_remove(rloc))
                } else {
                    ReadPerm::Owned(ledger.owned.borrow_mut().tracked_remove(rloc))
                };
                let tracked mut local_tok = ledger.owned.borrow_mut().tracked_remove(lloc);
                let tracked s = ledger.citations.borrow_mut();
                post_rdma_read(
                    qp,
                    cq,
                    i,
                    &r.locals[0],
                    &r.remote,
                    r.wr_id,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(&mut local_tok),
                    Tracked(&mut s.submitted),
                    Tracked(&s.knows),
                    Ghost(s.facts),
                ).unwrap();
                proof {
                    ledger.owned.borrow_mut().tracked_insert(lloc, local_tok);
                    match remote_perm {
                        ReadPerm::Shared(c) => {
                            ledger.shares.borrow_mut().tracked_insert(rloc, c);
                        },
                        ReadPerm::Owned(t) => {
                            ledger.owned.borrow_mut().tracked_insert(rloc, t);
                        },
                    }
                }
            } else if read {
                let ghost rloc = r.remote.slot_ref().loc;
                let tracked mut remote_perm = if r.remote.share() is Some {
                    ReadPerm::Shared(ledger.shares.borrow_mut().tracked_remove(rloc))
                } else {
                    ReadPerm::Owned(ledger.owned.borrow_mut().tracked_remove(rloc))
                };
                let tracked s = ledger.citations.borrow_mut();
                post_rdma_read_sg(
                    qp,
                    cq,
                    i,
                    r.locals,
                    &r.remote,
                    r.wr_id,
                    false,
                    Tracked(&mut remote_perm),
                    Tracked(ledger.owned.borrow_mut()),
                    Tracked(&mut s.submitted),
                    Tracked(&s.knows),
                    Ghost(s.facts),
                ).unwrap();
                proof {
                    match remote_perm {
                        ReadPerm::Shared(c) => {
                            ledger.shares.borrow_mut().tracked_insert(rloc, c);
                        },
                        ReadPerm::Owned(t) => {
                            ledger.owned.borrow_mut().tracked_insert(rloc, t);
                        },
                    }
                }
            } else {
                let ghost rloc = r.remote.slot_ref().loc;
                let tracked remote_tok = ledger.owned.borrow_mut().tracked_remove(rloc);
                let tracked s = ledger.citations.borrow_mut();
                let (_, _, Tracked(mut pending_write)) = post_rdma_write_sg(
                    qp,
                    cq,
                    i,
                    r.locals,
                    &r.remote,
                    r.wr_id,
                    false,
                    Tracked(ledger.owned.borrow_mut()),
                    Tracked(remote_tok),
                    Tracked(&mut s.submitted),
                    Tracked(&s.knows),
                    Ghost(s.facts),
                ).map_err(|e| e.0).unwrap();
                assert_eq!(ledger.observe(complete(cq, i)), r.wr_id);
                let ghost lloc = r.locals[0].slot_ref().loc;
                let tracked mut local_tok = ledger.owned.borrow_mut().tracked_remove(lloc);
                let tracked s = ledger.citations.borrow_mut();
                let flush_id = r.wr_id + 1000;
                post_rdma_write_flush_read(
                    qp,
                    cq,
                    i,
                    &r.locals[0].sub_slice(0, 1),
                    &r.remote.sub_slice(0, 1),
                    flush_id,
                    true,
                    Tracked(&mut pending_write),
                    Tracked(&mut local_tok),
                    Tracked(&mut s.submitted),
                    Tracked(&s.knows),
                    Ghost(s.facts),
                ).unwrap();
                let cqe = complete(cq, i);
                assert_eq!(ledger.observe(cqe), flush_id);
                proof {
                    ledger.owned.borrow_mut().tracked_insert(lloc, local_tok);
                    let flush_ev = ledger.latest@.unwrap();
                    let tracked s = ledger.citations.borrow();
                    let tracked succ = s.succeeded.get(flush_ev);
                    let tracked rtok = complete_remote_write_visibility(
                        pending_write,
                        Ghost(flush_ev),
                        &s.submitted,
                        succ,
                    );
                    ledger.owned.borrow_mut().tracked_insert(rloc, rtok);
                }
                continue;
            }
            assert_eq!(ledger.observe(complete(cq, i)), r.wr_id);
        }
    } else if mode.gpu() {
        let tracked s = ledger.citations.borrow_mut();
        let (result, pending_gdr) = post_send_batch_gdr(
            qp,
            cq,
            Tracked(_node.instance.borrow()),
            &requests,
            Tracked(ledger.owned.borrow_mut()),
            Tracked(ledger.shares.borrow_mut()),
            Tracked(&mut s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        );
        let posted = result.posted.len();
        for (item, request) in result.posted.iter().zip(&requests) {
            assert_eq!(item.wr_id, request.wr_id);
        }
        let deadline = start + Duration::from_secs(10);
        let mut seen = vec![false; posted];
        let mut remaining = posted;
        while remaining > 0 {
            assert!(Instant::now() < deadline, "timed out draining doorbell");
            for c in poll_completion_queue_batch(
                cq,
                remaining,
                Tracked(_node.instance.borrow()),
            ).unwrap() {
                let index = (first_id - ledger.observe(c)) as usize;
                assert!(index < posted && !seen[index], "unexpected or duplicate CQE");
                seen[index] = true;
                remaining -= 1;
            }
        }
        assert_eq!(result.error, None, "doorbell accepted {posted} of {count}");
        assert_eq!(posted, count);
        let tracked s = ledger.citations.borrow();
        let Tracked(mut gdr_toks) = mr.complete_gdr_slots(
            Tracked(_node.instance.borrow()),
            pending_gdr,
            Tracked(&s.knows),
        ).map_err(|e| e.0).unwrap();
        proof {
            ledger.owned.borrow_mut().tracked_union_prefer_right(gdr_toks);
        }
    } else {
        let tracked s = ledger.citations.borrow_mut();
        let result = post_send_batch(
            qp,
            cq,
            Tracked(_node.instance.borrow()),
            &requests,
            Tracked(ledger.owned.borrow_mut()),
            Tracked(ledger.shares.borrow_mut()),
            Tracked(&mut s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        );
        let posted = result.posted.len();
        for (item, request) in result.posted.iter().zip(&requests) {
            assert_eq!(item.wr_id, request.wr_id);
        }
        let deadline = start + Duration::from_secs(10);
        let mut seen = vec![false; posted];
        let mut remaining = posted;
        while remaining > 0 {
            assert!(Instant::now() < deadline, "timed out draining doorbell");
            for c in poll_completion_queue_batch(
                cq,
                remaining,
                Tracked(_node.instance.borrow()),
            ).unwrap() {
                let index = (first_id - ledger.observe(c)) as usize;
                assert!(index < posted && !seen[index], "unexpected or duplicate CQE");
                seen[index] = true;
                remaining -= 1;
            }
        }
        assert_eq!(result.error, None, "doorbell accepted {posted} of {count}");
        assert_eq!(posted, count);
    }
    if mode == Mode::Throughput {
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "{} bytes, max_rd_atomic={}, {}: {:.1} MB/s",
            count * block,
            qp.config().max_rd_atomic,
            if serial {
                "single WR/poll"
            } else {
                "chained WRs/batch poll"
            },
            (count * block) as f64 / 1e6 / elapsed
        );
    }
}

fn initiator(
    node: &mut Node,
    pd: &ProtectionDomain,
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    channel: &mut Channel<TcpTransport>,
    mode: Mode,
    serial: bool,
) {
    let block = mode.block_len();
    let count = mode.blocks();
    let n = count * block;
    let mut ledger = Ledger::new(node);
    let (buf, even_buf) = if mode == Mode::Batch {
        (
            Buffer::new(node, &mut ledger, mode, (count / 2) * 4, block / 4),
            Some(Buffer::new(node, &mut ledger, mode, count / 2, block)),
        )
    } else if mode == Mode::ScatterGather || mode == Mode::GdrBatch {
        (Buffer::new(node, &mut ledger, mode, count * 4, block / 4), None)
    } else {
        (Buffer::new(node, &mut ledger, mode, count, block), None)
    };
    let grant = match channel.recv_any(Tracked(node.instance.borrow())).unwrap() {
        ReceivedFrame::WriteGrant(grant) if mode == Mode::ScatterGather => grant,
        ReceivedFrame::ReadGrant(grant) if mode != Mode::ScatterGather => grant,
        _ => panic!("unexpected data-plane grant"),
    };
    let tag = grant.tag;
    let cap = ledger.receive(grant).unwrap();
    let flags = AccessFlags { local_write: true, mw_bind: false };
    let mr = buf.register(pd, flags).expect("register initiator allocation");
    let even_mr = even_buf.as_ref().map(|b| b.register(pd, flags).expect("register even slab"));
    transfer_blocks(node, qp, cq, &mr, even_mr.as_ref(), &cap, &mut ledger, mode, serial, true);
    #[cfg(feature = "cuda")]
    if mode.gpu() {
        if let Some(cuda::GdrWritesOrdering::None) = mr.gdr_ordering() {
            mr.flush_gdr_writes().expect("flush GDR writes");
        }
        assert!(mr.gdr_visible_after_flush_to_device(gpu_index(node.id)));
    }
    let expected = permuted(&pattern(n, 17), mode, serial);
    if let Some(eb) = &even_buf {
        let even_bytes = eb.read_all(node, &mut ledger);
        let odd_bytes = buf.read_all(node, &mut ledger);
        let mut combined = Vec::with_capacity(n);
        for i in 0..count {
            let k = i / 2;
            if i % 2 == 0 {
                combined.extend_from_slice(&even_bytes[k * block..(k + 1) * block]);
            } else {
                combined.extend_from_slice(&odd_bytes[k * block..(k + 1) * block]);
            }
        }
        assert!(combined == expected, "incorrect payload after batched DMA");
    } else {
        buf.check(node, &mut ledger, &expected);
    }
    if mode == Mode::ScatterGather {
        buf.write(node, &mut ledger, &pattern(n, 53));
        transfer_blocks(node, qp, cq, &mr, None, &cap, &mut ledger, mode, false, false);
    }
    if let Some(emr) = even_mr {
        ledger.deregister(node, emr);
    }
    ledger.deregister(node, mr);
    if let Some(eb) = even_buf {
        eb.free(node, &mut ledger);
    }
    buf.free(node, &mut ledger);
    if mode == Mode::ScatterGather {
        let remote = ledger.take_slot(Ghost(cap.slot_ref().loc));
        channel.send_remote_write_release(tag, b"verified", ledger.citations, remote).map_err(
            |e| e.0,
        ).unwrap();
    } else {
        let tracked share = ledger.shares.borrow_mut().tracked_remove(cap.slot_ref().loc);
        channel.send_remote_read_release(
            tag,
            b"verified",
            ledger.citations,
            Tracked(share),
        ).map_err(|e| e.0).unwrap();
    }
}

fn exercise(node: &mut Node, dev: &DeviceContext, t: TcpTransport, mode: Mode) {
    let attr = dev.attributes().unwrap();
    let pd = ProtectionDomain::new(dev, node.name()).unwrap();
    let mut cq = CompletionQueue::new(dev, 64, node.name()).unwrap();
    let mut channel = node.channel(t);
    let runs: &[bool] = if mode == Mode::Throughput {
        &[true, false]
    } else {
        &[false]
    };
    for &serial in runs {
        let depth = if serial {
            1
        } else if mode == Mode::Throughput {
            16
        } else {
            8
        };
        let config = QueuePairConfig {
            send_depth: 32,
            max_send_sge: if serial || mode == Mode::GdrSingle {
                1
            } else {
                4
            },
            max_rd_atomic: depth.min(attr.max_qp_init_rd_atomic),
            max_dest_rd_atomic: depth.min(attr.max_qp_rd_atomic),
        };
        let reset = UnconnectedQueuePair::new_with_config(
            &pd,
            &cq,
            config,
            1,
            gid_index(),
            0x420 + node.id as u32,
            node.name(),
        ).unwrap();
        let peer = channel.exchange_queue_pair_endpoint(
            &reset.endpoint(),
            Tracked(node.instance.borrow()),
        ).unwrap();
        let mut qp = connect_rc(reset, peer, 1 - node.id).unwrap();
        assert_eq!(qp.config(), config);
        if node.id == 0 {
            donor(node, &pd, &qp, &mut cq, &mut channel, mode);
        } else {
            initiator(node, &pd, &mut qp, &mut cq, &mut channel, mode, serial);
        }
        qp.destroy(&cq).unwrap();
    }
    cq.destroy().unwrap();
    pd.destroy().unwrap();
}

} // verus!
pub fn run(test: &str, mode: Mode) {
    match std::env::var(ROLE).as_deref() {
        Ok("donor") => {
            #[cfg(feature = "cuda")]
            if mode.gpu() {
                let count = match cuda::device_count() {
                    Ok(n) => n,
                    Err(cuda::CudaError(35 | 100)) => 0,
                    Err(e) => panic!("enumerating CUDA devices: {e:?}"),
                };
                if count < 2 {
                    eprintln!("SKIP {test}: {count} CUDA devices");
                    return;
                }
                assert!(
                    gpu_index(0) < count && gpu_index(1) < count && gpu_index(0) != gpu_index(1),
                    "select two distinct visible GPUs"
                );
            }
            let mut node = Node::new(0);
            if let Some(dev) = device(&mut node, mode.gpu()) {
                let (transport, mut child) = start_peer(test);
                exercise(&mut node, &dev, transport, mode);
                #[cfg(feature = "cuda")]
                let hca = mode.gpu().then(|| dev.name());
                dev.close().unwrap();
                assert!(child.wait().unwrap().success());
                #[cfg(feature = "cuda")]
                if let Some(hca) = hca {
                    let wrs = mode.blocks();
                    let sges: usize = (0..wrs).map(|i| mode.segments(i, false)).sum();
                    println!(
                        "{test}: donor_gpu={} initiator_gpu={} hca={hca} bytes={} data_wrs={wrs} sges={sges}",
                        gpu_index(0),
                        gpu_index(1),
                        wrs * mode.block_len()
                    );
                }
            }
        }
        Ok("initiator") => {
            let mut node = Node::new(1);
            let dev = device(&mut node, mode.gpu()).unwrap();
            exercise(&mut node, &dev, connect(), mode);
            dev.close().unwrap();
        }
        _ => assert!(child(test, "donor").status().unwrap().success()),
    }
}
