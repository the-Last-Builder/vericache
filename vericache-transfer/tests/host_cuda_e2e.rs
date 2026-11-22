//! Hardware exercises for Host and CUDA transfers. One process, one bootstrap.
use transfer::context::*;
use transfer::cuda::{
    self, CudaBuffer, CudaEvent, CudaHostBuffer, CudaHostRegistration, CudaSlice, CudaStream,
    EventStatus,
};
use transfer::domain::SlotLoc;
#[cfg(verus_only)]
use transfer::domain::TransferEvent;
use transfer::host::{self, HostBuffer, HostContext, HostSlice};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

verus! {

// Keep slot ownership and the most recent observed completion across calls.
// Each host access cites this completion, and each stream carries its own FIFO.
#[cfg_attr(not(verus_only), allow(dead_code))]
struct Fixture {
    instance: Tracked<TransferInstance>,
    allocations: Tracked<AllocSource>,
    sources: Tracked<EventSourceAllocator>,
    host: HostContext,
    tokens: Tracked<Map<SlotLoc, SlotToken>>,
    last: Tracked<Option<Submitted>>,
    knows: Tracked<Knows>,
}

impl Fixture {
    fn new() -> Self {
        let (Tracked(instance), allocations, Tracked(mut sources), _) = NodeContext::bootstrap(
            0,
        ).into_sources();
        let host = HostContext::new(Tracked(sources.mint(EventSourceKind::Host)));
        Self {
            host,
            allocations,
            sources: Tracked(sources),
            tokens: Tracked(Map::tracked_empty()),
            last: Tracked(None),
            knows: Tracked(Knows::empty(instance.id())),
            instance: Tracked(instance),
        }
    }

    spec fn cited(&self) -> ISet<TransferEvent> {
        match self.last@ {
            Some(s) => ISet::empty().insert(s.key()),
            None => ISet::empty(),
        }
    }

    proof fn infos(&self) -> (tracked infos: SubmittedMap) {
        let tracked mut infos = SubmittedMap::empty(self.instance@.id());
        if let Some(receipt) = self.last.get() {
            infos.insert(receipt);
        }
        infos
    }

    fn stream(&mut self, device: u32) -> CudaStream {
        CudaStream::new(
            device,
            Tracked(self.sources.borrow_mut().mint(EventSourceKind::CudaStream)),
        ).unwrap()
    }

    fn event(&self, device: u32) -> CudaEvent {
        CudaEvent::new(device, Ghost(0), Tracked(self.instance.borrow())).unwrap()
    }

    fn device(&mut self, device: u32, len: usize) -> CudaBuffer {
        let (buf, Tracked(tok)) = cuda::alloc_device(
            Tracked(self.allocations.borrow_mut()),
            device,
            len,
        ).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(buf.slot_loc(0), tok);
        }
        buf
    }

    fn pinned(&mut self, len: usize) -> CudaHostBuffer {
        let (buf, Tracked(tok)) = cuda::alloc_host(
            Tracked(self.allocations.borrow_mut()),
            len,
        ).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(buf.slot_loc(0), tok);
        }
        buf
    }

    fn pageable(&mut self, len: usize) -> HostBuffer {
        let (buf, Tracked(tok)) = host::alloc(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.allocations.borrow_mut()),
            len,
        ).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(buf.slot_loc(0), tok);
        }
        buf
    }

    fn completed(&mut self, Tracked(c): Tracked<host::Completion>) {
        proof {
            self.last = Tracked(Some(c.submitted));
            self.knows = Tracked(c.knows);
        }
    }

    fn write(&mut self, dst: &HostSlice, bytes: &[u8]) {
        let tracked infos = self.infos();
        let ghost cited = self.cited();
        let tracked mut tok = self.tokens.borrow_mut().tracked_remove(dst.slot_ref().loc);
        let c = host::write_slice(
            &mut self.host,
            Tracked(self.instance.borrow()),
            dst,
            bytes,
            Tracked(&mut tok),
            Tracked(&infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        );
        proof {
            self.tokens.borrow_mut().tracked_insert(dst.slot_ref().loc, tok);
        }
        self.completed(c);
    }

    fn read(&mut self, src: &HostSlice, out: &mut [u8]) {
        let tracked infos = self.infos();
        let ghost cited = self.cited();
        let tracked tok = self.tokens.borrow_mut().tracked_remove(src.slot_ref().loc);
        let tracked mut perm = ReadPerm::Owned(tok);
        let c = host::read_slice(
            &mut self.host,
            Tracked(self.instance.borrow()),
            src,
            out,
            Tracked(&mut perm),
            Tracked(&infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        );
        proof {
            self.tokens.borrow_mut().tracked_insert(src.slot_ref().loc, perm.into_owned());
        }
        self.completed(c);
    }

    fn copy(&mut self, stream: &mut CudaStream, dst: &CudaSlice, src: &CudaSlice) -> Tracked<
        Submitted,
    > {
        let tracked mut infos = self.infos();
        let ghost cited = self.cited();
        let tracked src_tok = self.tokens.borrow_mut().tracked_remove(src.slot_ref().loc);
        let tracked mut src_perm = ReadPerm::Owned(src_tok);
        let tracked mut dst_tok = self.tokens.borrow_mut().tracked_remove(dst.slot_ref().loc);
        let (_, receipt) = cuda::memcpy_async(
            stream,
            Tracked(self.instance.borrow()),
            dst,
            src,
            Tracked(&mut src_perm),
            Tracked(&mut dst_tok),
            Tracked(&mut infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        ).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(src.slot_ref().loc, src_perm.into_owned());
            self.tokens.borrow_mut().tracked_insert(dst.slot_ref().loc, dst_tok);
        }
        receipt
    }

    fn synchronize(&mut self, stream: &CudaStream, Tracked(last): Tracked<Submitted>) {
        self.knows = cuda::stream_synchronize(stream, Tracked(self.instance.borrow())).unwrap();
        proof {
            self.last = Tracked(Some(last));
        }
    }

    fn observed(&mut self, Tracked(last): Tracked<Submitted>, knows: Tracked<Knows>) {
        self.knows = knows;
        proof {
            self.last = Tracked(Some(last));
        }
    }

    fn query(&self, event: &CudaEvent) -> EventStatus {
        cuda::event_query(event, Tracked(self.instance.borrow())).unwrap()
    }

    fn unregister(&mut self, registration: CudaHostRegistration) {
        let tracked infos = self.infos();
        let ghost cited = self.cited();
        let ghost loc = registration.slot_loc(0);
        let tracked tok = self.tokens.borrow_mut().tracked_remove(loc);
        let (c, Tracked(returned)) = cuda::host_unregister(
            &mut self.host,
            Tracked(self.instance.borrow()),
            registration,
            Tracked(tok),
            Tracked(&infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        ).map_err(|e| e.0).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(loc, returned);
        }
        self.completed(c);
    }

    fn free_device(&mut self, buf: CudaBuffer) {
        let tracked infos = self.infos();
        let ghost cited = self.cited();
        let tracked tok = self.tokens.borrow_mut().tracked_remove(buf.slot_loc(0));
        let c = cuda::free_device(
            &mut self.host,
            Tracked(self.instance.borrow()),
            buf,
            Tracked(tok),
            Tracked(&infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        ).map_err(|e| e.0).unwrap();
        self.completed(c);
    }

    fn free_pinned(&mut self, buf: CudaHostBuffer) {
        let tracked infos = self.infos();
        let ghost cited = self.cited();
        let tracked tok = self.tokens.borrow_mut().tracked_remove(buf.slot_loc(0));
        let c = cuda::free_host(
            &mut self.host,
            Tracked(self.instance.borrow()),
            buf,
            Tracked(tok),
            Tracked(&infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        ).map_err(|e| e.0).unwrap();
        self.completed(c);
    }

    fn free_pageable(&mut self, buf: HostBuffer) {
        let tracked infos = self.infos();
        let ghost cited = self.cited();
        let tracked tok = self.tokens.borrow_mut().tracked_remove(buf.slot_loc(0));
        let c = host::free(
            &mut self.host,
            Tracked(self.instance.borrow()),
            buf,
            Tracked(tok),
            Tracked(&infos),
            Tracked(self.knows.borrow()),
            Ghost(cited),
        );
        self.completed(c);
    }
}

} // verus!
fn pattern(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 131 + i / 251 + 17) % 256) as u8)
        .collect()
}

fn multi_device_round_trip(f: &mut Fixture, destination: u32) {
    let bytes = pattern(65_539);
    let n = bytes.len();
    let host = f.pinned(2 * n + 2);
    let first = f.device(0, n);
    let second = f.device(destination, n);
    let mut producer = f.stream(0);
    let mut consumer = f.stream(destination);
    let mut copied = f.event(0);

    // Nonzero offsets and sentinel bytes exercise both kinds of slice.
    let cpu = host.as_host_slice();
    let dma = host.as_cuda_slice();
    f.write(&cpu, &vec![0xa5; cpu.len()]);
    f.write(&cpu.sub_slice(1, n), &bytes);
    f.copy(&mut producer, &first.as_cuda_slice(), &dma.sub_slice(1, n));
    f.copy(
        &mut producer,
        &second.as_cuda_slice(),
        &first.as_cuda_slice(),
    );
    cuda::event_record(&mut copied, &producer).unwrap();
    cuda::stream_wait_event(&mut consumer, &copied).unwrap();
    let done = f.copy(
        &mut consumer,
        &dma.sub_slice(n + 1, n),
        &second.as_cuda_slice(),
    );
    f.synchronize(&consumer, done);
    let mut out = vec![0; cpu.len()];
    f.read(&cpu, &mut out);
    assert_eq!(out[0], 0xa5);
    assert_eq!(out.last(), Some(&0xa5));
    assert_eq!(&out[1..n + 1], bytes);
    assert_eq!(&out[n + 1..2 * n + 1], bytes);

    copied.destroy().unwrap();
    consumer.destroy().unwrap();
    producer.destroy().unwrap();
    f.free_device(first);
    f.free_device(second);
    f.free_pinned(host);
}

fn registered_round_trip(f: &mut Fixture) {
    let bytes = pattern(16_387);
    let n = bytes.len();
    let buf = f.pageable(2 * n);
    let gpu = f.device(0, n);
    let mut stream = f.stream(0);
    f.write(&buf.as_slice().sub_slice(0, n), &bytes);
    let registration = cuda::host_register(&buf).unwrap();
    let dma = registration.as_cuda_slice();
    f.copy(&mut stream, &gpu.as_cuda_slice(), &dma.sub_slice(0, n));
    let done = f.copy(&mut stream, &dma.sub_slice(n, n), &gpu.as_cuda_slice());
    f.synchronize(&stream, done);
    f.unregister(registration);

    let mut out = vec![0; 2 * n];
    f.read(&buf.as_slice(), &mut out);
    assert_eq!(&out[..n], bytes);
    assert_eq!(&out[n..], bytes);
    stream.destroy().unwrap();
    f.free_device(gpu);
    f.free_pageable(buf);
}

fn wait_until_ready(event: &CudaEvent, f: &Fixture) -> Tracked<Knows> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match f.query(event) {
            EventStatus::Ready(known) => return known,
            EventStatus::NotReady => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "CUDA event did not complete"
                );
                std::thread::yield_now();
            }
        }
    }
}

fn ordered_handoff_and_rerecord(f: &mut Fixture) {
    let bytes = pattern(262_151);
    let n = bytes.len();
    let region = f.pinned(n);
    let staging = f.pinned(n);
    let first = f.device(0, n);
    let second = f.device(0, n);
    let replacement = f.device(0, n);
    let mut producer = f.stream(0);
    let mut writer = f.stream(0);
    let mut snapshot = f.event(0);
    let mut handoff = f.event(0);

    // A and B both read the fresh zero-filled pinned range. Only A belongs
    // to the first recording. The writer gets a distinct nonzero payload.
    f.copy(
        &mut producer,
        &first.as_cuda_slice(),
        &region.as_cuda_slice(),
    );
    cuda::event_record(&mut snapshot, &producer).unwrap();
    f.copy(
        &mut producer,
        &second.as_cuda_slice(),
        &region.as_cuda_slice(),
    );
    cuda::event_record(&mut handoff, &producer).unwrap();
    f.write(&staging.as_host_slice(), &bytes);
    f.copy(
        &mut writer,
        &replacement.as_cuda_slice(),
        &staging.as_cuda_slice(),
    );
    cuda::stream_wait_event(&mut writer, &handoff).unwrap();
    let written = f.copy(
        &mut writer,
        &region.as_cuda_slice(),
        &replacement.as_cuda_slice(),
    );

    // The entire handoff has already been enqueued when the host first polls.
    let _known_a = wait_until_ready(&snapshot, f);
    cuda::event_record(&mut snapshot, &writer).unwrap();
    let known_w = wait_until_ready(&snapshot, f);
    f.observed(written, known_w);

    let mut out = vec![0; n];
    f.read(&region.as_host_slice(), &mut out);
    assert_eq!(out, bytes);
    for saved in [&first, &second] {
        let done = f.copy(
            &mut writer,
            &staging.as_cuda_slice(),
            &saved.as_cuda_slice(),
        );
        f.synchronize(&writer, done);
        f.read(&staging.as_host_slice(), &mut out);
        assert!(out.iter().all(|&byte| byte == 0));
    }
    snapshot.destroy().unwrap();
    handoff.destroy().unwrap();
    writer.destroy().unwrap();
    producer.destroy().unwrap();
    f.free_device(replacement);
    f.free_device(second);
    f.free_device(first);
    f.free_pinned(staging);
    f.free_pinned(region);
}

#[test]
fn host_cuda_end_to_end() {
    let count = cuda::device_count().expect("query CUDA device count");
    assert!(
        count > 0,
        "the hardware integration test requires a CUDA device"
    );
    let destination = if count >= 2 { 1 } else { 0 };
    println!("CUDA devices: {count}, D2D route: 0 -> {destination}");
    let mut fixture = Fixture::new();
    ordered_handoff_and_rerecord(&mut fixture);
    multi_device_round_trip(&mut fixture, destination);
    registered_round_trip(&mut fixture);
}
