//! Real O_DIRECT reads and writes. Each test gets disjoint name supplies from one bootstrap.
use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};
use transfer::context::*;
use transfer::domain::SlotLoc;
use transfer::host::{self, HostBuffer, HostContext, HostSlice};
use transfer::nvme::{Completed, IoUring, NvmeFile, NvmeSlice, DIRECT_IO_ALIGNMENT};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

type Supplies = (
    Tracked<AllocSource>,
    Tracked<EventSourceAllocator>,
    Tracked<ShareSource>,
);
static NODE: OnceLock<(Tracked<TransferInstance>, Mutex<Supplies>)> = OnceLock::new();

verus! {

#[cfg_attr(not(verus_only), allow(dead_code))]
struct Fixture {
    instance: &'static Tracked<TransferInstance>,
    host: HostContext,
    ring: IoUring,
    allocations: Tracked<AllocSource>,
    names: Tracked<ShareSource>,
    tokens: Tracked<Map<SlotLoc, SlotToken>>,
    shares: Tracked<Map<SlotLoc, ShareCap>>,
    infos: Tracked<SubmittedMap>,
    knows: Tracked<Knows>,
}

impl Fixture {
    fn new(depth: u32) -> Self {
        let (instance, supplies) = NODE.get_or_init(
            ||
                {
                    let (instance, alloc, events, shares) = NodeContext::bootstrap(
                        0,
                    ).into_sources();
                    (instance, Mutex::new((alloc, events, shares)))
                },
        );
        #[allow(unused_mut)]
        let mut supplies = supplies.lock().unwrap();
        let tracked allocations = supplies.0.borrow_mut().split();
        let tracked mut events = supplies.1.borrow_mut().split();
        let tracked names = supplies.2.borrow_mut().split();
        drop(supplies);
        let host = HostContext::new(Tracked(events.mint(EventSourceKind::Host)));
        let ring = IoUring::new(Tracked(events.mint(EventSourceKind::IoUring)), depth).map_err(
            |(err, _)| err,
        ).expect("io_uring_setup");
        assert!(ring.sq_capacity_val() >= depth as usize);
        assert!(ring.max_in_flight_val() >= ring.sq_capacity_val());
        Self {
            instance,
            host,
            ring,
            allocations: Tracked(allocations),
            names: Tracked(names),
            tokens: Tracked(Map::tracked_empty()),
            shares: Tracked(Map::tracked_empty()),
            infos: Tracked(SubmittedMap::empty(instance@.id())),
            knows: Tracked(Knows::empty(instance@.id())),
        }
    }

    fn file(&mut self, len: usize) -> NvmeFile {
        let dir = std::env::var("VERICACHE_NVME_DIR").unwrap_or_else(|_| ".".to_owned());
        let (file, Tracked(tok)) = NvmeFile::create_temp(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.allocations.borrow_mut()),
            &dir,
            len as u64,
        ).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(file.slot_loc(0), tok);
        }
        assert_eq!(file.len(), len as u64);
        file
    }

    fn file_slab(&mut self, num_slots: usize, slot_len: usize) -> NvmeFile {
        let dir = std::env::var("VERICACHE_NVME_DIR").unwrap_or_else(|_| ".".to_owned());
        let (file, Tracked(mut tokens)) = NvmeFile::create_temp_slab(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.allocations.borrow_mut()),
            &dir,
            num_slots,
            slot_len as u64,
        ).unwrap();
        proof {
            let mut i: usize = 0;
            while i < num_slots {
                let tracked tok = tokens.tracked_remove(i);
                self.tokens.borrow_mut().tracked_insert(file.slot_loc(i as nat), tok);
                i = (i + 1) as usize;
            }
        }
        assert_eq!(file.len(), (num_slots * slot_len) as u64);
        file
    }

    fn buffer(&mut self, len: usize) -> HostBuffer {
        let (buf, Tracked(tok)) = host::alloc_aligned(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.allocations.borrow_mut()),
            len,
            DIRECT_IO_ALIGNMENT as usize,
        ).unwrap();
        proof {
            self.tokens.borrow_mut().tracked_insert(buf.slot_loc(0), tok);
        }
        assert_eq!(buf.as_slice().addr() % DIRECT_IO_ALIGNMENT as usize, 0);
        buf
    }

    fn buffer_slab(&mut self, num_slots: usize, slot_len: usize) -> HostBuffer {
        let (buf, Tracked(mut tokens)) = host::alloc_slab(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.allocations.borrow_mut()),
            num_slots,
            slot_len,
            DIRECT_IO_ALIGNMENT as usize,
        ).unwrap();
        proof {
            let mut i: usize = 0;
            while i < num_slots {
                let tracked tok = tokens.tracked_remove(i);
                self.tokens.borrow_mut().tracked_insert(buf.slot_loc(i as nat), tok);
                i = (i + 1) as usize;
            }
        }
        assert_eq!(buf.slot_slice(0).addr() % DIRECT_IO_ALIGNMENT as usize, 0);
        buf
    }

    proof fn record(&mut self, tracked receipt: Submitted, tracked knows: Knows) {
        let tracked mut knows = knows;
        self.knows.borrow_mut().insert(knows.remove((receipt.key().observer, receipt.key())));
        self.infos.borrow_mut().insert(receipt);
    }

    fn fill(&mut self, buf: &HostBuffer, data: &[u8]) {
        self.fill_slice(&buf.as_slice(), data);
    }

    fn fill_slice(&mut self, slice: &HostSlice, data: &[u8]) {
        let tracked mut tok = self.tokens.borrow_mut().tracked_remove(slice.slot_ref().loc);
        let Tracked(c) = host::write_slice(
            &mut self.host,
            Tracked(self.instance.borrow()),
            slice,
            data,
            Tracked(&mut tok),
            Tracked(self.infos.borrow()),
            Tracked(self.knows.borrow()),
            Ghost(self.infos@.dom()),
        );
        proof {
            self.tokens.borrow_mut().tracked_insert(slice.slot_ref().loc, tok);
            self.record(c.submitted, c.knows);
        }
    }

    fn check(&mut self, buf: &HostBuffer, data: &[u8]) {
        self.check_slice(&buf.as_slice(), data);
    }

    fn check_slice(&mut self, slice: &HostSlice, data: &[u8]) {
        let mut out = vec![0; data.len()];
        let tracked tok = self.tokens.borrow_mut().tracked_remove(slice.slot_ref().loc);
        let tracked mut perm = ReadPerm::Owned(tok);
        let Tracked(c) = host::read_slice(
            &mut self.host,
            Tracked(self.instance.borrow()),
            slice,
            &mut out,
            Tracked(&mut perm),
            Tracked(self.infos.borrow()),
            Tracked(self.knows.borrow()),
            Ghost(self.infos@.dom()),
        );
        proof {
            self.tokens.borrow_mut().tracked_insert(slice.slot_ref().loc, perm.into_owned());
            self.record(c.submitted, c.knows);
        }
        assert_eq!(out, data);
    }

    fn stage(&mut self, host: &HostSlice, file: &NvmeSlice, id: u64, write: bool) {
        let ghost src_loc = if write {
            host.slot_ref().loc
        } else {
            file.slot_ref().loc
        };
        let ghost dst_loc = if write {
            file.slot_ref().loc
        } else {
            host.slot_ref().loc
        };
        let tracked mut src_perm = if self.shares@.contains_key(src_loc) {
            ReadPerm::Shared(self.shares.borrow_mut().tracked_remove(src_loc))
        } else {
            ReadPerm::Owned(self.tokens.borrow_mut().tracked_remove(src_loc))
        };
        let tracked mut dst_tok = self.tokens.borrow_mut().tracked_remove(dst_loc);
        let r = if write {
            self.ring.stage_write(
                Tracked(self.instance.borrow()),
                file,
                host,
                id,
                Tracked(&mut src_perm),
                Tracked(&mut dst_tok),
                Tracked(self.infos.borrow()),
                Tracked(self.knows.borrow()),
                Ghost(self.infos@.dom()),
            )
        } else {
            self.ring.stage_read(
                Tracked(self.instance.borrow()),
                host,
                file,
                id,
                Tracked(&mut src_perm),
                Tracked(&mut dst_tok),
                Tracked(self.infos.borrow()),
                Tracked(self.knows.borrow()),
                Ghost(self.infos@.dom()),
            )
        };
        r.expect("stage direct I/O");
        proof {
            match src_perm {
                ReadPerm::Owned(t) => {
                    self.tokens.borrow_mut().tracked_insert(src_loc, t);
                },
                ReadPerm::Shared(c) => {
                    self.shares.borrow_mut().tracked_insert(src_loc, c);
                },
            }
            self.tokens.borrow_mut().tracked_insert(dst_loc, dst_tok);
        }
    }

    fn no_completion(&mut self) -> bool {
        self.ring.poll_one(Tracked(self.instance.borrow())).is_none()
    }

    fn reap(&mut self, bytes: usize, poll: bool) -> u64 {
        let ready = if poll {
            self.ring.poll_one(Tracked(self.instance.borrow()))
        } else {
            None
        };
        let c = ready.unwrap_or_else(
            || self.ring.wait_one(Tracked(self.instance.borrow())).unwrap(),
        );
        match c {
            Completed::Data {
                user_data,
                bytes: actual,
                submitted: _receipt,
                knows: _knowledge,
                ..
            } => {
                proof {
                    self.record(_receipt.get(), _knowledge.get());
                }
                assert_eq!(actual as usize, bytes);
                user_data
            },
            Completed::DataFailed {
                user_data,
                res,
                ..
            } => panic!("I/O {user_data} returned {res}"),
        }
    }

    fn finish<const N: usize>(mut self, file: NvmeFile, buffers: [HostBuffer; N]) {
        let tracked tok = self.tokens.borrow_mut().tracked_remove(file.slot_loc(0));
        NvmeFile::close(
            &mut self.host,
            Tracked(self.instance.borrow()),
            file,
            Tracked(tok),
            Tracked(self.infos.borrow()),
            Tracked(self.knows.borrow()),
            Ghost(self.infos@.dom()),
        );
        self.ring.destroy();
        for buf in buffers {
            let tracked tok = self.tokens.borrow_mut().tracked_remove(buf.slot_loc(0));
            host::free(
                &mut self.host,
                Tracked(self.instance.borrow()),
                buf,
                Tracked(tok),
                Tracked(self.infos.borrow()),
                Tracked(self.knows.borrow()),
                Ghost(self.infos@.dom()),
            );
        }
    }

    fn finish_slab(mut self, file: NvmeFile, buf: HostBuffer) {
        let ghost f_count = file.num_slots();
        let tracked mut file_tokens = Map::tracked_empty();
        proof {
            let mut i: usize = 0;
            while i < f_count {
                let tracked tok = self.tokens.borrow_mut().tracked_remove(file.slot_loc(i as nat));
                file_tokens.tracked_insert(i, tok);
                i = (i + 1) as usize;
            }
        }
        NvmeFile::close_slab(
            &mut self.host,
            Tracked(self.instance.borrow()),
            file,
            Tracked(file_tokens),
            Tracked(self.infos.borrow()),
            Tracked(self.knows.borrow()),
            Ghost(self.infos@.dom()),
        );
        self.ring.destroy();
        let ghost b_count = buf.num_slots();
        let tracked mut buf_tokens = Map::tracked_empty();
        proof {
            let mut i: usize = 0;
            while i < b_count {
                let tracked tok = self.tokens.borrow_mut().tracked_remove(buf.slot_loc(i as nat));
                buf_tokens.tracked_insert(i, tok);
                i = (i + 1) as usize;
            }
        }
        host::free_slab(
            &mut self.host,
            Tracked(self.instance.borrow()),
            buf,
            Tracked(buf_tokens),
            Tracked(self.infos.borrow()),
            Tracked(self.knows.borrow()),
            Ghost(self.infos@.dom()),
        );
    }

    fn lend(&mut self, _file: &NvmeFile) -> Tracked<SlotToken> {
        let tracked tok = self.tokens.borrow_mut().tracked_remove(_file.slot_loc(0));
        let tracked fresh = self.names.borrow_mut().allocate();
        let tracked (Tracked(lent), Tracked(cap)) = self.instance.borrow().lend_slot(tok, fresh);
        proof {
            self.shares.borrow_mut().tracked_insert(_file.slot_loc(0), cap.into_local());
        }
        Tracked(lent)
    }

    fn give_back(&mut self, Tracked(lent): Tracked<SlotToken>, _file: &NvmeFile) {
        let tracked cap = self.shares.borrow_mut().tracked_remove(_file.slot_loc(0));
        let tracked (Tracked(tok), _) = self.instance.borrow().give_back_slot(lent, cap);
        proof {
            self.tokens.borrow_mut().tracked_insert(_file.slot_loc(0), tok);
        }
    }
}

} // verus!
impl Fixture {
    fn submit_batch(&mut self, count: usize) {
        assert_eq!(self.ring.staged_len(), count);
        assert_eq!(self.ring.pending_len(), count);
        assert_eq!(self.ring.in_flight_len(), 0);
        assert!(self.no_completion());
        let mut left = count;
        while left > 0 {
            let n = self.ring.submit().unwrap();
            assert!(n > 0 && n <= left);
            left -= n;
            assert_eq!(self.ring.staged_len(), left);
        }
        assert_eq!(self.ring.in_flight_len(), count);
    }

    fn reap_batch(&mut self, ids: &[u64], bytes: usize, poll: bool) {
        let mut remaining: BTreeSet<_> = ids.iter().copied().collect();
        for _ in ids {
            assert!(
                remaining.remove(&self.reap(bytes, poll)),
                "unexpected or duplicate CQE"
            );
        }
        assert!(remaining.is_empty());
        assert_eq!(self.ring.pending_len(), 0);
        assert_eq!(self.ring.submit().unwrap(), 0);
    }
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i ^ (i >> 7) ^ (i >> 14)) as u8).collect()
}

#[test]
fn nvme_single_write_read_round_trip() {
    let len = 65536;
    let mut f = Fixture::new(1);
    let buf = f.buffer(len);
    let file = f.file(len);
    let bytes = pattern(len);
    f.fill(&buf, &bytes);
    f.stage(&buf.as_slice(), &file.as_slice(), u64::MAX, true);
    f.submit_batch(1);
    f.reap_batch(&[u64::MAX], len, false);
    f.fill(&buf, &vec![0xa5; len]);
    f.stage(&buf.as_slice(), &file.as_slice(), u64::MAX, false);
    f.submit_batch(1);
    f.reap_batch(&[u64::MAX], len, false);
    f.check(&buf, &bytes);
    f.finish(file, [buf]);
}

#[test]
fn nvme_batched_out_of_order_multi_block() {
    let block = 16384;
    let mut f = Fixture::new(8);
    let buf = f.buffer_slab(8, block);
    let file = f.file_slab(8, block);
    let bytes = pattern(8 * block);
    let ids: Vec<u64> = (0..8).map(|i| (1 << 56) + i).collect();
    for i in 0..8 {
        f.fill_slice(&buf.slot_slice(i), &bytes[i * block..(i + 1) * block]);
    }
    for write in [true, false] {
        if !write {
            let fill = vec![0x5a; block];
            for i in 0..8 {
                f.fill_slice(&buf.slot_slice(i), &fill);
            }
        }
        for j in 0..8 {
            let i = if write { j } else { 7 - j };
            let offset = i * block;
            let disk = file.slot_slice(i);
            let host = buf.slot_slice(i);
            assert_eq!(disk.offset(), offset as u64);
            assert_eq!(host.addr(), buf.slot_slice(0).addr() + offset);
            f.stage(&host, &disk, ids[i], write);
        }
        f.submit_batch(8);
        f.reap_batch(&ids, block, !write);
    }
    for i in 0..8 {
        f.check_slice(&buf.slot_slice(i), &bytes[i * block..(i + 1) * block]);
    }
    f.finish_slab(file, buf);
}

#[test]
fn nvme_concurrent_shared_read() {
    let len = 32768;
    let mut f = Fixture::new(2);
    let file = f.file(len);
    let first = f.buffer(len);
    let second = f.buffer(len);
    let bytes = pattern(len);
    f.fill(&first, &bytes);
    f.stage(&first.as_slice(), &file.as_slice(), 0, true);
    f.submit_batch(1);
    f.reap_batch(&[0], len, false);
    f.fill(&first, &vec![0xcc; len]);
    let lent = f.lend(&file);
    f.stage(&first.as_slice(), &file.as_slice(), 1, false);
    f.stage(&second.as_slice(), &file.as_slice(), 2, false);
    f.submit_batch(2);
    f.reap_batch(&[1, 2], len, true);
    f.check(&first, &bytes);
    f.check(&second, &bytes);
    f.give_back(lent, &file);
    f.finish(file, [first, second]);
}
