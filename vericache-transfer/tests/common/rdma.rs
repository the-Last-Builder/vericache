//! Protocol interoperability and physical RC transfers across process boundaries.
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use transfer::context::*;
use transfer::host::{self, Completion, HostBuffer, HostContext};
use transfer::rdma::*;
use transfer::transport::*;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

pub struct TcpTransport(TcpStream);

impl TcpTransport {
    fn new(stream: TcpStream) -> Self {
        stream.set_nodelay(true).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        Self(stream)
    }
}
impl ByteTransport for TcpTransport {
    type Error = io::Error;
    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        self.0.write_all(&n.to_be_bytes())?;
        self.0.write_all(frame)
    }
    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        let mut prefix = [0; 4];
        self.0.read_exact(&mut prefix)?;
        let n = u32::from_be_bytes(prefix) as usize;
        if n > 1024 * 1024 {
            return Err(io::Error::other("frame exceeds test limit"));
        }
        let mut frame = vec![0; n];
        self.0.read_exact(&mut frame)?;
        Ok(frame)
    }
}

pub const ROLE: &str = "VERICACHE_RDMA_TEST_ROLE";
const ADDRESS: &str = "VERICACHE_RDMA_TEST_ADDRESS";
const SESSION: &str = "VERICACHE_RDMA_TEST_SESSION";
pub fn child(test: &str, role: &str) -> Command {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", test, "--nocapture"]).env(ROLE, role);
    cmd
}
pub fn start_peer(test: &str) -> (TcpTransport, Child) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut secret = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut secret)
        .unwrap();
    let session: String = secret.iter().map(|byte| format!("{byte:02x}")).collect();
    let mut process = child(test, "initiator")
        .env(SESSION, &session)
        .env(ADDRESS, listener.local_addr().unwrap().to_string())
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        match listener.accept() {
            Ok((s, _)) => {
                let mut t = TcpTransport::new(s);
                // Each side demonstrates possession of its half of the one-use
                // subprocess secret before receiving any protocol frames.
                assert!(t.recv_frame().unwrap() == session.as_bytes()[..32]);
                t.send_frame(&session.as_bytes()[32..]).unwrap();
                return (t, process);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if let Some(status) = process.try_wait().unwrap() {
                    panic!("peer exited before connect: {status}");
                }
                if Instant::now() >= until {
                    process.kill().unwrap();
                    process.wait().unwrap();
                    panic!("peer connect timed out");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => panic!("accept: {e}"),
        }
    }
}
pub fn connect() -> TcpTransport {
    let mut t = TcpTransport::new(TcpStream::connect(std::env::var(ADDRESS).unwrap()).unwrap());
    let session = std::env::var(SESSION).unwrap();
    t.send_frame(&session.as_bytes()[..32]).unwrap();
    assert!(t.recv_frame().unwrap() == session.as_bytes()[32..]);
    t
}

verus! {

#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct Node {
    pub id: u64,
    pub instance: Tracked<TransferInstance>,
    pub alloc: Tracked<AllocSource>,
    pub sources: Tracked<EventSourceAllocator>,
    pub shares: Tracked<ShareSource>,
    pub host: HostContext,
}

impl Node {
    pub fn new(id: u64) -> Self {
        let (instance, alloc, Tracked(mut sources), shares) = NodeContext::bootstrap(
            id,
        ).into_sources();
        let host = HostContext::new(Tracked(sources.mint(EventSourceKind::host())));
        Self { id, instance, alloc, sources: Tracked(sources), shares, host }
    }

    pub fn name(&mut self) -> Tracked<EventSourceToken> {
        Tracked(self.sources.borrow_mut().mint(EventSourceKind::rdma_qp()))
    }

    pub fn channel(&self, t: TcpTransport) -> Channel<TcpTransport> {
        Channel::new(t, self.id, 1 - self.id, Tracked(self.instance.borrow()))
    }

    pub fn buffer(&mut self, n: usize) -> (HostBuffer, Tracked<SlotToken>) {
        host::alloc(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.alloc.borrow_mut()),
            n,
        ).unwrap()
    }

    pub fn buffer_slab(&mut self, num_slots: usize, slot_len: usize) -> (
        HostBuffer,
        Tracked<Map<usize, SlotToken>>,
    ) {
        host::alloc_slab(
            &self.host,
            Tracked(self.instance.borrow()),
            Tracked(self.alloc.borrow_mut()),
            num_slots,
            slot_len,
            1,
        ).unwrap()
    }

    pub fn empty(&self) -> Tracked<Statement> {
        Tracked(Statement::empty(self.instance@.id()))
    }

    pub fn write(
        &mut self,
        b: &HostBuffer,
        data: &[u8],
        Tracked(mut tok): Tracked<SlotToken>,
        Tracked(s): Tracked<Statement>,
    ) -> (Tracked<Completion>, Tracked<SlotToken>) {
        let result = host::write_slice(
            &mut self.host,
            Tracked(self.instance.borrow()),
            &b.as_slice(),
            data,
            Tracked(&mut tok),
            Tracked(&s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        );
        (result, Tracked(tok))
    }

    pub fn read(
        &mut self,
        b: &HostBuffer,
        data: &mut [u8],
        Tracked(tok): Tracked<SlotToken>,
        Tracked(s): Tracked<Statement>,
    ) -> (Tracked<Completion>, Tracked<SlotToken>) {
        let tracked mut perm = ReadPerm::Owned(tok);
        let result = host::read_slice(
            &mut self.host,
            Tracked(self.instance.borrow()),
            &b.as_slice(),
            data,
            Tracked(&mut perm),
            Tracked(&s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        );
        (result, Tracked(perm.into_owned()))
    }

    pub fn deregister(
        &mut self,
        mr: MemoryRegion,
        Tracked(tok): Tracked<SlotToken>,
        Tracked(s): Tracked<Statement>,
    ) -> (Tracked<Completion>, Tracked<SlotToken>) {
        deregister_memory(
            &mut self.host,
            Tracked(self.instance.borrow()),
            mr,
            Tracked(tok),
            Tracked(&s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        ).map_err(|e| e.0).unwrap()
    }

    pub fn free(
        &mut self,
        b: HostBuffer,
        Tracked(tok): Tracked<SlotToken>,
        Tracked(s): Tracked<Statement>,
    ) {
        host::free(
            &mut self.host,
            Tracked(self.instance.borrow()),
            b,
            Tracked(tok),
            Tracked(&s.submitted),
            Tracked(&s.knows),
            Ghost(s.facts),
        );
    }
}

pub fn statement(Tracked(c): Tracked<Completion>) -> Tracked<Statement> {
    Tracked(
        Statement {
            succeeded: SucceededMap::empty(c.submitted.instance_id()),
            facts: ISet::empty().insert(c.submitted.key()),
            submitted: c.infos(),
            knows: c.knows,
        },
    )
}

pub fn recv_notice(_node: &Node, c: &mut Channel<TcpTransport>) -> Received {
    c.recv_completion_notice(Tracked(_node.instance.borrow())).unwrap()
}

} // verus!
pub fn complete(cq: &mut CompletionQueue, i: Tracked<&TransferInstance>) -> Completed {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match poll_completion_queue(cq, i).unwrap() {
            PollStatus::Empty => {
                assert!(Instant::now() < deadline, "CQ timeout");
                std::thread::yield_now();
            }
            PollStatus::Completed(
                Completed::ControlFailed { wr_id, status }
                | Completed::BindFailed { wr_id, status, .. }
                | Completed::DataFailed { wr_id, status, .. },
            ) => {
                panic!("WR {wr_id} failed, ibv_wc_status {status}")
            }
            PollStatus::Completed(c) => return c,
        }
    }
}

fn select_device(node: &mut Node, requested: Option<&str>) -> Option<DeviceContext> {
    let count = DeviceContext::count().expect("enumerate verbs devices");
    for index in 0..count {
        let device = DeviceContext::open(index, node.id).expect("open verbs device");
        if requested.is_some_and(|name| name != device.name()) {
            device.close().unwrap();
            continue;
        }
        let type2 = device.supports_type_2().expect("query device capabilities");
        println!(
            "node {}, verbs device {} ({index} of {count}), Type-2B MW advertised {type2}",
            node.id,
            device.name()
        );
        if type2 {
            let pd = ProtectionDomain::new(&device, node.name()).expect("allocate probe PD");
            let supported = match alloc_window(&pd, node.name()) {
                Ok(window) => {
                    dealloc_window(window)
                        .map_err(|e| e.0)
                        .expect("destroy probe window");
                    true
                }
                Err(RdmaError(code @ (38 | 95))) => {
                    println!("device {index}: ibv_alloc_mw(TYPE_2) returned OS error {code}");
                    false
                }
                Err(e) => panic!("device {index}: ibv_alloc_mw(TYPE_2) failed: {e:?}"),
            };
            pd.destroy().expect("destroy probe PD");
            if supported {
                return Some(device);
            }
        }
        device.close().unwrap();
    }
    None
}
pub fn device(node: &mut Node, prefer_gdr: bool) -> Option<DeviceContext> {
    let requested = std::env::var("VERICACHE_RDMA_DEV").ok();
    let found = match requested.as_deref() {
        Some(name) => select_device(node, Some(name)),
        None if prefer_gdr => {
            select_device(node, Some("mlx5_4")).or_else(|| select_device(node, None))
        }
        None => select_device(node, None),
    };
    if found.is_none() {
        eprintln!(
            "SKIP hardware RDMA: no usable Type-2B device matching {}",
            requested.as_deref().unwrap_or("any name")
        );
    }
    found
}

pub fn gid_index() -> u8 {
    std::env::var("VERICACHE_RDMA_GID_INDEX")
        .map(|s| s.parse().expect("invalid VERICACHE_RDMA_GID_INDEX"))
        .unwrap_or(0)
}
