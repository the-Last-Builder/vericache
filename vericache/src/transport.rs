//! Concrete `ByteTransport` implementations for the ergonomic `vericache` facade:
//! - [`TcpTransport`]: framed TCP stream kept permanently non-blocking with an internal
//!   receive buffer so `try_recv_frame` drains burst frames in 1 `read()` syscall with zero
//!   `set_nonblocking`/`MSG_PEEK` toggles.
//! - [`NoTransport`]: zero-cost placeholder transport for local-only caches (0 peers).

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use transfer::transport::{ByteTransport, GrantTag};
use vericache_core::backends::rdma::protocol::{
    PROTOCOL_MAGIC, READ_GRANT_META_LEN, ReadGrantMeta, decode_read_grant_meta,
};

const NVME_WARM_FLAG: u32 = 1 << 31;
const MAX_TELEMETRY_REQUESTS: usize = 4096;

/// Metadata on the TCP envelope only. The verified channel receives the
/// original bytes, including its original integrity trailer.
#[derive(Default)]
pub(crate) struct PeerPathTelemetry {
    outgoing: BTreeMap<u64, (u64, u64)>,
    incoming: BTreeMap<u64, GrantTag>,
}

impl PeerPathTelemetry {
    pub(crate) fn mark_donor_warm(&mut self, request_id: u64, address: u64, version: u64) {
        self.outgoing.insert(request_id, (address, version));
        if self.outgoing.len() > MAX_TELEMETRY_REQUESTS {
            self.outgoing.pop_first();
        }
    }

    pub(crate) fn take_donor_warm(&mut self, request_id: u64, tag: GrantTag) -> bool {
        self.incoming.remove(&request_id) == Some(tag)
    }
}

fn read_grant(frame: &[u8]) -> Option<(ReadGrantMeta, GrantTag, u64)> {
    // Transfer channel framing: magic prefix, READ_GRANT header, payload, CRC32.
    const MAGIC_PREFIX: usize = 5;
    const GRANT_HEADER: usize = 29;
    let meta_start = MAGIC_PREFIX + GRANT_HEADER;
    if frame.len() != meta_start + READ_GRANT_META_LEN + 4
        || frame[0] != 0xff
        || frame[1..5] != PROTOCOL_MAGIC.to_le_bytes()
        || frame[5] != 1
    {
        return None;
    }
    let tag = u64::from_le_bytes(frame[6..14].try_into().ok()?);
    let address = u64::from_le_bytes(frame[14..22].try_into().ok()?);
    let meta = decode_read_grant_meta(&frame[meta_start..meta_start + READ_GRANT_META_LEN])?;
    Some((meta, tag, address))
}

const MAX_FRAME_LEN: usize = 32 * 1024 * 1024;
const READ_CHUNK_SIZE: usize = 16 * 1024;
const READ_DRAIN_CAP: usize = 64 * 1024;
const COMPACT_THRESHOLD: usize = 64 * 1024;

/// Framed TCP control-plane transport supporting both blocking `recv_frame`
/// and zero-toggle non-blocking `try_recv_frame`.
pub struct TcpTransport {
    stream: TcpStream,
    timeout: Duration,
    read_buf: Vec<u8>,
    read_pos: usize,
    write_buf: Vec<u8>,
    recv_paused: Arc<AtomicBool>,
    path_telemetry: Arc<Mutex<PeerPathTelemetry>>,
}

impl TcpTransport {
    pub fn new(stream: TcpStream) -> io::Result<Self> {
        Self::with_timeout(stream, Duration::from_secs(30))
    }

    pub fn with_timeout(stream: TcpStream, timeout: Duration) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            timeout,
            read_buf: Vec::with_capacity(READ_CHUNK_SIZE),
            read_pos: 0,
            write_buf: Vec::with_capacity(256),
            recv_paused: Arc::new(AtomicBool::new(false)),
            path_telemetry: Arc::default(),
        })
    }

    /// Return a shared handle that pauses non-blocking `try_recv_frame` while `true`
    /// without affecting blocking `recv_frame` or `send_frame`.
    pub fn recv_pause_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.recv_paused)
    }

    /// Toggle whether non-blocking `try_recv_frame` suppresses reading new frames.
    pub fn set_recv_paused(&self, paused: bool) {
        self.recv_paused.store(paused, Ordering::Release);
    }

    pub(crate) fn path_telemetry(&self) -> Arc<Mutex<PeerPathTelemetry>> {
        Arc::clone(&self.path_telemetry)
    }

    pub fn inner(&self) -> &TcpStream {
        &self.stream
    }

    #[inline]
    fn buffered_len(&self) -> usize {
        self.read_buf.len() - self.read_pos
    }

    fn maybe_compact(&mut self) {
        if self.read_pos == self.read_buf.len() {
            self.read_buf.clear();
            self.read_pos = 0;
        } else if self.read_pos >= COMPACT_THRESHOLD && self.read_pos * 2 >= self.read_buf.len() {
            self.read_buf.drain(..self.read_pos);
            self.read_pos = 0;
        }
    }

    /// Extract one complete length-prefixed frame if already present in `read_buf`.
    fn try_extract_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        let avail = self.buffered_len();
        if avail < 4 {
            return Ok(None);
        }
        let len_bytes: [u8; 4] = self.read_buf[self.read_pos..self.read_pos + 4]
            .try_into()
            .expect("4-byte prefix");
        let wire_len = u32::from_be_bytes(len_bytes);
        let n = (wire_len & !NVME_WARM_FLAG) as usize;
        if n > MAX_FRAME_LEN {
            return Err(io::Error::other("frame exceeds limit"));
        }
        if avail < 4 + n {
            return Ok(None);
        }
        let start = self.read_pos + 4;
        let end = start + n;
        let frame = self.read_buf[start..end].to_vec();
        if wire_len & NVME_WARM_FLAG != 0 {
            if let Some((meta, tag, _)) = read_grant(&frame) {
                let mut telemetry = self.path_telemetry.lock().unwrap();
                telemetry.incoming.insert(meta.request_id, tag);
                if telemetry.incoming.len() > MAX_TELEMETRY_REQUESTS {
                    telemetry.incoming.pop_first();
                }
            }
        }
        self.read_pos = end;
        self.maybe_compact();
        Ok(Some(frame))
    }

    /// Expected bytes needed to finish the next frame (or 4 if length prefix incomplete).
    fn next_frame_needed(&self) -> io::Result<usize> {
        let avail = self.buffered_len();
        if avail < 4 {
            return Ok(4);
        }
        let len_bytes: [u8; 4] = self.read_buf[self.read_pos..self.read_pos + 4]
            .try_into()
            .expect("4-byte prefix");
        let n = (u32::from_be_bytes(len_bytes) & !NVME_WARM_FLAG) as usize;
        if n > MAX_FRAME_LEN {
            return Err(io::Error::other("frame exceeds limit"));
        }
        Ok(4 + n)
    }

    /// Non-blockingly drain immediately readable socket bytes into `read_buf`.
    /// Returns `(bytes_read, hit_eof)`.
    fn drain_socket_nonblocking(&mut self) -> io::Result<(usize, bool)> {
        let mut tmp = [0u8; READ_CHUNK_SIZE];
        let mut total = 0usize;
        let mut eof = false;
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => {
                    eof = true;
                    break;
                }
                Ok(k) => {
                    self.read_buf.extend_from_slice(&tmp[..k]);
                    total += k;
                    if k < tmp.len() {
                        break;
                    }
                    let needed = self.next_frame_needed()?;
                    if self.buffered_len() >= needed && total >= READ_DRAIN_CAP {
                        break;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok((total, eof))
    }
}

impl ByteTransport for TcpTransport {
    type Error = io::Error;

    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        let mut n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        if frame.len() > MAX_FRAME_LEN {
            return Err(io::Error::other("frame exceeds limit"));
        }
        if let Some((meta, _, address)) = read_grant(frame) {
            let mut telemetry = self.path_telemetry.lock().unwrap();
            if telemetry.outgoing.remove(&meta.request_id) == Some((address, meta.version.pack())) {
                n |= NVME_WARM_FLAG;
            }
        }
        // Coalesce 4-byte length prefix + payload so control frames go out in a single write().
        self.write_buf.clear();
        self.write_buf.reserve(4 + frame.len());
        self.write_buf.extend_from_slice(&n.to_be_bytes());
        self.write_buf.extend_from_slice(frame);

        let mut written = 0usize;
        let mut spins = 0u32;
        let deadline = Instant::now() + self.timeout;
        while written < self.write_buf.len() {
            match self.stream.write(&self.write_buf[written..]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole frame to TCP stream",
                    ));
                }
                Ok(k) => {
                    written += k;
                    spins = 0;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "TCP send_frame timed out",
                        ));
                    }
                    spins += 1;
                    if spins <= 16 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(Duration::from_micros(50));
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        let mut spins = 0u32;
        let deadline = Instant::now() + self.timeout;
        loop {
            if let Some(frame) = self.try_extract_frame()? {
                return Ok(frame);
            }
            let (read_bytes, eof) = self.drain_socket_nonblocking()?;
            if let Some(frame) = self.try_extract_frame()? {
                return Ok(frame);
            }
            if eof {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed TCP connection",
                ));
            }
            if read_bytes > 0 {
                spins = 0;
                continue;
            }
            if Instant::now() >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "TCP recv_frame timed out",
                ));
            }
            spins += 1;
            if spins <= 16 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    fn try_recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        if self.recv_paused.load(Ordering::Acquire) {
            return Ok(None);
        }
        if let Some(frame) = self.try_extract_frame()? {
            return Ok(Some(frame));
        }
        let (_read_bytes, eof) = self.drain_socket_nonblocking()?;
        if let Some(frame) = self.try_extract_frame()? {
            return Ok(Some(frame));
        }
        if eof && self.buffered_len() > 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated frame before TCP EOF",
            ));
        }
        Ok(None)
    }
}

/// Placeholder `ByteTransport` for single-node local caches with 0 RDMA peers.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoTransport;

impl ByteTransport for NoTransport {
    type Error = io::Error;

    fn send_frame(&mut self, _frame: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::NotConnected,
            "NoTransport cannot send peer frames",
        ))
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        Err(io::Error::new(
            io::ErrorKind::NotConnected,
            "NoTransport cannot receive peer frames",
        ))
    }

    fn try_recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn donor_warm_metadata_crosses_tcp_without_changing_verified_grant_bytes() {
        use vericache_core::backends::rdma::protocol::encode_read_grant_meta;
        use vericache_core::spec::{ObjectKey, VersionId};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let mut tx = TcpTransport::new(client).unwrap();
        let mut rx = TcpTransport::new(server).unwrap();
        let meta = ReadGrantMeta {
            request_id: 7,
            key: ObjectKey([11, 12, 13, 14]),
            version: VersionId { origin_node: 1, seq: 3 },
            frame_size: 4096,
        };
        let (tag, address) = (21u64, 8192u64);
        let mut frame = vec![0xff];
        frame.extend_from_slice(&PROTOCOL_MAGIC.to_le_bytes());
        frame.push(1);
        frame.extend_from_slice(&tag.to_le_bytes());
        frame.extend_from_slice(&address.to_le_bytes());
        frame.extend_from_slice(&meta.frame_size.to_le_bytes());
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.extend_from_slice(&encode_read_grant_meta(meta));
        // The transport deliberately leaves the CRC bytes opaque.
        frame.extend_from_slice(&[1, 2, 3, 4]);

        tx.path_telemetry().lock().unwrap().mark_donor_warm(7, address + 4096, meta.version.pack());
        tx.send_frame(&frame).unwrap();
        assert_eq!(rx.recv_frame().unwrap(), frame);
        assert!(!rx.path_telemetry().lock().unwrap().take_donor_warm(7, tag));

        tx.path_telemetry().lock().unwrap().mark_donor_warm(7, address, meta.version.pack());
        tx.send_frame(&frame).unwrap();
        assert_eq!(rx.recv_frame().unwrap(), frame);
        assert!(rx.path_telemetry().lock().unwrap().take_donor_warm(7, tag));
        assert!(!rx.path_telemetry().lock().unwrap().take_donor_warm(7, tag));
    }

    #[test]
    fn tcp_transport_nonblocking_buffered_burst_and_large_frame() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");

        let client_stream = TcpStream::connect(addr).expect("connect");
        let (server_stream, _) = listener.accept().expect("accept");

        let mut tx = TcpTransport::with_timeout(client_stream, Duration::from_secs(2))
            .expect("tx transport");
        let mut rx = TcpTransport::with_timeout(server_stream, Duration::from_secs(2))
            .expect("rx transport");

        // Empty socket returns Ok(None) immediately without blocking.
        assert_eq!(rx.try_recv_frame().expect("empty try_recv"), None);

        // Send a burst of 32 frames; receiver drains them cleanly via internal buffer.
        for i in 0..32u32 {
            let payload = format!("frame-{i:04}-payload").into_bytes();
            tx.send_frame(&payload).expect("send burst frame");
        }
        for i in 0..32u32 {
            let expected = format!("frame-{i:04}-payload").into_bytes();
            let got = rx.recv_frame().expect("recv burst frame");
            assert_eq!(got, expected);
        }
        assert_eq!(rx.try_recv_frame().expect("drained try_recv"), None);

        // Send a large 256 KiB frame (> READ_DRAIN_CAP).
        let big: Vec<u8> = (0..256 * 1024).map(|i| (i % 251) as u8).collect();
        let big_clone = big.clone();
        let sender = std::thread::spawn(move || {
            tx.send_frame(&big_clone).expect("send big frame");
            tx
        });
        let got_big = rx.recv_frame().expect("recv big frame");
        assert_eq!(got_big, big);
        let _tx = sender.join().expect("join sender");
    }

    #[test]
    fn tcp_transport_recv_pause_suppresses_try_recv_without_dropping_frames() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");

        let client_stream = TcpStream::connect(addr).expect("connect");
        let (server_stream, _) = listener.accept().expect("accept");

        let mut tx = TcpTransport::with_timeout(client_stream, Duration::from_secs(2))
            .expect("tx transport");
        let mut rx = TcpTransport::with_timeout(server_stream, Duration::from_secs(2))
            .expect("rx transport");
        let pause = rx.recv_pause_handle();

        // Send 2 frames so frame 1 is read into internal read_buf along with frame 2.
        tx.send_frame(b"first-frame").expect("send 1");
        tx.send_frame(b"second-frame").expect("send 2");
        assert_eq!(rx.recv_frame().expect("recv 1"), b"first-frame");

        // Pause non-blocking try_recv_frame: buffered frame 2 AND newly sent frame 3 must
        // NOT be consumed by try_recv_frame while paused.
        pause.store(true, Ordering::Release);
        tx.send_frame(b"third-frame").expect("send 3");
        assert_eq!(rx.try_recv_frame().expect("paused try_recv"), None);
        assert_eq!(rx.try_recv_frame().expect("paused try_recv 2"), None);

        // Unpause: both frames arrive completely intact in order.
        pause.store(false, Ordering::Release);
        assert_eq!(rx.recv_frame().expect("recv 2"), b"second-frame");
        assert_eq!(rx.recv_frame().expect("recv 3"), b"third-frame");
        assert_eq!(rx.try_recv_frame().expect("drained"), None);
    }
}
