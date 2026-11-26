//! Concrete `ByteTransport` implementations for the ergonomic `vericache` facade:
//! - [`TcpTransport`]: framed TCP stream kept permanently non-blocking with an internal
//!   receive buffer so `try_recv_frame` drains burst frames in 1 `read()` syscall with zero
//!   `set_nonblocking`/`MSG_PEEK` toggles.
//! - [`NoTransport`]: zero-cost placeholder transport for local-only caches (0 peers).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};
use transfer::transport::ByteTransport;

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
        })
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
        let n = u32::from_be_bytes(len_bytes) as usize;
        if n > MAX_FRAME_LEN {
            return Err(io::Error::other("frame exceeds limit"));
        }
        if avail < 4 + n {
            return Ok(None);
        }
        let start = self.read_pos + 4;
        let end = start + n;
        let frame = self.read_buf[start..end].to_vec();
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
        let n = u32::from_be_bytes(len_bytes) as usize;
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
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        if frame.len() > MAX_FRAME_LEN {
            return Err(io::Error::other("frame exceeds limit"));
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
}
