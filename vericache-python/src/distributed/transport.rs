//! Hardened framed TCP `ByteTransport` for Stage 2 cross-instance RDMA control plane.
//!
//! Guarantees:
//! - Non-blocking `try_recv_frame` buffers partial reads and NEVER blocks mid-frame
//!   (unlike `MSG_PEEK` + blocking `read_exact`).
//! - Explicit peer EOF (`read() == 0`) is surfaced immediately as `UnexpectedEof`
//!   and latches the transport closed so progress loops never hang.
//! - Blocking `recv_frame` and `send_frame` enforce a tight bounded wall-clock deadline
//!   so a crashed or partitioned peer triggers clean teardown instead of stalling vLLM.

use std::io::{self, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};
use transfer::transport::ByteTransport;

const MAX_FRAME_LEN: usize = 32 * 1024 * 1024;
const COMPACT_THRESHOLD: usize = 64 * 1024;

pub struct BufferedTcpTransport {
    stream: TcpStream,
    timeout: Duration,
    read_buf: Vec<u8>,
    read_pos: usize,
    closed: bool,
}

impl BufferedTcpTransport {
    pub fn new(stream: TcpStream, timeout: Duration) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            timeout,
            read_buf: Vec::with_capacity(4096),
            read_pos: 0,
            closed: false,
        })
    }

    #[allow(dead_code)]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn buffered_len(&self) -> usize {
        self.read_buf.len().saturating_sub(self.read_pos)
    }

    fn maybe_compact(&mut self) {
        if self.read_pos == self.read_buf.len() {
            self.read_buf.clear();
            self.read_pos = 0;
        } else if self.read_pos >= COMPACT_THRESHOLD {
            self.read_buf.drain(..self.read_pos);
            self.read_pos = 0;
        }
    }

    /// Drain all currently available bytes from the non-blocking socket into `read_buf`.
    fn drain_socket(&mut self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                ErrorKind::BrokenPipe,
                "peer TCP transport already closed",
            ));
        }
        let mut tmp = [0u8; 8192];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => {
                    self.closed = true;
                    return Err(io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "peer closed TCP control connection",
                    ));
                }
                Ok(n) => {
                    self.read_buf.extend_from_slice(&tmp[..n]);
                    if n < tmp.len() {
                        break;
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.closed = true;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Pop one complete length-prefixed frame if already buffered.
    fn pop_buffered_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        if self.buffered_len() < 4 {
            return Ok(None);
        }
        let p = self.read_pos;
        let n = u32::from_be_bytes([
            self.read_buf[p],
            self.read_buf[p + 1],
            self.read_buf[p + 2],
            self.read_buf[p + 3],
        ]) as usize;
        if n > MAX_FRAME_LEN {
            self.closed = true;
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("control frame exceeds limit ({n} > {MAX_FRAME_LEN})"),
            ));
        }
        let total = 4 + n;
        if self.buffered_len() < total {
            return Ok(None);
        }
        let frame = self.read_buf[p + 4..p + total].to_vec();
        self.read_pos += total;
        self.maybe_compact();
        Ok(Some(frame))
    }
}

impl ByteTransport for BufferedTcpTransport {
    type Error = io::Error;

    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                ErrorKind::BrokenPipe,
                "peer TCP transport already closed",
            ));
        }
        let n = u32::try_from(frame.len()).map_err(|_| {
            self.closed = true;
            io::Error::new(ErrorKind::InvalidInput, "frame too large")
        })?;
        if frame.len() > MAX_FRAME_LEN {
            self.closed = true;
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "frame exceeds MAX_FRAME_LEN",
            ));
        }
        let mut wire = Vec::with_capacity(4 + frame.len());
        wire.extend_from_slice(&n.to_be_bytes());
        wire.extend_from_slice(frame);

        let deadline = Instant::now() + self.timeout;
        let mut written = 0usize;
        let mut spins = 0u32;
        while written < wire.len() {
            match self.stream.write(&wire[written..]) {
                Ok(0) => {
                    self.closed = true;
                    return Err(io::Error::new(
                        ErrorKind::WriteZero,
                        "peer closed TCP control connection during write",
                    ));
                }
                Ok(k) => {
                    written += k;
                    spins = 0;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        self.closed = true;
                        return Err(io::Error::new(
                            ErrorKind::TimedOut,
                            "timed out writing peer TCP control frame",
                        ));
                    }
                    if spins < 32 {
                        thread::yield_now();
                    } else {
                        thread::sleep(Duration::from_micros(50));
                    }
                    spins = spins.saturating_add(1);
                }
                Err(e) => {
                    self.closed = true;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        if let Some(frame) = self.pop_buffered_frame()? {
            return Ok(frame);
        }
        let deadline = Instant::now() + self.timeout;
        let mut spins = 0u32;
        loop {
            match self.try_recv_frame()? {
                Some(frame) => return Ok(frame),
                None => {
                    if Instant::now() >= deadline {
                        self.closed = true;
                        return Err(io::Error::new(
                            ErrorKind::TimedOut,
                            "timed out reading peer TCP control frame",
                        ));
                    }
                    if spins < 64 {
                        thread::yield_now();
                    } else {
                        thread::sleep(Duration::from_micros(50));
                    }
                    spins = spins.saturating_add(1);
                }
            }
        }
    }

    fn try_recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        if let Some(frame) = self.pop_buffered_frame()? {
            return Ok(Some(frame));
        }
        self.drain_socket()?;
        self.pop_buffered_frame()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn socket_pair(timeout: Duration) -> (BufferedTcpTransport, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let transport = BufferedTcpTransport::new(server, timeout).unwrap();
        (transport, client)
    }

    #[test]
    fn partial_reads_never_block_and_assemble_cleanly() {
        let (mut rx, mut tx) = socket_pair(Duration::from_millis(200));
        assert_eq!(rx.try_recv_frame().unwrap(), None);

        let payload = b"hello-vericache-rdma";
        let mut wire = Vec::new();
        wire.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        wire.extend_from_slice(payload);

        // Drip the header + payload in small chunks.
        tx.write_all(&wire[..2]).unwrap();
        thread::sleep(Duration::from_millis(5));
        assert_eq!(rx.try_recv_frame().unwrap(), None);

        tx.write_all(&wire[2..10]).unwrap();
        thread::sleep(Duration::from_millis(5));
        assert_eq!(rx.try_recv_frame().unwrap(), None);

        tx.write_all(&wire[10..]).unwrap();
        let got = rx.recv_frame().unwrap();
        assert_eq!(got, payload);
        assert_eq!(rx.try_recv_frame().unwrap(), None);
    }

    #[test]
    fn coalesced_frames_unpack_in_order() {
        let (mut rx, tx_stream) = socket_pair(Duration::from_millis(200));
        let mut tx = BufferedTcpTransport::new(tx_stream, Duration::from_millis(200)).unwrap();
        tx.send_frame(b"first").unwrap();
        tx.send_frame(b"second-frame").unwrap();
        tx.send_frame(b"").unwrap();
        assert_eq!(rx.recv_frame().unwrap(), b"first");
        assert_eq!(rx.recv_frame().unwrap(), b"second-frame");
        assert_eq!(rx.recv_frame().unwrap(), b"");
    }

    #[test]
    fn peer_disconnect_returns_eof_immediately() {
        let (mut rx, tx) = socket_pair(Duration::from_millis(200));
        drop(tx);
        thread::sleep(Duration::from_millis(5));
        let err = rx.try_recv_frame().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
        assert!(rx.is_closed());
    }

    #[test]
    fn stalled_peer_hits_bounded_timeout() {
        let (mut rx, _tx) = socket_pair(Duration::from_millis(40));
        let start = Instant::now();
        let err = rx.recv_frame().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TimedOut);
        assert!(start.elapsed() < Duration::from_millis(300));
        assert!(rx.is_closed());
    }
}
