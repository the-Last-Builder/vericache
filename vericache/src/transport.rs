//! Concrete `ByteTransport` implementations for the ergonomic `vericache` facade:
//! - [`TcpTransport`]: framed TCP stream with non-blocking `try_recv_frame` for
//!   deadlock-free full-duplex RDMA control-plane progress.
//! - [`NoTransport`]: zero-cost placeholder transport for local-only caches (0 peers).

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use transfer::transport::ByteTransport;

const MAX_FRAME_LEN: usize = 32 * 1024 * 1024;

/// Framed TCP control-plane transport supporting both blocking `recv_frame`
/// and non-blocking `try_recv_frame` (via `MSG_PEEK`).
pub struct TcpTransport {
    stream: TcpStream,
}

impl TcpTransport {
    pub fn new(stream: TcpStream) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        stream.set_write_timeout(Some(Duration::from_secs(30)))?;
        Ok(Self { stream })
    }

    pub fn with_timeout(stream: TcpStream, timeout: Duration) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        Ok(Self { stream })
    }

    pub fn inner(&self) -> &TcpStream {
        &self.stream
    }

    /// Non-blocking check whether at least 1 byte is queued on the TCP peer socket.
    fn has_pending_data(&self) -> io::Result<bool> {
        let mut probe = [0u8; 1];
        self.stream.set_nonblocking(true)?;
        let res = match self.stream.peek(&mut probe) {
            Ok(n) => Ok(n > 0),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(false),
            Err(e) => Err(e),
        };
        let restore = self.stream.set_nonblocking(false);
        res.and_then(|ready| restore.map(|()| ready))
    }
}

impl ByteTransport for TcpTransport {
    type Error = io::Error;

    fn send_frame(&mut self, frame: &[u8]) -> io::Result<()> {
        let n = u32::try_from(frame.len()).map_err(|_| io::Error::other("frame too large"))?;
        self.stream.write_all(&n.to_be_bytes())?;
        self.stream.write_all(frame)
    }

    fn recv_frame(&mut self) -> io::Result<Vec<u8>> {
        let mut prefix = [0u8; 4];
        self.stream.read_exact(&mut prefix)?;
        let n = u32::from_be_bytes(prefix) as usize;
        if n > MAX_FRAME_LEN {
            return Err(io::Error::other("frame exceeds limit"));
        }
        let mut frame = vec![0u8; n];
        self.stream.read_exact(&mut frame)?;
        Ok(frame)
    }

    fn try_recv_frame(&mut self) -> io::Result<Option<Vec<u8>>> {
        if !self.has_pending_data()? {
            return Ok(None);
        }
        self.recv_frame().map(Some)
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
