//! Verified hardware drivers for Host CPU, CUDA, NVMe, and RDMA.
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod host;
#[cfg(feature = "rdma")]
pub mod integration_proofs;
#[cfg(feature = "nvme")]
pub mod nvme;
#[cfg(feature = "rdma")]
pub mod rdma;

pub use transfer_core::{context, domain, memory, transport};
