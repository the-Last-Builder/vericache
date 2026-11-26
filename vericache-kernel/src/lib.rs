//! Verified cache state, transactions, and the interface for pluggable backends.
pub mod catalog;
#[cfg(verus_only)]
pub mod compose;
pub mod frame;
pub mod kernel;
pub mod provenance;
pub mod spec;
pub mod spi;
pub mod timeline;
