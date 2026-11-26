//! Verified hardware backends and composable local and distributed cache engines.
pub mod backends;
pub mod engine;

pub use vericache_kernel::{catalog, frame, kernel, provenance, spec, timeline};
