//! VeriCache-foundation: when asynchronous memory accesses do not race, and
//! what values they then read.
//!
//! Modules, in dependency order:
//!
//! - `domain`: the five types a consumer chooses.
//! - `space`: accesses and conflicts.
//! - `order`: strict partial orders and predecessor-closed sets.
//! - `history`: the `History` state machine and `race_free`.
//! - `value`: runs, `every_event_isolated` and `read_returns_last_write`.
//! - `refinement`: how backends drive the history.
//! - `tracked`: the history as tokens that many threads and nodes can hold.
//! - `write_once`: write-once persistent key-value map tokens.
pub mod domain;
pub mod history;
pub mod order;
pub mod refinement;
pub mod space;
pub mod tracked;
pub mod value;
pub mod write_once;
