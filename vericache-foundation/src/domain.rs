//! The types a consumer chooses. Foundation never inspects them; it only
//! compares them for equality.
use vstd::prelude::*;

verus! {

pub trait Domain {
    /// A physical storage location, e.g. `(node, memory space, byte offset)`.
    /// Every handle that reaches the same physical bytes maps to the same
    /// `Loc`, so conflicts across APIs are visible.
    type Loc;

    /// One occurrence of an access.
    type Event;

    /// What one location holds; `u8` for byte-addressed locations.
    type Value;

    /// An agent that submits and observes events, e.g. one node's controller.
    type Observer;

    /// Names one revocable read share.
    type Share;
}

} // verus!
