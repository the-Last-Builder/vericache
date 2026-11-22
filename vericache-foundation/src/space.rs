//! Space: what an event touches, and when two events conflict.
//!
//! Sets are `ISet`, because vstd's `Set` holds only finite sets and no
//! theorem here needs a footprint to be finite.
use crate::domain::Domain;
use vstd::prelude::*;

verus! {

/// The locations one event reads and writes.
///
/// A copy is one event (reads = source, writes = destination), and so is a
/// kernel. Reclamation (free, unmap, deregistration) is an event with no
/// reads, the reclaimed region as its writes, and an effect that allows any
/// output; premature reclamation is therefore an ordinary conflict.
pub struct Access<D: Domain> {
    pub reads: ISet<D::Loc>,
    pub writes: ISet<D::Loc>,
}

/// The values of a set of locations.
pub type Snapshot<D> = IMap<<D as Domain>::Loc, <D as Domain>::Value>;

/// Allowed behaviors of one event: `effect(input, output)` holds when the
/// event, having read `input` from its read set, may leave `output` in its
/// write set. A relation rather than a function, so nondeterministic and
/// partial writes are expressible.
pub type Effect<D> = spec_fn(Snapshot<D>, Snapshot<D>) -> bool;

/// The values of `locs` in a whole memory.
pub open spec fn restrict<D: Domain>(
    memory: spec_fn(D::Loc) -> D::Value,
    locs: ISet<D::Loc>,
) -> Snapshot<D> {
    IMap::new(|loc: D::Loc| locs.contains(loc), |loc: D::Loc| memory(loc))
}

pub open spec fn overlaps<D: Domain>(a: ISet<D::Loc>, b: ISet<D::Loc>) -> bool {
    exists|loc: D::Loc| a.contains(loc) && b.contains(loc)
}

pub open spec fn conflict_at<D: Domain>(a: Access<D>, b: Access<D>, loc: D::Loc) -> bool {
    ||| a.writes.contains(loc) && b.reads.contains(loc)
    ||| a.writes.contains(loc) && b.writes.contains(loc)
    ||| a.reads.contains(loc) && b.writes.contains(loc)
}

/// Two accesses conflict when they share a location and at least one of them
/// writes it.
pub open spec fn conflict<D: Domain>(a: Access<D>, b: Access<D>) -> bool {
    ||| overlaps::<D>(a.writes, b.reads)
    ||| overlaps::<D>(a.writes, b.writes)
    ||| overlaps::<D>(a.reads, b.writes)
}

pub proof fn conflict_is_symmetric<D: Domain>(a: Access<D>, b: Access<D>)
    ensures
        conflict(a, b) == conflict(b, a),
{
}

pub proof fn conflict_has_location<D: Domain>(a: Access<D>, b: Access<D>)
    requires
        conflict(a, b),
    ensures
        exists|loc: D::Loc| #[trigger] conflict_at(a, b, loc),
{
    let (writer, other) = if overlaps::<D>(a.writes, b.reads) || overlaps::<D>(a.writes, b.writes) {
        (a, b)
    } else {
        conflict_is_symmetric(a, b);
        (b, a)
    };
    let loc = choose|loc: D::Loc|
        writer.writes.contains(loc) && (other.reads.contains(loc) || other.writes.contains(loc));
    assert(conflict_at(a, b, loc));
}

/// A writer and a reader of the same location conflict.
pub proof fn write_conflicts_with_read<D: Domain>(writer: Access<D>, reader: Access<D>, loc: D::Loc)
    requires
        writer.writes.contains(loc),
        reader.reads.contains(loc),
    ensures
        conflict(writer, reader),
        conflict_at(writer, reader, loc),
{
}

/// Two writers of the same location conflict.
pub proof fn writes_conflict<D: Domain>(a: Access<D>, b: Access<D>, loc: D::Loc)
    requires
        a.writes.contains(loc),
        b.writes.contains(loc),
    ensures
        conflict(a, b),
        conflict_at(a, b, loc),
{
}

} // verus!
