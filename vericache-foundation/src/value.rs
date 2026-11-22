//! Value: what actually happened to memory, and what each read returns.
//!
//! A `Run` is a discrete timeline of memory states plus when each event ran.
//! `valid_run` ties it to a history by its frame, effect, order, fencing
//! and interval conditions. In every reachable history, in a valid run
//! every event is isolated and every accepted read sees its last ordered write.
use crate::domain::Domain;
#[cfg(verus_only)]
use crate::history::*;
#[cfg(verus_only)]
use crate::order::*;
#[cfg(verus_only)]
use crate::space::*;
use vstd::prelude::*;

verus! {

pub struct Run<D: Domain> {
    /// `memory(time)(loc)`: the value of `loc` at `time`.
    pub memory: spec_fn(nat) -> spec_fn(D::Loc) -> D::Value,
    pub start: spec_fn(D::Event) -> nat,
    /// `None`: the event never ends.
    pub end: spec_fn(D::Event) -> Option<nat>,
    pub succeeded: spec_fn(D::Event) -> bool,
    /// The fence rejected this read before the event touched the location.
    pub rejected_read: spec_fn(D::Event, D::Loc) -> bool,
}

pub open spec fn active<D: Domain>(run: Run<D>, event: D::Event, time: nat) -> bool {
    &&& (run.start)(event) <= time
    &&& match (run.end)(event) {
        None => true,
        Some(end) => time < end,
    }
}

/// `event` has ended at or before `time`.
pub open spec fn finished_by<D: Domain>(run: Run<D>, event: D::Event, time: nat) -> bool {
    exists|end: nat| (run.end)(event) == Some(end) && end <= time
}

pub open spec fn intervals_overlap<D: Domain>(run: Run<D>, a: D::Event, b: D::Event) -> bool {
    exists|time: nat| active(run, a, time) && active(run, b, time)
}

pub open spec fn effective_access<D: Domain>(h: History<D>, run: Run<D>, event: D::Event) -> Access<
    D,
> {
    Access {
        reads: ISet::new(
            |loc: D::Loc| (h.access)(event).reads.contains(loc) && !(run.rejected_read)(event, loc),
        ),
        writes: (h.access)(event).writes,
    }
}

/// No effectively conflicting event runs at the same time as `event`.
pub open spec fn isolated<D: Domain>(h: History<D>, run: Run<D>, event: D::Event) -> bool {
    forall|other: D::Event|
        h.events.contains(other) && other != event && #[trigger] conflict(
            effective_access(h, run, event),
            effective_access(h, run, other),
        ) ==> !intervals_overlap(run, event, other)
}

/// Some submitted event that writes `loc` is active at `time`.
pub open spec fn writer_active<D: Domain>(
    h: History<D>,
    run: Run<D>,
    loc: D::Loc,
    time: nat,
) -> bool {
    exists|writer: D::Event|
        h.events.contains(writer) && (h.access)(writer).writes.contains(loc) && #[trigger] active(
            run,
            writer,
            time,
        )
}

// ---------------------------------------------------------------------------
// The conditions of a valid run
// ---------------------------------------------------------------------------
/// An event ends no earlier than it starts.
pub open spec fn intervals_well_formed<D: Domain>(run: Run<D>) -> bool {
    forall|event: D::Event|
        match #[trigger] (run.end)(event) {
            Some(end) => (run.start)(event) <= end,
            None => true,
        }
}

/// Frame: a location changes only while some event that writes it is active.
pub open spec fn frame_condition<D: Domain>(h: History<D>, run: Run<D>) -> bool {
    forall|time: nat, loc: D::Loc| #[trigger]
        changes_at(run, loc, time) ==> writer_active(h, run, loc, time)
}

/// `loc` holds a different value at `time + 1` than at `time`.
pub open spec fn changes_at<D: Domain>(run: Run<D>, loc: D::Loc, time: nat) -> bool {
    (run.memory)(time + 1)(loc) != (run.memory)(time)(loc)
}

/// Effect: an isolated event that succeeds behaves as its effect allows.
pub open spec fn effect_condition<D: Domain>(h: History<D>, run: Run<D>) -> bool {
    forall|event: D::Event|
        h.events.contains(event) && (run.succeeded)(event) && #[trigger] isolated(h, run, event)
            ==> match (run.end)(event) {
            Some(end) => ((h.effect)(event))(
                restrict::<D>((run.memory)((run.start)(event)), (h.access)(event).reads),
                restrict::<D>((run.memory)(end), (h.access)(event).writes),
            ),
            None => true,
        }
}

/// Order: happens-before means the earlier event ends before the later starts.
pub open spec fn order_condition<D: Domain>(h: History<D>, run: Run<D>) -> bool {
    forall|a: D::Event, b: D::Event| #[trigger]
        (h.happens_before)(a, b) ==> finished_by(run, a, (run.start)(b))
}

/// A guarded read finishes by its fence or is rejected before it touches the location.
pub open spec fn fencing_condition<D: Domain>(h: History<D>, run: Run<D>) -> bool {
    &&& forall|event: D::Event, loc: D::Loc| #[trigger]
        (run.rejected_read)(event, loc) ==> !(run.succeeded)(event)
    &&& forall|event: D::Event, loc: D::Loc|
        h.events.contains(event) && (h.access)(event).reads.contains(loc) && #[trigger] (
        h.read_guards)(event).dom().contains(loc) && h.revoked.dom().contains(
            (loc, (h.read_guards)(event)[loc]),
        ) ==> (run.rejected_read)(event, loc) || exists|fence_end: nat|
            (run.end)(h.revoked[(loc, (h.read_guards)(event)[loc])]) == Some(fence_end)
                && #[trigger] finished_by(run, event, fence_end)
}

pub open spec fn valid_run<D: Domain>(h: History<D>, run: Run<D>) -> bool {
    &&& intervals_well_formed(run)
    &&& frame_condition(h, run)
    &&& effect_condition(h, run)
    &&& fencing_condition(h, run)
    &&& order_condition(h, run)
}

// ---------------------------------------------------------------------------
// Isolation
// ---------------------------------------------------------------------------
/// An accepted read fenced before a writer has ended by the writer's start.
proof fn fenced_read_finishes_before_writer<D: Domain>(
    h: History<D>,
    run: Run<D>,
    reader: D::Event,
    writer: D::Event,
    loc: D::Loc,
)
    requires
        valid_run(h, run),
        h.events.contains(reader),
        effective_access(h, run, reader).reads.contains(loc),
        read_fenced_before(h, reader, writer, loc),
    ensures
        finished_by(run, reader, (run.start)(writer)),
{
    let fence = h.revoked[(loc, (h.read_guards)(reader)[loc])];
    assert(finished_by(run, fence, (run.start)(writer)));
}

/// In a reachable history, every event of a valid run is isolated.
pub proof fn every_event_isolated<D: Domain>(h: History<D>, run: Run<D>, event: D::Event)
    requires
        reachable(h),
        valid_run(h, run),
        h.events.contains(event),
    ensures
        isolated(h, run, event),
{
    race_free(h);
    assert forall|other: D::Event|
        h.events.contains(other) && other != event && #[trigger] conflict(
            effective_access(h, run, event),
            effective_access(h, run, other),
        ) implies !intervals_overlap(run, event, other) by {
        let first = effective_access(h, run, event);
        let second = effective_access(h, run, other);
        conflict_has_location(first, second);
        let loc = choose|loc: D::Loc| #[trigger] conflict_at(first, second, loc);
        assert(conflict_at((h.access)(event), (h.access)(other), loc));
        if !(h.happens_before)(event, other) && !(h.happens_before)(other, event) {
            reachable_is_well_formed(h);
            if read_fenced_before(h, event, other, loc) {
                fenced_read_finishes_before_writer(h, run, event, other, loc);
            } else {
                fenced_read_finishes_before_writer(h, run, other, event, loc);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Read values
// ---------------------------------------------------------------------------
/// Submitted events that write `loc` and happen before `reader`.
pub open spec fn writers_before<D: Domain>(h: History<D>, reader: D::Event, loc: D::Loc) -> ISet<
    D::Event,
> {
    ISet::new(
        |writer: D::Event|
            h.events.contains(writer) && (h.access)(writer).writes.contains(loc) && (
            h.happens_before)(writer, reader),
    )
}

/// The value of `loc` when `event` starts.
pub open spec fn value_at_start<D: Domain>(run: Run<D>, event: D::Event, loc: D::Loc) -> D::Value {
    (run.memory)((run.start)(event))(loc)
}

/// What `reader` sees in `loc`: the initial value if no write happens before
/// it, otherwise the value left by the last such write when it ended.
pub open spec fn reads_last_write<D: Domain>(
    h: History<D>,
    run: Run<D>,
    reader: D::Event,
    loc: D::Loc,
) -> bool {
    let writers = writers_before(h, reader, loc);
    if writers == ISet::<D::Event>::empty() {
        value_at_start(run, reader, loc) == (run.memory)(0)(loc)
    } else {
        exists|last_writer: D::Event, last_end: nat| #[trigger]
            is_last(h.happens_before, writers, last_writer) && (run.end)(last_writer) == Some(
                last_end,
            ) && value_at_start(run, reader, loc) == #[trigger] (run.memory)(last_end)(loc)
    }
}

/// If no writer of `loc` is active at any time in `[from, to)`, `loc` holds
/// the same value at `to` as at `from`.
proof fn unchanged_without_active_writer<D: Domain>(
    h: History<D>,
    run: Run<D>,
    loc: D::Loc,
    from: nat,
    to: nat,
)
    requires
        frame_condition(h, run),
        from <= to,
        forall|time: nat| from <= time < to ==> !#[trigger] writer_active(h, run, loc, time),
    ensures
        (run.memory)(to)(loc) == (run.memory)(from)(loc),
    decreases to - from,
{
    if from < to {
        let previous = (to - 1) as nat;
        unchanged_without_active_writer(h, run, loc, from, previous);
        assert(!writer_active(h, run, loc, previous));
        assert(!changes_at(run, loc, previous));
    }
}

/// Between `from` and the start of `reader`, no writer of `loc` is active,
/// provided every write ordered before `reader` has finished by `from`.
///
/// A conflicting writer either happens before the read or starts after it
/// finishes. Fencing gives the same finish bound for an accepted read.
proof fn no_writer_active_before_read<D: Domain>(
    h: History<D>,
    run: Run<D>,
    reader: D::Event,
    loc: D::Loc,
    from: nat,
)
    requires
        reachable(h),
        valid_run(h, run),
        h.events.contains(reader),
        (h.access)(reader).reads.contains(loc),
        !(run.rejected_read)(reader, loc),
        forall|writer: D::Event| #[trigger]
            writers_before(h, reader, loc).contains(writer) ==> finished_by(run, writer, from),
    ensures
        forall|time: nat|
            from <= time < (run.start)(reader) ==> !#[trigger] writer_active(h, run, loc, time),
{
    race_free(h);
    assert forall|time: nat| from <= time < (run.start)(reader) implies !#[trigger] writer_active(
        h,
        run,
        loc,
        time,
    ) by {
        if writer_active(h, run, loc, time) {
            let writer = choose|writer: D::Event|
                h.events.contains(writer) && (h.access)(writer).writes.contains(loc)
                    && #[trigger] active(run, writer, time);
            assert(writer != reader);
            reachable_is_well_formed(h);
            write_conflicts_with_read((h.access)(writer), (h.access)(reader), loc);
            assert(!(h.read_guards)(writer).dom().contains(loc));
            if read_fenced_before(h, reader, writer, loc) {
                fenced_read_finishes_before_writer(h, run, reader, writer, loc);
            }
            if !(h.happens_before)(writer, reader) {
                // `reader` would end by `writer`'s start, yet it has not started.
                assert(match (run.end)(reader) {
                    Some(end) => (run.start)(reader) <= end,
                    None => true,
                });
            } else {
                assert(writers_before(h, reader, loc).contains(writer));
            }
        }
    }
}

/// Writers of one location are pairwise ordered, since they conflict.
proof fn writers_form_chain<D: Domain>(h: History<D>, reader: D::Event, loc: D::Loc)
    requires
        reachable(h),
    ensures
        is_chain(h.happens_before, writers_before(h, reader, loc)),
{
    race_free(h);
    let writers = writers_before(h, reader, loc);
    assert forall|a: D::Event, b: D::Event|
        writers.contains(a) && writers.contains(b) && a != b implies #[trigger] (h.happens_before)(
        a,
        b,
    ) || #[trigger] (h.happens_before)(b, a) by {
        reachable_is_well_formed(h);
        assert(!(h.read_guards)(a).dom().contains(loc) && !(h.read_guards)(b).dom().contains(loc));
        writes_conflict((h.access)(a), (h.access)(b), loc);
    }
}

/// In a reachable history and a valid run, each accepted read of `loc` returns the
/// last write to `loc` ordered before it, or the initial value if there is
/// none.
pub proof fn read_returns_last_write<D: Domain>(
    h: History<D>,
    run: Run<D>,
    reader: D::Event,
    loc: D::Loc,
)
    requires
        reachable(h),
        valid_run(h, run),
        h.events.contains(reader),
        (h.access)(reader).reads.contains(loc),
        !(run.rejected_read)(reader, loc),
    ensures
        reads_last_write(h, run, reader, loc),
{
    let writers = writers_before(h, reader, loc);
    if writers == ISet::<D::Event>::empty() {
        no_writer_active_before_read(h, run, reader, loc, 0);
        unchanged_without_active_writer(h, run, loc, 0, (run.start)(reader));
    } else {
        reachable_is_well_formed(h);
        vstd::iset_lib::lemma_iset_subset_finite(h.events, writers);
        writers_form_chain(h, reader, loc);
        finite_chain_has_last(h.happens_before, writers);
        let last_writer = choose|last_writer: D::Event|
            is_last(h.happens_before, writers, last_writer);
        assert(writers.contains(last_writer));
        // The last writer ends before `reader` starts.
        assert(finished_by(run, last_writer, (run.start)(reader)));
        let last_end = choose|end: nat|
            (run.end)(last_writer) == Some(end) && end <= (run.start)(reader);

        // Every writer has finished by `last_end`: the others end before
        // `last_writer` starts.
        assert forall|writer: D::Event| #[trigger] writers.contains(writer) implies finished_by(
            run,
            writer,
            last_end,
        ) by {
            if writer != last_writer {
                assert((h.happens_before)(writer, last_writer));
                assert(match (run.end)(last_writer) {
                    Some(end) => (run.start)(last_writer) <= end,
                    None => true,
                });
            }
        }
        no_writer_active_before_read(h, run, reader, loc, last_end);
        unchanged_without_active_writer(h, run, loc, last_end, (run.start)(reader));
        assert(is_last(h.happens_before, writers, last_writer));
    }
}

pub proof fn succeeded_read_returns_last_write<D: Domain>(
    h: History<D>,
    run: Run<D>,
    reader: D::Event,
    loc: D::Loc,
)
    requires
        reachable(h),
        valid_run(h, run),
        h.events.contains(reader),
        (h.access)(reader).reads.contains(loc),
        (run.succeeded)(reader),
    ensures
        reads_last_write(h, run, reader, loc),
{
    read_returns_last_write(h, run, reader, loc);
}

/// A succeeded read ordered after `writer` observes `writer`'s final value for
/// `loc`, provided every other writer before `reader` either happens before
/// `writer` or is fenced after `reader`.
pub proof fn lemma_guarded_read_observes_writer<D: Domain>(
    h: History<D>,
    run: Run<D>,
    reader: D::Event,
    writer: D::Event,
    loc: D::Loc,
)
    requires
        reachable(h),
        valid_run(h, run),
        h.events.contains(reader),
        h.events.contains(writer),
        (h.access)(writer).writes.contains(loc),
        (h.access)(reader).reads.contains(loc),
        (run.succeeded)(reader),
        (h.happens_before)(writer, reader),
        forall|e: D::Event| #[trigger]
            writers_before(h, reader, loc).contains(e) && e != writer ==> ((h.happens_before)(
                e,
                writer,
            ) || read_fenced_before(h, reader, e, loc)),
    ensures
        (run.end)(writer).is_some(),
        value_at_start(run, reader, loc) == (run.memory)((run.end)(writer).unwrap())(loc),
{
    assert(!(run.rejected_read)(reader, loc));
    assert(finished_by(run, writer, (run.start)(reader)));
    let w_end = (run.end)(writer).unwrap();
    assert(w_end <= (run.start)(reader));

    race_free(h);
    assert forall|time: nat| w_end <= time < (run.start)(reader) implies !#[trigger] writer_active(
        h,
        run,
        loc,
        time,
    ) by {
        if writer_active(h, run, loc, time) {
            let other = choose|other: D::Event|
                h.events.contains(other) && (h.access)(other).writes.contains(loc)
                    && #[trigger] active(run, other, time);
            if other == writer {
                assert(time < w_end);
            } else {
                reachable_is_well_formed(h);
                write_conflicts_with_read((h.access)(other), (h.access)(reader), loc);
                assert(other != reader);
                assert(!(h.read_guards)(other).dom().contains(loc));
                if read_fenced_before(h, reader, other, loc) {
                    fenced_read_finishes_before_writer(h, run, reader, other, loc);
                    assert(match (run.end)(reader) {
                        Some(end) => (run.start)(reader) <= end,
                        None => true,
                    });
                } else if (h.happens_before)(reader, other) {
                    assert(finished_by(run, reader, (run.start)(other)));
                    assert(match (run.end)(reader) {
                        Some(end) => (run.start)(reader) <= end,
                        None => true,
                    });
                } else {
                    assert((h.happens_before)(other, reader));
                    assert(writers_before(h, reader, loc).contains(other));
                    assert((h.happens_before)(other, writer));
                    assert(finished_by(run, other, (run.start)(writer)));
                    assert(match (run.end)(writer) {
                        Some(end) => (run.start)(writer) <= end,
                        None => true,
                    });
                }
            }
        }
    }
    unchanged_without_active_writer(h, run, loc, w_end, (run.start)(reader));
}

} // verus!
