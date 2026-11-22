//! History: the ghost record of submitted events and of what is ordered
//! before what. Its steps are the only way to add events, and admission is
//! checked at `Submit`, so every reachable history is race-free.
use crate::domain::Domain;
use crate::order::*;
use crate::space::*;
use vstd::prelude::*;

verus! {

broadcast use vstd::iset_lib::group_iset_properties;

pub struct History<D: Domain> {
    /// Submitted events.
    pub events: ISet<D::Event>,
    /// Meaningful only for submitted events.
    pub access: spec_fn(D::Event) -> Access<D>,
    /// Meaningful only for submitted events.
    pub effect: spec_fn(D::Event) -> Effect<D>,
    /// The share authorizing each guarded read of an event.
    pub read_guards: spec_fn(D::Event) -> IMap<D::Loc, D::Share>,
    /// The event that fenced off each revoked share at a location.
    pub revoked: IMap<(D::Loc, D::Share), D::Event>,
    /// `happens_before(a, b)`: every effect of `a` is complete and visible
    /// before `b` begins.
    pub happens_before: Relation<D::Event>,
    /// Events each observer knows to have completed.
    pub known_complete: spec_fn(D::Observer) -> ISet<D::Event>,
}

/// What an event that was never submitted is taken to access.
pub open spec fn no_access<D: Domain>() -> Access<D> {
    Access { reads: ISet::empty(), writes: ISet::empty() }
}

pub open spec fn initial<D: Domain>() -> History<D> {
    History {
        events: ISet::empty(),
        access: |event: D::Event| no_access::<D>(),
        effect: |event: D::Event| |input: Snapshot<D>, output: Snapshot<D>| true,
        read_guards: |event: D::Event| IMap::empty(),
        revoked: IMap::empty(),
        happens_before: |a: D::Event, b: D::Event| false,
        known_complete: |observer: D::Observer| ISet::empty(),
    }
}

pub open spec fn read_fenced_before<D: Domain>(
    h: History<D>,
    reader: D::Event,
    writer: D::Event,
    loc: D::Loc,
) -> bool {
    let guards = (h.read_guards)(reader);
    &&& guards.dom().contains(loc)
    &&& h.revoked.dom().contains((loc, guards[loc]))
    &&& (h.happens_before)(h.revoked[(loc, guards[loc])], writer)
}

pub open spec fn conflict_resolved_at<D: Domain>(
    h: History<D>,
    a: D::Event,
    b: D::Event,
    loc: D::Loc,
) -> bool {
    ||| (h.happens_before)(a, b)
    ||| (h.happens_before)(b, a)
    ||| (h.access)(a).reads.contains(loc) && (h.access)(b).writes.contains(loc)
        && read_fenced_before(h, a, b, loc)
    ||| (h.access)(b).reads.contains(loc) && (h.access)(a).writes.contains(loc)
        && read_fenced_before(h, b, a, loc)
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------
/// Events ordered before a new event that `observer` submits with device
/// ordering `after` (queue order, device-side waits) and cited knowledge
/// `cited` (completions the submitter relies on): everything ordered before
/// `after` and `cited`.
pub open spec fn ordered_before<D: Domain>(
    h: History<D>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
) -> ISet<D::Event> {
    with_predecessors(h.happens_before, after.union(cited))
}

pub open spec fn conflict_admitted_at<D: Domain>(
    h: History<D>,
    old: D::Event,
    access: Access<D>,
    read_guards: IMap<D::Loc, D::Share>,
    loc: D::Loc,
    ordered: ISet<D::Event>,
) -> bool {
    let old_guards = (h.read_guards)(old);
    ||| ordered.contains(old)
    ||| (h.access)(old).reads.contains(loc) && access.writes.contains(loc)
        && old_guards.dom().contains(loc) && h.revoked.dom().contains((loc, old_guards[loc]))
        && ordered.contains(h.revoked[(loc, old_guards[loc])])
    ||| (h.access)(old).writes.contains(loc) && access.reads.contains(loc)
        && read_guards.dom().contains(loc) && h.revoked.dom().contains((loc, read_guards[loc])) && (
    h.happens_before)(h.revoked[(loc, read_guards[loc])], old)
}

pub open spec fn can_submit<D: Domain>(
    h: History<D>,
    observer: D::Observer,
    new_event: D::Event,
    access: Access<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
) -> bool {
    let ordered = ordered_before(h, after, cited);
    &&& !h.events.contains(new_event)
    &&& after.subset_of(h.events)
    &&& cited.subset_of((h.known_complete)(observer))
    &&& read_guards.dom().subset_of(access.reads)
    &&& read_guards.dom().disjoint(access.writes)
    &&& forall|old: D::Event, loc: D::Loc|
        h.events.contains(old) && #[trigger] conflict_at((h.access)(old), access, loc)
            ==> conflict_admitted_at(h, old, access, read_guards, loc, ordered)
}

/// `observer` is not needed here; it matters only to `can_submit`.
pub open spec fn submit<D: Domain>(
    h: History<D>,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
) -> History<D> {
    let before_new = ordered_before(h, after, cited);
    History {
        events: h.events.insert(new_event),
        access: |event: D::Event|
            if event == new_event {
                access
            } else {
                (h.access)(event)
            },
        effect: |event: D::Event|
            if event == new_event {
                effect
            } else {
                (h.effect)(event)
            },
        happens_before: |a: D::Event, b: D::Event|
            (h.happens_before)(a, b) || (b == new_event && before_new.contains(a)),
        read_guards: |event: D::Event|
            if event == new_event {
                read_guards
            } else {
                (h.read_guards)(event)
            },
        ..h
    }
}

pub open spec fn can_revoke_read<D: Domain>(
    h: History<D>,
    fence_event: D::Event,
    share: D::Share,
    locs: ISet<D::Loc>,
) -> bool {
    &&& !h.events.contains(fence_event)
    &&& forall|loc: D::Loc|
        locs.contains(loc) ==> !#[trigger] h.revoked.dom().contains((loc, share))
}

pub open spec fn revoked_entries<D: Domain>(
    locs: ISet<D::Loc>,
    share: D::Share,
    fence_event: D::Event,
) -> IMap<(D::Loc, D::Share), D::Event> {
    IMap::new(|key: (D::Loc, D::Share)| locs.contains(key.0) && key.1 == share, |_| fence_event)
}

pub open spec fn revoke_read<D: Domain>(
    h: History<D>,
    fence_event: D::Event,
    share: D::Share,
    locs: ISet<D::Loc>,
) -> History<D> {
    History {
        events: h.events.insert(fence_event),
        access: |event: D::Event|
            if event == fence_event {
                no_access()
            } else {
                (h.access)(event)
            },
        effect: |event: D::Event|
            if event == fence_event {
                |_, _| true
            } else {
                (h.effect)(event)
            },
        read_guards: |event: D::Event|
            if event == fence_event {
                IMap::empty()
            } else {
                (h.read_guards)(event)
            },
        revoked: h.revoked.union_prefer_right(revoked_entries::<D>(locs, share, fence_event)),
        ..h
    }
}

/// `observer` additionally knows `newly_known` to have completed.
pub open spec fn add_knowledge<D: Domain>(
    h: History<D>,
    observer: D::Observer,
    newly_known: ISet<D::Event>,
) -> History<D> {
    History {
        known_complete: |o: D::Observer|
            if o == observer {
                (h.known_complete)(o).union(newly_known)
            } else {
                (h.known_complete)(o)
            },
        ..h
    }
}

/// `observer` sees `completed` finish. Anything ordered before a completed
/// event has itself completed, so its predecessors become known too.
pub open spec fn observe<D: Domain>(
    h: History<D>,
    observer: D::Observer,
    completed: ISet<D::Event>,
) -> History<D> {
    add_knowledge(h, observer, with_predecessors(h.happens_before, completed))
}

/// `receiver` is told by `sender` that `facts` have completed.
pub open spec fn learn<D: Domain>(
    h: History<D>,
    receiver: D::Observer,
    facts: ISet<D::Event>,
) -> History<D> {
    add_knowledge(h, receiver, with_predecessors(h.happens_before, facts))
}

pub enum Action<D: Domain> {
    Submit {
        observer: D::Observer,
        new_event: D::Event,
        access: Access<D>,
        effect: Effect<D>,
        read_guards: IMap<D::Loc, D::Share>,
        after: ISet<D::Event>,
        cited: ISet<D::Event>,
    },
    RevokeRead { fence_event: D::Event, share: D::Share, locs: ISet<D::Loc> },
    Observe { observer: D::Observer, completed: ISet<D::Event> },
    Learn { receiver: D::Observer, sender: D::Observer, facts: ISet<D::Event> },
    /// A backend step that touches no memory and changes no knowledge.
    Internal,
}

/// `next` is the result of taking `action` in `h`, and `action` is allowed.
pub open spec fn history_step<D: Domain>(
    h: History<D>,
    action: Action<D>,
    next: History<D>,
) -> bool {
    match action {
        Action::Submit { observer, new_event, access, effect, read_guards, after, cited } => {
            &&& can_submit(h, observer, new_event, access, read_guards, after, cited)
            &&& next == submit(h, new_event, access, effect, read_guards, after, cited)
        },
        Action::RevokeRead { fence_event, share, locs } => {
            &&& can_revoke_read(h, fence_event, share, locs)
            &&& next == revoke_read(h, fence_event, share, locs)
        },
        Action::Observe { observer, completed } => {
            &&& completed.subset_of(h.events)
            &&& next == observe(h, observer, completed)
        },
        Action::Learn { receiver, sender, facts } => {
            // A sender can only pass on what it knows.
            &&& facts.subset_of((h.known_complete)(sender))
            &&& next == learn(h, receiver, facts)
        },
        Action::Internal => next == h,
    }
}

// ---------------------------------------------------------------------------
// Reachability
// ---------------------------------------------------------------------------
/// Step `i` of `trace` is a history step under some action.
pub open spec fn step_at<D: Domain>(trace: Seq<History<D>>, i: int) -> bool {
    exists|action: Action<D>| #[trigger] history_step(trace[i], action, trace[i + 1])
}

/// `trace` starts at `initial()` and each history follows from the previous one.
pub open spec fn is_trace<D: Domain>(trace: Seq<History<D>>) -> bool {
    &&& trace.len() > 0
    &&& trace[0] == initial::<D>()
    &&& forall|i: int| 0 <= i < trace.len() - 1 ==> #[trigger] step_at(trace, i)
}

pub open spec fn reachable<D: Domain>(h: History<D>) -> bool {
    exists|trace: Seq<History<D>>| is_trace(trace) && trace.last() == h
}

// ---------------------------------------------------------------------------
// Invariant
// ---------------------------------------------------------------------------
/// What `observer` knows has been submitted, and is closed under predecessors.
pub open spec fn knowledge_well_formed<D: Domain>(h: History<D>, observer: D::Observer) -> bool {
    &&& (h.known_complete)(observer).subset_of(h.events)
    &&& includes_predecessors(h.happens_before, (h.known_complete)(observer))
}

pub open spec fn well_formed<D: Domain>(h: History<D>) -> bool {
    &&& h.events.finite()
    &&& is_strict_partial_order(h.happens_before, h.events)
    &&& forall|key: (D::Loc, D::Share)| #[trigger]
        h.revoked.dom().contains(key) ==> h.events.contains(h.revoked[key])
    &&& forall|event: D::Event|
        h.events.contains(event) ==> {
            let guards = #[trigger] (h.read_guards)(event);
            &&& guards.dom().subset_of((h.access)(event).reads)
            &&& guards.dom().disjoint((h.access)(event).writes)
        }
    &&& forall|observer: D::Observer| #[trigger] knowledge_well_formed(h, observer)
}

/// Each conflict is ordered directly or resolved by a read fence.
pub open spec fn conflicts_are_ordered<D: Domain>(h: History<D>) -> bool {
    forall|a: D::Event, b: D::Event, loc: D::Loc|
        h.events.contains(a) && h.events.contains(b) && a != b && #[trigger] conflict_at(
            (h.access)(a),
            (h.access)(b),
            loc,
        ) ==> conflict_resolved_at(h, a, b, loc)
}

pub open spec fn history_invariant<D: Domain>(h: History<D>) -> bool {
    well_formed(h) && conflicts_are_ordered(h)
}

pub proof fn initial_satisfies_invariant<D: Domain>()
    ensures
        history_invariant(initial::<D>()),
{
    let h = initial::<D>();
    assert(h.events.finite()) by {
        broadcast use vstd::iset::lemma_iset_empty_finite;

    }
    assert forall|observer: D::Observer| #[trigger] knowledge_well_formed(h, observer) by {}
}

/// The predecessors of submitted events are submitted, and the result
/// includes its own predecessors. Used for `ordered_before` and for what
/// `observe` and `learn` add to knowledge.
proof fn closure_of_submitted<D: Domain>(h: History<D>, set: ISet<D::Event>)
    requires
        well_formed(h),
        set.subset_of(h.events),
    ensures
        with_predecessors(h.happens_before, set).subset_of(h.events),
        includes_predecessors(h.happens_before, with_predecessors(h.happens_before, set)),
{
    with_predecessors_stays_within(h.happens_before, set, h.events);
    with_predecessors_is_closed(h.happens_before, set);
}

proof fn submit_preserves_invariant<D: Domain>(
    h: History<D>,
    observer: D::Observer,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
)
    requires
        history_invariant(h),
        can_submit(h, observer, new_event, access, read_guards, after, cited),
    ensures
        history_invariant(submit(h, new_event, access, effect, read_guards, after, cited)),
{
    let before_new = ordered_before(h, after, cited);
    let next = submit(h, new_event, access, effect, read_guards, after, cited);
    // Cited events are known, hence submitted.
    assert(knowledge_well_formed(h, observer));
    closure_of_submitted(h, after.union(cited));
    vstd::iset_lib::lemma_iset_insert_finite_iff(h.events, new_event);

    // The new event is not yet submitted, so it has no old edges and is not
    // in `before_new`. Hence the only new edges point into it.
    assert(!before_new.contains(new_event));

    assert(is_strict_partial_order(next.happens_before, next.events)) by {
        assert forall|a: D::Event, b: D::Event, c: D::Event| #[trigger]
            (next.happens_before)(a, b) && #[trigger] (next.happens_before)(b, c) implies (
        next.happens_before)(a, c) by {
            // `b` has an outgoing edge, so `b` is old and `a -> b` is an old edge.
            assert((h.happens_before)(a, b));
            if c == new_event && before_new.contains(b) {
                // `before_new` includes predecessors.
                assert(before_new.contains(a));
            }
        }
    }

    assert forall|o: D::Observer| #[trigger] knowledge_well_formed(next, o) by {
        assert(knowledge_well_formed(h, o));
        // Known events are old, and new edges only point to the new event.
        assert forall|member: D::Event, x: D::Event|
            (next.known_complete)(o).contains(member) && #[trigger] (next.happens_before)(
                x,
                member,
            ) implies (next.known_complete)(o).contains(x) by {
            assert((h.happens_before)(x, member));
        }
    }

    assert forall|a: D::Event, b: D::Event, loc: D::Loc|
        next.events.contains(a) && next.events.contains(b) && a != b && #[trigger] conflict_at(
            (next.access)(a),
            (next.access)(b),
            loc,
        ) implies conflict_resolved_at(next, a, b, loc) by {
        if a == new_event {
            assert(conflict_at((h.access)(b), access, loc));
        } else if b == new_event {
            assert(conflict_at((h.access)(a), access, loc));
        } else {
            assert(conflict_at((h.access)(a), (h.access)(b), loc));
        }
    }
}

/// Adding knowledge preserves the invariant when the added events are
/// submitted and closed under predecessors.
proof fn add_knowledge_preserves_invariant<D: Domain>(
    h: History<D>,
    observer: D::Observer,
    newly_known: ISet<D::Event>,
)
    requires
        history_invariant(h),
        newly_known.subset_of(h.events),
        includes_predecessors(h.happens_before, newly_known),
    ensures
        history_invariant(add_knowledge(h, observer, newly_known)),
{
    let next = add_knowledge(h, observer, newly_known);
    assert forall|o: D::Observer| #[trigger] knowledge_well_formed(next, o) by {
        assert(knowledge_well_formed(h, o));
        if o == observer {
            union_keeps_predecessors(h.happens_before, (h.known_complete)(o), newly_known);
        }
    }
}

pub proof fn step_preserves_invariant<D: Domain>(h: History<D>, action: Action<D>, next: History<D>)
    requires
        history_invariant(h),
        history_step(h, action, next),
    ensures
        history_invariant(next),
{
    match action {
        Action::Submit { observer, new_event, access, effect, read_guards, after, cited } => {
            submit_preserves_invariant(
                h,
                observer,
                new_event,
                access,
                effect,
                read_guards,
                after,
                cited,
            );
        },
        Action::RevokeRead { fence_event, share, locs } => {
            vstd::iset_lib::lemma_iset_insert_finite_iff(h.events, fence_event);
            assert forall|observer: D::Observer| #[trigger]
                knowledge_well_formed(next, observer) by {
                assert(knowledge_well_formed(h, observer));
            }
            assert forall|key: (D::Loc, D::Share)| #[trigger]
                h.revoked.dom().contains(key) implies next.revoked.dom().contains(key)
                && next.revoked[key] == h.revoked[key] by {
                assert(!revoked_entries::<D>(locs, share, fence_event).dom().contains(key));
            }
            assert forall|a: D::Event, b: D::Event, loc: D::Loc|
                next.events.contains(a) && next.events.contains(b) && a != b
                    && #[trigger] conflict_at(
                    (next.access)(a),
                    (next.access)(b),
                    loc,
                ) implies conflict_resolved_at(next, a, b, loc) by {
                assert(conflict_at((h.access)(a), (h.access)(b), loc));
            }
        },
        Action::Observe { observer, completed } => {
            closure_of_submitted(h, completed);
            add_knowledge_preserves_invariant(
                h,
                observer,
                with_predecessors(h.happens_before, completed),
            );
        },
        Action::Learn { receiver, sender, facts } => {
            // The sender's knowledge consists of submitted events.
            assert(knowledge_well_formed(h, sender));
            closure_of_submitted(h, facts);
            add_knowledge_preserves_invariant(
                h,
                receiver,
                with_predecessors(h.happens_before, facts),
            );
        },
        Action::Internal => {},
    }
}

proof fn trace_satisfies_invariant<D: Domain>(trace: Seq<History<D>>, i: int)
    requires
        is_trace(trace),
        0 <= i < trace.len(),
    ensures
        history_invariant(trace[i]),
    decreases i,
{
    if i == 0 {
        initial_satisfies_invariant::<D>();
    } else {
        trace_satisfies_invariant(trace, i - 1);
        assert(step_at(trace, i - 1));
        let action = choose|action: Action<D>| #[trigger]
            history_step(trace[i - 1], action, trace[i]);
        step_preserves_invariant(trace[i - 1], action, trace[i]);
    }
}

pub proof fn reachable_satisfies_invariant<D: Domain>(h: History<D>)
    requires
        reachable(h),
    ensures
        history_invariant(h),
{
    let trace = choose|trace: Seq<History<D>>| is_trace(trace) && trace.last() == h;
    trace_satisfies_invariant(trace, trace.len() - 1);
}

/// One more step from a reachable history reaches another.
pub proof fn step_keeps_reachable<D: Domain>(h: History<D>, action: Action<D>, next: History<D>)
    requires
        reachable(h),
        history_step(h, action, next),
    ensures
        reachable(next),
{
    let trace = choose|trace: Seq<History<D>>| is_trace(trace) && trace.last() == h;
    let longer = trace.push(next);
    assert forall|i: int| 0 <= i < longer.len() - 1 implies #[trigger] step_at(longer, i) by {
        if i < trace.len() - 1 {
            assert(step_at(trace, i));
            let a = choose|a: Action<D>| #[trigger] history_step(trace[i], a, trace[i + 1]);
            assert(history_step(longer[i], a, longer[i + 1]));
        } else {
            assert(history_step(longer[i], action, longer[i + 1]));
        }
    }
    assert(longer.last() == next);
}

pub proof fn initial_is_reachable<D: Domain>()
    ensures
        reachable(initial::<D>()),
{
    let trace = seq![initial::<D>()];
    assert(is_trace(trace));
    assert(trace.last() == initial::<D>());
}

pub proof fn reachable_is_well_formed<D: Domain>(h: History<D>)
    requires
        reachable(h),
    ensures
        well_formed(h),
{
    reachable_satisfies_invariant(h);
}

/// In every reachable history, each conflicting location is ordered or fenced.
pub proof fn race_free<D: Domain>(h: History<D>)
    requires
        reachable(h),
    ensures
        conflicts_are_ordered(h),
{
    reachable_satisfies_invariant(h);
}

} // verus!
