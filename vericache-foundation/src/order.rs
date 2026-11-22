//! Order: strict partial orders on events and predecessor-closed sets.
use vstd::prelude::*;

verus! {

/// A binary relation on events, read as `relation(earlier, later)`.
pub type Relation<Event> = spec_fn(Event, Event) -> bool;

pub open spec fn is_transitive<Event>(before: Relation<Event>) -> bool {
    forall|a: Event, b: Event, c: Event| #[trigger]
        before(a, b) && #[trigger] before(b, c) ==> before(a, c)
}

/// `before` is a strict partial order whose related events all lie in `events`.
pub open spec fn is_strict_partial_order<Event>(
    before: Relation<Event>,
    events: ISet<Event>,
) -> bool {
    &&& forall|a: Event, b: Event| #[trigger]
        before(a, b) ==> events.contains(a) && events.contains(b)
    &&& forall|a: Event| !#[trigger] before(a, a)
    &&& is_transitive(before)
}

/// `x` is ordered before some member of `set`.
pub open spec fn precedes_member<Event>(
    before: Relation<Event>,
    set: ISet<Event>,
    x: Event,
) -> bool {
    exists|member: Event| set.contains(member) && #[trigger] before(x, member)
}

/// `set` plus everything ordered before some member of it.
pub open spec fn with_predecessors<Event>(before: Relation<Event>, set: ISet<Event>) -> ISet<
    Event,
> {
    ISet::new(|x: Event| set.contains(x) || precedes_member(before, set, x))
}

/// Every predecessor of a member is also a member.
pub open spec fn includes_predecessors<Event>(before: Relation<Event>, set: ISet<Event>) -> bool {
    forall|member: Event, x: Event|
        set.contains(member) && #[trigger] before(x, member) ==> set.contains(x)
}

pub proof fn with_predecessors_is_closed<Event>(before: Relation<Event>, set: ISet<Event>)
    requires
        is_transitive(before),
    ensures
        includes_predecessors(before, with_predecessors(before, set)),
{
    let closed = with_predecessors(before, set);
    assert forall|member: Event, x: Event|
        closed.contains(member) && #[trigger] before(x, member) implies closed.contains(x) by {
        if !set.contains(member) {
            // `member` precedes some `y` in `set`, so `x` does too.
            let y = choose|y: Event| set.contains(y) && before(member, y);
            assert(before(x, y));
        }
        assert(precedes_member(before, set, x));
    }
}

pub proof fn with_predecessors_contains_set<Event>(before: Relation<Event>, set: ISet<Event>)
    ensures
        set.subset_of(with_predecessors(before, set)),
{
}

/// If every member of `set` is in `domain` and `before` relates only members
/// of `domain`, then `with_predecessors(before, set)` stays in `domain`.
pub proof fn with_predecessors_stays_within<Event>(
    before: Relation<Event>,
    set: ISet<Event>,
    domain: ISet<Event>,
)
    requires
        set.subset_of(domain),
        forall|a: Event, b: Event| #[trigger] before(a, b) ==> domain.contains(a),
    ensures
        with_predecessors(before, set).subset_of(domain),
{
}

pub proof fn union_keeps_predecessors<Event>(
    before: Relation<Event>,
    a: ISet<Event>,
    b: ISet<Event>,
)
    requires
        includes_predecessors(before, a),
        includes_predecessors(before, b),
    ensures
        includes_predecessors(before, a.union(b)),
{
}

/// Any two distinct members are ordered one way or the other.
pub open spec fn is_chain<Event>(before: Relation<Event>, chain: ISet<Event>) -> bool {
    forall|a: Event, b: Event|
        chain.contains(a) && chain.contains(b) && a != b ==> #[trigger] before(a, b)
            || #[trigger] before(b, a)
}

/// Every other member of `chain` is ordered before `last`.
pub open spec fn is_last<Event>(before: Relation<Event>, chain: ISet<Event>, last: Event) -> bool {
    &&& chain.contains(last)
    &&& forall|other: Event|
        chain.contains(other) && other != last ==> #[trigger] before(other, last)
}

/// A finite, nonempty chain of a transitive relation has a last member.
pub proof fn finite_chain_has_last<Event>(before: Relation<Event>, chain: ISet<Event>)
    requires
        is_transitive(before),
        chain.finite(),
        chain != ISet::<Event>::empty(),
        is_chain(before, chain),
    ensures
        exists|last: Event| is_last(before, chain, last),
    decreases chain.len(),
{
    broadcast use vstd::iset_lib::group_iset_properties, vstd::iset::group_iset_lemmas;

    let first = chain.choose();
    let rest = chain.remove(first);
    if rest =~= ISet::<Event>::empty() {
        assert(is_last(before, chain, first));
    } else {
        vstd::iset::lemma_iset_remove_len(chain, first);
        vstd::iset_lib::lemma_iset_remove_finite_iff(chain, first);
        finite_chain_has_last(before, rest);
        let last_of_rest = choose|last: Event| is_last(before, rest, last);
        if before(first, last_of_rest) {
            assert(is_last(before, chain, last_of_rest));
        } else {
            // `first` comes after `last_of_rest`, hence after all of `rest`.
            assert(rest.contains(last_of_rest));
            assert(before(first, last_of_rest) || before(last_of_rest, first));
            assert forall|other: Event|
                chain.contains(other) && other != first implies #[trigger] before(other, first) by {
                if other != last_of_rest {
                    assert(rest.contains(other));
                    assert(before(other, last_of_rest));
                }
            }
            assert(is_last(before, chain, first));
        }
    }
}

} // verus!
