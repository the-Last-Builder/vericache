//! Lemmas connecting the token representation to `History` steps and frontier
//! coverage.
#[cfg(verus_only)]
use super::*;
#[cfg(verus_only)]
use crate::domain::Domain;
#[cfg(verus_only)]
use crate::history::*;
#[cfg(verus_only)]
use crate::order::*;
#[cfg(verus_only)]
use crate::space::*;
use vstd::prelude::*;

verus! {

broadcast use vstd::iset_lib::group_iset_properties;
// ---------------------------------------------------------------------------
// Lemmas: the tokens' history follows the `History` steps
// ---------------------------------------------------------------------------

pub(super) proof fn no_tokens_is_initial<D: Domain>()
    ensures
        history_of::<D>(IMap::empty(), ISet::empty(), IMap::empty()) == initial::<D>(),
{
    broadcast use vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas;

    let h = history_of::<D>(IMap::empty(), ISet::empty(), IMap::empty());
    let init = initial::<D>();
    assert(h.events =~= init.events);
    assert(h.access =~= init.access);
    assert(h.read_guards =~= init.read_guards);
    assert(h.effect =~= init.effect);
    assert(h.happens_before =~= init.happens_before);
    assert forall|observer: D::Observer| #[trigger]
        (h.known_complete)(observer) == (init.known_complete)(observer) by {
        assert((h.known_complete)(observer) =~= (init.known_complete)(observer));
    }
    assert(h.known_complete =~= init.known_complete);
}

/// When `infos` are `submitted` facts for every member of `set`,
/// `closure_from` computes `with_predecessors` in the tokens' history.
pub(super) proof fn closure_from_is_with_predecessors<D: Domain>(
    submitted: IMap<D::Event, EventInfo<D>>,
    knows: ISet<(D::Observer, D::Event)>,
    revoked: IMap<(D::Loc, D::Share), D::Event>,
    infos: IMap<D::Event, EventInfo<D>>,
    set: ISet<D::Event>,
)
    requires
        infos.submap_of(submitted),
        set.subset_of(infos.dom()),
    ensures
        closure_from(infos, set) == with_predecessors(
            history_of(submitted, knows, revoked).happens_before,
            set,
        ),
{
    let before = history_of(submitted, knows, revoked).happens_before;
    assert forall|x: D::Event|
        closure_from(infos, set).contains(x) == with_predecessors(before, set).contains(x) by {
        if exists|m: D::Event| set.contains(m) && #[trigger] infos[m].preds.contains(x) {
            let m = choose|m: D::Event| set.contains(m) && #[trigger] infos[m].preds.contains(x);
            assert(before(x, m));
        }
        if precedes_member(before, set, x) {
            let m = choose|m: D::Event| set.contains(m) && #[trigger] before(x, m);
            assert(infos[m].preds.contains(x));
        }
    }
    assert(closure_from(infos, set) =~= with_predecessors(before, set));
}

/// The known completions of `observer` grow by `newly_known` when the
/// matching `knows` tokens are added.
pub(super) proof fn adding_knows_is_add_knowledge<D: Domain>(
    submitted: IMap<D::Event, EventInfo<D>>,
    knows: ISet<(D::Observer, D::Event)>,
    revoked: IMap<(D::Loc, D::Share), D::Event>,
    observer: D::Observer,
    newly_known: ISet<D::Event>,
)
    ensures
        history_of(submitted, knows.union(knowledge_of(observer, newly_known)), revoked)
            == add_knowledge(history_of(submitted, knows, revoked), observer, newly_known),
{
    let h = history_of(submitted, knows, revoked);
    let next = history_of(submitted, knows.union(knowledge_of(observer, newly_known)), revoked);
    let expected = add_knowledge(h, observer, newly_known);
    assert forall|o: D::Observer| #[trigger]
        (next.known_complete)(o) == (expected.known_complete)(o) by {
        assert((next.known_complete)(o) =~= (expected.known_complete)(o));
    }
    assert(next.known_complete =~= expected.known_complete);
}

/// Adding the `submitted` token of `new_event` takes the `submit` step.
pub(super) proof fn adding_submitted_is_submit<D: Domain>(
    submitted: IMap<D::Event, EventInfo<D>>,
    knows: ISet<(D::Observer, D::Event)>,
    revoked: IMap<(D::Loc, D::Share), D::Event>,
    new_event: D::Event,
    access: Access<D>,
    effect: Effect<D>,
    read_guards: IMap<D::Loc, D::Share>,
    after: ISet<D::Event>,
    cited: ISet<D::Event>,
)
    requires
        !submitted.dom().contains(new_event),
    ensures
        history_of(
            submitted.insert(
                new_event,
                EventInfo {
                    access,
                    effect,
                    read_guards,
                    preds: ordered_before(history_of(submitted, knows, revoked), after, cited),
                },
            ),
            knows,
            revoked,
        ) == submit(
            history_of(submitted, knows, revoked),
            new_event,
            access,
            effect,
            read_guards,
            after,
            cited,
        ),
{
    broadcast use vstd::imap::group_imap_lemmas;

    let h = history_of(submitted, knows, revoked);
    let info = EventInfo { access, effect, read_guards, preds: ordered_before(h, after, cited) };
    let next = history_of(submitted.insert(new_event, info), knows, revoked);
    let expected = submit(h, new_event, access, effect, read_guards, after, cited);
    assert(next.events =~= expected.events);
    assert(next.access =~= expected.access);
    assert(next.effect =~= expected.effect);
    assert(next.read_guards =~= expected.read_guards);
    assert(next.happens_before =~= expected.happens_before);
}

// ---------------------------------------------------------------------------
// Lemmas: frontiers
// ---------------------------------------------------------------------------
/// Waiting for the last writer also orders each of its predecessors.
pub(super) proof fn waiting_orders_predecessors<D: Domain>(
    h: History<D>,
    event: D::Event,
    last_writer: Option<D::Event>,
    ordered: ISet<D::Event>,
)
    requires
        at_or_before(h, event, last_writer),
        ordered_or_absent(last_writer, ordered),
        includes_predecessors(h.happens_before, ordered),
    ensures
        ordered.contains(event),
{
}

/// A frontier with the same last writer and more readers still covers.
pub(super) proof fn covers_with_more_readers<D: Domain>(
    h: History<D>,
    loc: D::Loc,
    frontier: Frontier<D>,
    more: Frontier<D>,
)
    requires
        covers(h, loc, frontier),
        more.last_writer == frontier.last_writer,
        frontier.readers.subset_of(more.readers),
    ensures
        covers(h, loc, more),
{
}

} // verus!
