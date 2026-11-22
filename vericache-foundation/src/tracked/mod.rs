//! Tracked: the history as PCM tokens, so that executable code on many
//! threads and nodes can submit events without a lock around one shared
//! history.
//!
//! `HistoryTokens` is a VerusSync state machine. The history is not a field;
//! `history_of` reads it off the `submitted`, `knows` and `revoked` tokens. Every
//! transition that changes those tokens takes exactly one `history_step`, so
//! the history stays `reachable` and the theorems about reachable histories
//! apply to it.
//!
//! Each location has one `region` token, which is `Owned` by one holder or
//! `Lent` out as read shares. The token's `Frontier` names the events a new
//! access must wait for, so `submit` checks admission against the tokens
//! passed in and never against the whole history. Its proof cost does not
//! grow with the pool.
mod examples;
mod lemmas;
mod machine;
mod submit_lemmas;
mod transitions;

pub use machine::HistoryTokens;
pub use transitions::HistoryTokenTransitions;

use crate::domain::Domain;
#[cfg(verus_only)]
use crate::history::*;
use crate::space::*;
use vstd::prelude::*;

verus! {

broadcast use vstd::iset_lib::group_iset_properties;
// ---------------------------------------------------------------------------
// Token values
// ---------------------------------------------------------------------------

/// What a `submitted` token records about one event.
pub struct EventInfo<D: Domain> {
    pub access: Access<D>,
    pub effect: Effect<D>,
    pub read_guards: IMap<D::Loc, D::Share>,
    /// Exactly the events that happen before this one.
    pub preds: ISet<D::Event>,
}

/// The writer waits for the fence recorded for this reader's share.
pub open spec fn writer_waits_for_fence<D: Domain>(
    reader: EventInfo<D>,
    writer: EventInfo<D>,
    loc: D::Loc,
    fences: IMap<(D::Loc, D::Share), D::Event>,
) -> bool {
    let key = (loc, reader.read_guards[loc]);
    &&& reader.read_guards.dom().contains(loc)
    &&& writer.access.writes.contains(loc)
    &&& fences.dom().contains(key)
    &&& writer.preds.contains(fences[key])
}

/// For one location, the events a new access there must wait for.
pub struct Frontier<D: Domain> {
    /// The last submitted event that writes the location.
    pub last_writer: Option<D::Event>,
    /// Events submitted after `last_writer` that read the location without
    /// writing it, plus fences revoking read shares of the location.
    pub readers: ISet<D::Event>,
}

/// The value of the `region` token of one location.
pub enum Slot<D: Domain> {
    /// The holder alone may submit events that touch the location.
    Owned(Frontier<D>),
    /// The location is lent out as the read shares `shares`. Readers that
    /// already returned their share are folded into `base`.
    Lent { base: Frontier<D>, shares: ISet<D::Share> },
}

pub open spec fn untouched<D: Domain>() -> Frontier<D> {
    Frontier { last_writer: None, readers: ISet::empty() }
}

/// The `(observer, event)` pairs that say `observer` knows `events` completed.
pub open spec fn knowledge_of<Observer, Event>(observer: Observer, events: ISet<Event>) -> ISet<
    (Observer, Event),
> {
    ISet::new(|pair: (Observer, Event)| pair.0 == observer && events.contains(pair.1))
}

/// `set` and everything the `infos` of its members record as earlier. When
/// `infos` are `submitted` tokens covering `set`, this is
/// `with_predecessors(h.happens_before, set)` (`closure_from_is_with_predecessors`).
pub open spec fn closure_from<D: Domain>(
    infos: IMap<D::Event, EventInfo<D>>,
    set: ISet<D::Event>,
) -> ISet<D::Event> {
    ISet::new(
        |x: D::Event|
            set.contains(x) || exists|member: D::Event|
                set.contains(member) && #[trigger] infos[member].preds.contains(x),
    )
}

// ---------------------------------------------------------------------------
// The frontier invariant
// ---------------------------------------------------------------------------
/// `event` is `last_writer` or happens before it.
pub open spec fn at_or_before<D: Domain>(
    h: History<D>,
    event: D::Event,
    last_writer: Option<D::Event>,
) -> bool {
    match last_writer {
        Some(writer) => event == writer || (h.happens_before)(event, writer),
        None => false,
    }
}

pub open spec fn frontier_contains<D: Domain>(
    h: History<D>,
    frontier: Frontier<D>,
    event: D::Event,
) -> bool {
    at_or_before(h, event, frontier.last_writer) || frontier.readers.contains(event)
}

pub open spec fn reader_is_covered<D: Domain>(
    h: History<D>,
    loc: D::Loc,
    frontier: Frontier<D>,
    event: D::Event,
) -> bool {
    let guards = (h.read_guards)(event);
    ||| frontier_contains(h, frontier, event)
    ||| guards.dom().contains(loc) && h.revoked.dom().contains((loc, guards[loc]))
        && frontier_contains(h, frontier, h.revoked[(loc, guards[loc])])
}

/// Every submitted event that touches `loc` is accounted for by `frontier`:
/// writers are at or before `last_writer`, and readers are too or are listed
/// in `readers`, or their revocation fence is covered.
pub open spec fn covers<D: Domain>(h: History<D>, loc: D::Loc, frontier: Frontier<D>) -> bool {
    &&& forall|event: D::Event|
        h.events.contains(event) && #[trigger] (h.access)(event).writes.contains(loc)
            ==> at_or_before(h, event, frontier.last_writer)
    &&& forall|event: D::Event|
        h.events.contains(event) && #[trigger] (h.access)(event).reads.contains(loc)
            ==> reader_is_covered(h, loc, frontier, event)
}

/// The readers recorded in the read shares `ids` of `loc`.
pub open spec fn share_readers<D: Domain>(
    read_share: IMap<(D::Loc, D::Share), Frontier<D>>,
    loc: D::Loc,
    ids: ISet<D::Share>,
) -> ISet<D::Event> {
    ISet::new(
        |event: D::Event|
            exists|id: D::Share|
                ids.contains(id) && #[trigger] read_share[(loc, id)].readers.contains(event),
    )
}

/// The whole frontier of `loc`: the slot's own, plus the readers recorded in
/// shares that are still out.
pub open spec fn frontier_of<D: Domain>(
    slot: Slot<D>,
    read_share: IMap<(D::Loc, D::Share), Frontier<D>>,
    loc: D::Loc,
) -> Frontier<D> {
    match slot {
        Slot::Owned(frontier) => frontier,
        Slot::Lent { base, shares } => Frontier {
            last_writer: base.last_writer,
            readers: base.readers.union(share_readers(read_share, loc, shares)),
        },
    }
}

// ---------------------------------------------------------------------------
// The history the tokens describe
// ---------------------------------------------------------------------------
/// The history recorded by all `submitted`, `knows` and `revoked` tokens together.
pub open spec fn history_of<D: Domain>(
    submitted: IMap<D::Event, EventInfo<D>>,
    knows: ISet<(D::Observer, D::Event)>,
    revoked: IMap<(D::Loc, D::Share), D::Event>,
) -> History<D> {
    History {
        events: submitted.dom(),
        access: |event: D::Event|
            if submitted.dom().contains(event) {
                submitted[event].access
            } else {
                no_access::<D>()
            },
        effect: |event: D::Event|
            if submitted.dom().contains(event) {
                submitted[event].effect
            } else {
                (initial::<D>().effect)(event)
            },
        read_guards: |event: D::Event|
            if submitted.dom().contains(event) {
                submitted[event].read_guards
            } else {
                IMap::empty()
            },
        revoked,
        happens_before: |a: D::Event, b: D::Event|
            submitted.dom().contains(b) && submitted[b].preds.contains(a),
        known_complete: |observer: D::Observer|
            ISet::new(|event: D::Event| knows.contains((observer, event))),
    }
}

// ---------------------------------------------------------------------------
// Admission and frontier updates
// ---------------------------------------------------------------------------
pub open spec fn ordered_or_absent<Event>(
    last_writer: Option<Event>,
    ordered: ISet<Event>,
) -> bool {
    match last_writer {
        Some(writer) => ordered.contains(writer),
        None => true,
    }
}

/// An access to `loc` may follow `frontier` if every event it must wait for
/// is in `ordered`. A writer waits for the last writer and all later readers;
/// a reader waits only for the last writer.
pub open spec fn admitted<D: Domain>(
    frontier: Frontier<D>,
    loc: D::Loc,
    access: Access<D>,
    ordered: ISet<D::Event>,
) -> bool {
    if access.writes.contains(loc) {
        ordered_or_absent(frontier.last_writer, ordered) && frontier.readers.subset_of(ordered)
    } else if access.reads.contains(loc) {
        ordered_or_absent(frontier.last_writer, ordered)
    } else {
        true
    }
}

/// The frontier of `loc` once `new_event` with `access` is submitted.
pub open spec fn advance<D: Domain>(
    frontier: Frontier<D>,
    loc: D::Loc,
    new_event: D::Event,
    access: Access<D>,
) -> Frontier<D> {
    if access.writes.contains(loc) {
        Frontier { last_writer: Some(new_event), readers: ISet::empty() }
    } else if access.reads.contains(loc) {
        Frontier { last_writer: frontier.last_writer, readers: frontier.readers.insert(new_event) }
    } else {
        frontier
    }
}

pub open spec fn owned_slots<D: Domain>(frontiers: IMap<D::Loc, Frontier<D>>) -> IMap<
    D::Loc,
    Slot<D>,
> {
    IMap::new(
        |loc: D::Loc| frontiers.dom().contains(loc),
        |loc: D::Loc| Slot::Owned(frontiers[loc]),
    )
}

pub open spec fn advance_owned<D: Domain>(
    frontiers: IMap<D::Loc, Frontier<D>>,
    new_event: D::Event,
    access: Access<D>,
) -> IMap<D::Loc, Frontier<D>> {
    IMap::new(
        |loc: D::Loc| frontiers.dom().contains(loc),
        |loc: D::Loc| advance(frontiers[loc], loc, new_event, access),
    )
}

pub open spec fn advance_shares<D: Domain>(
    shares: IMap<(D::Loc, D::Share), Frontier<D>>,
    new_event: D::Event,
    read_guards: IMap<D::Loc, D::Share>,
) -> IMap<(D::Loc, D::Share), Frontier<D>> {
    IMap::new(
        |key: (D::Loc, D::Share)| shares.dom().contains(key),
        |key: (D::Loc, D::Share)|
            if read_guards.dom().contains(key.0) && read_guards[key.0] == key.1 {
                Frontier { readers: shares[key].readers.insert(new_event), ..shares[key] }
            } else {
                shares[key]
            },
    )
}

// ---------------------------------------------------------------------------
// Lending and returning read shares
// ---------------------------------------------------------------------------
/// The read shares `ids` of every location in `frontiers`. A share starts
/// with the location's last writer and no readers of its own.
pub open spec fn new_shares<D: Domain>(
    frontiers: IMap<D::Loc, Frontier<D>>,
    ids: ISet<D::Share>,
) -> IMap<(D::Loc, D::Share), Frontier<D>> {
    IMap::new(
        |key: (D::Loc, D::Share)| frontiers.dom().contains(key.0) && ids.contains(key.1),
        |key: (D::Loc, D::Share)|
            Frontier { last_writer: frontiers[key.0].last_writer, readers: ISet::empty() },
    )
}

pub open spec fn lent_slots<D: Domain>(
    frontiers: IMap<D::Loc, Frontier<D>>,
    ids: ISet<D::Share>,
) -> IMap<D::Loc, Slot<D>> {
    IMap::new(
        |loc: D::Loc| frontiers.dom().contains(loc),
        |loc: D::Loc| Slot::Lent { base: frontiers[loc], shares: ids },
    )
}

pub open spec fn lent_bases<D: Domain>(lent: IMap<D::Loc, Slot<D>>) -> IMap<D::Loc, Frontier<D>> {
    lent.dom().mk_map(|loc| lent[loc]->Lent_base)
}

pub open spec fn add_lent_shares<D: Domain>(
    lent: IMap<D::Loc, Slot<D>>,
    ids: ISet<D::Share>,
) -> IMap<D::Loc, Slot<D>> {
    lent.dom().mk_map(
        |loc| Slot::Lent { base: lent[loc]->Lent_base, shares: lent[loc]->Lent_shares.union(ids) },
    )
}

/// The keys of share `id` for every location in `locs`.
pub open spec fn share_keys<D: Domain>(locs: ISet<D::Loc>, id: D::Share) -> ISet<
    (D::Loc, D::Share),
> {
    ISet::new(|key: (D::Loc, D::Share)| locs.contains(key.0) && key.1 == id)
}

/// Folds the readers of a returned share into the slot. When it was the last
/// share out, the slot is owned again.
pub open spec fn fold_share<D: Domain>(slot: Slot<D>, id: D::Share, returned: Frontier<D>) -> Slot<
    D,
> {
    match slot {
        Slot::Lent { base, shares } => {
            let base = Frontier {
                last_writer: base.last_writer,
                readers: base.readers.union(returned.readers),
            };
            if shares.remove(id) == ISet::<D::Share>::empty() {
                Slot::Owned(base)
            } else {
                Slot::Lent { base, shares: shares.remove(id) }
            }
        },
        Slot::Owned(frontier) => slot,
    }
}

pub open spec fn fold_shares<D: Domain>(
    lent: IMap<D::Loc, Slot<D>>,
    id: D::Share,
    returned: IMap<(D::Loc, D::Share), Frontier<D>>,
) -> IMap<D::Loc, Slot<D>> {
    IMap::new(
        |loc: D::Loc| lent.dom().contains(loc),
        |loc: D::Loc| fold_share(lent[loc], id, returned[(loc, id)]),
    )
}

pub open spec fn revoke_slots<D: Domain>(
    lent: IMap<D::Loc, Slot<D>>,
    id: D::Share,
    fence_event: D::Event,
) -> IMap<D::Loc, Slot<D>> {
    IMap::new(
        |loc: D::Loc| lent.dom().contains(loc),
        |loc: D::Loc|
            fold_share(
                lent[loc],
                id,
                Frontier { last_writer: None, readers: ISet::empty().insert(fence_event) },
            ),
    )
}

pub open spec fn is_lent_with<D: Domain>(slot: Slot<D>, id: D::Share) -> bool {
    match slot {
        Slot::Lent { base, shares } => shares.contains(id),
        Slot::Owned(_) => false,
    }
}

pub open spec fn slot_base_frontier<D: Domain>(slot: Slot<D>) -> Frontier<D> {
    match slot {
        Slot::Owned(frontier) => frontier,
        Slot::Lent { base, .. } => base,
    }
}

} // verus!
