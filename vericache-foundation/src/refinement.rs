//! Refinement: how backends drive the shared history.
//!
//! A backend is a state machine whose steps are labeled with history actions.
//! A system pairs one shared `History` with a backend; every backend step
//! takes the history step with the same label. Several backends combine into
//! one with `Both`, so a system with any number of backends still has a
//! single history, and admission is checked against all of their events.
use crate::domain::Domain;
use crate::history::*;
use core::marker::PhantomData;
use vstd::prelude::*;

verus! {

pub trait Backend<D: Domain> {
    type State;

    spec fn init(state: Self::State) -> bool;

    spec fn step(state: Self::State, action: Action<D>, next: Self::State) -> bool;
}

/// Two backends side by side: either one steps, the other stays unchanged.
pub struct Both<First, Second> {
    _backends: PhantomData<(First, Second)>,
}

impl<D: Domain, First: Backend<D>, Second: Backend<D>> Backend<D> for Both<First, Second> {
    type State = (First::State, Second::State);

    open spec fn init(state: Self::State) -> bool {
        First::init(state.0) && Second::init(state.1)
    }

    open spec fn step(state: Self::State, action: Action<D>, next: Self::State) -> bool {
        ||| First::step(state.0, action, next.0) && next.1 == state.1
        ||| Second::step(state.1, action, next.1) && next.0 == state.0
    }
}

pub struct SystemState<D: Domain, BackendState> {
    pub history: History<D>,
    pub backend: BackendState,
}

pub open spec fn system_init<D: Domain, B: Backend<D>>(state: SystemState<D, B::State>) -> bool {
    state.history == initial::<D>() && B::init(state.backend)
}

/// The backend and the history take a step with the same action.
pub open spec fn system_step<D: Domain, B: Backend<D>>(
    state: SystemState<D, B::State>,
    action: Action<D>,
    next: SystemState<D, B::State>,
) -> bool {
    &&& B::step(state.backend, action, next.backend)
    &&& history_step(state.history, action, next.history)
}

pub open spec fn system_step_at<D: Domain, B: Backend<D>>(
    trace: Seq<SystemState<D, B::State>>,
    i: int,
) -> bool {
    exists|action: Action<D>| #[trigger] system_step::<D, B>(trace[i], action, trace[i + 1])
}

pub open spec fn is_system_trace<D: Domain, B: Backend<D>>(
    trace: Seq<SystemState<D, B::State>>,
) -> bool {
    &&& trace.len() > 0
    &&& system_init::<D, B>(trace[0])
    &&& forall|i: int| 0 <= i < trace.len() - 1 ==> #[trigger] system_step_at::<D, B>(trace, i)
}

pub open spec fn system_reachable<D: Domain, B: Backend<D>>(
    state: SystemState<D, B::State>,
) -> bool {
    exists|trace: Seq<SystemState<D, B::State>>| #[trigger]
        is_system_trace::<D, B>(trace) && trace.last() == state
}

/// In every reachable system state, the history is a reachable `History`,
/// so `race_free`, `every_event_isolated` and `read_returns_last_write`
/// apply to it.
pub proof fn system_history_reachable<D: Domain, B: Backend<D>>(state: SystemState<D, B::State>)
    requires
        system_reachable::<D, B>(state),
    ensures
        reachable(state.history),
{
    let trace = choose|trace: Seq<SystemState<D, B::State>>| #[trigger]
        is_system_trace::<D, B>(trace) && trace.last() == state;
    // Forget the backend states.
    let histories = trace.map_values(|system: SystemState<D, B::State>| system.history);
    assert forall|i: int| 0 <= i < histories.len() - 1 implies #[trigger] step_at(histories, i) by {
        assert(system_step_at::<D, B>(trace, i));
        let action = choose|action: Action<D>| #[trigger]
            system_step::<D, B>(trace[i], action, trace[i + 1]);
        assert(history_step(histories[i], action, histories[i + 1]));
    }
    assert(is_trace(histories));
    assert(histories.last() == state.history);
}

} // verus!
