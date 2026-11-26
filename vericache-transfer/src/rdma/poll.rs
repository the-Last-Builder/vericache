use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

#[allow(inconsistent_fields)]
pub enum Completed {
    Data {
        succeeded: Tracked<Succeeded>,
        wr_id: u64,
        event: Ghost<TransferEvent>,
        submitted: Tracked<Submitted>,
        knows: Tracked<Knows>,
    },
    DataFailed {
        wr_id: u64,
        status: i32,
        event: Ghost<TransferEvent>,
        submitted: Tracked<Submitted>,
        knows: Tracked<Knows>,
    },
    WindowBound { wr_id: u64, witness: Tracked<BindWitness> },
    BindFailed { wr_id: u64, status: i32, witness: Tracked<BindFailedWitness> },
    WindowInvalidated { wr_id: u64, witness: Tracked<InvalidateWitness> },
    ControlFailed { wr_id: u64, status: i32 },
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct BindWitness {
    pub(super) ghost window_id: nat,
    pub(super) ghost wr_id: u64,
    pub(super) ghost node: NodeId,
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct BindFailedWitness {
    pub(super) ghost window_id: nat,
    pub(super) ghost wr_id: u64,
    pub(super) ghost node: NodeId,
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct InvalidateWitness {
    pub(super) ghost window_id: nat,
    pub(super) ghost wr_id: u64,
    pub(super) ghost node: NodeId,
}

impl BindWitness {
    pub open(super) spec fn matches(self, w: MemoryWindow) -> bool {
        self.window_id == w.id() && self.node == w.node() && w.state() is Binding && self.wr_id
            == w.state()->Binding_wr_id
    }
}

impl BindFailedWitness {
    pub open(super) spec fn matches(self, w: MemoryWindow) -> bool {
        self.window_id == w.id() && self.node == w.node() && w.state() is Binding && self.wr_id
            == w.state()->Binding_wr_id
    }
}

impl InvalidateWitness {
    pub open(super) spec fn matches(self, w: MemoryWindow) -> bool {
        self.window_id == w.id() && self.node == w.node() && w.state() is Invalidating && self.wr_id
            == w.state()->Invalidating_wr_id
    }
}

pub enum PollStatus {
    Empty,
    Completed(Completed),
}

pub ghost struct CqSnapshot {
    pub observer: NodeId,
    pub pending: Map<u64, PendingWorkRequest>,
    pub infos: IMap<TransferEvent, EventInfo<TransferDomain>>,
}

impl CqSnapshot {
    pub open spec fn of(cq: CompletionQueue) -> Self {
        Self { observer: cq.observer(), pending: cq.pending(), infos: cq.infos().map() }
    }

    pub open spec fn reap(self, id: u64) -> Self {
        let pending = self.pending.remove(id);
        let kind = self.pending[id].kind;
        let infos = if kind is DataTransfer && !(exists|k: u64|
            pending.contains_key(k) && (#[trigger] pending[k]).kind == kind) {
            self.infos.remove(kind->DataTransfer_event)
        } else {
            self.infos
        };
        Self { pending, infos, ..self }
    }
}

impl Completed {
    pub open spec fn id(self) -> u64 {
        match self {
            Self::Data { wr_id, .. }
            | Self::DataFailed { wr_id, .. }
            | Self::WindowBound { wr_id, .. }
            | Self::BindFailed { wr_id, .. }
            | Self::WindowInvalidated { wr_id, .. }
            | Self::ControlFailed { wr_id, .. } => wr_id,
        }
    }

    pub open spec fn describes(
        self,
        before: CompletionQueue,
        after: CompletionQueue,
        id: InstanceId,
    ) -> bool {
        self.describes_snapshot(CqSnapshot::of(before), CqSnapshot::of(after), id)
    }

    pub open spec fn describes_snapshot(
        self,
        before: CqSnapshot,
        after: CqSnapshot,
        id: InstanceId,
    ) -> bool {
        let data = |
            wr_id: u64,
            event: Ghost<TransferEvent>,
            submitted: Tracked<Submitted>,
            knows: Tracked<Knows>,
        |
            before.pending.contains_key(wr_id) && before.pending[wr_id].kind == (
            PendingKind::DataTransfer { event: event@ }) && after.pending == before.pending.remove(
                wr_id,
            ) && submitted@.instance_id() == id && submitted@.key() == event@ && submitted@.value()
                == before.infos[event@] && knows@.instance_id() == id && knows@.set()
                == knowledge_of(
                before.observer,
                closure_from(before.infos, ISet::empty().insert(event@)),
            );
        match self {
            Completed::Data { wr_id, event, submitted, knows, succeeded } => data(
                wr_id,
                event,
                submitted,
                knows,
            ) && succeeded@.event() == event@ && succeeded@.instance_id() == id,
            Completed::DataFailed { wr_id, status, event, submitted, knows } => status != 0 && data(
                wr_id,
                event,
                submitted,
                knows,
            ),
            Completed::WindowBound { wr_id, witness } => before.pending.contains_key(wr_id)
                && before.pending[wr_id].kind is BindWindow && after.pending
                == before.pending.remove(wr_id) && (forall|w: MemoryWindow|
                w.state() is Binding && w.state()->Binding_wr_id == wr_id && w.id()
                    == before.pending[wr_id].kind->BindWindow_window_id && w.node()
                    == before.observer ==> #[trigger] witness@.matches(w)),
            Completed::BindFailed { wr_id, status, witness } => status != 0
                && before.pending.contains_key(wr_id) && before.pending[wr_id].kind is BindWindow
                && after.pending == before.pending.remove(wr_id) && (forall|w: MemoryWindow|
                w.state() is Binding && w.state()->Binding_wr_id == wr_id && w.id()
                    == before.pending[wr_id].kind->BindWindow_window_id && w.node()
                    == before.observer ==> #[trigger] witness@.matches(w)),
            Completed::WindowInvalidated { wr_id, witness } => before.pending.contains_key(wr_id)
                && before.pending[wr_id].kind is LocalInvalidate && after.pending
                == before.pending.remove(wr_id) && (forall|w: MemoryWindow|
                w.state() is Invalidating && w.state()->Invalidating_wr_id == wr_id && w.id()
                    == before.pending[wr_id].kind->LocalInvalidate_window_id && w.node()
                    == before.observer ==> #[trigger] witness@.matches(w)),
            Completed::ControlFailed { wr_id, status } => status != 0
                && before.pending.contains_key(wr_id) && !(
            before.pending[wr_id].kind is DataTransfer) && after.pending == before.pending.remove(
                wr_id,
            ),
        }
    }
}

pub fn poll_completion_queue(
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
) -> (r: Result<PollStatus, RdmaError>)
    requires
        old(cq).well_formed(),
        old(cq).infos().instance_id() == instance.id(),
    ensures
        final(cq).well_formed(),
        final(cq).same_queue(*old(cq)),
        match r {
            Ok(PollStatus::Completed(c)) => c.describes(*old(cq), *final(cq), instance.id()),
            _ => *final(cq) == *old(cq),
        },
{
    let Some((wr_id, status, work, evidence)) = ffi::poll(cq)? else {
        return Ok(PollStatus::Empty);
    };
    Ok(
        PollStatus::Completed(
            complete_one_cqe(
                cq,
                wr_id,
                status,
                work,
                Tracked(instance),
                Tracked(evidence.borrow()),
            ),
        ),
    )
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn complete_one_cqe(
    cq: &mut CompletionQueue,
    wr_id: u64,
    status: i32,
    work: ffi::WorkKind,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(evidence): Tracked<&ObservedEvents>,
) -> (c: Completed)
    requires
        old(cq).well_formed(),
        old(cq).infos().instance_id() == instance.id(),
        old(cq).pending().contains_key(wr_id),
        ffi::matches_pending(work, old(cq).pending()[wr_id].kind),
        ffi::observed_cqe(
            *evidence,
            instance.id(),
            old(cq).observer(),
            old(cq).pending()[wr_id].kind,
            status,
        ),
    ensures
        final(cq).well_formed(),
        final(cq).same_queue(*old(cq)),
        c.id() == wr_id,
        c.describes(*old(cq), *final(cq), instance.id()),
        CqSnapshot::of(*final(cq)) == CqSnapshot::of(*old(cq)).reap(wr_id),
{
    let ghost kind = cq.pending()[wr_id].kind;
    let ghost node = cq.observer();
    let ghost data = kind is DataTransfer;
    let ghost ev = kind->DataTransfer_event;
    let tracked mut receipt = None;
    let tracked mut observation = None;
    cq.in_flight_count -= 1;
    proof {
        cq.in_flight = Ghost(cq.pending().remove(wr_id));
        if data {
            observation =
            Some(
                instance.observe_completed(
                    node,
                    ISet::empty().insert(ev),
                    cq.in_flight_infos.borrow(),
                    evidence,
                ),
            );
            let tracked s = cq.in_flight_infos.borrow_mut().remove(ev);
            if exists|k: u64|
                cq.pending().contains_key(k) && (#[trigger] cq.pending()[k]).kind == (
                PendingKind::DataTransfer { event: ev }) {
                cq.in_flight_infos.borrow_mut().insert(s);
            }
            assert(cq.infos().map() =~= CqSnapshot::of(*old(cq)).reap(wr_id).infos);
            receipt = Some(s);
        }
    }
    let c = match work {
        ffi::WorkKind::Data => {
            let submitted = Tracked(receipt.tracked_unwrap());
            let knows = Tracked(observation.tracked_unwrap());
            if status == 0 {
                Completed::Data {
                    wr_id,
                    event: Ghost(ev),
                    submitted,
                    knows,
                    succeeded: Tracked(evidence.success(instance.id(), ev)),
                }
            } else {
                Completed::DataFailed { wr_id, status, event: Ghost(ev), submitted, knows }
            }
        },
        ffi::WorkKind::BindWindow if status != 0 => Completed::BindFailed {
            wr_id,
            status,
            witness: Tracked(
                BindFailedWitness { window_id: kind->BindWindow_window_id, wr_id, node },
            ),
        },
        _ if status != 0 => Completed::ControlFailed { wr_id, status },
        ffi::WorkKind::BindWindow => Completed::WindowBound {
            wr_id,
            witness: Tracked(BindWitness { window_id: kind->BindWindow_window_id, wr_id, node }),
        },
        ffi::WorkKind::LocalInvalidate => Completed::WindowInvalidated {
            wr_id,
            witness: Tracked(
                InvalidateWitness { window_id: kind->LocalInvalidate_window_id, wr_id, node },
            ),
        },
    };
    c
}

/// Fold the observations in poll order, checking each witness against the CQ
/// immediately before its removal. The empty prefix preserves the entire CQ.
pub open spec fn completion_prefix(
    cs: Seq<Completed>,
    n: int,
    before: CqSnapshot,
    id: InstanceId,
) -> (CqSnapshot, bool)
    decreases n,
{
    if n <= 0 || n > cs.len() {
        (before, true)
    } else {
        let (prior, valid) = completion_prefix(cs, n - 1, before, id);
        let c = cs[n - 1];
        let after = prior.reap(c.id());
        (after, valid && c.describes_snapshot(prior, after, id))
    }
}

pub open spec fn completions_describe(
    cs: Seq<Completed>,
    before_cq: CompletionQueue,
    after_cq: CompletionQueue,
    instance_id: InstanceId,
) -> bool {
    completion_prefix(cs, cs.len() as int, CqSnapshot::of(before_cq), instance_id) == (
        CqSnapshot::of(after_cq),
        true,
    )
}

proof fn completion_prefix_push(
    cs: Seq<Completed>,
    c: Completed,
    n: int,
    before: CqSnapshot,
    id: InstanceId,
)
    requires
        0 <= n <= cs.len(),
    ensures
        completion_prefix(cs.push(c), n, before, id) == completion_prefix(cs, n, before, id),
    decreases n,
{
    if n > 0 {
        completion_prefix_push(cs, c, n - 1, before, id);
    }
}

pub(super) proof fn completion_at(
    cs: Seq<Completed>,
    i: int,
    n: int,
    before: CqSnapshot,
    id: InstanceId,
)
    requires
        0 <= i < n <= cs.len(),
        completion_prefix(cs, n, before, id).1,
    ensures
        cs[i].describes_snapshot(
            completion_prefix(cs, i, before, id).0,
            completion_prefix(cs, i + 1, before, id).0,
            id,
        ),
    decreases n - i,
{
    if i + 1 < n {
        completion_at(cs, i, n - 1, before, id);
    }
}

pub fn poll_completion_queue_batch(
    cq: &mut CompletionQueue,
    max_entries: usize,
    Tracked(instance): Tracked<&TransferInstance>,
) -> (r: Result<Vec<Completed>, RdmaError>)
    requires
        old(cq).well_formed(),
        old(cq).infos().instance_id() == instance.id(),
    ensures
        final(cq).well_formed(),
        final(cq).same_queue(*old(cq)),
        match r {
            Err(_) => *final(cq) == *old(cq),
            Ok(v) => v.len() <= max_entries && completions_describe(
                v@,
                *old(cq),
                *final(cq),
                instance.id(),
            ),
        },
{
    let raw = ffi::poll_batch(cq, max_entries)?;
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    let ghost before = CqSnapshot::of(*cq);
    while i < raw.len()
        invariant
            i == out.len() <= raw.len() <= max_entries,
            cq.well_formed(),
            cq.same_queue(*old(cq)),
            old(cq).infos().instance_id() == instance.id(),
            before == CqSnapshot::of(*old(cq)),
            completions_describe(out@, *old(cq), *cq, instance.id()),
            forall|j: int|
                #![trigger raw@[j]]
                i <= j < raw.len() ==> cq.pending().contains_key(raw@[j].0) && ffi::matches_pending(
                    raw@[j].2,
                    cq.pending()[raw@[j].0].kind,
                ) && ffi::observed_cqe(
                    raw@[j].3@,
                    instance.id(),
                    cq.observer(),
                    cq.pending()[raw@[j].0].kind,
                    raw@[j].1,
                ),
            forall|j: int, k: int|
                0 <= j < k < raw.len() ==> (#[trigger] raw@[j]).0 != (#[trigger] raw@[k]).0,
        decreases raw.len() - i,
    {
        let (id, status, kind, evidence) = &raw [i];
        let ghost current = CqSnapshot::of(*cq);
        assert(completion_prefix(out@, i as int, before, instance.id()) == (current, true));
        let c = complete_one_cqe(
            cq,
            *id,
            *status,
            *kind,
            Tracked(instance),
            Tracked(evidence.borrow()),
        );
        let ghost previous = out@;
        out.push(c);
        proof {
            completion_prefix_push(previous, c, i as int, before, instance.id());
            assert(out@[i as int] == c);
            assert(completion_prefix(out@, i + 1, before, instance.id()) == (
                CqSnapshot::of(*cq),
                true,
            ));
        }
        i += 1;
    }
    Ok(out)
}

} // verus!
