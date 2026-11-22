use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::set::group_set_lemmas,
};

pub struct TransferRequest<'a> {
    pub locals: &'a [MemoryRegionSlice<'a>],
    pub remote: RemoteCapability,
    pub wr_id: u64,
    pub read: bool,
    pub fence: bool,
}

pub struct PostedBatchItem {
    pub wr_id: u64,
    pub event: Ghost<TransferEvent>,
    pub submitted: Tracked<Submitted>,
}

pub struct BatchPostOutcome {
    pub posted: Vec<PostedBatchItem>,
    pub error: Option<RdmaError>,
}

/// Citations held fixed for the duration of a doorbell.
pub ghost struct BatchInputs {
    pub infos: SubmittedMap,
    pub cited: ISet<TransferEvent>,
}

pub ghost struct BatchState {
    pub next: TransferEvent,
    pub reads: ISet<TransferEvent>,
    pub read_infos: IMap<TransferEvent, EventInfo<TransferDomain>>,
    pub pending: Map<u64, PendingWorkRequest>,
    pub cq_infos: IMap<TransferEvent, EventInfo<TransferDomain>>,
    pub owned: Map<SlotLoc, Frontier<TransferDomain>>,
    pub shares: Map<SlotLoc, (TransferShare, Frontier<TransferDomain>)>,
}

impl BatchState {
    pub open spec fn of(
        q: QueuePair,
        c: CompletionQueue,
        owned: Map<SlotLoc, SlotToken>,
        shares: Map<SlotLoc, ShareCap>,
    ) -> Self {
        Self {
            next: q.source().next_event(),
            reads: q.reads(),
            read_infos: q.read_infos().map(),
            pending: c.pending(),
            cq_infos: c.infos().map(),
            owned: owned_frontiers(owned),
            shares: share_frontiers(shares),
        }
    }

    pub open spec fn request_guards(r: TransferRequest) -> ReadGuards {
        if r.read {
            match r.remote.share() {
                Some(sh) => IMap::empty().insert(r.remote.slot_ref().loc, sh),
                None => IMap::empty(),
            }
        } else {
            IMap::empty()
        }
    }

    pub open spec fn ordered(self, r: TransferRequest, b: BatchInputs) -> ISet<TransferEvent> {
        closure_from(
            if r.fence {
                b.infos.map().union_prefer_right(self.read_infos)
            } else {
                b.infos.map()
            },
            (if r.fence {
                self.reads
            } else {
                ISet::empty()
            }).union(b.cited),
        )
    }

    pub open spec fn info(self, r: TransferRequest, b: BatchInputs) -> EventInfo<TransferDomain> {
        EventInfo {
            access: sge_transfer_access(r.read, r.locals@, r.remote.slot_ref()),
            effect: sge_copy_effect(r.read, r.locals@, r.remote.slot_ref()),
            read_guards: Self::request_guards(r),
            preds: self.ordered(r, b),
        }
    }

    pub open spec fn advance(self, r: TransferRequest, b: BatchInputs) -> Self {
        let e = self.next;
        let info = self.info(r, b);
        let rloc = r.remote.slot_ref().loc;
        Self {
            next: TransferEvent { seq: e.seq + 1, ..e },
            reads: if r.read {
                self.reads.insert(e)
            } else {
                self.reads
            },
            read_infos: if r.read {
                self.read_infos.insert(e, info)
            } else {
                self.read_infos
            },
            pending: self.pending.insert(
                r.wr_id,
                PendingWorkRequest { kind: PendingKind::DataTransfer { event: e } },
            ),
            cq_infos: self.cq_infos.insert(e, info),
            owned: Map::new(
                self.owned.dom(),
                |l: SlotLoc| advance(self.owned[l], l, e, info.access),
            ),
            shares: if r.read && r.remote.share() is Some {
                let (sh, fr) = self.shares[rloc];
                self.shares.insert(
                    rloc,
                    (sh, Frontier { last_writer: fr.last_writer, readers: fr.readers.insert(e) }),
                )
            } else {
                self.shares
            },
        }
    }

    pub open spec fn accepts(
        self,
        r: TransferRequest,
        b: BatchInputs,
        q: QueuePair,
        id: InstanceId,
        knows: Knows,
    ) -> bool {
        let access = sge_transfer_access(r.read, r.locals@, r.remote.slot_ref());
        let ordered = self.ordered(r, b);
        let rloc = r.remote.slot_ref().loc;
        &&& !self.pending.contains_key(r.wr_id)
        &&& r.read ==> q.effective_max_rd_atomic() > 0
        &&& q.valid_sges(id, r.locals@, r.remote, r.read)
        &&& b.cited.subset_of(b.infos.dom())
        &&& knowledge_of(q.source().node(), b.cited).subset_of(knows.set())
        &&& forall|i: int|
            0 <= i < r.locals@.len() ==> self.owned.contains_key(
                (#[trigger] r.locals@[i]).slot_ref().loc,
            )
        &&& forall|l: SlotLoc| #[trigger]
            self.owned.contains_key(l) ==> admitted(self.owned[l], l, access, ordered)
        &&& if r.read && r.remote.share() is Some {
            &&& self.shares.contains_key(rloc)
            &&& !self.owned.contains_key(rloc)
            &&& self.shares[rloc].0 == r.remote.share()->Some_0
            &&& read_admitted(self.shares[rloc].1, ordered)
        } else {
            self.owned.contains_key(rloc)
        }
    }
}

pub open spec fn batch_prefix(
    initial: BatchState,
    requests: Seq<TransferRequest>,
    n: int,
    b: BatchInputs,
) -> BatchState
    decreases n,
{
    if n <= 0 || n > requests.len() {
        initial
    } else {
        batch_prefix(initial, requests, n - 1, b).advance(requests[n - 1], b)
    }
}

pub open spec fn batch_admitted(
    q: QueuePair,
    c: CompletionQueue,
    requests: Seq<TransferRequest>,
    owned: Map<SlotLoc, SlotToken>,
    shares: Map<SlotLoc, ShareCap>,
    b: BatchInputs,
    id: InstanceId,
    knows: Knows,
) -> bool {
    &&& q.ready() && c.well_formed() && q.cq_id() == c.id() && c.observer() == q.source().node()
    &&& c.pending().len() + requests.len() <= c.capacity()
    &&& q.source().id() == id && c.infos().instance_id() == id && b.infos.instance_id() == id
    &&& knows.instance_id() == id
    &&& owned_map_valid(owned, id) && shares_map_valid(shares, id)
    &&& owned.dom().disjoint(shares.dom())
    &&& forall|i: int| 0 <= i < requests.len() ==> (#[trigger] requests[i]).read
    &&& forall|i: int|
        0 <= i < requests.len() ==> batch_prefix(
            BatchState::of(q, c, owned, shares),
            requests,
            i,
            b,
        ).accepts(#[trigger] requests[i], b, q, id, knows)
}

pub open spec fn batch_frame(
    q: QueuePair,
    c: CompletionQueue,
    owned: Map<SlotLoc, SlotToken>,
    shares: Map<SlotLoc, ShareCap>,
    q0: QueuePair,
    c0: CompletionQueue,
    owned0: Map<SlotLoc, SlotToken>,
    shares0: Map<SlotLoc, ShareCap>,
    id: InstanceId,
) -> bool {
    &&& q.ready() && c.well_formed() && c.same_queue(c0)
    &&& q.source().id() == q0.source().id() && q.source().source() == q0.source().source()
        && q.source().node() == q0.source().node()
    &&& q.peer() == q0.peer() && q.cq_id() == q0.cq_id() && q.pd_id() == q0.pd_id()
    &&& q.configuration() == q0.configuration()
    &&& q.effective_max_rd_atomic() == q0.effective_max_rd_atomic()
    &&& owned_map_valid(owned, id) && shares_map_valid(shares, id)
    &&& owned.dom() == owned0.dom() && shares.dom() == shares0.dom()
}

impl PostedBatchItem {
    pub open spec fn describes(
        self,
        state: BatchState,
        request: TransferRequest,
        b: BatchInputs,
        id: InstanceId,
    ) -> bool {
        self.wr_id == request.wr_id && self.event@ == state.next && self.submitted@.key()
            == state.next && self.submitted@.instance_id() == id && self.submitted@.value()
            == state.info(request, b)
    }
}

pub open spec fn batch_items(
    items: Seq<PostedBatchItem>,
    initial: BatchState,
    requests: Seq<TransferRequest>,
    b: BatchInputs,
    id: InstanceId,
) -> bool {
    forall|i: int|
        0 <= i < items.len() ==> (#[trigger] items[i]).describes(
            batch_prefix(initial, requests, i, b),
            requests[i],
            b,
            id,
        )
}

pub fn post_send_batch(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    requests: &[TransferRequest],
    Tracked(owned): Tracked<&mut Map<SlotLoc, SlotToken>>,
    Tracked(shares): Tracked<&mut Map<SlotLoc, ShareCap>>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (outcome: BatchPostOutcome)
    requires
        forall|i: int, j: int|
            0 <= i < requests@.len() && 0 <= j < (#[trigger] requests@[i]).locals@.len() ==> (
            #[trigger] requests@[i].locals@[j]).registration().gpu_device_spec().is_none(),
        batch_admitted(
            *old(qp),
            *old(cq),
            requests@,
            *old(owned),
            *old(shares),
            BatchInputs { infos: *old(infos), cited },
            instance.id(),
            *knows,
        ),
    ensures
        ({
            let b = BatchInputs { infos: *old(infos), cited };
            let initial = BatchState::of(*old(qp), *old(cq), *old(owned), *old(shares));
            let n = outcome.posted@.len();
            &&& if outcome.error.is_none() {
                n == requests.len()
            } else {
                n <= requests.len()
            }
            &&& final(infos).map() == old(infos).map() && final(infos).instance_id()
                == instance.id()
            &&& batch_frame(
                *final(qp),
                *final(cq),
                *final(owned),
                *final(shares),
                *old(qp),
                *old(cq),
                *old(owned),
                *old(shares),
                instance.id(),
            )
            &&& BatchState::of(*final(qp), *final(cq), *final(owned), *final(shares))
                == batch_prefix(initial, requests@, n as int, b)
            &&& batch_items(outcome.posted@, initial, requests@, b, instance.id())
            &&& n == 0 ==> *final(qp) == *old(qp) && *final(cq) == *old(cq) && *final(owned)
                == *old(owned) && *final(shares) == *old(shares) && *final(infos) == *old(infos)
        }),
{
    hide(QueuePair::valid_sges);
    hide(sge_transfer_access);
    hide(sge_copy_effect);
    hide(closure_from);
    let ghost b = BatchInputs { infos: *infos, cited };
    let ghost initial = BatchState::of(*qp, *cq, *owned, *shares);
    assert forall|i: int| 0 <= i < requests.len() implies {
        let r = #[trigger] requests@[i];
        qp.valid_sges(instance.id(), r.locals@, r.remote, r.read)
    } by {
        assert(batch_prefix(initial, requests@, i, b).accepts(
            requests@[i],
            b,
            *qp,
            instance.id(),
            *knows,
        ));
    }
    let (count, error, Tracked(mut prefix)) = ffi::post_batch(qp, requests);
    let mut posted = Vec::with_capacity(count);
    let mut i = 0;
    while i < count
        invariant
            if error.is_none() {
                count == requests.len()
            } else {
                count <= requests.len()
            },
            prefix.descriptors() == requests@.take(count as int),
            prefix.cursor() == i as int,
            i == posted.len() <= count,
            b == (BatchInputs { infos: *old(infos), cited }),
            initial == BatchState::of(*old(qp), *old(cq), *old(owned), *old(shares)),
            batch_admitted(
                *old(qp),
                *old(cq),
                requests@,
                *old(owned),
                *old(shares),
                b,
                instance.id(),
                *knows,
            ),
            infos.map() == b.infos.map(),
            infos.instance_id() == instance.id(),
            batch_frame(
                *qp,
                *cq,
                *owned,
                *shares,
                *old(qp),
                *old(cq),
                *old(owned),
                *old(shares),
                instance.id(),
            ),
            cq.pending().len() == old(cq).pending().len() + i,
            BatchState::of(*qp, *cq, *owned, *shares) == batch_prefix(
                initial,
                requests@,
                i as int,
                b,
            ),
            batch_items(posted@, initial, requests@, b, instance.id()),
            i == 0 ==> *qp == *old(qp) && *cq == *old(cq) && *owned == *old(owned) && *shares
                == *old(shares) && *infos == *old(infos),
        decreases count - i,
    {
        let r = &requests[i];
        let ghost state = BatchState::of(*qp, *cq, *owned, *shares);
        let ghost info = state.info(*r, b);
        let ghost rloc = r.remote.slot_ref().loc;
        let ghost use_share = r.read && r.remote.share() is Some;
        let tracked mut opt_share: Option<ShareCap> = None;
        proof {
            assert(state.accepts(*r, b, *old(qp), instance.id(), *knows));
            assert(prefix.descriptors()[i as int] == *r);
            reveal(sge_transfer_access);
            if use_share {
                opt_share = Some(shares.tracked_remove(rloc));
            }
            assert forall|l: SlotLoc| #[trigger]
                info.access.writes.contains(l) implies owned.contains_key(l) by {
                if r.read {
                    let k = choose|k: int|
                        0 <= k < r.locals@.len() && (#[trigger] r.locals@[k].slot_ref()).loc == l;
                    assert(state.owned.contains_key(r.locals@[k].slot_ref().loc));
                }
            }
            assert forall|l: SlotLoc| #[trigger]
                info.access.reads.contains(l) implies owned.contains_key(l) || (use_share && l
                == rloc) by {
                if r.read {
                    if l != rloc {
                        let k = choose|k: int|
                            0 <= k < r.locals@.len() && (#[trigger] r.locals@[k].slot_ref()).loc
                                == l && !r.locals@[k].slot_ref().is_full();
                        assert(state.owned.contains_key(r.locals@[k].slot_ref().loc));
                    }
                } else if l != rloc {
                    let k = choose|k: int|
                        0 <= k < r.locals@.len() && (#[trigger] r.locals@[k].slot_ref()).loc == l;
                    assert(state.owned.contains_key(r.locals@[k].slot_ref().loc));
                }
            }
        }
        let tracked work = prefix.work(i as int, qp.reads());
        let (event, submitted, Tracked(out_share)) = submit_transfer_multi(
            qp,
            cq,
            Tracked(instance),
            Tracked(work),
            r.wr_id,
            r.fence,
            r.read,
            Tracked(owned),
            Tracked(opt_share),
            Tracked(infos),
            Tracked(knows),
            Ghost(cited),
        );
        proof {
            if use_share {
                let tracked cap = out_share.tracked_unwrap();
                shares.tracked_insert(rloc, cap);
                assert(shares.dom() =~= old(shares).dom());
            }
            assert(owned_frontiers(*owned) =~= state.advance(*r, b).owned);
            assert(share_frontiers(*shares) =~= state.advance(*r, b).shares);
            assert(BatchState::of(*qp, *cq, *owned, *shares) == state.advance(*r, b));
            assert(batch_prefix(initial, requests@, i + 1, b) == state.advance(*r, b));
        }
        let ghost prior = posted@;
        posted.push(PostedBatchItem { wr_id: r.wr_id, event, submitted });
        proof {
            assert forall|j: int| 0 <= j < posted.len() implies (#[trigger] posted@[j]).describes(
                batch_prefix(initial, requests@, j, b),
                requests@[j],
                b,
                instance.id(),
            ) by {
                if j < i {
                    assert(posted@[j] == prior[j]);
                }
            }
        }
        i += 1;
    }
    BatchPostOutcome { posted, error }
}

proof fn record_sge_writers(
    writers: Map<SlotLoc, TransferEvent>,
    locals: Seq<MemoryRegionSlice>,
    ev: TransferEvent,
    n: int,
    owned_dom: Set<SlotLoc>,
) -> (out: Map<SlotLoc, TransferEvent>)
    requires
        0 <= n <= locals.len(),
        writers.dom().subset_of(owned_dom),
        forall|i: int|
            0 <= i < locals.len() ==> owned_dom.contains((#[trigger] locals[i]).slot_ref().loc),
    ensures
        out.dom().subset_of(owned_dom),
        forall|l: SlotLoc| #[trigger]
            out.contains_key(l) ==> (out[l] == ev || (writers.contains_key(l) && out[l]
                == writers[l])),
    decreases n,
{
    if n == 0 {
        writers
    } else {
        let prev = record_sge_writers(writers, locals, ev, n - 1, owned_dom);
        prev.insert(locals[n - 1].slot_ref().loc, ev)
    }
}

proof fn extract_written_gdr_tokens(
    tracked owned: &mut Map<SlotLoc, SlotToken>,
    writers: Map<SlotLoc, TransferEvent>,
) -> (tracked extracted: Map<SlotLoc, SlotToken>)
    requires
        writers.dom().subset_of(old(owned).dom()),
    ensures
        extracted.dom() == writers.dom(),
        final(owned).dom() == old(owned).dom().difference(writers.dom()),
        forall|l: SlotLoc| #[trigger] extracted.contains_key(l) ==> extracted[l] == old(owned)[l],
        forall|l: SlotLoc| #[trigger]
            final(owned).contains_key(l) ==> final(owned)[l] == old(owned)[l],
    decreases writers.dom().len(),
{
    if writers.dom().is_empty() {
        let tracked m = Map::tracked_empty();
        assert(m.dom() =~= writers.dom());
        assert(owned.dom() =~= old(owned).dom().difference(writers.dom()));
        m
    } else {
        let l = writers.dom().choose();
        let rest_w = writers.remove(l);
        assert(rest_w.dom().len() < writers.dom().len());
        let tracked tok = owned.tracked_remove(l);
        let tracked mut extracted = extract_written_gdr_tokens(owned, rest_w);
        extracted.tracked_insert(l, tok);
        assert(extracted.dom() =~= writers.dom());
        assert(owned.dom() =~= old(owned).dom().difference(writers.dom()));
        extracted
    }
}

/// GPU-target batched RDMA READ doorbell: seals every written GPU `SlotToken`
/// out of `owned` into `PendingGdrSlots` until `MemoryRegion::complete_gdr_slots`.
pub fn post_send_batch_gdr(
    qp: &mut QueuePair,
    cq: &mut CompletionQueue,
    Tracked(instance): Tracked<&TransferInstance>,
    requests: &[TransferRequest],
    Tracked(owned): Tracked<&mut Map<SlotLoc, SlotToken>>,
    Tracked(shares): Tracked<&mut Map<SlotLoc, ShareCap>>,
    Tracked(infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: (BatchPostOutcome, Tracked<PendingGdrSlots>))
    requires
        batch_admitted(
            *old(qp),
            *old(cq),
            requests@,
            *old(owned),
            *old(shares),
            BatchInputs { infos: *old(infos), cited },
            instance.id(),
            *knows,
        ),
    ensures
        ({
            let b = BatchInputs { infos: *old(infos), cited };
            let initial = BatchState::of(*old(qp), *old(cq), *old(owned), *old(shares));
            let outcome = result.0;
            let pending = result.1@;
            let n = outcome.posted@.len();
            let final_state = batch_prefix(initial, requests@, n as int, b);
            &&& if outcome.error.is_none() {
                n == requests.len()
            } else {
                n <= requests.len()
            }
            &&& final(infos).map() == old(infos).map() && final(infos).instance_id()
                == instance.id()
            &&& final(qp).ready() && final(cq).well_formed() && final(cq).same_queue(*old(cq))
            &&& final(qp).source().id() == old(qp).source().id() && final(qp).source().source()
                == old(qp).source().source() && final(qp).source().node() == old(qp).source().node()
            &&& final(qp).peer() == old(qp).peer() && final(qp).cq_id() == old(qp).cq_id()
            &&& final(qp).pd_id() == old(qp).pd_id()
            &&& final(qp).configuration() == old(qp).configuration()
            &&& final(qp).effective_max_rd_atomic() == old(qp).effective_max_rd_atomic()
            &&& owned_map_valid(*final(owned), instance.id()) && shares_map_valid(
                *final(shares),
                instance.id(),
            )
            &&& final(shares).dom() == old(shares).dom()
            &&& pending.well_formed() && pending.instance_id() == instance.id()
            &&& pending.dom().subset_of(old(owned).dom())
            &&& final(owned).dom() == old(owned).dom().difference(pending.dom())
            &&& forall|l: SlotLoc| #[trigger]
                pending.dom().contains(l) ==> pending.frontiers()[l] == final_state.owned[l]
            &&& forall|l: SlotLoc| #[trigger]
                final(owned).contains_key(l) ==> slot_frontier(final(owned)[l])
                    == final_state.owned[l]
            &&& forall|l: SlotLoc| #[trigger]
                pending.dom().contains(l) ==> exists|k: int|
                    0 <= k < n && (#[trigger] outcome.posted@[k]).event@ == pending.writers()[l]
            &&& batch_items(outcome.posted@, initial, requests@, b, instance.id())
        }),
{
    hide(QueuePair::valid_sges);
    hide(sge_transfer_access);
    hide(sge_copy_effect);
    hide(closure_from);
    let ghost b = BatchInputs { infos: *infos, cited };
    let ghost initial = BatchState::of(*qp, *cq, *owned, *shares);
    assert forall|i: int| 0 <= i < requests.len() implies {
        let r = #[trigger] requests@[i];
        qp.valid_sges(instance.id(), r.locals@, r.remote, r.read)
    } by {
        assert(batch_prefix(initial, requests@, i, b).accepts(
            requests@[i],
            b,
            *qp,
            instance.id(),
            *knows,
        ));
    }
    let (count, error, Tracked(mut prefix)) = ffi::post_batch(qp, requests);
    let mut posted = Vec::with_capacity(count);
    let ghost mut writers = Map::<SlotLoc, TransferEvent>::empty();
    let mut i = 0;
    while i < count
        invariant
            if error.is_none() {
                count == requests.len()
            } else {
                count <= requests.len()
            },
            prefix.descriptors() == requests@.take(count as int),
            prefix.cursor() == i as int,
            i == posted.len() <= count,
            b == (BatchInputs { infos: *old(infos), cited }),
            initial == BatchState::of(*old(qp), *old(cq), *old(owned), *old(shares)),
            batch_admitted(
                *old(qp),
                *old(cq),
                requests@,
                *old(owned),
                *old(shares),
                b,
                instance.id(),
                *knows,
            ),
            infos.map() == b.infos.map(),
            infos.instance_id() == instance.id(),
            batch_frame(
                *qp,
                *cq,
                *owned,
                *shares,
                *old(qp),
                *old(cq),
                *old(owned),
                *old(shares),
                instance.id(),
            ),
            cq.pending().len() == old(cq).pending().len() + i,
            BatchState::of(*qp, *cq, *owned, *shares) == batch_prefix(
                initial,
                requests@,
                i as int,
                b,
            ),
            batch_items(posted@, initial, requests@, b, instance.id()),
            writers.dom().subset_of(old(owned).dom()),
            forall|l: SlotLoc| #[trigger]
                writers.contains_key(l) ==> exists|k: int|
                    0 <= k < i && (#[trigger] posted@[k]).event@ == writers[l],
        decreases count - i,
    {
        let r = &requests[i];
        let ghost state = BatchState::of(*qp, *cq, *owned, *shares);
        let ghost info = state.info(*r, b);
        let ghost rloc = r.remote.slot_ref().loc;
        let ghost use_share = r.read && r.remote.share() is Some;
        let tracked mut opt_share: Option<ShareCap> = None;
        proof {
            assert(state.accepts(*r, b, *old(qp), instance.id(), *knows));
            assert(prefix.descriptors()[i as int] == *r);
            reveal(sge_transfer_access);
            if use_share {
                opt_share = Some(shares.tracked_remove(rloc));
            }
            assert forall|l: SlotLoc| #[trigger]
                info.access.writes.contains(l) implies owned.contains_key(l) by {
                let k = choose|k: int|
                    0 <= k < r.locals@.len() && (#[trigger] r.locals@[k].slot_ref()).loc == l;
                assert(state.owned.contains_key(r.locals@[k].slot_ref().loc));
            }
            assert forall|l: SlotLoc| #[trigger]
                info.access.reads.contains(l) implies owned.contains_key(l) || (use_share && l
                == rloc) by {
                if l != rloc {
                    let k = choose|k: int|
                        0 <= k < r.locals@.len() && (#[trigger] r.locals@[k].slot_ref()).loc == l
                            && !r.locals@[k].slot_ref().is_full();
                    assert(state.owned.contains_key(r.locals@[k].slot_ref().loc));
                }
            }
        }
        let tracked work = prefix.work(i as int, qp.reads());
        let (event, submitted, Tracked(out_share)) = submit_transfer_multi(
            qp,
            cq,
            Tracked(instance),
            Tracked(work),
            r.wr_id,
            r.fence,
            r.read,
            Tracked(owned),
            Tracked(opt_share),
            Tracked(infos),
            Tracked(knows),
            Ghost(cited),
        );
        proof {
            if use_share {
                let tracked cap = out_share.tracked_unwrap();
                shares.tracked_insert(rloc, cap);
                assert(shares.dom() =~= old(shares).dom());
            }
            assert(owned_frontiers(*owned) =~= state.advance(*r, b).owned);
            assert(share_frontiers(*shares) =~= state.advance(*r, b).shares);
            assert(BatchState::of(*qp, *cq, *owned, *shares) == state.advance(*r, b));
            assert(batch_prefix(initial, requests@, i + 1, b) == state.advance(*r, b));
        }
        let ghost prior = posted@;
        posted.push(PostedBatchItem { wr_id: r.wr_id, event, submitted });
        proof {
            let ghost prev_writers = writers;
            assert forall|idx: int| 0 <= idx < r.locals@.len() implies old(owned).dom().contains(
                (#[trigger] r.locals@[idx]).slot_ref().loc,
            ) by {
                assert(state.owned.contains_key(r.locals@[idx].slot_ref().loc));
            }
            writers =
            record_sge_writers(
                prev_writers,
                r.locals@,
                event@,
                r.locals@.len() as int,
                old(owned).dom(),
            );
            assert forall|j: int| 0 <= j < posted.len() implies (#[trigger] posted@[j]).describes(
                batch_prefix(initial, requests@, j, b),
                requests@[j],
                b,
                instance.id(),
            ) by {
                if j < i {
                    assert(posted@[j] == prior[j]);
                }
            }
            assert forall|l: SlotLoc| #[trigger] writers.contains_key(l) implies exists|k: int|
                0 <= k < i + 1 && (#[trigger] posted@[k]).event@ == writers[l] by {
                if writers[l] == event@ {
                    assert(posted@[i as int].event@ == event@);
                } else {
                    let k = choose|k: int|
                        0 <= k < i && (#[trigger] prior[k]).event@ == prev_writers[l];
                    assert(posted@[k] == prior[k]);
                }
            }
        }
        i += 1;
    }
    let tracked gdr_tokens = extract_written_gdr_tokens(owned, writers);
    let tracked pending = PendingGdrSlots { instance: instance.id(), writers, tokens: gdr_tokens };
    (BatchPostOutcome { posted, error }, Tracked(pending))
}

} // verus!
