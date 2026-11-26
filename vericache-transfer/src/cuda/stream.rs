use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

verus! {

broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

exec static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_stream_id() -> Result<u64, CudaError> {
    let mut attempt = 0usize;
    while attempt < 64
        invariant
            attempt <= 64,
        decreases 64 - attempt,
    {
        let id = NEXT_STREAM_ID.load(Ordering::Relaxed);
        if id == u64::MAX {
            break;
        }
        if NEXT_STREAM_ID.compare_exchange(
            id,
            id + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ).is_ok() {
            return Ok(id);
        }
        attempt += 1;
    }
    Err(CudaError(1))
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct CudaStream {
    pub(super) raw: ffi::RawStreamGuard,
    pub(super) device: DeviceId,
    runtime_id: u64,
    source: Tracked<EventSourceToken>,
    tail: Ghost<ISet<TransferEvent>>,
    tail_infos: Tracked<SubmittedMap>,
    receipts: Tracked<Seq<Submitted>>,
}

impl CudaStream {
    pub closed spec fn tail(&self) -> ISet<TransferEvent> {
        self.tail@
    }

    pub closed spec fn infos(&self) -> SubmittedMap {
        self.tail_infos@
    }

    pub closed spec fn source(&self) -> EventSourceToken {
        self.source@
    }

    pub closed spec fn device(&self) -> DeviceId {
        self.device
    }

    pub fn device_id(&self) -> (device: DeviceId)
        ensures
            device == self.device(),
    {
        self.device
    }

    /// Process-local identity that survives moves and CUDA handle reuse.
    pub fn runtime_id(&self) -> u64 {
        self.runtime_id
    }

    pub closed spec fn well_formed(&self) -> bool {
        &&& self.source@.well_formed() && self.source@.source().is_cuda_stream()
        &&& self.source@.id() == self.tail_infos@.instance_id()
        &&& self.tail@.subset_of(self.tail_infos@.dom())
        &&& valid_receipts(self.receipts@, self.tail_infos@)
    }

    /// Every event in `submitted` from this stream's source has sequence strictly below
    /// `self.source().next_seq()`.
    pub fn lemma_source_fresh(
        &self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(submitted): Tracked<&SubmittedMap>,
    )
        requires
            self.well_formed(),
            self.source().id() == instance.id(),
            submitted.instance_id() == instance.id(),
        ensures
            forall|e: TransferEvent| #[trigger]
                submitted.dom().contains(e) && e.source == self.source().source() && e.observer
                    == self.source().node() ==> e.seq < self.source().next_seq(),
    {
        proof {
            self.source.borrow().lemma_all_future_unsubmitted(instance, submitted);
        }
    }

    pub fn new(device: DeviceId, Tracked(source): Tracked<EventSourceToken>) -> (r: Result<
        Self,
        CudaError,
    >)
        requires
            source.well_formed(),
            source.source().is_cuda_stream(),
            source.next_seq() == 0,
        ensures
            r matches Ok(s) ==> s.well_formed() && s.source() == source && s.device() == device
                && s.tail() == ISet::empty() && s.infos().map() == IMap::empty(),
    {
        let runtime_id = allocate_stream_id()?;
        let raw = ffi::raw_stream_create(device)?;
        Ok(
            Self {
                raw,
                device,
                runtime_id,
                source: Tracked(source),
                tail: Ghost(ISet::empty()),
                tail_infos: Tracked(SubmittedMap::empty(source.id())),
                receipts: Tracked(Seq::tracked_empty()),
            },
        )
    }

    pub fn destroy(self) -> Result<(), CudaError> {
        ffi::raw_stream_destroy(self.device, self.raw)
    }
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct CudaEvent {
    pub(super) raw: ffi::RawEventGuard,
    pub(super) device: DeviceId,
    observer: Ghost<NodeId>,
    recorded: Ghost<ISet<TransferEvent>>,
    recorded_infos: Tracked<SubmittedMap>,
    receipts: Tracked<Seq<Submitted>>,
}

impl CudaEvent {
    pub closed spec fn observer(&self) -> NodeId {
        self.observer@
    }

    pub closed spec fn device(&self) -> DeviceId {
        self.device
    }

    pub closed spec fn recorded(&self) -> ISet<TransferEvent> {
        self.recorded@
    }

    pub closed spec fn infos(&self) -> SubmittedMap {
        self.recorded_infos@
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.recorded@.subset_of(self.recorded_infos@.dom()) && valid_receipts(
            self.receipts@,
            self.recorded_infos@,
        )
    }

    pub fn new(
        device: DeviceId,
        observer: Ghost<NodeId>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Self, CudaError>)
        ensures
            r matches Ok(e) ==> e.well_formed() && e.device() == device && e.observer() == observer@
                && e.infos().instance_id() == instance.id() && e.recorded() == ISet::empty()
                && e.infos().map() == IMap::empty(),
    {
        let raw = ffi::raw_event_create(device)?;
        Ok(
            Self {
                raw,
                device,
                observer,
                recorded: Ghost(ISet::empty()),
                recorded_infos: Tracked(SubmittedMap::empty(instance.id())),
                receipts: Tracked(Seq::tracked_empty()),
            },
        )
    }

    pub fn destroy(self) -> Result<(), CudaError> {
        ffi::raw_event_destroy(self.device, self.raw)
    }
}

pub fn event_record(event: &mut CudaEvent, stream: &CudaStream) -> (r: Result<(), CudaError>)
    requires
        old(event).well_formed(),
        stream.well_formed(),
        old(event).device() == stream.device(),
        old(event).observer() == stream.source().node(),
        old(event).infos().instance_id() == stream.source().id(),
    ensures
        match r {
            Err(_) => *final(event) == *old(event),
            Ok(()) => final(event).well_formed() && final(event).recorded() == stream.tail()
                && final(event).infos().map() == stream.infos().map() && final(event).device()
                == stream.device() && final(event).observer() == stream.source().node()
                && final(event).infos().instance_id() == stream.source().id(),
        },
{
    ffi::raw_event_record(event, stream)?;
    proof {
        let tracked mut receipts = Seq::tracked_empty();
        let tracked mut infos = SubmittedMap::empty(stream.source@.id());
        append_receipts(
            stream.receipts.borrow(),
            stream.receipts@.len(),
            &mut receipts,
            &mut infos,
        );
        assert(receipts =~= stream.receipts@);
        assert(infos.map() =~= stream.tail_infos@.map());
        event.recorded = stream.tail;
        event.recorded_infos = Tracked(infos);
        event.receipts = Tracked(receipts);
    }
    Ok(())
}

pub fn stream_wait_event(stream: &mut CudaStream, event: &CudaEvent) -> (r: Result<(), CudaError>)
    requires
        old(stream).well_formed(),
        event.well_formed(),
        event.observer() == old(stream).source().node(),
        event.infos().instance_id() == old(stream).source().id(),
    ensures
        match r {
            Err(_) => *final(stream) == *old(stream),
            Ok(()) => final(stream).well_formed() && final(stream).source() == old(stream).source()
                && final(stream).device() == old(stream).device() && final(stream).tail() == old(
                stream,
            ).tail().union(event.recorded()) && final(stream).infos().map() == old(
                stream,
            ).infos().map().union_prefer_right(event.infos().map()),
        },
{
    ffi::raw_stream_wait_event(stream, event)?;
    proof {
        let ghost before = stream.receipts@;
        append_receipts(
            event.receipts.borrow(),
            event.receipts@.len(),
            stream.receipts.borrow_mut(),
            stream.tail_infos.borrow_mut(),
        );
        append_receipt_values(before, event.receipts@, event.receipts@.len());
        assert(event.receipts@.take(event.receipts@.len() as int) =~= event.receipts@);
        assert forall|i: int|
            0 <= i < stream.receipts@.len() implies #[trigger] stream.receipts@[i].instance_id()
            == stream.source@.id() by {
            if i >= before.len() {
                assert(stream.receipts@[i] == event.receipts@[i - before.len()]);
            }
        }
        stream.tail = Ghost(stream.tail@.union(event.recorded@));
    }
    Ok(())
}

pub fn event_query(event: &CudaEvent, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
    EventStatus,
    CudaError,
>)
    requires
        event.well_formed(),
        event.infos().instance_id() == instance.id(),
    ensures
        r matches Ok(EventStatus::Ready(k)) ==> k@.instance_id() == instance.id() && k@.set()
            == knowledge_of(event.observer(), closure_from(event.infos().map(), event.recorded())),
{
    if let Some(observed) = ffi::raw_event_query(event)? {
        let Tracked(evidence) = observed;
        Ok(
            EventStatus::Ready(
                Tracked(
                    instance.observe_completed(
                        event.observer@,
                        event.recorded@,
                        event.recorded_infos.borrow(),
                        &evidence,
                    ),
                ),
            ),
        )
    } else {
        Ok(EventStatus::NotReady)
    }
}

pub fn stream_synchronize(stream: &CudaStream, Tracked(instance): Tracked<&TransferInstance>) -> (r:
    Result<Tracked<Knows>, CudaError>)
    requires
        stream.well_formed(),
        stream.source().id() == instance.id(),
    ensures
        r matches Ok(k) ==> k@.instance_id() == instance.id() && k@.set() == knowledge_of(
            stream.source().node(),
            closure_from(stream.infos().map(), stream.tail()),
        ),
{
    let Tracked(evidence) = ffi::raw_stream_synchronize(stream)?;
    Ok(
        Tracked(
            instance.observe_completed(
                stream.source@.node(),
                stream.tail@,
                stream.tail_infos.borrow(),
                &evidence,
            ),
        ),
    )
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub(super) tracked struct EnqueuedCopy {
    pub(super) ghost access: Access<TransferDomain>,
    pub(super) ghost effect: Effect<TransferDomain>,
}

pub open spec fn copy_admitted(
    stream: CudaStream,
    src_perm: ReadPerm,
    dst_tok: SlotToken,
    cited_infos: SubmittedMap,
    knows: Knows,
    cited: ISet<TransferEvent>,
) -> bool {
    let id = stream.source().id();
    let merged = stream.infos().map().union_prefer_right(cited_infos.map());
    let waits = stream.tail().union(cited);
    let ordered = closure_from(merged, waits);
    &&& cited_infos.instance_id() == id
    &&& knows.instance_id() == id
    &&& waits.subset_of(merged.dom())
    &&& knowledge_of(stream.source().node(), cited).subset_of(knows.set())
    &&& read_admitted(src_perm.frontier(), ordered)
    &&& write_admitted(slot_frontier(dst_tok), ordered)
}

pub fn memcpy_async(
    stream: &mut CudaStream,
    Tracked(instance): Tracked<&TransferInstance>,
    dst: &CudaSlice,
    src: &CudaSlice,
    Tracked(src_perm): Tracked<&mut ReadPerm>,
    Tracked(dst_tok): Tracked<&mut SlotToken>,
    Tracked(cited_infos): Tracked<&mut SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<(Ghost<TransferEvent>, Tracked<Submitted>), CudaError>)
    requires
        old(stream).well_formed(),
        old(stream).source().id() == instance.id(),
        dst.well_formed(),
        src.well_formed(),
        dst.instance_id() == instance.id(),
        src.instance_id() == instance.id(),
        src.slot_ref().sub_len == dst.slot_ref().sub_len,
        src.slot_ref().loc.node == old(stream).source().node(),
        dst.slot_ref().loc.node == src.slot_ref().loc.node,
        src.slot_ref().loc != dst.slot_ref().loc,
        old(src_perm).valid_cooperative(instance.id(), src.slot_ref().loc),
        slot_is_owned(*old(dst_tok), instance.id(), dst.slot_ref().loc),
        src.slot_ref().loc.space.is_gpu() && dst.slot_ref().loc.space.is_gpu()
            && src.slot_ref().loc.space.gpu_device() != dst.slot_ref().loc.space.gpu_device() ==> (
        old(src_perm).frontier().last_writer matches Some(w) ==> !(w.source.is_rdma_qp())),
        copy_admitted(
            *old(stream),
            *old(src_perm),
            *old(dst_tok),
            *old(cited_infos),
            *knows,
            cited,
        ),
    ensures
        final(cited_infos).map() == old(cited_infos).map(),
        final(cited_infos).instance_id() == old(cited_infos).instance_id(),
        r.is_err() ==> *final(cited_infos) == *old(cited_infos),
        match r {
            Err(_) => *final(stream) == *old(stream) && *final(src_perm) == *old(src_perm)
                && *final(dst_tok) == *old(dst_tok),
            Ok((name, receipt)) => {
                let ev = old(stream).source().next_event();
                let access = copy_access(src.slot_ref(), dst.slot_ref());
                &&& final(stream).well_formed() && final(stream).device() == old(stream).device()
                &&& final(stream).source().source() == old(stream).source().source()
                &&& final(stream).source().node() == old(stream).source().node()
                    && final(stream).source().id() == instance.id()
                &&& final(stream).source().next_seq() == old(stream).source().next_seq() + 1
                &&& name@ == ev && receipt@.key() == ev && receipt@.instance_id() == instance.id()
                &&& receipt@.value() == (EventInfo {
                    access,
                    read_guards: old(src_perm).read_guards(),
                    effect: copy_effect(src.slot_ref(), dst.slot_ref()),
                    preds: closure_from(
                        old(stream).infos().map().union_prefer_right(old(cited_infos).map()),
                        old(stream).tail().union(cited),
                    ),
                })
                &&& final(stream).tail() == ISet::empty().insert(ev)
                &&& final(stream).infos().map() == IMap::empty().insert(ev, receipt@.value())
                &&& final(src_perm).advanced_read(*old(src_perm), ev)
                &&& slot_owned(
                    *final(dst_tok),
                    instance.id(),
                    dst.slot_ref().loc,
                    Frontier { last_writer: Some(ev), readers: ISet::empty() },
                )
            },
        },
{
    assert!(src.memory.len == dst.memory.len);
    match (src.memory.kind, dst.memory.kind) {
        (CudaPointerKind::Device(a), CudaPointerKind::Device(b)) if stream.device == a
            || stream.device == b => {},
        (CudaPointerKind::Device(a), CudaPointerKind::PinnedHost) if stream.device == a => {},
        (CudaPointerKind::PinnedHost, CudaPointerKind::Device(a)) if stream.device == a => {},
        _ => return Err(CudaError::INVALID_VALUE),
    }
    let Tracked(copy) = ffi::raw_memcpy_async(stream, dst, src)?;
    let ghost ev = stream.source@.next_event();
    let tracked receipt;
    proof {
        // Borrow both maps for submission. Every original citation is put
        // back after the step, including keys that also occur in the tail.
        let tracked mut tail = SubmittedMap::empty(instance.id());
        let tracked mut citations = SubmittedMap::empty(instance.id());
        vstd::modes::tracked_swap(stream.tail_infos.borrow_mut(), &mut tail);
        vstd::modes::tracked_swap(cited_infos, &mut citations);
        let ghost keys = citations.dom();
        let tracked mut entries = tail.into_map();
        entries.tracked_union_prefer_right(citations.into_map());
        let tracked infos = SubmittedMap::from_map(instance.id(), entries);
        assert(infos.map() =~= old(stream).infos().map().union_prefer_right(
            old(cited_infos).map(),
        ));
        let tracked name = stream.source.borrow_mut().take_next();
        let tracked submitted = instance.submit_2(
            ev.observer,
            copy.access,
            copy.effect,
            stream.tail@,
            cited,
            &infos,
            knows,
            name,
            src_perm,
            dst_tok,
        );
        let tracked mut entries = infos.into_map();
        *cited_infos = SubmittedMap::from_map(instance.id(), entries.tracked_remove_keys(keys));
        assert(cited_infos.map() =~= old(cited_infos).map());
        receipt = submitted;
        let tracked mut tail_infos = SubmittedMap::empty(instance.id());
        tail_infos.insert(submitted);
        let tracked mut receipts = Seq::tracked_empty();
        receipts.tracked_push(submitted);
        stream.receipts = Tracked(receipts);
        stream.tail_infos = Tracked(tail_infos);
        stream.tail = Ghost(ISet::empty().insert(ev));
        reveal_with_fuel(receipt_values, 2);
        assert(valid_receipts(stream.receipts@, stream.tail_infos@));
        assert(stream.well_formed());
    }
    Ok((Ghost(ev), Tracked(receipt)))
}

} // verus!
