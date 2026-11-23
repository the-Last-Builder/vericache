use super::*;
#[cfg(verus_only)]
use foundation::tracked::closure_from;
use transfer::cuda::{self, CudaEvent, CudaStream, EventStatus};

verus! {

/// One slot in the bounded CUDA event ring.
pub struct EventSlot {
    pub event: CudaEvent,
    pub generation: u64,
    pub recorded_seq: u64,
    pub in_use: bool,
}

impl EventSlot {
    pub open spec fn valid(
        &self,
        table: &TimelineTable,
        t: u8,
        device: DeviceId,
        node: NodeId,
        ledger: &KnowledgeLedger,
    ) -> bool {
        &&& self.event.well_formed()
        &&& self.event.device() == device
        &&& self.event.observer() == node
        &&& self.event.infos().instance_id() == ledger.submitted.instance_id()
        &&& self.event.infos().map().submap_of(ledger.submitted.map())
        &&& self.event.recorded().subset_of(self.event.infos().dom())
        &&& self.recorded_seq < table.next_seq[t as int]
        &&& table.prefix_in(
            t,
            self.recorded_seq,
            closure_from(self.event.infos().map(), self.event.recorded()),
        )
        &&& !self.in_use ==> self.recorded_seq <= table.confirmed_seq[t as int]
    }

    pub fn new(event: CudaEvent) -> (slot: Self)
        ensures
            slot.event == event,
            slot.generation == 0,
            slot.recorded_seq == 0,
            !slot.in_use,
    {
        Self { event, generation: 0, recorded_seq: 0, in_use: false }
    }
}

#[verifier::opaque]
pub open spec fn event_ring_valid(
    ring: Seq<EventSlot>,
    table: &TimelineTable,
    t: u8,
    device: DeviceId,
    node: NodeId,
    ledger: &KnowledgeLedger,
) -> bool {
    forall|i: int| 0 <= i < ring.len() ==> #[trigger] ring[i].valid(table, t, device, node, ledger)
}

#[verifier::opaque]
pub open spec fn ring_latest_covers(ring: Seq<EventSlot>, newest: int) -> bool {
    0 <= newest < ring.len() && forall|i: int|
        0 <= i < ring.len() ==> (#[trigger] ring[i]).recorded_seq <= ring[newest].recorded_seq
}

/// Ordered CUDA stream timeline backed by a bounded `CudaEvent` ring.
pub struct CudaTimeline {
    pub timeline_id: u8,
    pub device: DeviceId,
    pub stream: CudaStream,
    pub event_ring: Vec<EventSlot>,
    pub ring_head: usize,
    pub ring_tail: usize,
}

impl CudaTimeline {
    pub open spec fn newest(&self) -> int {
        if self.ring_tail == 0 {
            self.event_ring.len() - 1
        } else {
            self.ring_tail - 1
        }
    }

    pub open spec fn closure(&self) -> ISet<TransferEvent> {
        closure_from(self.stream.infos().map(), self.stream.tail())
    }

    pub open spec fn well_formed(
        &self,
        table: &TimelineTable,
        node: NodeId,
        ledger: &KnowledgeLedger,
    ) -> bool {
        let t = self.timeline_id;
        &&& t < table.count && table.count <= MAX_TIMELINES
        &&& table.kinds[t as int] == (TimelineKind::CudaStream { device: self.device })
        &&& source_fresh_for_ledger(self.stream.source(), *ledger)
        &&& self.stream.source().source() is CudaStream
        &&& self.stream.well_formed()
        &&& self.stream.device() == self.device
        &&& self.stream.source().node() == node
        &&& self.stream.source().id() == ledger.submitted.instance_id()
        &&& self.stream.infos().map().submap_of(ledger.submitted.map())
        &&& self.stream.tail().subset_of(self.stream.infos().dom())
        &&& table.prefix_in(t, (table.next_seq[t as int] - 1) as u64, self.closure())
        &&& self.ring_head < self.event_ring.len() <= u32::MAX
        &&& self.ring_tail < self.event_ring.len()
        &&& ring_latest_covers(self.event_ring@, self.newest())
        &&& event_ring_valid(self.event_ring@, table, t, self.device, node, ledger)
    }

    fn next_slot(i: usize, len: usize) -> (r: usize)
        requires
            i < len,
        ensures
            r < len,
            r == if i + 1 == len {
                0
            } else {
                i + 1
            },
    {
        if i + 1 == len {
            0
        } else {
            i + 1
        }
    }

    fn destroy_slots(mut unused: Vec<EventSlot>) -> (err: Option<CoreError>) {
        let mut error: Option<CoreError> = None;
        while unused.len() > 0
            decreases unused.len(),
        {
            let slot = unused.pop().unwrap();
            if let Err(e) = slot.event.destroy() {
                error = Some(cuda_error(e));
            }
        }
        error
    }

    /// Allocate a CUDA stream and bounded event pool without touching `core`.
    pub fn create_hardware(
        Tracked(instance): Tracked<&TransferInstance>,
        device: DeviceId,
        Ghost(node): Ghost<NodeId>,
        Tracked(stream_source): Tracked<EventSourceToken>,
        ring_len: usize,
    ) -> (r: BackendResult<(CudaStream, Vec<EventSlot>)>)
        requires
            stream_source.well_formed(),
            stream_source.id() == instance.id(),
            stream_source.node() == node,
            stream_source.source() is CudaStream,
            stream_source.next_seq() == 0,
            0 < ring_len <= u32::MAX,
        ensures
            r matches Ok((stream, slots)) ==> {
                &&& stream.well_formed()
                &&& stream.source() == stream_source
                &&& stream.device() == device
                &&& stream.tail() == ISet::empty()
                &&& stream.infos().map() == IMap::empty()
                &&& slots.len() == ring_len
                &&& forall|j: int|
                    0 <= j < ring_len ==> {
                        let s = #[trigger] slots[j];
                        &&& s.event.well_formed()
                        &&& s.event.device() == device
                        &&& s.event.observer() == node
                        &&& s.event.infos().instance_id() == instance.id()
                        &&& s.event.recorded() == ISet::empty()
                        &&& s.event.infos().map() == IMap::empty()
                        &&& !s.in_use
                        &&& s.recorded_seq == 0
                    }
            },
    {
        let stream = match CudaStream::new(device, Tracked(stream_source)) {
            Ok(s) => s,
            Err(e) => return Err(cuda_error(e)),
        };
        let mut slots: Vec<EventSlot> = Vec::new();
        let mut i: usize = 0;
        while i < ring_len
            invariant
                i <= ring_len,
                slots.len() == i,
                forall|j: int|
                    0 <= j < i ==> {
                        let s = #[trigger] slots[j];
                        &&& s.event.well_formed()
                        &&& s.event.device() == device
                        &&& s.event.observer() == node
                        &&& s.event.infos().instance_id() == instance.id()
                        &&& s.event.recorded() == ISet::empty()
                        &&& s.event.infos().map() == IMap::empty()
                        &&& !s.in_use
                        &&& s.recorded_seq == 0
                    },
            decreases ring_len - i,
        {
            match CudaEvent::new(device, Ghost(node), Tracked(instance)) {
                Ok(ev) => {
                    slots.push(EventSlot::new(ev));
                    i += 1;
                },
                Err(e) => {
                    let _ = Self::destroy_slots(slots);
                    let _ = stream.destroy();
                    return Err(cuda_error(e));
                },
            }
        }
        Ok((stream, slots))
    }

    pub fn destroy_hardware(stream: CudaStream, slots: Vec<EventSlot>) {
        let _ = Self::destroy_slots(slots);
        let _ = stream.destroy();
    }

    /// Register pre-allocated CUDA stream + event pool into `core`.
    pub fn register(
        core: &mut CacheCore,
        device: DeviceId,
        stream: CudaStream,
        slots: Vec<EventSlot>,
    ) -> (tl: Self)
        requires
            old(core).well_formed(),
            old(core).timelines.count < MAX_TIMELINES,
            stream.well_formed(),
            stream.device() == device,
            stream.source().source() is CudaStream,
            stream.source().id() == old(core).instance_id(),
            stream.source().node() == old(core).node(),
            source_fresh_for_ledger(stream.source(), old(core).ledger@),
            stream.tail() == ISet::empty(),
            stream.infos().map() == IMap::empty(),
            0 < slots.len() <= u32::MAX,
            forall|j: int|
                0 <= j < slots.len() ==> {
                    let s = #[trigger] slots[j];
                    &&& s.event.well_formed()
                    &&& s.event.device() == device
                    &&& s.event.observer() == old(core).node()
                    &&& s.event.infos().instance_id() == old(core).instance_id()
                    &&& s.event.recorded() == ISet::empty()
                    &&& s.event.infos().map() == IMap::empty()
                    &&& !s.in_use
                    &&& s.recorded_seq == 0
                },
        ensures
            final(core).well_formed(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).ledger == old(core).ledger,
            final(core).spec == old(core).spec,
            tl.well_formed(&final(core).timelines, final(core).node(), &final(core).ledger@),
            tl.timeline_id == old(core).timelines.count,
            tl.device == device,
            tl.event_ring.len() == slots.len(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost node = core.node();
        let timeline_id = core.register_timeline(TimelineKind::CudaStream { device }).unwrap();
        let tl = Self {
            timeline_id,
            device,
            stream,
            event_ring: slots,
            ring_head: 0,
            ring_tail: 0,
        };
        proof {
            lemma_cuda_timeline_init_wf(
                &tl.event_ring@,
                &core.timelines,
                timeline_id,
                device,
                node,
                &core.ledger@,
                tl.newest(),
            );
            lemma_empty_imap_submap(core.ledger@.submitted.map());
        }
        tl
    }

    /// Create a CUDA stream and bounded event pool and register its timeline in `core`.
    pub fn new(
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        device: DeviceId,
        Tracked(stream_source): Tracked<EventSourceToken>,
        ring_len: usize,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            stream_source.well_formed(),
            stream_source.id() == old(core).instance_id(),
            stream_source.node() == old(core).node(),
            stream_source.source() is CudaStream,
            stream_source.next_seq() == 0,
            source_fresh_for_ledger(stream_source, old(core).ledger@),
        ensures
            final(core).well_formed(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).ledger == old(core).ledger,
            final(core).spec == old(core).spec,
            match r {
                Err(_) => *final(core) == *old(core),
                Ok(tl) => {
                    &&& tl.well_formed(
                        &final(core).timelines,
                        final(core).node(),
                        &final(core).ledger@,
                    )
                    &&& tl.timeline_id == old(core).timelines.count
                    &&& tl.device == device
                    &&& tl.event_ring.len() == ring_len
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_core_wf_facts(core);
        }
        if ring_len == 0 || ring_len > u32::MAX as usize {
            return Err(CoreError::InvalidArgument);
        }
        if core.timelines.count >= MAX_TIMELINES {
            return Err(CoreError::CapacityExhausted);
        }
        let (stream, slots) = Self::create_hardware(
            Tracked(instance),
            device,
            Ghost(core.node()),
            Tracked(stream_source),
            ring_len,
        )?;
        Ok(Self::register(core, device, stream, slots))
    }

    /// Record a CUDA event at the tail of this timeline's stream, returning a
    /// `SyncPoint::TimelineEvent` handle with generation guard.
    pub fn record_event(&mut self, core: &CacheCore) -> (r: BackendResult<SyncPoint>)
        requires
            core.well_formed(),
            old(self).well_formed(&core.timelines, core.node(), &core.ledger@),
        ensures
            final(self).well_formed(&core.timelines, core.node(), &core.ledger@),
            final(self).stream == old(self).stream,
            final(self).timeline_id == old(self).timeline_id,
            final(self).device == old(self).device,
            final(self).event_ring.len() == old(self).event_ring.len(),
            match r {
                Err(_) => final(self).event_ring@ == old(self).event_ring@ && final(self).ring_head
                    == old(self).ring_head && final(self).ring_tail == old(self).ring_tail,
                Ok(point) => point matches SyncPoint::TimelineEvent {
                    timeline,
                    event_slot,
                    generation,
                } && {
                    let slot = final(self).event_ring[event_slot as int];
                    &&& timeline == final(self).timeline_id
                    &&& (event_slot as usize) < final(self).event_ring.len()
                    &&& generation != 0
                    &&& slot.in_use
                    &&& slot.generation == generation
                    &&& slot.recorded_seq == core.timelines.next_seq[timeline as int] - 1
                    &&& slot.event.recorded() == final(self).stream.tail()
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_ring = self.event_ring@;
        let i = self.ring_tail;
        proof {
            lemma_core_wf_facts(core);
            lemma_event_ring_index(
                before_ring,
                &core.timelines,
                self.timeline_id,
                self.device,
                core.node(),
                &core.ledger@,
                i as int,
            );
        }
        if self.event_ring[i].in_use {
            return Err(CoreError::CapacityExhausted);
        }
        if self.event_ring[i].generation == u64::MAX {
            return Err(CoreError::SequenceExhausted);
        }
        let slot = &mut self.event_ring[i];
        if let Err(e) = cuda::event_record(&mut slot.event, &self.stream) {
            proof {
                assert(self.event_ring[i as int] == before_ring[i as int]);
                assert(self.event_ring@ =~= before_ring);
            }
            return Err(cuda_error(e));
        }
        slot.generation += 1;
        slot.recorded_seq = core.timelines.next_seq[self.timeline_id as usize] - 1;
        slot.in_use = true;
        let generation = slot.generation;
        self.ring_tail = Self::next_slot(i, self.event_ring.len());
        proof {
            assert(self.newest() == i as int);
            lemma_event_ring_after_record(
                before_ring,
                self.event_ring@,
                &core.timelines,
                self.timeline_id,
                self.device,
                core.node(),
                &core.ledger@,
                i as int,
            );
        }
        Ok(
            SyncPoint::TimelineEvent {
                timeline: self.timeline_id,
                event_slot: i as u32,
                generation,
            },
        )
    }

    /// M2 readiness check: returns `true` ONLY when `point` is `HostComplete`,
    /// or when its CUDA event slot has matching non-zero generation, is NOT
    /// still `in_use`/recycled, and its `recorded_seq` is confirmed on `core`.
    pub fn is_sync_point_ready(&self, core: &CacheCore, point: SyncPoint) -> (ready: bool)
        requires
            self.timeline_id < core.timelines.count <= MAX_TIMELINES,
        ensures
            ready == match point {
                SyncPoint::HostComplete => true,
                SyncPoint::TimelineEvent { timeline, event_slot, generation } => timeline
                    == self.timeline_id && (event_slot as usize) < self.event_ring.len()
                    && generation != 0 && !self.event_ring[event_slot as int].in_use
                    && self.event_ring[event_slot as int].generation == generation
                    && self.event_ring[event_slot as int].recorded_seq
                    <= core.timelines.confirmed_seq[timeline as int],
            },
    {
        match point {
            SyncPoint::HostComplete => true,
            SyncPoint::TimelineEvent { timeline, event_slot, generation } => timeline
                == self.timeline_id && (event_slot as usize) < self.event_ring.len() && generation
                != 0 && !self.event_ring[event_slot as usize].in_use
                && self.event_ring[event_slot as usize].generation == generation
                && self.event_ring[event_slot as usize].recorded_seq
                <= core.timelines.confirmed_seq[timeline as usize],
        }
    }

    /// Ordered Handoff: if `point` references a live in-use event on this
    /// timeline, enqueue `cuda::stream_wait_event` on `dst_stream` without
    /// blocking the host CPU. If already ready, this is a no-op.
    pub fn wait_sync_point(
        &self,
        dst_stream: &mut CudaStream,
        core: &CacheCore,
        point: SyncPoint,
    ) -> (r: BackendResult<()>)
        requires
            core.well_formed(),
            self.well_formed(&core.timelines, core.node(), &core.ledger@),
            old(dst_stream).well_formed(),
            old(dst_stream).source().node() == core.node(),
            old(dst_stream).source().id() == core.instance_id(),
            old(dst_stream).infos().map().submap_of(core.ledger@.submitted.map()),
            old(dst_stream).tail().subset_of(old(dst_stream).infos().dom()),
        ensures
            final(dst_stream).well_formed(),
            final(dst_stream).source() == old(dst_stream).source(),
            final(dst_stream).device() == old(dst_stream).device(),
            final(dst_stream).infos().map().submap_of(core.ledger@.submitted.map()),
            final(dst_stream).tail().subset_of(final(dst_stream).infos().dom()),
            old(dst_stream).tail().subset_of(final(dst_stream).tail()),
            r.is_err() ==> *final(dst_stream) == *old(dst_stream),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_core_wf_facts(core);
        }
        if self.is_sync_point_ready(core, point) {
            return Ok(());
        }
        match point {
            SyncPoint::HostComplete => Ok(()),
            SyncPoint::TimelineEvent { timeline, event_slot, generation } => {
                if timeline != self.timeline_id || (event_slot as usize) >= self.event_ring.len() {
                    return Err(CoreError::WouldConflict);
                }
                let idx = event_slot as usize;
                proof {
                    lemma_event_ring_index(
                        self.event_ring@,
                        &core.timelines,
                        self.timeline_id,
                        self.device,
                        core.node(),
                        &core.ledger@,
                        idx as int,
                    );
                }
                let slot = &self.event_ring[idx];
                if !slot.in_use || slot.generation != generation {
                    return Err(CoreError::WouldConflict);
                }
                let ghost before_stream = *dst_stream;
                match cuda::stream_wait_event(dst_stream, &slot.event) {
                    Ok(()) => {
                        proof {
                            lemma_stream_wait_preserves_submap(
                                before_stream.infos().map(),
                                slot.event.infos().map(),
                                core.ledger@.submitted.map(),
                                before_stream.tail(),
                                slot.event.recorded(),
                            );
                        }
                        Ok(())
                    },
                    Err(e) => Err(cuda_error(e)),
                }
            },
        }
    }

    /// Poll completed `CudaEvent`s from the ring head, advancing `core`'s
    /// timeline confirmation watermark and recording `Knows` into `core.ledger`.
    pub fn poll_events(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<usize, (CoreError, usize)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(&old(core).timelines, old(core).node(), &old(core).ledger@),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(
                &final(core).timelines,
                final(core).node(),
                &final(core).ledger@,
            ),
            final(self).stream == old(self).stream,
            final(self).timeline_id == old(self).timeline_id,
            final(self).device == old(self).device,
            final(self).event_ring.len() == old(self).event_ring.len(),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).version_seq_start == old(core).version_seq_start,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).free_lease_slots == old(core).free_lease_slots,
            final(core).lease_generation == old(core).lease_generation,
            final(core).version_registry == old(core).version_registry,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.events == old(core).timelines.events,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            final(core).timelines.confirmed_seq[old(self).timeline_id as int] >= old(
                core,
            ).timelines.confirmed_seq[old(self).timeline_id as int],
            forall|u: u8|
                u < old(core).timelines.count && u != old(self).timeline_id
                    ==> #[trigger] final(core).timelines.confirmed_seq[u as int] == old(
                    core,
                ).timelines.confirmed_seq[u as int],
            final(core).ledger@.submitted == old(core).ledger@.submitted,
            final(core).ledger@.succeeded == old(core).ledger@.succeeded,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            final(core).kernel_view().extends(old(core).kernel_view()),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let t = self.timeline_id;
        let ghost node = core.node();
        let mut polled: usize = 0;
        let ring_len = self.event_ring.len();
        proof {
            lemma_cuda_extends_refl(core);
        }
        while polled < ring_len
            invariant
                polled <= ring_len,
                ring_len == self.event_ring.len() == old(self).event_ring.len(),
                t == self.timeline_id == old(self).timeline_id,
                self.device == old(self).device,
                self.stream == old(self).stream,
                core.well_formed(),
                self.well_formed(&core.timelines, core.node(), &core.ledger@),
                instance.id() == core.instance_id(),
                node == core.node(),
                core.node_id == old(core).node_id,
                core.tiers == old(core).tiers,
                core.frames == old(core).frames,
                core.catalog == old(core).catalog,
                core.spec == old(core).spec,
                core.version_seq_start == old(core).version_seq_start,
                core.next_version_seq == old(core).next_version_seq,
                core.version_seq_limit == old(core).version_seq_limit,
                core.active_leases == old(core).active_leases,
                core.free_lease_slots == old(core).free_lease_slots,
                core.lease_generation == old(core).lease_generation,
                core.version_registry == old(core).version_registry,
                core.timelines.count == old(core).timelines.count,
                core.timelines.kinds == old(core).timelines.kinds,
                core.timelines.events == old(core).timelines.events,
                core.timelines.next_seq == old(core).timelines.next_seq,
                core.timelines.confirmed_seq[t as int] >= old(
                    core,
                ).timelines.confirmed_seq[t as int],
                forall|u: u8|
                    u < old(core).timelines.count && u != t
                        ==> #[trigger] core.timelines.confirmed_seq[u as int] == old(
                        core,
                    ).timelines.confirmed_seq[u as int],
                core.ledger@.submitted == old(core).ledger@.submitted,
                core.ledger@.succeeded == old(core).ledger@.succeeded,
                core.ledger@.revoked == old(core).ledger@.revoked,
                old(core).ledger@.knows.set().subset_of(core.ledger@.knows.set()),
                core.kernel_view().extends(old(core).kernel_view()),
            decreases ring_len - polled,
        {
            let i = self.ring_head;
            if !self.event_ring[i].in_use {
                break;
            }
            let ghost before_core = *core;
            let ghost before_ring = self.event_ring@;
            let ghost before_newest = self.newest();
            proof {
                lemma_core_wf_facts(core);
                lemma_event_ring_index(
                    before_ring,
                    &core.timelines,
                    t,
                    self.device,
                    node,
                    &core.ledger@,
                    i as int,
                );
            }
            let status = match cuda::event_query(&self.event_ring[i].event, Tracked(instance)) {
                Err(e) => return Err((cuda_error(e), polled)),
                Ok(s) => s,
            };
            let Tracked(k) = match status {
                EventStatus::NotReady => break,
                EventStatus::Ready(w) => w,
            };
            let cur_confirmed = core.timelines.confirmed_seq[t as usize];
            let rec_seq = self.event_ring[i].recorded_seq;
            let prefix = if rec_seq > cur_confirmed {
                rec_seq
            } else {
                cur_confirmed
            };
            proof {
                lemma_event_ready_observes_prefix(
                    &before_ring[i as int],
                    &core.timelines,
                    t,
                    self.device,
                    node,
                    &core.ledger@,
                    k,
                    prefix,
                );
            }
            core.mark_timeline_complete(t, prefix, Tracked(k));
            let slot = &mut self.event_ring[i];
            slot.in_use = false;
            self.ring_head = Self::next_slot(i, ring_len);
            proof {
                lemma_event_ring_after_poll(
                    before_ring,
                    self.event_ring@,
                    before_newest,
                    &before_core.timelines,
                    &core.timelines,
                    t,
                    self.device,
                    node,
                    &before_core.ledger@,
                    &core.ledger@,
                    i as int,
                );
                lemma_cuda_extends_trans(
                    old(core).kernel_view(),
                    before_core.kernel_view(),
                    core.kernel_view(),
                    old(core).ledger@.knows.set(),
                    before_core.ledger@.knows.set(),
                    core.ledger@.knows.set(),
                );
            }
            polled += 1;
        }
        match self.record_uncovered_tail(core) {
            Ok(_) => Ok(polled),
            Err(e) => Err((e, polled)),
        }
    }

    /// Retry recording a CUDA event for any issued-but-uncovered stream tail.
    pub fn record_uncovered_tail(&mut self, core: &CacheCore) -> (r: BackendResult<
        Option<SyncPoint>,
    >)
        requires
            core.well_formed(),
            old(self).well_formed(&core.timelines, core.node(), &core.ledger@),
        ensures
            final(self).well_formed(&core.timelines, core.node(), &core.ledger@),
            final(self).stream == old(self).stream,
            final(self).timeline_id == old(self).timeline_id,
            final(self).device == old(self).device,
            final(self).event_ring.len() == old(self).event_ring.len(),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_core_wf_facts(core);
        }
        let t = self.timeline_id as usize;
        let last = core.timelines.next_seq[t] - 1;
        if last <= core.timelines.confirmed_seq[t] {
            return Ok(None);
        }
        let newest = if self.ring_tail == 0 {
            self.event_ring.len() - 1
        } else {
            self.ring_tail - 1
        };
        if self.event_ring[newest].recorded_seq >= last || self.event_ring[self.ring_tail].in_use {
            return Ok(None);
        }
        Ok(Some(self.record_event(core)?))
    }
}

} // verus!
