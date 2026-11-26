use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
#[cfg(verus_only)]
use transfer::cuda::CudaPointerKind;
use transfer::cuda::{self, CudaBuffer, CudaError, CudaHostRegistration, CudaSlice};

mod copy;
mod copy_park;
mod copy_sync;
mod device;
mod get;
mod get_scatter;
#[cfg(verus_only)]
mod proofs;
mod put;
mod put_park;
#[cfg(verus_only)]
mod put_proofs;
mod put_step;
mod put_sync;
mod scatter;
mod scatter_ops;
#[cfg(verus_only)]
mod scatter_proofs;
#[cfg(verus_only)]
mod scatter_step;
pub mod timeline;
mod timeline_ops;
#[cfg(verus_only)]
mod timeline_proofs;

#[cfg(verus_only)]
pub use copy::caller_stream_valid;
#[cfg(verus_only)]
use proofs::*;
#[cfg(verus_only)]
pub use proofs::{
    lemma_cuda_puts_bindings_empty, lemma_cuda_puts_bindings_extract,
    lemma_cuda_puts_bindings_push_abort, lemma_cuda_puts_bindings_push_publish,
};
#[cfg(verus_only)]
pub use put::scatter_offsets_valid;
#[cfg(verus_only)]
use put_park::{cuda_backend_pre_park_wf, put_gpu_committed_ready};
#[cfg(verus_only)]
use put_proofs::*;
#[cfg(verus_only)]
use put_step::put_gpu_enqueue_ready;
#[cfg(verus_only)]
use scatter_proofs::*;
#[cfg(verus_only)]
use scatter_step::*;
pub use timeline::{CudaTimeline, EventSlot};
#[cfg(verus_only)]
use timeline_proofs::*;

#[cfg(verus_only)]
mod preserve;
#[cfg(verus_only)]
pub use preserve::*;

verus! {

pub fn cuda_error(e: CudaError) -> (r: CoreError)
    ensures
        r == (CoreError::Backend { kind: BACKEND_CUDA, code: e.0 }),
{
    CoreError::Backend { kind: BACKEND_CUDA, code: e.0 }
}

pub type CudaPollReport = DevicePollReport;

/// Action taken once an asynchronous GPU write's ticket retires.
pub enum PendingGpuAction {
    Publish {
        key: ObjectKey,
        version: VersionId,
        proof: Ghost<PendingPutProof>,
        only_if_current: bool,
    },
    Abort,
}

impl PendingGpuAction {
    pub open(crate) spec fn publishes_version(self, v: VersionId) -> bool {
        self matches Self::Publish { version, .. } && version == v
    }
}

/// Existential witness that one event `ev` appears in `timeline`'s prefix `[1, s]`
/// and succeeds on observation.
pub open spec fn cuda_event_at_prefix(
    ev: TransferEvent,
    ev_map: Map<(TimelineId, u64), TransferEvent>,
    timeline: TimelineId,
    s: u64,
) -> bool {
    exists|q: u64|
        1 <= q <= s && (#[trigger] ev_map[(timeline, q)]) == ev && ev.source.succeeds_on_observe()
}

/// Opaque coverage predicate linking a `PendingPutProof`'s write events to a
/// prefix `[1, s]` of `timeline` (opaque to avoid CVC5 `forall ... exists` blowup).
#[verifier::opaque]
pub open spec fn cuda_put_events_covered(
    events: Seq<TransferEvent>,
    ev_map: Map<(TimelineId, u64), TransferEvent>,
    timeline: TimelineId,
    s: u64,
) -> bool {
    forall|p: int|
        0 <= p < events.len() ==> cuda_event_at_prefix(#[trigger] events[p], ev_map, timeline, s)
}

/// One in-flight asynchronous CUDA write parked in `CacheCore`.
pub struct PendingGpuWrite {
    pub frame: FrameId,
    pub tier: TierId,
    pub slot: Ghost<SlotRef>,
    pub ticket: Ticket,
    pub sync_point: SyncPoint,
    pub action: PendingGpuAction,
}

impl PendingGpuWrite {
    pub open(crate) spec fn valid(&self, core: &CacheCore, timeline: TimelineId) -> bool {
        let s = self.ticket.sequence();
        &&& self.ticket != Ticket::NONE
        &&& self.ticket.timeline_id() == timeline
        &&& core.timelines_spec().issued(timeline, s)
        &&& self.frame.0 < core.frames_spec().metas_spec().len()
        &&& core.frames_spec().meta(self.frame).state == STATE_INITIALIZING
        &&& !core.frames_spec().meta(self.frame).queued()
        &&& core.frames_spec().meta(self.frame).last_write_ticket == self.ticket
        &&& core.frames_spec().frame(self.frame).tier == self.tier
        &&& core.frames_spec().frame(self.frame).slot == self.slot@
        &&& core.frames_spec().frame(self.frame).token matches Some(tok) && {
            &&& slot_is_owned(tok, core.instance_id(), self.slot@.loc)
            &&& slot_frontier(tok).readers.subset_of(core.citations())
            &&& match self.action {
                PendingGpuAction::Abort => slot_frontier(tok).last_writer matches Some(w) ==> (
                core.citations().contains(w) || core.timelines_spec().ticket_covers(
                    self.ticket,
                    w,
                )),
                PendingGpuAction::Publish { key, version, proof, only_if_current } => {
                    let cert = proof@.certificate;
                    &&& core.frames_spec().frame(self.frame).bound_key == Some(key)
                    &&& core.frames_spec().frame(self.frame).bound_version == Some(version)
                    &&& (version.origin_node != core.node_id_spec() || version.seq
                        < core.next_version_seq_spec() || version.seq
                        >= core.version_seq_limit_spec())
                    &&& (only_if_current ==> core.frames_spec().versions_spec()@.contains_key(
                        version,
                    ))
                    &&& (core.frames_spec().versions_spec()@.contains_key(version)
                        ==> core.frames_spec().versions_spec()@[version] == cert.bytes)
                    &&& cert.slot == self.slot@
                    &&& cert.bytes.len() == self.slot@.slot_len
                    &&& cert.writer == core.timelines_spec().events_spec()@[(timeline, s)]
                    &&& slot_frontier(tok).last_writer == Some(cert.writer)
                    &&& slot_frontier(tok).readers.is_empty()
                    &&& proof@.pending(core.ledger_spec()@, core.node())
                    &&& cuda_put_events_covered(
                        proof@.events,
                        core.timelines_spec().events_spec()@,
                        timeline,
                        s,
                    )
                },
            }
        }
    }
}

#[verifier::opaque]
pub open spec fn cuda_puts_disjoint(puts: Seq<PendingGpuWrite>) -> bool {
    forall|i: int, j: int|
        0 <= i < puts.len() && 0 <= j < puts.len() && i != j ==> {
            &&& (#[trigger] puts[i]).frame != (#[trigger] puts[j]).frame
            &&& puts[i].slot@.loc != puts[j].slot@.loc
            &&& match (puts[i].action, puts[j].action) {
                (
                    PendingGpuAction::Publish { version: v1, only_if_current: false, .. },
                    PendingGpuAction::Publish { version: v2, only_if_current: false, .. },
                ) => v1 != v2,
                _ => true,
            }
        }
}

#[verifier::opaque]
pub open spec fn cuda_puts_valid(
    puts: Seq<PendingGpuWrite>,
    core: &CacheCore,
    timeline: TimelineId,
) -> bool {
    &&& cuda_puts_disjoint(puts)
    &&& forall|i: int| 0 <= i < puts.len() ==> (#[trigger] puts[i]).valid(core, timeline)
}

#[verifier::opaque]
pub open spec fn cuda_puts_bindings_covered(
    puts: Seq<PendingGpuWrite>,
    bindings: Map<VersionId, VersionBinding>,
) -> bool {
    forall|i: int|
        0 <= i < puts.len() ==> match (#[trigger] puts[i]).action {
            PendingGpuAction::Publish { version, proof, only_if_current: false, .. } => {
                &&& bindings.contains_key(version)
                &&& bindings[version].instance_id() == cluster_version_instance_id()
                &&& bindings[version].key() == (version.origin_node as NodeId, version.seq)
                &&& bindings[version].value() == proof@.certificate.bytes
            },
            _ => true,
        }
}

/// Optional CUDA-registered pinned host DRAM slab for H2D/D2H transfers.
pub struct PinnedHostSlab<'a> {
    pub(crate) reg: CudaHostRegistration<'a>,
    pub(crate) tier: TierId,
    pub(crate) first_frame: FrameId,
    pub(crate) num_frames: u32,
    pub(crate) sync_points: Vec<SyncPoint>,
}

impl<'a> PinnedHostSlab<'a> {
    pub open(crate) spec fn first_frame_spec(&self) -> FrameId {
        self.first_frame
    }

    pub open(crate) spec fn num_frames_spec(&self) -> u32 {
        self.num_frames
    }

    pub open(crate) spec fn tier_spec(&self) -> TierId {
        self.tier
    }

    pub open(crate) spec fn reg_spec(&self) -> CudaHostRegistration<'a> {
        self.reg
    }

    pub open(crate) spec fn sync_points_spec(&self) -> Vec<SyncPoint> {
        self.sync_points
    }

    pub open(crate) spec fn well_formed(&self, core: &CacheCore) -> bool {
        &&& self.reg.well_formed()
        &&& self.reg.instance_id() == core.instance_id()
        &&& self.reg.node() == core.node()
        &&& !(self.reg.space().is_gpu())
        &&& self.reg.num_slots() == self.num_frames as usize
        &&& self.reg.slot_len() == core.frames_spec().tier_frame_size(self.tier)
        &&& self.tier.valid(core.tiers_spec().len())
        &&& core.tiers_spec()[self.tier.index() as int].kind == TierKind::HostDram
        &&& self.sync_points.len() == self.num_frames as usize
        &&& slab_bound_to_core(core, self.tier, self.first_frame, self.num_frames, self.reg.space())
    }

    pub open(crate) spec fn owns(&self, frame: FrameId) -> bool {
        self.first_frame.0 <= frame.0 && frame.0 < self.first_frame.0 + self.num_frames
    }
}

/// Pluggable CUDA GPU VRAM backend with an ordered `CudaTimeline`, optional
/// pinned host slab registration, and O(1)-idle active async write tracking.
pub struct CudaBackend<'a> {
    pub(crate) core_id: Option<u64>,
    pub(crate) device: DeviceId,
    pub(crate) gpu_buffer: CudaBuffer,
    pub(crate) gpu_tier: TierId,
    pub(crate) gpu_first_frame: FrameId,
    pub(crate) gpu_num_frames: u32,
    pub(crate) gpu_sync_points: Vec<SyncPoint>,
    pub(crate) pinned_host: Option<PinnedHostSlab<'a>>,
    pub(crate) timeline: CudaTimeline,
    pub(crate) active_gpu_puts: Vec<PendingGpuWrite>,
}

impl<'a> CudaBackend<'a> {
    pub open(crate) spec fn gpu_first_frame_spec(&self) -> FrameId {
        self.gpu_first_frame
    }

    pub open(crate) spec fn timeline_spec(&self) -> CudaTimeline {
        self.timeline
    }

    pub open(crate) spec fn gpu_num_frames_spec(&self) -> u32 {
        self.gpu_num_frames
    }

    pub open(crate) spec fn device_spec(&self) -> DeviceId {
        self.device
    }

    pub open(crate) spec fn pinned_host_spec(&self) -> Option<PinnedHostSlab<'a>> {
        self.pinned_host
    }

    pub open(crate) spec fn gpu_buffer_spec(&self) -> CudaBuffer {
        self.gpu_buffer
    }

    pub open(crate) spec fn gpu_tier_spec(&self) -> TierId {
        self.gpu_tier
    }

    pub open(crate) spec fn active_gpu_puts_spec(&self) -> Vec<PendingGpuWrite> {
        self.active_gpu_puts
    }

    pub open(crate) spec fn gpu_sync_points_spec(&self) -> Vec<SyncPoint> {
        self.gpu_sync_points
    }

    pub open(crate) spec fn well_formed(&self, core: &CacheCore) -> bool {
        &&& self.gpu_buffer.well_formed()
        &&& self.gpu_buffer.instance_id() == core.instance_id()
        &&& self.gpu_buffer.node() == core.node()
        &&& self.gpu_buffer.space().is_gpu()
        &&& self.gpu_buffer.space().gpu_device() == self.device
        &&& self.gpu_buffer.num_slots() == self.gpu_num_frames as usize
        &&& self.gpu_buffer.slot_len() == core.frames_spec().tier_frame_size(self.gpu_tier)
        &&& self.gpu_tier.valid(core.tiers_spec().len())
        &&& core.tiers_spec()[self.gpu_tier.index() as int].kind == TierKind::GpuVram
        &&& self.gpu_sync_points.len() == self.gpu_num_frames as usize
        &&& slab_bound_to_core(
            core,
            self.gpu_tier,
            self.gpu_first_frame,
            self.gpu_num_frames,
            self.gpu_buffer.space(),
        )
        &&& match self.pinned_host {
            None => true,
            Some(h) => h.well_formed(core) && h.tier != self.gpu_tier
                && core.frames_spec().tier_frame_size(h.tier) == core.frames_spec().tier_frame_size(
                self.gpu_tier,
            ),
        }
        &&& self.timeline.device == self.device
        &&& self.timeline.well_formed(&core.timelines_spec(), core.node(), &core.ledger_spec()@)
        &&& cuda_puts_valid(self.active_gpu_puts@, core, self.timeline.timeline_id)
    }

    pub open(crate) spec fn owns_gpu(&self, frame: FrameId) -> bool {
        self.gpu_first_frame.0 <= frame.0 && frame.0 < self.gpu_first_frame.0 + self.gpu_num_frames
    }

    pub open(crate) spec fn owns_pinned_host(&self, frame: FrameId) -> bool {
        match self.pinned_host {
            None => false,
            Some(h) => h.owns(frame),
        }
    }

    pub open(crate) spec fn same_pinned_layout(&self, other: &Self) -> bool {
        match (self.pinned_host, other.pinned_host) {
            (None, None) => true,
            (Some(a), Some(b)) => a.reg == b.reg && a.tier == b.tier && a.first_frame
                == b.first_frame && a.num_frames == b.num_frames,
            _ => false,
        }
    }

    /// Allocate a GPU VRAM slab using the node default `frame_size`, create its
    /// `CudaTimeline` with `event_ring_len` events, and register both the
    /// `TierKind::GpuVram` tier and CUDA timeline in `core`.
    pub fn new(
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(stream_source): Tracked<EventSourceToken>,
        device: DeviceId,
        num_frames: u32,
        event_ring_len: usize,
        rdma_capable: bool,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            stream_source.well_formed(),
            stream_source.id() == old(core).instance_id(),
            stream_source.node() == old(core).node(),
            stream_source.source().is_cuda_stream(),
            stream_source.next_seq() == 0,
            source_fresh_for_ledger(stream_source, old(core).ledger_spec()@),
        ensures
            final(core).well_formed(),
            final(alloc).well_formed(),
            final(alloc).id() == old(alloc).id(),
            final(alloc).node() == old(alloc).node(),
            alloc_fresh_for_core(*final(alloc), final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            match r {
                Err(_) => *final(core) == *old(core) && *final(alloc) == *old(alloc),
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.device_spec() == device
                    &&& b.gpu_num_frames_spec() == num_frames
                    &&& b.gpu_tier_spec() == TierId(old(core).tiers_spec().len() as u16)
                    &&& b.gpu_first_frame_spec().0 == old(core).frames_spec().metas_spec().len()
                    &&& b.timeline_spec().timeline_id_spec() == old(
                        core,
                    ).timelines_spec().count_spec()
                    &&& b.pinned_host_spec() is None
                    &&& final(core).frames_spec().tier_frame_size(b.gpu_tier_spec()) == old(
                        core,
                    ).frames_spec().frame_size_spec()
                },
            },
    {
        let frame_size = core.frame_size();
        Self::new_with_frame_size(
            core,
            Tracked(instance),
            Tracked(alloc),
            Tracked(stream_source),
            device,
            num_frames,
            event_ring_len,
            rdma_capable,
            frame_size,
        )
    }

    fn finish_init(
        core: &mut CacheCore,
        Tracked(alloc): Tracked<&AllocSource>,
        device: DeviceId,
        stream: transfer::cuda::CudaStream,
        event_slots: Vec<EventSlot>,
        gpu_buffer: CudaBuffer,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
        Ghost(before_alloc): Ghost<AllocSource>,
        frame_size: usize,
        num_frames: u32,
        rdma_capable: bool,
    ) -> (r: Self)
        requires
            old(core).well_formed(),
            before_alloc.well_formed(),
            before_alloc.id() == old(core).instance_id(),
            before_alloc.node() == old(core).node(),
            alloc_fresh_for_core(before_alloc, old(core)),
            alloc.well_formed(),
            alloc.id() == old(core).instance_id(),
            alloc.node() == old(core).node(),
            alloc.names() == before_alloc.names().tail(),
            !alloc.names().contains(before_alloc.next_alloc_id()),
            gpu_buffer.well_formed(),
            gpu_buffer.instance_id() == old(core).instance_id(),
            gpu_buffer.node() == old(core).node(),
            gpu_buffer.space().is_gpu(),
            gpu_buffer.space().gpu_device() == device,
            gpu_buffer.space().alloc_id() == before_alloc.next_alloc_id(),
            gpu_buffer.num_slots() == num_frames as usize,
            gpu_buffer.slot_len() == frame_size,
            0 < frame_size,
            alloc.allocated_slab(before_alloc, gpu_buffer.space(), num_frames as usize, tokens),
            stream.well_formed(),
            stream.device() == device,
            stream.source().source().is_cuda_stream(),
            stream.source().id() == old(core).instance_id(),
            stream.source().node() == old(core).node(),
            source_fresh_for_ledger(stream.source(), old(core).ledger_spec()@),
            stream.tail() == ISet::empty(),
            stream.infos().map() == IMap::empty(),
            0 < event_slots.len() <= u32::MAX,
            forall|j: int|
                0 <= j < event_slots.len() ==> {
                    let s = #[trigger] event_slots[j];
                    &&& s.event.well_formed()
                    &&& s.event.device() == device
                    &&& s.event.observer() == old(core).node()
                    &&& s.event.infos().instance_id() == old(core).instance_id()
                    &&& s.event.recorded() == ISet::empty()
                    &&& s.event.infos().map() == IMap::empty()
                    &&& !s.in_use
                    &&& s.recorded_seq == 0
                },
            num_frames > 0,
            old(core).tiers_spec().len() < MAX_TIERS,
            old(core).timelines_spec().count_spec() < MAX_TIMELINES,
            old(core).frames_spec().metas_spec().len() + (num_frames as usize) <= u32::MAX,
        ensures
            final(core).well_formed(),
            alloc_fresh_for_core(*alloc, final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            r.well_formed(final(core)),
            r.device == device,
            r.gpu_num_frames == num_frames,
            r.gpu_tier == TierId(old(core).tiers_spec().len() as u16),
            r.gpu_first_frame.0 == old(core).frames_spec().metas_spec().len(),
            r.timeline.timeline_id == old(core).timelines_spec().count_spec(),
            r.pinned_host is None,
            final(core).frames_spec().tier_frame_size(r.gpu_tier) == frame_size,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let count = num_frames as usize;
        let ghost slots;
        proof {
            slots =
            lemma_allocated_slab_disjoint_with_size(
                before_alloc,
                *alloc,
                core,
                gpu_buffer.space(),
                count,
                frame_size,
                tokens,
            );
        }
        let timeline = CudaTimeline::register(core, device, stream, event_slots);
        let ghost after_tl = *core;
        let desc = TierDescriptor {
            id: TierId(0),
            kind: TierKind::GpuVram,
            device_id: Some(device),
            host_accessible: false,
            rdma_capable,
        };
        let (gpu_tier, gpu_first_frame) = core.register_tier_with_frame_size(
            desc,
            frame_size,
            Ghost(slots),
            num_frames,
            Tracked(tokens),
        ).unwrap();
        let mut gpu_sync_points: Vec<SyncPoint> = Vec::with_capacity(count);
        let mut i: usize = 0;
        while i < count
            invariant
                i <= count,
                gpu_sync_points.len() == i,
            decreases count - i,
        {
            gpu_sync_points.push(SyncPoint::HostComplete);
            i += 1;
        }
        let active_gpu_puts: Vec<PendingGpuWrite> = Vec::new();
        proof {
            lemma_core_wf_facts(&after_tl);
            lemma_core_wf_facts(core);
            lemma_after_register_tier(
                before_alloc,
                *alloc,
                &after_tl,
                core,
                gpu_tier,
                gpu_first_frame,
                num_frames,
                gpu_buffer.space(),
                slots,
            );
            lemma_cuda_puts_empty(active_gpu_puts@, core, timeline.timeline_id);
        }
        Self {
            core_id: core.runtime_id(),
            device,
            gpu_buffer,
            gpu_tier,
            gpu_first_frame,
            gpu_num_frames: num_frames,
            gpu_sync_points,
            pinned_host: None,
            timeline,
            active_gpu_puts,
        }
    }

    /// Allocate a GPU VRAM slab with an explicit per-tier `frame_size`, create its
    /// `CudaTimeline` with `event_ring_len` events, and register both the
    /// `TierKind::GpuVram` tier and CUDA timeline in `core`.
    pub fn new_with_frame_size(
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(alloc): Tracked<&mut AllocSource>,
        Tracked(stream_source): Tracked<EventSourceToken>,
        device: DeviceId,
        num_frames: u32,
        event_ring_len: usize,
        rdma_capable: bool,
        frame_size: usize,
    ) -> (r: BackendResult<Self>)
        requires
            old(core).well_formed(),
            instance.id() == old(core).instance_id(),
            old(alloc).well_formed(),
            old(alloc).id() == old(core).instance_id(),
            old(alloc).node() == old(core).node(),
            alloc_fresh_for_core(*old(alloc), old(core)),
            stream_source.well_formed(),
            stream_source.id() == old(core).instance_id(),
            stream_source.node() == old(core).node(),
            stream_source.source().is_cuda_stream(),
            stream_source.next_seq() == 0,
            source_fresh_for_ledger(stream_source, old(core).ledger_spec()@),
        ensures
            final(core).well_formed(),
            final(alloc).well_formed(),
            final(alloc).id() == old(alloc).id(),
            final(alloc).node() == old(alloc).node(),
            alloc_fresh_for_core(*final(alloc), final(core)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).ledger_spec() == old(core).ledger_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            match r {
                Err(_) => *final(core) == *old(core) && *final(alloc) == *old(alloc),
                Ok(b) => {
                    &&& b.well_formed(final(core))
                    &&& b.device_spec() == device
                    &&& b.gpu_num_frames_spec() == num_frames
                    &&& b.gpu_tier_spec() == TierId(old(core).tiers_spec().len() as u16)
                    &&& b.gpu_first_frame_spec().0 == old(core).frames_spec().metas_spec().len()
                    &&& b.timeline_spec().timeline_id_spec() == old(
                        core,
                    ).timelines_spec().count_spec()
                    &&& b.pinned_host_spec() is None
                    &&& final(core).frames_spec().tier_frame_size(b.gpu_tier_spec()) == frame_size
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        proof {
            lemma_core_wf_facts(core);
        }
        if frame_size == 0 || num_frames == 0 || event_ring_len == 0 || event_ring_len
            > u32::MAX as usize {
            return Err(CoreError::InvalidArgument);
        }
        let count = num_frames as usize;
        if count > usize::MAX / frame_size {
            return Err(CoreError::InvalidArgument);
        }
        if core.tier_count() >= MAX_TIERS || core.timeline_count() >= MAX_TIMELINES || count > (
        u32::MAX as usize) - core.frame_count() {
            return Err(CoreError::CapacityExhausted);
        }
        proof {
            assert(count * frame_size <= usize::MAX) by {
                vstd::arithmetic::div_mod::lemma_fundamental_div_mod(
                    usize::MAX as int,
                    frame_size as int,
                );
                vstd::arithmetic::mul::lemma_mul_inequality(
                    count as int,
                    (usize::MAX / frame_size) as int,
                    frame_size as int,
                );
            }
        }
        let (stream, event_slots) = CudaTimeline::create_hardware(
            Tracked(instance),
            device,
            Ghost(core.node()),
            Tracked(stream_source),
            event_ring_len,
        )?;
        let ghost before_alloc = *alloc;
        let (gpu_buffer, Tracked(tokens)) = match cuda::alloc_device_slab(
            Tracked(alloc),
            device,
            count,
            frame_size,
        ) {
            Ok(pair) => pair,
            Err(e) => {
                CudaTimeline::destroy_hardware(stream, event_slots);
                return Err(cuda_error(e));
            },
        };
        Ok(
            Self::finish_init(
                core,
                Tracked(alloc),
                device,
                stream,
                event_slots,
                gpu_buffer,
                Tracked(tokens),
                Ghost(before_alloc),
                frame_size,
                num_frames,
                rdma_capable,
            ),
        )
    }

    /// Pin a host slab for CUDA DMA (H2D / D2H) via
    /// `cuda::host_register`.
    pub fn attach_pinned_host<H: HostSlabBackend>(&mut self, core: &CacheCore, host: &'a H) -> (r:
        BackendResult<()>)
        requires
            core.well_formed(),
            old(self).well_formed(core),
            old(self).pinned_host_spec() is None,
            host.host_well_formed(core),
        ensures
            final(self).well_formed(core),
            final(self).device_spec() == old(self).device_spec(),
            final(self).gpu_tier_spec() == old(self).gpu_tier_spec(),
            final(self).gpu_first_frame_spec() == old(self).gpu_first_frame_spec(),
            final(self).gpu_num_frames_spec() == old(self).gpu_num_frames_spec(),
            final(self).timeline_spec() == old(self).timeline_spec(),
            r.is_ok() ==> (final(self).pinned_host_spec() matches Some(h) && h.tier_spec()
                == host.host_tier_spec() && h.first_frame_spec() == host.host_first_frame_spec()
                && h.num_frames_spec() == host.host_num_frames_spec()),
            r.is_err() ==> *final(self) == *old(self),
    {
        proof {
            host.lemma_host_layout(core);
        }
        if core.registered_tier_frame_size(host.host_tier_id()) != core.registered_tier_frame_size(
            self.gpu_tier,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let reg = match cuda::host_register(host.host_buffer()) {
            Ok(reg) => reg,
            Err(e) => return Err(cuda_error(e)),
        };
        let num_frames = host.host_num_frames();
        let count = num_frames as usize;
        let mut sync_points: Vec<SyncPoint> = Vec::with_capacity(count);
        let mut i: usize = 0;
        while i < count
            invariant
                i <= count,
                sync_points.len() == i,
            decreases count - i,
        {
            sync_points.push(SyncPoint::HostComplete);
            i += 1;
        }
        self.pinned_host = Some(
            PinnedHostSlab {
                reg,
                tier: host.host_tier_id(),
                first_frame: host.host_first_frame(),
                num_frames,
                sync_points,
            },
        );
        Ok(())
    }

    pub fn stream(&self) -> (stream: &cuda::CudaStream)
        ensures
            *stream == self.timeline_spec().stream_spec(),
    {
        &self.timeline.stream
    }

    pub fn gpu_first_frame(&self) -> (frame: FrameId)
        ensures
            frame == self.gpu_first_frame_spec(),
    {
        self.gpu_first_frame
    }

    pub fn gpu_num_frames(&self) -> (count: u32)
        ensures
            count == self.gpu_num_frames_spec(),
    {
        self.gpu_num_frames
    }

    pub fn has_pinned_host(&self) -> (pinned: bool)
        ensures
            pinned == self.pinned_host_spec().is_some(),
    {
        self.pinned_host.is_some()
    }

    pub fn device(&self) -> (d: DeviceId)
        ensures
            d == self.device_spec(),
    {
        self.device
    }

    pub fn gpu_buffer(&self) -> (b: &CudaBuffer)
        ensures
            b == &self.gpu_buffer_spec(),
    {
        &self.gpu_buffer
    }

    pub fn gpu_tier_id(&self) -> (t: TierId)
        ensures
            t == self.gpu_tier_spec(),
    {
        self.gpu_tier
    }

    pub fn timeline_id(&self) -> (t: TimelineId)
        ensures
            t == self.timeline_spec().timeline_id_spec(),
    {
        self.timeline.timeline_id
    }

    pub(crate) fn active_put_count(&self) -> (n: usize)
        ensures
            n == self.active_gpu_puts.len(),
    {
        self.active_gpu_puts.len()
    }

    pub fn owns_gpu_frame(&self, frame: FrameId) -> (r: bool)
        ensures
            r == self.owns_gpu(frame),
    {
        frame.0 >= self.gpu_first_frame.0 && frame.0 - self.gpu_first_frame.0 < self.gpu_num_frames
    }

    pub fn owns_pinned_host_frame(&self, frame: FrameId) -> (r: bool)
        ensures
            r == self.owns_pinned_host(frame),
    {
        match &self.pinned_host {
            None => false,
            Some(h) => frame.0 >= h.first_frame.0 && frame.0 - h.first_frame.0 < h.num_frames,
        }
    }

    pub fn gpu_frame_slice(&self, frame: FrameId) -> (s: CudaSlice<'_>)
        requires
            self.gpu_buffer_spec().well_formed(),
            self.gpu_buffer_spec().num_slots() == self.gpu_num_frames_spec() as usize,
            self.owns_gpu(frame),
        ensures
            s.well_formed(),
            s.slot_ref() == self.gpu_buffer_spec().slot_ref(
                (frame.0 - self.gpu_first_frame_spec().0) as nat,
            ),
            s.instance_id() == self.gpu_buffer_spec().instance_id(),
            s.kind() == CudaPointerKind::Device(self.gpu_buffer_spec().space().gpu_device()),
    {
        let idx = (frame.0 - self.gpu_first_frame.0) as usize;
        self.gpu_buffer.slot_slice(idx)
    }

    pub fn pinned_host_frame_slice(&self, frame: FrameId) -> (s: CudaSlice<'_>)
        requires
            self.pinned_host_spec() matches Some(h) && h.reg_spec().well_formed()
                && h.reg_spec().num_slots() == h.num_frames_spec() as usize && h.owns(frame),
        ensures
            s.well_formed(),
            s.slot_ref() == self.pinned_host_spec().unwrap().reg_spec().slot_ref(
                (frame.0 - self.pinned_host_spec().unwrap().first_frame_spec().0) as nat,
            ),
            s.instance_id() == self.pinned_host_spec().unwrap().reg_spec().instance_id(),
            s.kind() == CudaPointerKind::PinnedHost,
    {
        match &self.pinned_host {
            Some(h) => h.reg.slot_slice((frame.0 - h.first_frame.0) as usize),
            None => {
                assert(false);
                self.gpu_buffer.slot_slice(0)
            },
        }
    }

    pub open(crate) spec fn frame_sync_point_spec(&self, frame: FrameId) -> SyncPoint {
        if self.owns_gpu(frame) {
            self.gpu_sync_points[(frame.0 - self.gpu_first_frame.0) as int]
        } else if self.owns_pinned_host(frame) {
            let h = self.pinned_host.unwrap();
            h.sync_points[(frame.0 - h.first_frame.0) as int]
        } else {
            SyncPoint::HostComplete
        }
    }

    pub(crate) fn frame_sync_point(&self, frame: FrameId) -> (p: SyncPoint)
        requires
            self.gpu_sync_points.len() == self.gpu_num_frames as usize,
            self.pinned_host matches Some(h) ==> h.sync_points.len() == h.num_frames as usize,
        ensures
            p == self.frame_sync_point_spec(frame),
    {
        if self.owns_gpu_frame(frame) {
            self.gpu_sync_points[(frame.0 - self.gpu_first_frame.0) as usize]
        } else if let Some(h) = &self.pinned_host {
            if frame.0 >= h.first_frame.0 && frame.0 - h.first_frame.0 < h.num_frames {
                h.sync_points[(frame.0 - h.first_frame.0) as usize]
            } else {
                SyncPoint::HostComplete
            }
        } else {
            SyncPoint::HostComplete
        }
    }

    pub(super) fn set_frame_sync_point(&mut self, frame: FrameId, point: SyncPoint)
        requires
            old(self).gpu_sync_points.len() == old(self).gpu_num_frames as usize,
            old(self).pinned_host matches Some(h) ==> h.sync_points.len() == h.num_frames as usize,
        ensures
            final(self).device == old(self).device,
            final(self).gpu_buffer == old(self).gpu_buffer,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).gpu_sync_points.len() == old(self).gpu_sync_points.len(),
            final(self).timeline == old(self).timeline,
            final(self).active_gpu_puts == old(self).active_gpu_puts,
            match (old(self).pinned_host, final(self).pinned_host) {
                (None, None) => true,
                (Some(oh), Some(nh)) => oh.reg == nh.reg && oh.tier == nh.tier && oh.first_frame
                    == nh.first_frame && oh.num_frames == nh.num_frames && nh.sync_points.len()
                    == oh.sync_points.len(),
                _ => false,
            },
    {
        if self.owns_gpu_frame(frame) {
            let idx = (frame.0 - self.gpu_first_frame.0) as usize;
            self.gpu_sync_points.set(idx, point);
        } else if let Some(mut h) = self.pinned_host.take() {
            if frame.0 >= h.first_frame.0 && frame.0 - h.first_frame.0 < h.num_frames {
                let idx = (frame.0 - h.first_frame.0) as usize;
                h.sync_points.set(idx, point);
            }
            self.pinned_host = Some(h);
        }
    }

    pub fn is_frame_ready(&self, core: &CacheCore, frame: FrameId) -> (ready: bool)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            ready == (frame.0 < core.frames_spec().metas_spec().len() && core.frames_spec().meta(
                frame,
            ).state == STATE_PUBLISHED && self.timeline_spec().sync_point_ready(
                core,
                self.frame_sync_point_spec(frame),
            )),
    {
        if (frame.0 as usize) >= core.frame_count() || core.frame_meta(frame).unwrap().state
            != crate::frame::STATE_PUBLISHED {
            return false;
        }
        let p = self.frame_sync_point(frame);
        proof {
            lemma_core_wf_facts(core);
        }
        self.timeline.is_sync_point_ready(core, p)
    }

    /// Poll completed CUDA events and retire ready parked GPU puts in ascending
    /// ticket sequence order, returning partial progress on any event-query error.
    pub fn poll_completions(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<CudaPollReport, (CoreError, CudaPollReport)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).device_spec() == old(self).device_spec(),
            final(self).gpu_tier_spec() == old(self).gpu_tier_spec(),
            final(self).gpu_first_frame_spec() == old(self).gpu_first_frame_spec(),
            final(self).gpu_num_frames_spec() == old(self).gpu_num_frames_spec(),
            final(self).same_pinned_layout(old(self)),
            final(self).timeline_spec().timeline_id_spec() == old(
                self,
            ).timeline_spec().timeline_id_spec(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is Some,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_poll_core = *core;
        let ghost fp = self.poll_footprint();
        proof {
            lemma_cuda_poll_footprint_frames(self, core);
        }
        let poll_res = self.timeline.poll_events(core, Tracked(instance));
        proof {
            lemma_cuda_backend_after_poll_events(
                self.active_gpu_puts@,
                &before_poll_core,
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                &self.pinned_host,
                self.timeline.timeline_id,
            );
            assert(core.frames_spec().same_layout(&before_poll_core.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            assert(core.frames_spec().preserves_initializing(&before_poll_core.frames_spec(), None))
                by {
                reveal(FrameTable::preserves_initializing);
            }
            lemma_core_transition_after_retirement(&before_poll_core, core, fp, None, None);
        }
        let (events_completed, poll_err) = match poll_res {
            Ok(n) => (n, None),
            Err((e, n)) => (n, Some(e)),
        };
        let mut report = CudaPollReport {
            events_completed,
            writes_published: 0,
            writes_aborted: 0,
        };
        if self.active_gpu_puts.is_empty() {
            return match poll_err {
                Some(e) => Err((e, report)),
                None => Ok(report),
            };
        }
        let tl = self.timeline.timeline_id;
        while !self.active_gpu_puts.is_empty()
            invariant
                fp == old(self).poll_footprint(),
                forall|i: int|
                    0 <= i < self.active_gpu_puts@.len() ==> old(self).active_gpu_puts@.contains(
                        #[trigger] self.active_gpu_puts@[i],
                    ),
                tl == self.timeline.timeline_id == old(self).timeline.timeline_id,
                self.device == old(self).device,
                self.gpu_tier == old(self).gpu_tier,
                self.gpu_first_frame == old(self).gpu_first_frame,
                self.gpu_num_frames == old(self).gpu_num_frames,
                self.same_pinned_layout(old(self)),
                core.well_formed(),
                self.well_formed(core),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.tiers_spec() == old(core).tiers_spec(),
                old(core).spec_spec()@.versions_monotonic(core.spec_spec()@),
                core_transition(old(core), core, old(self).poll_footprint()),
            decreases self.active_gpu_puts.len(),
        {
            proof {
                lemma_core_wf_facts(core);
            }
            let confirmed = core.timelines().confirmed_sequences()[tl as usize];
            let idx = match self.find_min_ready_put(confirmed) {
                Some(i) => i,
                None => break,
            };
            let ghost before_puts = self.active_gpu_puts@;
            let ghost before_step_core = *core;
            let entry = self.active_gpu_puts.swap_remove(idx);
            proof {
                lemma_cuda_put_extract_valid(before_puts, core, tl, idx as int);
                lemma_cuda_poll_keeps_scope(
                    old(self),
                    before_puts,
                    self.active_gpu_puts@,
                    entry,
                    idx as int,
                );
                lemma_core_catalog_indexed(core);
            }
            match entry.action {
                PendingGpuAction::Publish { key, version, proof: _proof, only_if_current } => {
                    let should_publish = if only_if_current {
                        match core.catalog().lookup(key) {
                            Some(e) => e.version.origin_node == version.origin_node && e.version.seq
                                == version.seq,
                            None => false,
                        }
                    } else {
                        true
                    };
                    if should_publish {
                        let ghost cert;
                        proof {
                            cert = lemma_cuda_certify_ready_put(&entry, _proof@, core, tl);
                        }
                        core.publish_parked_write(entry.frame, entry.ticket, Ghost(cert));
                        self.set_frame_sync_point(entry.frame, SyncPoint::HostComplete);
                        if report.writes_published < usize::MAX {
                            report.writes_published += 1;
                        }
                    } else {
                        proof {
                            lemma_cuda_ready_put_writable(&entry, core, tl);
                        }
                        core.abort_parked_write(entry.frame, entry.tier);
                        self.set_frame_sync_point(entry.frame, SyncPoint::HostComplete);
                        if report.writes_aborted < usize::MAX {
                            report.writes_aborted += 1;
                        }
                    }
                },
                PendingGpuAction::Abort => {
                    proof {
                        lemma_cuda_ready_put_writable(&entry, core, tl);
                    }
                    core.abort_parked_write(entry.frame, entry.tier);
                    self.set_frame_sync_point(entry.frame, SyncPoint::HostComplete);
                    if report.writes_aborted < usize::MAX {
                        report.writes_aborted += 1;
                    }
                },
            }
            proof {
                lemma_versions_monotonic_trans(
                    old(core).spec_spec()@,
                    before_step_core.spec_spec()@,
                    core.spec_spec()@,
                );
                lemma_cuda_backend_after_retire_put(
                    before_puts,
                    self.active_gpu_puts@,
                    idx as int,
                    &entry,
                    &before_step_core,
                    core,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                    &self.pinned_host,
                    tl,
                );
                lemma_core_wf_facts(&before_step_core);
                lemma_core_wf_facts(core);
                lemma_cuda_retired_version(&before_step_core, core, entry, fp);
                lemma_core_transition_after_retirement(
                    &before_step_core,
                    core,
                    fp,
                    Some(entry.frame),
                    None,
                );
                lemma_core_transition_trans_same(old(core), &before_step_core, core, fp);
            }
        }
        match poll_err {
            Some(e) => Err((e, report)),
            None => Ok(report),
        }
    }

    /// Poll completed CUDA events, retire ready parked GPU puts in FIFO order, and
    /// register pre-bound `VersionBinding`s into `core.version_registry_spec()` so
    /// `core.all_versions_bound()` is preserved across async GPU publication.
    pub(crate) fn poll_completions_with_bindings(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(pending_bindings): Tracked<&mut Map<VersionId, VersionBinding>>,
    ) -> (r: Result<CudaPollReport, (CoreError, CudaPollReport)>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            cuda_puts_bindings_covered(old(self).active_gpu_puts@, *old(pending_bindings)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            cuda_puts_bindings_covered(final(self).active_gpu_puts@, *final(pending_bindings)),
            final(self).device == old(self).device,
            final(self).gpu_tier == old(self).gpu_tier,
            final(self).gpu_first_frame == old(self).gpu_first_frame,
            final(self).gpu_num_frames == old(self).gpu_num_frames,
            final(self).same_pinned_layout(old(self)),
            final(self).timeline.timeline_id == old(self).timeline.timeline_id,
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is Some,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let ghost before_poll_core = *core;
        let ghost fp = self.poll_footprint();
        proof {
            lemma_cuda_poll_footprint_frames(self, core);
        }
        let poll_res = self.timeline.poll_events(core, Tracked(instance));
        proof {
            lemma_cuda_backend_after_poll_events(
                self.active_gpu_puts@,
                &before_poll_core,
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                &self.pinned_host,
                self.timeline.timeline_id,
            );
            assert(core.frames_spec().same_layout(&before_poll_core.frames_spec())) by {
                reveal(FrameTable::same_layout);
            }
            assert(core.frames_spec().preserves_initializing(&before_poll_core.frames_spec(), None))
                by {
                reveal(FrameTable::preserves_initializing);
            }
            lemma_core_transition_after_retirement(&before_poll_core, core, fp, None, None);
            assert(core.all_versions_bound());
        }
        let (events_completed, poll_err) = match poll_res {
            Ok(n) => (n, None),
            Err((e, n)) => (n, Some(e)),
        };
        let mut report = CudaPollReport {
            events_completed,
            writes_published: 0,
            writes_aborted: 0,
        };
        if self.active_gpu_puts.is_empty() {
            return match poll_err {
                Some(e) => Err((e, report)),
                None => Ok(report),
            };
        }
        let tl = self.timeline.timeline_id;
        while !self.active_gpu_puts.is_empty()
            invariant
                fp == old(self).poll_footprint(),
                forall|i: int|
                    0 <= i < self.active_gpu_puts@.len() ==> old(self).active_gpu_puts@.contains(
                        #[trigger] self.active_gpu_puts@[i],
                    ),
                tl == self.timeline.timeline_id == old(self).timeline.timeline_id,
                self.device == old(self).device,
                self.gpu_tier == old(self).gpu_tier,
                self.gpu_first_frame == old(self).gpu_first_frame,
                self.gpu_num_frames == old(self).gpu_num_frames,
                self.same_pinned_layout(old(self)),
                core.well_formed(),
                core.all_versions_bound(),
                self.well_formed(core),
                cuda_puts_bindings_covered(self.active_gpu_puts@, *pending_bindings),
                instance.id() == core.instance_id(),
                core.node_id_spec() == old(core).node_id_spec(),
                core.tiers_spec() == old(core).tiers_spec(),
                old(core).spec_spec()@.versions_monotonic(core.spec_spec()@),
                core_transition(old(core), core, old(self).poll_footprint()),
            decreases self.active_gpu_puts.len(),
        {
            proof {
                lemma_core_wf_facts(core);
            }
            let confirmed = core.timelines().confirmed_sequences()[tl as usize];
            let idx = match self.find_min_ready_put(confirmed) {
                Some(i) => i,
                None => break,
            };
            let ghost before_puts = self.active_gpu_puts@;
            let ghost before_step_core = *core;
            let ghost before_bindings = *pending_bindings;
            let entry = self.active_gpu_puts.swap_remove(idx);
            proof {
                lemma_cuda_put_extract_valid(before_puts, core, tl, idx as int);
                lemma_cuda_poll_keeps_scope(
                    old(self),
                    before_puts,
                    self.active_gpu_puts@,
                    entry,
                    idx as int,
                );
                lemma_core_catalog_indexed(core);
            }
            match entry.action {
                PendingGpuAction::Publish { key, version, proof: _proof, only_if_current } => {
                    let should_publish = if only_if_current {
                        match core.catalog().lookup(key) {
                            Some(e) => e.version.origin_node == version.origin_node && e.version.seq
                                == version.seq,
                            None => false,
                        }
                    } else {
                        true
                    };
                    if should_publish {
                        let ghost cert;
                        proof {
                            cert = lemma_cuda_certify_ready_put(&entry, _proof@, core, tl);
                        }
                        core.publish_parked_write(entry.frame, entry.ticket, Ghost(cert));
                        let ghost after_pub_core = *core;
                        proof {
                            lemma_core_wf_facts(&after_pub_core);
                        }
                        if !only_if_current {
                            proof {
                                lemma_cuda_puts_bindings_extract(
                                    before_puts,
                                    before_bindings,
                                    idx as int,
                                );
                            }
                            let tracked b = pending_bindings.tracked_remove(version);
                            core.register_version_binding(version, Tracked(b));
                            proof {
                                lemma_cuda_puts_bindings_swap_remove(
                                    before_puts,
                                    self.active_gpu_puts@,
                                    before_bindings,
                                    *pending_bindings,
                                    idx as int,
                                    Some(version),
                                    &before_step_core,
                                    tl,
                                );
                                assert(core.all_versions_bound()) by {
                                    assert forall|v: VersionId| #[trigger]
                                        core.frames_spec().versions_spec()@.contains_key(
                                            v,
                                        ) implies core.has_version_binding(v) by {
                                        if v != version {
                                            assert(before_step_core.frames_spec().versions_spec()@.contains_key(
                                            v));
                                            assert(before_step_core.has_version_binding(v));
                                            assert(after_pub_core.has_version_binding(v));
                                        }
                                    }
                                }
                            }
                        } else {
                            proof {
                                lemma_cuda_puts_bindings_swap_remove(
                                    before_puts,
                                    self.active_gpu_puts@,
                                    before_bindings,
                                    *pending_bindings,
                                    idx as int,
                                    None,
                                    &before_step_core,
                                    tl,
                                );
                                assert(core.all_versions_bound()) by {
                                    assert(before_step_core.frames_spec().versions_spec()@.contains_key(
                                    version));
                                    assert forall|v: VersionId| #[trigger]
                                        core.frames_spec().versions_spec()@.contains_key(
                                            v,
                                        ) implies core.has_version_binding(v) by {
                                        assert(before_step_core.frames_spec().versions_spec()@.contains_key(
                                        v));
                                        assert(before_step_core.has_version_binding(v));
                                    }
                                }
                            }
                        }
                        self.set_frame_sync_point(entry.frame, SyncPoint::HostComplete);
                        if report.writes_published < usize::MAX {
                            report.writes_published += 1;
                        }
                    } else {
                        proof {
                            lemma_cuda_ready_put_writable(&entry, core, tl);
                        }
                        core.abort_parked_write(entry.frame, entry.tier);
                        proof {
                            lemma_cuda_puts_bindings_swap_remove(
                                before_puts,
                                self.active_gpu_puts@,
                                before_bindings,
                                *pending_bindings,
                                idx as int,
                                None,
                                &before_step_core,
                                tl,
                            );
                            assert(core.frames_spec().versions_spec()@
                                == before_step_core.frames_spec().versions_spec()@) by {
                                reveal(FrameTable::same_directory);
                            }
                            assert(core.all_versions_bound());
                        }
                        self.set_frame_sync_point(entry.frame, SyncPoint::HostComplete);
                        if report.writes_aborted < usize::MAX {
                            report.writes_aborted += 1;
                        }
                    }
                },
                PendingGpuAction::Abort => {
                    proof {
                        lemma_cuda_ready_put_writable(&entry, core, tl);
                    }
                    core.abort_parked_write(entry.frame, entry.tier);
                    proof {
                        lemma_cuda_puts_bindings_swap_remove(
                            before_puts,
                            self.active_gpu_puts@,
                            before_bindings,
                            *pending_bindings,
                            idx as int,
                            None,
                            &before_step_core,
                            tl,
                        );
                        assert(core.frames_spec().versions_spec()@
                            == before_step_core.frames_spec().versions_spec()@) by {
                            reveal(FrameTable::same_directory);
                        }
                        assert(core.all_versions_bound());
                    }
                    self.set_frame_sync_point(entry.frame, SyncPoint::HostComplete);
                    if report.writes_aborted < usize::MAX {
                        report.writes_aborted += 1;
                    }
                },
            }
            proof {
                lemma_versions_monotonic_trans(
                    old(core).spec_spec()@,
                    before_step_core.spec_spec()@,
                    core.spec_spec()@,
                );
                lemma_cuda_backend_after_retire_put(
                    before_puts,
                    self.active_gpu_puts@,
                    idx as int,
                    &entry,
                    &before_step_core,
                    core,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                    &self.pinned_host,
                    tl,
                );
                lemma_core_wf_facts(&before_step_core);
                lemma_core_wf_facts(core);
                lemma_cuda_retired_version(&before_step_core, core, entry, fp);
                lemma_core_transition_after_retirement(
                    &before_step_core,
                    core,
                    fp,
                    Some(entry.frame),
                    None,
                );
                lemma_core_transition_trans_same(old(core), &before_step_core, core, fp);
            }
        }
        match poll_err {
            Some(e) => Err((e, report)),
            None => Ok(report),
        }
    }
}

} // verus!
