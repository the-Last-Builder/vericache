#[cfg(verus_only)]
use super::{
    core_transition, slab_bound_to_core, source_fresh_for_ledger, CoreFootprint, VersionFootprint,
};
use super::{BackendResult, DevicePollReport, HostSlabBackend};
#[cfg(verus_only)]
use crate::frame::{FrameGhost, STATE_EMPTY, STATE_INITIALIZING, STATE_PUBLISHED};
#[cfg(verus_only)]
use crate::kernel::VersionRegistry;
use crate::kernel::{CacheCore, ReadLeaseHandle};
use crate::provenance::ValueCertificate;
#[cfg(verus_only)]
use crate::provenance::{
    preserves_destination, replica_valid, revoked_submap, source_slice_bytes, source_slice_valid,
    KnowledgeLedger,
};
use crate::spec::*;
#[cfg(verus_only)]
use crate::timeline::TimelineKind;
#[cfg(verus_only)]
use foundation::tracked::{closure_from, knowledge_of, Frontier};
use transfer_core::context::*;
use transfer_core::domain::{DeviceId, SlotRef, TransferEvent};
#[cfg(verus_only)]
use transfer_core::domain::{EventSource, SlotLoc};
use transfer_core::memory::RdmaRegisterableSlab;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

mod none;
pub use none::{NoDevice, NoDeviceBuffer, NoDeviceSlice, NoDeviceStream};

#[cfg(feature = "cuda")]
pub type DefaultDeviceBackend<'a> = super::cuda::CudaBackend<'a>;
#[cfg(not(feature = "cuda"))]
pub type DefaultDeviceBackend<'a> = NoDevice;

verus! {

#[cfg(feature = "cuda")]
pub type CudaAttachment<'a> = DeviceAttachment<'a, DefaultDeviceBackend<'a>>;

/// A caller-borrowed DMA range with its transfer instance and physical slot identity.
pub trait DeviceSlice {
    spec fn well_formed(&self) -> bool;

    spec fn slot_ref(&self) -> SlotRef;

    spec fn instance_id(&self) -> InstanceId;

    fn len(&self) -> (len: usize)
        requires
            self.well_formed(),
        ensures
            len == self.slot_ref().sub_len,
    ;

    fn device_addr(&self) -> usize
        requires
            self.well_formed(),
    ;
}

/// An ordered device stream, including the submissions it can attest to completing.
pub trait DeviceStream {
    spec fn well_formed(&self) -> bool;

    spec fn source(&self) -> EventSourceToken;

    spec fn infos(&self) -> SubmittedMap;

    spec fn tail(&self) -> ISet<TransferEvent>;

    spec fn device_spec(&self) -> DeviceId;

    spec fn timeline_kind_spec(&self) -> TimelineKind;

    spec fn source_kind_valid(&self) -> bool;

    proof fn lemma_source_kind(&self)
        requires
            self.source_kind_valid(),
        ensures
            !self.source().source().is_host(),
            !self.source().source().is_io_uring(),
            !self.source().source().is_rdma_qp(),
    ;

    fn device_id(&self) -> (device: DeviceId)
        ensures
            device == self.device_spec(),
    ;

    fn runtime_id(&self) -> u64;

    fn lemma_source_fresh(
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
    ;
}

/// The part of a parked device write needed by engine composition.
pub ghost struct PendingDeviceWrite {
    pub frame: FrameId,
    pub slot: SlotRef,
    pub publish: Option<PendingDevicePublication>,
}

pub ghost struct PendingDevicePublication {
    pub version: VersionId,
    pub bytes: Seq<u8>,
    pub only_if_current: bool,
}

impl PendingDeviceWrite {
    pub open spec fn publishes_version(self, version: VersionId) -> bool {
        self.publish matches Some(p) && p.version == version
    }

    pub open spec fn binding_covered(self, bindings: Map<VersionId, VersionBinding>) -> bool {
        match self.publish {
            Some(PendingDevicePublication { version, bytes, only_if_current: false }) => {
                &&& bindings.contains_key(version)
                &&& bindings[version].instance_id() == cluster_version_instance_id()
                &&& bindings[version].key() == (version.origin_node as u64, version.seq)
                &&& bindings[version].value() == bytes
            },
            _ => true,
        }
    }
}

pub open spec fn device_put_frames(puts: Seq<PendingDeviceWrite>) -> ISet<FrameId> {
    ISet::new(|f: FrameId| exists|i: int| 0 <= i < puts.len() && (#[trigger] puts[i]).frame == f)
}

pub open spec fn device_put_versions(puts: Seq<PendingDeviceWrite>) -> ISet<VersionId> {
    ISet::new(
        |v: VersionId|
            exists|i: int| 0 <= i < puts.len() && (#[trigger] puts[i]).publishes_version(v),
    )
}

pub open spec fn device_put_bindings_covered(
    puts: Seq<PendingDeviceWrite>,
    bindings: Map<VersionId, VersionBinding>,
) -> bool {
    forall|i: int| 0 <= i < puts.len() ==> (#[trigger] puts[i]).binding_covered(bindings)
}

pub open spec fn device_put_frames_untouched(
    puts: Seq<PendingDeviceWrite>,
    before: &CacheCore,
    frames: ISet<FrameId>,
) -> bool {
    frames.is_empty() || (forall|f: FrameId| #[trigger]
        frames.contains(f) && f.0 < before.frames_spec().metas_spec().len()
            ==> before.frames_spec().frame(f).token is None) || (forall|i: int|
        0 <= i < puts.len() ==> !frames.contains((#[trigger] puts[i]).frame))
}

pub open spec fn device_puts_untouched(
    puts: Seq<PendingDeviceWrite>,
    before: &CacheCore,
    fp: CoreFootprint,
) -> bool {
    &&& fp.allowed_sources is Some
    &&& device_put_frames_untouched(puts, before, fp.touched_init_frames)
    &&& device_put_frames_untouched(puts, before, fp.written_init_frames)
    &&& fp.versions.specific matches Some(versions) && (versions.is_empty() || forall|i: int|
        0 <= i < puts.len() ==> match (#[trigger] puts[i]).publish {
            Some(publish) => !versions.contains(publish.version),
            None => true,
        })
}

pub open spec fn device_put_destinations_preserved(
    puts: Seq<PendingDeviceWrite>,
    before: &CacheCore,
    ledger: &KnowledgeLedger,
) -> bool {
    forall|i: int|
        0 <= i < puts.len() && (#[trigger] puts[i]).publish is Some ==> preserves_destination(
            before.ledger_spec()@,
            *ledger,
            before.node(),
            puts[i].slot.loc,
        )
}

pub open spec fn device_poll_footprint(
    timeline: TimelineId,
    puts: Seq<PendingDeviceWrite>,
) -> CoreFootprint {
    CoreFootprint::completion(
        timeline,
        device_put_frames(puts),
        ISet::empty(),
        VersionFootprint::only(device_put_versions(puts)),
    )
}

/// Validates the byte ranges represented by a gather or scatter offset table.
pub open spec fn device_scatter_offsets_valid<S: DeviceSlice>(
    slices: Seq<S>,
    offsets: Seq<usize>,
    total_len: usize,
) -> bool {
    &&& slices.len() >= 1
    &&& offsets.len() == slices.len() + 1
    &&& offsets[0] == 0
    &&& offsets[slices.len() as int] == total_len
    &&& forall|i: int|
        0 <= i < slices.len() ==> {
            &&& (#[trigger] slices[i]).well_formed()
            &&& slices[i].slot_ref().sub_len > 0
            &&& offsets[i] + slices[i].slot_ref().sub_len == offsets[i + 1]
        }
}

pub open spec fn valid_device_caller_stream<S: DeviceStream>(
    stream: S,
    core: &CacheCore,
    timeline: TimelineId,
    backend_source: EventSource,
) -> bool {
    &&& stream.well_formed()
    &&& stream.source().id() == core.instance_id()
    &&& stream.source().node() == core.node()
    &&& stream.source_kind_valid()
    &&& stream.source().source() != backend_source
    &&& source_fresh_for_ledger(stream.source(), core.ledger_spec()@)
    &&& stream.infos().map().submap_of(core.ledger_spec()@.submitted.map())
    &&& stream.tail().subset_of(stream.infos().dom())
    &&& timeline < core.timelines_spec().count_spec() <= MAX_TIMELINES
    &&& core.timelines_spec().kinds_spec()[timeline as int] == stream.timeline_kind_spec()
}

/// Device operations and composition guarantees used by the cache engines.
///
/// `'a` covers the lifetime of resources borrowed by a backend. The independent
/// `Slice<'s>` names the range borrowed by a particular call, including each
/// element of a scatter/gather list.
pub trait DeviceBackend<'a>: Sized {
    type Buffer: RdmaRegisterableSlab;

    type Stream: DeviceStream;

    type Slice<'s>: DeviceSlice;

    spec fn well_formed(&self, core: &CacheCore) -> bool;

    spec fn same_config(&self, other: &Self) -> bool;

    spec fn device_spec(&self) -> DeviceId;

    spec fn device_tier_spec(&self) -> TierId;

    spec fn device_first_frame_spec(&self) -> FrameId;

    spec fn device_num_frames_spec(&self) -> u32;

    spec fn timeline_id_spec(&self) -> TimelineId;

    spec fn event_source_spec(&self) -> EventSource;

    spec fn stream_spec(&self) -> Self::Stream;

    spec fn buffer_spec(&self) -> Self::Buffer;

    spec fn has_pinned_host_spec(&self) -> bool;

    spec fn pinned_host_layout_spec(&self) -> Option<(TierId, FrameId, u32)>;

    spec fn pinned_host_matches(&self, tier: TierId, first_frame: FrameId, num_frames: u32) -> bool;

    spec fn owns_device(&self, frame: FrameId) -> bool;

    spec fn owns_pinned_host(&self, frame: FrameId) -> bool;

    spec fn active_put_count_spec(&self) -> usize;

    spec fn active_put_frames(&self) -> ISet<FrameId>;

    spec fn pending_puts_spec(&self) -> Seq<PendingDeviceWrite>;

    spec fn pending_put_bindings_covered(&self, bindings: Map<VersionId, VersionBinding>) -> bool;

    spec fn pending_put_excludes_version(&self, version: VersionId) -> bool;

    spec fn pending_put_excludes_frame(&self, frame: FrameId) -> bool;

    spec fn poll_footprint(&self) -> CoreFootprint;

    spec fn caller_stream_valid(
        &self,
        stream: Self::Stream,
        core: &CacheCore,
        timeline: TimelineId,
    ) -> bool;

    spec fn sync_points_valid(&self) -> bool;

    spec fn frame_sync_point_spec(&self, frame: FrameId) -> SyncPoint;

    spec fn sync_point_ready(&self, core: &CacheCore, point: SyncPoint) -> bool;

    proof fn lemma_device_layout(&self, core: &CacheCore)
        requires
            self.well_formed(core),
        ensures
            device_slab_layout(
                core,
                self.buffer_spec(),
                self.device_spec(),
                self.device_tier_spec(),
                self.device_first_frame_spec(),
                self.device_num_frames_spec(),
            ),
            self.sync_points_valid(),
            self.has_pinned_host_spec() == self.pinned_host_layout_spec().is_some(),
            self.event_source_spec() == self.stream_spec().source().source(),
            !self.event_source_spec().is_host(),
            !self.event_source_spec().is_io_uring(),
            !self.event_source_spec().is_rdma_qp(),
            self.stream_spec().well_formed(),
            self.stream_spec().source_kind_valid(),
            self.stream_spec().device_spec() == self.device_spec(),
            self.stream_spec().source().id() == core.instance_id(),
            self.stream_spec().source().node() == core.node(),
            source_fresh_for_ledger(self.stream_spec().source(), core.ledger_spec()@),
            self.timeline_id_spec() < core.timelines_spec().count_spec() <= MAX_TIMELINES,
            core.timelines_spec().kinds_spec()[self.timeline_id_spec() as int]
                == self.stream_spec().timeline_kind_spec(),
    ;

    proof fn lemma_frame_ownership(&self, frame: FrameId)
        ensures
            self.owns_device(frame) == (self.device_first_frame_spec().0 <= frame.0 && frame.0
                < self.device_first_frame_spec().0 + self.device_num_frames_spec()),
            self.owns_pinned_host(frame) == (self.pinned_host_layout_spec() matches Some(
                (_, first, count),
            ) && first.0 <= frame.0 && frame.0 < first.0 + count),
    ;

    proof fn lemma_pinned_host_matches(&self, tier: TierId, first_frame: FrameId, num_frames: u32)
        ensures
            self.pinned_host_matches(tier, first_frame, num_frames) == (
            self.pinned_host_layout_spec() == Some((tier, first_frame, num_frames))),
    ;

    proof fn lemma_same_config(&self, other: &Self)
        ensures
            self.same_config(self),
            self.same_config(other) ==> {
                &&& self.device_spec() == other.device_spec()
                &&& self.device_tier_spec() == other.device_tier_spec()
                &&& self.device_first_frame_spec() == other.device_first_frame_spec()
                &&& self.device_num_frames_spec() == other.device_num_frames_spec()
                &&& self.timeline_id_spec() == other.timeline_id_spec()
                &&& self.has_pinned_host_spec() == other.has_pinned_host_spec()
                &&& self.pinned_host_layout_spec() == other.pinned_host_layout_spec()
            },
    ;

    proof fn lemma_pending_put_model(&self)
        ensures
            self.active_put_count_spec() == self.pending_puts_spec().len(),
            self.active_put_frames() == device_put_frames(self.pending_puts_spec()),
            self.poll_footprint() == device_poll_footprint(
                self.timeline_id_spec(),
                self.pending_puts_spec(),
            ),
            forall|v: VersionId| #[trigger]
                self.pending_put_excludes_version(v) <==> !device_put_versions(
                    self.pending_puts_spec(),
                ).contains(v),
            forall|f: FrameId| #[trigger]
                self.pending_put_excludes_frame(f) <==> !self.active_put_frames().contains(f),
    ;

    proof fn lemma_pending_put_bindings(&self, bindings: Map<VersionId, VersionBinding>)
        ensures
            self.pending_put_bindings_covered(bindings) == device_put_bindings_covered(
                self.pending_puts_spec(),
                bindings,
            ),
    ;

    proof fn lemma_pending_put_valid(&self, core: &CacheCore, i: int)
        requires
            self.well_formed(core),
            0 <= i < self.pending_puts_spec().len(),
        ensures
            ({
                let put = self.pending_puts_spec()[i];
                &&& put.frame.0 < core.frames_spec().metas_spec().len()
                &&& core.frames_spec().meta(put.frame).state == STATE_INITIALIZING
                &&& core.frames_spec().frame(put.frame).slot == put.slot
                &&& core.frames_spec().frame(put.frame).token is Some
                &&& put.publish matches Some(p) ==> {
                    &&& (p.only_if_current ==> core.frames_spec().versions_spec()@.contains_key(
                        p.version,
                    ))
                    &&& (p.version.origin_node != core.node_id_spec() || p.version.seq
                        < core.next_version_seq_spec() || p.version.seq
                        >= core.version_seq_limit_spec())
                }
            }),
    ;

    proof fn preserve_across_core_transition(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            self.well_formed(before),
            core_transition(before, after, fp),
            !fp.touched_timelines.contains(self.timeline_id_spec()),
            fp.excludes_source(self.event_source_spec()) || source_fresh_for_ledger(
                self.stream_spec().source(),
                after.ledger_spec()@,
            ),
            device_puts_untouched(self.pending_puts_spec(), before, fp),
        ensures
            self.well_formed(after),
    ;

    /// Preserve an idle borrowed backend across a transition that can import events.
    proof fn preserve_idle_across_core_transition(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            self.well_formed(before),
            core_transition(before, after, fp),
            !fp.touched_timelines.contains(self.timeline_id_spec()),
            self.active_put_count_spec() == 0,
            source_fresh_for_ledger(self.stream_spec().source(), after.ledger_spec()@),
        ensures
            self.well_formed(after),
    ;

    /// Check parked destinations using the tracked slot tokens retained by the core.
    proof fn prove_pending_destinations_preserved(
        &self,
        before: &CacheCore,
        tracked ghosts: &mut Map<nat, FrameGhost>,
        tracked ledger: &KnowledgeLedger,
        tracked instance: &TransferInstance,
    )
        requires
            before.well_formed(),
            self.well_formed(before),
            instance.id() == before.instance_id(),
            ledger.valid(before.instance_id(), before.node()),
            before.ledger_spec()@.submitted.map().submap_of(ledger.submitted.map()),
            revoked_submap(before.ledger_spec()@.revoked, ledger.revoked),
            before.ledger_spec()@.knows.set().subset_of(ledger.knows.set()),
            before.ledger_spec()@.succeeded.dom().subset_of(ledger.succeeded.dom()),
            forall|f: FrameId|
                f.0 < before.frames_spec().metas_spec().len() && (
                #[trigger] before.frames_spec().meta(f)).state == STATE_INITIALIZING ==> old(
                    ghosts,
                ).contains_key(f.0 as nat) && old(ghosts)[f.0 as nat] == before.frames_spec().frame(
                    f,
                ),
        ensures
            *final(ghosts) == *old(ghosts),
            device_put_destinations_preserved(self.pending_puts_spec(), before, ledger),
    ;

    /// Preserve a backend with pending publications across a peer import.
    proof fn preserve_across_core_transition_with_bindings(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        tracked registry: &Option<VersionRegistry>,
        tracked bindings: &Map<VersionId, VersionBinding>,
        fp: CoreFootprint,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            after.all_versions_bound(),
            *registry == after.version_registry_spec()@,
            self.well_formed(before),
            self.pending_put_bindings_covered(*bindings),
            core_transition(before, after, fp),
            !fp.touched_timelines.contains(self.timeline_id_spec()),
            fp.touched_init_frames.is_empty(),
            source_fresh_for_ledger(self.stream_spec().source(), after.ledger_spec()@),
            device_put_destinations_preserved(
                self.pending_puts_spec(),
                before,
                &after.ledger_spec()@,
            ),
        ensures
            self.well_formed(after),
            self.pending_put_bindings_covered(*bindings),
    ;

    proof fn lemma_caller_stream_valid(
        &self,
        stream: Self::Stream,
        core: &CacheCore,
        timeline: TimelineId,
    )
        ensures
            self.caller_stream_valid(stream, core, timeline) == valid_device_caller_stream(
                stream,
                core,
                timeline,
                self.event_source_spec(),
            ),
    ;

    proof fn preserve_caller_stream(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
        stream: Self::Stream,
        timeline: TimelineId,
    )
        requires
            core_transition(before, after, fp),
            self.caller_stream_valid(stream, before, timeline),
            fp.excludes_source(stream.source().source()),
        ensures
            self.caller_stream_valid(stream, after, timeline),
    ;

    fn matches_core_host<H: HostSlabBackend>(&self, core: &CacheCore, host: &H) -> (matches: bool)
        requires
            core.well_formed(),
            self.well_formed(core),
            host.host_well_formed(core),
        ensures
            matches ==> self.active_put_count_spec() == 0 && (self.has_pinned_host_spec()
                ==> self.pinned_host_matches(
                host.host_tier_spec(),
                host.host_first_frame_spec(),
                host.host_num_frames_spec(),
            )),
    ;

    fn is_caller_timeline(
        core: &CacheCore,
        timeline: TimelineId,
        stream: &Self::Stream,
    ) -> (matches: bool)
        ensures
            matches ==> timeline < core.timelines_spec().kinds_spec().len()
                && core.timelines_spec().kinds_spec()[timeline as int]
                == stream.timeline_kind_spec() && !(stream.timeline_kind_spec() is Host) && !(
            stream.timeline_kind_spec() is IoUring) && !(
            stream.timeline_kind_spec() is RdmaQueuePair),
    ;

    fn device(&self) -> (device: DeviceId)
        ensures
            device == self.device_spec(),
    ;

    fn device_tier_id(&self) -> (tier: TierId)
        ensures
            tier == self.device_tier_spec(),
    ;

    fn device_first_frame(&self) -> (frame: FrameId)
        ensures
            frame == self.device_first_frame_spec(),
    ;

    fn device_num_frames(&self) -> (count: u32)
        ensures
            count == self.device_num_frames_spec(),
    ;

    fn timeline_id(&self) -> (timeline: TimelineId)
        ensures
            timeline == self.timeline_id_spec(),
    ;

    fn stream(&self) -> (stream: &Self::Stream)
        ensures
            *stream == self.stream_spec(),
    ;

    fn device_buffer(&self) -> (buffer: &Self::Buffer)
        ensures
            *buffer == self.buffer_spec(),
    ;

    fn has_pinned_host(&self) -> (pinned: bool)
        ensures
            pinned == self.has_pinned_host_spec(),
    ;

    fn owns_device_frame(&self, frame: FrameId) -> (owns: bool)
        ensures
            owns == self.owns_device(frame),
    ;

    fn owns_pinned_host_frame(&self, frame: FrameId) -> (owns: bool)
        ensures
            owns == self.owns_pinned_host(frame),
    ;

    fn device_frame_slice<'s>(&'s self, frame: FrameId) -> (r: Option<Self::Slice<'s>>)
        requires
            self.buffer_spec().reg_well_formed(),
            self.buffer_spec().reg_num_slots() == self.device_num_frames_spec() as usize,
        ensures
            r.is_some() == self.owns_device(frame),
            r matches Some(slice) ==> {
                &&& slice.well_formed()
                &&& slice.instance_id() == self.buffer_spec().reg_instance_id()
                &&& slice.slot_ref() == SlotRef::full(
                    SlotLoc {
                        node: self.buffer_spec().reg_node(),
                        space: self.buffer_spec().reg_space(),
                        slot_index: (frame.0 - self.device_first_frame_spec().0) as nat,
                    },
                    self.buffer_spec().reg_slot_len() as u64,
                )
            },
    ;

    fn device_frame_addr(&self, frame: FrameId) -> (r: Option<usize>)
        requires
            self.buffer_spec().reg_well_formed(),
            self.buffer_spec().reg_num_slots() == self.device_num_frames_spec() as usize,
        ensures
            r.is_some() == self.owns_device(frame),
    {
        match self.device_frame_slice(frame) {
            Some(slice) => Some(slice.device_addr()),
            None => None,
        }
    }

    fn active_put_count(&self) -> (count: usize)
        ensures
            count == self.active_put_count_spec(),
    ;

    fn is_frame_ready(&self, core: &CacheCore, frame: FrameId) -> (ready: bool)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            ready == (frame.0 < core.frames_spec().metas_spec().len() && core.frames_spec().meta(
                frame,
            ).state == STATE_PUBLISHED && self.sync_point_ready(
                core,
                self.frame_sync_point_spec(frame),
            )),
    ;

    fn frame_sync_point(&self, frame: FrameId) -> (point: SyncPoint)
        requires
            self.sync_points_valid(),
        ensures
            point == self.frame_sync_point_spec(frame),
    ;

    fn is_sync_point_ready(&self, core: &CacheCore, point: SyncPoint) -> (ready: bool)
        requires
            core.well_formed(),
            self.well_formed(core),
        ensures
            ready == self.sync_point_ready(core, point),
    ;

    fn synchronize_caller_stream(
        stream: &Self::Stream,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<Tracked<Knows>>)
        requires
            stream.well_formed(),
            stream.source().id() == instance.id(),
        ensures
            r matches Ok(knows) ==> knows@.instance_id() == instance.id() && knows@.set()
                == knowledge_of(
                stream.source().node(),
                closure_from(stream.infos().map(), stream.tail()),
            ),
    ;

    /// Submit an asynchronous write. Its publication is deferred until polling.
    fn put_device<'s>(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &Self::Slice<'s>,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            src.well_formed(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().loc.node == old(core).node(),
            src.slot_ref().sub_len == old(core).frames_spec().tier_frame_size(
                old(self).device_tier_spec(),
            ),
            old(src_perm).valid_cooperative(old(core).instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(core).citations()),
            src.slot_ref().loc.space.is_gpu() && src.slot_ref().loc.space.gpu_device() != old(
                self,
            ).device_spec() ==> (old(src_perm).frontier().last_writer matches Some(w) ==> !(
            w.source.is_rdma_qp())),
            replica_valid(old(core).ledger_spec()@, old(core).node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(core).frames_spec().metas_spec().len() && (#[trigger] old(
                    core,
                ).frames_spec().meta(f)).state == STATE_EMPTY ==> src.slot_ref().loc != old(
                    core,
                ).frames_spec().frame(f).slot.loc,
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).same_config(old(self)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline_id_spec()),
                    Some(old(self).event_source_spec()),
                ),
            ),
            match r {
                Ok((version, frame, ticket, _)) => {
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == old(self).timeline_id_spec()
                    &&& version.origin_node == old(core).node_id_spec()
                    &&& version.seq == old(core).next_version_seq_spec()
                    &&& final(core).next_version_seq_spec() == old(core).next_version_seq_spec() + 1
                    &&& ((dst_tier == old(self).device_tier_spec() && final(self).owns_device(
                        frame,
                    )) || final(self).owns_pinned_host(frame))
                    &&& final(self).pending_puts_spec() == old(self).pending_puts_spec().push(
                        final(self).pending_puts_spec().last(),
                    )
                    &&& final(self).pending_puts_spec().last().publish == Some(
                        PendingDevicePublication {
                            version,
                            bytes: src_cert.bytes,
                            only_if_current: false,
                        },
                    )
                },
                Err(_) => final(self).pending_puts_spec() == old(self).pending_puts_spec() || (
                final(self).pending_puts_spec() == old(self).pending_puts_spec().push(
                    final(self).pending_puts_spec().last(),
                ) && final(self).pending_puts_spec().last().publish is None),
            },
    ;

    fn put_device_sync<'s>(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        src_stream: &mut Self::Stream,
        src_timeline: TimelineId,
        key: ObjectKey,
        src: &Self::Slice<'s>,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            self.caller_stream_valid(*old(src_stream), old(core), src_timeline),
            src_timeline != self.timeline_id_spec(),
            old(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1 == old(
                core,
            ).timelines_spec().next_seq_spec()[src_timeline as int],
            old(src_stream).device_spec() == self.device_spec(),
            src.slot_ref().loc.space.is_gpu() && src.slot_ref().loc.space.gpu_device()
                != self.device_spec() ==> (old(src_perm).frontier().last_writer matches Some(w)
                ==> !(w.source.is_rdma_qp())),
            instance.id() == old(core).instance_id(),
            src.well_formed(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().loc.node == old(core).node(),
            src.slot_ref().sub_len == old(core).frames_spec().tier_frame_size(
                self.device_tier_spec(),
            ),
            old(src_perm).valid_cooperative(old(core).instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(core).citations()),
            replica_valid(old(core).ledger_spec()@, old(core).node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(core).frames_spec().metas_spec().len() && (#[trigger] old(
                    core,
                ).frames_spec().meta(f)).state == STATE_EMPTY ==> src.slot_ref().loc != old(
                    core,
                ).frames_spec().frame(f).slot.loc,
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            self.caller_stream_valid(*final(src_stream), final(core), src_timeline),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(src_timeline),
                    Some(old(src_stream).source().source()),
                ),
            ),
            r.is_err() ==> final(core).frames_spec().versions_spec()@ == old(
                core,
            ).frames_spec().versions_spec()@,
            r matches Ok((ver, frame)) ==> {
                &&& self.owns_device(frame)
                &&& ver.origin_node == old(core).node_id_spec()
                &&& old(core).next_version_seq_spec() <= ver.seq
                &&& ver.seq < old(core).version_seq_limit_spec()
                &&& ver.seq < final(core).next_version_seq_spec()
                &&& final(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1
                    == final(core).timelines_spec().next_seq_spec()[src_timeline as int]
                &&& final(core).spec_spec()@.versions == old(core).spec_spec()@.versions.insert(
                    ver,
                    src_cert.bytes,
                )
                &&& final(core).spec_spec()@.catalog == old(core).spec_spec()@.catalog.insert(
                    key,
                    ver,
                )
                &&& final(core).frames_spec().meta(frame).state == STATE_PUBLISHED
                &&& final(core).frames_spec().versions_spec()@.contains_key(ver)
                &&& final(core).frames_spec().versions_spec()@.dom() == old(
                    core,
                ).frames_spec().versions_spec()@.dom().insert(ver)
            },
    ;

    fn put_device_scatter<'s>(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        srcs: &[Self::Slice<'s>],
        Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            device_scatter_offsets_valid(
                srcs@,
                offsets,
                old(core).frames_spec().tier_frame_size(old(self).device_tier_spec()),
            ),
            bytes.len() == old(core).frames_spec().tier_frame_size(old(self).device_tier_spec()),
            src_certs.len() == srcs@.len(),
            forall|i: int|
                0 <= i < srcs@.len() ==> {
                    &&& (#[trigger] srcs@[i]).instance_id() == old(core).instance_id()
                    &&& srcs@[i].slot_ref().loc.node == old(core).node()
                    &&& old(src_perms).dom().contains(i)
                    &&& old(src_perms)[i].valid_cooperative(
                        old(core).instance_id(),
                        srcs@[i].slot_ref().loc,
                    )
                    &&& read_admitted(old(src_perms)[i].frontier(), old(core).citations())
                    &&& (srcs@[i].slot_ref().loc.space.is_gpu()
                        && srcs@[i].slot_ref().loc.space.gpu_device() != old(self).device_spec()
                        ==> (old(src_perms)[i].frontier().last_writer matches Some(w) ==> !(
                    w.source.is_rdma_qp())))
                    &&& replica_valid(old(core).ledger_spec()@, old(core).node(), src_certs[i])
                    &&& source_slice_valid(src_certs[i], srcs@[i].slot_ref())
                    &&& src_certs[i].slot.well_formed()
                    &&& src_certs[i].bytes.len() == src_certs[i].slot.sub_len
                    &&& source_slice_bytes(src_certs[i], srcs@[i].slot_ref()) == bytes.subrange(
                        offsets[i] as int,
                        offsets[i + 1] as int,
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(core).frames_spec().metas_spec().len() && (#[trigger] old(
                            core,
                        ).frames_spec().meta(f)).state == STATE_EMPTY ==> srcs@[i].slot_ref().loc
                            != old(core).frames_spec().frame(f).slot.loc
                },
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).same_config(old(self)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline_id_spec()),
                    Some(old(self).event_source_spec()),
                ),
            ),
            r matches Ok((version, frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == old(self).timeline_id_spec()
                &&& version.origin_node == old(core).node_id_spec()
                &&& version.seq == old(core).next_version_seq_spec()
                &&& final(self).owns_device(frame)
            },
    ;

    fn get_into_device<'s>(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut Self::Stream,
        dst_timeline: TimelineId,
        dst: &Self::Slice<'s>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            self.caller_stream_valid(*old(dst_stream), old(core), dst_timeline),
            dst_timeline != self.timeline_id_spec(),
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            dst.well_formed(),
            dst.instance_id() == old(core).instance_id(),
            dst.slot_ref().loc.node == old(core).node(),
            dst.slot_ref().sub_len == old(core).frames_spec().tier_frame_size(
                self.device_tier_spec(),
            ),
            dst.slot_ref().loc.space.is_gpu() ==> dst.slot_ref().loc.space.gpu_device()
                == self.device_spec(),
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(core).citations()),
            old(core).disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(core).frames_spec().metas_spec().len() ==> dst.slot_ref().loc != (
                #[trigger] old(core).frames_spec().frame(f)).slot.loc,
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            self.caller_stream_valid(*final(dst_stream), final(core), dst_timeline),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(dst_timeline),
                    Some(old(dst_stream).source().source()),
                ),
            ),
            r matches Ok((version, ticket, ev)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == dst_timeline
                &&& final(core).spec_spec()@.versions.contains_key(version)
                &&& slot_owned(
                    *final(dst_tok),
                    old(core).instance_id(),
                    dst.slot_ref().loc,
                    Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                )
            },
    ;

    /// On partial submission failure, return the last ticket and the submitted slice count.
    fn get_into_device_scatter<'s>(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut Self::Stream,
        dst_timeline: TimelineId,
        dsts: &[Self::Slice<'s>],
        Tracked(dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(offsets): Ghost<Seq<usize>>,
    ) -> (r: Result<(VersionId, Ticket), (CoreError, Option<Ticket>, usize)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            self.caller_stream_valid(*old(dst_stream), old(core), dst_timeline),
            dst_timeline != self.timeline_id_spec(),
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
            device_scatter_offsets_valid(
                dsts@,
                offsets,
                old(core).frames_spec().tier_frame_size(self.device_tier_spec()),
            ),
            forall|i: int|
                0 <= i < dsts@.len() ==> {
                    &&& (#[trigger] dsts@[i]).instance_id() == old(core).instance_id()
                    &&& dsts@[i].slot_ref().loc.node == old(core).node()
                    &&& (dsts@[i].slot_ref().loc.space.is_gpu()
                        ==> dsts@[i].slot_ref().loc.space.gpu_device() == self.device_spec())
                    &&& old(dst_toks).dom().contains(i)
                    &&& slot_is_owned(
                        old(dst_toks)[i],
                        old(core).instance_id(),
                        dsts@[i].slot_ref().loc,
                    )
                    &&& write_admitted(slot_frontier(old(dst_toks)[i]), old(core).citations())
                    &&& old(core).disjoint_from_published(
                        ISet::empty().insert(dsts@[i].slot_ref().loc),
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(core).frames_spec().metas_spec().len() ==> dsts@[i].slot_ref().loc
                            != (#[trigger] old(core).frames_spec().frame(f)).slot.loc
                },
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            self.caller_stream_valid(*final(dst_stream), final(core), dst_timeline),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(dst_timeline),
                    Some(old(dst_stream).source().source()),
                ),
            ),
            r matches Ok((version, ticket)) ==> ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline && final(core).spec_spec()@.versions.contains_key(version),
            r matches Err((_, Some(ticket), completed_slices)) ==> completed_slices > 0
                && completed_slices <= dsts.len() && ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline,
            r matches Err((_, None, completed_slices)) ==> completed_slices == 0,
    ;

    fn copy_hop_device(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(FrameId, Ticket, SyncPoint)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(self).same_config(old(self)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).spec_spec() == old(core).spec_spec(),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(old(self).timeline_id_spec()),
                    Some(old(self).event_source_spec()),
                ),
            ),
            r matches Ok((dst_frame, ticket, _)) ==> {
                &&& ticket != Ticket::NONE
                &&& ticket.timeline_id() == old(self).timeline_id_spec()
                &&& ((dst_tier == old(self).device_tier_spec() && final(self).owns_device(
                    dst_frame,
                )) || final(self).owns_pinned_host(dst_frame))
            },
    ;

    fn copy_hop_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_stream: &mut Self::Stream,
        src_timeline: TimelineId,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            self.caller_stream_valid(*old(src_stream), old(core), src_timeline),
            src_timeline != self.timeline_id_spec(),
            old(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1 == old(
                core,
            ).timelines_spec().next_seq_spec()[src_timeline as int],
            old(src_stream).device_spec() == self.device_spec(),
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            self.caller_stream_valid(*final(src_stream), final(core), src_timeline),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(src_timeline),
                    Some(old(src_stream).source().source()),
                ),
            ),
            r matches Ok((ver, dst_frame)) ==> {
                &&& ((dst_tier == self.device_tier_spec() && self.owns_device(dst_frame))
                    || self.owns_pinned_host(dst_frame))
                &&& final(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1
                    == final(core).timelines_spec().next_seq_spec()[src_timeline as int]
                &&& final(core).frames_spec().meta(dst_frame).state == STATE_PUBLISHED
                &&& final(core).frames_spec().frame(dst_frame).tier == dst_tier
                &&& final(core).spec_spec()@.versions.contains_key(ver)
            },
    ;

    /// Return partial progress if polling fails after retiring any work.
    fn poll_completions(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<DevicePollReport, (CoreError, DevicePollReport)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).same_config(old(self)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is Some,
    ;

    fn poll_completions_with_bindings(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(pending_bindings): Tracked<&mut Map<VersionId, VersionBinding>>,
    ) -> (r: Result<DevicePollReport, (CoreError, DevicePollReport)>)
        requires
            old(core).well_formed(),
            old(core).all_versions_bound(),
            old(self).well_formed(old(core)),
            old(self).pending_put_bindings_covered(*old(pending_bindings)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(core).all_versions_bound(),
            final(self).well_formed(final(core)),
            final(self).pending_put_bindings_covered(*final(pending_bindings)),
            final(self).same_config(old(self)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is Some,
    ;

    fn synchronize_device(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<DevicePollReport>)
        requires
            old(core).well_formed(),
            old(self).well_formed(old(core)),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            final(self).well_formed(final(core)),
            final(self).same_config(old(self)),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(old(core), final(core), old(self).poll_footprint()),
            forall|f: FrameId| #[trigger]
                old(self).poll_footprint().touched_init_frames.contains(f) ==> old(
                    core,
                ).frames_spec().frame(f).token is Some,
    ;
}

pub open spec fn device_slab_layout<B: RdmaRegisterableSlab>(
    core: &CacheCore,
    buffer: B,
    device: DeviceId,
    tier: TierId,
    first_frame: FrameId,
    num_frames: u32,
) -> bool {
    &&& buffer.reg_well_formed()
    &&& buffer.reg_instance_id() == core.instance_id()
    &&& buffer.reg_node() == core.node()
    &&& buffer.reg_space().is_gpu()
    &&& buffer.reg_space().gpu_device() == device
    &&& buffer.reg_gpu_device() == Some(device)
    &&& buffer.reg_num_slots() == num_frames as usize
    &&& buffer.reg_slot_len() == core.frames_spec().tier_frame_size(tier)
    &&& tier.valid(core.tiers_spec().len())
    &&& core.tiers_spec()[tier.index() as int].kind == TierKind::GpuVram
    &&& slab_bound_to_core(core, tier, first_frame, num_frames, buffer.reg_space())
}

/// Attachment and ownership of the optional device tier.
pub enum DeviceAttachment<'a, D: DeviceBackend<'a>> {
    None,
    Borrowed(&'a D),
    Owned(D),
}

impl<'a, D: DeviceBackend<'a>> DeviceAttachment<'a, D> {
    pub open spec fn spec_ref(&self) -> Option<D> {
        match *self {
            Self::None => None,
            Self::Borrowed(device) => Some(*device),
            Self::Owned(device) => Some(device),
        }
    }

    pub open spec fn same_config(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::None, Self::None) => true,
            (Self::Borrowed(a), Self::Borrowed(b)) => a == b,
            (Self::Owned(a), Self::Owned(b)) => a.same_config(&b),
            _ => false,
        }
    }

    pub fn from_option_borrowed(opt: Option<&'a D>) -> (r: Self)
        ensures
            match opt {
                None => r is None && r.spec_ref() is None,
                Some(device) => r == Self::Borrowed(device) && r.spec_ref() == Some(*device),
            },
    {
        match opt {
            None => Self::None,
            Some(device) => Self::Borrowed(device),
        }
    }

    pub fn as_ref(&self) -> (r: Option<&D>)
        ensures
            match (self, r) {
                (Self::None, None) => self.spec_ref() is None,
                (Self::Borrowed(device), Some(borrowed)) => borrowed == *device && self.spec_ref()
                    == Some(*borrowed),
                (Self::Owned(device), Some(borrowed)) => borrowed == device && self.spec_ref()
                    == Some(*borrowed),
                _ => false,
            },
    {
        match self {
            Self::None => None,
            Self::Borrowed(device) => Some(*device),
            Self::Owned(device) => Some(device),
        }
    }

    pub fn device_tier_id(&self) -> (tier: Option<TierId>)
        ensures
            tier == match self.spec_ref() {
                Some(device) => Some(device.device_tier_spec()),
                None => None,
            },
    {
        match self.as_ref() {
            Some(device) => Some(device.device_tier_id()),
            None => None,
        }
    }

    pub fn take(&mut self) -> (r: Self)
        ensures
            r == *old(self),
            *final(self) is None,
    {
        let mut out = Self::None;
        core::mem::swap(self, &mut out);
        out
    }
}

/// An active read lease for ordered handoff from a device frame.
pub struct DeviceLease {
    pub handle: ReadLeaseHandle,
    pub version: VersionId,
    pub frame: FrameId,
    pub sync_point: SyncPoint,
    pub cap: Tracked<ShareCap>,
}

#[cfg(verus_only)]
fn no_device_attachment_smoke<'a>() {
    let attachment: DeviceAttachment<'a, NoDevice> = DeviceAttachment::None;
    let tier = attachment.device_tier_id();
    assert(tier is None);
}

} // verus!
