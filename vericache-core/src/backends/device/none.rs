use super::*;
#[cfg(verus_only)]
use transfer_core::domain::{MemorySpace, NodeId};
use transfer_core::memory::DmaAllocation;

verus! {

/// Selects a host-only engine through `DeviceAttachment::None`.
pub struct NoDevice;

pub struct NoDeviceBuffer;

pub struct NoDeviceStream;

pub struct NoDeviceSlice;

impl RdmaRegisterableSlab for NoDeviceBuffer {
    open spec fn reg_node(&self) -> NodeId {
        0
    }

    open spec fn reg_space(&self) -> MemorySpace {
        MemorySpace::host(0)
    }

    open spec fn reg_num_slots(&self) -> usize {
        0
    }

    open spec fn reg_slot_len(&self) -> usize {
        0
    }

    open spec fn reg_total_len(&self) -> usize {
        0
    }

    open spec fn reg_instance_id(&self) -> InstanceId {
        arbitrary()
    }

    open spec fn reg_well_formed(&self) -> bool {
        false
    }

    open spec fn reg_gpu_device(&self) -> Option<DeviceId> {
        None
    }

    fn reg_allocation(&self) -> &DmaAllocation {
        &DmaAllocation::UNAVAILABLE
    }

    fn reg_num_slots_exec(&self) -> usize {
        0
    }

    fn reg_slot_len_exec(&self) -> usize {
        0
    }
}

impl DeviceStream for NoDeviceStream {
    open spec fn well_formed(&self) -> bool {
        false
    }

    open spec fn source(&self) -> EventSourceToken {
        arbitrary()
    }

    open spec fn infos(&self) -> SubmittedMap {
        arbitrary()
    }

    open spec fn tail(&self) -> ISet<TransferEvent> {
        ISet::empty()
    }

    open spec fn device_spec(&self) -> DeviceId {
        0
    }

    open spec fn timeline_kind_spec(&self) -> TimelineKind {
        TimelineKind::Custom(0)
    }

    open spec fn source_kind_valid(&self) -> bool {
        false
    }

    proof fn lemma_source_kind(&self) {
    }

    fn device_id(&self) -> DeviceId {
        0
    }

    fn runtime_id(&self) -> u64 {
        0
    }

    fn lemma_source_fresh(
        &self,
        Tracked(_instance): Tracked<&TransferInstance>,
        Tracked(_submitted): Tracked<&SubmittedMap>,
    ) {
    }
}

impl DeviceSlice for NoDeviceSlice {
    open spec fn well_formed(&self) -> bool {
        false
    }

    open spec fn slot_ref(&self) -> SlotRef {
        arbitrary()
    }

    open spec fn instance_id(&self) -> InstanceId {
        arbitrary()
    }

    fn len(&self) -> usize {
        0
    }

    fn device_addr(&self) -> usize {
        0
    }
}

impl<'a> DeviceBackend<'a> for NoDevice {
    type Buffer = NoDeviceBuffer;

    type Stream = NoDeviceStream;

    type Slice<'s> = NoDeviceSlice;

    open spec fn well_formed(&self, core: &CacheCore) -> bool {
        false
    }

    open spec fn same_config(&self, other: &Self) -> bool {
        true
    }

    open spec fn device_spec(&self) -> DeviceId {
        0
    }

    open spec fn device_tier_spec(&self) -> TierId {
        TierId(0)
    }

    open spec fn device_first_frame_spec(&self) -> FrameId {
        FrameId(0)
    }

    open spec fn device_num_frames_spec(&self) -> u32 {
        0
    }

    open spec fn timeline_id_spec(&self) -> TimelineId {
        0
    }

    open spec fn event_source_spec(&self) -> EventSource {
        self.stream_spec().source().source()
    }

    open spec fn stream_spec(&self) -> Self::Stream {
        NoDeviceStream
    }

    open spec fn buffer_spec(&self) -> Self::Buffer {
        NoDeviceBuffer
    }

    open spec fn has_pinned_host_spec(&self) -> bool {
        false
    }

    open spec fn pinned_host_layout_spec(&self) -> Option<(TierId, FrameId, u32)> {
        None
    }

    open spec fn pinned_host_matches(
        &self,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
    ) -> bool {
        false
    }

    open spec fn owns_device(&self, frame: FrameId) -> bool {
        false
    }

    open spec fn owns_pinned_host(&self, frame: FrameId) -> bool {
        false
    }

    open spec fn active_put_count_spec(&self) -> usize {
        0
    }

    open spec fn active_put_frames(&self) -> ISet<FrameId> {
        device_put_frames(self.pending_puts_spec())
    }

    open spec fn pending_puts_spec(&self) -> Seq<PendingDeviceWrite> {
        Seq::empty()
    }

    open spec fn pending_put_bindings_covered(
        &self,
        bindings: Map<VersionId, VersionBinding>,
    ) -> bool {
        device_put_bindings_covered(self.pending_puts_spec(), bindings)
    }

    open spec fn pending_put_excludes_version(&self, version: VersionId) -> bool {
        !device_put_versions(self.pending_puts_spec()).contains(version)
    }

    open spec fn pending_put_excludes_frame(&self, frame: FrameId) -> bool {
        !self.active_put_frames().contains(frame)
    }

    open spec fn poll_footprint(&self) -> CoreFootprint {
        device_poll_footprint(self.timeline_id_spec(), self.pending_puts_spec())
    }

    open spec fn caller_stream_valid(
        &self,
        stream: Self::Stream,
        core: &CacheCore,
        timeline: TimelineId,
    ) -> bool {
        false
    }

    open spec fn sync_points_valid(&self) -> bool {
        true
    }

    open spec fn frame_sync_point_spec(&self, frame: FrameId) -> SyncPoint {
        SyncPoint::HostComplete
    }

    open spec fn sync_point_ready(&self, core: &CacheCore, point: SyncPoint) -> bool {
        point is HostComplete
    }

    proof fn lemma_device_layout(&self, core: &CacheCore) {
    }

    proof fn lemma_frame_ownership(&self, frame: FrameId) {
    }

    proof fn lemma_pinned_host_matches(&self, tier: TierId, first_frame: FrameId, num_frames: u32) {
    }

    proof fn lemma_same_config(&self, other: &Self) {
    }

    proof fn lemma_pending_put_model(&self) {
    }

    proof fn lemma_pending_put_bindings(&self, bindings: Map<VersionId, VersionBinding>) {
    }

    proof fn lemma_pending_put_valid(&self, core: &CacheCore, i: int) {
    }

    proof fn preserve_across_core_transition(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
    ) {
    }

    proof fn preserve_idle_across_core_transition(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
    ) {
    }

    proof fn prove_pending_destinations_preserved(
        &self,
        before: &CacheCore,
        tracked ghosts: &mut Map<nat, FrameGhost>,
        tracked ledger: &KnowledgeLedger,
        tracked instance: &TransferInstance,
    ) {
    }

    proof fn preserve_across_core_transition_with_bindings(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        tracked registry: &Option<VersionRegistry>,
        tracked bindings: &Map<VersionId, VersionBinding>,
        fp: CoreFootprint,
    ) {
    }

    proof fn lemma_caller_stream_valid(
        &self,
        stream: Self::Stream,
        core: &CacheCore,
        timeline: TimelineId,
    ) {
    }

    proof fn preserve_caller_stream(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
        stream: Self::Stream,
        timeline: TimelineId,
    ) {
    }

    fn matches_core_host<H: HostSlabBackend>(&self, _core: &CacheCore, _host: &H) -> bool {
        false
    }

    fn is_caller_timeline(
        _core: &CacheCore,
        _timeline: TimelineId,
        _stream: &Self::Stream,
    ) -> bool {
        false
    }

    fn device(&self) -> DeviceId {
        0
    }

    fn device_tier_id(&self) -> TierId {
        TierId(0)
    }

    fn device_first_frame(&self) -> FrameId {
        FrameId(0)
    }

    fn device_num_frames(&self) -> u32 {
        0
    }

    fn timeline_id(&self) -> TimelineId {
        0
    }

    fn stream(&self) -> &Self::Stream {
        &NoDeviceStream
    }

    fn device_buffer(&self) -> &Self::Buffer {
        &NoDeviceBuffer
    }

    fn has_pinned_host(&self) -> bool {
        false
    }

    fn owns_device_frame(&self, _frame: FrameId) -> bool {
        false
    }

    fn owns_pinned_host_frame(&self, _frame: FrameId) -> bool {
        false
    }

    fn device_frame_slice<'s>(&'s self, _frame: FrameId) -> Option<Self::Slice<'s>> {
        None
    }

    fn active_put_count(&self) -> usize {
        0
    }

    fn is_frame_ready(&self, _core: &CacheCore, _frame: FrameId) -> bool {
        false
    }

    fn frame_sync_point(&self, _frame: FrameId) -> SyncPoint {
        SyncPoint::HostComplete
    }

    fn is_sync_point_ready(&self, _core: &CacheCore, point: SyncPoint) -> bool {
        matches!(point, SyncPoint::HostComplete)
    }

    fn synchronize_caller_stream(
        _stream: &Self::Stream,
        Tracked(_instance): Tracked<&TransferInstance>,
    ) -> BackendResult<Tracked<Knows>> {
        Err(CoreError::UnsupportedTier)
    }

    fn put_device<'s>(
        &mut self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        _dst_tier: TierId,
        _key: ObjectKey,
        _src: &Self::Slice<'s>,
        Tracked(_src_perm): Tracked<&mut ReadPerm>,
        Ghost(_src_cert): Ghost<ValueCertificate>,
    ) -> BackendResult<(VersionId, FrameId, Ticket, SyncPoint)> {
        Err(CoreError::UnsupportedTier)
    }

    fn put_device_sync<'s>(
        &self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        _src_stream: &mut Self::Stream,
        _src_timeline: TimelineId,
        _key: ObjectKey,
        _src: &Self::Slice<'s>,
        Tracked(_src_perm): Tracked<&mut ReadPerm>,
        Ghost(_src_cert): Ghost<ValueCertificate>,
    ) -> BackendResult<(VersionId, FrameId)> {
        Err(CoreError::UnsupportedTier)
    }

    fn put_device_scatter<'s>(
        &mut self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        _key: ObjectKey,
        _srcs: &[Self::Slice<'s>],
        Tracked(_src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Ghost(_src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(_offsets): Ghost<Seq<usize>>,
        Ghost(_bytes): Ghost<Seq<u8>>,
    ) -> BackendResult<(VersionId, FrameId, Ticket, SyncPoint)> {
        Err(CoreError::UnsupportedTier)
    }

    fn get_into_device<'s>(
        &self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        Tracked(_shares): Tracked<&mut ShareSource>,
        _frame: FrameId,
        _dst_stream: &mut Self::Stream,
        _dst_timeline: TimelineId,
        _dst: &Self::Slice<'s>,
        Tracked(_dst_tok): Tracked<&mut SlotToken>,
    ) -> BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)> {
        Err(CoreError::UnsupportedTier)
    }

    fn get_into_device_scatter<'s>(
        &self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        Tracked(_shares): Tracked<&mut ShareSource>,
        _frame: FrameId,
        _dst_stream: &mut Self::Stream,
        _dst_timeline: TimelineId,
        _dsts: &[Self::Slice<'s>],
        Tracked(_dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(_offsets): Ghost<Seq<usize>>,
    ) -> Result<(VersionId, Ticket), (CoreError, Option<Ticket>, usize)> {
        Err((CoreError::UnsupportedTier, None, 0))
    }

    fn copy_hop_device(
        &mut self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        Tracked(_shares): Tracked<&mut ShareSource>,
        _src_frame: FrameId,
        _dst_tier: TierId,
    ) -> BackendResult<(FrameId, Ticket, SyncPoint)> {
        Err(CoreError::UnsupportedTier)
    }

    fn copy_hop_sync(
        &self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        Tracked(_shares): Tracked<&mut ShareSource>,
        _src_stream: &mut Self::Stream,
        _src_timeline: TimelineId,
        _src_frame: FrameId,
        _dst_tier: TierId,
    ) -> BackendResult<(VersionId, FrameId)> {
        Err(CoreError::UnsupportedTier)
    }

    fn poll_completions(
        &mut self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
    ) -> Result<DevicePollReport, (CoreError, DevicePollReport)> {
        Err((CoreError::UnsupportedTier, DevicePollReport::empty()))
    }

    fn poll_completions_with_bindings(
        &mut self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
        Tracked(_pending_bindings): Tracked<&mut Map<VersionId, VersionBinding>>,
    ) -> Result<DevicePollReport, (CoreError, DevicePollReport)> {
        Err((CoreError::UnsupportedTier, DevicePollReport::empty()))
    }

    fn synchronize_device(
        &mut self,
        _core: &mut CacheCore,
        Tracked(_instance): Tracked<&TransferInstance>,
    ) -> BackendResult<DevicePollReport> {
        Err(CoreError::UnsupportedTier)
    }
}

} // verus!
