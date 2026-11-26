use super::*;
#[cfg(verus_only)]
use crate::backends::device::{
    device_poll_footprint, device_put_bindings_covered, device_put_destinations_preserved,
    device_put_frames, device_put_frames_untouched, device_put_versions,
    valid_device_caller_stream, PendingDevicePublication, PendingDeviceWrite,
};
use transfer::cuda::CudaStream;

verus! {

impl DeviceSlice for CudaSlice<'_> {
    open spec fn well_formed(&self) -> bool {
        CudaSlice::well_formed(self)
    }

    open spec fn slot_ref(&self) -> SlotRef {
        CudaSlice::slot_ref(self)
    }

    open spec fn instance_id(&self) -> InstanceId {
        CudaSlice::instance_id(self)
    }

    fn len(&self) -> usize {
        CudaSlice::len(self)
    }

    fn device_addr(&self) -> usize {
        CudaSlice::device_addr(self)
    }
}

impl DeviceStream for CudaStream {
    open spec fn well_formed(&self) -> bool {
        CudaStream::well_formed(self)
    }

    open spec fn source(&self) -> EventSourceToken {
        CudaStream::source(self)
    }

    open spec fn infos(&self) -> SubmittedMap {
        CudaStream::infos(self)
    }

    open spec fn tail(&self) -> ISet<TransferEvent> {
        CudaStream::tail(self)
    }

    open spec fn device_spec(&self) -> DeviceId {
        self.device()
    }

    open spec fn timeline_kind_spec(&self) -> TimelineKind {
        TimelineKind::CudaStream { device: self.device() }
    }

    open spec fn source_kind_valid(&self) -> bool {
        self.source().source().is_cuda_stream()
    }

    proof fn lemma_source_kind(&self) {
    }

    fn device_id(&self) -> DeviceId {
        CudaStream::device_id(self)
    }

    fn runtime_id(&self) -> u64 {
        CudaStream::runtime_id(self)
    }

    fn lemma_source_fresh(
        &self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(submitted): Tracked<&SubmittedMap>,
    ) {
        CudaStream::lemma_source_fresh(self, Tracked(instance), Tracked(submitted));
    }
}

pub open spec fn device_write(put: PendingGpuWrite) -> PendingDeviceWrite {
    PendingDeviceWrite {
        frame: put.frame,
        slot: put.slot@,
        publish: match put.action {
            PendingGpuAction::Publish { version, proof, only_if_current, .. } => Some(
                PendingDevicePublication {
                    version,
                    bytes: proof@.certificate.bytes,
                    only_if_current,
                },
            ),
            PendingGpuAction::Abort => None,
        },
    }
}

impl<'a> DeviceBackend<'a> for CudaBackend<'a> {
    type Buffer = CudaBuffer;

    type Stream = CudaStream;

    type Slice<'s> = CudaSlice<'s>;

    open spec fn well_formed(&self, core: &CacheCore) -> bool {
        CudaBackend::well_formed(self, core)
    }

    open spec fn same_config(&self, other: &Self) -> bool {
        &&& self.device_spec() == other.device_spec()
        &&& self.gpu_tier_spec() == other.gpu_tier_spec()
        &&& self.gpu_first_frame_spec() == other.gpu_first_frame_spec()
        &&& self.gpu_num_frames_spec() == other.gpu_num_frames_spec()
        &&& self.same_pinned_layout(other)
        &&& self.timeline_spec().timeline_id_spec() == other.timeline_spec().timeline_id_spec()
    }

    open spec fn device_spec(&self) -> DeviceId {
        CudaBackend::device_spec(self)
    }

    open spec fn device_tier_spec(&self) -> TierId {
        self.gpu_tier_spec()
    }

    open spec fn device_first_frame_spec(&self) -> FrameId {
        self.gpu_first_frame_spec()
    }

    open spec fn device_num_frames_spec(&self) -> u32 {
        self.gpu_num_frames_spec()
    }

    open spec fn timeline_id_spec(&self) -> TimelineId {
        self.timeline_spec().timeline_id_spec()
    }

    open spec fn event_source_spec(&self) -> EventSource {
        self.stream_spec().source().source()
    }

    open spec fn stream_spec(&self) -> Self::Stream {
        self.timeline_spec().stream_spec()
    }

    open spec fn buffer_spec(&self) -> Self::Buffer {
        self.gpu_buffer_spec()
    }

    open spec fn has_pinned_host_spec(&self) -> bool {
        self.pinned_host_spec().is_some()
    }

    open spec fn pinned_host_layout_spec(&self) -> Option<(TierId, FrameId, u32)> {
        match self.pinned_host_spec() {
            Some(host) => Some((host.tier_spec(), host.first_frame_spec(), host.num_frames_spec())),
            None => None,
        }
    }

    open spec fn pinned_host_matches(
        &self,
        tier: TierId,
        first_frame: FrameId,
        num_frames: u32,
    ) -> bool {
        self.pinned_host_layout_spec() == Some((tier, first_frame, num_frames))
    }

    open spec fn owns_device(&self, frame: FrameId) -> bool {
        self.owns_gpu(frame)
    }

    open spec fn owns_pinned_host(&self, frame: FrameId) -> bool {
        CudaBackend::owns_pinned_host(self, frame)
    }

    open spec fn active_put_count_spec(&self) -> usize {
        self.active_gpu_puts_spec().len()
    }

    open spec fn active_put_frames(&self) -> ISet<FrameId> {
        device_put_frames(self.pending_puts_spec())
    }

    open spec fn pending_puts_spec(&self) -> Seq<PendingDeviceWrite> {
        let puts = self.active_gpu_puts_spec()@;
        Seq::new(puts.len(), |i: int| device_write(puts[i]))
    }

    open spec fn pending_put_bindings_covered(
        &self,
        bindings: Map<VersionId, VersionBinding>,
    ) -> bool {
        cuda_puts_bindings_covered(self.active_gpu_puts_spec()@, bindings)
    }

    open spec fn pending_put_excludes_version(&self, version: VersionId) -> bool {
        !device_put_versions(self.pending_puts_spec()).contains(version)
    }

    open spec fn pending_put_excludes_frame(&self, frame: FrameId) -> bool {
        !self.active_put_frames().contains(frame)
    }

    open spec fn poll_footprint(&self) -> CoreFootprint {
        CudaBackend::poll_footprint(self)
    }

    open spec fn caller_stream_valid(
        &self,
        stream: Self::Stream,
        core: &CacheCore,
        timeline: TimelineId,
    ) -> bool {
        caller_stream_valid(stream, core, timeline, self.stream_spec())
    }

    open spec fn sync_points_valid(&self) -> bool {
        &&& self.gpu_sync_points_spec().len() == self.gpu_num_frames_spec() as usize
        &&& self.pinned_host_spec() matches Some(h) ==> h.sync_points_spec().len()
            == h.num_frames_spec() as usize
    }

    open spec fn frame_sync_point_spec(&self, frame: FrameId) -> SyncPoint {
        CudaBackend::frame_sync_point_spec(self, frame)
    }

    open spec fn sync_point_ready(&self, core: &CacheCore, point: SyncPoint) -> bool {
        self.timeline_spec().sync_point_ready(core, point)
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
        broadcast use vstd::iset::group_iset_lemmas;

        let puts = self.active_gpu_puts@;
        let model = self.pending_puts_spec();
        let concrete = CudaBackend::poll_footprint(self);
        let generic = device_poll_footprint(self.timeline_id_spec(), model);
        let concrete_frames = concrete.touched_init_frames;
        let generic_frames = generic.touched_init_frames;
        assert(concrete_frames =~= generic_frames) by {
            assert forall|frame: FrameId| #[trigger]
                concrete_frames.contains(frame) implies generic_frames.contains(frame) by {
                let i = choose|i: int| 0 <= i < puts.len() && (#[trigger] puts[i]).frame == frame;
                assert(model[i].frame == frame);
            }
            assert forall|frame: FrameId| #[trigger]
                generic_frames.contains(frame) implies concrete_frames.contains(frame) by {
                let i = choose|i: int| 0 <= i < model.len() && (#[trigger] model[i]).frame == frame;
                assert(puts[i].frame == frame);
            }
        }
        let concrete_versions = concrete.versions.specific.unwrap();
        let generic_versions = generic.versions.specific.unwrap();
        assert(concrete_versions =~= generic_versions) by {
            assert forall|version: VersionId| #[trigger]
                concrete_versions.contains(version) implies generic_versions.contains(version) by {
                let i = choose|i: int|
                    0 <= i < puts.len() && (#[trigger] puts[i]).action.publishes_version(version);
                assert(model[i].publishes_version(version));
            }
            assert forall|version: VersionId| #[trigger]
                generic_versions.contains(version) implies concrete_versions.contains(version) by {
                let i = choose|i: int|
                    0 <= i < model.len() && (#[trigger] model[i]).publishes_version(version);
                assert(puts[i].action.publishes_version(version));
            }
        }
    }

    proof fn lemma_pending_put_bindings(&self, bindings: Map<VersionId, VersionBinding>) {
        reveal(cuda_puts_bindings_covered);
        let puts = self.active_gpu_puts@;
        let model = self.pending_puts_spec();
        if cuda_puts_bindings_covered(puts, bindings) {
            assert forall|i: int| 0 <= i < model.len() implies match (#[trigger] model[i]).publish {
                Some(PendingDevicePublication { version, bytes, only_if_current: false }) => {
                    &&& bindings.contains_key(version)
                    &&& bindings[version].instance_id() == cluster_version_instance_id()
                    &&& bindings[version].key() == (version.origin_node as u64, version.seq)
                    &&& bindings[version].value() == bytes
                },
                _ => true,
            } by {
                assert(model[i] == device_write(puts[i]));
            }
        }
        if device_put_bindings_covered(model, bindings) {
            assert forall|i: int| 0 <= i < puts.len() implies match (#[trigger] puts[i]).action {
                PendingGpuAction::Publish { version, proof, only_if_current: false, .. } => {
                    &&& bindings.contains_key(version)
                    &&& bindings[version].instance_id() == cluster_version_instance_id()
                    &&& bindings[version].key() == (version.origin_node as u64, version.seq)
                    &&& bindings[version].value() == proof@.certificate.bytes
                },
                _ => true,
            } by {
                assert(model[i] == device_write(puts[i]));
            }
        }
    }

    proof fn lemma_pending_put_valid(&self, core: &CacheCore, i: int) {
        reveal(cuda_puts_valid);
        assert(self.active_gpu_puts@[i].valid(core, self.timeline.timeline_id));
    }

    proof fn preserve_across_core_transition(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
    ) {
        let puts = self.active_gpu_puts@;
        let model = self.pending_puts_spec();
        if !fp.touched_init_frames.is_empty() {
            assert(self.put_frames_untouched(before, fp.touched_init_frames)) by {
                assert forall|i: int|
                    0 <= i < puts.len() && !(forall|f: FrameId| #[trigger]
                        fp.touched_init_frames.contains(f) && f.0
                            < before.frames_spec().metas_spec().len()
                            ==> before.frames_spec().frame(
                            f,
                        ).token is None) implies !fp.touched_init_frames.contains(
                    (#[trigger] puts[i]).frame,
                ) by {
                    assert(model[i] == device_write(puts[i]));
                }
            }
        }
        if !fp.written_init_frames.is_empty() {
            assert(self.put_frames_untouched(before, fp.written_init_frames)) by {
                assert forall|i: int|
                    0 <= i < puts.len() && !(forall|f: FrameId| #[trigger]
                        fp.written_init_frames.contains(f) && f.0
                            < before.frames_spec().metas_spec().len()
                            ==> before.frames_spec().frame(
                            f,
                        ).token is None) implies !fp.written_init_frames.contains(
                    (#[trigger] puts[i]).frame,
                ) by {
                    assert(model[i] == device_write(puts[i]));
                }
            }
        }
        let versions = fp.versions.specific.unwrap();
        if !versions.is_empty() {
            assert forall|i: int, v: VersionId|
                0 <= i < puts.len() && (#[trigger] puts[i].action.publishes_version(
                    v,
                )) implies !versions.contains(v) by {
                assert(model[i] == device_write(puts[i]));
                assert(model[i].publishes_version(v));
            }
        }
        lemma_core_transition_preserves_cuda(before, after, fp, self);
    }

    proof fn preserve_idle_across_core_transition(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
    ) {
        lemma_core_transition_preserves_idle_cuda(before, after, fp, self);
    }

    proof fn prove_pending_destinations_preserved(
        &self,
        before: &CacheCore,
        tracked ghosts: &mut Map<nat, FrameGhost>,
        tracked ledger: &KnowledgeLedger,
        tracked instance: &TransferInstance,
    ) {
        let puts = self.active_gpu_puts@;
        lemma_prove_cuda_puts_dest_preserved(
            before,
            ghosts,
            ledger,
            instance,
            puts,
            self.timeline.timeline_id,
            puts.len(),
        );
        let model = self.pending_puts_spec();
        assert forall|i: int|
            0 <= i < model.len() && (
            #[trigger] model[i]).publish is Some implies preserves_destination(
            before.ledger_spec()@,
            *ledger,
            before.node(),
            model[i].slot.loc,
        ) by {
            assert(model[i] == device_write(puts[i]));
        }
    }

    proof fn preserve_across_core_transition_with_bindings(
        &self,
        before: &CacheCore,
        after: &CacheCore,
        tracked registry: &Option<VersionRegistry>,
        tracked bindings: &Map<VersionId, VersionBinding>,
        fp: CoreFootprint,
    ) {
        let puts = self.active_gpu_puts@;
        let model = self.pending_puts_spec();
        assert forall|i: int|
            0 <= i < puts.len() && (
            #[trigger] puts[i]).action is Publish implies preserves_destination(
            before.ledger_spec()@,
            after.ledger_spec()@,
            after.node(),
            puts[i].slot@.loc,
        ) by {
            reveal(core_transition);
            assert(model[i] == device_write(puts[i]));
        }
        lemma_core_transition_preserves_cuda_after_import(
            before,
            after,
            registry,
            bindings,
            fp,
            self,
        );
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
        lemma_core_transition_preserves_caller_stream(
            before,
            after,
            fp,
            stream,
            timeline,
            self.stream_spec(),
        );
    }

    fn matches_core_host<H: HostSlabBackend>(&self, core: &CacheCore, host: &H) -> bool {
        let kinds = core.timelines().kinds();
        let timeline = self.timeline_id() as usize;
        if self.core_id != core.runtime_id() || timeline >= kinds.len() || self.active_put_count()
            != 0 {
            return false;
        }
        let kind = kinds[timeline];
        if !matches!(kind, TimelineKind::CudaStream { device } if device == self.device)
            || !core.matches_slab(
            self.gpu_tier,
            self.gpu_first_frame,
            self.gpu_num_frames,
            self.gpu_buffer.slot_len_val() as u64,
        ) {
            return false;
        }
        if let Some(pinned) = &self.pinned_host {
            pinned.tier.0 == host.host_tier_id().0 && pinned.first_frame.0
                == host.host_first_frame().0 && pinned.num_frames == host.host_num_frames()
        } else {
            true
        }
    }

    fn is_caller_timeline(core: &CacheCore, timeline: TimelineId, stream: &Self::Stream) -> bool {
        let index = timeline as usize;
        let kinds = core.timelines().kinds();
        if index >= kinds.len() {
            return false;
        }
        let kind = kinds[index];
        matches!(kind, TimelineKind::CudaStream { device } if device == stream.device_id())
    }

    fn device(&self) -> DeviceId {
        CudaBackend::device(self)
    }

    fn device_tier_id(&self) -> TierId {
        self.gpu_tier_id()
    }

    fn device_first_frame(&self) -> FrameId {
        self.gpu_first_frame()
    }

    fn device_num_frames(&self) -> u32 {
        self.gpu_num_frames()
    }

    fn timeline_id(&self) -> TimelineId {
        CudaBackend::timeline_id(self)
    }

    fn stream(&self) -> &Self::Stream {
        CudaBackend::stream(self)
    }

    fn device_buffer(&self) -> &Self::Buffer {
        self.gpu_buffer()
    }

    fn has_pinned_host(&self) -> bool {
        CudaBackend::has_pinned_host(self)
    }

    fn owns_device_frame(&self, frame: FrameId) -> bool {
        self.owns_gpu_frame(frame)
    }

    fn owns_pinned_host_frame(&self, frame: FrameId) -> bool {
        CudaBackend::owns_pinned_host_frame(self, frame)
    }

    fn device_frame_slice<'s>(&'s self, frame: FrameId) -> Option<Self::Slice<'s>> {
        if self.owns_gpu_frame(frame) {
            Some(self.gpu_frame_slice(frame))
        } else {
            None
        }
    }

    fn active_put_count(&self) -> usize {
        CudaBackend::active_put_count(self)
    }

    fn is_frame_ready(&self, core: &CacheCore, frame: FrameId) -> bool {
        CudaBackend::is_frame_ready(self, core, frame)
    }

    fn frame_sync_point(&self, frame: FrameId) -> SyncPoint {
        CudaBackend::frame_sync_point(self, frame)
    }

    fn is_sync_point_ready(&self, core: &CacheCore, point: SyncPoint) -> bool {
        proof {
            lemma_core_wf_facts(core);
        }
        self.timeline.is_sync_point_ready(core, point)
    }

    fn synchronize_caller_stream(
        stream: &Self::Stream,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> BackendResult<Tracked<Knows>> {
        match cuda::stream_synchronize(stream, Tracked(instance)) {
            Ok(knows) => Ok(knows),
            Err(error) => Err(cuda_error(error)),
        }
    }

    fn put_device<'s>(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &Self::Slice<'s>,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> BackendResult<(VersionId, FrameId, Ticket, SyncPoint)> {
        let ghost before = self.active_gpu_puts@;
        let ghost before_model = self.pending_puts_spec();
        let result = CudaBackend::put_gpu(
            self,
            core,
            Tracked(instance),
            dst_tier,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        );
        proof {
            let after = self.active_gpu_puts@;
            let after_model = self.pending_puts_spec();
            if after.len() == before.len() + 1 {
                assert(after_model =~= before_model.push(after_model.last()));
            }
        }
        result
    }

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
    ) -> BackendResult<(VersionId, FrameId)> {
        CudaBackend::put_gpu_sync(
            self,
            core,
            Tracked(instance),
            src_stream,
            src_timeline,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        )
    }

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
    ) -> BackendResult<(VersionId, FrameId, Ticket, SyncPoint)> {
        CudaBackend::put_gpu_scatter(
            self,
            core,
            Tracked(instance),
            key,
            srcs,
            Tracked(src_perms),
            Ghost(src_certs),
            Ghost(offsets),
            Ghost(bytes),
        )
    }

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
    ) -> BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)> {
        CudaBackend::get_into_gpu(
            self,
            core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst_stream,
            dst_timeline,
            dst,
            Tracked(dst_tok),
        )
    }

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
    ) -> Result<(VersionId, Ticket), (CoreError, Option<Ticket>, usize)> {
        CudaBackend::get_into_gpu_scatter(
            self,
            core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst_stream,
            dst_timeline,
            dsts,
            Tracked(dst_toks),
            Ghost(offsets),
        )
    }

    fn copy_hop_device(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> BackendResult<(FrameId, Ticket, SyncPoint)> {
        CudaBackend::copy_hop_cuda(
            self,
            core,
            Tracked(instance),
            Tracked(shares),
            src_frame,
            dst_tier,
        )
    }

    fn copy_hop_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_stream: &mut Self::Stream,
        src_timeline: TimelineId,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> BackendResult<(VersionId, FrameId)> {
        CudaBackend::copy_hop_sync(
            self,
            core,
            Tracked(instance),
            Tracked(shares),
            src_stream,
            src_timeline,
            src_frame,
            dst_tier,
        )
    }

    fn poll_completions(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> Result<DevicePollReport, (CoreError, DevicePollReport)> {
        CudaBackend::poll_completions(self, core, Tracked(instance))
    }

    fn poll_completions_with_bindings(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(pending_bindings): Tracked<&mut Map<VersionId, VersionBinding>>,
    ) -> Result<DevicePollReport, (CoreError, DevicePollReport)> {
        CudaBackend::poll_completions_with_bindings(
            self,
            core,
            Tracked(instance),
            Tracked(pending_bindings),
        )
    }

    fn synchronize_device(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> BackendResult<DevicePollReport> {
        CudaBackend::synchronize(self, core, Tracked(instance))
    }
}

} // verus!
