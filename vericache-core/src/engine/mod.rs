use crate::backends::cuda::CudaBackend;
use crate::backends::host::HostBackend;
use crate::backends::nvme::NvmeBackend;
use crate::backends::*;
#[cfg(verus_only)]
use crate::frame::*;
use crate::kernel::*;
use crate::spec::*;
#[cfg(verus_only)]
use crate::timeline::*;
use transfer::context::*;
use transfer::cuda::CudaSlice;
#[cfg(verus_only)]
use transfer::domain::EventSource;
use transfer::host::{HostContext, HostSlice};
#[cfg(verus_only)]
use transfer::nvme::DIRECT_IO_ALIGNMENT;
use vstd::prelude::*;

pub mod distributed;
mod gpu_routes;
mod routes;

pub use distributed::{CudaAttachment, DistributedEngine, DistributedEvent, PendingReadGrant};

verus! {

/// Active local Host-DRAM read lease borrowing directly from the outer
/// `HostBackend` slab (`'a`), so multiple concurrent leases coexist with
/// `&mut LocalEngine<'a>` operations.
pub struct HostLease<'a> {
    pub handle: ReadLeaseHandle,
    pub version: VersionId,
    pub slice: HostSlice<'a>,
    pub cap: Tracked<ShareCap>,
}

/// Active local GPU-VRAM read lease carrying the frame id and `SyncPoint`
/// for Ordered Handoff without borrowing `&LocalEngine`.
pub struct GpuLease {
    pub handle: ReadLeaseHandle,
    pub version: VersionId,
    pub frame: FrameId,
    pub sync_point: SyncPoint,
    pub cap: Tracked<ShareCap>,
}

/// Thin, composable multi-tier local engine coordinating `CacheCore` with
/// `HostBackend` and optional `CudaBackend` / `NvmeBackend` plugins.
pub struct LocalEngine<'a> {
    pub core: CacheCore,
    pub host: &'a HostBackend,
    pub host_ctx: HostContext,
    pub cuda: Option<CudaBackend<'a>>,
    pub nvme: Option<NvmeBackend>,
}

impl<'a> LocalEngine<'a> {
    pub open spec fn well_formed(&self) -> bool {
        &&& self.core.well_formed()
        &&& self.host.well_formed(&self.core)
        &&& self.host_ctx.ready(self.core.instance_id(), self.core.node())
        &&& source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger@)
        &&& self.host_ctx.source@.source() != self.host.ctx.source@.source()
        &&& match self.cuda {
            None => true,
            Some(c) => {
                &&& c.well_formed(&self.core)
                &&& c.timeline.timeline_id != self.host.timeline
                &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier && h.first_frame
                    == self.host.first_frame && h.num_frames == self.host.num_frames
            },
        }
        &&& match self.nvme {
            None => true,
            Some(n) => {
                &&& n.well_formed(&self.core)
                &&& self.host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                &&& n.timeline != self.host.timeline
                &&& match self.cuda {
                    None => true,
                    Some(c) => n.timeline != c.timeline.timeline_id,
                }
            },
        }
    }

    /// Assemble a `LocalEngine` from an already-initialized `CacheCore`, shared
    /// `HostBackend`, dedicated `HostContext`, and optional `CudaBackend` /
    /// `NvmeBackend`.
    pub fn new(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: HostContext,
        cuda: Option<CudaBackend<'a>>,
        nvme: Option<NvmeBackend>,
    ) -> (engine: Self)
        requires
            core.well_formed(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger@),
            host_ctx.source@.source() != host.ctx.source@.source(),
            cuda matches Some(c) ==> {
                &&& c.well_formed(&core)
                &&& c.timeline.timeline_id != host.timeline
                &&& c.pinned_host matches Some(h) ==> h.tier == host.tier && h.first_frame
                    == host.first_frame && h.num_frames == host.num_frames
            },
            nvme matches Some(n) ==> {
                &&& n.well_formed(&core)
                &&& host.buffer.align() % (DIRECT_IO_ALIGNMENT as usize) == 0
                &&& n.timeline != host.timeline
                &&& (cuda matches Some(c) ==> n.timeline != c.timeline.timeline_id)
            },
        ensures
            engine.well_formed(),
            engine.core == core,
            engine.host == host,
            engine.cuda == cuda,
            engine.nvme == nvme,
    {
        Self { core, host, host_ctx, cuda, nvme }
    }

    pub fn host_tier(&self) -> (t: TierId)
        ensures
            t == self.host.tier,
    {
        self.host.tier_id()
    }

    pub fn gpu_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.cuda {
                Some(c) => Some(c.gpu_tier),
                None => None,
            },
    {
        match &self.cuda {
            Some(c) => Some(c.gpu_tier_id()),
            None => None,
        }
    }

    pub fn nvme_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.nvme {
                Some(n) => Some(n.tier),
                None => None,
            },
    {
        match &self.nvme {
            Some(n) => Some(n.tier_id()),
            None => None,
        }
    }

    pub fn gpu_frame_slice(&self, frame: FrameId) -> (s: Option<CudaSlice<'_>>)
        requires
            self.well_formed(),
    {
        match &self.cuda {
            Some(c) => {
                if c.owns_gpu_frame(frame) {
                    Some(c.gpu_frame_slice(frame))
                } else {
                    None
                }
            },
            None => None,
        }
    }

    pub(super) proof fn lemma_preserve_siblings(
        before: &CacheCore,
        after: &CacheCore,
        touched_timeline: Option<u8>,
        allowed_source: Option<EventSource>,
        mode: FrameTouchMode,
        host: &HostBackend,
        host_ctx: &HostContext,
        cuda: &Option<CudaBackend<'a>>,
        nvme: &Option<NvmeBackend>,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            core_transition(before, after, touched_timeline, allowed_source, mode),
            host.well_formed(before),
            touched_timeline != Some(host.timeline),
            allowed_source != Some(host.ctx.source@.source()),
            host_ctx.ready(before.instance_id(), before.node()),
            source_fresh_for_ledger(host_ctx.source@, before.ledger@),
            allowed_source != Some(host_ctx.source@.source()),
            cuda matches Some(c) ==> {
                &&& c.well_formed(before)
                &&& touched_timeline != Some(c.timeline.timeline_id)
                &&& allowed_source != Some(c.timeline.stream.source().source())
                &&& (mode is PreserveAllInit || mode is PreserveParkedInit)
            },
            nvme matches Some(n) ==> {
                &&& n.well_formed(before)
                &&& touched_timeline != Some(n.timeline)
                &&& allowed_source != Some(n.ring.source().source())
                &&& (mode is PreserveAllInit || mode is PreserveUnparkedInit)
            },
        ensures
            host.well_formed(after),
            host_ctx.ready(after.instance_id(), after.node()),
            source_fresh_for_ledger(host_ctx.source@, after.ledger@),
            cuda matches Some(c) ==> c.well_formed(after),
            nvme matches Some(n) ==> n.well_formed(after),
    {
        hide(CacheCore::well_formed);
        lemma_core_transition_preserves_host(
            before,
            after,
            touched_timeline,
            allowed_source,
            mode,
            host,
        );
        lemma_core_transition_preserves_source(
            before,
            after,
            touched_timeline,
            allowed_source,
            mode,
            host_ctx.source@,
        );
        if let Some(ref c) = *cuda {
            lemma_core_transition_preserves_cuda(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                c,
            );
        }
        if let Some(ref n) = *nvme {
            lemma_core_transition_preserves_nvme(
                before,
                after,
                touched_timeline,
                allowed_source,
                mode,
                n,
            );
        }
    }

    /// Evict a published frame (`Published -> Draining`).
    pub fn evict_frame(&mut self, frame: FrameId) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r.is_ok() ==> final(self).core.frames.meta(frame).state == STATE_DRAINING,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let res = self.core.evict_frame(frame);
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert(core_transition(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
            )) by {
                reveal(core_transition);
                broadcast use {
                    vstd::map::group_map_lemmas,
                    vstd::imap::group_imap_lemmas,
                    vstd::iset::group_iset_lemmas,
                };

            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
            );
        }
        res
    }

    /// Reclaim up to `budget` retired draining frames on `tier`.
    pub fn reclaim_draining(&mut self, tier: TierId, budget: usize) -> (reclaimed: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec == old(self).core.spec,
            reclaimed <= budget,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let reclaimed = self.core.reclaim_draining(tier, budget);
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
            );
        }
        reclaimed
    }

    /// Reclaim retired draining frames across every registered tier.
    pub fn reclaim_all_draining(&mut self) -> (total: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            final(self).core.spec == old(self).core.spec,
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let total = self.core.reclaim_all_draining();
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
            );
        }
        total
    }

    /// Acquire a zero-copy local read lease on a published Host-DRAM frame.
    pub fn get_lease_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: BackendResult<Option<HostLease<'a>>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(Some(lease)) ==> {
                &&& lease.slice.well_formed()
                &&& lease.slice.instance_id() == final(self).core.instance_id()
                &&& lease.slice.slot_ref() == final(self).core.frames.frame(lease.handle.frame).slot
                &&& final(self).core.active_lease_matches(lease.handle)
                &&& !final(self).core.active_leases[lease.handle.slot as int]->Some_0.internal
                &&& lease.cap@.is_local()
                &&& lease.cap@.instance_id() == final(self).core.instance_id()
                &&& lease.cap@.loc() == lease.slice.slot_ref().loc
                &&& lease.cap@.share()
                    == final(self).core.active_leases[lease.handle.slot as int]->Some_0.share@
                &&& read_admitted(lease.cap@.frontier(), final(self).core.citations())
                &&& lease.cap@.frontier().readers.is_empty()
                &&& final(self).core.frames.returned_reads(
                    lease.handle.frame,
                    lease.cap@,
                    Ticket::NONE,
                    final(self).core.kernel_view(),
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some((frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Ok(None);
        };
        let ghost before = self.core;
        let (handle, version, _cert, cap) = self.core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        )?;
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
            );
            lemma_slab_owns_tier_frame(
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
                frame,
            );
            lemma_slab_frame_slot(
                &self.core,
                self.host.tier,
                self.host.first_frame,
                self.host.num_frames,
                self.host.buffer.space(),
                frame,
            );
        }
        let slice = self.host.frame_slice(frame);
        Ok(Some(HostLease { handle, version, slice, cap }))
    }

    /// Acquire a zero-copy local read lease on a published GPU-VRAM frame.
    pub fn get_lease_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: BackendResult<Option<GpuLease>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(Some(lease)) ==> {
                &&& lease.handle.frame == lease.frame
                &&& final(self).cuda matches Some(c) && c.owns_gpu(lease.frame)
                &&& final(self).core.active_lease_matches(lease.handle)
                &&& !final(self).core.active_leases[lease.handle.slot as int]->Some_0.internal
                &&& lease.cap@.is_local()
                &&& lease.cap@.instance_id() == final(self).core.instance_id()
                &&& lease.cap@.loc() == final(self).core.frames.frame(lease.frame).slot.loc
                &&& lease.cap@.share()
                    == final(self).core.active_leases[lease.handle.slot as int]->Some_0.share@
                &&& read_admitted(lease.cap@.frontier(), final(self).core.citations())
                &&& lease.cap@.frontier().readers.is_empty()
                &&& final(self).core.frames.returned_reads(
                    lease.frame,
                    lease.cap@,
                    Ticket::NONE,
                    final(self).core.kernel_view(),
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some(gpu_tier) = self.gpu_tier() else {
            return Err(CoreError::UnsupportedTier);
        };
        let Some((frame, _ver)) = self.core.lookup_tier(key, gpu_tier) else {
            return Ok(None);
        };
        let ghost before = self.core;
        let (handle, version, _cert, cap) = self.core.acquire_local_lease(
            frame,
            false,
            Tracked(shares),
            Tracked(instance),
        )?;
        let sync_point = match &self.cuda {
            Some(c) => c.frame_sync_point(frame),
            None => SyncPoint::HostComplete,
        };
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
            );
            let c = self.cuda.unwrap();
            lemma_slab_owns_tier_frame(
                &self.core,
                c.gpu_tier,
                c.gpu_first_frame,
                c.gpu_num_frames,
                c.gpu_buffer.space(),
                frame,
            );
        }
        Ok(Some(GpuLease { handle, version, frame, sync_point, cap }))
    }

    /// Release a caller-owned (`!internal`) local read lease, recording its
    /// optional trailing `read_ticket` for Observed Reclamation.
    pub fn release_lease(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        handle: ReadLeaseHandle,
        Tracked(cap): Tracked<ShareCap>,
        read_ticket: Ticket,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(self).core.timelines.valid_ticket(read_ticket),
            old(self).core.active_lease_matches(handle) ==> {
                let share = old(self).core.active_leases[handle.slot as int]->Some_0.share@;
                &&& cap.share() == share
                &&& old(self).core.frames.returned_reads(
                    handle.frame,
                    cap,
                    read_ticket,
                    old(self).core.kernel_view(),
                )
            },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec == old(self).core.spec,
    {
        hide(CacheCore::well_formed);
        let idx = handle.slot as usize;
        if idx >= self.core.active_leases.len() {
            return Err(CoreError::InvalidArgument);
        }
        let is_internal = match &self.core.active_leases[idx] {
            Some(lease) => lease.internal,
            None => false,
        };
        if is_internal {
            return Err(CoreError::WouldConflict);
        }
        let ghost before = self.core;
        let res = self.core.release_local_lease(
            handle,
            Tracked(cap),
            read_ticket,
            Tracked(instance),
        );
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                None,
                None,
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &self.nvme,
            );
        }
        res
    }

    /// Non-blocking progress sweep across CUDA completions, NVMe completions,
    /// and draining-frame reclamation with M8 non-dropping progress reporting.
    pub fn poll_progress(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        ProgressReport,
        (CoreError, ProgressReport),
    >)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
    {
        hide(CacheCore::well_formed);
        let mut report = ProgressReport::empty();
        let mut first_err: Option<CoreError> = None;
        if let Some(mut cuda) = self.cuda.take() {
            let ghost before = self.core;
            let res = cuda.poll_completions(&mut self.core, Tracked(instance));
            let ghost tl = cuda.timeline.timeline_id;
            proof {
                let ghost none_cuda: Option<CudaBackend<'a>> = None;
                Self::lemma_preserve_siblings(
                    &before,
                    &self.core,
                    Some(tl),
                    None,
                    FrameTouchMode::PreserveUnparkedInit,
                    self.host,
                    &self.host_ctx,
                    &none_cuda,
                    &self.nvme,
                );
            }
            self.cuda = Some(cuda);
            match res {
                Ok(cr) => {
                    report.cuda_events = cr.events_completed;
                    report.published_gpu_puts = cr.writes_published;
                },
                Err((e, cr)) => {
                    report.cuda_events = cr.events_completed;
                    report.published_gpu_puts = cr.writes_published;
                    first_err = Some(e);
                },
            }
        }
        if let Some(mut nvme) = self.nvme.take() {
            let ghost before = self.core;
            let res = nvme.poll_completions(&mut self.core, Tracked(instance));
            let ghost tl = nvme.timeline;
            proof {
                let ghost none_nvme: Option<NvmeBackend> = None;
                Self::lemma_preserve_siblings(
                    &before,
                    &self.core,
                    Some(tl),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                    self.host,
                    &self.host_ctx,
                    &self.cuda,
                    &none_nvme,
                );
                lemma_versions_monotonic_trans(old(self).core.spec@, before.spec@, self.core.spec@);
            }
            self.nvme = Some(nvme);
            match res {
                Ok(np) => {
                    report.nvme_completed = np.completed;
                    report.nvme_failed = np.failed;
                },
                Err((e, np)) => {
                    report.nvme_completed = np.completed;
                    report.nvme_failed = np.failed;
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                },
            }
        }
        report.reclaimed_frames = self.reclaim_all_draining();
        match first_err {
            Some(e) => Err((e, report)),
            None => Ok(report),
        }
    }
}

} // verus!
