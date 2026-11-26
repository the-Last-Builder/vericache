#[cfg(verus_only)]
use crate::backends::device::device_puts_untouched;
use crate::backends::device::DefaultDeviceBackend;
#[cfg(verus_only)]
use crate::backends::host::lemma_core_transition_preserves_host;
use crate::backends::host::HostBackend;
#[cfg(all(feature = "nvme", verus_only))]
use crate::backends::nvme::NvmeBackend;
#[cfg(feature = "rdma")]
use crate::backends::rdma::peer::PeerSession;
#[cfg(feature = "rdma")]
use crate::backends::DeviceAttachment;
use crate::backends::{DeviceBackend, DeviceLease};
#[cfg(verus_only)]
use crate::frame::*;
use crate::kernel::*;
use crate::spec::*;
#[cfg(verus_only)]
use crate::timeline::*;
use transfer::context::*;
use transfer::host::HostSlice;
#[cfg(feature = "rdma")]
use transfer::transport::ByteTransport;
#[cfg(verus_only)]
use transfer_core::memory::RdmaRegisterableSlab;
#[cfg(verus_only)]
use vericache_kernel::compose::*;
use vericache_kernel::spi::*;
use vstd::prelude::*;

mod boundary;
use boundary::{is_caller_device_timeline, matches_backends};
mod caller;
#[cfg(feature = "rdma")]
pub mod distributed;
#[cfg(feature = "rdma")]
mod fetch;
mod gpu_routes;
mod inspect;
#[cfg(any(test, verus_only))]
mod no_device_tests;
mod nvme_storage;
mod routes;

#[cfg(feature = "cuda")]
pub use crate::backends::CudaAttachment;
#[cfg(feature = "rdma")]
pub use distributed::{DistributedEvent, HostWriteReservation, PendingReadGrant};
use nvme_storage::nvme_tier;
pub use nvme_storage::NvmeStorage;
#[cfg(verus_only)]
use nvme_storage::{
    lemma_preserve_nvme, nvme_compatible, nvme_tier_spec, nvme_untouched, nvme_well_formed,
};

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

pub type GpuLease = DeviceLease;

/// Verified composable 4-tier N-peer distributed KV-cache engine unifying
/// `CacheCore`, `HostBackend`, `DeviceAttachment`, optional `NvmeBackend`,
/// and `peers: Vec<PeerSession<'a, T>>`.
#[cfg(feature = "rdma")]
pub struct DistributedEngine<
    'a,
    T: ByteTransport,
    D: DeviceBackend<'a> = DefaultDeviceBackend<'a>,
> {
    pub(crate) core: CacheCore,
    pub(crate) host: &'a HostBackend,
    pub(crate) host_ctx: SyncContext,
    pub(crate) device: DeviceAttachment<'a, D>,
    pub(crate) pending_device_bindings: Tracked<Map<VersionId, VersionBinding>>,
    pub(crate) nvme: NvmeStorage,
    pub(crate) peers: Vec<PeerSession<'a, T>>,
    pub(crate) caller_device_streams: Vec<Option<u64>>,
}

/// Thin, composable multi-tier local engine coordinating `CacheCore` with
/// `HostBackend` and optional device and `NvmeBackend` plugins.
pub struct LocalEngine<'a, D: DeviceBackend<'a> = DefaultDeviceBackend<'a>> {
    pub(crate) core: CacheCore,
    pub(crate) host: &'a HostBackend,
    pub(crate) host_ctx: SyncContext,
    pub(crate) device: Option<D>,
    pub(crate) nvme: NvmeStorage,
    pub(crate) caller_device_streams: Vec<Option<u64>>,
}

impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    pub open(crate) spec fn host_ctx_spec(&self) -> SyncContext {
        self.host_ctx
    }

    pub open(crate) spec fn core_spec(&self) -> CacheCore {
        self.core
    }

    pub open(crate) spec fn device_spec(&self) -> Option<D> {
        self.device
    }

    pub open(crate) spec fn nvme_spec(&self) -> NvmeStorage {
        self.nvme
    }

    pub open(crate) spec fn host_spec(&self) -> &'a HostBackend {
        self.host
    }

    pub open(crate) spec fn well_formed(&self) -> bool {
        &&& self.core.well_formed()
        &&& self.host.well_formed(&self.core)
        &&& self.host_ctx.ready(self.core.instance_id(), self.core.node())
        &&& source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger_spec()@)
        &&& self.host_ctx.source@.source() != self.host.ctx.source@.source()
        &&& match self.device {
            None => true,
            Some(c) => {
                &&& c.well_formed(&self.core)
                &&& c.timeline_id_spec() != self.host.timeline
                &&& c.pinned_host_layout_spec() matches Some((tier, first, count)) ==> tier
                    == self.host.tier && first == self.host.first_frame && count
                    == self.host.num_frames
            },
        }
        &&& nvme_compatible(&self.nvme, &self.core, self.host, self.device)
    }

    /// Assemble a `LocalEngine` from an already-initialized `CacheCore`, shared
    /// `HostBackend`, dedicated `SyncContext`, and optional device and
    /// `NvmeBackend`.
    pub fn new(
        core: CacheCore,
        host: &'a HostBackend,
        host_ctx: SyncContext,
        device: Option<D>,
        nvme: NvmeStorage,
    ) -> (r: BackendResult<Self>)
        requires
            core.well_formed(),
            host.well_formed(&core),
            host_ctx.ready(core.instance_id(), core.node()),
            source_fresh_for_ledger(host_ctx.source@, core.ledger_spec()@),
            host_ctx.source@.source() != host.ctx_spec().source@.source(),
            device matches Some(c) ==> {
                &&& c.well_formed(&core)
                &&& c.timeline_id_spec() != host.timeline_spec()
                &&& c.pinned_host_layout_spec() matches Some((tier, first, count)) ==> tier
                    == host.tier_spec() && first == host.first_frame_spec() && count
                    == host.num_frames_spec()
            },
            nvme_compatible(&nvme, &core, host, device),
        ensures
            r matches Ok(engine) ==> {
                &&& engine.well_formed()
                &&& engine.core_spec() == core
                &&& engine.host_spec() == host
                &&& engine.device_spec() == device
                &&& engine.nvme_spec() == nvme
            },
    {
        if !matches_backends(&core, host, device.as_ref(), &nvme) {
            return Err(CoreError::InvalidArgument);
        }
        Ok(Self { core, host, host_ctx, device, nvme, caller_device_streams: Vec::new() })
    }

    pub fn host_tier(&self) -> (t: TierId)
        ensures
            t == self.host_spec().tier_spec(),
    {
        self.host.tier_id()
    }

    pub fn gpu_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.device_spec() {
                Some(c) => Some(c.device_tier_spec()),
                None => None,
            },
    {
        self.device_tier()
    }

    pub fn device_tier(&self) -> (t: Option<TierId>)
        ensures
            t == match self.device_spec() {
                Some(c) => Some(c.device_tier_spec()),
                None => None,
            },
    {
        match &self.device {
            Some(c) => Some(c.device_tier_id()),
            None => None,
        }
    }

    pub fn nvme_tier(&self) -> (t: Option<TierId>)
        ensures
            t == nvme_tier_spec(&self.nvme_spec()),
    {
        nvme_tier(&self.nvme)
    }

    pub fn gpu_frame_slice(&self, frame: FrameId) -> (s: Option<D::Slice<'_>>)
        requires
            self.well_formed(),
    {
        self.device_frame_slice(frame)
    }

    pub fn device_frame_slice(&self, frame: FrameId) -> (s: Option<D::Slice<'_>>)
        requires
            self.well_formed(),
    {
        match &self.device {
            Some(c) => {
                proof {
                    c.lemma_device_layout(&self.core);
                }
                c.device_frame_slice(frame)
            },
            None => None,
        }
    }

    pub(super) proof fn lemma_preserve_siblings(
        before: &CacheCore,
        after: &CacheCore,
        fp: CoreFootprint,
        host: &HostBackend,
        host_ctx: &SyncContext,
        device: &Option<D>,
        nvme: &NvmeStorage,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            core_transition(before, after, fp),
            host.well_formed(before),
            !fp.touched_timelines.contains(host.timeline),
            fp.excludes_source(host.ctx.source@.source()),
            host_ctx.ready(before.instance_id(), before.node()),
            source_fresh_for_ledger(host_ctx.source@, before.ledger_spec()@),
            fp.excludes_source(host_ctx.source@.source()),
            device matches Some(c) ==> {
                &&& c.well_formed(before)
                &&& !fp.touched_timelines.contains(c.timeline_id_spec())
                &&& fp.excludes_source(c.event_source_spec())
                &&& device_puts_untouched(c.pending_puts_spec(), before, fp)
            },
            nvme_well_formed(nvme, before),
            nvme_untouched(nvme, before, fp),
        ensures
            host.well_formed(after),
            host_ctx.ready(after.instance_id(), after.node()),
            source_fresh_for_ledger(host_ctx.source@, after.ledger_spec()@),
            device matches Some(c) ==> c.well_formed(after),
            nvme_well_formed(nvme, after),
    {
        hide(CacheCore::well_formed);
        lemma_core_transition_preserves_host(before, after, fp, host);
        lemma_core_transition_preserves_source(before, after, fp, host_ctx.source@);
        if let Some(ref c) = *device {
            c.preserve_across_core_transition(before, after, fp);
        }
        lemma_preserve_nvme(before, after, fp, nvme);
    }

    /// Evict a published frame (`Published -> Draining`).
    pub fn evict_frame(&mut self, frame: FrameId) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r.is_ok() ==> final(self).core_spec().frames_spec().meta(frame).state == STATE_DRAINING,
    {
        hide(CacheCore::well_formed);
        self.core.check_published_frame(frame)?;
        let ghost before = self.core;
        let res = self.core.evict_frame(frame);
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(&self.core);
            assert(core_transition(&before, &self.core, CoreFootprint::preserve_all(None, None)))
                by {
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
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
        }
        res
    }

    /// Reclaim up to `budget` retired draining frames on `tier`.
    pub fn reclaim_draining(&mut self, tier: TierId, budget: usize) -> (r: BackendResult<usize>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            r matches Ok(reclaimed) ==> reclaimed <= budget,
    {
        hide(CacheCore::well_formed);
        if tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let reclaimed = self.core.reclaim_draining(tier, budget);
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
        }
        Ok(reclaimed)
    }

    /// Reclaim retired draining frames across every registered tier.
    pub fn reclaim_all_draining(&mut self) -> (total: usize)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let total = self.core.reclaim_all_draining();
        proof {
            lemma_core_transition_no_event(&before, &self.core, None, None);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
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
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(Some(lease)) ==> {
                &&& lease.slice.well_formed()
                &&& lease.slice.instance_id() == final(self).core_spec().instance_id()
                &&& lease.slice.slot_ref() == final(self).core_spec().frames_spec().frame(
                    lease.handle.frame_spec(),
                ).slot
                &&& final(self).core_spec().active_lease_matches(lease.handle)
                &&& !final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.internal
                &&& lease.cap@.is_cooperative()
                &&& lease.cap@.instance_id() == final(self).core_spec().instance_id()
                &&& lease.cap@.loc() == lease.slice.slot_ref().loc
                &&& lease.cap@.share()
                    == final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.share@
                &&& read_admitted(lease.cap@.frontier(), final(self).core_spec().citations())
                &&& lease.cap@.frontier().readers.is_empty()
                &&& final(self).core_spec().frames_spec().returned_reads(
                    lease.handle.frame_spec(),
                    lease.cap@,
                    Ticket::NONE,
                    final(self).core_spec().kernel_view(),
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
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
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
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok(Some(lease)) ==> {
                &&& lease.handle.frame_spec() == lease.frame
                &&& final(self).device_spec() matches Some(c) && c.owns_device(lease.frame)
                &&& final(self).core_spec().active_lease_matches(lease.handle)
                &&& !final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.internal
                &&& lease.cap@.is_cooperative()
                &&& lease.cap@.instance_id() == final(self).core_spec().instance_id()
                &&& lease.cap@.loc() == final(self).core_spec().frames_spec().frame(
                    lease.frame,
                ).slot.loc
                &&& lease.cap@.share()
                    == final(self).core_spec().active_leases_spec()[lease.handle.slot_spec() as int]->Some_0.share@
                &&& read_admitted(lease.cap@.frontier(), final(self).core_spec().citations())
                &&& lease.cap@.frontier().readers.is_empty()
                &&& final(self).core_spec().frames_spec().returned_reads(
                    lease.frame,
                    lease.cap@,
                    Ticket::NONE,
                    final(self).core_spec().kernel_view(),
                )
            },
    {
        hide(CacheCore::well_formed);
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        let Some((frame, _ver)) = self.core.lookup_tier(key, device.device_tier_id()) else {
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
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
            device.lemma_device_layout(&self.core);
            lemma_slab_owns_tier_frame(
                &self.core,
                device.device_tier_spec(),
                device.device_first_frame_spec(),
                device.device_num_frames_spec(),
                device.buffer_spec().reg_space(),
                frame,
            );
            device.lemma_frame_ownership(frame);
        }
        let sync_point = device.frame_sync_point(frame);
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
            instance.id() == old(self).core_spec().instance_id(),
            old(self).core_spec().active_lease_matches(handle) ==> {
                let share = old(
                    self,
                ).core_spec().active_leases_spec()[handle.slot_spec() as int]->Some_0.share@;
                &&& cap.share() == share
                &&& old(self).core_spec().frames_spec().returned_reads(
                    handle.frame_spec(),
                    cap,
                    read_ticket,
                    old(self).core_spec().kernel_view(),
                )
            },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        if handle.core_id() != self.core.runtime_id() || handle.frame().0 as usize
            >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        proof {
            lemma_core_wf_facts(&self.core);
        }
        if !self.core.timelines().contains_ticket(read_ticket) {
            return Err(CoreError::InvalidArgument);
        }
        let idx = handle.slot() as usize;
        if idx >= self.core.local_leases().len() {
            return Err(CoreError::InvalidArgument);
        }
        let is_internal = match &self.core.local_leases()[idx] {
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
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
        }
        res
    }

    /// Non-blocking progress sweep across device completions, NVMe completions,
    /// and draining-frame reclamation with M8 non-dropping progress reporting.
    pub fn poll_progress(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        ProgressReport,
        (CoreError, ProgressReport),
    >)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
    {
        hide(CacheCore::well_formed);
        let mut report = ProgressReport::empty();
        let mut first_err: Option<CoreError> = None;
        if let Some(mut device) = self.device.take() {
            let ghost before_device = device;
            let ghost before = self.core;
            let ghost poll_fp = device.poll_footprint();
            let res = device.poll_completions(&mut self.core, Tracked(instance));
            proof {
                before_device.lemma_pending_put_model();
                device.lemma_same_config(&before_device);
                let ghost no_device: Option<D> = None;
                Self::lemma_preserve_siblings(
                    &before,
                    &self.core,
                    poll_fp,
                    self.host,
                    &self.host_ctx,
                    &no_device,
                    &self.nvme,
                );
            }
            self.device = Some(device);
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
        #[cfg(feature = "nvme")]
        if let Some(mut nvme) = self.nvme.take() {
            let ghost before = self.core;
            let ghost poll_fp = nvme.poll_footprint();
            let res = nvme.poll_completions(&mut self.core, Tracked(instance));
            proof {
                let ghost none_nvme: Option<NvmeBackend> = None;
                Self::lemma_preserve_siblings(
                    &before,
                    &self.core,
                    poll_fp,
                    self.host,
                    &self.host_ctx,
                    &self.device,
                    &none_nvme,
                );
                lemma_versions_monotonic_trans(
                    old(self).core.spec_spec()@,
                    before.spec_spec()@,
                    self.core.spec_spec()@,
                );
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
