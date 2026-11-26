#[cfg(verus_only)]
use crate::backends::device::device_scatter_offsets_valid;
#[cfg(verus_only)]
use crate::backends::host::host_read_witness;
#[cfg(all(feature = "nvme", verus_only))]
use crate::backends::nvme::NvmeBackend;
#[cfg(feature = "nvme")]
use crate::backends::nvme::NvmeProgress;
#[cfg(all(feature = "nvme", verus_only))]
use crate::backends::nvme::{lemma_nvme_pending_from_trans, lemma_nvme_widen_poll_footprint};
use crate::backends::{DeviceBackend, DeviceSlice};
#[cfg(verus_only)]
use crate::frame::*;
#[cfg(verus_only)]
use crate::kernel::CacheCore;
use crate::provenance::ValueCertificate;
#[cfg(verus_only)]
use crate::provenance::*;
use crate::spec::*;
#[cfg(feature = "nvme")]
use crate::timeline::BITMAP_CAPACITY;
use transfer::context::*;
#[cfg(verus_only)]
use transfer::domain::TransferEvent;
#[cfg(verus_only)]
use vericache_kernel::compose::*;
use vericache_kernel::spi::*;
use vstd::prelude::*;

#[cfg(verus_only)]
use super::nvme_storage::lemma_preserve_nvme;
use super::LocalEngine;

verus! {

impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    /// Synchronously write `data` into Host DRAM and publish into `CacheCore`.
    pub fn put_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        data: &[u8],
    ) -> (r: BackendResult<(VersionId, FrameId)>)
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
            match r {
                Err(_) => final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
                Ok((ver, frame)) => {
                    &&& final(self).host_spec().owns(frame)
                    &&& final(self).core_spec().spec_spec()@.versions == old(
                        self,
                    ).core_spec().spec_spec()@.versions.insert(ver, data@)
                    &&& final(self).core_spec().spec_spec()@.catalog == old(
                        self,
                    ).core_spec().spec_spec()@.catalog.insert(key, ver)
                    &&& final(self).core_spec().catalog_spec().slot(
                        key,
                        final(self).host_spec().tier_spec().index() as int,
                    ) == Some(frame)
                    &&& final(self).core_spec().frames_spec().meta(frame).state == STATE_PUBLISHED
                },
            },
    {
        hide(CacheCore::well_formed);
        let ghost before = self.core;
        let ghost src = self.host_ctx.source@.source();
        let res = self.host.put_host_with(
            &mut self.host_ctx,
            &mut self.core,
            Tracked(instance),
            key,
            data,
        );
        proof {
            if let Some(ref c) = self.device {
                c.lemma_device_layout(&before);
                c.preserve_across_core_transition(
                    &before,
                    &self.core,
                    CoreFootprint::preserve_all(Some(self.host.timeline), Some(src)),
                );
            }
            lemma_preserve_nvme(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(self.host.timeline), Some(src)),
                &self.nvme,
            );
        }
        res
    }

    /// Asynchronously write one full frame from a caller device slice into
    /// `dst_tier` (GPU VRAM or attached pinned Host DRAM).
    pub fn put_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &D::Slice<'_>,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            src.well_formed(),
            src.instance_id() == old(self).core_spec().instance_id(),
            src.slot_ref().loc.node == old(self).core_spec().node(),
            old(self).device_spec() matches Some(c) ==> src.slot_ref().sub_len == old(
                self,
            ).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
            old(src_perm).valid_cooperative(
                old(self).core_spec().instance_id(),
                src.slot_ref().loc,
            ),
            read_admitted(old(src_perm).frontier(), old(self).core_spec().citations()),
            old(self).device_spec() matches Some(c) ==> (src.slot_ref().loc.space.is_gpu()
                && src.slot_ref().loc.space.gpu_device() != c.device_spec() ==> (old(
                src_perm,
            ).frontier().last_writer matches Some(w) ==> !(w.source.is_rdma_qp()))),
            replica_valid(
                old(self).core_spec().ledger_spec()@,
                old(self).core_spec().node(),
                src_cert,
            ),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(self).core_spec().frames_spec().metas_spec().len() && (#[trigger] old(
                    self,
                ).core_spec().frames_spec().meta(f)).state == STATE_EMPTY ==> src.slot_ref().loc
                    != old(self).core_spec().frames_spec().frame(f).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        if dst_tier.as_index() >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(mut device) = self.device.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        if self.core.tier_frame_size(device.device_tier_id()) != Some(src.len()) {
            self.device = Some(device);
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost tl = device.timeline_id_spec();
        let ghost src_ev = device.event_source_spec();
        let ghost before_device = device;
        let res = device.put_device(
            &mut self.core,
            Tracked(instance),
            dst_tier,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        );
        proof {
            before_device.lemma_device_layout(&before);
            device.lemma_same_config(&before_device);
            let ghost no_device: Option<D> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &no_device,
                &self.nvme,
            );
        }
        self.device = Some(device);
        res
    }

    /// Asynchronously gather multiple caller device slices into one GPU-VRAM frame.
    pub fn put_gpu_scatter(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        srcs: &[D::Slice<'_>],
        Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(self).device_spec() matches Some(c) ==> {
                &&& device_scatter_offsets_valid(
                    srcs@,
                    offsets,
                    old(self).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
                )
                &&& bytes.len() == old(self).core_spec().frames_spec().tier_frame_size(
                    c.device_tier_spec(),
                )
            },
            src_certs.len() == srcs@.len(),
            forall|i: int|
                0 <= i < srcs@.len() ==> {
                    &&& (#[trigger] srcs@[i]).instance_id() == old(self).core_spec().instance_id()
                    &&& srcs@[i].slot_ref().loc.node == old(self).core_spec().node()
                    &&& old(src_perms).dom().contains(i)
                    &&& old(src_perms)[i].valid_cooperative(
                        old(self).core_spec().instance_id(),
                        srcs@[i].slot_ref().loc,
                    )
                    &&& read_admitted(
                        old(src_perms)[i].frontier(),
                        old(self).core_spec().citations(),
                    )
                    &&& (old(self).device_spec() matches Some(c) ==> (
                    srcs@[i].slot_ref().loc.space.is_gpu()
                        && srcs@[i].slot_ref().loc.space.gpu_device() != c.device_spec() ==> (old(
                        src_perms,
                    )[i].frontier().last_writer matches Some(w) ==> !(w.source.is_rdma_qp()))))
                    &&& replica_valid(
                        old(self).core_spec().ledger_spec()@,
                        old(self).core_spec().node(),
                        src_certs[i],
                    )
                    &&& source_slice_valid(src_certs[i], srcs@[i].slot_ref())
                    &&& src_certs[i].slot.well_formed()
                    &&& src_certs[i].bytes.len() == src_certs[i].slot.sub_len
                    &&& source_slice_bytes(src_certs[i], srcs@[i].slot_ref()) == bytes.subrange(
                        offsets[i] as int,
                        offsets[i + 1] as int,
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(self).core_spec().frames_spec().metas_spec().len() && (
                        #[trigger] old(self).core_spec().frames_spec().meta(f)).state == STATE_EMPTY
                            ==> srcs@[i].slot_ref().loc != old(
                            self,
                        ).core_spec().frames_spec().frame(f).slot.loc
                },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        hide(CacheCore::well_formed);
        let Some(mut device) = self.device.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        if !super::boundary::device_slices_cover_frame(
            srcs,
            self.core.tier_frame_size(device.device_tier_id()),
        ) {
            self.device = Some(device);
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost tl = device.timeline_id_spec();
        let ghost src_ev = device.event_source_spec();
        let ghost before_device = device;
        let res = device.put_device_scatter(
            &mut self.core,
            Tracked(instance),
            key,
            srcs,
            Tracked(src_perms),
            Ghost(src_certs),
            Ghost(offsets),
            Ghost(bytes),
        );
        proof {
            before_device.lemma_device_layout(&before);
            device.lemma_same_config(&before_device);
            let ghost no_device: Option<D> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &no_device,
                &self.nvme,
            );
        }
        self.device = Some(device);
        res
    }
}

#[cfg(feature = "nvme")]
impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    /// Stage and submit an async Host-DRAM -> Local-NVMe tier-down copy hop.
    pub fn tier_dram_to_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
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
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Err((CoreError::Miss, None));
        };
        let (dst_frame, ticket) = match self.stage_dram_to_nvme(
            Tracked(instance),
            Tracked(shares),
            src_frame,
        ) {
            Ok(pair) => pair,
            Err(e) => return Err((e, None)),
        };
        let submit_res = self.submit_staged_nvme();
        let rem_staged = match &self.nvme {
            Some(n) => n.staged_len(),
            None => 0,
        };
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
    }

    /// Submit the currently staged NVMe operations.
    pub fn submit_staged_nvme(&mut self) -> (r: BackendResult<usize>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).core_spec() == old(self).core_spec(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            match old(self).nvme_spec() {
                Some(n) => final(self).nvme_spec() matches Some(new) && new.timeline_spec()
                    == n.timeline_spec(),
                None => final(self).nvme_spec() is None,
            },
    {
        match &mut self.nvme {
            Some(nvme) => nvme.submit(Ghost(&self.core)),
            None => Ok(0),
        }
    }

    pub fn stage_dram_to_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
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
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        if src_frame.0 as usize >= self.core.frame_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(mut nvme) = self.nvme.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let ghost src_ev = nvme.ring.source().source();
        let staged = nvme.stage_dram_to_nvme(
            self.host,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            src_frame,
        );
        proof {
            if let Some(ref device) = self.device {
                device.lemma_device_layout(&before);
            }
            let ghost none_nvme: Option<NvmeBackend> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &self.device,
                &none_nvme,
            );
        }
        self.nvme = Some(nvme);
        staged
    }

    /// Stage and submit an async Local-NVMe -> Host-DRAM warm-up copy hop.
    pub fn warm_host_from_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
            old(self).nvme_spec() matches Some(n) ==> (final(self).nvme_spec() matches Some(n2)
                && n2.timeline_spec() == n.timeline_spec()),
            old(self).nvme_spec() matches Some(n) ==> core_transition(
                &old(self).core_spec(),
                &final(self).core_spec(),
                CoreFootprint::preserve_all(
                    Some(n.timeline_spec()),
                    Some(n.ring_spec().source().source()),
                ),
            ),
    {
        hide(CacheCore::well_formed);
        let Some(nvme_tier) = self.nvme_tier() else {
            return Err((CoreError::UnsupportedTier, None));
        };
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, nvme_tier) else {
            proof {
                let n = self.nvme.unwrap();
                lemma_core_transition_refl(
                    &self.core,
                    CoreFootprint::preserve_all(Some(n.timeline), Some(n.ring.source().source())),
                );
            }
            return Err((CoreError::Miss, None));
        };
        let (dst_frame, ticket) = match self.stage_nvme_to_dram(
            Tracked(instance),
            Tracked(shares),
            src_frame,
        ) {
            Ok(pair) => pair,
            Err(e) => return Err((e, None)),
        };
        let submit_res = self.submit_staged_nvme();
        let rem_staged = match &self.nvme {
            Some(n) => n.staged_len(),
            None => 0,
        };
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
    }

    pub fn stage_nvme_to_dram(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
    ) -> (r: BackendResult<(FrameId, Ticket)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (
            final(self).nvme_spec() matches Some(n) && ticket.timeline_id() == n.timeline_spec())
                && final(self).core_spec().timelines_spec().valid_ticket(ticket),
            old(self).nvme_spec() matches Some(n) ==> (final(self).nvme_spec() matches Some(n2)
                && n2.timeline_spec() == n.timeline_spec()),
            old(self).nvme_spec() matches Some(n) ==> core_transition(
                &old(self).core_spec(),
                &final(self).core_spec(),
                CoreFootprint::preserve_all(
                    Some(n.timeline_spec()),
                    Some(n.ring_spec().source().source()),
                ),
            ),
    {
        hide(CacheCore::well_formed);
        if src_frame.0 as usize >= self.core.frame_count() {
            proof {
                if let Some(ref n) = self.nvme {
                    lemma_core_transition_refl(
                        &self.core,
                        CoreFootprint::preserve_all(
                            Some(n.timeline),
                            Some(n.ring.source().source()),
                        ),
                    );
                }
            }
            return Err(CoreError::InvalidArgument);
        }
        let Some(mut nvme) = self.nvme.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = nvme.timeline;
        let ghost src_ev = nvme.ring.source().source();
        let staged = nvme.stage_nvme_to_dram(
            self.host,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            src_frame,
        );
        proof {
            if let Some(ref device) = self.device {
                device.lemma_device_layout(&before);
            }
            let ghost none_nvme: Option<NvmeBackend> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &self.device,
                &none_nvme,
            );
        }
        self.nvme = Some(nvme);
        staged
    }

    /// Synchronously wait for an NVMe `io_uring` `ticket` to retire, preserving
    /// M8 non-dropping `NvmeProgress` on error.
    pub fn wait_nvme_ticket(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        ticket: Ticket,
    ) -> (r: Result<NvmeProgress, (CoreError, NvmeProgress)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).core_spec().node_id_spec() == old(self).core_spec().node_id_spec(),
            final(self).core_spec().instance_id() == old(self).core_spec().instance_id(),
            final(self).core_spec().tiers_spec() == old(self).core_spec().tiers_spec(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            old(self).nvme_spec() matches Some(n) ==> (final(self).nvme_spec() matches Some(n2)
                && n2.timeline_spec() == n.timeline_spec()),
            old(self).nvme_spec() matches Some(n) ==> core_transition(
                &old(self).core_spec(),
                &final(self).core_spec(),
                n.poll_footprint(),
            ),
            r.is_ok() ==> final(self).core_spec().timelines_spec().retired(ticket),
    {
        hide(CacheCore::well_formed);
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, NvmeProgress::empty()));
        };
        let ghost tl = nvme.timeline;
        let ghost initial_nvme = nvme;
        let ghost poll_fp = nvme.poll_footprint();
        let mut progress = NvmeProgress::empty();
        let mut steps: usize = 0;
        proof {
            lemma_core_wf_facts(&self.core);
            lemma_core_transition_refl(&self.core, poll_fp);
        }
        if ticket.is_none() || ticket.timeline() != nvme.timeline
            || !self.core.timelines().contains_ticket(ticket) {
            self.nvme = Some(nvme);
            return Err((CoreError::InvalidArgument, progress));
        }
        while steps < BITMAP_CAPACITY as usize && (nvme.in_flight_len() > 0 || nvme.staged_len()
            > 0) && !self.core.timelines().is_retired(ticket)
            invariant
                steps <= BITMAP_CAPACITY as usize,
                progress.completed + progress.failed == steps,
                self.core.well_formed(),
                self.core.timelines_spec().bounds(),
                self.host.well_formed(&self.core),
                self.host_ctx.ready(self.core.instance_id(), self.core.node()),
                source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger_spec()@),
                self.host_ctx.source@.source() != self.host.ctx.source@.source(),
                self.host.buffer.align() % (transfer::nvme::DIRECT_IO_ALIGNMENT as usize) == 0,
                nvme.well_formed(&self.core),
                old(self).nvme matches Some(n) && n.timeline == tl,
                nvme.timeline == tl,
                initial_nvme == old(self).nvme.unwrap(),
                initial_nvme.poll_footprint() == poll_fp,
                nvme.pending_from(&initial_nvme),
                nvme.timeline != self.host.timeline,
                self.device == old(self).device,
                self.device matches Some(c) ==> {
                    &&& c.well_formed(&self.core)
                    &&& c.timeline_id_spec() != self.host.timeline
                    &&& nvme.timeline != c.timeline_id_spec()
                    &&& c.pinned_host_layout_spec() matches Some((tier, first, count)) ==> tier
                        == self.host.tier && first == self.host.first_frame && count
                        == self.host.num_frames
                },
                self.nvme is None,
                self.host == old(self).host,
                self.core.node_id_spec() == old(self).core.node_id_spec(),
                self.core.instance_id() == old(self).core.instance_id(),
                self.core.tiers_spec() == old(self).core.tiers_spec(),
                instance.id() == self.core.instance_id(),
                self.core.timelines_spec().valid_ticket(ticket),
                old(self).core.spec_spec()@.versions_monotonic(self.core.spec_spec()@),
                core_transition(&old(self).core, &self.core, poll_fp),
            decreases (BITMAP_CAPACITY as usize) - steps,
        {
            if nvme.staged_len() > 0 && !self.core.timelines().is_retired(ticket) {
                match nvme.submit(Ghost(&self.core)) {
                    Ok(n) => {
                        if n == 0 && nvme.in_flight_len() == 0 {
                            self.nvme = Some(nvme);
                            return Err((CoreError::WouldConflict, progress));
                        }
                    },
                    Err(e) => {
                        self.nvme = Some(nvme);
                        return Err((e, progress));
                    },
                }
            }
            if nvme.in_flight_len() == 0 {
                self.nvme = Some(nvme);
                return Err((CoreError::WouldConflict, progress));
            }
            let ghost before_step = self.core;
            let ghost before_nvme = nvme;
            let step_res = nvme.wait_one_completion(&mut self.core, Tracked(instance), progress);
            proof {
                let ghost none_nvme: Option<NvmeBackend> = None;
                Self::lemma_preserve_siblings(
                    &before_step,
                    &self.core,
                    before_nvme.poll_footprint(),
                    self.host,
                    &self.host_ctx,
                    &self.device,
                    &none_nvme,
                );
                lemma_versions_monotonic_trans(
                    old(self).core.spec_spec()@,
                    before_step.spec_spec()@,
                    self.core.spec_spec()@,
                );
                lemma_nvme_pending_from_trans(&initial_nvme, &before_nvme, &nvme);
                lemma_nvme_widen_poll_footprint(
                    &before_step,
                    &self.core,
                    &before_nvme,
                    &initial_nvme,
                );
                lemma_core_transition_trans_same(
                    &old(self).core,
                    &before_step,
                    &self.core,
                    poll_fp,
                );
                lemma_core_wf_facts(&self.core);
            }
            match step_res {
                Ok(p) => {
                    progress = p;
                    steps += 1;
                },
                Err((e, p)) => {
                    self.nvme = Some(nvme);
                    return Err((e, p));
                },
            }
        }
        self.nvme = Some(nvme);
        if self.core.timelines().is_retired(ticket) {
            Ok(progress)
        } else {
            Err((CoreError::WouldConflict, progress))
        }
    }
}

impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    /// Synchronize the backend device stream and publish all completed GPU writes.
    pub fn synchronize_cuda(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        BackendResult<DevicePollReport>)
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
        let Some(mut device) = self.device.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost poll_fp = device.poll_footprint();
        let ghost before_device = device;
        let res = device.synchronize_device(&mut self.core, Tracked(instance));
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
        res
    }

    /// Read `key` into caller host slice `dst`: hits Host DRAM directly (1-hop),
    /// or warms Host DRAM from GPU VRAM (2-hop D2H) or Local NVMe (2-hop `io_uring`)
    /// before reading.
    pub fn get_into_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst: &mut [u8],
    ) -> (r: BackendResult<Option<VersionId>>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r matches Ok(Some(v)) ==> final(self).core_spec().spec_spec()@.versions.contains_key(v)
                && final(dst)@.len() == final(self).core_spec().spec_spec()@.versions[v].len()
                && exists|read_ev: TransferEvent, cert: ValueCertificate| #[trigger]
                host_read_witness(&final(self).core_spec(), read_ev, cert, v, final(dst)@),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        if dst.len() != self.core.registered_tier_frame_size(self.host.tier) {
            return Err(CoreError::InvalidArgument);
        }
        // 1-hop: direct Host DRAM hit.

        if let Some((frame, _ver)) = self.core.lookup_tier(key, self.host.tier) {
            let ghost reader = self.host_ctx.source@.next_event();
            let ghost cert = self.core.frames_spec().frame(frame).cert.unwrap();
            let v = self.read_host_frame(Tracked(instance), Tracked(shares), frame, dst)?;
            assert(host_read_witness(&self.core, reader, cert, v, dst@));
            return Ok(Some(v));
        }
        // 2-hop: GPU VRAM -> Pinned Host DRAM -> Caller host slice.

        if let Some(gpu_tier) = self.gpu_tier() {
            let has_pinned = match &self.device {
                Some(c) => c.has_pinned_host(),
                None => false,
            };
            if has_pinned {
                if let Some((gpu_frame, _ver)) = self.core.lookup_tier(key, gpu_tier) {
                    let Some(mut device) = self.device.take() else {
                        return Err(CoreError::UnsupportedTier);
                    };
                    let ghost before_hop = self.core;
                    let ghost tl = device.timeline_id_spec();
                    let ghost src_ev = device.event_source_spec();
                    let ghost before_device = device;
                    let hop_res = device.copy_hop_device(
                        &mut self.core,
                        Tracked(instance),
                        Tracked(shares),
                        gpu_frame,
                        self.host.tier,
                    );
                    proof {
                        before_device.lemma_device_layout(&before_hop);
                        device.lemma_same_config(&before_device);
                        let ghost no_device: Option<D> = None;
                        Self::lemma_preserve_siblings(
                            &before_hop,
                            &self.core,
                            CoreFootprint::preserve_all(Some(tl), Some(src_ev)),
                            self.host,
                            &self.host_ctx,
                            &no_device,
                            &self.nvme,
                        );
                    }
                    self.device = Some(device);
                    let _ = hop_res?;
                    let _ = self.synchronize_cuda(Tracked(instance))?;
                    if let Some((dram_frame, _v)) = self.core.lookup_tier(key, self.host.tier) {
                        let v = self.read_host_frame(
                            Tracked(instance),
                            Tracked(shares),
                            dram_frame,
                            dst,
                        )?;
                        return Ok(Some(v));
                    }
                    return Err(CoreError::StaleReplica);
                }
            }
        }
        // 2-hop: Local NVMe -> Host DRAM -> Caller host slice.

        #[cfg(feature = "nvme")]
        if let Some(nvme_tier) = self.nvme_tier() {
            if self.core.lookup_tier(key, nvme_tier).is_some() {
                let (_dram_frame, ticket) = match self.warm_host_from_nvme(
                    Tracked(instance),
                    Tracked(shares),
                    key,
                ) {
                    Ok(pair) => pair,
                    Err((e, _)) => return Err(e),
                };
                if let Err((e, _)) = self.wait_nvme_ticket(Tracked(instance), ticket) {
                    return Err(e);
                }
                if let Some((dram_frame, _v)) = self.core.lookup_tier(key, self.host.tier) {
                    let v = self.read_host_frame(
                        Tracked(instance),
                        Tracked(shares),
                        dram_frame,
                        dst,
                    )?;
                    return Ok(Some(v));
                }
                return Err(CoreError::StaleReplica);
            }
        }
        Ok(None)
    }

    fn read_host_frame(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst: &mut [u8],
    ) -> (r: BackendResult<VersionId>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec_spec() == old(self).core.spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            r matches Ok(ver) ==> old(self).core.frames_spec().meta(frame).state == STATE_PUBLISHED
                && old(self).core.frames_spec().frame(frame).bound_version == Some(ver) && old(
                self,
            ).core.frames_spec().frame(frame).cert is Some && host_read_witness(
                &final(self).core,
                old(self).host_ctx.source@.next_event(),
                old(self).core.frames_spec().frame(frame).cert.unwrap(),
                ver,
                final(dst)@,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        let ghost before = self.core;
        let ghost src = self.host_ctx.source@.source();
        let res = self.host.read_frame_into_with(
            &mut self.host_ctx,
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst,
        );
        proof {
            if let Some(ref c) = self.device {
                c.lemma_device_layout(&before);
                c.preserve_across_core_transition(
                    &before,
                    &self.core,
                    CoreFootprint::preserve_all(None, Some(src)),
                );
            }
            lemma_preserve_nvme(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, Some(src)),
                &self.nvme,
            );
        }
        res
    }
}

} // verus!
