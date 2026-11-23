use crate::backends::cuda::CudaPollReport;
#[cfg(verus_only)]
use crate::backends::cuda::{scatter_offsets_valid, CudaBackend};
#[cfg(verus_only)]
use crate::backends::nvme::NvmeBackend;
use crate::backends::nvme::NvmeProgress;
use crate::backends::*;
#[cfg(verus_only)]
use crate::frame::*;
#[cfg(verus_only)]
use crate::kernel::CacheCore;
use crate::provenance::ValueCertificate;
#[cfg(verus_only)]
use crate::provenance::*;
use crate::spec::*;
use crate::timeline::BITMAP_CAPACITY;
use transfer::context::*;
use transfer::cuda::CudaSlice;
#[cfg(verus_only)]
use transfer::domain::TransferEvent;
use vstd::prelude::*;

use super::LocalEngine;

verus! {

impl<'a> LocalEngine<'a> {
    /// Synchronously write `data` into Host DRAM and publish into `CacheCore`.
    pub fn put_host(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        data: &[u8],
    ) -> (r: BackendResult<(VersionId, FrameId)>)
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
            match r {
                Err(_) => final(self).core.spec == old(self).core.spec,
                Ok((ver, frame)) => {
                    &&& final(self).host.owns(frame)
                    &&& final(self).core.spec@.versions == old(self).core.spec@.versions.insert(
                        ver,
                        data@,
                    )
                    &&& final(self).core.spec@.catalog == old(self).core.spec@.catalog.insert(
                        key,
                        ver,
                    )
                    &&& final(self).core.catalog.slot(key, final(self).host.tier.index() as int)
                        == Some(frame)
                    &&& final(self).core.frames.meta(frame).state == STATE_PUBLISHED
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
            if let Some(ref c) = self.cuda {
                lemma_core_transition_preserves_cuda(
                    &before,
                    &self.core,
                    Some(self.host.timeline),
                    Some(src),
                    FrameTouchMode::PreserveAllInit,
                    c,
                );
            }
            if let Some(ref n) = self.nvme {
                lemma_core_transition_preserves_nvme(
                    &before,
                    &self.core,
                    Some(self.host.timeline),
                    Some(src),
                    FrameTouchMode::PreserveAllInit,
                    n,
                );
            }
        }
        res
    }

    /// Asynchronously write one full frame from caller's `CudaSlice` into
    /// `dst_tier` (GPU VRAM or attached pinned Host DRAM).
    pub fn put_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        dst_tier: TierId,
        key: ObjectKey,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            src.well_formed(),
            src.instance_id() == old(self).core.instance_id(),
            src.slot_ref().loc.node == old(self).core.node(),
            old(self).cuda matches Some(c) ==> src.slot_ref().sub_len == old(
                self,
            ).core.frames.tier_frame_size(c.gpu_tier),
            old(src_perm).valid_local(old(self).core.instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(self).core.citations()),
            old(self).cuda matches Some(c) ==> (src.slot_ref().loc.space is GpuAlloc
                && src.slot_ref().loc.space->GpuAlloc_device != c.device ==> (old(
                src_perm,
            ).frontier().last_writer matches Some(w) ==> !(w.source is RdmaQueuePair))),
            replica_valid(old(self).core.ledger@, old(self).core.node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(self).core.frames.metas.len() && (#[trigger] old(self).core.frames.meta(
                    f,
                )).state == STATE_EMPTY ==> src.slot_ref().loc != old(self).core.frames.frame(
                    f,
                ).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            final(self).core.spec == old(self).core.spec,
    {
        hide(CacheCore::well_formed);
        let Some(mut cuda) = self.cuda.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = cuda.timeline.timeline_id;
        let ghost src_ev = cuda.timeline.stream.source().source();
        let res = cuda.put_gpu(
            &mut self.core,
            Tracked(instance),
            dst_tier,
            key,
            src,
            Tracked(src_perm),
            Ghost(src_cert),
        );
        proof {
            let ghost none_cuda: Option<CudaBackend<'a>> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &none_cuda,
                &self.nvme,
            );
        }
        self.cuda = Some(cuda);
        res
    }

    /// Asynchronously gather multiple caller `CudaSlice`s into one GPU-VRAM frame.
    pub fn put_gpu_scatter(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        key: ObjectKey,
        srcs: &[CudaSlice<'_>],
        Tracked(src_perms): Tracked<&mut Map<int, ReadPerm>>,
        Ghost(src_certs): Ghost<Seq<ValueCertificate>>,
        Ghost(offsets): Ghost<Seq<usize>>,
        Ghost(bytes): Ghost<Seq<u8>>,
    ) -> (r: BackendResult<(VersionId, FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core.instance_id(),
            old(self).cuda matches Some(c) ==> {
                &&& scatter_offsets_valid(
                    srcs@,
                    offsets,
                    old(self).core.frames.tier_frame_size(c.gpu_tier),
                )
                &&& bytes.len() == old(self).core.frames.tier_frame_size(c.gpu_tier)
            },
            src_certs.len() == srcs@.len(),
            forall|i: int|
                0 <= i < srcs@.len() ==> {
                    &&& (#[trigger] srcs@[i]).instance_id() == old(self).core.instance_id()
                    &&& srcs@[i].slot_ref().loc.node == old(self).core.node()
                    &&& old(src_perms).dom().contains(i)
                    &&& old(src_perms)[i].valid_local(
                        old(self).core.instance_id(),
                        srcs@[i].slot_ref().loc,
                    )
                    &&& read_admitted(old(src_perms)[i].frontier(), old(self).core.citations())
                    &&& (old(self).cuda matches Some(c) ==> (
                    srcs@[i].slot_ref().loc.space is GpuAlloc
                        && srcs@[i].slot_ref().loc.space->GpuAlloc_device != c.device ==> (old(
                        src_perms,
                    )[i].frontier().last_writer matches Some(w) ==> !(w.source is RdmaQueuePair))))
                    &&& replica_valid(old(self).core.ledger@, old(self).core.node(), src_certs[i])
                    &&& source_slice_valid(src_certs[i], srcs@[i].slot_ref())
                    &&& src_certs[i].slot.well_formed()
                    &&& src_certs[i].bytes.len() == src_certs[i].slot.sub_len
                    &&& source_slice_bytes(src_certs[i], srcs@[i].slot_ref()) == bytes.subrange(
                        offsets[i] as int,
                        offsets[i + 1] as int,
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(self).core.frames.metas.len() && (#[trigger] old(
                            self,
                        ).core.frames.meta(f)).state == STATE_EMPTY ==> srcs@[i].slot_ref().loc
                            != old(self).core.frames.frame(f).slot.loc
                },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            final(self).core.spec == old(self).core.spec,
    {
        hide(CacheCore::well_formed);
        let Some(mut cuda) = self.cuda.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = cuda.timeline.timeline_id;
        let ghost src_ev = cuda.timeline.stream.source().source();
        let res = cuda.put_gpu_scatter(
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
            let ghost none_cuda: Option<CudaBackend<'a>> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &none_cuda,
                &self.nvme,
            );
        }
        self.cuda = Some(cuda);
        res
    }

    /// Stage and submit an async Host-DRAM -> Local-NVMe tier-down copy hop.
    pub fn tier_dram_to_nvme(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
    ) -> (r: Result<(FrameId, Ticket), (CoreError, Option<Ticket>)>)
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
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (final(self).nvme matches Some(
                n,
            ) && ticket.timeline_id() == n.timeline) && final(self).core.timelines.valid_ticket(
                ticket,
            ),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme matches Some(n) && ticket.timeline_id() == n.timeline)
                && final(self).core.timelines.valid_ticket(ticket),
    {
        hide(CacheCore::well_formed);
        let Some((src_frame, _ver)) = self.core.lookup_tier(key, self.host.tier) else {
            return Err((CoreError::Miss, None));
        };
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, None));
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
            let ghost none_nvme: Option<NvmeBackend> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &none_nvme,
            );
        }
        let (dst_frame, ticket) = match staged {
            Ok(pair) => pair,
            Err(e) => {
                self.nvme = Some(nvme);
                return Err((e, None));
            },
        };
        let submit_res = nvme.submit(Ghost(&self.core));
        let rem_staged = nvme.staged_len();
        self.nvme = Some(nvme);
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
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
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Ok((_, ticket)) ==> ticket != Ticket::NONE && (final(self).nvme matches Some(
                n,
            ) && ticket.timeline_id() == n.timeline) && final(self).core.timelines.valid_ticket(
                ticket,
            ),
            r matches Err((_, Some(ticket))) ==> ticket != Ticket::NONE && (
            final(self).nvme matches Some(n) && ticket.timeline_id() == n.timeline)
                && final(self).core.timelines.valid_ticket(ticket),
            old(self).nvme matches Some(n) ==> (final(self).nvme matches Some(n2) && n2.timeline
                == n.timeline),
            old(self).nvme matches Some(n) ==> core_transition(
                &old(self).core,
                &final(self).core,
                Some(n.timeline),
                Some(n.ring.source().source()),
                FrameTouchMode::PreserveAllInit,
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
                    Some(n.timeline),
                    Some(n.ring.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err((CoreError::Miss, None));
        };
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, None));
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
            let ghost none_nvme: Option<NvmeBackend> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(tl),
                Some(src_ev),
                FrameTouchMode::PreserveAllInit,
                self.host,
                &self.host_ctx,
                &self.cuda,
                &none_nvme,
            );
        }
        let (dst_frame, ticket) = match staged {
            Ok(pair) => pair,
            Err(e) => {
                self.nvme = Some(nvme);
                return Err((e, None));
            },
        };
        let submit_res = nvme.submit(Ghost(&self.core));
        let rem_staged = nvme.staged_len();
        self.nvme = Some(nvme);
        match submit_res {
            Ok(n) if n > 0 && rem_staged == 0 => Ok((dst_frame, ticket)),
            Ok(_) => Err((CoreError::WouldConflict, Some(ticket))),
            Err(e) => Err((e, Some(ticket))),
        }
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
            instance.id() == old(self).core.instance_id(),
            old(self).core.timelines.valid_ticket(ticket),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda == old(self).cuda,
            final(self).core.node_id == old(self).core.node_id,
            final(self).core.instance_id() == old(self).core.instance_id(),
            final(self).core.tiers == old(self).core.tiers,
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            old(self).nvme matches Some(n) ==> (final(self).nvme matches Some(n2) && n2.timeline
                == n.timeline),
            old(self).nvme matches Some(n) ==> core_transition(
                &old(self).core,
                &final(self).core,
                Some(n.timeline),
                None,
                FrameTouchMode::PreserveParkedInit,
            ),
            r.is_ok() ==> final(self).core.timelines.retired(ticket),
    {
        hide(CacheCore::well_formed);
        let Some(mut nvme) = self.nvme.take() else {
            return Err((CoreError::UnsupportedTier, NvmeProgress::empty()));
        };
        let ghost tl = nvme.timeline;
        let mut progress = NvmeProgress::empty();
        let mut steps: usize = 0;
        proof {
            lemma_core_wf_facts(&self.core);
            lemma_core_transition_refl(
                &self.core,
                Some(tl),
                None,
                FrameTouchMode::PreserveParkedInit,
            );
        }
        if ticket.timeline() != nvme.timeline {
            self.nvme = Some(nvme);
            return Err((CoreError::InvalidArgument, progress));
        }
        while steps < BITMAP_CAPACITY as usize && (nvme.in_flight_len() > 0 || nvme.staged_len()
            > 0) && !self.core.timelines.is_retired(ticket)
            invariant
                steps <= BITMAP_CAPACITY as usize,
                progress.completed + progress.failed == steps,
                self.core.well_formed(),
                self.core.timelines.bounds(),
                self.host.well_formed(&self.core),
                self.host_ctx.ready(self.core.instance_id(), self.core.node()),
                source_fresh_for_ledger(self.host_ctx.source@, self.core.ledger@),
                self.host_ctx.source@.source() != self.host.ctx.source@.source(),
                self.host.buffer.align() % (transfer::nvme::DIRECT_IO_ALIGNMENT as usize) == 0,
                nvme.well_formed(&self.core),
                old(self).nvme matches Some(n) && n.timeline == tl,
                nvme.timeline == tl,
                nvme.timeline != self.host.timeline,
                self.cuda == old(self).cuda,
                self.cuda matches Some(c) ==> {
                    &&& c.well_formed(&self.core)
                    &&& c.timeline.timeline_id != self.host.timeline
                    &&& nvme.timeline != c.timeline.timeline_id
                    &&& c.pinned_host matches Some(h) ==> h.tier == self.host.tier && h.first_frame
                        == self.host.first_frame && h.num_frames == self.host.num_frames
                },
                self.nvme is None,
                self.host == old(self).host,
                self.core.node_id == old(self).core.node_id,
                self.core.instance_id() == old(self).core.instance_id(),
                self.core.tiers == old(self).core.tiers,
                instance.id() == self.core.instance_id(),
                self.core.timelines.valid_ticket(ticket),
                old(self).core.spec@.versions_monotonic(self.core.spec@),
                core_transition(
                    &old(self).core,
                    &self.core,
                    Some(tl),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                ),
            decreases (BITMAP_CAPACITY as usize) - steps,
        {
            if nvme.staged_len() > 0 && !self.core.timelines.is_retired(ticket) {
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
            let step_res = nvme.wait_one_completion(&mut self.core, Tracked(instance), progress);
            proof {
                let ghost none_nvme: Option<NvmeBackend> = None;
                Self::lemma_preserve_siblings(
                    &before_step,
                    &self.core,
                    Some(tl),
                    None,
                    FrameTouchMode::PreserveParkedInit,
                    self.host,
                    &self.host_ctx,
                    &self.cuda,
                    &none_nvme,
                );
                lemma_versions_monotonic_trans(
                    old(self).core.spec@,
                    before_step.spec@,
                    self.core.spec@,
                );
                lemma_core_transition_trans(
                    &old(self).core,
                    &before_step,
                    &self.core,
                    Some(tl),
                    None,
                    FrameTouchMode::PreserveParkedInit,
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
        if self.core.timelines.is_retired(ticket) {
            Ok(progress)
        } else {
            Err((CoreError::WouldConflict, progress))
        }
    }

    /// Synchronize the backend CUDA stream and publish all completed GPU writes.
    pub fn synchronize_cuda(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        BackendResult<CudaPollReport>)
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
        let Some(mut cuda) = self.cuda.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost tl = cuda.timeline.timeline_id;
        let res = cuda.synchronize(&mut self.core, Tracked(instance));
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
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r matches Ok(Some(v)) ==> final(self).core.spec@.versions.contains_key(v)
                && final(dst)@.len() == final(self).core.spec@.versions[v].len() && exists|
                read_ev: TransferEvent,
                cert: ValueCertificate,
            | #[trigger]
                host_read_witness(&final(self).core, read_ev, cert, v, final(dst)@),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        if dst.len() != self.core.tier_frame_size(self.host.tier) {
            return Err(CoreError::InvalidArgument);
        }
        // 1-hop: direct Host DRAM hit.

        if let Some((frame, _ver)) = self.core.lookup_tier(key, self.host.tier) {
            let v = self.read_host_frame(Tracked(instance), Tracked(shares), frame, dst)?;
            return Ok(Some(v));
        }
        // 2-hop: GPU VRAM -> Pinned Host DRAM -> Caller host slice.

        if let Some(gpu_tier) = self.gpu_tier() {
            let has_pinned = match &self.cuda {
                Some(c) => c.pinned_host.is_some(),
                None => false,
            };
            if has_pinned {
                if let Some((gpu_frame, _ver)) = self.core.lookup_tier(key, gpu_tier) {
                    let Some(mut cuda) = self.cuda.take() else {
                        return Err(CoreError::UnsupportedTier);
                    };
                    let ghost before_hop = self.core;
                    let ghost tl = cuda.timeline.timeline_id;
                    let ghost src_ev = cuda.timeline.stream.source().source();
                    let hop_res = cuda.copy_hop_cuda(
                        &mut self.core,
                        Tracked(instance),
                        Tracked(shares),
                        gpu_frame,
                        self.host.tier,
                    );
                    proof {
                        let ghost none_cuda: Option<CudaBackend<'a>> = None;
                        Self::lemma_preserve_siblings(
                            &before_hop,
                            &self.core,
                            Some(tl),
                            Some(src_ev),
                            FrameTouchMode::PreserveAllInit,
                            self.host,
                            &self.host_ctx,
                            &none_cuda,
                            &self.nvme,
                        );
                    }
                    self.cuda = Some(cuda);
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
            final(self).core.spec == old(self).core.spec,
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(dst)@.len() == old(dst)@.len(),
            r matches Ok(ver) ==> old(self).core.frames.meta(frame).state == STATE_PUBLISHED && old(
                self,
            ).core.frames.frame(frame).bound_version == Some(ver) && old(self).core.frames.frame(
                frame,
            ).cert is Some && host_read_witness(
                &final(self).core,
                old(self).host_ctx.source@.next_event(),
                old(self).core.frames.frame(frame).cert.unwrap(),
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
            if let Some(ref c) = self.cuda {
                lemma_core_transition_preserves_cuda(
                    &before,
                    &self.core,
                    None,
                    Some(src),
                    FrameTouchMode::PreserveAllInit,
                    c,
                );
            }
            if let Some(ref n) = self.nvme {
                lemma_core_transition_preserves_nvme(
                    &before,
                    &self.core,
                    None,
                    Some(src),
                    FrameTouchMode::PreserveAllInit,
                    n,
                );
            }
        }
        res
    }
}

} // verus!
