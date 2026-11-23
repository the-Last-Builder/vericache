#[cfg(verus_only)]
use crate::backends::cuda::{caller_stream_valid, scatter_offsets_valid, CudaBackend};
use crate::backends::*;
#[cfg(verus_only)]
use crate::kernel::CacheCore;
use crate::spec::*;
use transfer::context::*;
use transfer::cuda::{CudaSlice, CudaStream};
use transfer::domain::TransferEvent;
use vstd::prelude::*;

use super::LocalEngine;

verus! {

impl<'a> LocalEngine<'a> {
    /// Read `key` into caller's GPU `dst` on `dst_stream`:
    /// - 1-hop GPU VRAM hit (Ordered Handoff, zero host stall)
    /// - 1-hop Pinned Host DRAM hit (H2D Ordered Handoff, zero host stall)
    /// - 2-hop Local NVMe -> Pinned Host DRAM -> Caller GPU stream
    pub fn get_into_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dst: &CudaSlice,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<Option<(VersionId, Ticket, Ghost<TransferEvent>)>>)
        requires
            old(self).well_formed(),
            old(self).cuda matches Some(c) && caller_stream_valid(
                *old(dst_stream),
                &old(self).core,
                dst_timeline,
                c.timeline.stream,
            ) && dst_timeline != c.timeline.timeline_id,
            dst_timeline != old(self).host.timeline,
            old(self).nvme matches Some(n) ==> dst_timeline != n.timeline,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core.instance_id(),
            dst.slot_ref().loc.node == old(self).core.node(),
            old(self).cuda matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core.frames.tier_frame_size(c.gpu_tier),
            old(self).cuda matches Some(c) ==> (dst.slot_ref().loc.space is GpuAlloc
                ==> dst.slot_ref().loc.space->GpuAlloc_device == c.device),
            slot_is_owned(*old(dst_tok), old(self).core.instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core.citations()),
            old(self).core.disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core.frames.metas.len() ==> dst.slot_ref().loc != (#[trigger] old(
                    self,
                ).core.frames.frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda matches Some(c) && caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
                c.timeline.stream,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
    {
        hide(CacheCore::well_formed);
        let Some(gpu_tier) = self.gpu_tier() else {
            return Err(CoreError::UnsupportedTier);
        };
        let has_pinned = match &self.cuda {
            Some(c) => c.pinned_host.is_some(),
            None => false,
        };
        // 1-hop: GPU VRAM hit.
        if let Some((gpu_frame, _ver)) = self.core.lookup_tier(key, gpu_tier) {
            let v = self.gpu_read_frame(
                Tracked(instance),
                Tracked(shares),
                gpu_frame,
                dst_stream,
                dst_timeline,
                dst,
                Tracked(dst_tok),
            )?;
            return Ok(Some(v));
        }
        // 1-hop: Pinned Host DRAM hit.

        if has_pinned {
            if let Some((dram_frame, _ver)) = self.core.lookup_tier(key, self.host.tier) {
                let v = self.gpu_read_frame(
                    Tracked(instance),
                    Tracked(shares),
                    dram_frame,
                    dst_stream,
                    dst_timeline,
                    dst,
                    Tracked(dst_tok),
                )?;
                return Ok(Some(v));
            }
        }
        // 2-hop: Local NVMe -> Pinned Host DRAM -> Caller GPU stream.

        if has_pinned {
            if let Some(nvme_tier) = self.nvme_tier() {
                if self.core.lookup_tier(key, nvme_tier).is_some() {
                    let ghost before_warm = self.core;
                    let ghost nvme_tl = self.nvme.unwrap().timeline;
                    let ghost nvme_src = self.nvme.unwrap().ring.source().source();
                    let ghost backend_stream = self.cuda.unwrap().timeline.stream;
                    let warm_res = self.warm_host_from_nvme(
                        Tracked(instance),
                        Tracked(shares),
                        key,
                    );
                    proof {
                        lemma_core_transition_preserves_caller_stream(
                            &before_warm,
                            &self.core,
                            Some(nvme_tl),
                            Some(nvme_src),
                            FrameTouchMode::PreserveAllInit,
                            *dst_stream,
                            dst_timeline,
                            backend_stream,
                        );
                        lemma_core_transition_preserves_slot_disjoint_and_admitted(
                            &before_warm,
                            &self.core,
                            Some(nvme_tl),
                            Some(nvme_src),
                            FrameTouchMode::PreserveAllInit,
                            *dst_tok,
                            dst.slot_ref().loc,
                        );
                    }
                    let (_dram_frame, ticket) = match warm_res {
                        Ok(pair) => pair,
                        Err((e, _)) => return Err(e),
                    };
                    let ghost before_wait = self.core;
                    let wait_res = self.wait_nvme_ticket(Tracked(instance), ticket);
                    proof {
                        lemma_core_transition_preserves_caller_stream(
                            &before_wait,
                            &self.core,
                            Some(nvme_tl),
                            None,
                            FrameTouchMode::PreserveParkedInit,
                            *dst_stream,
                            dst_timeline,
                            backend_stream,
                        );
                        lemma_core_transition_preserves_slot_disjoint_and_admitted(
                            &before_wait,
                            &self.core,
                            Some(nvme_tl),
                            None,
                            FrameTouchMode::PreserveParkedInit,
                            *dst_tok,
                            dst.slot_ref().loc,
                        );
                    }
                    if let Err((e, _)) = wait_res {
                        return Err(e);
                    }
                    if let Some((dram_frame, _v)) = self.core.lookup_tier(key, self.host.tier) {
                        let v = self.gpu_read_frame(
                            Tracked(instance),
                            Tracked(shares),
                            dram_frame,
                            dst_stream,
                            dst_timeline,
                            dst,
                            Tracked(dst_tok),
                        )?;
                        return Ok(Some(v));
                    }
                    return Err(CoreError::StaleReplica);
                }
            }
        }
        Ok(None)
    }

    fn gpu_read_frame(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dst: &CudaSlice,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)>)
        requires
            old(self).well_formed(),
            old(self).cuda matches Some(c) && caller_stream_valid(
                *old(dst_stream),
                &old(self).core,
                dst_timeline,
                c.timeline.stream,
            ) && dst_timeline != c.timeline.timeline_id,
            dst_timeline != old(self).host.timeline,
            old(self).nvme matches Some(n) ==> dst_timeline != n.timeline,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core.instance_id(),
            dst.slot_ref().loc.node == old(self).core.node(),
            old(self).cuda matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core.frames.tier_frame_size(c.gpu_tier),
            old(self).cuda matches Some(c) ==> (dst.slot_ref().loc.space is GpuAlloc
                ==> dst.slot_ref().loc.space->GpuAlloc_device == c.device),
            slot_is_owned(*old(dst_tok), old(self).core.instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core.citations()),
            old(self).core.disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core.frames.metas.len() ==> dst.slot_ref().loc != (#[trigger] old(
                    self,
                ).core.frames.frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec == old(self).core.spec,
            final(self).cuda matches Some(c) && caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
                c.timeline.stream,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        let Some(cuda) = self.cuda.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        let res = cuda.get_into_gpu(
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst_stream,
            dst_timeline,
            dst,
            Tracked(dst_tok),
        );
        proof {
            let ghost none_cuda: Option<CudaBackend<'a>> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(dst_timeline),
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

    /// Scatter-read `key` into multiple caller GPU destination `CudaSlice`s on
    /// `dst_stream` (1-hop GPU / pinned DRAM Ordered Handoff, or 2-hop NVMe ->
    /// pinned DRAM -> GPU scatter).
    pub fn get_into_gpu_scatter(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dsts: &[CudaSlice<'_>],
        Tracked(dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(offsets): Ghost<Seq<usize>>,
    ) -> (r: Result<Option<(VersionId, Ticket)>, (CoreError, Option<Ticket>, usize)>)
        requires
            old(self).well_formed(),
            old(self).cuda matches Some(c) && caller_stream_valid(
                *old(dst_stream),
                &old(self).core,
                dst_timeline,
                c.timeline.stream,
            ) && dst_timeline != c.timeline.timeline_id,
            dst_timeline != old(self).host.timeline,
            old(self).nvme matches Some(n) ==> dst_timeline != n.timeline,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            old(self).cuda matches Some(c) ==> scatter_offsets_valid(
                dsts@,
                offsets,
                old(self).core.frames.tier_frame_size(c.gpu_tier),
            ),
            forall|i: int|
                0 <= i < dsts@.len() ==> {
                    &&& (#[trigger] dsts@[i]).instance_id() == old(self).core.instance_id()
                    &&& dsts@[i].slot_ref().loc.node == old(self).core.node()
                    &&& (old(self).cuda matches Some(c) ==> (
                    dsts@[i].slot_ref().loc.space is GpuAlloc
                        ==> dsts@[i].slot_ref().loc.space->GpuAlloc_device == c.device))
                    &&& old(dst_toks).dom().contains(i)
                    &&& slot_is_owned(
                        old(dst_toks)[i],
                        old(self).core.instance_id(),
                        dsts@[i].slot_ref().loc,
                    )
                    &&& write_admitted(slot_frontier(old(dst_toks)[i]), old(self).core.citations())
                    &&& old(self).core.disjoint_from_published(
                        ISet::empty().insert(dsts@[i].slot_ref().loc),
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(self).core.frames.metas.len() ==> dsts@[i].slot_ref().loc != (
                        #[trigger] old(self).core.frames.frame(f)).slot.loc
                },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).cuda matches Some(c) && caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
                c.timeline.stream,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            old(self).core.spec@.versions_monotonic(final(self).core.spec@),
            r matches Err((_, Some(ticket), completed_slices)) ==> completed_slices > 0
                && completed_slices <= dsts.len() && ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline,
            r matches Err((_, None, completed_slices)) ==> completed_slices == 0,
    {
        hide(CacheCore::well_formed);
        let Some(gpu_tier) = self.gpu_tier() else {
            return Err((CoreError::UnsupportedTier, None, 0));
        };
        let has_pinned = match &self.cuda {
            Some(c) => c.pinned_host.is_some(),
            None => false,
        };
        if let Some((gpu_frame, _ver)) = self.core.lookup_tier(key, gpu_tier) {
            let v = self.gpu_scatter_frame(
                Tracked(instance),
                Tracked(shares),
                gpu_frame,
                dst_stream,
                dst_timeline,
                dsts,
                Tracked(dst_toks),
                Ghost(offsets),
            )?;
            return Ok(Some(v));
        }
        if has_pinned {
            if let Some((dram_frame, _ver)) = self.core.lookup_tier(key, self.host.tier) {
                let v = self.gpu_scatter_frame(
                    Tracked(instance),
                    Tracked(shares),
                    dram_frame,
                    dst_stream,
                    dst_timeline,
                    dsts,
                    Tracked(dst_toks),
                    Ghost(offsets),
                )?;
                return Ok(Some(v));
            }
        }
        if has_pinned {
            if let Some(nvme_tier) = self.nvme_tier() {
                if self.core.lookup_tier(key, nvme_tier).is_some() {
                    let ghost before_warm = self.core;
                    let ghost nvme_tl = self.nvme.unwrap().timeline;
                    let ghost nvme_src = self.nvme.unwrap().ring.source().source();
                    let ghost backend_stream = self.cuda.unwrap().timeline.stream;
                    let warm_res = self.warm_host_from_nvme(
                        Tracked(instance),
                        Tracked(shares),
                        key,
                    );
                    proof {
                        lemma_core_transition_preserves_caller_stream(
                            &before_warm,
                            &self.core,
                            Some(nvme_tl),
                            Some(nvme_src),
                            FrameTouchMode::PreserveAllInit,
                            *dst_stream,
                            dst_timeline,
                            backend_stream,
                        );
                        assert forall|k: int| 0 <= k < dsts@.len() implies {
                            &&& write_admitted(slot_frontier(dst_toks[k]), self.core.citations())
                            &&& self.core.disjoint_from_published(
                                ISet::empty().insert((#[trigger] dsts@[k]).slot_ref().loc),
                            )
                            &&& forall|f: FrameId|
                                f.0 < self.core.frames.metas.len() ==> dsts@[k].slot_ref().loc != (
                                #[trigger] self.core.frames.frame(f)).slot.loc
                        } by {
                            lemma_core_transition_preserves_slot_disjoint_and_admitted(
                                &before_warm,
                                &self.core,
                                Some(nvme_tl),
                                Some(nvme_src),
                                FrameTouchMode::PreserveAllInit,
                                dst_toks[k],
                                dsts@[k].slot_ref().loc,
                            );
                        }
                    }
                    let (_dram_frame, ticket) = match warm_res {
                        Ok(pair) => pair,
                        Err((e, _)) => return Err((e, None, 0)),
                    };
                    let ghost before_wait = self.core;
                    let wait_res = self.wait_nvme_ticket(Tracked(instance), ticket);
                    proof {
                        lemma_core_transition_preserves_caller_stream(
                            &before_wait,
                            &self.core,
                            Some(nvme_tl),
                            None,
                            FrameTouchMode::PreserveParkedInit,
                            *dst_stream,
                            dst_timeline,
                            backend_stream,
                        );
                        assert forall|k: int| 0 <= k < dsts@.len() implies {
                            &&& write_admitted(slot_frontier(dst_toks[k]), self.core.citations())
                            &&& self.core.disjoint_from_published(
                                ISet::empty().insert((#[trigger] dsts@[k]).slot_ref().loc),
                            )
                            &&& forall|f: FrameId|
                                f.0 < self.core.frames.metas.len() ==> dsts@[k].slot_ref().loc != (
                                #[trigger] self.core.frames.frame(f)).slot.loc
                        } by {
                            lemma_core_transition_preserves_slot_disjoint_and_admitted(
                                &before_wait,
                                &self.core,
                                Some(nvme_tl),
                                None,
                                FrameTouchMode::PreserveParkedInit,
                                dst_toks[k],
                                dsts@[k].slot_ref().loc,
                            );
                        }
                    }
                    if let Err((e, _)) = wait_res {
                        return Err((e, None, 0));
                    }
                    if let Some((dram_frame, _v)) = self.core.lookup_tier(key, self.host.tier) {
                        let v = self.gpu_scatter_frame(
                            Tracked(instance),
                            Tracked(shares),
                            dram_frame,
                            dst_stream,
                            dst_timeline,
                            dsts,
                            Tracked(dst_toks),
                            Ghost(offsets),
                        )?;
                        return Ok(Some(v));
                    }
                    return Err((CoreError::StaleReplica, None, 0));
                }
            }
        }
        Ok(None)
    }

    fn gpu_scatter_frame(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        frame: FrameId,
        dst_stream: &mut CudaStream,
        dst_timeline: u8,
        dsts: &[CudaSlice<'_>],
        Tracked(dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(offsets): Ghost<Seq<usize>>,
    ) -> (r: Result<(VersionId, Ticket), (CoreError, Option<Ticket>, usize)>)
        requires
            old(self).well_formed(),
            old(self).cuda matches Some(c) && caller_stream_valid(
                *old(dst_stream),
                &old(self).core,
                dst_timeline,
                c.timeline.stream,
            ) && dst_timeline != c.timeline.timeline_id,
            dst_timeline != old(self).host.timeline,
            old(self).nvme matches Some(n) ==> dst_timeline != n.timeline,
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            old(self).cuda matches Some(c) ==> scatter_offsets_valid(
                dsts@,
                offsets,
                old(self).core.frames.tier_frame_size(c.gpu_tier),
            ),
            forall|i: int|
                0 <= i < dsts@.len() ==> {
                    &&& (#[trigger] dsts@[i]).instance_id() == old(self).core.instance_id()
                    &&& dsts@[i].slot_ref().loc.node == old(self).core.node()
                    &&& (old(self).cuda matches Some(c) ==> (
                    dsts@[i].slot_ref().loc.space is GpuAlloc
                        ==> dsts@[i].slot_ref().loc.space->GpuAlloc_device == c.device))
                    &&& old(dst_toks).dom().contains(i)
                    &&& slot_is_owned(
                        old(dst_toks)[i],
                        old(self).core.instance_id(),
                        dsts@[i].slot_ref().loc,
                    )
                    &&& write_admitted(slot_frontier(old(dst_toks)[i]), old(self).core.citations())
                    &&& old(self).core.disjoint_from_published(
                        ISet::empty().insert(dsts@[i].slot_ref().loc),
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(self).core.frames.metas.len() ==> dsts@[i].slot_ref().loc != (
                        #[trigger] old(self).core.frames.frame(f)).slot.loc
                },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec == old(self).core.spec,
            final(self).cuda matches Some(c) && caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
                c.timeline.stream,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            r matches Err((_, Some(ticket), completed_slices)) ==> completed_slices > 0
                && completed_slices <= dsts.len() && ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline,
            r matches Err((_, None, completed_slices)) ==> completed_slices == 0,
    {
        hide(CacheCore::well_formed);
        let Some(cuda) = self.cuda.take() else {
            return Err((CoreError::UnsupportedTier, None, 0));
        };
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        let res = cuda.get_into_gpu_scatter(
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            frame,
            dst_stream,
            dst_timeline,
            dsts,
            Tracked(dst_toks),
            Ghost(offsets),
        );
        proof {
            let ghost none_cuda: Option<CudaBackend<'a>> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                Some(dst_timeline),
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
}

} // verus!
