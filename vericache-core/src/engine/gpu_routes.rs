#[cfg(verus_only)]
use crate::backends::device::device_scatter_offsets_valid;
#[cfg(verus_only)]
use crate::backends::DeviceStream;
use crate::backends::{BackendResult, DeviceBackend, DeviceSlice};
#[cfg(verus_only)]
use crate::kernel::CacheCore;
use crate::spec::*;
use transfer::context::*;
use transfer::domain::TransferEvent;
#[cfg(verus_only)]
use transfer_core::memory::RdmaRegisterableSlab;
#[cfg(verus_only)]
use vericache_kernel::compose::{
    lemma_core_transition_preserves_slot_disjoint_and_admitted, CoreFootprint,
};
use vstd::prelude::*;

#[cfg(verus_only)]
use super::nvme_storage::nvme_timeline_distinct;
use super::{is_caller_device_timeline, LocalEngine};

verus! {

impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    /// Read `key` into caller's GPU `dst` on `dst_stream`:
    /// - 1-hop GPU VRAM hit (Ordered Handoff, zero host stall)
    /// - 1-hop Pinned Host DRAM hit (H2D Ordered Handoff, zero host stall)
    /// - 2-hop Local NVMe -> Pinned Host DRAM -> Caller GPU stream
    pub fn get_into_gpu(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst_stream: &mut D::Stream,
        dst_timeline: TimelineId,
        dst: &D::Slice<'_>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<Option<(VersionId, Ticket, Ghost<TransferEvent>)>>)
        requires
            old(self).well_formed(),
            old(self).device_spec() matches Some(c) && c.caller_stream_valid(
                *old(dst_stream),
                &old(self).core_spec(),
                dst_timeline,
            ) && dst_timeline != c.timeline_id_spec(),
            dst_timeline != old(self).host_spec().timeline_spec(),
            nvme_timeline_distinct(&old(self).nvme_spec(), dst_timeline),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core_spec().instance_id(),
            dst.slot_ref().loc.node == old(self).core_spec().node(),
            old(self).device_spec() matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
            old(self).device_spec() matches Some(c) ==> (dst.slot_ref().loc.space.is_gpu()
                ==> dst.slot_ref().loc.space.gpu_device() == c.device_spec()),
            slot_is_owned(*old(dst_tok), old(self).core_spec().instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core_spec().citations()),
            old(self).core_spec().disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core_spec().frames_spec().metas_spec().len() ==> dst.slot_ref().loc
                    != (#[trigger] old(self).core_spec().frames_spec().frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() matches Some(c) && c.caller_stream_valid(
                *final(dst_stream),
                &final(self).core_spec(),
                dst_timeline,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
    {
        hide(CacheCore::well_formed);
        let Some(device) = self.device.as_ref() else {
            return Err(CoreError::UnsupportedTier);
        };
        let gpu_tier = device.device_tier_id();
        if dst_timeline == device.timeline_id() || !is_caller_device_timeline::<D>(
            &self.core,
            dst_timeline,
            dst_stream,
        ) || self.core.tier_frame_size(gpu_tier) != Some(dst.len()) {
            return Err(CoreError::InvalidArgument);
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            dst_timeline,
            dst_stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let has_pinned = device.has_pinned_host();
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

        #[cfg(feature = "nvme")]
        if has_pinned {
            if let Some(nvme_tier) = self.nvme_tier() {
                if self.core.lookup_tier(key, nvme_tier).is_some() {
                    let ghost before_warm = self.core;
                    let ghost nvme_tl = self.nvme.unwrap().timeline;
                    let ghost nvme_src = self.nvme.unwrap().ring.source().source();
                    let ghost backend = self.device.unwrap();
                    proof {
                        backend.lemma_device_layout(&before_warm);
                        backend.lemma_caller_stream_valid(*dst_stream, &before_warm, dst_timeline);
                        dst_stream.lemma_source_kind();
                    }
                    let warm_res = self.warm_host_from_nvme(
                        Tracked(instance),
                        Tracked(shares),
                        key,
                    );
                    proof {
                        backend.preserve_caller_stream(
                            &before_warm,
                            &self.core,
                            CoreFootprint::preserve_all(Some(nvme_tl), Some(nvme_src)),
                            *dst_stream,
                            dst_timeline,
                        );
                        lemma_core_transition_preserves_slot_disjoint_and_admitted(
                            &before_warm,
                            &self.core,
                            CoreFootprint::preserve_all(Some(nvme_tl), Some(nvme_src)),
                            *dst_tok,
                            dst.slot_ref().loc,
                        );
                    }
                    let (_dram_frame, ticket) = match warm_res {
                        Ok(pair) => pair,
                        Err((e, _)) => return Err(e),
                    };
                    let ghost before_wait = self.core;
                    let ghost poll_fp = self.nvme.unwrap().poll_footprint();
                    let wait_res = self.wait_nvme_ticket(Tracked(instance), ticket);
                    proof {
                        backend.lemma_device_layout(&self.core);
                        assert(self.core.frames_spec().tier_frame_size(backend.device_tier_spec())
                            == backend.buffer_spec().reg_slot_len());
                        backend.preserve_caller_stream(
                            &before_wait,
                            &self.core,
                            poll_fp,
                            *dst_stream,
                            dst_timeline,
                        );
                        lemma_core_transition_preserves_slot_disjoint_and_admitted(
                            &before_wait,
                            &self.core,
                            poll_fp,
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
        dst_stream: &mut D::Stream,
        dst_timeline: TimelineId,
        dst: &D::Slice<'_>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(VersionId, Ticket, Ghost<TransferEvent>)>)
        requires
            old(self).well_formed(),
            old(self).device matches Some(c) && c.caller_stream_valid(
                *old(dst_stream),
                &old(self).core,
                dst_timeline,
            ) && dst_timeline != c.timeline_id_spec(),
            dst_timeline != old(self).host.timeline,
            nvme_timeline_distinct(&old(self).nvme, dst_timeline),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            dst.well_formed(),
            dst.instance_id() == old(self).core.instance_id(),
            dst.slot_ref().loc.node == old(self).core.node(),
            old(self).device matches Some(c) ==> dst.slot_ref().sub_len == old(
                self,
            ).core.frames_spec().tier_frame_size(c.device_tier_spec()),
            old(self).device matches Some(c) ==> (dst.slot_ref().loc.space.is_gpu()
                ==> dst.slot_ref().loc.space.gpu_device() == c.device_spec()),
            slot_is_owned(*old(dst_tok), old(self).core.instance_id(), dst.slot_ref().loc),
            write_admitted(slot_frontier(*old(dst_tok)), old(self).core.citations()),
            old(self).core.disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            forall|f: FrameId|
                f.0 < old(self).core.frames_spec().metas_spec().len() ==> dst.slot_ref().loc != (
                #[trigger] old(self).core.frames_spec().frame(f)).slot.loc,
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec_spec() == old(self).core.spec_spec(),
            final(self).device matches Some(c) && c.caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        hide(CacheCore::well_formed);
        let Some(device) = self.device.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        proof {
            device.lemma_caller_stream_valid(*dst_stream, &before, dst_timeline);
            dst_stream.lemma_source_kind();
        }
        let res = device.get_into_device(
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
            let ghost no_device: Option<D> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(dst_timeline), Some(src_ev)),
                self.host,
                &self.host_ctx,
                &no_device,
                &self.nvme,
            );
        }
        self.device = Some(device);
        res
    }

    /// Scatter-read `key` into multiple caller GPU destination device slices on
    /// `dst_stream` (1-hop GPU / pinned DRAM Ordered Handoff, or 2-hop NVMe ->
    /// pinned DRAM -> GPU scatter).
    pub fn get_into_gpu_scatter(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        key: ObjectKey,
        dst_stream: &mut D::Stream,
        dst_timeline: TimelineId,
        dsts: &[D::Slice<'_>],
        Tracked(dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(offsets): Ghost<Seq<usize>>,
    ) -> (r: Result<Option<(VersionId, Ticket)>, (CoreError, Option<Ticket>, usize)>)
        requires
            old(self).well_formed(),
            old(self).device_spec() matches Some(c) && c.caller_stream_valid(
                *old(dst_stream),
                &old(self).core_spec(),
                dst_timeline,
            ) && dst_timeline != c.timeline_id_spec(),
            dst_timeline != old(self).host_spec().timeline_spec(),
            nvme_timeline_distinct(&old(self).nvme_spec(), dst_timeline),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
            old(self).device_spec() matches Some(c) ==> device_scatter_offsets_valid(
                dsts@,
                offsets,
                old(self).core_spec().frames_spec().tier_frame_size(c.device_tier_spec()),
            ),
            forall|i: int|
                0 <= i < dsts@.len() ==> {
                    &&& (#[trigger] dsts@[i]).instance_id() == old(self).core_spec().instance_id()
                    &&& dsts@[i].slot_ref().loc.node == old(self).core_spec().node()
                    &&& (old(self).device_spec() matches Some(c) ==> (
                    dsts@[i].slot_ref().loc.space.is_gpu()
                        ==> dsts@[i].slot_ref().loc.space.gpu_device() == c.device_spec()))
                    &&& old(dst_toks).dom().contains(i)
                    &&& slot_is_owned(
                        old(dst_toks)[i],
                        old(self).core_spec().instance_id(),
                        dsts@[i].slot_ref().loc,
                    )
                    &&& write_admitted(
                        slot_frontier(old(dst_toks)[i]),
                        old(self).core_spec().citations(),
                    )
                    &&& old(self).core_spec().disjoint_from_published(
                        ISet::empty().insert(dsts@[i].slot_ref().loc),
                    )
                    &&& forall|f: FrameId|
                        f.0 < old(self).core_spec().frames_spec().metas_spec().len()
                            ==> dsts@[i].slot_ref().loc != (#[trigger] old(
                            self,
                        ).core_spec().frames_spec().frame(f)).slot.loc
                },
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() matches Some(c) && c.caller_stream_valid(
                *final(dst_stream),
                &final(self).core_spec(),
                dst_timeline,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            old(self).core_spec().spec_spec()@.versions_monotonic(
                final(self).core_spec().spec_spec()@,
            ),
            r matches Err((_, Some(ticket), completed_slices)) ==> completed_slices > 0
                && completed_slices <= dsts.len() && ticket != Ticket::NONE && ticket.timeline_id()
                == dst_timeline,
            r matches Err((_, None, completed_slices)) ==> completed_slices == 0,
    {
        hide(CacheCore::well_formed);
        let Some(device) = self.device.as_ref() else {
            return Err((CoreError::UnsupportedTier, None, 0));
        };
        let gpu_tier = device.device_tier_id();
        if dst_timeline == device.timeline_id() || !is_caller_device_timeline::<D>(
            &self.core,
            dst_timeline,
            dst_stream,
        ) || !super::boundary::device_slices_cover_frame(
            dsts,
            self.core.tier_frame_size(gpu_tier),
        ) {
            return Err((CoreError::InvalidArgument, None, 0));
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            dst_timeline,
            dst_stream,
        ) {
            return Err((CoreError::InvalidArgument, None, 0));
        }
        let has_pinned = device.has_pinned_host();
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
        #[cfg(feature = "nvme")]
        if has_pinned {
            if let Some(nvme_tier) = self.nvme_tier() {
                if self.core.lookup_tier(key, nvme_tier).is_some() {
                    let ghost before_warm = self.core;
                    let ghost nvme_tl = self.nvme.unwrap().timeline;
                    let ghost nvme_src = self.nvme.unwrap().ring.source().source();
                    let ghost backend = self.device.unwrap();
                    proof {
                        backend.lemma_device_layout(&before_warm);
                        backend.lemma_caller_stream_valid(*dst_stream, &before_warm, dst_timeline);
                        dst_stream.lemma_source_kind();
                    }
                    let warm_res = self.warm_host_from_nvme(
                        Tracked(instance),
                        Tracked(shares),
                        key,
                    );
                    proof {
                        backend.preserve_caller_stream(
                            &before_warm,
                            &self.core,
                            CoreFootprint::preserve_all(Some(nvme_tl), Some(nvme_src)),
                            *dst_stream,
                            dst_timeline,
                        );
                        assert forall|k: int| 0 <= k < dsts@.len() implies {
                            &&& write_admitted(slot_frontier(dst_toks[k]), self.core.citations())
                            &&& self.core.disjoint_from_published(
                                ISet::empty().insert((#[trigger] dsts@[k]).slot_ref().loc),
                            )
                            &&& forall|f: FrameId|
                                f.0 < self.core.frames_spec().metas_spec().len()
                                    ==> dsts@[k].slot_ref().loc != (
                                #[trigger] self.core.frames_spec().frame(f)).slot.loc
                        } by {
                            lemma_core_transition_preserves_slot_disjoint_and_admitted(
                                &before_warm,
                                &self.core,
                                CoreFootprint::preserve_all(Some(nvme_tl), Some(nvme_src)),
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
                    let ghost poll_fp = self.nvme.unwrap().poll_footprint();
                    let wait_res = self.wait_nvme_ticket(Tracked(instance), ticket);
                    proof {
                        backend.lemma_device_layout(&self.core);
                        assert(self.core.frames_spec().tier_frame_size(backend.device_tier_spec())
                            == backend.buffer_spec().reg_slot_len());
                        backend.preserve_caller_stream(
                            &before_wait,
                            &self.core,
                            poll_fp,
                            *dst_stream,
                            dst_timeline,
                        );
                        assert forall|k: int| 0 <= k < dsts@.len() implies {
                            &&& write_admitted(slot_frontier(dst_toks[k]), self.core.citations())
                            &&& self.core.disjoint_from_published(
                                ISet::empty().insert((#[trigger] dsts@[k]).slot_ref().loc),
                            )
                            &&& forall|f: FrameId|
                                f.0 < self.core.frames_spec().metas_spec().len()
                                    ==> dsts@[k].slot_ref().loc != (
                                #[trigger] self.core.frames_spec().frame(f)).slot.loc
                        } by {
                            lemma_core_transition_preserves_slot_disjoint_and_admitted(
                                &before_wait,
                                &self.core,
                                poll_fp,
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
        dst_stream: &mut D::Stream,
        dst_timeline: TimelineId,
        dsts: &[D::Slice<'_>],
        Tracked(dst_toks): Tracked<&mut Map<int, SlotToken>>,
        Ghost(offsets): Ghost<Seq<usize>>,
    ) -> (r: Result<(VersionId, Ticket), (CoreError, Option<Ticket>, usize)>)
        requires
            old(self).well_formed(),
            old(self).device matches Some(c) && c.caller_stream_valid(
                *old(dst_stream),
                &old(self).core,
                dst_timeline,
            ) && dst_timeline != c.timeline_id_spec(),
            dst_timeline != old(self).host.timeline,
            nvme_timeline_distinct(&old(self).nvme, dst_timeline),
            instance.id() == old(self).core.instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core.instance_id(),
            old(shares).node() == old(self).core.node(),
            old(self).device matches Some(c) ==> device_scatter_offsets_valid(
                dsts@,
                offsets,
                old(self).core.frames_spec().tier_frame_size(c.device_tier_spec()),
            ),
            forall|i: int|
                0 <= i < dsts@.len() ==> {
                    &&& (#[trigger] dsts@[i]).instance_id() == old(self).core.instance_id()
                    &&& dsts@[i].slot_ref().loc.node == old(self).core.node()
                    &&& (old(self).device matches Some(c) ==> (
                    dsts@[i].slot_ref().loc.space.is_gpu()
                        ==> dsts@[i].slot_ref().loc.space.gpu_device() == c.device_spec()))
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
                        f.0 < old(self).core.frames_spec().metas_spec().len()
                            ==> dsts@[i].slot_ref().loc != (#[trigger] old(
                            self,
                        ).core.frames_spec().frame(f)).slot.loc
                },
        ensures
            final(self).well_formed(),
            final(self).host == old(self).host,
            final(self).core.spec_spec() == old(self).core.spec_spec(),
            final(self).device matches Some(c) && c.caller_stream_valid(
                *final(dst_stream),
                &final(self).core,
                dst_timeline,
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
        let Some(device) = self.device.take() else {
            return Err((CoreError::UnsupportedTier, None, 0));
        };
        let ghost before = self.core;
        let ghost src_ev = dst_stream.source().source();
        proof {
            device.lemma_caller_stream_valid(*dst_stream, &before, dst_timeline);
            dst_stream.lemma_source_kind();
        }
        let res = device.get_into_device_scatter(
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
            let ghost no_device: Option<D> = None;
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(dst_timeline), Some(src_ev)),
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

} // verus!
