#[cfg(feature = "rdma")]
use super::DistributedEngine;
use super::{is_caller_device_timeline, LocalEngine};
use crate::backends::{BackendResult, DeviceBackend, DeviceStream};
use crate::spec::{CoreError, FrameId, SyncPoint, Ticket, TierId, TimelineId};
use crate::timeline::TimelineKind;
use transfer_core::context::{Completion, ShareSource, Submitted, TransferInstance};
#[cfg(feature = "rdma")]
use transfer_core::transport::ByteTransport;
#[cfg(verus_only)]
use vericache_kernel::compose::*;
use vstd::prelude::*;

verus! {

impl<'a, D: DeviceBackend<'a>> LocalEngine<'a, D> {
    pub fn copy_hop_cuda(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(FrameId, Ticket, SyncPoint)>)
        requires
            old(self).well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(self).core_spec().instance_id(),
            old(shares).node() == old(self).core_spec().node(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
    {
        if src_frame.0 as usize >= self.core.frame_count() || dst_tier.as_index()
            >= self.core.tier_count() {
            return Err(CoreError::InvalidArgument);
        }
        let Some(mut device) = self.device.take() else {
            return Err(CoreError::UnsupportedTier);
        };
        let ghost before = self.core;
        let ghost timeline = device.timeline_id_spec();
        let ghost source = device.event_source_spec();
        let ghost before_device = device;
        let result = device.copy_hop_device(
            &mut self.core,
            Tracked(instance),
            Tracked(shares),
            src_frame,
            dst_tier,
        );
        proof {
            before_device.lemma_device_layout(&before);
            device.lemma_same_config(&before_device);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(timeline), Some(source)),
                self.host,
                &self.host_ctx,
                &None,
                &self.nvme,
            );
        }
        self.device = Some(device);
        result
    }

    /// Register a timeline for a caller-owned stream or a custom adapter.
    pub fn register_timeline(&mut self, kind: TimelineKind) -> (r: BackendResult<TimelineId>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        let ghost before = self.core;
        let timeline = self.core.register_timeline(kind)?;
        proof {
            lemma_core_transition_after_timeline_append(&before, &self.core);
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
        Ok(timeline)
    }

    /// Record the observations from an operation on caller-owned Host memory.
    pub fn record_host_completion(&mut self, Tracked(completion): Tracked<Completion>)
        requires
            old(self).well_formed(),
            completion.submitted.instance_id() == old(self).core_spec().instance_id(),
            completion.knows.instance_id() == old(self).core_spec().instance_id(),
            completion.submitted.key().source.is_host(),
            completion.submitted.key().source != old(self).host_spec().ctx_spec().source@.source(),
            completion.submitted.key().source != old(self).host_ctx_spec().source@.source(),
            completion.submitted.value().preds.subset_of(
                old(self).core_spec().ledger_spec()@.submitted.dom(),
            ),
            old(self).core_spec().writes_disjoint_from_frames(
                completion.submitted.value().access.writes,
            ),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
    {
        let ghost before = self.core;
        let ghost source = completion.submitted.key().source;
        let ghost event = completion.submitted.key();
        self.core.record_disjoint_event(
            Tracked(completion.submitted),
            Tracked(Some(completion.knows)),
            Tracked(None),
        );
        proof {
            lemma_core_transition_external_event(&before, &self.core, None, event);
            if let Some(ref device) = self.device {
                device.lemma_device_layout(&before);
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, Some(source)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
        }
    }

    /// Assign a ticket to a submission on a caller-owned device timeline.
    pub fn record_cuda_submission(
        &mut self,
        stream: &D::Stream,
        timeline: TimelineId,
        Tracked(receipt): Tracked<Submitted>,
    ) -> (r: BackendResult<Ticket>)
        requires
            old(self).well_formed(),
            stream.well_formed(),
            stream.source_kind_valid(),
            receipt.instance_id() == old(self).core_spec().instance_id(),
            receipt.key().observer == old(self).core_spec().node(),
            receipt.key().source == stream.source().source(),
            old(self).device_spec() matches Some(c) ==> stream.source().source()
                != c.event_source_spec(),
            receipt.value().preds.subset_of(old(self).core_spec().ledger_spec()@.submitted.dom()),
            old(self).core_spec().writes_disjoint_from_frames(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            r matches Ok(ticket) ==> final(self).core_spec().timelines_spec().valid_ticket(ticket)
                && ticket != Ticket::NONE && ticket.timeline_id() == timeline,
    {
        if !is_caller_device_timeline::<D>(&self.core, timeline, stream) {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(device) = self.device() {
            if timeline == device.timeline_id() || stream.runtime_id()
                == device.stream().runtime_id() {
                return Err(CoreError::InvalidArgument);
            }
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            timeline,
            stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost event = receipt.key();
        let ticket = self.core.issue_ticket(timeline, Tracked(receipt))?;
        proof {
            lemma_core_transition_external_event(&before, &self.core, Some(timeline), event);
            stream.lemma_source_kind();
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(timeline), Some(event.source)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
        }
        Ok(ticket)
    }

    /// Synchronize a caller stream before retiring its ticket.
    pub fn synchronize_caller_stream(
        &mut self,
        stream: &D::Stream,
        timeline: TimelineId,
        ticket: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            stream.well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            stream.source().id() == instance.id(),
            stream.source().node() == old(self).core_spec().node(),
            ticket != Ticket::NONE ==> old(self).core_spec().timelines_spec().prefix_in(
                timeline,
                ticket.sequence(),
                foundation::tracked::closure_from(stream.infos().map(), stream.tail()),
            ),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            r.is_ok() ==> final(self).core_spec().timelines_spec().retired(ticket),
    {
        if !is_caller_device_timeline::<D>(&self.core, timeline, stream) {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(device) = self.device() {
            if timeline == device.timeline_id() || stream.runtime_id()
                == device.stream().runtime_id() {
                return Err(CoreError::InvalidArgument);
            }
        }
        if !ticket.is_none() && (ticket.timeline() != timeline
            || !self.core.timelines().contains_ticket(ticket)
            || !super::boundary::owns_caller_device_timeline(
            &self.caller_device_streams,
            timeline,
            stream,
        )) {
            return Err(CoreError::InvalidArgument);
        }
        let Tracked(knows) = D::synchronize_caller_stream(stream, Tracked(instance))?;
        if ticket.is_none() || self.core.timelines().is_retired(ticket) {
            return Ok(());
        }
        let ghost before = self.core;
        let seq = ticket.seq();
        self.core.mark_timeline_complete(timeline, seq, Tracked(knows));
        proof {
            lemma_core_transition_after_external_completion(&before, &self.core, timeline);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(timeline), None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
            );
        }
        Ok(())
    }
}

#[cfg(feature = "rdma")]
impl<'a, T: ByteTransport, D: DeviceBackend<'a>> DistributedEngine<'a, T, D> {
    /// Register a timeline for a caller-owned stream or a custom adapter.
    pub fn register_timeline(&mut self, kind: TimelineKind) -> (r: BackendResult<TimelineId>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
    {
        let ghost before = self.core;
        let timeline = self.core.register_timeline(kind)?;
        proof {
            lemma_core_transition_after_timeline_append(&before, &self.core);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
        }
        Ok(timeline)
    }

    /// Record the observations from an operation on caller-owned Host memory.
    pub fn record_host_completion(&mut self, Tracked(completion): Tracked<Completion>)
        requires
            old(self).well_formed(),
            completion.submitted.instance_id() == old(self).core_spec().instance_id(),
            completion.knows.instance_id() == old(self).core_spec().instance_id(),
            completion.submitted.key().source.is_host(),
            completion.submitted.key().source != old(self).host_spec().ctx_spec().source@.source(),
            completion.submitted.key().source != old(self).host_ctx_spec().source@.source(),
            completion.submitted.value().preds.subset_of(
                old(self).core_spec().ledger_spec()@.submitted.dom(),
            ),
            old(self).core_spec().writes_disjoint_from_frames(
                completion.submitted.value().access.writes,
            ),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
    {
        let ghost before = self.core;
        let ghost source = completion.submitted.key().source;
        let ghost event = completion.submitted.key();
        self.core.record_disjoint_event(
            Tracked(completion.submitted),
            Tracked(Some(completion.knows)),
            Tracked(None),
        );
        proof {
            lemma_core_transition_external_event(&before, &self.core, None, event);
            if let Some(ref device) = self.device.spec_ref() {
                device.lemma_device_layout(&before);
            }
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(None, Some(source)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
        }
    }

    /// Assign a ticket to a submission on a caller-owned device timeline.
    pub fn record_cuda_submission(
        &mut self,
        stream: &D::Stream,
        timeline: TimelineId,
        Tracked(receipt): Tracked<Submitted>,
    ) -> (r: BackendResult<Ticket>)
        requires
            old(self).well_formed(),
            stream.well_formed(),
            stream.source_kind_valid(),
            receipt.instance_id() == old(self).core_spec().instance_id(),
            receipt.key().observer == old(self).core_spec().node(),
            receipt.key().source == stream.source().source(),
            old(self).device_spec().spec_ref() matches Some(c) ==> stream.source().source()
                != c.event_source_spec(),
            receipt.value().preds.subset_of(old(self).core_spec().ledger_spec()@.submitted.dom()),
            old(self).core_spec().writes_disjoint_from_frames(receipt.value().access.writes),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            r matches Ok(ticket) ==> final(self).core_spec().timelines_spec().valid_ticket(ticket)
                && ticket != Ticket::NONE && ticket.timeline_id() == timeline,
    {
        if !is_caller_device_timeline::<D>(&self.core, timeline, stream) {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(device) = self.device() {
            if timeline == device.timeline_id() || stream.runtime_id()
                == device.stream().runtime_id() {
                return Err(CoreError::InvalidArgument);
            }
        }
        if !super::boundary::bind_caller_device_stream(
            &mut self.caller_device_streams,
            timeline,
            stream,
        ) {
            return Err(CoreError::InvalidArgument);
        }
        let ghost before = self.core;
        let ghost event = receipt.key();
        let ticket = self.core.issue_ticket(timeline, Tracked(receipt))?;
        proof {
            lemma_core_transition_external_event(&before, &self.core, Some(timeline), event);
            stream.lemma_source_kind();
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(timeline), Some(event.source)),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
        }
        Ok(ticket)
    }

    /// Synchronize a caller stream before retiring its ticket.
    pub fn synchronize_caller_stream(
        &mut self,
        stream: &D::Stream,
        timeline: TimelineId,
        ticket: Ticket,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(self).well_formed(),
            stream.well_formed(),
            instance.id() == old(self).core_spec().instance_id(),
            stream.source().id() == instance.id(),
            stream.source().node() == old(self).core_spec().node(),
            ticket != Ticket::NONE ==> old(self).core_spec().timelines_spec().prefix_in(
                timeline,
                ticket.sequence(),
                foundation::tracked::closure_from(stream.infos().map(), stream.tail()),
            ),
        ensures
            final(self).well_formed(),
            final(self).host_spec() == old(self).host_spec(),
            final(self).device_spec() == old(self).device_spec(),
            final(self).nvme_spec() == old(self).nvme_spec(),
            final(self).core_spec().spec_spec() == old(self).core_spec().spec_spec(),
            final(self).peers_spec() == old(self).peers_spec(),
            r.is_ok() ==> final(self).core_spec().timelines_spec().retired(ticket),
    {
        if !is_caller_device_timeline::<D>(&self.core, timeline, stream) {
            return Err(CoreError::InvalidArgument);
        }
        if let Some(device) = self.device() {
            if timeline == device.timeline_id() || stream.runtime_id()
                == device.stream().runtime_id() {
                return Err(CoreError::InvalidArgument);
            }
        }
        if !ticket.is_none() && (ticket.timeline() != timeline
            || !self.core.timelines().contains_ticket(ticket)
            || !super::boundary::owns_caller_device_timeline(
            &self.caller_device_streams,
            timeline,
            stream,
        )) {
            return Err(CoreError::InvalidArgument);
        }
        let Tracked(knows) = D::synchronize_caller_stream(stream, Tracked(instance))?;
        if ticket.is_none() || self.core.timelines().is_retired(ticket) {
            return Ok(());
        }
        let ghost before = self.core;
        let seq = ticket.seq();
        self.core.mark_timeline_complete(timeline, seq, Tracked(knows));
        proof {
            lemma_core_transition_after_external_completion(&before, &self.core, timeline);
            Self::lemma_preserve_siblings(
                &before,
                &self.core,
                CoreFootprint::preserve_all(Some(timeline), None),
                self.host,
                &self.host_ctx,
                &self.device,
                &self.nvme,
                self.peers@,
            );
        }
        Ok(())
    }
}

} // verus!
