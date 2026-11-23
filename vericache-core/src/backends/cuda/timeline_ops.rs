use super::*;
#[cfg(verus_only)]
use foundation::tracked::{closure_from, EventInfo, Frontier};
use transfer::cuda::{self, CudaSlice};

verus! {

impl CudaTimeline {
    /// Enqueue an async `cuda::memcpy_async` on this timeline's stream, issue
    /// its FIFO ticket on `core`, and preserve `CudaTimeline::well_formed`.
    pub fn enqueue_memcpy(
        &mut self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        dst: &CudaSlice,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
    ) -> (r: BackendResult<(Ghost<TransferEvent>, Ticket)>)
        requires
            old(core).well_formed(),
            old(self).well_formed(&old(core).timelines, old(core).node(), &old(core).ledger@),
            instance.id() == old(core).instance_id(),
            dst.well_formed(),
            src.well_formed(),
            dst.instance_id() == old(core).instance_id(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().sub_len == dst.slot_ref().sub_len,
            src.slot_ref().loc.node == old(core).node(),
            dst.slot_ref().loc.node == old(core).node(),
            src.slot_ref().loc != dst.slot_ref().loc,
            old(src_perm).valid_local(old(core).instance_id(), src.slot_ref().loc),
            slot_is_owned(*old(dst_tok), old(core).instance_id(), dst.slot_ref().loc),
            read_admitted(
                old(src_perm).frontier(),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).stream.tail().union(old(core).citations()),
                ),
            ),
            write_admitted(
                slot_frontier(*old(dst_tok)),
                closure_from(
                    old(core).ledger@.submitted.map(),
                    old(self).stream.tail().union(old(core).citations()),
                ),
            ),
            old(core).disjoint_from_published(ISet::empty().insert(dst.slot_ref().loc)),
            src.slot_ref().loc.space is GpuAlloc && dst.slot_ref().loc.space is GpuAlloc
                && src.slot_ref().loc.space->GpuAlloc_device
                != dst.slot_ref().loc.space->GpuAlloc_device ==> (old(
                src_perm,
            ).frontier().last_writer matches Some(w) ==> !(w.source is RdmaQueuePair)),
        ensures
            final(core).well_formed(),
            final(self).well_formed(
                &final(core).timelines,
                final(core).node(),
                &final(core).ledger@,
            ),
            final(self).timeline_id == old(self).timeline_id,
            final(self).device == old(self).device,
            final(self).stream.source().source() == old(self).stream.source().source(),
            final(self).event_ring == old(self).event_ring,
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).version_seq_start == old(core).version_seq_start,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).free_lease_slots == old(core).free_lease_slots,
            final(core).lease_generation == old(core).lease_generation,
            final(core).version_registry == old(core).version_registry,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.confirmed_seq == old(core).timelines.confirmed_seq,
            final(core).ledger@.knows == old(core).ledger@.knows,
            final(core).ledger@.succeeded == old(core).ledger@.succeeded,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            match r {
                Err(_) => *final(core) == *old(core) && *final(self) == *old(self)
                    && *final(src_perm) == *old(src_perm) && *final(dst_tok) == *old(dst_tok),
                Ok((ev, ticket)) => {
                    let s = old(core).timelines.next_seq[old(self).timeline_id as int];
                    &&& ev@ == old(self).stream.source().next_event()
                    &&& !old(core).ledger@.submitted.dom().contains(ev@)
                    &&& ticket != Ticket::NONE
                    &&& ticket.timeline_id() == old(self).timeline_id
                    &&& ticket.sequence() == s
                    &&& final(core).timelines.issued(old(self).timeline_id, s)
                    &&& final(core).timelines.events@ == old(core).timelines.events@.insert(
                        (old(self).timeline_id, s),
                        ev@,
                    )
                    &&& final(core).timelines.events@[(old(self).timeline_id, s)] == ev@
                    &&& final(core).timelines.next_seq@ == old(core).timelines.next_seq@.update(
                        old(self).timeline_id as int,
                        (s + 1) as u64,
                    )
                    &&& final(core).ledger@.submitted.instance_id() == old(
                        core,
                    ).ledger@.submitted.instance_id()
                    &&& final(core).ledger@.submitted.dom().contains(ev@)
                    &&& final(core).ledger@.submitted.map() == old(
                        core,
                    ).ledger@.submitted.map().insert(
                        ev@,
                        EventInfo {
                            access: copy_access(src.slot_ref(), dst.slot_ref()),
                            read_guards: old(src_perm).read_guards(),
                            effect: copy_effect(src.slot_ref(), dst.slot_ref()),
                            preds: closure_from(
                                old(core).ledger@.submitted.map(),
                                old(self).stream.tail().union(old(core).citations()),
                            ),
                        },
                    )
                    &&& final(core).ledger@.submitted.map()[ev@].preds.subset_of(
                        old(core).ledger@.submitted.dom(),
                    )
                    &&& old(core).ledger@.submitted.map().submap_of(
                        final(core).ledger@.submitted.map(),
                    )
                    &&& final(self).stream.tail() == ISet::empty().insert(ev@)
                    &&& final(src_perm).advanced_read(*old(src_perm), ev@)
                    &&& slot_owned(
                        *final(dst_tok),
                        old(core).instance_id(),
                        dst.slot_ref().loc,
                        Frontier { last_writer: Some(ev@), readers: ISet::empty() },
                    )
                    &&& final(core).kernel_view().extends(old(core).kernel_view())
                },
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let t = self.timeline_id;
        proof {
            lemma_core_wf_facts(core);
        }
        if core.timelines.next_seq[t as usize] > MAX_TICKET_SEQ {
            return Err(CoreError::SequenceExhausted);
        }
        let ghost before_core = *core;
        let ghost before_stream = self.stream;
        let ghost cited = core.citations();
        proof {
            lemma_cuda_memcpy_admitted(before_stream, *src_perm, *dst_tok, &before_core);
        }
        let copy_res = {
            let tracked ledger_ref = core.ledger.borrow_mut();
            cuda::memcpy_async(
                &mut self.stream,
                Tracked(instance),
                dst,
                src,
                Tracked(src_perm),
                Tracked(dst_tok),
                Tracked(&mut ledger_ref.submitted),
                Tracked(&ledger_ref.knows),
                Ghost(cited),
            )
        };
        let (ev, receipt) = match copy_res {
            Ok(pair) => pair,
            Err(e) => {
                proof {
                    assert(*core == before_core);
                }
                return Err(cuda_error(e));
            },
        };
        proof {
            lemma_cuda_memcpy_receipt_wf(
                before_stream,
                self.stream,
                &before_core,
                dst.slot_ref(),
                src.slot_ref(),
                ev@,
                receipt@,
            );
            lemma_core_wf_same_submitted_map(&before_core, core);
        }
        let ticket = core.issue_ticket(t, receipt).unwrap();
        proof {
            lemma_cuda_timeline_after_memcpy(
                before_stream,
                self.stream,
                self.event_ring@,
                &before_core,
                core,
                t,
                self.device,
                ev@,
                receipt@,
            );
        }
        Ok((ev, ticket))
    }

    /// Block until all work previously enqueued on this CUDA stream completes
    /// on the device, advancing `core.timelines.confirmed_seq[self.timeline_id]`
    /// to `next_seq - 1`.
    pub fn synchronize(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: BackendResult<()>)
        requires
            old(core).well_formed(),
            self.well_formed(&old(core).timelines, old(core).node(), &old(core).ledger@),
            instance.id() == old(core).instance_id(),
        ensures
            final(core).well_formed(),
            self.well_formed(&final(core).timelines, final(core).node(), &final(core).ledger@),
            final(core).node_id == old(core).node_id,
            final(core).tiers == old(core).tiers,
            final(core).frames == old(core).frames,
            final(core).catalog == old(core).catalog,
            final(core).spec == old(core).spec,
            final(core).version_seq_start == old(core).version_seq_start,
            final(core).next_version_seq == old(core).next_version_seq,
            final(core).version_seq_limit == old(core).version_seq_limit,
            final(core).active_leases == old(core).active_leases,
            final(core).free_lease_slots == old(core).free_lease_slots,
            final(core).lease_generation == old(core).lease_generation,
            final(core).version_registry == old(core).version_registry,
            final(core).timelines.count == old(core).timelines.count,
            final(core).timelines.kinds == old(core).timelines.kinds,
            final(core).timelines.events == old(core).timelines.events,
            final(core).timelines.next_seq == old(core).timelines.next_seq,
            forall|u: u8|
                u < old(core).timelines.count && u != self.timeline_id
                    ==> #[trigger] final(core).timelines.confirmed_seq[u as int] == old(
                    core,
                ).timelines.confirmed_seq[u as int],
            final(core).ledger@.submitted == old(core).ledger@.submitted,
            final(core).ledger@.succeeded == old(core).ledger@.succeeded,
            final(core).ledger@.revoked == old(core).ledger@.revoked,
            old(core).ledger@.knows.set().subset_of(final(core).ledger@.knows.set()),
            final(core).kernel_view().extends(old(core).kernel_view()),
            r.is_ok() ==> final(core).timelines.confirmed_seq[self.timeline_id as int] == old(
                core,
            ).timelines.next_seq[self.timeline_id as int] - 1,
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        let t = self.timeline_id;
        let ghost before_core = *core;
        let ghost node = core.node();
        proof {
            lemma_core_wf_facts(core);
        }
        let Tracked(k) = match cuda::stream_synchronize(&self.stream, Tracked(instance)) {
            Ok(w) => w,
            Err(e) => {
                proof {
                    lemma_cuda_extends_refl(core);
                }
                return Err(cuda_error(e));
            },
        };
        let target = core.timelines.next_seq[t as usize] - 1;
        proof {
            assert forall|s: u64| 1 <= s <= target implies core.timelines.observed(
                t,
                s,
                node,
                k,
            ) by {
                assert(self.closure().contains(core.timelines.events@[(t, s)]));
                assert(k.set().contains((node, core.timelines.events@[(t, s)])));
            }
        }
        core.mark_timeline_complete(t, target, Tracked(k));
        proof {
            lemma_cuda_timeline_preserved(self, &before_core, core);
        }
        Ok(())
    }
}

} // verus!
