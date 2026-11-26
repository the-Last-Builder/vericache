#[cfg(verus_only)]
use super::copy::caller_stream_valid;
use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::cuda::CudaStream;

verus! {

impl<'a> CudaBackend<'a> {
    proof fn lemma_copy_hop_sync_abort_transition(
        before: &CacheCore,
        mid: &CacheCore,
        after: &CacheCore,
        dst_frame: FrameId,
        lease_slot: u32,
        src_timeline: TimelineId,
        source: EventSource,
    )
        requires
            before.well_formed(),
            mid.well_formed(),
            after.well_formed(),
            dst_frame.0 < before.frames_spec().metas_spec().len(),
            before.frames_spec().meta(dst_frame).state != STATE_INITIALIZING,
            mid.frames_spec().same_directory(&before.frames_spec()),
            after.frames_spec().same_directory(&mid.frames_spec()),
            mid.frames_spec().preserves_initializing(&before.frames_spec(), None),
            after.frames_spec().preserves_initializing(&mid.frames_spec(), Some(dst_frame)),
            mid.frames_spec().preserves_fenced_leases(&before.frames_spec()),
            after.frames_spec().preserves_fenced_leases(&mid.frames_spec()),
            after.node_id_spec() == before.node_id_spec(),
            after.tiers_spec() == before.tiers_spec(),
            after.timelines_spec().only_watermarks(&before.timelines_spec()),
            after.ledger_spec() == before.ledger_spec(),
            after.next_version_seq_spec() >= before.next_version_seq_spec(),
            after.version_seq_limit_spec() == before.version_seq_limit_spec(),
            after.active_leases_spec().len() == before.active_leases_spec().len(),
            (lease_slot as usize) < before.active_leases_spec().len(),
            before.active_leases_spec()[lease_slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases_spec().len() && s != lease_slot as int
                    ==> #[trigger] mid.active_leases_spec()[s] == before.active_leases_spec()[s],
            forall|s: int|
                0 <= s < before.active_leases_spec().len() && s != lease_slot as int
                    ==> #[trigger] after.active_leases_spec()[s] == mid.active_leases_spec()[s],
        ensures
            core_transition(
                before,
                after,
                CoreFootprint::preserve_all(Some(src_timeline), Some(source)),
            ),
    {
        hide(CacheCore::well_formed);
        assert(after.frames_spec().same_directory(&before.frames_spec())) by {
            reveal(FrameTable::same_directory);
        }
        assert(after.frames_spec().preserves_initializing(&before.frames_spec(), None)) by {
            reveal(FrameTable::preserves_initializing);
            assert forall|f: FrameId|
                f.0 < before.frames_spec().metas_spec().len() && (
                #[trigger] before.frames_spec().meta(f)).state == STATE_INITIALIZING implies {
                &&& after.frames_spec().meta(f) == before.frames_spec().meta(f)
                &&& after.frames_spec().frame(f) == before.frames_spec().frame(f)
            } by {
                assert(f != dst_frame);
                assert(mid.frames_spec().meta(f) == before.frames_spec().meta(f));
                assert(mid.frames_spec().frame(f) == before.frames_spec().frame(f));
            }
        }
        assert(after.frames_spec().preserves_fenced_leases(&before.frames_spec())) by {
            assert forall|f: FrameId| f.0 < before.frames_spec().metas_spec().len() implies {
                &&& (#[trigger] after.frames_spec().frame(f)).fenced_shares
                    == before.frames_spec().frame(f).fenced_shares
                &&& after.frames_spec().meta(f).active_fenced_leases == before.frames_spec().meta(
                    f,
                ).active_fenced_leases
            } by {
                assert(mid.frames_spec().frame(f).fenced_shares == before.frames_spec().frame(
                    f,
                ).fenced_shares);
                assert(mid.frames_spec().meta(f).active_fenced_leases == before.frames_spec().meta(
                    f,
                ).active_fenced_leases);
            }
        }
        assert forall|s: int|
            0 <= s < before.active_leases_spec().len() && (
            #[trigger] before.active_leases_spec()[s] matches Some(l)
                && l.internal) implies after.active_leases_spec()[s]
            == before.active_leases_spec()[s] by {
            assert(s != lease_slot as int);
            assert(mid.active_leases_spec()[s] == before.active_leases_spec()[s]);
        }
        lemma_core_transition_no_event(before, after, Some(src_timeline), Some(source));
    }

    proof fn lemma_copy_hop_sync_step_transition(
        &self,
        before: &CacheCore,
        mid: &CacheCore,
        after: &CacheCore,
        before_stream: CudaStream,
        after_stream: CudaStream,
        src_timeline: TimelineId,
        ev: TransferEvent,
        dst_frame: FrameId,
        dst_loc: SlotLoc,
        lease_slot: u32,
    )
        requires
            before.well_formed(),
            mid.well_formed(),
            after.well_formed(),
            self.well_formed(before),
            before.timelines_spec().bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            dst_frame.0 < before.frames_spec().metas_spec().len(),
            before.frames_spec().meta(dst_frame).state != STATE_INITIALIZING,
            before.frames_spec().frame(dst_frame).slot.loc == dst_loc,
            after.node_id_spec() == before.node_id_spec(),
            after.instance_id() == before.instance_id(),
            after.tiers_spec() == before.tiers_spec(),
            after.next_version_seq_spec() >= before.next_version_seq_spec(),
            after.version_seq_limit_spec() == before.version_seq_limit_spec(),
            after.frames_spec().versions_spec()@ == before.frames_spec().versions_spec()@,
            after.active_leases_spec().len() == before.active_leases_spec().len(),
            (lease_slot as usize) < before.active_leases_spec().len(),
            before.active_leases_spec()[lease_slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases_spec().len() && s != lease_slot as int
                    ==> #[trigger] mid.active_leases_spec()[s] == before.active_leases_spec()[s],
            forall|s: int|
                0 <= s < before.active_leases_spec().len() && s != lease_slot as int
                    ==> #[trigger] after.active_leases_spec()[s] == mid.active_leases_spec()[s],
            after.timelines_spec().count_spec() == before.timelines_spec().count_spec(),
            after.timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
            forall|tl: TimelineId|
                tl < before.timelines_spec().count_spec() && tl != src_timeline ==> {
                    &&& (#[trigger] after.timelines_spec().next_seq_spec()[tl as int])
                        == before.timelines_spec().next_seq_spec()[tl as int]
                    &&& after.timelines_spec().confirmed_seq_spec()[tl as int]
                        == before.timelines_spec().confirmed_seq_spec()[tl as int]
                    &&& forall|s: u64|
                        1 <= s < before.timelines_spec().next_seq_spec()[tl as int]
                            ==> #[trigger] after.timelines_spec().events_spec()@[(tl, s)]
                            == before.timelines_spec().events_spec()@[(tl, s)]
                },
            mid.frames_spec().same_directory(&before.frames_spec()),
            after.frames_spec().same_layout(&mid.frames_spec()),
            mid.frames_spec().preserves_initializing(&before.frames_spec(), None),
            after.frames_spec().preserves_initializing(&mid.frames_spec(), Some(dst_frame)),
            mid.frames_spec().preserves_fenced_leases(&before.frames_spec()),
            after.frames_spec().preserves_fenced_leases(&mid.frames_spec()),
            before.ledger_spec()@.submitted.map().submap_of(after.ledger_spec()@.submitted.map()),
            after.ledger_spec()@.submitted.dom() == before.ledger_spec()@.submitted.dom().insert(
                ev,
            ),
            before.ledger_spec()@.knows.set().subset_of(after.ledger_spec()@.knows.set()),
            before.ledger_spec()@.succeeded.dom().subset_of(after.ledger_spec()@.succeeded.dom()),
            revoked_submap(before.ledger_spec()@.revoked, after.ledger_spec()@.revoked),
            ev == before_stream.source().next_event(),
            after.ledger_spec()@.submitted.map()[ev].access.writes == ISet::empty().insert(dst_loc),
            after_stream.source().source() == before_stream.source().source(),
            after_stream.source().node() == before_stream.source().node(),
            after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
            source_fresh_for_ledger(before_stream.source(), before.ledger_spec()@),
            before.spec_spec()@.versions_monotonic(after.spec_spec()@),
        ensures
            source_fresh_for_ledger(after_stream.source(), after.ledger_spec()@),
            self.well_formed(after),
            core_transition(
                before,
                after,
                CoreFootprint::preserve_all(
                    Some(src_timeline),
                    Some(before_stream.source().source()),
                ),
            ),
    {
        hide(CacheCore::well_formed);
        broadcast use {
            vstd::imap::group_imap_lemmas,
            vstd::iset::group_iset_lemmas,
            vstd::map::group_map_lemmas,
        };

        lemma_core_wf_facts(before);
        lemma_core_wf_facts(after);
        lemma_core_frames_separated(before);
        lemma_source_fresh_stepped(
            before_stream.source(),
            after_stream.source(),
            before.ledger_spec()@,
            after.ledger_spec()@,
            ev,
        );
        assert(after.frames_spec().same_layout(&before.frames_spec())) by {
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
        }
        assert(after.frames_spec().preserves_initializing(&before.frames_spec(), None)) by {
            reveal(FrameTable::preserves_initializing);
            assert forall|f: FrameId|
                f.0 < before.frames_spec().metas_spec().len() && (
                #[trigger] before.frames_spec().meta(f)).state == STATE_INITIALIZING implies {
                &&& after.frames_spec().meta(f) == before.frames_spec().meta(f)
                &&& after.frames_spec().frame(f) == before.frames_spec().frame(f)
            } by {
                assert(f != dst_frame);
                assert(mid.frames_spec().meta(f) == before.frames_spec().meta(f));
                assert(mid.frames_spec().frame(f) == before.frames_spec().frame(f));
            }
        }
        assert(after.frames_spec().preserves_fenced_leases(&before.frames_spec())) by {
            assert forall|f: FrameId| f.0 < before.frames_spec().metas_spec().len() implies {
                &&& (#[trigger] after.frames_spec().frame(f)).fenced_shares
                    == before.frames_spec().frame(f).fenced_shares
                &&& after.frames_spec().meta(f).active_fenced_leases == before.frames_spec().meta(
                    f,
                ).active_fenced_leases
            } by {
                assert(mid.frames_spec().frame(f).fenced_shares == before.frames_spec().frame(
                    f,
                ).fenced_shares);
                assert(mid.frames_spec().meta(f).active_fenced_leases == before.frames_spec().meta(
                    f,
                ).active_fenced_leases);
            }
        }
        assert forall|s: int|
            0 <= s < before.active_leases_spec().len() && (
            #[trigger] before.active_leases_spec()[s] matches Some(l)
                && l.internal) implies after.active_leases_spec()[s]
            == before.active_leases_spec()[s] by {
            assert(s != lease_slot as int);
            assert(mid.active_leases_spec()[s] == before.active_leases_spec()[s]);
        }
        lemma_non_init_frame_disjoint_from_init(before, dst_frame);
        lemma_core_transition_single_event(
            before,
            after,
            Some(src_timeline),
            before_stream.source().source(),
            ev,
            Some(dst_loc),
            None,
        );
        lemma_core_transition_preserves_cuda(
            before,
            after,
            CoreFootprint::preserve_all(Some(src_timeline), Some(before_stream.source().source())),
            self,
        );
    }

    fn park_failed_copy_hop_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
        ticket: Ticket,
        src_timeline: TimelineId,
        Ghost(before): Ghost<CacheCore>,
        Ghost(after_hop): Ghost<CacheCore>,
        Ghost(before_stream): Ghost<CudaStream>,
        Ghost(after_stream): Ghost<CudaStream>,
        Ghost(ev): Ghost<TransferEvent>,
    )
        requires
            before.well_formed(),
            after_hop.well_formed(),
            old(core).well_formed(),
            self.well_formed(&before),
            before.timelines_spec().bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            after_stream.well_formed(),
            after_stream.source().id() == old(core).instance_id(),
            after_stream.source().node() == old(core).node(),
            after_stream.source().source().is_cuda_stream(),
            after_stream.source().source() == before_stream.source().source(),
            after_stream.source().node() == before_stream.source().node(),
            after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
            source_fresh_for_ledger(before_stream.source(), before.ledger_spec()@),
            after_stream.infos().map().submap_of(old(core).ledger_spec()@.submitted.map()),
            after_stream.tail().subset_of(after_stream.infos().dom()),
            src_timeline < old(core).timelines_spec().count_spec() <= MAX_TIMELINES,
            old(core).timelines_spec().kinds_spec()[src_timeline as int] == (
            TimelineKind::CudaStream { device: after_stream.device() }),
            instance.id() == old(core).instance_id(),
            old(core).node_id_spec() == before.node_id_spec(),
            old(core).instance_id() == before.instance_id(),
            old(core).tiers_spec() == before.tiers_spec(),
            old(core).next_version_seq_spec() == before.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before.version_seq_limit_spec(),
            old(core).frames_spec() == after_hop.frames_spec(),
            old(core).spec_spec() == before.spec_spec(),
            old(core).version_registry_spec() == before.version_registry_spec(),
            old(core).active_leases_spec() == after_hop.active_leases_spec(),
            after_hop.frames_spec().same_directory(&before.frames_spec()),
            after_hop.frames_spec().preserves_initializing(&before.frames_spec(), None),
            after_hop.frames_spec().preserves_fenced_leases(&before.frames_spec()),
            after_hop.active_leases_spec().len() == before.active_leases_spec().len(),
            (hop.src_lease.slot_spec() as usize) < before.active_leases_spec().len(),
            before.active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases_spec().len() && s != hop.src_lease.slot_spec() as int
                    ==> #[trigger] after_hop.active_leases_spec()[s]
                    == before.active_leases_spec()[s],
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@
                == src_cap.share(),
            hop.src_lease.frame_spec() == hop.src_frame,
            hop.src_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < before.frames_spec().metas_spec().len(),
            hop.dst_frame.0 < old(core).frames_spec().metas_spec().len(),
            before.frames_spec().meta(hop.dst_frame).state != STATE_INITIALIZING,
            before.frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(hop.dst_frame).queued(),
            old(core).frames_spec().frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames_spec().frame(hop.dst_frame).token is None,
            src_cap.well_formed(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == old(core).frames_spec().frame(hop.src_frame).slot.loc,
            src_cap.frontier().readers == ISet::empty().insert(ev),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                hop.dst_slot@.loc,
                Frontier { last_writer: Some(ev), readers: ISet::empty() },
            ),
            old(core).timelines_spec().count_spec() == before.timelines_spec().count_spec(),
            old(core).timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
            forall|tl: TimelineId|
                tl < before.timelines_spec().count_spec() && tl != src_timeline ==> {
                    &&& (#[trigger] old(core).timelines_spec().next_seq_spec()[tl as int])
                        == before.timelines_spec().next_seq_spec()[tl as int]
                    &&& old(core).timelines_spec().confirmed_seq_spec()[tl as int]
                        == before.timelines_spec().confirmed_seq_spec()[tl as int]
                    &&& forall|s: u64|
                        1 <= s < before.timelines_spec().next_seq_spec()[tl as int]
                            ==> #[trigger] old(core).timelines_spec().events_spec()@[(tl, s)]
                            == before.timelines_spec().events_spec()@[(tl, s)]
                },
            ticket != Ticket::NONE,
            ticket.timeline_id() == src_timeline,
            1 <= ticket.sequence(),
            old(core).timelines_spec().issued(src_timeline, ticket.sequence()),
            old(core).timelines_spec().events_spec()@[(src_timeline, ticket.sequence())] == ev,
            ev == before_stream.source().next_event(),
            before.ledger_spec()@.submitted.map().submap_of(
                old(core).ledger_spec()@.submitted.map(),
            ),
            old(core).ledger_spec()@.submitted.dom()
                == before.ledger_spec()@.submitted.dom().insert(ev),
            before.ledger_spec()@.knows.set().subset_of(old(core).ledger_spec()@.knows.set()),
            before.ledger_spec()@.succeeded.dom().subset_of(
                old(core).ledger_spec()@.succeeded.dom(),
            ),
            revoked_submap(before.ledger_spec()@.revoked, old(core).ledger_spec()@.revoked),
            old(core).ledger_spec()@.submitted.map()[ev].access.writes == ISet::empty().insert(
                hop.dst_slot@.loc,
            ),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            caller_stream_valid(after_stream, final(core), src_timeline, self.timeline.stream),
            final(core).node_id_spec() == before.node_id_spec(),
            final(core).instance_id() == before.instance_id(),
            final(core).tiers_spec() == before.tiers_spec(),
            final(core).frames_spec().versions_spec()@ == before.frames_spec().versions_spec()@,
            final(core).version_registry_spec() == before.version_registry_spec(),
            before.spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(
                &before,
                final(core),
                CoreFootprint::preserve_all(
                    Some(src_timeline),
                    Some(before_stream.source().source()),
                ),
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CudaBackend::well_formed);
        hide(FrameTable::same_directory);
        hide(FrameTable::same_layout);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::preserves_fenced_leases);
        let ghost dst_frame = hop.dst_frame;
        let ghost dst_loc = hop.dst_slot@.loc;
        let ghost lease_slot = hop.src_lease.slot_spec();
        proof {
            assert forall|x: TransferEvent| #[trigger]
                src_cap.frontier().readers.contains(x) implies core.citations().contains(x)
                || exists|s: u64|
                1 <= s <= ticket.sequence() && #[trigger] core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == x by {
                assert(x == ev);
                let s = ticket.sequence();
                assert(1 <= s <= ticket.sequence() && core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == x);
            }
            lemma_returned_reads_by_ticket(core, hop.src_frame, src_cap, ticket);
        }
        core.release_local_lease(
            hop.src_lease,
            Tracked(src_cap),
            ticket,
            Tracked(instance),
        ).unwrap();
        let ghost after_rel = *core;
        proof {
            assert(core.frames_spec().metas_spec().len()
                == after_hop.frames_spec().metas_spec().len() && core.frames_spec().meta(
                dst_frame,
            ).state == STATE_INITIALIZING && !core.frames_spec().meta(dst_frame).queued()
                && core.frames_spec().frame(dst_frame) == after_hop.frames_spec().frame(dst_frame))
                by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            assert(slot_frontier(dst_tok).readers.subset_of(core.citations())) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            Self::lemma_ticket_covers_self(&core.timelines_spec(), ticket, ev);
        }
        let write_handle = WriteHandle {
            frame: hop.dst_frame,
            tier: hop.dst_tier,
            key: hop.key,
            version: hop.version,
            slot: hop.dst_slot,
        };
        core.park_write(write_handle, Tracked(dst_tok), ticket);
        proof {
            assert(core.frames_spec().same_layout(&after_hop.frames_spec())) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::same_layout);
            }
            assert(core.frames_spec().preserves_initializing(
                &after_hop.frames_spec(),
                Some(dst_frame),
            )) by {
                reveal(FrameTable::preserves_initializing);
                assert forall|f: FrameId|
                    f.0 < after_hop.frames_spec().metas_spec().len() && (
                    #[trigger] after_hop.frames_spec().meta(f)).state == STATE_INITIALIZING && f
                        != dst_frame implies {
                    &&& core.frames_spec().meta(f) == after_hop.frames_spec().meta(f)
                    &&& core.frames_spec().frame(f) == after_hop.frames_spec().frame(f)
                } by {
                    assert(after_rel.frames_spec().meta(f) == after_hop.frames_spec().meta(f));
                    assert(after_rel.frames_spec().frame(f) == after_hop.frames_spec().frame(f));
                }
            }
            assert(core.frames_spec().preserves_fenced_leases(&after_hop.frames_spec())) by {
                reveal(FrameTable::preserves_fenced_leases);
                assert forall|f: FrameId| f.0 < after_hop.frames_spec().metas_spec().len() implies {
                    &&& (#[trigger] core.frames_spec().frame(f)).fenced_shares
                        == after_hop.frames_spec().frame(f).fenced_shares
                    &&& core.frames_spec().meta(f).active_fenced_leases
                        == after_hop.frames_spec().meta(f).active_fenced_leases
                } by {
                    assert(after_rel.frames_spec().frame(f).fenced_shares
                        == after_hop.frames_spec().frame(f).fenced_shares);
                    assert(after_rel.frames_spec().meta(f).active_fenced_leases
                        == after_hop.frames_spec().meta(f).active_fenced_leases);
                }
            }
            assert(core.frames_spec().versions_spec()@ == before.frames_spec().versions_spec()@)
                by {
                reveal(FrameTable::same_directory);
            }
            self.lemma_copy_hop_sync_step_transition(
                &before,
                &after_hop,
                core,
                before_stream,
                after_stream,
                src_timeline,
                ev,
                dst_frame,
                dst_loc,
                lease_slot,
            );
        }
    }

    fn finish_copy_hop_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        hop: CopyHopHandle,
        Tracked(src_cap): Tracked<ShareCap>,
        Tracked(dst_tok): Tracked<SlotToken>,
        Tracked(knows): Tracked<Knows>,
        ticket: Ticket,
        src_timeline: TimelineId,
        dst_tier: TierId,
        dst_is_gpu: bool,
        Ghost(before): Ghost<CacheCore>,
        Ghost(after_hop): Ghost<CacheCore>,
        Ghost(before_stream): Ghost<CudaStream>,
        Ghost(after_stream): Ghost<CudaStream>,
        Ghost(ev): Ghost<TransferEvent>,
        Ghost(sub_val): Ghost<foundation::tracked::EventInfo<TransferDomain>>,
        Ghost(cited): Ghost<ISet<TransferEvent>>,
    ) -> (r: (VersionId, FrameId))
        requires
            before.well_formed(),
            after_hop.well_formed(),
            old(core).well_formed(),
            self.well_formed(&before),
            before.timelines_spec().bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            after_stream.well_formed(),
            after_stream.source().id() == old(core).instance_id(),
            after_stream.source().node() == old(core).node(),
            after_stream.source().source().is_cuda_stream(),
            after_stream.source().source() == before_stream.source().source(),
            after_stream.source().node() == before_stream.source().node(),
            after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
            source_fresh_for_ledger(before_stream.source(), before.ledger_spec()@),
            after_stream.infos().map().submap_of(old(core).ledger_spec()@.submitted.map()),
            after_stream.tail() == ISet::empty().insert(ev),
            after_stream.infos().map().dom().contains(ev),
            after_stream.infos().map()[ev] == sub_val,
            src_timeline < old(core).timelines_spec().count_spec() <= MAX_TIMELINES,
            old(core).timelines_spec().kinds_spec()[src_timeline as int] == (
            TimelineKind::CudaStream { device: after_stream.device() }),
            old(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 2 == old(
                core,
            ).timelines_spec().next_seq_spec()[src_timeline as int],
            ticket.sequence() + 1 == old(
                core,
            ).timelines_spec().next_seq_spec()[src_timeline as int],
            instance.id() == old(core).instance_id(),
            old(core).node_id_spec() == before.node_id_spec() && after_hop.node_id_spec()
                == before.node_id_spec(),
            old(core).instance_id() == before.instance_id() && after_hop.instance_id()
                == before.instance_id(),
            old(core).tiers_spec() == before.tiers_spec(),
            old(core).next_version_seq_spec() == before.next_version_seq_spec(),
            old(core).version_seq_limit_spec() == before.version_seq_limit_spec(),
            old(core).frames_spec() == after_hop.frames_spec(),
            old(core).spec_spec() == before.spec_spec(),
            old(core).version_registry_spec() == before.version_registry_spec(),
            old(core).active_leases_spec() == after_hop.active_leases_spec(),
            after_hop.frames_spec().same_directory(&before.frames_spec()),
            after_hop.frames_spec().preserves_initializing(&before.frames_spec(), None),
            after_hop.frames_spec().preserves_fenced_leases(&before.frames_spec()),
            after_hop.active_leases_spec().len() == before.active_leases_spec().len(),
            (hop.src_lease.slot_spec() as usize) < before.active_leases_spec().len(),
            before.active_leases_spec()[hop.src_lease.slot_spec() as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases_spec().len() && s != hop.src_lease.slot_spec() as int
                    ==> #[trigger] after_hop.active_leases_spec()[s]
                    == before.active_leases_spec()[s],
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases_spec()[hop.src_lease.slot_spec() as int]->Some_0.share@
                == src_cap.share(),
            hop.src_lease.frame_spec() == hop.src_frame,
            hop.src_frame.0 < old(core).frames_spec().metas_spec().len(),
            hop.dst_frame.0 < before.frames_spec().metas_spec().len(),
            hop.dst_frame.0 < old(core).frames_spec().metas_spec().len(),
            before.frames_spec().meta(hop.dst_frame).state != STATE_INITIALIZING,
            before.frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames_spec().meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames_spec().meta(hop.dst_frame).queued(),
            old(core).frames_spec().frame(hop.dst_frame).tier == dst_tier,
            hop.dst_tier == dst_tier,
            old(core).frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames_spec().frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames_spec().frame(hop.dst_frame).token is None,
            old(core).frames_spec().versions_spec()@.contains_key(hop.version),
            old(core).frames_spec().versions_spec()@[hop.version] == hop.src_cert@.bytes,
            hop.version.origin_node != old(core).node_id_spec() || hop.version.seq < old(
                core,
            ).next_version_seq_spec() || hop.version.seq >= old(core).version_seq_limit_spec(),
            dst_is_gpu ==> dst_tier == self.gpu_tier && self.owns_gpu(hop.dst_frame),
            !dst_is_gpu ==> self.owns_pinned_host(hop.dst_frame),
            src_cap.well_formed(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == old(core).frames_spec().frame(hop.src_frame).slot.loc,
            src_cap.frontier().readers == ISet::empty().insert(ev),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                hop.dst_slot@.loc,
                Frontier { last_writer: Some(ev), readers: ISet::empty() },
            ),
            old(core).timelines_spec().count_spec() == before.timelines_spec().count_spec(),
            old(core).timelines_spec().kinds_spec() == before.timelines_spec().kinds_spec(),
            forall|tl: TimelineId|
                tl < before.timelines_spec().count_spec() && tl != src_timeline ==> {
                    &&& (#[trigger] old(core).timelines_spec().next_seq_spec()[tl as int])
                        == before.timelines_spec().next_seq_spec()[tl as int]
                    &&& old(core).timelines_spec().confirmed_seq_spec()[tl as int]
                        == before.timelines_spec().confirmed_seq_spec()[tl as int]
                    &&& forall|s: u64|
                        1 <= s < before.timelines_spec().next_seq_spec()[tl as int]
                            ==> #[trigger] old(core).timelines_spec().events_spec()@[(tl, s)]
                            == before.timelines_spec().events_spec()@[(tl, s)]
                },
            ticket != Ticket::NONE,
            ticket.timeline_id() == src_timeline,
            1 <= ticket.sequence(),
            old(core).timelines_spec().issued(src_timeline, ticket.sequence()),
            old(core).timelines_spec().events_spec()@[(src_timeline, ticket.sequence())] == ev,
            ev == before_stream.source().next_event(),
            ev.source.succeeds_on_observe(),
            !after_hop.ledger_spec()@.submitted.dom().contains(ev),
            after_hop.ledger_spec()@ == before.ledger_spec()@,
            cited == after_hop.citations(),
            cited.subset_of(sub_val.preds),
            sub_val.preds.subset_of(after_hop.ledger_spec()@.submitted.dom()),
            sub_val.access.reads.contains(hop.src_slot@.loc),
            sub_val.access.writes == ISet::empty().insert(hop.dst_slot@.loc),
            sub_val.effect == copy_effect(hop.src_slot@, hop.dst_slot@),
            old(core).ledger_spec()@.submitted.map()
                == after_hop.ledger_spec()@.submitted.map().insert(ev, sub_val),
            old(core).ledger_spec()@.knows == after_hop.ledger_spec()@.knows,
            old(core).ledger_spec()@.succeeded == after_hop.ledger_spec()@.succeeded,
            old(core).ledger_spec()@.revoked == after_hop.ledger_spec()@.revoked,
            knows.instance_id() == old(core).instance_id(),
            knows.set() == foundation::tracked::knowledge_of(
                old(core).node(),
                closure_from(after_stream.infos().map(), after_stream.tail()),
            ),
            replica_valid(after_hop.ledger_spec()@, after_hop.node(), hop.src_cert@),
            source_slice_valid(hop.src_cert@, hop.src_slot@),
            hop.src_cert@.slot == hop.src_slot@,
            hop.src_slot@.well_formed(),
            hop.src_cert@.bytes.len() == hop.src_slot@.sub_len,
            hop.dst_slot@.well_formed(),
            hop.dst_slot@.is_full(),
            hop.src_slot@.sub_len == hop.dst_slot@.sub_len,
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            caller_stream_valid(after_stream, final(core), src_timeline, self.timeline.stream),
            final(core).node_id_spec() == before.node_id_spec(),
            final(core).instance_id() == before.instance_id(),
            final(core).tiers_spec() == before.tiers_spec(),
            final(core).frames_spec().versions_spec()@ == before.frames_spec().versions_spec()@,
            final(core).version_registry_spec() == before.version_registry_spec(),
            before.spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(
                &before,
                final(core),
                CoreFootprint::preserve_all(
                    Some(src_timeline),
                    Some(before_stream.source().source()),
                ),
            ),
            ((dst_tier == self.gpu_tier && self.owns_gpu(r.1)) || self.owns_pinned_host(r.1)),
            final(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1
                == final(core).timelines_spec().next_seq_spec()[src_timeline as int],
            final(core).frames_spec().meta(r.1).state == STATE_PUBLISHED,
            final(core).frames_spec().frame(r.1).tier == dst_tier,
            final(core).spec_spec()@.versions.contains_key(r.0),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CudaBackend::well_formed);
        hide(FrameTable::same_directory);
        hide(FrameTable::same_layout);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::preserves_fenced_leases);
        let ghost knows_val = knows;
        proof {
            assert(closure_from(after_stream.infos().map(), after_stream.tail()).contains(ev)) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            assert(core.timelines_spec().observed(
                src_timeline,
                ticket.sequence(),
                core.node(),
                knows,
            ));
        }
        core.mark_timeline_complete(src_timeline, ticket.seq(), Tracked(knows));
        let ver = hop.version;
        let dst_frame = hop.dst_frame;
        let ghost dst_slot = hop.dst_slot@;
        let ghost lease_slot = hop.src_lease.slot_spec();
        let ghost cert;
        proof {
            cert =
            Self::lemma_put_sync_copy_cert(
                &after_hop,
                core,
                after_stream,
                hop.src_cert@,
                hop.src_slot@,
                dst_slot,
                ev,
                sub_val,
                knows_val,
                cited,
            );
            assert forall|x: TransferEvent| #[trigger]
                src_cap.frontier().readers.contains(x) implies core.citations().contains(x)
                || exists|s: u64|
                1 <= s <= ticket.sequence() && #[trigger] core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == x by {
                assert(x == ev);
                let s = ticket.sequence();
                assert(1 <= s <= ticket.sequence() && core.timelines_spec().events_spec()@[(
                    ticket.timeline_id(),
                    s,
                )] == x);
            }
            lemma_returned_reads_by_ticket(core, hop.src_frame, src_cap, ticket);
        }
        let ghost before_finish = *core;
        core.finish_copy_hop(
            hop,
            Tracked(src_cap),
            ticket,
            Tracked(dst_tok),
            ticket,
            Ghost(cert),
            Tracked(instance),
        );
        proof {
            assert(core.ledger_spec()@.submitted.dom()
                =~= before.ledger_spec()@.submitted.dom().insert(ev)) by {
                broadcast use vstd::imap::group_imap_lemmas;

            }
            assert(core.frames_spec().versions_spec()@ =~= before.frames_spec().versions_spec()@)
                by {
                broadcast use vstd::map::group_map_lemmas;

                reveal(CacheCore::well_formed);
            }
            assert(core.spec_spec()@.versions.contains_key(ver)) by {
                broadcast use vstd::map::group_map_lemmas;

            }
            assert(core.frames_spec().frame(dst_frame).tier == dst_tier) by {
                reveal(FrameTable::same_layout);
            }
            self.lemma_copy_hop_sync_step_transition(
                &before,
                &before_finish,
                core,
                before_stream,
                after_stream,
                src_timeline,
                ev,
                dst_frame,
                dst_slot.loc,
                lease_slot,
            );
        }
        (ver, dst_frame)
    }

    /// Synchronously copy a published GPU-VRAM or Pinned-Host frame into the
    /// opposite tier (`GPU <-> Pinned Host`) using caller's idle `src_stream` /
    /// `src_timeline`, keeping `&self` shared so RDMA GPU/Host MRs remain valid.
    /// Preserves the existing `VersionId` and `VersionRegistry` without minting
    /// a new version.
    pub(crate) fn copy_hop_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_stream: &mut CudaStream,
        src_timeline: TimelineId,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            caller_stream_valid(*old(src_stream), old(core), src_timeline, self.timeline.stream),
            src_timeline != self.timeline.timeline_id,
            old(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1 == old(
                core,
            ).timelines_spec().next_seq_spec()[src_timeline as int],
            old(src_stream).device() == self.device,
            instance.id() == old(core).instance_id(),
            old(shares).well_formed(),
            old(shares).id() == old(core).instance_id(),
            old(shares).node() == old(core).node(),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            caller_stream_valid(
                *final(src_stream),
                final(core),
                src_timeline,
                self.timeline.stream,
            ),
            final(shares).well_formed(),
            final(shares).id() == old(shares).id(),
            final(shares).node() == old(shares).node(),
            final(core).node_id_spec() == old(core).node_id_spec(),
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers_spec() == old(core).tiers_spec(),
            final(core).frames_spec().versions_spec()@ == old(core).frames_spec().versions_spec()@,
            final(core).version_registry_spec() == old(core).version_registry_spec(),
            old(core).spec_spec()@.versions_monotonic(final(core).spec_spec()@),
            core_transition(
                old(core),
                final(core),
                CoreFootprint::preserve_all(
                    Some(src_timeline),
                    Some(old(src_stream).source().source()),
                ),
            ),
            r matches Ok((ver, dst_frame)) ==> {
                &&& ((dst_tier == self.gpu_tier && self.owns_gpu(dst_frame))
                    || self.owns_pinned_host(dst_frame))
                &&& final(core).timelines_spec().confirmed_seq_spec()[src_timeline as int] + 1
                    == final(core).timelines_spec().next_seq_spec()[src_timeline as int]
                &&& final(core).frames_spec().meta(dst_frame).state == STATE_PUBLISHED
                &&& final(core).frames_spec().frame(dst_frame).tier == dst_tier
                &&& final(core).spec_spec()@.versions.contains_key(ver)
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CudaBackend::well_formed);
        hide(FrameTable::same_directory);
        hide(FrameTable::same_layout);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::preserves_fenced_leases);
        let ghost before = *core;
        let ghost before_stream = *src_stream;
        let src_is_gpu = self.owns_gpu_frame(src_frame);
        let src_is_pinned = self.owns_pinned_host_frame(src_frame);
        let dst_is_gpu = dst_tier.0 == self.gpu_tier.0;
        let dst_is_pinned = match &self.pinned_host {
            Some(h) => dst_tier.0 == h.tier.0,
            None => false,
        };
        if !(src_is_gpu || src_is_pinned) || !(dst_is_gpu || dst_is_pinned) || (src_is_pinned
            && dst_is_pinned) {
            proof {
                lemma_core_transition_refl(
                    core,
                    CoreFootprint::preserve_all(
                        Some(src_timeline),
                        Some(before_stream.source().source()),
                    ),
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        proof {
            assert({
                &&& self.gpu_buffer.well_formed()
                &&& self.gpu_buffer.node() == before.node()
                &&& self.gpu_buffer.instance_id() == before.instance_id()
                &&& self.gpu_buffer.space().is_gpu()
                &&& self.gpu_buffer.space().gpu_device() == self.device
                &&& self.gpu_buffer.num_slots() == self.gpu_num_frames
                &&& self.gpu_buffer.slot_len() == before.frames_spec().tier_frame_size(
                    self.gpu_tier,
                )
                &&& slab_bound_to_core(
                    &before,
                    self.gpu_tier,
                    self.gpu_first_frame,
                    self.gpu_num_frames,
                    self.gpu_buffer.space(),
                )
                &&& (self.pinned_host matches Some(h) ==> {
                    &&& h.reg.well_formed()
                    &&& h.reg.node() == before.node()
                    &&& h.reg.instance_id() == before.instance_id()
                    &&& h.reg.num_slots() == h.num_frames
                    &&& h.reg.slot_len() == before.frames_spec().tier_frame_size(h.tier)
                    &&& before.frames_spec().tier_frame_size(h.tier)
                        == before.frames_spec().tier_frame_size(self.gpu_tier)
                    &&& slab_bound_to_core(
                        &before,
                        h.tier,
                        h.first_frame,
                        h.num_frames,
                        h.reg.space(),
                    )
                })
            }) by {
                reveal(CudaBackend::well_formed);
            }
        }
        let (hop, Tracked(src_cap), Tracked(mut dst_tok)) = match core.begin_copy_hop(
            src_frame,
            dst_tier,
            Tracked(shares),
            Tracked(instance),
        ) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    assert(core.frames_spec().versions_spec()@
                        == before.frames_spec().versions_spec()@) by {
                        reveal(FrameTable::same_directory);
                    }
                    lemma_core_transition_no_event(
                        &before,
                        core,
                        Some(src_timeline),
                        Some(before_stream.source().source()),
                    );
                    lemma_core_transition_preserves_cuda(
                        &before,
                        core,
                        CoreFootprint::preserve_all(
                            Some(src_timeline),
                            Some(before_stream.source().source()),
                        ),
                        self,
                    );
                }
                return Err(e);
            },
        };
        let ghost after_hop = *core;
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(core);
            lemma_copy_hop_slots_ready(
                self,
                &before,
                core,
                hop.src_frame,
                hop.dst_frame,
                src_is_gpu,
                dst_is_gpu,
                src_cap.frontier(),
                slot_frontier(dst_tok),
            );
            assert(before.frames_spec().frame(hop.dst_frame).slot == hop.dst_slot@) by {
                reveal(FrameTable::same_directory);
            }
        }
        if core.timelines().next_sequences()[src_timeline as usize] > MAX_TICKET_SEQ {
            let ghost dst_frame = hop.dst_frame;
            let ghost lease_slot = hop.src_lease.slot_spec();
            core.abort_copy_hop(
                hop,
                Tracked(src_cap),
                Ticket::NONE,
                Tracked(dst_tok),
                Tracked(instance),
            );
            proof {
                assert(core.frames_spec().versions_spec()@ == before.frames_spec().versions_spec()@)
                    by {
                    reveal(FrameTable::same_directory);
                }
                Self::lemma_copy_hop_sync_abort_transition(
                    &before,
                    &after_hop,
                    core,
                    dst_frame,
                    lease_slot,
                    src_timeline,
                    before_stream.source().source(),
                );
                lemma_core_transition_preserves_cuda(
                    &before,
                    core,
                    CoreFootprint::preserve_all(
                        Some(src_timeline),
                        Some(before_stream.source().source()),
                    ),
                    self,
                );
            }
            return Err(CoreError::SequenceExhausted);
        }
        let src_slice = if src_is_gpu {
            self.gpu_buffer.slot_slice((hop.src_frame.0 - self.gpu_first_frame.0) as usize)
        } else {
            match &self.pinned_host {
                Some(h) => h.reg.slot_slice((hop.src_frame.0 - h.first_frame.0) as usize),
                None => {
                    assert(false);
                    self.gpu_buffer.slot_slice(0)
                },
            }
        };
        let dst_slice = if dst_is_gpu {
            self.gpu_buffer.slot_slice((hop.dst_frame.0 - self.gpu_first_frame.0) as usize)
        } else {
            match &self.pinned_host {
                Some(h) => h.reg.slot_slice((hop.dst_frame.0 - h.first_frame.0) as usize),
                None => {
                    assert(false);
                    self.gpu_buffer.slot_slice(0)
                },
            }
        };
        let tracked mut src_perm = ReadPerm::Shared(src_cap);
        let ghost cited = core.citations();
        let ghost ev = src_stream.source().next_event();
        proof {
            if src_is_gpu && dst_is_gpu {
                assert(src_slice.slot_ref().loc.space == self.gpu_buffer.space());
                assert(dst_slice.slot_ref().loc.space == self.gpu_buffer.space());
            }
            Self::lemma_put_sync_copy_admitted(core, *src_stream, src_perm, dst_tok, cited);
        }
        let copy_res = {
            let tracked ledger_ref = core.borrow_ledger_mut();
            cuda::memcpy_async(
                src_stream,
                Tracked(instance),
                &dst_slice,
                &src_slice,
                Tracked(&mut src_perm),
                Tracked(&mut dst_tok),
                Tracked(&mut ledger_ref.submitted),
                Tracked(&ledger_ref.knows),
                Ghost(cited),
            )
        };
        let tracked src_cap = src_perm.into_shared();
        let (Ghost(_ev), Tracked(receipt)) = match copy_res {
            Ok(v) => v,
            Err(e) => {
                let ghost dst_frame = hop.dst_frame;
                let ghost lease_slot = hop.src_lease.slot_spec();
                core.abort_copy_hop(
                    hop,
                    Tracked(src_cap),
                    Ticket::NONE,
                    Tracked(dst_tok),
                    Tracked(instance),
                );
                proof {
                    assert(core.frames_spec().versions_spec()@
                        == before.frames_spec().versions_spec()@) by {
                        reveal(FrameTable::same_directory);
                    }
                    Self::lemma_copy_hop_sync_abort_transition(
                        &before,
                        &after_hop,
                        core,
                        dst_frame,
                        lease_slot,
                        src_timeline,
                        before_stream.source().source(),
                    );
                    lemma_core_transition_preserves_cuda(
                        &before,
                        core,
                        CoreFootprint::preserve_all(
                            Some(src_timeline),
                            Some(before_stream.source().source()),
                        ),
                        self,
                    );
                }
                return Err(cuda_error(e));
            },
        };
        let ghost sub_val = receipt.value();
        proof {
            assert(sub_val.access.writes =~= ISet::empty().insert(hop.dst_slot@.loc)) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            Self::lemma_put_sync_preds_closed(
                &after_hop,
                core,
                before_stream,
                cited,
                sub_val,
                hop.dst_frame,
            );
        }
        let ticket = core.issue_ticket(src_timeline, Tracked(receipt)).unwrap();
        let sync_res = cuda::stream_synchronize(src_stream, Tracked(instance));
        match sync_res {
            Err(e) => {
                proof {
                    assert(core.ledger_spec()@.submitted.dom()
                        =~= before.ledger_spec()@.submitted.dom().insert(ev)) by {
                        broadcast use vstd::imap::group_imap_lemmas;

                    }
                }
                self.park_failed_copy_hop_sync(
                    core,
                    Tracked(instance),
                    hop,
                    Tracked(src_cap),
                    Tracked(dst_tok),
                    ticket,
                    src_timeline,
                    Ghost(before),
                    Ghost(after_hop),
                    Ghost(before_stream),
                    Ghost(*src_stream),
                    Ghost(ev),
                );
                Err(cuda_error(e))
            },
            Ok(knows_tracked) => {
                let Tracked(knows) = knows_tracked;
                proof {
                    assert(cited.subset_of(sub_val.preds)) by {
                        broadcast use vstd::iset::group_iset_lemmas;

                    }
                    assert(sub_val.preds.subset_of(after_hop.ledger_spec()@.submitted.dom())) by {
                        broadcast use vstd::iset::group_iset_lemmas;

                        assert forall|x: TransferEvent| #[trigger]
                            sub_val.preds.contains(
                                x,
                            ) implies after_hop.ledger_spec()@.submitted.dom().contains(x) by {
                            assert(x != ev);
                        }
                    }
                }
                let res = self.finish_copy_hop_sync(
                    core,
                    Tracked(instance),
                    hop,
                    Tracked(src_cap),
                    Tracked(dst_tok),
                    Tracked(knows),
                    ticket,
                    src_timeline,
                    dst_tier,
                    dst_is_gpu,
                    Ghost(before),
                    Ghost(after_hop),
                    Ghost(before_stream),
                    Ghost(*src_stream),
                    Ghost(ev),
                    Ghost(sub_val),
                    Ghost(cited),
                );
                Ok(res)
            },
        }
    }
}

} // verus!
