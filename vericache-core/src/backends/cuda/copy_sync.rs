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
        src_timeline: u8,
        source: EventSource,
    )
        requires
            before.well_formed(),
            mid.well_formed(),
            after.well_formed(),
            dst_frame.0 < before.frames.metas.len(),
            before.frames.meta(dst_frame).state != STATE_INITIALIZING,
            mid.frames.same_directory(&before.frames),
            after.frames.same_directory(&mid.frames),
            mid.frames.preserves_initializing(&before.frames, None),
            after.frames.preserves_initializing(&mid.frames, Some(dst_frame)),
            mid.frames.preserves_remote_leases(&before.frames),
            after.frames.preserves_remote_leases(&mid.frames),
            after.node_id == before.node_id,
            after.tiers == before.tiers,
            after.timelines.only_watermarks(&before.timelines),
            after.ledger == before.ledger,
            after.next_version_seq >= before.next_version_seq,
            after.version_seq_limit == before.version_seq_limit,
            after.active_leases.len() == before.active_leases.len(),
            (lease_slot as usize) < before.active_leases.len(),
            before.active_leases[lease_slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases.len() && s != lease_slot as int
                    ==> #[trigger] mid.active_leases[s] == before.active_leases[s],
            forall|s: int|
                0 <= s < before.active_leases.len() && s != lease_slot as int
                    ==> #[trigger] after.active_leases[s] == mid.active_leases[s],
        ensures
            core_transition(
                before,
                after,
                Some(src_timeline),
                Some(source),
                FrameTouchMode::PreserveAllInit,
            ),
    {
        hide(CacheCore::well_formed);
        assert(after.frames.same_directory(&before.frames)) by {
            reveal(FrameTable::same_directory);
        }
        assert(after.frames.preserves_initializing(&before.frames, None)) by {
            reveal(FrameTable::preserves_initializing);
            assert forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING implies {
                &&& after.frames.meta(f) == before.frames.meta(f)
                &&& after.frames.frame(f) == before.frames.frame(f)
            } by {
                assert(f != dst_frame);
                assert(mid.frames.meta(f) == before.frames.meta(f));
                assert(mid.frames.frame(f) == before.frames.frame(f));
            }
        }
        assert(after.frames.preserves_remote_leases(&before.frames)) by {
            assert forall|f: FrameId| f.0 < before.frames.metas.len() implies {
                &&& (#[trigger] after.frames.frame(f)).remote_shares == before.frames.frame(
                    f,
                ).remote_shares
                &&& after.frames.meta(f).active_remote_leases == before.frames.meta(
                    f,
                ).active_remote_leases
            } by {
                assert(mid.frames.frame(f).remote_shares == before.frames.frame(f).remote_shares);
                assert(mid.frames.meta(f).active_remote_leases == before.frames.meta(
                    f,
                ).active_remote_leases);
            }
        }
        assert forall|s: int|
            0 <= s < before.active_leases.len() && (#[trigger] before.active_leases[s] matches Some(
                l,
            ) && l.internal) implies after.active_leases[s] == before.active_leases[s] by {
            assert(s != lease_slot as int);
            assert(mid.active_leases[s] == before.active_leases[s]);
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
        src_timeline: u8,
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
            before.timelines.bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            dst_frame.0 < before.frames.metas.len(),
            before.frames.meta(dst_frame).state != STATE_INITIALIZING,
            before.frames.frame(dst_frame).slot.loc == dst_loc,
            after.node_id == before.node_id,
            after.instance_id() == before.instance_id(),
            after.tiers == before.tiers,
            after.next_version_seq >= before.next_version_seq,
            after.version_seq_limit == before.version_seq_limit,
            after.frames.versions@ == before.frames.versions@,
            after.active_leases.len() == before.active_leases.len(),
            (lease_slot as usize) < before.active_leases.len(),
            before.active_leases[lease_slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases.len() && s != lease_slot as int
                    ==> #[trigger] mid.active_leases[s] == before.active_leases[s],
            forall|s: int|
                0 <= s < before.active_leases.len() && s != lease_slot as int
                    ==> #[trigger] after.active_leases[s] == mid.active_leases[s],
            after.timelines.count == before.timelines.count,
            after.timelines.kinds == before.timelines.kinds,
            forall|tl: u8|
                tl < before.timelines.count && tl != src_timeline ==> {
                    &&& (#[trigger] after.timelines.next_seq[tl as int])
                        == before.timelines.next_seq[tl as int]
                    &&& after.timelines.confirmed_seq[tl as int]
                        == before.timelines.confirmed_seq[tl as int]
                    &&& forall|s: u64|
                        1 <= s < before.timelines.next_seq[tl as int]
                            ==> #[trigger] after.timelines.events@[(tl, s)]
                            == before.timelines.events@[(tl, s)]
                },
            mid.frames.same_directory(&before.frames),
            after.frames.same_layout(&mid.frames),
            mid.frames.preserves_initializing(&before.frames, None),
            after.frames.preserves_initializing(&mid.frames, Some(dst_frame)),
            mid.frames.preserves_remote_leases(&before.frames),
            after.frames.preserves_remote_leases(&mid.frames),
            before.ledger@.submitted.map().submap_of(after.ledger@.submitted.map()),
            after.ledger@.submitted.dom() == before.ledger@.submitted.dom().insert(ev),
            before.ledger@.knows.set().subset_of(after.ledger@.knows.set()),
            before.ledger@.succeeded.dom().subset_of(after.ledger@.succeeded.dom()),
            revoked_submap(before.ledger@.revoked, after.ledger@.revoked),
            ev == before_stream.source().next_event(),
            after.ledger@.submitted.map()[ev].access.writes == ISet::empty().insert(dst_loc),
            after_stream.source().source() == before_stream.source().source(),
            after_stream.source().node() == before_stream.source().node(),
            after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
            source_fresh_for_ledger(before_stream.source(), before.ledger@),
            before.spec@.versions_monotonic(after.spec@),
        ensures
            source_fresh_for_ledger(after_stream.source(), after.ledger@),
            self.well_formed(after),
            core_transition(
                before,
                after,
                Some(src_timeline),
                Some(before_stream.source().source()),
                FrameTouchMode::PreserveAllInit,
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
            before.ledger@,
            after.ledger@,
            ev,
        );
        assert(after.frames.same_layout(&before.frames)) by {
            reveal(FrameTable::same_directory);
            reveal(FrameTable::same_layout);
        }
        assert(after.frames.preserves_initializing(&before.frames, None)) by {
            reveal(FrameTable::preserves_initializing);
            assert forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING implies {
                &&& after.frames.meta(f) == before.frames.meta(f)
                &&& after.frames.frame(f) == before.frames.frame(f)
            } by {
                assert(f != dst_frame);
                assert(mid.frames.meta(f) == before.frames.meta(f));
                assert(mid.frames.frame(f) == before.frames.frame(f));
            }
        }
        assert(after.frames.preserves_remote_leases(&before.frames)) by {
            assert forall|f: FrameId| f.0 < before.frames.metas.len() implies {
                &&& (#[trigger] after.frames.frame(f)).remote_shares == before.frames.frame(
                    f,
                ).remote_shares
                &&& after.frames.meta(f).active_remote_leases == before.frames.meta(
                    f,
                ).active_remote_leases
            } by {
                assert(mid.frames.frame(f).remote_shares == before.frames.frame(f).remote_shares);
                assert(mid.frames.meta(f).active_remote_leases == before.frames.meta(
                    f,
                ).active_remote_leases);
            }
        }
        assert forall|s: int|
            0 <= s < before.active_leases.len() && (#[trigger] before.active_leases[s] matches Some(
                l,
            ) && l.internal) implies after.active_leases[s] == before.active_leases[s] by {
            assert(s != lease_slot as int);
            assert(mid.active_leases[s] == before.active_leases[s]);
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
            Some(src_timeline),
            Some(before_stream.source().source()),
            FrameTouchMode::PreserveAllInit,
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
        src_timeline: u8,
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
            before.timelines.bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            after_stream.well_formed(),
            after_stream.source().id() == old(core).instance_id(),
            after_stream.source().node() == old(core).node(),
            after_stream.source().source() is CudaStream,
            after_stream.source().source() == before_stream.source().source(),
            after_stream.source().node() == before_stream.source().node(),
            after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
            source_fresh_for_ledger(before_stream.source(), before.ledger@),
            after_stream.infos().map().submap_of(old(core).ledger@.submitted.map()),
            after_stream.tail().subset_of(after_stream.infos().dom()),
            src_timeline < old(core).timelines.count <= MAX_TIMELINES,
            old(core).timelines.kinds[src_timeline as int] == (TimelineKind::CudaStream {
                device: after_stream.device(),
            }),
            instance.id() == old(core).instance_id(),
            old(core).node_id == before.node_id,
            old(core).instance_id() == before.instance_id(),
            old(core).tiers == before.tiers,
            old(core).next_version_seq == before.next_version_seq,
            old(core).version_seq_limit == before.version_seq_limit,
            old(core).frames == after_hop.frames,
            old(core).spec == before.spec,
            old(core).version_registry == before.version_registry,
            old(core).active_leases == after_hop.active_leases,
            after_hop.frames.same_directory(&before.frames),
            after_hop.frames.preserves_initializing(&before.frames, None),
            after_hop.frames.preserves_remote_leases(&before.frames),
            after_hop.active_leases.len() == before.active_leases.len(),
            (hop.src_lease.slot as usize) < before.active_leases.len(),
            before.active_leases[hop.src_lease.slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases.len() && s != hop.src_lease.slot as int
                    ==> #[trigger] after_hop.active_leases[s] == before.active_leases[s],
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases[hop.src_lease.slot as int]->Some_0.share@ == src_cap.share(),
            hop.src_lease.frame == hop.src_frame,
            hop.src_frame.0 < old(core).frames.metas.len(),
            hop.dst_frame.0 < before.frames.metas.len(),
            hop.dst_frame.0 < old(core).frames.metas.len(),
            before.frames.meta(hop.dst_frame).state != STATE_INITIALIZING,
            before.frames.frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames.meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(hop.dst_frame).queued(),
            old(core).frames.frame(hop.dst_frame).tier == hop.dst_tier,
            old(core).frames.frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames.frame(hop.dst_frame).token is None,
            src_cap.well_formed(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == old(core).frames.frame(hop.src_frame).slot.loc,
            src_cap.frontier().readers == ISet::empty().insert(ev),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                hop.dst_slot@.loc,
                Frontier { last_writer: Some(ev), readers: ISet::empty() },
            ),
            old(core).timelines.count == before.timelines.count,
            old(core).timelines.kinds == before.timelines.kinds,
            forall|tl: u8|
                tl < before.timelines.count && tl != src_timeline ==> {
                    &&& (#[trigger] old(core).timelines.next_seq[tl as int])
                        == before.timelines.next_seq[tl as int]
                    &&& old(core).timelines.confirmed_seq[tl as int]
                        == before.timelines.confirmed_seq[tl as int]
                    &&& forall|s: u64|
                        1 <= s < before.timelines.next_seq[tl as int] ==> #[trigger] old(
                            core,
                        ).timelines.events@[(tl, s)] == before.timelines.events@[(tl, s)]
                },
            ticket != Ticket::NONE,
            ticket.timeline_id() == src_timeline,
            1 <= ticket.sequence(),
            old(core).timelines.issued(src_timeline, ticket.sequence()),
            old(core).timelines.events@[(src_timeline, ticket.sequence())] == ev,
            ev == before_stream.source().next_event(),
            before.ledger@.submitted.map().submap_of(old(core).ledger@.submitted.map()),
            old(core).ledger@.submitted.dom() == before.ledger@.submitted.dom().insert(ev),
            before.ledger@.knows.set().subset_of(old(core).ledger@.knows.set()),
            before.ledger@.succeeded.dom().subset_of(old(core).ledger@.succeeded.dom()),
            revoked_submap(before.ledger@.revoked, old(core).ledger@.revoked),
            old(core).ledger@.submitted.map()[ev].access.writes == ISet::empty().insert(
                hop.dst_slot@.loc,
            ),
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            caller_stream_valid(after_stream, final(core), src_timeline, self.timeline.stream),
            final(core).node_id == before.node_id,
            final(core).instance_id() == before.instance_id(),
            final(core).tiers == before.tiers,
            final(core).frames.versions@ == before.frames.versions@,
            final(core).version_registry == before.version_registry,
            before.spec@.versions_monotonic(final(core).spec@),
            core_transition(
                &before,
                final(core),
                Some(src_timeline),
                Some(before_stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CudaBackend::well_formed);
        hide(FrameTable::same_directory);
        hide(FrameTable::same_layout);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::preserves_remote_leases);
        let ghost dst_frame = hop.dst_frame;
        let ghost dst_loc = hop.dst_slot@.loc;
        let ghost lease_slot = hop.src_lease.slot;
        proof {
            assert forall|x: TransferEvent| #[trigger]
                src_cap.frontier().readers.contains(x) implies core.citations().contains(x)
                || exists|s: u64|
                1 <= s <= ticket.sequence() && #[trigger] core.timelines.events@[(
                    ticket.timeline_id(),
                    s,
                )] == x by {
                assert(x == ev);
                let s = ticket.sequence();
                assert(1 <= s <= ticket.sequence() && core.timelines.events@[(
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
            assert(core.frames.metas.len() == after_hop.frames.metas.len() && core.frames.meta(
                dst_frame,
            ).state == STATE_INITIALIZING && !core.frames.meta(dst_frame).queued()
                && core.frames.frame(dst_frame) == after_hop.frames.frame(dst_frame)) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::preserves_initializing);
            }
            assert(slot_frontier(dst_tok).readers.subset_of(core.citations())) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            Self::lemma_ticket_covers_self(&core.timelines, ticket, ev);
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
            assert(core.frames.same_layout(&after_hop.frames)) by {
                reveal(FrameTable::same_directory);
                reveal(FrameTable::same_layout);
            }
            assert(core.frames.preserves_initializing(&after_hop.frames, Some(dst_frame))) by {
                reveal(FrameTable::preserves_initializing);
                assert forall|f: FrameId|
                    f.0 < after_hop.frames.metas.len() && (#[trigger] after_hop.frames.meta(
                        f,
                    )).state == STATE_INITIALIZING && f != dst_frame implies {
                    &&& core.frames.meta(f) == after_hop.frames.meta(f)
                    &&& core.frames.frame(f) == after_hop.frames.frame(f)
                } by {
                    assert(after_rel.frames.meta(f) == after_hop.frames.meta(f));
                    assert(after_rel.frames.frame(f) == after_hop.frames.frame(f));
                }
            }
            assert(core.frames.preserves_remote_leases(&after_hop.frames)) by {
                reveal(FrameTable::preserves_remote_leases);
                assert forall|f: FrameId| f.0 < after_hop.frames.metas.len() implies {
                    &&& (#[trigger] core.frames.frame(f)).remote_shares == after_hop.frames.frame(
                        f,
                    ).remote_shares
                    &&& core.frames.meta(f).active_remote_leases == after_hop.frames.meta(
                        f,
                    ).active_remote_leases
                } by {
                    assert(after_rel.frames.frame(f).remote_shares == after_hop.frames.frame(
                        f,
                    ).remote_shares);
                    assert(after_rel.frames.meta(f).active_remote_leases == after_hop.frames.meta(
                        f,
                    ).active_remote_leases);
                }
            }
            assert(core.frames.versions@ == before.frames.versions@) by {
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
        src_timeline: u8,
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
            before.timelines.bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            after_stream.well_formed(),
            after_stream.source().id() == old(core).instance_id(),
            after_stream.source().node() == old(core).node(),
            after_stream.source().source() is CudaStream,
            after_stream.source().source() == before_stream.source().source(),
            after_stream.source().node() == before_stream.source().node(),
            after_stream.source().next_seq() == before_stream.source().next_seq() + 1,
            source_fresh_for_ledger(before_stream.source(), before.ledger@),
            after_stream.infos().map().submap_of(old(core).ledger@.submitted.map()),
            after_stream.tail() == ISet::empty().insert(ev),
            after_stream.infos().map().dom().contains(ev),
            after_stream.infos().map()[ev] == sub_val,
            src_timeline < old(core).timelines.count <= MAX_TIMELINES,
            old(core).timelines.kinds[src_timeline as int] == (TimelineKind::CudaStream {
                device: after_stream.device(),
            }),
            old(core).timelines.confirmed_seq[src_timeline as int] + 2 == old(
                core,
            ).timelines.next_seq[src_timeline as int],
            ticket.sequence() + 1 == old(core).timelines.next_seq[src_timeline as int],
            instance.id() == old(core).instance_id(),
            old(core).node_id == before.node_id && after_hop.node_id == before.node_id,
            old(core).instance_id() == before.instance_id() && after_hop.instance_id()
                == before.instance_id(),
            old(core).tiers == before.tiers,
            old(core).next_version_seq == before.next_version_seq,
            old(core).version_seq_limit == before.version_seq_limit,
            old(core).frames == after_hop.frames,
            old(core).spec == before.spec,
            old(core).version_registry == before.version_registry,
            old(core).active_leases == after_hop.active_leases,
            after_hop.frames.same_directory(&before.frames),
            after_hop.frames.preserves_initializing(&before.frames, None),
            after_hop.frames.preserves_remote_leases(&before.frames),
            after_hop.active_leases.len() == before.active_leases.len(),
            (hop.src_lease.slot as usize) < before.active_leases.len(),
            before.active_leases[hop.src_lease.slot as int].is_none(),
            forall|s: int|
                0 <= s < before.active_leases.len() && s != hop.src_lease.slot as int
                    ==> #[trigger] after_hop.active_leases[s] == before.active_leases[s],
            old(core).active_lease_matches(hop.src_lease),
            old(core).active_leases[hop.src_lease.slot as int]->Some_0.share@ == src_cap.share(),
            hop.src_lease.frame == hop.src_frame,
            hop.src_frame.0 < old(core).frames.metas.len(),
            hop.dst_frame.0 < before.frames.metas.len(),
            hop.dst_frame.0 < old(core).frames.metas.len(),
            before.frames.meta(hop.dst_frame).state != STATE_INITIALIZING,
            before.frames.frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames.meta(hop.dst_frame).state == STATE_INITIALIZING,
            !old(core).frames.meta(hop.dst_frame).queued(),
            old(core).frames.frame(hop.dst_frame).tier == dst_tier,
            hop.dst_tier == dst_tier,
            old(core).frames.frame(hop.dst_frame).slot == hop.dst_slot@,
            old(core).frames.frame(hop.dst_frame).bound_version == Some(hop.version),
            old(core).frames.frame(hop.dst_frame).token is None,
            old(core).frames.versions@.contains_key(hop.version),
            old(core).frames.versions@[hop.version] == hop.src_cert@.bytes,
            hop.version.origin_node != old(core).node_id || hop.version.seq < old(
                core,
            ).next_version_seq || hop.version.seq >= old(core).version_seq_limit,
            dst_is_gpu ==> dst_tier == self.gpu_tier && self.owns_gpu(hop.dst_frame),
            !dst_is_gpu ==> self.owns_pinned_host(hop.dst_frame),
            src_cap.well_formed(),
            src_cap.instance_id() == old(core).instance_id(),
            src_cap.loc() == old(core).frames.frame(hop.src_frame).slot.loc,
            src_cap.frontier().readers == ISet::empty().insert(ev),
            slot_owned(
                dst_tok,
                old(core).instance_id(),
                hop.dst_slot@.loc,
                Frontier { last_writer: Some(ev), readers: ISet::empty() },
            ),
            old(core).timelines.count == before.timelines.count,
            old(core).timelines.kinds == before.timelines.kinds,
            forall|tl: u8|
                tl < before.timelines.count && tl != src_timeline ==> {
                    &&& (#[trigger] old(core).timelines.next_seq[tl as int])
                        == before.timelines.next_seq[tl as int]
                    &&& old(core).timelines.confirmed_seq[tl as int]
                        == before.timelines.confirmed_seq[tl as int]
                    &&& forall|s: u64|
                        1 <= s < before.timelines.next_seq[tl as int] ==> #[trigger] old(
                            core,
                        ).timelines.events@[(tl, s)] == before.timelines.events@[(tl, s)]
                },
            ticket != Ticket::NONE,
            ticket.timeline_id() == src_timeline,
            1 <= ticket.sequence(),
            old(core).timelines.issued(src_timeline, ticket.sequence()),
            old(core).timelines.events@[(src_timeline, ticket.sequence())] == ev,
            ev == before_stream.source().next_event(),
            ev.source.succeeds_on_observe(),
            !after_hop.ledger@.submitted.dom().contains(ev),
            after_hop.ledger@ == before.ledger@,
            cited == after_hop.citations(),
            cited.subset_of(sub_val.preds),
            sub_val.preds.subset_of(after_hop.ledger@.submitted.dom()),
            sub_val.access.reads.contains(hop.src_slot@.loc),
            sub_val.access.writes == ISet::empty().insert(hop.dst_slot@.loc),
            sub_val.effect == copy_effect(hop.src_slot@, hop.dst_slot@),
            old(core).ledger@.submitted.map() == after_hop.ledger@.submitted.map().insert(
                ev,
                sub_val,
            ),
            old(core).ledger@.knows == after_hop.ledger@.knows,
            old(core).ledger@.succeeded == after_hop.ledger@.succeeded,
            old(core).ledger@.revoked == after_hop.ledger@.revoked,
            knows.instance_id() == old(core).instance_id(),
            knows.set() == foundation::tracked::knowledge_of(
                old(core).node(),
                closure_from(after_stream.infos().map(), after_stream.tail()),
            ),
            replica_valid(after_hop.ledger@, after_hop.node(), hop.src_cert@),
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
            final(core).node_id == before.node_id,
            final(core).instance_id() == before.instance_id(),
            final(core).tiers == before.tiers,
            final(core).frames.versions@ == before.frames.versions@,
            final(core).version_registry == before.version_registry,
            before.spec@.versions_monotonic(final(core).spec@),
            core_transition(
                &before,
                final(core),
                Some(src_timeline),
                Some(before_stream.source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            ((dst_tier == self.gpu_tier && self.owns_gpu(r.1)) || self.owns_pinned_host(r.1)),
            final(core).timelines.confirmed_seq[src_timeline as int] + 1
                == final(core).timelines.next_seq[src_timeline as int],
            final(core).frames.meta(r.1).state == STATE_PUBLISHED,
            final(core).frames.frame(r.1).tier == dst_tier,
            final(core).spec@.versions.contains_key(r.0),
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CudaBackend::well_formed);
        hide(FrameTable::same_directory);
        hide(FrameTable::same_layout);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::preserves_remote_leases);
        let ghost knows_val = knows;
        proof {
            assert(closure_from(after_stream.infos().map(), after_stream.tail()).contains(ev)) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            assert(core.timelines.observed(src_timeline, ticket.sequence(), core.node(), knows));
        }
        core.mark_timeline_complete(src_timeline, ticket.seq(), Tracked(knows));
        let ver = hop.version;
        let dst_frame = hop.dst_frame;
        let ghost dst_slot = hop.dst_slot@;
        let ghost lease_slot = hop.src_lease.slot;
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
                1 <= s <= ticket.sequence() && #[trigger] core.timelines.events@[(
                    ticket.timeline_id(),
                    s,
                )] == x by {
                assert(x == ev);
                let s = ticket.sequence();
                assert(1 <= s <= ticket.sequence() && core.timelines.events@[(
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
            assert(core.ledger@.submitted.dom() =~= before.ledger@.submitted.dom().insert(ev)) by {
                broadcast use vstd::imap::group_imap_lemmas;

            }
            assert(core.frames.versions@ =~= before.frames.versions@) by {
                broadcast use vstd::map::group_map_lemmas;

                reveal(CacheCore::well_formed);
            }
            assert(core.spec@.versions.contains_key(ver)) by {
                broadcast use vstd::map::group_map_lemmas;

            }
            assert(core.frames.frame(dst_frame).tier == dst_tier) by {
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
    pub fn copy_hop_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(shares): Tracked<&mut ShareSource>,
        src_stream: &mut CudaStream,
        src_timeline: u8,
        src_frame: FrameId,
        dst_tier: TierId,
    ) -> (r: BackendResult<(VersionId, FrameId)>)
        requires
            old(core).well_formed(),
            self.well_formed(old(core)),
            caller_stream_valid(*old(src_stream), old(core), src_timeline, self.timeline.stream),
            src_timeline != self.timeline.timeline_id,
            old(core).timelines.confirmed_seq[src_timeline as int] + 1 == old(
                core,
            ).timelines.next_seq[src_timeline as int],
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
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).frames.versions@ == old(core).frames.versions@,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(src_timeline),
                Some(old(src_stream).source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r matches Ok((ver, dst_frame)) ==> {
                &&& ((dst_tier == self.gpu_tier && self.owns_gpu(dst_frame))
                    || self.owns_pinned_host(dst_frame))
                &&& final(core).timelines.confirmed_seq[src_timeline as int] + 1
                    == final(core).timelines.next_seq[src_timeline as int]
                &&& final(core).frames.meta(dst_frame).state == STATE_PUBLISHED
                &&& final(core).frames.frame(dst_frame).tier == dst_tier
                &&& final(core).spec@.versions.contains_key(ver)
            },
    {
        hide(CacheCore::well_formed);
        hide(replica_valid);
        hide(complete_receipts);
        hide(CudaBackend::well_formed);
        hide(FrameTable::same_directory);
        hide(FrameTable::same_layout);
        hide(FrameTable::preserves_initializing);
        hide(FrameTable::preserves_remote_leases);
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
                    Some(src_timeline),
                    Some(before_stream.source().source()),
                    FrameTouchMode::PreserveAllInit,
                );
            }
            return Err(CoreError::UnsupportedTier);
        }
        proof {
            assert({
                &&& self.gpu_buffer.well_formed()
                &&& self.gpu_buffer.node() == before.node()
                &&& self.gpu_buffer.instance_id() == before.instance_id()
                &&& self.gpu_buffer.space() is GpuAlloc
                &&& self.gpu_buffer.space()->GpuAlloc_device == self.device
                &&& self.gpu_buffer.num_slots() == self.gpu_num_frames
                &&& self.gpu_buffer.slot_len() == before.frames.tier_frame_size(self.gpu_tier)
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
                    &&& h.reg.slot_len() == before.frames.tier_frame_size(h.tier)
                    &&& before.frames.tier_frame_size(h.tier) == before.frames.tier_frame_size(
                        self.gpu_tier,
                    )
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
                    assert(core.frames.versions@ == before.frames.versions@) by {
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
                        Some(src_timeline),
                        Some(before_stream.source().source()),
                        FrameTouchMode::PreserveAllInit,
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
            assert(before.frames.frame(hop.dst_frame).slot == hop.dst_slot@) by {
                reveal(FrameTable::same_directory);
            }
        }
        if core.timelines.next_seq[src_timeline as usize] > MAX_TICKET_SEQ {
            let ghost dst_frame = hop.dst_frame;
            let ghost lease_slot = hop.src_lease.slot;
            core.abort_copy_hop(
                hop,
                Tracked(src_cap),
                Ticket::NONE,
                Tracked(dst_tok),
                Tracked(instance),
            );
            proof {
                assert(core.frames.versions@ == before.frames.versions@) by {
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
                    Some(src_timeline),
                    Some(before_stream.source().source()),
                    FrameTouchMode::PreserveAllInit,
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
            let tracked ledger_ref = core.ledger.borrow_mut();
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
                let ghost lease_slot = hop.src_lease.slot;
                core.abort_copy_hop(
                    hop,
                    Tracked(src_cap),
                    Ticket::NONE,
                    Tracked(dst_tok),
                    Tracked(instance),
                );
                proof {
                    assert(core.frames.versions@ == before.frames.versions@) by {
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
                        Some(src_timeline),
                        Some(before_stream.source().source()),
                        FrameTouchMode::PreserveAllInit,
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
                    assert(core.ledger@.submitted.dom() =~= before.ledger@.submitted.dom().insert(
                        ev,
                    )) by {
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
                    assert(sub_val.preds.subset_of(after_hop.ledger@.submitted.dom())) by {
                        broadcast use vstd::iset::group_iset_lemmas;

                        assert forall|x: TransferEvent| #[trigger]
                            sub_val.preds.contains(
                                x,
                            ) implies after_hop.ledger@.submitted.dom().contains(x) by {
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
