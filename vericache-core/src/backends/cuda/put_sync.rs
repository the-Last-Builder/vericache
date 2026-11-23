#[cfg(verus_only)]
use super::copy::caller_stream_valid;
use super::*;
#[cfg(verus_only)]
use foundation::tracked::Frontier;
use transfer::cuda::{CudaSlice, CudaStream};

verus! {

impl<'a> CudaBackend<'a> {
    pub(super) proof fn lemma_ticket_covers_self(
        timelines: &TimelineTable,
        k: Ticket,
        e: TransferEvent,
    )
        requires
            k != Ticket::NONE,
            1 <= k.sequence(),
            timelines.events@[(k.timeline_id(), k.sequence())] == e,
        ensures
            timelines.ticket_covers(k, e),
    {
        let s = k.sequence();
        assert(1 <= s <= k.sequence() && timelines.events@[(k.timeline_id(), s)] == e);
    }

    pub(super) proof fn lemma_put_sync_copy_admitted(
        core: &CacheCore,
        src_stream: CudaStream,
        src_perm: ReadPerm,
        dst_tok: SlotToken,
        cited: ISet<TransferEvent>,
    )
        requires
            core.well_formed(),
            src_stream.well_formed(),
            src_stream.source().id() == core.instance_id(),
            src_stream.source().node() == core.node(),
            src_stream.infos().map().submap_of(core.ledger@.submitted.map()),
            src_stream.tail().subset_of(src_stream.infos().dom()),
            source_fresh_for_ledger(src_stream.source(), core.ledger@),
            cited == core.citations(),
            read_admitted(src_perm.frontier(), cited),
            write_admitted(slot_frontier(dst_tok), cited),
        ensures
            !core.ledger@.submitted.dom().contains(src_stream.source().next_event()),
            src_stream.infos().map().union_prefer_right(core.ledger@.submitted.map())
                == core.ledger@.submitted.map(),
            src_stream.tail().union(cited).subset_of(core.ledger@.submitted.map().dom()),
            cuda::copy_admitted(
                src_stream,
                src_perm,
                dst_tok,
                core.ledger@.submitted,
                core.ledger@.knows,
                cited,
            ),
    {
        hide(CacheCore::well_formed);
        broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

        lemma_core_wf_facts(core);
        lemma_citations_closure(core);
        lemma_fresh_source_event_absent(src_stream.source(), core.ledger@);
        core.lemma_citations_admit_closure(src_perm.frontier(), src_stream.tail());
        core.lemma_citations_admit_closure(slot_frontier(dst_tok), src_stream.tail());
        assert(src_stream.infos().map().union_prefer_right(core.ledger@.submitted.map())
            =~= core.ledger@.submitted.map());
        assert(src_stream.tail().union(cited).subset_of(core.ledger@.submitted.map().dom()));
    }

    proof fn lemma_put_sync_abort_transition(
        before: &CacheCore,
        mid: &CacheCore,
        after: &CacheCore,
        frame: FrameId,
        src_timeline: u8,
        source: EventSource,
    )
        requires
            before.well_formed(),
            mid.well_formed(),
            after.well_formed(),
            frame.0 < before.frames.metas.len(),
            before.frames.meta(frame).state == STATE_EMPTY,
            mid.frames.same_directory(&before.frames),
            after.frames.same_directory(&mid.frames),
            mid.frames.preserves_initializing(&before.frames, None),
            after.frames.preserves_initializing(&mid.frames, Some(frame)),
            mid.frames.preserves_remote_leases(&before.frames),
            after.frames.preserves_remote_leases(&mid.frames),
            after.node_id == before.node_id,
            after.tiers == before.tiers,
            after.timelines == before.timelines,
            after.ledger == before.ledger,
            after.next_version_seq >= before.next_version_seq,
            after.version_seq_limit == before.version_seq_limit,
            after.active_leases == before.active_leases,
        ensures
            core_transition(
                before,
                after,
                Some(src_timeline),
                Some(source),
                FrameTouchMode::PreserveAllInit,
            ),
            after.frames.versions@ == before.frames.versions@,
    {
        hide(CacheCore::well_formed);
        assert(after.frames.same_directory(&before.frames));
        assert(after.frames.preserves_initializing(&before.frames, None)) by {
            assert forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING implies {
                &&& after.frames.meta(f) == before.frames.meta(f)
                &&& after.frames.frame(f) == before.frames.frame(f)
            } by {
                assert(f != frame);
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
        lemma_core_transition_no_event(before, after, Some(src_timeline), Some(source));
    }

    pub(super) proof fn lemma_put_sync_preds_closed(
        after_bw: &CacheCore,
        cur: &CacheCore,
        before_stream: CudaStream,
        cited: ISet<TransferEvent>,
        sub_val: foundation::tracked::EventInfo<TransferDomain>,
        frame: FrameId,
    )
        requires
            after_bw.well_formed(),
            cur.node_id == after_bw.node_id,
            cur.tiers == after_bw.tiers,
            cur.timelines == after_bw.timelines,
            cur.frames == after_bw.frames,
            cur.catalog == after_bw.catalog,
            cur.next_version_seq == after_bw.next_version_seq,
            cur.version_seq_start == after_bw.version_seq_start,
            cur.version_seq_limit == after_bw.version_seq_limit,
            cur.active_leases == after_bw.active_leases,
            cur.free_lease_slots == after_bw.free_lease_slots,
            cur.lease_generation == after_bw.lease_generation,
            cur.spec == after_bw.spec,
            cur.version_registry == after_bw.version_registry,
            cur.ledger@.submitted.instance_id() == after_bw.ledger@.submitted.instance_id(),
            cur.ledger@.submitted.map() == after_bw.ledger@.submitted.map(),
            cur.ledger@.knows == after_bw.ledger@.knows,
            cur.ledger@.succeeded == after_bw.ledger@.succeeded,
            cur.ledger@.revoked == after_bw.ledger@.revoked,
            cited == after_bw.citations(),
            before_stream.tail().union(cited).subset_of(after_bw.ledger@.submitted.map().dom()),
            sub_val.preds == closure_from(
                after_bw.ledger@.submitted.map(),
                before_stream.tail().union(cited),
            ),
            frame.0 < cur.frames.metas.len(),
            cur.frames.meta(frame).state == STATE_INITIALIZING,
            sub_val.access.writes == ISet::empty().insert(cur.frames.frame(frame).slot.loc),
        ensures
            cur.well_formed(),
            sub_val.preds.subset_of(cur.ledger@.submitted.dom()),
            forall|g: FrameId|
                g.0 < cur.frames.metas.len() && cur.frames.meta(g).state == STATE_PUBLISHED ==> !(
                #[trigger] sub_val.access.writes.contains(cur.frames.frame(g).slot.loc)),
    {
        broadcast use {
            vstd::imap::group_imap_lemmas,
            vstd::iset::group_iset_lemmas,
            vstd::map::group_map_lemmas,
        };

        cur.preserve_after_ledger_growth(*after_bw);
        let dom = cur.ledger@.submitted.dom();
        let waits = before_stream.tail().union(cited);
        assert(waits.subset_of(dom));
        assert forall|e: TransferEvent| #[trigger] sub_val.preds.contains(e) implies dom.contains(
            e,
        ) by {
            if !waits.contains(e) {
                let m = choose|m: TransferEvent|
                    waits.contains(m) && (#[trigger] cur.ledger@.submitted.map()[m]).preds.contains(
                        e,
                    );
                assert(dom.contains(m));
                assert(cur.ledger@.submitted.map()[m].preds.subset_of(dom));
            }
        }
        lemma_write_disjoint_from_published(cur, frame, sub_val.access.writes);
    }

    pub(super) proof fn lemma_put_sync_copy_cert(
        before: &CacheCore,
        after: &CacheCore,
        src_stream: CudaStream,
        src_cert: ValueCertificate,
        src_slot: SlotRef,
        dst_slot: SlotRef,
        ev: TransferEvent,
        sub_val: foundation::tracked::EventInfo<TransferDomain>,
        knows_val: Knows,
        cited: ISet<TransferEvent>,
    ) -> (cert: ValueCertificate)
        requires
            before.well_formed(),
            after.well_formed(),
            after.node_id == before.node_id,
            after.instance_id() == before.instance_id(),
            cited == before.citations(),
            cited.subset_of(sub_val.preds),
            !before.ledger@.submitted.dom().contains(ev),
            after.ledger@.submitted.map() == before.ledger@.submitted.map().insert(ev, sub_val),
            after.ledger@.knows.set() == before.ledger@.knows.set().union(knows_val.set()),
            after.ledger@.succeeded == before.ledger@.succeeded,
            after.ledger@.revoked == before.ledger@.revoked,
            sub_val.preds.subset_of(before.ledger@.submitted.dom()),
            sub_val.access.reads.contains(src_slot.loc),
            sub_val.access.writes.contains(dst_slot.loc),
            sub_val.effect == copy_effect(src_slot, dst_slot),
            ev.source.succeeds_on_observe(),
            src_stream.tail() == ISet::empty().insert(ev),
            src_stream.infos().map().dom().contains(ev),
            src_stream.infos().map()[ev] == sub_val,
            knows_val.set() == knowledge_of(
                after.node(),
                closure_from(src_stream.infos().map(), src_stream.tail()),
            ),
            replica_valid(before.ledger@, before.node(), src_cert),
            source_slice_valid(src_cert, src_slot),
            src_cert.slot == src_slot,
            src_slot.well_formed(),
            src_cert.bytes.len() == src_slot.sub_len,
            dst_slot.well_formed(),
            dst_slot.is_full(),
            src_slot.sub_len == dst_slot.sub_len,
        ensures
            replica_valid(after.ledger@, after.node(), cert),
            cert.slot == dst_slot,
            cert.writer == ev,
            cert.bytes == src_cert.bytes,
    {
        broadcast use {
            vstd::imap::group_imap_lemmas,
            vstd::iset::group_iset_lemmas,
            vstd::map::group_map_lemmas,
        };

        assert(closure_from(src_stream.infos().map(), src_stream.tail()) =~= sub_val.preds.insert(
            ev,
        )) by {
            assert forall|x: TransferEvent|
                closure_from(src_stream.infos().map(), src_stream.tail()).contains(x) == (
                #[trigger] sub_val.preds.insert(ev).contains(x)) by {
                if x == ev {
                    assert(src_stream.tail().contains(ev));
                } else if sub_val.preds.contains(x) {
                    assert(src_stream.tail().contains(ev));
                    assert(src_stream.infos().map().dom().contains(ev));
                    assert(src_stream.infos().map()[ev].preds.contains(x));
                }
                if closure_from(src_stream.infos().map(), src_stream.tail()).contains(x) && x
                    != ev {
                    let m = choose|m: TransferEvent|
                        src_stream.tail().contains(m) && src_stream.infos().map().dom().contains(m)
                            && (#[trigger] src_stream.infos().map()[m]).preds.contains(x);
                    assert(m == ev);
                    assert(sub_val.preds.contains(x));
                }
            }
        }
        assert(knows_val.set() =~= knowledge_of(after.node(), sub_val.preds.insert(ev)));
        lemma_observed_event_in_citations(
            before.ledger@,
            after.ledger@,
            after.node(),
            ev,
            sub_val,
            knows_val,
            sub_val.preds,
        );
        assert(before.ledger@.submitted.map().submap_of(after.ledger@.submitted.map()));
        assert(revoked_submap(before.ledger@.revoked, after.ledger@.revoked));
        assert(event_succeeded(after.ledger@, after.node(), ev));
        let copy = ReplicaCopy {
            before: before.ledger@,
            source: src_cert,
            source_slot: src_slot,
            writer: ev,
            destination: dst_slot,
        };
        assert(effect_implies(sub_val.effect, copy_effect(src_slot, dst_slot)));
        assert forall|e: TransferEvent| #[trigger]
            after.ledger@.submitted.dom().contains(e) && !before.ledger@.submitted.dom().contains(e)
                && e != ev implies !after.ledger@.submitted.map()[e].access.writes.contains(
            dst_slot.loc,
        ) by {
            assert(e == ev);
        }
        assert(copy.retained(after.ledger@, after.node()));
        let cert = copy.certify(after.ledger@, after.node());
        assert(source_slice_bytes(src_cert, src_slot) =~= src_cert.bytes);
        assert(cert.bytes =~= src_cert.bytes);
        cert
    }

    proof fn lemma_put_sync_step_transition(
        &self,
        before: &CacheCore,
        mid: &CacheCore,
        after: &CacheCore,
        before_stream: CudaStream,
        after_stream: CudaStream,
        src_timeline: u8,
        ev: TransferEvent,
        frame: FrameId,
        dst_loc: SlotLoc,
        new_ver: Option<VersionId>,
    )
        requires
            before.well_formed(),
            after.well_formed(),
            self.well_formed(before),
            before.timelines.bounds(),
            src_timeline != self.timeline.timeline_id,
            before_stream.source().source() != self.timeline.stream.source().source(),
            frame.0 < before.frames.metas.len(),
            before.frames.meta(frame).state == STATE_EMPTY,
            before.frames.frame(frame).slot.loc == dst_loc,
            after.node_id == before.node_id,
            after.instance_id() == before.instance_id(),
            after.tiers == before.tiers,
            after.next_version_seq >= before.next_version_seq,
            after.version_seq_limit == before.version_seq_limit,
            after.active_leases == before.active_leases,
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
            after.frames.preserves_initializing(&mid.frames, Some(frame)),
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
            match new_ver {
                None => after.frames.versions@ == before.frames.versions@,
                Some(v) => {
                    &&& v.origin_node == before.node_id
                    &&& before.next_version_seq <= v.seq
                    &&& v.seq < before.version_seq_limit
                    &&& after.frames.versions@.dom() == before.frames.versions@.dom().insert(v)
                },
            },
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
        assert(after.frames.same_layout(&before.frames));
        assert(after.frames.preserves_initializing(&before.frames, None)) by {
            assert forall|f: FrameId|
                f.0 < before.frames.metas.len() && (#[trigger] before.frames.meta(f)).state
                    == STATE_INITIALIZING implies {
                &&& after.frames.meta(f) == before.frames.meta(f)
                &&& after.frames.frame(f) == before.frames.frame(f)
            } by {
                assert(f != frame);
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
        lemma_non_init_frame_disjoint_from_init(before, frame);
        lemma_core_transition_single_event(
            before,
            after,
            Some(src_timeline),
            before_stream.source().source(),
            ev,
            Some(dst_loc),
            new_ver,
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

    /// Synchronously publish a GPU-VRAM frame from caller's `CudaSlice` (`src`)
    /// using caller's idle `src_stream` / `src_timeline`, keeping `self` shared
    /// (`&self`) so RDMA GPU MRs can simultaneously borrow `self.gpu_buffer`.
    pub fn put_gpu_sync(
        &self,
        core: &mut CacheCore,
        Tracked(instance): Tracked<&TransferInstance>,
        src_stream: &mut CudaStream,
        src_timeline: u8,
        key: ObjectKey,
        src: &CudaSlice,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Ghost(src_cert): Ghost<ValueCertificate>,
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
            src.slot_ref().loc.space is GpuAlloc && src.slot_ref().loc.space->GpuAlloc_device
                != self.device ==> (old(src_perm).frontier().last_writer matches Some(w) ==> !(
            w.source is RdmaQueuePair)),
            instance.id() == old(core).instance_id(),
            src.well_formed(),
            src.instance_id() == old(core).instance_id(),
            src.slot_ref().loc.node == old(core).node(),
            src.slot_ref().sub_len == old(core).frames.tier_frame_size(self.gpu_tier),
            old(src_perm).valid_local(old(core).instance_id(), src.slot_ref().loc),
            read_admitted(old(src_perm).frontier(), old(core).citations()),
            replica_valid(old(core).ledger@, old(core).node(), src_cert),
            source_slice_valid(src_cert, src.slot_ref()),
            src_cert.slot == src.slot_ref(),
            src_cert.bytes.len() == src.slot_ref().sub_len,
            forall|f: FrameId|
                f.0 < old(core).frames.metas.len() && (#[trigger] old(core).frames.meta(f)).state
                    == STATE_EMPTY ==> src.slot_ref().loc != old(core).frames.frame(f).slot.loc,
        ensures
            final(core).well_formed(),
            self.well_formed(final(core)),
            caller_stream_valid(
                *final(src_stream),
                final(core),
                src_timeline,
                self.timeline.stream,
            ),
            final(core).node_id == old(core).node_id,
            final(core).instance_id() == old(core).instance_id(),
            final(core).tiers == old(core).tiers,
            final(core).version_registry == old(core).version_registry,
            old(core).spec@.versions_monotonic(final(core).spec@),
            core_transition(
                old(core),
                final(core),
                Some(src_timeline),
                Some(old(src_stream).source().source()),
                FrameTouchMode::PreserveAllInit,
            ),
            r.is_err() ==> final(core).frames.versions@ == old(core).frames.versions@,
            r matches Ok((ver, frame)) ==> {
                &&& self.owns_gpu(frame)
                &&& ver.origin_node == old(core).node_id
                &&& old(core).next_version_seq <= ver.seq
                &&& ver.seq < old(core).version_seq_limit
                &&& ver.seq < final(core).next_version_seq
                &&& final(core).timelines.confirmed_seq[src_timeline as int] + 1
                    == final(core).timelines.next_seq[src_timeline as int]
                &&& final(core).spec@.versions == old(core).spec@.versions.insert(
                    ver,
                    src_cert.bytes,
                )
                &&& final(core).spec@.catalog == old(core).spec@.catalog.insert(key, ver)
                &&& final(core).frames.meta(frame).state == STATE_PUBLISHED
                &&& final(core).frames.versions@.contains_key(ver)
                &&& final(core).frames.versions@.dom() == old(core).frames.versions@.dom().insert(
                    ver,
                )
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
            }) by {
                reveal(CudaBackend::well_formed);
            }
        }
        let (handle, Tracked(mut dst_tok)) = match core.begin_write(self.gpu_tier, key) {
            Ok(v) => v,
            Err(e) => {
                proof {
                    lemma_core_transition_refl(
                        core,
                        Some(src_timeline),
                        Some(before_stream.source().source()),
                        FrameTouchMode::PreserveAllInit,
                    );
                }
                return Err(e);
            },
        };
        let ghost after_bw = *core;
        proof {
            lemma_core_wf_facts(&before);
            lemma_core_wf_facts(core);
            lemma_initializing_frame_slot_valid(core, handle.frame);
            lemma_slab_bound_same_directory(
                &before,
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
            );
            lemma_slab_owns_tier_frame(
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                handle.frame,
            );
            lemma_slab_frame_slot(
                core,
                self.gpu_tier,
                self.gpu_first_frame,
                self.gpu_num_frames,
                self.gpu_buffer.space(),
                handle.frame,
            );
            assert(before.frames.meta(handle.frame).state == STATE_EMPTY);
            assert(before.frames.frame(handle.frame).slot == handle.slot@) by {
                reveal(FrameTable::same_directory);
            }
            assert(src.slot_ref().loc != handle.slot@.loc);
        }
        if core.timelines.next_seq[src_timeline as usize] > MAX_TICKET_SEQ {
            let ghost frame = handle.frame;
            core.abort_write(handle, Tracked(dst_tok));
            proof {
                Self::lemma_put_sync_abort_transition(
                    &before,
                    &after_bw,
                    core,
                    frame,
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
        let dst = self.gpu_frame_slice(handle.frame);
        let ghost cited = core.citations();
        let ghost ev = src_stream.source().next_event();
        proof {
            Self::lemma_put_sync_copy_admitted(core, *src_stream, *src_perm, dst_tok, cited);
        }
        let copy_res = {
            let tracked ledger_ref = core.ledger.borrow_mut();
            cuda::memcpy_async(
                src_stream,
                Tracked(instance),
                &dst,
                src,
                Tracked(src_perm),
                Tracked(&mut dst_tok),
                Tracked(&mut ledger_ref.submitted),
                Tracked(&ledger_ref.knows),
                Ghost(cited),
            )
        };
        let (Ghost(_ev), Tracked(receipt)) = match copy_res {
            Ok(v) => v,
            Err(e) => {
                let ghost frame = handle.frame;
                core.abort_write(handle, Tracked(dst_tok));
                proof {
                    Self::lemma_put_sync_abort_transition(
                        &before,
                        &after_bw,
                        core,
                        frame,
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
            assert(sub_val.access.writes =~= ISet::empty().insert(handle.slot@.loc)) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            Self::lemma_put_sync_preds_closed(
                &after_bw,
                core,
                before_stream,
                cited,
                sub_val,
                handle.frame,
            );
        }
        let ticket = core.issue_ticket(src_timeline, Tracked(receipt)).unwrap();
        let sync_res = cuda::stream_synchronize(src_stream, Tracked(instance));
        let Tracked(knows) = match sync_res {
            Ok(k) => k,
            Err(e) => {
                proof {
                    assert(slot_frontier(dst_tok).readers.subset_of(core.citations())) by {
                        broadcast use vstd::iset::group_iset_lemmas;

                    }
                    assert(ticket.timeline_id() == src_timeline);
                    assert(1 <= ticket.sequence());
                    assert(core.timelines.events@[(ticket.timeline_id(), ticket.sequence())] == ev);
                    Self::lemma_ticket_covers_self(&core.timelines, ticket, ev);
                }
                core.park_write(handle, Tracked(dst_tok), ticket);
                proof {
                    assert(core.frames.same_layout(&after_bw.frames)) by {
                        reveal(FrameTable::same_directory);
                        reveal(FrameTable::same_layout);
                    }
                    assert(core.frames.versions@ == before.frames.versions@) by {
                        reveal(FrameTable::same_directory);
                    }
                    assert(core.ledger@.submitted.dom() =~= before.ledger@.submitted.dom().insert(
                        ev,
                    )) by {
                        broadcast use vstd::imap::group_imap_lemmas;

                    }
                    self.lemma_put_sync_step_transition(
                        &before,
                        &after_bw,
                        core,
                        before_stream,
                        *src_stream,
                        src_timeline,
                        ev,
                        handle.frame,
                        handle.slot@.loc,
                        None,
                    );
                }
                return Err(cuda_error(e));
            },
        };
        let ghost knows_val = knows;
        proof {
            assert(core.timelines.events@[(src_timeline, ticket.sequence())] == ev);
            assert(closure_from(src_stream.infos().map(), src_stream.tail()).contains(ev)) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            assert(core.timelines.observed(src_timeline, ticket.sequence(), core.node(), knows));
        }
        core.mark_timeline_complete(src_timeline, ticket.seq(), Tracked(knows));
        let ver = handle.version;
        let frame = handle.frame;
        let ghost dst_slot = handle.slot@;
        let ghost cert;
        proof {
            assert(cited.subset_of(sub_val.preds)) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
            assert(sub_val.preds.subset_of(before.ledger@.submitted.dom())) by {
                broadcast use vstd::iset::group_iset_lemmas;

                assert forall|x: TransferEvent| #[trigger]
                    sub_val.preds.contains(x) implies before.ledger@.submitted.dom().contains(
                    x,
                ) by {
                    assert(x != ev);
                }
            }
            cert =
            Self::lemma_put_sync_copy_cert(
                &before,
                core,
                *src_stream,
                src_cert,
                src.slot_ref(),
                dst_slot,
                ev,
                sub_val,
                knows_val,
                cited,
            );
        }
        let ghost before_finish = *core;
        proof {
            assert(slot_frontier(dst_tok).readers.subset_of(core.citations())) by {
                broadcast use vstd::iset::group_iset_lemmas;

            }
        }
        core.finish_write(handle, Tracked(dst_tok), ticket, Ghost(cert));
        proof {
            assert(core.ledger@.submitted.dom() =~= before.ledger@.submitted.dom().insert(ev)) by {
                broadcast use vstd::imap::group_imap_lemmas;

            }
            assert(core.frames.versions@.contains_key(ver)) by {
                reveal(CacheCore::well_formed);
            }
            assert(core.frames.versions@.dom() =~= before.frames.versions@.dom().insert(ver)) by {
                broadcast use vstd::map::group_map_lemmas;

                reveal(CacheCore::well_formed);
            }
            self.lemma_put_sync_step_transition(
                &before,
                &before_finish,
                core,
                before_stream,
                *src_stream,
                src_timeline,
                ev,
                frame,
                dst_slot.loc,
                Some(ver),
            );
        }
        Ok((ver, frame))
    }
}

} // verus!
