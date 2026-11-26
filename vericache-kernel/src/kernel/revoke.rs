use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

impl CacheCore {
    pub fn begin_fenced_revocation(&mut self, f: FrameId, Ghost(share): Ghost<TransferShare>) -> (r:
        Result<PendingRevocation, CoreError>)
        requires
            old(self).well_formed(),
            f.0 < old(self).frames_spec().metas_spec().len() && old(self).frames_spec().meta(
                f,
            ).active_fenced_leases > 0 ==> old(self).frames_spec().frame(f).has_fenced_share(share),
        ensures
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).same_runtime_setup(old(self)),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            f.0 < old(self).frames_spec().metas_spec().len() && old(self).frames_spec().meta(
                f,
            ).active_fenced_leases > 0 ==> r.is_ok(),
            match r {
                Err(_) => *final(self) == *old(self) && final(self).well_formed(),
                Ok(pending) => {
                    &&& final(self).well_formed_with_pending_revocation(pending)
                    &&& pending.frame == f
                    &&& pending.share@ == share
                    &&& pending.slot_ref@ == old(self).frames_spec().frame(f).slot
                    &&& pending.token@ == old(self).frames_spec().frame(f).token
                    &&& pending.token@ is Some
                    &&& slot_is_lent_with(
                        pending.token@->Some_0,
                        old(self).instance_id(),
                        pending.slot_ref@.loc,
                        share,
                    )
                    &&& pending.snapshot@ == old(self).frames_spec()
                },
            },
    {
        let ghost v = self.kernel_view();
        self.frames.begin_fenced_revocation(f, Ghost(share), Ghost(v))
    }

    pub fn cancel_fenced_revocation(&mut self, pending: PendingRevocation)
        requires
            old(self).well_formed_with_pending_revocation(pending),
            pending.token@ == pending.snapshot@.frame(pending.frame).token,
        ensures
            final(self).well_formed(),
            final(self).frames_spec() == pending.snapshot@,
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).same_runtime_setup(old(self)),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
    {
        let ghost v = self.kernel_view();
        self.frames.cancel_fenced_revocation(pending, Ghost(v));
    }

    proof fn lemma_fence_view_extends(
        before_view: KernelView,
        v_after: KernelView,
        fence_event: TransferEvent,
    )
        requires
            before_view.timelines.well_formed(
                before_view.id(),
                before_view.node,
                &before_view.ledger,
            ),
            complete_receipts(before_view.ledger),
            v_after.node == before_view.node,
            v_after.timelines == before_view.timelines,
            v_after.id() == before_view.id(),
            v_after.ledger.valid(before_view.id(), before_view.node),
            v_after.ledger.submitted.dom().contains(fence_event),
            v_after.ledger.submitted.map()[fence_event].preds.subset_of(
                before_view.ledger.submitted.dom(),
            ),
            v_after.ledger.submitted.map() == before_view.ledger.submitted.map().insert(
                fence_event,
                v_after.ledger.submitted.map()[fence_event],
            ),
            before_view.ledger.submitted.map().submap_of(v_after.ledger.submitted.map()),
            revoked_submap(before_view.ledger.revoked, v_after.ledger.revoked),
            before_view.ledger.knows.set().subset_of(v_after.ledger.knows.set()),
            v_after.ledger.succeeded == before_view.ledger.succeeded,
        ensures
            complete_receipts(v_after.ledger),
            v_after.extends(before_view),
    {
        assert forall|e: TransferEvent| #[trigger]
            v_after.ledger.submitted.dom().contains(
                e,
            ) implies v_after.ledger.submitted.map()[e].preds.subset_of(
            v_after.ledger.submitted.dom(),
        ) by {
            if e == fence_event {
                assert(v_after.ledger.submitted.map()[fence_event].preds.subset_of(
                    before_view.ledger.submitted.dom(),
                ));
            } else {
                assert(before_view.ledger.submitted.dom().contains(e));
                assert(before_view.ledger.submitted.map()[e] == v_after.ledger.submitted.map()[e]);
                assert(before_view.ledger.submitted.map()[e].preds.subset_of(
                    before_view.ledger.submitted.dom(),
                ));
            }
        }
        assert forall|t: TimelineId, s: u64|
            v_after.timelines.issued(
                t,
                s,
            ) implies #[trigger] v_after.timelines.events_spec()@.contains_key((t, s))
            && v_after.ledger.submitted.dom().contains(
            v_after.timelines.events_spec()@[(t, s)],
        ) by {
            assert(before_view.timelines.events_spec()@.contains_key((t, s)));
            assert(before_view.ledger.submitted.dom().contains(
                before_view.timelines.events_spec()@[(t, s)],
            ));
        }
        assert forall|t: TimelineId, s: u64|
            t < v_after.timelines.count && 1 <= s
                <= v_after.timelines.confirmed_seq[t as int] implies #[trigger] v_after.timelines.observed(
        t, s, v_after.node, v_after.ledger.knows) by {
            assert(before_view.timelines.observed(
                t,
                s,
                before_view.node,
                before_view.ledger.knows,
            ));
        }
        assert(before_view.ledger.citations(v_after.node).subset_of(
            v_after.ledger.citations(v_after.node),
        ));
    }

    proof fn lemma_fence_certified(
        snap: &FrameTable,
        before_view: KernelView,
        v_after: KernelView,
        fence_event: TransferEvent,
    )
        requires
            before_view.ledger.valid(before_view.id(), before_view.node),
            v_after.ledger.valid(before_view.id(), before_view.node),
            complete_receipts(before_view.ledger),
            complete_receipts(v_after.ledger),
            CacheCore::certified(snap, before_view),
            v_after.node == before_view.node,
            v_after.id() == before_view.id(),
            v_after.ledger.submitted.dom().contains(fence_event),
            v_after.ledger.submitted.map()[fence_event].access.writes == ISet::<SlotLoc>::empty(),
            v_after.ledger.submitted.map() == before_view.ledger.submitted.map().insert(
                fence_event,
                v_after.ledger.submitted.map()[fence_event],
            ),
            before_view.ledger.submitted.map().submap_of(v_after.ledger.submitted.map()),
            revoked_submap(before_view.ledger.revoked, v_after.ledger.revoked),
            before_view.ledger.knows.set().subset_of(v_after.ledger.knows.set()),
            v_after.ledger.succeeded == before_view.ledger.succeeded,
        ensures
            CacheCore::certified(snap, v_after),
    {
        assert(before_view.ledger.valid(before_view.ledger.submitted.instance_id(), v_after.node));
        assert forall|g: FrameId|
            g.0 < snap.metas.len() && (#[trigger] snap.meta(g)).state == STATE_PUBLISHED implies {
            let fg = snap.frame(g);
            fg.cert matches Some(cert) && {
                &&& replica_valid(v_after.ledger, v_after.node, cert)
                &&& event_succeeded(v_after.ledger, v_after.node, cert.writer)
                &&& v_after.ledger.citations(v_after.node).contains(cert.writer)
                &&& fg.bound_version matches Some(ver)
                &&& snap.versions_spec()@.contains_key(ver)
                &&& snap.versions_spec()@[ver] == cert.bytes
            }
        } by {
            let cert = snap.frame(g).cert.unwrap();
            assert forall|e: TransferEvent| #[trigger]
                v_after.ledger.submitted.dom().contains(e)
                    && !before_view.ledger.submitted.dom().contains(
                    e,
                ) implies !v_after.ledger.submitted.map()[e].access.writes.contains(
                cert.slot.loc,
            ) by {
                assert(e == fence_event);
            }
            assert(before_view.ledger.succeeded.dom().subset_of(v_after.ledger.succeeded.dom()));
            replica_survives_disjoint_writes(
                before_view.ledger,
                v_after.ledger,
                v_after.node,
                cert,
            );
        }
    }

    /// Record the hardware revocation fence event, verify the returned `RevokedToken`
    /// and updated `SlotToken`, and restore `CacheCore::well_formed()` atomically.
    pub fn finish_fenced_revocation(
        &mut self,
        pending: PendingRevocation,
        Tracked(fence_submitted): Tracked<Submitted>,
        Tracked(fence_knows): Tracked<Knows>,
        Tracked(revoked_slot): Tracked<SlotToken>,
        Tracked(revoked_token): Tracked<&RevokedToken>,
    )
        requires
            old(self).well_formed_with_pending_revocation(pending),
            fence_submitted.instance_id() == old(self).instance_id(),
            fence_knows.instance_id() == old(self).instance_id(),
            fence_submitted.value().access.writes == ISet::<SlotLoc>::empty(),
            fence_submitted.value().preds.subset_of(old(self).ledger_spec()@.submitted.dom()),
            fence_knows.set().contains((old(self).node(), fence_submitted.key())),
            revoked_token.instance_id() == old(self).instance_id(),
            revoked_token.key() == (
                pending.snapshot@.frame(pending.frame).slot.loc,
                pending.share@,
            ),
            revoked_token.value() == fence_submitted.key(),
            revoked_slot.well_formed(),
            revoked_slot.instance_id() == old(self).instance_id(),
            revoked_slot.key() == pending.snapshot@.frame(pending.frame).slot.loc,
            revoked_slot.value() == fold_share(
                pending.snapshot@.frame(pending.frame).slot_tok().value(),
                pending.share@,
                Frontier {
                    last_writer: None,
                    readers: ISet::empty().insert(fence_submitted.key()),
                },
            ),
        ensures
            final(self).well_formed(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).frames_spec().same_directory(&pending.snapshot@),
            final(self).frames_spec().preserves_initializing(&pending.snapshot@, None),
            final(self).frames_spec().same_other_pins(&pending.snapshot@, Some(pending.frame)),
            final(self).frames_spec().frame(pending.frame).fenced_shares == pending.snapshot@.frame(
                pending.frame,
            ).fenced_shares.remove(pending.share@),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec()@.submitted.map() == old(
                self,
            ).ledger_spec()@.submitted.map().insert(fence_submitted.key(), fence_submitted.value()),
            old(self).ledger_spec()@.submitted.map().submap_of(
                final(self).ledger_spec()@.submitted.map(),
            ),
            final(self).ledger_spec()@.knows.set() == old(self).ledger_spec()@.knows.set().union(
                fence_knows.set(),
            ),
            final(self).ledger_spec()@.succeeded == old(self).ledger_spec()@.succeeded,
            final(self).ledger_spec()@.revoked.map() == old(
                self,
            ).ledger_spec()@.revoked.map().insert(revoked_token.key(), revoked_token.value()),
            revoked_submap(old(self).ledger_spec()@.revoked, final(self).ledger_spec()@.revoked),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
    {
        let ghost before_view = self.kernel_view();
        let ghost snap = pending.snapshot@;
        let ghost f = pending.frame;
        let ghost fence_event = fence_submitted.key();
        proof {
            self.ledger.borrow_mut().record_observed_event(fence_submitted, fence_knows);
            self.ledger.borrow_mut().record_revoked(*revoked_token);
            assert(self.ledger@.citations(self.node()).contains(fence_event));
        }
        self.frames.finish_fenced_revocation(
            pending,
            Tracked(revoked_slot),
            Tracked(revoked_token),
            Ghost(fence_event),
            Tracked(self.ledger.borrow()),
            Ghost(before_view),
        );
        proof {
            let v_after = self.kernel_view();
            Self::lemma_fence_view_extends(before_view, v_after, fence_event);
            Self::lemma_fence_certified(&snap, before_view, v_after, fence_event);
            snap.lemma_timeline_progress(before_view, v_after);
            Self::lemma_same_directory_preserves_state(
                &self.catalog,
                &snap,
                &self.frames,
                v_after,
                v_after,
            );
            assert(self.frames.frame(f).local_shares == snap.frame(f).local_shares);
        }
    }
}

} // verus!
