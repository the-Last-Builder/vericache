use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// The submitted event and the observer's knowledge of its completion and predecessors.
pub tracked struct Completion {
    pub tracked submitted: Submitted,
    pub tracked knows: Knows,
}

impl Completion {
    pub open spec fn records(
        self,
        id: InstanceId,
        event: TransferEvent,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        preds: ISet<TransferEvent>,
        read_guards: ReadGuards,
    ) -> bool {
        &&& self.submitted.key() == event && self.submitted.instance_id() == id
        &&& self.submitted.value() == (EventInfo { access, effect, preds, read_guards })
        &&& self.knows.instance_id() == id
        &&& self.knows.set() == knowledge_of(event.observer, preds.insert(event))
    }

    pub proof fn infos(tracked &self) -> (tracked infos: SubmittedMap)
        ensures
            infos.instance_id() == self.submitted.instance_id(),
            infos.map() == IMap::empty().insert(self.submitted.key(), self.submitted.value()),
    {
        let tracked mut infos = SubmittedMap::empty(self.submitted.instance_id());
        infos.insert(self.submitted);
        infos
    }
}

/// The access that a synchronous shim completed, with its actual arguments.
/// Recording it consumes the witness. Ordinary read/write permissions remain
/// separate so a hardware operation can never create a foundation token.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub(crate) tracked struct CompletedAccess {
    pub(crate) ghost access: Access<TransferDomain>,
    pub(crate) ghost effect: Effect<TransferDomain>,
}

proof fn observe_single(
    tracked instance: &TransferInstance,
    observer: NodeId,
    tracked submitted: Submitted,
) -> (tracked knowledge: Knows)
    requires
        submitted.instance_id() == instance.id(),
    ensures
        knowledge.instance_id() == instance.id(),
        knowledge.set() == knowledge_of(observer, submitted.value().preds.insert(submitted.key())),
{
    let event = submitted.key();
    let tracked mut completed = SubmittedMap::empty(instance.id());
    completed.insert(submitted);
    let singleton = ISet::empty().insert(event);
    let tracked knowledge = instance.observe(observer, singleton, &completed);
    assert forall|e: TransferEvent| #[trigger]
        submitted.value().preds.contains(e) implies closure_from(
        completed.map(),
        singleton,
    ).contains(e) by {
        assert(completed.map()[event].preds.contains(e));
    }
    assert(closure_from(completed.map(), singleton) =~= submitted.value().preds.insert(event));
    knowledge
}

pub(crate) proof fn finish_owned(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    tracked completed_access: CompletedAccess,
    tracked slot: &mut SlotToken,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: Completion)
    requires
        old(source).well_formed(),
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        old(slot).well_formed(),
        old(slot).instance_id() == instance.id(),
        old(slot).value() is Owned,
        cited.subset_of(infos.dom()),
        knowledge_of(old(source).node(), cited).subset_of(knows.set()),
        completed_access.access.writes.subset_of(ISet::empty().insert(old(slot).key())),
        completed_access.access.reads.subset_of(ISet::empty().insert(old(slot).key())),
        admitted(
            slot_frontier(*old(slot)),
            old(slot).key(),
            completed_access.access,
            closure_from(infos.map(), cited),
        ),
    ensures
        final(source).well_formed(),
        final(source).node() == old(source).node(),
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        final(slot).well_formed(),
        final(slot).instance_id() == instance.id(),
        final(slot).key() == old(slot).key(),
        final(slot).value() == Slot::Owned(
            advance(
                slot_frontier(*old(slot)),
                old(slot).key(),
                old(source).next_event(),
                completed_access.access,
            ),
        ),
        result.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access,
            completed_access.effect,
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked submitted = instance.submit_1_owned(
        source.node(),
        completed_access.access,
        completed_access.effect,
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        slot,
    );
    let tracked knowledge = observe_single(instance, source.node(), submitted);
    Completion { submitted, knows: knowledge }
}

pub(crate) proof fn finish_read(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    tracked completed_access: CompletedAccess,
    tracked src_perm: &mut ReadPerm,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: Completion)
    requires
        old(source).well_formed(),
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        old(src_perm).instance_id() == instance.id(),
        old(src_perm).is_owned() || old(src_perm).is_shared_wf(),
        cited.subset_of(infos.dom()),
        knowledge_of(old(source).node(), cited).subset_of(knows.set()),
        completed_access.access == (Access {
            reads: ISet::empty().insert(old(src_perm).loc()),
            writes: ISet::empty(),
        }),
        read_admitted(old(src_perm).frontier(), closure_from(infos.map(), cited)),
    ensures
        final(source).well_formed(),
        final(source).node() == old(source).node(),
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        final(src_perm).advanced_read(*old(src_perm), old(source).next_event()),
        result.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access,
            completed_access.effect,
            closure_from(infos.map(), cited),
            old(src_perm).read_guards(),
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked submitted = instance.submit_1_read(
        source.node(),
        completed_access.effect,
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        src_perm,
    );
    let tracked knowledge = observe_single(instance, source.node(), submitted);
    Completion { submitted, knows: knowledge }
}

pub(crate) proof fn finish_copy(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    tracked completed_access: CompletedAccess,
    tracked src_perm: &mut ReadPerm,
    tracked dst_tok: &mut SlotToken,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: Completion)
    requires
        old(source).well_formed(),
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        old(src_perm).instance_id() == instance.id(),
        old(dst_tok).well_formed(),
        old(dst_tok).instance_id() == instance.id(),
        old(src_perm).is_owned() || old(src_perm).is_shared_wf(),
        old(dst_tok).value() is Owned,
        old(src_perm).loc() != old(dst_tok).key(),
        cited.subset_of(infos.dom()),
        knowledge_of(old(source).node(), cited).subset_of(knows.set()),
        completed_access.access.writes == ISet::empty().insert(old(dst_tok).key()),
        completed_access.access.reads.contains(old(src_perm).loc()),
        completed_access.access.reads.subset_of(
            ISet::empty().insert(old(src_perm).loc()).insert(old(dst_tok).key()),
        ),
        read_admitted(old(src_perm).frontier(), closure_from(infos.map(), cited)),
        write_admitted(slot_frontier(*old(dst_tok)), closure_from(infos.map(), cited)),
    ensures
        final(source).well_formed(),
        final(source).node() == old(source).node(),
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        final(src_perm).advanced_read(*old(src_perm), old(source).next_event()),
        slot_owned(
            *final(dst_tok),
            instance.id(),
            old(dst_tok).key(),
            Frontier { last_writer: Some(old(source).next_event()), readers: ISet::empty() },
        ),
        result.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access,
            completed_access.effect,
            closure_from(infos.map(), cited),
            old(src_perm).read_guards(),
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked submitted = instance.submit_2(
        source.node(),
        completed_access.access,
        completed_access.effect,
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        src_perm,
        dst_tok,
    );
    let tracked knowledge = observe_single(instance, source.node(), submitted);
    Completion { submitted, knows: knowledge }
}

pub(crate) proof fn finish_slab_owned(
    tracked source: &mut EventSourceToken,
    tracked instance: &TransferInstance,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    tracked completed_access: CompletedAccess,
    tracked tokens: Map<usize, SlotToken>,
    tracked infos: &SubmittedMap,
    tracked knows: &Knows,
    cited: ISet<TransferEvent>,
) -> (tracked result: (Completion, Map<usize, SlotToken>))
    requires
        old(source).well_formed(),
        old(source).node() == node,
        old(source).id() == instance.id(),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(infos.dom()),
        knowledge_of(node, cited).subset_of(knows.set()),
        completed_access.access.writes == slab_locs(node, space, count as nat),
        completed_access.access.reads.subset_of(slab_locs(node, space, count as nat)),
        slab_tokens_valid(tokens, instance.id(), node, space, count),
        forall|i: usize|
            i < count ==> write_admitted(
                slot_frontier(#[trigger] tokens[i]),
                closure_from(infos.map(), cited),
            ),
    ensures
        final(source).well_formed(),
        final(source).node() == node,
        final(source).source() == old(source).source(),
        final(source).next_seq() == old(source).next_seq() + 1,
        final(source).id() == instance.id(),
        result.0.records(
            instance.id(),
            old(source).next_event(),
            completed_access.access,
            completed_access.effect,
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
        slab_tokens_owned(
            result.1,
            instance.id(),
            node,
            space,
            count,
            Frontier { last_writer: Some(old(source).next_event()), readers: ISet::empty() },
        ),
{
    let tracked name = source.take_next();
    assert(ISet::empty().union(cited) =~= cited);
    let tracked (Tracked(submitted), Tracked(out_tokens)) = instance.submit_slab_owned(
        node,
        node,
        space,
        count,
        completed_access.access,
        completed_access.effect,
        ISet::empty(),
        cited,
        infos,
        knows,
        name,
        tokens,
    );
    let tracked knowledge = observe_single(instance, node, submitted);
    (Completion { submitted, knows: knowledge }, out_tokens)
}

pub open spec fn can_reclaim_slot(
    id: InstanceId,
    loc: SlotLoc,
    tok: SlotToken,
    infos: SubmittedMap,
    knows: Knows,
    cited: ISet<TransferEvent>,
) -> bool {
    &&& slot_is_owned(tok, id, loc)
    &&& infos.instance_id() == id
    &&& knows.instance_id() == id
    &&& cited.subset_of(infos.dom())
    &&& knowledge_of(loc.node, cited).subset_of(knows.set())
    &&& write_admitted(slot_frontier(tok), closure_from(infos.map(), cited))
}

pub open spec fn can_reclaim_slab(
    id: InstanceId,
    node: NodeId,
    space: MemorySpace,
    count: usize,
    tokens: Map<usize, SlotToken>,
    infos: SubmittedMap,
    knows: Knows,
    cited: ISet<TransferEvent>,
) -> bool {
    &&& infos.instance_id() == id
    &&& knows.instance_id() == id
    &&& cited.subset_of(infos.dom())
    &&& knowledge_of(node, cited).subset_of(knows.set())
    &&& slab_tokens_valid(tokens, id, node, space, count)
    &&& forall|i: usize|
        i < count ==> write_admitted(
            slot_frontier(#[trigger] tokens[i]),
            closure_from(infos.map(), cited),
        )
}

/// A fresh allocation is written and then read through the same borrowed view.
/// In a valid hardware run the write is the only predecessor of the read.
/// Consequently the observed byte sequence equals the input sequence with zero quantifiers.
#[allow(dead_code)]
fn host_round_trip(node: NodeContext, data: &[u8], out: &mut [u8]) -> (allocated: bool)
    requires
        node.initial_partition(),
        data@.len() == old(out)@.len(),
{
    let (Tracked(instance), Tracked(mut allocations), Tracked(mut sources), _) =
        node.into_sources();
    let tracked source = sources.mint(EventSourceKind::Host);
    let mut ctx = HostContext::new(Tracked(source));
    let Some((buf, Tracked(mut tok))) = alloc(
        &ctx,
        Tracked(&instance),
        Tracked(&mut allocations),
        data.len(),
    ) else { return false };
    let tracked empty_infos = SubmittedMap::empty(instance.id());
    let tracked empty_knows = Knows::empty(instance.id());
    let slice = buf.as_slice();
    let ghost sref = slice.slot_ref();
    let ghost loc = sref.loc;

    let ghost writer = ctx.source@.next_event();
    proof {
        assert(knowledge_of(ctx.source@.node(), ISet::<TransferEvent>::empty()) =~= ISet::empty());
    }
    let Tracked(written) = write_slice(
        &mut ctx,
        Tracked(&instance),
        &slice,
        data,
        Tracked(&mut tok),
        Tracked(&empty_infos),
        Tracked(&empty_knows),
        Ghost(ISet::empty()),
    );
    let tracked writer_infos = written.infos();
    let ghost writer_only = ISet::empty().insert(writer);
    let ghost reader = ctx.source@.next_event();
    let tracked mut read_perm = ReadPerm::Owned(tok);
    proof {
        assert(written.submitted.value().preds =~= ISet::empty());
    }
    let Tracked(observed) = read_slice(
        &mut ctx,
        Tracked(&instance),
        &slice,
        out,
        Tracked(&mut read_perm),
        Tracked(&writer_infos),
        Tracked(&written.knows),
        Ghost(writer_only),
    );
    let tracked tok = read_perm.into_owned();
    let tracked reader_infos = observed.infos();
    let ghost reader_only = ISet::empty().insert(reader);
    proof {
        assert(observed.submitted.value().preds =~= writer_only);
        let initial = history::initial::<TransferDomain>();
        let empty = ISet::empty();
        let w = written.submitted.value();
        let r = observed.submitted.value();
        let after_write = history::submit(
            initial,
            writer,
            w.access,
            w.effect,
            IMap::empty(),
            empty,
            empty,
        );
        let completed = history::observe(after_write, loc.node, writer_only);
        let h = history::submit(
            completed,
            reader,
            r.access,
            r.effect,
            IMap::empty(),
            empty,
            writer_only,
        );
        history::initial_is_reachable::<TransferDomain>();
        history::step_keeps_reachable(
            initial,
            Action::Submit {
                observer: loc.node,
                new_event: writer,
                access: w.access,
                effect: w.effect,
                read_guards: IMap::empty(),
                after: empty,
                cited: empty,
            },
            after_write,
        );
        history::step_keeps_reachable(
            after_write,
            Action::Observe { observer: loc.node, completed: writer_only },
            completed,
        );
        history::step_keeps_reachable(
            completed,
            Action::Submit {
                observer: loc.node,
                new_event: reader,
                access: r.access,
                effect: r.effect,
                read_guards: IMap::empty(),
                after: empty,
                cited: writer_only,
            },
            h,
        );
        assert forall|run: Run<TransferDomain>| #[trigger]
            valid_run(h, run) && (run.succeeded)(writer) && (run.succeeded)(reader) && (run.end)(
                writer,
            ) is Some && (run.end)(reader) is Some implies out@ == data@ by {
            every_event_isolated(h, run, writer);
            every_event_isolated(h, run, reader);
            read_returns_last_write(h, run, reader, loc);
            assert(writers_before(h, reader, loc) =~= writer_only);
            assert(writers_before(h, reader, loc).contains(writer));
            let w_in = restrict::<TransferDomain>(
                (run.memory)((run.start)(writer)),
                w.access.reads,
            );
            let w_out = restrict::<TransferDomain>(
                (run.memory)((run.end)(writer).unwrap()),
                w.access.writes,
            );
            let r_in = restrict::<TransferDomain>(
                (run.memory)((run.start)(reader)),
                r.access.reads,
            );
            let r_out = restrict::<TransferDomain>(
                (run.memory)((run.end)(reader).unwrap()),
                r.access.writes,
            );
            splice_bytes_full(sref, data@, w_in, w_out);
            assert(r_in[loc] == w_out[loc]);
            read_bytes_full(sref, out@, r_in, r_out);
        }
        assert forall|e: TransferEvent| #[trigger] writer_only.contains(e) implies closure_from(
            reader_infos.map(),
            reader_only,
        ).contains(e) by {
            assert(reader_infos.map()[reader].preds.contains(e));
        }
        assert(closure_from(reader_infos.map(), reader_only) =~= writer_only.insert(reader));
        assert(knowledge_of(loc.node, reader_only).subset_of(observed.knows.set()));
    }
    free(
        &mut ctx,
        Tracked(&instance),
        buf,
        Tracked(tok),
        Tracked(&reader_infos),
        Tracked(&observed.knows),
        Ghost(reader_only),
    );
    true
}

} // verus!
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_round_trip_bytes() {
        let data = [0, 127, 255, 1];
        let mut out = [42; 4];
        assert!(host_round_trip(NodeContext::bootstrap(0), &data, &mut out));
        assert_eq!(out, data);
    }

    #[test]
    #[should_panic]
    fn host_slab_slot_slice_oob_panics() {
        let mut ctx = HostContext::new(Tracked::assume_new());
        let (buf, _) =
            alloc_slab(&ctx, Tracked::assume_new(), Tracked::assume_new(), 2, 64, 1).unwrap();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = buf.slot_slice(2);
        }));
        free_slab(
            &mut ctx,
            Tracked::assume_new(),
            buf,
            Tracked::assume_new(),
            Tracked::assume_new(),
            Tracked::assume_new(),
            Ghost::assume_new(),
        );
        if let Err(payload) = res {
            std::panic::resume_unwind(payload);
        }
    }

    #[test]
    fn host_slab_alloc_overflow_returns_none() {
        let ctx = HostContext::new(Tracked::assume_new());
        assert!(alloc_slab(
            &ctx,
            Tracked::assume_new(),
            Tracked::assume_new(),
            2,
            usize::MAX,
            1
        )
        .is_none());
        assert!(alloc_slab(&ctx, Tracked::assume_new(), Tracked::assume_new(), 0, 64, 1).is_none());
        assert!(alloc_aligned(&ctx, Tracked::assume_new(), Tracked::assume_new(), 64, 3).is_none());
    }
}
