use super::*;
#[cfg(verus_only)]
use foundation::{
    history::{self, Action},
    space::restrict,
    value::*,
};

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

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
    let tracked source = sources.mint(EventSourceKind::host());
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
    #[cfg(verus_only)]
    use foundation::{
        history::{self, Action},
        space::restrict,
        value::*,
    };

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
