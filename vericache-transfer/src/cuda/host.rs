//! CUDA page-locked host allocations and host-buffer registrations.
use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub struct CudaHostBuffer {
    pub(super) memory: SlabMemory,
}

pub struct CudaHostRegistration<'a> {
    pub(super) buffer: &'a HostBuffer,
    pub(super) guard: ffi::RawHostRegGuard,
}

impl CudaHostBuffer {
    #[allow(dead_code)]
    pub(crate) fn raw_parts(&self) -> (p: (*mut u8, usize))
        requires
            self.well_formed(),
        ensures
            p.1 == self.total_len(),
            self.num_slots() >= 1,
            p.1 == self.num_slots() * self.slot_len(),
            p.0.addr() + p.1 <= usize::MAX,
    {
        (self.memory.ptr, self.memory.len)
    }

    pub closed spec fn node(&self) -> NodeId {
        self.memory.node@
    }

    pub closed spec fn space(&self) -> MemorySpace {
        self.memory.space@
    }

    pub closed spec fn num_slots(&self) -> usize {
        self.memory.num_slots
    }

    pub closed spec fn slot_len(&self) -> usize {
        self.memory.slot_len
    }

    pub fn num_slots_val(&self) -> (r: usize)
        requires
            self.well_formed(),
        ensures
            r == self.num_slots(),
    {
        self.memory.num_slots
    }

    pub fn slot_len_val(&self) -> (r: usize)
        requires
            self.well_formed(),
        ensures
            r == self.slot_len(),
    {
        self.memory.slot_len
    }

    pub closed spec fn total_len(&self) -> usize {
        self.memory.len
    }

    pub open spec fn slot_loc(&self, idx: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space(), slot_index: idx }
    }

    pub open spec fn slot_ref(&self, idx: nat) -> SlotRef {
        SlotRef::full(self.slot_loc(idx), self.slot_len() as u64)
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.memory.instance@
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.memory.well_formed() && self.memory.kind is PinnedHost
    }

    pub fn as_cuda_slice(&self) -> (s: CudaSlice<'_>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(0),
            s.instance_id() == self.instance_id(),
            s.kind() == CudaPointerKind::PinnedHost,
    {
        self.memory.borrow_slot(0)
    }

    pub fn slot_slice(&self, idx: usize) -> (s: CudaSlice<'_>)
        requires
            self.well_formed(),
            idx < self.num_slots(),
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(idx as nat),
            s.instance_id() == self.instance_id(),
            s.kind() == CudaPointerKind::PinnedHost,
    {
        assert!(idx < self.memory.num_slots);
        self.memory.borrow_slot(idx)
    }

    pub fn as_host_slice(&self) -> (s: HostSlice<'_>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(0),
            s.instance_id() == self.instance_id(),
    {
        self.host_slot_slice(0)
    }

    pub fn host_slot_slice(&self, idx: usize) -> (s: HostSlice<'_>)
        requires
            self.well_formed(),
            idx < self.num_slots(),
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(idx as nat),
            s.instance_id() == self.instance_id(),
    {
        assert!(idx < self.memory.num_slots);
        let m = &self.memory;
        proof {
            lemma_mul_is_distributive_add_other_way(m.slot_len as int, idx as int, 1);
            lemma_mul_inequality((idx + 1) as int, m.num_slots as int, m.slot_len as int);
        }
        let off = idx * m.slot_len;
        let ptr = m.ptr.with_addr(m.ptr.addr() + off);
        HostSlice::from_raw_parts(
            self,
            ptr,
            m.slot_len,
            Ghost(self.slot_ref(idx as nat)),
            m.instance,
        )
    }
}

pub fn alloc_host(Tracked(source): Tracked<&mut AllocSource>, len: usize) -> (r: Result<
    (CudaHostBuffer, Tracked<SlotToken>),
    CudaError,
>)
    requires
        old(source).well_formed(),
    ensures
        match r {
            Err(_) => *final(source) == *old(source),
            Ok((buf, tok)) => {
                &&& buf.well_formed() && buf.instance_id() == old(source).id()
                &&& buf.num_slots() == 1 && buf.slot_len() == len
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(None)
                &&& buf.slot_ref(0) == old(source).slot_ref(None, len as u64)
                &&& final(source).allocated_slot(*old(source), buf.slot_loc(0), tok@)
            },
        },
{
    let ptr = ffi::raw_alloc_host(len)?;
    let ghost node = source.node();
    let ghost space = source.space(None);
    let tracked tok = source.allocate_slot(None);
    Ok(
        (
            CudaHostBuffer {
                memory: SlabMemory {
                    ptr,
                    num_slots: 1,
                    slot_len: len,
                    len,
                    kind: CudaPointerKind::PinnedHost,
                    node: Ghost(node),
                    space: Ghost(space),
                    instance: Ghost(source.id()),
                },
            },
            Tracked(tok),
        ),
    )
}

pub fn alloc_host_slab(
    Tracked(source): Tracked<&mut AllocSource>,
    num_slots: usize,
    slot_len: usize,
) -> (r: Result<(CudaHostBuffer, Tracked<Map<usize, SlotToken>>), CudaError>)
    requires
        num_slots >= 1,
        num_slots * slot_len <= usize::MAX,
        old(source).well_formed(),
    ensures
        match r {
            Err(_) => *final(source) == *old(source),
            Ok((buf, tokens)) => {
                &&& buf.well_formed() && buf.instance_id() == old(source).id()
                &&& buf.num_slots() == num_slots && buf.slot_len() == slot_len
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(None)
                &&& final(source).allocated_slab(*old(source), buf.space(), num_slots, tokens@)
            },
        },
{
    if num_slots == 0 {
        return Err(CudaError::INVALID_VALUE);
    }
    let Some(total) = num_slots.checked_mul(slot_len) else {
        return Err(CudaError::INVALID_VALUE);
    };
    let ptr = ffi::raw_alloc_host(total)?;
    let ghost node = source.node();
    let ghost space = source.space(None);
    let tracked tokens = source.allocate_slab_slots(None, num_slots);
    Ok(
        (
            CudaHostBuffer {
                memory: SlabMemory {
                    ptr,
                    num_slots,
                    slot_len,
                    len: total,
                    kind: CudaPointerKind::PinnedHost,
                    node: Ghost(node),
                    space: Ghost(space),
                    instance: Ghost(source.id()),
                },
            },
            Tracked(tokens),
        ),
    )
}

impl<'a> CudaHostRegistration<'a> {
    pub closed spec fn node(&self) -> NodeId {
        self.buffer.node()
    }

    pub closed spec fn space(&self) -> MemorySpace {
        self.buffer.space()
    }

    pub closed spec fn num_slots(&self) -> usize {
        self.buffer.num_slots()
    }

    pub closed spec fn slot_len(&self) -> usize {
        self.buffer.slot_len()
    }

    pub open spec fn slot_loc(&self, idx: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space(), slot_index: idx }
    }

    pub open spec fn slot_ref(&self, idx: nat) -> SlotRef {
        SlotRef::full(self.slot_loc(idx), self.slot_len() as u64)
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.buffer.instance_id()
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.buffer.well_formed()
    }

    pub fn as_cuda_slice(&self) -> (s: CudaSlice<'_>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(0),
            s.instance_id() == self.instance_id(),
            s.kind() == CudaPointerKind::PinnedHost,
    {
        self.slot_slice(0)
    }

    pub fn slot_slice(&self, idx: usize) -> (s: CudaSlice<'_>)
        requires
            self.well_formed(),
            idx < self.num_slots(),
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(idx as nat),
            s.instance_id() == self.instance_id(),
            s.kind() == CudaPointerKind::PinnedHost,
    {
        assert!(idx < self.buffer.num_slots_val());
        let h = self.buffer.slot_slice(idx);
        let (ptr, len) = h.raw_parts();
        CudaSlice {
            memory: SliceMemory {
                ptr,
                len,
                kind: CudaPointerKind::PinnedHost,
                slot: Ghost(self.slot_ref(idx as nat)),
                instance: Ghost(self.instance_id()),
            },
            borrow: PhantomData,
        }
    }
}

pub fn host_register(buffer: &HostBuffer) -> (r: Result<CudaHostRegistration<'_>, CudaError>)
    requires
        buffer.well_formed(),
    ensures
        r matches Ok(reg) ==> reg.well_formed() && reg.node() == buffer.node() && reg.space()
            == buffer.space() && reg.num_slots() == buffer.num_slots() && reg.slot_len()
            == buffer.slot_len() && reg.instance_id() == buffer.instance_id(),
{
    let guard = ffi::raw_host_register(buffer)?;
    Ok(CudaHostRegistration { buffer, guard })
}

pub fn free_host(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: CudaHostBuffer,
    Tracked(tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<Tracked<Completion>, (CudaError, CudaHostBuffer, Tracked<SlotToken>)>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.num_slots() == 1,
        buf.instance_id() == instance.id(),
        cpu_host::can_reclaim_slot(instance.id(), buf.slot_loc(0), tok, *infos, *knows, cited),
    ensures
        match r {
            Err((_, returned, ret_tok)) => returned == buf && ret_tok@ == tok && *final(_ctx)
                == *old(_ctx),
            Ok(completion) => {
                &&& final(_ctx).stepped(*old(_ctx))
                &&& completion@.records(
                    instance.id(),
                    old(_ctx).source@.next_event(),
                    reclaim_access(buf.slot_loc(0)),
                    havoc_effect(),
                    closure_from(infos.map(), cited),
                    IMap::empty(),
                )
            },
        },
{
    let ghost loc = buf.slot_loc(0);
    let tracked mut tok = tok;
    let Tracked(completed) = match ffi::raw_free_host(&buf) {
        Err(e) => return Err((e, buf, Tracked(tok))),
        Ok(w) => w,
    };
    proof {
        assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
        assert(completed.access == reclaim_access(loc));
    }
    let tracked completion = cpu_host::finish_owned(
        _ctx.source.borrow_mut(),
        instance,
        completed,
        &mut tok,
        infos,
        knows,
        cited,
    );
    Ok(Tracked(completion))
}

pub fn free_host_slab(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: CudaHostBuffer,
    Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<Tracked<Completion>, (CudaError, CudaHostBuffer, Tracked<Map<usize, SlotToken>>)>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.instance_id() == instance.id(),
        cpu_host::can_reclaim_slab(
            instance.id(),
            buf.node(),
            buf.space(),
            buf.num_slots(),
            tokens,
            *infos,
            *knows,
            cited,
        ),
    ensures
        match r {
            Err((_, returned, ret_toks)) => returned == buf && ret_toks@ == tokens && *final(_ctx)
                == *old(_ctx),
            Ok(completion) => {
                &&& final(_ctx).stepped(*old(_ctx))
                &&& completion@.records(
                    instance.id(),
                    old(_ctx).source@.next_event(),
                    reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
                    havoc_effect(),
                    closure_from(infos.map(), cited),
                    IMap::empty(),
                )
            },
        },
{
    let ghost node = buf.node();
    let ghost space = buf.space();
    let ghost count = buf.num_slots();
    let Tracked(completed) = match ffi::raw_free_host(&buf) {
        Err(e) => return Err((e, buf, Tracked(tokens))),
        Ok(w) => w,
    };
    let tracked (completion, _) = cpu_host::finish_slab_owned(
        _ctx.source.borrow_mut(),
        instance,
        node,
        space,
        count,
        completed,
        tokens,
        infos,
        knows,
        cited,
    );
    Ok(Tracked(completion))
}

pub fn host_unregister<'a>(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: CudaHostRegistration<'a>,
    Tracked(tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Tracked<Completion>, Tracked<SlotToken>),
    (CudaError, CudaHostRegistration<'a>, Tracked<SlotToken>),
>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.num_slots() == 1,
        buf.instance_id() == instance.id(),
        cpu_host::can_reclaim_slot(instance.id(), buf.slot_loc(0), tok, *infos, *knows, cited),
    ensures
        match r {
            Err((_, returned, ret_tok)) => returned == buf && ret_tok@ == tok && *final(_ctx)
                == *old(_ctx),
            Ok((completion, out_tok)) => {
                &&& final(_ctx).stepped(*old(_ctx))
                &&& completion@.records(
                    instance.id(),
                    old(_ctx).source@.next_event(),
                    dereg_access(buf.slot_loc(0)),
                    identity_effect(buf.slot_loc(0)),
                    closure_from(infos.map(), cited),
                    IMap::empty(),
                )
                &&& slot_owned(
                    out_tok@,
                    instance.id(),
                    buf.slot_loc(0),
                    Frontier {
                        last_writer: Some(old(_ctx).source@.next_event()),
                        readers: ISet::empty(),
                    },
                )
            },
        },
{
    let ghost loc = buf.slot_loc(0);
    let tracked mut tok = tok;
    let Tracked(completed) = match ffi::raw_host_unregister(&buf) {
        Err(e) => return Err((e, buf, Tracked(tok))),
        Ok(w) => w,
    };
    let _ = buf.guard.into_raw();
    proof {
        assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
        assert(completed.access == dereg_access(loc));
    }
    let tracked completion = cpu_host::finish_owned(
        _ctx.source.borrow_mut(),
        instance,
        completed,
        &mut tok,
        infos,
        knows,
        cited,
    );
    Ok((Tracked(completion), Tracked(tok)))
}

pub fn host_unregister_slab<'a>(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: CudaHostRegistration<'a>,
    Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Tracked<Completion>, Tracked<Map<usize, SlotToken>>),
    (CudaError, CudaHostRegistration<'a>, Tracked<Map<usize, SlotToken>>),
>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.instance_id() == instance.id(),
        cpu_host::can_reclaim_slab(
            instance.id(),
            buf.node(),
            buf.space(),
            buf.num_slots(),
            tokens,
            *infos,
            *knows,
            cited,
        ),
    ensures
        match r {
            Err((_, returned, ret_toks)) => returned == buf && ret_toks@ == tokens && *final(_ctx)
                == *old(_ctx),
            Ok((completion, out_toks)) => {
                &&& final(_ctx).stepped(*old(_ctx))
                &&& completion@.records(
                    instance.id(),
                    old(_ctx).source@.next_event(),
                    dereg_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
                    if buf.num_slots() == 1 {
                        identity_effect(buf.slot_loc(0))
                    } else {
                        identity_effect_slab()
                    },
                    closure_from(infos.map(), cited),
                    IMap::empty(),
                )
                &&& slab_tokens_owned(
                    out_toks@,
                    instance.id(),
                    buf.node(),
                    buf.space(),
                    buf.num_slots(),
                    Frontier {
                        last_writer: Some(old(_ctx).source@.next_event()),
                        readers: ISet::empty(),
                    },
                )
            },
        },
{
    let ghost node = buf.node();
    let ghost space = buf.space();
    let ghost count = buf.num_slots();
    let Tracked(completed) = match ffi::raw_host_unregister(&buf) {
        Err(e) => return Err((e, buf, Tracked(tokens))),
        Ok(w) => w,
    };
    let _ = buf.guard.into_raw();
    let tracked (completion, out_tokens) = cpu_host::finish_slab_owned(
        _ctx.source.borrow_mut(),
        instance,
        node,
        space,
        count,
        completed,
        tokens,
        infos,
        knows,
        cited,
    );
    Ok((Tracked(completion), Tracked(out_tokens)))
}

} // verus!
