//! Synchronous operations on host allocations (standalone buffers and slabs).
mod ffi;
mod proofs;

use crate::context::*;
use crate::domain::*;
use core::marker::PhantomData;
use foundation::space::{Access, Effect};
#[cfg(verus_only)]
use foundation::{
    history::{self, Action},
    space::restrict,
    tracked::*,
    value::*,
};
#[cfg(verus_only)]
use vstd::arithmetic::mul::{lemma_mul_inequality, lemma_mul_is_distributive_add_other_way};
use vstd::prelude::*;
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

pub(crate) use proofs::CompletedAccess;
pub use proofs::Completion;
#[cfg(verus_only)]
pub use proofs::{can_reclaim_slab, can_reclaim_slot};
#[cfg(verus_only)]
pub(crate) use proofs::{finish_copy, finish_owned, finish_read, finish_slab_owned};

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub struct HostContext {
    pub source: Tracked<EventSourceToken>,
}

impl HostContext {
    pub fn new(Tracked(source): Tracked<EventSourceToken>) -> (result: Self)
        requires
            source.well_formed(),
            source.source() is Host,
        ensures
            result.source@ == source,
    {
        Self { source: Tracked(source) }
    }

    pub open spec fn ready(&self, instance: InstanceId, node: NodeId) -> bool {
        self.source@.well_formed() && self.source@.node() == node && self.source@.source() is Host
            && self.source@.id() == instance
    }

    pub open spec fn stepped(&self, previous: Self) -> bool {
        &&& self.ready(previous.source@.id(), previous.source@.node())
        &&& self.source@.source() == previous.source@.source()
        &&& self.source@.next_seq() == previous.source@.next_seq() + 1
    }
}

pub open spec fn is_power_of_two_usize(align: usize) -> bool {
    align > 0 && (align & ((align - 1) as usize)) == 0
}

pub struct HostBuffer {
    align: usize,
    ptr: *mut u8,
    num_slots: usize,
    slot_len: usize,
    len: usize,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    node: Ghost<NodeId>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    space: Ghost<MemorySpace>,
    instance: Ghost<InstanceId>,
}

pub struct HostSlice<'a> {
    ptr: *mut u8,
    len: usize,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    slot: Ghost<SlotRef>,
    instance: Ghost<InstanceId>,
    borrow: PhantomData<&'a HostBuffer>,
}

impl HostBuffer {
    pub closed spec fn address(&self) -> usize {
        self.ptr.addr()
    }

    pub closed spec fn align(&self) -> usize {
        self.align
    }

    pub closed spec fn node(&self) -> NodeId {
        self.node@
    }

    pub closed spec fn space(&self) -> MemorySpace {
        self.space@
    }

    pub closed spec fn num_slots(&self) -> usize {
        self.num_slots
    }

    pub closed spec fn slot_len(&self) -> usize {
        self.slot_len
    }

    pub closed spec fn total_len(&self) -> usize {
        self.len
    }

    pub open spec fn slot_loc(&self, idx: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space(), slot_index: idx }
    }

    pub open spec fn slot_ref(&self, idx: nat) -> SlotRef {
        SlotRef::full(self.slot_loc(idx), self.slot_len() as u64)
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.instance@
    }

    pub closed spec fn well_formed(&self) -> bool {
        &&& self.space@ is HostAlloc
        &&& self.num_slots >= 1
        &&& self.len == self.num_slots * self.slot_len
        &&& is_power_of_two_usize(self.align)
        &&& self.ptr.addr() % self.align == 0
        &&& self.ptr.addr() + self.len <= usize::MAX
    }

    pub(crate) fn raw_parts(&self) -> (parts: (*mut u8, usize))
        requires
            self.well_formed(),
        ensures
            parts.0.addr() == self.address(),
            parts.1 == self.total_len(),
            self.num_slots() >= 1,
            parts.1 == self.num_slots() * self.slot_len(),
            parts.0.addr() + parts.1 <= usize::MAX,
            self.space() is HostAlloc,
    {
        (self.ptr, self.len)
    }

    pub fn len(&self) -> (result: usize)
        requires
            self.well_formed(),
        ensures
            result == self.total_len(),
            result == self.num_slots() * self.slot_len(),
    {
        self.len
    }

    pub fn num_slots_val(&self) -> (result: usize)
        requires
            self.well_formed(),
        ensures
            result == self.num_slots(),
    {
        self.num_slots
    }

    pub fn slot_len_val(&self) -> (result: usize)
        requires
            self.well_formed(),
        ensures
            result == self.slot_len(),
    {
        self.slot_len
    }

    pub fn as_slice(&self) -> (result: HostSlice<'_>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            result.well_formed(),
            result.slot_ref() == self.slot_ref(0),
            result.address() == self.address(),
            result.instance_id() == self.instance_id(),
    {
        self.slot_slice(0)
    }

    pub fn slot_slice(&self, idx: usize) -> (result: HostSlice<'_>)
        requires
            self.well_formed(),
            idx < self.num_slots(),
        ensures
            result.well_formed(),
            result.slot_ref() == self.slot_ref(idx as nat),
            result.address() == self.address() + idx * self.slot_len(),
            result.instance_id() == self.instance_id(),
    {
        assert!(idx < self.num_slots);
        proof {
            lemma_mul_is_distributive_add_other_way(self.slot_len as int, idx as int, 1);
            lemma_mul_inequality((idx + 1) as int, self.num_slots as int, self.slot_len as int);
        }
        let off = idx * self.slot_len;
        let ptr = self.ptr.with_addr(self.ptr.addr() + off);
        HostSlice::from_raw_parts(
            self,
            ptr,
            self.slot_len,
            Ghost(self.slot_ref(idx as nat)),
            self.instance,
        )
    }
}

impl<'a> HostSlice<'a> {
    pub closed spec fn address(&self) -> usize {
        self.ptr.addr()
    }

    pub fn addr(&self) -> (r: usize)
        ensures
            r == self.address(),
    {
        self.ptr.addr()
    }

    pub(crate) fn raw_parts(&self) -> (r: (*mut u8, usize))
        requires
            self.well_formed(),
        ensures
            r.0.addr() == self.address(),
            r.1 == self.slot_ref().sub_len,
            r.0.addr() + r.1 <= usize::MAX,
            self.slot_ref().loc.space is HostAlloc,
    {
        (self.ptr, self.len)
    }

    pub(crate) fn from_raw_parts<T>(
        _owner: &'a T,
        ptr: *mut u8,
        len: usize,
        slot: Ghost<SlotRef>,
        instance: Ghost<InstanceId>,
    ) -> (slice: Self)
        requires
            slot@.well_formed(),
            slot@.loc.space is HostAlloc,
            slot@.sub_len == len,
            ptr.addr() + len <= usize::MAX,
        ensures
            slice.well_formed(),
            slice.slot_ref() == slot@,
            slice.address() == ptr.addr(),
            slice.instance_id() == instance@,
    {
        Self { ptr, len, slot, instance, borrow: PhantomData }
    }

    pub closed spec fn slot_ref(&self) -> SlotRef {
        self.slot@
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.instance@
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.slot@.well_formed() && self.slot@.loc.space is HostAlloc && self.slot@.sub_len
            == self.len && self.ptr.addr() + self.len <= usize::MAX
    }

    pub fn len(&self) -> (result: usize)
        requires
            self.well_formed(),
        ensures
            result == self.slot_ref().sub_len,
    {
        self.len
    }

    pub fn sub_slice(&self, offset: usize, len: usize) -> (result: HostSlice<'a>)
        requires
            self.well_formed(),
            offset + len <= self.slot_ref().sub_len,
        ensures
            result.well_formed(),
            result.instance_id() == self.instance_id(),
            result.address() == self.address() + offset,
            result.slot_ref() == self.slot_ref().sub_slice(offset as u64, len as u64),
    {
        assert!(offset <= self.len && len <= self.len - offset);
        let ptr = self.ptr.with_addr(self.ptr.addr() + offset);
        HostSlice {
            ptr,
            len,
            slot: Ghost(self.slot@.sub_slice(offset as u64, len as u64)),
            instance: self.instance,
            borrow: PhantomData,
        }
    }
}

/// A failed allocation leaves the identifier supply and all its tokens intact.
pub fn alloc(
    _ctx: &HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(source): Tracked<&mut AllocSource>,
    len: usize,
) -> (result: Option<(HostBuffer, Tracked<SlotToken>)>)
    requires
        old(source).well_formed(),
        _ctx.ready(instance.id(), old(source).node()),
        old(source).id() == instance.id(),
    ensures
        match result {
            None => *final(source) == *old(source),
            Some((buf, tok)) => {
                &&& buf.well_formed() && buf.instance_id() == instance.id()
                &&& buf.num_slots() == 1 && buf.slot_len() == len
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(None)
                &&& buf.slot_ref(0) == old(source).slot_ref(None, len as u64)
                &&& final(source).allocated_slot(*old(source), buf.slot_loc(0), tok@)
            },
        },
{
    assert((1usize & 0usize) == 0usize) by (bit_vector);
    alloc_aligned(_ctx, Tracked(instance), Tracked(source), len, 1)
}

pub fn alloc_aligned(
    _ctx: &HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(source): Tracked<&mut AllocSource>,
    len: usize,
    align: usize,
) -> (result: Option<(HostBuffer, Tracked<SlotToken>)>)
    requires
        is_power_of_two_usize(align),
        old(source).well_formed(),
        _ctx.ready(instance.id(), old(source).node()),
        old(source).id() == instance.id(),
    ensures
        match result {
            None => *final(source) == *old(source),
            Some((buf, tok)) => {
                &&& buf.align() == align && buf.address() % align == 0
                &&& buf.well_formed() && buf.instance_id() == instance.id()
                &&& buf.num_slots() == 1 && buf.slot_len() == len
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(None)
                &&& buf.slot_ref(0) == old(source).slot_ref(None, len as u64)
                &&& final(source).allocated_slot(*old(source), buf.slot_loc(0), tok@)
            },
        },
{
    if align == 0 || (align & (align - 1)) != 0 {
        return None;
    }
    let ptr = ffi::raw_alloc_aligned(len, align)?;
    let ghost node = source.node();
    let ghost space = source.space(None);
    let tracked tok = source.allocate_slot(None);
    Some(
        (
            HostBuffer {
                ptr,
                num_slots: 1,
                slot_len: len,
                len,
                align,
                node: Ghost(node),
                space: Ghost(space),
                instance: Ghost(instance.id()),
            },
            Tracked(tok),
        ),
    )
}

pub fn alloc_slab(
    _ctx: &HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(source): Tracked<&mut AllocSource>,
    num_slots: usize,
    slot_len: usize,
    align: usize,
) -> (result: Option<(HostBuffer, Tracked<Map<usize, SlotToken>>)>)
    requires
        num_slots >= 1,
        num_slots * slot_len <= usize::MAX,
        is_power_of_two_usize(align),
        old(source).well_formed(),
        _ctx.ready(instance.id(), old(source).node()),
        old(source).id() == instance.id(),
    ensures
        match result {
            None => *final(source) == *old(source),
            Some((buf, tokens)) => {
                &&& buf.align() == align && buf.address() % align == 0
                &&& buf.well_formed() && buf.instance_id() == instance.id()
                &&& buf.num_slots() == num_slots && buf.slot_len() == slot_len
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(None)
                &&& final(source).allocated_slab(*old(source), buf.space(), num_slots, tokens@)
            },
        },
{
    if num_slots == 0 || align == 0 || (align & (align - 1)) != 0 {
        return None;
    }
    let Some(total) = num_slots.checked_mul(slot_len) else {
        return None;
    };
    let ptr = ffi::raw_alloc_aligned(total, align)?;
    let ghost node = source.node();
    let ghost space = source.space(None);
    let tracked tokens = source.allocate_slab_slots(None, num_slots);
    Some(
        (
            HostBuffer {
                ptr,
                num_slots,
                slot_len,
                len: total,
                align,
                node: Ghost(node),
                space: Ghost(space),
                instance: Ghost(instance.id()),
            },
            Tracked(tokens),
        ),
    )
}

pub fn write_slice(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    dst: &HostSlice,
    data: &[u8],
    Tracked(dst_tok): Tracked<&mut SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: Tracked<Completion>)
    requires
        old(_ctx).ready(instance.id(), dst.slot_ref().loc.node),
        dst.well_formed(),
        dst.instance_id() == instance.id(),
        data@.len() == dst.slot_ref().sub_len,
        slot_is_owned(*old(dst_tok), instance.id(), dst.slot_ref().loc),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(infos.dom()),
        knowledge_of(dst.slot_ref().loc.node, cited).subset_of(knows.set()),
        write_admitted(slot_frontier(*old(dst_tok)), closure_from(infos.map(), cited)),
    ensures
        final(_ctx).stepped(*old(_ctx)),
        slot_owned(
            *final(dst_tok),
            instance.id(),
            dst.slot_ref().loc,
            Frontier { last_writer: Some(old(_ctx).source@.next_event()), readers: ISet::empty() },
        ),
        result@.records(
            instance.id(),
            old(_ctx).source@.next_event(),
            write_access(dst.slot_ref()),
            splice_bytes(dst.slot_ref(), data@),
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
{
    assert!(dst.len == data.len());
    let Tracked(completed) = ffi::raw_write(dst, data);
    let tracked result = finish_owned(
        _ctx.source.borrow_mut(),
        instance,
        completed,
        dst_tok,
        infos,
        knows,
        cited,
    );
    Tracked(result)
}

pub fn read_slice(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    src: &HostSlice,
    out: &mut [u8],
    Tracked(src_perm): Tracked<&mut ReadPerm>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: Tracked<Completion>)
    requires
        old(_ctx).ready(instance.id(), src.slot_ref().loc.node),
        src.well_formed(),
        src.instance_id() == instance.id(),
        old(out)@.len() == src.slot_ref().sub_len,
        old(src_perm).valid_local(instance.id(), src.slot_ref().loc),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(infos.dom()),
        knowledge_of(src.slot_ref().loc.node, cited).subset_of(knows.set()),
        read_admitted(old(src_perm).frontier(), closure_from(infos.map(), cited)),
    ensures
        final(_ctx).stepped(*old(_ctx)),
        final(src_perm).advanced_read(*old(src_perm), old(_ctx).source@.next_event()),
        final(out)@.len() == old(out)@.len(),
        result@.records(
            instance.id(),
            old(_ctx).source@.next_event(),
            read_access(src.slot_ref()),
            read_bytes_effect(src.slot_ref(), final(out)@),
            closure_from(infos.map(), cited),
            old(src_perm).read_guards(),
        ),
{
    assert!(src.len == out.len());
    let Tracked(completed) = ffi::raw_read(src, out);
    let tracked result = finish_read(
        _ctx.source.borrow_mut(),
        instance,
        completed,
        src_perm,
        infos,
        knows,
        cited,
    );
    Tracked(result)
}

pub fn copy(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    dst: &HostSlice,
    src: &HostSlice,
    Tracked(src_perm): Tracked<&mut ReadPerm>,
    Tracked(dst_tok): Tracked<&mut SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: Tracked<Completion>)
    requires
        old(_ctx).ready(instance.id(), src.slot_ref().loc.node),
        src.well_formed(),
        src.instance_id() == instance.id(),
        dst.well_formed(),
        dst.instance_id() == instance.id(),
        src.slot_ref().sub_len == dst.slot_ref().sub_len,
        src.slot_ref().loc.node == dst.slot_ref().loc.node,
        src.slot_ref().loc != dst.slot_ref().loc,
        old(src_perm).valid_local(instance.id(), src.slot_ref().loc),
        slot_is_owned(*old(dst_tok), instance.id(), dst.slot_ref().loc),
        infos.instance_id() == instance.id(),
        knows.instance_id() == instance.id(),
        cited.subset_of(infos.dom()),
        knowledge_of(src.slot_ref().loc.node, cited).subset_of(knows.set()),
        read_admitted(old(src_perm).frontier(), closure_from(infos.map(), cited)),
        write_admitted(slot_frontier(*old(dst_tok)), closure_from(infos.map(), cited)),
    ensures
        final(_ctx).stepped(*old(_ctx)),
        final(src_perm).advanced_read(*old(src_perm), old(_ctx).source@.next_event()),
        slot_owned(
            *final(dst_tok),
            instance.id(),
            dst.slot_ref().loc,
            Frontier { last_writer: Some(old(_ctx).source@.next_event()), readers: ISet::empty() },
        ),
        result@.records(
            instance.id(),
            old(_ctx).source@.next_event(),
            copy_access(src.slot_ref(), dst.slot_ref()),
            copy_effect(src.slot_ref(), dst.slot_ref()),
            closure_from(infos.map(), cited),
            old(src_perm).read_guards(),
        ),
{
    assert!(src.len == dst.len);
    let Tracked(completed) = ffi::raw_copy(dst, src);
    let tracked result = finish_copy(
        _ctx.source.borrow_mut(),
        instance,
        completed,
        src_perm,
        dst_tok,
        infos,
        knows,
        cited,
    );
    Tracked(result)
}

pub fn free(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: HostBuffer,
    Tracked(tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: Tracked<Completion>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.num_slots() == 1,
        buf.instance_id() == instance.id(),
        can_reclaim_slot(instance.id(), buf.slot_loc(0), tok, *infos, *knows, cited),
    ensures
        final(_ctx).stepped(*old(_ctx)),
        result@.records(
            instance.id(),
            old(_ctx).source@.next_event(),
            reclaim_access(buf.slot_loc(0)),
            havoc_effect(),
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
{
    let ghost loc = buf.slot_loc(0);
    let tracked mut tok = tok;
    let Tracked(completed) = ffi::raw_free(buf);
    proof {
        assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
        assert(completed.access == reclaim_access(loc));
    }
    Tracked(
        finish_owned(_ctx.source.borrow_mut(), instance, completed, &mut tok, infos, knows, cited),
    )
}

pub fn free_slab(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: HostBuffer,
    Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: Tracked<Completion>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.instance_id() == instance.id(),
        can_reclaim_slab(
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
        final(_ctx).stepped(*old(_ctx)),
        result@.records(
            instance.id(),
            old(_ctx).source@.next_event(),
            reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
            havoc_effect(),
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
{
    let ghost node = buf.node();
    let ghost space = buf.space();
    let ghost count = buf.num_slots();
    let Tracked(completed) = ffi::raw_free(buf);
    let tracked (completion, _) = finish_slab_owned(
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
    Tracked(completion)
}

} // verus!
