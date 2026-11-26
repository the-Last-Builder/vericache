//! Synchronous operations on host allocations (standalone buffers and slabs).
mod ffi;
mod proofs;

use crate::context::*;
use crate::domain::*;
use crate::memory::{DmaAllocation, HostDmaSlab, RdmaRegisterableSlab};
#[cfg(verus_only)]
use foundation::tracked::*;
pub use transfer_core::memory::HostSlice;
use vstd::prelude::*;
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

#[cfg(verus_only)]
pub use crate::context::{can_reclaim_slab, can_reclaim_slot};
pub use crate::context::{Completion, HostContext, SyncContext};

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub open spec fn is_power_of_two_usize(align: usize) -> bool {
    align > 0 && (align & ((align - 1) as usize)) == 0
}

pub struct HostBuffer {
    align: usize,
    allocation: DmaAllocation,
    num_slots: usize,
    slot_len: usize,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    node: Ghost<NodeId>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    space: Ghost<MemorySpace>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    instance: Ghost<InstanceId>,
}

impl HostBuffer {
    pub closed spec fn address(&self) -> usize {
        self.allocation.address()
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
        self.allocation.total_len()
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
        &&& self.allocation.well_formed()
        &&& self.allocation.device_spec().is_none()
        &&& self.space@.is_host()
        &&& self.num_slots >= 1
        &&& self.allocation.total_len() == self.num_slots * self.slot_len
        &&& is_power_of_two_usize(self.align)
        &&& self.allocation.address() % self.align == 0
        &&& self.allocation.address() + self.allocation.total_len() <= usize::MAX
    }

    pub fn len(&self) -> (result: usize)
        requires
            self.well_formed(),
        ensures
            result == self.total_len(),
            result == self.num_slots() * self.slot_len(),
    {
        self.allocation.raw_parts().1
    }

    pub fn alignment(&self) -> (align: usize)
        ensures
            align == self.align(),
    {
        self.align
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
        self.host_slice(idx)
    }
}

impl HostDmaSlab for HostBuffer {
    open spec fn slab_node(&self) -> NodeId {
        self.node()
    }

    open spec fn slab_space(&self) -> MemorySpace {
        self.space()
    }

    open spec fn slab_num_slots(&self) -> usize {
        self.num_slots()
    }

    open spec fn slab_slot_len(&self) -> usize {
        self.slot_len()
    }

    open spec fn slab_total_len(&self) -> usize {
        self.total_len()
    }

    open spec fn slab_instance_id(&self) -> InstanceId {
        self.instance_id()
    }

    open spec fn slab_well_formed(&self) -> bool {
        self.well_formed()
    }

    open spec fn slab_address(&self) -> usize {
        self.address()
    }

    fn slab_allocation(&self) -> &DmaAllocation {
        &self.allocation
    }

    fn slab_num_slots_exec(&self) -> usize {
        self.num_slots_val()
    }

    fn slab_slot_len_exec(&self) -> usize {
        self.slot_len_val()
    }
}

impl RdmaRegisterableSlab for HostBuffer {
    open spec fn reg_node(&self) -> NodeId {
        self.node()
    }

    open spec fn reg_space(&self) -> MemorySpace {
        self.space()
    }

    open spec fn reg_num_slots(&self) -> usize {
        self.num_slots()
    }

    open spec fn reg_slot_len(&self) -> usize {
        self.slot_len()
    }

    open spec fn reg_total_len(&self) -> usize {
        self.total_len()
    }

    open spec fn reg_instance_id(&self) -> InstanceId {
        self.instance_id()
    }

    open spec fn reg_well_formed(&self) -> bool {
        self.well_formed()
    }

    open spec fn reg_gpu_device(&self) -> Option<DeviceId> {
        None
    }

    fn reg_allocation(&self) -> &DmaAllocation {
        &self.allocation
    }

    fn reg_num_slots_exec(&self) -> usize {
        self.num_slots_val()
    }

    fn reg_slot_len_exec(&self) -> usize {
        self.slot_len_val()
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
                &&& buf.node() == old(source).node() && buf.space() == old(source).space_for(
                    SPACE_HOST,
                    0,
                )
                &&& buf.slot_ref(0) == old(source).slot_ref_for(SPACE_HOST, 0, len as u64)
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
                &&& buf.node() == old(source).node() && buf.space() == old(source).space_for(
                    SPACE_HOST,
                    0,
                )
                &&& buf.slot_ref(0) == old(source).slot_ref_for(SPACE_HOST, 0, len as u64)
                &&& final(source).allocated_slot(*old(source), buf.slot_loc(0), tok@)
            },
        },
{
    if align == 0 || (align & (align - 1)) != 0 {
        return None;
    }
    let allocation = ffi::raw_alloc_aligned(len, align)?;
    let ghost node = source.node();
    let ghost space = MemorySpace::host(source.next_alloc_id());
    let tracked tok = source.allocate_slot_in(space);
    Some(
        (
            HostBuffer {
                allocation,
                num_slots: 1,
                slot_len: len,
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
                &&& buf.node() == old(source).node() && buf.space() == old(source).space_for(
                    SPACE_HOST,
                    0,
                )
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
    let allocation = ffi::raw_alloc_aligned(total, align)?;
    let ghost node = source.node();
    let ghost space = MemorySpace::host(source.next_alloc_id());
    let tracked tokens = source.allocate_slab_in(space, num_slots);
    Some(
        (
            HostBuffer {
                allocation,
                num_slots,
                slot_len,
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
    assert!(dst.len() == data.len());
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

/// Record a completed external write to an exclusively owned host slice.
/// Finish and synchronize all external writes to `dst` before calling this.
#[inline]
pub fn record_external_write(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    dst: &HostSlice,
    Tracked(dst_tok): Tracked<&mut SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (result: (Tracked<Completion>, Ghost<Seq<u8>>))
    requires
        old(_ctx).ready(instance.id(), dst.slot_ref().loc.node),
        dst.well_formed(),
        dst.instance_id() == instance.id(),
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
        result.1@.len() == dst.slot_ref().sub_len,
        result.0@.records(
            instance.id(),
            old(_ctx).source@.next_event(),
            write_access(dst.slot_ref()),
            splice_bytes(dst.slot_ref(), result.1@),
            closure_from(infos.map(), cited),
            IMap::empty(),
        ),
{
    let (Tracked(completed), bytes) = ffi::raw_observe_write(dst);
    let tracked result = finish_owned(
        _ctx.source.borrow_mut(),
        instance,
        completed,
        dst_tok,
        infos,
        knows,
        cited,
    );
    (Tracked(result), bytes)
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
        old(src_perm).valid_cooperative(instance.id(), src.slot_ref().loc),
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
    assert!(src.len() == out.len());
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
        old(src_perm).valid_cooperative(instance.id(), src.slot_ref().loc),
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
    assert!(src.len() == dst.len());
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
        assert(completed.access() == reclaim_access(loc));
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
