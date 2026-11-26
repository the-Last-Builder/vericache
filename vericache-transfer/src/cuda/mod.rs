//! CUDA allocations (single-slot and slab), non-blocking streams, and snapshot events.
//! Release allocations, registrations, streams and events explicitly. A failed
//! free or unregister returns the handle and its original slot permissions.
mod ffi;
mod host;
mod proofs;
mod stream;

use crate::context::*;
use crate::domain::*;
pub use crate::memory::GdrWritesOrdering;
use crate::memory::{DmaAllocation, HostDmaSlab, HostSlice, RdmaRegisterableSlab};
use core::marker::PhantomData;
use foundation::space::{Access, Effect};
#[cfg(verus_only)]
use foundation::tracked::*;
#[cfg(verus_only)]
use vstd::arithmetic::mul::{lemma_mul_inequality, lemma_mul_is_distributive_add_other_way};
use vstd::prelude::*;
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

pub use host::{
    alloc_host, alloc_host_slab, free_host, free_host_slab, host_register, host_unregister,
    host_unregister_slab, CudaHostBuffer, CudaHostRegistration,
};
#[cfg(verus_only)]
use proofs::*;
#[cfg(verus_only)]
pub use stream::copy_admitted;
use stream::EnqueuedCopy;
pub use stream::{
    event_query, event_record, memcpy_async, stream_synchronize, stream_wait_event, CudaEvent,
    CudaStream,
};

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

#[derive(Debug, PartialEq, Eq)]
pub struct CudaError(pub i32);

impl CudaError {
    pub(super) const INVALID_VALUE: Self = CudaError(1);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CudaPointerKind {
    Device(DeviceId),
    PinnedHost,
}

pub enum EventStatus {
    NotReady,
    Ready(Tracked<Knows>),
}

// Private slab backing metadata for `CudaBuffer` and `CudaHostBuffer`.
pub(super) struct SlabMemory {
    pub(super) allocation: DmaAllocation,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) num_slots: usize,
    pub(super) slot_len: usize,
    pub(super) kind: CudaPointerKind,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) node: Ghost<NodeId>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) space: Ghost<MemorySpace>,
    pub(super) instance: Ghost<InstanceId>,
}

impl SlabMemory {
    pub(super) open spec fn slot_loc(&self, idx: nat) -> SlotLoc {
        SlotLoc { node: self.node@, space: self.space@, slot_index: idx }
    }

    pub(super) open spec fn slot_ref(&self, idx: nat) -> SlotRef {
        SlotRef::full(self.slot_loc(idx), self.slot_len as u64)
    }

    pub(super) open spec fn well_formed(&self) -> bool {
        &&& self.num_slots >= 1
        &&& self.allocation.total_len() == self.num_slots * self.slot_len
        &&& self.allocation.well_formed()
        &&& match self.kind {
            CudaPointerKind::PinnedHost => self.space@.is_host()
                && self.allocation.device_spec().is_none(),
            CudaPointerKind::Device(d) => self.space@.is_gpu() && self.space@.gpu_device() == d
                && self.allocation.device_spec() == Some(d),
        }
    }

    pub(super) fn borrow_slot(&self, idx: usize) -> (s: CudaSlice<'_>)
        requires
            self.well_formed(),
            idx < self.num_slots,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(idx as nat),
            s.instance_id() == self.instance@,
            s.kind() == self.kind,
    {
        assert!(idx < self.num_slots);
        proof {
            lemma_mul_is_distributive_add_other_way(self.slot_len as int, idx as int, 1);
            lemma_mul_inequality((idx + 1) as int, self.num_slots as int, self.slot_len as int);
        }
        let off = idx * self.slot_len;
        let (base, _) = self.allocation.raw_parts();
        let ptr = base.with_addr(base.addr() + off);
        CudaSlice {
            memory: SliceMemory {
                ptr,
                len: self.slot_len,
                kind: self.kind,
                slot: Ghost(self.slot_ref(idx as nat)),
                instance: self.instance,
            },
            borrow: PhantomData,
        }
    }
}

// Kept private so a caller can only borrow a view from a live allocation.
pub(super) struct SliceMemory {
    pub(super) ptr: *mut u8,
    pub(super) len: usize,
    pub(super) kind: CudaPointerKind,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) slot: Ghost<SlotRef>,
    pub(super) instance: Ghost<InstanceId>,
}

impl SliceMemory {
    pub(super) open spec fn well_formed(&self) -> bool {
        &&& self.slot@.well_formed()
        &&& self.slot@.sub_len == self.len
        &&& self.ptr.addr() + self.len <= usize::MAX
        &&& match self.kind {
            CudaPointerKind::PinnedHost => self.slot@.loc.space.is_host(),
            CudaPointerKind::Device(d) => self.slot@.loc.space.is_gpu()
                && self.slot@.loc.space.gpu_device() == d,
        }
    }
}

pub struct CudaBuffer {
    pub(super) memory: SlabMemory,
}

pub struct CudaSlice<'a> {
    pub(super) memory: SliceMemory,
    pub(super) borrow: PhantomData<&'a u8>,
}

impl<'a> CudaSlice<'a> {
    pub closed spec fn slot_ref(&self) -> SlotRef {
        self.memory.slot@
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.memory.instance@
    }

    pub closed spec fn kind(&self) -> CudaPointerKind {
        self.memory.kind
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.memory.well_formed()
    }

    pub fn len(&self) -> (n: usize)
        requires
            self.well_formed(),
        ensures
            n == self.slot_ref().sub_len,
    {
        self.memory.len
    }

    /// Return the raw device/pinned virtual address of this slice as an integer
    /// (read-only address inspection for external CUDA runtime interop such as
    /// PyTorch/vLLM `cudaMemcpyAsync`).
    pub fn device_addr(&self) -> (addr: usize)
        requires
            self.well_formed(),
    {
        self.memory.ptr.addr()
    }

    pub fn sub_slice(&self, offset: usize, len: usize) -> (s: CudaSlice<'a>)
        requires
            self.well_formed(),
            offset + len <= self.slot_ref().sub_len,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref().sub_slice(offset as u64, len as u64),
            s.instance_id() == self.instance_id(),
            s.kind() == self.kind(),
    {
        let m = &self.memory;
        assert!(offset <= m.len && len <= m.len - offset);
        CudaSlice {
            memory: SliceMemory {
                ptr: m.ptr.with_addr(m.ptr.addr() + offset),
                len,
                kind: m.kind,
                slot: Ghost(m.slot@.sub_slice(offset as u64, len as u64)),
                instance: m.instance,
            },
            borrow: PhantomData,
        }
    }
}

impl CudaBuffer {
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
        self.memory.allocation.total_len()
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
        self.memory.well_formed() && self.memory.kind is Device
    }

    pub fn device(&self) -> (d: DeviceId)
        requires
            self.well_formed(),
        ensures
            d == self.space().gpu_device(),
    {
        match self.memory.kind {
            CudaPointerKind::Device(d) => d,
            CudaPointerKind::PinnedHost => 0,
        }
    }

    pub fn as_cuda_slice(&self) -> (s: CudaSlice<'_>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(0),
            s.instance_id() == self.instance_id(),
            s.kind() == CudaPointerKind::Device(self.space().gpu_device()),
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
            s.kind() == CudaPointerKind::Device(self.space().gpu_device()),
    {
        assert!(idx < self.memory.num_slots);
        self.memory.borrow_slot(idx)
    }
}

impl RdmaRegisterableSlab for CudaBuffer {
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
        Some(self.space().gpu_device())
    }

    fn reg_allocation(&self) -> &DmaAllocation {
        &self.memory.allocation
    }

    fn reg_num_slots_exec(&self) -> usize {
        self.num_slots_val()
    }

    fn reg_slot_len_exec(&self) -> usize {
        self.slot_len_val()
    }

    fn prepare_rdma_registration(&self) -> Result<(), i32> {
        match enable_sync_memops(self) {
            Ok(()) => Ok(()),
            Err(e) => Err(e.0),
        }
    }

    fn query_gdr_ordering(&self) -> Result<Option<GdrWritesOrdering>, i32> {
        let device = gdr_device_index(self.device())?;
        match gdr_writes_ordering(device) {
            Ok(order) => Ok(Some(order)),
            Err(e) => Err(e.0),
        }
    }

    fn query_gdr_flush_supported(&self) -> Result<bool, i32> {
        let device = gdr_device_index(self.device())?;
        match gdr_flush_writes_supported(device) {
            Ok(supported) => Ok(supported),
            Err(e) => Err(e.0),
        }
    }

    fn flush_gdr_writes(&self, device: i32) -> Result<(), i32> {
        match flush_gdr_writes(device) {
            Ok(()) => Ok(()),
            Err(e) => Err(e.0),
        }
    }
}

fn gdr_device_index(device: DeviceId) -> Result<i32, i32> {
    if device > i32::MAX as u32 {
        Err(22)
    } else {
        Ok(device as i32)
    }
}

pub fn alloc_device(Tracked(source): Tracked<&mut AllocSource>, device: DeviceId, len: usize) -> (r:
    Result<(CudaBuffer, Tracked<SlotToken>), CudaError>)
    requires
        old(source).well_formed(),
    ensures
        match r {
            Err(_) => *final(source) == *old(source),
            Ok((buf, tok)) => {
                &&& buf.well_formed() && buf.instance_id() == old(source).id()
                &&& buf.num_slots() == 1 && buf.slot_len() == len
                &&& buf.node() == old(source).node() && buf.space() == old(source).space_for(
                    SPACE_GPU,
                    device,
                )
                &&& buf.slot_ref(0) == old(source).slot_ref_for(SPACE_GPU, device, len as u64)
                &&& final(source).allocated_slot(*old(source), buf.slot_loc(0), tok@)
            },
        },
{
    let allocation = ffi::raw_alloc_device(device, len)?;
    let ghost node = source.node();
    let ghost space = MemorySpace::gpu(device, source.next_alloc_id());
    let tracked tok = source.allocate_slot_in(space);
    Ok(
        (
            CudaBuffer {
                memory: SlabMemory {
                    allocation,
                    num_slots: 1,
                    slot_len: len,
                    kind: CudaPointerKind::Device(device),
                    node: Ghost(node),
                    space: Ghost(space),
                    instance: Ghost(source.id()),
                },
            },
            Tracked(tok),
        ),
    )
}

pub fn alloc_device_slab(
    Tracked(source): Tracked<&mut AllocSource>,
    device: DeviceId,
    num_slots: usize,
    slot_len: usize,
) -> (r: Result<(CudaBuffer, Tracked<Map<usize, SlotToken>>), CudaError>)
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
                &&& buf.node() == old(source).node() && buf.space() == old(source).space_for(
                    SPACE_GPU,
                    device,
                )
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
    let allocation = ffi::raw_alloc_device(device, total)?;
    let ghost node = source.node();
    let ghost space = MemorySpace::gpu(device, source.next_alloc_id());
    let tracked tokens = source.allocate_slab_in(space, num_slots);
    Ok(
        (
            CudaBuffer {
                memory: SlabMemory {
                    allocation,
                    num_slots,
                    slot_len,
                    kind: CudaPointerKind::Device(device),
                    node: Ghost(node),
                    space: Ghost(space),
                    instance: Ghost(source.id()),
                },
            },
            Tracked(tokens),
        ),
    )
}

fn enable_sync_memops(buf: &CudaBuffer) -> Result<(), CudaError>
    requires
        buf.well_formed(),
{
    ffi::raw_enable_sync_memops(buf)
}

pub fn device_count() -> Result<u32, CudaError> {
    ffi::raw_device_count()
}

pub fn gdr_writes_ordering(device: i32) -> Result<GdrWritesOrdering, CudaError> {
    ffi::raw_gdr_writes_ordering(device)
}

pub fn gdr_flush_writes_supported(device: i32) -> Result<bool, CudaError> {
    ffi::raw_gdr_flush_writes_supported(device)
}

pub fn flush_gdr_writes(device: i32) -> Result<(), CudaError> {
    ffi::raw_flush_gdr_writes(device)
}

pub fn free_device(
    _ctx: &mut SyncContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: CudaBuffer,
    Tracked(tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<Tracked<Completion>, (CudaError, CudaBuffer, Tracked<SlotToken>)>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.num_slots() == 1,
        buf.instance_id() == instance.id(),
        can_reclaim_slot(instance.id(), buf.slot_loc(0), tok, *infos, *knows, cited),
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
    let Tracked(completed) = match ffi::raw_free_device(&buf) {
        Err(e) => return Err((e, buf, Tracked(tok))),
        Ok(w) => w,
    };
    proof {
        assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
        assert(completed.access() == reclaim_access(loc));
    }
    let tracked completion = finish_owned(
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

pub fn free_device_slab(
    _ctx: &mut SyncContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: CudaBuffer,
    Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<Tracked<Completion>, (CudaError, CudaBuffer, Tracked<Map<usize, SlotToken>>)>)
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
    let Tracked(completed) = match ffi::raw_free_device(&buf) {
        Err(e) => return Err((e, buf, Tracked(tokens))),
        Ok(w) => w,
    };
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
    Ok(Tracked(completion))
}

} // verus!
