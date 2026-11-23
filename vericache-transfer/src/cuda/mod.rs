//! CUDA allocations (single-slot and slab), non-blocking streams, and snapshot events.
//! Release allocations, registrations, streams and events explicitly. A failed
//! free or unregister returns the handle and its original slot permissions.
mod ffi;
mod host;
mod proofs;
mod stream;

use crate::context::*;
use crate::domain::*;
#[cfg(verus_only)]
use crate::host as cpu_host;
use crate::host::{CompletedAccess, Completion, HostBuffer, HostContext, HostSlice};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdrWritesOrdering {
    None,
    Owner,
    AllDevices,
}

pub enum EventStatus {
    NotReady,
    Ready(Tracked<Knows>),
}

// Private slab backing metadata for `CudaBuffer` and `CudaHostBuffer`.
pub(super) struct SlabMemory {
    pub(super) ptr: *mut u8,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) num_slots: usize,
    pub(super) slot_len: usize,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) len: usize,
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
        &&& self.len == self.num_slots * self.slot_len
        &&& self.ptr.addr() + self.len <= usize::MAX
        &&& match self.kind {
            CudaPointerKind::PinnedHost => self.space@ is HostAlloc,
            CudaPointerKind::Device(d) => self.space@ is GpuAlloc && self.space@->GpuAlloc_device
                == d,
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
        let ptr = self.ptr.with_addr(self.ptr.addr() + off);
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
            CudaPointerKind::PinnedHost => self.slot@.loc.space is HostAlloc,
            CudaPointerKind::Device(d) => self.slot@.loc.space is GpuAlloc
                && self.slot@.loc.space->GpuAlloc_device == d,
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
        self.memory.well_formed() && self.memory.kind is Device
    }

    pub fn device(&self) -> (d: DeviceId)
        requires
            self.well_formed(),
        ensures
            d == self.space()->GpuAlloc_device,
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
            s.kind() == CudaPointerKind::Device(self.space()->GpuAlloc_device),
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
            s.kind() == CudaPointerKind::Device(self.space()->GpuAlloc_device),
    {
        assert!(idx < self.memory.num_slots);
        self.memory.borrow_slot(idx)
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
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(
                    Some(device),
                )
                &&& buf.slot_ref(0) == old(source).slot_ref(Some(device), len as u64)
                &&& final(source).allocated_slot(*old(source), buf.slot_loc(0), tok@)
            },
        },
{
    let ptr = ffi::raw_alloc_device(device, len)?;
    let ghost node = source.node();
    let ghost space = source.space(Some(device));
    let tracked tok = source.allocate_slot(Some(device));
    Ok(
        (
            CudaBuffer {
                memory: SlabMemory {
                    ptr,
                    num_slots: 1,
                    slot_len: len,
                    len,
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
                &&& buf.node() == old(source).node() && buf.space() == old(source).space(
                    Some(device),
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
    let ptr = ffi::raw_alloc_device(device, total)?;
    let ghost node = source.node();
    let ghost space = source.space(Some(device));
    let tracked tokens = source.allocate_slab_slots(Some(device), num_slots);
    Ok(
        (
            CudaBuffer {
                memory: SlabMemory {
                    ptr,
                    num_slots,
                    slot_len,
                    len: total,
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

#[allow(dead_code)]
pub(crate) fn enable_sync_memops(buf: &CudaBuffer) -> Result<(), CudaError>
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
    _ctx: &mut HostContext,
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
    let Tracked(completed) = match ffi::raw_free_device(&buf) {
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

pub fn free_device_slab(
    _ctx: &mut HostContext,
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
    let Tracked(completed) = match ffi::raw_free_device(&buf) {
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

} // verus!
