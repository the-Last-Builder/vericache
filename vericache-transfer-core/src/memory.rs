//! Borrowed host views and memory capabilities shared by transfer drivers.
use crate::domain::*;
use core::marker::PhantomData;
#[cfg(verus_only)]
use vstd::arithmetic::mul::{lemma_mul_inequality, lemma_mul_is_distributive_add_other_way};
use vstd::prelude::*;
use vstd::tokens::InstanceId;

verus! {

pub struct HostSlice<'a> {
    ptr: *mut u8,
    len: usize,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    slot: Ghost<SlotRef>,
    instance: Ghost<InstanceId>,
    borrow: PhantomData<&'a ()>,
}

impl<'a> HostSlice<'a> {
    pub closed spec fn address(&self) -> usize {
        self.ptr.addr()
    }

    /// Expose the address for raw-pointer and DMA interop.
    pub fn addr(&self) -> (r: usize)
        ensures
            r == self.address(),
    {
        self.ptr as usize
    }

    pub closed spec fn slot_ref(&self) -> SlotRef {
        self.slot@
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.instance@
    }

    pub closed spec fn well_formed(&self) -> bool {
        self.slot@.well_formed() && self.slot@.loc.space.is_host() && self.slot@.sub_len == self.len
            && self.ptr.addr() + self.len <= usize::MAX
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

/// A live allocation range vouched for by the driver that owns it.
/// Borrowing the range ties views and registrations to the owner's lifetime.
pub struct DmaAllocation {
    ptr: *mut u8,
    len: usize,
    device: Option<DeviceId>,
}

impl DmaAllocation {
    /// An empty capability for adapters with no backing allocation.
    pub exec const UNAVAILABLE: Self
        ensures
            !Self::UNAVAILABLE.well_formed(),
    {
        Self { ptr: core::ptr::null_mut(), len: 0, device: None }
    }

    pub(crate) fn new(ptr: *mut u8, len: usize, device: Option<DeviceId>) -> (a: Self)
        requires
            ptr.addr() != 0,
            ptr.addr() + len <= usize::MAX,
        ensures
            a.well_formed(),
            a.address() == ptr.addr(),
            a.total_len() == len,
            a.device_spec() == device,
    {
        Self { ptr, len, device }
    }

    pub closed spec fn address(&self) -> usize {
        self.ptr.addr()
    }

    pub closed spec fn total_len(&self) -> usize {
        self.len
    }

    pub closed spec fn device_spec(&self) -> Option<DeviceId> {
        self.device
    }

    pub open spec fn well_formed(&self) -> bool {
        self.address() != 0 && self.address() + self.total_len() <= usize::MAX
    }

    pub fn raw_parts(&self) -> (r: (*mut u8, usize))
        ensures
            r.0.addr() == self.address(),
            r.1 == self.total_len(),
    {
        (self.ptr, self.len)
    }

    pub fn device(&self) -> (r: Option<DeviceId>)
        ensures
            r == self.device_spec(),
    {
        self.device
    }
}

/// A borrowed CPU-addressable transfer range with its slot identity.
pub trait HostDmaSlice {
    spec fn dma_address(&self) -> usize;

    spec fn dma_slot_ref(&self) -> SlotRef;

    spec fn dma_instance_id(&self) -> InstanceId;

    spec fn dma_well_formed(&self) -> bool;

    fn dma_raw_parts(&self) -> (r: (*mut u8, usize))
        requires
            self.dma_well_formed(),
        ensures
            r.0.addr() == self.dma_address(),
            r.1 == self.dma_slot_ref().sub_len,
            r.0.addr() + r.1 <= usize::MAX,
            self.dma_slot_ref().well_formed(),
            self.dma_slot_ref().loc.space.is_host(),
    ;
}

impl HostDmaSlice for HostSlice<'_> {
    open spec fn dma_address(&self) -> usize {
        self.address()
    }

    open spec fn dma_slot_ref(&self) -> SlotRef {
        self.slot_ref()
    }

    open spec fn dma_instance_id(&self) -> InstanceId {
        self.instance_id()
    }

    open spec fn dma_well_formed(&self) -> bool {
        self.well_formed()
    }

    fn dma_raw_parts(&self) -> (r: (*mut u8, usize)) {
        (self.ptr, self.len)
    }
}

/// An allocation of contiguous, equal-sized host slots.
pub trait HostDmaSlab {
    spec fn slab_node(&self) -> NodeId;

    spec fn slab_space(&self) -> MemorySpace;

    spec fn slab_num_slots(&self) -> usize;

    spec fn slab_slot_len(&self) -> usize;

    spec fn slab_total_len(&self) -> usize;

    spec fn slab_address(&self) -> usize;

    spec fn slab_instance_id(&self) -> InstanceId;

    spec fn slab_well_formed(&self) -> bool;

    fn slab_allocation(&self) -> (a: &DmaAllocation)
        requires
            self.slab_well_formed(),
        ensures
            a.well_formed(),
            a.device_spec().is_none(),
            a.address() == self.slab_address(),
            a.total_len() == self.slab_total_len(),
            self.slab_space().is_host(),
            self.slab_num_slots() >= 1,
            self.slab_total_len() == self.slab_num_slots() * self.slab_slot_len(),
    ;

    fn slab_num_slots_exec(&self) -> (n: usize)
        requires
            self.slab_well_formed(),
        ensures
            n == self.slab_num_slots(),
    ;

    fn slab_slot_len_exec(&self) -> (n: usize)
        requires
            self.slab_well_formed(),
        ensures
            n == self.slab_slot_len(),
    ;

    fn slab_raw_parts(&self) -> (r: (*mut u8, usize))
        requires
            self.slab_well_formed(),
        ensures
            r.0.addr() == self.slab_address(),
            r.1 == self.slab_total_len(),
            r.0.addr() + r.1 <= usize::MAX,
            self.slab_space().is_host(),
            self.slab_num_slots() >= 1,
            r.1 == self.slab_num_slots() * self.slab_slot_len(),
    {
        self.slab_allocation().raw_parts()
    }

    fn host_slice(&self, idx: usize) -> (s: HostSlice<'_>)
        requires
            self.slab_well_formed(),
            idx < self.slab_num_slots(),
        ensures
            s.well_formed(),
            s.address() == self.slab_address() + idx * self.slab_slot_len(),
            s.slot_ref() == SlotRef::full(
                SlotLoc {
                    node: self.slab_node(),
                    space: self.slab_space(),
                    slot_index: idx as nat,
                },
                self.slab_slot_len() as u64,
            ),
            s.instance_id() == self.slab_instance_id(),
    {
        let allocation = self.slab_allocation();
        let (base, total_len) = allocation.raw_parts();
        let count = self.slab_num_slots_exec();
        let len = self.slab_slot_len_exec();
        assert!(allocation.device().is_none() && idx < count);
        proof {
            lemma_mul_is_distributive_add_other_way(len as int, idx as int, 1);
            lemma_mul_inequality((idx + 1) as int, count as int, len as int);
        }
        let offset = idx.checked_mul(len).unwrap();
        assert!(offset <= total_len && len <= total_len - offset);
        let ptr = base.with_addr(base.addr() + offset);
        HostSlice {
            ptr,
            len,
            slot: Ghost(
                SlotRef::full(
                    SlotLoc {
                        node: self.slab_node(),
                        space: self.slab_space(),
                        slot_index: idx as nat,
                    },
                    self.slab_slot_len() as u64,
                ),
            ),
            instance: Ghost(self.slab_instance_id()),
            borrow: PhantomData,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdrWritesOrdering {
    None,
    Owner,
    AllDevices,
}

/// An allocation that can be registered with an RDMA protection domain.
/// Device implementations report their ordering and flush support for the
/// underlying allocation and satisfy the corresponding visibility contract.
pub trait RdmaRegisterableSlab {
    spec fn reg_node(&self) -> NodeId;

    spec fn reg_space(&self) -> MemorySpace;

    spec fn reg_num_slots(&self) -> usize;

    spec fn reg_slot_len(&self) -> usize;

    spec fn reg_total_len(&self) -> usize;

    spec fn reg_instance_id(&self) -> InstanceId;

    spec fn reg_well_formed(&self) -> bool;

    spec fn reg_gpu_device(&self) -> Option<DeviceId>;

    fn reg_allocation(&self) -> (a: &DmaAllocation)
        requires
            self.reg_well_formed(),
        ensures
            a.well_formed(),
            a.total_len() == self.reg_total_len(),
            a.device_spec() == self.reg_gpu_device(),
            self.reg_num_slots() >= 1,
            self.reg_total_len() == self.reg_num_slots() * self.reg_slot_len(),
            match self.reg_gpu_device() {
                None => self.reg_space().is_host(),
                Some(d) => self.reg_space().is_gpu() && self.reg_space().gpu_device() == d,
            },
    ;

    fn reg_raw_parts(&self) -> (r: (*mut u8, usize))
        requires
            self.reg_well_formed(),
        ensures
            r.1 == self.reg_total_len(),
            r.0.addr() + r.1 <= usize::MAX,
            self.reg_num_slots() >= 1,
            r.1 == self.reg_num_slots() * self.reg_slot_len(),
    {
        self.reg_allocation().raw_parts()
    }

    fn reg_num_slots_exec(&self) -> (n: usize)
        requires
            self.reg_well_formed(),
        ensures
            n == self.reg_num_slots(),
    ;

    fn reg_slot_len_exec(&self) -> (n: usize)
        requires
            self.reg_well_formed(),
        ensures
            n == self.reg_slot_len(),
    ;

    fn reg_gpu_device_exec(&self) -> (d: Option<DeviceId>)
        requires
            self.reg_well_formed(),
        ensures
            d == self.reg_gpu_device(),
    {
        self.reg_allocation().device()
    }

    fn prepare_rdma_registration(&self) -> Result<(), i32>
        requires
            self.reg_well_formed(),
    {
        Ok(())
    }

    fn query_gdr_ordering(&self) -> (r: Result<Option<GdrWritesOrdering>, i32>)
        requires
            self.reg_well_formed(),
        ensures
            r matches Ok(ordering) ==> ordering.is_some() == self.reg_gpu_device().is_some(),
    {
        if self.reg_gpu_device_exec().is_some() {
            Err(95)
        } else {
            Ok(None)
        }
    }

    fn query_gdr_flush_supported(&self) -> Result<bool, i32>
        requires
            self.reg_well_formed(),
    {
        Ok(false)
    }

    /// Flush writes to the registered device when explicit visibility is required.
    fn flush_gdr_writes(&self, _device: i32) -> Result<(), i32> {
        Err(95)
    }
}

} // verus!
