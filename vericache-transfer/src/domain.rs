//! Slot-level memory identities, event identities, and zero-`forall` effects
//! shared by all backends.
#[cfg(verus_only)]
use foundation::space::{Access, Effect, Snapshot};
use vstd::prelude::*;
verus! {

pub type NodeId = u64;

pub type DeviceId = u32;

/// Pure ghost identifier for one allocation; never overflows, never reused.
pub type AllocId = nat;

pub ghost enum MemorySpace {
    HostAlloc { alloc: AllocId },
    NvmeFile { alloc: AllocId },
    GpuAlloc { device: DeviceId, alloc: AllocId },
}

/// Slot-level physical location: standalone buffers use `slot_index == 0`;
/// slab allocations use `slot_index` in `0..num_slots`.
pub ghost struct SlotLoc {
    pub node: NodeId,
    pub space: MemorySpace,
    pub slot_index: nat,
}

/// A contiguous byte range inside one logical slot.
pub ghost struct SlotRef {
    pub loc: SlotLoc,
    pub slot_len: u64,
    pub sub_offset: u64,
    pub sub_len: u64,
}

pub ghost enum EventSource {
    Host { id: nat },
    CudaStream { id: nat },
    IoUring { id: nat },
    RdmaQueuePair { id: nat },
}

pub ghost struct TransferEvent {
    pub observer: NodeId,
    pub source: EventSource,
    /// Pure ghost sequence number; never overflows.
    pub seq: nat,
}

pub ghost struct TransferShare {
    pub owner: NodeId,
    pub seq: nat,
}

pub struct TransferDomain;

impl foundation::domain::Domain for TransferDomain {
    type Loc = SlotLoc;

    type Event = TransferEvent;

    type Value = Seq<u8>;

    type Observer = NodeId;

    type Share = TransferShare;
}

impl MemorySpace {
    pub open spec fn alloc_id(self) -> AllocId {
        match self {
            Self::HostAlloc { alloc }
            | Self::NvmeFile { alloc }
            | Self::GpuAlloc { device: _, alloc } => alloc,
        }
    }
}

impl EventSource {
    pub open spec fn id(self) -> nat {
        match self {
            Self::Host { id }
            | Self::IoUring { id }
            | Self::CudaStream { id }
            | Self::RdmaQueuePair { id } => id,
        }
    }

    /// True for event sources that only produce an observation when the
    /// underlying operation succeeded.
    pub open spec fn succeeds_on_observe(self) -> bool {
        match self {
            Self::Host { .. } | Self::CudaStream { .. } => true,
            Self::IoUring { .. } | Self::RdmaQueuePair { .. } => false,
        }
    }
}

impl SlotRef {
    pub open spec fn full(loc: SlotLoc, len: u64) -> Self {
        Self { loc, slot_len: len, sub_offset: 0, sub_len: len }
    }

    pub open spec fn well_formed(self) -> bool {
        self.sub_offset + self.sub_len <= self.slot_len
    }

    pub open spec fn is_full(self) -> bool {
        self.sub_offset == 0 && self.sub_len == self.slot_len
    }

    pub open spec fn sub_slice(self, off: u64, len: u64) -> Self {
        Self {
            loc: self.loc,
            slot_len: self.slot_len,
            sub_offset: (self.sub_offset + off) as u64,
            sub_len: len,
        }
    }
}

/// Access set for writing `dst`. A partial-slot splice reads the prior slot
/// value to preserve untouched prefix/suffix bytes.
pub open spec fn write_access(dst: SlotRef) -> Access<TransferDomain> {
    Access {
        reads: if dst.is_full() {
            ISet::empty()
        } else {
            ISet::empty().insert(dst.loc)
        },
        writes: ISet::empty().insert(dst.loc),
    }
}

/// Access set for reading `src`.
pub open spec fn read_access(src: SlotRef) -> Access<TransferDomain> {
    Access { reads: ISet::empty().insert(src.loc), writes: ISet::empty() }
}

/// Access set for copying from `src` to `dst`. A partial-slot destination also
/// reads `dst.loc` to preserve untouched prefix/suffix bytes.
pub open spec fn copy_access(src: SlotRef, dst: SlotRef) -> Access<TransferDomain> {
    Access {
        reads: if dst.is_full() {
            ISet::empty().insert(src.loc)
        } else {
            ISet::empty().insert(src.loc).insert(dst.loc)
        },
        writes: ISet::empty().insert(dst.loc),
    }
}

/// Access set for reclaiming one slot.
pub open spec fn reclaim_access(loc: SlotLoc) -> Access<TransferDomain> {
    Access { reads: ISet::empty(), writes: ISet::empty().insert(loc) }
}

/// Access set for deregistering one slot (preserves logical bytes).
pub open spec fn dereg_access(loc: SlotLoc) -> Access<TransferDomain> {
    Access { reads: ISet::empty().insert(loc), writes: ISet::empty().insert(loc) }
}

/// The set of slot locations belonging to a slab of `num_slots` slots.
pub open spec fn slab_locs(node: NodeId, space: MemorySpace, num_slots: nat) -> ISet<SlotLoc> {
    ISet::new(|l: SlotLoc| l.node == node && l.space == space && l.slot_index < num_slots)
}

/// Access set for reclaiming an entire slab.
pub open spec fn reclaim_slab_access(node: NodeId, space: MemorySpace, num_slots: nat) -> Access<
    TransferDomain,
> {
    Access { reads: ISet::empty(), writes: slab_locs(node, space, num_slots) }
}

/// Access set for deregistering an entire slab (preserves all slot values).
pub open spec fn dereg_slab_access(node: NodeId, space: MemorySpace, num_slots: nat) -> Access<
    TransferDomain,
> {
    let locs = slab_locs(node, space, num_slots);
    Access { reads: locs, writes: locs }
}

/// Write concrete bytes into `dst`, preserving prefix and suffix when `dst` is
/// a sub-slice of an already-sized slot.
pub open spec fn splice_bytes(dst: SlotRef, bytes: Seq<u8>) -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>|
        output[dst.loc].len() == dst.slot_len && output[dst.loc].subrange(
            dst.sub_offset as int,
            (dst.sub_offset + dst.sub_len) as int,
        ) == bytes && (input[dst.loc].len() == dst.slot_len ==> (output[dst.loc].subrange(
            0,
            dst.sub_offset as int,
        ) == input[dst.loc].subrange(0, dst.sub_offset as int) && output[dst.loc].subrange(
            (dst.sub_offset + dst.sub_len) as int,
            dst.slot_len as int,
        ) == input[dst.loc].subrange((dst.sub_offset + dst.sub_len) as int, dst.slot_len as int)))
}

/// For a full-slot write, `splice_bytes` directly fixes `output[dst.loc] == bytes`.
pub proof fn splice_bytes_full(
    dst: SlotRef,
    bytes: Seq<u8>,
    input: Snapshot<TransferDomain>,
    output: Snapshot<TransferDomain>,
)
    requires
        dst.is_full(),
        (splice_bytes(dst, bytes))(input, output),
    ensures
        output[dst.loc] == bytes,
{
    assert(output[dst.loc].subrange(0, dst.slot_len as int) =~= output[dst.loc]);
}

/// Synchronous read tying executable output bytes `observed` to `src`.
pub open spec fn read_bytes_effect(src: SlotRef, observed: Seq<u8>) -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, _output: Snapshot<TransferDomain>|
        input[src.loc].len() >= src.sub_offset + src.sub_len && input[src.loc].subrange(
            src.sub_offset as int,
            (src.sub_offset + src.sub_len) as int,
        ) == observed
}

/// For a full-slot read of a full-length slot, `observed` equals the slot value.
pub proof fn read_bytes_full(
    src: SlotRef,
    observed: Seq<u8>,
    input: Snapshot<TransferDomain>,
    output: Snapshot<TransferDomain>,
)
    requires
        src.is_full(),
        input[src.loc].len() == src.slot_len,
        (read_bytes_effect(src, observed))(input, output),
    ensures
        observed == input[src.loc],
{
    assert(input[src.loc].subrange(0, src.slot_len as int) =~= input[src.loc]);
}

/// Byte-for-byte copy from `src` to `dst` (`src.sub_len == dst.sub_len`),
/// preserving prefix and suffix of `dst` when `dst` is a sub-slice.
pub open spec fn copy_effect(src: SlotRef, dst: SlotRef) -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>|
        input[src.loc].len() >= src.sub_offset + src.sub_len && output[dst.loc].len()
            == dst.slot_len && output[dst.loc].subrange(
            dst.sub_offset as int,
            (dst.sub_offset + dst.sub_len) as int,
        ) == input[src.loc].subrange(src.sub_offset as int, (src.sub_offset + src.sub_len) as int)
            && (input[dst.loc].len() == dst.slot_len ==> (output[dst.loc].subrange(
            0,
            dst.sub_offset as int,
        ) == input[dst.loc].subrange(0, dst.sub_offset as int) && output[dst.loc].subrange(
            (dst.sub_offset + dst.sub_len) as int,
            dst.slot_len as int,
        ) == input[dst.loc].subrange((dst.sub_offset + dst.sub_len) as int, dst.slot_len as int)))
}

/// For a full-slot copy between equal-length slots, `output[dst.loc] == input[src.loc]`.
pub proof fn copy_effect_full(
    src: SlotRef,
    dst: SlotRef,
    input: Snapshot<TransferDomain>,
    output: Snapshot<TransferDomain>,
)
    requires
        src.is_full(),
        dst.is_full(),
        src.slot_len == dst.slot_len,
        input[src.loc].len() == src.slot_len,
        (copy_effect(src, dst))(input, output),
    ensures
        output[dst.loc] == input[src.loc],
{
    assert(input[src.loc].subrange(0, src.slot_len as int) =~= input[src.loc]);
    assert(output[dst.loc].subrange(0, dst.slot_len as int) =~= output[dst.loc]);
}

/// Preserves the logical value of `loc` while conflicting with all prior accesses.
/// Used for single-slot memory deregistration (`cudaHostUnregister`, `ibv_dereg_mr`).
pub open spec fn identity_effect(loc: SlotLoc) -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>| output[loc] == input[loc]
}

/// Preserves every slot in a deregistered slab (`reads == writes == slab_locs`).
pub open spec fn identity_effect_slab() -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>| output == input
}

/// Allows any resulting slot value. Used for reclamation (`free`, `close`).
pub open spec fn havoc_effect() -> Effect<TransferDomain> {
    |_input: Snapshot<TransferDomain>, _output: Snapshot<TransferDomain>| true
}

} // verus!
