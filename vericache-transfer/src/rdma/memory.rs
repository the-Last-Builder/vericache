use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// Sealed GPU destination slot token held until GPUDirect RDMA BAR1 writes
/// have been flushed to the owning GPU via `MemoryRegion::complete_gdr_slot`.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct PendingGdrSlot {
    pub(super) ghost writer: TransferEvent,
    pub(super) tracked token: SlotToken,
}

impl PendingGdrSlot {
    pub open(super) spec fn writer(self) -> TransferEvent {
        self.writer
    }

    pub open(super) spec fn instance_id(self) -> InstanceId {
        self.token.instance_id()
    }

    pub open(super) spec fn loc(self) -> SlotLoc {
        self.token.key()
    }

    pub open(super) spec fn frontier(self) -> Frontier<TransferDomain> {
        slot_frontier(self.token)
    }

    pub open(super) spec fn well_formed(self) -> bool {
        slot_is_owned(self.token, self.token.instance_id(), self.token.key())
    }
}

/// Sealed map of GPU destination slot tokens held until GPUDirect RDMA BAR1
/// writes are flushed via `MemoryRegion::complete_gdr_slots`.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct PendingGdrSlots {
    pub(super) ghost instance: InstanceId,
    pub(super) ghost writers: Map<SlotLoc, TransferEvent>,
    pub(super) tracked tokens: Map<SlotLoc, SlotToken>,
}

impl PendingGdrSlots {
    pub open(super) spec fn instance_id(self) -> InstanceId {
        self.instance
    }

    pub open(super) spec fn dom(self) -> Set<SlotLoc> {
        self.tokens.dom()
    }

    pub open(super) spec fn writers(self) -> Map<SlotLoc, TransferEvent> {
        self.writers
    }

    pub open(super) spec fn frontiers(self) -> Map<SlotLoc, Frontier<TransferDomain>> {
        owned_frontiers(self.tokens)
    }

    pub open(super) spec fn well_formed(self) -> bool {
        self.writers.dom() == self.tokens.dom() && owned_map_valid(self.tokens, self.instance)
    }
}

pub enum RegistrationBuffer<'a> {
    Host(&'a HostBuffer),
    CudaHost(&'a CudaHostBuffer),
    Cuda(&'a CudaBuffer),
}

impl<'a> RegistrationBuffer<'a> {
    pub open spec fn node(&self) -> NodeId {
        match self {
            Self::Host(b) => b.node(),
            Self::CudaHost(b) => b.node(),
            Self::Cuda(b) => b.node(),
        }
    }

    pub open spec fn space(&self) -> MemorySpace {
        match self {
            Self::Host(b) => b.space(),
            Self::CudaHost(b) => b.space(),
            Self::Cuda(b) => b.space(),
        }
    }

    pub open spec fn num_slots(&self) -> usize {
        match self {
            Self::Host(b) => b.num_slots(),
            Self::CudaHost(b) => b.num_slots(),
            Self::Cuda(b) => b.num_slots(),
        }
    }

    pub open spec fn slot_len(&self) -> usize {
        match self {
            Self::Host(b) => b.slot_len(),
            Self::CudaHost(b) => b.slot_len(),
            Self::Cuda(b) => b.slot_len(),
        }
    }

    pub open spec fn total_len(&self) -> usize {
        match self {
            Self::Host(b) => b.total_len(),
            Self::CudaHost(b) => b.total_len(),
            Self::Cuda(b) => b.total_len(),
        }
    }

    pub open spec fn slot_loc(&self, idx: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space(), slot_index: idx }
    }

    pub open spec fn slot_ref(&self, idx: nat) -> SlotRef {
        SlotRef::full(self.slot_loc(idx), self.slot_len() as u64)
    }

    pub open spec fn instance_id(&self) -> InstanceId {
        match self {
            Self::Host(b) => b.instance_id(),
            Self::CudaHost(b) => b.instance_id(),
            Self::Cuda(b) => b.instance_id(),
        }
    }

    pub open spec fn well_formed(&self) -> bool {
        match self {
            Self::Host(b) => b.well_formed(),
            Self::CudaHost(b) => b.well_formed(),
            Self::Cuda(b) => b.well_formed(),
        }
    }

    pub open spec fn gpu_device(&self) -> Option<DeviceId> {
        match self {
            Self::Cuda(b) => Some(b.space()->GpuAlloc_device),
            _ => None,
        }
    }
}

pub struct MemoryRegion<'a> {
    pub(super) raw: *mut u8,
    pub(super) addr: u64,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) len: u64,
    pub(super) num_slots: usize,
    pub(super) slot_len: usize,
    pub(super) lkey: u32,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) buffer: RegistrationBuffer<'a>,
    pub(super) pd: &'a ProtectionDomain<'a>,
    pub(super) flags: AccessFlags,
    pub(super) gdr_ordering: Option<GdrWritesOrdering>,
}

pub struct MemoryRegionSlice<'a> {
    pub(super) mr: &'a MemoryRegion<'a>,
    pub(super) addr: u64,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) len: u64,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) slot: Ghost<SlotRef>,
}

impl<'a> MemoryRegion<'a> {
    pub closed spec fn address(&self) -> u64 {
        self.addr
    }

    pub closed spec fn local_key(&self) -> u32 {
        self.lkey
    }

    pub closed spec fn node(&self) -> NodeId {
        self.buffer.node()
    }

    pub closed spec fn space(&self) -> MemorySpace {
        self.buffer.space()
    }

    pub closed spec fn num_slots(&self) -> usize {
        self.num_slots
    }

    pub closed spec fn slot_len(&self) -> usize {
        self.slot_len
    }

    pub closed spec fn total_len(&self) -> u64 {
        self.len
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

    pub closed spec fn pd_id(&self) -> nat {
        self.pd.id()
    }

    pub closed spec fn access(&self) -> AccessFlags {
        self.flags
    }

    pub closed spec fn gdr_ordering_spec(&self) -> Option<GdrWritesOrdering> {
        self.gdr_ordering
    }

    pub closed spec fn gpu_device_spec(&self) -> Option<DeviceId> {
        self.buffer.gpu_device()
    }

    pub open spec fn gdr_visible_to_device_spec(&self, consumer_device: DeviceId) -> bool {
        match self.gdr_ordering_spec() {
            None | Some(GdrWritesOrdering::AllDevices) => true,
            Some(GdrWritesOrdering::Owner) => self.gpu_device_spec() == Some(consumer_device),
            Some(GdrWritesOrdering::None) => false,
        }
    }

    pub open spec fn gdr_visible_after_flush_to_device_spec(
        &self,
        consumer_device: DeviceId,
    ) -> bool {
        match self.gdr_ordering_spec() {
            None | Some(GdrWritesOrdering::AllDevices) => true,
            Some(GdrWritesOrdering::Owner) | Some(GdrWritesOrdering::None) => self.gpu_device_spec()
                == Some(consumer_device),
        }
    }

    pub fn gdr_ordering(&self) -> (o: Option<GdrWritesOrdering>)
        ensures
            o == self.gdr_ordering_spec(),
    {
        self.gdr_ordering
    }

    pub fn gpu_device(&self) -> (d: Option<DeviceId>)
        requires
            self.well_formed(),
        ensures
            d == self.gpu_device_spec(),
    {
        match self.buffer {
            RegistrationBuffer::Cuda(b) => Some(b.device()),
            _ => None,
        }
    }

    /// True if GPUDirect RDMA writes into this region are natively visible to
    /// `consumer_device` without a host-side flush.
    pub fn gdr_visible_to_device(&self, consumer_device: DeviceId) -> (v: bool)
        requires
            self.well_formed(),
        ensures
            v == self.gdr_visible_to_device_spec(consumer_device),
    {
        match self.gdr_ordering {
            None | Some(GdrWritesOrdering::AllDevices) => true,
            Some(GdrWritesOrdering::Owner) => match self.gpu_device() {
                Some(dev) => dev == consumer_device,
                None => false,
            },
            Some(GdrWritesOrdering::None) => false,
        }
    }

    /// True if GPUDirect RDMA writes are visible to `consumer_device` after
    /// `self.flush_gdr_writes()` (which flushes with `ToOwner` scope).
    pub fn gdr_visible_after_flush_to_device(&self, consumer_device: DeviceId) -> (v: bool)
        requires
            self.well_formed(),
        ensures
            v == self.gdr_visible_after_flush_to_device_spec(consumer_device),
    {
        match self.gdr_ordering {
            None | Some(GdrWritesOrdering::AllDevices) => true,
            Some(GdrWritesOrdering::Owner)
            | Some(GdrWritesOrdering::None) => match self.gpu_device() {
                Some(dev) => dev == consumer_device,
                None => false,
            },
        }
    }

    /// Flush GPUDirect RDMA writes to the owning GPU derived from `self.gpu_device()`
    /// when `self.gdr_ordering()` is `Some(GdrWritesOrdering::None)`.
    pub fn flush_gdr_writes(&self) -> (r: Result<(), CudaError>)
        requires
            self.well_formed(),
        ensures
            self.gdr_ordering_spec() != Some(GdrWritesOrdering::None) ==> r is Ok,
    {
        if let Some(GdrWritesOrdering::None) = self.gdr_ordering {
            if let Some(dev_u32) = self.gpu_device() {
                if dev_u32 > i32::MAX as u32 {
                    return Err(CudaError(1));
                }
                gdr_flush_writes(dev_u32 as i32)?;
            }
        }
        Ok(())
    }

    /// Release a single GPU destination `SlotToken` after observing its RDMA READ
    /// completion and flushing GPUDirect RDMA BAR1 writes to the owning GPU.
    pub fn complete_gdr_slot(
        &self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(pending): Tracked<PendingGdrSlot>,
        Tracked(knows): Tracked<&Knows>,
    ) -> (r: Result<Tracked<SlotToken>, (CudaError, Tracked<PendingGdrSlot>)>)
        requires
            self.well_formed(),
            self.gpu_device_spec() is Some,
            pending.well_formed(),
            pending.instance_id() == instance.id(),
            pending.loc().node == self.node(),
            pending.loc().space == self.space(),
            knows.instance_id() == instance.id(),
            knows.set().contains((self.node(), pending.writer())),
        ensures
            r matches Err((_, ret)) ==> ret@ == pending,
            r matches Ok(tok) ==> slot_owned(
                tok@,
                instance.id(),
                pending.loc(),
                pending.frontier(),
            ),
    {
        if let Err(e) = self.flush_gdr_writes() {
            return Err((e, Tracked(pending)));
        }
        Ok(Tracked(pending.token))
    }

    /// Release a batch/SG map of GPU destination `SlotToken`s after observing
    /// their RDMA READ completions and flushing GPUDirect RDMA BAR1 writes once.
    pub fn complete_gdr_slots(
        &self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(pending): Tracked<PendingGdrSlots>,
        Tracked(knows): Tracked<&Knows>,
    ) -> (r: Result<Tracked<Map<SlotLoc, SlotToken>>, (CudaError, Tracked<PendingGdrSlots>)>)
        requires
            self.well_formed(),
            self.gpu_device_spec() is Some,
            pending.well_formed(),
            pending.instance_id() == instance.id(),
            knows.instance_id() == instance.id(),
            forall|l: SlotLoc| #[trigger]
                pending.dom().contains(l) ==> l.node == self.node() && l.space == self.space()
                    && knows.set().contains((self.node(), pending.writers()[l])),
        ensures
            r matches Err((_, ret)) ==> ret@ == pending,
            r matches Ok(toks) ==> owned_map_valid(toks@, instance.id()) && toks@.dom()
                == pending.dom() && owned_frontiers(toks@) == pending.frontiers(),
    {
        if let Err(e) = self.flush_gdr_writes() {
            return Err((e, Tracked(pending)));
        }
        Ok(Tracked(pending.tokens))
    }

    pub fn num_slots_val(&self) -> (n: usize)
        ensures
            n == self.num_slots(),
    {
        self.num_slots
    }

    pub fn slot_len_val(&self) -> (n: usize)
        ensures
            n == self.slot_len(),
    {
        self.slot_len
    }

    pub open(super) spec fn well_formed(&self) -> bool {
        &&& self.buffer.well_formed()
        &&& match self.buffer {
            RegistrationBuffer::Cuda(_) => self.gdr_ordering is Some,
            _ => self.gdr_ordering is None,
        }
        &&& self.num_slots >= 1
        &&& self.num_slots == self.buffer.num_slots()
        &&& self.slot_len == self.buffer.slot_len()
        &&& self.len == (self.num_slots * self.slot_len) as u64
        &&& self.num_slots * self.slot_len <= usize::MAX
        &&& self.addr + self.len <= u64::MAX
    }

    pub fn as_slice(&'a self) -> (s: MemoryRegionSlice<'a>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(0),
            s.registration() == *self,
            s.address() == self.address(),
            s.local_key() == self.local_key(),
    {
        self.slot_slice(0)
    }

    pub fn slot_slice(&'a self, idx: usize) -> (s: MemoryRegionSlice<'a>)
        requires
            self.well_formed(),
            idx < self.num_slots(),
        ensures
            s.well_formed(),
            s.slot_ref() == self.slot_ref(idx as nat),
            s.registration() == *self,
            s.address() == (self.address() + idx * self.slot_len()) as u64,
            s.local_key() == self.local_key(),
    {
        assert!(idx < self.num_slots);
        proof {
            lemma_mul_is_distributive_add_other_way(self.slot_len as int, idx as int, 1);
            lemma_mul_inequality((idx + 1) as int, self.num_slots as int, self.slot_len as int);
        }
        let off = (idx * self.slot_len) as u64;
        MemoryRegionSlice {
            mr: self,
            addr: self.addr + off,
            len: self.slot_len as u64,
            slot: Ghost(self.slot_ref(idx as nat)),
        }
    }
}

impl<'a> MemoryRegionSlice<'a> {
    pub closed spec fn address(&self) -> u64 {
        self.addr
    }

    pub closed spec fn local_key(&self) -> u32 {
        self.mr.local_key()
    }

    pub open(super) spec fn slot_ref(&self) -> SlotRef {
        self.slot@
    }

    pub open(super) spec fn registration(&self) -> MemoryRegion<'a> {
        *self.mr
    }

    pub open(super) spec fn well_formed(&self) -> bool {
        &&& self.addr + self.len <= u64::MAX
        &&& self.slot@.well_formed()
        &&& self.len == self.slot@.sub_len
        &&& self.slot@.slot_len == self.mr.slot_len() as u64
        &&& self.slot@.loc.node == self.mr.node()
        &&& self.slot@.loc.space == self.mr.space()
        &&& self.slot@.loc.slot_index < self.mr.num_slots()
    }

    pub fn len(&self) -> (n: u64)
        requires
            self.well_formed(),
        ensures
            n == self.slot_ref().sub_len,
    {
        self.len
    }

    pub fn sub_slice(&self, offset: u64, len: u64) -> (s: Self)
        requires
            self.well_formed(),
            offset + len <= self.slot_ref().sub_len,
        ensures
            s.well_formed(),
            s.registration() == self.registration(),
            s.address() == self.address() + offset,
            s.local_key() == self.local_key(),
            s.slot_ref() == self.slot_ref().sub_slice(offset, len),
    {
        assert!(offset <= self.len && len <= self.len - offset);
        Self {
            mr: self.mr,
            addr: self.addr + offset,
            len,
            slot: Ghost(self.slot_ref().sub_slice(offset, len)),
        }
    }
}

pub fn register_memory<'a>(
    pd: &'a ProtectionDomain<'a>,
    buffer: RegistrationBuffer<'a>,
    flags: AccessFlags,
) -> (r: Result<MemoryRegion<'a>, RdmaError>)
    requires
        buffer.well_formed(),
        buffer.node() == pd.node(),
    ensures
        r matches Ok(m) ==> m.well_formed() && m.node() == buffer.node() && m.space()
            == buffer.space() && m.num_slots() == buffer.num_slots() && m.slot_len()
            == buffer.slot_len() && m.total_len() == buffer.total_len() as u64 && m.instance_id()
            == buffer.instance_id() && m.pd_id() == pd.id() && m.access() == flags
            && m.gpu_device_spec() == buffer.gpu_device(),
{
    let (ptr, len, num_slots, slot_len, gdr_ordering) = match &buffer {
        RegistrationBuffer::Host(b) => {
            let (p, l) = b.raw_parts();
            (p, l, b.num_slots_val(), b.slot_len_val(), None)
        },
        RegistrationBuffer::CudaHost(b) => {
            let (p, l) = b.raw_parts();
            (p, l, b.num_slots_val(), b.slot_len_val(), None)
        },
        RegistrationBuffer::Cuda(b) => {
            if let Err(e) = crate::cuda::enable_sync_memops(b) {
                return Err(RdmaError(e.0));
            }
            let dev_u32 = b.device();
            if dev_u32 > i32::MAX as u32 {
                return Err(RdmaError(22));
            }
            let dev = dev_u32 as i32;
            let ord = match gdr_writes_ordering(dev) {
                Ok(o) => o,
                Err(e) => return Err(RdmaError(e.0)),
            };
            if let GdrWritesOrdering::None = ord {
                match gdr_flush_writes_supported(dev) {
                    Ok(true) => {},
                    Ok(false) => return Err(RdmaError(95)),
                    Err(e) => return Err(RdmaError(e.0)),
                }
            }
            let (p, l) = b.raw_parts();
            (p, l, b.num_slots_val(), b.slot_len_val(), Some(ord))
        },
    };
    let (raw, lkey) = ffi::register(pd.raw, ptr, len, flags)?;
    Ok(
        MemoryRegion {
            raw,
            addr: ptr.addr() as u64,
            len: len as u64,
            num_slots,
            slot_len,
            lkey,
            buffer,
            pd,
            flags,
            gdr_ordering,
        },
    )
}

pub fn deregister_memory<'a>(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: MemoryRegion<'a>,
    Tracked(tok): Tracked<SlotToken>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Tracked<Completion>, Tracked<SlotToken>),
    (RdmaError, MemoryRegion<'a>, Tracked<SlotToken>),
>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.num_slots() == 1,
        buf.instance_id() == instance.id(),
        host::can_reclaim_slot(instance.id(), buf.slot_loc(0), tok, *infos, *knows, cited),
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
    let Tracked(completed) = match ffi::raw_deregister_memory(&buf) {
        Err(e) => return Err((e, buf, Tracked(tok))),
        Ok(w) => w,
    };
    proof {
        assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
        assert(completed.access == dereg_access(loc));
    }
    let tracked completion = host::finish_owned(
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

pub fn deregister_memory_slab<'a>(
    _ctx: &mut HostContext,
    Tracked(instance): Tracked<&TransferInstance>,
    buf: MemoryRegion<'a>,
    Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    Tracked(infos): Tracked<&SubmittedMap>,
    Tracked(knows): Tracked<&Knows>,
    Ghost(cited): Ghost<ISet<TransferEvent>>,
) -> (r: Result<
    (Tracked<Completion>, Tracked<Map<usize, SlotToken>>),
    (RdmaError, MemoryRegion<'a>, Tracked<Map<usize, SlotToken>>),
>)
    requires
        old(_ctx).ready(instance.id(), buf.node()),
        buf.well_formed(),
        buf.instance_id() == instance.id(),
        host::can_reclaim_slab(
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
    let Tracked(completed) = match ffi::raw_deregister_memory(&buf) {
        Err(e) => return Err((e, buf, Tracked(tokens))),
        Ok(w) => w,
    };
    let tracked (completion, out_tokens) = host::finish_slab_owned(
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
