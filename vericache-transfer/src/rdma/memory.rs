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
    pub proof fn empty(instance: InstanceId) -> (tracked batch: Self)
        ensures
            batch.well_formed(),
            batch.instance_id() == instance,
            batch.dom().is_empty(),
    {
        Self { instance, writers: Map::empty(), tokens: Map::tracked_empty() }
    }

    pub proof fn insert(tracked &mut self, tracked slot: PendingGdrSlot)
        requires
            old(self).well_formed(),
            slot.well_formed(),
            slot.instance_id() == old(self).instance_id(),
            !old(self).dom().contains(slot.loc()),
        ensures
            final(self).well_formed(),
            final(self).instance_id() == old(self).instance_id(),
            final(self).dom() == old(self).dom().insert(slot.loc()),
            final(self).writers() == old(self).writers().insert(slot.loc(), slot.writer()),
            final(self).frontiers() == old(self).frontiers().insert(slot.loc(), slot.frontier()),
    {
        let loc = slot.loc();
        self.writers = self.writers.insert(loc, slot.writer());
        self.tokens.tracked_insert(loc, slot.token);
        assert(self.frontiers() =~= old(self).frontiers().insert(loc, slot.frontier()));
    }

    pub proof fn remove(tracked &mut self, loc: SlotLoc) -> (tracked slot: PendingGdrSlot)
        requires
            old(self).well_formed(),
            old(self).dom().contains(loc),
        ensures
            final(self).well_formed(),
            final(self).instance_id() == old(self).instance_id(),
            final(self).dom() == old(self).dom().remove(loc),
            final(self).writers() == old(self).writers().remove(loc),
            final(self).frontiers() == old(self).frontiers().remove(loc),
            slot.well_formed(),
            slot.instance_id() == old(self).instance_id(),
            slot.loc() == loc,
            slot.writer() == old(self).writers()[loc],
            slot.frontier() == old(self).frontiers()[loc],
    {
        let writer = self.writers[loc];
        self.writers = self.writers.remove(loc);
        let tracked token = self.tokens.tracked_remove(loc);
        assert(self.frontiers() =~= old(self).frontiers().remove(loc));
        PendingGdrSlot { writer, token }
    }

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

/// Erase the slab type for registrations that need a post-CQE flush.
pub(super) trait GdrFlush {
    fn flush(&self, device: i32) -> Result<(), i32>;
}

impl<B: RdmaRegisterableSlab> GdrFlush for B {
    fn flush(&self, device: i32) -> Result<(), i32> {
        self.flush_gdr_writes(device)
    }
}

pub struct MemoryRegion<'a> {
    pub(super) raw: ffi::RawMrGuard,
    pub(super) addr: u64,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) len: u64,
    pub(super) num_slots: usize,
    pub(super) slot_len: usize,
    pub(super) lkey: u32,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) node: Ghost<NodeId>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) space: Ghost<MemorySpace>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) instance: Ghost<InstanceId>,
    pub(super) gpu_device: Option<DeviceId>,
    pub(super) borrow: PhantomData<&'a ()>,
    pub(super) pd: &'a ProtectionDomain<'a>,
    pub(super) flags: AccessFlags,
    pub(super) gdr_ordering: Option<GdrWritesOrdering>,
    pub(super) gdr_flush_supported: bool,
    pub(super) gdr_flush: Option<&'a dyn GdrFlush>,
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
    pub fn protection_domain(&self) -> &'a ProtectionDomain<'a> {
        self.pd
    }

    pub closed spec fn address(&self) -> u64 {
        self.addr
    }

    pub closed spec fn local_key(&self) -> u32 {
        self.lkey
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
        self.instance@
    }

    pub closed spec fn pd_id(&self) -> nat {
        self.pd.id()
    }

    pub fn access_flags(&self) -> (flags: AccessFlags)
        ensures
            flags == self.access(),
    {
        self.flags
    }

    pub closed spec fn access(&self) -> AccessFlags {
        self.flags
    }

    pub closed spec fn gdr_ordering_spec(&self) -> Option<GdrWritesOrdering> {
        self.gdr_ordering
    }

    pub closed spec fn gpu_device_spec(&self) -> Option<DeviceId> {
        self.gpu_device
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
        self.gpu_device
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
    pub fn flush_gdr_writes(&self) -> (r: Result<(), RdmaError>)
        requires
            self.well_formed(),
        ensures
            self.gdr_ordering_spec() != Some(GdrWritesOrdering::None) ==> r is Ok,
    {
        let Some(GdrWritesOrdering::None) = self.gdr_ordering else {
            return Ok(());
        };
        if let (Some(device), Some(flush)) = (self.gpu_device, self.gdr_flush) {
            if self.gdr_flush_supported && device <= i32::MAX as u32 {
                return flush.flush(device as i32).map_err(|code| RdmaError(code));
            }
        }
        Err(RdmaError(95))
    }

    /// Release a single GPU destination `SlotToken` after observing its RDMA READ
    /// completion and flushing GPUDirect RDMA BAR1 writes to the owning GPU.
    pub fn complete_gdr_slot(
        &self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(pending): Tracked<PendingGdrSlot>,
        Tracked(knows): Tracked<&Knows>,
    ) -> (r: Result<Tracked<SlotToken>, (RdmaError, Tracked<PendingGdrSlot>)>)
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
    ) -> (r: Result<Tracked<Map<SlotLoc, SlotToken>>, (RdmaError, Tracked<PendingGdrSlots>)>)
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

    pub open spec fn well_formed(&self) -> bool {
        &&& match self.gpu_device_spec() {
            None => self.space().is_host() && self.gdr_ordering_spec().is_none(),
            Some(d) => self.space().is_gpu() && self.space().gpu_device() == d && d <= i32::MAX
                && self.gdr_ordering_spec().is_some(),
        }
        &&& (self.gdr_ordering_spec() == Some(GdrWritesOrdering::None) ==> self.has_flush_hook())
        &&& self.num_slots() >= 1
        &&& self.total_len() == (self.num_slots() * self.slot_len()) as u64
        &&& self.num_slots() * self.slot_len() <= usize::MAX
        &&& self.address() + self.total_len() <= u64::MAX
    }

    pub closed spec fn has_flush_hook(&self) -> bool {
        self.gdr_flush_supported && self.gdr_flush.is_some()
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

pub fn register_memory<'a, B: RdmaRegisterableSlab>(
    pd: &'a ProtectionDomain<'a>,
    buffer: &'a B,
    flags: AccessFlags,
) -> (r: Result<MemoryRegion<'a>, RdmaError>)
    requires
        buffer.reg_well_formed(),
        buffer.reg_node() == pd.node(),
    ensures
        r matches Ok(m) ==> m.well_formed() && m.node() == buffer.reg_node() && m.space()
            == buffer.reg_space() && m.num_slots() == buffer.reg_num_slots() && m.slot_len()
            == buffer.reg_slot_len() && m.total_len() == buffer.reg_total_len() as u64
            && m.instance_id() == buffer.reg_instance_id() && m.pd_id() == pd.id() && m.access()
            == flags && m.gpu_device_spec() == buffer.reg_gpu_device(),
{
    let allocation = buffer.reg_allocation();
    let (ptr, len) = allocation.raw_parts();
    let num_slots = buffer.reg_num_slots_exec();
    let slot_len = buffer.reg_slot_len_exec();
    let gpu_device = allocation.device();
    if num_slots == 0 || num_slots.checked_mul(slot_len) != Some(len) {
        return Err(RdmaError(22));
    }
    buffer.prepare_rdma_registration().map_err(|code| RdmaError(code))?;
    if let Some(device) = gpu_device {
        if device > i32::MAX as u32 {
            return Err(RdmaError(22));
        }
    }
    let gdr_ordering = buffer.query_gdr_ordering().map_err(|code| RdmaError(code))?;
    let (gdr_flush_supported, gdr_flush) = if let Some(GdrWritesOrdering::None) = gdr_ordering {
        let supported = buffer.query_gdr_flush_supported().map_err(|code| RdmaError(code))?;
        if !supported {
            return Err(RdmaError(95));
        }
        (true, Some(buffer as &dyn GdrFlush))
    } else {
        (false, None)
    };
    let (raw, lkey) = ffi::register(pd.raw.as_ptr(), ptr, len, flags)?;
    Ok(
        MemoryRegion {
            raw,
            addr: ptr.addr() as u64,
            len: len as u64,
            num_slots,
            slot_len,
            lkey,
            node: Ghost(buffer.reg_node()),
            space: Ghost(buffer.reg_space()),
            instance: Ghost(buffer.reg_instance_id()),
            gpu_device,
            borrow: PhantomData,
            pd,
            flags,
            gdr_ordering,
            gdr_flush_supported,
            gdr_flush,
        },
    )
}

pub fn deregister_memory<'a>(
    _ctx: &mut SyncContext,
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
        can_reclaim_slot(instance.id(), buf.slot_loc(0), tok, *infos, *knows, cited),
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
    let _ = buf.raw.into_raw();
    proof {
        assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
        assert(completed.access() == dereg_access(loc));
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
    Ok((Tracked(completion), Tracked(tok)))
}

pub fn deregister_memory_slab<'a>(
    _ctx: &mut SyncContext,
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
    let _ = buf.raw.into_raw();
    let tracked (completion, out_tokens) = finish_slab_owned(
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
