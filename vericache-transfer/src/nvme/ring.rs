use super::*;

verus! {

broadcast use {vstd::imap::group_imap_lemmas, vstd::iset::group_iset_lemmas};

pub ghost struct PendingIo {
    pub event: TransferEvent,
    pub access: Access<TransferDomain>,
    pub effect: Effect<TransferDomain>,
    pub preds: ISet<TransferEvent>,
    pub read_guards: ReadGuards,
    pub expected_len: u32,
}

impl PendingIo {
    pub open spec fn info(self) -> EventInfo<TransferDomain> {
        EventInfo {
            access: self.access,
            effect: self.effect,
            preds: self.preds,
            read_guards: self.read_guards,
        }
    }
}

pub struct IoUring {
    pub(super) raw: ffi::IoUringGuard,
    sq_capacity: usize,
    max_in_flight: usize,
    source: Tracked<EventSourceToken>,
    pending: Ghost<Map<u64, PendingIo>>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pending_submitted: Tracked<SubmittedMap>,
    staged: Ghost<Seq<u64>>,
    pub(super) staged_count: usize,
    pending_count: usize,
}

pub enum Completed {
    Data {
        user_data: u64,
        bytes: u32,
        event: Ghost<TransferEvent>,
        submitted: Tracked<Submitted>,
        knows: Tracked<Knows>,
        succeeded: Tracked<Succeeded>,
    },
    DataFailed {
        user_data: u64,
        res: i32,
        event: Ghost<TransferEvent>,
        submitted: Tracked<Submitted>,
        knows: Tracked<Knows>,
    },
}

impl Completed {
    pub open spec fn describes(self, before: IoUring, after: IoUring) -> bool {
        let (id, ev, submitted, knows) = match self {
            Self::Data { user_data, event, submitted, knows, .. }
            | Self::DataFailed { user_data, event, submitted, knows, .. } => (
                user_data,
                event@,
                submitted@,
                knows@,
            ),
        };
        let io = before.pending()[id];
        &&& before.pending().contains_key(id) && !before.staged().contains(id)
        &&& after.pending() == before.pending().remove(id) && after.staged() == before.staged()
        &&& io.event == ev && submitted.key() == ev && submitted.value() == io.info()
        &&& submitted.instance_id() == before.instance_id() && knows.instance_id()
            == before.instance_id()
        &&& knows.set() == knowledge_of(before.node(), io.preds.insert(ev))
        &&& match self {
            Self::Data { bytes, succeeded, .. } => bytes == io.expected_len && succeeded@.event()
                == ev && succeeded@.instance_id() == before.instance_id(),
            Self::DataFailed { res, .. } => res != io.expected_len,
        }
    }
}

impl IoUring {
    pub closed spec fn source(&self) -> EventSourceToken {
        self.source@
    }

    pub open spec fn instance_id(&self) -> InstanceId {
        self.source().id()
    }

    pub open spec fn node(&self) -> NodeId {
        self.source().node()
    }

    pub closed spec fn sq_capacity(&self) -> usize {
        self.sq_capacity
    }

    pub closed spec fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    pub closed spec fn staged(&self) -> Seq<u64> {
        self.staged@
    }

    pub closed spec fn pending(&self) -> Map<u64, PendingIo> {
        self.pending@
    }

    pub open spec fn staged_count(&self) -> nat {
        self.staged().len()
    }

    pub open spec fn pending_count(&self) -> nat {
        self.pending().len()
    }

    pub open spec fn in_flight(&self) -> nat {
        (self.pending_count() - self.staged_count()) as nat
    }

    pub fn sq_capacity_val(&self) -> (r: usize)
        ensures
            r == self.sq_capacity(),
    {
        self.sq_capacity
    }

    pub fn max_in_flight_val(&self) -> (r: usize)
        ensures
            r == self.max_in_flight(),
    {
        self.max_in_flight
    }

    pub fn staged_len(&self) -> (r: usize)
        requires
            self.well_formed(),
        ensures
            r == self.staged_count(),
    {
        self.staged_count
    }

    pub fn pending_len(&self) -> (r: usize)
        requires
            self.well_formed(),
        ensures
            r == self.pending_count(),
    {
        self.pending_count
    }

    pub fn in_flight_len(&self) -> (r: usize)
        requires
            self.well_formed(),
        ensures
            r == self.in_flight(),
    {
        self.pending_count - self.staged_count
    }

    pub closed spec fn well_formed(&self) -> bool {
        &&& self.source@.well_formed() && self.source@.source() is IoUring
        &&& self.pending_submitted@.instance_id() == self.instance_id()
        &&& 0 < self.sq_capacity <= self.max_in_flight <= u32::MAX
        &&& self.staged_count == self.staged().len() && self.pending_count == self.pending().len()
        &&& self.staged_count <= self.sq_capacity && self.staged_count <= self.pending_count
            <= self.max_in_flight
        &&& self.staged().no_duplicates()
        &&& forall|i: int|
            0 <= i < self.staged_count ==> self.pending().contains_key(#[trigger] self.staged()[i])
        &&& forall|k: u64|
            self.pending().contains_key(k) ==> {
                let io = #[trigger] self.pending()[k];
                &&& io.event.seq < self.source@.next_seq() && io.event.observer == self.node()
                    && io.event.source == self.source@.source()
                &&& self.pending_submitted@.dom().contains(io.event)
                    && self.pending_submitted@.map()[io.event] == io.info()
                &&& 0 < io.expected_len <= MAX_IO_LEN
            }
        &&& forall|a: u64, b: u64|
            self.pending().contains_key(a) && self.pending().contains_key(b) && (
            #[trigger] self.pending()[a]).event == (#[trigger] self.pending()[b]).event ==> a == b
    }

    pub open spec fn same_ring(self, prior: Self) -> bool {
        &&& self.source().source() == prior.source().source() && self.node() == prior.node()
        &&& self.instance_id() == prior.instance_id() && self.sq_capacity() == prior.sq_capacity()
        &&& self.max_in_flight() == prior.max_in_flight()
    }

    pub fn new(Tracked(source): Tracked<EventSourceToken>, entries: u32) -> (r: Result<
        Self,
        (NvmeError, Tracked<EventSourceToken>),
    >)
        requires
            source.well_formed(),
            source.source() is IoUring,
            entries > 0,
        ensures
            match r {
                Ok(q) => q.well_formed() && q.source() == source && q.pending() == Map::empty()
                    && q.staged() == Seq::empty() && 0 < q.sq_capacity() <= q.max_in_flight(),
                Err((_, s)) => s@ == source,
            },
    {
        let (raw, sq, cq) = match ffi::new(entries) {
            Ok(v) => v,
            Err(e) => return Err((e, Tracked(source))),
        };
        Ok(
            Self {
                raw,
                sq_capacity: sq as usize,
                max_in_flight: cq as usize,
                pending_submitted: Tracked(SubmittedMap::empty(source.id())),
                source: Tracked(source),
                pending: Ghost(Map::empty()),
                staged: Ghost(Seq::empty()),
                staged_count: 0,
                pending_count: 0,
            },
        )
    }

    pub fn destroy(self) -> (s: Tracked<EventSourceToken>)
        requires
            self.well_formed(),
            self.pending().is_empty(),
        ensures
            s@ == self.source(),
            s@.well_formed(),
            s@.source() is IoUring,
    {
        ffi::destroy(self.raw);
        self.source
    }

    pub open spec fn can_stage(
        self,
        src: SlotRef,
        dst: SlotRef,
        host_address: usize,
        file_offset: u64,
        id: u64,
        src_perm: ReadPerm,
        dst_tok: SlotToken,
        infos: SubmittedMap,
        knows: Knows,
        cited: ISet<TransferEvent>,
    ) -> bool {
        let ordered = closure_from(infos.map(), cited);
        &&& self.well_formed() && src.loc.node == self.node() && dst.loc.node == self.node()
        &&& src.loc != dst.loc
        &&& 0 < src.sub_len <= MAX_IO_LEN && src.sub_len == dst.sub_len
        &&& host_address % (DIRECT_IO_ALIGNMENT as usize) == 0 && file_offset % DIRECT_IO_ALIGNMENT
            == 0 && src.sub_len % DIRECT_IO_ALIGNMENT == 0
        &&& !self.pending().contains_key(id) && self.staged_count() < self.sq_capacity()
            && self.pending_count() < self.max_in_flight()
        &&& infos.instance_id() == self.instance_id()
        &&& knows.instance_id() == self.instance_id()
        &&& cited.subset_of(infos.dom())
        &&& knowledge_of(self.node(), cited).subset_of(knows.set())
        &&& src_perm.valid_local(self.instance_id(), src.loc)
        &&& slot_is_owned(dst_tok, self.instance_id(), dst.loc)
        &&& read_admitted(src_perm.frontier(), ordered)
        &&& write_admitted(slot_frontier(dst_tok), ordered)
    }

    pub open spec fn stage_result(
        self,
        prior: Self,
        src: SlotRef,
        dst: SlotRef,
        id: u64,
        old_src_perm: ReadPerm,
        new_src_perm: ReadPerm,
        old_dst_tok: SlotToken,
        new_dst_tok: SlotToken,
        infos: SubmittedMap,
        cited: ISet<TransferEvent>,
        result: Result<Tracked<Submitted>, NvmeError>,
    ) -> bool {
        let access = copy_access(src, dst);
        let ev = prior.source().next_event();
        let guards = old_src_perm.read_guards();
        match result {
            Err(_) => self == prior && new_src_perm == old_src_perm && new_dst_tok == old_dst_tok,
            Ok(receipt) => {
                let io = PendingIo {
                    event: ev,
                    access,
                    effect: copy_effect(src, dst),
                    preds: closure_from(infos.map(), cited),
                    read_guards: guards,
                    expected_len: src.sub_len as u32,
                };
                &&& self.well_formed() && self.same_ring(prior) && self.source().next_seq()
                    == ev.seq + 1
                &&& self.staged() == prior.staged().push(id)
                &&& self.pending() == prior.pending().insert(id, io)
                &&& receipt@.instance_id() == self.instance_id()
                &&& receipt@.key() == ev && receipt@.value() == io.info()
                &&& new_src_perm.advanced_read(old_src_perm, ev)
                &&& slot_owned(
                    new_dst_tok,
                    self.instance_id(),
                    dst.loc,
                    Frontier { last_writer: Some(ev), readers: ISet::empty() },
                )
            },
        }
    }

    pub fn stage_read(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        dst: &HostSlice,
        src: &NvmeSlice,
        user_data: u64,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
        Tracked(infos): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Ghost(cited): Ghost<ISet<TransferEvent>>,
    ) -> (result: Result<Tracked<Submitted>, NvmeError>)
        requires
            old(self).instance_id() == instance.id(),
            dst.well_formed(),
            src.well_formed(),
            dst.instance_id() == instance.id(),
            src.instance_id() == instance.id(),
            old(self).can_stage(
                src.slot_ref(),
                dst.slot_ref(),
                dst.address(),
                src.file_offset(),
                user_data,
                *old(src_perm),
                *old(dst_tok),
                *infos,
                *knows,
                cited,
            ),
        ensures
            final(self).stage_result(
                *old(self),
                src.slot_ref(),
                dst.slot_ref(),
                user_data,
                *old(src_perm),
                *final(src_perm),
                *old(dst_tok),
                *final(dst_tok),
                *infos,
                cited,
                result,
            ),
    {
        self.stage(
            Tracked(instance),
            dst,
            src,
            false,
            user_data,
            Tracked(src_perm),
            Tracked(dst_tok),
            Tracked(infos),
            Tracked(knows),
            Ghost(cited),
        )
    }

    pub fn stage_write(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        dst: &NvmeSlice,
        src: &HostSlice,
        user_data: u64,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
        Tracked(infos): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Ghost(cited): Ghost<ISet<TransferEvent>>,
    ) -> (result: Result<Tracked<Submitted>, NvmeError>)
        requires
            old(self).instance_id() == instance.id(),
            src.well_formed(),
            dst.well_formed(),
            src.instance_id() == instance.id(),
            dst.instance_id() == instance.id(),
            old(self).can_stage(
                src.slot_ref(),
                dst.slot_ref(),
                src.address(),
                dst.file_offset(),
                user_data,
                *old(src_perm),
                *old(dst_tok),
                *infos,
                *knows,
                cited,
            ),
        ensures
            final(self).stage_result(
                *old(self),
                src.slot_ref(),
                dst.slot_ref(),
                user_data,
                *old(src_perm),
                *final(src_perm),
                *old(dst_tok),
                *final(dst_tok),
                *infos,
                cited,
                result,
            ),
    {
        self.stage(
            Tracked(instance),
            src,
            dst,
            true,
            user_data,
            Tracked(src_perm),
            Tracked(dst_tok),
            Tracked(infos),
            Tracked(knows),
            Ghost(cited),
        )
    }

    spec fn endpoints(host: SlotRef, file: SlotRef, write: bool) -> (SlotRef, SlotRef) {
        if write {
            (host, file)
        } else {
            (file, host)
        }
    }

    fn stage(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
        host: &HostSlice,
        file: &NvmeSlice,
        write: bool,
        user_data: u64,
        Tracked(src_perm): Tracked<&mut ReadPerm>,
        Tracked(dst_tok): Tracked<&mut SlotToken>,
        Tracked(infos): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Ghost(cited): Ghost<ISet<TransferEvent>>,
    ) -> (result: Result<Tracked<Submitted>, NvmeError>)
        requires
            old(self).instance_id() == instance.id(),
            host.well_formed(),
            file.well_formed(),
            host.instance_id() == instance.id(),
            file.instance_id() == instance.id(),
            old(self).can_stage(
                Self::endpoints(host.slot_ref(), file.slot_ref(), write).0,
                Self::endpoints(host.slot_ref(), file.slot_ref(), write).1,
                host.address(),
                file.file_offset(),
                user_data,
                *old(src_perm),
                *old(dst_tok),
                *infos,
                *knows,
                cited,
            ),
        ensures
            final(self).stage_result(
                *old(self),
                Self::endpoints(host.slot_ref(), file.slot_ref(), write).0,
                Self::endpoints(host.slot_ref(), file.slot_ref(), write).1,
                user_data,
                *old(src_perm),
                *final(src_perm),
                *old(dst_tok),
                *final(dst_tok),
                *infos,
                cited,
                result,
            ),
    {
        let (ptr, len) = host.raw_parts();
        let ghost (src, dst) = Self::endpoints(host.slot_ref(), file.slot_ref(), write);
        if let Err(e) = ffi::stage(self, file, ptr, len as u32, user_data, write) {
            return Err(e);
        }
        let ghost access = copy_access(src, dst);
        let ghost guards = src_perm.read_guards();
        let tracked name = self.source.borrow_mut().take_next();
        proof {
            assert(ISet::empty().union(cited) =~= cited);
        }
        let tracked receipt = instance.submit_2(
            self.node(),
            access,
            copy_effect(src, dst),
            ISet::empty(),
            cited,
            infos,
            knows,
            name,
            src_perm,
            dst_tok,
        );
        proof {
            self.pending_submitted.borrow_mut().insert(receipt);
            self.pending = Ghost(
                self.pending().insert(
                    user_data,
                    PendingIo {
                        event: receipt.key(),
                        access,
                        effect: copy_effect(src, dst),
                        preds: receipt.value().preds,
                        read_guards: guards,
                        expected_len: len as u32,
                    },
                ),
            );
            self.staged = Ghost(self.staged().push(user_data));
        }
        self.pending_count += 1;
        self.staged_count += 1;
        Ok(Tracked(receipt))
    }

    pub fn submit(&mut self) -> (r: Result<usize, NvmeError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).same_ring(*old(self)),
            final(self).source() == old(self).source(),
            final(self).pending() == old(self).pending(),
            match r {
                Ok(n) => n <= old(self).staged_count() && final(self).staged() == old(
                    self,
                ).staged().skip(n as int),
                Err(_) => *final(self) == *old(self),
            },
    {
        if self.staged_count == 0 {
            assert(self.staged().skip(0) =~= self.staged());
            return Ok(0);
        }
        let n = ffi::submit(self)?;
        self.staged = Ghost(self.staged().skip(n as int));
        self.staged_count -= n;
        Ok(n)
    }

    pub fn poll_one(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Option<
        Completed,
    >)
        requires
            old(self).well_formed(),
            old(self).instance_id() == instance.id(),
        ensures
            final(self).well_formed(),
            final(self).same_ring(*old(self)),
            final(self).source() == old(self).source(),
            match r {
                Some(c) => c.describes(*old(self), *final(self)),
                None => *final(self) == *old(self),
            },
    {
        if self.pending_count == self.staged_count {
            return None;
        }
        let (id, len, res) = ffi::poll(self)?;
        let ghost io = self.pending()[id];
        let ghost ev = io.event;
        let ghost one = ISet::empty().insert(ev);
        let tracked knows = instance.observe(self.node(), one, self.pending_submitted.borrow());
        proof {
            assert forall|e: TransferEvent| #[trigger] io.preds.contains(e) implies closure_from(
                self.pending_submitted@.map(),
                one,
            ).contains(e) by {
                assert(self.pending_submitted@.map()[ev].preds.contains(e));
            }
            assert(closure_from(self.pending_submitted@.map(), one) =~= io.preds.insert(ev));
        }
        let tracked submitted = self.pending_submitted.borrow_mut().remove(ev);
        self.pending = Ghost(self.pending().remove(id));
        self.pending_count -= 1;
        if res == len as i32 {
            Some(
                Completed::Data {
                    user_data: id,
                    bytes: len,
                    event: Ghost(ev),
                    submitted: Tracked(submitted),
                    knows: Tracked(knows),
                    succeeded: Tracked(Succeeded::mint(instance.id(), ev)),
                },
            )
        } else {
            Some(
                Completed::DataFailed {
                    user_data: id,
                    res,
                    event: Ghost(ev),
                    submitted: Tracked(submitted),
                    knows: Tracked(knows),
                },
            )
        }
    }

    #[verifier::exec_allows_no_decreases_clause]
    pub fn wait_one(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        Completed,
        NvmeError,
    >)
        requires
            old(self).well_formed(),
            old(self).instance_id() == instance.id(),
            old(self).in_flight() > 0,
        ensures
            final(self).well_formed(),
            final(self).same_ring(*old(self)),
            final(self).source() == old(self).source(),
            match r {
                Ok(c) => c.describes(*old(self), *final(self)),
                Err(_) => *final(self) == *old(self),
            },
    {
        loop
            invariant
                *self == *old(self),
                self.well_formed(),
                self.instance_id() == instance.id(),
                self.in_flight() > 0,
        {
            if let Some(c) = self.poll_one(Tracked(instance)) {
                return Ok(c);
            }
            ffi::wait(self)?;
        }
    }
}

} // verus!
