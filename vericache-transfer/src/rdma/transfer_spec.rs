use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
    vstd::set::group_set_lemmas,
};

pub open spec fn transfer_access(read: bool, local: SlotRef, remote: SlotRef) -> Access<
    TransferDomain,
> {
    if read {
        copy_access(remote, local)
    } else {
        copy_access(local, remote)
    }
}

pub open spec fn sge_total_len(slices: Seq<MemoryRegionSlice>, n: int) -> nat
    decreases n,
{
    if n <= 0 || n > slices.len() {
        0
    } else {
        sge_total_len(slices, n - 1) + slices[n - 1].slot_ref().sub_len as nat
    }
}

pub open spec fn sge_locs(slices: Seq<MemoryRegionSlice>) -> ISet<SlotLoc> {
    ISet::new(
        |l: SlotLoc|
            exists|i: int| 0 <= i < slices.len() && (#[trigger] slices[i].slot_ref()).loc == l,
    )
}

pub open spec fn sge_partial_locs(slices: Seq<MemoryRegionSlice>) -> ISet<SlotLoc> {
    ISet::new(
        |l: SlotLoc|
            exists|i: int|
                0 <= i < slices.len() && (#[trigger] slices[i].slot_ref()).loc == l
                    && !slices[i].slot_ref().is_full(),
    )
}

pub open spec fn sges_distinct_locs(slices: Seq<MemoryRegionSlice>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < slices.len() ==> (#[trigger] slices[i].slot_ref()).loc != (
        #[trigger] slices[j].slot_ref()).loc
}

pub open spec fn sge_transfer_access(
    read: bool,
    slices: Seq<MemoryRegionSlice>,
    remote: SlotRef,
) -> Access<TransferDomain> {
    let local = sge_locs(slices);
    if read {
        Access { reads: sge_partial_locs(slices).insert(remote.loc), writes: local }
    } else {
        Access {
            reads: if remote.is_full() {
                local
            } else {
                local.insert(remote.loc)
            },
            writes: ISet::empty().insert(remote.loc),
        }
    }
}

/// Zero-byte-quantifier scatter/gather copy effect over SGE slice indices.
pub open spec fn sge_copy_effect(
    read: bool,
    slices: Seq<MemoryRegionSlice>,
    remote: SlotRef,
) -> Effect<TransferDomain> {
    |input: Snapshot<TransferDomain>, output: Snapshot<TransferDomain>|
        if read {
            forall|i: int|
                0 <= i < slices.len() ==> {
                    let dst = (#[trigger] slices[i]).slot_ref();
                    let off = sge_total_len(slices, i) as u64;
                    (copy_effect(remote.sub_slice(off, dst.sub_len), dst))(input, output)
                }
        } else {
            &&& output[remote.loc].len() == remote.slot_len
            &&& (input[remote.loc].len() == remote.slot_len ==> (output[remote.loc].subrange(
                0,
                remote.sub_offset as int,
            ) == input[remote.loc].subrange(0, remote.sub_offset as int)
                && output[remote.loc].subrange(
                (remote.sub_offset + remote.sub_len) as int,
                remote.slot_len as int,
            ) == input[remote.loc].subrange(
                (remote.sub_offset + remote.sub_len) as int,
                remote.slot_len as int,
            )))
            &&& forall|i: int|
                0 <= i < slices.len() ==> {
                    let src = (#[trigger] slices[i]).slot_ref();
                    let off = sge_total_len(slices, i) as u64;
                    &&& input[src.loc].len() >= src.sub_offset + src.sub_len
                    &&& output[remote.loc].subrange(
                        (remote.sub_offset + off) as int,
                        (remote.sub_offset + off + src.sub_len) as int,
                    ) == input[src.loc].subrange(
                        src.sub_offset as int,
                        (src.sub_offset + src.sub_len) as int,
                    )
                }
        }
}

pub open spec fn owned_map_valid(owned: Map<SlotLoc, SlotToken>, id: InstanceId) -> bool {
    forall|l: SlotLoc| #[trigger] owned.contains_key(l) ==> slot_is_owned(owned[l], id, l)
}

pub open spec fn shares_map_valid(shares: Map<SlotLoc, ShareCap>, id: InstanceId) -> bool {
    forall|l: SlotLoc| #[trigger]
        shares.contains_key(l) ==> shares[l].is_rdma_bound() && shares[l].instance_id() == id
            && shares[l].loc() == l
}

pub open spec fn owned_frontiers(owned: Map<SlotLoc, SlotToken>) -> Map<
    SlotLoc,
    Frontier<TransferDomain>,
> {
    Map::new(owned.dom(), |l: SlotLoc| slot_frontier(owned[l]))
}

pub open spec fn share_frontiers(shares: Map<SlotLoc, ShareCap>) -> Map<
    SlotLoc,
    (TransferShare, Frontier<TransferDomain>),
> {
    Map::new(shares.dom(), |l: SlotLoc| (shares[l].share(), shares[l].frontier()))
}

pub open spec fn owned_map_advanced(
    after: Map<SlotLoc, SlotToken>,
    before: Map<SlotLoc, SlotToken>,
    id: InstanceId,
    ev: TransferEvent,
    access: Access<TransferDomain>,
) -> bool {
    &&& after.dom() == before.dom()
    &&& forall|l: SlotLoc| #[trigger]
        before.contains_key(l) ==> slot_owned(
            after[l],
            id,
            l,
            advance(slot_frontier(before[l]), l, ev, access),
        )
}

impl<'a> QueuePair<'a> {
    pub open spec fn fence_infos(&self, fence: bool, infos: SubmittedMap) -> IMap<
        TransferEvent,
        EventInfo<TransferDomain>,
    > {
        if fence {
            infos.map().union_prefer_right(self.read_infos().map())
        } else {
            infos.map()
        }
    }

    pub open spec fn fence_after(&self, fence: bool) -> ISet<TransferEvent> {
        if fence {
            self.reads()
        } else {
            ISet::empty()
        }
    }

    pub open spec fn transfer_ordered(
        &self,
        fence: bool,
        infos: SubmittedMap,
        cited: ISet<TransferEvent>,
    ) -> ISet<TransferEvent> {
        closure_from(self.fence_infos(fence, infos), self.fence_after(fence).union(cited))
    }

    pub open spec fn read_transfer_admitted(
        &self,
        cq: CompletionQueue,
        instance: InstanceId,
        local: MemoryRegionSlice,
        remote: RemoteCapability,
        wr_id: u64,
        fence: bool,
        remote_perm: ReadPerm,
        local_tok: SlotToken,
        infos: SubmittedMap,
        knows: Knows,
        cited: ISet<TransferEvent>,
    ) -> bool {
        let ordered = self.transfer_ordered(fence, infos, cited);
        &&& self.can_post(cq, wr_id) && self.source().id() == instance
        &&& self.effective_max_rd_atomic() > 0
        &&& local.well_formed() && remote.well_formed()
        &&& local.registration().pd_id() == self.pd_id()
        &&& local.registration().instance_id() == instance
        &&& local.registration().access().local_write
        &&& local.slot_ref().loc.node == self.source().node()
        &&& remote.slot_ref().loc.node == self.peer()
        &&& remote.slot_ref().loc.node != local.slot_ref().loc.node
        &&& local.slot_ref().sub_len == remote.slot_ref().sub_len
        &&& local.slot_ref().sub_len <= MAX_RDMA_MSG_LEN
        &&& remote_perm.valid_rdma(instance, remote.slot_ref().loc)
        &&& remote_perm.share_opt() == remote.share()
        &&& slot_is_owned(local_tok, instance, local.slot_ref().loc)
        &&& infos.instance_id() == instance && knows.instance_id() == instance
        &&& cited.subset_of(infos.dom())
        &&& knowledge_of(self.source().node(), cited).subset_of(knows.set())
        &&& read_admitted(remote_perm.frontier(), ordered)
        &&& write_admitted(slot_frontier(local_tok), ordered)
    }

    pub open spec fn write_transfer_admitted(
        &self,
        cq: CompletionQueue,
        instance: InstanceId,
        local: MemoryRegionSlice,
        remote: RemoteCapability,
        wr_id: u64,
        fence: bool,
        local_perm: ReadPerm,
        remote_tok: SlotToken,
        infos: SubmittedMap,
        knows: Knows,
        cited: ISet<TransferEvent>,
    ) -> bool {
        let ordered = self.transfer_ordered(fence, infos, cited);
        &&& self.can_post(cq, wr_id) && self.source().id() == instance
        &&& local.well_formed() && remote.well_formed()
        &&& local.registration().pd_id() == self.pd_id()
        &&& local.registration().instance_id() == instance
        &&& remote.access() == RemoteAccess::ReadWrite
        &&& local.slot_ref().loc.node == self.source().node()
        &&& remote.slot_ref().loc.node == self.peer()
        &&& remote.slot_ref().loc.node != local.slot_ref().loc.node
        &&& local.slot_ref().sub_len == remote.slot_ref().sub_len
        &&& local.slot_ref().sub_len <= MAX_RDMA_MSG_LEN
        &&& local_perm.valid_local(instance, local.slot_ref().loc)
        &&& slot_is_owned(remote_tok, instance, remote.slot_ref().loc)
        &&& infos.instance_id() == instance && knows.instance_id() == instance
        &&& cited.subset_of(infos.dom())
        &&& knowledge_of(self.source().node(), cited).subset_of(knows.set())
        &&& read_admitted(local_perm.frontier(), ordered)
        &&& write_admitted(slot_frontier(remote_tok), ordered)
    }

    pub open spec fn valid_sges(
        &self,
        instance: InstanceId,
        locals: Seq<MemoryRegionSlice>,
        remote: RemoteCapability,
        read: bool,
    ) -> bool {
        &&& 1 <= locals.len() <= self.max_send_sge()
        &&& remote.well_formed() && remote.slot_ref().loc.node == self.peer() && self.peer()
            != self.source().node()
        &&& read || remote.access() == RemoteAccess::ReadWrite
        &&& sge_total_len(locals, locals.len() as int) == remote.slot_ref().sub_len
            <= MAX_RDMA_MSG_LEN
        &&& !read || sges_distinct_locs(locals)
        &&& forall|i: int|
            0 <= i < locals.len() ==> {
                let l = #[trigger] locals[i];
                &&& l.well_formed() && 0 < l.slot_ref().sub_len <= MAX_RDMA_MSG_LEN
                &&& l.registration().pd_id() == self.pd_id() && l.registration().instance_id()
                    == instance
                &&& l.slot_ref().loc.node == self.source().node() && (!read
                    || l.registration().access().local_write)
            }
    }

    pub open spec fn read_sg_admitted(
        &self,
        cq: CompletionQueue,
        instance: InstanceId,
        locals: Seq<MemoryRegionSlice>,
        remote: RemoteCapability,
        wr_id: u64,
        fence: bool,
        remote_perm: ReadPerm,
        local_tokens: Map<SlotLoc, SlotToken>,
        infos: SubmittedMap,
        knows: Knows,
        cited: ISet<TransferEvent>,
    ) -> bool {
        let ordered = self.transfer_ordered(fence, infos, cited);
        let access = sge_transfer_access(true, locals, remote.slot_ref());
        &&& self.can_post(cq, wr_id) && self.source().id() == instance
        &&& self.effective_max_rd_atomic() > 0
        &&& self.valid_sges(instance, locals, remote, true)
        &&& infos.instance_id() == instance && knows.instance_id() == instance
        &&& cited.subset_of(infos.dom())
        &&& knowledge_of(self.source().node(), cited).subset_of(knows.set())
        &&& remote_perm.valid_rdma(instance, remote.slot_ref().loc)
        &&& remote_perm.share_opt() == remote.share()
        &&& read_admitted(remote_perm.frontier(), ordered)
        &&& owned_map_valid(local_tokens, instance)
        &&& !local_tokens.contains_key(remote.slot_ref().loc)
        &&& forall|i: int|
            0 <= i < locals.len() ==> local_tokens.contains_key(
                (#[trigger] locals[i]).slot_ref().loc,
            )
        &&& forall|l: SlotLoc| #[trigger]
            local_tokens.contains_key(l) ==> admitted(
                slot_frontier(local_tokens[l]),
                l,
                access,
                ordered,
            )
    }

    pub open spec fn write_sg_admitted(
        &self,
        cq: CompletionQueue,
        instance: InstanceId,
        locals: Seq<MemoryRegionSlice>,
        remote: RemoteCapability,
        wr_id: u64,
        fence: bool,
        local_tokens: Map<SlotLoc, SlotToken>,
        remote_tok: SlotToken,
        infos: SubmittedMap,
        knows: Knows,
        cited: ISet<TransferEvent>,
    ) -> bool {
        let ordered = self.transfer_ordered(fence, infos, cited);
        let access = sge_transfer_access(false, locals, remote.slot_ref());
        &&& self.can_post(cq, wr_id) && self.source().id() == instance
        &&& self.valid_sges(instance, locals, remote, false)
        &&& infos.instance_id() == instance && knows.instance_id() == instance
        &&& cited.subset_of(infos.dom())
        &&& knowledge_of(self.source().node(), cited).subset_of(knows.set())
        &&& slot_is_owned(remote_tok, instance, remote.slot_ref().loc)
        &&& write_admitted(slot_frontier(remote_tok), ordered)
        &&& owned_map_valid(local_tokens, instance)
        &&& !local_tokens.contains_key(remote.slot_ref().loc)
        &&& forall|i: int|
            0 <= i < locals.len() ==> local_tokens.contains_key(
                (#[trigger] locals[i]).slot_ref().loc,
            )
        &&& forall|l: SlotLoc| #[trigger]
            local_tokens.contains_key(l) ==> admitted(
                slot_frontier(local_tokens[l]),
                l,
                access,
                ordered,
            )
    }

    pub open spec fn qp_cq_stepped(
        &self,
        before: Self,
        cq: CompletionQueue,
        before_cq: CompletionQueue,
        access: Access<TransferDomain>,
        effect: Effect<TransferDomain>,
        guards: ReadGuards,
        wr_id: u64,
        fence: bool,
        read: bool,
        infos: SubmittedMap,
        cited: ISet<TransferEvent>,
        name: TransferEvent,
        receipt: Submitted,
    ) -> bool {
        let ev = before.source().next_event();
        &&& name == ev && receipt.key() == ev && receipt.instance_id() == before.source().id()
        &&& receipt.value() == (EventInfo {
            access,
            read_guards: guards,
            effect,
            preds: before.transfer_ordered(fence, infos, cited),
        })
        &&& self.ready() && self.source().node() == before.source().node() && self.source().id()
            == before.source().id() && self.source().source() == before.source().source()
            && self.source().next_seq() == before.source().next_seq() + 1 && self.peer()
            == before.peer() && self.cq_id() == before.cq_id() && self.pd_id() == before.pd_id()
            && self.reads() == (if read {
            before.reads().insert(ev)
        } else {
            before.reads()
        })
        &&& self.configuration() == before.configuration()
        &&& self.effective_max_rd_atomic() == before.effective_max_rd_atomic()
        &&& self.read_infos().map() == (if read {
            before.read_infos().map().insert(ev, receipt.value())
        } else {
            before.read_infos().map()
        })
        &&& cq.well_formed() && cq.same_queue(before_cq) && cq.pending()
            == before_cq.pending().insert(
            wr_id,
            PendingWorkRequest { kind: PendingKind::DataTransfer { event: ev } },
        ) && cq.infos().map() == before_cq.infos().map().insert(ev, receipt.value())
    }
}

} // verus!
