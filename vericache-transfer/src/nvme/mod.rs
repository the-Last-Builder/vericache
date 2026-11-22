//! Direct I/O over process-exclusive files (single-slot and slab) and a single-owner Linux io_uring.
mod ffi;
#[cfg(verus_only)]
mod proofs;
mod ring;

use crate::context::*;
use crate::domain::*;
use crate::host::{self, HostContext, HostSlice};
use foundation::space::{Access, Effect};
#[cfg(verus_only)]
use foundation::{
    history::{self, History},
    space::restrict,
    tracked::*,
    value::*,
};
#[cfg(verus_only)]
use vstd::arithmetic::div_mod::lemma_mul_mod_noop_right;
#[cfg(verus_only)]
use vstd::arithmetic::mul::{lemma_mul_inequality, lemma_mul_is_distributive_add_other_way};
use vstd::prelude::*;
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::KeyValueToken;

pub use ring::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub const DIRECT_IO_ALIGNMENT: u64 = 4096;

pub const MAX_IO_LEN: usize = 0x7fff_f000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmeError(pub i32);

#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct NvmeFile {
    fd: i32,
    num_slots: usize,
    slot_len: u64,
    len: u64,
    node: Ghost<NodeId>,
    space: Ghost<MemorySpace>,
    instance: Ghost<InstanceId>,
}

pub struct NvmeSlice<'a> {
    file: &'a NvmeFile,
    start: u64,
    len: u64,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    slot: Ghost<SlotRef>,
}

impl NvmeFile {
    pub closed spec fn node(&self) -> NodeId {
        self.node@
    }

    pub closed spec fn space(&self) -> MemorySpace {
        self.space@
    }

    pub closed spec fn num_slots(&self) -> usize {
        self.num_slots
    }

    pub closed spec fn slot_len(&self) -> u64 {
        self.slot_len
    }

    pub closed spec fn total_len(&self) -> u64 {
        self.len
    }

    pub open spec fn slot_loc(&self, idx: nat) -> SlotLoc {
        SlotLoc { node: self.node(), space: self.space(), slot_index: idx }
    }

    pub open spec fn slot_ref(&self, idx: nat) -> SlotRef {
        SlotRef::full(self.slot_loc(idx), self.slot_len())
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.instance@
    }

    pub closed spec fn well_formed(&self) -> bool {
        &&& self.space@ is NvmeFile
        &&& self.num_slots >= 1
        &&& self.slot_len > 0 && self.slot_len % DIRECT_IO_ALIGNMENT == 0
        &&& self.len == (self.num_slots as nat) * (self.slot_len as nat)
        &&& self.len > 0 && self.len % DIRECT_IO_ALIGNMENT == 0
    }

    pub fn create_temp(
        _ctx: &HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(source): Tracked<&mut AllocSource>,
        dir: &str,
        len: u64,
    ) -> (result: Result<(NvmeFile, Tracked<SlotToken>), NvmeError>)
        requires
            len > 0 && len % DIRECT_IO_ALIGNMENT == 0,
            old(source).well_formed(),
            _ctx.ready(instance.id(), old(source).node()),
            old(source).id() == instance.id(),
        ensures
            match result {
                Err(_) => *final(source) == *old(source),
                Ok((file, tok)) => {
                    &&& file.well_formed() && file.instance_id() == instance.id()
                    &&& file.num_slots() == 1 && file.slot_len() == len
                    &&& file.node() == old(source).node() && file.space() == old(
                        source,
                    ).nvme_space()
                    &&& file.slot_ref(0) == old(source).nvme_slot_ref(len)
                    &&& final(source).allocated_slot(*old(source), file.slot_loc(0), tok@)
                },
            },
    {
        if len == 0 || len % DIRECT_IO_ALIGNMENT != 0 {
            return Err(NvmeError(22));
        }
        let fd = ffi::create(dir, len)?;
        let ghost node = source.node();
        let ghost space = source.nvme_space();
        let tracked tok = source.allocate_nvme_slot();
        Ok(
            (
                Self {
                    fd,
                    num_slots: 1,
                    slot_len: len,
                    len,
                    node: Ghost(node),
                    space: Ghost(space),
                    instance: Ghost(instance.id()),
                },
                Tracked(tok),
            ),
        )
    }

    pub fn create_temp_slab(
        _ctx: &HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(source): Tracked<&mut AllocSource>,
        dir: &str,
        num_slots: usize,
        slot_len: u64,
    ) -> (result: Result<(NvmeFile, Tracked<Map<usize, SlotToken>>), NvmeError>)
        requires
            num_slots >= 1,
            slot_len > 0 && slot_len % DIRECT_IO_ALIGNMENT == 0,
            (num_slots as nat) * (slot_len as nat) <= u64::MAX,
            old(source).well_formed(),
            _ctx.ready(instance.id(), old(source).node()),
            old(source).id() == instance.id(),
        ensures
            match result {
                Err(_) => *final(source) == *old(source),
                Ok((file, tokens)) => {
                    &&& file.well_formed() && file.instance_id() == instance.id()
                    &&& file.num_slots() == num_slots && file.slot_len() == slot_len
                    &&& file.node() == old(source).node() && file.space() == old(
                        source,
                    ).nvme_space()
                    &&& final(source).allocated_slab(*old(source), file.space(), num_slots, tokens@)
                },
            },
    {
        if num_slots == 0 || slot_len == 0 || slot_len % DIRECT_IO_ALIGNMENT != 0 {
            return Err(NvmeError(22));
        }
        let Some(total) = (num_slots as u64).checked_mul(slot_len) else {
            return Err(NvmeError(22));
        };
        proof {
            lemma_mul_inequality(1, num_slots as int, slot_len as int);
            lemma_mul_mod_noop_right(num_slots as int, slot_len as int, DIRECT_IO_ALIGNMENT as int);
        }
        let fd = ffi::create(dir, total)?;
        let ghost node = source.node();
        let ghost space = source.nvme_space();
        let tracked tokens = source.allocate_nvme_slab_slots(num_slots);
        Ok(
            (
                Self {
                    fd,
                    num_slots,
                    slot_len,
                    len: total,
                    node: Ghost(node),
                    space: Ghost(space),
                    instance: Ghost(instance.id()),
                },
                Tracked(tokens),
            ),
        )
    }

    pub fn as_slice(&self) -> (r: NvmeSlice<'_>)
        requires
            self.well_formed(),
            self.num_slots() == 1,
        ensures
            r.well_formed(),
            r.slot_ref() == self.slot_ref(0),
            r.file_offset() == 0,
            r.instance_id() == self.instance_id(),
    {
        self.slot_slice(0)
    }

    pub fn slot_slice(&self, idx: usize) -> (r: NvmeSlice<'_>)
        requires
            self.well_formed(),
            idx < self.num_slots(),
        ensures
            r.well_formed(),
            r.slot_ref() == self.slot_ref(idx as nat),
            r.file_offset() == (idx as nat) * (self.slot_len() as nat),
            r.file_offset() % DIRECT_IO_ALIGNMENT == 0,
            r.instance_id() == self.instance_id(),
    {
        assert!(idx < self.num_slots);
        proof {
            lemma_mul_is_distributive_add_other_way(self.slot_len as int, idx as int, 1);
            lemma_mul_inequality((idx + 1) as int, self.num_slots as int, self.slot_len as int);
            lemma_mul_mod_noop_right(idx as int, self.slot_len as int, DIRECT_IO_ALIGNMENT as int);
        }
        let start = (idx as u64) * self.slot_len;
        NvmeSlice { file: self, start, len: self.slot_len, slot: Ghost(self.slot_ref(idx as nat)) }
    }

    pub fn len(&self) -> (r: u64)
        requires
            self.well_formed(),
        ensures
            r == self.total_len(),
            r > 0 && r % DIRECT_IO_ALIGNMENT == 0,
    {
        self.len
    }

    pub fn num_slots_val(&self) -> (r: usize)
        requires
            self.well_formed(),
        ensures
            r == self.num_slots(),
    {
        self.num_slots
    }

    pub fn slot_len_val(&self) -> (r: u64)
        requires
            self.well_formed(),
        ensures
            r == self.slot_len(),
            r > 0 && r % DIRECT_IO_ALIGNMENT == 0,
    {
        self.slot_len
    }

    pub fn close(
        _ctx: &mut HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        file: NvmeFile,
        Tracked(tok): Tracked<SlotToken>,
        Tracked(infos): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Ghost(cited): Ghost<ISet<TransferEvent>>,
    ) -> (result: Tracked<host::Completion>)
        requires
            old(_ctx).ready(instance.id(), file.node()),
            file.well_formed(),
            file.num_slots() == 1,
            file.instance_id() == instance.id(),
            host::can_reclaim_slot(instance.id(), file.slot_loc(0), tok, *infos, *knows, cited),
        ensures
            final(_ctx).stepped(*old(_ctx)),
            result@.records(
                instance.id(),
                old(_ctx).source@.next_event(),
                reclaim_access(file.slot_loc(0)),
                havoc_effect(),
                closure_from(infos.map(), cited),
                IMap::empty(),
            ),
    {
        let ghost loc = file.slot_loc(0);
        let tracked mut tok = tok;
        let Tracked(completed) = ffi::close(file);
        proof {
            assert(slab_locs(loc.node, loc.space, 1nat) =~= ISet::empty().insert(loc));
            assert(completed.access == reclaim_access(loc));
        }
        Tracked(
            host::finish_owned(
                _ctx.source.borrow_mut(),
                instance,
                completed,
                &mut tok,
                infos,
                knows,
                cited,
            ),
        )
    }

    pub fn close_slab(
        _ctx: &mut HostContext,
        Tracked(instance): Tracked<&TransferInstance>,
        file: NvmeFile,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
        Tracked(infos): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Ghost(cited): Ghost<ISet<TransferEvent>>,
    ) -> (result: Tracked<host::Completion>)
        requires
            old(_ctx).ready(instance.id(), file.node()),
            file.well_formed(),
            file.instance_id() == instance.id(),
            host::can_reclaim_slab(
                instance.id(),
                file.node(),
                file.space(),
                file.num_slots(),
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
                reclaim_slab_access(file.node(), file.space(), file.num_slots() as nat),
                havoc_effect(),
                closure_from(infos.map(), cited),
                IMap::empty(),
            ),
    {
        let ghost node = file.node();
        let ghost space = file.space();
        let ghost count = file.num_slots();
        let Tracked(completed) = ffi::close(file);
        let tracked (completion, _) = host::finish_slab_owned(
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
}

impl<'a> NvmeSlice<'a> {
    pub closed spec fn slot_ref(&self) -> SlotRef {
        self.slot@
    }

    pub closed spec fn file_offset(&self) -> u64 {
        self.start
    }

    pub closed spec fn instance_id(&self) -> InstanceId {
        self.file.instance_id()
    }

    pub closed spec fn well_formed(&self) -> bool {
        &&& self.file.well_formed()
        &&& self.slot@.well_formed()
        &&& self.slot@.loc.node == self.file.node()
        &&& self.slot@.loc.space == self.file.space()
        &&& self.slot@.loc.slot_index < self.file.num_slots()
        &&& self.slot@.slot_len == self.file.slot_len()
        &&& self.slot@.sub_len == self.len
        &&& self.start == self.slot@.loc.slot_index * (self.file.slot_len() as nat)
            + self.slot@.sub_offset
        &&& self.start + self.len <= self.file.len
    }

    pub fn len(&self) -> (r: u64)
        requires
            self.well_formed(),
        ensures
            r == self.slot_ref().sub_len,
    {
        self.len
    }

    pub fn offset(&self) -> (r: u64)
        requires
            self.well_formed(),
        ensures
            r == self.file_offset(),
    {
        self.start
    }

    pub fn sub_slice(&self, offset: u64, len: u64) -> (r: Self)
        requires
            self.well_formed(),
            offset + len <= self.slot_ref().sub_len,
        ensures
            r.well_formed(),
            r.slot_ref() == self.slot_ref().sub_slice(offset, len),
            r.file_offset() == self.file_offset() + offset,
            r.instance_id() == self.instance_id(),
    {
        assert!(offset <= self.len && len <= self.len - offset);
        Self {
            file: self.file,
            start: self.start + offset,
            len,
            slot: Ghost(self.slot@.sub_slice(offset, len)),
        }
    }
}

} // verus!
