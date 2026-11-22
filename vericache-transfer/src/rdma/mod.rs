//! Reliable-connected verbs operations. WR identifiers belong to the send CQ.
mod ffi;

use crate::context::*;
use crate::cuda::{
    gdr_flush_writes_supported, gdr_writes_ordering, CudaBuffer, CudaError, CudaHostBuffer,
    GdrWritesOrdering,
};
use crate::domain::*;
#[cfg(verus_only)]
use crate::host;
use crate::host::{CompletedAccess, Completion, HostBuffer, HostContext};
#[cfg(verus_only)]
use crate::transport::{covers, share_matches, slot_matches, Statement};
use crate::transport::{ByteTransport, Channel};
#[cfg(verus_only)]
use foundation::history::no_access;
#[cfg(verus_only)]
use foundation::space::{restrict, Snapshot};
use foundation::space::{Access, Effect};
#[cfg(verus_only)]
use foundation::tracked::*;
#[cfg(not(verus_only))]
use foundation::tracked::{EventInfo, Frontier};
#[cfg(verus_only)]
use foundation::{
    history::{self, History},
    value::*,
};
#[cfg(verus_only)]
use vstd::arithmetic::mul::{lemma_mul_inequality, lemma_mul_is_distributive_add_other_way};
use vstd::prelude::*;
use vstd::tokens::InstanceId;
#[cfg(verus_only)]
use vstd::tokens::{ElementToken, KeyValueToken};

mod batch;
mod lifecycle;
mod memory;
mod poll;
mod proofs;
mod queue;
mod transfer;
#[cfg(verus_only)]
mod transfer_spec;
mod window;

pub use crate::cuda::{
    flush_gdr_writes as gdr_flush_writes, gdr_flush_writes_supported as gdr_flush_supported,
    gdr_writes_ordering as gdr_ordering, GdrWritesOrdering as RdmaGdrWritesOrdering,
};
pub use batch::*;
pub use lifecycle::*;
pub use memory::*;
pub use poll::*;
#[cfg(verus_only)]
pub use proofs::scatter_read_returns_last_write;
pub use queue::*;
pub use transfer::*;
#[cfg(verus_only)]
pub use transfer_spec::*;
pub use window::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

/// Maximum single IBTA RDMA message payload length (`2^31` bytes).
pub const MAX_RDMA_MSG_LEN: u64 = 0x8000_0000;

#[derive(Debug, PartialEq, Eq)]
pub struct RdmaError(pub i32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceAttributes {
    pub supports_type_2: bool,
    pub max_qp_wr: u32,
    pub max_send_sge: u32,
    pub max_read_sge: u32,
    pub max_qp_rd_atomic: u8,
    pub max_qp_init_rd_atomic: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueuePairConfig {
    pub send_depth: u32,
    pub max_send_sge: u32,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
}

pub struct DeviceContext {
    raw: *mut u8,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    node: NodeId,
}

pub struct ProtectionDomain<'a> {
    raw: *mut u8,
    context: &'a DeviceContext,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pd_id: Ghost<nat>,
}

impl DeviceContext {
    pub closed spec fn node(&self) -> NodeId {
        self.node
    }

    pub fn count() -> Result<u32, RdmaError> {
        ffi::device_count()
    }

    pub fn open(index: u32, node: NodeId) -> (r: Result<Self, RdmaError>)
        ensures
            r matches Ok(d) ==> d.node() == node,
    {
        Ok(Self { raw: ffi::open(index)?, node })
    }

    pub fn name(&self) -> String {
        ffi::device_name(self.raw)
    }

    pub fn supports_type_2(&self) -> Result<bool, RdmaError> {
        Ok(self.attributes()?.supports_type_2)
    }

    pub fn attributes(&self) -> Result<DeviceAttributes, RdmaError> {
        ffi::attributes(self.raw)
    }

    pub fn close(self) -> Result<(), RdmaError> {
        ffi::close(self.raw)
    }
}

impl<'a> ProtectionDomain<'a> {
    pub closed spec fn id(&self) -> nat {
        self.pd_id@
    }

    pub fn new(context: &'a DeviceContext, Tracked(name): Tracked<EventSourceToken>) -> (r: Result<
        Self,
        RdmaError,
    >)
        requires
            name.well_formed(),
            name.node() == context.node(),
            name.source() is RdmaQueuePair,
        ensures
            r matches Ok(pd) ==> pd.node() == context.node() && pd.id()
                == name.source()->RdmaQueuePair_id,
    {
        Ok(
            Self {
                raw: ffi::alloc_pd(context.raw)?,
                context,
                pd_id: Ghost(name.source()->RdmaQueuePair_id),
            },
        )
    }

    pub closed spec fn node(&self) -> NodeId {
        self.context.node()
    }

    pub fn destroy(self) -> Result<(), RdmaError> {
        ffi::dealloc_pd(self.raw)
    }
}

#[derive(Copy, Clone)]
pub struct AccessFlags {
    pub local_write: bool,
    pub mw_bind: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RemoteAccess {
    ReadOnly,
    ReadWrite,
}

} // verus!
