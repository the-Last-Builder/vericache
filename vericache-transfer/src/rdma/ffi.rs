//! Native resource coverage, RC ordering, successful copy behavior and CQ observation.
use super::*;

#[cfg_attr(not(verus_only), link(name = "ibverbs"))]
unsafe extern "C" {
    fn ibv_get_device_list(n: *mut i32) -> *mut *mut u8;
    fn ibv_free_device_list(p: *mut *mut u8);
    fn ibv_open_device(p: *mut u8) -> *mut u8;
    fn ibv_close_device(p: *mut u8) -> i32;
    fn ibv_alloc_pd(p: *mut u8) -> *mut u8;
    fn ibv_dealloc_pd(p: *mut u8) -> i32;
    fn vc_create_cq(ctx: *mut u8, depth: i32) -> *mut u8;
    fn vc_destroy_cq(p: *mut u8) -> i32;
    fn ibv_dereg_mr(p: *mut u8) -> i32;
    fn ibv_destroy_qp(p: *mut u8) -> i32;
    fn vc_gid(ctx: *mut u8, port: u8, index: i32, gid: *mut u8) -> i32;
    fn vc_device_at(list: *mut *mut u8, index: u32) -> *mut u8;
    fn vc_device_name(ctx: *mut u8) -> *const std::ffi::c_char;
    fn vc_device_attr(
        ctx: *mut u8,
        supported: *mut i32,
        depth: *mut u32,
        sge: *mut u32,
        sge_rd: *mut u32,
        responder: *mut u8,
        initiator: *mut u8,
    ) -> i32;
    fn vc_port(
        ctx: *mut u8,
        port: u8,
        lid: *mut u16,
        mtu: *mut u8,
        state: *mut u8,
        ethernet: *mut u8,
    ) -> i32;
    fn vc_register(pd: *mut u8, addr: *mut u8, len: usize, write: i32, bind: i32) -> *mut u8;
    fn vc_lkey(mr: *mut u8) -> u32;
    fn vc_alloc_window(pd: *mut u8) -> *mut u8;
    fn vc_rkey(mw: *mut u8) -> u32;
    fn vc_dealloc_window(mw: *mut u8) -> i32;
    fn vc_create_qp(pd: *mut u8, cq: *mut u8, depth: u32, sge: u32) -> *mut u8;
    fn vc_qpn(qp: *mut u8) -> u32;
    fn vc_init(qp: *mut u8, port: u8) -> i32;
    fn vc_rtr(
        qp: *mut u8,
        port: u8,
        qpn: u32,
        psn: u32,
        lid: u16,
        mtu: u8,
        global: u8,
        index: u8,
        gid: *const u8,
        responder: u8,
    ) -> i32;
    fn vc_rts(qp: *mut u8, psn: u32, initiator: u8) -> i32;
    fn vc_error(qp: *mut u8) -> i32;
    fn vc_reset(qp: *mut u8) -> i32;
    fn vc_bind(
        qp: *mut u8,
        mw: *mut u8,
        mr: *mut u8,
        addr: u64,
        len: u64,
        rkey: u32,
        write: i32,
        id: u64,
    ) -> i32;
    fn vc_invalidate(qp: *mut u8, rkey: u32, id: u64) -> i32;
    fn vc_transfer(
        qp: *mut u8,
        local: u64,
        len: u32,
        lkey: u32,
        remote: u64,
        rkey: u32,
        read: i32,
        fence: i32,
        id: u64,
    ) -> i32;
    fn vc_transfer_sg(qp: *mut u8, d: *const WrDesc, sges: *const Sge) -> i32;
    fn vc_post_batch(
        qp: *mut u8,
        reqs: *const WrDesc,
        count: usize,
        sges: *const Sge,
        total: usize,
        posted: *mut usize,
    ) -> i32;
    fn vc_poll(cq: *mut u8, id: *mut u64, status: *mut i32, kind: *mut WorkKind) -> i32;
    fn vc_poll_batch(cq: *mut u8, max: i32) -> i32;
    fn vc_take_batch(cq: *mut u8, out: *mut RawCompletion, count: i32);
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Sge {
    addr: u64,
    len: u32,
    lkey: u32,
}

impl Sge {
    fn from_slice(s: &MemoryRegionSlice) -> Self {
        Self {
            addr: s.addr,
            len: s.len as u32,
            lkey: s.mr.lkey,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WrDesc {
    id: u64,
    remote: u64,
    rkey: u32,
    num_sge: u32,
    read: u8,
    fence: u8,
}

impl WrDesc {
    fn new(cap: &RemoteCapability, id: u64, count: u32, read: bool, fence: bool) -> Self {
        Self {
            id,
            remote: cap.addr,
            rkey: cap.rkey,
            num_sge: count,
            read: read as u8,
            fence: fence as u8,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawCompletion {
    id: u64,
    status: i32,
    kind: WorkKind,
}

impl Default for RawCompletion {
    fn default() -> Self {
        Self {
            id: 0,
            status: 0,
            kind: WorkKind::Data,
        }
    }
}

// Stage the ABI descriptors inline on the common path, spilling only larger lists.
struct Scratch<T, const N: usize> {
    inline: [T; N],
    overflow: Vec<T>,
}
impl<T: Default + Copy, const N: usize> Scratch<T, N> {
    fn new() -> Self {
        Self {
            inline: [T::default(); N],
            overflow: Vec::new(),
        }
    }
    fn buffer(&mut self, n: usize) -> Result<&mut [T], RdmaError> {
        if n <= N {
            return Ok(&mut self.inline[..n]);
        }
        self.overflow
            .try_reserve_exact(n)
            .map_err(|_| RdmaError(12))?;
        self.overflow.resize(n, T::default());
        Ok(&mut self.overflow)
    }
}

fn encode_sges(
    out: &mut [Sge],
    slices: &[MemoryRegionSlice],
    remote: &RemoteCapability,
) -> Result<(), RdmaError> {
    let mut total = 0u64;
    for (dst, src) in out.iter_mut().zip(slices) {
        if src.len == 0 || src.len > MAX_RDMA_MSG_LEN {
            return Err(RdmaError(22));
        }
        total = total.checked_add(src.len).ok_or(RdmaError(22))?;
        if total > MAX_RDMA_MSG_LEN {
            return Err(RdmaError(22));
        }
        *dst = Sge::from_slice(src);
    }
    if slices.is_empty() || total != remote.len {
        return Err(RdmaError(22));
    }
    Ok(())
}

fn disarm_rdma_ptr(slot: &mut *mut u8) -> *mut u8 {
    let raw = *slot;
    *slot = std::ptr::null_mut();
    raw
}

fn drop_rdma_ptr(slot: &mut *mut u8, destroy: unsafe extern "C" fn(*mut u8) -> i32) {
    let raw = disarm_rdma_ptr(slot);
    if !raw.is_null() {
        let _ = unsafe { destroy(raw) };
    }
}

verus! {

#[verifier::external_body]
pub(super) struct RawDeviceCtxGuard(*mut u8);

impl RawDeviceCtxGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = disarm_rdma_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawDeviceCtxGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_rdma_ptr(&mut self.0, ibv_close_device);
    }
}

#[verifier::external_body]
pub(super) struct RawPdGuard(*mut u8);

impl RawPdGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = disarm_rdma_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawPdGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_rdma_ptr(&mut self.0, ibv_dealloc_pd);
    }
}

#[verifier::external_body]
pub(super) struct RawCqGuard(*mut u8);

impl RawCqGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = disarm_rdma_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawCqGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_rdma_ptr(&mut self.0, vc_destroy_cq);
    }
}

#[verifier::external_body]
pub(super) struct RawQpGuard(*mut u8);

impl RawQpGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = disarm_rdma_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawQpGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_rdma_ptr(&mut self.0, ibv_destroy_qp);
    }
}

#[verifier::external_body]
pub(super) struct RawMrGuard(*mut u8);

impl RawMrGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = disarm_rdma_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawMrGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_rdma_ptr(&mut self.0, ibv_dereg_mr);
    }
}

#[verifier::external_body]
pub(super) struct RawMwGuard(*mut u8);

impl RawMwGuard {
    #[verifier::external_body]
    pub(super) fn new(raw: *mut u8) -> Self {
        Self(raw)
    }

    #[verifier::external_body]
    pub(super) fn as_ptr(&self) -> *mut u8 {
        self.0
    }

    #[verifier::external_body]
    pub(super) fn into_raw(mut self) -> *mut u8 {
        let raw = disarm_rdma_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawMwGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_rdma_ptr(&mut self.0, vc_dealloc_window);
    }
}

fn status(code: i32) -> Result<(), RdmaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(RdmaError(code))
    }
}

#[verifier::external_body]
fn checked<T>(p: *mut T) -> Result<*mut T, RdmaError> {
    if p.is_null() {
        Err(RdmaError(std::io::Error::last_os_error().raw_os_error().unwrap_or(5)))
    } else {
        Ok(p)
    }
}

#[verifier::external_body]
pub(super) fn device_count() -> Result<u32, RdmaError> {
    let mut n = 0;
    let list = checked(unsafe { ibv_get_device_list(&mut n) })?;
    unsafe {
        ibv_free_device_list(list);
    }
    Ok(n as u32)
}

#[verifier::external_body]
pub(super) fn open(index: u32) -> Result<RawDeviceCtxGuard, RdmaError> {
    let mut n = 0;
    let list = checked(unsafe { ibv_get_device_list(&mut n) })?;
    let result = if index < n as u32 {
        checked(unsafe { ibv_open_device(vc_device_at(list, index)) }).map(RawDeviceCtxGuard::new)
    } else {
        Err(RdmaError(19))
    };
    unsafe {
        ibv_free_device_list(list);
    }
    result
}

#[verifier::external_body]
pub(super) fn device_name(ctx: *mut u8) -> String {
    let name = unsafe { std::ffi::CStr::from_ptr(vc_device_name(ctx)) };
    name.to_string_lossy().into_owned()
}

#[verifier::external_body]
pub(super) fn attributes(ctx: *mut u8) -> Result<DeviceAttributes, RdmaError> {
    let (mut supported, mut depth, mut sge, mut sge_rd, mut responder, mut initiator) = (
        0,
        0,
        0,
        0,
        0,
        0,
    );
    status(
        unsafe {
            vc_device_attr(
                ctx,
                &mut supported,
                &mut depth,
                &mut sge,
                &mut sge_rd,
                &mut responder,
                &mut initiator,
            )
        },
    )?;
    Ok(
        DeviceAttributes {
            supports_type_2: supported != 0,
            max_qp_wr: depth,
            max_send_sge: sge,
            max_read_sge: sge_rd,
            max_qp_rd_atomic: responder,
            max_qp_init_rd_atomic: initiator,
        },
    )
}

#[verifier::external_body]
pub(super) fn port(ctx: *mut u8, p: u8, index: u8) -> Result<(u16, u8, bool, [u8; 16]), RdmaError> {
    let (mut lid, mut mtu, mut state, mut ethernet) = (0, 0, 0, 0);
    status(unsafe { vc_port(ctx, p, &mut lid, &mut mtu, &mut state, &mut ethernet) })?;
    if state != 4 {
        return Err(RdmaError(100));
    }
    let mut gid = [0u8;16];
    if ethernet != 0 {
        status(unsafe { vc_gid(ctx, p, index as i32, gid.as_mut_ptr()) })?;
    }
    Ok((lid, mtu, ethernet != 0, gid))
}

#[verifier::external_body]
pub(super) fn register(pd: *mut u8, ptr: *mut u8, len: usize, flags: AccessFlags) -> Result<
    (RawMrGuard, u32),
    RdmaError,
> {
    let mr = checked(
        unsafe { vc_register(pd, ptr, len, flags.local_write as i32, flags.mw_bind as i32) },
    )?;
    let lkey = unsafe { vc_lkey(mr) };
    Ok((RawMrGuard::new(mr), lkey))
}

#[verifier::external_body]
pub(super) fn raw_deregister_memory(m: &MemoryRegion) -> (r: Result<
    Tracked<CompletedAccess>,
    RdmaError,
>)
    ensures
        r matches Ok(w) ==> w@.access == dereg_slab_access(
            m.node(),
            m.space(),
            m.num_slots() as nat,
        ) && w@.effect == (if m.num_slots() == 1 {
            identity_effect(m.slot_loc(0))
        } else {
            identity_effect_slab()
        }),
{
    status(unsafe { ibv_dereg_mr(m.raw.as_ptr()) })?;
    Ok(
        Tracked(
            CompletedAccess {
                access: dereg_slab_access(m.node(), m.space(), m.num_slots() as nat),
                effect: if m.num_slots() == 1 {
                    identity_effect(m.slot_loc(0))
                } else {
                    identity_effect_slab()
                },
            },
        ),
    )
}

#[verifier::external_body]
pub(super) fn alloc_window(pd: *mut u8) -> Result<(RawMwGuard, u32), RdmaError> {
    let mw = checked(unsafe { vc_alloc_window(pd) })?;
    let rkey = unsafe { vc_rkey(mw) };
    Ok((RawMwGuard::new(mw), rkey))
}

#[verifier::external_body]
pub(super) fn create_cq(ctx: *mut u8, depth: i32) -> Result<RawCqGuard, RdmaError> {
    checked(unsafe { vc_create_cq(ctx, depth) }).map(RawCqGuard::new)
}

#[verifier::external_body]
pub(super) fn create_qp(pd: *mut u8, cq: *mut u8, config: QueuePairConfig) -> Result<
    (RawQpGuard, u32),
    RdmaError,
> {
    let qp = checked(unsafe { vc_create_qp(pd, cq, config.send_depth, config.max_send_sge) })?;
    let qpn = unsafe { vc_qpn(qp) };
    Ok((RawQpGuard::new(qp), qpn))
}

#[verifier::external_body]
pub(super) fn connect(qp: &UnconnectedQueuePair, peer: QueuePairEndpoint) -> Result<(), RdmaError> {
    let raw = qp.raw.as_ptr();
    let initiator = qp.config.max_rd_atomic.min(peer.responder_resources);
    status(unsafe { vc_init(raw, qp.port) })?;
    status(
        unsafe {
            vc_rtr(
                raw,
                qp.port,
                peer.number,
                peer.psn,
                peer.lid,
                qp.endpoint.mtu.min(peer.mtu),
                qp.global as u8,
                qp.gid_index,
                peer.gid.as_ptr(),
                qp.config.max_dest_rd_atomic,
            )
        },
    )?;
    status(unsafe { vc_rts(raw, qp.endpoint.psn, initiator) })
}

#[verifier::external_body]
pub(super) fn reconnect(qp: &UnconnectedQueuePair, psn: u32, peer: QueuePairEndpoint) -> Result<
    (),
    RdmaError,
> {
    let raw = qp.raw.as_ptr();
    let initiator = qp.config.max_rd_atomic.min(peer.responder_resources);
    status(unsafe { vc_init(raw, qp.port) })?;
    status(
        unsafe {
            vc_rtr(
                raw,
                qp.port,
                peer.number,
                peer.psn,
                peer.lid,
                qp.endpoint.mtu.min(peer.mtu),
                qp.global as u8,
                qp.gid_index,
                peer.gid.as_ptr(),
                qp.config.max_dest_rd_atomic,
            )
        },
    )?;
    status(unsafe { vc_rts(raw, psn, initiator) })
}

#[verifier::external_body]
pub(super) fn bind(
    q: *mut u8,
    mw: *mut u8,
    mr: *mut u8,
    addr: u64,
    len: u64,
    rkey: u32,
    access: RemoteAccess,
    id: u64,
) -> Result<(), RdmaError> {
    status(
        unsafe { vc_bind(q, mw, mr, addr, len, rkey, (access == RemoteAccess::ReadWrite) as i32, id)
        },
    )
}

#[verifier::external_body]
pub(super) fn invalidate(q: *mut u8, rkey: u32, id: u64) -> Result<(), RdmaError> {
    status(unsafe { vc_invalidate(q, rkey, id) })
}

/// The signaled WR copies exactly this SGE. A fence waits for preceding RC reads.
#[verifier::external_body]
pub(super) fn transfer(
    q: &QueuePair,
    local: &MemoryRegionSlice,
    remote: &RemoteCapability,
    id: u64,
    read: bool,
    fence: bool,
) -> (r: Result<Tracked<PostedTransfer>, RdmaError>)
    requires
        local.len == remote.len,
        local.len <= MAX_RDMA_MSG_LEN,
        local.well_formed(),
        remote.well_formed(),
    ensures
        r matches Ok(w) ==> w@.access == transfer_access(read, local.slot_ref(), remote.slot_ref())
            && w@.effect == (if read {
            copy_effect(remote.slot_ref(), local.slot_ref())
        } else {
            copy_effect(local.slot_ref(), remote.slot_ref())
        }) && w@.after == (if fence {
            q.reads()
        } else {
            ISet::empty()
        }),
{
    status(
        unsafe {
            vc_transfer(
                q.connection.raw.as_ptr(),
                local.addr,
                local.len as u32,
                local.mr.lkey,
                remote.addr,
                remote.rkey,
                read as i32,
                fence as i32,
                id,
            )
        },
    )?;
    Ok(
        Tracked(
            PostedTransfer {
                access: transfer_access(read, local.slot_ref(), remote.slot_ref()),
                effect: if read {
                    copy_effect(remote.slot_ref(), local.slot_ref())
                } else {
                    copy_effect(local.slot_ref(), remote.slot_ref())
                },
                after: if fence {
                    q.reads()
                } else {
                    ISet::empty()
                },
            },
        ),
    )
}

/// Coverage and ordering of one signaled scatter/gather WR.
#[verifier::external_body]
pub(super) fn transfer_sg(
    q: &QueuePair,
    locals: &[MemoryRegionSlice],
    remote: &RemoteCapability,
    id: u64,
    read: bool,
    fence: bool,
) -> (r: Result<Tracked<PostedTransfer>, RdmaError>)
    requires
        q.valid_sges(q.source().id(), locals@, *remote, read),
    ensures
        r matches Ok(w) ==> w@.access == sge_transfer_access(read, locals@, remote.slot_ref())
            && w@.effect == sge_copy_effect(read, locals@, remote.slot_ref()) && w@.after == (
        if fence {
            q.reads()
        } else {
            ISet::empty()
        }),
{
    if locals.len() > q.connection.config.max_send_sge as usize {
        return Err(RdmaError(22));
    }
    let mut storage = Scratch::<Sge, 32>::new();
    let sges = storage.buffer(locals.len())?;
    encode_sges(sges, locals, remote)?;
    let d = WrDesc::new(remote, id, sges.len() as u32, read, fence);
    status(unsafe { vc_transfer_sg(q.connection.raw.as_ptr(), &d, sges.as_ptr()) })?;
    Ok(
        Tracked(
            PostedTransfer {
                access: sge_transfer_access(read, locals@, remote.slot_ref()),
                effect: sge_copy_effect(read, locals@, remote.slot_ref()),
                after: if fence {
                    q.reads()
                } else {
                    ISet::empty()
                },
            },
        ),
    )
}

/// Evidence from the native doorbell. Only the accepted descriptors have a
/// copy effect. The caller supplies the QP read frontier at this WR's position.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub(super) tracked struct EnqueuedPrefix<'a> {
    ghost descriptors: Seq<TransferRequest<'a>>,
    ghost cursor: int,
}

impl<'a> EnqueuedPrefix<'a> {
    pub(super) closed spec fn descriptors(&self) -> Seq<TransferRequest<'a>> {
        self.descriptors
    }

    pub(super) closed spec fn cursor(&self) -> int {
        self.cursor
    }

    pub(super) proof fn work(
        tracked &mut self,
        i: int,
        reads: ISet<TransferEvent>,
    ) -> (tracked work: PostedTransfer)
        requires
            0 <= i < old(self).descriptors().len(),
            i == old(self).cursor(),
        ensures
            final(self).descriptors() == old(self).descriptors(),
            final(self).cursor() == old(self).cursor() + 1,
            ({
                let r = old(self).descriptors()[i];
                &&& work.access == sge_transfer_access(r.read, r.locals@, r.remote.slot_ref())
                &&& work.effect == sge_copy_effect(r.read, r.locals@, r.remote.slot_ref())
                &&& work.after == (if r.fence {
                    reads
                } else {
                    ISet::empty()
                })
            }),
    {
        let r = self.descriptors[i];
        self.cursor = self.cursor + 1;
        PostedTransfer {
            access: sge_transfer_access(r.read, r.locals@, r.remote.slot_ref()),
            effect: sge_copy_effect(r.read, r.locals@, r.remote.slot_ref()),
            after: if r.fence {
                reads
            } else {
                ISet::empty()
            },
        }
    }
}

/// Coverage, behavior and precedence: exactly the returned prefix was enqueued,
/// in descriptor order, with each descriptor's scatter list and fence bit.
#[verifier::external_body]
pub(super) fn post_batch<'a>(q: &QueuePair, requests: &[TransferRequest<'a>]) -> (r: (
    usize,
    Option<RdmaError>,
    Tracked<EnqueuedPrefix<'a>>,
))
    requires
        forall|i: int|
            0 <= i < requests.len() ==> {
                let r = #[trigger] requests@[i];
                q.valid_sges(q.source().id(), r.locals@, r.remote, r.read)
            },
    ensures
        if r.1.is_none() {
            r.0 == requests.len()
        } else {
            r.0 <= requests.len()
        },
        r.2@.descriptors() == requests@.take(r.0 as int),
        r.2@.cursor() == 0,
{
    let post = || -> Result<(usize, Option<RdmaError>), RdmaError>
        {
            if requests.is_empty() {
                return Ok((0, None));
            }
            let total = requests.iter().try_fold(
                0usize,
                |n, r| n.checked_add(r.locals.len()).ok_or(RdmaError(22)),
            )?;
            let mut descriptors = Scratch::<WrDesc, 32>::new();
            let mut segments = Scratch::<Sge, 64>::new();
            let descs = descriptors.buffer(requests.len())?;
            let sges = segments.buffer(total)?;
            let mut offset = 0;
            for (r, d) in requests.iter().zip(descs.iter_mut()) {
                let n = r.locals.len();
                if n > q.connection.config.max_send_sge as usize {
                    return Err(RdmaError(22));
                }
                encode_sges(&mut sges[offset..offset + n], r.locals, &r.remote)?;
                *d = WrDesc::new(&r.remote, r.wr_id, n as u32, r.read, r.fence);
                offset += n;
            }
            let mut posted = 0;
            let rc = unsafe {
                vc_post_batch(
                    q.connection.raw.as_ptr(),
                    descs.as_ptr(),
                    descs.len(),
                    sges.as_ptr(),
                    total,
                    &mut posted,
                )
            };
            Ok((posted, status(rc).err()))
        };
    let (count, error) = post().unwrap_or_else(|e| (0, Some(e)));
    (count, error, Tracked(EnqueuedPrefix { descriptors: requests@.take(count as int), cursor: 0 }))
}

#[derive(Clone, Copy)]
#[repr(u32)]
#[cfg_attr(not(verus_only), allow(dead_code))]
pub(super) enum WorkKind {
    Data,
    BindWindow,
    LocalInvalidate,
}

pub(super) open spec fn matches_pending(kind: WorkKind, pending: PendingKind) -> bool {
    match pending {
        PendingKind::DataTransfer { .. } => kind == WorkKind::Data,
        PendingKind::BindWindow { .. } => kind == WorkKind::BindWindow,
        PendingKind::LocalInvalidate { .. } => kind == WorkKind::LocalInvalidate,
    }
}

/// One poll consumes distinct cookies from this CQ, including failed WRs.
#[verifier::external_body]
pub(super) fn poll_batch(cq: &CompletionQueue, max_entries: usize) -> (r: Result<
    Vec<(u64, i32, WorkKind)>,
    RdmaError,
>)
    ensures
        r matches Ok(v) ==> v.len() <= max_entries && (forall|i: int|
            #![trigger v@[i]]
            0 <= i < v.len() ==> cq.pending().contains_key(v@[i].0) && matches_pending(
                v@[i].2,
                cq.pending()[v@[i].0].kind,
            )) && (forall|i: int, j: int|
            0 <= i < j < v.len() ==> (#[trigger] v@[i]).0 != (#[trigger] v@[j]).0),
{
    if max_entries == 0 {
        return Ok(Vec::new());
    }
    let max = i32::try_from(max_entries).map_err(|_| RdmaError(22))?;
    let n = unsafe { vc_poll_batch(cq.raw.as_ptr(), max) };
    if n == 0 {
        return Ok(Vec::new());
    }
    if n < 0 {
        return Err(RdmaError(n));
    }
    let mut result = Vec::new();
    result.try_reserve_exact(n as usize).map_err(|_| RdmaError(12))?;
    let mut raw = [RawCompletion::default();64];
    while result.len() < n as usize {
        let count = raw.len().min(n as usize - result.len());
        unsafe { vc_take_batch(cq.raw.as_ptr(), raw.as_mut_ptr(), count as i32) };
        for wc in &raw [..count] {
            result.push((wc.id, wc.status, wc.kind));
        }
    }
    Ok(result)
}

/// The HCA returns the cookie posted for this WR on this CQ, including on error.
#[verifier::external_body]
pub(super) fn poll(cq: &CompletionQueue) -> (r: Result<Option<(u64, i32, WorkKind)>, RdmaError>)
    ensures
        r matches Ok(Some((id, _, kind))) ==> cq.pending().contains_key(id) && matches_pending(
            kind,
            cq.pending()[id].kind,
        ),
{
    let mut wc = RawCompletion::default();
    match unsafe { vc_poll(cq.raw.as_ptr(), &mut wc.id, &mut wc.status, &mut wc.kind) } {
        0 => Ok(None),
        1 => Ok(Some((wc.id, wc.status, wc.kind))),
        e => Err(RdmaError(e)),
    }
}

#[verifier::external_body]
pub(super) fn close(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { ibv_close_device(p) })
}

#[verifier::external_body]
pub(super) fn alloc_pd(p: *mut u8) -> Result<RawPdGuard, RdmaError> {
    checked(unsafe { ibv_alloc_pd(p) }).map(RawPdGuard::new)
}

#[verifier::external_body]
pub(super) fn dealloc_pd(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { ibv_dealloc_pd(p) })
}

#[verifier::external_body]
pub(super) fn destroy_qp(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { ibv_destroy_qp(p) })
}

#[verifier::external_body]
pub(super) fn destroy_cq(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { vc_destroy_cq(p) })
}

#[verifier::external_body]
pub(super) fn dealloc_window(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { vc_dealloc_window(p) })
}

#[verifier::external_body]
pub(super) fn error(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { vc_error(p) })
}

#[verifier::external_body]
pub(super) fn reset(p: *mut u8) -> Result<(), RdmaError> {
    status(unsafe { vc_reset(p) })
}

} // verus!
