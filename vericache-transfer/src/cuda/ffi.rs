//! CUDA Runtime ABI and hardware premises for CUDA streams, events, and
//! GPUDirect RDMA visibility.
use super::*;
use std::ffi::c_void;
use std::ptr;

#[cfg_attr(not(verus_only), link(name = "cudart"))]
unsafe extern "C" {
    fn cudaGetDriverEntryPointByVersion(
        symbol: *const std::ffi::c_char,
        function: *mut *mut c_void,
        version: u32,
        flags: u64,
        query_status: *mut i32,
    ) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
    fn cudaEventQuery(event: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaDeviceFlushGPUDirectRDMAWrites(target: i32, scope: i32) -> i32;
    fn cudaDeviceCanAccessPeer(can_access: *mut i32, device: i32, peer: i32) -> i32;
    fn cudaDeviceEnablePeerAccess(peer: i32, flags: u32) -> i32;
}

const CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: i32 = 6;
const DEV_ATTR_GDR_FLUSH_WRITES_OPTIONS: i32 = 117;
const DEV_ATTR_GDR_WRITES_ORDERING: i32 = 118;
const GDR_FLUSH_WRITES_OPTION_HOST: i32 = 1;
const GDR_FLUSH_TARGET_CURRENT_DEVICE: i32 = 0;
const GDR_FLUSH_SCOPE_TO_OWNER: i32 = 100;
const STREAM_NON_BLOCKING: u32 = 1;
const EVENT_DISABLE_TIMING: u32 = 2;
const HOST_REGISTER_PORTABLE: u32 = 1;
const MEMCPY_DEFAULT: i32 = 4;
const ERROR_INVALID_DEVICE: i32 = 101;
const ERROR_NOT_READY: i32 = 600;
const ERROR_PEER_ACCESS_ALREADY_ENABLED: i32 = 704;

fn disarm_cuda_ptr(slot: &mut *mut u8) -> *mut u8 {
    let raw = *slot;
    *slot = ptr::null_mut();
    raw
}

fn drop_cuda_ptr(slot: &mut *mut u8, destroy: unsafe extern "C" fn(*mut c_void) -> i32) {
    let raw = disarm_cuda_ptr(slot);
    if !raw.is_null() {
        let _ = unsafe { destroy(raw.cast()) };
    }
}

verus! {

#[verifier::external_body]
pub(super) struct RawStreamGuard(*mut u8);

impl RawStreamGuard {
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
        let raw = disarm_cuda_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawStreamGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_cuda_ptr(&mut self.0, cudaStreamDestroy);
    }
}

#[verifier::external_body]
pub(super) struct RawEventGuard(*mut u8);

impl RawEventGuard {
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
        let raw = disarm_cuda_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawEventGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_cuda_ptr(&mut self.0, cudaEventDestroy);
    }
}

#[verifier::external_body]
pub(super) struct RawHostRegGuard(*mut u8);

impl RawHostRegGuard {
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
        let raw = disarm_cuda_ptr(&mut self.0);
        std::mem::forget(self);
        raw
    }
}

impl Drop for RawHostRegGuard {
    #[verifier::external_body]
    fn drop(&mut self)
        opens_invariants none
        no_unwind
    {
        drop_cuda_ptr(&mut self.0, cudaHostUnregister);
    }
}

fn status(code: i32) -> (r: Result<(), CudaError>)
    ensures
        r.is_ok() == (code == 0),
{
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError(code))
    }
}

/// Select the device on this calling thread before using its handles.
#[verifier::external_body]
fn set_device(device: DeviceId) -> Result<(), CudaError> {
    if device > i32::MAX as u32 {
        return Err(CudaError(ERROR_INVALID_DEVICE));
    }
    status(unsafe { cudaSetDevice(device as i32) })
}

#[verifier::external_body]
pub(super) fn raw_device_count() -> Result<u32, CudaError> {
    let mut count = 0;
    status(unsafe { cudaGetDeviceCount(&mut count) })?;
    Ok(count as u32)
}

#[verifier::external_body]
pub(super) fn raw_gdr_writes_ordering(device: i32) -> Result<GdrWritesOrdering, CudaError> {
    let mut val = 0;
    status(unsafe { cudaDeviceGetAttribute(&mut val, DEV_ATTR_GDR_WRITES_ORDERING, device) })?;
    Ok(
        if val >= 200 {
            GdrWritesOrdering::AllDevices
        } else if val >= 100 {
            GdrWritesOrdering::Owner
        } else {
            GdrWritesOrdering::None
        },
    )
}

#[verifier::external_body]
pub(super) fn raw_gdr_flush_writes_supported(device: i32) -> Result<bool, CudaError> {
    let mut val = 0;
    status(unsafe { cudaDeviceGetAttribute(&mut val, DEV_ATTR_GDR_FLUSH_WRITES_OPTIONS, device) })?;
    Ok((val & GDR_FLUSH_WRITES_OPTION_HOST) != 0)
}

#[verifier::external_body]
pub(super) fn raw_flush_gdr_writes(device: i32) -> Result<(), CudaError> {
    status(unsafe { cudaSetDevice(device) })?;
    status(
        unsafe {
            cudaDeviceFlushGPUDirectRDMAWrites(
                GDR_FLUSH_TARGET_CURRENT_DEVICE,
                GDR_FLUSH_SCOPE_TO_OWNER,
            )
        },
    )
}

/// Enable synchronous CUDA visibility for peer-memory access to this allocation.
#[allow(dead_code)]
#[verifier::external_body]
pub(super) fn raw_enable_sync_memops(buf: &CudaBuffer) -> Result<(), CudaError>
    requires
        buf.well_formed(),
{
    let CudaPointerKind::Device(device) = buf.memory.kind else { unreachable!() };
    set_device(device)?;
    let mut function = ptr::null_mut();
    status(
        unsafe {
            cudaGetDriverEntryPointByVersion(
                b"cuPointerSetAttribute\0".as_ptr().cast(),
                &mut function,
                11000,
                0,
                ptr::null_mut(),
            )
        },
    )?;
    if function.is_null() {
        return Err(CudaError(801));
    }
    type SetAttribute = unsafe extern "C" fn (*const c_void, i32, u64) -> i32;
    let set: SetAttribute = unsafe { std::mem::transmute(function) };
    let sync_val: u32 = 1;
    status(
        unsafe {
            set(
                (&sync_val as *const u32).cast(),
                CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                buf.memory.ptr.addr() as u64,
            )
        },
    )
}

/// Coverage: a fresh live device allocation on the selected GPU, zero-initialized.
#[verifier::external_body]
pub(super) fn raw_alloc_device(device: DeviceId, len: usize) -> (r: Result<*mut u8, CudaError>)
    ensures
        r matches Ok(ptr) ==> ptr.addr() != 0 && ptr.addr() + len <= usize::MAX,
{
    set_device(device)?;
    let mut ptr = ptr::null_mut();
    let alloc_len = len.max(1);
    status(unsafe { cudaMalloc(&mut ptr, alloc_len) })?;
    if let Err(e) = status(unsafe { cudaMemset(ptr, 0, alloc_len) }) {
        let _ = unsafe { cudaFree(ptr) };
        return Err(e);
    }
    Ok(ptr.cast())
}

/// Coverage: a fresh page-locked allocation, initialized for CPU byte reads.
#[verifier::external_body]
pub(super) fn raw_alloc_host(len: usize) -> (r: Result<*mut u8, CudaError>)
    ensures
        r matches Ok(ptr) ==> ptr.addr() != 0 && ptr.addr() + len <= usize::MAX,
{
    let mut ptr = ptr::null_mut();
    status(unsafe { cudaHostAlloc(&mut ptr, len.max(1), 0) })?;
    unsafe {
        ptr.cast::<u8>().write_bytes(0, len);
    }
    Ok(ptr.cast())
}

/// Coverage: pin the existing CPU byte range in place.
#[verifier::external_body]
pub(super) fn raw_host_register(buffer: &HostBuffer) -> Result<RawHostRegGuard, CudaError>
    requires
        buffer.well_formed(),
{
    let (ptr, len) = buffer.raw_parts();
    if len == 0 {
        return Err(CudaError::INVALID_VALUE);
    }
    status(unsafe { cudaHostRegister(ptr.cast(), len, HOST_REGISTER_PORTABLE) })?;
    Ok(RawHostRegGuard::new(ptr))
}

/// Precedence: non-blocking FIFO, independent of the legacy default stream.
#[verifier::external_body]
pub(super) fn raw_stream_create(device: DeviceId) -> Result<RawStreamGuard, CudaError> {
    set_device(device)?;
    let mut stream = ptr::null_mut();
    status(unsafe { cudaStreamCreateWithFlags(&mut stream, STREAM_NON_BLOCKING) })?;
    Ok(RawStreamGuard::new(stream.cast()))
}

#[verifier::external_body]
pub(super) fn raw_event_create(device: DeviceId) -> Result<RawEventGuard, CudaError> {
    set_device(device)?;
    let mut event = ptr::null_mut();
    status(unsafe { cudaEventCreateWithFlags(&mut event, EVENT_DISABLE_TIMING) })?;
    Ok(RawEventGuard::new(event.cast()))
}

#[verifier::external_body]
pub(super) fn raw_stream_destroy(device: DeviceId, guard: RawStreamGuard) -> Result<(), CudaError> {
    set_device(device)?;
    let raw = guard.into_raw();
    status(unsafe { cudaStreamDestroy(raw.cast()) })
}

#[verifier::external_body]
pub(super) fn raw_event_destroy(device: DeviceId, guard: RawEventGuard) -> Result<(), CudaError> {
    set_device(device)?;
    let raw = guard.into_raw();
    status(unsafe { cudaEventDestroy(raw.cast()) })
}

/// Precedence: capture exactly the work preceding this record.
#[verifier::external_body]
pub(super) fn raw_event_record(event: &CudaEvent, stream: &CudaStream) -> Result<(), CudaError> {
    set_device(stream.device)?;
    status(unsafe { cudaEventRecord(event.raw.as_ptr().cast(), stream.raw.as_ptr().cast()) })
}

/// Precedence: order future stream work after the recording current at this call.
#[verifier::external_body]
pub(super) fn raw_stream_wait_event(stream: &CudaStream, event: &CudaEvent) -> Result<
    (),
    CudaError,
> {
    set_device(stream.device)?;
    status(unsafe { cudaStreamWaitEvent(stream.raw.as_ptr().cast(), event.raw.as_ptr().cast(), 0) })
}

/// Observation: only success establishes host visibility of the recorded work.
#[verifier::external_body]
pub(super) fn raw_event_query(event: &CudaEvent) -> Result<bool, CudaError> {
    set_device(event.device)?;
    match unsafe { cudaEventQuery(event.raw.as_ptr().cast()) } {
        0 => Ok(true),
        ERROR_NOT_READY => Ok(false),
        code => Err(CudaError(code)),
    }
}

/// Observation: success establishes host visibility of all preceding stream work.
#[verifier::external_body]
pub(super) fn raw_stream_synchronize(stream: &CudaStream) -> Result<(), CudaError> {
    set_device(stream.device)?;
    status(unsafe { cudaStreamSynchronize(stream.raw.as_ptr().cast()) })
}

/// Enable the stream GPU to access the remote allocation when the topology
/// supports it. CUDA also supports staged inter-device copies without P2P.
#[verifier::external_body]
fn enable_peer_access(device: DeviceId, peer: DeviceId) -> Result<(), CudaError> {
    if device > i32::MAX as u32 || peer > i32::MAX as u32 {
        return Err(CudaError(ERROR_INVALID_DEVICE));
    }
    let mut accessible = 0;
    status(unsafe { cudaDeviceCanAccessPeer(&mut accessible, device as i32, peer as i32) })?;
    if accessible != 0 {
        let code = unsafe { cudaDeviceEnablePeerAccess(peer as i32, 0) };
        if code != ERROR_PEER_ACCESS_ALREADY_ENABLED {
            status(code)?;
        }
    }
    Ok(())
}

/// Coverage, behavior, precedence: enqueue a byte-for-byte copy on this FIFO.
/// This witness describes submission only. Visibility requires a later observation.
#[verifier::external_body]
pub(super) fn raw_memcpy_async(stream: &CudaStream, dst: &CudaSlice, src: &CudaSlice) -> (r: Result<
    Tracked<EnqueuedCopy>,
    CudaError,
>)
    requires
        dst.well_formed(),
        src.well_formed(),
        dst.slot_ref().sub_len == src.slot_ref().sub_len,
        src.slot_ref().loc != dst.slot_ref().loc,
    ensures
        r matches Ok(w) ==> w@.access == copy_access(src.slot_ref(), dst.slot_ref()) && w@.effect
            == copy_effect(src.slot_ref(), dst.slot_ref()),
{
    set_device(stream.device)?;
    if let (CudaPointerKind::Device(a), CudaPointerKind::Device(b)) = (
        src.memory.kind,
        dst.memory.kind,
    ) {
        if a != b {
            enable_peer_access(
                stream.device,
                if stream.device == a {
                    b
                } else {
                    a
                },
            )?;
        }
    }
    status(
        unsafe {
            cudaMemcpyAsync(
                dst.memory.ptr.cast(),
                src.memory.ptr.cast(),
                src.memory.len,
                MEMCPY_DEFAULT,
                stream.raw.as_ptr().cast(),
            )
        },
    )?;
    Ok(
        Tracked(
            EnqueuedCopy {
                access: copy_access(src.slot_ref(), dst.slot_ref()),
                effect: copy_effect(src.slot_ref(), dst.slot_ref()),
            },
        ),
    )
}

/// Synchronous reclamation of the base allocation on success.
#[verifier::external_body]
pub(super) fn raw_free_device(buf: &CudaBuffer) -> (r: Result<Tracked<CompletedAccess>, CudaError>)
    requires
        buf.well_formed(),
    ensures
        r matches Ok(w) ==> w@.access == reclaim_slab_access(
            buf.node(),
            buf.space(),
            buf.num_slots() as nat,
        ) && w@.effect == havoc_effect(),
{
    let CudaPointerKind::Device(device) = buf.memory.kind else { unreachable!() };
    set_device(device)?;
    status(unsafe { cudaFree(buf.memory.ptr.cast()) })?;
    Ok(
        Tracked(
            CompletedAccess {
                access: reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
                effect: havoc_effect(),
            },
        ),
    )
}

/// Synchronous reclamation of the base allocation on success.
#[verifier::external_body]
pub(super) fn raw_free_host(buf: &CudaHostBuffer) -> (r: Result<
    Tracked<CompletedAccess>,
    CudaError,
>)
    requires
        buf.well_formed(),
    ensures
        r matches Ok(w) ==> w@.access == reclaim_slab_access(
            buf.node(),
            buf.space(),
            buf.num_slots() as nat,
        ) && w@.effect == havoc_effect(),
{
    status(unsafe { cudaFreeHost(buf.memory.ptr.cast()) })?;
    Ok(
        Tracked(
            CompletedAccess {
                access: reclaim_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
                effect: havoc_effect(),
            },
        ),
    )
}

/// Synchronous deregistration preserves every slot on success.
#[verifier::external_body]
pub(super) fn raw_host_unregister(buf: &CudaHostRegistration<'_>) -> (r: Result<
    Tracked<CompletedAccess>,
    CudaError,
>)
    requires
        buf.well_formed(),
    ensures
        r matches Ok(w) ==> w@.access == dereg_slab_access(
            buf.node(),
            buf.space(),
            buf.num_slots() as nat,
        ) && w@.effect == (if buf.num_slots() == 1 {
            identity_effect(buf.slot_loc(0))
        } else {
            identity_effect_slab()
        }),
{
    status(unsafe { cudaHostUnregister(buf.guard.as_ptr().cast()) })?;
    Ok(
        Tracked(
            CompletedAccess {
                access: dereg_slab_access(buf.node(), buf.space(), buf.num_slots() as nat),
                effect: if buf.num_slots() == 1 {
                    identity_effect(buf.slot_loc(0))
                } else {
                    identity_effect_slab()
                },
            },
        ),
    )
}

} // verus!
