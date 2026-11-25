//! The C handles are opaque. See the safety contract on each export.
use super::engine::{Error, Lease, Options, object_key};
use super::{StagingBuffer, VericacheVllmEngine};
use crate::set_last_error;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;
use vericache_core::spec::ObjectKey;

const INVALID: i32 = -1;
const BACKEND: i32 = -2;
const SHORT_BUFFER: i32 = -3;
const BUSY: i32 = -4;

fn status(error: Error) -> i32 {
    match error {
        Error::Invalid => INVALID,
        Error::Backend => BACKEND,
        Error::Capacity(_) => SHORT_BUFFER,
        Error::Busy => BUSY,
    }
}

fn boundary<T: Copy>(fallback: T, call: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or(fallback)
}

/// # Safety
/// A nonempty path must point to readable UTF-8 bytes for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_create(
    num_shards: usize,
    host_frames_per_shard: usize,
    nvme_frames_per_shard: usize,
    frame_size: usize,
    nvme_path_ptr: *const u8,
    nvme_path_len: usize,
) -> *mut VericacheVllmEngine {
    set_last_error("");
    boundary(ptr::null_mut(), || {
        let Ok(path) = (unsafe { path_from_raw(nvme_path_ptr, nvme_path_len) }) else {
            return ptr::null_mut();
        };
        VericacheVllmEngine::new(
            num_shards,
            host_frames_per_shard,
            nvme_frames_per_shard,
            frame_size,
            path,
        )
        .map_or(ptr::null_mut(), |engine| Box::into_raw(Box::new(engine)))
    })
}

/// # Safety
/// Pass null or the sole ownership of a live engine, after all other calls finish.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_destroy(engine: *mut VericacheVllmEngine) {
    if !engine.is_null() {
        boundary((), || drop(unsafe { Box::from_raw(engine) }));
    }
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_frame_size(engine: *const VericacheVllmEngine) -> usize {
    unsafe { engine.as_ref() }.map_or(0, VericacheVllmEngine::frame_size)
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_num_shards(engine: *const VericacheVllmEngine) -> usize {
    unsafe { engine.as_ref() }.map_or(0, VericacheVllmEngine::num_shards)
}

/// # Safety
/// A nonnull engine must stay live. A nonempty source must be readable and immutable for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_put(
    engine: *mut VericacheVllmEngine,
    key: u64,
    src_ptr: *const u8,
    len: usize,
) -> i32 {
    unsafe { put_impl(engine, object_key(key), src_ptr, len) }
}

/// # Safety
/// Follows the contract of `vericache_engine_put`. The key
/// pointer must address four readable, aligned little-endian u64 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_put_key256(
    engine: *mut VericacheVllmEngine,
    key4_ptr: *const u64,
    src_ptr: *const u8,
    len: usize,
) -> i32 {
    let Some(key) = (unsafe { read_key256(key4_ptr) }) else {
        return INVALID;
    };
    unsafe { put_impl(engine, key, src_ptr, len) }
}

unsafe fn put_impl(
    engine: *mut VericacheVllmEngine,
    key: ObjectKey,
    src_ptr: *const u8,
    len: usize,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if len > engine.frame_size() || (len != 0 && src_ptr.is_null()) {
            return INVALID;
        }
        let src = if len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(src_ptr, len) }
        };
        engine.put(key, src).map_or_else(status, |()| 0)
    })
}

/// # Safety
/// A nonnull engine must stay live. The destination and output length must be
/// disjoint storage, exclusive to this call. Destination bytes must be initialized
/// and writable. A zero capacity may use null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_get(
    engine: *mut VericacheVllmEngine,
    key: u64,
    dst_ptr: *mut u8,
    dst_cap: usize,
    out_len: *mut usize,
) -> i32 {
    unsafe { get_impl(engine, object_key(key), dst_ptr, dst_cap, out_len) }
}

/// # Safety
/// Follows the contract of `vericache_engine_get`. The key
/// pointer must address four readable, aligned little-endian u64 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_get_key256(
    engine: *mut VericacheVllmEngine,
    key4_ptr: *const u64,
    dst_ptr: *mut u8,
    dst_cap: usize,
    out_len: *mut usize,
) -> i32 {
    let Some(key) = (unsafe { read_key256(key4_ptr) }) else {
        return INVALID;
    };
    unsafe { get_impl(engine, key, dst_ptr, dst_cap, out_len) }
}

unsafe fn get_impl(
    engine: *mut VericacheVllmEngine,
    key: ObjectKey,
    dst_ptr: *mut u8,
    dst_cap: usize,
    out_len: *mut usize,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if out_len.is_null() || (dst_cap != 0 && dst_ptr.is_null()) || dst_cap > isize::MAX as usize
        {
            return INVALID;
        }
        unsafe { out_len.write(0) };
        let dst = if dst_cap == 0 {
            &mut []
        } else {
            unsafe { slice::from_raw_parts_mut(dst_ptr, dst_cap) }
        };
        match engine.get(key, dst) {
            Ok(Some(len)) => {
                unsafe { out_len.write(len) };
                0
            }
            Ok(None) => 1,
            Err(Error::Capacity(required)) => {
                unsafe { out_len.write(required) };
                SHORT_BUFFER
            }
            Err(error) => status(error),
        }
    })
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_contains(
    engine: *mut VericacheVllmEngine,
    key: u64,
) -> i32 {
    unsafe { contains_impl(engine, object_key(key)) }
}

/// # Safety
/// Follows the contract of `vericache_engine_contains`. The key
/// pointer must address four readable, aligned little-endian u64 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_contains_key256(
    engine: *mut VericacheVllmEngine,
    key4_ptr: *const u64,
) -> i32 {
    let Some(key) = (unsafe { read_key256(key4_ptr) }) else {
        return INVALID;
    };
    unsafe { contains_impl(engine, key) }
}

unsafe fn contains_impl(engine: *mut VericacheVllmEngine, key: ObjectKey) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        engine.contains(key).map_or_else(status, i32::from)
    })
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_has_cold_data(engine: *const VericacheVllmEngine) -> i32 {
    unsafe { engine.as_ref() }.map_or(INVALID, |engine| i32::from(engine.has_cold_data()))
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_evict(engine: *mut VericacheVllmEngine, key: u64) -> i32 {
    unsafe { evict_impl(engine, object_key(key)) }
}

/// # Safety
/// Follows the contract of `vericache_engine_evict`. The key
/// pointer must address four readable, aligned little-endian u64 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_evict_key256(
    engine: *mut VericacheVllmEngine,
    key4_ptr: *const u64,
) -> i32 {
    let Some(key) = (unsafe { read_key256(key4_ptr) }) else {
        return INVALID;
    };
    unsafe { evict_impl(engine, key) }
}

unsafe fn evict_impl(engine: *mut VericacheVllmEngine, key: ObjectKey) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        engine
            .evict(key)
            .map_or_else(status, |found| i32::from(!found))
    })
}

/// # Safety
/// A nonnull engine must stay live. Output pointers must be writable and exclusive for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_stats(
    engine: *const VericacheVllmEngine,
    out_puts: *mut u64,
    out_gets: *mut u64,
    out_hits: *mut u64,
    out_misses: *mut u64,
    out_evictions: *mut u64,
    out_bytes_written: *mut u64,
    out_bytes_read: *mut u64,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let pointers = [
            out_puts,
            out_gets,
            out_hits,
            out_misses,
            out_evictions,
            out_bytes_written,
            out_bytes_read,
        ];
        if pointers.iter().any(|pointer| pointer.is_null()) {
            return INVALID;
        }
        let s = match engine.stats() {
            Ok(s) => s,
            Err(e) => return status(e),
        };
        let values = [
            s.puts,
            s.gets,
            s.hits,
            s.misses,
            s.evictions,
            s.bytes_written,
            s.bytes_read,
        ];
        for (output, value) in pointers.into_iter().zip(values) {
            unsafe { output.write(value) };
        }
        0
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn vericache_staging_alloc(
    size: usize,
    prefer_cuda_pinned: i32,
) -> *mut StagingBuffer {
    set_last_error("");
    boundary(ptr::null_mut(), || {
        StagingBuffer::allocate(size, prefer_cuda_pinned != 0)
            .map_or(ptr::null_mut(), |buf| Box::into_raw(Box::new(buf)))
    })
}

/// # Safety
/// A nonnull buffer must stay live while callers use the returned storage.
/// Callers must coordinate any reads and writes, including device transfers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_staging_ptr(buf: *mut StagingBuffer) -> *mut u8 {
    unsafe { buf.as_ref() }.map_or(ptr::null_mut(), |buf| {
        ptr::with_exposed_provenance_mut(buf.address())
    })
}

/// # Safety
/// A nonnull buffer must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_staging_len(buf: *const StagingBuffer) -> usize {
    unsafe { buf.as_ref() }.map_or(0, StagingBuffer::len)
}

/// # Safety
/// A nonnull buffer must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_staging_is_cuda_pinned(buf: *const StagingBuffer) -> i32 {
    unsafe { buf.as_ref() }.map_or(INVALID, |buf| i32::from(buf.is_cuda_pinned()))
}

/// # Safety
/// Pass null or the sole ownership of a live buffer, with no remaining users.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_staging_free(buf: *mut StagingBuffer) {
    if !buf.is_null() {
        boundary((), || drop(unsafe { Box::from_raw(buf) }));
    }
}

unsafe fn path_from_raw<'a>(path: *const u8, len: usize) -> Result<Option<&'a str>, Error> {
    if len == 0 {
        return Ok(None);
    }
    if !valid_array(path, len) {
        return Err(Error::Invalid);
    }
    let bytes = unsafe { slice::from_raw_parts(path, len) };
    std::str::from_utf8(bytes)
        .ok()
        .filter(|path| !path.contains('\0'))
        .map(Some)
        .ok_or(Error::Invalid)
}

/// # Safety
/// The path follows the contract of vericache_engine_create.
/// Flag bit 0 requests CUDA registration. Bit 1 enables eviction reporting.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_create_with_options(
    num_shards: usize,
    host_frames: usize,
    nvme_frames: usize,
    frame_size: usize,
    path: *const u8,
    path_len: usize,
    flags: u32,
) -> *mut VericacheVllmEngine {
    set_last_error("");
    boundary(ptr::null_mut(), || {
        let Ok(path) = (unsafe { path_from_raw(path, path_len) }) else {
            return ptr::null_mut();
        };
        if flags & !3 != 0 {
            return ptr::null_mut();
        }
        let options = Options {
            pin_cuda: flags & 1 != 0,
            track_evictions: flags & 2 != 0,
        };
        VericacheVllmEngine::with_options(
            num_shards,
            host_frames,
            nvme_frames,
            frame_size,
            path,
            options,
        )
        .map_or(ptr::null_mut(), |engine| Box::into_raw(Box::new(engine)))
    })
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_is_slab_cuda_pinned(
    engine: *const VericacheVllmEngine,
) -> i32 {
    unsafe { engine.as_ref() }.map_or(INVALID, |engine| i32::from(engine.is_cuda_pinned()))
}

unsafe fn write_lease(
    lease: Result<Option<Lease>, Error>,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
    out_id: *mut u64,
) -> i32 {
    let (ptr, len, id, code) = match lease {
        Ok(Some(lease)) => (
            ptr::with_exposed_provenance(lease.address),
            lease.len,
            lease.id,
            0,
        ),
        Ok(None) => (ptr::null(), 0, 0, 1),
        Err(error) => (ptr::null(), 0, 0, status(error)),
    };
    unsafe {
        out_ptr.write(ptr);
        out_len.write(len);
        out_id.write(id);
    }
    code
}

/// # Safety
/// The live engine and all outputs follow the contract of vericache_engine_get.
/// A hit returns immutable bytes. Finish every use, including asynchronous
/// transfers, before releasing the lease or destroying the engine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_acquire_read_lease(
    engine: *mut VericacheVllmEngine,
    key: u64,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
    out_lease_id: *mut u64,
) -> i32 {
    unsafe { acquire_read_lease_impl(engine, object_key(key), out_ptr, out_len, out_lease_id) }
}

/// # Safety
/// Follows the contract of `vericache_engine_acquire_read_lease`. The key
/// pointer must address four readable, aligned little-endian u64 words.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_acquire_read_lease_key256(
    engine: *mut VericacheVllmEngine,
    key4_ptr: *const u64,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
    out_lease_id: *mut u64,
) -> i32 {
    let Some(key) = (unsafe { read_key256(key4_ptr) }) else {
        return INVALID;
    };
    unsafe { acquire_read_lease_impl(engine, key, out_ptr, out_len, out_lease_id) }
}

unsafe fn acquire_read_lease_impl(
    engine: *mut VericacheVllmEngine,
    key: ObjectKey,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
    out_lease_id: *mut u64,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if out_ptr.is_null() || out_len.is_null() || out_lease_id.is_null() {
            return INVALID;
        }
        unsafe { write_lease(engine.acquire(key), out_ptr, out_len, out_lease_id) }
    })
}

/// # Safety
/// The engine must stay live and users of the lease must have finished.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_release_read_lease(
    engine: *mut VericacheVllmEngine,
    lease_id: u64,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        engine.release(lease_id).map_or_else(status, |()| 0)
    })
}

fn valid_array<T>(ptr: *const T, count: usize) -> bool {
    count == 0
        || (!ptr.is_null() && ptr.is_aligned() && count <= (isize::MAX as usize) / size_of::<T>())
}

trait AbiKey: Copy {
    fn into_key(self) -> ObjectKey;
}

impl AbiKey for u64 {
    fn into_key(self) -> ObjectKey {
        object_key(self)
    }
}

impl AbiKey for [u64; 4] {
    fn into_key(self) -> ObjectKey {
        ObjectKey(self.map(u64::from_le))
    }
}

unsafe fn read_key256(key: *const u64) -> Option<ObjectKey> {
    if key.is_null() || !key.is_aligned() {
        return None;
    }
    Some(unsafe { key.cast::<[u64; 4]>().read() }.into_key())
}

unsafe fn input_array<'a, T>(ptr: *const T, count: usize) -> Option<&'a [T]> {
    if count == 0 {
        Some(&[])
    } else if valid_array(ptr, count) {
        Some(unsafe { slice::from_raw_parts(ptr, count) })
    } else {
        None
    }
}

/// # Safety
/// All arrays have count elements. Inputs are readable and outputs are
/// disjoint writable storage for the call. Individual statuses match the
/// single-key operation. A zero count permits null arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_acquire_read_lease_batch(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const u64,
    count: usize,
    out_ptrs: *mut *const u8,
    out_lens: *mut usize,
    out_lease_ids: *mut u64,
    out_statuses: *mut i32,
) -> i32 {
    unsafe {
        acquire_read_lease_batch_impl(
            engine,
            keys_ptr,
            count,
            out_ptrs,
            out_lens,
            out_lease_ids,
            out_statuses,
        )
    }
}

/// # Safety
/// Follows the contract of `vericache_engine_acquire_read_lease_batch`, using
/// `count` contiguous keys of four little-endian u64 words each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_acquire_read_lease_batch_key256(
    engine: *mut VericacheVllmEngine,
    keys4_ptr: *const u64,
    count: usize,
    out_ptrs: *mut *const u8,
    out_lens: *mut usize,
    out_lease_ids: *mut u64,
    out_statuses: *mut i32,
) -> i32 {
    unsafe {
        acquire_read_lease_batch_impl(
            engine,
            keys4_ptr.cast::<[u64; 4]>(),
            count,
            out_ptrs,
            out_lens,
            out_lease_ids,
            out_statuses,
        )
    }
}

unsafe fn acquire_read_lease_batch_impl<K: AbiKey>(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const K,
    count: usize,
    out_ptrs: *mut *const u8,
    out_lens: *mut usize,
    out_lease_ids: *mut u64,
    out_statuses: *mut i32,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let Some(keys) = (unsafe { input_array(keys_ptr, count) }) else {
            return INVALID;
        };
        if !valid_array(out_ptrs, count)
            || !valid_array(out_lens, count)
            || !valid_array(out_lease_ids, count)
            || !valid_array(out_statuses, count)
        {
            return INVALID;
        }
        for (i, lease) in engine
            .acquire_batch(keys.iter().copied().map(AbiKey::into_key))
            .into_iter()
            .enumerate()
        {
            unsafe {
                let code = write_lease(
                    lease,
                    out_ptrs.add(i),
                    out_lens.add(i),
                    out_lease_ids.add(i),
                );
                out_statuses.add(i).write(code);
            }
        }
        0
    })
}

/// # Safety
/// The live engine and lease IDs follow the single-key release contract.
/// The input array has count readable elements. Every ID is attempted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_release_read_lease_batch(
    engine: *mut VericacheVllmEngine,
    lease_ids_ptr: *const u64,
    count: usize,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let Some(ids) = (unsafe { input_array(lease_ids_ptr, count) }) else {
            return INVALID;
        };
        engine.release_batch(ids).map_or_else(status, |()| 0)
    })
}

/// # Safety
/// With a live engine, supply count readable keys and count writable bytes.
/// These arrays must be disjoint. A zero count permits null arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_contains_batch(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const u64,
    count: usize,
    out_present: *mut u8,
) -> i32 {
    unsafe { contains_batch_impl(engine, keys_ptr, count, out_present) }
}

/// # Safety
/// Follows the contract of `vericache_engine_contains_batch`, using
/// `count` contiguous keys of four little-endian u64 words each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_contains_batch_key256(
    engine: *mut VericacheVllmEngine,
    keys4_ptr: *const u64,
    count: usize,
    out_present: *mut u8,
) -> i32 {
    unsafe { contains_batch_impl(engine, keys4_ptr.cast::<[u64; 4]>(), count, out_present) }
}

unsafe fn contains_batch_impl<K: AbiKey>(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const K,
    count: usize,
    out_present: *mut u8,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let Some(keys) = (unsafe { input_array(keys_ptr, count) }) else {
            return INVALID;
        };
        if !valid_array(out_present, count) {
            return INVALID;
        }
        for (i, present) in engine
            .contains_batch(keys.iter().copied().map(AbiKey::into_key))
            .into_iter()
            .enumerate()
        {
            match present {
                Ok(present) => unsafe { out_present.add(i).write(u8::from(present)) },
                Err(error) => return status(error),
            }
        }
        0
    })
}

/// # Safety
/// With a live engine, all arrays have count elements. Inputs and source
/// buffers follow the single-key put contract and must not alias statuses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_put_batch(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const u64,
    src_ptrs: *const *const u8,
    lens_ptr: *const usize,
    count: usize,
    out_statuses: *mut i32,
) -> i32 {
    unsafe { put_batch_impl(engine, keys_ptr, src_ptrs, lens_ptr, count, out_statuses) }
}

/// # Safety
/// Follows the contract of `vericache_engine_put_batch`, using
/// `count` contiguous keys of four little-endian u64 words each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_put_batch_key256(
    engine: *mut VericacheVllmEngine,
    keys4_ptr: *const u64,
    src_ptrs: *const *const u8,
    lens_ptr: *const usize,
    count: usize,
    out_statuses: *mut i32,
) -> i32 {
    unsafe {
        put_batch_impl(
            engine,
            keys4_ptr.cast::<[u64; 4]>(),
            src_ptrs,
            lens_ptr,
            count,
            out_statuses,
        )
    }
}

unsafe fn put_batch_impl<K: AbiKey>(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const K,
    src_ptrs: *const *const u8,
    lens_ptr: *const usize,
    count: usize,
    out_statuses: *mut i32,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let (Some(keys), Some(ptrs), Some(lens)) = (unsafe {
            (
                input_array(keys_ptr, count),
                input_array(src_ptrs, count),
                input_array(lens_ptr, count),
            )
        }) else {
            return INVALID;
        };
        if !valid_array(out_statuses, count) {
            return INVALID;
        }
        let sources = ptrs
            .iter()
            .zip(lens)
            .map(|(&ptr, &len)| {
                if len > engine.frame_size() {
                    return None;
                }
                unsafe { input_array(ptr, len) }
            })
            .collect();
        for (i, result) in engine
            .put_batch(keys.iter().copied().map(AbiKey::into_key), sources)
            .into_iter()
            .enumerate()
        {
            unsafe {
                out_statuses
                    .add(i)
                    .write(result.map_or_else(status, |()| 0))
            };
        }
        0
    })
}

/// # Safety
/// With a live engine, all arrays have count elements. Destinations must be
/// initialized, writable and pairwise disjoint from all other buffers or arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_get_batch(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const u64,
    dst_ptrs: *const *mut u8,
    dst_caps: *const usize,
    count: usize,
    out_lens: *mut usize,
    out_statuses: *mut i32,
) -> i32 {
    unsafe {
        get_batch_impl(
            engine,
            keys_ptr,
            dst_ptrs,
            dst_caps,
            count,
            out_lens,
            out_statuses,
        )
    }
}

/// # Safety
/// Follows the contract of `vericache_engine_get_batch`, using
/// `count` contiguous keys of four little-endian u64 words each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_get_batch_key256(
    engine: *mut VericacheVllmEngine,
    keys4_ptr: *const u64,
    dst_ptrs: *const *mut u8,
    dst_caps: *const usize,
    count: usize,
    out_lens: *mut usize,
    out_statuses: *mut i32,
) -> i32 {
    unsafe {
        get_batch_impl(
            engine,
            keys4_ptr.cast::<[u64; 4]>(),
            dst_ptrs,
            dst_caps,
            count,
            out_lens,
            out_statuses,
        )
    }
}

unsafe fn get_batch_impl<K: AbiKey>(
    engine: *mut VericacheVllmEngine,
    keys_ptr: *const K,
    dst_ptrs: *const *mut u8,
    dst_caps: *const usize,
    count: usize,
    out_lens: *mut usize,
    out_statuses: *mut i32,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let (Some(keys), Some(ptrs), Some(caps)) = (unsafe {
            (
                input_array(keys_ptr, count),
                input_array(dst_ptrs, count),
                input_array(dst_caps, count),
            )
        }) else {
            return INVALID;
        };
        if !valid_array(out_lens, count) || !valid_array(out_statuses, count) {
            return INVALID;
        }
        let destinations = ptrs
            .iter()
            .zip(caps)
            .map(|(&ptr, &cap)| {
                if cap == 0 {
                    Some(&mut [][..])
                } else if valid_array(ptr, cap) {
                    Some(unsafe { slice::from_raw_parts_mut(ptr, cap) })
                } else {
                    None
                }
            })
            .collect();
        for (i, result) in engine
            .get_batch(keys.iter().copied().map(AbiKey::into_key), destinations)
            .into_iter()
            .enumerate()
        {
            let (len, code) = match result {
                Ok(Some(len)) => (len, 0),
                Ok(None) => (0, 1),
                Err(Error::Capacity(len)) => (len, SHORT_BUFFER),
                Err(error) => (0, status(error)),
            };
            unsafe {
                out_lens.add(i).write(len);
                out_statuses.add(i).write(code);
            }
        }
        0
    })
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_poll_writeback(engine: *mut VericacheVllmEngine) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        engine
            .poll_writeback()
            .map_or_else(status, |count| i32::try_from(count).unwrap_or(i32::MAX))
    })
}

/// # Safety
/// A nonnull engine must stay live for this call. The percent is in 0..=100. Zero disables eager writeback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_set_writeback_policy(
    engine: *mut VericacheVllmEngine,
    high_watermark_percent: usize,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        engine
            .set_writeback_policy(high_watermark_percent)
            .map_or_else(status, |()| 0)
    })
}

/// # Safety
/// A nonnull engine must stay live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_set_nvme_policy(
    engine: *mut VericacheVllmEngine,
    depth: usize,
    probation: i32,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if !matches!(probation, 0 | 1) {
            return INVALID;
        }
        engine
            .set_nvme_policy(depth, probation != 0)
            .map_or_else(status, |()| 0)
    })
}

/// # Safety
/// A nonnull engine must stay live. Outputs must be disjoint writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_nvme_io_stats(
    engine: *const VericacheVllmEngine,
    read_bytes: *mut u64,
    write_bytes: *mut u64,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if read_bytes.is_null() || write_bytes.is_null() {
            return INVALID;
        }
        let stats = match engine.extended_stats() {
            Ok(stats) => stats,
            Err(error) => return status(error),
        };
        unsafe {
            read_bytes.write(stats.nvme_read_bytes);
            write_bytes.write(stats.nvme_write_bytes);
        }
        0
    })
}

/// # Safety
/// A nonnull engine must stay live. Outputs must be disjoint writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_extended_stats(
    engine: *const VericacheVllmEngine,
    dram_hits: *mut u64,
    nvme_hits: *mut u64,
    nvme_promotions: *mut u64,
    async_writebacks: *mut u64,
    clean_dram_evictions: *mut u64,
    active_leases: *mut u64,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        let outputs = [
            dram_hits,
            nvme_hits,
            nvme_promotions,
            async_writebacks,
            clean_dram_evictions,
            active_leases,
        ];
        if outputs.iter().any(|ptr| ptr.is_null()) {
            return INVALID;
        }
        let s = match engine.extended_stats() {
            Ok(s) => s,
            Err(error) => return status(error),
        };
        let values = [
            s.dram_hits,
            s.nvme_hits,
            s.nvme_promotions,
            s.async_writebacks,
            s.clean_dram_evictions,
            s.active_leases,
        ];
        for (out, value) in outputs.into_iter().zip(values) {
            unsafe { out.write(value) };
        }
        0
    })
}

/// # Safety
/// The engine must stay live. Output has capacity elements and is exclusive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_drain_evictions(
    engine: *mut VericacheVllmEngine,
    out: *mut u64,
    capacity: usize,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if !valid_array(out, capacity) || capacity > i32::MAX as usize {
            return INVALID;
        }
        engine
            .drain_evictions(capacity, |i, key| unsafe { out.add(i).write(key.0[0]) })
            .map_or_else(status, |count| count as i32)
    })
}

/// # Safety
/// The engine must stay live. The exclusive output has room for `max_count`
/// keys of four little-endian u64 words each, and may be uninitialized.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_drain_evictions_key256(
    engine: *mut VericacheVllmEngine,
    out_keys4: *mut u64,
    max_count: usize,
) -> isize {
    boundary(BACKEND as isize, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID as isize;
        };
        let out = out_keys4.cast::<[u64; 4]>();
        if !valid_array(out, max_count) {
            return INVALID as isize;
        }
        engine
            .drain_evictions(max_count, |i, key| unsafe {
                out.add(i).write(key.0.map(u64::to_le));
            })
            .map_or_else(|error| status(error) as isize, |count| count as isize)
    })
}

/// Acquire zero-copy read leases for `count` keys and issue asynchronous pinned
/// H2D `cudaMemcpyAsync` copies for both K and V slices directly into a vLLM
/// paged KV cache layer tensor without Python/GIL overhead.
///
/// # Safety
/// `engine` must be live; input/output arrays must have `count` valid elements;
/// `kv_layer_ptr` must address a live CUDA tensor large enough for all `block_ids`
/// with strides `k_stride_bytes` (dim 0) and `block_stride_bytes` (block dim).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_engine_load_kv_h2d_async(
    engine: *mut VericacheVllmEngine,
    keys4_ptr: *const u64,
    block_ids_ptr: *const u32,
    count: usize,
    kv_layer_ptr: usize,
    k_stride_bytes: usize,
    block_stride_bytes: usize,
    kv_slice_bytes: usize,
    cuda_stream_ptr: usize,
    out_lease_ids: *mut u64,
    out_statuses: *mut i32,
) -> i32 {
    boundary(BACKEND, || {
        let Some(engine) = (unsafe { engine.as_ref() }) else {
            return INVALID;
        };
        if kv_layer_ptr == 0 || kv_slice_bytes == 0 || block_stride_bytes < kv_slice_bytes {
            return INVALID;
        }
        let (Some(keys), Some(block_ids)) = (unsafe {
            (
                input_array(keys4_ptr.cast::<[u64; 4]>(), count),
                input_array(block_ids_ptr, count),
            )
        }) else {
            return INVALID;
        };
        if !valid_array(out_lease_ids, count) || !valid_array(out_statuses, count) {
            return INVALID;
        }
        let expected_len = match kv_slice_bytes.checked_mul(2) {
            Some(v) => v,
            None => return INVALID,
        };
        let results = engine.acquire_batch(keys.iter().copied().map(AbiKey::into_key));
        for (i, result) in results.into_iter().enumerate() {
            let (lease_id, code) = match result {
                Ok(Some(lease)) => {
                    let bid = block_ids[i] as usize;
                    let addrs = if lease.len == expected_len {
                        bid.checked_mul(block_stride_bytes)
                            .and_then(|off| kv_layer_ptr.checked_add(off))
                            .and_then(|dst_k| {
                                dst_k.checked_add(k_stride_bytes).and_then(|dst_v| {
                                    lease
                                        .address
                                        .checked_add(kv_slice_bytes)
                                        .map(|src_v| (dst_k, dst_v, src_v))
                                })
                            })
                    } else {
                        None
                    };
                    if let Some((dst_k, dst_v, src_v)) = addrs {
                        let rc = crate::cuda_copy::memcpy_h2d_async(
                            dst_k,
                            lease.address,
                            kv_slice_bytes,
                            cuda_stream_ptr,
                        )
                        .and_then(|()| {
                            crate::cuda_copy::memcpy_h2d_async(
                                dst_v,
                                src_v,
                                kv_slice_bytes,
                                cuda_stream_ptr,
                            )
                        });
                        if rc.is_ok() {
                            (lease.id, 0)
                        } else {
                            let _ = engine.release(lease.id);
                            (0, BACKEND)
                        }
                    } else {
                        let _ = engine.release(lease.id);
                        (0, INVALID)
                    }
                }
                Ok(None) => (0, 1),
                Err(error) => (0, status(error)),
            };
            unsafe {
                out_lease_ids.add(i).write(lease_id);
                out_statuses.add(i).write(code);
            }
        }
        0
    })
}

/// Issue asynchronous pinned D2H `cudaMemcpyAsync` copies for `count` KV blocks
/// (both K and V slices) from a vLLM paged KV cache layer tensor directly into
/// a contiguous pinned host staging buffer without GPU gather kernels or GIL.
///
/// # Safety
/// `dst_host_base` must address at least `count * 2 * kv_slice_bytes` writable
/// host bytes; `kv_layer_ptr` must address a live CUDA tensor large enough for
/// all `block_ids`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_cuda_gather_kv_d2h_async(
    dst_host_base: *mut u8,
    kv_layer_ptr: usize,
    k_stride_bytes: usize,
    block_stride_bytes: usize,
    kv_slice_bytes: usize,
    block_ids_ptr: *const u32,
    count: usize,
    cuda_stream_ptr: usize,
) -> i32 {
    boundary(BACKEND, || {
        if count == 0 {
            return 0;
        }
        if dst_host_base.is_null()
            || kv_layer_ptr == 0
            || kv_slice_bytes == 0
            || block_stride_bytes < kv_slice_bytes
        {
            return INVALID;
        }
        let Some(block_ids) = (unsafe { input_array(block_ids_ptr, count) }) else {
            return INVALID;
        };
        let Some(frame_bytes) = kv_slice_bytes.checked_mul(2) else {
            return INVALID;
        };
        let dst_base = dst_host_base as usize;
        for (i, &bid_u32) in block_ids.iter().enumerate() {
            let bid = bid_u32 as usize;
            let Some(block_off) = bid.checked_mul(block_stride_bytes) else {
                return INVALID;
            };
            let Some(src_k) = kv_layer_ptr.checked_add(block_off) else {
                return INVALID;
            };
            let Some(src_v) = src_k.checked_add(k_stride_bytes) else {
                return INVALID;
            };
            let Some(dst_k) = i.checked_mul(frame_bytes).and_then(|o| dst_base.checked_add(o))
            else {
                return INVALID;
            };
            let Some(dst_v) = dst_k.checked_add(kv_slice_bytes) else {
                return INVALID;
            };
            if crate::cuda_copy::memcpy_d2h_async(dst_k, src_k, kv_slice_bytes, cuda_stream_ptr)
                .is_err()
                || crate::cuda_copy::memcpy_d2h_async(
                    dst_v,
                    src_v,
                    kv_slice_bytes,
                    cuda_stream_ptr,
                )
                .is_err()
            {
                return BACKEND;
            }
        }
        0
    })
}
