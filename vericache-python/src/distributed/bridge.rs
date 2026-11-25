//! C ABI exports for `DistributedVllmEngine` (Scheme A multi-shard Type-2B MW RDMA).

use super::engine::{DistEngineConfig, DistPeerSpec, DistStats, DistributedVllmEngine};
use crate::{cuda_copy, set_last_error};
use core::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::slice;
use vericache_core::spec::ObjectKey;

#[repr(C)]
pub struct VeriCacheDistPeerSpecC {
    pub peer_node_id: u64,
    pub tcp_host: *const c_char,
    pub base_port: u16,
    pub is_listener: u8,
    pub psn_hint: u32,
}

#[repr(C)]
pub struct VeriCacheDistConfigC {
    pub node_id: u64,
    pub num_shards: usize,
    pub host_frames_per_shard: u32,
    pub nvme_frames_per_shard: u32,
    pub frame_size: usize,
    pub nvme_path: *const c_char,
    pub prefer_cuda_pinned: u8,
    pub rdma_device: *const c_char,
    pub rdma_gid_index: u8,
    pub rdma_windows_per_peer: usize,
    pub peer_fetch_enabled: u8,
    pub peers: *const VeriCacheDistPeerSpecC,
    pub num_peers: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VeriCacheDistStatsC {
    pub puts: u64,
    pub gets: u64,
    pub hits: u64,
    pub misses: u64,
    pub dram_hits: u64,
    pub nvme_hits: u64,
    pub remote_hits: u64,
    pub dram_evictions: u64,
    pub nvme_evictions: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub nvme_read_bytes: u64,
    pub nvme_write_bytes: u64,
    pub rdma_fetch_bytes: u64,
    pub rdma_fetches_ok: u64,
    pub rdma_fetches_miss: u64,
    pub rdma_fetches_err: u64,
    pub rdma_grants_served: u64,
    pub mw_revocations_cooperative: u64,
    pub mw_revocations_unilateral: u64,
    pub mw_revocation_latency_us_sum: u64,
    pub mw_revocation_latency_us_max: u64,
    pub peer_disconnects: u64,
    pub peer_reconnects: u64,
    pub peers_connected: u64,
    pub active_leases: u64,
    pub dram_resident_keys: u64,
}

impl From<DistStats> for VeriCacheDistStatsC {
    fn from(s: DistStats) -> Self {
        Self {
            puts: s.puts,
            gets: s.gets,
            hits: s.hits,
            misses: s.misses,
            dram_hits: s.dram_hits,
            nvme_hits: s.nvme_hits,
            remote_hits: s.remote_hits,
            dram_evictions: s.dram_evictions,
            nvme_evictions: s.nvme_evictions,
            bytes_written: s.bytes_written,
            bytes_read: s.bytes_read,
            nvme_read_bytes: s.nvme_read_bytes,
            nvme_write_bytes: s.nvme_write_bytes,
            rdma_fetch_bytes: s.rdma_fetch_bytes,
            rdma_fetches_ok: s.rdma_fetches_ok,
            rdma_fetches_miss: s.rdma_fetches_miss,
            rdma_fetches_err: s.rdma_fetches_err,
            rdma_grants_served: s.rdma_grants_served,
            mw_revocations_cooperative: s.mw_revocations_cooperative,
            mw_revocations_unilateral: s.mw_revocations_unilateral,
            mw_revocation_latency_us_sum: s.mw_revocation_latency_us_sum,
            mw_revocation_latency_us_max: s.mw_revocation_latency_us_max,
            peer_disconnects: s.peer_disconnects,
            peer_reconnects: s.peer_reconnects,
            peers_connected: s.peers_connected,
            active_leases: s.active_leases,
            dram_resident_keys: s.dram_resident_keys,
        }
    }
}

fn boundary<T: Copy>(fallback: T, call: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(call)).unwrap_or(fallback)
}

unsafe fn opt_cstr(ptr: *const c_char) -> Result<Option<String>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| format!("invalid UTF-8: {e}"))?;
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s.to_owned()))
    }
}

unsafe fn read_keys256(keys4_ptr: *const u64, count: usize) -> Option<Vec<ObjectKey>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if keys4_ptr.is_null() {
        return None;
    }
    let raw = unsafe { slice::from_raw_parts(keys4_ptr, count * 4) };
    let mut keys = Vec::with_capacity(count);
    for i in 0..count {
        let base = i * 4;
        keys.push(ObjectKey([
            raw[base],
            raw[base + 1],
            raw[base + 2],
            raw[base + 3],
        ]));
    }
    Some(keys)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_create(
    cfg: *const VeriCacheDistConfigC,
) -> *mut DistributedVllmEngine {
    set_last_error("");
    boundary(ptr::null_mut(), || {
        if cfg.is_null() {
            set_last_error("null dist config");
            return ptr::null_mut();
        }
        let c = unsafe { &*cfg };
        let nvme_path = match unsafe { opt_cstr(c.nvme_path) } {
            Ok(p) => p,
            Err(e) => {
                set_last_error(e);
                return ptr::null_mut();
            }
        };
        let rdma_device = match unsafe { opt_cstr(c.rdma_device) } {
            Ok(d) => d,
            Err(e) => {
                set_last_error(e);
                return ptr::null_mut();
            }
        };
        let mut peers = Vec::with_capacity(c.num_peers);
        if c.num_peers > 0 {
            if c.peers.is_null() {
                set_last_error("num_peers > 0 but peers pointer is null");
                return ptr::null_mut();
            }
            let raw = unsafe { slice::from_raw_parts(c.peers, c.num_peers) };
            for p in raw {
                let host = match unsafe { opt_cstr(p.tcp_host) } {
                    Ok(Some(h)) => h,
                    Ok(None) => {
                        set_last_error("peer tcp_host is empty");
                        return ptr::null_mut();
                    }
                    Err(e) => {
                        set_last_error(e);
                        return ptr::null_mut();
                    }
                };
                peers.push(DistPeerSpec {
                    peer_node_id: p.peer_node_id,
                    tcp_host: host,
                    base_port: p.base_port,
                    is_listener: p.is_listener != 0,
                    psn_hint: if p.psn_hint == 0 { 0x1000 } else { p.psn_hint },
                });
            }
        }
        let dcfg = DistEngineConfig {
            node_id: c.node_id,
            num_shards: c.num_shards,
            host_frames_per_shard: c.host_frames_per_shard,
            nvme_frames_per_shard: c.nvme_frames_per_shard,
            frame_size: c.frame_size,
            nvme_path,
            prefer_cuda_pinned: c.prefer_cuda_pinned != 0,
            rdma_device,
            rdma_gid_index: c.rdma_gid_index,
            rdma_windows_per_peer: if c.rdma_windows_per_peer == 0 {
                16
            } else {
                c.rdma_windows_per_peer
            },
            peer_fetch_enabled: c.peer_fetch_enabled != 0,
            peers,
        };
        match DistributedVllmEngine::new(dcfg) {
            Ok(engine) => Box::into_raw(Box::new(engine)),
            Err(e) => {
                set_last_error(e);
                ptr::null_mut()
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_destroy(engine: *mut DistributedVllmEngine) {
    if !engine.is_null() {
        boundary((), || drop(unsafe { Box::from_raw(engine) }));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_frame_size(
    engine: *const DistributedVllmEngine,
) -> usize {
    unsafe { engine.as_ref() }.map_or(0, DistributedVllmEngine::frame_size)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_num_shards(
    engine: *const DistributedVllmEngine,
) -> usize {
    unsafe { engine.as_ref() }.map_or(0, DistributedVllmEngine::num_shards)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_is_slab_cuda_pinned(
    engine: *const DistributedVllmEngine,
) -> i32 {
    unsafe { engine.as_ref() }.map_or(0, |e| i32::from(e.is_cuda_pinned()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_peers_connected(
    engine: *const DistributedVllmEngine,
) -> u64 {
    unsafe { engine.as_ref() }.map_or(0, DistributedVllmEngine::peers_connected)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_set_peer_fetch_enabled(
    engine: *mut DistributedVllmEngine,
    enabled: u8,
) -> i32 {
    boundary(-1, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        e.set_peer_fetch_enabled(enabled != 0);
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_put_batch_key256(
    engine: *mut DistributedVllmEngine,
    keys4_ptr: *const u64,
    src_ptrs: *const *const u8,
    lens: *const usize,
    count: usize,
    out_statuses: *mut i32,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if count == 0 {
            return 0;
        }
        let Some(keys) = (unsafe { read_keys256(keys4_ptr, count) }) else {
            return -1;
        };
        if src_ptrs.is_null() || lens.is_null() || out_statuses.is_null() {
            return -1;
        }
        let ptrs = unsafe { slice::from_raw_parts(src_ptrs, count) };
        let lengths = unsafe { slice::from_raw_parts(lens, count) };
        let items: Vec<(ObjectKey, usize, usize)> = keys
            .into_iter()
            .zip(ptrs.iter().zip(lengths.iter()))
            .map(|(k, (&p, &l))| (k, p as usize, l))
            .collect();
        let statuses = e.put_batch(&items);
        let out = unsafe { slice::from_raw_parts_mut(out_statuses, count) };
        out.copy_from_slice(&statuses);
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_contains_batch_key256(
    engine: *const DistributedVllmEngine,
    keys4_ptr: *const u64,
    count: usize,
    dram_only: u8,
    out_present: *mut u8,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if count == 0 {
            return 0;
        }
        let Some(keys) = (unsafe { read_keys256(keys4_ptr, count) }) else {
            return -1;
        };
        if out_present.is_null() {
            return -1;
        }
        let flags = e.contains_batch(&keys, dram_only != 0);
        let out = unsafe { slice::from_raw_parts_mut(out_present, count) };
        out.copy_from_slice(&flags);
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_acquire_read_lease_batch_key256(
    engine: *mut DistributedVllmEngine,
    keys4_ptr: *const u64,
    allow_remote_ptr: *const u8,
    count: usize,
    out_lease_ids: *mut u64,
    out_addrs: *mut usize,
    out_lens: *mut usize,
    out_locs: *mut i32,
    out_statuses: *mut i32,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if count == 0 {
            return 0;
        }
        let Some(keys) = (unsafe { read_keys256(keys4_ptr, count) }) else {
            return -1;
        };
        if out_lease_ids.is_null()
            || out_addrs.is_null()
            || out_lens.is_null()
            || out_locs.is_null()
            || out_statuses.is_null()
        {
            return -1;
        }
        let remote_flags = if allow_remote_ptr.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(allow_remote_ptr, count) })
        };
        let items: Vec<(ObjectKey, bool)> = keys
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, remote_flags.is_none_or(|f| f[i] != 0)))
            .collect();
        let results = e.acquire_lease_batch(&items);
        let ids = unsafe { slice::from_raw_parts_mut(out_lease_ids, count) };
        let addrs = unsafe { slice::from_raw_parts_mut(out_addrs, count) };
        let lens = unsafe { slice::from_raw_parts_mut(out_lens, count) };
        let locs = unsafe { slice::from_raw_parts_mut(out_locs, count) };
        let statuses = unsafe { slice::from_raw_parts_mut(out_statuses, count) };
        for (i, r) in results.into_iter().enumerate() {
            ids[i] = r.lease_id;
            addrs[i] = r.address;
            lens[i] = r.len;
            locs[i] = r.location;
            statuses[i] = r.status;
        }
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_load_kv_h2d_async(
    engine: *mut DistributedVllmEngine,
    keys4_ptr: *const u64,
    block_ids_ptr: *const usize,
    allow_remote_ptr: *const u8,
    count: usize,
    kv_base_ptr: usize,
    k_v_stride: usize,
    block_stride: usize,
    kv_slice_bytes: usize,
    cuda_stream: usize,
    out_lease_ids: *mut u64,
    out_locs: *mut i32,
    out_statuses: *mut i32,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if count == 0 {
            return 0;
        }
        let Some(keys) = (unsafe { read_keys256(keys4_ptr, count) }) else {
            return -1;
        };
        if block_ids_ptr.is_null()
            || out_lease_ids.is_null()
            || out_locs.is_null()
            || out_statuses.is_null()
            || kv_base_ptr == 0
            || kv_slice_bytes == 0
        {
            return -1;
        }
        let block_ids = unsafe { slice::from_raw_parts(block_ids_ptr, count) };
        let remote_flags = if allow_remote_ptr.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(allow_remote_ptr, count) })
        };
        let items: Vec<(ObjectKey, bool)> = keys
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, remote_flags.is_none_or(|f| f[i] != 0)))
            .collect();
        let results = e.acquire_lease_batch(&items);
        let ids = unsafe { slice::from_raw_parts_mut(out_lease_ids, count) };
        let locs = unsafe { slice::from_raw_parts_mut(out_locs, count) };
        let statuses = unsafe { slice::from_raw_parts_mut(out_statuses, count) };
        let expected_len = kv_slice_bytes.saturating_mul(2);

        for (i, r) in results.into_iter().enumerate() {
            ids[i] = r.lease_id;
            locs[i] = r.location;
            if r.status != 0 {
                statuses[i] = r.status;
                continue;
            }
            if r.len != expected_len {
                statuses[i] = -3;
                continue;
            }
            let bid = block_ids[i];
            let k_dst = kv_base_ptr + bid * block_stride;
            let v_dst = k_dst + k_v_stride;
            if let Err(code) =
                cuda_copy::memcpy_h2d_async(k_dst, r.address, kv_slice_bytes, cuda_stream)
            {
                set_last_error(format!("dist H2D K cudaMemcpyAsync failed: {code}"));
                statuses[i] = -2;
                continue;
            }
            if let Err(code) = cuda_copy::memcpy_h2d_async(
                v_dst,
                r.address + kv_slice_bytes,
                kv_slice_bytes,
                cuda_stream,
            ) {
                set_last_error(format!("dist H2D V cudaMemcpyAsync failed: {code}"));
                statuses[i] = -2;
                continue;
            }
            statuses[i] = 0;
        }
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_release_read_lease_batch(
    engine: *mut DistributedVllmEngine,
    lease_ids_ptr: *const u64,
    count: usize,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if count == 0 {
            return 0;
        }
        if lease_ids_ptr.is_null() {
            return -1;
        }
        let ids = unsafe { slice::from_raw_parts(lease_ids_ptr, count) };
        e.release_leases(ids);
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_get_batch_key256(
    engine: *mut DistributedVllmEngine,
    keys4_ptr: *const u64,
    allow_remote_ptr: *const u8,
    dst_ptrs: *const *mut u8,
    dst_caps: *const usize,
    count: usize,
    out_lens: *mut usize,
    out_locs: *mut i32,
    out_statuses: *mut i32,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if count == 0 {
            return 0;
        }
        let Some(keys) = (unsafe { read_keys256(keys4_ptr, count) }) else {
            return -1;
        };
        if dst_ptrs.is_null()
            || dst_caps.is_null()
            || out_lens.is_null()
            || out_locs.is_null()
            || out_statuses.is_null()
        {
            return -1;
        }
        let ptrs = unsafe { slice::from_raw_parts(dst_ptrs, count) };
        let caps = unsafe { slice::from_raw_parts(dst_caps, count) };
        let remote_flags = if allow_remote_ptr.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(allow_remote_ptr, count) })
        };
        let items: Vec<(ObjectKey, usize, usize, bool)> = keys
            .into_iter()
            .enumerate()
            .map(|(i, k)| {
                (
                    k,
                    ptrs[i] as usize,
                    caps[i],
                    remote_flags.is_none_or(|f| f[i] != 0),
                )
            })
            .collect();
        let results = e.get_batch(&items);
        let ol = unsafe { slice::from_raw_parts_mut(out_lens, count) };
        let oloc = unsafe { slice::from_raw_parts_mut(out_locs, count) };
        let ost = unsafe { slice::from_raw_parts_mut(out_statuses, count) };
        for (i, (len, st, loc)) in results.into_iter().enumerate() {
            ol[i] = len;
            ost[i] = st;
            oloc[i] = loc;
        }
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_drain_evictions_key256(
    engine: *mut DistributedVllmEngine,
    out_keys4: *mut u64,
    cap: usize,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if cap == 0 {
            return 0;
        }
        if out_keys4.is_null() {
            return -1;
        }
        let keys = e.drain_evictions(cap);
        let out = unsafe { slice::from_raw_parts_mut(out_keys4, keys.len() * 4) };
        for (i, k) in keys.iter().enumerate() {
            let base = i * 4;
            out[base] = k.0[0];
            out[base + 1] = k.0[1];
            out[base + 2] = k.0[2];
            out[base + 3] = k.0[3];
        }
        keys.len() as i32
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_set_dram_budget_frames(
    engine: *mut DistributedVllmEngine,
    max_frames_per_shard: usize,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        e.set_dram_budget_frames(max_frames_per_shard) as i32
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_set_coop_revoke_spin_budget(
    engine: *mut DistributedVllmEngine,
    spins: u32,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        e.set_coop_revoke_spin_budget(spins);
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_evict_dram_keys256(
    engine: *mut DistributedVllmEngine,
    keys4_ptr: *const u64,
    count: usize,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        let Some(keys) = (unsafe { read_keys256(keys4_ptr, count) }) else {
            return -1;
        };
        e.evict_local_dram_keys(&keys) as i32
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_poll(engine: *mut DistributedVllmEngine) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        e.poll();
        0
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vericache_dist_engine_stats(
    engine: *const DistributedVllmEngine,
    out_stats: *mut VeriCacheDistStatsC,
) -> i32 {
    boundary(-2, || {
        let Some(e) = (unsafe { engine.as_ref() }) else {
            return -1;
        };
        if out_stats.is_null() {
            return -1;
        }
        let s: VeriCacheDistStatsC = e.stats().into();
        unsafe { *out_stats = s };
        0
    })
}
