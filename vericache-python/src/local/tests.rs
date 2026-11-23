use super::engine::{Error, object_key};
use super::*;
use std::fs;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

#[test]
fn reports_exact_lengths_and_supports_replacement_and_empty_payload() {
    let engine = VericacheVllmEngine::new(1, 2, 0, 32, None).unwrap();
    engine.put(object_key(7), b"hello").unwrap();
    assert_eq!(
        engine.get(object_key(7), &mut [0; 4]),
        Err(Error::Capacity(5))
    );
    let mut small = [0; 5];
    assert_eq!(engine.get(object_key(7), &mut small).unwrap(), Some(5));
    assert_eq!(&small, b"hello");

    engine.put(object_key(7), b"new").unwrap();
    let mut frame = [42; 32];
    assert_eq!(engine.get(object_key(7), &mut frame).unwrap(), Some(3));
    assert_eq!(&frame[..3], b"new");
    engine.put(object_key(8), &[]).unwrap();
    assert_eq!(engine.get(object_key(8), &mut []).unwrap(), Some(0));
    assert_eq!(engine.put(object_key(9), &[0; 33]), Err(Error::Invalid));
    assert!(engine.evict(object_key(7)).unwrap());
    assert!(!engine.evict(object_key(7)).unwrap());
    assert!(!engine.contains(object_key(7)).unwrap());
    assert_eq!(engine.get(object_key(7), &mut frame).unwrap(), None);

    let stats = engine.stats().unwrap();
    assert_eq!(
        (stats.puts, stats.gets, stats.hits, stats.misses),
        (3, 5, 3, 1)
    );
    assert_eq!(
        (stats.bytes_written, stats.bytes_read, stats.evictions),
        (8, 8, 1)
    );
}

#[test]
fn sustained_writes_reclaim_host_frames() {
    let engine = VericacheVllmEngine::new(1, 3, 0, 8, None).unwrap();
    for key in 0u64..300 {
        engine.put(object_key(key), &key.to_le_bytes()).unwrap();
    }
    for key in 0u64..300 {
        let mut dst = [0; 8];
        let result = engine.get(object_key(key), &mut dst).unwrap();
        if key < 297 {
            assert_eq!(result, None);
        } else {
            assert_eq!(result, Some(8));
            assert_eq!(dst, key.to_le_bytes());
        }
    }
    assert_eq!(engine.stats().unwrap().evictions, 297);
}

struct TestFile(PathBuf);

impl TestFile {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
        fs::create_dir_all(&root).unwrap();
        Self(root.join(format!(
            "nvme-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        )))
    }

    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        for index in 0..2 {
            let _ = fs::remove_file(format!("{}.shard-{index}", self.path()));
        }
    }
}

#[test]
fn nvme_spill_and_reads_preserve_host_capacity() {
    let file = TestFile::new();
    let engine = VericacheVllmEngine::new(1, 1, 2, 37, Some(file.path())).unwrap();
    assert_eq!(fs::metadata(&file.0).unwrap().len(), 8192);
    assert!(VericacheVllmEngine::new(1, 1, 2, 37, Some(file.path())).is_err());
    for key in 1..=3 {
        engine.put(object_key(key), &[key as u8; 37]).unwrap();
    }
    for key in [1, 3, 2, 1, 3] {
        let mut dst = [0; 37];
        assert_eq!(engine.get(object_key(key), &mut dst).unwrap(), Some(37));
        assert_eq!(dst, [key as u8; 37]);
    }
    assert_eq!(engine.stats().unwrap().evictions, 0);
    engine.put(object_key(4), b"new").unwrap();
    assert!(!engine.contains(object_key(2)).unwrap());
    assert!(engine.evict(object_key(1)).unwrap());
    assert!(engine.contains(object_key(3)).unwrap());
    drop(engine);
    assert!(VericacheVllmEngine::new(1, 1, 2, 37, Some(file.path())).is_ok());
}

#[test]
fn nvme_shards_use_independent_backing_files() {
    let file = TestFile::new();
    let engine = VericacheVllmEngine::new(2, 1, 2, 17, Some(file.path())).unwrap();
    for index in 0..2 {
        assert_eq!(
            fs::metadata(format!("{}.shard-{index}", file.path()))
                .unwrap()
                .len(),
            8192,
        );
    }
    for key in 0u64..200 {
        engine.put(object_key(key), &key.to_be_bytes()).unwrap();
        let mut dst = [0; 8];
        assert_eq!(engine.get(object_key(key), &mut dst).unwrap(), Some(8));
        assert_eq!(dst, key.to_be_bytes());
    }
    assert_eq!(engine.stats().unwrap().evictions, 194);
}

#[test]
fn concurrent_reads_and_writes_share_an_engine() {
    let engine = VericacheVllmEngine::new(8, 256, 0, 32, None).unwrap();
    thread::scope(|scope| {
        for worker in 0u64..8 {
            let engine = &engine;
            scope.spawn(move || {
                for index in 0..200 {
                    let key = worker * 200 + index;
                    let data = key.to_le_bytes().repeat(4);
                    engine.put(object_key(key), &data).unwrap();
                    let mut dst = [0; 32];
                    assert_eq!(engine.get(object_key(key), &mut dst).unwrap(), Some(32));
                    assert_eq!(dst.as_slice(), data);
                }
            });
        }
    });
    let stats = engine.stats().unwrap();
    assert_eq!((stats.puts, stats.hits, stats.evictions), (1600, 1600, 0));
}

#[test]
fn abi_reports_nulls_misses_and_required_bytes() {
    unsafe {
        assert!(vericache_engine_create(0, 1, 0, 32, ptr::null(), 0).is_null());
        assert!(vericache_engine_create(1, 1, 1, 32, ptr::null(), 1).is_null());
        assert_eq!(vericache_engine_frame_size(ptr::null()), 0);
        assert_eq!(vericache_engine_num_shards(ptr::null()), 0);
        assert!(vericache_engine_contains(ptr::null_mut(), 1) < 0);
        assert!(vericache_staging_ptr(ptr::null_mut()).is_null());
        assert!(vericache_staging_alloc(0, 0).is_null());
        let engine = vericache_engine_create(1, 1, 0, 32, ptr::null(), 0);
        assert!(!engine.is_null());
        let mut len = 15;
        assert_eq!(
            vericache_engine_get(engine, 1, ptr::null_mut(), 0, &mut len),
            1
        );
        assert_eq!(len, 0);
        assert_eq!(vericache_engine_put(engine, 1, b"data".as_ptr(), 4), 0);
        assert_eq!(
            vericache_engine_get(engine, 1, ptr::null_mut(), 0, &mut len),
            -3
        );
        assert_eq!(len, 4);
        assert_eq!(vericache_engine_contains(engine, 1), 1);
        assert_eq!(vericache_engine_evict(engine, 1), 0);
        assert_eq!(vericache_engine_evict(engine, 1), 1);
        vericache_engine_destroy(engine);
        vericache_engine_destroy(ptr::null_mut());
        vericache_staging_free(ptr::null_mut());
    }
}

#[test]
fn writable_staging_uses_the_same_rust_and_c_storage() {
    unsafe {
        let engine = vericache_engine_create(1, 2, 0, 4096, ptr::null(), 0);
        let stage = vericache_staging_alloc(4096, 0);
        assert!(!stage.is_null());
        let address = vericache_staging_ptr(stage);
        assert!(!address.is_null());
        assert_eq!(address.addr() % 4096, 0);
        assert_eq!(vericache_staging_len(stage), 4096);
        assert_eq!(vericache_staging_is_cuda_pinned(stage), 0);
        address.write_bytes(42, 4096);
        assert_eq!(vericache_engine_put(engine, 1, address, 4096), 0);
        address.write_bytes(0, 4096);
        let mut len = 0;
        assert_eq!(vericache_engine_get(engine, 1, address, 4096, &mut len), 0);
        assert_eq!(len, 4096);
        assert_eq!(address.read(), 42);
        assert_eq!(address.add(4095).read(), 42);
        vericache_staging_free(stage);
        vericache_engine_destroy(engine);
    }

    let stage = vericache_staging_alloc(17, 1);
    assert!(
        !stage.is_null(),
        "CUDA pinning or aligned host fallback is available"
    );
    unsafe {
        assert_eq!(vericache_staging_len(stage), 17);
        let address = vericache_staging_ptr(stage);
        assert_eq!(address.read(), 0);
        if vericache_staging_is_cuda_pinned(stage) == 0 {
            assert_eq!(address.addr() % 4096, 0);
        }
        vericache_staging_free(stage);
    }
}

#[test]
fn leases_alias_slab_and_prevent_eviction_or_replacement() {
    let engine = VericacheVllmEngine::new(1, 2, 0, 4096, None).unwrap();
    engine.put(object_key(1), b"first").unwrap();
    engine.put(object_key(2), b"second").unwrap();
    let lease = engine.acquire(object_key(1)).unwrap().unwrap();
    assert!(engine.slab_range(0).contains(&lease.address));
    assert_eq!(lease.address % 4096, 0);
    let duplicate = engine.acquire(object_key(1)).unwrap().unwrap();
    assert_eq!(lease.address, duplicate.address);
    assert_ne!(lease.id, duplicate.id);
    assert_eq!(engine.evict(object_key(1)), Err(Error::Busy));
    assert_eq!(engine.put(object_key(1), b"replacement"), Err(Error::Busy));
    engine.release(lease.id).unwrap();
    assert_eq!(engine.evict(object_key(1)), Err(Error::Busy));
    engine.get(object_key(2), &mut [0; 6]).unwrap();
    engine.put(object_key(3), b"third").unwrap();
    assert!(!engine.contains(object_key(2)).unwrap());
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            ptr::with_exposed_provenance(duplicate.address),
            duplicate.len,
        )
    };
    assert_eq!(bytes, b"first");
    engine.release_batch(&[duplicate.id]).unwrap();
    assert_eq!(engine.extended_stats().unwrap().active_leases, 0);
    assert!(engine.evict(object_key(1)).unwrap());
    assert_eq!(engine.release(duplicate.id), Err(Error::Invalid));
    // The destructor must also return any leases abandoned by C callers.
    assert!(engine.acquire(object_key(3)).unwrap().is_some());
}

#[test]
fn reads_and_leases_refresh_lru() {
    let engine = VericacheVllmEngine::new(1, 2, 0, 8, None).unwrap();
    engine.put(object_key(1), b"one").unwrap();
    engine.put(object_key(2), b"two").unwrap();
    assert_eq!(engine.get(object_key(1), &mut [0; 3]), Ok(Some(3)));
    engine.put(object_key(3), b"three").unwrap();
    assert!(engine.contains(object_key(1)).unwrap());
    assert!(!engine.contains(object_key(2)).unwrap());

    let lease = engine.acquire(object_key(1)).unwrap().unwrap();
    engine.release(lease.id).unwrap();
    engine.put(object_key(4), b"four").unwrap();
    assert!(engine.contains(object_key(1)).unwrap());
    assert!(!engine.contains(object_key(3)).unwrap());
}

#[test]
fn completed_writeback_enables_clean_eviction_and_promotion() {
    let file = TestFile::new();
    let engine = VericacheVllmEngine::new(1, 4, 8, 4096, Some(file.path())).unwrap();
    engine.set_writeback_policy(75).unwrap();
    engine.put(object_key(1), b"one").unwrap();
    engine.put(object_key(2), b"two").unwrap();
    assert_eq!(engine.poll_writeback(), Ok(0));
    engine.put(object_key(3), b"three").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        engine.poll_writeback().unwrap();
        if engine.extended_stats().unwrap().async_writebacks >= 3 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "writeback did not retire"
        );
        thread::yield_now();
    }

    engine.put(object_key(4), b"four").unwrap();
    engine.put(object_key(5), b"five").unwrap();
    assert!(engine.extended_stats().unwrap().clean_dram_evictions > 0);
    let mut dst = [0; 3];
    assert_eq!(engine.get(object_key(1), &mut dst).unwrap(), Some(3));
    assert_eq!(&dst, b"one");
    let promoted = engine.extended_stats().unwrap();
    assert_eq!(promoted.nvme_promotions, 1);
    assert_eq!(promoted.nvme_hits, 1);
    let lease = engine.acquire(object_key(1)).unwrap().unwrap();
    engine.release(lease.id).unwrap();
    let reread = engine.extended_stats().unwrap();
    assert_eq!(reread.dram_hits, promoted.dram_hits + 1);
    assert_eq!(reread.nvme_promotions, promoted.nvme_promotions);
}

#[test]
fn batch_abi_preserves_positions_and_exact_lengths() {
    unsafe {
        let engine = vericache_engine_create(3, 8, 0, 32, ptr::null(), 0);
        assert!(!engine.is_null());
        let keys = [1, 7, 53, 67];
        let sources = [b"abc".as_ptr(), ptr::null(), ptr::null(), b"wxyz".as_ptr()];
        let lens = [3, 0, 5, 4];
        let mut statuses = [99; 4];
        assert_eq!(
            vericache_engine_put_batch(
                engine,
                keys.as_ptr(),
                sources.as_ptr(),
                lens.as_ptr(),
                keys.len(),
                statuses.as_mut_ptr()
            ),
            0
        );
        assert_eq!(statuses, [0, 0, -1, 0]);
        let mut present = [2; 4];
        assert_eq!(
            vericache_engine_contains_batch(
                engine,
                keys.as_ptr(),
                keys.len(),
                present.as_mut_ptr()
            ),
            0
        );
        assert_eq!(present, [1, 1, 0, 1]);
        let mut abc = [0u8; 3];
        let mut short = [0u8; 2];
        let destinations = [
            abc.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            short.as_mut_ptr(),
        ];
        let mut read_lens = [99; 4];
        assert_eq!(
            vericache_engine_get_batch(
                engine,
                keys.as_ptr(),
                destinations.as_ptr(),
                [3, 0, 0, 2].as_ptr(),
                keys.len(),
                read_lens.as_mut_ptr(),
                statuses.as_mut_ptr()
            ),
            0
        );
        assert_eq!(statuses, [0, 0, 1, -3]);
        assert_eq!(read_lens, [3, 0, 0, 4]);
        assert_eq!(&abc, b"abc");

        let mut ptrs = [ptr::null(); 4];
        let mut ids = [0; 4];
        assert_eq!(
            vericache_engine_acquire_read_lease_batch(
                engine,
                keys.as_ptr(),
                keys.len(),
                ptrs.as_mut_ptr(),
                read_lens.as_mut_ptr(),
                ids.as_mut_ptr(),
                statuses.as_mut_ptr()
            ),
            0
        );
        assert_eq!(statuses, [0, 0, 1, 0]);
        assert_eq!(read_lens, [3, 0, 0, 4]);
        assert_eq!(std::slice::from_raw_parts(ptrs[0], 3), b"abc");
        assert_eq!(std::slice::from_raw_parts(ptrs[3], 4), b"wxyz");
        assert_eq!(ids[2], 0);
        let live = [ids[0], ids[1], ids[3]];
        assert_eq!(
            vericache_engine_release_read_lease_batch(engine, live.as_ptr(), live.len()),
            0
        );
        assert_eq!(vericache_engine_release_read_lease(engine, ids[0]), -1);
        assert_eq!(
            vericache_engine_acquire_read_lease_batch(
                engine,
                ptr::null(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut()
            ),
            0
        );
        vericache_engine_destroy(engine);
    }
}

#[test]
fn concurrent_leases_stay_immutable_during_reclamation() {
    let engine = VericacheVllmEngine::new(4, 16, 0, 32, None).unwrap();
    for key in 0..8 {
        engine.put(object_key(key), &[key as u8; 32]).unwrap();
    }
    let permanently_pinned = engine.acquire_batch((0..8).map(|key| object_key(key as u64)));
    let permanently_pinned: Vec<_> = permanently_pinned
        .into_iter()
        .map(|lease| lease.unwrap().unwrap().id)
        .collect();
    thread::scope(|scope| {
        for worker in 0..8 {
            let engine = &engine;
            scope.spawn(move || {
                for i in 0..300 {
                    let lease = engine.acquire(object_key(worker)).unwrap().unwrap();
                    let key = 8 + worker * 300 + i;
                    engine.put(object_key(key), &[key as u8; 32]).unwrap();
                    assert_eq!(engine.evict(object_key(worker)), Err(Error::Busy));
                    let leased: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            ptr::with_exposed_provenance(lease.address),
                            lease.len,
                        )
                    };
                    assert_eq!(leased, &[worker as u8; 32]);
                    engine.release(lease.id).unwrap();
                    engine.evict(object_key(key)).unwrap();
                }
            });
        }
    });
    engine.release_batch(&permanently_pinned).unwrap();
    assert_eq!(engine.extended_stats().unwrap().active_leases, 0);
}

#[test]
fn lease_batch_promotes_multiple_cold_frames_on_one_shard() {
    let file = TestFile::new();
    let engine = VericacheVllmEngine::new(1, 4, 8, 4096, Some(file.path())).unwrap();
    engine.set_writeback_policy(0).unwrap();
    for key in 1..=8 {
        engine.put(object_key(key), &[key as u8; 64]).unwrap();
    }
    engine.set_writeback_policy(75).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while engine.extended_stats().unwrap().async_writebacks < 4 {
        engine.poll_writeback().unwrap();
        assert!(
            std::time::Instant::now() < deadline,
            "writeback did not retire"
        );
        thread::yield_now();
    }

    let before = engine.extended_stats().unwrap();
    let keys = [3, 1, 4, 2];
    let leases = engine.acquire_batch(keys.iter().copied().map(object_key));
    let mut ids = Vec::new();
    for (key, lease) in keys.into_iter().zip(leases) {
        let lease = lease.unwrap().unwrap();
        let payload = unsafe {
            std::slice::from_raw_parts(ptr::with_exposed_provenance::<u8>(lease.address), lease.len)
        };
        assert_eq!(payload, &[key as u8; 64]);
        ids.push(lease.id);
    }
    let after = engine.extended_stats().unwrap();
    assert_eq!(after.nvme_promotions - before.nvme_promotions, 4);
    assert_eq!(after.nvme_hits - before.nvme_hits, 4);
    assert_eq!(after.active_leases, 4);
    engine.release_batch(&ids).unwrap();
}

#[test]
fn scans_reuse_promotion_frames_and_keep_requested_probation() {
    let file = TestFile::new();
    let engine = VericacheVllmEngine::new(1, 4, 16, 4096, Some(file.path())).unwrap();
    engine.set_writeback_policy(0).unwrap();
    for key in 1..=12 {
        engine.put(object_key(key), &[key as u8]).unwrap();
    }
    engine.set_writeback_policy(75).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while engine.extended_stats().unwrap().async_writebacks < 4 {
        engine.poll_writeback().unwrap();
        assert!(std::time::Instant::now() < deadline);
        thread::yield_now();
    }

    let ids: Vec<_> = engine
        .acquire_batch([1, 2, 3, 4].into_iter().map(object_key))
        .into_iter()
        .map(|lease| lease.unwrap().unwrap().id)
        .collect();
    engine.release_batch(&ids).unwrap();
    let mut ids = Vec::new();
    for (key, lease) in [5, 6, 7, 8, 4]
        .into_iter()
        .zip(engine.acquire_batch([5, 6, 7, 8, 4].into_iter().map(object_key)))
    {
        if key == 8 {
            assert!(matches!(lease, Err(Error::Busy)));
        } else {
            let lease = lease.unwrap().unwrap();
            assert_eq!(
                unsafe { *ptr::with_exposed_provenance::<u8>(lease.address) },
                key as u8
            );
            ids.push(lease.id);
        }
    }
    engine.release_batch(&ids).unwrap();
    let mut dst = [0; 1];
    assert_eq!(engine.get(object_key(8), &mut dst), Ok(Some(1)));
    assert_eq!(dst, [8]);

    let before = engine.extended_stats().unwrap();
    for key in [4, 9, 10, 11, 12] {
        assert_eq!(engine.get(object_key(key), &mut dst), Ok(Some(1)));
        assert_eq!(dst, [key as u8]);
    }
    let after = engine.extended_stats().unwrap();
    assert_eq!(after.nvme_promotions, before.nvme_promotions);
    assert_eq!(after.dram_hits - before.dram_hits, 5);
}

#[test]
fn reads_do_not_start_writeback() {
    let file = TestFile::new();
    let engine = VericacheVllmEngine::new(1, 4, 8, 4096, Some(file.path())).unwrap();
    engine.set_writeback_policy(0).unwrap();
    for key in 1..=5 {
        engine.put(object_key(key), &[key as u8]).unwrap();
    }
    engine.set_writeback_policy(75).unwrap();
    let mut dst = [0; 1];
    assert_eq!(engine.get(object_key(5), &mut dst), Ok(Some(1)));
    engine
        .release(engine.acquire(object_key(5)).unwrap().unwrap().id)
        .unwrap();
    let ids: Vec<_> = engine
        .acquire_batch([1, 5].into_iter().map(object_key))
        .into_iter()
        .map(|lease| lease.unwrap().unwrap().id)
        .collect();
    engine.release_batch(&ids).unwrap();
    // Eviction waits for any existing spill of this dirty key.
    assert_eq!(engine.evict(object_key(5)), Ok(true));
    assert_eq!(engine.extended_stats().unwrap().async_writebacks, 0);
}

#[test]
fn key256_abi_distinguishes_keys_with_the_same_first_word() {
    let first = [42u64, 1, 0, 0].map(u64::to_le);
    let second = [42u64, 2, 0, 0].map(u64::to_le);
    unsafe {
        let engine = vericache_engine_create_with_options(1, 3, 0, 32, ptr::null(), 0, 2);
        assert!(!engine.is_null());
        assert_eq!(
            vericache_engine_put_key256(engine, first.as_ptr(), b"first".as_ptr(), 5),
            0
        );
        assert_eq!(
            vericache_engine_put_key256(engine, second.as_ptr(), b"second".as_ptr(), 6),
            0
        );

        let mut dst = [0; 32];
        let mut len = 0;
        assert_eq!(
            vericache_engine_get_key256(
                engine,
                first.as_ptr(),
                dst.as_mut_ptr(),
                dst.len(),
                &mut len
            ),
            0
        );
        assert_eq!(&dst[..len], b"first");

        let mut address = ptr::null();
        let mut lease = 0;
        assert_eq!(
            vericache_engine_acquire_read_lease_key256(
                engine,
                second.as_ptr(),
                &mut address,
                &mut len,
                &mut lease,
            ),
            0
        );
        assert_eq!(std::slice::from_raw_parts(address, len), b"second");
        assert_eq!(vericache_engine_evict_key256(engine, second.as_ptr()), -4);
        assert_eq!(vericache_engine_evict_key256(engine, first.as_ptr()), 0);
        assert_eq!(vericache_engine_contains_key256(engine, first.as_ptr()), 0);
        assert_eq!(vericache_engine_contains_key256(engine, second.as_ptr()), 1);

        let mut evicted = [[0; 4]; 2];
        let out = evicted.as_mut_ptr().cast();
        assert_eq!(vericache_engine_drain_evictions_key256(engine, out, 2), 1);
        assert_eq!(evicted[0], first);
        assert_eq!(vericache_engine_release_read_lease(engine, lease), 0);
        assert_eq!(vericache_engine_evict_key256(engine, second.as_ptr()), 0);
        assert_eq!(vericache_engine_drain_evictions_key256(engine, out, 2), 1);
        assert_eq!(evicted[0], second);

        // The legacy and wide entry points agree on the canonical legacy key.
        let legacy = object_key(42).0.map(u64::to_le);
        assert_eq!(vericache_engine_put(engine, 42, b"legacy".as_ptr(), 6), 0);
        assert_eq!(vericache_engine_contains_key256(engine, legacy.as_ptr()), 1);
        assert_eq!(vericache_engine_evict_key256(engine, legacy.as_ptr()), 0);
        let mut legacy_evicted = 0;
        assert_eq!(
            vericache_engine_drain_evictions(engine, &mut legacy_evicted, 1),
            1
        );
        assert_eq!(legacy_evicted, 42);
        vericache_engine_destroy(engine);
    }
}
