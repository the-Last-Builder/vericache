use vericache::transport::NoTransport;
use vericache::{CacheError, DefaultPolicy, GpuConfig, StorageConfig, VeriCache};
use vericache_core::spec::ObjectKey;

fn key(id: u64) -> ObjectKey {
    ObjectKey([id, 0, 0, 0])
}

#[test]
fn host_write_batch_dmas_into_published_pinned_frames() {
    use std::cell::Cell;
    use std::ffi::{c_int, c_uint, c_void};

    if transfer::cuda::device_count().unwrap_or(0) == 0 {
        assert!(
            std::env::var_os("VERICACHE_REQUIRE_HW").is_none(),
            "CUDA GPU required"
        );
        eprintln!("SKIP host_write_batch_dmas_into_published_pinned_frames: no CUDA GPU");
        return;
    }

    #[repr(C)]
    struct CudaMemcpyAttributes {
        src_access_order: c_int,
        src_loc_hint: [c_int; 2],
        dst_loc_hint: [c_int; 2],
        flags: c_uint,
    }

    #[link(name = "cudart")]
    unsafe extern "C" {
        fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: c_uint) -> c_int;
        fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
        fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
        fn cudaMemcpyBatchAsync(
            dsts: *const *mut c_void,
            srcs: *const *const c_void,
            sizes: *const usize,
            count: usize,
            attrs: *mut CudaMemcpyAttributes,
            attrs_idxs: *mut usize,
            num_attrs: usize,
            stream: *mut c_void,
        ) -> c_int;
    }

    struct ExternalStream(*mut c_void);
    impl Drop for ExternalStream {
        fn drop(&mut self) {
            unsafe { cudaStreamDestroy(self.0) };
        }
    }

    const SIZE: usize = 4096;
    let cfg = StorageConfig::new(0, SIZE, 4).with_gpu(GpuConfig::new(0, 2));
    VeriCache::<NoTransport, _, _, _>::with_session(
        cfg,
        Vec::new(),
        DefaultPolicy::default(),
        |cache| {
            assert!(cache.is_host_pinned());
            let mut upload_stream = cache.create_gpu_stream()?;
            let mut slots = [cache.alloc_gpu_slot()?, cache.alloc_gpu_slot()?];
            let payloads = [[0x51; SIZE], [0xA2; SIZE]];
            for (slot, bytes) in slots.iter_mut().zip(&payloads) {
                cache.write_gpu_slot(slot, &mut upload_stream, bytes)?;
            }
            let srcs = slots
                .each_ref()
                .map(|slot| slot.device_addr() as *const c_void);
            let mut stream = std::ptr::null_mut();
            assert_eq!(unsafe { cudaStreamCreateWithFlags(&mut stream, 1) }, 0);
            let stream = ExternalStream(stream);
            let destinations = [Cell::new(0), Cell::new(0)];
            let keys = [key(1), key(2)];
            let written = cache.with_host_write_batch(
                &keys,
                |i, addr| {
                    destinations[i].set(addr);
                    Ok(())
                },
                || {
                    let dsts = destinations
                        .each_ref()
                        .map(|addr| addr.get() as *mut c_void);
                    let mut attr = CudaMemcpyAttributes {
                        src_access_order: 1, // cudaMemcpySrcAccessOrderStream
                        src_loc_hint: [0; 2],
                        dst_loc_hint: [0; 2],
                        flags: 0,
                    };
                    let mut attr_idx = 0;
                    let submitted = unsafe {
                        cudaMemcpyBatchAsync(
                            dsts.as_ptr(),
                            srcs.as_ptr(),
                            [SIZE; 2].as_ptr(),
                            dsts.len(),
                            &mut attr,
                            &mut attr_idx,
                            1,
                            stream.0,
                        )
                    };
                    let synced = unsafe { cudaStreamSynchronize(stream.0) };
                    if submitted != 0 || synced != 0 {
                        return Err(CacheError::Init(format!(
                            "D2H batch returned {submitted}, stream sync returned {synced}"
                        )));
                    }
                    Ok(())
                },
            )?;
            assert_eq!(written, keys.len());
            for (i, key) in keys.into_iter().enumerate() {
                let mut lease = cache.lease_host(key)?.unwrap();
                assert_eq!(lease.slice().addr(), destinations[i].get());
                let mut actual = [0; SIZE];
                lease.copy_into(&mut actual)?;
                assert_eq!(actual, payloads[i]);
            }
            Ok(())
        },
    )
    .expect("DMA directly into published pinned frames");
}
