//! CUDA runtime resources and caller DMA for the facade.
#[cfg(verus_only)]
use transfer::context::EventSourceKind;
use transfer::context::SlotToken;
use transfer::cuda::{self, CudaBuffer, CudaHostBuffer, CudaSlice, CudaStream};
use transfer::domain::DeviceId;
use transfer::host::{self, HostContext};
use transfer::transport::ByteTransport;
use vericache_core::backends::cuda::CudaBackend;
use vericache_core::backends::{host::HostBackend, DeviceBackend};
use vericache_core::engine::DistributedEngine;
use vericache_core::kernel::CacheCore;
use vericache_core::provenance::ValueCertificate;
#[cfg(verus_only)]
use vericache_core::provenance::ValueProof;
use vericache_core::spec::{CoreError, TimelineId};
use vericache_core::timeline::TimelineKind;
use vstd::prelude::{verus, Ghost, Tracked};

use crate::device::{DeviceRuntime, DeviceSlotParts};
use crate::lease::DeviceLeaseGuard;
use crate::policy::{EvictionPolicy, PeerRoutingPolicy, TieringPolicy};
use crate::resources::{GpuConfig, ShardCaps};
use crate::CacheError;

/// CUDA implementation of the facade's device runtime.
pub struct CudaRuntime;

/// Reusable pinned host buffer and its transfer capabilities.
pub struct CudaStaging {
    buffer: CudaHostBuffer,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    token: Tracked<SlotToken>,
    context: HostContext,
}

verus! {

/// Caller-owned GPU buffer slot managed by `VeriCache`, hiding `SlotToken` and
/// `ValueCertificate` provenance from ergonomic callers.
#[cfg_attr(not(verus_only), allow(dead_code))]
pub struct GpuSlot {
    buffer: CudaBuffer,
    token: Tracked<SlotToken>,
    cert: Option<Ghost<ValueCertificate>>,
    frame_size: usize,
    device_id: DeviceId,
}

impl GpuSlot {
    pub fn len(&self) -> usize {
        self.frame_size
    }

    pub fn is_empty(&self) -> bool {
        self.frame_size == 0
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Raw CUDA device address of this slot (for external CUDA interop such as
    /// PyTorch/vLLM `cudaMemcpyAsync` into/out of a managed `GpuSlot`).
    pub fn device_addr(&self) -> usize {
        self.buffer.slot_slice(0).device_addr()
    }
}

/// Record external GPU-producer provenance for a `GpuSlot` populated via an
/// external CUDA stream (e.g., PyTorch/vLLM tensor D2D copy into `slot.device_addr()`).
#[cfg_attr(not(verus_only), allow(unused_variables))]
fn mark_gpu_slot_external_write(slot: &mut GpuSlot) {
    let slice = slot.buffer.slot_slice(0);
    let ghost cert = ValueCertificate {
        slot: slice.slot_ref(),
        writer: transfer::domain::TransferEvent {
            source: EventSourceKind::host().source(0),
            observer: 0,
            seq: 0,
        },
        bytes: vstd::prelude::Seq::empty(),
        proof: ValueProof::HostWrite,
    };
    slot.cert = Some(Ghost(cert));
}

/// Caller-owned CUDA stream paired with its registered `CacheCore` timeline id.
pub struct GpuStream {
    stream: CudaStream,
    timeline: TimelineId,
    device_id: DeviceId,
}

impl GpuStream {
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn timeline_id(&self) -> TimelineId {
        self.timeline
    }
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn alloc_gpu_slot_internal(caps: &mut ShardCaps, device_id: DeviceId, frame_size: usize) -> Result<
    GpuSlot,
    CacheError,
> {
    let (buffer, token) = cuda::alloc_device(
        Tracked(caps.alloc.borrow_mut()),
        device_id,
        frame_size,
    ).map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
    Ok(GpuSlot { buffer, token, cert: None, frame_size, device_id })
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn upload_gpu_slot_internal<'a, T: ByteTransport>(
    engine: &mut DistributedEngine<'a, T, CudaBackend<'a>>,
    caps: &mut ShardCaps,
    staging: &mut CudaStaging,
    stream: &mut GpuStream,
    slot: &mut GpuSlot,
    data: &[u8],
) -> Result<(), CacheError> {
    if data.len() != slot.frame_size {
        return Err(CacheError::Core(CoreError::InvalidArgument));
    }
    let host_slice = staging.buffer.as_host_slice();
    let Tracked(w_comp) = host::write_slice(
        &mut staging.context,
        Tracked(caps.instance.borrow()),
        &host_slice,
        data,
        Tracked(staging.token.borrow_mut()),
        Tracked(&engine.core().ledger().borrow().submitted),
        Tracked(&engine.core().ledger().borrow().knows),
        Ghost(engine.core().citations()),
    );
    let ghost w_ev = w_comp.submitted.key();
    engine.record_host_completion(Tracked(w_comp));

    let src_cuda = staging.buffer.as_cuda_slice();
    let dst_cuda = slot.buffer.slot_slice(0);
    let tracked mut tmp_tok = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(staging.token.borrow_mut(), &mut tmp_tok);
    }
    let tracked mut src_perm = transfer::context::ReadPerm::Owned(tmp_tok);
    let copy_res = {
        let tracked mut submitted = caps.instance.borrow().duplicate_submitted(
            &engine.core().ledger().borrow().submitted,
        );
        cuda::memcpy_async(
            &mut stream.stream,
            Tracked(caps.instance.borrow()),
            &dst_cuda,
            &src_cuda,
            Tracked(&mut src_perm),
            Tracked(slot.token.borrow_mut()),
            Tracked(&mut submitted),
            Tracked(&engine.core().ledger().borrow().knows),
            Ghost(engine.core().citations()),
        )
    };
    proof {
        *staging.token.borrow_mut() = src_perm.into_owned();
    }
    let (Ghost(_copy_ev), Tracked(receipt)) = copy_res.map_err(
        |e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }),
    )?;
    let t_copy = engine.record_cuda_submission(
        &stream.stream,
        stream.timeline,
        Tracked(receipt),
    ).map_err(CacheError::Core)?;
    engine.synchronize_caller_stream(
        &stream.stream,
        stream.timeline,
        t_copy,
        Tracked(caps.instance.borrow()),
    ).map_err(CacheError::Core)?;
    let ghost slot_cert = ValueCertificate {
        slot: dst_cuda.slot_ref(),
        writer: w_ev,
        bytes: data@,
        proof: ValueProof::HostWrite,
    };
    slot.cert = Some(Ghost(slot_cert));
    Ok(())
}

#[cfg_attr(not(verus_only), allow(unused_variables))]
fn download_gpu_slot_internal<'a, T: ByteTransport>(
    engine: &mut DistributedEngine<'a, T, CudaBackend<'a>>,
    caps: &mut ShardCaps,
    staging: &mut CudaStaging,
    stream: &mut GpuStream,
    slot: &mut GpuSlot,
    dst: &mut [u8],
) -> Result<(), CacheError> {
    if dst.len() != slot.frame_size {
        return Err(CacheError::Core(CoreError::InvalidArgument));
    }
    let src_cuda = slot.buffer.slot_slice(0);
    let dst_cuda = staging.buffer.as_cuda_slice();
    let tracked mut tmp_tok = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(slot.token.borrow_mut(), &mut tmp_tok);
    }
    let tracked mut src_perm = transfer::context::ReadPerm::Owned(tmp_tok);
    let copy_res = {
        let tracked mut submitted = caps.instance.borrow().duplicate_submitted(
            &engine.core().ledger().borrow().submitted,
        );
        cuda::memcpy_async(
            &mut stream.stream,
            Tracked(caps.instance.borrow()),
            &dst_cuda,
            &src_cuda,
            Tracked(&mut src_perm),
            Tracked(staging.token.borrow_mut()),
            Tracked(&mut submitted),
            Tracked(&engine.core().ledger().borrow().knows),
            Ghost(engine.core().citations()),
        )
    };
    proof {
        *slot.token.borrow_mut() = src_perm.into_owned();
    }
    let (_, Tracked(receipt)) = copy_res.map_err(
        |e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }),
    )?;
    let t_copy = engine.record_cuda_submission(
        &stream.stream,
        stream.timeline,
        Tracked(receipt),
    ).map_err(CacheError::Core)?;
    engine.synchronize_caller_stream(
        &stream.stream,
        stream.timeline,
        t_copy,
        Tracked(caps.instance.borrow()),
    ).map_err(CacheError::Core)?;

    let host_slice = staging.buffer.as_host_slice();
    let tracked mut tmp_stage = SlotToken::empty(caps.instance.borrow().id());
    proof {
        vstd::modes::tracked_swap(staging.token.borrow_mut(), &mut tmp_stage);
    }
    let tracked mut stage_perm = transfer::context::ReadPerm::Owned(tmp_stage);
    let Tracked(r_comp) = host::read_slice(
        &mut staging.context,
        Tracked(caps.instance.borrow()),
        &host_slice,
        dst,
        Tracked(&mut stage_perm),
        Tracked(&engine.core().ledger().borrow().submitted),
        Tracked(&engine.core().ledger().borrow().knows),
        Ghost(engine.core().citations()),
    );
    proof {
        *staging.token.borrow_mut() = stage_perm.into_owned();
    }
    engine.record_host_completion(Tracked(r_comp));
    Ok(())
}

} // verus!
verus! {

impl DeviceRuntime for CudaRuntime {
    type Backend<'a> = CudaBackend<'a>;

    type Slot = GpuSlot;

    type Stream = GpuStream;

    type Staging = CudaStaging;

    fn new_backend<'a>(core: &mut CacheCore, caps: &mut ShardCaps, config: &GpuConfig) -> Result<
        CudaBackend<'a>,
        CacheError,
    > {
        CudaBackend::new(
            core,
            Tracked(caps.instance.borrow()),
            Tracked(caps.alloc.borrow_mut()),
            caps.mint_event(Ghost(EventSourceKind::cuda_stream())),
            config.device_id,
            config.hbm_frames,
            config.event_ring_size,
            config.pin_host_dram,
        ).map_err(CacheError::Core)
    }

    fn attach_pinned_host<'a>(
        backend: &mut CudaBackend<'a>,
        core: &CacheCore,
        host: &'a HostBackend,
    ) -> Result<(), CacheError> {
        backend.attach_pinned_host(core, host).map_err(CacheError::Core)
    }

    fn borrow_backend<'a, 'storage: 'a>(backend: &'a CudaBackend<'storage>) -> &'a CudaBackend<'a> {
        backend
    }

    fn new_stream<'a, T: ByteTransport>(
        engine: &mut DistributedEngine<'a, T, CudaBackend<'a>>,
        caps: &mut ShardCaps,
    ) -> Result<GpuStream, CacheError> {
        let device_id = engine.device().ok_or(CoreError::UnsupportedTier)?.device();
        let timeline = engine.register_timeline(TimelineKind::CudaStream { device: device_id })?;
        let stream = CudaStream::new(
            device_id,
            caps.mint_event(Ghost(EventSourceKind::cuda_stream())),
        ).map_err(|e| CacheError::Core(CoreError::Backend { kind: 1, code: e.0 }))?;
        Ok(GpuStream { stream, timeline, device_id })
    }

    fn new_staging(caps: &mut ShardCaps, frame_size: usize) -> Result<CudaStaging, CacheError> {
        let (buffer, token) = cuda::alloc_host(
            Tracked(caps.alloc.borrow_mut()),
            frame_size,
        ).map_err(|e| CacheError::Init(format!("alloc pinned staging failed: {e:?}")))?;
        let context = HostContext::new(caps.mint_host());
        Ok(CudaStaging { buffer, token, context })
    }

    fn alloc_slot(caps: &mut ShardCaps, device: DeviceId, frame_size: usize) -> Result<
        GpuSlot,
        CacheError,
    > {
        alloc_gpu_slot_internal(caps, device, frame_size)
    }

    fn slot_len(slot: &GpuSlot) -> usize {
        slot.len()
    }

    fn upload_slot<'a, T: ByteTransport>(
        engine: &mut DistributedEngine<'a, T, CudaBackend<'a>>,
        caps: &mut ShardCaps,
        staging: &mut CudaStaging,
        stream: &mut GpuStream,
        slot: &mut GpuSlot,
        data: &[u8],
    ) -> Result<(), CacheError> {
        upload_gpu_slot_internal(engine, caps, staging, stream, slot, data)
    }

    fn download_slot<'a, T: ByteTransport>(
        engine: &mut DistributedEngine<'a, T, CudaBackend<'a>>,
        caps: &mut ShardCaps,
        staging: &mut CudaStaging,
        stream: &mut GpuStream,
        slot: &mut GpuSlot,
        dst: &mut [u8],
    ) -> Result<(), CacheError> {
        download_gpu_slot_internal(engine, caps, staging, stream, slot, dst)
    }

    fn stream_parts<'a, 'stream>(stream: &'stream mut GpuStream) -> Result<
        (&'stream mut <Self::Backend<'a> as DeviceBackend<'a>>::Stream, TimelineId),
        CacheError,
    > {
        Ok((&mut stream.stream, stream.timeline))
    }

    fn slot_parts<'a, 'slot>(slot: &'slot mut GpuSlot) -> Result<
        DeviceSlotParts<'a, 'slot, CudaBackend<'a>>,
        CacheError,
    > {
        Ok(
            DeviceSlotParts {
                slice: slot.buffer.slot_slice(0),
                token: &mut slot.token,
                certificate: slot.cert,
            },
        )
    }

    fn mark_external_write(slot: &mut GpuSlot) -> Result<(), CacheError> {
        mark_gpu_slot_external_write(slot);
        Ok(())
    }
}

} // verus!
impl<'cache, 'storage, T, E, P, R> DeviceLeaseGuard<'cache, 'storage, T, E, P, R, CudaRuntime>
where
    T: ByteTransport,
    E: EvictionPolicy,
    P: TieringPolicy,
    R: PeerRoutingPolicy,
{
    /// Borrow the CUDA slice for the leased GPU frame.
    pub fn cuda_slice(&self) -> Option<CudaSlice<'_>> {
        self.device_slice()
    }
}
