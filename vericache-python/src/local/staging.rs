use super::shard_cap::ShardCapabilityBundle;
use crate::{backend_mode, set_last_error};
use std::sync::{Mutex, OnceLock};
use transfer::context::SlotToken;
use transfer::cuda::{self, CudaHostBuffer};
use transfer::host::{self, HostBuffer, HostContext};
use vericache_core::kernel::CacheCore;
use vstd::prelude::*;

enum Storage {
    Cuda(CudaHostBuffer),
    Host(HostBuffer),
}

#[cfg_attr(not(verus_only), allow(dead_code))]
struct StagingAllocator {
    caps: ShardCapabilityBundle,
    ctx: HostContext,
    core: CacheCore,
}

static ALLOCATOR: OnceLock<Option<Mutex<StagingAllocator>>> = OnceLock::new();

pub struct StagingBuffer {
    storage: Option<(Storage, Tracked<SlotToken>)>,
    len: usize,
}

verus! {

impl StagingAllocator {
    fn new() -> Option<Self> {
        let mut caps = ShardCapabilityBundle::take()?;
        let core = caps.make_local_core(1);
        let ctx = HostContext::new(caps.mint_host());
        Some(Self { caps, ctx, core })
    }

    fn allocate(&mut self, len: usize, prefer_cuda: bool) -> Option<StagingBuffer> {
        let pinned = if prefer_cuda {
            cuda::alloc_host(Tracked(self.caps.alloc.borrow_mut()), len).ok()
        } else {
            None
        };
        let (storage, token) = match pinned {
            Some((buf, token)) => (Storage::Cuda(buf), token),
            None => {
                let (buf, token) = host::alloc_aligned(
                    &self.ctx, Tracked(self.caps.instance.borrow()),
                    Tracked(self.caps.alloc.borrow_mut()), len, 4096,
                )?;
                (Storage::Host(buf), token)
            },
        };
        Some(StagingBuffer { storage: Some((storage, token)), len })
    }

    fn free(&mut self, storage: Storage, token: Tracked<SlotToken>) {
        match storage {
            Storage::Cuda(buf) => {
                let _ = cuda::free_host(
                    &mut self.ctx,
                    Tracked(self.caps.instance.borrow()),
                    buf,
                    token,
                    Tracked(&self.core.ledger.borrow().submitted),
                    Tracked(&self.core.ledger.borrow().knows),
                    Ghost(self.core.citations()),
                );
            },
            Storage::Host(buf) => {
                let _ = host::free(
                    &mut self.ctx,
                    Tracked(self.caps.instance.borrow()),
                    buf,
                    token,
                    Tracked(&self.core.ledger.borrow().submitted),
                    Tracked(&self.core.ledger.borrow().knows),
                    Ghost(self.core.citations()),
                );
            },
        }
    }
}

} // verus!

impl StagingBuffer {
    pub(crate) fn allocate(len: usize, prefer_cuda: bool) -> Option<Self> {
        if len == 0 || len > isize::MAX as usize {
            return None;
        }
        if let Err(error) = backend_mode::claim_local() {
            set_last_error(error);
            return None;
        }
        ALLOCATOR
            .get_or_init(|| StagingAllocator::new().map(Mutex::new))
            .as_ref()?
            .lock()
            .ok()?
            .allocate(len, prefer_cuda)
    }

    pub(crate) fn address(&self) -> usize {
        match self.storage.as_ref() {
            Some((Storage::Cuda(buf), _)) => buf.as_host_slice().addr(),
            Some((Storage::Host(buf), _)) => buf.as_slice().addr(),
            None => 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_cuda_pinned(&self) -> bool {
        matches!(&self.storage, Some((Storage::Cuda(_), _)))
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        if let Some((storage, token)) = self.storage.take() {
            if let Some(Some(allocator)) = ALLOCATOR.get() {
                allocator
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .free(storage, token);
            }
        }
    }
}
