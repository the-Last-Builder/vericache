use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendMode {
    Distributed,
    DistributedVllm,
    Local,
}

// Each backend owns a separate shard allocator, but all draw from the
// same one-shot NodeContext. The first claim lasts for the process lifetime.
static MODE: OnceLock<BackendMode> = OnceLock::new();

pub(crate) fn claim_distributed() -> Result<(), &'static str> {
    match *MODE.get_or_init(|| BackendMode::Distributed) {
        BackendMode::Distributed => Ok(()),
        BackendMode::Local => Err(
            "cannot initialize Distributed VeriCache backend after Local backend in the same process",
        ),
        BackendMode::DistributedVllm => Err(
            "cannot initialize Distributed VeriCache backend after DistributedVllm backend in the same process",
        ),
    }
}

pub(crate) fn claim_distributed_vllm() -> Result<(), &'static str> {
    match *MODE.get_or_init(|| BackendMode::DistributedVllm) {
        BackendMode::DistributedVllm => Ok(()),
        BackendMode::Local => Err(
            "cannot initialize DistributedVllm VeriCache backend after Local backend in the same process",
        ),
        BackendMode::Distributed => Err(
            "cannot initialize DistributedVllm VeriCache backend after Distributed backend in the same process",
        ),
    }
}

pub(crate) fn is_distributed_vllm() -> bool {
    matches!(MODE.get(), Some(BackendMode::DistributedVllm))
}

pub(crate) fn claim_local() -> Result<(), &'static str> {
    match *MODE.get_or_init(|| BackendMode::Local) {
        BackendMode::Local => Ok(()),
        BackendMode::Distributed => Err(
            "cannot initialize Local VeriCache backend after Distributed backend in the same process",
        ),
        BackendMode::DistributedVllm => Err(
            "cannot initialize Local VeriCache backend after DistributedVllm backend in the same process",
        ),
    }
}
