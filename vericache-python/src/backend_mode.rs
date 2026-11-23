use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendMode {
    Distributed,
    Local,
}

// Each backend owns a separate shard allocator, but both draw from the
// same one-shot NodeContext. The first claim lasts for the process lifetime.
static MODE: OnceLock<BackendMode> = OnceLock::new();

pub(crate) fn claim_distributed() -> Result<(), &'static str> {
    if *MODE.get_or_init(|| BackendMode::Distributed) == BackendMode::Distributed {
        Ok(())
    } else {
        Err(
            "cannot initialize Distributed VeriCache backend after Local backend in the same process",
        )
    }
}

pub(crate) fn claim_local() -> Result<(), &'static str> {
    if *MODE.get_or_init(|| BackendMode::Local) == BackendMode::Local {
        Ok(())
    } else {
        Err(
            "cannot initialize Local VeriCache backend after Distributed backend in the same process",
        )
    }
}
