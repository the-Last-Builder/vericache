use vstd::prelude::*;

verus! {

/// Non-dropping progress counters returned across backend polling steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgressReport {
    pub cuda_events: usize,
    pub published_gpu_puts: usize,
    pub nvme_completed: usize,
    pub nvme_failed: usize,
    pub reclaimed_frames: usize,
}

impl ProgressReport {
    pub fn empty() -> (r: Self)
        ensures
            r.cuda_events == 0,
            r.published_gpu_puts == 0,
            r.nvme_completed == 0,
            r.nvme_failed == 0,
            r.reclaimed_frames == 0,
    {
        Self {
            cuda_events: 0,
            published_gpu_puts: 0,
            nvme_completed: 0,
            nvme_failed: 0,
            reclaimed_frames: 0,
        }
    }
}

} // verus!
