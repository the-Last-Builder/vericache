#[cfg(all(feature = "nvme", verus_only))]
use crate::backends::nvme::lemma_core_transition_preserves_nvme;
#[cfg(feature = "nvme")]
use crate::backends::nvme::NvmeBackend;
#[cfg(verus_only)]
use crate::backends::{core_transition, CoreFootprint};
#[cfg(verus_only)]
use crate::backends::{host::HostBackend, DeviceBackend};
#[cfg(verus_only)]
use crate::kernel::CacheCore;
use crate::spec::TierId;
#[cfg(verus_only)]
use crate::spec::TimelineId;
#[cfg(all(feature = "nvme", verus_only))]
use transfer::nvme::DIRECT_IO_ALIGNMENT;
use vstd::prelude::*;

/// The optional tier passed to engine constructors. Without NVMe support, pass `None`.
#[cfg(feature = "nvme")]
pub type NvmeStorage = Option<NvmeBackend>;
#[cfg(not(feature = "nvme"))]
pub type NvmeStorage = Option<()>;

verus! {

pub open(crate) spec fn nvme_well_formed(nvme: &NvmeStorage, core: &CacheCore) -> bool {
    #[cfg(feature = "nvme")]
    let valid = *nvme matches Some(n) ==> n.well_formed(core);
    #[cfg(not(feature = "nvme"))]
    let valid = nvme is None;
    valid
}

pub open(crate) spec fn nvme_compatible<'a, D: DeviceBackend<'a>>(
    nvme: &NvmeStorage,
    core: &CacheCore,
    host: &HostBackend,
    device: Option<D>,
) -> bool {
    #[cfg(feature = "nvme")]
    let compatible = *nvme matches Some(n) ==> {
        &&& host.buffer_spec().align() % (DIRECT_IO_ALIGNMENT as usize) == 0
        &&& n.timeline_spec() != host.timeline_spec()
        &&& (device matches Some(c) ==> n.timeline_spec() != c.timeline_id_spec())
    };
    #[cfg(not(feature = "nvme"))]
    let compatible = true;
    nvme_well_formed(nvme, core) && compatible
}

pub open(crate) spec fn nvme_timeline_distinct(nvme: &NvmeStorage, timeline: TimelineId) -> bool {
    #[cfg(feature = "nvme")]
    let distinct = *nvme matches Some(n) ==> n.timeline_spec() != timeline;
    #[cfg(not(feature = "nvme"))]
    let distinct = true;
    distinct
}

pub open(crate) spec fn nvme_pending_untouched(
    nvme: &NvmeStorage,
    before: &CacheCore,
    fp: CoreFootprint,
) -> bool {
    #[cfg(feature = "nvme")]
    let untouched = *nvme matches Some(n) ==> n.pending_untouched(before, fp);
    #[cfg(not(feature = "nvme"))]
    let untouched = true;
    untouched
}

pub open(crate) spec fn nvme_untouched(
    nvme: &NvmeStorage,
    before: &CacheCore,
    fp: CoreFootprint,
) -> bool {
    #[cfg(feature = "nvme")]
    let timeline_untouched = *nvme matches Some(n) ==> !fp.touched_timelines.contains(n.timeline);
    #[cfg(not(feature = "nvme"))]
    let timeline_untouched = true;
    timeline_untouched && nvme_pending_untouched(nvme, before, fp)
}

pub(super) proof fn lemma_preserve_nvme(
    before: &CacheCore,
    after: &CacheCore,
    fp: CoreFootprint,
    nvme: &NvmeStorage,
)
    requires
        before.well_formed(),
        after.well_formed(),
        core_transition(before, after, fp),
        nvme_well_formed(nvme, before),
        nvme_untouched(nvme, before, fp),
    ensures
        nvme_well_formed(nvme, after),
{
    #[cfg(feature = "nvme")]
    if let Some(ref n) = *nvme {
        lemma_core_transition_preserves_nvme(before, after, fp, n);
    }
}

pub open(crate) spec fn nvme_tier_spec(nvme: &NvmeStorage) -> Option<TierId> {
    #[cfg(feature = "nvme")]
    let tier = match *nvme {
        Some(n) => Some(n.tier_spec()),
        None => None,
    };
    #[cfg(not(feature = "nvme"))]
    let tier = None;
    tier
}

#[cfg_attr(not(feature = "nvme"), allow(unused_variables))]
pub(super) fn nvme_tier(nvme: &NvmeStorage) -> (t: Option<TierId>)
    ensures
        t == nvme_tier_spec(nvme),
{
    #[cfg(feature = "nvme")]
    let tier = match nvme {
        Some(n) => Some(n.tier_id()),
        None => None,
    };
    #[cfg(not(feature = "nvme"))]
    let tier = None;
    tier
}

} // verus!
