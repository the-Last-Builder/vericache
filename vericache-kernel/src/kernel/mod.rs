#![cfg_attr(verus_only, verifier::loop_isolation(false))]
use crate::{catalog::*, frame::*, provenance::*, spec::*, timeline::*};
#[cfg(verus_only)]
use foundation::tracked::{fold_share, untouched, Frontier};
use std::sync::atomic::{AtomicU64, Ordering};
use transfer_core::{context::*, domain::*};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

mod backend;
mod copy;
mod inspect;
mod lease;
mod ledger;
mod reclaim;
mod revoke;
mod write;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadLeaseHandle {
    pub(crate) slot: u32,
    pub(crate) generation: u64,
    pub(crate) frame: FrameId,
    pub(crate) core_id: Option<u64>,
}

impl ReadLeaseHandle {
    pub open(crate) spec fn frame_spec(&self) -> FrameId {
        self.frame
    }

    pub open(crate) spec fn slot_spec(&self) -> u32 {
        self.slot
    }

    pub open(crate) spec fn generation_spec(&self) -> u64 {
        self.generation
    }

    pub open(crate) spec fn core_id_spec(&self) -> Option<u64> {
        self.core_id
    }

    pub fn core_id(&self) -> (id: Option<u64>)
        ensures
            id == self.core_id_spec(),
    {
        self.core_id
    }

    pub fn frame(&self) -> (frame: FrameId)
        ensures
            frame == self.frame_spec(),
    {
        self.frame
    }

    pub fn slot(&self) -> (slot: u32)
        ensures
            slot == self.slot_spec(),
    {
        self.slot
    }

    pub fn generation(&self) -> (generation: u64)
        ensures
            generation == self.generation_spec(),
    {
        self.generation
    }
}

pub struct ActiveLocalLease {
    pub internal: bool,
    pub handle: ReadLeaseHandle,
    pub share: Ghost<TransferShare>,
}

pub struct WriteHandle {
    pub frame: FrameId,
    pub tier: TierId,
    pub key: ObjectKey,
    pub version: VersionId,
    pub slot: Ghost<SlotRef>,
}

pub struct CopyHopHandle {
    pub src_frame: FrameId,
    pub src_tier: TierId,
    pub src_slot: Ghost<SlotRef>,
    pub src_lease: ReadLeaseHandle,
    pub dst_frame: FrameId,
    pub dst_tier: TierId,
    pub dst_slot: Ghost<SlotRef>,
    pub key: ObjectKey,
    pub version: VersionId,
    pub src_cert: Ghost<ValueCertificate>,
}

pub tracked struct VersionRegistry {
    pub tracked instance: VersionInstance,
    pub tracked source: VersionSource,
    pub tracked bindings: Map<VersionId, VersionBinding>,
}

// Live backend resources carry this process-local identity across assembly.
exec static NEXT_CORE_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_core_id() -> Option<u64> {
    let mut attempts = 0usize;
    while attempts < 64
        invariant
            attempts <= 64,
        decreases 64 - attempts,
    {
        let id = NEXT_CORE_ID.load(Ordering::Relaxed);
        if id == u64::MAX {
            return None;
        }
        if NEXT_CORE_ID.compare_exchange(id, id + 1, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            return Some(id);
        }
        attempts += 1;
    }
    None
}

pub struct CacheCore {
    pub(crate) runtime_id: Option<u64>,
    pub(crate) version_registry_enabled: bool,
    pub(crate) node_id: u16,
    pub(crate) tiers: Vec<TierDescriptor>,
    pub(crate) timelines: TimelineTable,
    pub(crate) frames: FrameTable,
    pub(crate) catalog: Catalog,
    pub(crate) ledger: Tracked<KnowledgeLedger>,
    pub(crate) version_seq_start: u64,
    pub(crate) next_version_seq: u64,
    pub(crate) version_seq_limit: u64,
    pub(crate) active_leases: Vec<Option<ActiveLocalLease>>,
    pub(crate) free_lease_slots: Vec<u32>,
    pub(crate) lease_generation: u64,
    pub(crate) spec: Ghost<ServiceSpec>,
    pub(crate) version_registry: Tracked<Option<VersionRegistry>>,
}

impl CacheCore {
    pub open(crate) spec fn lease_generation_spec(&self) -> u64 {
        self.lease_generation
    }

    pub open(crate) spec fn free_lease_slots_spec(&self) -> Vec<u32> {
        self.free_lease_slots
    }

    pub open(crate) spec fn version_seq_start_spec(&self) -> u64 {
        self.version_seq_start
    }

    pub open(crate) spec fn version_seq_limit_spec(&self) -> u64 {
        self.version_seq_limit
    }

    pub open(crate) spec fn ledger_spec(&self) -> Tracked<KnowledgeLedger> {
        self.ledger
    }

    pub open(crate) spec fn version_registry_spec(&self) -> Tracked<Option<VersionRegistry>> {
        self.version_registry
    }

    pub open(crate) spec fn node_id_spec(&self) -> u16 {
        self.node_id
    }

    pub open(crate) spec fn frames_spec(&self) -> FrameTable {
        self.frames
    }

    pub open(crate) spec fn timelines_spec(&self) -> TimelineTable {
        self.timelines
    }

    pub open(crate) spec fn tiers_spec(&self) -> Vec<TierDescriptor> {
        self.tiers
    }

    pub open(crate) spec fn catalog_spec(&self) -> Catalog {
        self.catalog
    }

    pub open(crate) spec fn active_leases_spec(&self) -> Vec<Option<ActiveLocalLease>> {
        self.active_leases
    }

    pub open(crate) spec fn spec_spec(&self) -> Ghost<ServiceSpec> {
        self.spec
    }

    pub open(crate) spec fn next_version_seq_spec(&self) -> u64 {
        self.next_version_seq
    }

    pub open spec fn node(&self) -> NodeId {
        self.node_id_spec() as NodeId
    }

    pub open spec fn instance_id(&self) -> InstanceId {
        self.ledger_spec()@.submitted.instance_id()
    }

    pub open spec fn citations(&self) -> ISet<TransferEvent> {
        self.ledger_spec()@.citations(self.node())
    }

    pub open spec fn tier_frame_size_spec(&self, t: TierId) -> usize {
        self.frames_spec().tier_frame_size(t)
    }

    #[verifier::when_used_as_spec(tier_frame_size_spec)]
    pub fn registered_tier_frame_size(&self, t: TierId) -> (s: usize)
        requires
            self.well_formed(),
            t.valid(self.tiers_spec().len()),
        ensures
            s == self.frames_spec().tier_frame_size(t),
            0 < s <= u64::MAX,
    {
        self.frames.tier_size(t)
    }

    pub open spec fn frame_slot(&self, f: FrameId) -> SlotRef {
        self.frames_spec().frame(f).slot
    }

    pub open spec fn frame_state_spec(&self, f: FrameId) -> u8 {
        self.frames_spec().meta(f).state
    }

    pub open spec fn frame_tier_spec(&self, f: FrameId) -> TierId {
        self.frames_spec().frame(f).tier
    }

    pub open spec fn frame_cert(&self, f: FrameId) -> Option<ValueCertificate> {
        self.frames_spec().frame(f).cert
    }

    pub open spec fn disjoint_from_published(&self, writes: ISet<SlotLoc>) -> bool {
        forall|f: FrameId|
            f.0 < self.frames_spec().metas_spec().len() && (#[trigger] self.frames_spec().meta(
                f,
            )).state == STATE_PUBLISHED ==> !writes.contains(self.frames_spec().frame(f).slot.loc)
    }

    pub open spec fn kernel_view(&self) -> KernelView {
        KernelView::of(&self.timelines_spec(), &self.ledger_spec()@, self.node())
    }

    pub open spec fn service(frames: &FrameTable, catalog: &Catalog) -> ServiceSpec {
        ServiceSpec {
            frame_size: frames.frame_size_spec() as nat,
            versions: frames.versions_spec()@,
            catalog: Map::new(catalog.view().dom(), |k: ObjectKey| catalog.view()[k].version),
        }
    }

    pub open spec fn certified(frames: &FrameTable, v: KernelView) -> bool {
        forall|f: FrameId|
            f.0 < frames.metas_spec().len() && (#[trigger] frames.meta(f)).state == STATE_PUBLISHED
                ==> {
                let g = frames.frame(f);
                g.cert matches Some(cert) && {
                    &&& replica_valid(v.ledger, v.node, cert)
                    &&& event_succeeded(v.ledger, v.node, cert.writer)
                    &&& v.ledger.citations(v.node).contains(cert.writer)
                    &&& g.bound_version matches Some(ver)
                    &&& frames.versions_spec()@.contains_key(ver)
                    &&& frames.versions_spec()@[ver] == cert.bytes
                }
            }
    }

    pub open spec fn tiers_valid(&self) -> bool {
        &&& self.tiers_spec().len() <= MAX_TIERS
        &&& self.frames_spec().num_tiers_spec() == self.tiers_spec().len()
        &&& self.catalog_spec().num_tiers() == self.tiers_spec().len()
        &&& forall|i: int|
            0 <= i < self.tiers_spec().len() ==> (#[trigger] self.tiers_spec()[i]).id.0 as usize
                == i
    }

    pub open spec fn version_range_valid_for(&self, frames: &FrameTable) -> bool {
        &&& 1 <= self.version_seq_start_spec() <= self.next_version_seq_spec()
            <= self.version_seq_limit_spec() <= MAX_VERSION_SEQ + 1
        &&& forall|v: VersionId| #[trigger]
            frames.versions_spec()@.contains_key(v) && v.origin_node == self.node_id_spec() ==> (
            v.seq < self.next_version_seq_spec() || v.seq >= self.version_seq_limit_spec())
    }

    pub open spec fn version_range_valid(&self) -> bool {
        self.version_range_valid_for(&self.frames_spec())
    }

    pub open spec fn leases_valid_for(&self, frames: &FrameTable) -> bool {
        &&& self.active_leases_spec().len() <= u32::MAX
        &&& self.free_lease_slots_spec()@.no_duplicates()
        &&& forall|i: int|
            0 <= i < self.free_lease_slots_spec().len() ==> {
                let s = #[trigger] self.free_lease_slots_spec()[i];
                (s as usize) < self.active_leases_spec().len()
                    && self.active_leases_spec()[s as int].is_none()
            }
        &&& forall|s: int|
            0 <= s < self.active_leases_spec().len() ==> {
                match #[trigger] self.active_leases_spec()[s] {
                    None => self.free_lease_slots_spec()@.contains(s as u32),
                    Some(lease) => {
                        &&& !self.free_lease_slots_spec()@.contains(s as u32)
                        &&& lease.handle.slot_spec() as int == s
                        &&& 1 <= lease.handle.generation_spec() <= self.lease_generation_spec()
                        &&& lease.handle.frame_spec().0 < frames.metas_spec().len()
                        &&& frames.frame(lease.handle.frame_spec()).local_shares.contains_key(
                            lease.share@,
                        )
                    },
                }
            }
        &&& forall|s1: int, s2: int|
            0 <= s1 < self.active_leases_spec().len() && 0 <= s2 < self.active_leases_spec().len()
                && s1 != s2 ==> match (#[trigger]
                self.active_leases_spec()[s1],
                #[trigger] self.active_leases_spec()[s2],
            ) {
                (Some(l1), Some(l2)) => l1.handle.frame_spec() == l2.handle.frame_spec()
                    ==> l1.share@ != l2.share@,
                _ => true,
            }
    }

    pub open spec fn leases_valid(&self) -> bool {
        self.leases_valid_for(&self.frames_spec())
    }

    pub open spec fn active_lease_matches(&self, handle: ReadLeaseHandle) -> bool {
        (handle.slot_spec() as usize) < self.active_leases_spec().len() && (
        self.active_leases_spec()[handle.slot_spec() as int] matches Some(lease) && lease.handle
            == handle)
    }

    pub open spec fn has_version_registry(&self) -> bool {
        self.version_registry_spec()@ is Some
    }

    pub open spec fn has_version_binding(&self, v: VersionId) -> bool {
        self.version_registry_spec()@ matches Some(reg) && reg.bindings.contains_key(v)
    }

    pub open spec fn all_versions_bound(&self) -> bool {
        &&& self.has_version_registry()
        &&& forall|v: VersionId| #[trigger]
            self.frames_spec().versions_spec()@.contains_key(v) ==> self.has_version_binding(v)
    }

    pub fn verify_version_binding(
        &self,
        version: VersionId,
        Tracked(binding): Tracked<&VersionBinding>,
    )
        requires
            self.well_formed(),
            self.all_versions_bound(),
            binding.instance_id() == cluster_version_instance_id(),
            binding.key() == (version.origin_node as NodeId, version.seq),
        ensures
            !self.frames_spec().versions_spec()@.contains_key(version)
                || self.frames_spec().versions_spec()@[version] == binding.value(),
    {
        proof {
            if self.frames.versions_spec()@.contains_key(version) {
                let tracked reg = self.version_registry.borrow().tracked_borrow();
                let tracked existing = reg.bindings.tracked_borrow(version);
                version_binding_agree(existing, binding);
            }
        }
    }

    pub open spec fn version_registry_valid_for(&self, frames: &FrameTable) -> bool {
        match self.version_registry_spec()@ {
            None => true,
            Some(reg) => {
                &&& reg.instance.id() == cluster_version_instance_id()
                &&& reg.source.well_formed()
                &&& reg.source.id() == reg.instance.id()
                &&& reg.source.node() == self.node()
                &&& reg.source.min_seq() <= self.next_version_seq_spec() as nat
                &&& reg.source.max_seq() matches Some(lim) ==> lim
                    >= self.version_seq_limit_spec() as nat
                &&& forall|v: VersionId| #[trigger]
                    reg.bindings.contains_key(v) ==> {
                        &&& reg.bindings[v].instance_id() == reg.instance.id()
                        &&& reg.bindings[v].key() == (v.origin_node as NodeId, v.seq)
                        &&& frames.versions_spec()@.contains_key(v)
                        &&& reg.bindings[v].value() == frames.versions_spec()@[v]
                    }
            },
        }
    }

    pub open spec fn version_registry_valid(&self) -> bool {
        self.version_registry_valid_for(&self.frames_spec())
    }

    pub open spec fn well_formed(&self) -> bool {
        &&& self.tiers_valid()
        &&& self.frames_spec().consistent(self.kernel_view())
        &&& self.catalog_spec().well_formed(&self.frames_spec())
        &&& Self::certified(&self.frames_spec(), self.kernel_view())
        &&& complete_receipts(self.ledger_spec()@)
        &&& self.version_range_valid()
        &&& self.leases_valid()
        &&& self.version_registry_valid()
        &&& ServiceSpec::valid_versions(
            self.frames_spec().frame_size_spec() as nat,
            self.frames_spec().versions_spec()@,
        )
        &&& self.spec_spec()@ == Self::service(&self.frames_spec(), &self.catalog_spec())
        &&& self.spec_spec()@.well_formed()
    }

    pub open spec fn well_formed_with_pending_revocation(
        &self,
        pending: PendingRevocation,
    ) -> bool {
        &&& self.tiers_valid()
        &&& pending.snapshot@.consistent(self.kernel_view())
        &&& self.frames_spec().replaced_token(pending.snapshot@, pending.frame)
        &&& pending.frame.0 < pending.snapshot@.metas_spec().len()
        &&& pending.snapshot@.meta(pending.frame).active_fenced_leases > 0
        &&& pending.snapshot@.frame(pending.frame).has_fenced_share(pending.share@)
        &&& self.catalog_spec().well_formed(&pending.snapshot@)
        &&& Self::certified(&pending.snapshot@, self.kernel_view())
        &&& complete_receipts(self.ledger_spec()@)
        &&& self.version_range_valid_for(&pending.snapshot@)
        &&& self.leases_valid_for(&pending.snapshot@)
        &&& self.version_registry_valid_for(&pending.snapshot@)
        &&& ServiceSpec::valid_versions(
            pending.snapshot@.frame_size_spec() as nat,
            pending.snapshot@.versions_spec()@,
        )
        &&& self.spec_spec()@ == Self::service(&pending.snapshot@, &self.catalog_spec())
        &&& self.spec_spec()@.well_formed()
    }

    pub fn new(
        node_id: u16,
        frame_size: usize,
        tiers: Vec<TierDescriptor>,
        max_leases: usize,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (core: Self)
        requires
            0 < frame_size,
            tiers.len() <= MAX_TIERS,
            forall|i: int| 0 <= i < tiers.len() ==> (#[trigger] tiers[i]).id.0 as usize == i,
            max_leases <= u32::MAX,
        ensures
            core.well_formed(),
            core.node_id_spec() == node_id,
            core.instance_id() == instance.id(),
            core.frames_spec().frame_size_spec() == frame_size,
            core.tiers_spec()@ == tiers@,
            core.frames_spec().metas_spec().len() == 0,
            core.timelines_spec().count_spec() == 0,
            core.version_seq_start_spec() == 1,
            core.next_version_seq_spec() == 1,
            core.version_seq_limit_spec() == MAX_VERSION_SEQ + 1,
            core.active_leases_spec().len() == max_leases,
            core.free_lease_slots_spec().len() == max_leases,
            core.spec_spec()@.versions == Map::empty(),
            core.spec_spec()@.catalog == Map::empty(),
            !core.has_version_registry(),
    {
        Self::new_with_version_range(
            node_id,
            frame_size,
            tiers,
            max_leases,
            1,
            MAX_VERSION_SEQ + 1,
            Tracked(instance),
        )
    }

    pub fn new_with_version_range(
        node_id: u16,
        frame_size: usize,
        tiers: Vec<TierDescriptor>,
        max_leases: usize,
        version_seq_start: u64,
        version_seq_limit: u64,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (core: Self)
        requires
            0 < frame_size,
            tiers.len() <= MAX_TIERS,
            forall|i: int| 0 <= i < tiers.len() ==> (#[trigger] tiers[i]).id.0 as usize == i,
            max_leases <= u32::MAX,
            1 <= version_seq_start <= version_seq_limit <= MAX_VERSION_SEQ + 1,
        ensures
            core.well_formed(),
            core.node_id_spec() == node_id,
            core.instance_id() == instance.id(),
            core.frames_spec().frame_size_spec() == frame_size,
            core.tiers_spec()@ == tiers@,
            core.frames_spec().metas_spec().len() == 0,
            core.timelines_spec().count_spec() == 0,
            core.version_seq_start_spec() == version_seq_start,
            core.next_version_seq_spec() == version_seq_start,
            core.version_seq_limit_spec() == version_seq_limit,
            core.active_leases_spec().len() == max_leases,
            core.free_lease_slots_spec().len() == max_leases,
            core.spec_spec()@.versions == Map::empty(),
            core.spec_spec()@.catalog == Map::empty(),
            !core.has_version_registry(),
    {
        Self::new_with_version_range_and_buckets(
            node_id,
            frame_size,
            tiers,
            max_leases,
            version_seq_start,
            version_seq_limit,
            DEFAULT_BUCKETS,
            Tracked(instance),
        )
    }

    pub fn new_for_capacity(
        node_id: u16,
        frame_size: usize,
        tiers: Vec<TierDescriptor>,
        max_leases: usize,
        total_frames: usize,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (core: Self)
        requires
            0 < frame_size,
            tiers.len() <= MAX_TIERS,
            forall|i: int| 0 <= i < tiers.len() ==> (#[trigger] tiers[i]).id.0 as usize == i,
            max_leases <= u32::MAX,
        ensures
            core.well_formed(),
            core.node_id_spec() == node_id,
            core.instance_id() == instance.id(),
            core.frames_spec().frame_size_spec() == frame_size,
            core.tiers_spec()@ == tiers@,
            core.frames_spec().metas_spec().len() == 0,
            core.timelines_spec().count_spec() == 0,
            core.version_seq_start_spec() == 1,
            core.next_version_seq_spec() == 1,
            core.version_seq_limit_spec() == MAX_VERSION_SEQ + 1,
            core.active_leases_spec().len() == max_leases,
            core.free_lease_slots_spec().len() == max_leases,
            DEFAULT_BUCKETS <= core.catalog_spec().num_buckets() <= MAX_RECOMMENDED_BUCKETS,
            core.catalog_spec().num_buckets() >= vstd::math::min(
                2 * total_frames as int,
                MAX_RECOMMENDED_BUCKETS as int,
            ),
            vstd::arithmetic::power2::is_pow2(core.catalog_spec().num_buckets() as int),
            core.spec_spec()@.versions == Map::empty(),
            core.spec_spec()@.catalog == Map::empty(),
            !core.has_version_registry(),
    {
        Self::new_with_version_range_and_buckets(
            node_id,
            frame_size,
            tiers,
            max_leases,
            1,
            MAX_VERSION_SEQ + 1,
            Catalog::recommended_buckets(total_frames),
            Tracked(instance),
        )
    }

    pub fn new_with_version_range_and_buckets(
        node_id: u16,
        frame_size: usize,
        tiers: Vec<TierDescriptor>,
        max_leases: usize,
        version_seq_start: u64,
        version_seq_limit: u64,
        num_buckets: usize,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (core: Self)
        requires
            0 < frame_size,
            tiers.len() <= MAX_TIERS,
            forall|i: int| 0 <= i < tiers.len() ==> (#[trigger] tiers[i]).id.0 as usize == i,
            max_leases <= u32::MAX,
            1 <= version_seq_start <= version_seq_limit <= MAX_VERSION_SEQ + 1,
            num_buckets > 0,
        ensures
            core.well_formed(),
            core.node_id_spec() == node_id,
            core.instance_id() == instance.id(),
            core.frames_spec().frame_size_spec() == frame_size,
            core.tiers_spec()@ == tiers@,
            core.frames_spec().metas_spec().len() == 0,
            core.timelines_spec().count_spec() == 0,
            core.version_seq_start_spec() == version_seq_start,
            core.next_version_seq_spec() == version_seq_start,
            core.version_seq_limit_spec() == version_seq_limit,
            core.active_leases_spec().len() == max_leases,
            core.free_lease_slots_spec().len() == max_leases,
            core.catalog_spec().num_buckets() == num_buckets,
            core.spec_spec()@.versions == Map::empty(),
            core.spec_spec()@.catalog == Map::empty(),
            !core.has_version_registry(),
    {
        let num_tiers = tiers.len();
        let timelines = TimelineTable::new();
        let frames = FrameTable::new(num_tiers, frame_size);
        let catalog = Catalog::with_capacity(num_tiers, num_buckets);
        let tracked ledger = KnowledgeLedger::empty(instance.id());
        let mut active_leases = Vec::<Option<ActiveLocalLease>>::with_capacity(max_leases);
        let mut free_lease_slots = Vec::<u32>::with_capacity(max_leases);
        let mut i: usize = 0;
        while i < max_leases
            invariant
                i <= max_leases <= u32::MAX,
                active_leases.len() == i,
                free_lease_slots.len() == i,
                forall|k: int| 0 <= k < i ==> (#[trigger] active_leases[k]).is_none(),
                forall|k: int|
                    0 <= k < i ==> (#[trigger] free_lease_slots[k]) == (max_leases - 1 - k) as u32,
            decreases max_leases - i,
        {
            active_leases.push(None);
            free_lease_slots.push((max_leases - 1 - i) as u32);
            i += 1;
        }
        let ghost spec = Self::service(&frames, &catalog);
        let core = Self {
            runtime_id: allocate_core_id(),
            version_registry_enabled: false,
            node_id,
            tiers,
            timelines,
            frames,
            catalog,
            ledger: Tracked(ledger),
            version_seq_start,
            next_version_seq: version_seq_start,
            version_seq_limit,
            active_leases,
            free_lease_slots,
            lease_generation: 0,
            spec: Ghost(spec),
            version_registry: Tracked(None),
        };
        proof {
            assert(core.ledger@.valid(instance.id(), core.node()));
            assert(core.spec@.catalog =~= Map::empty());
            assert forall|s: int|
                0 <= s < max_leases implies #[trigger] core.active_leases[s].is_none()
                && core.free_lease_slots@.contains(s as u32) by {
                let idx = max_leases - 1 - s;
                assert(core.free_lease_slots[idx] == s as u32);
            }
            assert(core.free_lease_slots@.no_duplicates());
            assert(core.leases_valid());
            assert(core.frames.consistent(core.kernel_view()));
            assert(core.catalog.well_formed(&core.frames));
            assert(Self::certified(&core.frames, core.kernel_view()));
            assert(complete_receipts(core.ledger@));
            assert(core.version_registry_valid());
            Self::lemma_service_well_formed(&core.frames, &core.catalog, core.kernel_view());
        }
        core
    }

    pub fn from_shard_versions(
        node_id: u16,
        frame_size: usize,
        tiers: Vec<TierDescriptor>,
        max_leases: usize,
        version_seq_start: u64,
        version_seq_limit: u64,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(ver_inst): Tracked<&VersionInstance>,
        Tracked(ver_src): Tracked<VersionSource>,
    ) -> (core: Self)
        requires
            0 < frame_size,
            tiers.len() <= MAX_TIERS,
            forall|i: int| 0 <= i < tiers.len() ==> (#[trigger] tiers[i]).id.0 as usize == i,
            max_leases <= u32::MAX,
            1 <= version_seq_start <= version_seq_limit <= MAX_VERSION_SEQ + 1,
            ver_inst.id() == cluster_version_instance_id(),
            ver_src.well_formed(),
            ver_src.id() == ver_inst.id(),
            ver_src.node() == node_id as NodeId,
            ver_src.min_seq() <= version_seq_start as nat,
            ver_src.max_seq() matches Some(lim) ==> lim >= version_seq_limit as nat,
        ensures
            core.well_formed(),
            core.has_version_registry(),
            core.node_id_spec() == node_id,
            core.instance_id() == instance.id(),
            core.frames_spec().frame_size_spec() == frame_size,
            core.tiers_spec()@ == tiers@,
            core.frames_spec().metas_spec().len() == 0,
            core.timelines_spec().count_spec() == 0,
            core.version_seq_start_spec() == version_seq_start,
            core.next_version_seq_spec() == version_seq_start,
            core.version_seq_limit_spec() == version_seq_limit,
            core.active_leases_spec().len() == max_leases,
            core.free_lease_slots_spec().len() == max_leases,
            core.spec_spec()@.versions == Map::empty(),
            core.spec_spec()@.catalog == Map::empty(),
    {
        Self::from_shard_versions_with_buckets(
            node_id,
            frame_size,
            tiers,
            max_leases,
            version_seq_start,
            version_seq_limit,
            DEFAULT_BUCKETS,
            Tracked(instance),
            Tracked(ver_inst),
            Tracked(ver_src),
        )
    }

    pub fn from_shard_versions_with_buckets(
        node_id: u16,
        frame_size: usize,
        tiers: Vec<TierDescriptor>,
        max_leases: usize,
        version_seq_start: u64,
        version_seq_limit: u64,
        num_buckets: usize,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(ver_inst): Tracked<&VersionInstance>,
        Tracked(ver_src): Tracked<VersionSource>,
    ) -> (core: Self)
        requires
            0 < frame_size,
            tiers.len() <= MAX_TIERS,
            forall|i: int| 0 <= i < tiers.len() ==> (#[trigger] tiers[i]).id.0 as usize == i,
            max_leases <= u32::MAX,
            1 <= version_seq_start <= version_seq_limit <= MAX_VERSION_SEQ + 1,
            num_buckets > 0,
            ver_inst.id() == cluster_version_instance_id(),
            ver_src.well_formed(),
            ver_src.id() == ver_inst.id(),
            ver_src.node() == node_id as NodeId,
            ver_src.min_seq() <= version_seq_start as nat,
            ver_src.max_seq() matches Some(lim) ==> lim >= version_seq_limit as nat,
        ensures
            core.well_formed(),
            core.has_version_registry(),
            core.node_id_spec() == node_id,
            core.instance_id() == instance.id(),
            core.frames_spec().frame_size_spec() == frame_size,
            core.tiers_spec()@ == tiers@,
            core.frames_spec().metas_spec().len() == 0,
            core.timelines_spec().count_spec() == 0,
            core.version_seq_start_spec() == version_seq_start,
            core.next_version_seq_spec() == version_seq_start,
            core.version_seq_limit_spec() == version_seq_limit,
            core.active_leases_spec().len() == max_leases,
            core.free_lease_slots_spec().len() == max_leases,
            core.catalog_spec().num_buckets() == num_buckets,
            core.spec_spec()@.versions == Map::empty(),
            core.spec_spec()@.catalog == Map::empty(),
    {
        let mut core = Self::new_with_version_range_and_buckets(
            node_id,
            frame_size,
            tiers,
            max_leases,
            version_seq_start,
            version_seq_limit,
            num_buckets,
            Tracked(instance),
        );
        let tracked dup_inst = ver_inst.duplicate();
        core.enable_version_registry(Tracked(dup_inst), Tracked(ver_src));
        core
    }

    /// Attach an optional cluster-wide write-once version registry to this `CacheCore`.
    pub fn enable_version_registry(
        &mut self,
        Tracked(version_instance): Tracked<VersionInstance>,
        Tracked(version_source): Tracked<VersionSource>,
    )
        requires
            old(self).well_formed(),
            !old(self).has_version_registry(),
            version_instance.id() == cluster_version_instance_id(),
            version_source.well_formed(),
            version_source.id() == version_instance.id(),
            version_source.node() == old(self).node(),
            version_source.min_seq() <= old(self).next_version_seq_spec() as nat,
            version_source.max_seq() matches Some(lim) ==> lim >= old(
                self,
            ).version_seq_limit_spec() as nat,
        ensures
            final(self).well_formed(),
            final(self).has_version_registry(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
    {
        self.version_registry_enabled = true;
        proof {
            self.version_registry = Tracked(
                Some(
                    VersionRegistry {
                        instance: version_instance,
                        source: version_source,
                        bindings: Map::tracked_empty(),
                    },
                ),
            );
        }
    }

    /// Mint and store a cluster-wide `VersionBinding` for a locally originated version.
    pub fn bind_local_version(&mut self, version: VersionId) -> (r: Tracked<VersionBinding>)
        requires
            old(self).well_formed(),
            old(self).has_version_registry(),
            old(self).frames_spec().versions_spec()@.contains_key(version),
            version.origin_node == old(self).node_id_spec(),
            version.seq as nat >= old(self).version_registry_spec()@->Some_0.source.min_seq(),
            version.seq < old(self).next_version_seq_spec(),
        ensures
            final(self).well_formed(),
            final(self).has_version_registry(),
            final(self).has_version_binding(version),
            forall|v: VersionId|
                old(self).has_version_binding(v) ==> #[trigger] final(self).has_version_binding(v),
            r@.instance_id() == cluster_version_instance_id(),
            r@.key() == (version.origin_node as NodeId, version.seq),
            r@.value() == old(self).frames_spec().versions_spec()@[version],
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
    {
        let tracked out;
        proof {
            let tracked mut reg_opt: Option<VersionRegistry> = None;
            vstd::modes::tracked_swap(self.version_registry.borrow_mut(), &mut reg_opt);
            let tracked mut reg = reg_opt.tracked_unwrap();
            let tracked unused = reg.source.take_seq(version.seq);
            let tracked binding = reg.instance.bind(
                (version.origin_node as NodeId, version.seq),
                self.frames.versions_spec()@[version],
                unused,
            );
            out = version_binding_duplicate(&binding);
            reg.bindings.tracked_insert(version, binding);
            self.version_registry = Tracked(Some(reg));
        }
        Tracked(out)
    }

    /// Mint a cluster-wide `VersionBinding` for an in-flight locally originated version
    /// (`version.seq < self.next_version_seq`) before the frame is published into `self.frames.versions`.
    pub fn prebind_local_version(&mut self, version: VersionId, Ghost(bytes): Ghost<Seq<u8>>) -> (r:
        Tracked<VersionBinding>)
        requires
            old(self).well_formed(),
            old(self).has_version_registry(),
            version.origin_node == old(self).node_id_spec(),
            version.seq as nat >= old(self).version_registry_spec()@->Some_0.source.min_seq(),
            version.seq < old(self).next_version_seq_spec(),
        ensures
            final(self).well_formed(),
            final(self).has_version_registry(),
            forall|v: VersionId|
                old(self).has_version_binding(v) <==> #[trigger] final(self).has_version_binding(v),
            r@.instance_id() == cluster_version_instance_id(),
            r@.key() == (version.origin_node as NodeId, version.seq),
            r@.value() == bytes,
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
    {
        let tracked out;
        proof {
            let tracked mut reg_opt: Option<VersionRegistry> = None;
            vstd::modes::tracked_swap(self.version_registry.borrow_mut(), &mut reg_opt);
            let tracked mut reg = reg_opt.tracked_unwrap();
            let tracked unused = reg.source.take_seq(version.seq);
            out = reg.instance.bind((version.origin_node as NodeId, version.seq), bytes, unused);
            self.version_registry = Tracked(Some(reg));
        }
        Tracked(out)
    }

    /// Register a peer-supplied `VersionBinding` for a version already recorded in `self.frames.versions`.
    pub fn register_version_binding(
        &mut self,
        version: VersionId,
        Tracked(binding): Tracked<VersionBinding>,
    )
        requires
            old(self).well_formed(),
            old(self).has_version_registry(),
            old(self).frames_spec().versions_spec()@.contains_key(version),
            binding.instance_id() == cluster_version_instance_id(),
            binding.key() == (version.origin_node as NodeId, version.seq),
            binding.value() == old(self).frames_spec().versions_spec()@[version],
        ensures
            final(self).well_formed(),
            final(self).has_version_registry(),
            final(self).has_version_binding(version),
            forall|v: VersionId|
                old(self).has_version_binding(v) ==> #[trigger] final(self).has_version_binding(v),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
    {
        proof {
            let tracked mut reg_opt: Option<VersionRegistry> = None;
            vstd::modes::tracked_swap(self.version_registry.borrow_mut(), &mut reg_opt);
            let tracked mut reg = reg_opt.tracked_unwrap();
            reg.bindings.tracked_insert(version, binding);
            self.version_registry = Tracked(Some(reg));
        }
    }

    /// Register an optional peer-supplied `VersionBinding` if the version registry is enabled.
    pub fn try_register_version_binding(
        &mut self,
        version: VersionId,
        Tracked(binding_opt): Tracked<Option<VersionBinding>>,
    )
        requires
            old(self).well_formed(),
            old(self).frames_spec().versions_spec()@.contains_key(version),
            binding_opt matches Some(binding) ==> binding.instance_id()
                == cluster_version_instance_id() && binding.key() == (
                version.origin_node as NodeId,
                version.seq,
            ) && binding.value() == old(self).frames_spec().versions_spec()@[version],
        ensures
            final(self).well_formed(),
            final(self).has_version_registry() == old(self).has_version_registry(),
            old(self).has_version_registry() && binding_opt is Some
                ==> final(self).has_version_binding(version),
            forall|v: VersionId| #[trigger]
                old(self).has_version_binding(v) ==> final(self).has_version_binding(v),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
    {
        proof {
            if let Some(binding) = binding_opt {
                if self.version_registry@ is Some {
                    let tracked mut reg_opt: Option<VersionRegistry> = None;
                    vstd::modes::tracked_swap(self.version_registry.borrow_mut(), &mut reg_opt);
                    let tracked mut reg = reg_opt.tracked_unwrap();
                    reg.bindings.tracked_insert(version, binding);
                    self.version_registry = Tracked(Some(reg));
                }
            }
        }
    }

    /// Retrieve a duplicated `VersionBinding` for `version` if registered.
    pub fn version_binding_for(&self, version: VersionId) -> (r: Tracked<Option<VersionBinding>>)
        requires
            self.well_formed(),
        ensures
            self.has_version_binding(version) ==> (r@ matches Some(b) && b.instance_id()
                == cluster_version_instance_id() && b.key() == (
                version.origin_node as NodeId,
                version.seq,
            ) && self.frames_spec().versions_spec()@.contains_key(version) && b.value()
                == self.frames_spec().versions_spec()@[version]),
            !self.has_version_binding(version) ==> r@ is None,
    {
        let tracked out = match self.version_registry.borrow() {
            Some(reg) => if reg.bindings.contains_key(version) {
                Some(version_binding_duplicate(reg.bindings.tracked_borrow(version)))
            } else {
                None
            },
            None => None,
        };
        Tracked(out)
    }

    pub fn register_timeline(&mut self, kind: TimelineKind) -> (r: Result<TimelineId, CoreError>)
        requires
            old(self).well_formed(),
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).frames_spec() == old(self).frames_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).timelines_spec().count_spec() >= old(self).timelines_spec().count_spec(),
            final(self).timelines_spec().events_spec() == old(self).timelines_spec().events_spec(),
            forall|t: TimelineId|
                t < old(self).timelines_spec().count_spec() ==> {
                    &&& (#[trigger] final(self).timelines_spec().kinds_spec()[t as int]) == old(
                        self,
                    ).timelines_spec().kinds_spec()[t as int]
                    &&& final(self).timelines_spec().next_seq_spec()[t as int] == old(
                        self,
                    ).timelines_spec().next_seq_spec()[t as int]
                    &&& final(self).timelines_spec().confirmed_seq_spec()[t as int] == old(
                        self,
                    ).timelines_spec().confirmed_seq_spec()[t as int]
                },
            match r {
                Err(_) => *final(self) == *old(self) && old(self).timelines_spec().count_spec()
                    == MAX_TIMELINES,
                Ok(t) => t == old(self).timelines_spec().count_spec()
                    && final(self).timelines_spec().count_spec() == t + 1
                    && final(self).timelines_spec().kinds_spec()[t as int] == kind
                    && final(self).timelines_spec().next_seq_spec()[t as int] == 1
                    && final(self).timelines_spec().confirmed_seq_spec()[t as int] == 0
                    && final(self).kernel_view().extends(old(self).kernel_view()),
            },
    {
        let ghost before = *self;
        let t = self.timelines.register_timeline(kind)?;
        proof {
            assert forall|i: int|
                0 <= i < before.timelines.count implies #[trigger] self.timelines.confirmed_seq[i]
                >= before.timelines.confirmed_seq[i] && self.timelines.next_seq[i]
                >= before.timelines.next_seq[i] && self.timelines.kinds[i]
                == before.timelines.kinds[i] by {
                assert(self.timelines.kinds[i] == before.timelines.kinds[i]);
            }
            assert forall|u: TimelineId, s: u64|
                self.timelines.issued(
                    u,
                    s,
                ) implies #[trigger] self.timelines.events_spec()@.contains_key((u, s))
                && self.ledger@.submitted.dom().contains(self.timelines.events_spec()@[(u, s)]) by {
                if u < before.timelines.count {
                    assert(self.timelines.kinds[u as int] == before.timelines.kinds[u as int]);
                    assert(self.timelines.next_seq[u as int]
                        == before.timelines.next_seq[u as int]);
                    assert(before.timelines.issued(u, s));
                    assert(before.timelines.events_spec()@.contains_key((u, s)));
                } else {
                    assert(u == t);
                    assert(self.timelines.next_seq[t as int] == 1);
                }
            }
            assert forall|u: TimelineId, s: u64|
                u < self.timelines.count && 1 <= s
                    <= self.timelines.confirmed_seq[u as int] implies #[trigger] self.timelines.observed(
            u, s, self.node(), self.ledger@.knows) by {
                if u < before.timelines.count {
                    assert(self.timelines.kinds[u as int] == before.timelines.kinds[u as int]);
                    assert(before.timelines.observed(u, s, before.node(), before.ledger@.knows));
                } else {
                    assert(u == t);
                    assert(self.timelines.confirmed_seq[t as int] == 0);
                }
            }
            assert(self.kernel_view().extends(before.kernel_view()));
            self.preserve_after_ledger_growth(before);
        }
        Ok(t)
    }

    pub fn register_tier_slab(
        &mut self,
        tier: TierId,
        slots: Ghost<Seq<SlotRef>>,
        num_frames: u32,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    ) -> (r: Result<FrameId, CoreError>)
        requires
            old(self).well_formed(),
            slots@.len() == num_frames,
            slab_slots_ready(
                tokens,
                slots@,
                0,
                num_frames as usize,
                old(self).instance_id(),
                old(self).node(),
                if tier.valid(old(self).tiers_spec().len()) {
                    Some(old(self).frames_spec().tier_frame_size(tier))
                } else {
                    None
                },
            ),
            forall|i: int, j: int|
                0 <= i < j < num_frames ==> (#[trigger] slots@[i]).loc != (
                #[trigger] slots@[j]).loc,
            forall|i: nat, j: int|
                i < old(self).frames_spec().metas_spec().len() && 0 <= j < num_frames ==> (
                #[trigger] old(self).frames_spec().ghosts_spec()@[i].slot).loc != (
                #[trigger] slots@[j]).loc,
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).tiers_spec() == old(self).tiers_spec(),
            final(self).frames_spec().frame_size_spec() == old(
                self,
            ).frames_spec().frame_size_spec(),
            final(self).frames_spec().tier_frame_sizes_spec() == old(
                self,
            ).frames_spec().tier_frame_sizes_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).catalog_spec() == old(self).catalog_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            r.is_ok() == (tier.valid(old(self).tiers_spec().len()) && num_frames > 0 && old(
                self,
            ).frames_spec().metas_spec().len() + num_frames <= u32::MAX),
            match r {
                Err(_) => *final(self) == *old(self),
                Ok(first) => {
                    &&& tier.valid(old(self).tiers_spec().len())
                    &&& num_frames > 0
                    &&& first.0 == old(self).frames_spec().metas_spec().len()
                    &&& final(self).frames_spec().metas_spec().len() == old(
                        self,
                    ).frames_spec().metas_spec().len() + num_frames
                    &&& final(self).frames_spec().ready_queue_spec()[tier.index() as int].len()
                        == old(self).frames_spec().ready_queue_spec()[tier.index() as int].len()
                        + num_frames
                    &&& forall|j: nat|
                        j < old(self).frames_spec().metas_spec().len()
                            ==> #[trigger] final(self).frames_spec().ghosts_spec()@[j] == old(
                            self,
                        ).frames_spec().ghosts_spec()@[j]
                            && final(self).frames_spec().metas_spec()[j as int] == old(
                            self,
                        ).frames_spec().metas_spec()[j as int]
                    &&& forall|j: int|
                        0 <= j < num_frames ==> {
                            (#[trigger] final(self).frames_spec().frame(
                                FrameId((first.0 + j) as u32),
                            )).tier == tier && final(self).frames_spec().frame(
                                FrameId((first.0 + j) as u32),
                            ).slot == slots@[j] && final(self).frames_spec().meta(
                                FrameId((first.0 + j) as u32),
                            ).state == STATE_EMPTY && final(self).frames_spec().meta(
                                FrameId((first.0 + j) as u32),
                            ).queued()
                        }
                },
            },
    {
        if tier.as_index() >= self.tiers.len() {
            return Err(CoreError::UnsupportedTier);
        }
        if num_frames == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let count = num_frames as usize;
        if count > (u32::MAX as usize) - self.frames.metas.len() {
            return Err(CoreError::CapacityExhausted);
        }
        let first = self.frames.metas.len() as u32;
        let ghost node = self.node();
        self.frames.register_slab(
            tier,
            slots,
            Tracked(tokens),
            count,
            &self.timelines,
            Tracked(self.ledger.borrow()),
            Ghost(node),
        );
        proof {
            assert forall|j: nat|
                j < old(self).frames.metas.len() implies #[trigger] self.frames.ghosts_spec()@[j]
                == old(self).frames.ghosts_spec()@[j] && self.frames.metas[j as int] == old(
                self,
            ).frames.metas[j as int] by {
                assert(self.frames.ghosts_spec()@[j] == old(self).frames.ghosts_spec()@[j]);
            }
            self.preserve_after_slab_append(*old(self), tier, first, slots@, num_frames);
        }
        Ok(FrameId(first))
    }

    /// Dynamically register a new storage tier with an explicit per-tier frame size and initial slab.
    pub fn register_tier_with_frame_size(
        &mut self,
        desc: TierDescriptor,
        tier_frame_size: usize,
        slots: Ghost<Seq<SlotRef>>,
        num_frames: u32,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    ) -> (r: Result<(TierId, FrameId), CoreError>)
        requires
            old(self).well_formed(),
            0 < tier_frame_size,
            slots@.len() == num_frames,
            slab_slots_ready(
                tokens,
                slots@,
                0,
                num_frames as usize,
                old(self).instance_id(),
                old(self).node(),
                Some(tier_frame_size),
            ),
            forall|i: int, j: int|
                0 <= i < j < num_frames ==> (#[trigger] slots@[i]).loc != (
                #[trigger] slots@[j]).loc,
            forall|i: nat, j: int|
                i < old(self).frames_spec().metas_spec().len() && 0 <= j < num_frames ==> (
                #[trigger] old(self).frames_spec().ghosts_spec()@[i].slot).loc != (
                #[trigger] slots@[j]).loc,
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).frames_spec().frame_size_spec() == old(
                self,
            ).frames_spec().frame_size_spec(),
            r.is_ok() == (old(self).tiers_spec().len() < MAX_TIERS && num_frames > 0 && old(
                self,
            ).frames_spec().metas_spec().len() + num_frames <= u32::MAX),
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((tier, first)) => {
                    &&& old(self).tiers_spec().len() < MAX_TIERS
                    &&& num_frames > 0
                    &&& tier == TierId(old(self).tiers_spec().len() as u16)
                    &&& final(self).tiers_spec().len() == old(self).tiers_spec().len() + 1
                    &&& final(self).tiers_spec()[tier.index() as int] == (TierDescriptor {
                        id: tier,
                        ..desc
                    })
                    &&& final(self).frames_spec().tier_frame_sizes_spec()@ == old(
                        self,
                    ).frames_spec().tier_frame_sizes_spec()@.push(tier_frame_size)
                    &&& first.0 == old(self).frames_spec().metas_spec().len()
                    &&& final(self).frames_spec().metas_spec().len() == old(
                        self,
                    ).frames_spec().metas_spec().len() + num_frames
                    &&& final(self).frames_spec().ready_queue_spec()[tier.index() as int].len()
                        == num_frames
                    &&& forall|j: nat|
                        j < old(self).frames_spec().metas_spec().len()
                            ==> #[trigger] final(self).frames_spec().ghosts_spec()@[j] == old(
                            self,
                        ).frames_spec().ghosts_spec()@[j]
                            && final(self).frames_spec().metas_spec()[j as int] == old(
                            self,
                        ).frames_spec().metas_spec()[j as int]
                    &&& forall|j: int|
                        0 <= j < num_frames ==> {
                            (#[trigger] final(self).frames_spec().frame(
                                FrameId((first.0 + j) as u32),
                            )).tier == tier && final(self).frames_spec().frame(
                                FrameId((first.0 + j) as u32),
                            ).slot == slots@[j] && final(self).frames_spec().meta(
                                FrameId((first.0 + j) as u32),
                            ).state == STATE_EMPTY && final(self).frames_spec().meta(
                                FrameId((first.0 + j) as u32),
                            ).queued()
                        }
                },
            },
    {
        if self.tiers.len() >= MAX_TIERS {
            return Err(CoreError::CapacityExhausted);
        }
        if num_frames == 0 {
            return Err(CoreError::InvalidArgument);
        }
        let count = num_frames as usize;
        if count > (u32::MAX as usize) - self.frames.metas.len() {
            return Err(CoreError::CapacityExhausted);
        }
        let ghost v = self.kernel_view();
        let tier = self.frames.add_tier_with_size(tier_frame_size, Ghost(v));
        self.catalog.add_tier();
        let mut stored = desc;
        stored.id = tier;
        self.tiers.push(stored);
        proof {
            assert forall|k: ObjectKey, t: int| 0 <= t < self.catalog.num_tiers() implies (
            #[trigger] self.catalog.slot(k, t) matches Some(f) ==> {
                &&& f.0 < self.frames.metas.len()
                &&& self.frames.meta(f).state == STATE_PUBLISHED
                &&& self.frames.frame(f).tier.index() == t
                &&& self.frames.frame(f).bound_key == Some(k)
                &&& self.frames.frame(f).bound_version == Some(self.catalog.view()[k].version)
            }) by {
                if t == old(self).catalog.num_tiers() {
                    assert(old(self).catalog.slot(k, t).is_none());
                } else {
                    assert(self.catalog.slot(k, t) == old(self).catalog.slot(k, t));
                }
            }
            assert forall|k: ObjectKey, t: int| self.catalog.num_tiers() <= t < MAX_TIERS implies (
            #[trigger] self.catalog.slot(k, t)).is_none() by {
                assert(old(self).catalog.slot(k, t).is_none());
            }
            assert forall|f: FrameId|
                f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                    == STATE_PUBLISHED implies {
                let g = self.frames.frame(f);
                &&& g.bound_key.is_some()
                &&& g.tier.valid(self.catalog.num_tiers())
                &&& self.catalog.slot(g.bound_key.unwrap(), g.tier.index() as int) == Some(f)
            } by {
                assert(old(self).frames.meta(f).state == STATE_PUBLISHED);
            }
            assert forall|k| self.catalog.view().contains_key(k) implies exists|t: int|
                0 <= t < self.catalog.num_tiers() && #[trigger] self.catalog.slot(
                    k,
                    t,
                ).is_some() by {
                let t = choose|t: int|
                    0 <= t < old(self).catalog.num_tiers() && #[trigger] old(self).catalog.slot(
                        k,
                        t,
                    ).is_some();
                assert(self.catalog.slot(k, t).is_some());
            }
            assert(self.catalog.well_formed(&self.frames));
            assert(self.tiers_valid());
            assert(stored == (TierDescriptor { id: tier, ..desc }));
            assert forall|f: FrameId|
                f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                    == STATE_PUBLISHED implies {
                let g = self.frames.frame(f);
                g.cert matches Some(cert) && {
                    &&& replica_valid(self.ledger@, self.node(), cert)
                    &&& event_succeeded(self.ledger@, self.node(), cert.writer)
                    &&& self.ledger@.citations(self.node()).contains(cert.writer)
                    &&& g.bound_version matches Some(ver)
                    &&& self.frames.versions_spec()@.contains_key(ver)
                    &&& self.frames.versions_spec()@[ver] == cert.bytes
                }
            } by {
                assert(old(self).frames.meta(f).state == STATE_PUBLISHED);
            }
            assert(self.version_range_valid());
            assert(self.leases_valid());
            assert(Self::service(&self.frames, &self.catalog) == old(self).spec@);
            assert(self.well_formed());
        }
        let first = self.register_tier_slab(tier, slots, num_frames, Tracked(tokens)).unwrap();
        proof {
            assert forall|j: nat| j < old(self).frames.metas.len() implies {
                &&& (#[trigger] self.frames.ghosts_spec()@[j]) == old(self).frames.ghosts_spec()@[j]
                &&& self.frames.metas[j as int] == old(self).frames.metas[j as int]
            } by {
                assert(self.frames_spec().ghosts_spec()@[j] == old(
                    self,
                ).frames_spec().ghosts_spec()@[j]);
            }
        }

        Ok((tier, first))
    }

    /// Dynamically register a new storage tier using the node default `frame_size`.
    pub fn register_tier(
        &mut self,
        desc: TierDescriptor,
        slots: Ghost<Seq<SlotRef>>,
        num_frames: u32,
        Tracked(tokens): Tracked<Map<usize, SlotToken>>,
    ) -> (r: Result<(TierId, FrameId), CoreError>)
        requires
            old(self).well_formed(),
            slots@.len() == num_frames,
            slab_slots_ready(
                tokens,
                slots@,
                0,
                num_frames as usize,
                old(self).instance_id(),
                old(self).node(),
                Some(old(self).frames_spec().frame_size_spec()),
            ),
            forall|i: int, j: int|
                0 <= i < j < num_frames ==> (#[trigger] slots@[i]).loc != (
                #[trigger] slots@[j]).loc,
            forall|i: nat, j: int|
                i < old(self).frames_spec().metas_spec().len() && 0 <= j < num_frames ==> (
                #[trigger] old(self).frames_spec().ghosts_spec()@[i].slot).loc != (
                #[trigger] slots@[j]).loc,
        ensures
            final(self).well_formed(),
            final(self).node_id_spec() == old(self).node_id_spec(),
            final(self).timelines_spec() == old(self).timelines_spec(),
            final(self).ledger_spec() == old(self).ledger_spec(),
            final(self).version_seq_start_spec() == old(self).version_seq_start_spec(),
            final(self).next_version_seq_spec() == old(self).next_version_seq_spec(),
            final(self).version_seq_limit_spec() == old(self).version_seq_limit_spec(),
            final(self).active_leases_spec() == old(self).active_leases_spec(),
            final(self).free_lease_slots_spec() == old(self).free_lease_slots_spec(),
            final(self).lease_generation_spec() == old(self).lease_generation_spec(),
            final(self).spec_spec() == old(self).spec_spec(),
            final(self).version_registry_spec() == old(self).version_registry_spec(),
            final(self).frames_spec().frame_size_spec() == old(
                self,
            ).frames_spec().frame_size_spec(),
            r.is_ok() == (old(self).tiers_spec().len() < MAX_TIERS && num_frames > 0 && old(
                self,
            ).frames_spec().metas_spec().len() + num_frames <= u32::MAX),
            match r {
                Err(_) => *final(self) == *old(self),
                Ok((tier, first)) => {
                    &&& old(self).tiers_spec().len() < MAX_TIERS
                    &&& num_frames > 0
                    &&& tier == TierId(old(self).tiers_spec().len() as u16)
                    &&& final(self).tiers_spec().len() == old(self).tiers_spec().len() + 1
                    &&& final(self).tiers_spec()[tier.index() as int] == (TierDescriptor {
                        id: tier,
                        ..desc
                    })
                    &&& final(self).frames_spec().tier_frame_sizes_spec()@ == old(
                        self,
                    ).frames_spec().tier_frame_sizes_spec()@.push(
                        old(self).frames_spec().frame_size_spec(),
                    )
                    &&& first.0 == old(self).frames_spec().metas_spec().len()
                    &&& final(self).frames_spec().metas_spec().len() == old(
                        self,
                    ).frames_spec().metas_spec().len() + num_frames
                    &&& final(self).frames_spec().ready_queue_spec()[tier.index() as int].len()
                        == num_frames
                    &&& forall|j: nat|
                        j < old(self).frames_spec().metas_spec().len()
                            ==> #[trigger] final(self).frames_spec().ghosts_spec()@[j] == old(
                            self,
                        ).frames_spec().ghosts_spec()@[j]
                            && final(self).frames_spec().metas_spec()[j as int] == old(
                            self,
                        ).frames_spec().metas_spec()[j as int]
                    &&& forall|j: int|
                        0 <= j < num_frames ==> {
                            (#[trigger] final(self).frames_spec().frame(
                                FrameId((first.0 + j) as u32),
                            )).tier == tier && final(self).frames_spec().frame(
                                FrameId((first.0 + j) as u32),
                            ).slot == slots@[j] && final(self).frames_spec().meta(
                                FrameId((first.0 + j) as u32),
                            ).state == STATE_EMPTY && final(self).frames_spec().meta(
                                FrameId((first.0 + j) as u32),
                            ).queued()
                        }
                },
            },
    {
        proof {
            assert forall|i: nat, j: int|
                i < old(self).frames.metas.len() && 0 <= j < num_frames implies (#[trigger] old(
                self,
            ).frames.ghosts_spec()@[i].slot).loc != (#[trigger] slots@[j]).loc by {
                assert(old(self).frames_spec().ghosts_spec()@[i].slot.loc != slots@[j].loc);
            }
        }
        let r = self.register_tier_with_frame_size(
            desc,
            self.frames.frame_size,
            slots,
            num_frames,
            Tracked(tokens),
        );
        proof {
            if let Ok((tier, first)) = r {
                assert(self.frames.tier_frame_sizes@ =~= old(self).frames.tier_frame_sizes@.push(
                    old(self).frames.frame_size,
                ));
                assert forall|j: nat| j < old(self).frames.metas.len() implies {
                    &&& (#[trigger] self.frames.ghosts_spec()@[j]) == old(
                        self,
                    ).frames.ghosts_spec()@[j]
                    &&& self.frames.metas[j as int] == old(self).frames.metas[j as int]
                } by {
                    assert(self.frames_spec().ghosts_spec()@[j] == old(
                        self,
                    ).frames_spec().ghosts_spec()@[j]);
                }
                assert forall|j: int| 0 <= j < num_frames implies (#[trigger] self.frames.frame(
                    FrameId((first.0 + j) as u32),
                )).tier == tier && self.frames.frame(FrameId((first.0 + j) as u32)).slot
                    == slots@[j] && self.frames.meta(FrameId((first.0 + j) as u32)).state
                    == STATE_EMPTY && self.frames.meta(
                    FrameId((first.0 + j) as u32),
                ).queued() by {};
            }
        }
        r
    }

    proof fn preserve_after_slab_append(
        &self,
        before: Self,
        tier: TierId,
        first: u32,
        slots: Seq<SlotRef>,
        num_frames: u32,
    )
        requires
            before.well_formed(),
            self.node_id == before.node_id,
            self.tiers == before.tiers,
            self.timelines == before.timelines,
            self.catalog == before.catalog,
            self.ledger == before.ledger,
            self.version_seq_start == before.version_seq_start,
            self.next_version_seq == before.next_version_seq,
            self.version_seq_limit == before.version_seq_limit,
            self.active_leases == before.active_leases,
            self.free_lease_slots == before.free_lease_slots,
            self.lease_generation == before.lease_generation,
            self.spec == before.spec,
            self.version_registry == before.version_registry,
            first == before.frames.metas.len(),
            self.frames.well_formed(
                self.instance_id(),
                self.node(),
                &self.timelines,
                &self.ledger@,
            ),
            self.frames.num_tiers == before.frames.num_tiers,
            self.frames.frame_size == before.frames.frame_size,
            self.frames.tier_frame_sizes == before.frames.tier_frame_sizes,
            self.frames.versions == before.frames.versions,
            self.frames.metas.len() == before.frames.metas.len() + num_frames,
            forall|j: nat|
                j < before.frames.metas.len() ==> #[trigger] self.frames.ghosts_spec()@[j]
                    == before.frames.ghosts_spec()@[j] && self.frames.metas[j as int]
                    == before.frames.metas[j as int],
            forall|j: nat|
                before.frames.metas.len() <= j < self.frames.metas.len() ==> (
                #[trigger] self.frames.ghosts_spec()@[j]).tier == tier
                    && self.frames.metas[j as int].state == STATE_EMPTY
                    && self.frames.metas[j as int].queued(),
            forall|j: int|
                0 <= j < num_frames ==> (#[trigger] self.frames.ghosts_spec()@[(
                before.frames.metas.len() + j) as nat]).slot == slots[j],
        ensures
            self.well_formed(),
            forall|j: int|
                0 <= j < num_frames ==> {
                    (#[trigger] self.frames.frame(FrameId((first + j) as u32))).tier == tier
                        && self.frames.frame(FrameId((first + j) as u32)).slot == slots[j]
                        && self.frames.meta(FrameId((first + j) as u32)).state == STATE_EMPTY
                        && self.frames.meta(FrameId((first + j) as u32)).queued()
                },
    {
        assert forall|k: ObjectKey, t: int| 0 <= t < self.catalog.num_tiers() implies (
        #[trigger] self.catalog.slot(k, t) matches Some(f) ==> {
            &&& f.0 < self.frames.metas.len()
            &&& self.frames.meta(f).state == STATE_PUBLISHED
            &&& self.frames.frame(f).tier.index() == t
            &&& self.frames.frame(f).bound_key == Some(k)
            &&& self.frames.frame(f).bound_version == Some(self.catalog.view()[k].version)
        }) by {
            if let Some(f) = self.catalog.slot(k, t) {
                assert(f.0 < before.frames.metas.len());
                assert(self.frames.ghosts_spec()@[f.0 as nat]
                    == before.frames.ghosts_spec()@[f.0 as nat]);
                assert(self.frames.metas[f.0 as int] == before.frames.metas[f.0 as int]);
            }
        }
        assert forall|f: FrameId|
            f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                == STATE_PUBLISHED implies {
            let g = self.frames.frame(f);
            &&& g.bound_key.is_some()
            &&& g.tier.valid(self.catalog.num_tiers())
            &&& self.catalog.slot(g.bound_key.unwrap(), g.tier.index() as int) == Some(f)
        } by {
            assert(f.0 < before.frames.metas.len()) by {
                if f.0 >= before.frames.metas.len() {
                    assert(self.frames.metas[f.0 as int].state == STATE_EMPTY);
                }
            }
            assert(self.frames.ghosts_spec()@[f.0 as nat]
                == before.frames.ghosts_spec()@[f.0 as nat]);
            assert(self.frames.metas[f.0 as int] == before.frames.metas[f.0 as int]);
            assert(before.frames.meta(f).state == STATE_PUBLISHED);
        }
        assert forall|f: FrameId|
            f.0 < self.frames.metas.len() && (#[trigger] self.frames.meta(f)).state
                == STATE_PUBLISHED implies {
            let g = self.frames.frame(f);
            g.cert matches Some(cert) && {
                &&& replica_valid(self.ledger@, self.node(), cert)
                &&& event_succeeded(self.ledger@, self.node(), cert.writer)
                &&& self.ledger@.citations(self.node()).contains(cert.writer)
                &&& g.bound_version matches Some(ver)
                &&& self.frames.versions_spec()@.contains_key(ver)
                &&& self.frames.versions_spec()@[ver] == cert.bytes
            }
        } by {
            assert(f.0 < before.frames.metas.len()) by {
                if f.0 >= before.frames.metas.len() {
                    assert(self.frames.metas[f.0 as int].state == STATE_EMPTY);
                }
            }
            assert(self.frames.ghosts_spec()@[f.0 as nat]
                == before.frames.ghosts_spec()@[f.0 as nat]);
            assert(self.frames.metas[f.0 as int] == before.frames.metas[f.0 as int]);
            assert(before.frames.meta(f).state == STATE_PUBLISHED);
        }
        assert forall|j: int| 0 <= j < num_frames implies {
            (#[trigger] self.frames.frame(FrameId((first + j) as u32))).tier == tier
                && self.frames.frame(FrameId((first + j) as u32)).slot == slots[j]
                && self.frames.meta(FrameId((first + j) as u32)).state == STATE_EMPTY
                && self.frames.meta(FrameId((first + j) as u32)).queued()
        } by {
            let idx = (before.frames.metas.len() + j) as nat;
            assert(idx == (first + j) as nat);
            assert(self.frames.ghosts_spec()@[idx].slot == slots[j]);
            assert(self.frames.ghosts_spec()@[idx].tier == tier);
            assert(self.frames.metas[idx as int].state == STATE_EMPTY);
            assert(self.frames.metas[idx as int].queued());
        }
        assert forall|s: int|
            0 <= s < self.active_leases.len() implies match #[trigger] self.active_leases[s] {
            None => self.free_lease_slots@.contains(s as u32),
            Some(lease) => {
                &&& !self.free_lease_slots@.contains(s as u32)
                &&& lease.handle.slot as int == s
                &&& 1 <= lease.handle.generation <= self.lease_generation
                &&& lease.handle.frame.0 < self.frames.metas.len()
                &&& self.frames.frame(lease.handle.frame).local_shares.contains_key(lease.share@)
            },
        } by {
            if let Some(lease) = self.active_leases[s] {
                let f = lease.handle.frame;
                assert(f.0 < before.frames.metas.len());
                assert(self.frames.ghosts_spec()@[f.0 as nat]
                    == before.frames.ghosts_spec()@[f.0 as nat]);
            }
        }
    }
}

} // verus!
