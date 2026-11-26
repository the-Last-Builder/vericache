use super::*;

verus! {

impl CacheCore {
    pub fn matches_slab(&self, tier: TierId, first: FrameId, count: u32, slot_size: u64) -> bool {
        let t = tier.as_index();
        let f = first.0 as usize;
        let n = count as usize;
        if t >= self.tiers.len() || t >= self.frames.tier_frame_sizes.len() || n == 0 || f
            >= self.frames.bindings.len() || n > self.frames.bindings.len() - f {
            return false;
        }
        self.frames.tier_frame_sizes[t] as u64 == slot_size && self.frames.bindings[f].1.0 == tier.0
            && self.frames.bindings[f + (n - 1)].1.0 == tier.0
    }

    pub fn check_published_frame(&self, frame: FrameId) -> Result<(), CoreError> {
        match self.frame_state(frame) {
            None => Err(CoreError::InvalidArgument),
            Some(STATE_PUBLISHED) => Ok(()),
            _ => Err(CoreError::WouldConflict),
        }
    }

    pub fn timeline_is_idle(&self, timeline: TimelineId) -> bool {
        let t = timeline as usize;
        t < self.timelines.confirmed_seq.len() && t < self.timelines.next_seq.len()
            && self.timelines.confirmed_seq[t].checked_add(1) == Some(self.timelines.next_seq[t])
    }

    pub open spec fn same_runtime_setup(&self, other: &Self) -> bool {
        self.runtime_id_spec() == other.runtime_id_spec() && self.version_registry_enabled_spec()
            == other.version_registry_enabled_spec()
    }

    pub open(crate) spec fn runtime_id_spec(&self) -> Option<u64> {
        self.runtime_id
    }

    #[verifier::when_used_as_spec(runtime_id_spec)]
    pub fn runtime_id(&self) -> (id: Option<u64>)
        ensures
            id == self.runtime_id_spec(),
    {
        self.runtime_id
    }

    pub open(crate) spec fn version_registry_enabled_spec(&self) -> bool {
        self.version_registry_enabled
    }

    pub fn version_registry_enabled(&self) -> (enabled: bool)
        ensures
            enabled == self.version_registry_enabled_spec(),
    {
        self.version_registry_enabled
    }

    pub fn node_id(&self) -> (id: u16)
        ensures
            id == self.node_id_spec(),
    {
        self.node_id
    }

    pub fn frame_size(&self) -> (size: usize)
        ensures
            size == self.frames_spec().frame_size_spec(),
    {
        self.frames.frame_size
    }

    pub fn catalog(&self) -> (catalog: &Catalog)
        ensures
            *catalog == self.catalog_spec(),
    {
        &self.catalog
    }

    pub fn local_leases(&self) -> (leases: &[Option<ActiveLocalLease>])
        ensures
            leases@ == self.active_leases_spec()@,
    {
        self.active_leases.as_slice()
    }

    pub fn frames(&self) -> (frames: &FrameTable)
        ensures
            *frames == self.frames_spec(),
    {
        &self.frames
    }

    pub fn timelines(&self) -> (timelines: &TimelineTable)
        ensures
            *timelines == self.timelines_spec(),
    {
        &self.timelines
    }

    pub fn ledger(&self) -> (ledger: &Tracked<KnowledgeLedger>)
        ensures
            *ledger == self.ledger_spec(),
    {
        &self.ledger
    }

    pub fn tier_frame_size(&self, tier: TierId) -> (r: Option<usize>)
        ensures
            r matches Some(size) ==> tier.valid(self.tier_count_spec()) && size
                == self.tier_frame_size_spec(tier),
    {
        let t = tier.as_index();
        if t < self.tiers.len() && t < self.frames.tier_frame_sizes.len() {
            Some(self.frames.tier_frame_sizes[t])
        } else {
            None
        }
    }

    pub open spec fn frame_count_spec(&self) -> usize {
        self.frames_spec().metas_spec().len()
    }

    #[verifier::when_used_as_spec(frame_count_spec)]
    pub fn frame_count(&self) -> (n: usize)
        ensures
            n == self.frame_count_spec(),
    {
        self.frames.metas.len()
    }

    pub open spec fn tier_count_spec(&self) -> usize {
        self.tiers_spec().len()
    }

    #[verifier::when_used_as_spec(tier_count_spec)]
    pub fn tier_count(&self) -> (n: usize)
        ensures
            n == self.tier_count_spec(),
    {
        self.tiers.len()
    }

    pub fn frame_meta(&self, frame: FrameId) -> (r: Option<FrameSyncMeta>)
        ensures
            r == if frame.0 < self.frame_count_spec() {
                Some(self.frames_spec().meta(frame))
            } else {
                None
            },
    {
        let idx = frame.0 as usize;
        if idx < self.frames.metas.len() {
            Some(self.frames.metas[idx])
        } else {
            None
        }
    }

    pub fn frame_state(&self, frame: FrameId) -> (r: Option<u8>)
        ensures
            r == if frame.0 < self.frame_count_spec() {
                Some(self.frame_state_spec(frame))
            } else {
                None
            },
    {
        match self.frame_meta(frame) {
            Some(meta) => Some(meta.state),
            None => None,
        }
    }

    pub open spec fn frame_binding_spec(&self, frame: FrameId) -> Option<(ObjectKey, TierId)> {
        if frame.0 < self.frames_spec().bindings_spec().len() {
            Some(self.frames_spec().bindings_spec()[frame.0 as int])
        } else {
            None
        }
    }

    pub fn frame_binding(&self, frame: FrameId) -> (r: Option<(ObjectKey, TierId)>)
        ensures
            r == self.frame_binding_spec(frame),
    {
        let idx = frame.0 as usize;
        if idx < self.frames.bindings.len() {
            Some(self.frames.bindings[idx])
        } else {
            None
        }
    }

    pub fn frame_tier(&self, frame: FrameId) -> (r: Option<TierId>)
        ensures
            match self.frame_binding_spec(frame) {
                Some((_, tier)) => r == Some(tier),
                None => r is None,
            },
    {
        let idx = frame.0 as usize;
        if idx < self.frames.bindings.len() {
            Some(self.frames.bindings[idx].1)
        } else {
            None
        }
    }

    /// Key of a published or draining frame. Initializing frames are private to writers.
    pub fn frame_bound_key(&self, frame: FrameId) -> (r: Option<ObjectKey>)
        ensures
            r is Some ==> (self.frame_binding_spec(frame) matches Some((k, _)) && r == Some(k)),
    {
        let idx = frame.0 as usize;
        if idx < self.frames.metas.len() && idx < self.frames.bindings.len() {
            let state = self.frames.metas[idx].state;
            if state == STATE_PUBLISHED || state == STATE_DRAINING {
                return Some(self.frames.bindings[idx].0);
            }
        }
        None
    }

    pub fn tier_ready_count(&self, tier: TierId) -> (r: Option<usize>)
        ensures
            r == self.frames_spec().tier_ready_count_spec(tier),
    {
        self.frames.tier_ready_count(tier)
    }

    pub fn tier_draining_count(&self, tier: TierId) -> (r: Option<usize>)
        ensures
            r == self.frames_spec().tier_draining_count_spec(tier),
    {
        self.frames.tier_draining_count(tier)
    }

    pub fn next_version_sequence(&self) -> (r: u64)
        ensures
            r == self.next_version_seq_spec(),
    {
        self.next_version_seq
    }

    pub fn free_lease_count(&self) -> (n: usize)
        ensures
            n == self.free_lease_slots_spec().len(),
    {
        self.free_lease_slots.len()
    }

    pub fn catalog_bucket_count(&self) -> (n: usize)
        ensures
            n == self.catalog_spec().num_buckets(),
    {
        self.catalog.bucket_count()
    }

    pub fn timeline_count(&self) -> (n: usize)
        ensures
            n == self.timelines_spec().count_spec(),
    {
        self.timelines.count()
    }

    pub open spec fn timeline_event(&self, timeline: TimelineId, sequence: u64) -> TransferEvent {
        self.timelines_spec().events_spec()@[(timeline, sequence)]
    }

    pub fn is_ticket_retired(&self, ticket: Ticket) -> (r: bool)
        requires
            self.timelines_spec().bounds(),
        ensures
            r == (self.timelines_spec().valid_ticket(ticket) && self.timelines_spec().retired(
                ticket,
            )),
    {
        self.timelines.is_retired(ticket)
    }

    pub fn frame_overflow_sequences(&self, frame: FrameId) -> (r: Option<&[u64]>)
        ensures
            r matches Some(sequences) ==> sequences@ == self.frames_spec().ovf(frame),
    {
        let idx = frame.0 as usize;
        if idx < self.frames.overflow_seqs.len() {
            Some(self.frames.overflow_seqs[idx].as_slice())
        } else {
            None
        }
    }
}

} // verus!
