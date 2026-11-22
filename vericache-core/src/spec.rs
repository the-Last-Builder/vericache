use vstd::prelude::*;

verus! {

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObjectKey(pub [u64; 4]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VersionId {
    pub origin_node: u16,
    pub seq: u64,
}

pub const MAX_VERSION_SEQ: u64 = 0x0000_ffff_ffff_ffff;

impl VersionId {
    pub const NONE: Self = Self { origin_node: 0, seq: 0 };

    pub open spec fn valid_sequence(&self) -> bool {
        self.seq <= MAX_VERSION_SEQ
    }

    #[verifier::when_used_as_spec(valid_sequence)]
    pub fn well_formed(&self) -> (valid: bool)
        ensures
            valid == self.valid_sequence(),
    {
        self.seq <= MAX_VERSION_SEQ
    }

    pub open spec fn packed(&self) -> u64 {
        ((self.origin_node as u64) << 48) | self.seq
    }

    pub open spec fn decoded(raw: u64) -> Self {
        Self { origin_node: (raw >> 48) as u16, seq: raw & MAX_VERSION_SEQ }
    }

    pub fn pack(&self) -> (raw: u64)
        requires
            self.well_formed(),
        ensures
            raw == self.packed(),
    {
        ((self.origin_node as u64) << 48) | self.seq
    }

    pub fn unpack(raw: u64) -> (v: Self)
        ensures
            v == Self::decoded(raw),
            v.well_formed(),
    {
        proof {
            assert((raw & 0xffffffffffffu64) <= 0xffffffffffffu64) by (bit_vector);
        }
        Self { origin_node: (raw >> 48) as u16, seq: raw & MAX_VERSION_SEQ }
    }
}

pub proof fn lemma_version_pack_unpack(v: VersionId, raw: u64)
    requires
        v.well_formed(),
    ensures
        VersionId::decoded(v.packed()) == v,
        VersionId::decoded(raw).packed() == raw,
{
    let n = v.origin_node;
    let s = v.seq;
    assert(s <= 0xffffffffffffu64);
    assert(((((n as u64) << 48) | s) >> 48) as u16 == n && ((((n as u64) << 48) | s)
        & 0xffffffffffffu64) == s) by (bit_vector)
        requires
            s <= 0xffffffffffffu64,
    ;
    assert((((((raw >> 48) as u16) as u64) << 48) | (raw & 0xffffffffffffu64)) == raw)
        by (bit_vector);
}

pub const MAX_TIERS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TierId(pub u16);

impl TierId {
    pub open spec fn valid(self, num_tiers: usize) -> bool {
        (self.0 as usize) < num_tiers
    }

    pub open spec fn index(self) -> usize {
        self.0 as usize
    }

    pub fn as_index(&self) -> (i: usize)
        ensures
            i == self.index(),
    {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierKind {
    GpuVram,
    HostDram,
    LocalNvme,
    RemotePool,
    Custom(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierDescriptor {
    pub id: TierId,
    pub kind: TierKind,
    pub device_id: Option<u32>,
    pub host_accessible: bool,
    pub rdma_capable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameId(pub u32);

pub type TimelineId = u8;

pub const MAX_TIMELINES: usize = 64;

pub const MAX_TICKET_SEQ: u64 = 0x00ff_ffff_ffff_ffff;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket(pub u64);

impl Ticket {
    pub const NONE: Self = Ticket(0);

    pub open spec fn timeline_id(self) -> u8 {
        (self.0 >> 56) as u8
    }

    pub open spec fn sequence(self) -> u64 {
        self.0 & MAX_TICKET_SEQ
    }

    pub open spec fn encoded(t: u8, s: u64) -> Self {
        Ticket(((t as u64) << 56) | s)
    }

    pub fn pack(timeline: u8, seq: u64) -> (r: Self)
        requires
            timeline < MAX_TIMELINES,
            1 <= seq <= MAX_TICKET_SEQ,
        ensures
            r == Self::encoded(timeline, seq),
            r.timeline_id() == timeline,
            r.sequence() == seq,
            r != Self::NONE,
    {
        proof {
            lemma_ticket_pack_unpack(timeline, seq, 0);
            assert(Ticket::NONE.sequence() == 0) by (compute);
        }
        Ticket(((timeline as u64) << 56) | seq)
    }

    pub fn timeline(&self) -> (r: u8)
        ensures
            r == self.timeline_id(),
    {
        (self.0 >> 56) as u8
    }

    pub fn seq(&self) -> (r: u64)
        ensures
            r == self.sequence(),
    {
        self.0 & MAX_TICKET_SEQ
    }

    pub fn is_none(&self) -> (r: bool)
        ensures
            r == (*self == Self::NONE),
    {
        self.0 == 0
    }
}

pub proof fn lemma_ticket_pack_unpack(t: u8, s: u64, raw: u64)
    requires
        s <= MAX_TICKET_SEQ,
    ensures
        Ticket::encoded(t, s).timeline_id() == t,
        Ticket::encoded(t, s).sequence() == s,
        Ticket::encoded(Ticket(raw).timeline_id(), Ticket(raw).sequence()).0 == raw,
{
    assert(((((t as u64) << 56) | s) >> 56) as u8 == t && ((((t as u64) << 56) | s)
        & 0xffffffffffffffu64) == s) by (bit_vector)
        requires
            s <= 0xffffffffffffffu64,
    ;
    assert((((((raw >> 56) as u8) as u64) << 56) | (raw & 0xffffffffffffffu64)) == raw)
        by (bit_vector);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncPoint {
    HostComplete,
    TimelineEvent { timeline: u8, event_slot: u32, generation: u64 },
}

pub const BACKEND_HOST: u16 = 0;

pub const BACKEND_CUDA: u16 = 1;

pub const BACKEND_NVME: u16 = 2;

pub const BACKEND_RDMA: u16 = 3;

pub const BACKEND_CUSTOM_BASE: u16 = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreError {
    Miss,
    CapacityExhausted,
    WouldConflict,
    StaleReplica,
    InvalidArgument,
    SequenceExhausted,
    UnsupportedTier,
    Backend { kind: u16, code: i32 },
}

pub ghost struct ServiceSpec {
    pub frame_size: nat,
    pub versions: Map<VersionId, Seq<u8>>,
    pub catalog: Map<ObjectKey, VersionId>,
}

impl ServiceSpec {
    pub open spec fn well_formed(self) -> bool {
        &&& 0 < self.frame_size <= u64::MAX
        &&& Self::valid_versions(self.frame_size, self.versions)
        &&& Self::directory_valid(self.catalog, self.versions)
    }

    pub open spec fn valid_versions(size: nat, versions: Map<VersionId, Seq<u8>>) -> bool {
        forall|v| #[trigger]
            versions.contains_key(v) ==> v != VersionId::NONE && v.valid_sequence()
                && versions[v].len() == size
    }

    pub proof fn insert_version(
        size: nat,
        versions: Map<VersionId, Seq<u8>>,
        version: VersionId,
        bytes: Seq<u8>,
    )
        requires
            Self::valid_versions(size, versions),
            version != VersionId::NONE,
            version.well_formed(),
            bytes.len() == size,
        ensures
            Self::valid_versions(size, versions.insert(version, bytes)),
    {
        let next = versions.insert(version, bytes);
        assert forall|v: VersionId| #[trigger] next.contains_key(v) implies {
            v != VersionId::NONE && v.valid_sequence() && next[v].len() == size
        } by {
            if v != version {
                assert(versions.contains_key(v));
                assert(versions[v].len() == size);
            }
        }
    }

    pub open spec fn directory_valid(
        catalog: Map<ObjectKey, VersionId>,
        versions: Map<VersionId, Seq<u8>>,
    ) -> bool {
        forall|k: ObjectKey| #[trigger]
            catalog.contains_key(k) ==> versions.contains_key(catalog[k])
    }

    pub proof fn restrict_directory(self, earlier: Self)
        requires
            earlier.well_formed(),
            self.versions == earlier.versions,
            self.frame_size == earlier.frame_size,
            self.catalog.submap_of(earlier.catalog),
        ensures
            self.well_formed(),
    {
        assert forall|k: ObjectKey| #[trigger]
            self.catalog.contains_key(k) implies self.versions.contains_key(self.catalog[k]) by {
            assert(self.catalog.dom().contains(k));
            assert(earlier.catalog.dom().contains(k));
            assert(earlier.catalog.contains_key(k));
        }
        assert(Self::directory_valid(self.catalog, self.versions));
    }

    pub open spec fn versions_monotonic(self, later: Self) -> bool {
        self.versions.submap_of(later.versions)
    }
}

} // verus!
