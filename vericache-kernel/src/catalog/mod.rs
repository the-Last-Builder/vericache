use crate::{frame::*, spec::*};
mod lifecycle;
#[cfg(verus_only)]
use vstd::arithmetic::power2::{
    is_pow2_equiv, is_pow2_exists, lemma_pow2, lemma_pow2_strictly_increases, lemma_pow2_unfold,
    pow2,
};
use vstd::prelude::*;

verus! {

pub const DEFAULT_BUCKETS: usize = 4096;

pub const MAX_RECOMMENDED_BUCKETS: usize = 4_194_304;

#[derive(Clone, Copy)]
pub struct CatalogEntry {
    pub version: VersionId,
    pub slots: [Option<FrameId>; MAX_TIERS],
}

pub struct Catalog {
    num_tiers: usize,
    num_buckets: usize,
    buckets: Vec<Vec<(ObjectKey, CatalogEntry)>>,
    _positions: Ghost<Map<ObjectKey, usize>>,
}

impl ObjectKey {
    fn same_key(&self, other: &Self) -> (b: bool)
        ensures
            b == (*self == *other),
    {
        let equal = self.0[0] == other.0[0] && self.0[1] == other.0[1] && self.0[2] == other.0[2]
            && self.0[3] == other.0[3];
        assert(equal ==> self.0 =~= other.0);
        equal
    }
}

impl Catalog {
    pub closed spec fn num_tiers(&self) -> usize {
        self.num_tiers
    }

    pub fn tier_count(&self) -> (n: usize)
        ensures
            n == self.num_tiers(),
    {
        self.num_tiers
    }

    pub closed spec fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    pub fn bucket_count(&self) -> (n: usize)
        ensures
            n == self.num_buckets(),
    {
        self.num_buckets
    }

    pub open spec fn bucket_index_spec(key: ObjectKey, num_buckets: usize) -> usize {
        (((key.0[0] ^ key.0[1]) as usize) % num_buckets) as usize
    }

    fn bucket_idx(&self, key: &ObjectKey) -> (idx: usize)
        requires
            self.num_buckets > 0,
            self.buckets@.len() == self.num_buckets,
        ensures
            idx == Self::bucket_index_spec(*key, self.num_buckets),
            idx < self.buckets@.len(),
    {
        ((key.0[0] ^ key.0[1]) as usize) % self.num_buckets
    }

    pub closed spec fn view(&self) -> Map<ObjectKey, CatalogEntry> {
        Map::new(
            self._positions@.dom(),
            |k: ObjectKey|
                {
                    let b = Self::bucket_index_spec(k, self.num_buckets) as int;
                    self.buckets@[b]@[self._positions@[k] as int].1
                },
        )
    }

    pub closed spec fn indexed(&self) -> bool {
        &&& self.num_tiers <= MAX_TIERS
        &&& self.num_buckets > 0
        &&& self.buckets@.len() == self.num_buckets
        &&& forall|k: ObjectKey| #[trigger]
            self._positions@.contains_key(k) ==> {
                let b = Self::bucket_index_spec(k, self.num_buckets) as int;
                let i = self._positions@[k] as int;
                &&& i < self.buckets@[b]@.len()
                &&& self.buckets@[b]@[i].0 == k
            }
        &&& forall|b: int, i: int|
            0 <= b < self.num_buckets && 0 <= i < self.buckets@[b]@.len() ==> {
                let k = (#[trigger] self.buckets@[b]@[i]).0;
                &&& Self::bucket_index_spec(k, self.num_buckets) == b
                &&& self._positions@.contains_key(k)
                &&& self._positions@[k] == i
            }
    }

    pub open spec fn slot(&self, k: ObjectKey, t: int) -> Option<FrameId> {
        if self.view().contains_key(k) {
            self.view()[k].slots[t]
        } else {
            None
        }
    }

    pub open spec fn well_formed(&self, frames: &FrameTable) -> bool {
        self.covers(frames, None)
    }

    /// Publication can briefly leave the new frame awaiting its directory slot.
    pub open spec fn covers(&self, frames: &FrameTable, publishing: Option<FrameId>) -> bool {
        &&& self.indexed()
        &&& self.num_tiers() == frames.num_tiers_spec()
        &&& forall|k: ObjectKey, t: int|
            0 <= t < self.num_tiers() ==> (#[trigger] self.slot(k, t) matches Some(f) ==> {
                &&& Some(f) != publishing
                &&& f.0 < frames.metas_spec().len()
                &&& frames.meta(f).state == STATE_PUBLISHED
                &&& frames.frame(f).tier.index() == t
                &&& frames.frame(f).bound_key == Some(k)
                &&& frames.frame(f).bound_version == Some(self.view()[k].version)
            })
        &&& forall|f: FrameId|
            Some(f) != publishing && f.0 < frames.metas_spec().len() && (#[trigger] frames.meta(
                f,
            )).state == STATE_PUBLISHED ==> {
                let g = frames.frame(f);
                &&& g.bound_key.is_some()
                &&& g.tier.valid(self.num_tiers())
                &&& self.slot(g.bound_key.unwrap(), g.tier.index() as int) == Some(f)
            }
        &&& forall|k: ObjectKey, t: int|
            self.num_tiers() <= t < MAX_TIERS ==> (#[trigger] self.slot(k, t)).is_none()
        &&& forall|k|
            self.view().contains_key(k) ==> exists|t: int|
                0 <= t < self.num_tiers() && #[trigger] self.slot(k, t).is_some()
    }

    /// Size the catalog for up to half a frame per bucket, subject to the bucket limits.
    pub fn recommended_buckets(total_frames: usize) -> (b: usize)
        ensures
            b > 0,
            DEFAULT_BUCKETS <= b <= MAX_RECOMMENDED_BUCKETS,
            b >= vstd::math::min(2 * total_frames as int, MAX_RECOMMENDED_BUCKETS as int),
            vstd::arithmetic::power2::is_pow2(b as int),
    {
        let target = if total_frames >= MAX_RECOMMENDED_BUCKETS / 2 {
            MAX_RECOMMENDED_BUCKETS
        } else {
            2 * total_frames
        };
        let mut b = DEFAULT_BUCKETS;
        let ghost mut exponent: nat = 12;
        proof {
            reveal(vstd::arithmetic::power::pow);
            assert(pow2(12) == DEFAULT_BUCKETS) by (compute);
            assert(pow2(22) == MAX_RECOMMENDED_BUCKETS) by (compute);
        }
        while b < target
            invariant
                DEFAULT_BUCKETS <= b <= MAX_RECOMMENDED_BUCKETS,
                target <= MAX_RECOMMENDED_BUCKETS,
                12 <= exponent <= 22,
                b == pow2(exponent),
                pow2(22) == MAX_RECOMMENDED_BUCKETS,
            decreases 22 - exponent,
        {
            proof {
                exponent = exponent + 1;
                lemma_pow2_unfold(exponent);
                if exponent < 22 {
                    lemma_pow2_strictly_increases(exponent, 22);
                }
            }
            b *= 2;
        }
        proof {
            lemma_pow2(exponent);
            assert(is_pow2_exists(b as int));
            is_pow2_equiv(b as int);
        }
        b
    }

    pub fn with_capacity(num_tiers: usize, num_buckets: usize) -> (c: Self)
        requires
            num_tiers <= MAX_TIERS,
            num_buckets > 0,
        ensures
            c.indexed(),
            c.num_tiers() == num_tiers,
            c.num_buckets() == num_buckets,
            c.view() == Map::<ObjectKey, CatalogEntry>::empty(),
    {
        let mut buckets = Vec::<Vec<(ObjectKey, CatalogEntry)>>::with_capacity(num_buckets);
        while buckets.len() < num_buckets
            invariant
                buckets@.len() <= num_buckets,
                forall|b: int| 0 <= b < buckets@.len() ==> (#[trigger] buckets@[b])@.len() == 0,
            decreases num_buckets - buckets@.len(),
        {
            buckets.push(Vec::new());
        }
        let c = Self { num_tiers, num_buckets, buckets, _positions: Ghost(Map::empty()) };
        assert(c.view() =~= Map::empty());
        c
    }

    pub fn new(num_tiers: usize) -> (c: Self)
        requires
            num_tiers <= MAX_TIERS,
        ensures
            c.indexed(),
            c.num_tiers() == num_tiers,
            c.view() == Map::<ObjectKey, CatalogEntry>::empty(),
    {
        Self::with_capacity(num_tiers, DEFAULT_BUCKETS)
    }

    pub fn add_tier(&mut self)
        requires
            old(self).indexed(),
            old(self).num_tiers() < MAX_TIERS,
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers() + 1,
            final(self).view() == old(self).view(),
    {
        self.num_tiers = self.num_tiers + 1;
        assert(self.view() =~= old(self).view());
    }

    fn position_in_bucket(&self, b: usize, key: ObjectKey) -> (r: Option<usize>)
        requires
            self.indexed(),
            b == Self::bucket_index_spec(key, self.num_buckets),
        ensures
            r == if self._positions@.contains_key(key) {
                Some(self._positions@[key])
            } else {
                None
            },
    {
        let bucket = &self.buckets[b];
        let len = bucket.len();
        let mut idx: usize = 0;
        while idx < len
            invariant
                self.indexed(),
                b == Self::bucket_index_spec(key, self.num_buckets),
                b < self.buckets@.len(),
                bucket@ == self.buckets@[b as int]@,
                len == bucket@.len(),
                idx <= len,
                forall|j: int| 0 <= j < idx ==> (#[trigger] bucket@[j]).0 != key,
            decreases len - idx,
        {
            if bucket[idx].0.same_key(&key) {
                assert(self.buckets@[b as int]@[idx as int].0 == key);
                return Some(idx);
            }
            idx = idx + 1;
        }
        proof {
            if self._positions@.contains_key(key) {
                let pos = self._positions@[key] as int;
                assert(bucket@[pos].0 == key);
            }
        }
        None
    }

    fn position(&self, key: ObjectKey) -> (r: Option<usize>)
        requires
            self.indexed(),
        ensures
            r == if self._positions@.contains_key(key) {
                Some(self._positions@[key])
            } else {
                None
            },
    {
        let b = self.bucket_idx(&key);
        self.position_in_bucket(b, key)
    }

    pub fn lookup(&self, key: ObjectKey) -> (r: Option<&CatalogEntry>)
        requires
            self.indexed(),
        ensures
            r.is_some() == self.view().contains_key(key),
            r matches Some(entry) ==> *entry == self.view()[key],
    {
        let b = self.bucket_idx(&key);
        let i = self.position_in_bucket(b, key)?;
        Some(&self.buckets[b][i].1)
    }

    pub fn lookup_tier(&self, key: ObjectKey, tier: TierId) -> (r: Option<(VersionId, FrameId)>)
        requires
            self.indexed(),
            tier.valid(self.num_tiers()),
        ensures
            r == match self.slot(key, tier.index() as int) {
                Some(f) => Some((self.view()[key].version, f)),
                None => None,
            },
    {
        let entry = self.lookup(key)?;
        let f = entry.slots[tier.as_index()]?;
        Some((entry.version, f))
    }

    /// Return the first available replica for `key` following the caller-supplied
    /// `priority` order of tiers. Invalid tier IDs in `priority` are skipped.
    pub fn best_replica(&self, key: ObjectKey, priority: &[TierId]) -> (r: Option<
        (FrameId, TierId, VersionId),
    >)
        requires
            self.indexed(),
        ensures
            r matches Some((f, tier, version)) ==> {
                &&& tier.valid(self.num_tiers())
                &&& self.slot(key, tier.index() as int) == Some(f)
                &&& self.view()[key].version == version
                &&& exists|idx: int|
                    0 <= idx < priority.len() && #[trigger] priority[idx] == tier && forall|
                        prev: int,
                    |
                        0 <= prev < idx && (#[trigger] priority[prev]).valid(self.num_tiers())
                            ==> self.slot(key, priority[prev].index() as int).is_none()
            },
            r.is_none() ==> forall|idx: int|
                0 <= idx < priority.len() && (#[trigger] priority[idx]).valid(self.num_tiers())
                    ==> self.slot(key, priority[idx].index() as int).is_none(),
    {
        let Some(entry) = self.lookup(key) else {
            return None;
        };
        for i in 0..priority.len()
            invariant
                self.indexed(),
                self.view().contains_key(key),
                *entry == self.view()[key],
                forall|prev: int|
                    0 <= prev < i && (#[trigger] priority[prev]).valid(self.num_tiers())
                        ==> self.slot(key, priority[prev].index() as int).is_none(),
        {
            let tier = priority[i];
            let idx = tier.as_index();
            if idx < self.num_tiers {
                if let Some(f) = entry.slots[idx] {
                    assert(priority[i as int] == tier);
                    return Some((f, tier, entry.version));
                }
            }
        }
        None
    }

    fn replace_existing(&mut self, b: usize, key: ObjectKey, i: usize, value: CatalogEntry)
        requires
            old(self).indexed(),
            b == Self::bucket_index_spec(key, old(self).num_buckets),
            old(self)._positions@.contains_key(key),
            old(self)._positions@[key] == i,
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == old(self).view().insert(key, value),
    {
        let ghost original = self.buckets@[b as int]@;
        self.buckets[b].set(i, (key, value));
        proof {
            let ghost updated = self.buckets@[b as int]@;
            assert forall|k: ObjectKey| #[trigger] self._positions@.contains_key(k) implies {
                let bk = Self::bucket_index_spec(k, self.num_buckets) as int;
                let idx = self._positions@[k] as int;
                &&& idx < self.buckets@[bk]@.len()
                &&& self.buckets@[bk]@[idx].0 == k
            } by {
                let bk = Self::bucket_index_spec(k, self.num_buckets) as int;
                let idx = old(self)._positions@[k] as int;
                if bk == b as int && k != key {
                    assert(original[idx].0 == k);
                    assert(original[i as int].0 == key);
                    assert(idx != i as int);
                    assert(updated[idx] == original[idx]);
                }
            }
            assert forall|b2: int, j: int|
                0 <= b2 < self.num_buckets && 0 <= j < self.buckets@[b2]@.len() implies {
                let k = (#[trigger] self.buckets@[b2]@[j]).0;
                &&& Self::bucket_index_spec(k, self.num_buckets) == b2
                &&& self._positions@.contains_key(k)
                &&& self._positions@[k] == j
            } by {
                if b2 == b as int && j != i as int {
                    assert(updated[j] == original[j]);
                    assert(old(self).buckets@[b2]@[j].0 == updated[j].0);
                }
            }
            assert(self.view() =~= old(self).view().insert(key, value)) by {
                assert forall|k: ObjectKey|
                    k != key && self._positions@.contains_key(k) implies #[trigger] self.view()[k]
                    == old(self).view()[k] by {
                    if Self::bucket_index_spec(k, self.num_buckets) == b {
                        let idx = old(self)._positions@[k] as int;
                        assert(original[idx].0 == k);
                        assert(original[i as int].0 == key);
                        assert(idx != i as int);
                    }
                }
            }
        }
    }

    fn insert_new(&mut self, b: usize, key: ObjectKey, value: CatalogEntry)
        requires
            old(self).indexed(),
            b == Self::bucket_index_spec(key, old(self).num_buckets),
            !old(self)._positions@.contains_key(key),
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == old(self).view().insert(key, value),
    {
        let ghost original = self.buckets@[b as int]@;
        let i = self.buckets[b].len();
        self.buckets[b].push((key, value));
        proof {
            self._positions = Ghost(self._positions@.insert(key, i));
            let ghost updated = self.buckets@[b as int]@;
            assert forall|k: ObjectKey| #[trigger] self._positions@.contains_key(k) implies {
                let bk = Self::bucket_index_spec(k, self.num_buckets) as int;
                let idx = self._positions@[k] as int;
                &&& idx < self.buckets@[bk]@.len()
                &&& self.buckets@[bk]@[idx].0 == k
            } by {
                if k != key && Self::bucket_index_spec(k, self.num_buckets) == b {
                    let idx = old(self)._positions@[k] as int;
                    assert(original[idx].0 == k);
                    assert(updated[idx] == original[idx]);
                }
            }
            assert forall|b2: int, j: int|
                0 <= b2 < self.num_buckets && 0 <= j < self.buckets@[b2]@.len() implies {
                let k = (#[trigger] self.buckets@[b2]@[j]).0;
                &&& Self::bucket_index_spec(k, self.num_buckets) == b2
                &&& self._positions@.contains_key(k)
                &&& self._positions@[k] == j
            } by {
                if b2 != b as int {
                    let k = old(self).buckets@[b2]@[j].0;
                    assert(Self::bucket_index_spec(k, self.num_buckets) == b2);
                    assert(k != key);
                } else if j < i as int {
                    assert(updated[j] == original[j]);
                    let k = original[j].0;
                    assert(old(self).buckets@[b as int]@[j].0 == k);
                    assert(old(self)._positions@.contains_key(k));
                    assert(k != key);
                }
            }
            assert(self.view() =~= old(self).view().insert(key, value)) by {
                assert forall|k: ObjectKey|
                    k != key && self._positions@.contains_key(k) implies #[trigger] self.view()[k]
                    == old(self).view()[k] by {
                    if Self::bucket_index_spec(k, self.num_buckets) == b {
                        let idx = old(self)._positions@[k] as int;
                        assert(original[idx].0 == k);
                        assert(updated[idx] == original[idx]);
                    }
                }
            }
        }
    }

    fn remove_existing(&mut self, b: usize, key: ObjectKey, i: usize)
        requires
            old(self).indexed(),
            b == Self::bucket_index_spec(key, old(self).num_buckets),
            old(self)._positions@.contains_key(key),
            old(self)._positions@[key] == i,
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == old(self).view().remove(key),
    {
        let ghost original = self.buckets@[b as int]@;
        let last = self.buckets[b].len() - 1;
        let moved = self.buckets[b][last].0;
        let _ = self.buckets[b].swap_remove(i);
        proof {
            assert(old(self).buckets@[b as int]@[last as int].0 == moved);
            assert(old(self)._positions@.contains_key(moved));
            assert(old(self)._positions@[moved] == last);
            if i != last {
                assert(moved != key);
                self._positions = Ghost(self._positions@.insert(moved, i));
            }
            self._positions = Ghost(self._positions@.remove(key));
            let ghost updated = self.buckets@[b as int]@;
            assert forall|k: ObjectKey| #[trigger] self._positions@.contains_key(k) implies {
                let bk = Self::bucket_index_spec(k, self.num_buckets) as int;
                let idx = self._positions@[k] as int;
                &&& idx < self.buckets@[bk]@.len()
                &&& self.buckets@[bk]@[idx].0 == k
            } by {
                if Self::bucket_index_spec(k, self.num_buckets) == b {
                    if i != last && k == moved {
                        assert(updated[i as int] == original[last as int]);
                    } else {
                        let idx = old(self)._positions@[k] as int;
                        assert(original[idx].0 == k);
                        assert(original[i as int].0 == key);
                        assert(idx != i as int);
                        assert(idx < last as int);
                        assert(updated[idx] == original[idx]);
                    }
                } else {
                    assert(Self::bucket_index_spec(moved, self.num_buckets) == b);
                    assert(k != moved);
                }
            }
            assert forall|b2: int, j: int|
                0 <= b2 < self.num_buckets && 0 <= j < self.buckets@[b2]@.len() implies {
                let k = (#[trigger] self.buckets@[b2]@[j]).0;
                &&& Self::bucket_index_spec(k, self.num_buckets) == b2
                &&& self._positions@.contains_key(k)
                &&& self._positions@[k] == j
            } by {
                if b2 != b as int {
                    let k = old(self).buckets@[b2]@[j].0;
                    assert(Self::bucket_index_spec(k, self.num_buckets) == b2);
                    assert(Self::bucket_index_spec(moved, self.num_buckets) == b);
                    assert(k != key && k != moved);
                } else if i != last && j == i as int {
                    assert(updated[j] == original[last as int]);
                } else {
                    assert(updated[j] == original[j]);
                    let k = original[j].0;
                    assert(old(self).buckets@[b as int]@[j].0 == k);
                    assert(old(self)._positions@.contains_key(k));
                    assert(k != key && k != moved);
                }
            }
            assert(self.view() =~= old(self).view().remove(key)) by {
                assert forall|k: ObjectKey|
                    k != key && self._positions@.contains_key(k) implies #[trigger] self.view()[k]
                    == old(self).view()[k] by {
                    if Self::bucket_index_spec(k, self.num_buckets) == b {
                        if i != last && k == moved {
                            assert(updated[i as int] == original[last as int]);
                        } else {
                            let idx = old(self)._positions@[k] as int;
                            assert(original[idx].0 == k);
                            assert(updated[idx] == original[idx]);
                        }
                    } else {
                        assert(Self::bucket_index_spec(moved, self.num_buckets) == b);
                        assert(k != moved);
                    }
                }
            }
        }
    }

    /// Edit one full key while keeping every collider at its logical key.
    pub(super) fn set_entry(&mut self, key: ObjectKey, entry: Option<CatalogEntry>)
        requires
            old(self).indexed(),
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == match entry {
                Some(value) => old(self).view().insert(key, value),
                None => old(self).view().remove(key),
            },
    {
        let b = self.bucket_idx(&key);
        match (self.position_in_bucket(b, key), entry) {
            (Some(i), Some(value)) => self.replace_existing(b, key, i, value),
            (None, Some(value)) => self.insert_new(b, key, value),
            (Some(i), None) => self.remove_existing(b, key, i),
            (None, None) => {
                assert(self.view() =~= old(self).view().remove(key));
            },
        }
    }
}

} // verus!
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_half_collisions_and_custom_tier_priority() {
        let mut catalog = Catalog::new(16);
        let tier_dram = TierId(0);
        let tier_nvme = TierId(1);
        let tier_custom = TierId(12);
        // All keys map to the same bucket as `0x1234 ^ 0x5678 = 0x444C`:
        // - identical low halves (differing in [2], [3])
        // - distinct low halves with identical XOR (`(0x1234 ^ 0xAAAA) ^ (0x5678 ^ 0xAAAA)`)
        // - distinct XOR colliding modulo `DEFAULT_BUCKETS` (`0x0234 ^ 0x5678 = 0x544C = 0x444C + 4096`)
        let keys = [
            ObjectKey([0x1234, 0x5678, 1, 2]),
            ObjectKey([0x1234, 0x5678, 1, 3]),
            ObjectKey([0x1234, 0x5678, 4, 2]),
            ObjectKey([0x1234 ^ 0xAAAA, 0x5678 ^ 0xAAAA, 7, 8]),
            ObjectKey([0x0234, 0x5678, 9, 10]),
        ];
        let expected_bucket = catalog.bucket_idx(&keys[0]);
        for key in keys {
            assert_eq!(catalog.bucket_idx(&key), expected_bucket);
        }
        for (i, key) in keys.into_iter().enumerate() {
            let mut slots = [None; MAX_TIERS];
            slots[tier_dram.as_index()] = Some(FrameId(i as u32));
            slots[tier_nvme.as_index()] = Some(FrameId(10 + i as u32));
            slots[tier_custom.as_index()] = Some(FrameId(20 + i as u32));
            catalog.set_entry(
                key,
                Some(CatalogEntry {
                    version: VersionId {
                        origin_node: 0,
                        seq: i as u64 + 1,
                    },
                    slots,
                }),
            );
        }
        assert_eq!(catalog.buckets[expected_bucket].len(), keys.len());
        assert!(catalog.lookup(ObjectKey([0x1234, 0x5678, 8, 9])).is_none());
        catalog.set_entry(keys[1], None);
        assert!(catalog.lookup(keys[1]).is_none());
        for i in [0usize, 2, 3, 4] {
            let version = VersionId {
                origin_node: 0,
                seq: i as u64 + 1,
            };
            assert_eq!(
                catalog.best_replica(keys[i], &[TierId(99), tier_custom, tier_dram, tier_nvme]),
                Some((FrameId(20 + i as u32), tier_custom, version))
            );
            assert_eq!(
                catalog.best_replica(keys[i], &[TierId(5), tier_dram, tier_nvme]),
                Some((FrameId(i as u32), tier_dram, version))
            );
            let mut entry = *catalog.lookup(keys[i]).unwrap();
            entry.slots[tier_dram.as_index()] = None;
            catalog.set_entry(keys[i], Some(entry));
            assert_eq!(
                catalog.best_replica(keys[i], &[tier_dram, tier_nvme]),
                Some((FrameId(10 + i as u32), tier_nvme, version))
            );
            catalog.set_entry(keys[i], None);
        }
        assert_eq!(catalog.buckets.len(), DEFAULT_BUCKETS);
        assert!(catalog.buckets.iter().all(|b| b.is_empty()));
    }

    #[test]
    fn with_capacity_stress_and_empty_after_removal() {
        for num_buckets in [1usize, 16usize] {
            let mut catalog = Catalog::with_capacity(4, num_buckets);
            assert_eq!(catalog.buckets.len(), num_buckets);
            let count = 200usize;
            for i in 0..count {
                let key = ObjectKey([i as u64, (i as u64).wrapping_mul(0x9e37_79b9), 1, 2]);
                let mut slots = [None; MAX_TIERS];
                slots[0] = Some(FrameId(i as u32));
                catalog.set_entry(
                    key,
                    Some(CatalogEntry {
                        version: VersionId {
                            origin_node: 1,
                            seq: i as u64 + 1,
                        },
                        slots,
                    }),
                );
            }
            // Overwrite even keys in place and verify updated version/frame.
            for i in (0..count).step_by(2) {
                let key = ObjectKey([i as u64, (i as u64).wrapping_mul(0x9e37_79b9), 1, 2]);
                let mut slots = [None; MAX_TIERS];
                slots[1] = Some(FrameId(1000 + i as u32));
                catalog.set_entry(
                    key,
                    Some(CatalogEntry {
                        version: VersionId {
                            origin_node: 2,
                            seq: i as u64 + 100,
                        },
                        slots,
                    }),
                );
            }
            for i in 0..count {
                let key = ObjectKey([i as u64, (i as u64).wrapping_mul(0x9e37_79b9), 1, 2]);
                if i % 2 == 0 {
                    assert_eq!(
                        catalog.lookup_tier(key, TierId(1)),
                        Some((
                            VersionId {
                                origin_node: 2,
                                seq: i as u64 + 100,
                            },
                            FrameId(1000 + i as u32),
                        ))
                    );
                    assert!(catalog.lookup_tier(key, TierId(0)).is_none());
                } else {
                    assert_eq!(
                        catalog.lookup_tier(key, TierId(0)),
                        Some((
                            VersionId {
                                origin_node: 1,
                                seq: i as u64 + 1,
                            },
                            FrameId(i as u32),
                        ))
                    );
                }
            }
            // Remove in mixed order (first evens backwards, then odds forwards).
            for i in (0..count).step_by(2).rev() {
                let key = ObjectKey([i as u64, (i as u64).wrapping_mul(0x9e37_79b9), 1, 2]);
                catalog.set_entry(key, None);
                assert!(catalog.lookup(key).is_none());
            }
            for i in (1..count).step_by(2) {
                let key = ObjectKey([i as u64, (i as u64).wrapping_mul(0x9e37_79b9), 1, 2]);
                catalog.set_entry(key, None);
                assert!(catalog.lookup(key).is_none());
            }
            assert_eq!(catalog.buckets.len(), num_buckets);
            assert!(catalog.buckets.iter().all(|b| b.is_empty()));
        }
    }
}
