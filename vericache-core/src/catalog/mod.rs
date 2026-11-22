use crate::{frame::*, spec::*};
mod lifecycle;
use std::collections::HashMap;
use vstd::prelude::*;

verus! {

#[derive(Clone, Copy)]
pub struct CatalogEntry {
    pub version: VersionId,
    pub slots: [Option<FrameId>; MAX_TIERS],
}

pub struct Catalog {
    num_tiers: usize,
    buckets: HashMap<u128, Vec<(ObjectKey, CatalogEntry)>>,
    collision_indices: HashMap<u128, HashMap<u128, usize>>,
    _positions: Ghost<Map<ObjectKey, usize>>,
}

impl ObjectKey {
    #[verifier::opaque]
    pub open spec fn bucket(self) -> u128 {
        ((self.0[1] as u128) << 64) | self.0[0] as u128
    }

    fn bucket_number(&self) -> (n: u128)
        ensures
            n == self.bucket(),
    {
        proof {
            reveal(ObjectKey::bucket);
        }
        ((self.0[1] as u128) << 64) | self.0[0] as u128
    }

    #[verifier::opaque]
    pub open spec fn upper(self) -> u128 {
        ((self.0[3] as u128) << 64) | self.0[2] as u128
    }

    fn upper_half(&self) -> (value: u128)
        ensures
            value == self.upper(),
    {
        proof {
            reveal(ObjectKey::upper);
        }
        ((self.0[3] as u128) << 64) | self.0[2] as u128
    }

    proof fn halves_identify(a: Self, b: Self)
        requires
            a.bucket() == b.bucket(),
            a.upper() == b.upper(),
        ensures
            a == b,
    {
        reveal(ObjectKey::bucket);
        reveal(ObjectKey::upper);
        Self::unpack_half(a.0[0], a.0[1]);
        Self::unpack_half(b.0[0], b.0[1]);
        Self::unpack_half(a.0[2], a.0[3]);
        Self::unpack_half(b.0[2], b.0[3]);
        assert(a.0 =~= b.0);
    }

    proof fn unpack_half(lo: u64, hi: u64)
        ensures
            (((hi as u128) << 64) | lo as u128) as u64 == lo,
            ((((hi as u128) << 64) | lo as u128) >> 64) as u64 == hi,
    {
        assert((((hi as u128) << 64) | lo as u128) as u64 == lo && ((((hi as u128) << 64)
            | lo as u128) >> 64) as u64 == hi) by (bit_vector);
    }

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

    pub closed spec fn view(&self) -> Map<ObjectKey, CatalogEntry> {
        Map::new(
            self._positions@.dom(),
            |k: ObjectKey| self.buckets@[k.bucket()][self._positions@[k] as int].1,
        )
    }

    pub closed spec fn indexed(&self) -> bool {
        &&& self.num_tiers <= MAX_TIERS
        &&& forall|k: ObjectKey| #[trigger]
            self._positions@.contains_key(k) ==> {
                let i = self._positions@[k] as int;
                &&& self.buckets@.contains_key(k.bucket())
                &&& i < self.buckets@[k.bucket()].len()
                &&& self.buckets@[k.bucket()][i].0 == k
            }
        &&& self.collisions_indexed()
        &&& forall|b: u128, i: int|
            self.buckets@.contains_key(b) && 0 <= i < self.buckets@[b].len() ==> {
                let k = (#[trigger] self.buckets@[b][i]).0;
                &&& k.bucket() == b
                &&& self._positions@.contains_key(k)
                &&& self._positions@[k] == i
            }
    }

    closed spec fn collisions_indexed(&self) -> bool {
        forall|k: ObjectKey| #[trigger]
            self._positions@.contains_key(k) && self.buckets@[k.bucket()].len() > 1 ==> {
                &&& self.collision_indices@.contains_key(k.bucket())
                &&& self.collision_indices@[k.bucket()]@.contains_key(k.upper())
                &&& self.collision_indices@[k.bucket()]@[k.upper()] == self._positions@[k]
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
        &&& self.num_tiers() == frames.num_tiers
        &&& forall|k: ObjectKey, t: int|
            0 <= t < self.num_tiers() ==> (#[trigger] self.slot(k, t) matches Some(f) ==> {
                &&& Some(f) != publishing
                &&& f.0 < frames.metas.len()
                &&& frames.meta(f).state == STATE_PUBLISHED
                &&& frames.frame(f).tier.index() == t
                &&& frames.frame(f).bound_key == Some(k)
                &&& frames.frame(f).bound_version == Some(self.view()[k].version)
            })
        &&& forall|f: FrameId|
            Some(f) != publishing && f.0 < frames.metas.len() && (#[trigger] frames.meta(f)).state
                == STATE_PUBLISHED ==> {
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

    pub fn new(num_tiers: usize) -> (c: Self)
        requires
            num_tiers <= MAX_TIERS,
        ensures
            c.indexed(),
            c.num_tiers() == num_tiers,
            c.view() == Map::<ObjectKey, CatalogEntry>::empty(),
    {
        let c = Self {
            num_tiers,
            buckets: HashMap::new(),
            collision_indices: HashMap::new(),
            _positions: Ghost(Map::empty()),
        };
        assert(c.view() =~= Map::empty());
        c
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
        let number = key.bucket_number();
        let bucket = self.buckets.get(&number)?;
        let i = if bucket.len() == 1 {
            0
        } else {
            let index = self.collision_indices.get(&number)?;
            *index.get(&key.upper_half())?
        };
        if i < bucket.len() && bucket[i].0.same_key(&key) {
            Some(i)
        } else {
            None
        }
    }

    pub fn lookup(&self, key: ObjectKey) -> (r: Option<&CatalogEntry>)
        requires
            self.indexed(),
        ensures
            r.is_some() == self.view().contains_key(key),
            r matches Some(entry) ==> *entry == self.view()[key],
    {
        let i = self.position(key)?;
        Some(&self.buckets.get(&key.bucket_number()).unwrap()[i].1)
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

    fn replace_existing(&mut self, key: ObjectKey, i: usize, value: CatalogEntry)
        requires
            old(self).indexed(),
            old(self)._positions@.contains_key(key),
            old(self)._positions@[key] == i,
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == old(self).view().insert(key, value),
    {
        let b = key.bucket_number();
        let mut bucket = self.buckets.remove(&b).unwrap();
        let ghost original = bucket@;
        bucket.set(i, (key, value));
        self.buckets.insert(b, bucket);
        proof {
            assert forall|k: ObjectKey| #[trigger] self._positions@.contains_key(k) implies {
                let idx = self._positions@[k] as int;
                &&& self.buckets@.contains_key(k.bucket())
                &&& idx < self.buckets@[k.bucket()].len()
                &&& self.buckets@[k.bucket()][idx].0 == k
            } by {
                if k != key && k.bucket() == b {
                    let idx = old(self)._positions@[k] as int;
                    assert(original[idx].0 == k);
                }
            }
            assert(self.collisions_indexed()) by {
                assert forall|k: ObjectKey| #[trigger]
                    self._positions@.contains_key(k) && self.buckets@[k.bucket()].len()
                        > 1 implies {
                    &&& self.collision_indices@.contains_key(k.bucket())
                    &&& self.collision_indices@[k.bucket()]@.contains_key(k.upper())
                    &&& self.collision_indices@[k.bucket()]@[k.upper()] == self._positions@[k]
                } by {
                    if k.bucket() == b {
                        assert(old(self).buckets@[b].len() > 1);
                    }
                }
            }
            assert(self.view() =~= old(self).view().insert(key, value)) by {
                assert forall|k: ObjectKey|
                    k != key && self._positions@.contains_key(k) implies #[trigger] self.view()[k]
                    == old(self).view()[k] by {
                    if k.bucket() == b {
                        assert(original[old(self)._positions@[k] as int].0 == k);
                    }
                }
            }
        }
    }

    fn insert_new(&mut self, key: ObjectKey, value: CatalogEntry)
        requires
            old(self).indexed(),
            !old(self)._positions@.contains_key(key),
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == old(self).view().insert(key, value),
    {
        let b = key.bucket_number();
        let mut bucket = match self.buckets.remove(&b) {
            Some(bucket) => bucket,
            None => Vec::new(),
        };
        let ghost original = bucket@;
        let mut index = self.collision_indices.remove(&b).unwrap_or(HashMap::new());
        let i = bucket.len();
        if i == 1 {
            index.insert(bucket[0].0.upper_half(), 0);
        }
        if i > 0 {
            index.insert(key.upper_half(), i);
        }
        bucket.push((key, value));
        proof {
            self._positions = Ghost(self._positions@.insert(key, i));
        }
        if bucket.len() > 1 {
            self.collision_indices.insert(b, index);
        }
        self.buckets.insert(b, bucket);
        proof {
            assert forall|k: ObjectKey| #[trigger] self._positions@.contains_key(k) implies {
                let idx = self._positions@[k] as int;
                &&& self.buckets@.contains_key(k.bucket())
                &&& idx < self.buckets@[k.bucket()].len()
                &&& self.buckets@[k.bucket()][idx].0 == k
            } by {
                if k != key && k.bucket() == b {
                    let idx = old(self)._positions@[k] as int;
                    assert(original[idx].0 == k);
                }
            }
            assert(self.collisions_indexed()) by {
                assert forall|k: ObjectKey| #[trigger]
                    self._positions@.contains_key(k) && self.buckets@[k.bucket()].len()
                        > 1 implies {
                    &&& self.collision_indices@.contains_key(k.bucket())
                    &&& self.collision_indices@[k.bucket()]@.contains_key(k.upper())
                    &&& self.collision_indices@[k.bucket()]@[k.upper()] == self._positions@[k]
                } by {
                    if k.bucket() == b && k != key {
                        if k.upper() == key.upper() {
                            ObjectKey::halves_identify(k, key);
                        }
                        let idx = old(self)._positions@[k] as int;
                        assert(original[idx].0 == k);
                    }
                }
            }
            assert(self.view() =~= old(self).view().insert(key, value)) by {
                assert forall|k: ObjectKey|
                    k != key && self._positions@.contains_key(k) implies #[trigger] self.view()[k]
                    == old(self).view()[k] by {
                    if k.bucket() == b {
                        assert(original[old(self)._positions@[k] as int].0 == k);
                    }
                }
            }
        }
    }

    fn remove_existing(&mut self, key: ObjectKey, i: usize)
        requires
            old(self).indexed(),
            old(self)._positions@.contains_key(key),
            old(self)._positions@[key] == i,
        ensures
            final(self).indexed(),
            final(self).num_tiers() == old(self).num_tiers(),
            final(self).view() == old(self).view().remove(key),
    {
        let b = key.bucket_number();
        let mut bucket = self.buckets.remove(&b).unwrap();
        let ghost original = bucket@;
        let mut index = self.collision_indices.remove(&b).unwrap_or(HashMap::new());
        let last = bucket.len() - 1;
        let moved = bucket[last].0;
        if last > 1 {
            index.remove(&key.upper_half());
            if i != last {
                index.insert(moved.upper_half(), i);
            }
        }
        bucket.swap_remove(i);
        proof {
            if i != last {
                self._positions = Ghost(self._positions@.insert(moved, i));
            }
            self._positions = Ghost(self._positions@.remove(key));
        }
        let ghost updated = bucket@;
        if bucket.len() > 1 {
            self.collision_indices.insert(b, index);
        }
        if !bucket.is_empty() {
            self.buckets.insert(b, bucket);
        }
        proof {
            assert forall|k: ObjectKey| #[trigger] self._positions@.contains_key(k) implies {
                let idx = self._positions@[k] as int;
                &&& self.buckets@.contains_key(k.bucket())
                &&& idx < self.buckets@[k.bucket()].len()
                &&& self.buckets@[k.bucket()][idx].0 == k
            } by {
                if k.bucket() == b {
                    let idx = old(self)._positions@[k] as int;
                    assert(original[idx].0 == k);
                    assert(updated[self._positions@[k] as int].0 == k);
                }
            }
            assert(self.collisions_indexed()) by {
                assert forall|k: ObjectKey| #[trigger]
                    self._positions@.contains_key(k) && self.buckets@[k.bucket()].len()
                        > 1 implies {
                    &&& self.collision_indices@.contains_key(k.bucket())
                    &&& self.collision_indices@[k.bucket()]@.contains_key(k.upper())
                    &&& self.collision_indices@[k.bucket()]@[k.upper()] == self._positions@[k]
                } by {
                    if k.bucket() == b {
                        if k.upper() == key.upper() {
                            ObjectKey::halves_identify(k, key);
                        }
                        let idx = old(self)._positions@[k] as int;
                        assert(original[idx].0 == k);
                        if moved.upper() == k.upper() {
                            assert(original[last as int].0.bucket() == b);
                            ObjectKey::halves_identify(k, moved);
                        }
                    }
                }
            }
            assert(self.view() =~= old(self).view().remove(key)) by {
                assert forall|k: ObjectKey|
                    k != key && self._positions@.contains_key(k) implies #[trigger] self.view()[k]
                    == old(self).view()[k] by {
                    if k.bucket() == b {
                        assert(original[old(self)._positions@[k] as int].0 == k);
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
        match (self.position(key), entry) {
            (Some(i), Some(value)) => self.replace_existing(key, i, value),
            (None, Some(value)) => self.insert_new(key, value),
            (Some(i), None) => self.remove_existing(key, i),
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
        let keys = [
            ObjectKey([0x1234, 0x5678, 1, 2]),
            ObjectKey([0x1234, 0x5678, 1, 3]),
            ObjectKey([0x1234, 0x5678, 4, 2]),
        ];
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
        assert!(catalog.lookup(ObjectKey([0x1234, 0x5678, 8, 9])).is_none());
        catalog.set_entry(keys[1], None);
        assert!(catalog.lookup(keys[1]).is_none());
        for i in [0usize, 2] {
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
        assert!(catalog.buckets.is_empty());
        assert!(catalog.collision_indices.is_empty());
    }
}
