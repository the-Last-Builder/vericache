use super::*;

verus! {

broadcast use {
    vstd::map::group_map_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::imap::group_imap_lemmas,
};

pub const BITMAP_WORDS: usize = 64;

pub const BITMAP_CAPACITY: u64 = 4096;

pub const BITMAP_SLOTS: usize = 4096;

pub struct SlidingBitmap {
    words: [u64; BITMAP_WORDS],
}

spec fn word_bit(word: u64, bit: int) -> bool {
    ((word >> (bit as u64)) & 1) == 1
}

impl SlidingBitmap {
    pub closed spec fn bit_set(&self, offset: int) -> bool {
        0 <= offset < BITMAP_CAPACITY && word_bit(self.words[offset / 64], offset % 64)
    }

    pub open spec fn well_formed(&self) -> bool {
        !self.bit_set(0)
    }

    pub open spec fn observations(
        &self,
        table: &TimelineTable,
        t: u8,
        node: NodeId,
        ledger: &KnowledgeLedger,
    ) -> bool {
        forall|i: int| #[trigger]
            self.bit_set(i) ==> {
                let s = (table.confirmed_seq[t as int] + 1 + i) as u64;
                table.issued(t, s) && table.observed(t, s, node, ledger.knows)
            }
    }

    pub proof fn preserve_observations(
        &self,
        before: &TimelineTable,
        after: &TimelineTable,
        tier: u8,
        node: NodeId,
        earlier: &KnowledgeLedger,
        later: &KnowledgeLedger,
    )
        requires
            self.observations(before, tier, node, earlier),
            before.bounds(),
            tier < before.count,
            before.confirmed_seq[tier as int] == after.confirmed_seq[tier as int],
            before.count == after.count,
            before.next_seq == after.next_seq,
            before.events == after.events,
            earlier.knows.set().subset_of(later.knows.set()),
        ensures
            self.observations(after, tier, node, later),
    {
        assert forall|i: int| #[trigger] self.bit_set(i) implies {
            let s = (after.confirmed_seq[tier as int] + 1 + i) as u64;
            after.issued(tier, s) && after.observed(tier, s, node, later.knows)
        } by {
            assert(0 <= i < BITMAP_CAPACITY);
            let s = (before.confirmed_seq[tier as int] + 1 + i) as u64;
            assert(before.observed(tier, s, node, earlier.knows));
        }
    }

    fn mark(&mut self, offset: usize)
        requires
            offset < BITMAP_CAPACITY,
        ensures
            forall|i: int| #[trigger]
                final(self).bit_set(i) == (old(self).bit_set(i) || i == offset),
    {
        let index = offset / 64;
        let bit = (offset % 64) as u64;
        let word = self.words[index];
        self.words[index] = word | (1u64 << bit);
        assert forall|i: int| #[trigger]
            self.bit_set(i) == (old(self).bit_set(i) || i == offset) by {
            if 0 <= i < BITMAP_CAPACITY && i / 64 == index {
                let b = (i % 64) as u64;
                assert((((word | (1u64 << bit)) >> b) & 1u64 == 1u64) == (((word >> b) & 1u64
                    == 1u64) || b == bit)) by (bit_vector)
                    requires
                        b < 64,
                        bit < 64,
                ;
                assert((b == bit) == (i == offset));
            }
        }
    }

    fn prefix(&self) -> (n: usize)
        ensures
            n <= BITMAP_CAPACITY,
            !self.bit_set(n as int),
            forall|i: int| 0 <= i < n ==> #[trigger] self.bit_set(i),
    {
        let mut index = 0;
        while index < BITMAP_WORDS
            invariant
                index <= BITMAP_WORDS,
                forall|i: int| 0 <= i < index * 64 ==> #[trigger] self.bit_set(i),
            decreases BITMAP_WORDS - index,
        {
            let word = self.words[index];
            let bits = word.trailing_ones();
            proof {
                vstd::std_specs::bits::axiom_u64_trailing_ones(word);
            }
            assert forall|i: int|
                index * 64 <= i < index * 64 + bits implies #[trigger] self.bit_set(i) by {
                assert((word >> ((i % 64) as u64)) & 1u64 == 1u64);
            }
            if bits < 64 {
                return index * 64 + bits as usize;
            }
            index += 1;
        }
        BITMAP_CAPACITY as usize
    }

    proof fn lemma_shift_word_bit(low: u64, high: u64, merged: u64, bits: u64, b: u64)
        requires
            b < 64,
            bits < 64,
            merged == (if bits == 0 {
                low
            } else {
                (low >> bits) | (high << ((64 - bits) as u64))
            }),
        ensures
            ((merged >> b) & 1u64) == (if b + bits < 64 {
                (low >> ((b + bits) as u64)) & 1u64
            } else {
                (high >> ((b + bits - 64) as u64)) & 1u64
            }),
    {
        assert(((merged >> b) & 1u64) == (if b + bits < 64 {
            (low >> ((b + bits) as u64)) & 1u64
        } else {
            (high >> ((b + bits - 64) as u64)) & 1u64
        })) by (bit_vector)
            requires
                b < 64,
                bits < 64,
                merged == (if bits == 0 {
                    low
                } else {
                    (low >> bits) | (high << ((64 - bits) as u64))
                }),
        ;
    }

    proof fn lemma_zero_word_bit(pos: u64)
        requires
            pos < 64,
        ensures
            ((0u64 >> pos) & 1u64) == 0u64,
    {
        assert(((0u64 >> pos) & 1u64) == 0u64) by (bit_vector)
            requires
                pos < 64,
        ;
    }

    fn merge_words(low: u64, high: u64, bits: u64) -> (merged: u64)
        requires
            bits < 64,
        ensures
            forall|b: u64|
                b < 64 ==> #[trigger] word_bit(merged, b as int) == (if b + bits < 64 {
                    word_bit(low, (b + bits) as int)
                } else {
                    word_bit(high, (b + bits - 64) as int)
                }),
    {
        let merged = if bits == 0 {
            low
        } else {
            (low >> bits) | (high << (64 - bits))
        };
        proof {
            assert forall|b: u64| b < 64 implies #[trigger] word_bit(merged, b as int) == (if b
                + bits < 64 {
                word_bit(low, (b + bits) as int)
            } else {
                word_bit(high, (b + bits - 64) as int)
            }) by {
                Self::lemma_shift_word_bit(low, high, merged, bits, b);
            }
        }
        merged
    }

    proof fn lemma_shift_word_step(
        orig: Self,
        before: Self,
        after: Self,
        dest: usize,
        distance: usize,
        whole: usize,
        bits: u64,
        low: u64,
        high: u64,
        merged: u64,
    )
        requires
            dest < BITMAP_WORDS,
            distance <= BITMAP_CAPACITY,
            whole == distance / 64,
            bits == distance % 64,
            low == (if dest + whole < BITMAP_WORDS {
                orig.words[(dest + whole) as int]
            } else {
                0
            }),
            high == (if dest + whole + 1 < BITMAP_WORDS {
                orig.words[(dest + whole + 1) as int]
            } else {
                0
            }),
            forall|b: u64|
                b < 64 ==> #[trigger] word_bit(merged, b as int) == (if b + bits < 64 {
                    word_bit(low, (b + bits) as int)
                } else {
                    word_bit(high, (b + bits - 64) as int)
                }),
            after.words@ == before.words@.update(dest as int, merged),
            forall|i: int|
                0 <= i < dest * 64 ==> #[trigger] before.bit_set(i) == orig.bit_set(i + distance),
        ensures
            forall|i: int|
                0 <= i < (dest + 1) * 64 ==> #[trigger] after.bit_set(i) == orig.bit_set(
                    i + distance,
                ),
    {
        assert forall|i: int| 0 <= i < (dest + 1) * 64 implies #[trigger] after.bit_set(i)
            == orig.bit_set(i + distance) by {
            if i < dest * 64 {
                assert(i / 64 < dest);
                assert(after.words[i / 64] == before.words[i / 64]);
                assert(before.bit_set(i) == after.bit_set(i));
            } else {
                let b = (i % 64) as u64;
                assert(i / 64 == dest);
                assert(after.words[dest as int] == merged);
                assert(word_bit(merged, b as int) == (if b + bits < 64 {
                    word_bit(low, (b + bits) as int)
                } else {
                    word_bit(high, (b + bits - 64) as int)
                }));
                let shifted = i + distance;
                assert(shifted == (dest + whole) * 64 + (b + bits));
                if b + bits < 64 {
                    assert(shifted / 64 == dest + whole);
                    assert(shifted % 64 == b + bits);
                    if dest + whole >= BITMAP_WORDS {
                        assert(shifted >= BITMAP_CAPACITY);
                        Self::lemma_zero_word_bit((b + bits) as u64);
                    }
                } else {
                    assert(shifted / 64 == dest + whole + 1);
                    assert(shifted % 64 == b + bits - 64);
                    if dest + whole + 1 >= BITMAP_WORDS {
                        assert(shifted >= BITMAP_CAPACITY);
                        Self::lemma_zero_word_bit((b + bits - 64) as u64);
                    }
                }
            }
        }
    }

    fn shift(&mut self, distance: usize)
        requires
            distance <= BITMAP_CAPACITY,
        ensures
            forall|i: int|
                0 <= i ==> #[trigger] final(self).bit_set(i) == old(self).bit_set(i + distance),
    {
        let whole = distance / 64;
        let bits = (distance % 64) as u64;
        let mut dest = 0;
        while dest < BITMAP_WORDS
            invariant
                dest <= BITMAP_WORDS,
                distance <= BITMAP_CAPACITY,
                whole == distance / 64,
                bits == distance % 64,
                forall|w: int|
                    dest <= w < BITMAP_WORDS ==> #[trigger] self.words[w] == old(self).words[w],
                forall|i: int|
                    0 <= i < dest * 64 ==> #[trigger] self.bit_set(i) == old(self).bit_set(
                        i + distance,
                    ),
            decreases BITMAP_WORDS - dest,
        {
            let src = dest + whole;
            let low = if src < BITMAP_WORDS {
                self.words[src]
            } else {
                0
            };
            let high = if src + 1 < BITMAP_WORDS {
                self.words[src + 1]
            } else {
                0
            };
            let merged = Self::merge_words(low, high, bits);
            let ghost before = *self;
            self.words[dest] = merged;
            proof {
                Self::lemma_shift_word_step(
                    *old(self),
                    before,
                    *self,
                    dest,
                    distance,
                    whole,
                    bits,
                    low,
                    high,
                    merged,
                );
            }
            dest += 1;
        }
    }

    pub fn new() -> (r: Self)
        ensures
            forall|i: int| !#[trigger] r.bit_set(i),
    {
        assert forall|b: u64| #![auto] b < 64 implies ((0u64 >> b) & 1u64) != 1u64 by {
            assert(((0u64 >> b) & 1u64) != 1u64) by (bit_vector)
                requires
                    b < 64,
            ;
        }
        Self { words: [0;BITMAP_WORDS] }
    }

    pub fn mark_and_advance(&mut self, confirmed_seq: &mut u64, completed_seq: u64) -> (advanced:
        u64)
        requires
            old(self).well_formed(),
            *old(confirmed_seq) <= u64::MAX - BITMAP_CAPACITY,
            *old(confirmed_seq) < completed_seq <= *old(confirmed_seq) + BITMAP_CAPACITY,
        ensures
            final(self).well_formed(),
            advanced <= BITMAP_CAPACITY,
            *final(confirmed_seq) == *old(confirmed_seq) + advanced,
            forall|i: int|
                0 <= i < advanced ==> #[trigger] old(self).bit_set(i) || *old(confirmed_seq) + 1 + i
                    == completed_seq,
            forall|i: int|
                0 <= i ==> #[trigger] final(self).bit_set(i) == (old(self).bit_set(i + advanced)
                    || *final(confirmed_seq) + 1 + i == completed_seq),
    {
        self.mark((completed_seq - *confirmed_seq - 1) as usize);
        let n = self.prefix();
        assert forall|i: int| 0 <= i < n implies #[trigger] old(self).bit_set(i) || *confirmed_seq
            + 1 + i == completed_seq by {
            assert(self.bit_set(i));
        }
        if n != 0 {
            self.shift(n);
        }
        *confirmed_seq += n as u64;
        n as u64
    }
}

pub fn apply_bitmap_completion(
    bitmap: &mut SlidingBitmap,
    table: &mut TimelineTable,
    t: u8,
    s: u64,
    Tracked(observation): Tracked<Knows>,
    Tracked(ledger): Tracked<&mut KnowledgeLedger>,
    Ghost(node): Ghost<NodeId>,
) -> (advanced: u64)
    requires
        old(table).well_formed(observation.instance_id(), node, old(ledger)),
        old(table).issued(t, s),
        old(table).confirmed_seq[t as int] < s <= old(table).confirmed_seq[t as int]
            + BITMAP_CAPACITY,
        old(table).observed(t, s, node, observation),
        old(bitmap).well_formed(),
        old(bitmap).observations(old(table), t, node, old(ledger)),
    ensures
        final(table).well_formed(observation.instance_id(), node, final(ledger)),
        final(bitmap).well_formed(),
        final(bitmap).observations(final(table), t, node, final(ledger)),
        final(table).confirmed_seq[t as int] == old(table).confirmed_seq[t as int] + advanced,
        advanced <= BITMAP_CAPACITY,
        final(table).confirmed_seq@ == old(table).confirmed_seq@.update(
            t as int,
            (old(table).confirmed_seq[t as int] + advanced) as u64,
        ),
        final(table).events == old(table).events,
        final(table).next_seq == old(table).next_seq,
        final(table).kinds == old(table).kinds,
        final(table).count == old(table).count,
        final(ledger).submitted == old(ledger).submitted,
        final(ledger).succeeded == old(ledger).succeeded,
        final(ledger).revoked == old(ledger).revoked,
        final(ledger).knows.set() == old(ledger).knows.set().union(observation.set()),
        forall|i: int|
            0 <= i < advanced ==> #[trigger] old(bitmap).bit_set(i) || old(
                table,
            ).confirmed_seq[t as int] + 1 + i == s,
        forall|i: int|
            0 <= i ==> #[trigger] final(bitmap).bit_set(i) == (old(bitmap).bit_set(i + advanced)
                || final(table).confirmed_seq[t as int] + 1 + i == s),
{
    let mut prefix = table.confirmed_seq[t as usize];
    let advanced = bitmap.mark_and_advance(&mut prefix, s);
    let ghost base = old(table).confirmed_seq[t as int];
    assert forall|q: u64| base < q <= prefix implies #[trigger] table.observed(
        t,
        q,
        node,
        observation,
    ) || table.observed(t, q, node, ledger.knows) by {
        if q != s {
            assert(old(bitmap).bit_set(q - base - 1));
        }
    }
    proof {
        if prefix > base && prefix != s {
            assert(old(bitmap).bit_set(prefix - base - 1));
        }
    }
    table.advance_confirmed(t, prefix, Tracked(observation), Tracked(ledger), Ghost(node));
    advanced
}

} // verus!
