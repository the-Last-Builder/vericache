use vericache_core::spec::ObjectKey;

/// Deterministically map a 64-bit trace `hash_id` to a 256-bit VeriCache `ObjectKey`.
#[inline]
pub fn key_from_hash_id(hash_id: u64) -> ObjectKey {
    let h1 = splitmix64(hash_id ^ 0x9e37_79b9_7f4a_7c15);
    let h2 = splitmix64(h1 ^ 0xbf58_476d_1ce4_e5b9);
    let h3 = splitmix64(h2 ^ 0x94d0_49bb_1331_11eb);
    ObjectKey([hash_id, h1, h2, h3])
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Key-hash multi-engine shard router driving independent `LocalEngine` or
/// `DistributedEngine` shards with disjoint key spaces.
pub struct ShardRouter<T> {
    shards: Vec<T>,
}

impl<T> ShardRouter<T> {
    pub fn new(shards: Vec<T>) -> Self {
        assert!(!shards.is_empty(), "ShardRouter requires at least 1 shard");
        Self { shards }
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    #[inline]
    pub fn shard_index(&self, key: &ObjectKey) -> usize {
        let mixed = splitmix64(key.0[0] ^ key.0[1]);
        (mixed as usize) % self.shards.len()
    }

    #[inline]
    pub fn shard_for(&self, key: &ObjectKey) -> &T {
        let idx = self.shard_index(key);
        &self.shards[idx]
    }

    #[inline]
    pub fn shard_for_mut(&mut self, key: &ObjectKey) -> &mut T {
        let idx = self.shard_index(key);
        &mut self.shards[idx]
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.shards.iter_mut()
    }
}

/// Deterministic payload checksum header + trailer for Suite 4 integrity verification.
pub const FRAME_MAGIC: u64 = 0x5643_4143_4845_3034; // "VCACHE04"

/// Stamp a frame buffer of length `>= 32` with `(magic, hash_id, version_seq)` header,
/// deterministic body bytes, and trailing 8-byte FNV-1a checksum.
pub fn stamp_checksummed_frame(buf: &mut [u8], hash_id: u64, version_seq: u64) {
    assert!(buf.len() >= 32, "checksummed frame must be >= 32 bytes");
    buf[0..8].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
    buf[8..16].copy_from_slice(&hash_id.to_le_bytes());
    buf[16..24].copy_from_slice(&version_seq.to_le_bytes());
    let body_end = buf.len() - 8;
    // Fast 8-byte deterministic fill so large frames stamp at memory bandwidth.
    let mut state = splitmix64(hash_id ^ version_seq.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut off = 24;
    while off + 8 <= body_end {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        buf[off..off + 8].copy_from_slice(&state.to_le_bytes());
        off += 8;
    }
    while off < body_end {
        buf[off] = (state & 0xff) as u8;
        state >>= 8;
        off += 1;
    }
    let csum = frame_checksum(&buf[..body_end]);
    buf[body_end..].copy_from_slice(&csum.to_le_bytes());
}

/// Verify magic, expected `hash_id` (if `Some`), and full-payload checksum.
/// Returns `Ok(version_seq)` on intact frame, `Err(reason)` on corruption/tearing.
pub fn verify_checksummed_frame(buf: &[u8], expected_hash_id: Option<u64>) -> Result<u64, &'static str> {
    if buf.len() < 32 {
        return Err("short_frame");
    }
    let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    if magic != FRAME_MAGIC {
        return Err("bad_magic");
    }
    let hash_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    if let Some(exp) = expected_hash_id {
        if hash_id != exp {
            return Err("hash_id_mismatch");
        }
    }
    let version_seq = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    let body_end = buf.len() - 8;
    let stored = u64::from_le_bytes(buf[body_end..].try_into().unwrap());
    let actual = frame_checksum(&buf[..body_end]);
    if stored != actual {
        return Err("checksum_mismatch");
    }
    Ok(version_seq)
}

#[inline]
pub fn frame_checksum(data: &[u8]) -> u64 {
    // Fast word-granularity FNV-1a-64 mixing over every 8-byte word.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut chunks = data.chunks_exact(8);
    for c in chunks.by_ref() {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h ^= w;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &b in chunks.remainder() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
