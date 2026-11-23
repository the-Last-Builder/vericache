use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub const FRAME_MAGIC: u64 = 0x5643_4143_4845_3034;

#[derive(Debug, Deserialize)]
pub struct TraceRequest {
    #[allow(dead_code)]
    pub timestamp: u64,
    #[allow(dead_code)]
    pub input_length: usize,
    #[allow(dead_code)]
    pub output_length: usize,
    pub hash_ids: Vec<u64>,
}

pub fn load_trace(path: &str, max_requests: Option<usize>) -> Vec<TraceRequest> {
    let f = File::open(path).unwrap_or_else(|e| panic!("failed to open trace {path}: {e}"));
    let reader = BufReader::new(f);
    let limit = max_requests.unwrap_or(usize::MAX);
    let mut reqs = Vec::new();
    for line in reader.lines() {
        if reqs.len() >= limit {
            break;
        }
        let line = line.expect("read trace line");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(req) = serde_json::from_str::<TraceRequest>(trimmed) {
            if !req.hash_ids.is_empty() {
                reqs.push(req);
            }
        }
    }
    assert!(!reqs.is_empty(), "trace {path} yielded 0 valid requests");
    reqs
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Deterministic 2-node 70/30 session-sticky routing + 8-shard affinity.
/// Exact bit-for-bit match with `vericache-bench/src/suites/scaleup_cluster.rs`.
#[inline]
pub fn route_request(
    req_idx: usize,
    hash_ids: &[u64],
    shards_per_node: usize,
    sticky_pct: u64,
) -> (u64, usize) {
    let pos = if hash_ids.len() > 12 {
        12
    } else {
        hash_ids.len() - 1
    };
    let sk = hash_ids[pos];
    let h = splitmix64(sk);
    let home_node = h & 1;
    let shard_id = ((h >> 1) as usize) % shards_per_node;
    let drift_hash = splitmix64((req_idx as u64) ^ sk.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let node_id = if (drift_hash % 100) < sticky_pct {
        home_node
    } else {
        1 - home_node
    };
    (node_id, shard_id)
}

#[inline]
pub fn should_revoke(op_idx: usize, revoke_rate: f64) -> bool {
    if revoke_rate <= 0.0 {
        return false;
    }
    let period = ((1.0 / revoke_rate).round() as usize).max(1);
    ((op_idx + 1) % period) == 0
}

pub fn stamp_checksummed_frame(arr: &mut [u64], hash_id: u64, version_seq: u64) {
    let n = arr.len();
    assert!(n >= 4);
    arr[0] = FRAME_MAGIC;
    arr[1] = hash_id;
    arr[2] = version_seq;
    let fill = hash_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ version_seq.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    for slot in &mut arr[3..n - 1] {
        *slot = fill;
    }
    let mut csum = 0u64;
    for &v in &arr[..n - 1] {
        csum ^= v;
    }
    arr[n - 1] = csum;
}

pub fn verify_checksummed_frame(arr: &[u64], expected_hash_id: u64) -> bool {
    let n = arr.len();
    if n < 4 || arr[0] != FRAME_MAGIC {
        return false;
    }
    let hid = arr[1];
    if hid != expected_hash_id {
        return false;
    }
    let ver = arr[2];
    let expected_fill =
        hid.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ver.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let mut csum = arr[0] ^ arr[1] ^ arr[2];
    for &v in &arr[3..n - 1] {
        if v != expected_fill {
            return false;
        }
        csum ^= v;
    }
    arr[n - 1] == csum
}
