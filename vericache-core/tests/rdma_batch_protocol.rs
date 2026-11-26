use vericache_core::backends::rdma::protocol::{
    decode_control_payload, encode_control_payload, ControlPayload, CONTROL_TAG_FETCH_BATCH,
    MAX_FETCH_BATCH,
};
use vericache_core::spec::ObjectKey;

#[test]
fn fetch_batch_wire_order_and_little_endian_fields() {
    let entries = vec![
        (0x0123_4567_89ab_cdef, ObjectKey([u64::MAX, 0, 17, 42])),
        (u64::MAX, ObjectKey([9, 10, 11, 12])),
    ];
    let mut wire = vec![CONTROL_TAG_FETCH_BATCH];
    for (id, key) in &entries {
        wire.extend(id.to_le_bytes());
        for word in key.0 {
            wire.extend(word.to_le_bytes());
        }
    }
    assert_eq!(
        wire,
        encode_control_payload(ControlPayload::FetchBatch {
            entries: entries.clone()
        })
    );
    assert_eq!(
        decode_control_payload(&wire),
        Some(ControlPayload::FetchBatch { entries })
    );
    assert_eq!(decode_control_payload(&wire[..wire.len() - 1]), None);
    assert_eq!(decode_control_payload(&wire[..1]), None);
    assert_eq!(decode_control_payload(&[0xff]), None);
}

#[test]
fn fetch_batch_bound_preserves_legacy_fetch_wire_format() {
    let key = ObjectKey([1, 2, 3, 4]);
    let max = vec![(7, key); MAX_FETCH_BATCH];
    let at_bound = encode_control_payload(ControlPayload::FetchBatch {
        entries: max.clone(),
    });
    assert_eq!(
        decode_control_payload(&at_bound),
        Some(ControlPayload::FetchBatch { entries: max })
    );
    let too_many = encode_control_payload(ControlPayload::FetchBatch {
        entries: vec![(7, key); MAX_FETCH_BATCH + 1],
    });
    assert_eq!(decode_control_payload(&too_many), None);

    let single = encode_control_payload(ControlPayload::FetchRequest { request_id: 7, key });
    assert_eq!(single.len(), 41);
    assert_eq!(single[0], 1);
    assert_eq!(
        decode_control_payload(&single),
        Some(ControlPayload::FetchRequest { request_id: 7, key })
    );
}
