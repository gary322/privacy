from __future__ import annotations

from uvcc_party.fss_block import (
    FSSBlockV1,
    FSSRecordV1,
    fss_id_v1,
    fssblock_hash32_v1,
    build_fss_block_v1,
)


def test_fss_id_v1_is_deterministic() -> None:
    job_nonce32 = b"\x01" * 32
    a = fss_id_v1(job_nonce32=job_nonce32, epoch=1, step=2, sgir_op_id=3, lane=0xFFFF, fss_kind=2)
    b = fss_id_v1(job_nonce32=job_nonce32, epoch=1, step=2, sgir_op_id=3, lane=0xFFFF, fss_kind=2)
    c = fss_id_v1(job_nonce32=job_nonce32, epoch=1, step=2, sgir_op_id=3, lane=0xFFFF, fss_kind=1)
    assert a == b
    assert a != c


def test_fss_block_v1_build_parse_lookup() -> None:
    kb0 = b"key0"
    kb1 = b"key1-longer"
    # Intentionally out of order: builder must sort by (fss_id,fss_kind,sgir_op_id,lane).
    r1 = FSSRecordV1(
        fss_id=200,
        sgir_op_id=7,
        lane=0,
        fss_kind=2,
        domain_bits=8,
        range_bits=1,
        share_mode=1,
        edge_mode=1,
        edge_id=0,
        flags=1,
        key_bytes=0,
        key_offset=0,
        key_hash_hi=0,
        key_hash_lo=0,
    )
    r0 = FSSRecordV1(
        fss_id=100,
        sgir_op_id=7,
        lane=0xFFFF,
        fss_kind=1,
        domain_bits=8,
        range_bits=1,
        share_mode=1,
        edge_mode=1,
        edge_id=0,
        flags=1,
        key_bytes=0,
        key_offset=0,
        key_hash_hi=0,
        key_hash_lo=0,
    )
    block = build_fss_block_v1(epoch=2, step=3, records=[r1, r0], key_blobs=[kb1, kb0])
    b = FSSBlockV1(block)
    assert b.header.epoch == 2
    assert b.header.step == 3
    assert b.header.record_count == 2
    assert b.find_record(fss_id=100) is not None
    assert b.find_record(fss_id=200) is not None
    assert b.key_blob(fss_id=100) == kb0
    assert b.key_blob(fss_id=200) == kb1
    h = fssblock_hash32_v1(block)
    assert isinstance(h, (bytes, bytearray)) and len(h) == 32


