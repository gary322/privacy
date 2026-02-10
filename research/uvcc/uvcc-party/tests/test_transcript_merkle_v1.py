from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_805dc0ce43c13243,uvcc_group_a215f6bfd7a10303,uvcc_group_35ac5da9edf180e2

import hashlib

from uvcc_party.transcript import (
    DS_EMPTY_EPOCH,
    DS_FINAL,
    DS_LEAF,
    DS_NODE,
    SegmentDescV1,
    TranscriptStoreV1,
    merkle_root_v1,
    sha256,
)


def test_merkle_root_v1_empty_and_singleton() -> None:
    assert merkle_root_v1([]) == sha256(DS_EMPTY_EPOCH)
    h = b"\xAA" * 32
    assert merkle_root_v1([h]) == sha256(DS_NODE + h + h)


def test_transcript_epoch_root_is_order_independent_via_sorting() -> None:
    job_id32 = b"\x99" * 32
    ts = TranscriptStoreV1(job_id32=job_id32)

    # Create two leaves with different ordering keys, insert in reverse order.
    segs = (SegmentDescV1(seg_kind=10, object_id=1, sub_id=0, dtype=9, offset=0, length=8, fxp_frac_bits=0),)
    l2 = ts.record_frame(
        epoch=0,
        step=2,
        round=0,
        msg_kind=0x0102,
        sender=1,
        receiver=2,
        dir=0,
        seq_no=0,
        payload_bytes=8,
        payload_hash32=hashlib.sha256(b"p2").digest(),
        header_hash32=hashlib.sha256(b"h2").digest(),
        segments=segs,
    )
    l1 = ts.record_frame(
        epoch=0,
        step=1,
        round=0,
        msg_kind=0x0102,
        sender=0,
        receiver=1,
        dir=0,
        seq_no=0,
        payload_bytes=8,
        payload_hash32=hashlib.sha256(b"p1").digest(),
        header_hash32=hashlib.sha256(b"h1").digest(),
        segments=segs,
    )

    # Epoch root should be based on sorted leaves (l1 then l2), independent of insertion order.
    want = merkle_root_v1([l1.leaf_hash32, l2.leaf_hash32])
    got = ts.epoch_root(epoch=0)
    assert got == want

    # Final root with epoch_count=1 must match DS_FINAL + E + epoch_root[0].
    fr = ts.final_root(epoch_count=1)
    assert fr == sha256(DS_FINAL + (1).to_bytes(4, "little") + got)


def test_transcript_exactly_once_acceptance_by_ordering_key() -> None:
    job_id32 = b"\x55" * 32
    ts = TranscriptStoreV1(job_id32=job_id32)
    segs = (SegmentDescV1(seg_kind=10, object_id=1, sub_id=0, dtype=9, offset=0, length=8, fxp_frac_bits=0),)
    payload_hash32 = hashlib.sha256(b"payload").digest()
    header_hash32 = hashlib.sha256(b"header").digest()

    l1 = ts.record_frame(
        epoch=0,
        step=1,
        round=0,
        msg_kind=0x0102,
        sender=0,
        receiver=1,
        dir=0,
        seq_no=7,
        payload_bytes=8,
        payload_hash32=payload_hash32,
        header_hash32=header_hash32,
        segments=segs,
    )
    l2 = ts.record_frame(
        epoch=0,
        step=1,
        round=0,
        msg_kind=0x0102,
        sender=0,
        receiver=1,
        dir=0,
        seq_no=7,
        payload_bytes=8,
        payload_hash32=payload_hash32,
        header_hash32=header_hash32,
        segments=segs,
    )
    assert l2 is l1
    assert len(ts.leaves()) == 1

    try:
        ts.record_frame(
            epoch=0,
            step=1,
            round=0,
            msg_kind=0x0102,
            sender=0,
            receiver=1,
            dir=0,
            seq_no=7,
            payload_bytes=8,
            payload_hash32=hashlib.sha256(b"payload2").digest(),
            header_hash32=header_hash32,
            segments=segs,
        )
        raise AssertionError("expected duplicate leaf key mismatch to raise")
    except ValueError:
        pass


