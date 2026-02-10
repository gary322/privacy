from __future__ import annotations

import os

from uvcc_party.netframe import (
    DT_U32,
    SEG_OPEN_SHARE_LO,
    SEG_PAD,
    SegmentPayloadV1,
    build_netframe_v1,
    parse_netframe_v1,
    payload_hash32_v1,
    relay_msg_id_v1,
)


def test_netframe_build_parse_roundtrip_and_padding() -> None:
    job_id32 = b"\x11" * 32
    sid_hash32 = b"\x22" * 32

    segs = [
        SegmentPayloadV1(seg_kind=SEG_OPEN_SHARE_LO, object_id=5, sub_id=0, dtype=DT_U32, fxp_frac_bits=0, payload=b"\xAA\xBB\xCC"),
        SegmentPayloadV1(seg_kind=SEG_OPEN_SHARE_LO, object_id=6, sub_id=0, dtype=DT_U32, fxp_frac_bits=0, payload=b"\xDD\xEE"),
    ]
    fr = build_netframe_v1(
        job_id32=job_id32,
        epoch=7,
        step=8,
        round=9,
        msg_kind=0x0101,
        flags=0,
        sender=0,
        receiver=1,
        seq_no=0,
        segments=segs,
    )
    raw = fr.to_bytes()
    fr2 = parse_netframe_v1(raw)
    assert fr2.header.job_id32 == job_id32
    assert fr2.header.epoch == 7
    assert fr2.header.step == 8
    assert fr2.header.round == 9
    assert fr2.header.msg_kind == 0x0101
    assert fr2.header.sender == 0
    assert fr2.header.receiver == 1
    assert fr2.header.seq_no == 0
    assert fr2.header.payload_bytes == len(fr.payload)

    # Should have inserted a PAD segment between the two payloads because first is 3 bytes.
    seg_kinds = [s.seg_kind for s in fr2.segments]
    assert SEG_PAD in seg_kinds
    # Payload hash should match.
    assert payload_hash32_v1(fr2) == payload_hash32_v1(fr)

    # Relay msg_id must be derivable without seeing payload bytes (only metadata + sid/job hashes).
    mid = relay_msg_id_v1(
        domain=b"uvcc.netframe.relay.v1",
        sid_hash32=sid_hash32,
        job_id32=job_id32,
        epoch=7,
        step=8,
        round=9,
        msg_kind=0x0101,
        sender=0,
        receiver=1,
        seq_no=0,
        frame_no=0,
    )
    assert isinstance(mid, str) and len(mid) == 64

    # Changing payload must not change msg_id.
    segs2 = [
        SegmentPayloadV1(seg_kind=SEG_OPEN_SHARE_LO, object_id=5, sub_id=0, dtype=DT_U32, fxp_frac_bits=0, payload=os.urandom(3)),
        SegmentPayloadV1(seg_kind=SEG_OPEN_SHARE_LO, object_id=6, sub_id=0, dtype=DT_U32, fxp_frac_bits=0, payload=os.urandom(2)),
    ]
    fr3 = build_netframe_v1(
        job_id32=job_id32,
        epoch=7,
        step=8,
        round=9,
        msg_kind=0x0101,
        flags=0,
        sender=0,
        receiver=1,
        seq_no=0,
        segments=segs2,
    )
    mid2 = relay_msg_id_v1(
        domain=b"uvcc.netframe.relay.v1",
        sid_hash32=sid_hash32,
        job_id32=job_id32,
        epoch=7,
        step=8,
        round=9,
        msg_kind=0x0101,
        sender=0,
        receiver=1,
        seq_no=0,
        frame_no=0,
    )
    assert mid2 == mid
    assert payload_hash32_v1(fr3) != payload_hash32_v1(fr)


