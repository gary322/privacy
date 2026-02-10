from __future__ import annotations

import hashlib

from uvcc_party.dpf_dcf import PRG_CHACHA12, PRIM_DCF, keygen_dpf_dcf_keyrecs_v1
from uvcc_party.fss_dir import FSSDirEntryV1, FSSDirectoryV1, build_fss_dir_v1


def test_fss_dir_v1_build_parse_lookup_keyrec() -> None:
    sid = b"sid-fss-dir"
    sid_hash32 = hashlib.sha256(sid).digest()
    master_seed32 = b"\x44" * 32

    # Build two different DCF keyrecs (edge keys are out of scope here; we just need stable bytes).
    k0a, _ = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=100,
        alpha=7,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=master_seed32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )
    k0b, _ = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=200,
        alpha=9,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=master_seed32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    e1 = FSSDirEntryV1(
        fss_id=200,
        prim_type=PRIM_DCF,
        w=8,
        out_kind=0,
        prg_id=PRG_CHACHA12,
        keyrec_off=0,
        keyrec_len=0,
        aux_off=0,
        aux_len=0,
        stream_id32=0,
    )
    e0 = FSSDirEntryV1(
        fss_id=100,
        prim_type=PRIM_DCF,
        w=8,
        out_kind=0,
        prg_id=PRG_CHACHA12,
        keyrec_off=0,
        keyrec_len=0,
        aux_off=0,
        aux_len=0,
        stream_id32=0,
    )

    blob = build_fss_dir_v1(
        party_id=0,
        sid_hash32=sid_hash32,
        epoch=1,
        entries=[
            (e1, k0b, None),
            (e0, k0a, b"aux"),
        ],
    )

    d = FSSDirectoryV1(blob)
    assert d.header.party_id == 0
    assert d.header.sid_hash32 == sid_hash32
    assert d.header.entry_count == 2

    # Sorted lookup
    assert d.find_entry(fss_id=100) is not None
    assert d.find_entry(fss_id=200) is not None
    assert d.find_entry(fss_id=9999) is None

    assert d.keyrec_bytes(fss_id=100) == k0a
    assert d.keyrec_bytes(fss_id=200) == k0b
    assert d.aux_bytes(fss_id=100) == b"aux"
    assert d.aux_bytes(fss_id=200) is None


