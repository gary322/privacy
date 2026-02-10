from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_c3fb595c9212b029,uvcc_group_ba7afac425406f12

import base64
import dataclasses
import json
import tempfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _add_paths() -> None:
    import sys

    root = _repo_root()
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-party"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-verifier"))


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def test_verifier_roundtrip_ok() -> None:
    _add_paths()

    from uvcc_party.netframe import DT_U64, SegmentPayloadV1, build_netframe_v1, payload_hash32_v1
    from uvcc_party.proof_bundle import ProofBundlePartyV1, ProofBundleSignatureV1, ProofBundleV1, party_from_privkey, sign_final_root_v1
    from uvcc_party.transcript import SegmentDescV1, TranscriptStoreV1
    from uvcc_party.eip712 import EIP712DomainV1
    from uvcc_verifier.proof_bundle_v1 import parse_proof_bundle_json_v1, verify_proof_bundle_v1
    from uvcc_verifier.transcript_v1 import compute_epoch_roots_v1, compute_final_root_v1, parse_transcript_jsonl_v1, validate_transcript_leaves_v1

    job_id32 = b"\x11" * 32
    ts = TranscriptStoreV1(job_id32=job_id32)

    # Create a single NetFrame and record a SEND leaf for it (mimics Party.send_netframe).
    frame = build_netframe_v1(
        job_id32=job_id32,
        epoch=0,
        step=0,
        round=0,
        msg_kind=0x0102,
        flags=0,
        sender=0,
        receiver=1,
        seq_no=0,
        segments=[SegmentPayloadV1(seg_kind=10, object_id=7, sub_id=0, dtype=DT_U64, fxp_frac_bits=0, payload=b"\x00" * 8)],
    )
    segs = tuple(
        SegmentDescV1(
            seg_kind=int(s.seg_kind),
            object_id=int(s.object_id),
            sub_id=int(s.sub_id),
            dtype=int(s.dtype),
            offset=int(s.offset),
            length=int(s.length),
            fxp_frac_bits=int(s.fxp_frac_bits),
        )
        for s in frame.segments
    )
    leaf = ts.record_frame(
        epoch=int(frame.header.epoch),
        step=int(frame.header.step),
        round=int(frame.header.round),
        msg_kind=int(frame.header.msg_kind),
        sender=int(frame.header.sender),
        receiver=int(frame.header.receiver),
        dir=0,
        seq_no=int(frame.header.seq_no),
        payload_bytes=int(frame.header.payload_bytes),
        payload_hash32=payload_hash32_v1(frame),
        header_hash32=bytes(frame.header.header_hash32),
        segments=segs,
    )

    epoch_roots = [ts.epoch_root(epoch=0)]
    final_root32 = ts.final_root(epoch_count=1)
    assert final_root32 == compute_final_root_v1(epoch_roots=epoch_roots)

    policy_hash32 = b"\x22" * 32
    sgir_hash32 = b"\x33" * 32
    runtime_hash32 = b"\x44" * 32
    result_hash32 = b"\x55" * 32

    privs = {0: b"\x01" * 32, 1: b"\x02" * 32, 2: b"\x03" * 32}
    parties = [party_from_privkey(party_id=i, privkey32=privs[i]) for i in (0, 1, 2)]
    dom = EIP712DomainV1(chain_id=31337, verifying_contract=b"\x00" * 20)
    sigs = [
        sign_final_root_v1(
            party_id=i,
            privkey32=privs[i],
            policy_hash32=policy_hash32,
            final_root32=final_root32,
            result_hash32=result_hash32,
            job_id32=job_id32,
            eip712_domain=dom,
        )
        for i in (0, 1, 2)
    ]

    pb = ProofBundleV1(
        uvcc_version="1.0",
        job_id32=job_id32,
        policy_hash32=policy_hash32,
        eip712_domain=dom,
        sgir_hash32=sgir_hash32,
        runtime_hash32=runtime_hash32,
        backend="CRYPTO_CC_3PC",
        parties=parties,
        epoch_roots=epoch_roots,
        final_root32=final_root32,
        signatures=sigs,
        result_hash32=result_hash32,
        status="OK",
    )
    proof_json = pb.to_json_bytes()
    proof = parse_proof_bundle_json_v1(proof_json)

    # Dump transcript jsonl
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.jsonl"
        rec = {"body_b64": _b64(leaf.body_bytes), "leaf_hash_hex": "0x" + leaf.leaf_hash32.hex()}
        path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

        leaves = parse_transcript_jsonl_v1(str(path))
        validate_transcript_leaves_v1(leaves, strict_unknown_msg_kind=True, strict_netframe_header_hash=True)
        roots_by_epoch = compute_epoch_roots_v1(leaves)
        assert roots_by_epoch[0] == epoch_roots[0]
        final2 = compute_final_root_v1(epoch_roots=[roots_by_epoch[0]])

        verify_proof_bundle_v1(proof=proof, transcript_epoch_roots=[roots_by_epoch[0]], transcript_final_root32=final2)


def test_verifier_rejects_bad_signature() -> None:
    _add_paths()

    from uvcc_party.proof_bundle import ProofBundleV1, party_from_privkey, sign_final_root_v1
    from uvcc_party.transcript import TranscriptStoreV1
    from uvcc_party.eip712 import EIP712DomainV1
    from uvcc_verifier.proof_bundle_v1 import parse_proof_bundle_json_v1, verify_proof_bundle_v1

    job_id32 = b"\x11" * 32
    ts = TranscriptStoreV1(job_id32=job_id32)
    epoch_roots = [ts.epoch_root(epoch=0)]
    final_root32 = ts.final_root(epoch_count=1)
    policy_hash32 = b"\x22" * 32
    result_hash32 = b"\x55" * 32
    dom = EIP712DomainV1(chain_id=31337, verifying_contract=b"\x00" * 20)

    privs = {0: b"\x01" * 32, 1: b"\x02" * 32, 2: b"\x03" * 32}
    parties = [party_from_privkey(party_id=i, privkey32=privs[i]) for i in (0, 1, 2)]
    sigs = [
        sign_final_root_v1(
            party_id=i,
            privkey32=privs[i],
            policy_hash32=policy_hash32,
            final_root32=final_root32,
            result_hash32=result_hash32,
            job_id32=job_id32,
            eip712_domain=dom,
        )
        for i in (0, 1, 2)
    ]
    # Corrupt one signature byte.
    s0 = bytearray(sigs[0].sig65)
    s0[0] ^= 1
    sigs = [dataclasses.replace(sigs[0], sig65=bytes(s0)), sigs[1], sigs[2]]

    pb = ProofBundleV1(
        uvcc_version="1.0",
        job_id32=job_id32,
        policy_hash32=policy_hash32,
        eip712_domain=dom,
        sgir_hash32=b"\x33" * 32,
        runtime_hash32=b"\x44" * 32,
        backend="CRYPTO_CC_3PC",
        parties=parties,
        epoch_roots=epoch_roots,
        final_root32=final_root32,
        signatures=sigs,
        result_hash32=result_hash32,
        status="OK",
    )
    proof = parse_proof_bundle_json_v1(pb.to_json_bytes())
    with pytest.raises(ValueError):
        verify_proof_bundle_v1(proof=proof, transcript_epoch_roots=epoch_roots, transcript_final_root32=final_root32)


