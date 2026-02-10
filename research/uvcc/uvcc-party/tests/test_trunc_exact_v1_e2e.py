from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_4bf28fa973d7eb82,uvcc_group_25803b2067d6fd77,uvcc_group_f4494bb8307a6da6

import concurrent.futures as cf
import hashlib
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from uvcc_party.b2a import build_b2a_packs_det_v1
from uvcc_party.dpf_dcf import EDGE_01, PRG_CHACHA12, PRIM_DCF, PRIM_DPF, keygen_dpf_dcf_keyrecs_v1
from uvcc_party.gf2_triples import generate_gf2_triples_packs_v1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import make_rss_arith_u64_triple
from uvcc_party.trunc import (
    MSG_TRUNC_CARRY_RESULT,
    MSG_TRUNC_OPEN_ARITH_RESULT,
    MSG_TRUNC_OPEN_ARITH_SEND,
    MSG_TRUNC_OUTPUT_COMMIT,
    TruncFSSKeysV1,
    op_trunc_exact_v1,
    op_trunc_prob_v1,
    parse_trunc_pack_v1,
    trunc_fss_id_carry_v1,
    trunc_fss_id_eq_chunk_v1,
    trunc_fss_id_lt_chunk_v1,
)


def _sid_hash32(sid: bytes) -> bytes:
    return hashlib.sha256(sid).digest()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _start_relay(port: int, db_path: str) -> subprocess.Popen:
    repo_root = Path(__file__).resolve().parents[4]
    relay_py = repo_root / "research" / "uvcc" / "uvcc-relay" / "relay_server.py"
    assert relay_py.exists()
    return subprocess.Popen(
        [
            sys.executable,
            str(relay_py),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--db",
            db_path,
            "--require-token",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_health(base_url: str) -> None:
    rc = RelayClient(base_url=base_url, group_id="health", token=None, timeout_s=2.0)
    for _ in range(200):
        try:
            rc.healthz()
            return
        except Exception:
            time.sleep(0.02)
    raise RuntimeError("relay never became healthy")


_TRHDR = struct.Struct("<8sHHBBBB32sIQI")


def _pack_u64_pair(lo_u64: int, hi_u64: int) -> bytes:
    return (int(lo_u64) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False) + (int(hi_u64) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)


def _build_trunc_pack_blobs_det(
    *,
    sid: bytes,
    sgir_op_id: int,
    base_fss_id: int,
    f_bits: int,
    signed_mode: bool,
    lanes: int,
    seed: int,
) -> Tuple[Dict[int, bytes], List[int], List[int]]:
    """
    Deterministic TRUNC pack builder for tests.
    Returns per-party pack blobs and the public secrets (R0_u64[lanes], R_u64[lanes]).
    """
    sid_hash32 = _sid_hash32(sid)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    F = int(f_bits)
    assert 0 <= F <= 63
    maskF = (1 << F) - 1 if F > 0 else 0

    r0_pub = [int(torch.randint(0, 1 << F, (1,), dtype=torch.int64, generator=gen).item()) & maskF if F > 0 else 0 for _ in range(lanes)]
    r1_pub = []
    for _ in range(lanes):
        lo = int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item())
        hi = int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item())
        r1 = ((hi << 32) | lo) & 0xFFFFFFFFFFFFFFFF
        # IMPORTANT (spec): R1 == (R >> F), so R1 must fit in (64-F) bits to avoid overflow on (R1<<F).
        if F != 0:
            r1 &= (1 << (64 - F)) - 1
        r1_pub.append(int(r1))
    r_pub = [((r1_pub[i] << F) + r0_pub[i]) & 0xFFFFFFFFFFFFFFFF for i in range(lanes)]

    # Share R1 and R0 into 3 additive components, then derive R comps as (R1<<F)+R0 componentwise.
    comps: Dict[str, List[Tuple[int, int, int]]] = {"R1": [], "R0": [], "R": []}
    for i in range(lanes):
        a1 = int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) | (int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) << 32)
        b1 = int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) | (int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) << 32)
        a1 &= 0xFFFFFFFFFFFFFFFF
        b1 &= 0xFFFFFFFFFFFFFFFF
        c1 = (r1_pub[i] - a1 - b1) & 0xFFFFFFFFFFFFFFFF

        a0 = int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) | (int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) << 32)
        b0 = int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) | (int(torch.randint(0, 2**32, (1,), dtype=torch.int64, generator=gen).item()) << 32)
        a0 &= 0xFFFFFFFFFFFFFFFF
        b0 &= 0xFFFFFFFFFFFFFFFF
        c0 = (r0_pub[i] - a0 - b0) & 0xFFFFFFFFFFFFFFFF

        comps["R1"].append((a1, b1, c1))
        comps["R0"].append((a0, b0, c0))
        comps["R"].append((((a1 << F) + a0) & 0xFFFFFFFFFFFFFFFF, ((b1 << F) + b0) & 0xFFFFFFFFFFFFFFFF, ((c1 << F) + c0) & 0xFFFFFFFFFFFFFFFF))

    hdr = _TRHDR.pack(
        b"UVCCTRN1",
        1,
        1 if signed_mode else 0,
        64,
        int(F) & 0xFF,
        16,
        0,
        sid_hash32,
        int(sgir_op_id) & 0xFFFFFFFF,
        int(base_fss_id) & 0xFFFFFFFFFFFFFFFF,
        0,
    )

    outs: Dict[int, bytearray] = {0: bytearray(hdr), 1: bytearray(hdr), 2: bytearray(hdr)}
    for i in range(lanes):
        # For each secret, emit this party's RSS pair (comp_i, comp_{i+1})
        for name in ("R", "R1", "R0"):
            c0, c1, c2 = comps[name][i]
            pairs = {
                0: (c0, c1),
                1: (c1, c2),
                2: (c2, c0),
            }
            for pid in (0, 1, 2):
                lo, hi = pairs[pid]
                outs[pid] += _pack_u64_pair(lo, hi)

    return {pid: bytes(b) for pid, b in outs.items()}, r0_pub, r_pub


def _build_trunc_keys_det(
    *,
    sid: bytes,
    sid_hash32: bytes,
    base_fss_id: int,
    f_bits: int,
    r0_pub: List[int],
    r_pub: List[int],
    seed_edge01_32: bytes,
) -> Tuple[Dict[int, List[bytes]], Dict[int, Tuple[List[bytes], List[bytes], List[bytes], List[bytes]]], Dict[int, Tuple[List[bytes], List[bytes], List[bytes], List[bytes]]]]:
    lanes = len(r_pub)
    fss_id_carry = trunc_fss_id_carry_v1(base_fss_id)
    w_carry = 8 if int(f_bits) <= 8 else 16

    carry0: List[bytes] = []
    carry1: List[bytes] = []
    lt0 = [[], [], [], []]  # type: ignore[var-annotated]
    lt1 = [[], [], [], []]  # type: ignore[var-annotated]
    eq0 = [[], [], [], []]  # type: ignore[var-annotated]
    eq1 = [[], [], [], []]  # type: ignore[var-annotated]

    for lane in range(lanes):
        alpha_carry = int(r0_pub[lane]) & ((1 << int(f_bits)) - 1) if int(f_bits) > 0 else 0
        k0, k1 = keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=int(fss_id_carry),
            alpha=int(alpha_carry),
            w=int(w_carry),
            prg_id=PRG_CHACHA12,
            party_edge=EDGE_01,
            master_seed32=seed_edge01_32,
            prim_type=PRIM_DCF,
            dcf_invert=True,
            payload_mask_u64=1,
        )
        carry0.append(k0)
        carry1.append(k1)

        for j in range(4):
            alpha16 = int((int(r_pub[lane]) >> (16 * j)) & 0xFFFF)
            fss_id_lt = trunc_fss_id_lt_chunk_v1(base_fss_id, j)
            fss_id_eq = trunc_fss_id_eq_chunk_v1(base_fss_id, j)
            k0lt, k1lt = keygen_dpf_dcf_keyrecs_v1(
                sid=sid,
                sid_hash32=sid_hash32,
                fss_id=int(fss_id_lt),
                alpha=alpha16,
                w=16,
                prg_id=PRG_CHACHA12,
                party_edge=EDGE_01,
                master_seed32=seed_edge01_32,
                prim_type=PRIM_DCF,
                dcf_invert=True,
                payload_mask_u64=1,
            )
            k0eq, k1eq = keygen_dpf_dcf_keyrecs_v1(
                sid=sid,
                sid_hash32=sid_hash32,
                fss_id=int(fss_id_eq),
                alpha=alpha16,
                w=16,
                prg_id=PRG_CHACHA12,
                party_edge=EDGE_01,
                master_seed32=seed_edge01_32,
                prim_type=PRIM_DPF,
            )
            lt0[j].append(k0lt)
            lt1[j].append(k1lt)
            eq0[j].append(k0eq)
            eq1[j].append(k1eq)

    return {0: carry0, 1: carry1, 2: []}, {0: tuple(lt0), 1: tuple(lt1), 2: ([], [], [], [])}, {0: tuple(eq0), 1: tuple(eq1), 2: ([], [], [], [])}


def _reconstruct_u64_from_pairs(y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    # y0 is [2,lanes] for party0, y1 is [2,lanes] for party1.
    share0 = y0[0]
    share1 = y0[1]
    share2 = y1[1]
    return (share0 + share1 + share2).to(torch.int64)


def test_trunc_prob_v1_reconstructs_shifted_value() -> None:
    sid = b"sid-trunc-prob"
    job_id32 = b"\x55" * 32
    epoch = 0
    step = 0
    sgir_op_id = 7
    lanes = 4
    F = 16

    gen = torch.Generator(device="cpu").manual_seed(2020)
    lo = torch.randint(0, 2**32, (lanes,), dtype=torch.int64, generator=gen)
    hi = torch.randint(0, 2**32, (lanes,), dtype=torch.int64, generator=gen)
    x_pub = (hi << 32) | lo
    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x_pub, generator=gen, device=torch.device("cpu"))

    base_fss_id = 0xABCDEF01_23456789
    packs, _r0, _r = _build_trunc_pack_blobs_det(sid=sid, sgir_op_id=sgir_op_id, base_fss_id=base_fss_id, f_bits=F, signed_mode=False, lanes=lanes, seed=991)

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            def run_party(pid: int) -> Tuple[torch.Tensor, List[int]]:
                relay = RelayClient(base_url=base, group_id="g-trunc-prob", token=None, timeout_s=20.0)
                p = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                y = op_trunc_prob_v1(
                    p,
                    x={0: x0, 1: x1, 2: x2}[pid],
                    trunc_pack_blob=packs[pid],
                    epoch=epoch,
                    step=step,
                    sgir_op_id=sgir_op_id,
                    f_bits=F,
                    signedness=0,
                )
                kinds = [int(l.prefix.msg_kind) for l in (p.transcript.leaves() if p.transcript is not None else [])]
                return torch.stack([y.lo, y.hi], dim=0), kinds

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                y0, k0 = f0.result(timeout=120)
                y1, k1 = f1.result(timeout=120)
                _, k2 = f2.result(timeout=120)

            for ks in (k0, k1, k2):
                assert MSG_TRUNC_OPEN_ARITH_SEND in ks
                assert MSG_TRUNC_OPEN_ARITH_RESULT in ks
                assert MSG_TRUNC_OUTPUT_COMMIT in ks
                assert MSG_TRUNC_CARRY_RESULT not in ks

            y_pub = _reconstruct_u64_from_pairs(y0, y1)

            # TRUNC_PROB output matches (x+r)>>F - (r>>F); equivalently, floor(x/2^F) or floor(x/2^F)+1 depending on randomized carry.
            # Note: this is defined modulo 2^(64-F) (embedded in u64), so it can differ by +/- 2^(64-F) when (x+r) overflows 2^64.
            # We assert the residue mod 2^(64-F) is in {0,1}.
            x_u64 = [int(v) & 0xFFFFFFFFFFFFFFFF for v in x_pub.tolist()]
            floor_u64 = [(v >> F) & 0xFFFFFFFFFFFFFFFF for v in x_u64]
            y_u64 = [int(v) & 0xFFFFFFFFFFFFFFFF for v in y_pub.tolist()]
            mod_mask = (1 << (64 - F)) - 1
            for yu, fu in zip(y_u64, floor_u64):
                d = (yu - fu) & mod_mask
                assert d in (0, 1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def test_trunc_exact_v1_matches_logical_shift_unsigned() -> None:
    sid = b"sid-trunc-exact"
    job_id32 = b"\x66" * 32
    epoch = 0
    step = 0
    sgir_op_id = 11
    lanes = 3
    F = 16
    base_fss_id = 0x11112222_33334444

    gen = torch.Generator(device="cpu").manual_seed(2021)
    lo = torch.randint(0, 2**32, (lanes,), dtype=torch.int64, generator=gen)
    hi = torch.randint(0, 2**32, (lanes,), dtype=torch.int64, generator=gen)
    x_pub = (hi << 32) | lo
    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x_pub, generator=gen, device=torch.device("cpu"))

    packs, r0_pub, r_pub = _build_trunc_pack_blobs_det(sid=sid, sgir_op_id=sgir_op_id, base_fss_id=base_fss_id, f_bits=F, signed_mode=False, lanes=lanes, seed=992)
    sid_hash32 = _sid_hash32(sid)
    edge01_seed32 = b"\x11" * 32
    carry, lt, eq = _build_trunc_keys_det(sid=sid, sid_hash32=sid_hash32, base_fss_id=base_fss_id, f_bits=F, r0_pub=r0_pub, r_pub=r_pub, seed_edge01_32=edge01_seed32)

    # B2A packs (carry / ov) and GF2 triples for boolean ANDs.
    b2a_c0, b2a_c1, b2a_c2 = build_b2a_packs_det_v1(sid=sid, sgir_op_id=sgir_op_id, base_stream_id=0, count_bits=lanes, seed32=b"\x22" * 32)
    b2a_o0, b2a_o1, b2a_o2 = build_b2a_packs_det_v1(sid=sid, sgir_op_id=sgir_op_id + 1, base_stream_id=0, count_bits=lanes, seed32=b"\x23" * 32)
    t0, t1, t2 = generate_gf2_triples_packs_v1(sid_hash32=sid_hash32, triple_id_base=0, count_triples=8 * lanes, seed32=b"\x33" * 32, sgir_op_id=sgir_op_id)

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            def run_party(pid: int) -> Tuple[torch.Tensor, List[int]]:
                relay = RelayClient(base_url=base, group_id="g-trunc-exact", token=None, timeout_s=30.0)
                p = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                keys = TruncFSSKeysV1(edge=EDGE_01, carry_keyrecs=carry[pid], lt_keyrecs=lt[pid], eq_keyrecs=eq[pid])
                y = op_trunc_exact_v1(
                    p,
                    x={0: x0, 1: x1, 2: x2}[pid],
                    trunc_pack_blob=packs[pid],
                    fss_keys=keys,
                    gf2_triples_blob={0: t0, 1: t1, 2: t2}[pid],
                    b2a_carry_blob={0: b2a_c0, 1: b2a_c1, 2: b2a_c2}[pid],
                    b2a_ov_blob={0: b2a_o0, 1: b2a_o1, 2: b2a_o2}[pid],
                    edge_key32=(b"\x99" * 32 if pid in (0, 1) else None),
                    epoch=epoch,
                    step=step,
                    sgir_op_id=sgir_op_id,
                    f_bits=F,
                    signedness=0,
                )
                kinds = [int(l.prefix.msg_kind) for l in (p.transcript.leaves() if p.transcript is not None else [])]
                return torch.stack([y.lo, y.hi], dim=0), kinds

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                y0, k0 = f0.result(timeout=120)
                y1, k1 = f1.result(timeout=120)
                _, k2 = f2.result(timeout=120)

            for ks in (k0, k1, k2):
                assert MSG_TRUNC_OPEN_ARITH_SEND in ks
                assert MSG_TRUNC_OPEN_ARITH_RESULT in ks
                assert MSG_TRUNC_CARRY_RESULT in ks
                assert MSG_TRUNC_OUTPUT_COMMIT in ks

            y_pub = _reconstruct_u64_from_pairs(y0, y1)
            x_u64 = [int(v) & 0xFFFFFFFFFFFFFFFF for v in x_pub.tolist()]
            expect = torch.tensor([_ for _ in [(v >> F) & 0xFFFFFFFFFFFFFFFF for v in x_u64]], dtype=torch.int64)
            assert torch.equal(y_pub, expect)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


