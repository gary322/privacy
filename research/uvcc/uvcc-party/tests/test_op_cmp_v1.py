from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_5a83f07bbef68c40,uvcc_group_2f305dfbd543379d

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.cmp import PRED_LE, PRED_LT, PRED_EQ, op_cmp_v1
from uvcc_party.edabit import A2BPackV1
from uvcc_party.gf2_triples import GF2TriplesPackV1, generate_gf2_triples_packs_v1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import RSSArithU64, make_rss_arith_u64_triple


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


def _rand_i64_bits(shape, gen: torch.Generator) -> torch.Tensor:
    lo = torch.randint(0, 2**32, shape, device="cpu", dtype=torch.int64, generator=gen)
    hi = torch.randint(0, 2**32, shape, device="cpu", dtype=torch.int64, generator=gen)
    return (hi << 32) | lo


def _make_edabit_packs(
    *,
    party_id: int,
    job_id32: bytes,
    sid_hash32: bytes,
    w: int,
    n: int,
    base_triple_id: int,
    seed: int,
) -> A2BPackV1:
    # Deterministic public r bits and arithmetic r from seed.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r_bits_pub = torch.randint(0, 2, (n, w), dtype=torch.int64, generator=gen)
    # r_arith = sum 2^j * rj
    r_val = torch.zeros((n,), dtype=torch.int64)
    for j in range(w):
        r_val += (r_bits_pub[:, j] << j)

    # Share arithmetic r in RSS (int64 mod 2^64).
    r0, r1, r2 = make_rss_arith_u64_triple(x_pub=r_val, generator=gen, device=torch.device("cpu"))
    if party_id == 0:
        r_arith_lo, r_arith_hi = r0.lo, r0.hi
    elif party_id == 1:
        r_arith_lo, r_arith_hi = r1.lo, r1.hi
    else:
        r_arith_lo, r_arith_hi = r2.lo, r2.hi

    # Boolean shares for bits: choose s0,s1 random; s2 = bit ^ s0 ^ s1.
    s0 = torch.randint(0, 2, (n, w), dtype=torch.int64, generator=gen)
    s1 = torch.randint(0, 2, (n, w), dtype=torch.int64, generator=gen)
    s2 = r_bits_pub ^ s0 ^ s1
    if party_id == 0:
        r_bits_lo, r_bits_hi = s0, s1
    elif party_id == 1:
        r_bits_lo, r_bits_hi = s1, s2
    else:
        r_bits_lo, r_bits_hi = s2, s0

    return A2BPackV1(
        w_bits=w,
        count_vals=n,
        sgir_op_id=0,
        base_triple_id=base_triple_id,
        sid_hash32=sid_hash32,
        r_arith_lo=r_arith_lo,
        r_arith_hi=r_arith_hi,
        r_bits_lo=r_bits_lo,
        r_bits_hi=r_bits_hi,
    )


def test_op_cmp_v1_le_lt_eq_small_w8() -> None:
    w = 8
    n = 65
    gen = torch.Generator(device="cpu").manual_seed(123)
    x = _rand_i64_bits((n,), gen) & ((1 << w) - 1)
    y = _rand_i64_bits((n,), gen) & ((1 << w) - 1)

    # Plaintext checks (unsigned)
    x_u = x.to(torch.int64)
    y_u = y.to(torch.int64)
    lt_expect = (x_u < y_u).to(torch.int64)
    le_expect = (x_u <= y_u).to(torch.int64)
    eq_expect = (x_u == y_u).to(torch.int64)

    # Secret-share x,y for each party.
    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))
    y0, y1, y2 = make_rss_arith_u64_triple(x_pub=y, generator=gen, device=torch.device("cpu"))

    # Relay + triples pool
    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            job_id32 = b"\x66" * 32
            sid = b"sid-cmp"
            sid_hash32 = torch.tensor(list(sid), dtype=torch.uint8).numpy().tobytes()  # just deterministic bytes
            sid_hash32 = __import__("hashlib").sha256(sid).digest()

            # Triple pool big enough for CMP macro:
            # - borrow-lookahead scan for x and y: per operand, gates = 2*sum_s (w-2^s)
            # - eq suffix-scan: gates = sum_s (w-2^s)
            # - u layer: w gates
            # - t layer: w gates
            L = int((w - 1).bit_length())
            sum_w_minus = sum(int(w - (1 << s)) for s in range(L))
            gates_borrow_xy = 2 * sum_w_minus * 2
            gates_suffix = sum_w_minus
            gates_u = w
            gates_t = w
            total_gates = int(gates_borrow_xy + gates_suffix + gates_u + gates_t)
            triple_base = 1000
            count_triples = triple_base + (total_gates * n) + 1024  # slack
            b0, b1, b2 = generate_gf2_triples_packs_v1(
                sid_hash32=sid_hash32,
                triple_id_base=triple_base,
                count_triples=count_triples,
                seed32=b"\x02" * 32,
                sgir_op_id=0,
            )
            t0 = GF2TriplesPackV1.from_bytes(b0)
            t1 = GF2TriplesPackV1.from_bytes(b1)
            t2 = GF2TriplesPackV1.from_bytes(b2)

            def run_party(party_id: int, xs, ys, triples_pack: GF2TriplesPackV1):
                relay = RelayClient(base_url=base, group_id="g-cmp", token=None, timeout_s=10.0)
                p = Party(party_id=party_id, job_id32=job_id32, sid=sid, relay=relay)
                ed_x = _make_edabit_packs(
                    party_id=party_id,
                    job_id32=job_id32,
                    sid_hash32=sid_hash32,
                    w=w,
                    n=n,
                    base_triple_id=triple_base + 0,
                    seed=777,
                )
                ed_y = _make_edabit_packs(
                    party_id=party_id,
                    job_id32=job_id32,
                    sid_hash32=sid_hash32,
                    w=w,
                    n=n,
                    base_triple_id=triple_base + (w * n),
                    seed=888,
                )
                cmp_base = triple_base + (w * n * 2)

                out_lt = op_cmp_v1(
                    p,
                    cmp_uid=0xABCDEF01,
                    x=xs,
                    y=ys,
                    bitwidth=w,
                    pred=PRED_LT,
                    signedness=0,
                    edabit_x=ed_x,
                    edabit_y=ed_y,
                    triples=triples_pack,
                    cmp_triple_cursor_base=cmp_base,
                    epoch=0,
                    step=0,
                )
                out_le = op_cmp_v1(
                    p,
                    cmp_uid=0xABCDEF02,
                    x=xs,
                    y=ys,
                    bitwidth=w,
                    pred=PRED_LE,
                    signedness=0,
                    edabit_x=ed_x,
                    edabit_y=ed_y,
                    triples=triples_pack,
                    cmp_triple_cursor_base=cmp_base,
                    epoch=0,
                    step=1,
                )
                out_eq = op_cmp_v1(
                    p,
                    cmp_uid=0xABCDEF03,
                    x=xs,
                    y=ys,
                    bitwidth=w,
                    pred=PRED_EQ,
                    signedness=0,
                    edabit_x=ed_x,
                    edabit_y=ed_y,
                    triples=triples_pack,
                    cmp_triple_cursor_base=cmp_base,
                    epoch=0,
                    step=2,
                )
                return out_lt, out_le, out_eq

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0, y0, t0)
                f1 = ex.submit(run_party, 1, x1, y1, t1)
                f2 = ex.submit(run_party, 2, x2, y2, t2)
                o0 = f0.result(timeout=60)
                o1 = f1.result(timeout=60)
                o2 = f2.result(timeout=60)

            # Reconstruct public outputs by XORing shares (open them locally for test):
            def rec(out0, out1, out2):
                # Reconstruct packed words -> bits
                pub_words = out0.lo_words ^ out0.hi_words ^ out2.lo_words  # NOTE: for party2, lo is share2, hi is share0; simplest is compute global shares from party0/1/2? Use bitwise from three share-0/1/2.
                # For correctness in this test, just open via relay would be overkill; instead compare hashes of shares are committed in transcript.
                return pub_words

            # Basic sanity: all parties should have same n_bits and consistent share shapes.
            for outs in (o0, o1, o2):
                for v in outs:
                    assert v.n_bits == n

            # We don't reconstruct bits here (share-layout reconstruction is tested elsewhere); instead validate that
            # all three parties produced identical share-pair hashes committed by OP_CMP leaf type.
            # As a weaker check, we verify that running party0 alone and opening via boolean reconstruction matches plaintext:
            # reconstruct from party0+party1+party2 components explicitly:
            def reconstruct_from_parties(a0, a1, a2):
                # Recover b0,b1,b2 words per lane from party views:
                # party0 has (b0,b1); party1 has (b1,b2); party2 has (b2,b0)
                b0 = a0.lo_words
                b1 = a0.hi_words
                b2 = a1.hi_words
                pub = b0 ^ b1 ^ b2
                return pub

            def words_to_bits(words: torch.Tensor, n_bits: int) -> torch.Tensor:
                bits = torch.zeros((n_bits,), dtype=torch.int64)
                for i in range(n_bits):
                    wi = i // 64
                    bi = i % 64
                    bits[i] = (words[wi] >> bi) & 1
                return bits

            lt_pub = words_to_bits(reconstruct_from_parties(o0[0], o1[0], o2[0]), n)
            le_pub = words_to_bits(reconstruct_from_parties(o0[1], o1[1], o2[1]), n)
            eq_pub = words_to_bits(reconstruct_from_parties(o0[2], o1[2], o2[2]), n)

            assert torch.equal(lt_pub, lt_expect)
            assert torch.equal(le_pub, le_expect)
            assert torch.equal(eq_pub, eq_expect)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


