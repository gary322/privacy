from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_d4a68ca07fd9802a,uvcc_group_2f305dfbd543379d

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.cmp import PRED_LT, op_cmp_v1
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
    sid_hash32: bytes,
    w: int,
    n: int,
    base_triple_id: int,
    seed: int,
) -> A2BPackV1:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r_bits_pub = torch.randint(0, 2, (n, w), dtype=torch.int64, generator=gen)
    r_val = torch.zeros((n,), dtype=torch.int64)
    for j in range(w):
        r_val += (r_bits_pub[:, j] << j)
    r0, r1, r2 = make_rss_arith_u64_triple(x_pub=r_val, generator=gen, device=torch.device("cpu"))
    if party_id == 0:
        r_arith_lo, r_arith_hi = r0.lo, r0.hi
    elif party_id == 1:
        r_arith_lo, r_arith_hi = r1.lo, r1.hi
    else:
        r_arith_lo, r_arith_hi = r2.lo, r2.hi

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


def test_op_cmp_v1_lt_w16_e2e() -> None:
    w = 16
    n = 65
    gen = torch.Generator(device="cpu").manual_seed(2026)
    x = _rand_i64_bits((n,), gen) & ((1 << w) - 1)
    y = _rand_i64_bits((n,), gen) & ((1 << w) - 1)
    lt_expect = (x < y).to(torch.int64)

    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))
    y0, y1, y2 = make_rss_arith_u64_triple(x_pub=y, generator=gen, device=torch.device("cpu"))

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)
            job_id32 = b"\x55" * 32
            sid = b"sid-cmp-w32"
            sid_hash32 = __import__("hashlib").sha256(sid).digest()

            # Allocate a deterministic GF2 triples pool large enough for the log-depth CMP.
            L = int((w - 1).bit_length())
            sum_w_minus = sum(int(w - (1 << s)) for s in range(L))
            gates_borrow_xy = 2 * sum_w_minus * 2
            gates_suffix = sum_w_minus
            gates_u = w
            gates_t = w
            total_gates = int(gates_borrow_xy + gates_suffix + gates_u + gates_t)
            triple_base = 1000
            count_triples = triple_base + (total_gates * n) + 2048
            b0, b1, b2 = generate_gf2_triples_packs_v1(
                sid_hash32=sid_hash32,
                triple_id_base=triple_base,
                count_triples=count_triples,
                seed32=b"\x07" * 32,
                sgir_op_id=0,
            )
            t0 = GF2TriplesPackV1.from_bytes(b0)
            t1 = GF2TriplesPackV1.from_bytes(b1)
            t2 = GF2TriplesPackV1.from_bytes(b2)

            def run_party(party_id: int, xs: RSSArithU64, ys: RSSArithU64, triples_pack: GF2TriplesPackV1):
                relay = RelayClient(base_url=base, group_id="g-cmp-w16", token=None, timeout_s=10.0)
                p = Party(party_id=party_id, job_id32=job_id32, sid=sid, relay=relay)
                ed_x = _make_edabit_packs(party_id=party_id, sid_hash32=sid_hash32, w=w, n=n, base_triple_id=triple_base, seed=7777)
                ed_y = _make_edabit_packs(party_id=party_id, sid_hash32=sid_hash32, w=w, n=n, base_triple_id=triple_base + 12345, seed=8888)
                out_lt = op_cmp_v1(
                    p,
                    cmp_uid=0xDEADBEEF,
                    x=xs,
                    y=ys,
                    bitwidth=w,
                    pred=PRED_LT,
                    signedness=0,
                    edabit_x=ed_x,
                    edabit_y=ed_y,
                    triples=triples_pack,
                    cmp_triple_cursor_base=triple_base,
                    epoch=0,
                    step=0,
                )
                return out_lt

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0, y0, t0)
                f1 = ex.submit(run_party, 1, x1, y1, t1)
                f2 = ex.submit(run_party, 2, x2, y2, t2)
                o0 = f0.result(timeout=120)
                o1 = f1.result(timeout=120)
                _ = f2.result(timeout=120)

            def reconstruct_from_parties(a0, a1):
                b0w = a0.lo_words
                b1w = a0.hi_words
                b2w = a1.hi_words
                return b0w ^ b1w ^ b2w

            def words_to_bits(words: torch.Tensor, n_bits: int) -> torch.Tensor:
                bits = torch.zeros((n_bits,), dtype=torch.int64)
                for i in range(n_bits):
                    wi = i // 64
                    bi = i % 64
                    bits[i] = (words[wi] >> bi) & 1
                return bits

            lt_pub = words_to_bits(reconstruct_from_parties(o0, o1), n)
            assert torch.equal(lt_pub, lt_expect)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


