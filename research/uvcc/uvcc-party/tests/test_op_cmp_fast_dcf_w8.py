from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_0287a0390f86b1c6,uvcc_group_740cc94010953fd5

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.cmp import op_cmp_lt_fast_dcf_v1
from uvcc_party.dpf_dcf import PRG_CHACHA12, PRIM_DCF, keygen_dpf_dcf_keyrecs_v1
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


def test_op_cmp_fast_dcf_w8_lt_e2e() -> None:
    w = 8
    n = 65
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = _rand_i64_bits((n,), gen) & ((1 << w) - 1)
    y = _rand_i64_bits((n,), gen) & ((1 << w) - 1)

    lt_expect = (x < y).to(torch.int64)

    # Secret-share x,y,r for each party.
    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))
    y0, y1, y2 = make_rss_arith_u64_triple(x_pub=y, generator=gen, device=torch.device("cpu"))
    r_val = torch.randint(0, 1 << w, (n,), dtype=torch.int64, generator=gen)
    r0, r1, r2 = make_rss_arith_u64_triple(x_pub=r_val, generator=gen, device=torch.device("cpu"))

    # Precompute edge01 DCF keys per element (dealer knows r_val here; in production this is offline/TEE).
    sid = b"sid-cmp-fast-dcf"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    master_seed32 = b"\x11" * 32
    fss_id0 = 0x0102030405060708

    keyrecs_p0 = []
    keyrecs_p1 = []
    for i in range(n):
        k0, k1 = keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=fss_id0 + i,
            alpha=int(r_val[i].item()),
            w=w,
            prg_id=PRG_CHACHA12,
            party_edge=0,
            master_seed32=master_seed32,
            prim_type=PRIM_DCF,
            dcf_invert=True,
            payload_mask_u64=1,
        )
        keyrecs_p0.append(k0)
        keyrecs_p1.append(k1)

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            job_id32 = b"\x77" * 32

            def run_party(party_id: int, xs: RSSArithU64, ys: RSSArithU64, rs: RSSArithU64):
                relay = RelayClient(base_url=base, group_id="g-cmp-fast", token=None, timeout_s=10.0)
                p = Party(party_id=party_id, job_id32=job_id32, sid=sid, relay=relay)
                keys = keyrecs_p0 if party_id == 0 else keyrecs_p1 if party_id == 1 else []
                edge01_key32 = b"\x99" * 32 if party_id in (0, 1) else None
                out = op_cmp_lt_fast_dcf_v1(
                    p,
                    cmp_uid=0x11112222,
                    x=xs,
                    y=ys,
                    bitwidth=w,
                    signedness=0,
                    r_mask=rs,
                    dcf_keyrecs_edge01=keys,
                    edge01_key32=edge01_key32,
                    epoch=0,
                    step=0,
                )
                return out

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0, y0, r0)
                f1 = ex.submit(run_party, 1, x1, y1, r1)
                f2 = ex.submit(run_party, 2, x2, y2, r2)
                o0 = f0.result(timeout=60)
                o1 = f1.result(timeout=60)
                o2 = f2.result(timeout=60)

            # Reconstruct bits from RSS pairs:
            def reconstruct_from_parties(a0, a1) -> torch.Tensor:
                b0 = a0.lo_words
                b1 = a0.hi_words
                b2 = a1.hi_words
                return b0 ^ b1 ^ b2

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


def test_op_cmp_fast_dcf_w16_lt_e2e_small() -> None:
    # Keep this tiny: w=16 fast-path does full-domain eval in the python reference.
    w = 16
    n = 3
    gen = torch.Generator(device="cpu").manual_seed(2027)
    x = _rand_i64_bits((n,), gen) & ((1 << w) - 1)
    y = _rand_i64_bits((n,), gen) & ((1 << w) - 1)
    lt_expect = (x < y).to(torch.int64)

    x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))
    y0, y1, y2 = make_rss_arith_u64_triple(x_pub=y, generator=gen, device=torch.device("cpu"))
    r_val = torch.randint(0, 1 << w, (n,), dtype=torch.int64, generator=gen)
    r0, r1, r2 = make_rss_arith_u64_triple(x_pub=r_val, generator=gen, device=torch.device("cpu"))

    sid = b"sid-cmp-fast-dcf-w16"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    master_seed32 = b"\x11" * 32
    fss_id0 = 0x0102030405060708

    keyrecs_p0 = []
    keyrecs_p1 = []
    for i in range(n):
        k0, k1 = keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=fss_id0 + i,
            alpha=int(r_val[i].item()),
            w=w,
            prg_id=PRG_CHACHA12,
            party_edge=0,
            master_seed32=master_seed32,
            prim_type=PRIM_DCF,
            dcf_invert=True,
            payload_mask_u64=1,
        )
        keyrecs_p0.append(k0)
        keyrecs_p1.append(k1)

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            job_id32 = b"\x78" * 32

            def run_party(party_id: int, xs: RSSArithU64, ys: RSSArithU64, rs: RSSArithU64):
                relay = RelayClient(base_url=base, group_id="g-cmp-fast-w16", token=None, timeout_s=10.0)
                p = Party(party_id=party_id, job_id32=job_id32, sid=sid, relay=relay)
                keys = keyrecs_p0 if party_id == 0 else keyrecs_p1 if party_id == 1 else []
                edge01_key32 = b"\x99" * 32 if party_id in (0, 1) else None
                out = op_cmp_lt_fast_dcf_v1(
                    p,
                    cmp_uid=0x22223333,
                    x=xs,
                    y=ys,
                    bitwidth=w,
                    signedness=0,
                    r_mask=rs,
                    dcf_keyrecs_edge01=keys,
                    edge01_key32=edge01_key32,
                    epoch=0,
                    step=0,
                )
                return out

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0, y0, r0)
                f1 = ex.submit(run_party, 1, x1, y1, r1)
                f2 = ex.submit(run_party, 2, x2, y2, r2)
                o0 = f0.result(timeout=120)
                o1 = f1.result(timeout=120)
                _ = f2.result(timeout=120)

            def reconstruct_from_parties(a0, a1) -> torch.Tensor:
                b0 = a0.lo_words
                b1 = a0.hi_words
                b2 = a1.hi_words
                return b0 ^ b1 ^ b2

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


