from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_30225604e2990a9f,uvcc_group_48a6f9c1656f1342,uvcc_group_e6b8d87d4097bede,uvcc_group_509a5eba42fcc1ce,uvcc_group_b69668db9263f95f

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.op_lut import op_lut_public_v1
from uvcc_party.op_lut_blob import build_oplut_record_blobs_v1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import make_rss_arith_u64_triple


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


def test_op_lut_public_v1_w8_e2e_two_lanes() -> None:
    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            sid = b"sid-oplut-w8"
            job_id32 = b"\x44" * 32
            epoch = 0
            step = 0
            sgir_op_id = 123
            lanes = 2

            # Public table T[0..255] (lifted into ring u64).
            T = torch.arange(256, dtype=torch.int64)
            # Secret indices x in [0,256).
            gen = torch.Generator(device="cpu").manual_seed(2029)
            x_pub = torch.randint(0, 256, (lanes,), dtype=torch.int64, generator=gen)
            x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x_pub, generator=gen, device=torch.device("cpu"))

            # Build per-party blobs.
            fss_id = 0xABCDEF0123456789
            p0_blob, p1_blob, p2_blob = build_oplut_record_blobs_v1(
                sid=sid,
                fss_id=fss_id,
                sgir_op_id=sgir_op_id,
                domain_w=8,
                lanes=lanes,
                lane_base=0,
                K_master32=b"\x11" * 32,
                seed_edge01_32=b"\x22" * 32,
                seed_edge12_32=b"\x33" * 32,
                seed_edge20_32=b"\x44" * 32,
            )
            blobs = {0: p0_blob, 1: p1_blob, 2: p2_blob}
            xs = {0: x0, 1: x1, 2: x2}

            def run_party(party_id: int) -> torch.Tensor:
                relay = RelayClient(base_url=base, group_id="g-oplut", token=None, timeout_s=20.0)
                p = Party(party_id=party_id, job_id32=job_id32, sid=sid, relay=relay)
                y = op_lut_public_v1(
                    p,
                    x=xs[party_id],
                    table_u64=T,
                    fss_blob=blobs[party_id],
                    epoch=epoch,
                    step=step,
                    sgir_op_id=sgir_op_id,
                )
                # Return this party's RSS pair (lo,hi) for reconstruction.
                return torch.stack([y.lo, y.hi], dim=0)

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                y0 = f0.result(timeout=120)
                y1 = f1.result(timeout=120)
                y2 = f2.result(timeout=120)

            # Reconstruct public y from RSS pairs:
            # share0 = P0.lo, share1 = P0.hi, share2 = P1.hi
            share0 = y0[0]
            share1 = y0[1]
            share2 = y1[1]
            y_pub = (share0 + share1 + share2).to(torch.int64)
            expect = T[x_pub]
            assert torch.equal(y_pub.to(torch.int64), expect.to(torch.int64))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


