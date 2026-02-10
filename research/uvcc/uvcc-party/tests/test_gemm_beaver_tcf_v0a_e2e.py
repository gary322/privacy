from __future__ import annotations

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import torch

from uvcc_party.gemm import op_gemm_tile_beaver_tcf_v0a_u64_v1
from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.rss import make_rss_arith_u64_triple
from uvcc_party.tcf import MSG_TCF_REPL_V1, tcf_gen_v1


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


def _u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _tensor_to_u64_matrix(x: torch.Tensor) -> List[List[int]]:
    x = x.contiguous().cpu()
    d0, d1 = x.shape
    out: List[List[int]] = []
    for i in range(int(d0)):
        row: List[int] = []
        for j in range(int(d1)):
            row.append(_u64(int(x[i, j].item())))
        out.append(row)
    return out


def _matmul_u64(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    d = len(A)
    out = [[0] * d for _ in range(d)]
    for i in range(d):
        for k in range(d):
            aik = A[i][k]
            for j in range(d):
                out[i][j] = _u64(out[i][j] + _u64(aik * B[k][j]))
    return out


def _reconstruct_from_pairs(p0_lohi: torch.Tensor, p1_hi: torch.Tensor) -> torch.Tensor:
    return (p0_lohi[0] + p0_lohi[1] + p1_hi).to(torch.int64)


def test_gemm_tile_beaver_tcf_v0a_e2e() -> None:
    sid = b"sid-gemm"
    job_id32 = b"\x88" * 32
    epoch = 0
    step = 0
    sgir_op_id = 99
    op_id = 7
    d = 16

    master_seed32 = b"\x11" * 32
    k0, k1, k2 = tcf_gen_v1(master_seed32=master_seed32, sid=sid)
    tcf_keys = {0: k0, 1: k1, 2: k2}

    gen = torch.Generator(device="cpu").manual_seed(2025)
    loX = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
    hiX = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
    X_pub = (hiX << 32) | loX
    loY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
    hiY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
    Y_pub = (hiY << 32) | loY

    X0, X1, X2 = make_rss_arith_u64_triple(x_pub=X_pub, generator=gen, device=torch.device("cpu"))
    Y0, Y1, Y2 = make_rss_arith_u64_triple(x_pub=Y_pub, generator=gen, device=torch.device("cpu"))

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            def run_party(pid: int) -> Tuple[torch.Tensor, List[int]]:
                relay = RelayClient(base_url=base, group_id="g-gemm", token=None, timeout_s=60.0)
                party = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                res = op_gemm_tile_beaver_tcf_v0a_u64_v1(
                    party,
                    X={0: X0, 1: X1, 2: X2}[pid],
                    Y={0: Y0, 1: Y1, 2: Y2}[pid],
                    tcf_key=tcf_keys[pid],
                    op_id=op_id,
                    tile_i=0,
                    tile_j=0,
                    tile_p=0,
                    epoch=epoch,
                    step=step,
                    sgir_op_id=sgir_op_id,
                    fxp_frac_bits=0,
                    d=d,
                )
                Z = res.Z
                kinds = [int(l.prefix.msg_kind) for l in (party.transcript.leaves() if party.transcript is not None else [])]
                return torch.stack([Z.lo, Z.hi], dim=0), kinds

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                Z0, k0s = f0.result(timeout=180)
                Z1, k1s = f1.result(timeout=180)
                _Z2, k2s = f2.result(timeout=180)

            # Expect TCF replication traffic.
            for ks in (k0s, k1s, k2s):
                assert int(MSG_TCF_REPL_V1) in ks

            Z_pub = _reconstruct_from_pairs(Z0, Z1[1])
            X_mat = _tensor_to_u64_matrix(X_pub)
            Y_mat = _tensor_to_u64_matrix(Y_pub)
            expect = _matmul_u64(X_mat, Y_mat)
            got = _tensor_to_u64_matrix(Z_pub)
            assert got == expect
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


