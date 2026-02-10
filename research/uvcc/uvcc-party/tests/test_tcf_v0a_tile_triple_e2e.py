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

from uvcc_party.party import Party
from uvcc_party.relay_client import RelayClient
from uvcc_party.tcf import MSG_TCF_REPL_V1, tcf_eval_v0a_tile_u64_v1, tcf_gen_v1


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


def _matmul_u64(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    d = len(A)
    out = [[0] * d for _ in range(d)]
    for i in range(d):
        for k in range(d):
            aik = A[i][k]
            for j in range(d):
                out[i][j] = _u64(out[i][j] + _u64(aik * B[k][j]))
    return out


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


def _reconstruct_from_pairs(p0_lohi: torch.Tensor, p1_hi: torch.Tensor) -> torch.Tensor:
    # Reconstruct u64 matrix from party0 (share0,share1) and party1 (share2 in hi).
    return (p0_lohi[0] + p0_lohi[1] + p1_hi).to(torch.int64)


def test_tcf_v0a_tile_triple_e2e() -> None:
    sid = b"sid-tcf"
    job_id32 = b"\x77" * 32
    epoch = 0
    step = 0
    round = 0
    op_id = 42
    i = 0
    j = 0
    p = 0
    d = 16

    master_seed32 = b"\x11" * 32
    k0, k1, k2 = tcf_gen_v1(master_seed32=master_seed32, sid=sid)
    keys = {0: k0, 1: k1, 2: k2}

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            def run_party(pid: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
                relay = RelayClient(base_url=base, group_id="g-tcf", token=None, timeout_s=30.0)
                party = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                A, B, C = tcf_eval_v0a_tile_u64_v1(
                    party,
                    key=keys[pid],
                    op_id=op_id,
                    i=i,
                    j=j,
                    p=p,
                    epoch=epoch,
                    step=step,
                    round=round,
                    fxp_frac_bits=0,
                    d=d,
                )
                kinds = [int(l.prefix.msg_kind) for l in (party.transcript.leaves() if party.transcript is not None else [])]
                # Return (A_lohi, B_lohi, C_lohi, kinds); for party1 we also need hi share2.
                return torch.stack([A.lo, A.hi], dim=0), torch.stack([B.lo, B.hi], dim=0), torch.stack([C.lo, C.hi], dim=0), kinds

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                A0, B0, C0, k0s = f0.result(timeout=120)
                A1, B1, C1, k1s = f1.result(timeout=120)
                _A2, _B2, _C2, k2s = f2.result(timeout=120)

            # Transcript should contain TCF replication frames (msg_kind=200).
            for ks in (k0s, k1s, k2s):
                assert int(MSG_TCF_REPL_V1) in ks

            A_pub = _reconstruct_from_pairs(A0, A1[1])
            B_pub = _reconstruct_from_pairs(B0, B1[1])
            C_pub = _reconstruct_from_pairs(C0, C1[1])

            A_mat = _tensor_to_u64_matrix(A_pub)
            B_mat = _tensor_to_u64_matrix(B_pub)
            C_mat = _tensor_to_u64_matrix(C_pub)
            expect = _matmul_u64(A_mat, B_mat)
            assert C_mat == expect
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


