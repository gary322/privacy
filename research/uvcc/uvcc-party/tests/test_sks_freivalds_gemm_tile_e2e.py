from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_ba7afac425406f12

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
from uvcc_party.rss import RSSArithU64, make_rss_arith_u64_triple
from uvcc_party.sks import (
    LEAF_SKS_CHECK_META_V1,
    LEAF_SKS_EPOCH_COMMIT_V1,
    LEAF_SKS_EPOCH_REVEAL_V1,
    LEAF_SKS_OPEN_COMMIT_V1,
    LEAF_SKS_OPEN_RESULT_V1,
    sks_epoch_setup_v1,
    sks_freivalds_check_tile_gemm_u64_v1,
)
from uvcc_party.tcf import tcf_gen_v1


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


def test_sks_freivalds_tile_gemm_pass_and_fail() -> None:
    sid = b"sid-sks"
    job_id32 = b"\x99" * 32
    epoch = 0
    d = 16

    master_seed32 = b"\x11" * 32
    k0, k1, k2 = tcf_gen_v1(master_seed32=master_seed32, sid=sid)
    tcf_keys = {0: k0, 1: k1, 2: k2}

    gen = torch.Generator(device="cpu").manual_seed(4242)
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

            def run_party(pid: int, *, tamper: bool) -> Tuple[bool, List[int]]:
                relay = RelayClient(base_url=base, group_id=("g-sks-tamper" if tamper else "g-sks-pass"), token=None, timeout_s=60.0)
                party = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)

                # Epoch randomness (commit/reveal).
                st = sks_epoch_setup_v1(party, sid=sid, epoch=epoch, step=1000)
                assert st.epoch_rand32 is not None

                # Run one secure GEMM tile (Beaver w/ TCF).
                res = op_gemm_tile_beaver_tcf_v0a_u64_v1(
                    party,
                    X={0: X0, 1: X1, 2: X2}[pid],
                    Y={0: Y0, 1: Y1, 2: Y2}[pid],
                    tcf_key=tcf_keys[pid],
                    op_id=7,
                    tile_i=0,
                    tile_j=0,
                    tile_p=0,
                    epoch=epoch,
                    step=0,
                    sgir_op_id=123,
                    fxp_frac_bits=0,
                    d=d,
                )

                Z = res.Z
                if tamper and pid == 0:
                    # Corrupt output shares deterministically.
                    Z = RSSArithU64(lo=(Z.lo + 1), hi=Z.hi, fxp_frac_bits=Z.fxp_frac_bits)

                ok = sks_freivalds_check_tile_gemm_u64_v1(
                    party,
                    sid=sid,
                    epoch_rand32=st.epoch_rand32,
                    epoch=epoch,
                    step=2000,
                    sgir_op_id=123,
                    kernel_instance_id=0,
                    sks_sample_log2=0,  # always selected
                    t_checks=3,
                    field_id=0,  # ring mode
                    Z=Z,
                    triple_A=res.triple_A,
                    triple_B=res.triple_B,
                    triple_C=res.triple_C,
                    E_pub=res.E_pub,
                    F_pub=res.F_pub,
                )
                assert ok is not None

                kinds = [int(l.prefix.msg_kind) for l in (party.transcript.leaves() if party.transcript is not None else [])]
                return bool(ok), kinds

            # PASS case
            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, tamper=False)
                f1 = ex.submit(run_party, 1, tamper=False)
                f2 = ex.submit(run_party, 2, tamper=False)
                ok0, k0s = f0.result(timeout=180)
                ok1, k1s = f1.result(timeout=180)
                ok2, k2s = f2.result(timeout=180)
            assert ok0 and ok1 and ok2
            for ks in (k0s, k1s, k2s):
                assert LEAF_SKS_EPOCH_COMMIT_V1 in ks
                assert LEAF_SKS_EPOCH_REVEAL_V1 in ks
                assert LEAF_SKS_CHECK_META_V1 in ks
                assert LEAF_SKS_OPEN_COMMIT_V1 in ks
                assert LEAF_SKS_OPEN_RESULT_V1 in ks

            # FAIL case (tampered)
            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, tamper=True)
                f1 = ex.submit(run_party, 1, tamper=True)
                f2 = ex.submit(run_party, 2, tamper=True)
                ok0, _ = f0.result(timeout=180)
                ok1, _ = f1.result(timeout=180)
                ok2, _ = f2.result(timeout=180)
            assert not (ok0 and ok1 and ok2)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


