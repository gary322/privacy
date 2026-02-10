from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_0bad2ad63695a9fd,uvcc_group_91b08a9d0e68235e

import concurrent.futures as cf
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from uvcc_party.open import OpenArithItemU64, open_arith_u64_round_v1
from uvcc_party.party import Party
from uvcc_party.recorder import PartyRecorderV1
from uvcc_party.relay_client import RelayClient
from uvcc_party.eip712 import EIP712DomainV1, FinalCommitV1
from uvcc_party.sig import secp256k1_pubkey_from_privkey, secp256k1_verify_hash
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


def test_party_recorder_builds_proof_bundle_and_signature_verifies() -> None:
    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            job_id32 = b"\x55" * 32
            sid = b"sid-recorder"
            gen = torch.Generator(device="cpu").manual_seed(7)
            n = 8
            x = torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) | (torch.randint(0, 2**32, (n,), dtype=torch.int64, generator=gen) << 32)
            x0, x1, x2 = make_rss_arith_u64_triple(x_pub=x, generator=gen, device=torch.device("cpu"))

            def run_party(pid: int, xs):
                relay = RelayClient(base_url=base, group_id="g-rec", token=None, timeout_s=10.0)
                p = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)
                pub = open_arith_u64_round_v1(
                    p,
                    items=[OpenArithItemU64(open_id=1, sub_id=0, x=xs)],
                    epoch=0,
                    step=0,
                    round=0,
                    sgir_op_id=0,
                )[(1, 0)]
                return p, pub

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0, x0)
                f1 = ex.submit(run_party, 1, x1)
                f2 = ex.submit(run_party, 2, x2)
                p0, pub0 = f0.result(timeout=60)
                p1, pub1 = f1.result(timeout=60)
                p2, pub2 = f2.result(timeout=60)

            assert torch.equal(pub0, x)
            assert torch.equal(pub1, x)
            assert torch.equal(pub2, x)

            policy_hash32 = b"\x01" * 32
            sgir_hash32 = b"\x02" * 32
            runtime_hash32 = b"\x03" * 32
            priv0 = b"\x11" * 32
            result_hash32 = b"\x99" * 32

            rec = PartyRecorderV1(party=p0, policy_hash32=policy_hash32, sgir_hash32=sgir_hash32, runtime_hash32=runtime_hash32)
            dom = EIP712DomainV1(chain_id=31337, verifying_contract=b"\x00" * 20)
            bundle = rec.build_proof_bundle(epoch_count=1, party_privkey32=priv0, result_hash32=result_hash32, eip712_domain=dom)
            js = bundle.to_json_bytes()
            assert js.startswith(b"{")

            # Verify signature
            msg_hash32 = FinalCommitV1(
                job_id32=job_id32,
                policy_hash32=policy_hash32,
                final_root32=bundle.final_root32,
                result_hash32=result_hash32,
            ).digest32(domain=dom)
            pubkey64 = secp256k1_pubkey_from_privkey(priv0)
            sig65 = bundle.signatures[0].sig65
            assert secp256k1_verify_hash(pubkey64, msg_hash32, sig65)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


