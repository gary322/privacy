from __future__ import annotations

# pyright: reportMissingImports=false

# UVCC_REQ_GROUP: uvcc_group_c3fb595c9212b029

import base64
import concurrent.futures as cf
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _hex32(b: bytes) -> str:
    return "0x" + bytes(b).hex()


@dataclass(frozen=True)
class DemoJobArtifactsV1:
    out_dir: Path
    policy_hash32: bytes
    client_nonce32: bytes
    job_id32: bytes
    final_root32: bytes
    result_hash32: bytes
    proof_bundle_json: bytes
    transcript_jsonl: bytes
    party_addresses: Tuple[str, str, str]
    party_privkeys_hex: Tuple[str, str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _import_uvcc_party_and_verifier() -> None:
    # Ensure we can import sibling projects without installation.
    root = _repo_root()
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-party"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-verifier"))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _start_relay(port: int, db_path: str) -> subprocess.Popen:
    repo_root = _repo_root()
    relay_py = repo_root / "research" / "uvcc" / "uvcc-relay" / "relay_server.py"
    if not relay_py.exists():
        raise FileNotFoundError(str(relay_py))
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
    from uvcc_party.relay_client import RelayClient

    rc = RelayClient(base_url=base_url, group_id="health", token=None, timeout_s=2.0)
    for _ in range(200):
        try:
            rc.healthz()
            return
        except Exception:
            time.sleep(0.02)
    raise RuntimeError("relay never became healthy")


def _det_nonce32(*, job_id32: bytes, pid: int) -> bytes:
    return _sha256(b"UVCC.demo.nonce.v1\0" + bytes(job_id32) + struct.pack("<I", int(pid) & 0xFFFFFFFF))


def _dump_transcript_jsonl(leaves) -> bytes:
    # leaves are uvcc_party.transcript.TranscriptLeafV1 objects.
    out_lines: List[str] = []
    for lf in leaves:
        rec = {"body_b64": _b64(lf.body_bytes), "leaf_hash_hex": _hex32(lf.leaf_hash32)}
        out_lines.append(json.dumps(rec, sort_keys=True, separators=(",", ":")))
    return ("\n".join(out_lines) + "\n").encode("utf-8")


def run_demo_job_v1(
    *,
    out_dir: str,
    sid: bytes = b"sid-uvcc-demo",
    policy_hash32: Optional[bytes] = None,
    client_nonce32: Optional[bytes] = None,
    party_privkeys32: Optional[Sequence[bytes]] = None,
    eip712_chain_id: int = 31337,
    eip712_verifying_contract20: Optional[bytes] = None,
) -> DemoJobArtifactsV1:
    """
    Runs a deterministic local UVCC demo job (CPU reference), producing:
    - transcript JSONL (union transcript)
    - proof bundle JSON signed by three parties
    """
    _import_uvcc_party_and_verifier()

    from eth_utils.crypto import keccak
    from uvcc_party.gemm import op_gemm_tile_beaver_tcf_v0a_u64_v1
    from uvcc_party.open import OpenArithItemU64, open_arith_u64_round_v1
    from uvcc_party.party import Party
    from uvcc_party.proof_bundle import ProofBundleV1, party_from_privkey, sign_final_root_v1
    from uvcc_party.eip712 import EIP712DomainV1
    from uvcc_party.relay_client import RelayClient
    from uvcc_party.rss import make_rss_arith_u64_triple
    from uvcc_party.sks import sks_epoch_setup_v1, sks_freivalds_check_tile_gemm_u64_v1
    from uvcc_party.tcf import tcf_gen_v1
    from uvcc_verifier.proof_bundle_v1 import parse_proof_bundle_json_v1, verify_proof_bundle_v1
    from uvcc_verifier.transcript_v1 import compute_epoch_roots_v1, compute_final_root_v1, parse_transcript_jsonl_v1, validate_transcript_leaves_v1

    outp = Path(out_dir).resolve()
    outp.mkdir(parents=True, exist_ok=True)

    if policy_hash32 is None:
        # On-chain policy hash is keccak (v1 profile).
        policy_hash32 = keccak(b"uvcc.demo.policy.v1")
    if client_nonce32 is None:
        client_nonce32 = keccak(b"uvcc.demo.client_nonce.v1")
    if len(policy_hash32) != 32 or len(client_nonce32) != 32:
        raise ValueError("policy_hash32/client_nonce32 must be 32 bytes")

    # job_id32 matches uvcc-contracts UVCCJobLedger.computeJobId().
    job_id32 = keccak(b"UVCC.jobid.v1\0" + policy_hash32 + client_nonce32)

    if party_privkeys32 is None:
        # Deterministic demo keys (NOT FOR PRODUCTION).
        party_privkeys32 = [b"\x01" * 32, b"\x02" * 32, b"\x03" * 32]
    if len(party_privkeys32) != 3:
        raise ValueError("party_privkeys32 must have length 3")
    privs = [bytes(k) for k in party_privkeys32]
    if any(len(k) != 32 for k in privs):
        raise ValueError("each privkey must be 32 bytes")

    # Derive Ethereum addresses (for uvcc-demo/contract interop).
    from uvcc_party.sig import secp256k1_eth_address_from_pubkey, secp256k1_pubkey_from_privkey

    addrs = []
    for k in privs:
        pub = secp256k1_pubkey_from_privkey(k)
        addr20 = secp256k1_eth_address_from_pubkey(pub)
        addrs.append("0x" + addr20.hex())

    # Start relay.
    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "relay.sqlite")
        proc = _start_relay(port, db_path)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_health(base)

            # TCF keys (deterministic).
            master_seed32 = keccak(b"uvcc.demo.tcf.seed.v1")
            k0, k1, k2 = tcf_gen_v1(master_seed32=bytes(master_seed32), sid=sid)
            tcf_keys = {0: k0, 1: k1, 2: k2}

            # Deterministic public inputs.
            d = 16
            gen = torch.Generator(device="cpu").manual_seed(424242)
            loX = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            hiX = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            X_pub = (hiX << 32) | loX
            loY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            hiY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            Y_pub = (hiY << 32) | loY

            X0, X1, X2 = make_rss_arith_u64_triple(x_pub=X_pub, generator=gen, device=torch.device("cpu"))
            Y0, Y1, Y2 = make_rss_arith_u64_triple(x_pub=Y_pub, generator=gen, device=torch.device("cpu"))

            def run_party(pid: int) -> Tuple[bytes, List[bytes], bytes]:
                relay = RelayClient(base_url=base, group_id="g-demo", token=None, timeout_s=120.0)
                party = Party(party_id=pid, job_id32=job_id32, sid=sid, relay=relay)

                # Epoch randomness (deterministic nonces for demo reproducibility).
                st = sks_epoch_setup_v1(party, sid=sid, epoch=0, step=1000, nonce32=_det_nonce32(job_id32=job_id32, pid=pid))
                assert st.epoch_rand32 is not None

                # Secure GEMM tile.
                res = op_gemm_tile_beaver_tcf_v0a_u64_v1(
                    party,
                    X={0: X0, 1: X1, 2: X2}[pid],
                    Y={0: Y0, 1: Y1, 2: Y2}[pid],
                    tcf_key=tcf_keys[pid],
                    op_id=7,
                    tile_i=0,
                    tile_j=0,
                    tile_p=0,
                    epoch=0,
                    step=0,
                    sgir_op_id=123,
                    fxp_frac_bits=0,
                    d=d,
                )

                # SKS Freivalds check (always selected).
                ok = sks_freivalds_check_tile_gemm_u64_v1(
                    party,
                    sid=sid,
                    epoch_rand32=st.epoch_rand32,
                    epoch=0,
                    step=2000,
                    sgir_op_id=123,
                    kernel_instance_id=0,
                    sks_sample_log2=0,
                    t_checks=3,
                    field_id=0,
                    Z=res.Z,
                    triple_A=res.triple_A,
                    triple_B=res.triple_B,
                    triple_C=res.triple_C,
                    E_pub=res.E_pub,
                    F_pub=res.F_pub,
                )
                if ok is not True:
                    raise RuntimeError("SKS check failed in demo")

                # OPEN the output tile to compute a deterministic result_hash32.
                open_id = 0xDEADBEEF
                out = open_arith_u64_round_v1(
                    party,
                    items=[OpenArithItemU64(open_id=open_id, sub_id=0, x=res.Z)],
                    epoch=0,
                    step=3000,
                    round=0,
                    sgir_op_id=999,
                )
                Z_pub = out[(open_id, 0)].contiguous().view(-1)
                z_bytes = bytearray()
                for v in Z_pub.cpu().tolist():
                    z_bytes += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
                result_hash32 = _sha256(bytes(z_bytes))

                # Return transcript leaves and result hash.
                leaves = party.transcript.leaves() if party.transcript is not None else []
                return bytes(result_hash32), [lf.body_bytes for lf in leaves], bytes(party.sid_hash32())

            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                f0 = ex.submit(run_party, 0)
                f1 = ex.submit(run_party, 1)
                f2 = ex.submit(run_party, 2)
                r0, bodies0, _ = f0.result(timeout=300)
                r1, bodies1, _ = f1.result(timeout=300)
                r2, bodies2, _ = f2.result(timeout=300)

            if r0 != r1 or r0 != r2:
                raise RuntimeError("result_hash mismatch across parties")
            result_hash32 = r0

            # Union transcript: concatenate all leaf bodies (verifier will recompute hashes).
            all_bodies = list(bodies0) + list(bodies1) + list(bodies2)
            transcript_jsonl = "\n".join(json.dumps({"body_b64": _b64(b)}, sort_keys=True, separators=(",", ":")) for b in all_bodies) + "\n"
            transcript_jsonl_bytes = transcript_jsonl.encode("utf-8")

            # Compute transcript roots via verifier logic.
            with tempfile.TemporaryDirectory() as td2:
                tpath = Path(td2) / "t.jsonl"
                tpath.write_bytes(transcript_jsonl_bytes)
                leaves_parsed = parse_transcript_jsonl_v1(str(tpath))
                validate_transcript_leaves_v1(leaves_parsed, strict_unknown_msg_kind=False, strict_netframe_header_hash=True)
                roots_by_epoch = compute_epoch_roots_v1(leaves_parsed)
                epoch_roots = [roots_by_epoch.get(0, b"")]
                if len(epoch_roots[0]) != 32:
                    raise RuntimeError("missing epoch root")
                final_root32 = compute_final_root_v1(epoch_roots=epoch_roots)

            # Build proof bundle (3 parties, 3 signatures).
            parties = [party_from_privkey(party_id=i, privkey32=privs[i]) for i in (0, 1, 2)]
            if eip712_verifying_contract20 is None:
                eip712_verifying_contract20 = b"\x00" * 20
            dom = EIP712DomainV1(chain_id=int(eip712_chain_id), verifying_contract=bytes(eip712_verifying_contract20))
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
                sgir_hash32=keccak(b"uvcc.demo.sgir.v1"),
                runtime_hash32=keccak(b"uvcc.demo.runtime.v1"),
                backend="CRYPTO_CC_3PC",
                parties=parties,
                epoch_roots=epoch_roots,
                final_root32=final_root32,
                signatures=sigs,
                result_hash32=result_hash32,
                status="OK",
            )
            proof_json = pb.to_json_bytes()

            # Self-check verifier on produced artifacts.
            proof_parsed = parse_proof_bundle_json_v1(proof_json)
            verify_proof_bundle_v1(proof=proof_parsed, transcript_epoch_roots=epoch_roots, transcript_final_root32=final_root32)

            # Write outputs.
            (outp / "proof_bundle.json").write_bytes(proof_json)
            (outp / "transcript_v1.jsonl").write_bytes(transcript_jsonl_bytes)

            return DemoJobArtifactsV1(
                out_dir=outp,
                policy_hash32=bytes(policy_hash32),
                client_nonce32=bytes(client_nonce32),
                job_id32=bytes(job_id32),
                final_root32=bytes(final_root32),
                result_hash32=bytes(result_hash32),
                proof_bundle_json=proof_json,
                transcript_jsonl=transcript_jsonl_bytes,
                party_addresses=(addrs[0], addrs[1], addrs[2]),
                party_privkeys_hex=("0x" + privs[0].hex(), "0x" + privs[1].hex(), "0x" + privs[2].hex()),
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


