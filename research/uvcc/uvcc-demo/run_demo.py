from __future__ import annotations

# pyright: reportMissingImports=false

# UVCC_REQ_GROUP: uvcc_group_c3fb595c9212b029

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-client"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-party"))


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stdout}")
    return str(p.stdout)


def _deploy_forge_create(*, cwd: Path, rpc_url: str, privkey_hex: str, contract: str, args: list[str]) -> str:
    cmd = ["forge", "create", "--broadcast", "--rpc-url", rpc_url, "--private-key", privkey_hex, contract]
    if args:
        cmd += ["--constructor-args"] + args
    out = _run(cmd, cwd=cwd)
    m = re.search(r"Deployed to:\s*(0x[0-9a-fA-F]{40})", out)
    if not m:
        raise RuntimeError(f"failed to parse deployed address for {contract}:\n{out}")
    return m.group(1)


def _cast_send(*, rpc_url: str, privkey_hex: str, to: str, sig: str, args: list[str]) -> str:
    cmd = ["cast", "send", "--rpc-url", rpc_url, "--private-key", privkey_hex, to, sig] + args
    return _run(cmd)


def _cast_call(*, rpc_url: str, to: str, sig: str, args: list[str]) -> str:
    cmd = ["cast", "call", "--rpc-url", rpc_url, to, sig] + args
    return _run(cmd).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for proof_bundle.json/transcript_v1.jsonl")
    args = ap.parse_args()

    out_dir = Path(str(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _add_paths()
    from uvcc_client import run_demo_job_v1
    from eth_utils.crypto import keccak
    from uvcc_party.eip712 import EIP712DomainV1, PolicyCommitV1

    # Start local anvil with deterministic config output.
    anvil_port = _free_port()
    rpc_url = f"http://127.0.0.1:{anvil_port}"

    with tempfile.TemporaryDirectory() as td:
        conf_path = Path(td) / "anvil.json"
        proc = subprocess.Popen(
            [
                "anvil",
                "--port",
                str(anvil_port),
                "--accounts",
                "2",  # deployer + client (parties are external signers)
                "--mnemonic",
                "test test test test test test test test test test test junk",
                "--config-out",
                str(conf_path),
                "--silent",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for config file.
            for _ in range(200):
                if conf_path.exists() and conf_path.stat().st_size > 0:
                    break
                time.sleep(0.02)
            if not conf_path.exists():
                raise RuntimeError("anvil did not write config-out")
            conf = json.loads(conf_path.read_text(encoding="utf-8"))
            # Foundry config-out format includes accounts with private_keys.
            # We accept either {"private_keys":[...]} or {"accounts":[{"private_key":...},...]}.
            priv0 = None
            if isinstance(conf, dict) and isinstance(conf.get("private_keys", None), list) and conf["private_keys"]:
                priv0 = str(conf["private_keys"][0])
            if priv0 is None and isinstance(conf, dict) and isinstance(conf.get("accounts", None), list) and conf["accounts"]:
                priv0 = str(conf["accounts"][0]["private_key"])
            if priv0 is None:
                raise RuntimeError(f"could not parse anvil private key from config-out: {conf.keys()}")

            # Build+deploy uvcc-contracts.
            contracts_dir = _repo_root() / "research" / "uvcc" / "uvcc-contracts"
            _run(["forge", "build"], cwd=contracts_dir)

            avl = _deploy_forge_create(cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/MockAVL.sol:MockAVL", args=[])
            staking = _deploy_forge_create(cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/AVLStakingManager.sol:AVLStakingManager", args=[avl])
            bonds = _deploy_forge_create(
                cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/ProviderBondRegistry.sol:ProviderBondRegistry", args=[avl, "0x0000000000000000000000000000000000000000"]
            )
            ledger = _deploy_forge_create(cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/UVCCJobLedger.sol:UVCCJobLedger", args=[staking, bonds])
            _cast_send(rpc_url=rpc_url, privkey_hex=priv0, to=bonds, sig="setJobLedger(address)", args=[ledger])

            # Run the MPC demo job and write artifacts into out_dir.
            # EIP-712 domain must match the deployed ledger contract.
            ledger_addr20 = bytes.fromhex(ledger[2:])  # 20 bytes
            dom = EIP712DomainV1(chain_id=31337, verifying_contract=ledger_addr20)
            art = run_demo_job_v1(out_dir=str(out_dir), eip712_chain_id=int(dom.chain_id), eip712_verifying_contract20=bytes(dom.verifying_contract))

            # Create job on-chain (client = deployer for demo).
            policy_hex = "0x" + art.policy_hash32.hex()
            nonce_hex = "0x" + art.client_nonce32.hex()
            p0, p1, p2 = art.party_addresses

            # PolicyCommit values (demo constants).
            job_id_hex = "0x" + art.job_id32.hex()
            sid_hash32 = bytes(keccak(bytes(b"sid-uvcc-demo")))
            sgir_hash32 = bytes(keccak(bytes(b"uvcc.demo.sgir.v1")))
            runtime_hash32 = bytes(keccak(bytes(b"uvcc.demo.runtime.v1")))
            fss_dir_hash32 = bytes(keccak(bytes(b"uvcc.demo.fssdir.v1")))
            preproc_hash32 = bytes(keccak(bytes(b"uvcc.demo.preproc.v1")))

            pc = PolicyCommitV1(
                job_id32=art.job_id32,
                policy_hash32=art.policy_hash32,
                sid_hash32=sid_hash32,
                sgir_hash32=sgir_hash32,
                runtime_hash32=runtime_hash32,
                fss_dir_hash32=fss_dir_hash32,
                preproc_hash32=preproc_hash32,
                backend_u8=0,
                epoch_u64=0,
            )
            digest_pc = pc.digest32(domain=dom)
            sig_pc = {}
            for pid_s, priv_hex in zip(("P0", "P1", "P2"), art.party_privkeys_hex):
                priv32 = bytes.fromhex(priv_hex[2:])
                from uvcc_party.sig import secp256k1_sign_hash

                sig_pc[pid_s] = secp256k1_sign_hash(priv32, digest_pc)
            sig0_pc = "0x" + sig_pc["P0"].hex()
            sig1_pc = "0x" + sig_pc["P1"].hex()
            sig2_pc = "0x" + sig_pc["P2"].hex()

            pc_tuple = (
                f"({job_id_hex},{policy_hex},0x{sid_hash32.hex()},0x{sgir_hash32.hex()},0x{runtime_hash32.hex()},"
                f"0x{fss_dir_hash32.hex()},0x{preproc_hash32.hex()},0,0)"
            )
            _cast_send(
                rpc_url=rpc_url,
                privkey_hex=priv0,
                to=ledger,
                sig="createJob((bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8,uint64),address[3],bytes,bytes,bytes)",
                args=[pc_tuple, f"[{p0},{p1},{p2}]", sig0_pc, sig1_pc, sig2_pc],
            )

            # Submit final proof on-chain.
            proof = json.loads((out_dir / "proof_bundle.json").read_text(encoding="utf-8"))
            final_root = proof["transcript"]["final_root"]
            result_hash = proof["verdict"]["result_hash"]
            sigs = {s["party_id"]: base64.b64decode(s["sig65_b64"]) for s in proof["signatures"]}
            sig0 = "0x" + sigs["P0"].hex()
            sig1 = "0x" + sigs["P1"].hex()
            sig2 = "0x" + sigs["P2"].hex()
            job_id_hex = proof["job"]["job_id"]

            _cast_send(
                rpc_url=rpc_url,
                privkey_hex=priv0,
                to=ledger,
                sig="submitFinal(bytes32,bytes32,bytes32,bytes,bytes,bytes)",
                args=[job_id_hex, final_root, result_hash, sig0, sig1, sig2],
            )

            # Check on-chain state matches proof.
            on_final_root = _cast_call(
                rpc_url=rpc_url,
                to=ledger,
                sig="jobs(bytes32)(bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8,uint64,address,uint64,bool,bytes32,bytes32,uint64,bool,address,uint64,bool,bool)",
                args=[job_id_hex],
            )
            # cast returns a tuple-like string; we only sanity check that final_root/result_hash appear in it.
            if final_root[2:].lower() not in on_final_root.lower():
                raise RuntimeError(f"on-chain final_root mismatch: {on_final_root}")
            if result_hash[2:].lower() not in on_final_root.lower():
                raise RuntimeError(f"on-chain result_hash mismatch: {on_final_root}")

            # Verify using uvcc-verifier CLI.
            verifier_dir = _repo_root() / "research" / "uvcc" / "uvcc-verifier"
            _run(
                [
                    sys.executable,
                    "-m",
                    "uvcc_verifier",
                    "verify",
                    "--proof",
                    str(out_dir / "proof_bundle.json"),
                    "--transcript",
                    str(out_dir / "transcript_v1.jsonl"),
                ],
                cwd=verifier_dir,
            )

            print("UVCC demo complete.")
            print(f"- rpc_url: {rpc_url}")
            print(f"- ledger:  {ledger}")
            print(f"- job_id:  {job_id_hex}")
            print(f"- final_root: {final_root}")
            print(f"- result_hash: {result_hash}")
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())


