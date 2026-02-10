from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-client"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-party"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-verifier"))


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


def _build_uvcc_bundle_tgz(out_path: Path) -> None:
    root = _repo_root()
    targets = [
        root / "research" / "uvcc" / "uvcc-client",
        root / "research" / "uvcc" / "uvcc-party",
        root / "research" / "uvcc" / "uvcc-relay",
        root / "research" / "uvcc" / "requirements-uvcc-base.txt",
    ]
    for t in targets:
        if not t.exists():
            raise FileNotFoundError(str(t))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(out_path), "w:gz") as tf:
        for t in targets:
            arc = t.relative_to(root)
            tf.add(str(t), arcname=str(arc))


def _tls_ca_and_server_cert(*, host_or_ip: str) -> Tuple[bytes, bytes, bytes]:
    """
    Return (ca_cert_pem, server_cert_pem, server_key_pem) for relay TLS.
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "UVCC Relay CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=14))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    srv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cn = str(host_or_ip).strip()
    srv_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    san_items = []
    try:
        san_items.append(x509.IPAddress(ipaddress.ip_address(cn)))
    except Exception:
        san_items.append(x509.DNSName(cn))
    # Also allow local access (party0 may use loopback to avoid hairpin NAT / port mapping issues).
    for extra in ("127.0.0.1", "localhost"):
        try:
            san_items.append(x509.IPAddress(ipaddress.ip_address(extra)))
        except Exception:
            san_items.append(x509.DNSName(extra))
    san = x509.SubjectAlternativeName(san_items)
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name)
        .issuer_name(ca_cert.subject)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=14))
        .add_extension(san, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    srv_pem = srv_cert.public_bytes(serialization.Encoding.PEM)
    key_pem = srv_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption())
    return ca_pem, srv_pem, key_pem


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


def _summarize_gpu_telemetry_csv(path: Path) -> Dict[str, Any]:
    """
    Best-effort summary of nvidia-smi CSV telemetry captured via:
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"rows": 0}
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        return {"rows": 0}
    # Parse simple numeric columns; tolerate headerless csv.
    util_g: List[float] = []
    mem_used: List[float] = []
    pwr: List[float] = []
    temp: List[float] = []
    for ln in lines:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 9:
            continue
        # parts[0]=timestamp, [1]=index, [2]=name
        try:
            util_g.append(float(parts[3]))
        except Exception:
            pass
        try:
            mem_used.append(float(parts[5]))
        except Exception:
            pass
        try:
            pwr.append(float(parts[7]))
        except Exception:
            pass
        try:
            temp.append(float(parts[8]))
        except Exception:
            pass
    def avg(xs: List[float]) -> Optional[float]:
        return (sum(xs) / float(len(xs))) if xs else None
    def mx(xs: List[float]) -> Optional[float]:
        return max(xs) if xs else None
    return {
        "rows": int(len(lines)),
        "util_gpu_avg": avg(util_g),
        "util_gpu_max": mx(util_g),
        "mem_used_max_mib": mx(mem_used),
        "power_draw_max_w": mx(pwr),
        "temp_max_c": mx(temp),
    }


def _trim(s: str, n: int = 2000) -> str:
    t = str(s)
    if len(t) <= int(n):
        return t
    return t[: int(n)] + f"...(truncated,{len(t)} chars)"


class RunLoggerV1:
    """
    Writes:
      - run_full.log   (human-readable, append-only)
      - run_full.jsonl (machine-readable, one JSON object per line)
    """

    def __init__(self, *, out_dir: Path):
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "run_full.log"
        self.jsonl_path = self.out_dir / "run_full.jsonl"
        self._t0 = time.monotonic()
        # Avoid mixing logs across reruns: rotate existing logs if present.
        if self.log_path.exists():
            ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            self.log_path.replace(self.out_dir / f"run_full.{ts}.log")
        if self.jsonl_path.exists():
            ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            self.jsonl_path.replace(self.out_dir / f"run_full.{ts}.jsonl")

    def _write_text(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def log(self, msg: str) -> None:
        line = f"[{_now_iso_utc()}] {msg}\n"
        # Print for operator visibility and persist to file.
        print(line, end="")
        self._write_text(line)

    def event(self, name: str, **fields: Any) -> None:
        # Enforce basic secret hygiene: drop any obvious secret-bearing keys.
        redacted: Dict[str, Any] = {}
        for k, v in fields.items():
            lk = str(k).lower()
            if "token" in lk or "api_key" in lk or "private_key" in lk or "privkey" in lk:
                continue
            redacted[str(k)] = v
        rec = {
            "ts": _now_iso_utc(),
            "t_rel_s": round(time.monotonic() - self._t0, 6),
            "event": str(name),
            "fields": redacted,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        # Also emit to the human-readable log for end-to-end operator visibility.
        # Keep it single-line to make `tail -f` usable during long runs.
        self._write_text(
            f"[{rec['ts']}] EVENT {rec['event']} {json.dumps(rec['fields'], sort_keys=True, separators=(',', ':'))}\n"
        )


def _pick_prime_image(*, requested: str, available_images: Optional[List[str]]) -> str:
    """
    Choose a Prime image name compatible with the selected provider+cloud offer.

    Preference: latest CUDA+PyTorch if available; otherwise fall back to ubuntu CUDA base.
    """
    req = str(requested or "").strip()
    imgs = list(available_images) if available_images else []
    imgs_l = {str(x).lower(): str(x) for x in imgs}
    if req:
        hit = imgs_l.get(req.lower())
        if hit is not None:
            return str(hit)
    # Prefer PyTorch images in descending preference order.
    pref = [
        "cuda_12_6_pytorch_2_7",
        "cuda_12_4_pytorch_2_7",
        "cuda_12_4_pytorch_2_6",
        "cuda_12_4_pytorch_2_5",
        "cuda_12_4_pytorch_2_4",
        "cuda_12_1_pytorch_2_2",
        "cuda_11_8_pytorch_2_1",
        "ubuntu_22_cuda_12",
    ]
    for p in pref:
        hit = imgs_l.get(p.lower())
        if hit is not None:
            return str(hit)
    # Last resort: first available image if present.
    if imgs:
        return str(imgs[0])
    # Absolute fallback: most common base image.
    return "ubuntu_22_cuda_12"


@dataclass
class RemoteNodeV1:
    party_id: int
    pod_id: str
    provider_type: str
    ssh_user: str
    ssh_host: str
    ssh_port: int
    home: str
    root_dir: str
    venv_dir: str

    address: Optional[str] = None
    pubkey64_hex: Optional[str] = None

    def venv_python(self) -> str:
        return f"{self.venv_dir}/bin/python3"

    def env_prefix(self) -> str:
        py_path = ":".join(
            [
                f"{self.root_dir}/research/uvcc/uvcc-client",
                f"{self.root_dir}/research/uvcc/uvcc-party",
                f"{self.root_dir}/research/uvcc/uvcc-verifier",
            ]
        )
        # Ensure venv binaries (e.g., ninja) are on PATH for torch.utils.cpp_extension.
        return f"PATH={self.venv_dir}/bin:$PATH PYTHONPATH={shlex.quote(py_path)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output dir for proof_bundle.json/transcript_v1.jsonl")
    ap.add_argument("--ssh-key-path", default=None, help="SSH private key PEM path for Prime pods (defaults to best-effort ~/.ssh/* key)")
    ap.add_argument("--prime-api-key-env", default="UVCC_PRIME_API_KEY", help="Env var containing Prime API key")
    ap.add_argument(
        "--prime-api-key-path",
        default=None,
        help="Optional path to a file containing the Prime API key (safer than putting the key in shell history). If set, overrides --prime-api-key-env.",
    )

    ap.add_argument("--job", default="train_v1", help="Job kind (default: train_v1)")
    ap.add_argument("--job-json", default=None, help="Optional JSON job spec overriding built-in job defaults")

    ap.add_argument("--reuse-pod-id", action="append", default=None, help="Reuse an existing Prime pod id (repeat 3 times). Skips provisioning.")

    ap.add_argument("--cloud-id", default=None, help="Prime pod.cloudId (if omitted, auto-selected from /availability/gpu)")
    ap.add_argument("--provider-type", default="auto", help="Prime provider.type (default: auto)")
    ap.add_argument(
        "--providers",
        default=None,
        help=(
            "Optional per-party provider types (comma-separated, exactly 3, must be distinct). "
            "Order is party0,party1,party2. Example: hyperstack,crusoecloud,runpod. "
            "If set, overrides --provider-type for initial provisioning and failover replacement."
        ),
    )
    ap.add_argument("--gpu-type", default=None, help="Prime pod.gpuType (if omitted, auto-selected from /availability/gpu)")
    ap.add_argument("--gpu-count", type=int, default=1, help="Prime pod.gpuCount")
    ap.add_argument("--socket", default="PCIe", help="Prime pod.socket")
    ap.add_argument("--image", default="cuda_12_6_pytorch_2_7", help="Prime pod.image")
    ap.add_argument("--max-price", type=float, default=None, help="Prime pod.maxPrice")

    ap.add_argument("--prefer-region", action="append", default=None, help="Optional region preference (repeatable)")
    ap.add_argument("--prefer-gpu-type", action="append", default=None, help="Optional gpuType preference (repeatable)")

    ap.add_argument("--relay-port", type=int, default=8443)
    ap.add_argument("--run-gpu-tests", choices=["true", "false"], default="true")
    ap.add_argument("--party-timeout-s", type=int, default=1800, help="Max seconds to wait for each party to produce result.json (default: 1800)")
    ap.add_argument("--party-log-level", default="info", choices=["quiet", "info", "debug", "trace"], help="Verbosity for per-party run.log (default: info)")
    ap.add_argument("--gpu-telemetry", choices=["true", "false"], default="true", help="If true, capture per-node nvidia-smi telemetry during training (default: true)")
    ap.add_argument("--gpu-telemetry-interval-s", type=float, default=1.0, help="Sampling interval for nvidia-smi telemetry (seconds, default: 1.0)")
    ap.add_argument("--keep-pods", choices=["true", "false"], default="false", help="If false (default), best-effort delete pods at end")

    # Reliability / failover controls (recommended ON for long runs and flaky providers).
    ap.add_argument("--enable-failover", choices=["true", "false"], default="true", help="If true, attempt failover+resume using checkpoints when a pod/party fails (default: true)")
    ap.add_argument("--failover-max-epochs", type=int, default=3, help="Max number of epochs/attempts (0=only one try). Default: 3")
    ap.add_argument("--checkpoint-enable", choices=["true", "false"], default="true", help="If true, parties write private per-step checkpoint shares for resume (default: true)")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint every N local steps within an epoch (default: 1)")
    ap.add_argument("--start-live-recorder", choices=["true", "false"], default="true", help="If true, start local append-only live recorder to persist logs/telemetry/checkpoints during the run (default: true)")
    ap.add_argument("--live-recorder-interval-s", type=float, default=10.0, help="Polling interval for live recorder (seconds, default: 10)")
    args = ap.parse_args()

    out_dir = Path(str(args.out)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLoggerV1(out_dir=out_dir)
    # Record the runner PID so watchdogs / operators can detect liveness and clean up if needed.
    try:
        (out_dir / "runner.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    except Exception:
        pass
    log.log("UVCC Prime 3-node run starting")
    private_keep = out_dir / "private_keep"
    private_keep.mkdir(parents=True, exist_ok=True)
    live_recorder_pid: Optional[int] = None
    log.event(
        "config",
        out_dir=str(out_dir),
        job=str(args.job),
        cloud_id=str(args.cloud_id) if args.cloud_id is not None else "",
        gpu_type=str(args.gpu_type) if args.gpu_type is not None else "",
        gpu_count=int(args.gpu_count),
        provider_type=str(args.provider_type),
        socket=str(args.socket),
        image=str(args.image),
        run_gpu_tests=str(args.run_gpu_tests),
        relay_port=int(args.relay_port),
        party_timeout_s=int(args.party_timeout_s),
        party_log_level=str(args.party_log_level),
        gpu_telemetry=str(args.gpu_telemetry),
        gpu_telemetry_interval_s=float(args.gpu_telemetry_interval_s),
        keep_pods=str(args.keep_pods),
    )

    _add_paths()

    from eth_utils.crypto import keccak
    from uvcc_client.prime_api import PrimePodSpecV1, prime_client_from_env_v1
    from uvcc_client.party_identity import load_or_create_party_privkey32_v1, party_identity_from_privkey_v1
    from uvcc_client.ssh_runner import load_private_key_from_file, sftp_get_file, sftp_put_bytes, sftp_put_file, ssh_connect_with_retries, ssh_exec
    from uvcc_party.eip712 import EIP712DomainV1, FinalCommitV1, PolicyCommitV1
    from uvcc_party.proof_bundle import ProofBundlePartyV1, ProofBundleSignatureV1, ProofBundleV1
    from uvcc_party.rss import make_rss_arith_u64_triple
    from uvcc_party.tcf import tcf_gen_v1
    from uvcc_verifier.proof_bundle_v1 import parse_proof_bundle_json_v1, verify_proof_bundle_v1
    from uvcc_verifier.transcript_v1 import compute_epoch_roots_v1, compute_final_root_v1, parse_transcript_jsonl_v1, validate_transcript_leaves_v1

    # Start local anvil with deterministic config output.
    prime = None
    created_pod_ids: List[str] = []
    anvil_port = _free_port()
    rpc_url = f"http://127.0.0.1:{anvil_port}"
    with tempfile.TemporaryDirectory() as td:
        conf_path = Path(td) / "anvil.json"
        proc_anvil = subprocess.Popen(
            [
                "anvil",
                "--port",
                str(anvil_port),
                "--accounts",
                "2",
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
            for _ in range(200):
                if conf_path.exists() and conf_path.stat().st_size > 0:
                    break
                time.sleep(0.02)
            if not conf_path.exists():
                raise RuntimeError("anvil did not write config-out")
            conf = json.loads(conf_path.read_text(encoding="utf-8"))
            priv0 = None
            if isinstance(conf, dict) and isinstance(conf.get("private_keys", None), list) and conf["private_keys"]:
                priv0 = str(conf["private_keys"][0])
            if priv0 is None and isinstance(conf, dict) and isinstance(conf.get("accounts", None), list) and conf["accounts"]:
                priv0 = str(conf["accounts"][0]["private_key"])
            if priv0 is None:
                raise RuntimeError("could not parse anvil private key")

            # Build+deploy uvcc-contracts.
            contracts_dir = _repo_root() / "research" / "uvcc" / "uvcc-contracts"
            t_contracts0 = time.monotonic()
            _run(["forge", "build"], cwd=contracts_dir)
            avl = _deploy_forge_create(cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/MockAVL.sol:MockAVL", args=[])
            staking = _deploy_forge_create(cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/AVLStakingManager.sol:AVLStakingManager", args=[avl])
            bonds = _deploy_forge_create(
                cwd=contracts_dir,
                rpc_url=rpc_url,
                privkey_hex=priv0,
                contract="src/ProviderBondRegistry.sol:ProviderBondRegistry",
                args=[avl, "0x0000000000000000000000000000000000000000"],
            )
            ledger = _deploy_forge_create(cwd=contracts_dir, rpc_url=rpc_url, privkey_hex=priv0, contract="src/UVCCJobLedger.sol:UVCCJobLedger", args=[staking, bonds])
            _cast_send(rpc_url=rpc_url, privkey_hex=priv0, to=bonds, sig="setJobLedger(address)", args=[ledger])
            log.event(
                "contracts_deployed",
                t_s=round(time.monotonic() - t_contracts0, 3),
                rpc_url=str(rpc_url),
                avl=str(avl),
                staking=str(staking),
                bonds=str(bonds),
                ledger=str(ledger),
            )

            ledger_addr20 = bytes.fromhex(ledger[2:])
            dom = EIP712DomainV1(chain_id=31337, verifying_contract=ledger_addr20)

            # Job ids / policy inputs.
            policy_hash32 = bytes(keccak(b"uvcc.prime3pc.policy.v1"))
            client_nonce32 = bytes(keccak(b"uvcc.prime3pc.client_nonce.v1"))
            job_id32 = bytes(keccak(b"UVCC.jobid.v1\0" + policy_hash32 + client_nonce32))
            sid = f"sid-uvcc-prime3pc-{job_id32.hex()[:16]}".encode("utf-8")
            sid_hash32 = bytes(keccak(bytes(sid)))

            sgir_hash32 = bytes(keccak(b"uvcc.prime3pc.sgir.v1"))
            runtime_hash32 = bytes(keccak(b"uvcc.prime3pc.runtime.v1"))
            fss_dir_hash32 = bytes(keccak(b"uvcc.prime3pc.fssdir.v1"))
            preproc_hash32 = bytes(keccak(b"uvcc.prime3pc.preproc.v1"))

            # Load job config.
            job_cfg: Dict[str, Any] = {}
            if args.job_json is not None:
                job_cfg = json.loads(Path(str(args.job_json)).expanduser().read_text(encoding="utf-8"))
                if not isinstance(job_cfg, dict):
                    raise RuntimeError("job-json must be an object")
            else:
                job_cfg = {"kind": str(args.job), "d": 16, "steps": 1, "seed": 424242, "require_cuda": True, "fxp_frac_bits": 0}
            if str(job_cfg.get("kind") or str(args.job)) != "train_v1":
                raise RuntimeError("only job kind train_v1 is supported in this runner")
            d_job = int(job_cfg.get("d") or 16)
            steps_job = int(job_cfg.get("steps") or 1)
            seed_job = int(job_cfg.get("seed") or 424242)
            require_cuda_job = bool(job_cfg.get("require_cuda") if "require_cuda" in job_cfg else True)
            fxp_job = int(job_cfg.get("fxp_frac_bits") or 0)
            sks_t_checks_job = int(job_cfg.get("sks_t_checks") or 3)
            sks_sample_log2_job = int(job_cfg.get("sks_sample_log2") or 0)
            if d_job <= 0 or d_job > 1024:
                raise RuntimeError("job d must be 1..1024")
            if steps_job <= 0 or steps_job > 10_000:
                raise RuntimeError("job steps must be 1..10000")
            if fxp_job < 0 or fxp_job > 63:
                raise RuntimeError("fxp_frac_bits must be 0..63")
            if sks_t_checks_job <= 0 or sks_t_checks_job > 255:
                raise RuntimeError("sks_t_checks must be 1..255")
            if sks_sample_log2_job < 0 or sks_sample_log2_job > 20:
                raise RuntimeError("sks_sample_log2 must be 0..20")
            log.event(
                "job_config",
                kind=str(job_cfg.get("kind") or str(args.job)),
                d=int(d_job),
                steps=int(steps_job),
                seed=int(seed_job),
                require_cuda=bool(require_cuda_job),
                fxp_frac_bits=int(fxp_job),
                sks_t_checks=int(sks_t_checks_job),
                sks_sample_log2=int(sks_sample_log2_job),
            )

            # Derive a conservative per-party completion timeout for long training runs.
            # Empirically, each step can take multiple minutes (GEMM + SKS + network), so a fixed default
            # (e.g. 1800s) is not sufficient for larger `steps`. Keep the CLI arg as a *minimum* and bump
            # it upward when necessary.
            #
            # Heuristic: allow ~6 minutes per step + 10 minutes slack for OPEN + finalization.
            # This is intentionally conservative to avoid premature runner failure at the end.
            party_timeout_min_s = max(1800, (int(steps_job) * 360) + 600)
            party_timeout_effective_s = max(int(args.party_timeout_s), int(party_timeout_min_s))
            if party_timeout_effective_s != int(args.party_timeout_s):
                log.log(
                    f"WARN: party_timeout_s={int(args.party_timeout_s)} is too small for steps={int(steps_job)}; "
                    f"bumping to party_timeout_s_effective={int(party_timeout_effective_s)}"
                )
            log.event(
                "party_timeout_effective",
                party_timeout_s=int(args.party_timeout_s),
                party_timeout_min_s=int(party_timeout_min_s),
                party_timeout_effective_s=int(party_timeout_effective_s),
            )

            # Prime pods: reuse (if provided) or provision 3 fresh pods.
            prime_api_env = str(args.prime_api_key_env)
            if args.prime_api_key_path is not None:
                key_path = Path(str(args.prime_api_key_path)).expanduser().resolve()
                if not key_path.exists():
                    raise RuntimeError(f"prime api key file missing: {key_path}")
                api_key = key_path.read_text(encoding="utf-8").strip()
                if not api_key:
                    raise RuntimeError("prime api key file is empty")
                os.environ[prime_api_env] = api_key
                log.event("prime_api_key_loaded", key_path=str(key_path), env_var=str(prime_api_env))
            prime = prime_client_from_env_v1(api_key_env=str(prime_api_env))
            cloud_id = str(args.cloud_id).strip() if args.cloud_id is not None else ""
            gpu_type = str(args.gpu_type).strip() if args.gpu_type is not None else ""
            # Consumer-first defaults (can be overridden by --prefer-gpu-type).
            default_gpu_prefs = [
                "RTX4090_24GB",
                "RTX3090_24GB",
                "RTX5090_32GB",
                "L40S_48GB",
                "L40_48GB",
                "RTX6000Ada_48GB",
                "A6000_48GB",
                "A10_24GB",
                "L4_24GB",
                "T4",
                "A100_80GB",
                "A100_40GB",
            ]
            gpu_prefs = list(args.prefer_gpu_type) if args.prefer_gpu_type else default_gpu_prefs
            region_prefs = list(args.prefer_region) if args.prefer_region else None

            # Optional: mixed-provider run (one provider per party).
            providers_by_pid: Optional[Dict[int, str]] = None
            providers_arg = str(getattr(args, "providers", "") or "").strip()
            if providers_arg:
                parts = [p.strip() for p in str(providers_arg).split(",") if p.strip()]
                if len(parts) != 3:
                    raise RuntimeError("--providers must be a comma-separated list of exactly 3 provider types")
                low = [p.lower() for p in parts]
                if len(set(low)) != 3:
                    raise RuntimeError("--providers must specify 3 distinct provider types (no duplicates)")
                if any(p in ("", "auto") for p in low):
                    raise RuntimeError("--providers entries cannot be empty/auto; pass explicit provider types")
                providers_by_pid = {0: parts[0], 1: parts[1], 2: parts[2]}
                log.event("providers_by_party", providers=[providers_by_pid[i] for i in (0, 1, 2)])

            reuse_ids = [str(x).strip() for x in (args.reuse_pod_id or []) if str(x).strip()]
            pods = None
            chosen_image = str(args.image)

            if reuse_ids:
                if len(reuse_ids) != 3:
                    raise RuntimeError("--reuse-pod-id must be provided exactly 3 times")
                created_pod_ids = list(reuse_ids)
                pods = [prime.wait_active(pid, timeout_s=1800) for pid in created_pod_ids]
                log.event(
                    "prime_pods_reused",
                    pod_ids=list(created_pod_ids),
                    ssh_hosts=[str(p.ssh_host) for p in pods],
                    provider_types=[str((p.status_row or {}).get("providerType") or (p.status_row or {}).get("provider") or "") for p in pods],
                )
            else:
                provider_arg = str(args.provider_type or "").strip()
                provider_filter = None if provider_arg.lower() in ("", "auto") else provider_arg

                # Select a concrete availability offer (cloudId+gpuType+images), then provision 3 pods.
                # If provisioning fails due to capacity, automatically try the next-best offer.
                offers_to_try = []
                if providers_by_pid is not None:
                    # Mixed-provider mode: we select one offer per party based on --providers.
                    # We still use the same outer retry loop, but typically only need one attempt.
                    if cloud_id or gpu_type:
                        raise RuntimeError("--cloud-id/--gpu-type overrides are not supported when using --providers")
                    offers_to_try = [None]  # type: ignore[list-item]
                elif cloud_id and gpu_type:
                    offers_to_try = [None]  # type: ignore[list-item]
                else:
                    offers_to_try = prime.candidate_offers_v1(
                        nodes=3,
                        gpu_count_per_node=int(args.gpu_count),
                        provider_type=provider_filter,
                        socket=str(args.socket),
                        prefer_gpu_types=gpu_prefs,
                        prefer_regions=region_prefs,
                        limit=16,
                    )
                    # If the user explicitly requested a provider, enforce it strictly.
                    if provider_filter is not None:
                        want = str(provider_filter).strip().lower()
                        offers_to_try = [o for o in offers_to_try if (o.provider is not None and str(o.provider).strip().lower() == want)]
                        if not offers_to_try:
                            raise RuntimeError(f"no Prime availability offers for requested provider_type={provider_filter}")
                    # If the job requires CUDA, exclude CPU-only offers (some providers advertise CPU nodes in /availability).
                    if bool(require_cuda_job):
                        before = len(list(offers_to_try))
                        offers_to_try = [
                            o
                            for o in offers_to_try
                            if (
                                not str(o.gpu_type or "").strip().upper().startswith("CPU")
                                and not str(o.gpu_type or "").strip().upper().endswith("CPU_NODE")
                                and not str(o.cloud_id or "").strip().lower().startswith("cpu-")
                            )
                        ]
                        after = len(list(offers_to_try))
                        if before != after:
                            log.event("prime_offers_filtered", reason="require_cuda_exclude_cpu", before=int(before), after=int(after))
                        if not offers_to_try:
                            raise RuntimeError(f"no GPU offers available for require_cuda=true (provider_type={provider_filter or 'auto'})")

                ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                last_err: Optional[str] = None

                for idx, offer in enumerate(offers_to_try):
                    if providers_by_pid is not None:
                        # Mixed-provider mode: one offer per party based on --providers.
                        # IMPORTANT: all 3 parties must run the same Prime image to avoid version skew
                        # (torch/python/cuda) across providers.
                        offers_all_by_pid: Dict[int, List[Any]] = {}
                        img_sets: List[set[str]] = []
                        for pid in (0, 1, 2):
                            prov = str(providers_by_pid[int(pid)]).strip()
                            offers_pid = prime.candidate_offers_v1(
                                nodes=1,
                                gpu_count_per_node=int(args.gpu_count),
                                provider_type=str(prov),
                                socket=str(args.socket),
                                prefer_gpu_types=gpu_prefs,
                                prefer_regions=region_prefs,
                                limit=64,
                            )
                            want = str(prov).strip().lower()
                            offers_pid = [o for o in offers_pid if (o.provider is not None and str(o.provider).strip().lower() == want)]
                            if not offers_pid:
                                raise RuntimeError(f"no Prime availability offers for requested provider_type={prov} (party_id={pid})")
                            if bool(require_cuda_job):
                                before = len(list(offers_pid))
                                offers_pid = [
                                    o
                                    for o in offers_pid
                                    if (
                                        not str(o.gpu_type or "").strip().upper().startswith("CPU")
                                        and not str(o.gpu_type or "").strip().upper().endswith("CPU_NODE")
                                        and not str(o.cloud_id or "").strip().lower().startswith("cpu-")
                                    )
                                ]
                                after = len(list(offers_pid))
                                if before != after:
                                    log.event(
                                        "prime_offers_filtered",
                                        reason="require_cuda_exclude_cpu",
                                        before=int(before),
                                        after=int(after),
                                        provider_type=str(prov),
                                        party_id=int(pid),
                                    )
                                if not offers_pid:
                                    raise RuntimeError(f"no GPU offers available for require_cuda=true (provider_type={prov} party_id={pid})")

                            offers_all_by_pid[int(pid)] = list(offers_pid)
                            s: set[str] = set()
                            for o in offers_pid:
                                for im in (o.images or []):
                                    s.add(str(im).strip().lower())
                            img_sets.append(s)

                        common_imgs = set.intersection(*img_sets) if img_sets else set()
                        # Prefer the requested image, then fall back through the global preference list.
                        pref_imgs: List[str] = []
                        req_img = str(args.image or "").strip()
                        if req_img:
                            pref_imgs.append(req_img)
                        for p in [
                            "cuda_12_6_pytorch_2_7",
                            "cuda_12_4_pytorch_2_7",
                            "cuda_12_4_pytorch_2_6",
                            "cuda_12_4_pytorch_2_5",
                            "cuda_12_4_pytorch_2_4",
                            "cuda_12_1_pytorch_2_2",
                            "cuda_11_8_pytorch_2_1",
                            "ubuntu_22_cuda_12",
                        ]:
                            if p not in pref_imgs:
                                pref_imgs.append(p)

                        common_image = None
                        for p in pref_imgs:
                            if str(p).strip().lower() in common_imgs:
                                common_image = str(p).strip()
                                break
                        if common_image is None:
                            raise RuntimeError(
                                "no common Prime image found across providers for mixed-provider run; "
                                f"providers={[providers_by_pid[i] for i in (0,1,2)]} common_images_count={len(common_imgs)}"
                            )
                        log.event(
                            "prime_common_image_selected",
                            attempt=int(idx),
                            image=str(common_image),
                            providers=[providers_by_pid[i] for i in (0, 1, 2)],
                            common_images_count=int(len(common_imgs)),
                        )

                        specs = []
                        for pid in (0, 1, 2):
                            prov = str(providers_by_pid[int(pid)]).strip()
                            offers_pid = offers_all_by_pid[int(pid)]

                            offer_pid = None
                            img_pid = None
                            for o in offers_pid:
                                for im in (o.images or []):
                                    if str(im).strip().lower() == str(common_image).strip().lower():
                                        offer_pid = o
                                        img_pid = str(im).strip()
                                        break
                                if offer_pid is not None:
                                    break
                            if offer_pid is None or img_pid is None:
                                raise RuntimeError(f"provider {prov} has no offer supporting common image {common_image} (party_id={pid})")

                            dc_id_pid = (
                                str(
                                    offer_pid.raw.get("dataCenterId")
                                    or offer_pid.raw.get("data_center_id")
                                    or offer_pid.raw.get("dataCenterID")
                                    or offer_pid.raw.get("dataCenter")
                                    or offer_pid.raw.get("data_center")
                                    or offer_pid.raw.get("datacenter")
                                    or offer_pid.raw.get("datacenter_id")
                                    or ""
                                ).strip()
                                or None
                            )
                            log.event(
                                "prime_offer_selected",
                                attempt=int(idx),
                                party_id=int(pid),
                                provider_type=str(prov),
                                cloud_id=str(offer_pid.cloud_id),
                                gpu_type=str(offer_pid.gpu_type),
                                gpu_count=int(args.gpu_count),
                                socket=str(args.socket),
                                image=str(img_pid),
                                offer_region=str(offer_pid.region) if offer_pid is not None else "",
                            )
                            specs.append(
                                PrimePodSpecV1(
                                    cloud_id=str(offer_pid.cloud_id),
                                    gpu_type=str(offer_pid.gpu_type),
                                    gpu_count=int(args.gpu_count),
                                    socket=str(args.socket),
                                    image=str(img_pid),
                                    name=f"uvcc-3pc-mixed-{ts}-p{pid}",
                                    provider_type=str(prov),
                                    data_center_id=dc_id_pid,
                                    max_price=float(args.max_price) if args.max_price is not None else None,
                                )
                            )

                        # Create pods sequentially so partial failures can be cleaned up.
                        created_pod_ids = []
                        try:
                            t_prime0 = time.monotonic()
                            for spec in specs:
                                created_pod_ids.append(prime.create_pod(spec))
                            pods = [prime.wait_active(pid) for pid in created_pod_ids]

                            # SSH gate: verify we can actually connect before proceeding.
                            ssh_key_path = str(args.ssh_key_path).strip() if args.ssh_key_path is not None else ""
                            if not ssh_key_path:
                                for cand in ("~/.ssh/uvcc_prime_runner_ed25519", "~/.ssh/vracu_prime_intellect_ed25519", "~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
                                    p = Path(cand).expanduser().resolve()
                                    if p.exists():
                                        ssh_key_path = str(p)
                                        break
                            if not ssh_key_path:
                                raise RuntimeError("ssh key not found; pass --ssh-key-path")
                            pkey = load_private_key_from_file(str(ssh_key_path))
                            for pp in pods:
                                ssh = ssh_connect_with_retries(
                                    hostname=pp.ssh_host,
                                    port=pp.ssh_port,
                                    username=pp.ssh_user,
                                    pkey=pkey,
                                    timeout_s=60,
                                )
                                try:
                                    code, out, err = ssh_exec(ssh, "bash -lc 'echo ok'", timeout_s=30)
                                    if code != 0 or "ok" not in out:
                                        raise RuntimeError(f"ssh probe failed: {out} {err}")
                                finally:
                                    try:
                                        ssh.close()
                                    except Exception:
                                        pass
                            log.event(
                                "prime_pods_active",
                                t_s=round(time.monotonic() - t_prime0, 3),
                                pod_ids=list(created_pod_ids),
                                ssh_hosts=[str(p.ssh_host) for p in pods],
                                provider_types=[
                                    str((p.status_row or {}).get("providerType") or (p.status_row or {}).get("provider") or "")
                                    for p in pods
                                ],
                            )
                            break
                        except Exception as exc:
                            last_err = str(exc)
                            log.log(f"WARN: mixed provisioning attempt {idx} failed: {exc}")
                            for pid in list(created_pod_ids):
                                try:
                                    prime.delete_pod(pid)
                                    log.event("prime_pod_deleted", pod_id=str(pid))
                                except Exception:
                                    pass
                            pods = None
                            created_pod_ids = []
                            time.sleep(2.0)
                            continue

                    if offer is not None:
                        cloud_id, gpu_type = offer.cloud_id, offer.gpu_type
                        chosen_image = _pick_prime_image(requested=str(args.image), available_images=offer.images)
                        provider_type = str(offer.provider) if offer.provider is not None else (provider_filter or "runpod")
                    else:
                        # Manual override path: keep provided cloud_id/gpu_type and requested image/provider.
                        chosen_image = str(args.image)
                        provider_type = provider_filter or "runpod"

                    dc_id: Optional[str] = None
                    if offer is not None:
                        dc_id = (
                            str(
                                offer.raw.get("dataCenterId")
                                or offer.raw.get("data_center_id")
                                or offer.raw.get("dataCenterID")
                                or offer.raw.get("dataCenter")
                                or offer.raw.get("data_center")
                                or offer.raw.get("datacenter")
                                or offer.raw.get("datacenter_id")
                                or ""
                            ).strip()
                            or None
                        )

                    log.event(
                        "prime_offer_try",
                        attempt=int(idx),
                        cloud_id=str(cloud_id),
                        gpu_type=str(gpu_type),
                        provider_type=str(provider_type),
                        gpu_count=int(args.gpu_count),
                        socket=str(args.socket),
                        image=str(chosen_image),
                        offer_region=str(offer.region) if offer is not None else "",
                    )

                    specs = [
                        PrimePodSpecV1(
                            cloud_id=str(cloud_id),
                            gpu_type=str(gpu_type),
                            gpu_count=int(args.gpu_count),
                            socket=str(args.socket),
                            image=str(chosen_image),
                            name=f"uvcc-3pc-{ts}-p{pid}",
                            provider_type=str(provider_type),
                            data_center_id=dc_id,
                            max_price=float(args.max_price) if args.max_price is not None else None,
                        )
                        for pid in (0, 1, 2)
                    ]

                    # Create pods sequentially so partial failures can be cleaned up.
                    created_pod_ids = []
                    try:
                        t_prime0 = time.monotonic()
                        for spec in specs:
                            created_pod_ids.append(prime.create_pod(spec))
                        pods = [prime.wait_active(pid) for pid in created_pod_ids]
                        # SSH gate: some Prime offers report ACTIVE but do not accept SSH keys.
                        # Verify we can actually connect before proceeding, otherwise delete and retry.
                        ssh_key_path = str(args.ssh_key_path).strip() if args.ssh_key_path is not None else ""
                        if not ssh_key_path:
                            for cand in ("~/.ssh/uvcc_prime_runner_ed25519", "~/.ssh/vracu_prime_intellect_ed25519", "~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
                                p = Path(cand).expanduser().resolve()
                                if p.exists():
                                    ssh_key_path = str(p)
                                    break
                        if not ssh_key_path:
                            raise RuntimeError("ssh key not found; pass --ssh-key-path")
                        pkey = load_private_key_from_file(str(ssh_key_path))
                        for pp in pods:
                            # fast-fail connectivity check per node
                            ssh = ssh_connect_with_retries(
                                hostname=pp.ssh_host,
                                port=pp.ssh_port,
                                username=pp.ssh_user,
                                pkey=pkey,
                                timeout_s=60,
                            )
                            try:
                                code, out, err = ssh_exec(ssh, "bash -lc 'echo ok'", timeout_s=30)
                                if code != 0 or "ok" not in out:
                                    raise RuntimeError(f"ssh probe failed: {out} {err}")
                            finally:
                                try:
                                    ssh.close()
                                except Exception:
                                    pass
                        log.event(
                            "prime_pods_active",
                            t_s=round(time.monotonic() - t_prime0, 3),
                            pod_ids=list(created_pod_ids),
                            ssh_hosts=[str(p.ssh_host) for p in pods],
                            provider_types=[
                                str((p.status_row or {}).get("providerType") or (p.status_row or {}).get("provider") or "")
                                for p in pods
                            ],
                        )
                        break
                    except Exception as exc:
                        last_err = str(exc)
                        log.log(f"WARN: provisioning attempt {idx} failed: {exc}")
                        # Best-effort cleanup before trying the next offer.
                        for pid in list(created_pod_ids):
                            try:
                                prime.delete_pod(pid)
                                log.event("prime_pod_deleted", pod_id=str(pid))
                            except Exception:
                                pass
                        pods = None
                        created_pod_ids = []
                        time.sleep(2.0)
                        continue

                if pods is None:
                    raise RuntimeError(f"unable to provision Prime pods after retries: {last_err}")

            # SSH connect and bootstrap.
            ssh_key_path = str(args.ssh_key_path).strip() if args.ssh_key_path is not None else ""
            if not ssh_key_path:
                # Best-effort default: prefer a Prime-specific key if present, then common defaults.
                for cand in ("~/.ssh/uvcc_prime_runner_ed25519", "~/.ssh/vracu_prime_intellect_ed25519", "~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
                    p = Path(cand).expanduser().resolve()
                    if p.exists():
                        ssh_key_path = str(p)
                        break
            if not ssh_key_path:
                raise RuntimeError("ssh key not found; pass --ssh-key-path")
            log.event("ssh_key_selected", ssh_key_path=str(ssh_key_path))
            pkey = load_private_key_from_file(str(ssh_key_path))
            nodes: List[RemoteNodeV1] = []
            bundle_path = Path(td) / "uvcc_bundle.tgz"
            _build_uvcc_bundle_tgz(bundle_path)

            for pid, pod in zip((0, 1, 2), pods):
                t_node0 = time.monotonic()
                ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    code, out, err = ssh_exec(ssh, "bash -lc 'echo -n $HOME'", timeout_s=30)
                    if code != 0 or not out:
                        raise RuntimeError(f"failed to get remote home: {err}")
                    home = out.strip()
                    root_dir = f"{home}/uvcc"
                    venv_dir = f"{root_dir}/venv"

                    # Capture a GPU snapshot for efficiency reporting.
                    smi_cmd = "bash -lc 'nvidia-smi -L || true; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true'"
                    code_smi, out_smi, err_smi = ssh_exec(ssh, smi_cmd, timeout_s=60)
                    smi_local = out_dir / f"node_p{int(pid)}_nvidia_smi.txt"
                    smi_local.write_text(out_smi + err_smi, encoding="utf-8")
                    log.event("node_gpu_snapshot", party_id=int(pid), ssh_host=str(pod.ssh_host), path=str(smi_local), exit_code=int(code_smi))

                    # Upload bundle to /tmp and extract into $HOME/uvcc.
                    remote_bundle = f"{home}/uvcc_bundle.tgz"
                    sftp_put_file(ssh, local_path=str(bundle_path), remote_path=str(remote_bundle), mode=0o600)
                    t_boot0 = time.monotonic()
                    bootstrap_cmd = (
                        "set -euo pipefail;"
                        " if command -v apt-get >/dev/null 2>&1; then"
                        "   export DEBIAN_FRONTEND=noninteractive;"
                        "   SUDO=''; if sudo -n true >/dev/null 2>&1; then SUDO='sudo -n'; fi;"
                        "   $SUDO apt-get update -y;"
                        "   $SUDO apt-get install -y python3-venv python3-pip build-essential git;"
                        " fi;"
                        f" mkdir -p {shlex.quote(root_dir)};"
                        f" tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(root_dir)};"
                        f" if [ ! -x {shlex.quote(venv_dir)}/bin/python3 ]; then python3 -m venv --system-site-packages {shlex.quote(venv_dir)}; fi;"
                        f" {shlex.quote(venv_dir)}/bin/pip install -U pip setuptools wheel;"
                        f" {shlex.quote(venv_dir)}/bin/pip install -r {shlex.quote(root_dir)}/research/uvcc/requirements-uvcc-base.txt;"
                        f" {shlex.quote(venv_dir)}/bin/python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())';"
                    )
                    code2, out2, err2 = ssh_exec(ssh, f"bash -lc {shlex.quote(bootstrap_cmd)}", timeout_s=1800)
                    if code2 != 0:
                        raise RuntimeError(f"bootstrap failed:\n{out2}\n{err2}")
                    boot_local = out_dir / f"node_p{int(pid)}_bootstrap.log"
                    boot_local.write_text(out2 + err2, encoding="utf-8")
                    log.event(
                        "node_bootstrap_done",
                        party_id=int(pid),
                        ssh_host=str(pod.ssh_host),
                        t_s=round(time.monotonic() - t_boot0, 3),
                        log_path=str(boot_local),
                    )

                    # Optionally run CUDA conformance tests on each node (compiles extensions).
                    if str(args.run_gpu_tests).lower() == "true":
                        envp = f"PATH={venv_dir}/bin:$PATH PYTHONPATH={root_dir}/research/uvcc/uvcc-party:{root_dir}/research/uvcc/uvcc-client"
                        test_cmd = (
                            f"set -euo pipefail; {envp} {venv_dir}/bin/python3 -m pytest -q "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_dpf_dcf_kernels.py "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_gf2_and_a2b_kernels.py "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_trunc_apply_u64.py "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_matmul_u64.py "
                        )
                        t_gpu0 = time.monotonic()
                        code3, out3, err3 = ssh_exec(ssh, f"bash -lc {shlex.quote(test_cmd)}", timeout_s=3600)
                        gpu_local = out_dir / f"node_p{int(pid)}_gpu_tests.log"
                        gpu_local.write_text(out3 + err3, encoding="utf-8")
                        if code3 != 0:
                            raise RuntimeError(f"gpu tests failed:\n{out3}\n{err3}")
                        log.event(
                            "node_gpu_tests_done",
                            party_id=int(pid),
                            ssh_host=str(pod.ssh_host),
                            t_s=round(time.monotonic() - t_gpu0, 3),
                            log_path=str(gpu_local),
                        )
                finally:
                    ssh.close()
                log.event("node_ready", party_id=int(pid), ssh_host=str(pod.ssh_host), t_total_s=round(time.monotonic() - t_node0, 3))

                nodes.append(
                    RemoteNodeV1(
                        party_id=int(pid),
                        pod_id=str(pod.pod_id),
                        provider_type=str((pod.status_row or {}).get("providerType") or (pod.status_row or {}).get("provider") or ""),
                        ssh_user=str(pod.ssh_user),
                        ssh_host=str(pod.ssh_host),
                        ssh_port=int(pod.ssh_port),
                        home=str(home),
                        root_dir=str(root_dir),
                        venv_dir=str(venv_dir),
                    )
                )

            # Stable per-party identities (so we can replace pods without changing on-chain party addresses).
            #
            # We generate (or reuse from disk) one secp256k1 privkey per party, store it under out_dir/private_keep,
            # and upload the corresponding key file onto each node. All `party-info` / `party-sign` commands use this
            # key file, so replacing a pod does NOT change the party address.
            party_keys_dir = private_keep / "party_keys"
            party_keys_dir.mkdir(parents=True, exist_ok=True)
            party_priv_by_pid: Dict[int, bytes] = {}
            for pid in (0, 1, 2):
                kp = party_keys_dir / f"party_privkey_p{int(pid)}.hex"
                priv = load_or_create_party_privkey32_v1(path=str(kp))
                party_priv_by_pid[int(pid)] = bytes(priv)
            # Log ONLY public identities (never log privkeys).
            party_identities_local = []
            for pid in (0, 1, 2):
                ident = party_identity_from_privkey_v1(party_id=int(pid), privkey32=party_priv_by_pid[int(pid)])
                party_identities_local.append({"party_id": int(pid), "address": "0x" + bytes(ident.address20).hex()})
            log.event("party_identity_keys_ready", parties=party_identities_local, key_dir=str(party_keys_dir))

            # Upload party key file to each node at a known path (used by party-info/party-sign).
            for n in nodes:
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    remote_key = f"{n.root_dir}/party_privkey.hex"
                    priv = party_priv_by_pid[int(n.party_id)]
                    sftp_put_bytes(ssh, remote_path=str(remote_key), data=("0x" + priv.hex()).encode("utf-8"), mode=0o600)
                finally:
                    ssh.close()

            # Start local append-only live recorder (best-effort) to preserve logs/telemetry/checkpoints even if pods die.
            if str(args.start_live_recorder).lower() == "true":
                try:
                    rec_py = _repo_root() / "research" / "uvcc" / "uvcc-demo" / "record_live_logs_append.py"
                    rec_out = out_dir / "live_keep" / "recorder_stdout.log"
                    rec_out.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        str(rec_py),
                        "--out",
                        str(out_dir),
                        "--ssh-key-path",
                        str(ssh_key_path),
                        "--interval-s",
                        str(max(1.0, float(args.live_recorder_interval_s))),
                    ]
                    for nn in nodes:
                        cmd += ["--node", f"{int(nn.party_id)},{nn.ssh_host},{int(nn.ssh_port)},{nn.ssh_user}"]
                    with open(rec_out, "a", encoding="utf-8") as f:
                        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                    live_recorder_pid = int(p.pid)
                    (out_dir / "live_keep" / "recorder.pid").write_text(str(live_recorder_pid) + "\n", encoding="utf-8")
                    log.event("live_recorder_started", pid=int(live_recorder_pid), interval_s=float(args.live_recorder_interval_s))
                except Exception as exc:
                    log.event("live_recorder_start_failed", error=str(exc))

            # Create relay TLS assets for node0, upload CA everywhere, start relay on node0.
            relay_token = base64.b64encode(os.urandom(24)).decode("ascii")
            relay_host = nodes[0].ssh_host
            # Prime hosted pods (e.g. runpod) often expose only a small set of port mappings.
            # Choose a relay bind port that is actually mapped, and advertise the mapped external port.
            relay_requested_internal = int(args.relay_port)
            relay_candidates: List[Tuple[int, int, Optional[Dict[str, Any]]]] = [(int(relay_requested_internal), int(relay_requested_internal), None)]
            try:
                pod0 = prime.pod_get(str(nodes[0].pod_id))
                pm = pod0.get("primePortMapping")
                pm_list = pm if isinstance(pm, list) else []
                by_internal: Dict[int, Tuple[int, Dict[str, Any]]] = {}
                for m in pm_list:
                    if not isinstance(m, dict):
                        continue
                    try:
                        i = int(str(m.get("internal", "")).strip())
                        e = int(str(m.get("external", "")).strip())
                    except Exception:
                        continue
                    by_internal[int(i)] = (int(e), dict(m))

                if by_internal:
                    # Some providers (e.g. datacrunch) report only the SSH mapping here; we must never
                    # attempt to run the relay on the SSH port. If we can't find a usable extra mapping,
                    # we fall back to trying common ports with internal==external and rely on the public
                    # health check to validate reachability.
                    ssh_port0 = int(nodes[0].ssh_port)
                    is_root0 = str(nodes[0].ssh_user) == "root"

                    def _mapping_usable(m: Dict[str, Any]) -> bool:
                        used = str(m.get("usedBy") or "").strip().upper()
                        desc = str(m.get("description") or "").strip().lower()
                        if used in ("SSH", "JUPYTER_NOTEBOOK"):
                            return False
                        if "ssh" in desc or "jupyter" in desc:
                            return False
                        return True

                    def _port_ok(pi: int) -> bool:
                        if int(pi) in (22, 8888, ssh_port0):
                            return False
                        if int(pi) < 1024 and not is_root0:
                            return False
                        return True

                    pref_ports: List[int] = []
                    for p in (relay_requested_internal, 3000, 8000, 54692, 19123):
                        pi = int(p)
                        if pi not in pref_ports:
                            pref_ports.append(pi)

                    cand: List[Tuple[int, int, Optional[Dict[str, Any]]]] = []
                    seen: set[int] = set()

                    def add(pi: int, pe: int, mm: Optional[Dict[str, Any]]) -> None:
                        if int(pi) in seen:
                            return
                        if not _port_ok(int(pi)):
                            return
                        cand.append((int(pi), int(pe), dict(mm) if isinstance(mm, dict) else None))
                        seen.add(int(pi))

                    # Try preferred ports first; if a usable mapping exists, use its external port, else
                    # fall back to internal==external (some providers expose public IPs without explicit mappings).
                    for pi in pref_ports:
                        if int(pi) in by_internal:
                            e, m = by_internal[int(pi)]
                            if _mapping_usable(dict(m)):
                                add(int(pi), int(e), dict(m))
                                continue
                        add(int(pi), int(pi), None)

                    # Add any other usable mappings after preferences (useful on hosted providers like runpod).
                    for i, (e, m) in by_internal.items():
                        if not _mapping_usable(dict(m)):
                            continue
                        add(int(i), int(e), dict(m))

                    if cand:
                        relay_candidates = cand
            except Exception:
                # Non-Prime environment or API change: fall back to internal==external.
                relay_candidates = [(int(relay_requested_internal), int(relay_requested_internal), None)]

            relay_port_internal = int(relay_requested_internal)
            relay_port_external = int(relay_requested_internal)
            relay_port_mapping: Optional[Dict[str, Any]] = None
            relay_url_public = ""
            relay_url_local = ""
            ca_pem, srv_pem, srv_key = _tls_ca_and_server_cert(host_or_ip=str(relay_host))

            ca_remote = f"{nodes[0].root_dir}/relay_ca.pem"
            srv_remote = f"{nodes[0].root_dir}/relay_srv.pem"
            key_remote = f"{nodes[0].root_dir}/relay_srv.key"
            token_remote0 = f"{nodes[0].root_dir}/relay_token.txt"

            for n in nodes:
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    if int(n.party_id) == 0:
                        sftp_put_bytes(ssh, remote_path=ca_remote, data=ca_pem, mode=0o644)
                        sftp_put_bytes(ssh, remote_path=srv_remote, data=srv_pem, mode=0o644)
                        sftp_put_bytes(ssh, remote_path=key_remote, data=srv_key, mode=0o600)
                        sftp_put_bytes(ssh, remote_path=token_remote0, data=relay_token.encode("utf-8"), mode=0o600)
                    else:
                        ca_path = f"{n.root_dir}/relay_ca.pem"
                        sftp_put_bytes(ssh, remote_path=ca_path, data=ca_pem, mode=0o644)
                        token_path = f"{n.root_dir}/relay_token.txt"
                        sftp_put_bytes(ssh, remote_path=token_path, data=relay_token.encode("utf-8"), mode=0o600)
                finally:
                    ssh.close()

            group_id = "g-" + job_id32.hex()[:16]

            # Wait for relay to become reachable (from node0 via loopback and from node1 via public mapping).
            def _remote_healthz(*, n: RemoteNodeV1, url: str, ca_path: str) -> None:
                py = (
                    "import ssl,urllib.request;"
                    f"ctx=ssl.create_default_context(cafile={ca_path!r});"
                    f"r=urllib.request.urlopen({(url + '/healthz')!r},context=ctx,timeout=3);"
                    "print(r.status);"
                )
                cmd = f"{n.env_prefix()} {shlex.quote(n.venv_python())} -c {shlex.quote(py)}"
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=120)
                try:
                    code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=15)
                    if code != 0 or "200" not in str(out):
                        raise RuntimeError(f"healthz failed ({n.party_id}) for {url}:\n{out}\n{err}")
                finally:
                    ssh.close()

            # Start relay (TLS + token) on node0 and validate reachability.
            relay_py = f"{nodes[0].root_dir}/research/uvcc/uvcc-relay/relay_server.py"
            relay_db = f"{nodes[0].root_dir}/relay.sqlite"
            relay_log = f"{nodes[0].root_dir}/relay.log"
            relay_pid = f"{nodes[0].root_dir}/relay.pid"

            last_relay_err = ""
            for cand_internal, cand_external, cand_mapping in list(relay_candidates):
                relay_port_internal = int(cand_internal)
                relay_port_external = int(cand_external)
                relay_port_mapping = dict(cand_mapping) if isinstance(cand_mapping, dict) else None
                relay_url_public = f"https://{relay_host}:{int(relay_port_external)}"
                relay_url_local = f"https://127.0.0.1:{int(relay_port_internal)}"
                log.event(
                    "relay_candidate_try",
                    relay_port_internal=int(relay_port_internal),
                    relay_port_external=int(relay_port_external),
                    relay_port_mapping=dict(relay_port_mapping) if relay_port_mapping is not None else None,
                    relay_url_public=str(relay_url_public),
                    relay_url_local=str(relay_url_local),
                )

                ssh0 = ssh_connect_with_retries(hostname=nodes[0].ssh_host, port=nodes[0].ssh_port, username=nodes[0].ssh_user, pkey=pkey, timeout_s=900)
                try:
                    start_cmd = (
                        "set -euo pipefail;"
                        f" rm -f {shlex.quote(relay_log)};"
                        # Stop any existing relay from a previous attempt on the same pod.
                        f" if [ -f {shlex.quote(relay_pid)} ]; then"
                        f"   (kill -9 $(cat {shlex.quote(relay_pid)}) >/dev/null 2>&1 || true);"
                        f"   rm -f {shlex.quote(relay_pid)};"
                        " fi;"
                        f" PYTHONUNBUFFERED=1 nohup {shlex.quote(nodes[0].venv_python())} {shlex.quote(relay_py)}"
                        f" --host 0.0.0.0 --port {int(relay_port_internal)} --db {shlex.quote(relay_db)}"
                        " --require-token true"
                        f" --token-file {shlex.quote(token_remote0)}"
                        f" --tls-cert {shlex.quote(srv_remote)} --tls-key {shlex.quote(key_remote)}"
                        f" > {shlex.quote(relay_log)} 2>&1 & echo $! > {shlex.quote(relay_pid)};"
                        " sleep 0.2;"
                        f" PID=$(cat {shlex.quote(relay_pid)});"
                        " kill -0 $PID >/dev/null 2>&1 || (echo 'relay_process_exited_early' >&2; tail -n 200 "
                        f"{shlex.quote(relay_log)} >&2 || true; exit 1);"
                    )
                    code, out, err = ssh_exec(ssh0, f"bash -lc {shlex.quote(start_cmd)}", timeout_s=30)
                    if code != 0:
                        last_relay_err = f"relay start failed (port={relay_port_internal}):\n{out}\n{err}"
                        log.event(
                            "relay_candidate_failed",
                            stage="start",
                            relay_port_internal=int(relay_port_internal),
                            relay_port_external=int(relay_port_external),
                            error=str(last_relay_err),
                        )
                        continue
                finally:
                    ssh0.close()

                last0 = ""
                for _ in range(60):
                    try:
                        _remote_healthz(n=nodes[0], url=str(relay_url_local), ca_path=str(ca_remote))
                        last0 = ""
                        break
                    except Exception as exc:
                        last0 = str(exc)
                        time.sleep(0.5)
                if last0:
                    last_relay_err = f"relay loopback health failed (port={relay_port_internal}): {last0}"
                    log.event(
                        "relay_candidate_failed",
                        stage="loopback_health",
                        relay_port_internal=int(relay_port_internal),
                        relay_port_external=int(relay_port_external),
                        error=str(last0),
                    )
                    continue

                last1 = ""
                for _ in range(60):
                    try:
                        _remote_healthz(n=nodes[1], url=str(relay_url_public), ca_path=str(f"{nodes[1].root_dir}/relay_ca.pem"))
                        last1 = ""
                        break
                    except Exception as exc:
                        last1 = str(exc)
                        time.sleep(0.5)
                if last1:
                    last_relay_err = f"relay public health failed (port={relay_port_external}): {last1}"
                    log.event(
                        "relay_candidate_failed",
                        stage="public_health",
                        relay_port_internal=int(relay_port_internal),
                        relay_port_external=int(relay_port_external),
                        error=str(last1),
                    )
                    continue

                last_relay_err = ""
                break
            else:
                raise RuntimeError(f"relay did not become healthy (last_error={last_relay_err})")

            log.event(
                "privacy_relay_started",
                relay_url_public=str(relay_url_public),
                relay_url_local=str(relay_url_local),
                relay_host=str(relay_host),
                relay_port_internal=int(relay_port_internal),
                relay_port_external=int(relay_port_external),
                relay_port_mapping=dict(relay_port_mapping) if relay_port_mapping is not None else None,
                tls_ca_sha256=str(_sha256_hex(ca_pem)),
                token_required=True,
                token_file=str(token_remote0),
            )

            # Party identities (pubkeys/addresses) from each node.
            for n in nodes:
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    key_path = f"{n.root_dir}/party_privkey.hex"
                    cmd = (
                        f"{n.env_prefix()} {shlex.quote(n.venv_python())} -m uvcc_client party-info"
                        f" --party-id {int(n.party_id)} --key-path {shlex.quote(str(key_path))}"
                    )
                    code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=60)
                    if code != 0:
                        raise RuntimeError(f"party-info failed:\n{out}\n{err}")
                    info = json.loads(out.strip())
                    n.address = str(info["address"])
                    n.pubkey64_hex = str(info["pubkey64_hex"])
                finally:
                    ssh.close()
            log.event(
                "party_identities",
                parties=[
                    {"party_id": int(n.party_id), "address": str(n.address), "ssh_host": str(n.ssh_host)} for n in nodes
                ],
            )

            # Policy commit digest and signatures from the three nodes.
            pc = PolicyCommitV1(
                job_id32=job_id32,
                policy_hash32=policy_hash32,
                sid_hash32=sid_hash32,
                sgir_hash32=sgir_hash32,
                runtime_hash32=runtime_hash32,
                fss_dir_hash32=fss_dir_hash32,
                preproc_hash32=preproc_hash32,
                backend_u8=0,
                epoch_u64=0,
            )
            digest_pc = pc.digest32(domain=dom)
            sig_pc_hex: Dict[int, str] = {}
            for n in nodes:
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    key_path = f"{n.root_dir}/party_privkey.hex"
                    cmd = (
                        f"{n.env_prefix()} {shlex.quote(n.venv_python())} -m uvcc_client party-sign"
                        f" --party-id {int(n.party_id)} --key-path {shlex.quote(str(key_path))} --digest-hex 0x{digest_pc.hex()}"
                    )
                    code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=60)
                    if code != 0:
                        raise RuntimeError(f"party-sign policy failed:\n{out}\n{err}")
                    info = json.loads(out.strip())
                    sig_pc_hex[int(n.party_id)] = str(info["sig65_hex"])
                finally:
                    ssh.close()
            log.event(
                "policy_commit_signed",
                digest_pc_hex="0x" + digest_pc.hex(),
                sig_p0=str(sig_pc_hex.get(0, "")),
                sig_p1=str(sig_pc_hex.get(1, "")),
                sig_p2=str(sig_pc_hex.get(2, "")),
            )

            # Create job on-chain.
            pc_tuple = (
                f"(0x{job_id32.hex()},0x{policy_hash32.hex()},0x{sid_hash32.hex()},0x{sgir_hash32.hex()},0x{runtime_hash32.hex()},"
                f"0x{fss_dir_hash32.hex()},0x{preproc_hash32.hex()},0,0)"
            )
            parties_arr = f"[{nodes[0].address},{nodes[1].address},{nodes[2].address}]"
            create_out = _cast_send(
                rpc_url=rpc_url,
                privkey_hex=priv0,
                to=ledger,
                sig="createJob((bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8,uint64),address[3],bytes,bytes,bytes)",
                args=[pc_tuple, parties_arr, sig_pc_hex[0], sig_pc_hex[1], sig_pc_hex[2]],
            )
            create_log = out_dir / "onchain_createJob.log"
            create_log.write_text(str(create_out), encoding="utf-8")
            log.event("onchain_createJob", log_path=str(create_log))

            # Prepare per-party inputs (secret shared) + per-party TCF keys.
            import torch

            d = int(d_job)
            gen = torch.Generator(device="cpu").manual_seed(int(seed_job))
            X_pub = torch.eye(d, dtype=torch.int64)
            loY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            hiY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            Y_pub = (hiY << 32) | loY
            loW = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            hiW = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen)
            W_pub = (hiW << 32) | loW

            X0, X1, X2 = make_rss_arith_u64_triple(x_pub=X_pub, generator=gen, device=torch.device("cpu"))
            Y0, Y1, Y2 = make_rss_arith_u64_triple(x_pub=Y_pub, generator=gen, device=torch.device("cpu"))
            W0, W1, W2 = make_rss_arith_u64_triple(x_pub=W_pub, generator=gen, device=torch.device("cpu"))

            master_seed32 = os.urandom(32)
            k0, k1, k2 = tcf_gen_v1(master_seed32=master_seed32, sid=sid)
            tcf_keys = {0: k0, 1: k1, 2: k2}

            def enc_u64(t: torch.Tensor) -> str:
                out = bytearray()
                for v in t.contiguous().view(-1).tolist():
                    out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
                return base64.b64encode(bytes(out)).decode("ascii")

            shares = {
                0: (X0, Y0, W0),
                1: (X1, Y1, W1),
                2: (X2, Y2, W2),
            }
            # Failover design:
            # - Each attempt runs as a distinct transcript epoch (epoch=0,1,2,...). This avoids any transcript key collisions.
            # - Parties write private checkpoint shares each step (if enabled). On failure, we resume from the last common checkpoint.
            # - Stable per-party signing keys ensure the on-chain party addresses remain unchanged across pod replacement/restarts.

            def _read_checkpoint_file(pid: int) -> List[Dict[str, Any]]:
                p = private_keep / f"checkpoints_W_p{int(pid)}.jsonl"
                if not p.exists():
                    return []
                out: List[Dict[str, Any]] = []
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
                return out

            def _last_ckpt_step(pid: int) -> int:
                recs = _read_checkpoint_file(int(pid))
                steps = []
                for r in recs:
                    try:
                        steps.append(int(r.get("step")))
                    except Exception:
                        continue
                return max(steps) if steps else -1

            def _ckpt_w_b64_for_step(pid: int, *, step: int) -> Optional[Tuple[str, str]]:
                recs = _read_checkpoint_file(int(pid))
                hit: Optional[Tuple[str, str]] = None
                for r in recs:
                    try:
                        if int(r.get("step")) != int(step):
                            continue
                        wlo = str(r.get("W_lo_b64") or "")
                        whi = str(r.get("W_hi_b64") or "")
                        if wlo and whi:
                            hit = (wlo, whi)
                    except Exception:
                        continue
                return hit

            # Precompute base (X/Y/TCF) inputs per party and record them privately for recovery.
            base_inputs_by_pid: Dict[int, Dict[str, Any]] = {}
            init_w_b64_by_pid: Dict[int, Tuple[str, str]] = {}
            for pid in (0, 1, 2):
                Xs, Ys, Ws = shares[int(pid)]
                tk = tcf_keys[int(pid)]
                base_inputs_by_pid[int(pid)] = {
                    "d": int(d),
                    "fxp_frac_bits": int(fxp_job),
                    "tcf_key": {
                        "sid_hash32_hex": "0x" + bytes(tk.sid_hash32).hex(),
                        "s01_hex": "0x" + bytes(tk.s01).hex(),
                        "s02_hex": "0x" + bytes(tk.s02).hex(),
                        "s12_hex": "0x" + bytes(tk.s12).hex(),
                    },
                    "X_lo_b64": enc_u64(Xs.lo),
                    "X_hi_b64": enc_u64(Xs.hi),
                    "Y_lo_b64": enc_u64(Ys.lo),
                    "Y_hi_b64": enc_u64(Ys.hi),
                }
                init_w_b64_by_pid[int(pid)] = (enc_u64(Ws.lo), enc_u64(Ws.hi))
                # Save initial private inputs snapshot locally (do not upload this to logs).
                (private_keep / f"inputs_initial_p{int(pid)}.json").write_text(
                    json.dumps({**base_inputs_by_pid[int(pid)], "W_lo_b64": init_w_b64_by_pid[int(pid)][0], "W_hi_b64": init_w_b64_by_pid[int(pid)][1]}, sort_keys=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )

            def _upload_inputs_for_epoch(*, start_step: int) -> None:
                # Determine which W shares to use at this epoch start.
                if int(start_step) <= 0:
                    w_for = dict(init_w_b64_by_pid)
                else:
                    ck_step = int(start_step) - 1
                    w_for = {}
                    for pid in (0, 1, 2):
                        hit = _ckpt_w_b64_for_step(int(pid), step=ck_step)
                        if hit is None:
                            raise RuntimeError(f"missing checkpoint shares for party {pid} at step={ck_step} (cannot resume safely)")
                        w_for[int(pid)] = hit
                for n in nodes:
                    pid = int(n.party_id)
                    wlo, whi = w_for[pid]
                    inp = dict(base_inputs_by_pid[pid])
                    inp["W_lo_b64"] = str(wlo)
                    inp["W_hi_b64"] = str(whi)
                    inp_bytes = (json.dumps(inp, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                    inp_hash32_hex = "0x" + hashlib.sha256(b"uvcc.inputs.v1\0" + inp_bytes).hexdigest()
                    ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                    try:
                        remote_inp = f"{n.root_dir}/inputs_p{int(n.party_id)}.json"
                        sftp_put_bytes(ssh, remote_path=remote_inp, data=inp_bytes, mode=0o600)
                    finally:
                        ssh.close()
                    log.event(
                        "party_inputs_uploaded",
                        party_id=int(n.party_id),
                        ssh_host=str(n.ssh_host),
                        remote_path=str(remote_inp),
                        bytes=int(len(inp_bytes)),
                        inputs_hash32_hex=str(inp_hash32_hex),
                        start_step=int(start_step),
                    )

            def _bootstrap_pod_as_node(*, party_id: int, pod: Any, epoch: int) -> RemoteNodeV1:
                """
                Bootstrap a newly provisioned pod into a RemoteNodeV1 (uvcc bundle + venv + deps + optional gpu tests).
                """
                pid = int(party_id)
                ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    code, out, err = ssh_exec(ssh, "bash -lc 'echo -n $HOME'", timeout_s=30)
                    if code != 0 or not out:
                        raise RuntimeError(f"failed to get remote home: {err}")
                    home = out.strip()
                    root_dir = f"{home}/uvcc"
                    venv_dir = f"{root_dir}/venv"

                    # Capture a GPU snapshot for evidence (suffix per epoch to avoid clobbering).
                    smi_cmd = "bash -lc 'nvidia-smi -L || true; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true'"
                    code_smi, out_smi, err_smi = ssh_exec(ssh, smi_cmd, timeout_s=60)
                    smi_local = out_dir / f"node_p{int(pid)}_nvidia_smi_epoch{int(epoch)}.txt"
                    smi_local.write_text(out_smi + err_smi, encoding="utf-8")
                    log.event("node_gpu_snapshot", party_id=int(pid), ssh_host=str(pod.ssh_host), path=str(smi_local), exit_code=int(code_smi))

                    remote_bundle = f"{home}/uvcc_bundle.tgz"
                    sftp_put_file(ssh, local_path=str(bundle_path), remote_path=str(remote_bundle), mode=0o600)
                    t_boot0 = time.monotonic()
                    bootstrap_cmd = (
                        "set -euo pipefail;"
                        " if command -v apt-get >/dev/null 2>&1; then"
                        "   export DEBIAN_FRONTEND=noninteractive;"
                        "   SUDO=''; if sudo -n true >/dev/null 2>&1; then SUDO='sudo -n'; fi;"
                        "   $SUDO apt-get update -y;"
                        "   $SUDO apt-get install -y python3-venv python3-pip build-essential git;"
                        " fi;"
                        f" mkdir -p {shlex.quote(root_dir)};"
                        f" tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(root_dir)};"
                        f" if [ ! -x {shlex.quote(venv_dir)}/bin/python3 ]; then python3 -m venv --system-site-packages {shlex.quote(venv_dir)}; fi;"
                        f" {shlex.quote(venv_dir)}/bin/pip install -U pip setuptools wheel;"
                        f" {shlex.quote(venv_dir)}/bin/pip install -r {shlex.quote(root_dir)}/research/uvcc/requirements-uvcc-base.txt;"
                        f" {shlex.quote(venv_dir)}/bin/python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())';"
                    )
                    code2, out2, err2 = ssh_exec(ssh, f"bash -lc {shlex.quote(bootstrap_cmd)}", timeout_s=1800)
                    if code2 != 0:
                        raise RuntimeError(f"bootstrap failed:\n{out2}\n{err2}")
                    boot_local = out_dir / f"node_p{int(pid)}_bootstrap_epoch{int(epoch)}.log"
                    boot_local.write_text(out2 + err2, encoding="utf-8")
                    log.event(
                        "node_bootstrap_done",
                        party_id=int(pid),
                        ssh_host=str(pod.ssh_host),
                        t_s=round(time.monotonic() - t_boot0, 3),
                        log_path=str(boot_local),
                        epoch=int(epoch),
                    )

                    if str(args.run_gpu_tests).lower() == "true":
                        envp = f"PATH={venv_dir}/bin:$PATH PYTHONPATH={root_dir}/research/uvcc/uvcc-party:{root_dir}/research/uvcc/uvcc-client"
                        test_cmd = (
                            f"set -euo pipefail; {envp} {venv_dir}/bin/python3 -m pytest -q "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_dpf_dcf_kernels.py "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_gf2_and_a2b_kernels.py "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_trunc_apply_u64.py "
                            f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_matmul_u64.py "
                        )
                        t_gpu0 = time.monotonic()
                        code3, out3, err3 = ssh_exec(ssh, f"bash -lc {shlex.quote(test_cmd)}", timeout_s=3600)
                        gpu_local = out_dir / f"node_p{int(pid)}_gpu_tests_epoch{int(epoch)}.log"
                        gpu_local.write_text(out3 + err3, encoding="utf-8")
                        if code3 != 0:
                            raise RuntimeError(f"gpu tests failed:\n{out3}\n{err3}")
                        log.event(
                            "node_gpu_tests_done",
                            party_id=int(pid),
                            ssh_host=str(pod.ssh_host),
                            t_s=round(time.monotonic() - t_gpu0, 3),
                            log_path=str(gpu_local),
                            epoch=int(epoch),
                        )

                    return RemoteNodeV1(
                        party_id=int(pid),
                        pod_id=str(pod.pod_id),
                        provider_type=str((pod.status_row or {}).get("providerType") or (pod.status_row or {}).get("provider") or ""),
                        ssh_user=str(pod.ssh_user),
                        ssh_host=str(pod.ssh_host),
                        ssh_port=int(pod.ssh_port),
                        home=str(home),
                        root_dir=str(root_dir),
                        venv_dir=str(venv_dir),
                    )
                finally:
                    ssh.close()

            def _replace_party_pod(*, party_id: int, epoch: int) -> RemoteNodeV1:
                """
                Replace a single party pod (best-effort) and return the new RemoteNodeV1.
                NOTE: For simplicity we only auto-replace party_id != 0 (relay host). If party0 dies, the run aborts.
                """
                pid = int(party_id)
                if pid == 0:
                    raise RuntimeError("party0 (relay host) replacement is not supported in this failover mode")
                old = nodes[pid]
                try:
                    prime.delete_pod(str(old.pod_id))
                    log.event("prime_pod_deleted", pod_id=str(old.pod_id), party_id=int(pid), reason="failover_replace")
                except Exception as exc:
                    log.log(f"WARN: failed to delete pod {old.pod_id} during failover: {exc}")

                # Prefer replacing with the same provider as the failed pod (preserves mixed-provider semantics).
                preferred_provider = str(old.provider_type or "").strip()
                provider_filter: Optional[str] = preferred_provider
                if not provider_filter and providers_by_pid is not None:
                    provider_filter = str(providers_by_pid.get(int(pid)) or "").strip() or None
                if not provider_filter:
                    provider_arg = str(args.provider_type or "").strip()
                    provider_filter = None if provider_arg.lower() in ("", "auto") else provider_arg

                def _pick_failover_offer(provider_type_opt: Optional[str]) -> Any:
                    offers = prime.candidate_offers_v1(
                        nodes=1,
                        gpu_count_per_node=int(args.gpu_count),
                        provider_type=provider_type_opt,
                        socket=str(args.socket),
                        prefer_gpu_types=gpu_prefs,
                        prefer_regions=region_prefs,
                        limit=64,
                    )
                    if provider_type_opt is not None:
                        want = str(provider_type_opt).strip().lower()
                        offers = [o for o in offers if (o.provider is not None and str(o.provider).strip().lower() == want)]
                        if not offers:
                            raise RuntimeError(f"no Prime availability offers for requested provider_type={provider_type_opt} (failover)")
                    if bool(require_cuda_job):
                        offers = [
                            o
                            for o in offers
                            if (
                                not str(o.gpu_type or "").strip().upper().startswith("CPU")
                                and not str(o.gpu_type or "").strip().upper().endswith("CPU_NODE")
                                and not str(o.cloud_id or "").strip().lower().startswith("cpu-")
                            )
                        ]
                        if not offers:
                            raise RuntimeError(
                                f"no GPU offers available for require_cuda=true (provider_type={provider_type_opt or 'auto'} failover)"
                            )
                    return offers[0]

                offer: Any
                try:
                    offer = _pick_failover_offer(provider_filter)
                except Exception as exc:
                    # Fallback: try to avoid providers currently used by other parties, then fall back to any provider.
                    log.event("failover_provider_pick_failed", party_id=int(pid), provider_type=str(provider_filter or ""), error=str(exc))
                    used = {str(n.provider_type or "").strip().lower() for n in nodes if int(n.party_id) != int(pid)}
                    try:
                        cand = _pick_failover_offer(None)
                        # If we can find a non-used provider among top candidates, prefer it.
                        cand_list = prime.candidate_offers_v1(
                            nodes=1,
                            gpu_count_per_node=int(args.gpu_count),
                            provider_type=None,
                            socket=str(args.socket),
                            prefer_gpu_types=gpu_prefs,
                            prefer_regions=region_prefs,
                            limit=128,
                        )
                        cand_list2 = [
                            o
                            for o in cand_list
                            if str(o.provider or "").strip().lower()
                            and str(o.provider or "").strip().lower() not in used
                        ]
                        if bool(require_cuda_job):
                            cand_list2 = [
                                o
                                for o in cand_list2
                                if (
                                    not str(o.gpu_type or "").strip().upper().startswith("CPU")
                                    and not str(o.gpu_type or "").strip().upper().endswith("CPU_NODE")
                                    and not str(o.cloud_id or "").strip().lower().startswith("cpu-")
                                )
                            ]
                        offer = cand_list2[0] if cand_list2 else cand
                    except Exception as exc2:
                        raise RuntimeError(f"failover could not pick a replacement offer: {exc2}") from exc2

                cloud_id_new, gpu_type_new = offer.cloud_id, offer.gpu_type
                image_new = _pick_prime_image(requested=str(args.image), available_images=offer.images)
                provider_new = str(offer.provider) if offer.provider is not None else (str(provider_filter) if provider_filter is not None else "runpod")
                dc_id_new = (
                    str(
                        offer.raw.get("dataCenterId")
                        or offer.raw.get("data_center_id")
                        or offer.raw.get("dataCenterID")
                        or offer.raw.get("dataCenter")
                        or offer.raw.get("data_center")
                        or offer.raw.get("datacenter")
                        or offer.raw.get("datacenter_id")
                        or ""
                    ).strip()
                    or None
                )
                ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                spec = PrimePodSpecV1(
                    cloud_id=str(cloud_id_new),
                    gpu_type=str(gpu_type_new),
                    gpu_count=int(args.gpu_count),
                    socket=str(args.socket),
                    image=str(image_new),
                    name=f"uvcc-3pc-failover-e{int(epoch)}-p{int(pid)}-{ts}",
                    provider_type=str(provider_new),
                    data_center_id=dc_id_new,
                    max_price=float(args.max_price) if args.max_price is not None else None,
                )
                new_pod_id = prime.create_pod(spec)
                created_pod_ids.append(str(new_pod_id))
                new_pod = prime.wait_active(str(new_pod_id), timeout_s=1800)

                # SSH gate
                ssh = ssh_connect_with_retries(hostname=new_pod.ssh_host, port=new_pod.ssh_port, username=new_pod.ssh_user, pkey=pkey, timeout_s=120)
                try:
                    ssh_exec(ssh, "bash -lc 'echo ok'", timeout_s=30)
                finally:
                    ssh.close()

                new_node = _bootstrap_pod_as_node(party_id=int(pid), pod=new_pod, epoch=int(epoch))

                # Re-upload party signing key and relay CA/token to the new node.
                ssh = ssh_connect_with_retries(hostname=new_node.ssh_host, port=new_node.ssh_port, username=new_node.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    sftp_put_bytes(
                        ssh,
                        remote_path=f"{new_node.root_dir}/party_privkey.hex",
                        data=("0x" + party_priv_by_pid[int(pid)].hex()).encode("utf-8"),
                        mode=0o600,
                    )
                    sftp_put_bytes(ssh, remote_path=f"{new_node.root_dir}/relay_ca.pem", data=ca_pem, mode=0o644)
                    sftp_put_bytes(ssh, remote_path=f"{new_node.root_dir}/relay_token.txt", data=relay_token.encode("utf-8"), mode=0o600)
                finally:
                    ssh.close()

                # Keep party identity stable across failover (address/pubkey derived from the stable privkey),
                # and ensure proof bundle generation later has pubkey64 available even if we never re-run party-info.
                try:
                    ident = party_identity_from_privkey_v1(party_id=int(pid), privkey32=party_priv_by_pid[int(pid)])
                    new_node.address = "0x" + bytes(ident.address20).hex()
                    new_node.pubkey64_hex = "0x" + bytes(ident.pubkey64).hex()
                except Exception:
                    # Best-effort: if derivation fails, we will fall back to any later party-info call (or error out).
                    pass

                nodes[int(pid)] = new_node
                log.event(
                    "failover_pod_replaced",
                    party_id=int(pid),
                    old_pod_id=str(old.pod_id),
                    new_pod_id=str(new_pod_id),
                    new_ssh_host=str(new_node.ssh_host),
                    provider_type=str(provider_new),
                    cloud_id=str(cloud_id_new),
                    gpu_type=str(gpu_type_new),
                    image=str(image_new),
                    epoch=int(epoch),
                )
                return new_node

            enable_failover = str(args.enable_failover).lower() == "true"
            max_epochs = max(1, int(args.failover_max_epochs))
            start_step = 0
            t_train0 = time.monotonic()
            transcripts_local: Dict[int, Path] = {}
            results: Dict[int, bytes] = {}

            for epoch_idx in range(int(max_epochs)):
                steps_remaining = int(steps_job) - int(start_step)
                if steps_remaining <= 0:
                    break

                # Ensure live recorder is tracking the current node list (important if a pod was replaced).
                if str(args.start_live_recorder).lower() == "true":
                    try:
                        if live_recorder_pid is not None:
                            try:
                                os.kill(int(live_recorder_pid), signal.SIGTERM)
                            except Exception:
                                pass
                        rec_py = _repo_root() / "research" / "uvcc" / "uvcc-demo" / "record_live_logs_append.py"
                        rec_out = out_dir / "live_keep" / "recorder_stdout.log"
                        rec_out.parent.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            sys.executable,
                            str(rec_py),
                            "--out",
                            str(out_dir),
                            "--ssh-key-path",
                            str(ssh_key_path),
                            "--interval-s",
                            str(max(1.0, float(args.live_recorder_interval_s))),
                        ]
                        for nn in nodes:
                            cmd += ["--node", f"{int(nn.party_id)},{nn.ssh_host},{int(nn.ssh_port)},{nn.ssh_user}"]
                        with open(rec_out, "a", encoding="utf-8") as f:
                            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                        live_recorder_pid = int(p.pid)
                        (out_dir / "live_keep" / "recorder.pid").write_text(str(live_recorder_pid) + "\n", encoding="utf-8")
                        log.event("live_recorder_restarted", epoch=int(epoch_idx), pid=int(live_recorder_pid))
                    except Exception as exc:
                        log.event("live_recorder_restart_failed", epoch=int(epoch_idx), error=str(exc))

                # Upload inputs for this epoch (initial W or checkpoint W).
                _upload_inputs_for_epoch(start_step=int(start_step))

                # Preflight: right before we launch parties, verify each node still has a healthy CUDA stack.
                if bool(require_cuda_job):
                    for n in nodes:
                        ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                        try:
                            code_smi, out_smi, err_smi = ssh_exec(ssh, "bash -lc 'nvidia-smi -L'", timeout_s=60)
                            smi_text = (out_smi + err_smi).strip()
                            smi_ok = (int(code_smi) == 0) and ("Failed to initialize NVML" not in smi_text)

                            torch_cmd = (
                                f"{n.env_prefix()} {shlex.quote(n.venv_python())} -c "
                                + shlex.quote(
                                    "import torch; "
                                    "print('TORCH_VERSION=' + str(torch.__version__) + ' TORCH_CUDA_AVAILABLE=' + str(torch.cuda.is_available()))"
                                )
                            )
                            code_t, out_t, err_t = ssh_exec(ssh, f"bash -lc {shlex.quote(torch_cmd)}", timeout_s=60)
                            torch_text = (out_t + err_t).strip()
                            torch_ok = (int(code_t) == 0) and ("TORCH_CUDA_AVAILABLE=True" in torch_text)

                            log.event(
                                "cuda_health_preflight",
                                party_id=int(n.party_id),
                                ssh_host=str(n.ssh_host),
                                nvidia_smi_ok=bool(smi_ok),
                                nvidia_smi=_trim(smi_text, 800),
                                torch_ok=bool(torch_ok),
                                torch_out=_trim(torch_text, 800),
                                epoch=int(epoch_idx),
                                start_step=int(start_step),
                            )
                            if not smi_ok or not torch_ok:
                                raise RuntimeError(f"CUDA preflight failed on party {int(n.party_id)}: nvidia_smi_ok={smi_ok} torch_ok={torch_ok}")
                        finally:
                            ssh.close()

                group_id_epoch = f"g-{job_id32.hex()[:16]}-e{int(epoch_idx)}"
                epoch_setup_step = 1000 + int(epoch_idx)
                log.event(
                    "training_launch",
                    relay_url_public=str(relay_url_public),
                    relay_url_local=str(relay_url_local),
                    group_id=str(group_id_epoch),
                    steps=int(steps_remaining),
                    d=int(d_job),
                    sks_t_checks=int(sks_t_checks_job),
                    sks_sample_log2=int(sks_sample_log2_job),
                    party_log_level=str(args.party_log_level),
                    epoch=int(epoch_idx),
                    start_step=int(start_step),
                    epoch_setup_step=int(epoch_setup_step),
                )

                telemetry_remote: Dict[int, Tuple[str, str]] = {}  # party_id -> (csv_path, pid_path)
                if str(args.gpu_telemetry).lower() == "true":
                    interval_s = max(0.2, float(args.gpu_telemetry_interval_s))
                    interval_ms = int(interval_s * 1000.0)
                    for n in nodes:
                        ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                        try:
                            out_remote = f"{n.root_dir}/out_party_{int(n.party_id)}"
                            csv_path = f"{out_remote}/gpu_telemetry.csv"
                            pid_path = f"{out_remote}/gpu_telemetry.pid"
                            cmd = (
                                "set -euo pipefail;"
                                f" mkdir -p {shlex.quote(out_remote)};"
                                f" rm -f {shlex.quote(csv_path)} {shlex.quote(pid_path)};"
                                f" nohup bash -lc {shlex.quote('nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -lms ' + str(interval_ms) + ' > ' + csv_path)}"
                                f" >/dev/null 2>&1 & echo $! > {shlex.quote(pid_path)};"
                            )
                            ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=20)
                            telemetry_remote[int(n.party_id)] = (str(csv_path), str(pid_path))
                            log.event(
                                "gpu_telemetry_started",
                                party_id=int(n.party_id),
                                ssh_host=str(n.ssh_host),
                                remote_csv=str(csv_path),
                                interval_s=float(interval_s),
                                epoch=int(epoch_idx),
                            )
                        finally:
                            ssh.close()

                # Launch parties for this epoch segment.
                for n in nodes:
                    ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                    try:
                        out_remote = f"{n.root_dir}/out_party_{int(n.party_id)}"
                        ca_path = f"{n.root_dir}/relay_ca.pem"
                        token_path = f"{n.root_dir}/relay_token.txt"
                        inp_path = f"{n.root_dir}/inputs_p{int(n.party_id)}.json"
                        log_path = f"{out_remote}/run.log"
                        pid_path = f"{out_remote}/run.pid"
                        party_relay_url = str(relay_url_local) if int(n.party_id) == 0 else str(relay_url_public)
                        inner = (
                            f"{n.env_prefix()} {shlex.quote(n.venv_python())} -m uvcc_client run-party-train"
                            f" --party-id {int(n.party_id)}"
                            f" --relay-url {shlex.quote(party_relay_url)}"
                            f" --group-id {shlex.quote(group_id_epoch)}"
                            f" --relay-token-file {shlex.quote(token_path)}"
                            f" --tls-ca-pem {shlex.quote(ca_path)}"
                            f" --job-id-hex 0x{job_id32.hex()}"
                            f" --sid {shlex.quote(sid.decode('utf-8'))}"
                            f" --inputs-json {shlex.quote(inp_path)}"
                            f" --out {shlex.quote(out_remote)}"
                            f" --device {'cuda' if require_cuda_job else 'auto'} --require-cuda {'true' if require_cuda_job else 'false'}"
                            f" --steps {int(steps_remaining)}"
                            f" --epoch {int(epoch_idx)} --step-offset {int(start_step)} --epoch-setup-step {int(epoch_setup_step)}"
                            f" --checkpoint-enable {shlex.quote(str(args.checkpoint_enable))} --checkpoint-every {int(args.checkpoint_every)}"
                            f" --sks-t-checks {int(sks_t_checks_job)} --sks-sample-log2 {int(sks_sample_log2_job)}"
                            f" --log-level {shlex.quote(str(args.party_log_level))}"
                        )
                        cmd = (
                            "set -euo pipefail;"
                            f" mkdir -p {shlex.quote(out_remote)};"
                            f" if [ -f {shlex.quote(pid_path)} ]; then"
                            f"   (kill -9 $(cat {shlex.quote(pid_path)}) >/dev/null 2>&1 || true);"
                            f"   rm -f {shlex.quote(pid_path)};"
                            " fi;"
                            f" rm -f {shlex.quote(log_path)};"
                            f" nohup bash -lc {shlex.quote(inner)}"
                            f" > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)};"
                            " sleep 0.2;"
                            f" PID=$(cat {shlex.quote(pid_path)});"
                            " kill -0 $PID >/dev/null 2>&1 || (echo 'party_process_exited_early' >&2; tail -n 200 "
                            f"{shlex.quote(log_path)} >&2 || true; exit 1);"
                        )
                        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
                        if code != 0:
                            raise RuntimeError(f"failed to launch party:\n{out}\n{err}")
                    finally:
                        ssh.close()

                # Poll completion and fetch artifacts for this epoch segment.
                try:
                    # Connect once per party and do round-robin polling so we can detect ANY dead party quickly.
                    # This prevents long hangs where party0/party1 wait for a dead peer, time out, and then appear
                    # as the "failed" party.
                    ssh_by_pid: Dict[int, Any] = {}
                    paths_by_pid: Dict[int, Dict[str, str]] = {}
                    try:
                        for n in nodes:
                            pid = int(n.party_id)
                            try:
                                ssh_by_pid[int(pid)] = ssh_connect_with_retries(
                                    hostname=n.ssh_host,
                                    port=n.ssh_port,
                                    username=n.ssh_user,
                                    pkey=pkey,
                                    timeout_s=900,
                                )
                            except Exception as exc:
                                raise RuntimeError(f"party {int(pid)} ssh_connect failed: {exc}") from exc
                            out_remote = f"{n.root_dir}/out_party_{int(pid)}"
                            paths_by_pid[int(pid)] = {
                                "out_remote": str(out_remote),
                                "res_path": str(f"{out_remote}/result.json"),
                                "tr_path": str(f"{out_remote}/transcript_v1.jsonl"),
                                "run_log_remote": str(f"{out_remote}/run.log"),
                                "run_pid_remote": str(f"{out_remote}/run.pid"),
                            }

                        deadline = time.time() + max(60, int(party_timeout_effective_s))
                        done: set[int] = set()
                        while time.time() < deadline:
                            for n in nodes:
                                pid = int(n.party_id)
                                if int(pid) in done:
                                    continue
                                ssh = ssh_by_pid[int(pid)]
                                p = paths_by_pid[int(pid)]
                                status_cmd = (
                                    f"if [ -f {p['res_path']} ]; then echo OK; "
                                    f"elif [ -f {p['run_pid_remote']} ]; then "
                                    f"PID=$(cat {p['run_pid_remote']}); kill -0 $PID >/dev/null 2>&1 && echo RUNNING || echo DEAD; "
                                    f"else echo NOPID; fi"
                                )
                                try:
                                    _c, out, _e = ssh_exec(ssh, f"bash -lc {shlex.quote(status_cmd)}", timeout_s=10)
                                except Exception as exc:
                                    raise RuntimeError(f"party {int(pid)} status check failed: {exc}") from exc
                                outp = str(out or "").strip()
                                if "OK" in outp:
                                    done.add(int(pid))
                                    continue
                                if "DEAD" in outp:
                                    run_local = out_dir / f"party_p{int(pid)}_run.log"
                                    try:
                                        sftp_get_file(ssh, remote_path=p["run_log_remote"], local_path=str(run_local))
                                    except Exception:
                                        run_local = out_dir / f"party_p{int(pid)}_run.log.missing"
                                        run_local.write_text("", encoding="utf-8")
                                    tail_out, tail_err = "", ""
                                    try:
                                        tail_cmd = f"tail -n 200 {p['run_log_remote']} || true"
                                        _tc, tail_out, tail_err = ssh_exec(
                                            ssh,
                                            f"bash -lc {shlex.quote(tail_cmd)}",
                                            timeout_s=20,
                                        )
                                    except Exception:
                                        pass
                                    log.event(
                                        "party_failed",
                                        party_id=int(pid),
                                        ssh_host=str(n.ssh_host),
                                        reason="process_exited",
                                        run_log_path=str(run_local),
                                        run_log_tail=_trim(tail_out + tail_err, 4000),
                                        epoch=int(epoch_idx),
                                        start_step=int(start_step),
                                    )
                                    raise RuntimeError(f"party {int(pid)} process exited (see {run_local})")

                            if len(done) == len(nodes):
                                break
                            time.sleep(0.5)

                        if len(done) != len(nodes):
                            # Emit a timeout event for each non-completed party.
                            for n in nodes:
                                pid = int(n.party_id)
                                if int(pid) in done:
                                    continue
                                ssh = ssh_by_pid[int(pid)]
                                p = paths_by_pid[int(pid)]
                                tail_out, tail_err = "", ""
                                try:
                                    tail_cmd = f"tail -n 200 {p['run_log_remote']} || true"
                                    _tc, tail_out, tail_err = ssh_exec(
                                        ssh,
                                        f"bash -lc {shlex.quote(tail_cmd)}",
                                        timeout_s=20,
                                    )
                                except Exception:
                                    pass
                                log.event(
                                    "party_timeout",
                                    party_id=int(pid),
                                    ssh_host=str(n.ssh_host),
                                    party_timeout_effective_s=int(party_timeout_effective_s),
                                    run_log_tail=_trim(tail_out + tail_err, 4000),
                                    epoch=int(epoch_idx),
                                    start_step=int(start_step),
                                )
                            raise RuntimeError(
                                f"timed out waiting for all parties result.json after {int(party_timeout_effective_s)}s "
                                f"(steps_total={int(steps_job)} start_step={int(start_step)} steps_remaining={int(steps_remaining)})."
                            )

                        # All parties done: fetch artifacts for each party.
                        for n in nodes:
                            pid = int(n.party_id)
                            ssh = ssh_by_pid[int(pid)]
                            p = paths_by_pid[int(pid)]

                            cat_cmd = f"cat {p['res_path']}"
                            code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cat_cmd)}", timeout_s=20)
                            if code != 0:
                                raise RuntimeError(f"party {int(pid)} missing result.json:\n{out}\n{err}")
                            obj = json.loads(out.strip())
                            rh = str(obj["result_hash32_hex"])
                            if not rh.startswith("0x") or len(rh) != 66:
                                raise RuntimeError("bad result_hash32_hex")
                            results[int(pid)] = bytes.fromhex(rh[2:])

                            tr_local = out_dir / f"transcript_p{int(pid)}.jsonl"
                            try:
                                sftp_get_file(ssh, remote_path=p["tr_path"], local_path=str(tr_local))
                                transcripts_local[int(pid)] = tr_local
                            except Exception:
                                # If transcript_v1.jsonl isn't present, we will fall back to the live recorder mirror.
                                transcripts_local[int(pid)] = tr_local
                                tr_local.write_text("", encoding="utf-8")

                            run_local = out_dir / f"party_p{int(pid)}_run.log"
                            try:
                                sftp_get_file(ssh, remote_path=p["run_log_remote"], local_path=str(run_local))
                            except Exception:
                                run_local = out_dir / f"party_p{int(pid)}_run.log.missing"
                                run_local.write_text("", encoding="utf-8")

                            if int(pid) in telemetry_remote:
                                csv_remote, pid_remote = telemetry_remote[int(pid)]
                                try:
                                    ssh_exec(
                                        ssh,
                                        f"bash -lc {shlex.quote(f'if [ -f {pid_remote} ]; then kill -9 $(cat {pid_remote}) >/dev/null 2>&1 || true; fi')}",
                                        timeout_s=10,
                                    )
                                except Exception:
                                    pass
                                tele_local = out_dir / f"gpu_telemetry_p{int(pid)}.csv"
                                try:
                                    sftp_get_file(ssh, remote_path=str(csv_remote), local_path=str(tele_local))
                                    log.event("gpu_telemetry_fetched", party_id=int(pid), path=str(tele_local), epoch=int(epoch_idx))
                                except Exception:
                                    tele_local.write_text("", encoding="utf-8")

                            log.event(
                                "party_done",
                                party_id=int(pid),
                                ssh_host=str(n.ssh_host),
                                result_hash32_hex=str(rh),
                                transcript_path=str(tr_local),
                                run_log_path=str(run_local),
                                epoch=int(epoch_idx),
                                start_step=int(start_step),
                            )
                    finally:
                        for _ssh in ssh_by_pid.values():
                            try:
                                _ssh.close()
                            except Exception:
                                pass
                    # Success!
                    break
                except Exception as exc:
                    err_s = str(exc) or exc.__class__.__name__
                    # Best-effort: identify which party failed from the error message (for targeted pod replacement).
                    fail_pid = None
                    m = re.search(r"party\s+(\d+)", err_s)
                    if m:
                        try:
                            fail_pid = int(m.group(1))
                        except Exception:
                            fail_pid = None
                    log.event("training_epoch_failed", epoch=int(epoch_idx), start_step=int(start_step), error=str(err_s), failed_party_id=fail_pid)

                    # Best-effort stop parties/telemetry for this epoch (avoid wasting GPU while we failover).
                    for nn in nodes:
                        try:
                            ssh = ssh_connect_with_retries(hostname=nn.ssh_host, port=nn.ssh_port, username=nn.ssh_user, pkey=pkey, timeout_s=60)
                        except Exception:
                            continue
                        try:
                            out_remote = f"{nn.root_dir}/out_party_{int(nn.party_id)}"
                            run_pid_remote = f"{out_remote}/run.pid"
                            try:
                                ssh_exec(
                                    ssh,
                                    f"bash -lc {shlex.quote(f'if [ -f {run_pid_remote} ]; then kill -9 $(cat {run_pid_remote}) >/dev/null 2>&1 || true; fi')}",
                                    timeout_s=10,
                                )
                            except Exception:
                                pass
                            if int(nn.party_id) in telemetry_remote:
                                _csv_remote, pid_remote = telemetry_remote[int(nn.party_id)]
                                try:
                                    ssh_exec(
                                        ssh,
                                        f"bash -lc {shlex.quote(f'if [ -f {pid_remote} ]; then kill -9 $(cat {pid_remote}) >/dev/null 2>&1 || true; fi')}",
                                        timeout_s=10,
                                    )
                                except Exception:
                                    pass
                        finally:
                            try:
                                ssh.close()
                            except Exception:
                                pass

                    if not enable_failover or int(epoch_idx) + 1 >= int(max_epochs):
                        raise

                    # Decide which pods to replace before retrying.
                    #
                    # Key nuance: party0 hosts the relay, but party0 *training process* can still be restarted on
                    # the same pod. We only abort if the party0 *pod itself* is unreachable.
                    unreachable: List[int] = []
                    for nn in nodes:
                        try:
                            ssh = ssh_connect_with_retries(
                                hostname=nn.ssh_host,
                                port=nn.ssh_port,
                                username=nn.ssh_user,
                                pkey=pkey,
                                timeout_s=20,
                            )
                            try:
                                ssh_exec(ssh, "bash -lc 'echo ok'", timeout_s=10)
                            finally:
                                ssh.close()
                        except Exception:
                            unreachable.append(int(nn.party_id))
                    if unreachable:
                        log.event("failover_unreachable_parties", parties=list(unreachable), epoch=int(epoch_idx), start_step=int(start_step))
                    if 0 in unreachable:
                        # Relay host *pod* is unreachable → we cannot recover in this simplified mode.
                        log.event("failover_replace_unsupported", party_id=0, reason="party0_unreachable_relay_host_pod")
                        raise

                    # Best-effort: sync remote checkpoint files into local private_keep BEFORE we delete/replace any pods.
                    # This avoids losing the newest checkpoint if the failed party pod is about to be deleted and the
                    # live recorder hasn't polled the checkpoint file yet.
                    ckpt_sync: Dict[str, Dict[str, Any]] = {}
                    for nn in nodes:
                        pid = int(nn.party_id)
                        remote_ckpt = f"{nn.root_dir}/out_party_{int(pid)}/private/checkpoints_W.jsonl"
                        local_ckpt = private_keep / f"checkpoints_W_p{int(pid)}.jsonl"
                        tmp_ckpt = private_keep / f".tmp_checkpoints_W_p{int(pid)}.jsonl"
                        try:
                            ssh = ssh_connect_with_retries(
                                hostname=nn.ssh_host,
                                port=nn.ssh_port,
                                username=nn.ssh_user,
                                pkey=pkey,
                                timeout_s=30,
                            )
                            try:
                                # Fetch to a temp file; then only overwrite if the remote is larger/equal.
                                # If remote is smaller (e.g., truncated due to failure), keep the local copy.
                                try:
                                    sftp_get_file(ssh, remote_path=str(remote_ckpt), local_path=str(tmp_ckpt))
                                except Exception as exc_get:
                                    ckpt_sync[str(pid)] = {"ok": False, "error": f"fetch_failed: {exc_get}"}
                                    continue
                                try:
                                    remote_sz = int(tmp_ckpt.stat().st_size) if tmp_ckpt.exists() else 0
                                    local_sz = int(local_ckpt.stat().st_size) if local_ckpt.exists() else 0
                                    if remote_sz > 0 and remote_sz >= local_sz:
                                        local_ckpt.write_bytes(tmp_ckpt.read_bytes())
                                        local_sz = remote_sz
                                    ckpt_sync[str(pid)] = {"ok": True, "remote_bytes": int(remote_sz), "local_bytes": int(local_sz)}
                                finally:
                                    try:
                                        tmp_ckpt.unlink()
                                    except Exception:
                                        pass
                            finally:
                                try:
                                    ssh.close()
                                except Exception:
                                    pass
                        except Exception as exc:
                            ckpt_sync[str(pid)] = {"ok": False, "error": f"ssh_failed: {exc}"}
                    if ckpt_sync:
                        log.event("failover_checkpoint_sync", epoch=int(epoch_idx), start_step=int(start_step), results=ckpt_sync)

                    # Choose resume point from last common checkpoint (after best-effort sync).
                    lasts = {pid: _last_ckpt_step(pid) for pid in (0, 1, 2)}
                    common = min(lasts.values())
                    new_start = max(0, int(common) + 1)
                    log.event("failover_resume_selected", last_ckpt_by_party=lasts, resume_from_step=int(new_start))
                    # Explicit audit note for operators: checkpoints + append-only live recorder are the mechanisms
                    # that prevent log/telemetry loss across failovers. Private artifacts remain in private_keep.
                    try:
                        live_state = out_dir / "live_keep" / "recorder_state.json"
                        live_state_sha256 = _sha256_hex(live_state.read_bytes()) if live_state.exists() else ""
                    except Exception:
                        live_state_sha256 = ""
                    log.event(
                        "failover_no_loss_assertion",
                        epoch=int(epoch_idx),
                        resume_from_step=int(new_start),
                        checkpoint_strategy="last_common_checkpoint_plus1",
                        append_only_recorder=True,
                        live_keep_dir=str(out_dir / "live_keep"),
                        private_keep_dir=str(private_keep),
                        recorder_state_sha256=str(live_state_sha256),
                        privacy_note="private_keep contains secret shares/checkpoints and is excluded from public bundles",
                    )
                    start_step = int(new_start)

                    pids_to_replace: List[int] = []
                    if fail_pid in (1, 2):
                        # If a non-relay party failed, replace that pod.
                        pids_to_replace = [int(fail_pid)]
                    else:
                        # If party0 failed (likely due to waiting on a dead peer), or we couldn't infer which party
                        # failed, replace any unreachable non-relay parties (if any) and restart the epoch.
                        pids_to_replace = [int(x) for x in unreachable if int(x) != 0]
                        if int(fail_pid or -1) == 0:
                            log.event(
                                "failover_party0_process_failed",
                                epoch=int(epoch_idx),
                                start_step=int(start_step),
                                note="party0 training process failed; relay host pod is retained and epoch will be restarted",
                            )

                    for pid_r in list(dict.fromkeys([int(x) for x in pids_to_replace])):
                        try:
                            _replace_party_pod(party_id=int(pid_r), epoch=int(epoch_idx) + 1)
                        except Exception as exc2:
                            log.event("failover_replace_failed", party_id=int(pid_r), error=str(exc2))
                            raise
                    # Continue to next epoch (attempt).
                    continue

            if results[0] != results[1] or results[0] != results[2]:
                raise RuntimeError("result_hash mismatch across parties")
            result_hash32 = results[0]
            log.event("training_done", t_s=round(time.monotonic() - t_train0, 3), result_hash32_hex="0x" + result_hash32.hex())

            # Summarize GPU telemetry (best-effort).
            if str(args.gpu_telemetry).lower() == "true":
                for pid in (0, 1, 2):
                    tele = out_dir / f"gpu_telemetry_p{int(pid)}.csv"
                    if tele.exists() and tele.stat().st_size > 0:
                        log.event("gpu_telemetry_summary", party_id=int(pid), path=str(tele), **_summarize_gpu_telemetry_csv(tele))

            # Union transcript file.
            union_lines: List[str] = []
            for pid in (0, 1, 2):
                # Prefer the local live recorder mirror (captures partial progress across failovers / pod loss).
                live_tr = out_dir / "live_keep" / f"party_p{int(pid)}_transcript.jsonl"
                if live_tr.exists() and live_tr.stat().st_size > 0:
                    union_lines += live_tr.read_text(encoding="utf-8", errors="replace").splitlines()
                else:
                    union_lines += transcripts_local[pid].read_text(encoding="utf-8", errors="replace").splitlines()
            union_text = "\n".join([ln for ln in union_lines if ln.strip()]) + "\n"
            union_path = out_dir / "transcript_v1.jsonl"
            union_path.write_text(union_text, encoding="utf-8")

            leaves = parse_transcript_jsonl_v1(str(union_path))
            validate_transcript_leaves_v1(leaves, strict_unknown_msg_kind=False, strict_netframe_header_hash=True)
            roots_by_epoch = compute_epoch_roots_v1(leaves)
            if not roots_by_epoch:
                raise RuntimeError("transcript had zero epoch roots")
            max_epoch = max(int(e) for e in roots_by_epoch.keys())
            epoch_roots = [roots_by_epoch.get(e, b"") for e in range(int(max_epoch) + 1)]
            if any(len(r) != 32 for r in epoch_roots):
                raise RuntimeError("missing one or more epoch roots (non-contiguous epochs?)")
            final_root32 = compute_final_root_v1(epoch_roots=epoch_roots)
            log.event(
                "transcript_roots",
                union_transcript_path=str(union_path),
                epoch0_root_hex="0x" + epoch_roots[0].hex(),
                epoch_roots_hex=["0x" + r.hex() for r in epoch_roots],
                final_root_hex="0x" + final_root32.hex(),
            )

            # Final commit signatures from parties.
            digest_final = FinalCommitV1(job_id32=job_id32, policy_hash32=policy_hash32, final_root32=final_root32, result_hash32=result_hash32).digest32(domain=dom)
            sig_final_hex: Dict[int, str] = {}
            for n in nodes:
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    key_path = f"{n.root_dir}/party_privkey.hex"
                    cmd = (
                        f"{n.env_prefix()} {shlex.quote(n.venv_python())} -m uvcc_client party-sign"
                        f" --party-id {int(n.party_id)} --key-path {shlex.quote(str(key_path))} --digest-hex 0x{digest_final.hex()}"
                    )
                    code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=60)
                    if code != 0:
                        raise RuntimeError(f"party-sign final failed:\n{out}\n{err}")
                    info = json.loads(out.strip())
                    sig_final_hex[int(n.party_id)] = str(info["sig65_hex"])
                finally:
                    ssh.close()
            log.event(
                "final_commit_signed",
                digest_final_hex="0x" + digest_final.hex(),
                sig_p0=str(sig_final_hex.get(0, "")),
                sig_p1=str(sig_final_hex.get(1, "")),
                sig_p2=str(sig_final_hex.get(2, "")),
            )

            parties = []
            sigs = []
            for n in nodes:
                if n.pubkey64_hex is None:
                    raise RuntimeError("missing pubkey64")
                pubhex = str(n.pubkey64_hex)
                if not pubhex.startswith("0x") or len(pubhex) != 130:
                    raise RuntimeError("bad pubkey64 hex")
                parties.append(ProofBundlePartyV1(party_id=int(n.party_id), pubkey64=bytes.fromhex(pubhex[2:])))
                s = sig_final_hex[int(n.party_id)]
                if not s.startswith("0x"):
                    raise RuntimeError("bad signature hex")
                sigs.append(ProofBundleSignatureV1(party_id=int(n.party_id), sig65=bytes.fromhex(s[2:])))

            pb = ProofBundleV1(
                uvcc_version="1.0",
                job_id32=job_id32,
                policy_hash32=policy_hash32,
                eip712_domain=dom,
                sgir_hash32=sgir_hash32,
                runtime_hash32=runtime_hash32,
                backend="CRYPTO_CC_3PC",
                parties=parties,
                epoch_roots=epoch_roots,
                final_root32=final_root32,
                signatures=sigs,
                result_hash32=result_hash32,
                status="OK",
            )
            proof_json = pb.to_json_bytes()
            (out_dir / "proof_bundle.json").write_bytes(proof_json)
            proof_hash32_hex = "0x" + hashlib.sha256(b"uvcc.proofbundle.v1" + proof_json).digest().hex()

            # Self-check verifier.
            proof_parsed = parse_proof_bundle_json_v1(proof_json)
            verify_proof_bundle_v1(proof=proof_parsed, transcript_epoch_roots=epoch_roots, transcript_final_root32=final_root32)
            log.event(
                "verifier_ok",
                proof_bundle_path=str(out_dir / "proof_bundle.json"),
                proof_bundle_hash32_hex=str(proof_hash32_hex),
            )

            # Submit final on-chain.
            final_out = _cast_send(
                rpc_url=rpc_url,
                privkey_hex=priv0,
                to=ledger,
                sig="submitFinal(bytes32,bytes32,bytes32,bytes,bytes,bytes)",
                args=[
                    "0x" + job_id32.hex(),
                    "0x" + final_root32.hex(),
                    "0x" + result_hash32.hex(),
                    sig_final_hex[0],
                    sig_final_hex[1],
                    sig_final_hex[2],
                ],
            )
            final_log = out_dir / "onchain_submitFinal.log"
            final_log.write_text(str(final_out), encoding="utf-8")
            log.event("onchain_submitFinal", log_path=str(final_log))

            on_job = _cast_call(
                rpc_url=rpc_url,
                to=ledger,
                sig="jobs(bytes32)(bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8,uint64,address,uint64,bool,bytes32,bytes32,uint64,bool,address,uint64,bool,bool)",
                args=["0x" + job_id32.hex()],
            )
            if final_root32.hex() not in on_job.lower():
                raise RuntimeError(f"on-chain final_root mismatch: {on_job}")
            if result_hash32.hex() not in on_job.lower():
                raise RuntimeError(f"on-chain result_hash mismatch: {on_job}")

            # Collect relay log (best-effort) for full run artifact completeness.
            relay_log_local = out_dir / "relay_node0.log"
            try:
                ssh = ssh_connect_with_retries(hostname=nodes[0].ssh_host, port=nodes[0].ssh_port, username=nodes[0].ssh_user, pkey=pkey, timeout_s=900)
                try:
                    sftp_get_file(ssh, remote_path=relay_log, local_path=str(relay_log_local))
                finally:
                    ssh.close()
            except Exception:
                relay_log_local.write_text("", encoding="utf-8")

            # Write a public verification recipe (does not contain secrets).
            verify_md = out_dir / "how_to_verify_public.md"
            verify_md.write_text(
                "\n".join(
                    [
                        "## UVCC public verification recipe (v1)",
                        "",
                        "This file is sufficient for a third party to verify the run artifacts and the on-chain receipts.",
                        "",
                        "### Inputs (from this run)",
                        f"- job_id32: 0x{job_id32.hex()}",
                        f"- epoch_roots: [{', '.join('0x' + r.hex() for r in epoch_roots)}]",
                        f"- final_root: 0x{final_root32.hex()}",
                        f"- result_hash32: 0x{result_hash32.hex()}",
                        f"- proof_bundle_hash32: {proof_hash32_hex}",
                        "",
                        "### Verify the proof bundle vs transcript (off-chain)",
                        "",
                        "Run on any machine with this repo checked out:",
                        "",
                        "```bash",
                        "PYTHONPATH=research/uvcc/uvcc-verifier \\",
                        "  python3 -m uvcc_verifier verify \\",
                        f"    --proof {shlex.quote(str(out_dir / 'proof_bundle.json'))} \\",
                        f"    --transcript {shlex.quote(str(union_path))}",
                        "```",
                        "",
                        "### Check on-chain receipts (local Anvil)",
                        "",
                        f"- rpc_url: {rpc_url}",
                        f"- ledger: {ledger}",
                        "",
                        "```bash",
                        f"cast call --rpc-url {shlex.quote(str(rpc_url))} {shlex.quote(str(ledger))} \"jobs(bytes32)(bytes32,bytes32,bytes32,bytes32,bytes32,bytes32,uint8,uint64,address,uint64,bool,bytes32,bytes32,uint64,bool,address,uint64,bool,bool)\" 0x{job_id32.hex()}",
                        "```",
                        "",
                        "Confirm the returned tuple contains the expected `final_root` and `result_hash32`.",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            log.log("UVCC Prime 3-node run complete.")
            log.log("=== Privacy ===")
            log.log(f"- relay_url_public: {relay_url_public}")
            log.log(f"- relay_url_local: {relay_url_local}")
            log.log(f"- relay_tls_ca_sha256: {_sha256_hex(ca_pem)}")
            log.log("- relay_token: stored in file on nodes (not in args/logs)")
            log.log("=== Verifiability ===")
            log.log(f"- job_id32: 0x{job_id32.hex()}")
            log.log(f"- epoch0_root: 0x{epoch_roots[0].hex()}")
            log.log(f"- final_root: 0x{final_root32.hex()}")
            log.log(f"- result_hash32: 0x{result_hash32.hex()}")
            log.log(f"- proof_bundle_hash32: {proof_hash32_hex}")
            log.log("=== Efficiency ===")
            log.log(f"- prime_cloud_id: {cloud_id}")
            log.log(f"- prime_gpu_type: {gpu_type}")
            log.log(f"- job_d: {d_job} steps: {steps_job} sks_t_checks: {sks_t_checks_job} sks_sample_log2: {sks_sample_log2_job}")
            log.log(f"- gpu_telemetry: {args.gpu_telemetry} interval_s: {args.gpu_telemetry_interval_s}")
            log.log("=== Artifacts ===")
            log.log(f"- run_full.log: {log.log_path}")
            log.log(f"- run_full.jsonl: {log.jsonl_path}")
            log.log(f"- transcript_v1.jsonl: {union_path}")
            log.log(f"- proof_bundle.json: {out_dir / 'proof_bundle.json'}")
            log.log(f"- relay_node0.log: {relay_log_local}")
            log.log(f"- onchain_createJob.log: {create_log}")
            log.log(f"- onchain_submitFinal.log: {final_log}")
            log.log(f"- how_to_verify_public.md: {verify_md}")
            log.log("- gpu_telemetry_p*.csv: per-node sampled nvidia-smi telemetry (if enabled)")
            log.event(
                "artifacts",
                run_full_log=str(log.log_path),
                run_full_jsonl=str(log.jsonl_path),
                proof_bundle=str(out_dir / "proof_bundle.json"),
                transcript=str(union_path),
                relay_log=str(relay_log_local),
                onchain_createJob=str(create_log),
                onchain_submitFinal=str(final_log),
                how_to_verify_public=str(verify_md),
            )
            return 0
        except Exception as exc:
            # Ensure failures are always visible in both logs and JSONL.
            log.log(f"ERROR: run failed: {exc}")
            log.event("run_failed", error=str(exc))
            raise
        finally:
            # Best-effort cleanup to avoid runaway spend.
            if str(args.keep_pods).lower() != "true":
                try:
                    if prime is not None and created_pod_ids:
                        for pid in list(created_pod_ids):
                            try:
                                prime.delete_pod(pid)
                                log.event("prime_pod_deleted", pod_id=str(pid))
                            except Exception as exc:
                                log.log(f"WARN: failed to delete pod {pid}: {exc}")
                except Exception:
                    pass
            proc_anvil.terminate()
            try:
                proc_anvil.wait(timeout=3)
            except Exception:
                proc_anvil.kill()


if __name__ == "__main__":
    raise SystemExit(main())


