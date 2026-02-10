from __future__ import annotations

# pyright: reportMissingImports=false

"""
Prime runner for SR-DP (Secure-Replica Data Parallel) on top of UVCC.

Target topology (v1):
- R replicas (default 8)
- 3 parties (P0,P1,P2) on 3 different Prime providers
- total pods: 3 * R (default 24), each pod has 1 GPU

MPC plane:
- Each replica r runs an independent UVCC 3PC triangle with its own sid_rep[r] and relay group_id.

DP plane:
- Within each party domain, all R replica pods join a torch.distributed NCCL group and allreduce the
  gradient shares per step (SUM).

This script is intentionally additive and does not modify `research/uvcc/uvcc-demo/run_prime_3node.py`.
"""

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import re
import shlex
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _repo_root() -> Path:
    # .../research/uvcc_parallel/run_prime_dp_3xR.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _add_paths() -> None:
    import sys

    root = _repo_root()
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-client"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-party"))
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-verifier"))
    # Make `uvcc_parallel` importable as a top-level package.
    # (We import `uvcc_parallel.*`, so the parent directory must be on sys.path.)
    sys.path.insert(0, str(root / "research"))


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _trim(s: str, n: int = 2000) -> str:
    t = str(s)
    if len(t) <= int(n):
        return t
    return t[: int(n)] + f"...(truncated,{len(t)} chars)"


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _hex32(b: bytes) -> str:
    if len(b) != 32:
        raise ValueError("expected 32 bytes")
    return "0x" + b.hex()


def _sanity_check_party_run_log(
    *,
    run_log_path: Path,
    expect_party_id: int,
    expect_steps: int,
    expect_result_hash32_hex: Optional[str],
) -> None:
    """
    Guard against false-positive completion when reusing pods.

    If a pod is reused (attach-prefix), a stale remote `result.json` could make the orchestrator
    think a run is complete immediately unless we both:
      (a) delete stale artifacts before launch, and
      (b) verify the downloaded per-party `run.log` matches the requested job.

    At `--party-log-level info`, the party worker emits:
      - party_start (includes steps)
      - step_start (for every step)
      - open_final_done (includes result_hash32_hex)

    We validate those signals here.
    """
    p = Path(run_log_path)
    if not p.is_file():
        raise RuntimeError(f"missing party run.log: {p}")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise RuntimeError(f"empty party run.log: {p}")

    saw_party_start = False
    saw_open_final = False
    open_final_hash: Optional[str] = None
    step_ids: set[int] = set()
    max_step = -1

    for ln in lines:
        t = str(ln).strip()
        if not t:
            continue
        try:
            obj = json.loads(t)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if int(obj.get("party_id", -1)) != int(expect_party_id):
            continue
        ev = str(obj.get("event") or "")
        fields = obj.get("fields")
        fd: Dict[str, Any] = fields if isinstance(fields, dict) else {}

        if ev == "party_start":
            if "steps" not in fd:
                raise RuntimeError(f"party_start missing steps (party_id={expect_party_id}): {p}")
            try:
                got_steps = int(fd.get("steps"))
            except Exception:
                raise RuntimeError(f"party_start steps invalid (party_id={expect_party_id}): {fd.get('steps')}")
            if int(got_steps) != int(expect_steps):
                raise RuntimeError(f"party_start.steps mismatch (party_id={expect_party_id}): got={got_steps} want={expect_steps}")
            saw_party_start = True

        if ev == "step_start":
            st = fd.get("step")
            try:
                si = int(st)
            except Exception:
                continue
            max_step = max(max_step, si)
            step_ids.add(si)

        if ev == "open_final_done":
            saw_open_final = True
            rh = fd.get("result_hash32_hex")
            if rh is not None:
                open_final_hash = str(rh)

    if not saw_party_start:
        raise RuntimeError(f"missing party_start in run.log (party_id={expect_party_id}) — rerun with --party-log-level info: {p}")
    if not saw_open_final:
        raise RuntimeError(f"missing open_final_done in run.log (party_id={expect_party_id}) — rerun with --party-log-level info: {p}")
    if int(max_step) < int(expect_steps) - 1:
        raise RuntimeError(f"incomplete steps in run.log (party_id={expect_party_id}): max_step={max_step} want>={int(expect_steps) - 1}")

    missing_steps = [i for i in range(int(expect_steps)) if int(i) not in step_ids]
    if missing_steps:
        raise RuntimeError(f"missing step_start entries in run.log (party_id={expect_party_id}): missing={missing_steps[:10]}")

    if expect_result_hash32_hex is not None and open_final_hash is not None:
        if str(open_final_hash).strip().lower() != str(expect_result_hash32_hex).strip().lower():
            raise RuntimeError(
                f"open_final_done.result_hash32_hex mismatch (party_id={expect_party_id}): got={open_final_hash} want={expect_result_hash32_hex}"
            )


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
        self._lock = threading.RLock()
        # rotate old
        if self.log_path.exists():
            ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            self.log_path.replace(self.out_dir / f"run_full.{ts}.log")
        if self.jsonl_path.exists():
            ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            self.jsonl_path.replace(self.out_dir / f"run_full.{ts}.jsonl")

    def _write_text(self, line: str) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def log(self, msg: str) -> None:
        with self._lock:
            line = f"[{_now_iso_utc()}] {msg}\n"
            print(line, end="")
            self._write_text(line)

    def event(self, name: str, **fields: Any) -> None:
        with self._lock:
            # Drop obvious secret keys.
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
            self._write_text(
                f"[{rec['ts']}] EVENT {rec['event']} {json.dumps(rec['fields'], sort_keys=True, separators=(',', ':'))}\n"
            )


def _build_uvcc_dp_bundle_tgz(out_path: Path) -> None:
    """
    Bundle the remote code needed for DP runs:
      - uvcc-client
      - uvcc-party
      - uvcc-relay
      - uvcc_parallel (this overlay)
      - requirements-uvcc-base.txt
    """
    root = _repo_root()
    targets = [
        root / "research" / "uvcc" / "uvcc-client",
        root / "research" / "uvcc" / "uvcc-party",
        root / "research" / "uvcc" / "uvcc-relay",
        root / "research" / "uvcc_parallel",
        root / "research" / "uvcc" / "requirements-uvcc-base.txt",
    ]
    for t in targets:
        if not t.exists():
            raise FileNotFoundError(str(t))
    def _tar_filter(ti: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        """
        Keep the remote bundle small and deterministic:
          - never include local output directories (out-*/), caches, or venvs
          - never include existing bundles/logs
        """
        name = str(ti.name).lstrip("./")
        parts = [p for p in name.split("/") if p]
        lower = name.lower()

        # Skip common caches/venvs and git metadata.
        for bad in (
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            "venv",
        ):
            if bad in parts:
                return None

        # Skip local run outputs (very large, and can otherwise make the bundle include itself).
        if any(str(p).startswith("out-") for p in parts):
            return None

        # Skip common large artifacts.
        if lower.endswith((".tgz", ".tar.gz", ".zip", ".pt", ".pth", ".bin", ".safetensors", ".onnx")):
            return None

        # Skip local logs/recordings (not needed on remote).
        if lower.endswith((".log", ".jsonl", ".csv")):
            return None

        # Skip OS/editor junk.
        if parts and parts[-1].lower() in (".ds_store",):
            return None

        return ti

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(out_path), "w:gz") as tf:
        for t in targets:
            arc = t.relative_to(root)
            tf.add(str(t), arcname=str(arc), filter=_tar_filter)


def _tls_ca_and_server_cert(*, host_or_ip: str) -> Tuple[bytes, bytes, bytes]:
    """
    Return (ca_cert_pem, server_cert_pem, server_key_pem) for relay TLS.
    Copied from the 3-node runner, kept local to avoid importing that script.
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
        ip = ipaddress.ip_address(cn)
        san_items.append(x509.IPAddress(ip))
    except Exception:
        san_items.append(x509.DNSName(cn))
    san_items.append(x509.DNSName("localhost"))
    san_items.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name)
        .issuer_name(ca_name)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=14))
        .add_extension(x509.SubjectAlternativeName(san_items), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    srv_pem = srv_cert.public_bytes(serialization.Encoding.PEM)
    srv_key_pem = srv_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return ca_pem, srv_pem, srv_key_pem


@dataclass
class DPRemoteNodeV1:
    party_id: int
    replica_id: int
    pod_id: str
    provider_type: str
    ssh_user: str
    ssh_host: str
    ssh_port: int
    home: str
    root_dir: str
    python_bin: str

    def venv_python(self) -> str:
        # Historical name: for v1 we run directly with system python3 (+ user site packages).
        return str(self.python_bin)

    def env_prefix(self) -> str:
        # Ensure user-level console scripts are visible (e.g. ninja) and our code is importable.
        py_path = ":".join(
            [
                f"{self.root_dir}/research/uvcc/uvcc-client",
                f"{self.root_dir}/research/uvcc/uvcc-party",
                f"{self.root_dir}/research/uvcc_parallel",
            ]
        )
        return f"PATH={self.home}/.local/bin:$PATH PYTHONPATH={shlex.quote(py_path)}"


def _read_prime_api_key(*, env_name: str, path_opt: Optional[str]) -> str:
    if path_opt is not None:
        p = Path(str(path_opt)).expanduser().resolve()
        return p.read_text(encoding="utf-8").strip()
    v = str(os.environ.get(str(env_name), "")).strip()
    if not v:
        raise RuntimeError(f"missing Prime API key (set {env_name} or pass --prime-api-key-path)")
    return v


def _best_effort_ssh_key_path(path_opt: Optional[str]) -> str:
    if path_opt is not None:
        return str(Path(str(path_opt)).expanduser().resolve())
    # best-effort fallback
    for cand in ("~/.ssh/uvcc_prime_runner_ed25519", "~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
        p = Path(cand).expanduser()
        if p.exists():
            return str(p.resolve())
    raise RuntimeError("missing ssh key (pass --ssh-key-path)")


def main() -> int:
    _add_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--replicas", default=8, type=int, help="DP replicas R (default: 8)")

    ap.add_argument(
        "--providers",
        required=True,
        help="Comma-separated provider types for party0,party1,party2 (must be 3 distinct). Example: hyperstack,crusoecloud,lambdalabs",
    )
    ap.add_argument("--image", default="", help="Requested Prime image (optional; otherwise common-image auto pick)")
    ap.add_argument(
        "--socket",
        default="PCIe",
        help="Prime offer socket filter (default: PCIe). Use 'auto' to not filter by socket (allows heterogeneous sockets across providers).",
    )
    ap.add_argument(
        "--prefer-gpu-types",
        default="",
        help="Optional comma-separated GPU types to rank offers (applies within each provider). Example: A10_24GB,RTX6000Ada_48GB,L40_48GB,A6000_48GB,A100_40GB,A100_80GB",
    )
    ap.add_argument("--gpu-count", default=1, type=int, help="GPUs per pod (must be 1 for v1 DP runner)")
    ap.add_argument("--max-price", default=None, type=float, help="Optional max price for Prime pods")
    ap.add_argument("--keep-pods", default="false", choices=["true", "false"])
    ap.add_argument(
        "--attach-prefix",
        default="",
        help="If set, reuse existing Prime pods whose name starts with this prefix (e.g. uvcc-dp-20251220T222934Z) instead of provisioning new pods.",
    )

    ap.add_argument("--ssh-key-path", default=None)
    ap.add_argument("--prime-api-key-env", default="UVCC_PRIME_API_KEY")
    ap.add_argument("--prime-api-key-path", default="~/.uvcc/prime_api_key.txt")

    ap.add_argument("--job-json", default=None, help="Optional JSON job spec (train_v1)")

    ap.add_argument("--relay-port", default=3000, type=int, help="Preferred internal relay port (default: 3000)")
    ap.add_argument(
        "--relay-lease-s",
        default=120,
        type=int,
        help="Relay lease duration in seconds (default: 120). Increase if you see ack 409 lease_expired under load.",
    )

    ap.add_argument("--dp-master-port", default=29500, type=int, help="Base DP master port per party (p adds +party_id)")
    ap.add_argument("--dp-timeout-s", default=900, type=int, help="torch.distributed init timeout seconds (default: 900)")
    ap.add_argument("--dp-preflight", default="true", choices=["true", "false"], help="If true, run a torch.distributed NCCL connectivity preflight before training (default: true)")
    ap.add_argument("--dp-preflight-timeout-s", default=600, type=int, help="Overall preflight deadline seconds (default: 600)")

    ap.add_argument("--party-timeout-s", default=7200, type=int, help="Overall timeout for parties to finish (default: 7200)")
    ap.add_argument("--run-gpu-tests", default="false", choices=["true", "false"])
    ap.add_argument("--party-log-level", default="info", choices=["quiet", "info", "debug", "trace"])

    # Image may not include torch (e.g. ubuntu_22_cuda_12); optionally install a CUDA wheel.
    ap.add_argument("--ensure-torch", default="true", choices=["true", "false"], help="If true, install torch into venv if missing (default: true)")
    ap.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu124", help="Index URL for torch CUDA wheels (default: cu124)")
    ap.add_argument("--torch-version", default="", help="Optional torch==<ver> pin (default: empty = latest from index)")
    ap.add_argument("--pod-active-timeout-s", default=1800, type=int, help="Timeout waiting for each pod to reach ACTIVE+SSH (default: 1800)")
    ap.add_argument(
        "--pod-active-attempt-timeout-s",
        default=900,
        type=int,
        help="Per-attempt cap when waiting for ACTIVE+SSH (default: 900). Helps slow providers without hanging forever on bad pods.",
    )
    ap.add_argument("--pod-active-poll-s", default=10.0, type=float, help="Poll interval for pod ACTIVE/SSH readiness (default: 10s)")

    args = ap.parse_args()

    if int(args.gpu_count) != 1:
        raise RuntimeError("v1 DP runner requires --gpu-count 1 (one GPU per pod)")
    R = int(args.replicas)
    if R <= 0 or R > 128:
        raise RuntimeError("--replicas must be 1..128")

    out_dir = Path(str(args.out)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = RunLoggerV1(out_dir=out_dir)
    log.log("UVCC DP runner starting")
    log.event("dp_config", replicas=int(R), gpu_count=int(args.gpu_count))

    providers = [str(x).strip() for x in str(args.providers).split(",") if str(x).strip()]
    if len(providers) != 3 or len(set([p.lower() for p in providers])) != 3:
        raise RuntimeError("--providers must be exactly 3 distinct provider types (p0,p1,p2)")
    providers_by_pid = {0: providers[0], 1: providers[1], 2: providers[2]}
    log.event("providers_selected", providers=[providers_by_pid[i] for i in (0, 1, 2)])

    prefer_gpu_types = [str(x).strip() for x in str(args.prefer_gpu_types or "").split(",") if str(x).strip()]
    if prefer_gpu_types:
        log.event("prefer_gpu_types", prefer_gpu_types=list(prefer_gpu_types))

    prime_key = _read_prime_api_key(env_name=str(args.prime_api_key_env), path_opt=str(args.prime_api_key_path) if args.prime_api_key_path else None)
    ssh_key_path = _best_effort_ssh_key_path(args.ssh_key_path)

    from uvcc_client.prime_api import PrimeClientV1, PrimePodSpecV1
    from uvcc_client.ssh_runner import load_private_key_from_file, sftp_get_file, sftp_put_bytes, sftp_put_file, ssh_connect_with_retries, ssh_exec

    from uvcc_parallel.dp_ids import new_sid_job, relay_group_id_for_replica, sid_replica_v1

    prime = PrimeClientV1(api_key=str(prime_key))
    pkey = load_private_key_from_file(str(ssh_key_path))

    # Build bundle.
    bundle_path = out_dir / "uvcc_dp_bundle.tgz"
    _build_uvcc_dp_bundle_tgz(bundle_path)
    log.event("bundle_built", path=str(bundle_path), bytes=int(bundle_path.stat().st_size))

    # Load job config (train_v1 only).
    job_cfg: Dict[str, Any] = {}
    if args.job_json is not None:
        job_cfg = json.loads(Path(str(args.job_json)).expanduser().read_text(encoding="utf-8"))
        if not isinstance(job_cfg, dict):
            raise RuntimeError("job-json must be an object")
    else:
        job_cfg = {"kind": "train_v1", "d": 16, "steps": 2, "seed": 424242, "require_cuda": True, "fxp_frac_bits": 0, "sks_t_checks": 3, "sks_sample_log2": 0}
    if str(job_cfg.get("kind") or "") != "train_v1":
        raise RuntimeError("only job kind train_v1 is supported")
    d_job = int(job_cfg.get("d") or 16)
    steps_job = int(job_cfg.get("steps") or 1)
    seed_job = int(job_cfg.get("seed") or 424242)
    require_cuda_job = bool(job_cfg.get("require_cuda") if "require_cuda" in job_cfg else True)
    fxp_job = int(job_cfg.get("fxp_frac_bits") or 0)
    sks_t_checks_job = int(job_cfg.get("sks_t_checks") or 3)
    sks_sample_log2_job = int(job_cfg.get("sks_sample_log2") or 0)
    log.event(
        "job_config",
        kind="train_v1",
        d=int(d_job),
        steps=int(steps_job),
        seed=int(seed_job),
        require_cuda=bool(require_cuda_job),
        fxp_frac_bits=int(fxp_job),
        sks_t_checks=int(sks_t_checks_job),
        sks_sample_log2=int(sks_sample_log2_job),
    )

    # Conservative completion timeout.
    party_timeout_min_s = max(1800, (int(steps_job) * 360) + 600)
    party_timeout_effective_s = max(int(args.party_timeout_s), int(party_timeout_min_s))
    log.event("party_timeout_effective", party_timeout_s=int(party_timeout_effective_s))

    # Select offers per provider and find a common image across providers.
    want_socket = str(args.socket or "").strip()
    socket_filter: Optional[str] = None if want_socket.lower() == "auto" or want_socket == "" else want_socket
    offers_by_pid: Dict[int, List[Any]] = {}
    img_sets: List[set[str]] = []
    for pid in (0, 1, 2):
        prov = str(providers_by_pid[int(pid)]).strip()
        offers_pid = prime.candidate_offers_v1(
            nodes=1,
            gpu_count_per_node=1,
            provider_type=str(prov),
            socket=str(socket_filter) if socket_filter is not None else None,
            prefer_gpu_types=list(prefer_gpu_types) if prefer_gpu_types else None,
            limit=64,
        )
        want = str(prov).strip().lower()
        offers_pid = [o for o in offers_pid if (o.provider is not None and str(o.provider).strip().lower() == want)]
        if not offers_pid:
            raise RuntimeError(f"no Prime offers for provider_type={prov} (party_id={pid})")
        if bool(require_cuda_job):
            offers_pid = [
                o
                for o in offers_pid
                if (
                    not str(o.gpu_type or "").strip().upper().startswith("CPU")
                    and not str(o.gpu_type or "").strip().upper().endswith("CPU_NODE")
                    and not str(o.cloud_id or "").strip().lower().startswith("cpu-")
                )
            ]
            if not offers_pid:
                raise RuntimeError(f"no GPU offers available for require_cuda=true (provider_type={prov} party_id={pid})")
        offers_by_pid[int(pid)] = list(offers_pid)
        s: set[str] = set()
        for o in offers_pid:
            for im in (o.images or []):
                s.add(str(im).strip().lower())
        img_sets.append(s)

    common_imgs = set.intersection(*img_sets) if img_sets else set()
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
        raise RuntimeError(f"no common Prime image found across providers: {providers} (common_images_count={len(common_imgs)})")
    log.event("prime_common_image_selected", image=str(common_image), providers=[providers_by_pid[i] for i in (0, 1, 2)], common_images_count=int(len(common_imgs)))

    # Pick one offer per party that supports the common image and ideally has capacity >= R.
    chosen_offer_by_pid: Dict[int, Any] = {}
    chosen_img_by_pid: Dict[int, str] = {}
    chosen_dc_by_pid: Dict[int, Optional[str]] = {}
    for pid in (0, 1, 2):
        prov = str(providers_by_pid[int(pid)]).strip()
        offers_pid = offers_by_pid[int(pid)]

        def supports(o: Any) -> bool:
            for im in (o.images or []):
                if str(im).strip().lower() == str(common_image).strip().lower():
                    return True
            return False

        def has_cap(o: Any) -> bool:
            try:
                return int(o.available or 0) >= int(R)
            except Exception:
                return False

        pick = None
        for o in offers_pid:
            if supports(o) and has_cap(o):
                pick = o
                break
        if pick is None:
            for o in offers_pid:
                if supports(o):
                    pick = o
                    break
        if pick is None:
            raise RuntimeError(f"provider {prov} has no offer supporting common image {common_image} (party_id={pid})")

        img_pid = None
        for im in (pick.images or []):
            if str(im).strip().lower() == str(common_image).strip().lower():
                img_pid = str(im).strip()
                break
        if img_pid is None:
            img_pid = str(common_image)

        dc_id_pid = (
            str(
                pick.raw.get("dataCenterId")
                or pick.raw.get("data_center_id")
                or pick.raw.get("dataCenterID")
                or pick.raw.get("dataCenter")
                or pick.raw.get("data_center")
                or pick.raw.get("datacenter")
                or pick.raw.get("datacenter_id")
                or ""
            ).strip()
            or None
        )
        chosen_offer_by_pid[int(pid)] = pick
        chosen_img_by_pid[int(pid)] = str(img_pid)
        chosen_dc_by_pid[int(pid)] = dc_id_pid
        log.event(
            "prime_offer_selected",
            party_id=int(pid),
            provider_type=str(prov),
            cloud_id=str(pick.cloud_id),
            gpu_type=str(pick.gpu_type),
            gpu_count=1,
            socket=str(pick.socket or socket_filter or ""),
            image=str(img_pid),
            offer_region=str(pick.region) if pick is not None else "",
            available=int(pick.available) if pick.available is not None else None,
        )

    # Provision pods: 8 per party/provider.
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log.event("prime_name_prefix", prefix=str(f"uvcc-dp-{ts}"))
    created: List[Tuple[int, int, str]] = []  # (party_id, replica_id, pod_id)
    pods_active: Dict[Tuple[int, int], Any] = {}
    try:
        attach_prefix = str(args.attach_prefix or "").strip()
        if attach_prefix:
            # Attach to existing pods (useful if the local runner dies after provisioning).
            # We still keep the rest of the flow (wait_active -> bootstrap -> relay -> dp_preflight -> training).
            log.event("prime_attach_start", attach_prefix=str(attach_prefix), replicas=int(R))

            url = f"{prime.api_base}/pods/"
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {prime_key}", "Content-Type": "application/json"},
                timeout=60,
            )
            if int(resp.status_code) >= 400:
                raise RuntimeError(f"prime list pods failed ({resp.status_code}): {resp.text}")
            payload = resp.json()
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise RuntimeError(f"prime list pods returned unexpected json: {type(payload)}")

            found: Dict[Tuple[int, int], str] = {}
            found_meta: List[Dict[str, Any]] = []
            pat = re.compile(r"-p(?P<pid>\d+)-r(?P<rid>\d+)-")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "")
                if not name.startswith(str(attach_prefix)):
                    continue
                m = pat.search(name)
                if not m:
                    continue
                pid = int(m.group("pid"))
                rid = int(m.group("rid"))
                pod_id = str(row.get("id") or row.get("podId") or row.get("pod_id") or "").strip()
                if not pod_id:
                    continue
                prov_row = row.get("provider") if isinstance(row.get("provider"), dict) else None
                ptype = str(row.get("providerType") or (prov_row.get("type") if prov_row else "") or "").strip().lower()
                want = str(providers_by_pid.get(int(pid), "")).strip().lower()
                if want and ptype and ptype != want:
                    raise RuntimeError(f"attach-prefix provider mismatch for {name}: want={want} got={ptype}")
                if int(pid) not in (0, 1, 2):
                    continue
                if int(rid) < 0 or int(rid) >= int(R):
                    continue
                key = (int(pid), int(rid))
                found[key] = str(pod_id)
                found_meta.append({"party_id": int(pid), "replica_id": int(rid), "pod_id": str(pod_id), "name": str(name), "provider_type": str(ptype)})

            expected = 3 * int(R)
            if len(found) != expected:
                raise RuntimeError(f"attach-prefix did not find expected pods: found={len(found)} expected={expected} prefix={attach_prefix}")

            created = [(pid, rid, found[(pid, rid)]) for pid in (0, 1, 2) for rid in range(int(R))]
            log.event(
                "prime_pods_attached",
                attach_prefix=str(attach_prefix),
                count=int(len(created)),
                pods=sorted(found_meta, key=lambda x: (x["party_id"], x["replica_id"])),
            )

        def _create_pod_retry(*, pid: int, rid: int, spec: PrimePodSpecV1, base_name: str) -> str:
            """
            Prime pod provisioning can return transient 5xx (e.g. 500: Error during pod provisioning).
            Retry a few times with backoff and a unique name per attempt.
            """
            last_err = ""
            for attempt in range(1, 6):
                try:
                    spec2 = dataclasses.replace(spec, name=f"{str(base_name)}-a{int(attempt)}")
                    return str(prime.create_pod(spec2))
                except Exception as exc:
                    last_err = str(exc) or exc.__class__.__name__
                    log.event(
                        "prime_create_pod_failed",
                        party_id=int(pid),
                        replica_id=int(rid),
                        attempt=int(attempt),
                        provider_type=str(spec.provider_type),
                        cloud_id=str(spec.cloud_id),
                        gpu_type=str(spec.gpu_type),
                        image=str(spec.image),
                        socket=str(spec.socket),
                        error=str(last_err),
                    )
                    time.sleep(min(20.0, 2.0 + attempt * 2.5))
            raise RuntimeError(f"unable to provision pod after retries (party_id={pid} replica_id={rid}): {last_err}")

        def _dc_id_for_offer(offer: Any) -> Optional[str]:
            try:
                raw = offer.raw if hasattr(offer, "raw") else {}
                if not isinstance(raw, dict):
                    raw = {}
                dc = (
                    str(
                        raw.get("dataCenterId")
                        or raw.get("data_center_id")
                        or raw.get("dataCenterID")
                        or raw.get("dataCenter")
                        or raw.get("data_center")
                        or raw.get("datacenter")
                        or raw.get("datacenter_id")
                        or ""
                    ).strip()
                    or None
                )
                return dc
            except Exception:
                return None

        def _image_for_offer(offer: Any, *, common_image: str) -> Optional[str]:
            try:
                for im in (offer.images or []):
                    if str(im).strip().lower() == str(common_image).strip().lower():
                        return str(im).strip()
            except Exception:
                pass
            return None

        def _create_one(*, pid: int, rid: int) -> str:
            prov = str(providers_by_pid[int(pid)]).strip()
            # Try candidate offers in order; if one offer systematically returns 5xx, fall back to another.
            offers_pid = list(offers_by_pid[int(pid)])
            for offer_idx, offer in enumerate(offers_pid):
                img_offer = _image_for_offer(offer, common_image=str(common_image))
                if img_offer is None:
                    continue
                dc = _dc_id_for_offer(offer)
                sock = str(getattr(offer, "socket", "") or "").strip() or (str(socket_filter) if socket_filter is not None else "PCIe")
                base_name = f"uvcc-dp-{ts}-p{int(pid)}-r{int(rid)}-o{int(offer_idx)}"
                spec = PrimePodSpecV1(
                    cloud_id=str(offer.cloud_id),
                    gpu_type=str(offer.gpu_type),
                    gpu_count=1,
                    socket=str(sock),
                    image=str(img_offer),
                    name=str(base_name),
                    provider_type=str(prov),
                    data_center_id=dc,
                    max_price=float(args.max_price) if args.max_price is not None else None,
                )
                try:
                    pod_id = _create_pod_retry(pid=int(pid), rid=int(rid), spec=spec, base_name=str(base_name))
                    log.event(
                        "prime_pod_created",
                        party_id=int(pid),
                        replica_id=int(rid),
                        pod_id=str(pod_id),
                        provider_type=str(prov),
                        cloud_id=str(offer.cloud_id),
                        gpu_type=str(offer.gpu_type),
                        image=str(img_offer),
                        offer_idx=int(offer_idx),
                    )
                    return str(pod_id)
                except Exception as exc:
                    log.event(
                        "prime_pod_create_offer_failed",
                        party_id=int(pid),
                        replica_id=int(rid),
                        provider_type=str(prov),
                        cloud_id=str(offer.cloud_id),
                        gpu_type=str(offer.gpu_type),
                        image=str(img_offer),
                        offer_idx=int(offer_idx),
                        error=str(exc),
                    )
                    continue
            raise RuntimeError(f"unable to provision pod with any offer (party_id={pid} replica_id={rid})")

        if not created:
            # Interleave provisioning across parties per replica to avoid creating many pods and then failing on a later provider.
            for rid in range(int(R)):
                for pid in (0, 1, 2):
                    pod_id = _create_one(pid=int(pid), rid=int(rid))
                    created.append((int(pid), int(rid), str(pod_id)))
            log.event("prime_pods_created", count=int(len(created)))

        # Wait ACTIVE+SSH for all, with auto-replace for stuck pods (e.g. provider returns ACTIVE but never assigns ip/sshConnection).
        for idx, (pid, rid, pod_id) in enumerate(list(created)):
            cur_pod_id = str(pod_id)
            last_err = ""
            for attempt in range(1, 4):
                try:
                    pod = prime.wait_active(
                        str(cur_pod_id),
                        # Avoid waiting the full timeout on a pod that is "ACTIVE but never ssh-ready".
                        timeout_s=min(int(args.pod_active_timeout_s), int(args.pod_active_attempt_timeout_s)),
                        poll_s=float(args.pod_active_poll_s),
                    )
                    pods_active[(int(pid), int(rid))] = pod
                    created[idx] = (int(pid), int(rid), str(cur_pod_id))
                    last_err = ""
                    break
                except Exception as exc:
                    last_err = str(exc) or exc.__class__.__name__
                    log.event(
                        "prime_wait_active_failed",
                        party_id=int(pid),
                        replica_id=int(rid),
                        pod_id=str(cur_pod_id),
                        attempt=int(attempt),
                        error=str(last_err),
                    )
                    # Best-effort delete of the stuck pod before recreating.
                    try:
                        prime.delete_pod(str(cur_pod_id))
                        log.event("pod_deleted", party_id=int(pid), replica_id=int(rid), pod_id=str(cur_pod_id), attempt=int(attempt), stage="wait_active_replace")
                    except Exception as de:
                        log.event("pod_delete_failed", party_id=int(pid), replica_id=int(rid), pod_id=str(cur_pod_id), attempt=int(attempt), error=str(de))

                    # Recreate pod for this slot unless we've exhausted attempts.
                    if attempt >= 3:
                        break
                    new_id = _create_one(pid=int(pid), rid=int(rid))
                    log.event(
                        "prime_pod_replaced",
                        party_id=int(pid),
                        replica_id=int(rid),
                        old_pod_id=str(cur_pod_id),
                        new_pod_id=str(new_id),
                        attempt=int(attempt),
                    )
                    cur_pod_id = str(new_id)
            if last_err:
                raise RuntimeError(f"pod never became ssh-ready (party_id={pid} replica_id={rid}): {last_err}")
        log.event(
            "prime_pods_active",
            pods=[{"party_id": pid, "replica_id": rid, "pod_id": pod_id} for (pid, rid, pod_id) in created],
            provider_types=[providers_by_pid[i] for i in (0, 1, 2)],
        )

        # Bootstrap all pods.
        nodes: List[DPRemoteNodeV1] = []

        def _bootstrap_one(*, pid: int, rid: int, pod: Any) -> DPRemoteNodeV1:
            ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
            try:
                code, out, err = ssh_exec(ssh, "bash -lc 'echo -n $HOME'", timeout_s=30)
                if code != 0 or not out:
                    raise RuntimeError(f"failed to get remote home: {err}")
                home = out.strip()
                root_dir = f"{home}/uvcc"

                # GPU snapshot
                smi_cmd = "bash -lc 'nvidia-smi -L || true; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true'"
                code_smi, out_smi, err_smi = ssh_exec(ssh, smi_cmd, timeout_s=60)
                smi_local = out_dir / f"node_p{int(pid)}_r{int(rid)}_nvidia_smi.txt"
                smi_local.write_text(out_smi + err_smi, encoding="utf-8")
                log.event("node_gpu_snapshot", party_id=int(pid), replica_id=int(rid), ssh_host=str(pod.ssh_host), path=str(smi_local), exit_code=int(code_smi))

                remote_bundle = f"{home}/uvcc_dp_bundle.tgz"
                sftp_put_file(ssh, local_path=str(bundle_path), remote_path=str(remote_bundle), mode=0o600)

                t_boot0 = time.monotonic()
                torch_pkg = "torch"
                if str(args.torch_version or "").strip():
                    torch_pkg = f"torch=={str(args.torch_version).strip()}"
                ensure_torch = str(args.ensure_torch).lower() == "true"
                torch_index_url = str(args.torch_index_url or "").strip() or "https://download.pytorch.org/whl/cu124"
                torch_install_snip = ""
                if ensure_torch:
                    torch_install_snip = (
                        f" if ! python3 -c {shlex.quote('import torch')} >/dev/null 2>&1; then"
                        f"   python3 -m pip install --user --no-cache-dir --index-url {shlex.quote(torch_index_url)} {shlex.quote(torch_pkg)} >/dev/null;"
                        " fi;"
                    )
                bootstrap_cmd = (
                    "set -euo pipefail;"
                    f" mkdir -p {shlex.quote(root_dir)};"
                    f" tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(root_dir)};"
                    # Prefer system python3 + user-site packages (avoids venv/ensurepip/apt-get issues across providers).
                    " if ! command -v python3 >/dev/null 2>&1; then echo 'missing_python3' >&2; exit 1; fi;"
                    " if ! python3 -m pip --version >/dev/null 2>&1; then"
                    "   if python3 -c 'import ensurepip' >/dev/null 2>&1; then python3 -m ensurepip --upgrade >/dev/null 2>&1 || true; fi;"
                    " fi;"
                    " if ! python3 -m pip --version >/dev/null 2>&1; then"
                    "   if command -v apt-get >/dev/null 2>&1; then"
                    "     export DEBIAN_FRONTEND=noninteractive;"
                    "     sudo -n apt-get update -y >/dev/null;"
                    "     sudo -n apt-get install -y --no-install-recommends python3-pip ca-certificates >/dev/null;"
                    "   fi;"
                    " fi;"
                    " if ! python3 -m pip --version >/dev/null 2>&1; then echo 'missing_pip' >&2; exit 1; fi;"
                    " export PATH=\"$HOME/.local/bin:$PATH\";"
                    f" python3 -m pip install --user --no-cache-dir -r {shlex.quote(root_dir + '/research/uvcc/requirements-uvcc-base.txt')} >/dev/null;"
                    + str(torch_install_snip)
                    + f" python3 -c {shlex.quote('import torch; print(torch.__version__); print(bool(torch.cuda.is_available()))')};"
                )
                code2, out2, err2 = ssh_exec(ssh, f"bash -lc {shlex.quote(bootstrap_cmd)}", timeout_s=3600)
                boot_local = out_dir / f"node_p{int(pid)}_r{int(rid)}_bootstrap.log"
                boot_local.write_text(out2 + err2, encoding="utf-8")
                if code2 != 0:
                    raise RuntimeError(f"bootstrap failed:\n{out2}\n{err2}")
                log.event("node_bootstrap_done", party_id=int(pid), replica_id=int(rid), ssh_host=str(pod.ssh_host), t_s=round(time.monotonic() - t_boot0, 3), log_path=str(boot_local))

                if str(args.run_gpu_tests).lower() == "true":
                    envp = f"PATH={home}/.local/bin:$PATH PYTHONPATH={root_dir}/research/uvcc/uvcc-party:{root_dir}/research/uvcc/uvcc-client"
                    test_cmd = (
                        f"set -euo pipefail; {envp} python3 -m pytest -q "
                        f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_dpf_dcf_kernels.py "
                        f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_gf2_and_a2b_kernels.py "
                        f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_trunc_apply_u64.py "
                        f"{root_dir}/research/uvcc/uvcc-party/tests/test_cuda_matmul_u64.py "
                    )
                    t_gpu0 = time.monotonic()
                    code3, out3, err3 = ssh_exec(ssh, f"bash -lc {shlex.quote(test_cmd)}", timeout_s=3600)
                    gpu_local = out_dir / f"node_p{int(pid)}_r{int(rid)}_gpu_tests.log"
                    gpu_local.write_text(out3 + err3, encoding="utf-8")
                    if code3 != 0:
                        raise RuntimeError(f"gpu tests failed:\n{out3}\n{err3}")
                    log.event("node_gpu_tests_done", party_id=int(pid), replica_id=int(rid), ssh_host=str(pod.ssh_host), t_s=round(time.monotonic() - t_gpu0, 3), log_path=str(gpu_local))

                return DPRemoteNodeV1(
                    party_id=int(pid),
                    replica_id=int(rid),
                    pod_id=str(pod.pod_id),
                    provider_type=str((pod.status_row or {}).get("providerType") or (pod.status_row or {}).get("provider") or ""),
                    ssh_user=str(pod.ssh_user),
                    ssh_host=str(pod.ssh_host),
                    ssh_port=int(pod.ssh_port),
                    home=str(home),
                    root_dir=str(root_dir),
                    python_bin="python3",
                )
            finally:
                ssh.close()

        # Bootstrap (and optional GPU tests) in parallel to keep wall-clock time bounded
        # even as replicas scale (R=8 -> 24 pods). Logging is thread-safe.
        bootstrap_items = sorted(pods_active.items(), key=lambda x: (x[0][0], x[0][1]))
        max_workers = min(8, max(1, len(bootstrap_items)))
        log.event("bootstrap_parallel_start", workers=int(max_workers), count=int(len(bootstrap_items)))
        nodes_tmp: List[DPRemoteNodeV1] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_bootstrap_one, pid=int(pid), rid=int(rid), pod=pod): (int(pid), int(rid))
                for (pid, rid), pod in bootstrap_items
            }
            for fut in as_completed(list(futs.keys())):
                pid, rid = futs[fut]
                try:
                    nodes_tmp.append(fut.result())
                except Exception as exc:
                    log.event("node_bootstrap_failed", party_id=int(pid), replica_id=int(rid), error=str(exc))
                    for f2 in futs.keys():
                        try:
                            f2.cancel()
                        except Exception:
                            pass
                    raise
        nodes = sorted(nodes_tmp, key=lambda n: (int(n.party_id), int(n.replica_id)))
        log.event("bootstrap_parallel_done", count=int(len(nodes)))

        # Stable per-party identity keys: one key per party, uploaded to all pods of that party.
        from uvcc_client.party_identity import load_or_create_party_privkey32_v1, party_identity_from_privkey_v1

        party_keys_dir = out_dir / "private_keep" / "party_keys"
        party_keys_dir.mkdir(parents=True, exist_ok=True)
        party_priv_by_pid: Dict[int, bytes] = {}
        for pid in (0, 1, 2):
            kp = party_keys_dir / f"party_privkey_p{int(pid)}.hex"
            priv = load_or_create_party_privkey32_v1(path=str(kp))
            party_priv_by_pid[int(pid)] = bytes(priv)
        party_identities_local = []
        for pid in (0, 1, 2):
            ident = party_identity_from_privkey_v1(party_id=int(pid), privkey32=party_priv_by_pid[int(pid)])
            party_identities_local.append({"party_id": int(pid), "address": "0x" + bytes(ident.address20).hex()})
        log.event("party_identity_keys_ready", parties=party_identities_local, key_dir=str(party_keys_dir))

        for n in nodes:
            ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
            try:
                # Some provider images can have a minimal home filesystem layout; be explicit.
                code_mk, out_mk, err_mk = ssh_exec(ssh, f"bash -lc 'mkdir -p {shlex.quote(str(n.root_dir))}'", timeout_s=30)
                if code_mk != 0:
                    raise RuntimeError(f"failed to ensure party key dir exists: {out_mk}{err_mk}")
                remote_key = f"{n.root_dir}/party_privkey.hex"
                priv = party_priv_by_pid[int(n.party_id)]
                sftp_put_bytes(ssh, remote_path=str(remote_key), data=("0x" + priv.hex()).encode("utf-8"), mode=0o600)
                log.event("party_identity_key_uploaded", party_id=int(n.party_id), replica_id=int(n.replica_id), ssh_host=str(n.ssh_host), path=str(remote_key))
            except Exception as exc:
                log.event("party_identity_key_upload_failed", party_id=int(n.party_id), replica_id=int(n.replica_id), ssh_host=str(n.ssh_host), error=str(exc))
                raise
            finally:
                ssh.close()

        # Start relay on party0 replica0.
        relay_node = next(nn for nn in nodes if int(nn.party_id) == 0 and int(nn.replica_id) == 0)
        relay_token = base64.b64encode(os.urandom(24)).decode("ascii")
        relay_host = str(relay_node.ssh_host)
        ca_pem, srv_pem, srv_key = _tls_ca_and_server_cert(host_or_ip=str(relay_host))

        ca_remote0 = f"{relay_node.root_dir}/relay_ca.pem"
        srv_remote0 = f"{relay_node.root_dir}/relay_srv.pem"
        key_remote0 = f"{relay_node.root_dir}/relay_srv.key"
        token_remote0 = f"{relay_node.root_dir}/relay_token.txt"

        # Upload CA + token to all nodes; server cert/key only to relay host.
        for n in nodes:
            ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
            try:
                ca_path = f"{n.root_dir}/relay_ca.pem"
                token_path = f"{n.root_dir}/relay_token.txt"
                sftp_put_bytes(ssh, remote_path=ca_path, data=ca_pem, mode=0o644)
                sftp_put_bytes(ssh, remote_path=token_path, data=relay_token.encode("utf-8"), mode=0o600)
                if int(n.party_id) == 0 and int(n.replica_id) == 0:
                    sftp_put_bytes(ssh, remote_path=srv_remote0, data=srv_pem, mode=0o644)
                    sftp_put_bytes(ssh, remote_path=key_remote0, data=srv_key, mode=0o600)
            finally:
                ssh.close()

        # Pick a relay port (reuse robust port mapping strategy from the 3-node runner in simplified form).
        relay_requested_internal = int(args.relay_port)
        relay_candidates: List[Tuple[int, int, Optional[Dict[str, Any]]]] = [(int(relay_requested_internal), int(relay_requested_internal), None)]
        try:
            pod0 = prime.pod_get(str(relay_node.pod_id))
            pm = pod0.get("primePortMapping")
            pm_list = pm if isinstance(pm, list) else []
            by_internal: Dict[int, Tuple[int, Dict[str, Any]]] = {}
            for m in pm_list:
                if not isinstance(m, dict):
                    continue
                try:
                    i = int(m.get("internal") or 0)
                    e = int(m.get("external") or 0)
                except Exception:
                    continue
                if i > 0 and e > 0:
                    by_internal[i] = (e, dict(m))

            def _mapping_usable(mm: Dict[str, Any]) -> bool:
                used_by = str(mm.get("usedBy") or "").strip().upper()
                desc = str(mm.get("description") or "").strip().upper()
                # Never select SSH ports.
                if used_by == "SSH" or "SSH" in desc:
                    return False
                return True

            def _port_ok(pi: int) -> bool:
                # avoid common restricted ports and the ssh port itself
                if pi in (0, 22, int(relay_node.ssh_port)):
                    return False
                return 1 <= int(pi) <= 65535

            pref_ports: List[int] = []
            for p in (relay_requested_internal, 3000, 8000, 54692, 19123):
                if int(p) not in pref_ports:
                    pref_ports.append(int(p))

            cand: List[Tuple[int, int, Optional[Dict[str, Any]]]] = []
            seen: set[int] = set()

            def add(pi: int, pe: int, mm: Optional[Dict[str, Any]]) -> None:
                if int(pi) in seen:
                    return
                if not _port_ok(int(pi)):
                    return
                cand.append((int(pi), int(pe), dict(mm) if isinstance(mm, dict) else None))
                seen.add(int(pi))

            for pi in pref_ports:
                if int(pi) in by_internal:
                    e, m = by_internal[int(pi)]
                    if _mapping_usable(dict(m)):
                        add(int(pi), int(e), dict(m))
                        continue
                add(int(pi), int(pi), None)

            for i, (e, m) in by_internal.items():
                if not _mapping_usable(dict(m)):
                    continue
                add(int(i), int(e), dict(m))

            if cand:
                relay_candidates = cand
        except Exception:
            relay_candidates = [(int(relay_requested_internal), int(relay_requested_internal), None)]

        def _remote_healthz(*, n: DPRemoteNodeV1, url: str, ca_path: str) -> None:
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
                    raise RuntimeError(f"healthz failed for {url}:\n{out}\n{err}")
            finally:
                ssh.close()

        relay_py = f"{relay_node.root_dir}/research/uvcc/uvcc-relay/relay_server.py"
        relay_db = f"{relay_node.root_dir}/relay.sqlite"
        relay_log = f"{relay_node.root_dir}/relay.log"
        relay_pid = f"{relay_node.root_dir}/relay.pid"

        relay_port_internal = int(args.relay_port)
        relay_port_external = int(args.relay_port)
        relay_url_public = ""
        relay_url_local = ""
        last_relay_err = ""

        probe_node = next(nn for nn in nodes if int(nn.party_id) == 1 and int(nn.replica_id) == 0)
        for cand_internal, cand_external, cand_mapping in list(relay_candidates):
            relay_port_internal = int(cand_internal)
            relay_port_external = int(cand_external)
            relay_url_public = f"https://{relay_host}:{int(relay_port_external)}"
            relay_url_local = f"https://127.0.0.1:{int(relay_port_internal)}"
            log.event(
                "relay_candidate_try",
                relay_port_internal=int(relay_port_internal),
                relay_port_external=int(relay_port_external),
                relay_port_mapping=dict(cand_mapping) if isinstance(cand_mapping, dict) else None,
                relay_url_public=str(relay_url_public),
                relay_url_local=str(relay_url_local),
            )

            ssh0 = ssh_connect_with_retries(hostname=relay_node.ssh_host, port=relay_node.ssh_port, username=relay_node.ssh_user, pkey=pkey, timeout_s=900)
            try:
                start_cmd = (
                    "set -euo pipefail;"
                    f" rm -f {shlex.quote(relay_log)};"
                    f" if [ -f {shlex.quote(relay_pid)} ]; then"
                    f"   (kill -9 $(cat {shlex.quote(relay_pid)}) >/dev/null 2>&1 || true);"
                    f"   rm -f {shlex.quote(relay_pid)};"
                    " fi;"
                    f" PYTHONUNBUFFERED=1 nohup {shlex.quote(relay_node.venv_python())} {shlex.quote(relay_py)}"
                    f" --host 0.0.0.0 --port {int(relay_port_internal)} --db {shlex.quote(relay_db)} --lease-s {int(args.relay_lease_s)}"
                    " --require-token true"
                    f" --token-file {shlex.quote(token_remote0)}"
                    f" --tls-cert {shlex.quote(srv_remote0)} --tls-key {shlex.quote(key_remote0)}"
                    f" > {shlex.quote(relay_log)} 2>&1 & echo $! > {shlex.quote(relay_pid)};"
                    " sleep 0.2;"
                    f" PID=$(cat {shlex.quote(relay_pid)});"
                    " kill -0 $PID >/dev/null 2>&1 || (echo 'relay_process_exited_early' >&2; tail -n 200 "
                    f"{shlex.quote(relay_log)} >&2 || true; exit 1);"
                )
                code, out, err = ssh_exec(ssh0, f"bash -lc {shlex.quote(start_cmd)}", timeout_s=30)
                if code != 0:
                    last_relay_err = f"relay start failed:\n{out}\n{err}"
                    log.event("relay_candidate_failed", stage="start", relay_port_internal=int(relay_port_internal), relay_port_external=int(relay_port_external), error=_trim(last_relay_err))
                    continue
            finally:
                ssh0.close()

            last0 = ""
            for _ in range(60):
                try:
                    _remote_healthz(n=relay_node, url=str(relay_url_local), ca_path=str(ca_remote0))
                    last0 = ""
                    break
                except Exception as exc:
                    last0 = str(exc)
                    time.sleep(0.5)
            if last0:
                last_relay_err = f"relay loopback health failed: {last0}"
                log.event("relay_candidate_failed", stage="loopback_health", relay_port_internal=int(relay_port_internal), relay_port_external=int(relay_port_external), error=_trim(last0))
                continue

            last1 = ""
            for _ in range(60):
                try:
                    _remote_healthz(n=probe_node, url=str(relay_url_public), ca_path=str(f"{probe_node.root_dir}/relay_ca.pem"))
                    last1 = ""
                    break
                except Exception as exc:
                    last1 = str(exc)
                    time.sleep(0.5)
            if last1:
                last_relay_err = f"relay public health failed: {last1}"
                log.event("relay_candidate_failed", stage="public_health", relay_port_internal=int(relay_port_internal), relay_port_external=int(relay_port_external), error=_trim(last1))
                continue

            # success
            break
        else:
            raise RuntimeError(f"unable to start relay: {last_relay_err}")

        log.event("privacy_relay_started", relay_url_public=str(relay_url_public), relay_url_local=str(relay_url_local), relay_host=str(relay_host))

        # DP job ids: create a fresh sid_job, and derive sid_rep[r].
        sid_job = new_sid_job()
        log.event("sid_job_selected", sid_job_hex=_hex32(bytes(sid_job)))

        # Prepare per-replica inputs (secret shared).
        import torch

        from uvcc_party.rss import make_rss_arith_u64_triple
        from uvcc_party.tcf import tcf_gen_v1

        # v1 job_id32 is not on-chain here; we just need a stable 32B identifier for transcripts and proof bundles.
        # Use sha256 to avoid extra deps.
        job_id32 = _sha256(b"UVCC.dp.jobid.v1\0" + os.urandom(32))
        log.event("job_id_selected", job_id_hex=_hex32(job_id32))

        d = int(d_job)
        gen0 = torch.Generator(device="cpu").manual_seed(int(seed_job))
        X_pub = torch.eye(d, dtype=torch.int64)
        loW = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen0)
        hiW = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen0)
        W_pub = (hiW << 32) | loW

        def enc_u64(t: torch.Tensor) -> str:
            out = bytearray()
            for v in t.contiguous().view(-1).tolist():
                out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
            return base64.b64encode(bytes(out)).decode("ascii")

        # Precompute per-replica inputs and upload.
        sid_rep_by_r: Dict[int, bytes] = {}
        for rid in range(int(R)):
            sid_rep = sid_replica_v1(sid_job=bytes(sid_job), replica_id=int(rid))
            sid_rep_by_r[int(rid)] = bytes(sid_rep)

        base_seed32 = os.urandom(32)
        for rid in range(int(R)):
            gen_r = torch.Generator(device="cpu").manual_seed(int(seed_job) + 10_000 * int(rid))
            loY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen_r)
            hiY = torch.randint(0, 2**32, (d, d), dtype=torch.int64, generator=gen_r)
            Y_pub = (hiY << 32) | loY

            # Shares (CPU) for this replica.
            X0, X1, X2 = make_rss_arith_u64_triple(x_pub=X_pub, generator=gen_r, device=torch.device("cpu"))
            Y0, Y1, Y2 = make_rss_arith_u64_triple(x_pub=Y_pub, generator=gen_r, device=torch.device("cpu"))
            W0, W1, W2 = make_rss_arith_u64_triple(x_pub=W_pub, generator=gen_r, device=torch.device("cpu"))

            # Per-replica TCF keys (independent triangles).
            ms = _sha256(b"UVCC.dp.master_seed.v1\0" + base_seed32 + int(rid).to_bytes(4, "little", signed=False))
            k0, k1, k2 = tcf_gen_v1(master_seed32=bytes(ms), sid=bytes(sid_rep_by_r[int(rid)]))
            tcf_keys = {0: k0, 1: k1, 2: k2}

            shares = {0: (X0, Y0, W0), 1: (X1, Y1, W1), 2: (X2, Y2, W2)}
            for pid in (0, 1, 2):
                tk = tcf_keys[int(pid)]
                Xs, Ys, Ws = shares[int(pid)]
                inp = {
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
                    "W_lo_b64": enc_u64(Ws.lo),
                    "W_hi_b64": enc_u64(Ws.hi),
                }
                inp_bytes = (json.dumps(inp, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                inp_hash32_hex = "0x" + hashlib.sha256(b"uvcc.inputs.v1\0" + inp_bytes).hexdigest()

                # Upload to the matching pod (pid,rid).
                n = next(nn for nn in nodes if int(nn.party_id) == int(pid) and int(nn.replica_id) == int(rid))
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    remote_inp = f"{n.root_dir}/inputs_p{int(pid)}_r{int(rid)}.json"
                    sftp_put_bytes(ssh, remote_path=remote_inp, data=inp_bytes, mode=0o600)
                finally:
                    ssh.close()
                log.event("party_inputs_uploaded", party_id=int(pid), replica_id=int(rid), ssh_host=str(n.ssh_host), remote_path=str(remote_inp), bytes=int(len(inp_bytes)), inputs_hash32_hex=str(inp_hash32_hex))

        # DP plane config: one NCCL group per party, with master at replica0 of that party.
        dp_master_by_pid: Dict[int, Tuple[str, int]] = {}

        def _pick_dp_master_port(*, master_pod_id: str, preferred_port: int) -> Tuple[int, str]:
            """
            Pick a DP rendezvous port that other replicas can reach.

            - If the provider exposes an all-ports mapping (primePortMapping internal='*', external='*'),
              we can safely use preferred_port.
            - Otherwise (hosted/NAT providers like runpod), pick a port mapping entry where internal==external
              and is not SSH/Jupyter.
            """
            try:
                full = prime.pod_get(str(master_pod_id))
                row = full.get("data") if isinstance(full, dict) and isinstance(full.get("data"), dict) else full
                pm = row.get("primePortMapping") or row.get("portMappings") or []
                pm_list = pm if isinstance(pm, list) else []

                def _as_int(x: Any) -> Optional[int]:
                    try:
                        if x is None:
                            return None
                        if isinstance(x, bool):
                            return int(x)
                        if isinstance(x, int):
                            return int(x)
                        t = str(x).strip()
                        if not t:
                            return None
                        return int(t, 10)
                    except Exception:
                        return None

                # Wildcard means "all other ports", so preferred_port should work.
                for m in pm_list:
                    if not isinstance(m, dict):
                        continue
                    if str(m.get("internal")).strip() == "*" and str(m.get("external")).strip() == "*":
                        return int(preferred_port), "wildcard_all_ports"

                # Otherwise, find internal==external non-SSH/Jupyter ports.
                for m in pm_list:
                    if not isinstance(m, dict):
                        continue
                    used_by = str(m.get("usedBy") or "").strip().upper()
                    desc = str(m.get("description") or "").strip().upper()
                    if used_by in {"SSH", "JUPYTER_NOTEBOOK"} or "SSH" in desc or "JUPYTER" in desc:
                        continue
                    pi = _as_int(m.get("internal"))
                    pe = _as_int(m.get("external"))
                    if pi is None or pe is None:
                        continue
                    if int(pi) != int(pe):
                        continue
                    if int(pi) in (22, 8888, 1234):
                        continue
                    if int(pi) <= 0 or int(pi) > 65535:
                        continue
                    return int(pi), "primePortMapping_internal_eq_external"
            except Exception:
                pass
            return int(preferred_port), "fallback_preferred"

        for pid in (0, 1, 2):
            master_node = next(nn for nn in nodes if int(nn.party_id) == int(pid) and int(nn.replica_id) == 0)
            preferred = int(args.dp_master_port) + int(pid)
            port, reason = _pick_dp_master_port(master_pod_id=str(master_node.pod_id), preferred_port=int(preferred))
            dp_master_by_pid[int(pid)] = (str(master_node.ssh_host), int(port))
            log.event("dp_master_port_selected", party_id=int(pid), addr=str(master_node.ssh_host), port=int(port), reason=str(reason))
        log.event("dp_masters_selected", masters=[{"party_id": pid, "addr": dp_master_by_pid[pid][0], "port": dp_master_by_pid[pid][1]} for pid in (0, 1, 2)])

        # Write a stable layout manifest (safe to share).
        layout = {
            "generated_ts": _now_iso_utc(),
            "replicas": int(R),
            "providers_by_party": {str(k): str(v) for k, v in providers_by_pid.items()},
            "relay": {"url_public": str(relay_url_public), "url_local": str(relay_url_local), "host": str(relay_host)},
            "dp_masters": [{"party_id": int(pid), "addr": dp_master_by_pid[int(pid)][0], "port": dp_master_by_pid[int(pid)][1]} for pid in (0, 1, 2)],
            "nodes": [
                {
                    "party_id": int(n.party_id),
                    "replica_id": int(n.replica_id),
                    "provider_type": str(n.provider_type),
                    "pod_id": str(n.pod_id),
                    "ssh_user": str(n.ssh_user),
                    "ssh_host": str(n.ssh_host),
                    "ssh_port": int(n.ssh_port),
                }
                for n in sorted(nodes, key=lambda x: (int(x.party_id), int(x.replica_id)))
            ],
        }
        layout_path = out_dir / "dp_layout.json"
        layout_path.write_text(json.dumps(layout, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        log.event("dp_layout_written", path=str(layout_path), node_count=int(len(nodes)))

        # Explicit DP NCCL connectivity preflight (staged smoke gate).
        if str(args.dp_preflight).lower() == "true":
            log.log("Running DP NCCL preflight (torch.distributed init + NCCL all_reduce on CUDA tensor)...")
            preflight_nodes = list(nodes)
            # Launch preflight as background processes that write ok.txt on success.
            for n in preflight_nodes:
                ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    out_remote = f"{n.root_dir}/out_dp_preflight/p{int(n.party_id)}/r{int(n.replica_id)}"
                    log_path = f"{out_remote}/preflight.log"
                    pid_path = f"{out_remote}/preflight.pid"
                    ok_path = f"{out_remote}/ok.txt"
                    dp_master_addr, dp_master_port = dp_master_by_pid[int(n.party_id)]
                    py = (
                        "import datetime,os,torch,torch.distributed as dist;"
                        "torch.cuda.set_device(0);"
                        "rank=int(os.environ['UVCC_DP_RANK']);"
                        "ws=int(os.environ['UVCC_DP_WORLD_SIZE']);"
                        "ma=os.environ['UVCC_DP_MASTER_ADDR'];"
                        "mp=int(os.environ['UVCC_DP_MASTER_PORT']);"
                        "ok=os.environ['UVCC_DP_OK_PATH'];"
                        "dist.init_process_group(backend='nccl',init_method=f'tcp://{ma}:{mp}',world_size=ws,rank=rank,timeout=datetime.timedelta(seconds=int(os.environ.get('UVCC_DP_TIMEOUT_S','900'))));"
                        # Use an explicit CUDA tensor op (matches what training needs) vs barrier(), which can hang
                        # if rank->device mapping is ambiguous on some provider images.
                        "t=torch.tensor([rank+1],device='cuda',dtype=torch.int64);"
                        "dist.all_reduce(t,op=dist.ReduceOp.SUM);"
                        "dist.destroy_process_group();"
                        "open(ok,'w',encoding='utf-8').write(f'ok sum={int(t.item())} ws={ws} rank={rank}\\n');"
                    )
                    # Ensure NCCL uses a routable NIC on heterogeneous provider images and avoids IB paths that
                    # often misbehave on commodity cloud networking.
                    nccl_env = (
                        "NCCL_IB_DISABLE=1 "
                        "NCCL_SOCKET_FAMILY=AF_INET "
                        # Use NCCL exclude-mode so we don't hardcode per-provider NIC names.
                        # This avoids failures like "Bootstrap : no socket interface found" on some images.
                        "NCCL_SOCKET_IFNAME=^lo,docker0 "
                        "NCCL_DEBUG=INFO "
                        "NCCL_DEBUG_SUBSYS=NET "
                    )
                    inner = (
                        f"{n.env_prefix()} {nccl_env}"
                        f"UVCC_DP_RANK={int(n.replica_id)} UVCC_DP_WORLD_SIZE={int(R)} "
                        f"UVCC_DP_MASTER_ADDR={shlex.quote(str(dp_master_addr))} UVCC_DP_MASTER_PORT={int(dp_master_port)} "
                        f"UVCC_DP_TIMEOUT_S={int(args.dp_timeout_s)} UVCC_DP_OK_PATH={shlex.quote(str(ok_path))} "
                        f"{shlex.quote(n.venv_python())} -c {shlex.quote(py)}"
                    )
                    cmd = (
                        "set -euo pipefail;"
                        f" mkdir -p {shlex.quote(out_remote)};"
                        f" rm -f {shlex.quote(ok_path)};"
                        f" if [ -f {shlex.quote(pid_path)} ]; then (kill -9 $(cat {shlex.quote(pid_path)}) >/dev/null 2>&1 || true); rm -f {shlex.quote(pid_path)}; fi;"
                        f" rm -f {shlex.quote(log_path)};"
                        f" : > {shlex.quote(log_path)};"
                        f" PYTHONUNBUFFERED=1 nohup bash -lc {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)};"
                        " sleep 0.2;"
                        f" PID=$(cat {shlex.quote(pid_path)});"
                        " kill -0 $PID >/dev/null 2>&1 || (echo 'dp_preflight_exited_early' >&2; tail -n 200 "
                        f"{shlex.quote(log_path)} >&2 || true; exit 1);"
                    )
                    code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
                    if code != 0:
                        raise RuntimeError(f"dp preflight launch failed for p{n.party_id} r{n.replica_id}:\n{out}\n{err}")
                finally:
                    ssh.close()

            # Poll for ok.txt on all ranks.
            deadline_pf = time.time() + max(60, int(args.dp_preflight_timeout_s))
            ok_done: set[Tuple[int, int]] = set()
            ssh_pf: Dict[Tuple[int, int], Any] = {}
            try:
                for n in preflight_nodes:
                    key = (int(n.party_id), int(n.replica_id))
                    ssh_pf[key] = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                while time.time() < deadline_pf:
                    for n in preflight_nodes:
                        key = (int(n.party_id), int(n.replica_id))
                        if key in ok_done:
                            continue
                        out_remote = f"{n.root_dir}/out_dp_preflight/p{int(n.party_id)}/r{int(n.replica_id)}"
                        ok_path = f"{out_remote}/ok.txt"
                        pid_path = f"{out_remote}/preflight.pid"
                        log_path = f"{out_remote}/preflight.log"
                        code, out, _ = ssh_exec(ssh_pf[key], f"bash -lc {shlex.quote('test -f ' + ok_path + ' && echo yes || echo no')}", timeout_s=10)
                        if code == 0 and "yes" in out:
                            ok_done.add(key)
                            log.event("dp_preflight_ok", party_id=int(n.party_id), replica_id=int(n.replica_id))
                            continue
                        # Fail fast if the preflight process already exited (avoids waiting full timeout).
                        code2, out2, _ = ssh_exec(
                            ssh_pf[key],
                            f"bash -lc {shlex.quote('if [ -f ' + pid_path + ' ]; then kill -0 $(cat ' + pid_path + ') >/dev/null 2>&1 && echo alive || echo dead; else echo no_pid; fi')}",
                            timeout_s=10,
                        )
                        if code2 == 0 and "dead" in out2:
                            # Save tail log and fail immediately.
                            code_t, out_t, err_t = ssh_exec(
                                ssh_pf[key],
                                f"bash -lc {shlex.quote('tail -n 200 ' + log_path + ' || true')}",
                                timeout_s=20,
                            )
                            local_tail = out_dir / f"dp_preflight_p{int(n.party_id)}_r{int(n.replica_id)}_tail.log"
                            local_tail.write_text(out_t + err_t, encoding="utf-8")
                            log.event(
                                "dp_preflight_log_saved",
                                party_id=int(n.party_id),
                                replica_id=int(n.replica_id),
                                path=str(local_tail),
                                exit_code=int(code_t),
                            )
                            raise RuntimeError(
                                f"dp preflight process died early (p{int(n.party_id)} r{int(n.replica_id)}); tail saved to {local_tail}"
                            )
                    if len(ok_done) == len(preflight_nodes):
                        break
                    time.sleep(1.0)
                if len(ok_done) != len(preflight_nodes):
                    # Best-effort dump some failing logs.
                    missing = [k for k in sorted(ssh_pf.keys()) if k not in ok_done]
                    log.event("dp_preflight_timeout", missing=missing)
                    # Save tail logs locally for debugging (pods may be deleted on failure).
                    for (pid_m, rid_m) in list(missing)[: min(6, len(missing))]:
                        try:
                            n_m = next(nn for nn in preflight_nodes if int(nn.party_id) == int(pid_m) and int(nn.replica_id) == int(rid_m))
                        except Exception:
                            continue
                        try:
                            out_remote_m = f"{n_m.root_dir}/out_dp_preflight/p{int(n_m.party_id)}/r{int(n_m.replica_id)}"
                            log_path_m = f"{out_remote_m}/preflight.log"
                            code_t, out_t, err_t = ssh_exec(
                                ssh_pf[(int(pid_m), int(rid_m))],
                                f"bash -lc {shlex.quote('tail -n 200 ' + log_path_m + ' || true')}",
                                timeout_s=20,
                            )
                            local_tail = out_dir / f"dp_preflight_p{int(pid_m)}_r{int(rid_m)}_tail.log"
                            local_tail.write_text(out_t + err_t, encoding="utf-8")
                            log.event(
                                "dp_preflight_log_saved",
                                party_id=int(pid_m),
                                replica_id=int(rid_m),
                                path=str(local_tail),
                                exit_code=int(code_t),
                            )
                        except Exception as exc:
                            log.event("dp_preflight_log_fetch_failed", party_id=int(pid_m), replica_id=int(rid_m), error=str(exc))
                    raise TimeoutError(f"dp preflight timeout: ok={len(ok_done)}/{len(preflight_nodes)}")
            finally:
                for ssh in ssh_pf.values():
                    try:
                        ssh.close()
                    except Exception:
                        pass
            log.log("DP NCCL preflight OK.")

        # Launch all 24 party workers.
        for n in nodes:
            ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
            try:
                out_remote = f"{n.root_dir}/out_dp/p{int(n.party_id)}/r{int(n.replica_id)}"
                ca_path = f"{n.root_dir}/relay_ca.pem"
                token_path = f"{n.root_dir}/relay_token.txt"
                inp_path = f"{n.root_dir}/inputs_p{int(n.party_id)}_r{int(n.replica_id)}.json"
                log_path = f"{out_remote}/run.log"
                pid_path = f"{out_remote}/run.pid"

                party_relay_url = str(relay_url_local) if (int(n.party_id) == 0 and int(n.replica_id) == 0) else str(relay_url_public)
                group_id_rep = relay_group_id_for_replica(sid_rep=sid_rep_by_r[int(n.replica_id)], replica_id=int(n.replica_id))
                sid_hex = _hex32(bytes(sid_rep_by_r[int(n.replica_id)]))
                dp_master_addr, dp_master_port = dp_master_by_pid[int(n.party_id)]

                # Ensure NCCL uses a routable NIC on heterogeneous provider images and avoids IB paths.
                nccl_env = (
                    "NCCL_IB_DISABLE=1 "
                    "NCCL_SOCKET_FAMILY=AF_INET "
                    "NCCL_SOCKET_IFNAME=^lo,docker0 "
                )
                inner = (
                    f"{n.env_prefix()} {nccl_env}{shlex.quote(n.venv_python())} {shlex.quote(n.root_dir + '/research/uvcc_parallel/party_train_dp.py')}"
                    f" --party-id {int(n.party_id)} --replica-id {int(n.replica_id)}"
                    f" --relay-url {shlex.quote(party_relay_url)} --relay-group-id {shlex.quote(group_id_rep)}"
                    f" --relay-token-file {shlex.quote(token_path)} --tls-ca-pem {shlex.quote(ca_path)}"
                    f" --job-id-hex {_hex32(job_id32)} --sid-hex {shlex.quote(sid_hex)}"
                    f" --inputs-json {shlex.quote(inp_path)} --out {shlex.quote(out_remote)}"
                    f" --device {'cuda' if require_cuda_job else 'auto'} --require-cuda {'true' if require_cuda_job else 'false'}"
                    f" --steps {int(steps_job)} --epoch 0 --step-offset 0 --epoch-setup-step 1000"
                    f" --checkpoint-enable false --checkpoint-every 1"
                    f" --sks-t-checks {int(sks_t_checks_job)} --sks-sample-log2 {int(sks_sample_log2_job)}"
                    f" --log-level {shlex.quote(str(args.party_log_level))}"
                    f" --dp-enable true --dp-world-size {int(R)} --dp-rank {int(n.replica_id)}"
                    f" --dp-master-addr {shlex.quote(str(dp_master_addr))} --dp-master-port {int(dp_master_port)} --dp-timeout-s {int(args.dp_timeout_s)}"
                )
                cmd = (
                    "set -euo pipefail;"
                    f" mkdir -p {shlex.quote(out_remote)};"
                    # IMPORTANT: when reusing pods (attach-prefix), prior runs may have left result/transcript files behind.
                    # If we don't delete them, the orchestrator can incorrectly treat the run as completed immediately.
                    f" rm -f {shlex.quote(str(out_remote) + '/result.json')} {shlex.quote(str(out_remote) + '/transcript_v1.jsonl')} {shlex.quote(str(out_remote) + '/transcript_v1_live.jsonl')};"
                    f" if [ -f {shlex.quote(pid_path)} ]; then (kill -9 $(cat {shlex.quote(pid_path)}) >/dev/null 2>&1 || true); rm -f {shlex.quote(pid_path)}; fi;"
                    f" rm -f {shlex.quote(log_path)};"
                    f" PYTHONUNBUFFERED=1 nohup bash -lc {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 & echo $! > {shlex.quote(pid_path)};"
                    " sleep 0.2;"
                    f" PID=$(cat {shlex.quote(pid_path)});"
                    " kill -0 $PID >/dev/null 2>&1 || (echo 'party_process_exited_early' >&2; tail -n 200 "
                    f"{shlex.quote(log_path)} >&2 || true; exit 1);"
                )
                code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
                if code != 0:
                    raise RuntimeError(f"failed to launch party p{n.party_id} r{n.replica_id}:\n{out}\n{err}")
            finally:
                ssh.close()
        log.event("parties_launched", count=int(len(nodes)))

        # Poll completion and fetch artifacts.
        ssh_by_key: Dict[Tuple[int, int], Any] = {}
        paths_by_key: Dict[Tuple[int, int], Dict[str, str]] = {}
        try:
            for n in nodes:
                key = (int(n.party_id), int(n.replica_id))
                ssh_by_key[key] = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
                out_remote = f"{n.root_dir}/out_dp/p{int(n.party_id)}/r{int(n.replica_id)}"
                paths_by_key[key] = {
                    "out_remote": str(out_remote),
                    "res_path": str(f"{out_remote}/result.json"),
                    "tr_path": str(f"{out_remote}/transcript_v1.jsonl"),
                    "run_log_remote": str(f"{out_remote}/run.log"),
                    "run_pid_remote": str(f"{out_remote}/run.pid"),
                    "tr_live_remote": str(f"{out_remote}/transcript_v1_live.jsonl"),
                }

            deadline = time.time() + max(60, int(party_timeout_effective_s))
            done: set[Tuple[int, int]] = set()
            while time.time() < deadline:
                for key, ssh in ssh_by_key.items():
                    if key in done:
                        continue
                    pths = paths_by_key[key]
                    pid = int(key[0])
                    rid = int(key[1])
                    # If result.json exists, mark done.
                    code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote('test -f ' + pths['res_path'] + ' && echo yes || echo no')}", timeout_s=10)
                    if code == 0 and "yes" in out:
                        done.add(key)
                        log.event("party_done", party_id=int(pid), replica_id=int(rid))
                        continue
                    # If process died, fail fast.
                    code2, out2, err2 = ssh_exec(
                        ssh,
                        f"bash -lc {shlex.quote('if [ -f ' + pths['run_pid_remote'] + ' ]; then kill -0 $(cat ' + pths['run_pid_remote'] + ') >/dev/null 2>&1 && echo alive || echo dead; else echo no_pid; fi')}",
                        timeout_s=10,
                    )
                    if code2 == 0 and "dead" in out2:
                        tail_cmd = f"bash -lc {shlex.quote('tail -n 200 ' + pths['run_log_remote'] + ' || true')}"
                        _, tout, _ = ssh_exec(ssh, tail_cmd, timeout_s=10)
                        raise RuntimeError(f"party process died early (p{pid} r{rid}); tail:\n{tout}")

                if len(done) == len(nodes):
                    break
                time.sleep(2.0)

            if len(done) != len(nodes):
                raise TimeoutError(f"timeout waiting for parties: done={len(done)}/{len(nodes)}")

        finally:
            for ssh in ssh_by_key.values():
                try:
                    ssh.close()
                except Exception:
                    pass

        # Download artifacts.
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        result_hashes_by_rep: Dict[int, List[str]] = {rid: [] for rid in range(int(R))}

        for n in nodes:
            key = (int(n.party_id), int(n.replica_id))
            pths = paths_by_key[key]
            local_base = artifacts_dir / f"p{int(n.party_id)}" / f"r{int(n.replica_id)}"
            local_base.mkdir(parents=True, exist_ok=True)
            ssh = ssh_connect_with_retries(hostname=n.ssh_host, port=n.ssh_port, username=n.ssh_user, pkey=pkey, timeout_s=900)
            try:
                sftp_get_file(ssh, remote_path=pths["res_path"], local_path=str(local_base / "result.json"))
                sftp_get_file(ssh, remote_path=pths["tr_path"], local_path=str(local_base / "transcript_v1.jsonl"))
                sftp_get_file(ssh, remote_path=pths["run_log_remote"], local_path=str(local_base / "run.log"))
                # best-effort live transcript
                try:
                    sftp_get_file(ssh, remote_path=pths["tr_live_remote"], local_path=str(local_base / "transcript_v1_live.jsonl"))
                except Exception:
                    pass
                # best-effort private checkpoints (do not fail if missing)
                try:
                    sftp_get_file(ssh, remote_path=str(Path(pths["out_remote"]) / "private" / "checkpoints_W.jsonl"), local_path=str(out_dir / "private_keep" / "checkpoints" / f"p{int(n.party_id)}" / f"r{int(n.replica_id)}.jsonl"))
                except Exception:
                    pass
            finally:
                ssh.close()

            rj = json.loads((local_base / "result.json").read_text(encoding="utf-8"))
            # Sanity check: ensure the party actually ran the requested job (prevents stale-result false positives).
            _sanity_check_party_run_log(
                run_log_path=(local_base / "run.log"),
                expect_party_id=int(n.party_id),
                expect_steps=int(steps_job),
                expect_result_hash32_hex=str(rj.get("result_hash32_hex") or "").strip() or None,
            )
            result_hashes_by_rep[int(n.replica_id)].append(str(rj.get("result_hash32_hex") or ""))

        # Consistency checks.
        bad_rep = []
        for rid, hs in result_hashes_by_rep.items():
            hs2 = [h for h in hs if h]
            if len(set(hs2)) != 1:
                bad_rep.append({"replica_id": int(rid), "hashes": hs2})
        if bad_rep:
            log.event("dp_result_mismatch_within_replica", bad=bad_rep)
            raise RuntimeError(f"DP run failed: result hash mismatch within replicas: {bad_rep}")

        all_hashes = sorted({hs[0] for hs in result_hashes_by_rep.values() if hs and hs[0]})
        log.event("dp_result_hashes", unique_hashes=all_hashes)
        if len(all_hashes) != 1:
            raise RuntimeError(f"DP run failed: replicas disagree on final result hash: {all_hashes}")

        # Build DP transcript-of-transcripts roots (PARALLEL.txt §14) and emit a proof_bundle.json.
        #
        # This keeps the v1 verifier intact by:
        # - committing the proof bundle to epoch_roots = [global_root(epoch=0), ...]
        # - setting final_root = compute_final_root_v1(epoch_roots)
        # - storing the per-replica detail under optional_proofs for audit drill-down.
        try:
            from uvcc_client.party_identity import load_or_create_party_privkey32_v1
            from uvcc_party.eip712 import EIP712DomainV1
            from uvcc_party.proof_bundle import ProofBundleV1, party_from_privkey, proof_bundle_hash32_v1, sign_final_root_v1
            from uvcc_verifier.proof_bundle_v1 import parse_proof_bundle_json_v1, verify_proof_bundle_v1
            from uvcc_verifier.transcript_v1 import (
                compute_epoch_roots_v1,
                compute_final_root_v1,
                parse_transcript_jsonl_v1,
                validate_transcript_leaves_v1,
            )
            from uvcc_parallel.dp_roots import global_root_v1, replica_root_from_map_v1
        except Exception as exc:
            log.event("dp_proof_bundle_import_failed", error=str(exc))
            raise

        # Load party privkeys (one per party) from the run output.
        party_keys_dir = out_dir / "private_keep" / "party_keys"
        party_priv_by_pid: Dict[int, bytes] = {}
        for pid in (0, 1, 2):
            kp = party_keys_dir / f"party_privkey_p{int(pid)}.hex"
            priv = load_or_create_party_privkey32_v1(path=str(kp))
            party_priv_by_pid[int(pid)] = bytes(priv)
        parties = [party_from_privkey(party_id=int(pid), privkey32=party_priv_by_pid[int(pid)]) for pid in (0, 1, 2)]

        # Union transcript per replica (p0+p1+p2) and compute per-replica epoch roots.
        replicas_dir = out_dir / "replicas"
        replicas_dir.mkdir(parents=True, exist_ok=True)
        roots_by_rid: Dict[int, Dict[int, bytes]] = {}
        max_epoch = -1
        for rid in range(int(R)):
            union_lines: List[str] = []
            for pid in (0, 1, 2):
                p = artifacts_dir / f"p{int(pid)}" / f"r{int(rid)}" / "transcript_v1.jsonl"
                union_lines += p.read_text(encoding="utf-8", errors="replace").splitlines()
            union_text = "\n".join([ln for ln in union_lines if ln.strip()]) + "\n"
            rep_dir = replicas_dir / f"r{int(rid)}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            union_path = rep_dir / "transcript_v1.jsonl"
            union_path.write_text(union_text, encoding="utf-8")

            leaves = parse_transcript_jsonl_v1(str(union_path))
            validate_transcript_leaves_v1(leaves, strict_unknown_msg_kind=False, strict_netframe_header_hash=True)
            roots_by_epoch = compute_epoch_roots_v1(leaves)
            if not roots_by_epoch:
                raise RuntimeError(f"replica {rid} transcript had zero epoch roots")
            roots_by_rid[int(rid)] = {int(k): bytes(v) for k, v in roots_by_epoch.items()}
            max_epoch = max(max_epoch, max(int(e) for e in roots_by_epoch.keys()))

        # Compute global epoch roots (one per epoch) using replica_root/global_root.
        epoch_roots_global: List[bytes] = []
        dp_detail: Dict[str, Any] = {
            "sid_job_hex": _hex32(bytes(sid_job)),
            "replicas": [],
            "epochs": [],
        }
        for rid in range(int(R)):
            dp_detail["replicas"].append(
                {
                    "replica_id": int(rid),
                    "sid_rep_hex": _hex32(bytes(sid_rep_by_r[int(rid)])),
                }
            )

        for e in range(int(max_epoch) + 1):
            replica_roots_e: List[bytes] = []
            epoch_entry: Dict[str, Any] = {
                "epoch": int(e),
                "replica_epoch_roots_hex": [],
                "replica_roots_hex": [],
                "global_root_hex": "",
            }
            for rid in range(int(R)):
                r_epoch_root = roots_by_rid[int(rid)].get(int(e))
                if r_epoch_root is None or len(r_epoch_root) != 32:
                    raise RuntimeError(f"missing epoch_root for replica {rid} epoch={e}")
                # DP v1 uses one subgroup (stage=0,tp=0) per replica.
                rep_root = replica_root_from_map_v1(
                    sid_rep=bytes(sid_rep_by_r[int(rid)]),
                    epoch=int(e),
                    roots_by_sub={(0, 0): bytes(r_epoch_root)},
                )
                replica_roots_e.append(bytes(rep_root))
                epoch_entry["replica_epoch_roots_hex"].append(_hex32(bytes(r_epoch_root)))
                epoch_entry["replica_roots_hex"].append(_hex32(bytes(rep_root)))
            glob = global_root_v1(sid_job=bytes(sid_job), epoch=int(e), replica_roots=replica_roots_e)
            epoch_roots_global.append(bytes(glob))
            epoch_entry["global_root_hex"] = _hex32(bytes(glob))
            dp_detail["epochs"].append(epoch_entry)

        final_root32 = compute_final_root_v1(epoch_roots=epoch_roots_global)
        (out_dir / "dp_roots.json").write_text(json.dumps(dp_detail, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        # Human-friendly DP matrix summary for quick auditing.
        try:
            lines: List[str] = []
            lines.append("# UVCC DP result matrix (R replicas)")
            lines.append("")
            lines.append(f"- Generated: `{_now_iso_utc()}`")
            lines.append(f"- Replicas: `{int(R)}`")
            lines.append(f"- Result hash: `{str(all_hashes[0])}`")
            lines.append("")
            if dp_detail.get("epochs"):
                e0 = dp_detail["epochs"][0]
                lines.append(f"- Global epoch0 root: `{e0.get('global_root_hex','')}`")
            lines.append("")
            lines.append("| replica_id | sid_rep | epoch0_root | replica_root_epoch0 |")
            lines.append("|---:|---|---|---|")
            epoch0 = dp_detail["epochs"][0] if dp_detail.get("epochs") else {}
            e0_roots = epoch0.get("replica_epoch_roots_hex", []) if isinstance(epoch0, dict) else []
            e0_rep_roots = epoch0.get("replica_roots_hex", []) if isinstance(epoch0, dict) else []
            for rid in range(int(R)):
                sid_rep_hex = str(dp_detail.get("replicas", [{}] * int(R))[rid].get("sid_rep_hex", "")) if isinstance(dp_detail.get("replicas"), list) and len(dp_detail["replicas"]) > rid else ""
                epoch_root_hex = str(e0_roots[rid]) if isinstance(e0_roots, list) and len(e0_roots) > rid else ""
                rep_root_hex = str(e0_rep_roots[rid]) if isinstance(e0_rep_roots, list) and len(e0_rep_roots) > rid else ""
                lines.append(f"| {rid} | `{sid_rep_hex}` | `{epoch_root_hex}` | `{rep_root_hex}` |")
            (out_dir / "dp_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass
        log.event(
            "dp_transcript_roots",
            epoch_roots_hex=[_hex32(r) for r in epoch_roots_global],
            final_root_hex=_hex32(bytes(final_root32)),
            dp_roots_path=str(out_dir / "dp_roots.json"),
        )

        # Build proof bundle (v1 format, with DP detail in optional_proofs).
        result_hash_hex = str(all_hashes[0])
        if result_hash_hex.startswith("0x"):
            result_hash_hex = result_hash_hex[2:]
        result_hash32 = bytes.fromhex(result_hash_hex)

        # Deterministic per-run identifiers (not on-chain here, but bound by signatures).
        policy_hash32 = _sha256(b"UVCC.dp.policy.v1\0" + bytes(job_id32))
        sgir_hash32 = _sha256(b"UVCC.dp.sgir.v1\0" + bytes(job_id32))
        runtime_hash32 = _sha256(b"UVCC.dp.runtime.v1\0" + bytes(job_id32))

        dom = EIP712DomainV1(chain_id=31337, verifying_contract=b"\x00" * 20)
        sigs = [
            sign_final_root_v1(
                party_id=int(pid),
                privkey32=party_priv_by_pid[int(pid)],
                policy_hash32=bytes(policy_hash32),
                final_root32=bytes(final_root32),
                result_hash32=bytes(result_hash32),
                job_id32=bytes(job_id32),
                eip712_domain=dom,
            )
            for pid in (0, 1, 2)
        ]

        pb = ProofBundleV1(
            uvcc_version="1.0",
            job_id32=bytes(job_id32),
            policy_hash32=bytes(policy_hash32),
            eip712_domain=dom,
            sgir_hash32=bytes(sgir_hash32),
            runtime_hash32=bytes(runtime_hash32),
            backend="CRYPTO_CC_3PC",
            parties=parties,
            epoch_roots=epoch_roots_global,
            final_root32=bytes(final_root32),
            signatures=sigs,
            result_hash32=bytes(result_hash32),
            status="OK",
            optional_proofs={"dp": dp_detail},
        )
        proof_json = pb.to_json_bytes()
        proof_path = out_dir / "proof_bundle.json"
        proof_path.write_bytes(proof_json)
        proof_hash32 = proof_bundle_hash32_v1(proof_json)
        log.event("proof_bundle_written", path=str(proof_path), proof_hash32_hex=_hex32(bytes(proof_hash32)))

        # Verifier self-check (structure + signatures + final-root relation).
        proof_parsed = parse_proof_bundle_json_v1(proof_json)
        verify_proof_bundle_v1(proof=proof_parsed, transcript_epoch_roots=epoch_roots_global, transcript_final_root32=final_root32)
        log.event("verifier_ok", proof_bundle_path=str(proof_path))

        log.log("UVCC DP run complete (artifacts downloaded).")
        log.event("dp_run_complete", result_hash32_hex=str(all_hashes[0]))

    finally:
        if str(args.keep_pods).lower() != "true":
            # best-effort cleanup
            def _delete_pod_retry(*, pid: int, rid: int, pod_id: str) -> None:
                last_err = ""
                for attempt in range(1, 9):
                    try:
                        prime.delete_pod(str(pod_id))
                        log.event("pod_deleted", party_id=int(pid), replica_id=int(rid), pod_id=str(pod_id), attempt=int(attempt))
                        return
                    except Exception as exc:
                        last_err = str(exc) or exc.__class__.__name__
                        log.event(
                            "pod_delete_retry",
                            party_id=int(pid),
                            replica_id=int(rid),
                            pod_id=str(pod_id),
                            attempt=int(attempt),
                            error=str(last_err),
                        )
                        time.sleep(min(30.0, 2.0 + attempt * 3.0))
                log.event("pod_delete_failed", party_id=int(pid), replica_id=int(rid), pod_id=str(pod_id), error=str(last_err))

            for pid, rid, pod_id in list(created):
                _delete_pod_retry(pid=int(pid), rid=int(rid), pod_id=str(pod_id))
            log.event("pods_deleted", keep_pods=False)
        else:
            log.event("pods_kept", keep_pods=True, pods=[{"party_id": pid, "replica_id": rid, "pod_id": pod_id} for (pid, rid, pod_id) in created])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


