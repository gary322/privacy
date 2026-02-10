#!/usr/bin/env python3
"""
Phase 6 bring-up runner for uvcc_native:

- Provisions 3 Prime pods (parties 0/1/2)
- Starts the Python relay (uvcc-relay) on party0
- Builds and runs `research/uvcc_native/build/uvcc_worker --mode toy_open` on each pod
- Collects each party's epoch_root output

This is intentionally a *toy* end-to-end to validate:
- relay connectivity (multi-host)
- transport exactly-once + retransmit behavior
- OpenEngine early-arrival buffering
- deterministic transcript roots on each party

It does NOT yet run the full R=2,S=2,T=2,M=8 GPU program; that becomes feasible once:
- nccl bindings are wired (Phase 6/7),
- PP send/recv inside party is implemented,
- stage programs for the real model are implemented (Phase 7+).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shlex
import struct
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _add_paths() -> None:
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "research" / "uvcc" / "uvcc-client"))


_add_paths()

from uvcc_client.prime_api import PrimeClientV1, PrimePodSpecV1  # noqa: E402
from uvcc_client.ssh_runner import (  # noqa: E402
    load_private_key_from_file,
    ssh_connect_with_retries,
    ssh_exec,
    sftp_get_file,
    sftp_put_file,
)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip()


def _load_prime_api_key() -> str:
    v = str(os.environ.get("UVCC_PRIME_API_KEY", "")).strip()
    if v:
        return v
    key_file = Path.home() / ".uvcc" / "prime_api_key.txt"
    if key_file.exists():
        return _read_text(key_file)
    raise RuntimeError("missing Prime API key (set UVCC_PRIME_API_KEY or ~/.uvcc/prime_api_key.txt)")


def _build_bundle(*, repo_root: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # Keep this bundle *tiny*; only ship sources needed to build uvcc_worker + relay on remote pods.
    # DO NOT include local build/outputs (they can be multi-GB and will kill upload time/cost).
    include = [
        repo_root / "research" / "uvcc_native" / "CMakeLists.txt",
        repo_root / "research" / "uvcc_native" / "uvcc",
        repo_root / "research" / "uvcc_native" / "tools",
        repo_root / "research" / "uvcc_native" / "tests",
        repo_root / "research" / "uvcc" / "uvcc-relay",
    ]
    with tarfile.open(out_path, "w:gz") as tf:
        for p in include:
            tf.add(p, arcname=str(p.relative_to(repo_root)))
    return out_path


@dataclass(frozen=True)
class RemotePod:
    party: int
    idx_in_party: int
    pod_id: str
    ssh_user: str
    ssh_host: str
    ssh_port: int


def _bootstrap_and_build(*, pod: RemotePod, bundle_path: Path, remote_root: str, with_nccl: bool, pkey: object) -> None:
    ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
    try:
        remote_bundle = f"{remote_root}/uvcc_native_bundle.tgz"
        # Ensure root exists.
        ssh_exec(ssh, f"bash -lc {shlex.quote(f'mkdir -p {remote_root} && rm -f {remote_bundle}')} ", timeout_s=120)
        sftp_put_file(ssh, local_path=str(bundle_path), remote_path=str(remote_bundle), mode=0o600)

        cmake_extra = " -DUVCC_WITH_CUDA_NCCL=ON" if bool(with_nccl) else ""

        cmd = (
            "set -euo pipefail;"
            f" cd {shlex.quote(remote_root)};"
            f" tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(remote_root)};"
            # Install minimal build deps (best-effort; images vary).
            " if command -v apt-get >/dev/null 2>&1; then"
            "   export DEBIAN_FRONTEND=noninteractive;"
            "   apt-get update -y >/dev/null;"
            "   apt-get install -y --no-install-recommends build-essential cmake pkg-config curl libcurl4-openssl-dev python3 ca-certificates >/dev/null;"
            " fi;"
            f" cd {shlex.quote(remote_root + '/research/uvcc_native')};"
            # Force a clean rebuild so updated sources always take effect (tar preserves mtimes).
            " rm -rf build;"
            f" cmake -S . -B build -DCMAKE_BUILD_TYPE=Release{cmake_extra} >/dev/null;"
            " cmake --build build -j 4 >/dev/null;"
            " echo 'build_ok';"
        )
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=3600)
        if code != 0:
            raise RuntimeError(f"bootstrap/build failed for party{pod.party}:\n{out}\n{err}")
    finally:
        ssh.close()


def _start_relay(*, pod0: RemotePod, remote_root: str, port: int, token: str, lease_s: int, ttl_s: int, pkey: object) -> Tuple[str, str]:
    ssh = ssh_connect_with_retries(hostname=pod0.ssh_host, port=pod0.ssh_port, username=pod0.ssh_user, pkey=pkey, timeout_s=900)
    try:
        relay_py = f"{remote_root}/research/uvcc/uvcc-relay/relay_server.py"
        db_path = f"{remote_root}/relay_db.sqlite"
        log_path = f"{remote_root}/relay.log"
        pid_path = f"{remote_root}/relay.pid"
        token_path = f"{remote_root}/relay_token.txt"

        inner = (
            f"python3 {shlex.quote(relay_py)} "
            f"--host 0.0.0.0 --port {int(port)} "
            f"--db {shlex.quote(db_path)} "
            f"--require-token true "
            f"--token-file {shlex.quote(token_path)} "
            f"--default-ttl-s {int(ttl_s)} "
            f"--lease-s {int(lease_s)} "
            f"--gc-interval-s 5.0"
        )
        # Important: use `setsid` + stdin redirection so the relay process is fully detached from the SSH PTY.
        # Some providers/SSH servers will SIGHUP/terminate background jobs tied to a PTY once the channel closes.
        cmd = (
            "set -euo pipefail;"
            f" if [ -f {shlex.quote(pid_path)} ]; then OLD=$(cat {shlex.quote(pid_path)} 2>/dev/null || true);"
            "   if [ -n \"$OLD\" ]; then (kill -9 \"$OLD\" >/dev/null 2>&1 || true); fi;"
            " fi;"
            f" rm -f {shlex.quote(pid_path)};"
            f" rm -f {shlex.quote(log_path)};"
            # Determinism / replay safety: clear the relay DB so reusing the same sid_job_hex won't see stale msg_ids.
            f" rm -f {shlex.quote(db_path)} {shlex.quote(db_path + '-wal')} {shlex.quote(db_path + '-shm')} >/dev/null 2>&1 || true;"
            f" : > {shlex.quote(log_path)};"
            f" nohup setsid bash -lc {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)};"
            " sleep 0.2;"
            f" kill -0 $(cat {shlex.quote(pid_path)}) >/dev/null 2>&1 || (echo 'relay_failed' >&2; tail -n 200 {shlex.quote(log_path)} >&2; exit 1);"
            " echo 'relay_ok';"
        )
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
        if code != 0:
            raise RuntimeError(f"failed to start relay:\n{out}\n{err}")
        return f"http://{pod0.ssh_host}:{int(port)}", str(token_path)
    finally:
        ssh.close()


def _write_relay_token_file(*, pod: RemotePod, remote_root: str, token: str, pkey: object) -> str:
    """
    Write relay token to a well-known path on *each* pod so uvcc_worker can read it.
    Returns remote token path.
    """
    ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
    try:
        token_path = f"{remote_root}/relay_token.txt"
        cmd = (
            "set -euo pipefail;"
            f" mkdir -p {shlex.quote(remote_root)};"
            " umask 077;"
            f" printf %s {shlex.quote(str(token))} > {shlex.quote(token_path)};"
            f" chmod 600 {shlex.quote(token_path)};"
        )
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
        if code != 0:
            raise RuntimeError(f"failed to write relay token file on {pod.ssh_host}:\n{out}\n{err}")
        return token_path
    finally:
        ssh.close()

@dataclass(frozen=True)
class WorkerLaunch:
    party: int
    replica: int
    stage: int
    tp: int
    local_rank: int
    pod: RemotePod
    gpu_idx: int
    group_id: str


def _launch_worker_bg(
    *,
    pod: RemotePod,
    launch: WorkerLaunch,
    remote_root: str,
    sid_job_hex: str,
    relay_url: str,
    relay_token_file: str,
    relay_timeout_s: float,
    debug_relay: bool,
    worker_mode: str,
    topo_r: int,
    topo_s: int,
    topo_t: int,
    microbatches: int,
    step_id: int,
    phase6_enable_dp: bool,
    phase6_timeout_s: int,
    pkey: object,
) -> Tuple[str, str]:
    """
    Launch one uvcc_worker process in the background.
    Returns (log_path, done_path).
    """
    ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
    try:
        out_dir = f"{remote_root}/out_native_toy/p{launch.party}/r{launch.replica}/s{launch.stage}/t{launch.tp}"
        log_path = f"{out_dir}/run.log"
        done_path = f"{out_dir}/done.txt"
        pid_path = f"{out_dir}/run.pid"
        exit_path = f"{out_dir}/exit_code.txt"
        cmd_path = f"{out_dir}/cmd.txt"
        worker = f"{remote_root}/research/uvcc_native/build/uvcc_worker"
        # Important: capture *all* output (including shell errors before exec) to run.log.
        # Otherwise we can end up stuck waiting for done.txt with no diagnostics.
        dbg = "UVCC_DEBUG_RELAY=1 " if bool(debug_relay) else ""
        wm = str(worker_mode or "").strip().lower()
        if wm not in ("toy_open", "nccl_smoke", "phase6_step"):
            raise RuntimeError(f"unknown worker_mode={worker_mode}")
        nccl_env = ""
        if wm in ("nccl_smoke", "phase6_step"):
            # Conservative defaults across providers.
            nccl_env = "NCCL_SOCKET_IFNAME=^lo,docker0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO "
        extra = ""
        if wm == "phase6_step":
            # Default behavior historically skipped DP init; enable DP only when explicitly requested.
            if not bool(phase6_enable_dp):
                extra = " --phase6-skip-dp"
            extra += f" --phase6-timeout-s {int(phase6_timeout_s)}"
        # Wrap the worker so we always record an exit code, and always emit at least one log line.
        # This makes failures diagnosable even when the worker crashes before producing output.
        inner = (
            "set -uo pipefail;"
            f" mkdir -p {shlex.quote(out_dir)};"
            f" rm -f {shlex.quote(done_path)} {shlex.quote(exit_path)} {shlex.quote(cmd_path)};"
            " umask 077;"
            f" export CUDA_VISIBLE_DEVICES={int(launch.gpu_idx)};"
            f" echo \"start_ts=$(date -Is) host=$(hostname) party={int(launch.party)} r={int(launch.replica)} s={int(launch.stage)} t={int(launch.tp)} gpu={int(launch.gpu_idx)}\";"
            f" printf %s {shlex.quote(str(worker))} > {shlex.quote(cmd_path)};"
            " set +e;"
            f" {nccl_env}{dbg}{shlex.quote(worker)}"
            f" --mode {shlex.quote(wm)}"
            f" --sid-job-hex {shlex.quote(sid_job_hex)}"
            f" --party {int(launch.party)} --replica {int(launch.replica)} --stage {int(launch.stage)} --tp {int(launch.tp)}"
            f" --replicas {int(topo_r)} --stages {int(topo_s)} --tp-ranks {int(topo_t)}"
            f" --relay-url {shlex.quote(relay_url)}"
            f" --relay-group-id {shlex.quote(launch.group_id)}"
            f" --relay-token-file {shlex.quote(relay_token_file)}"
            f" --relay-timeout-s {float(relay_timeout_s):.3f}"
            f" --microbatches {int(microbatches)} --step-id {int(step_id)}{extra};"
            " rc=$?;"
            " set -e;"
            f" echo $rc > {shlex.quote(exit_path)};"
            f" if [ \"$rc\" = \"0\" ]; then echo ok > {shlex.quote(done_path)}; fi;"
            " exit $rc;"
        )
        # Run in background to allow all 3 parties to start.
        # Important: use `setsid` + stdin redirection so the worker wrapper is fully detached from the SSH PTY.
        # Without this, some environments will terminate the backgrounded bash before it runs the worker,
        # leaving 0-byte run.log and no exit_code.txt diagnostics.
        bg = (
            "set -euo pipefail;"
            f" mkdir -p {shlex.quote(out_dir)};"
            f" rm -f {shlex.quote(pid_path)};"
            f" : > {shlex.quote(log_path)};"
            f" nohup setsid bash -lc {shlex.quote(inner)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $! > {shlex.quote(pid_path)};"
            " sleep 0.2;"
            f" kill -0 $(cat {shlex.quote(pid_path)}) >/dev/null 2>&1 || (echo 'worker_failed' >&2; tail -n 200 {shlex.quote(log_path)} >&2; exit 1);"
        )
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(bg)}", timeout_s=30)
        if code != 0:
            raise RuntimeError(f"failed to launch worker p{launch.party} r{launch.replica} s{launch.stage} t{launch.tp}:\n{out}\n{err}")
        return log_path, done_path
    finally:
        ssh.close()


def _run_worker_fg_once(
    *,
    pod: RemotePod,
    remote_root: str,
    sid_job_hex: str,
    relay_url: str,
    group_id: str,
    relay_token_file: str,
    relay_timeout_s: float,
    topo_r: int,
    topo_s: int,
    topo_t: int,
    debug_relay: bool,
    worker_mode: str,
    microbatches: int,
    step_id: int,
    phase6_enable_dp: bool,
    phase6_timeout_s: int,
    timeout_s: int,
    pkey: object,
) -> Tuple[int, str, str]:
    """
    Run uvcc_worker toy_open once on a pod (foreground), optionally bounded by `timeout_s`.
    Returns (exit_code, stdout, stderr).
    """
    ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
    try:
        worker = f"{remote_root}/research/uvcc_native/build/uvcc_worker"
        dbg = "UVCC_DEBUG_RELAY=1 " if bool(debug_relay) else ""
        wm = str(worker_mode or "").strip().lower()
        if wm not in ("toy_open", "nccl_smoke", "phase6_step"):
            raise RuntimeError(f"unknown worker_mode={worker_mode}")
        nccl_env = ""
        if wm in ("nccl_smoke", "phase6_step"):
            nccl_env = "NCCL_SOCKET_IFNAME=^lo,docker0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO "
        extra = ""
        if wm == "phase6_step":
            if not bool(phase6_enable_dp):
                extra = " --phase6-skip-dp"
            extra += f" --phase6-timeout-s {int(phase6_timeout_s)}"
        inner = (
            "set -euo pipefail;"
            f" {nccl_env}{dbg}{shlex.quote(worker)}"
            f" --mode {shlex.quote(wm)}"
            f" --sid-job-hex {shlex.quote(str(sid_job_hex))}"
            f" --party {int(pod.party)} --replica 0 --stage 0 --tp 0"
            f" --replicas {int(topo_r)} --stages {int(topo_s)} --tp-ranks {int(topo_t)}"
            f" --relay-url {shlex.quote(str(relay_url))}"
            f" --relay-group-id {shlex.quote(str(group_id))}"
            f" --relay-token-file {shlex.quote(str(relay_token_file))}"
            f" --relay-timeout-s {float(relay_timeout_s):.3f}"
            f" --microbatches {int(microbatches)} --step-id {int(step_id)}{extra}"
        )
        cmd = inner
        if int(timeout_s) > 0:
            # Use coreutils timeout; preserve-status is not necessary here.
            cmd = f"timeout {int(timeout_s)}s bash -lc {shlex.quote(inner)}"
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=max(10, int(timeout_s) + 15))
        return int(code), str(out), str(err)
    finally:
        ssh.close()


def _wait_done(*, pod: RemotePod, done_path: str, pkey: object, timeout_s: int = 3600) -> None:
    ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
    try:
        dp = str(done_path)
        # Convention: out_dir/{run.log,run.pid,done.txt}
        pid_path = dp.replace("/done.txt", "/run.pid")
        log_path = dp.replace("/done.txt", "/run.log")
        deadline = time.time() + int(timeout_s)
        while time.time() < deadline:
            code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(f'test -f {dp} && cat {dp} || true')}", timeout_s=20)
            if code == 0 and str(out).strip():
                return

            # If the worker exited early, fail fast with log tail.
            chk = (
                "set -euo pipefail;"
                f" if [ -f {shlex.quote(pid_path)} ]; then PID=$(cat {shlex.quote(pid_path)} 2>/dev/null || true); else PID=''; fi;"
                " if [ -n \"$PID\" ] && kill -0 \"$PID\" >/dev/null 2>&1; then echo RUNNING; exit 0; fi;"
                " echo EXITED;"
            )
            code2, out2, err2 = ssh_exec(ssh, f"bash -lc {shlex.quote(chk)}", timeout_s=20)
            if "EXITED" in str(out2):
                # Race fix: the wrapper can exit after successfully writing done.txt.
                # Re-check done.txt before treating EXITED as failure.
                _, out_done2, _ = ssh_exec(ssh, f"bash -lc {shlex.quote(f'test -f {dp} && cat {dp} || true')}", timeout_s=20)
                if str(out_done2).strip():
                    return
                exit_path = dp.replace("/done.txt", "/exit_code.txt")
                # Best-effort: include exit code and directory listing for fast diagnosis.
                _, out_ec, err_ec = ssh_exec(ssh, f"bash -lc {shlex.quote(f'cat {exit_path} 2>/dev/null || true')}", timeout_s=20)
                _, out_ls, err_ls = ssh_exec(
                    ssh,
                    f"bash -lc {shlex.quote(f'ls -lah {shlex.quote(str(Path(log_path).parent))} 2>/dev/null || true')}",
                    timeout_s=20,
                )
                tail_cmd = f"tail -n 200 {shlex.quote(log_path)} || true"
                _, out3, err3 = ssh_exec(ssh, f"bash -lc {shlex.quote(tail_cmd)}", timeout_s=20)
                raise RuntimeError(
                    f"worker exited before writing done.txt on {pod.ssh_host} (done={dp}).\n"
                    f"--- exit_code.txt ---\n{out_ec}{err_ec}\n"
                    f"--- ls -lah out_dir ---\n{out_ls}{err_ls}\n"
                    f"--- run.log tail ---\n{out3}{err3}"
                )
            time.sleep(1.0)
        raise RuntimeError(f"timeout waiting for {done_path} on {pod.ssh_host}")
    finally:
        ssh.close()


def _run_toy_open(*, pod: RemotePod, remote_root: str, sid_job_hex: str, relay_url: str, relay_group_id: str, relay_token: str, pkey: object) -> str:
    ssh = ssh_connect_with_retries(hostname=pod.ssh_host, port=pod.ssh_port, username=pod.ssh_user, pkey=pkey, timeout_s=900)
    try:
        worker = f"{remote_root}/research/uvcc_native/build/uvcc_worker"
        cmd = (
            "set -euo pipefail;"
            f" {shlex.quote(worker)}"
            f" --mode toy_open"
            f" --sid-job-hex {shlex.quote(sid_job_hex)}"
            f" --party {int(pod.party)} --replica 0 --stage 0 --tp 0"
            f" --replicas 1 --stages 1 --tp-ranks 1"
            f" --relay-url {shlex.quote(relay_url)}"
            f" --relay-group-id {shlex.quote(relay_group_id)}"
            f" --relay-token {shlex.quote(relay_token)}"
            f" --microbatches 1 --step-id 0"
        )
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=900)
        if code != 0:
            raise RuntimeError(f"toy_open failed for party{pod.party}:\n{out}\n{err}")
        return (out + err).strip()
    finally:
        ssh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider-type", default="datacrunch", help="Prime provider_type")
    ap.add_argument("--prefer-gpu-type", action="append", default=None, help="Preferred GPU type (repeatable)")
    ap.add_argument("--strict-prefer-gpu-type", action="store_true", help="If set, only try offers whose gpuType matches --prefer-gpu-type")
    ap.add_argument("--prefer-region", action="append", default=None, help="Preferred region (repeatable)")
    ap.add_argument("--data-center-id", default="auto", help="Prime dataCenterId / data_center_id routing field (default: auto from offer.region)")
    ap.add_argument("--max-price", type=float, default=None, help="Max price per GPU-hour (Prime maxPrice)")
    ap.add_argument("--pods-per-party", type=int, default=1, help="How many pods to provision per party (for multi-GPU parties)")
    ap.add_argument("--gpus-per-pod", type=int, default=1, help="GPUs per pod")
    ap.add_argument("--replicas", type=int, default=1, help="Topology R (replicas)")
    ap.add_argument("--stages", type=int, default=1, help="Topology S (pipeline stages)")
    ap.add_argument("--tp-ranks", type=int, default=1, help="Topology T (tensor ranks)")
    ap.add_argument("--socket", default="auto", help="Socket filter (auto/PCIe/SXM)")
    ap.add_argument("--image", default="auto", help="Prime image to use (auto uses offer first image)")
    ap.add_argument("--pod-name-prefix", default="uvcc-native-toy", help="Pod name prefix")
    ap.add_argument("--pod-active-timeout-s", type=int, default=900, help="Timeout for a pod to become ACTIVE+SSH-ready before retrying/replacing")
    ap.add_argument("--pod-poll-s", type=float, default=10.0, help="Polling interval for pod status")
    ap.add_argument("--relay-port", type=int, default=8080)
    ap.add_argument("--relay-lease-s", type=int, default=60)
    ap.add_argument("--relay-ttl-s", type=int, default=3600)
    ap.add_argument(
        "--worker-relay-timeout-s",
        type=float,
        default=60.0,
        help="uvcc_worker relay HTTP client timeout in seconds (passed as --relay-timeout-s). Default: 60.0",
    )
    ap.add_argument("--debug-relay", action="store_true", help="Enable verbose relay polling debug logs in uvcc_worker (writes to per-worker run.log)")
    ap.add_argument(
        "--worker-mode",
        default="toy_open",
        choices=["toy_open", "nccl_smoke", "phase6_step"],
        help="uvcc_worker --mode to run on each worker",
    )
    ap.add_argument("--with-nccl", action="store_true", help="Build uvcc_worker with CUDA+NCCL support (sets -DUVCC_WITH_CUDA_NCCL=ON)")
    ap.add_argument(
        "--phase6-enable-dp",
        action="store_true",
        help="For --worker-mode=phase6_step: enable DP NCCL groups and dp_allreduce checks (default: disabled for safety).",
    )
    ap.add_argument("--launch-mode", default="pump", choices=["pump", "bg"], help="How to drive toy_open workers (pump=short foreground bursts, bg=spawn background workers)")
    ap.add_argument("--pump-rounds", type=int, default=60, help="Max pump rounds (only used if --launch-mode=pump)")
    ap.add_argument("--pump-timeout-s", type=int, default=6, help="Per-party timeout per pump run (seconds)")
    ap.add_argument("--microbatches", type=int, default=1, help="Toy program microbatches M (default: 1)")
    ap.add_argument("--step-id", type=int, default=0, help="Toy program step_id (epoch_id32) (default: 0)")
    ap.add_argument("--sid-job-hex", default="auto", help="Optional fixed sid_job hex (0x..64 hex). Default: auto (random)")
    ap.add_argument(
        "--phase6-timeout-s",
        type=int,
        default=600,
        help="For --worker-mode=phase6_step: per-wait timeout in seconds (OPEN + PP waits). Default: 600.",
    )
    ap.add_argument(
        "--oversubscribe",
        action="store_true",
        help=(
            "Allow mapping multiple workers onto the same (pod,gpu) slot. "
            "For --worker-mode=nccl_smoke we use a group-safe mapping to avoid NCCL duplicate-GPU errors."
        ),
    )
    ap.add_argument("--skip-bootstrap", action="store_true", help="Skip remote bundle upload/build (use when attaching to existing pods)")
    ap.add_argument("--attach-pod-ids", default="", help="Attach to existing pods: comma-separated pod ids in party-major order (len=3*pods_per_party)")
    ap.add_argument("--attach-name-prefix", default="", help="Attach to existing pods by name prefix (expects names containing -p0-, -p1-, -p2-). Picks newest pods_per_party per party.")
    ap.add_argument("--out", default="auto", help="Local output directory (default: auto timestamp under research/uvcc_native/)")
    ap.add_argument("--remote-root", default="/root/uvcc", help="Remote root dir")
    ap.add_argument("--ssh-key-path", default=None, help="Path to SSH private key for Prime pods (defaults to best-effort ~/.ssh/* key)")
    ap.add_argument("--list-offers", action="store_true", help="List candidate offers and exit (no provisioning)")
    ap.add_argument("--keep-pods", action="store_true", help="Do not delete pods on exit")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_arg = str(args.out).strip().lower()
    if out_arg == "" or out_arg == "auto":
        out_dir = repo_root / "research" / "uvcc_native" / f"out-prime-native-toy-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    else:
        out_dir = Path(str(args.out)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[runner] out_dir={out_dir}", flush=True)
    bundle_path = out_dir / "uvcc_native_bundle.tgz"
    _build_bundle(repo_root=repo_root, out_path=bundle_path)
    print(f"[runner] bundle_path={bundle_path} bytes={bundle_path.stat().st_size}", flush=True)

    api_key = _load_prime_api_key()
    prime = PrimeClientV1(api_key=api_key)
    ssh_key_path = str(args.ssh_key_path).strip() if args.ssh_key_path is not None else ""
    if not ssh_key_path:
        for cand in ("~/.ssh/uvcc_prime_runner_ed25519", "~/.ssh/vracu_prime_intellect_ed25519", "~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
            p = Path(str(cand)).expanduser()
            if p.exists():
                ssh_key_path = str(p)
                break
    if not ssh_key_path:
        raise RuntimeError("ssh key not found; pass --ssh-key-path")
    pkey = load_private_key_from_file(str(ssh_key_path))

    attach_pod_ids_raw = str(args.attach_pod_ids or "").strip()
    attach_prefix = str(args.attach_name_prefix or "").strip()
    attach_mode = bool(attach_pod_ids_raw or attach_prefix)

    # Pick one offer and use it for all 3 parties (provision mode only).
    #
    # IMPORTANT: In attach mode we do *not* need to hit Prime availability endpoints at all.
    # They occasionally return malformed rows and can crash the runner even though attach mode
    # would otherwise succeed.
    offers: List[Any] = []
    if (not attach_mode) or bool(args.list_offers):
        offers = prime.candidate_offers_v1(
            nodes=1,
            gpu_count_per_node=int(args.gpus_per_pod),
            provider_type=str(args.provider_type),
            socket=str(args.socket),
            prefer_gpu_types=list(args.prefer_gpu_type) if args.prefer_gpu_type else None,
            prefer_regions=list(args.prefer_region) if args.prefer_region else None,
            limit=20,
        )
        print(f"[runner] candidate_offers={len(offers)} provider={args.provider_type} gpus_per_pod={int(args.gpus_per_pod)} socket={args.socket}", flush=True)
        if bool(args.strict_prefer_gpu_type) and args.prefer_gpu_type:
            want = {str(x).strip().lower() for x in (args.prefer_gpu_type or []) if str(x).strip()}
            offers = [o for o in offers if str(getattr(o, "gpu_type", "")).strip().lower() in want]
            if not offers:
                raise RuntimeError(f"no offers match --prefer-gpu-type with --strict-prefer-gpu-type (want={sorted(list(want))})")
        if bool(args.list_offers):
            for i, o in enumerate(offers[:20]):
                imgs = getattr(o, "images", None)
                imgs_s = ""
                try:
                    if isinstance(imgs, list) and imgs:
                        imgs_s = ",".join([str(x) for x in imgs[:3]])
                except Exception:
                    imgs_s = ""
                print(
                    f"[{i}] provider={o.provider} region={o.region} dc={getattr(o,'data_center_id',None)} cloud_id={o.cloud_id} gpu={o.gpu_type} gpu_count={o.gpu_count} socket={o.socket} avail={o.available} images0_2={imgs_s}"
                )
            return 0
    else:
        print("[runner] attach_mode=true: skipping candidate_offers lookup", flush=True)
    dc_arg = str(args.data_center_id).strip()
    img_req = str(args.image).strip()

    def _parse_sid_job_hex(s: str) -> str:
        t = str(s or "").strip()
        if t == "" or t.lower() == "auto":
            return "0x" + secrets.token_hex(32)
        if t.startswith("0x") or t.startswith("0X"):
            raw = t[2:]
            out = "0x" + raw
        else:
            raw = t
            out = "0x" + raw
        if len(raw) != 64:
            raise RuntimeError(f"--sid-job-hex must be 32 bytes hex (got {len(raw)} hex chars)")
        # Validate hex.
        int(raw, 16)
        return out

    def _sha256(b: bytes) -> bytes:
        return hashlib.sha256(bytes(b)).digest()

    def _sid_replica_bytes(*, sid_job32: bytes, replica_id: int) -> bytes:
        return _sha256(b"UVCC_SID_REPLICA_V1" + bytes(sid_job32) + struct.pack("<I", int(replica_id) & 0xFFFFFFFF))

    def _replica_root_bytes(*, sid_rep32: bytes, step_id: int, roots32_sorted: List[bytes]) -> bytes:
        return _sha256(b"UVCC_REPLICA_ROOT_V1" + bytes(sid_rep32) + struct.pack("<I", int(step_id) & 0xFFFFFFFF) + b"".join(roots32_sorted))

    def _global_root_bytes(*, sid_job32: bytes, step_id: int, replica_roots32: List[bytes]) -> bytes:
        return _sha256(b"UVCC_GLOBAL_ROOT_V1" + bytes(sid_job32) + struct.pack("<I", int(step_id) & 0xFFFFFFFF) + b"".join(replica_roots32))

    def _pick_dc_for_offer(o: Any) -> Optional[str]:
        if dc_arg and dc_arg.lower() != "auto":
            return str(dc_arg)
        dc2 = getattr(o, "data_center_id", None)
        if dc2 is not None and str(dc2).strip():
            return str(dc2).strip()
        if getattr(o, "region", None) is not None and str(getattr(o, "region", None)).strip():
            return str(getattr(o, "region", None)).strip()
        return None

    def _pick_img_for_offer(o: Any) -> str:
        if img_req and img_req.lower() != "auto":
            return str(img_req).strip()
        if o.images and len(o.images) > 0 and str(o.images[0]).strip():
            return str(o.images[0]).strip()
        raise RuntimeError("offer has no images; pass --image explicitly")

    pods: list[RemotePod] = []
    try:
        # Provisioning can be flaky across offers/providers. Try offers in order until we get a full 3-party set.
        fail_chain: List[str] = []
        selected_offer = None
        selected_img = None
        selected_dc = None

        if attach_mode:
            attach_ids: List[str] = []
            if attach_pod_ids_raw:
                attach_ids = [x.strip() for x in str(attach_pod_ids_raw).split(",") if str(x).strip()]
            elif attach_prefix:
                # Best-effort pod discovery by name prefix. Assumes names encode party as "-p0-", "-p1-", "-p2-".
                import requests

                from uvcc_client.prime_api import _prime_headers  # type: ignore

                url = f"{prime.api_base}/pods/"
                resp = requests.get(url, headers=_prime_headers(prime.api_key), timeout=max(10.0, float(prime.timeout_s)))
                if resp.status_code >= 400:
                    raise RuntimeError(f"attach-name-prefix failed to list pods ({resp.status_code}): {resp.text}")
                data = resp.json()
                rows = data.get("data") if isinstance(data, dict) else None
                rows_list = rows if isinstance(rows, list) else []
                by_party: Dict[int, List[Tuple[str, str]]] = {0: [], 1: [], 2: []}
                for row in rows_list:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("name") or "").strip()
                    status = str(row.get("status") or "").strip().upper()
                    pid = str(row.get("id") or "").strip()
                    if not name or not pid:
                        continue
                    if attach_prefix not in name:
                        continue
                    if status != "ACTIVE":
                        continue
                    for party in (0, 1, 2):
                        if f"-p{party}-" in name:
                            by_party[int(party)].append((name, pid))
                            break

                want_pp = int(args.pods_per_party)
                attach_ids = []
                for party in (0, 1, 2):
                    cands = sorted(by_party[int(party)], key=lambda x: x[0], reverse=True)
                    if len(cands) < want_pp:
                        raise RuntimeError(f"attach-name-prefix={attach_prefix} found only {len(cands)} ACTIVE pods for party{party}, need {want_pp}")
                    attach_ids.extend([pid for _, pid in cands[:want_pp]])

            want_ids = 3 * int(args.pods_per_party)
            if len(attach_ids) != want_ids:
                raise RuntimeError(f"attach expects len=3*pods_per_party ({want_ids}) pod ids, got {len(attach_ids)}")

            print(f"[runner] attach_mode pod_ids={attach_ids}", flush=True)
            attach_pods: List[RemotePod] = []
            k = 0
            for party in (0, 1, 2):
                for j in range(int(args.pods_per_party)):
                    pid = str(attach_ids[k])
                    k += 1
                    print(f"[runner] attach_wait_active pod_id={pid}", flush=True)
                    # Attach mode: pods are expected to already be ACTIVE, but Prime status endpoints can be transiently flaky.
                    # Do not cap the wait to 60s; honor the configured pod_active_timeout_s.
                    pod = prime.wait_active(str(pid), timeout_s=int(args.pod_active_timeout_s), poll_s=float(args.pod_poll_s))
                    attach_pods.append(
                        RemotePod(
                            party=int(party),
                            idx_in_party=int(j),
                            pod_id=str(pod.pod_id),
                            ssh_user=str(pod.ssh_user),
                            ssh_host=str(pod.ssh_host),
                            ssh_port=int(pod.ssh_port),
                        )
                    )
                    print(f"[runner] attached party={party} idx={j} pod_id={pod.pod_id} ssh={pod.ssh_user}@{pod.ssh_host}:{pod.ssh_port}", flush=True)

            pods = attach_pods
            selected_offer = True
            selected_img = "attached"
            selected_dc = None
            # Prevent provisioning loop from running.
            offers = []

        for o in offers:
            attempt_pods: List[RemotePod] = []
            attempt_pod_ids: List[str] = []
            try:
                dc = _pick_dc_for_offer(o)
                img = _pick_img_for_offer(o)
                print(
                    f"[runner] trying_offer cloud_id={getattr(o,'cloud_id',None)} gpu={getattr(o,'gpu_type',None)} socket={getattr(o,'socket',None)} region={getattr(o,'region',None)} dc={dc} img={img}",
                    flush=True,
                )
                for party in (0, 1, 2):
                    for j in range(int(args.pods_per_party)):
                        name = f"{str(args.pod_name_prefix)}-p{party}-{j}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                        print(f"[runner] create_pod party={party} idx={j} name={name}", flush=True)
                        spec = PrimePodSpecV1(
                            cloud_id=str(o.cloud_id),
                            gpu_type=str(o.gpu_type),
                            gpu_count=int(args.gpus_per_pod),
                            socket=str(args.socket if str(args.socket).lower() != "auto" else (o.socket or "auto")),
                            image=str(img),
                            name=str(name),
                            provider_type=str(args.provider_type),
                            data_center_id=str(dc) if dc is not None else None,
                            max_price=float(args.max_price) if args.max_price is not None else None,
                        )
                        pod_id = prime.create_pod(spec)
                        attempt_pod_ids.append(str(pod_id))
                        print(f"[runner] pod_created party={party} idx={j} pod_id={pod_id}", flush=True)
                        print(f"[runner] wait_active pod_id={pod_id}", flush=True)
                        pod = prime.wait_active(str(pod_id), timeout_s=int(args.pod_active_timeout_s), poll_s=float(args.pod_poll_s))
                        attempt_pods.append(
                            RemotePod(
                                party=party,
                                idx_in_party=j,
                                pod_id=pod.pod_id,
                                ssh_user=pod.ssh_user,
                                ssh_host=pod.ssh_host,
                                ssh_port=pod.ssh_port,
                            )
                        )
                        print(
                            f"[runner] pod_active party={party} idx={j} pod_id={pod.pod_id} ssh={pod.ssh_user}@{pod.ssh_host}:{pod.ssh_port}",
                            flush=True,
                        )
                # Success: adopt this attempt.
                pods = attempt_pods
                selected_offer = o
                selected_img = img
                selected_dc = dc
                print(
                    f"[runner] provision_ok offer_cloud_id={getattr(o,'cloud_id',None)} gpu={getattr(o,'gpu_type',None)} dc={selected_dc} img={selected_img} pods={len(pods)}",
                    flush=True,
                )
                break
            except Exception as e:
                fail_chain.append(
                    f"offer cloud_id={getattr(o, 'cloud_id', None)} gpu={getattr(o, 'gpu_type', None)} socket={getattr(o, 'socket', None)} region={getattr(o, 'region', None)}: {type(e).__name__}: {e}"
                )
                print(f"[runner] provision_failed {fail_chain[-1]}", flush=True)
                # Best-effort cleanup of any pods created in this failed attempt.
                for p in attempt_pods:
                    try:
                        prime.delete_pod(p.pod_id)
                    except Exception:
                        pass
                for pid in attempt_pod_ids:
                    if not pid:
                        continue
                    try:
                        prime.delete_pod(str(pid))
                    except Exception:
                        pass
                continue

        if selected_offer is None:
            msg = "unable to provision pods with any candidate offer"
            if fail_chain:
                msg += "\n" + "\n".join(fail_chain[:50])
            raise RuntimeError(msg)

        want_nccl = bool(args.with_nccl) or (str(args.worker_mode).strip().lower() == "nccl_smoke")

        # Attach-mode safety: kill any stale workers/relays and clear old output dirs.
        # This avoids mismatched NCCL participants (hangs) when reusing the same pods.
        if attach_mode:
            print("[runner] attach_mode: remote_preclean (kill uvcc_worker + relay + clear out_native_toy)", flush=True)
            for p in pods:
                ssh = ssh_connect_with_retries(hostname=p.ssh_host, port=p.ssh_port, username=p.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    out_party = f"{str(args.remote_root)}/out_native_toy/p{int(p.party)}"
                    cmd = (
                        "set -euo pipefail;"
                        " pkill -9 -f '/root/uvcc/research/uvcc_native/build/uvcc_worker' >/dev/null 2>&1 || true;"
                        " pkill -9 -f 'uvcc-relay/relay_server.py' >/dev/null 2>&1 || true;"
                        f" rm -rf {shlex.quote(out_party)} >/dev/null 2>&1 || true;"
                        " true;"
                    )
                    ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=60)
                finally:
                    ssh.close()

        # Bootstrap + build uvcc_worker on all pods (unless attaching and skipping bootstrap).
        if bool(args.skip_bootstrap):
            print("[runner] skip_bootstrap=true (using existing remote build)", flush=True)
        else:
            for p in pods:
                print(f"[runner] bootstrap_start pod_id={p.pod_id} ssh={p.ssh_user}@{p.ssh_host}:{p.ssh_port}", flush=True)
                _bootstrap_and_build(pod=p, bundle_path=bundle_path, remote_root=str(args.remote_root), with_nccl=want_nccl, pkey=pkey)
                print(f"[runner] bootstrap_done pod_id={p.pod_id}", flush=True)

        # Start relay on party0.
        token = secrets.token_hex(16)
        pod0 = [p for p in pods if int(p.party) == 0 and int(p.idx_in_party) == 0][0]
        # Ensure every pod has the relay token file (uvcc_worker can read it locally).
        for p in pods:
            _write_relay_token_file(pod=p, remote_root=str(args.remote_root), token=token, pkey=pkey)
        print(f"[runner] start_relay pod0_id={pod0.pod_id} ssh={pod0.ssh_user}@{pod0.ssh_host}:{pod0.ssh_port}", flush=True)
        relay_url, relay_token_file = _start_relay(
            pod0=pod0,
            remote_root=str(args.remote_root),
            port=int(args.relay_port),
            token=token,
            lease_s=int(args.relay_lease_s),
            ttl_s=int(args.relay_ttl_s),
            pkey=pkey,
        )

        # Use a fresh relay group id for this run.
        relay_group_id = f"g-native-toy-{secrets.token_hex(8)}"
        sid_job_hex = _parse_sid_job_hex(str(args.sid_job_hex))
        print(f"[runner] relay_ready relay_url={relay_url} relay_group_id={relay_group_id} sid_job={sid_job_hex}", flush=True)

        # Launch a matrix of independent subgroup workers:
        # total per party workers = R*S*T, mapped onto pods_per_party*gpus_per_pod slots.
        R = int(args.replicas)
        S = int(args.stages)
        T = int(args.tp_ranks)
        per_party = R * S * T
        cap = int(args.pods_per_party) * int(args.gpus_per_pod)
        if per_party > cap and not bool(args.oversubscribe):
            raise RuntimeError(
                f"need per-party capacity {per_party} (R*S*T) but have only {cap} (pods_per_party*gpus_per_pod). "
                "Pass --oversubscribe to allow multiple workers to share a GPU slot."
            )
        if per_party > cap and bool(args.oversubscribe):
            wm = str(args.worker_mode or "").strip().lower()
            if wm in ("nccl_smoke", "phase6_step"):
                need = max(R, S, T)
                if cap < need:
                    raise RuntimeError(
                        f"--oversubscribe with nccl_smoke requires at least max(R,S,T) GPU slots per party to avoid NCCL duplicate-GPU errors "
                        f"(need>={need}, have cap={cap}). Reduce R/S/T or increase pods_per_party*gpus_per_pod."
                    )
            print(f"[runner] oversubscribe enabled: per_party={per_party} cap={cap} worker_mode={wm}", flush=True)

        pods_by_party: Dict[int, List[RemotePod]] = {0: [], 1: [], 2: []}
        for p in pods:
            pods_by_party[int(p.party)].append(p)
        for k in pods_by_party:
            pods_by_party[k] = sorted(pods_by_party[k], key=lambda x: int(x.idx_in_party))

        launches: List[WorkerLaunch] = []
        for party in (0, 1, 2):
            for local_rank in range(per_party):
                r = local_rank // (S * T)
                rem = local_rank % (S * T)
                s = rem // T
                t = rem % T
                wm = str(args.worker_mode or "").strip().lower()
                if per_party <= cap:
                    pod_idx = local_rank // int(args.gpus_per_pod)
                    gpu_idx = local_rank % int(args.gpus_per_pod)
                else:
                    # Oversubscribe: map multiple workers onto available (pod,gpu) slots.
                    #
                    # IMPORTANT for nccl_smoke:
                    # NCCL communicators require that each *rank* in a communicator uses a distinct GPU device.
                    # If we map group members onto the same GPU, NCCL will fail with "Duplicate GPU detected".
                    #
                    # The nccl_smoke communicator groups are:
                    # - TP: fixed (r,s), varying t in [0..T-1], size=T
                    # - PP: fixed (r,t), varying s in [0..S-1], size=S
                    # - DP: fixed (s,t), varying r in [0..R-1], size=R
                    #
                    # With cap >= max(R,S,T), the simple mapping slot=(r+s+t)%cap guarantees distinct slots
                    # within each of these groups (because each group varies only one coordinate).
                    if wm in ("nccl_smoke", "phase6_step"):
                        slot = (int(r) + int(s) + int(t)) % int(cap)
                    else:
                        # Default: round-robin by local_rank.
                        slot = int(local_rank) % int(cap)
                    pod_idx = slot // int(args.gpus_per_pod)
                    gpu_idx = slot % int(args.gpus_per_pod)
                pod = pods_by_party[party][pod_idx]
                # Relay group id:
                # - toy_open: group id must be identical across parties for fixed (r,s,t)
                # - nccl_smoke: use a single shared base group id for the whole run (handshake code derives per-group ids)
                if wm in ("nccl_smoke", "phase6_step"):
                    group_id = str(relay_group_id)
                else:
                    group_id = f"g-native-sub-r{r}-s{s}-t{t}-{sid_job_hex[2:10]}"
                launches.append(
                    WorkerLaunch(
                        party=party,
                        replica=int(r),
                        stage=int(s),
                        tp=int(t),
                        local_rank=int(local_rank),
                        pod=pod,
                        gpu_idx=int(gpu_idx),
                        group_id=str(group_id),
                    )
                )

        # IMPORTANT (scale): for phase6_step we must avoid cross-party skew for each (r,s,t) OPEN subgroup.
        # If we launch all party0 workers first, party0 can reach OPEN while party1/2 workers for that subgroup
        # haven't even started yet, causing false OPEN timeouts and tons of unacked relay messages.
        wm_for_order = str(args.worker_mode or "").strip().lower()
        if wm_for_order == "phase6_step":
            by_key: Dict[Tuple[int, int], WorkerLaunch] = {(int(L.party), int(L.local_rank)): L for L in launches}
            launches2: List[WorkerLaunch] = []
            for lr in range(int(per_party)):
                for p in (0, 1, 2):
                    launches2.append(by_key[(int(p), int(lr))])
            launches = launches2

        if str(args.launch_mode).strip().lower() == "pump":
            if per_party != 1:
                raise RuntimeError("launch-mode=pump currently supports only R*S*T == 1 per party")
            # Pump the three parties in short bursts until each prints epoch_root.
            roots_by_party: Dict[int, str] = {}
            for rd in range(int(args.pump_rounds)):
                if len(roots_by_party) == 3:
                    break
                print(f"[runner] pump_round {rd}", flush=True)
                for L in launches:
                    code, out, err = _run_worker_fg_once(
                        pod=L.pod,
                        remote_root=str(args.remote_root),
                        sid_job_hex=str(sid_job_hex),
                        relay_url=str(relay_url),
                        group_id=str(L.group_id),
                        relay_token_file=str(relay_token_file),
                        relay_timeout_s=float(args.worker_relay_timeout_s),
                        topo_r=R,
                        topo_s=S,
                        topo_t=T,
                        debug_relay=bool(args.debug_relay),
                        worker_mode=str(args.worker_mode),
                        microbatches=int(args.microbatches),
                        step_id=int(args.step_id),
                        phase6_enable_dp=bool(args.phase6_enable_dp),
                        phase6_timeout_s=int(args.phase6_timeout_s),
                        timeout_s=int(args.pump_timeout_s),
                        pkey=pkey,
                    )
                    p_log = out_dir / f"pump_p{int(L.party)}_round{int(rd)}.log"
                    p_log.write_text(str(out) + str(err), encoding="utf-8")
                    txt = (str(out) + "\n" + str(err)).strip()
                    if int(L.party) not in roots_by_party and "epoch_root=0x" in txt:
                        # Take the last occurrence.
                        for ln in txt.splitlines()[::-1]:
                            if "epoch_root=0x" in ln:
                                roots_by_party[int(L.party)] = ln.strip()
                                break
                        print(f"[runner] party{int(L.party)} done: {roots_by_party[int(L.party)]}", flush=True)
                time.sleep(0.2)

            if len(roots_by_party) != 3:
                raise RuntimeError(f"pump did not converge: roots_by_party={roots_by_party}")
            roots_path = out_dir / "epoch_roots.txt"
            roots_path.write_text("\n".join([f"party{p}: {roots_by_party[p]}" for p in sorted(roots_by_party)]), encoding="utf-8")
        else:
            # Launch all workers in background and wait for done markers.
            done_paths: List[Tuple[WorkerLaunch, str, str]] = []
            for L in launches:
                print(
                    f"[runner] launch_worker party={L.party} r={L.replica} s={L.stage} t={L.tp} pod_id={L.pod.pod_id} gpu={L.gpu_idx} group_id={L.group_id}",
                    flush=True,
                )
                log_path, done = _launch_worker_bg(
                    pod=L.pod,
                    launch=L,
                    remote_root=str(args.remote_root),
                    sid_job_hex=sid_job_hex,
                    relay_url=relay_url,
                    relay_token_file=relay_token_file,
                    relay_timeout_s=float(args.worker_relay_timeout_s),
                    debug_relay=bool(args.debug_relay),
                    worker_mode=str(args.worker_mode),
                    topo_r=R,
                    topo_s=S,
                    topo_t=T,
                    microbatches=int(args.microbatches),
                    step_id=int(args.step_id),
                    phase6_enable_dp=bool(args.phase6_enable_dp),
                    phase6_timeout_s=int(args.phase6_timeout_s),
                    pkey=pkey,
                )
                done_paths.append((L, log_path, done))

            for L, log_path, done in done_paths:
                print(f"[runner] wait_done pod_id={L.pod.pod_id} done_path={done}", flush=True)
                _wait_done(pod=L.pod, done_path=done, pkey=pkey, timeout_s=3600)

            # Collect worker logs. For toy_open and phase6_step we also compute transcript-of-transcripts audit root.
            logs_dir = out_dir / "remote_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            roots: List[Dict[str, Any]] = []
            wm_run = str(args.worker_mode or "").strip().lower()
            for L, log_path, done in done_paths:
                remote_dir = str(Path(log_path).parent)
                ssh = ssh_connect_with_retries(hostname=L.pod.ssh_host, port=L.pod.ssh_port, username=L.pod.ssh_user, pkey=pkey, timeout_s=900)
                try:
                    local_prefix = logs_dir / f"p{int(L.party)}_r{int(L.replica)}_s{int(L.stage)}_t{int(L.tp)}"
                    # Best-effort download of key files.
                    for fname in ("run.log", "exit_code.txt", "done.txt", "cmd.txt"):
                        rp = f"{remote_dir}/{fname}"
                        lp = str(local_prefix) + f".{fname}"
                        try:
                            sftp_get_file(ssh, remote_path=rp, local_path=lp)
                        except Exception:
                            pass

                    epoch_root_hex: Optional[str] = None
                    if wm_run in ("toy_open", "phase6_step"):
                        try:
                            txt = Path(str(local_prefix) + ".run.log").read_text(encoding="utf-8", errors="replace")
                            for ln in txt.splitlines()[::-1]:
                                if "epoch_root=0x" in ln:
                                    epoch_root_hex = ln.strip().split("epoch_root=")[-1].strip()
                                    break
                        except Exception:
                            epoch_root_hex = None
                        if not epoch_root_hex:
                            # Fallback: grep remotely.
                            cmd = f"grep -a 'epoch_root=0x' -n {shlex.quote(str(log_path))} | tail -n 1 || true"
                            _, out_g, _ = ssh_exec(ssh, f"bash -lc {shlex.quote(cmd)}", timeout_s=30)
                            if "epoch_root=0x" in str(out_g):
                                epoch_root_hex = str(out_g).strip().split("epoch_root=")[-1].strip()

                    roots.append(
                        dict(
                            party=int(L.party),
                            replica=int(L.replica),
                            stage=int(L.stage),
                            tp=int(L.tp),
                            local_rank=int(L.local_rank),
                            pod_id=str(L.pod.pod_id),
                            ssh_host=str(L.pod.ssh_host),
                            remote_log=str(log_path),
                            remote_done=str(done),
                            epoch_root_hex=str(epoch_root_hex) if epoch_root_hex else None,
                        )
                    )
                finally:
                    ssh.close()

            (out_dir / "roots_by_coord.json").write_text(json.dumps(roots, indent=2, sort_keys=True), encoding="utf-8")

            if wm_run in ("toy_open", "phase6_step"):
                missing = [r for r in roots if not r.get("epoch_root_hex")]
                if missing:
                    raise RuntimeError(f"missing epoch_root for {len(missing)} workers; see {logs_dir}")

                sid_job32 = bytes.fromhex(str(sid_job_hex)[2:])
                step_id = int(args.step_id)
                roots_by_rep: Dict[int, List[Dict[str, Any]]] = {}
                for r in roots:
                    roots_by_rep.setdefault(int(r["replica"]), []).append(r)

                replica_roots: List[Dict[str, Any]] = []
                replica_roots32: List[bytes] = []
                for rid in sorted(roots_by_rep.keys()):
                    xs = sorted(roots_by_rep[int(rid)], key=lambda z: (int(z["party"]), int(z["stage"]), int(z["tp"])))
                    subs32 = [bytes.fromhex(str(z["epoch_root_hex"]).removeprefix("0x")) for z in xs]
                    sid_rep32 = _sid_replica_bytes(sid_job32=sid_job32, replica_id=int(rid))
                    rr32 = _replica_root_bytes(sid_rep32=sid_rep32, step_id=step_id, roots32_sorted=subs32)
                    replica_roots.append(dict(replica=int(rid), sid_rep_hex="0x" + sid_rep32.hex(), replica_root_hex="0x" + rr32.hex()))
                    replica_roots32.append(rr32)

                global_root32 = _global_root_bytes(sid_job32=sid_job32, step_id=step_id, replica_roots32=replica_roots32)
                (out_dir / "audit_bundle.json").write_text(
                    json.dumps(
                        dict(
                            sid_job_hex=str(sid_job_hex),
                            step_id=int(step_id),
                            topology=dict(R=int(R), S=int(S), T=int(T), M=int(args.microbatches)),
                            global_root_hex="0x" + global_root32.hex(),
                            replica_roots=replica_roots,
                        ),
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            else:
                # nccl_smoke: no transcript roots; we only validate completion via done/exit_code.
                (out_dir / "nccl_smoke_done.json").write_text(
                    json.dumps(
                        dict(
                            sid_job_hex=str(sid_job_hex),
                            step_id=int(args.step_id),
                            topology=dict(R=int(R), S=int(S), T=int(T), M=int(args.microbatches)),
                            pods=len(pods),
                            ok=True,
                        ),
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

        out_txt = out_dir / "toy_open_matrix_done.txt"
        out_txt.write_text(f"ok sid_job={sid_job_hex} relay_url={relay_url} pods={len(pods)}\n", encoding="utf-8")
        print(out_txt)
        return 0
    finally:
        # Safety: never delete pods in attach mode unless explicitly requested (keep-pods is implied).
        if (not bool(args.keep_pods)) and (not attach_mode):
            for p in pods:
                try:
                    print(f"[runner] delete_pod pod_id={p.pod_id}", flush=True)
                    prime.delete_pod(p.pod_id)
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())


