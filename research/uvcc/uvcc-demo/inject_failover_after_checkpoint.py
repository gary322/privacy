from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Make `uvcc_client` importable when running from a repo checkout (no venv / no pip install).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3] if len(_THIS_FILE.parents) >= 4 else _THIS_FILE.parent
_UVCC_CLIENT_DIR = _REPO_ROOT / "research" / "uvcc" / "uvcc-client"
if _UVCC_CLIENT_DIR.exists() and str(_UVCC_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_UVCC_CLIENT_DIR))

from uvcc_client.ssh_runner import (  # noqa: E402
    load_private_key_from_file,
    ssh_connect_with_retries,
    ssh_exec,
)


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _wait_file(path: Path, *, timeout_s: int = 3600, poll_s: float = 1.0) -> None:
    deadline = time.time() + max(1, int(timeout_s))
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(max(0.1, float(poll_s)))
    raise RuntimeError(f"timed out waiting for file: {path}")


def _find_node_ssh(out_dir: Path, *, party_id: int) -> Tuple[str, int, str]:
    """
    Prefer the live recorder's recorder_meta.json (has host/port/user).
    Falls back to (host from prime_pods_active, port=1234, user=root) if needed.
    """
    meta_path = out_dir / "live_keep" / "recorder_meta.json"
    if meta_path.exists():
        meta = _read_json(meta_path)
        nodes = meta.get("nodes")
        if isinstance(nodes, list):
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                if int(n.get("party_id", -1)) != int(party_id):
                    continue
                host = str(n.get("host") or "").strip()
                port = int(n.get("port") or 0)
                user = str(n.get("user") or "root").strip() or "root"
                if host and 0 < port <= 65535:
                    return host, port, user

    # Fallback: parse prime_pods_active from run_full.jsonl
    jsonl = out_dir / "run_full.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            if '"event":"prime_pods_active"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fields = obj.get("fields") if isinstance(obj, dict) else None
            if not isinstance(fields, dict):
                continue
            hosts = fields.get("ssh_hosts")
            if isinstance(hosts, list) and len(hosts) > int(party_id):
                host = str(hosts[int(party_id)]).strip()
                if host:
                    return host, 1234, "root"

    raise RuntimeError("could not determine party SSH details (missing recorder_meta.json and prime_pods_active)")


def _log_contains_checkpoint(path: Path, *, step: int) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    # The per-party log is JSON-lines with occasional non-json warnings.
    # We match by substrings to keep it robust to key ordering.
    return ('"event":"checkpoint_written"' in text) and (f'"step":{int(step)}' in text)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="inject_failover_after_checkpoint")
    ap.add_argument("--out", required=True, help="UVCC run out dir (where run_full.jsonl exists)")
    ap.add_argument("--ssh-key-path", required=True, help="SSH private key to reach pods")
    ap.add_argument("--party-id", type=int, default=2, choices=[1, 2], help="Which non-relay party to kill (default: 2)")
    ap.add_argument("--after-step", type=int, default=0, help="Kill after this checkpoint step is written (default: 0)")
    ap.add_argument("--timeout-s", type=int, default=7200, help="Max seconds to wait for checkpoint (default: 7200)")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out)).expanduser().resolve()
    if not out_dir.exists():
        raise SystemExit(f"out dir does not exist: {out_dir}")

    run_jsonl = out_dir / "run_full.jsonl"
    _wait_file(run_jsonl, timeout_s=600, poll_s=0.5)

    # Wait for recorder meta so we know correct host/port/user (and so live_keep logs exist).
    meta_path = out_dir / "live_keep" / "recorder_meta.json"
    _wait_file(meta_path, timeout_s=1800, poll_s=1.0)

    pid = int(args.party_id)
    step = int(args.after_step)

    party_log = out_dir / "live_keep" / f"party_p{pid}_run.log"
    deadline = time.time() + max(1, int(args.timeout_s))
    while time.time() < deadline:
        if _log_contains_checkpoint(party_log, step=step):
            break
        time.sleep(1.0)
    else:
        raise RuntimeError(f"timed out waiting for checkpoint_written step={step} in {party_log}")

    host, port, user = _find_node_ssh(out_dir, party_id=int(pid))
    print(f"[{_now_iso_utc()}] injecting failover: killing party={pid} on {user}@{host}:{port} after checkpoint step={step}")

    pkey = load_private_key_from_file(str(args.ssh_key_path))
    ssh = ssh_connect_with_retries(hostname=str(host), port=int(port), username=str(user), pkey=pkey, timeout_s=60)
    try:
        # Find the party process PID. Use pgrep if available; fallback to ps|grep.
        find_pid_cmd = (
            "set -euo pipefail; "
            f"PID=$(pgrep -f {shlex_quote(f'uvcc_client run-party-train.*--party-id {pid}')} | head -n 1 || true); "
            "if [ -z \"$PID\" ]; then "
            f"  PID=$(ps aux | grep -F {shlex_quote('uvcc_client run-party-train')} | grep -F {shlex_quote(f'--party-id {pid}')} | grep -v grep | awk '{{print $2}}' | head -n 1 || true); "
            "fi; "
            "echo ${PID:-}"
        )
        code, out, err = ssh_exec(ssh, f"bash -lc {shlex_quote(find_pid_cmd)}", timeout_s=20)
        pid_s = str(out or "").strip()
        if code != 0 or not pid_s.isdigit():
            raise RuntimeError(f"failed to find party process pid (code={code}): out={out!r} err={err!r}")
        proc_pid = int(pid_s)

        # Derive the remote --out dir from process args, and ensure run.pid matches.
        derive_out_cmd = (
            "set -euo pipefail; "
            f"PID={int(proc_pid)}; "
            "ARGS=$(ps -o args= -p $PID || true); "
            "OUT=$(echo \"$ARGS\" | sed -n 's/.*--out \\([^ ]*\\).*/\\1/p' | head -n 1); "
            f"if [ -z \"$OUT\" ]; then OUT=/root/uvcc/out_party_{int(pid)}; fi; "
            "mkdir -p \"$OUT\"; "
            "echo $PID > \"$OUT/run.pid\"; "
            "echo \"$OUT\""
        )
        code2, out2, err2 = ssh_exec(ssh, f"bash -lc {shlex_quote(derive_out_cmd)}", timeout_s=20)
        out_remote = str(out2 or "").strip()
        if code2 != 0 or not out_remote:
            raise RuntimeError(f"failed to derive remote out dir (code={code2}): out={out2!r} err={err2!r}")

        kill_cmd = f"kill -9 {int(proc_pid)} >/dev/null 2>&1 || true; echo killed_pid={int(proc_pid)} out_dir={out_remote}"
        code3, out3, err3 = ssh_exec(ssh, f"bash -lc {shlex_quote(kill_cmd)}", timeout_s=20)
        print(f"[{_now_iso_utc()}] {str(out3 or '').strip()} {str(err3 or '').strip()}")
        if code3 != 0:
            raise RuntimeError(f"kill command failed (code={code3}): out={out3!r} err={err3!r}")
    finally:
        try:
            ssh.close()
        except Exception:
            pass

    return 0


def shlex_quote(s: str) -> str:
    # Tiny local quote helper to avoid importing shlex in remote command construction.
    t = str(s)
    if not t:
        return "''"
    if all(c.isalnum() or c in "._/-=" for c in t):
        return t
    return "'" + t.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())


