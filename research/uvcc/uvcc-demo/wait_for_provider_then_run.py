from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _provider_offer_count(*, provider_type: str, prime_api_key_path: Path) -> int:
    """
    Best-effort: poll Prime /availability and count offers by provider.
    We intentionally do NOT log the api key.
    """
    import requests

    key = Path(prime_api_key_path).read_text(encoding="utf-8", errors="ignore").strip()
    base = os.environ.get("PRIME_API_BASE", "https://api.primeintellect.ai/api/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.get(base + "/availability", headers=headers, timeout=60)
    r.raise_for_status()

    obj = r.json()
    rows = []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                rows.extend([x for x in v if isinstance(x, dict)])
    elif isinstance(obj, list):
        rows = [x for x in obj if isinstance(x, dict)]

    c = Counter(str(x.get("provider") or x.get("providerType") or x.get("provider_type") or "").strip().lower() for x in rows)
    return int(c.get(str(provider_type).strip().lower(), 0))


def _start_watchdog(*, out_dir: Path, ssh_key_path: Path, interval_s: float) -> Optional[int]:
    script = Path(__file__).resolve().parent / "watch_prime_3node_status.py"
    live_keep = out_dir / "live_keep"
    live_keep.mkdir(parents=True, exist_ok=True)
    stdout_path = live_keep / "watch_stdout.log"
    pid_path = live_keep / "watch.pid"

    cmd = [
        sys.executable,
        str(script),
        "--out",
        str(out_dir),
        "--ssh-key-path",
        str(ssh_key_path),
        "--interval-s",
        str(max(5.0, float(interval_s))),
    ]
    with open(stdout_path, "ab") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        pid_path.write_text(str(int(p.pid)) + "\n", encoding="utf-8")
    except Exception:
        pass
    return int(p.pid)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="wait_for_provider_then_run")
    ap.add_argument("--provider-type", required=True, help="Prime provider type to wait for (e.g. datacrunch)")
    ap.add_argument("--out", required=True, help="UVCC run out dir")
    ap.add_argument("--prime-api-key-path", default="~/.uvcc/prime_api_key.txt", help="Path to file containing Prime API key")
    ap.add_argument("--poll-interval-s", type=float, default=60.0, help="Poll interval for provider availability (seconds, default: 60)")
    ap.add_argument("--max-wait-s", type=float, default=0.0, help="Max wait seconds (0 = wait forever)")
    ap.add_argument("--start-watchdog", choices=["true", "false"], default="true", help="Start watchdog to keep append-only recorder alive")
    ap.add_argument("--ssh-key-path", default="~/.ssh/uvcc_prime_runner_ed25519", help="SSH key path for watchdog recorder restarts")
    ap.add_argument("--watchdog-interval-s", type=float, default=30.0, help="Watchdog interval seconds (default: 30)")
    ap.add_argument("--bundler-output", default="all_logs_explained.md", help="Bundled explained log filename (default: all_logs_explained.md)")
    ap.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Args forwarded to run_prime_3node.py; prefix with `--` (example: -- --job-json ... --party-log-level trace)",
    )
    args = ap.parse_args(argv)

    provider_type = str(args.provider_type).strip().lower()
    out_dir = Path(str(args.out)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prime_api_key_path = Path(str(args.prime_api_key_path)).expanduser().resolve()
    if not prime_api_key_path.exists():
        raise SystemExit(f"prime api key file not found: {prime_api_key_path}")

    # Optional watchdog.
    watch_pid: Optional[int] = None
    if str(args.start_watchdog).lower() == "true":
        ssh_key_path = Path(str(args.ssh_key_path)).expanduser().resolve()
        if ssh_key_path.exists():
            try:
                watch_pid = _start_watchdog(out_dir=out_dir, ssh_key_path=ssh_key_path, interval_s=float(args.watchdog_interval_s))
                print(f"[{_now_iso_utc()}] watchdog_started pid={watch_pid} out_dir={out_dir}", flush=True)
            except Exception as exc:
                print(f"[{_now_iso_utc()}] watchdog_start_failed error={exc}", flush=True)
        else:
            print(f"[{_now_iso_utc()}] watchdog_skipped ssh_key_not_found={ssh_key_path}", flush=True)

    # Wait for provider availability.
    poll_s = max(5.0, float(args.poll_interval_s))
    deadline = (time.time() + float(args.max_wait_s)) if float(args.max_wait_s) > 0 else None
    while True:
        try:
            n = _provider_offer_count(provider_type=provider_type, prime_api_key_path=prime_api_key_path)
            print(f"[{_now_iso_utc()}] availability provider={provider_type} offers={n}", flush=True)
            if int(n) > 0:
                break
        except Exception as exc:
            print(f"[{_now_iso_utc()}] availability_check_failed provider={provider_type} error={exc}", flush=True)

        if deadline is not None and time.time() >= deadline:
            print(f"[{_now_iso_utc()}] wait_timeout provider={provider_type} max_wait_s={args.max_wait_s}", flush=True)
            return 2
        time.sleep(poll_s)

    # Launch the runner.
    repo = _repo_root()
    runner = repo / "research" / "uvcc" / "uvcc-demo" / "run_prime_3node.py"
    cmd = [
        sys.executable,
        str(runner),
        "--out",
        str(out_dir),
        "--provider-type",
        str(provider_type),
        "--prime-api-key-path",
        str(prime_api_key_path),
    ]
    # Forward remaining args (strip a leading "--" if present).
    fwd = list(args.runner_args or [])
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]
    cmd += fwd

    print(f"[{_now_iso_utc()}] runner_start cmd={' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(repo))
    print(f"[{_now_iso_utc()}] runner_exit rc={rc}", flush=True)

    # Bundle logs (even on failure) for single-file auditing.
    bundler = repo / "research" / "uvcc" / "uvcc-demo" / "bundle_logs_explained.py"
    bundle_cmd = [sys.executable, str(bundler), "--out", str(out_dir), "--output", str(args.bundler_output)]
    print(f"[{_now_iso_utc()}] bundler_start cmd={' '.join(bundle_cmd)}", flush=True)
    brc = subprocess.call(bundle_cmd, cwd=str(repo))
    print(f"[{_now_iso_utc()}] bundler_exit rc={brc}", flush=True)

    # Stop watchdog (best-effort).
    if watch_pid is not None:
        try:
            os.kill(int(watch_pid), signal.SIGTERM)
            print(f"[{_now_iso_utc()}] watchdog_stopped pid={watch_pid}", flush=True)
        except Exception:
            pass

    return int(rc) if int(rc) != 0 else int(brc)


if __name__ == "__main__":
    raise SystemExit(main())


