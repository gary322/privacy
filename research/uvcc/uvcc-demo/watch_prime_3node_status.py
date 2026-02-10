from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_int(path: Path) -> Optional[int]:
    try:
        s = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        return int(s) if s else None
    except Exception:
        return None


def _tail_lines(path: Path, *, max_bytes: int = 200_000) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("rb") as f:
            try:
                f.seek(-int(max_bytes), os.SEEK_END)
            except Exception:
                f.seek(0)
            data = f.read()
        return data.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return []


def _last_step_events(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (last_step_done_event, last_step_start_event) from a JSONL log file.
    """
    lines = _tail_lines(path)
    last_done: Optional[Dict[str, Any]] = None
    last_start: Optional[Dict[str, Any]] = None
    for line in reversed(lines):
        if last_done is None and '"event":"step_done"' in line:
            try:
                last_done = json.loads(line)
            except Exception:
                last_done = {"_raw": line}
        if last_start is None and '"event":"step_start"' in line:
            try:
                last_start = json.loads(line)
            except Exception:
                last_start = {"_raw": line}
        if last_done is not None and last_start is not None:
            break
    return last_done, last_start


def _fmt_step(ev: Optional[Dict[str, Any]]) -> str:
    if not ev:
        return "-"
    try:
        step = ev.get("fields", {}).get("step")
        ts = ev.get("ts")
        return f"{step}@{ts}" if ts else str(step)
    except Exception:
        return "?"


def _start_recorder(out_dir: Path, *, ssh_key_path: Path, recorder_meta_path: Path, recorder_pid_path: Path, stdout_path: Path) -> Optional[int]:
    try:
        meta = json.loads(Path(recorder_meta_path).read_text(encoding="utf-8"))
        interval_s = float(meta.get("interval_s") or 10.0)
        nodes = meta.get("nodes") or []
    except Exception:
        return None

    if not isinstance(nodes, list) or len(nodes) != 3:
        return None

    script = Path(__file__).resolve().parent / "record_live_logs_append.py"
    cmd = [
        sys.executable,
        str(script),
        "--out",
        str(out_dir),
        "--ssh-key-path",
        str(ssh_key_path),
        "--interval-s",
        str(interval_s),
    ]
    for n in nodes:
        try:
            pid = int(n.get("party_id"))
            host = str(n.get("host"))
            port = int(n.get("port"))
            user = str(n.get("user") or "root")
        except Exception:
            continue
        cmd += ["--node", f"{pid},{host},{port},{user}"]

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "ab") as out_f:
        p = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        recorder_pid_path.write_text(str(int(p.pid)) + "\n", encoding="utf-8")
    except Exception:
        pass
    return int(p.pid)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="watch_prime_3node_status")
    ap.add_argument("--out", required=True, help="UVCC run out dir")
    ap.add_argument("--ssh-key-path", default="", help="If set, can restart append-only recorder when it dies")
    ap.add_argument("--interval-s", type=float, default=30.0, help="Status polling interval seconds (default: 30)")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out)).expanduser().resolve()
    live_keep = out_dir / "live_keep"
    live_keep.mkdir(parents=True, exist_ok=True)

    runner_pid_path = out_dir / "runner.pid"
    recorder_pid_path = live_keep / "recorder.pid"
    recorder_meta_path = live_keep / "recorder_meta.json"
    recorder_stdout_path = live_keep / "recorder_stdout.log"

    status_log = live_keep / "watch_status.log"
    interval_s = max(5.0, float(args.interval_s))
    ssh_key_path = Path(str(args.ssh_key_path)).expanduser().resolve() if str(args.ssh_key_path).strip() else None

    with open(status_log, "a", encoding="utf-8") as f:
        f.write(f"[{_now_iso_utc()}] watch_status starting out_dir={out_dir}\n")

    while True:
        ts = _now_iso_utc()

        runner_pid = _read_int(runner_pid_path)
        runner_alive = bool(runner_pid and _pid_alive(runner_pid))

        recorder_pid = _read_int(recorder_pid_path)
        recorder_alive = bool(recorder_pid and _pid_alive(recorder_pid))

        if (not recorder_alive) and ssh_key_path and recorder_meta_path.exists():
            new_pid = _start_recorder(
                out_dir,
                ssh_key_path=ssh_key_path,
                recorder_meta_path=recorder_meta_path,
                recorder_pid_path=recorder_pid_path,
                stdout_path=recorder_stdout_path,
            )
            recorder_pid = new_pid
            recorder_alive = bool(new_pid and _pid_alive(new_pid))

        steps: Dict[str, str] = {}
        for p in (0, 1, 2):
            logp = live_keep / f"party_p{p}_run.log"
            done_ev, start_ev = _last_step_events(logp)
            steps[f"p{p}_done"] = _fmt_step(done_ev)
            steps[f"p{p}_start"] = _fmt_step(start_ev)

        msg = {
            "ts": ts,
            "runner_pid": runner_pid,
            "runner_alive": runner_alive,
            "recorder_pid": recorder_pid,
            "recorder_alive": recorder_alive,
            **steps,
        }
        with open(status_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, sort_keys=True) + "\n")

        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())


