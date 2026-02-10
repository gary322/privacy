from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `uvcc_client` importable when running from a repo checkout (no venv / no pip install).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3] if len(_THIS_FILE.parents) >= 4 else _THIS_FILE.parent
_UVCC_CLIENT_DIR = _REPO_ROOT / "research" / "uvcc" / "uvcc-client"
if _UVCC_CLIENT_DIR.exists() and str(_UVCC_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_UVCC_CLIENT_DIR))

from uvcc_client.ssh_runner import (
    load_private_key_from_file,
    sftp_get_file,
    ssh_connect_with_retries,
    ssh_exec,
)


@dataclass(frozen=True)
class NodeV1:
    party_id: int
    host: str
    port: int
    user: str = "root"


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_node(s: str) -> NodeV1:
    """
    Format: pid,host,port[,user]
    Example: 0,86.38.238.26,1234
    """
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) not in (3, 4):
        raise ValueError("node must be pid,host,port[,user]")
    pid = int(parts[0])
    host = str(parts[1])
    port = int(parts[2])
    user = str(parts[3]) if len(parts) == 4 else "root"
    if pid not in (0, 1, 2):
        raise ValueError("party_id must be 0..2")
    if not host:
        raise ValueError("host required")
    if port <= 0 or port > 65535:
        raise ValueError("port invalid")
    return NodeV1(party_id=pid, host=host, port=port, user=user)


def _atomic_write_text(path: Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(p))


def _load_json(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _state_key(host: str, pid: int, remote_path: str) -> str:
    return f"{host}|p{int(pid)}|{remote_path}"


def _local_part_path(base: Path, *, part: int) -> Path:
    """
    For a base path "foo.ext", return:
      part=0 -> foo.ext
      part=1 -> foo.part1.ext
      part=2 -> foo.part2.ext
    """
    base = Path(base)
    if part <= 0:
        return base
    return base.with_name(f"{base.stem}.part{int(part)}{base.suffix}")


def _append_remote_delta(
    ssh: Any,
    *,
    sftp: Any = None,
    host: str,
    party_id: int,
    remote_path: str,
    local_base_path: Path,
    state: Dict[str, Any],
    max_read_bytes: int = 4 * 1024 * 1024,
) -> Tuple[bool, str]:
    """
    Append-only mirror:
      - Reads remote file from last known byte offset and appends to local.
      - If remote size shrinks, starts a new local part file and continues from 0.
    Returns (changed, note)
    """
    key = _state_key(host, int(party_id), str(remote_path))
    cur = state.get(key) if isinstance(state.get(key), dict) else {}
    offset = int(cur.get("offset") or 0)
    part = int(cur.get("part") or 0)
    lp = Path(cur.get("local_path") or _local_part_path(local_base_path, part=part))

    close_sftp = False
    if sftp is None:
        sftp = ssh.open_sftp()
        close_sftp = True
    try:
        st = sftp.stat(str(remote_path))
        size = int(getattr(st, "st_size", 0) or 0)
        if size < offset:
            # Remote file truncated/replaced. Preserve local history by starting a new local part file.
            part += 1
            offset = 0
            lp = _local_part_path(local_base_path, part=part)
            state[key] = {"offset": 0, "part": part, "local_path": str(lp)}
            note = "remote_truncated_new_part"
        else:
            state[key] = {"offset": offset, "part": part, "local_path": str(lp)}
            note = ""

        if size == offset:
            return (False, note or "no_change")

        to_read = min(max(0, size - offset), int(max_read_bytes))
        if to_read <= 0:
            return (False, note or "no_change")

        with sftp.file(str(remote_path), mode="rb") as rf:
            rf.seek(offset)
            data = rf.read(to_read)
        if not isinstance(data, (bytes, bytearray)) or not data:
            return (False, note or "no_change")

        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "ab") as lf:
            lf.write(bytes(data))

        offset += len(data)
        state[key]["offset"] = offset
        return (True, note or "appended")
    finally:
        if close_sftp:
            try:
                sftp.close()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="record_live_logs_append")
    ap.add_argument("--out", required=True, help="Local UVCC run out dir to write append-only copies into")
    ap.add_argument("--ssh-key-path", required=True, help="SSH private key path")
    ap.add_argument("--interval-s", type=float, default=10.0, help="Polling interval seconds (default: 10)")
    ap.add_argument("--node", action="append", default=[], help="Repeat: pid,host,port[,user]")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out)).expanduser().resolve()
    live_dir = out_dir / "live_keep"
    live_dir.mkdir(parents=True, exist_ok=True)
    # Private mirror directory (DO NOT SHARE): may contain checkpoint shares used for failover/resume.
    private_dir = out_dir / "private_keep"
    private_dir.mkdir(parents=True, exist_ok=True)

    nodes: List[NodeV1] = [_parse_node(x) for x in (args.node or [])]
    if len(nodes) != 3:
        raise SystemExit("must pass exactly 3 --node entries (party 0/1/2)")
    nodes = sorted(nodes, key=lambda n: int(n.party_id))

    interval_s = max(1.0, float(args.interval_s))
    pkey = load_private_key_from_file(str(args.ssh_key_path))

    meta = {
        "ts": _now_iso_utc(),
        "out_dir": str(out_dir),
        "interval_s": float(interval_s),
        "nodes": [{"party_id": n.party_id, "host": n.host, "port": n.port, "user": n.user} for n in nodes],
        "mode": "append_only_parts",
    }
    _atomic_write_text(live_dir / "recorder_meta.json", json.dumps(meta, sort_keys=True, indent=2) + "\n")

    state_path = live_dir / "recorder_state.json"
    state: Dict[str, Any] = _load_json(state_path)

    log_path = live_dir / "recorder.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{_now_iso_utc()}] recorder append-only starting\n")

    while True:
        loop_ts = _now_iso_utc()
        # Best-effort: one ssh per node per interval.
        for n in nodes:
            try:
                ssh = ssh_connect_with_retries(hostname=n.host, port=n.port, username=n.user, pkey=pkey, timeout_s=60)
            except Exception as exc:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{loop_ts}] WARN ssh_connect failed party={n.party_id} {n.user}@{n.host}:{n.port}: {exc}\n")
                continue

            try:
                base = f"/root/uvcc/out_party_{int(n.party_id)}"
                # IMPORTANT: keep only one SFTP session per node per tick to avoid FD/channel leaks.
                sftp = None
                try:
                    sftp = ssh.open_sftp()
                except Exception:
                    sftp = None

                # Append-only mirrors (never shrink).
                for remote, local_name in [
                    (f"{base}/run.log", f"party_p{int(n.party_id)}_run.log"),
                    # Live transcript dump (append-only) so partial progress is preserved.
                    (f"{base}/transcript_v1_live.jsonl", f"party_p{int(n.party_id)}_transcript.jsonl"),
                    (f"{base}/gpu_telemetry.csv", f"gpu_telemetry_remote_p{int(n.party_id)}.csv"),
                ]:
                    try:
                        changed, note = _append_remote_delta(
                            ssh,
                            sftp=sftp,
                            host=str(n.host),
                            party_id=int(n.party_id),
                            remote_path=str(remote),
                            local_base_path=live_dir / local_name,
                            state=state,
                        )
                        if changed:
                            pass
                    except Exception:
                        # Missing file expected mid-run (transcript created at end).
                        pass

                # Private append-only mirrors (checkpoint shares for failover/resume).
                for remote, local_name in [
                    (f"{base}/private/checkpoints_W.jsonl", f"checkpoints_W_p{int(n.party_id)}.jsonl"),
                ]:
                    try:
                        _append_remote_delta(
                            ssh,
                            sftp=sftp,
                            host=str(n.host),
                            party_id=int(n.party_id),
                            remote_path=str(remote),
                            local_base_path=private_dir / local_name,
                            state=state,
                        )
                    except Exception:
                        pass

                # Small snapshots (overwrite is fine, used for liveness or "latest result").
                for remote, local_name in [
                    (f"{base}/run.pid", f"party_p{int(n.party_id)}_run.pid"),
                    (f"{base}/result.json", f"party_p{int(n.party_id)}_result_latest.json"),
                ]:
                    try:
                        if sftp is not None:
                            lp = Path(str(live_dir / local_name)).expanduser().resolve()
                            lp.parent.mkdir(parents=True, exist_ok=True)
                            sftp.get(str(remote), str(lp))
                        else:
                            # Fallback (opens/closes its own SFTP session).
                            sftp_get_file(ssh, remote_path=str(remote), local_path=str(live_dir / local_name))
                    except Exception:
                        pass

                # Telemetry poll (doesn't depend on a remote background nvidia-smi process).
                try:
                    cmd = (
                        "bash -lc "
                        + repr(
                            "nvidia-smi "
                            "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu "
                            "--format=csv,noheader,nounits"
                        )
                    )
                    code, out, _err = ssh_exec(ssh, cmd, timeout_s=10)
                    if code == 0 and str(out).strip():
                        p = live_dir / f"gpu_telemetry_polled_p{int(n.party_id)}.csv"
                        with open(p, "a", encoding="utf-8") as f:
                            for line in str(out).splitlines():
                                if line.strip():
                                    f.write(line.rstrip("\n") + "\n")
                except Exception:
                    pass

                # Relay log (node0 only) append-only.
                if int(n.party_id) == 0:
                    try:
                        _append_remote_delta(
                            ssh,
                            sftp=sftp,
                            host=str(n.host),
                            party_id=int(n.party_id),
                            remote_path="/root/uvcc/relay.log",
                            local_base_path=live_dir / "relay_node0.log",
                            state=state,
                        )
                    except Exception:
                        pass
            finally:
                try:
                    if sftp is not None:
                        sftp.close()
                except Exception:
                    pass
                try:
                    ssh.close()
                except Exception:
                    pass

        try:
            _atomic_write_text(state_path, json.dumps(state, sort_keys=True, indent=2) + "\n")
        except OSError as exc:
            # Don't crash the recorder if the system is temporarily under FD pressure.
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{loop_ts}] WARN state_write_failed: {exc}\n")
            except Exception:
                pass
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{loop_ts}] tick ok\n")
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())


