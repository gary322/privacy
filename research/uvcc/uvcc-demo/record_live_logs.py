from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from uvcc_client.ssh_runner import load_private_key_from_file, sftp_get_file, ssh_connect_with_retries


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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="record_live_logs")
    ap.add_argument("--out", required=True, help="Local UVCC run out dir to write live copies into")
    ap.add_argument("--ssh-key-path", required=True, help="SSH private key path")
    ap.add_argument("--interval-s", type=float, default=15.0, help="Polling interval seconds (default: 15)")
    ap.add_argument("--node", action="append", default=[], help="Repeat: pid,host,port[,user]")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out)).expanduser().resolve()
    live_dir = out_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

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
    }
    _atomic_write_text(live_dir / "recorder_meta.json", json.dumps(meta, sort_keys=True, indent=2) + "\n")

    log_path = live_dir / "recorder.log"
    _atomic_write_text(log_path, f"[{_now_iso_utc()}] recorder starting\n")

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
                # Party logs
                for remote, local_name in [
                    (f"{base}/run.log", f"party_p{int(n.party_id)}_run.log"),
                    (f"{base}/run.pid", f"party_p{int(n.party_id)}_run.pid"),
                    (f"{base}/result.json", f"party_p{int(n.party_id)}_result.json"),
                    (f"{base}/transcript_v1.jsonl", f"party_p{int(n.party_id)}_transcript.jsonl"),
                    (f"{base}/gpu_telemetry.csv", f"gpu_telemetry_p{int(n.party_id)}.csv"),
                ]:
                    lp = live_dir / local_name
                    try:
                        sftp_get_file(ssh, remote_path=str(remote), local_path=str(lp))
                    except Exception:
                        # Missing file is expected mid-run (result/transcript created at end).
                        pass

                # Relay log (node0 only)
                if int(n.party_id) == 0:
                    try:
                        sftp_get_file(ssh, remote_path="/root/uvcc/relay.log", local_path=str(live_dir / "relay_node0.log"))
                    except Exception:
                        pass
            finally:
                try:
                    ssh.close()
                except Exception:
                    pass

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{loop_ts}] tick ok\n")
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())


