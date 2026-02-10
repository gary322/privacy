from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _post_json(url: str, obj: dict) -> tuple[int, dict]:
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return int(e.code), json.loads(body)


def _get_json(url: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        return int(resp.status), json.loads(resp.read().decode("utf-8"))


def test_relay_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    relay_py = repo_root / "research" / "uvcc" / "uvcc-relay" / "relay_server.py"
    assert relay_py.exists()

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "relay.sqlite")
        p = subprocess.Popen(
            [
                sys.executable,
                str(relay_py),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--db",
                db,
                "--require-token",
                "false",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(200):
                try:
                    st, j = _get_json(base + "/healthz")
                    if st == 200 and j.get("ok") is True:
                        break
                except Exception:
                    time.sleep(0.02)
            else:
                raise RuntimeError("relay not healthy")

            payload = b"hello"
            payload_b64 = base64.b64encode(payload).decode("ascii")
            st, j = _post_json(
                base + "/enqueue",
                {"group_id": "g", "msg_id": "m1", "sender": 0, "receiver": 1, "payload_b64": payload_b64, "ttl_s": 60},
            )
            assert st == 200 and j["ok"] is True

            st, j = _post_json(base + "/poll", {"group_id": "g", "receiver": 1, "deadline_s": time.time() + 1.0})
            assert st == 200 and j["ok"] is True
            msg = j["msg"]
            assert msg is not None
            assert msg["msg_id"] == "m1"
            assert base64.b64decode(msg["payload_b64"]) == payload

            st, j = _post_json(
                base + "/ack",
                {"group_id": "g", "receiver": 1, "msg_id": "m1", "lease_token": msg["lease_token"]},
            )
            assert st == 200 and j["ok"] is True

            st, j = _post_json(base + "/poll", {"group_id": "g", "receiver": 1, "deadline_s": time.time() + 0.2})
            assert st == 200 and j["ok"] is True and j["msg"] is None

            # Dedup: enqueue same payload under same msg_id
            st, j = _post_json(
                base + "/enqueue",
                {"group_id": "g", "msg_id": "m1", "sender": 0, "receiver": 1, "payload_b64": payload_b64, "ttl_s": 60},
            )
            assert st == 200 and j["ok"] is True and j["status"] == "dedup"

            # Collision: same msg_id, different payload
            st, j = _post_json(
                base + "/enqueue",
                {"group_id": "g", "msg_id": "m1", "sender": 0, "receiver": 1, "payload_b64": base64.b64encode(b"bye").decode("ascii"), "ttl_s": 60},
            )
            assert st == 409 and j["ok"] is False
        finally:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


