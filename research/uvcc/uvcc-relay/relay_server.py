#!/usr/bin/env python3
"""
uvcc-relay: production-grade HTTP(S) relay hub for UVCC v1.

Design goals (aligned with research/privacy_new.txt + uvcc_profile_v1):
- Opaque frame carriage (relay does not interpret payloads)
- Idempotent enqueue (dedup by (group_id, msg_id))
- Leased delivery (poll grants a lease; ack finalizes)
- Bounded storage with TTL + GC
- Deterministic errors and stable JSON responses (so clients are reproducible)
- No external deps (stdlib only)
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import secrets
import sqlite3
import ssl
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple


def _now_s() -> float:
    return time.time()


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def _json_response(handler: BaseHTTPRequestHandler, *, status: int, obj: Dict[str, Any]) -> None:
    body = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler, *, max_bytes: int) -> Dict[str, Any]:
    n = int(handler.headers.get("Content-Length", "0") or "0")
    if n <= 0 or n > max_bytes:
        raise ValueError("bad_content_length")
    raw = handler.rfile.read(n)
    return json.loads(raw.decode("utf-8"))


@dataclass(frozen=True)
class RelayConfig:
    db_path: str
    require_token: bool
    token: Optional[str]
    max_body_bytes: int = 8 * 1024 * 1024
    default_ttl_s: int = 300
    lease_s: int = 15
    gc_interval_s: int = 5


class RelayStore:
    def __init__(self, cfg: RelayConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(cfg.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_db()

    def _init_db(self) -> None:
        # sqlite3.Connection is not safe for concurrent use across threads, even with
        # check_same_thread=False. Serialize ALL access to avoid undefined behavior.
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS msgs (
                      group_id TEXT NOT NULL,
                      msg_id   TEXT NOT NULL,
                      sender   INTEGER NOT NULL,
                      receiver INTEGER NOT NULL,
                      payload_b64 TEXT NOT NULL,
                      payload_hash32 BLOB NOT NULL,
                      created_at REAL NOT NULL,
                      expires_at REAL NOT NULL,
                      leased_until REAL NOT NULL,
                      lease_token TEXT NOT NULL,
                      acked INTEGER NOT NULL,
                      PRIMARY KEY (group_id, msg_id)
                    );
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_msgs_recv ON msgs(group_id, receiver, acked, leased_until, expires_at);"
                )

    def gc_once(self) -> int:
        now = _now_s()
        with self._lock:
            with self._conn:
                cur = self._conn.execute("DELETE FROM msgs WHERE acked=1 OR expires_at < ?", (now,))
                return int(cur.rowcount)

    def enqueue(
        self,
        *,
        group_id: str,
        msg_id: str,
        sender: int,
        receiver: int,
        payload_bytes: bytes,
        ttl_s: Optional[int],
    ) -> Tuple[bool, str]:
        ttl = int(ttl_s) if ttl_s is not None else int(self.cfg.default_ttl_s)
        if ttl <= 0 or ttl > 24 * 3600:
            raise ValueError("bad_ttl")

        ph = _sha256(payload_bytes)
        now = _now_s()
        expires = now + ttl
        payload_b64 = _b64e(payload_bytes)

        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT payload_hash32 FROM msgs WHERE group_id=? AND msg_id=?",
                    (group_id, msg_id),
                ).fetchone()
                if row is not None:
                    if bytes(row[0]) != ph:
                        return (False, "msg_id_collision")
                    return (True, "dedup")

                self._conn.execute(
                    """
                    INSERT INTO msgs(group_id,msg_id,sender,receiver,payload_b64,payload_hash32,created_at,expires_at,leased_until,lease_token,acked)
                    VALUES(?,?,?,?,?,?,?,?,?,?,0)
                    """,
                    (group_id, msg_id, int(sender), int(receiver), payload_b64, ph, now, expires, 0.0, ""),
                )
                return (True, "enqueued")

    def poll(
        self,
        *,
        group_id: str,
        receiver: int,
    ) -> Optional[Dict[str, Any]]:
        now = _now_s()
        lease_until = now + float(self.cfg.lease_s)
        lease_token = secrets.token_hex(16)

        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    """
                    SELECT msg_id, sender, payload_b64, payload_hash32
                    FROM msgs
                    WHERE group_id=?
                      AND receiver=?
                      AND acked=0
                      AND expires_at >= ?
                      AND (leased_until <= ? OR leased_until IS NULL)
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (group_id, int(receiver), now, now),
                ).fetchone()
                if row is None:
                    return None
                msg_id, sender, payload_b64, payload_hash32 = row
                # Acquire lease
                self._conn.execute(
                    """
                    UPDATE msgs SET leased_until=?, lease_token=?
                    WHERE group_id=? AND msg_id=?
                    """,
                    (lease_until, lease_token, group_id, msg_id),
                )
                return {
                    "msg_id": str(msg_id),
                    "sender": int(sender),
                    "receiver": int(receiver),
                    "lease_token": lease_token,
                    "lease_until": float(lease_until),
                    "payload_b64": str(payload_b64),
                    "payload_hash32_b64": _b64e(bytes(payload_hash32)),
                }

    def ack(
        self,
        *,
        group_id: str,
        receiver: int,
        msg_id: str,
        lease_token: str,
    ) -> Tuple[bool, str]:
        now = _now_s()
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT acked, leased_until, lease_token, receiver FROM msgs WHERE group_id=? AND msg_id=?",
                    (group_id, msg_id),
                ).fetchone()
                if row is None:
                    return (False, "not_found")
                acked, leased_until, tok, recv = row
                if int(recv) != int(receiver):
                    return (False, "wrong_receiver")
                if int(acked) == 1:
                    return (True, "already_acked")
                if str(tok) != str(lease_token):
                    return (False, "bad_lease_token")
                if float(leased_until) < now:
                    return (False, "lease_expired")

                self._conn.execute(
                    "UPDATE msgs SET acked=1 WHERE group_id=? AND msg_id=?",
                    (group_id, msg_id),
                )
                return (True, "acked")


class RelayHandler(BaseHTTPRequestHandler):
    server_version = "uvcc-relay/1"

    def _auth_ok(self) -> bool:
        cfg: RelayConfig = self.server.cfg  # type: ignore[attr-defined]
        if not cfg.require_token:
            return True
        if cfg.token is None:
            return False
        hdr = self.headers.get("Authorization", "")
        if not hdr.startswith("Bearer "):
            return False
        tok = hdr[len("Bearer ") :]
        return secrets.compare_digest(tok, cfg.token)

    def _not_found(self) -> None:
        _json_response(self, status=404, obj={"ok": False, "error": "not_found"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            _json_response(self, status=200, obj={"ok": True})
            return
        self._not_found()

    def do_POST(self) -> None:  # noqa: N802
        if not self._auth_ok():
            _json_response(self, status=401, obj={"ok": False, "error": "unauthorized"})
            return

        cfg: RelayConfig = self.server.cfg  # type: ignore[attr-defined]
        store: RelayStore = self.server.store  # type: ignore[attr-defined]

        try:
            req = _read_json(self, max_bytes=cfg.max_body_bytes)
        except Exception:
            _json_response(self, status=400, obj={"ok": False, "error": "bad_json"})
            return

        if self.path == "/enqueue":
            try:
                group_id = str(req["group_id"])
                msg_id = str(req["msg_id"])
                sender = int(req["sender"])
                receiver = int(req["receiver"])
                payload_b64 = str(req["payload_b64"])
                ttl_s = int(req["ttl_s"]) if "ttl_s" in req and req["ttl_s"] is not None else None
                payload = _b64d(payload_b64)
                ok, status = store.enqueue(
                    group_id=group_id,
                    msg_id=msg_id,
                    sender=sender,
                    receiver=receiver,
                    payload_bytes=payload,
                    ttl_s=ttl_s,
                )
                if not ok and status == "msg_id_collision":
                    _json_response(self, status=409, obj={"ok": False, "error": "msg_id_collision"})
                    return
                _json_response(self, status=200, obj={"ok": True, "status": status})
                return
            except Exception as e:
                _json_response(self, status=400, obj={"ok": False, "error": "enqueue_failed"})
                return

        if self.path == "/poll":
            try:
                group_id = str(req["group_id"])
                receiver = int(req["receiver"])
                deadline_s = float(req.get("deadline_s", 0.0))
                if deadline_s <= 0:
                    deadline_s = _now_s() + 2.0
                while True:
                    m = store.poll(group_id=group_id, receiver=receiver)
                    if m is not None:
                        _json_response(self, status=200, obj={"ok": True, "msg": m})
                        return
                    if _now_s() >= deadline_s:
                        _json_response(self, status=200, obj={"ok": True, "msg": None})
                        return
                    time.sleep(0.01)
            except Exception:
                _json_response(self, status=400, obj={"ok": False, "error": "poll_failed"})
                return

        if self.path == "/ack":
            try:
                group_id = str(req["group_id"])
                receiver = int(req["receiver"])
                msg_id = str(req["msg_id"])
                lease_token = str(req["lease_token"])
                ok, status = store.ack(group_id=group_id, receiver=receiver, msg_id=msg_id, lease_token=lease_token)
                if ok:
                    _json_response(self, status=200, obj={"ok": True, "status": status})
                    return
                _json_response(self, status=409, obj={"ok": False, "error": status})
                return
            except Exception:
                _json_response(self, status=400, obj={"ok": False, "error": "ack_failed"})
                return

        self._not_found()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Avoid noisy default logging; production should plug structured logging.
        return


def _gc_thread(store: RelayStore, cfg: RelayConfig, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            store.gc_once()
        except Exception:
            pass
        stop.wait(timeout=float(cfg.gc_interval_s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--db", required=True, help="SQLite db path")
    ap.add_argument("--require-token", default="false", choices=["true", "false"])
    ap.add_argument("--token", default=None, help="Bearer token (if require-token=true)")
    ap.add_argument("--token-file", default=None, help="Path to file containing bearer token (if require-token=true)")
    ap.add_argument("--tls-cert", default=None, help="PEM cert path (optional)")
    ap.add_argument("--tls-key", default=None, help="PEM key path (optional)")
    ap.add_argument("--default-ttl-s", type=int, default=300, help="Default message TTL in seconds (enqueue ttl_s fallback)")
    ap.add_argument("--lease-s", type=int, default=15, help="Receiver lease duration in seconds (must be >= worst-case poll->ack latency)")
    ap.add_argument("--gc-interval-s", type=float, default=5.0, help="GC interval in seconds (deletes acked/expired messages)")
    args = ap.parse_args()

    token: Optional[str] = str(args.token).strip() if args.token is not None else None
    if token == "":
        token = None
    if token is None and args.token_file is not None:
        try:
            token = open(str(args.token_file), "r", encoding="utf-8").read().strip()
        except Exception as exc:
            raise SystemExit(f"failed to read --token-file: {exc}")
        if token == "":
            token = None
    # Final fallback: allow env var injection so operators don't have to put tokens on the process list.
    if token is None:
        token = str(os.environ.get("UVCC_RELAY_TOKEN", "")).strip() or None
    if (args.require_token == "true") and not token:
        raise SystemExit("require-token=true but no token provided (use --token, --token-file, or UVCC_RELAY_TOKEN)")
    if int(args.default_ttl_s) <= 0:
        raise SystemExit("--default-ttl-s must be > 0")
    if int(args.lease_s) <= 0:
        raise SystemExit("--lease-s must be > 0")
    if float(args.gc_interval_s) <= 0:
        raise SystemExit("--gc-interval-s must be > 0")
    os.makedirs(os.path.dirname(str(args.db)) or ".", exist_ok=True)
    cfg = RelayConfig(
        db_path=str(args.db),
        require_token=(args.require_token == "true"),
        token=str(token) if token is not None else None,
        default_ttl_s=int(args.default_ttl_s),
        lease_s=int(args.lease_s),
        gc_interval_s=float(args.gc_interval_s),
    )
    store = RelayStore(cfg)

    httpd = ThreadingHTTPServer((str(args.host), int(args.port)), RelayHandler)
    httpd.cfg = cfg  # type: ignore[attr-defined]
    httpd.store = store  # type: ignore[attr-defined]

    if args.tls_cert and args.tls_key:
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(certfile=str(args.tls_cert), keyfile=str(args.tls_key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    stop = threading.Event()
    t = threading.Thread(target=_gc_thread, args=(store, cfg, stop), daemon=True)
    t.start()

    try:
        httpd.serve_forever(poll_interval=0.25)
    finally:
        stop.set()
        try:
            httpd.server_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


