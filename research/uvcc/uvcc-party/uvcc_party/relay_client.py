from __future__ import annotations

import base64
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class RelayError(RuntimeError):
    pass


def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _json_loads(b: bytes) -> Any:
    return json.loads(b.decode("utf-8"))


@dataclass(frozen=True)
class RelayClient:
    base_url: str
    group_id: str
    token: Optional[str]
    timeout_s: float = 10.0
    tls_ca_pem: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token is not None:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _ssl_context(self) -> Optional[ssl.SSLContext]:
        # Only relevant for https:// URLs. We allow specifying a CA bundle explicitly so
        # callers can use a private PKI (e.g. self-hosted relay with self-signed CA).
        if self.tls_ca_pem is None and not str(self.base_url).lower().startswith("https://"):
            return None
        cafile = str(self.tls_ca_pem) if self.tls_ca_pem is not None else None
        return ssl.create_default_context(cafile=cafile)

    def healthz(self) -> None:
        req = urllib.request.Request(self.base_url + "/healthz", method="GET", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s), context=self._ssl_context()) as resp:
                if int(resp.status) != 200:
                    raise RelayError(f"healthz bad status {resp.status}")
        except Exception as e:
            raise RelayError(f"healthz failed: {e}") from e

    def enqueue(self, *, msg_id: str, sender: int, receiver: int, payload: bytes, ttl_s: Optional[int] = None) -> str:
        obj: Dict[str, Any] = {
            "group_id": self.group_id,
            "msg_id": str(msg_id),
            "sender": int(sender),
            "receiver": int(receiver),
            "payload_b64": base64.b64encode(payload).decode("ascii"),
            "ttl_s": int(ttl_s) if ttl_s is not None else None,
        }
        req = urllib.request.Request(
            self.base_url + "/enqueue",
            data=_json_dumps(obj),
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s), context=self._ssl_context()) as resp:
                data = _json_loads(resp.read())
                if not data.get("ok", False):
                    raise RelayError(f"enqueue failed: {data}")
                return str(data.get("status", ""))
        except urllib.error.HTTPError as e:
            raise RelayError(f"enqueue failed: {e.code} {e.read().decode('utf-8', errors='replace')}") from e
        except Exception as e:
            raise RelayError(f"enqueue failed: {e}") from e

    def poll(self, *, receiver: int, deadline_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if deadline_s is None:
            deadline_s = time.time() + float(self.timeout_s)
        obj = {"group_id": self.group_id, "receiver": int(receiver), "deadline_s": float(deadline_s)}
        req = urllib.request.Request(
            self.base_url + "/poll",
            data=_json_dumps(obj),
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s), context=self._ssl_context()) as resp:
                data = _json_loads(resp.read())
                if not data.get("ok", False):
                    raise RelayError(f"poll failed: {data}")
                msg = data.get("msg", None)
                if msg is None:
                    return None
                if not isinstance(msg, dict):
                    raise RelayError("poll returned non-dict msg")
                return msg
        except Exception as e:
            raise RelayError(f"poll failed: {e}") from e

    def ack(self, *, receiver: int, msg_id: str, lease_token: str) -> str:
        obj = {"group_id": self.group_id, "receiver": int(receiver), "msg_id": str(msg_id), "lease_token": str(lease_token)}
        req = urllib.request.Request(
            self.base_url + "/ack",
            data=_json_dumps(obj),
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=float(self.timeout_s), context=self._ssl_context()) as resp:
                data = _json_loads(resp.read())
                if not data.get("ok", False):
                    raise RelayError(f"ack failed: {data}")
                return str(data.get("status", ""))
        except urllib.error.HTTPError as e:
            raise RelayError(f"ack failed: {e.code} {e.read().decode('utf-8', errors='replace')}") from e
        except Exception as e:
            raise RelayError(f"ack failed: {e}") from e


