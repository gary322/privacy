from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import paramiko


class SSHError(RuntimeError):
    pass


def _load_private_key_pem(pem: str) -> Any:
    if not isinstance(pem, str) or not pem.strip():
        raise SSHError("ssh private key pem is empty")
    buf = io.StringIO(pem)
    last_err: Optional[Exception] = None
    for key_cls in (getattr(paramiko, "Ed25519Key", None), getattr(paramiko, "RSAKey", None), getattr(paramiko, "ECDSAKey", None)):
        if not key_cls:
            continue
        try:
            buf.seek(0)
            return key_cls.from_private_key(buf)
        except Exception as exc:
            last_err = exc
            continue
    raise SSHError(f"unsupported ssh private key format: {last_err}")


def load_private_key_from_file(path: str) -> Any:
    p = Path(str(path)).expanduser().resolve()
    if not p.exists():
        raise SSHError(f"ssh private key file missing: {p}")
    pem = p.read_text(encoding="utf-8")
    return _load_private_key_pem(pem)


def ssh_connect_with_retries(
    *,
    hostname: str,
    port: int,
    username: str,
    pkey: Any,
    timeout_s: int = 900,
) -> paramiko.SSHClient:
    deadline = time.time() + max(10, int(timeout_s))
    attempt = 0
    last_err = ""
    while time.time() < deadline:
        attempt += 1
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                hostname=str(hostname),
                port=int(port),
                username=str(username),
                pkey=pkey,
                timeout=30,
                banner_timeout=60,
                auth_timeout=60,
                allow_agent=False,
                look_for_keys=False,
            )
            return ssh
        except Exception as exc:
            try:
                ssh.close()
            except Exception:
                pass
            err = str(exc) or exc.__class__.__name__
            last_err = err
            delay = min(15.0, max(2.0, 2.0 + (attempt * 0.5)))
            time.sleep(delay)
    raise SSHError(f"timed out connecting SSH to {username}@{hostname}:{port} (last_error={last_err})")


def ssh_exec(ssh: paramiko.SSHClient, cmd: str, *, timeout_s: int = 600) -> Tuple[int, str, str]:
    """
    Execute a remote command without deadlocking on large stdout/stderr.
    Returns (exit_code, stdout_text, stderr_text).
    """
    stdin, stdout, stderr = ssh.exec_command(str(cmd), get_pty=True)
    ch = stdout.channel
    deadline = time.time() + max(1, int(timeout_s))

    out_chunks: list[bytes] = []
    err_chunks: list[bytes] = []
    while True:
        if time.time() > deadline:
            try:
                ch.close()
            except Exception:
                pass
            raise SSHError(f"ssh command timed out after {timeout_s}s: {cmd}")

        if ch.recv_ready():
            out_chunks.append(ch.recv(65536))
            continue
        if ch.recv_stderr_ready():
            err_chunks.append(ch.recv_stderr(65536))
            continue
        if ch.exit_status_ready():
            while ch.recv_ready():
                out_chunks.append(ch.recv(65536))
            while ch.recv_stderr_ready():
                err_chunks.append(ch.recv_stderr(65536))
            break
        time.sleep(0.02)

    code = int(ch.recv_exit_status())
    out = b"".join(out_chunks).decode("utf-8", errors="replace")
    err = b"".join(err_chunks).decode("utf-8", errors="replace")
    try:
        stdin.close()
    except Exception:
        pass
    try:
        stdout.close()
    except Exception:
        pass
    try:
        stderr.close()
    except Exception:
        pass
    return code, out, err


def sftp_put_bytes(ssh: paramiko.SSHClient, *, remote_path: str, data: bytes, mode: int = 0o600) -> None:
    rp = str(remote_path)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(rp, mode="wb") as f:
            f.write(bytes(data))
        sftp.chmod(rp, int(mode) & 0o777)
    finally:
        try:
            sftp.close()
        except Exception:
            pass


def sftp_put_file(ssh: paramiko.SSHClient, *, local_path: str, remote_path: str, mode: Optional[int] = None) -> None:
    lp = Path(str(local_path)).expanduser().resolve()
    if not lp.exists():
        raise SSHError(f"local file missing: {lp}")
    rp = str(remote_path)
    # Some providers (notably datacrunch) can drop the SFTP channel mid-transfer during early provisioning.
    # Make this best-effort robust by retrying with a fresh SFTP session.
    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            sftp = ssh.open_sftp()
            try:
                sftp.put(str(lp), rp)
                if mode is not None:
                    try:
                        sftp.chmod(rp, int(mode) & 0o777)
                    except Exception:
                        pass
            finally:
                try:
                    sftp.close()
                except Exception:
                    pass
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0 + attempt * 2.0)
            continue
    if last_exc is not None:
        raise last_exc


def sftp_get_file(ssh: paramiko.SSHClient, *, remote_path: str, local_path: str) -> None:
    lp = Path(str(local_path)).expanduser().resolve()
    lp.parent.mkdir(parents=True, exist_ok=True)
    sftp = ssh.open_sftp()
    try:
        sftp.get(str(remote_path), str(lp))
    finally:
        try:
            sftp.close()
        except Exception:
            pass


@dataclass(frozen=True)
class RemoteHostV1:
    user: str
    host: str
    port: int = 22

    def __post_init__(self) -> None:
        if not str(self.user).strip():
            raise ValueError("user required")
        if not str(self.host).strip():
            raise ValueError("host required")
        p = int(self.port)
        if p <= 0 or p > 65535:
            raise ValueError("port invalid")


