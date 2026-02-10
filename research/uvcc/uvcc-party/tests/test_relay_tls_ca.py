from __future__ import annotations

# pyright: reportMissingImports=false

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from uvcc_party.relay_client import RelayClient, RelayError


def _repo_root() -> Path:
    # .../research/uvcc/uvcc-party/tests/test_relay_tls_ca.py -> repo root
    return Path(__file__).resolve().parents[4]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _write_ca_and_server_cert(tmp: Path) -> tuple[Path, Path, Path]:
    """
    Create a private CA + server cert for 127.0.0.1 using cryptography.
    Returns (ca_cert_pem, server_cert_pem, server_key_pem).
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "UVCC Test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    san = x509.SubjectAlternativeName([x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=7))
        .add_extension(san, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    ca_cert_pem = tmp / "ca.pem"
    server_cert_pem = tmp / "server.pem"
    server_key_pem = tmp / "server.key"

    ca_cert_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    server_cert_pem.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    server_key_pem.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    os.chmod(server_key_pem, 0o600)
    return ca_cert_pem, server_cert_pem, server_key_pem


def _start_relay_tls(*, port: int, db_path: str, cert_pem: Path, key_pem: Path) -> subprocess.Popen:
    relay_py = _repo_root() / "research" / "uvcc" / "uvcc-relay" / "relay_server.py"
    if not relay_py.exists():
        raise FileNotFoundError(str(relay_py))
    return subprocess.Popen(
        [
            sys.executable,
            str(relay_py),
            "--host",
            "127.0.0.1",
            "--port",
            str(int(port)),
            "--db",
            str(db_path),
            "--require-token",
            "false",
            "--tls-cert",
            str(cert_pem),
            "--tls-key",
            str(key_pem),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_health_tls(*, base_url: str, ca_pem: Path) -> None:
    rc = RelayClient(base_url=base_url, group_id="health", token=None, timeout_s=2.0, tls_ca_pem=str(ca_pem))
    for _ in range(250):
        try:
            rc.healthz()
            return
        except Exception:
            time.sleep(0.02)
    raise RuntimeError("relay never became healthy (tls)")


def test_relay_tls_ca_healthz_roundtrip() -> None:
    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ca_pem, server_pem, server_key = _write_ca_and_server_cert(tmp)
        db_path = str(tmp / "relay.sqlite")
        proc = _start_relay_tls(port=port, db_path=db_path, cert_pem=server_pem, key_pem=server_key)
        try:
            base = f"https://127.0.0.1:{port}"
            _wait_health_tls(base_url=base, ca_pem=ca_pem)

            # Without CA pinning, the self-signed chain should be rejected.
            rc_bad = RelayClient(base_url=base, group_id="health", token=None, timeout_s=2.0, tls_ca_pem=None)
            with pytest.raises(RelayError):
                rc_bad.healthz()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


