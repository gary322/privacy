from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from eth_keys import keys


class PartyIdentityError(RuntimeError):
    pass


def _load_privkey32_hex(path: Path) -> bytes:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("0x"):
        raw = raw[2:]
    b = bytes.fromhex(raw)
    if len(b) != 32:
        raise PartyIdentityError("privkey must be 32 bytes")
    # Validate by constructing eth_keys object (raises on invalid).
    _ = keys.PrivateKey(b)
    return b


def _save_privkey32_hex(path: Path, privkey32: bytes) -> None:
    if len(privkey32) != 32:
        raise PartyIdentityError("privkey must be 32 bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("0x" + privkey32.hex(), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)


def load_or_create_party_privkey32_v1(*, path: str, rng: Optional[Any] = None) -> bytes:
    """
    Load an existing secp256k1 private key (32 bytes) from disk, or create a new one.

    This is for party identity signatures (EIP-712 policy/final commits).
    """
    p = Path(str(path)).expanduser().resolve()
    if p.exists():
        return _load_privkey32_hex(p)

    # Generate until eth_keys accepts it (ensures 0 < k < curve_order).
    while True:
        priv = os.urandom(32)
        try:
            _ = keys.PrivateKey(priv)
            _save_privkey32_hex(p, priv)
            return priv
        except Exception:
            continue


@dataclass(frozen=True)
class PartyIdentityV1:
    party_id: int
    privkey32: bytes
    pubkey64: bytes
    address20: bytes

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "party_id": f"P{int(self.party_id)}",
            "pubkey64_hex": "0x" + self.pubkey64.hex(),
            "address": "0x" + self.address20.hex(),
        }


def party_identity_from_privkey_v1(*, party_id: int, privkey32: bytes) -> PartyIdentityV1:
    if len(privkey32) != 32:
        raise PartyIdentityError("privkey32 must be 32 bytes")
    pk = keys.PrivateKey(bytes(privkey32))
    pub64 = pk.public_key.to_bytes()
    addr20 = pk.public_key.to_canonical_address()
    return PartyIdentityV1(party_id=int(party_id), privkey32=bytes(privkey32), pubkey64=pub64, address20=addr20)


def party_sign_hash32_v1(*, privkey32: bytes, digest32: bytes) -> bytes:
    if len(privkey32) != 32 or len(digest32) != 32:
        raise PartyIdentityError("privkey32 and digest32 must be 32 bytes")
    sig = keys.PrivateKey(bytes(privkey32)).sign_msg_hash(bytes(digest32))
    return sig.to_bytes()


