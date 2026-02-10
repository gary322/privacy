from __future__ import annotations

"""
secp256k1 signatures for UVCC v1.

UVCC uses Ethereum-compatible secp256k1 keys for party identities and for signing
transcript roots / commits. This module implements deterministic signing and
verification over 32-byte message hashes.
"""

from eth_keys import keys


class SigError(Exception):
    pass


def _require_len(b: bytes, n: int, name: str) -> None:
    if not isinstance(b, (bytes, bytearray)) or len(b) != n:
        raise SigError(f"{name} must be {n} bytes")


def secp256k1_pubkey_from_privkey(privkey32: bytes) -> bytes:
    _require_len(privkey32, 32, "privkey32")
    pk = keys.PrivateKey(bytes(privkey32)).public_key
    # Uncompressed 64-byte (x||y) form is stable for proof bundle identity.
    return pk.to_bytes()


def secp256k1_eth_address_from_pubkey(pubkey64: bytes) -> bytes:
    _require_len(pubkey64, 64, "pubkey64")
    return keys.PublicKey(bytes(pubkey64)).to_canonical_address()


def secp256k1_sign_hash(privkey32: bytes, msg_hash32: bytes) -> bytes:
    """
    Sign a 32-byte message hash. Returns 65-byte signature: r(32)||s(32)||v(1) with v in {0,1}.
    """
    _require_len(privkey32, 32, "privkey32")
    _require_len(msg_hash32, 32, "msg_hash32")
    sig = keys.PrivateKey(bytes(privkey32)).sign_msg_hash(bytes(msg_hash32))
    return sig.to_bytes()


def secp256k1_verify_hash(pubkey64: bytes, msg_hash32: bytes, sig65: bytes) -> bool:
    _require_len(pubkey64, 64, "pubkey64")
    _require_len(msg_hash32, 32, "msg_hash32")
    _require_len(sig65, 65, "sig65")
    try:
        pk = keys.PublicKey(bytes(pubkey64))
        sig = keys.Signature(bytes(sig65))
        return bool(sig.verify_msg_hash(bytes(msg_hash32), pk))
    except Exception:
        return False


def secp256k1_recover_pubkey_from_hash(msg_hash32: bytes, sig65: bytes) -> bytes:
    _require_len(msg_hash32, 32, "msg_hash32")
    _require_len(sig65, 65, "sig65")
    pk = keys.Signature(bytes(sig65)).recover_public_key_from_msg_hash(bytes(msg_hash32))
    return pk.to_bytes()


