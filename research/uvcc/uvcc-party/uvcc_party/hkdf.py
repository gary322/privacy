from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b09f4ad9c27cdb7d

import hashlib
from typing import Optional


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    """
    HMAC-SHA256 (RFC2104).
    """
    if not isinstance(key, (bytes, bytearray)) or not isinstance(msg, (bytes, bytearray)):
        raise TypeError("key/msg must be bytes")
    k = bytes(key)
    if len(k) > 64:
        k = sha256(k)
    if len(k) < 64:
        k = k + b"\x00" * (64 - len(k))
    o_key_pad = bytes((x ^ 0x5C) for x in k)
    i_key_pad = bytes((x ^ 0x36) for x in k)
    return sha256(o_key_pad + sha256(i_key_pad + bytes(msg)))


def hkdf_extract(*, salt: Optional[bytes], ikm: bytes) -> bytes:
    """
    HKDF-Extract (RFC5869) using HMAC-SHA256.
    """
    if salt is None:
        salt = b""
    if not isinstance(salt, (bytes, bytearray)) or not isinstance(ikm, (bytes, bytearray)):
        raise TypeError("salt/ikm must be bytes")
    s = bytes(salt)
    if len(s) == 0:
        s = b"\x00" * 32
    return hmac_sha256(s, bytes(ikm))


def hkdf_expand(*, prk32: bytes, info: bytes, length: int) -> bytes:
    """
    HKDF-Expand (RFC5869) using HMAC-SHA256.
    """
    if not isinstance(prk32, (bytes, bytearray)) or not isinstance(info, (bytes, bytearray)):
        raise TypeError("prk32/info must be bytes")
    prk = bytes(prk32)
    if len(prk) != 32:
        raise ValueError("prk32 must be 32 bytes")
    L = int(length)
    if L < 0:
        raise ValueError("length must be >= 0")
    if L == 0:
        return b""
    n = (L + 31) // 32
    if n > 255:
        raise ValueError("length too large")
    okm = bytearray()
    t = b""
    info_b = bytes(info)
    for i in range(1, n + 1):
        t = hmac_sha256(prk, t + info_b + bytes([i & 0xFF]))
        okm += t
    return bytes(okm[:L])


