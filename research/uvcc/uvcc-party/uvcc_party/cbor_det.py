from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b2265a18869da013

from typing import Any, Dict, Iterable, List, Tuple


def _encode_type_and_len(major: int, n: int) -> bytes:
    if n < 0:
        raise ValueError("length must be >= 0")
    m = int(major) & 0x07
    nn = int(n)
    if nn <= 23:
        return bytes([(m << 5) | nn])
    if nn <= 0xFF:
        return bytes([(m << 5) | 24, nn & 0xFF])
    if nn <= 0xFFFF:
        return bytes([(m << 5) | 25]) + nn.to_bytes(2, "big", signed=False)
    if nn <= 0xFFFFFFFF:
        return bytes([(m << 5) | 26]) + nn.to_bytes(4, "big", signed=False)
    if nn <= 0xFFFFFFFFFFFFFFFF:
        return bytes([(m << 5) | 27]) + nn.to_bytes(8, "big", signed=False)
    raise ValueError("length too large")


def cbor_dumps_det_v1(x: Any) -> bytes:
    """
    Deterministic CBOR subset encoder (definite-length only).

    Supported:
      - int (signed 64-bit)
      - bytes/bytearray
      - str (UTF-8)
      - list/tuple
      - dict with str keys
      - bool
    """
    if x is False:
        return b"\xF4"
    if x is True:
        return b"\xF5"
    if isinstance(x, int):
        v = int(x)
        if v >= 0:
            return _encode_type_and_len(0, v)
        # negative: -1-n
        n = -1 - v
        return _encode_type_and_len(1, n)
    if isinstance(x, (bytes, bytearray)):
        b = bytes(x)
        return _encode_type_and_len(2, len(b)) + b
    if isinstance(x, str):
        b = x.encode("utf-8")
        return _encode_type_and_len(3, len(b)) + b
    if isinstance(x, (list, tuple)):
        out = bytearray(_encode_type_and_len(4, len(x)))
        for it in x:
            out += cbor_dumps_det_v1(it)
        return bytes(out)
    if isinstance(x, dict):
        # Keys must be str; sort by bytewise UTF-8 of the key.
        items: List[Tuple[bytes, str, Any]] = []
        for k, v in x.items():
            if not isinstance(k, str):
                raise TypeError("CBOR map keys must be str")
            kb = k.encode("utf-8")
            items.append((kb, k, v))
        items.sort(key=lambda t: t[0])
        out = bytearray(_encode_type_and_len(5, len(items)))
        for _kb, k, v in items:
            out += cbor_dumps_det_v1(k)
            out += cbor_dumps_det_v1(v)
        return bytes(out)
    raise TypeError(f"unsupported CBOR type: {type(x)}")


