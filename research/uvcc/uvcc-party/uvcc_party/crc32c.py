from __future__ import annotations

"""
CRC32C (Castagnoli) implementation.

UVCC uses CRC32C in some wire formats (optional but deterministic). This module
implements CRC32C with the Castagnoli polynomial (0x1EDC6F41) in pure Python
for portability (CPU and GPU hosts without extra deps).
"""

from functools import lru_cache


def _crc32c_table() -> list[int]:
    # Reflected CRC32C polynomial (Castagnoli): 0x82F63B78
    poly = 0x82F63B78
    tbl: list[int] = []
    for i in range(256):
        c = i
        for _ in range(8):
            if c & 1:
                c = (c >> 1) ^ poly
            else:
                c >>= 1
        tbl.append(c & 0xFFFFFFFF)
    return tbl


@lru_cache(maxsize=1)
def _tbl() -> tuple[int, ...]:
    return tuple(_crc32c_table())


def crc32c(data: bytes, *, init: int = 0) -> int:
    """
    Compute CRC32C(data) with initial value `init` (default 0).

    Returns an unsigned 32-bit integer.
    """
    c = int(init) ^ 0xFFFFFFFF
    tbl = _tbl()
    for b in data:
        c = tbl[(c ^ b) & 0xFF] ^ (c >> 8)
    return (c ^ 0xFFFFFFFF) & 0xFFFFFFFF


