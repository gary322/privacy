from __future__ import annotations

from uvcc_party.crc32c import crc32c


def test_crc32c_known_vector() -> None:
    # Standard CRC32C (Castagnoli) check value for ASCII "123456789".
    assert crc32c(b"123456789") == 0xE3069283


