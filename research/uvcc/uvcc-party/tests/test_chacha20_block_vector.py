from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_f1817a0260a2d9bb

from uvcc_party.dpf_dcf import chacha20_block_bytes_v1


def test_chacha20_block_rfc8439_vector() -> None:
    # RFC 8439 Section 2.3.2 test vector (widely used):
    # key = 00..1f
    # nonce = 000000090000004a00000000
    # counter = 1
    key32 = bytes(range(32))
    nonce12 = bytes.fromhex("000000090000004a00000000")
    out = chacha20_block_bytes_v1(key32=key32, nonce12=nonce12, counter32=1)
    assert len(out) == 64
    expected = bytes.fromhex(
        "10f1e7e4d13b5915500fdd1fa32071c4"
        "c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2"
        "b5129cd1de164eb9cbd083e8a2503c4e"
    )
    assert out == expected


