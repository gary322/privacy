from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_fce8a8bb25a7fd5f

import struct
from dataclasses import dataclass
from typing import ClassVar


MAGIC_KDF_INFO_V1 = b"UVCCKDF\0"
KDF_INFO_VERSION_V1 = 1

KIND_DPF_PRG_CTX = 0x01
KIND_DCF_PRG_CTX = 0x02
KIND_OPLUT_BUNDLE = 0x10
KIND_OPLUT_MASKVEC_ONLY = 0x11
KIND_OPLUT_OUTMASK_ONLY = 0x12

PRG_TYPE_AES128_FIXEDKEY = 1
PRG_TYPE_CHACHA20_FIXEDKEY = 2


_HDR = struct.Struct("<8sBBBBBBB4s")  # magic, ver, kind, prg_type, pair_id, share_idx, w_bits, elem_bits, reserved[4]


@dataclass(frozen=True)
class UVCCKDFInfoV1:
    """
    Byte-exact UVCC_KDF_INFO_v1 (83 bytes).
    """

    kind: int
    prg_type: int
    pair_id: int
    share_idx: int
    w_bits: int
    elem_bits: int
    sid32: bytes
    fss_id16: bytes
    table_hash16: bytes

    SIZE: ClassVar[int] = 83

    def __post_init__(self) -> None:
        if int(self.kind) not in (KIND_DPF_PRG_CTX, KIND_DCF_PRG_CTX, KIND_OPLUT_BUNDLE, KIND_OPLUT_MASKVEC_ONLY, KIND_OPLUT_OUTMASK_ONLY):
            raise ValueError("bad kind")
        if int(self.prg_type) not in (PRG_TYPE_AES128_FIXEDKEY, PRG_TYPE_CHACHA20_FIXEDKEY):
            raise ValueError("bad prg_type")
        if int(self.pair_id) not in (0, 1, 2, 255):
            raise ValueError("pair_id must be 0/1/2/255")
        if int(self.share_idx) not in (0, 1, 2, 255):
            raise ValueError("share_idx must be 0/1/2/255")
        if int(self.w_bits) not in (8, 16):
            raise ValueError("w_bits must be 8 or 16")
        if int(self.elem_bits) not in (8, 16):
            raise ValueError("elem_bits must be 8 or 16")
        if len(self.sid32) != 32:
            raise ValueError("sid32 must be 32 bytes")
        if len(self.fss_id16) != 16:
            raise ValueError("fss_id16 must be 16 bytes")
        if len(self.table_hash16) != 16:
            raise ValueError("table_hash16 must be 16 bytes")

    def to_bytes(self) -> bytes:
        hdr = _HDR.pack(
            MAGIC_KDF_INFO_V1,
            int(KDF_INFO_VERSION_V1) & 0xFF,
            int(self.kind) & 0xFF,
            int(self.prg_type) & 0xFF,
            int(self.pair_id) & 0xFF,
            int(self.share_idx) & 0xFF,
            int(self.w_bits) & 0xFF,
            int(self.elem_bits) & 0xFF,
            b"\x00\x00\x00\x00",
        )
        out = bytearray(hdr)
        out += bytes(self.sid32)
        out += bytes(self.fss_id16)
        out += bytes(self.table_hash16)
        b = bytes(out)
        if len(b) != self.SIZE:
            raise AssertionError("bad KDF info size")
        return b

    @staticmethod
    def from_bytes(buf: bytes) -> "UVCCKDFInfoV1":
        if not isinstance(buf, (bytes, bytearray)):
            raise TypeError("buf must be bytes")
        b = bytes(buf)
        if len(b) != UVCCKDFInfoV1.SIZE:
            raise ValueError("bad length")
        magic, ver, kind, prg_type, pair_id, share_idx, w_bits, elem_bits, reserved4 = _HDR.unpack_from(b, 0)
        if magic != MAGIC_KDF_INFO_V1 or int(ver) != KDF_INFO_VERSION_V1:
            raise ValueError("bad magic/version")
        if reserved4 != b"\x00\x00\x00\x00":
            raise ValueError("reserved must be 0")
        off = _HDR.size
        sid32 = b[off : off + 32]
        off += 32
        fss_id16 = b[off : off + 16]
        off += 16
        table_hash16 = b[off : off + 16]
        return UVCCKDFInfoV1(
            kind=int(kind),
            prg_type=int(prg_type),
            pair_id=int(pair_id),
            share_idx=int(share_idx),
            w_bits=int(w_bits),
            elem_bits=int(elem_bits),
            sid32=bytes(sid32),
            fss_id16=bytes(fss_id16),
            table_hash16=bytes(table_hash16),
        )


def kdf_info_zero_table_hash16() -> bytes:
    return b"\x00" * 16


def kdf_info_fss_id16_from_u128_le(x: int) -> bytes:
    """
    Deterministic helper for tests: pack a u128 namespace id into 16 bytes (LE).
    """
    v = int(x)
    if v < 0 or v >= (1 << 128):
        raise ValueError("u128 out of range")
    return v.to_bytes(16, "little", signed=False)


