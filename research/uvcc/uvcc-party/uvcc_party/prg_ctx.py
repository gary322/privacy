from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_fffe7f7fba4343b8,uvcc_group_6d0a46e2d04f891f,uvcc_group_732bc069128b3469

import hashlib
import struct
from dataclasses import dataclass
from typing import Tuple

from .hkdf import hkdf_expand, hkdf_extract, sha256
from .kdf_info import (
    KIND_DCF_PRG_CTX,
    KIND_DPF_PRG_CTX,
    PRG_TYPE_AES128_FIXEDKEY,
    PRG_TYPE_CHACHA20_FIXEDKEY,
    UVCCKDFInfoV1,
    kdf_info_zero_table_hash16,
)


DS_SALT = b"UVCC/SALT/V1\0"


def _domain_tag16_v1() -> bytes:
    return sha256(b"UVCC_G_V1")[0:16]


_AES_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

_AES_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def aes128_expand_rk_v1(key16: bytes) -> bytes:
    """
    AES-128 key expansion (176 bytes round keys) matching the v1 state/key schedule mapping.
    """
    if len(key16) != 16:
        raise ValueError("key16 must be 16 bytes")
    rk = [0] * 176
    for i, b in enumerate(key16):
        rk[i] = int(b)
    rcon_i = 0
    for i in range(16, 176, 4):
        t0, t1, t2, t3 = rk[i - 4], rk[i - 3], rk[i - 2], rk[i - 1]
        if (i % 16) == 0:
            # RotWord
            t0, t1, t2, t3 = t1, t2, t3, t0
            # SubWord
            t0 = _AES_SBOX[t0]
            t1 = _AES_SBOX[t1]
            t2 = _AES_SBOX[t2]
            t3 = _AES_SBOX[t3]
            # Rcon
            t0 ^= _AES_RCON[rcon_i]
            rcon_i += 1
        rk[i + 0] = rk[i - 16 + 0] ^ t0
        rk[i + 1] = rk[i - 16 + 1] ^ t1
        rk[i + 2] = rk[i - 16 + 2] ^ t2
        rk[i + 3] = rk[i - 16 + 3] ^ t3
    return bytes(int(x) & 0xFF for x in rk)


def _xtime(x: int) -> int:
    x &= 0xFF
    return ((x << 1) & 0xFF) ^ (0x1B if (x & 0x80) else 0x00)


def _aes_shift_rows(s: list[int]) -> None:
    t = s[:]
    # row0 unchanged
    s[0], s[4], s[8], s[12] = t[0], t[4], t[8], t[12]
    # row1 left shift 1
    s[1], s[5], s[9], s[13] = t[5], t[9], t[13], t[1]
    # row2 left shift 2
    s[2], s[6], s[10], s[14] = t[10], t[14], t[2], t[6]
    # row3 left shift 3
    s[3], s[7], s[11], s[15] = t[15], t[3], t[7], t[11]


def _aes_mix_columns(s: list[int]) -> None:
    for c in range(4):
        o = c * 4
        a0, a1, a2, a3 = s[o + 0], s[o + 1], s[o + 2], s[o + 3]
        r0 = _xtime(a0) ^ (_xtime(a1) ^ a1) ^ a2 ^ a3
        r1 = a0 ^ _xtime(a1) ^ (_xtime(a2) ^ a2) ^ a3
        r2 = a0 ^ a1 ^ _xtime(a2) ^ (_xtime(a3) ^ a3)
        r3 = (_xtime(a0) ^ a0) ^ a1 ^ a2 ^ _xtime(a3)
        s[o + 0] = r0 & 0xFF
        s[o + 1] = r1 & 0xFF
        s[o + 2] = r2 & 0xFF
        s[o + 3] = r3 & 0xFF


def aes128_encrypt_fixedrk_v1(*, rk176: bytes, block16: bytes) -> bytes:
    if len(rk176) != 176:
        raise ValueError("rk176 must be 176 bytes")
    if len(block16) != 16:
        raise ValueError("block16 must be 16 bytes")
    s = [int(block16[i]) ^ int(rk176[i]) for i in range(16)]
    for rnd in range(1, 10):
        s = [_AES_SBOX[x] for x in s]
        _aes_shift_rows(s)
        _aes_mix_columns(s)
        off = rnd * 16
        s = [(s[i] ^ int(rk176[off + i])) & 0xFF for i in range(16)]
    s = [_AES_SBOX[x] for x in s]
    _aes_shift_rows(s)
    out = bytes(((s[i] ^ int(rk176[160 + i])) & 0xFF) for i in range(16))
    return out


def prg_salt32_v1(*, sid32: bytes) -> bytes:
    if len(sid32) != 32:
        raise ValueError("sid32 must be 32 bytes")
    return sha256(DS_SALT + sid32)


def prk_pair_v1(*, sid32: bytes, kpair32: bytes) -> bytes:
    if len(kpair32) != 32:
        raise ValueError("kpair32 must be 32 bytes")
    salt32 = prg_salt32_v1(sid32=sid32)
    return hkdf_extract(salt=salt32, ikm=kpair32)


@dataclass(frozen=True)
class UVCCPRGCTXv1:
    prg_type: int
    impl: int
    domain_tag16: bytes
    aes_rk176: bytes
    chacha_k32: bytes  # 32 bytes (8 u32 le)

    def __post_init__(self) -> None:
        if int(self.prg_type) not in (PRG_TYPE_AES128_FIXEDKEY, PRG_TYPE_CHACHA20_FIXEDKEY):
            raise ValueError("bad prg_type")
        if len(self.domain_tag16) != 16:
            raise ValueError("domain_tag16 must be 16 bytes")
        if len(self.aes_rk176) != 176:
            raise ValueError("aes_rk176 must be 176 bytes")
        if len(self.chacha_k32) != 32:
            raise ValueError("chacha_k32 must be 32 bytes")

    def to_bytes(self) -> bytes:
        # Mirrors UVCC_PRG_CTX_v1 from privacy_new.txt (256 bytes).
        pad16 = b"\x00" * 16
        return (
            struct.pack("<IIII", int(self.prg_type) & 0xFFFFFFFF, int(self.impl) & 0xFFFFFFFF, 0, 0)
            + bytes(self.domain_tag16)
            + bytes(self.aes_rk176)
            + bytes(self.chacha_k32)
            + pad16
        )

    @staticmethod
    def from_bytes(buf: bytes) -> "UVCCPRGCTXv1":
        if len(buf) != 256:
            raise ValueError("ctx must be 256 bytes")
        prg_type, impl, flags, reserved0 = struct.unpack_from("<IIII", buf, 0)
        if int(flags) != 0 or int(reserved0) != 0:
            raise ValueError("flags/reserved0 must be 0")
        domain_tag16 = buf[16:32]
        aes_rk176 = buf[32:208]
        chacha_k32 = buf[208:240]
        pad = buf[240:256]
        if pad != b"\x00" * 16:
            raise ValueError("pad must be 0")
        return UVCCPRGCTXv1(prg_type=int(prg_type), impl=int(impl), domain_tag16=domain_tag16, aes_rk176=aes_rk176, chacha_k32=chacha_k32)


def derive_prg_ctx_v1(
    *,
    sid32: bytes,
    kpair32: bytes,
    kind: int,
    prg_type: int,
    impl: int,
    pair_id: int,
    share_idx: int,
    w_bits: int,
    elem_bits: int,
    fss_id16: bytes,
    table_hash16: bytes | None = None,
) -> UVCCPRGCTXv1:
    """
    Derive a v1 PRG context from a pair secret and fixed public parameters (HKDF).
    """
    if len(sid32) != 32:
        raise ValueError("sid32 must be 32 bytes")
    if len(fss_id16) != 16:
        raise ValueError("fss_id16 must be 16 bytes")
    if table_hash16 is None:
        table_hash16 = kdf_info_zero_table_hash16()
    if len(table_hash16) != 16:
        raise ValueError("table_hash16 must be 16 bytes")

    prk = prk_pair_v1(sid32=sid32, kpair32=kpair32)
    info = UVCCKDFInfoV1(
        kind=int(kind),
        prg_type=int(prg_type),
        pair_id=int(pair_id),
        share_idx=int(share_idx),
        w_bits=int(w_bits),
        elem_bits=int(elem_bits),
        sid32=bytes(sid32),
        fss_id16=bytes(fss_id16),
        table_hash16=bytes(table_hash16),
    ).to_bytes()
    okm32 = hkdf_expand(prk32=prk, info=info, length=32)
    domain_tag16 = _domain_tag16_v1()

    if int(prg_type) == PRG_TYPE_AES128_FIXEDKEY:
        aes_key16 = okm32[0:16]
        rk176 = aes128_expand_rk_v1(aes_key16)
        return UVCCPRGCTXv1(prg_type=int(prg_type), impl=int(impl), domain_tag16=domain_tag16, aes_rk176=rk176, chacha_k32=b"\x00" * 32)
    if int(prg_type) == PRG_TYPE_CHACHA20_FIXEDKEY:
        return UVCCPRGCTXv1(prg_type=int(prg_type), impl=int(impl), domain_tag16=domain_tag16, aes_rk176=b"\x00" * 176, chacha_k32=okm32)
    raise ValueError("bad prg_type")


def derive_dpf_prg_ctx_v1(
    *,
    sid32: bytes,
    kpair32: bytes,
    prg_type: int,
    impl: int,
    pair_id: int,
    w_bits: int,
    elem_bits: int,
    fss_id16: bytes,
    table_hash16: bytes | None = None,
) -> UVCCPRGCTXv1:
    return derive_prg_ctx_v1(
        sid32=sid32,
        kpair32=kpair32,
        kind=KIND_DPF_PRG_CTX,
        prg_type=prg_type,
        impl=impl,
        pair_id=pair_id,
        share_idx=255,
        w_bits=w_bits,
        elem_bits=elem_bits,
        fss_id16=fss_id16,
        table_hash16=table_hash16,
    )


def derive_dcf_prg_ctx_v1(
    *,
    sid32: bytes,
    kpair32: bytes,
    prg_type: int,
    impl: int,
    pair_id: int,
    w_bits: int,
    elem_bits: int,
    fss_id16: bytes,
    table_hash16: bytes | None = None,
) -> UVCCPRGCTXv1:
    return derive_prg_ctx_v1(
        sid32=sid32,
        kpair32=kpair32,
        kind=KIND_DCF_PRG_CTX,
        prg_type=prg_type,
        impl=impl,
        pair_id=pair_id,
        share_idx=255,
        w_bits=w_bits,
        elem_bits=elem_bits,
        fss_id16=fss_id16,
        table_hash16=table_hash16,
    )


def prg_expand2_seed16_aes_fixedkey_v1(*, ctx: UVCCPRGCTXv1, seed16: bytes, lvl: int) -> Tuple[bytes, int, bytes, int]:
    """
    Expand one parent seed into (seedL,tL,seedR,tR) for AES128_FIXEDKEY backend.
    """
    if int(ctx.prg_type) != PRG_TYPE_AES128_FIXEDKEY:
        raise ValueError("ctx.prg_type mismatch")
    if len(seed16) != 16:
        raise ValueError("seed16 must be 16 bytes")
    lvl16 = int(lvl) & 0xFFFF

    def tweak(side: int) -> bytes:
        t = bytearray(ctx.domain_tag16)
        t[0] ^= lvl16 & 0xFF
        t[1] ^= (lvl16 >> 8) & 0xFF
        t[2] ^= int(side) & 0xFF
        return bytes(t)

    blkL_in = bytes((seed16[i] ^ tweak(0)[i]) & 0xFF for i in range(16))
    blkR_in = bytes((seed16[i] ^ tweak(1)[i]) & 0xFF for i in range(16))
    rawL = bytearray(aes128_encrypt_fixedrk_v1(rk176=ctx.aes_rk176, block16=blkL_in))
    rawR = bytearray(aes128_encrypt_fixedrk_v1(rk176=ctx.aes_rk176, block16=blkR_in))
    tL = int(rawL[0] & 1)
    tR = int(rawR[0] & 1)
    rawL[0] &= 0xFE
    rawR[0] &= 0xFE
    return bytes(rawL), tL, bytes(rawR), tR


def prg_expand2_seed16_chacha20_fixedkey_v1(*, ctx: UVCCPRGCTXv1, seed16: bytes, lvl: int) -> Tuple[bytes, int, bytes, int]:
    """
    Expand one parent seed into (seedL,tL,seedR,tR) for CHACHA20_FIXEDKEY backend.
    """
    if int(ctx.prg_type) != PRG_TYPE_CHACHA20_FIXEDKEY:
        raise ValueError("ctx.prg_type mismatch")
    if len(seed16) != 16:
        raise ValueError("seed16 must be 16 bytes")

    # key32 is okm32 interpreted as 8 little-endian u32 words.
    key32 = bytes(ctx.chacha_k32)

    lvl32 = int(lvl) & 0xFFFFFFFF
    seed_lo12 = seed16[0:12]
    seed_ctr = int.from_bytes(seed16[12:16], "little", signed=False)

    nonce = bytearray(12)
    nonce[0:4] = int(lvl32).to_bytes(4, "little", signed=False)
    for i in range(4, 12):
        nonce[i] = seed_lo12[i] ^ ctx.domain_tag16[i]
    ctr = seed_ctr ^ int.from_bytes(ctx.domain_tag16[12:16], "little", signed=False)

    from .dpf_dcf import chacha20_block_bytes_v1

    out64 = chacha20_block_bytes_v1(key32=key32, nonce12=bytes(nonce), counter32=int(ctr) & 0xFFFFFFFF)
    rawL = bytearray(out64[0:16])
    rawR = bytearray(out64[16:32])
    tL = int(rawL[0] & 1)
    tR = int(rawR[0] & 1)
    rawL[0] &= 0xFE
    rawR[0] &= 0xFE
    return bytes(rawL), tL, bytes(rawR), tR


