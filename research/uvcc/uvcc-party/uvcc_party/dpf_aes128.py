from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_f1817a0260a2d9bb

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Tuple

try:
    from Crypto.Cipher import AES as _AES  # type: ignore
except Exception:  # pragma: no cover
    _AES = None  # type: ignore


DS_DPF_KEYGEN = b"uvcc.dpf.keygen.v1"


def _u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _require_len(b: bytes, n: int, name: str) -> None:
    if not isinstance(b, (bytes, bytearray)) or len(b) != n:
        raise ValueError(f"{name} must be {n} bytes")


_C0 = b"\x00" * 16
_C1 = b"\x00" * 15 + b"\x01"
_C2 = b"\x00" * 15 + b"\x02"
_C3 = b"\x00" * 15 + b"\x03"
_C4 = b"\x00" * 15 + b"\x04"


def aes128_enc_fk_v1(*, key16: bytes, block16: bytes) -> bytes:
    """
    AES-128 "foreign-key" mode: the 16-byte seed is the AES key, block16 is plaintext.
    """
    _require_len(key16, 16, "key16")
    _require_len(block16, 16, "block16")
    if _AES is None:
        raise RuntimeError("Crypto.Cipher.AES not available")
    cipher = _AES.new(bytes(key16), _AES.MODE_ECB)
    return cipher.encrypt(bytes(block16))


def g_expand_aes128_v1(seed16: bytes) -> Tuple[bytes, int, bytes, int]:
    """
    G_expand(seed) -> (seedL,tL,seedR,tR) per privacy_new.txt §1.2.
    """
    b0 = aes128_enc_fk_v1(key16=seed16, block16=_C0)
    b1 = aes128_enc_fk_v1(key16=seed16, block16=_C1)
    b2 = aes128_enc_fk_v1(key16=seed16, block16=_C2)
    b3 = aes128_enc_fk_v1(key16=seed16, block16=_C3)
    seedL = b0
    seedR = b1
    tL = int(b2[0] & 1)
    tR = int(b3[0] & 1)
    return seedL, tL, seedR, tR


def V_aes128_u64_v1(seed16: bytes) -> int:
    """
    V(seed) = LE64(AES(seed,C4)[0..7]) per privacy_new.txt §1.2.
    """
    b4 = aes128_enc_fk_v1(key16=seed16, block16=_C4)
    return int.from_bytes(b4[0:8], "little", signed=False)


@dataclass(frozen=True)
class DPFKeyAES128AR64V1:
    w: int
    root_seed16: bytes
    root_t: int
    # Per level (i=0..w-1): cw_seed_L[16], cw_seed_R[16], cw_tL, cw_tR
    # Stored in-order by level to match `privacy_new.txt` §4.1 (DPFKey_GGM_AES128_AR64_v1).
    cw_seed_L: Tuple[bytes, ...]  # len=w, each 16 bytes
    cw_seed_R: Tuple[bytes, ...]  # len=w, each 16 bytes
    cw_tL: Tuple[int, ...]  # len=w, each 0/1
    cw_tR: Tuple[int, ...]  # len=w, each 0/1
    cw_last_u64: int

    def __post_init__(self) -> None:
        if int(self.w) not in (8, 16):
            raise ValueError("w must be 8 or 16")
        _require_len(self.root_seed16, 16, "root_seed16")
        if int(self.root_t) not in (0, 1):
            raise ValueError("root_t must be 0/1")
        if (
            len(self.cw_seed_L) != int(self.w)
            or len(self.cw_seed_R) != int(self.w)
            or len(self.cw_tL) != int(self.w)
            or len(self.cw_tR) != int(self.w)
        ):
            raise ValueError("cw arrays must have length w")
        for s in self.cw_seed_L:
            _require_len(s, 16, "cw_seed_L[i]")
        for s in self.cw_seed_R:
            _require_len(s, 16, "cw_seed_R[i]")
        for b in list(self.cw_tL) + list(self.cw_tR):
            if int(b) not in (0, 1):
                raise ValueError("cw_t bits must be 0/1")

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += bytes(self.root_seed16)
        out += bytes([int(self.root_t) & 1])
        for i in range(int(self.w)):
            out += bytes(self.cw_seed_L[i])
            out += bytes(self.cw_seed_R[i])
            out += bytes([int(self.cw_tL[i]) & 1])
            out += bytes([int(self.cw_tR[i]) & 1])
        out += _u64(int(self.cw_last_u64)).to_bytes(8, "little", signed=False)
        return bytes(out)

    @staticmethod
    def from_bytes(buf: bytes, *, w: int) -> "DPFKeyAES128AR64V1":
        W = int(w)
        if W not in (8, 16):
            raise ValueError("w must be 8 or 16")
        # root_seed16(16) + root_t(1) + w*(seedL16+seedR16+tL+tR) + cw_last_u64(8)
        need = 16 + 1 + W * (16 + 16 + 1 + 1) + 8
        if len(buf) != need:
            raise ValueError("bad key length")
        off = 0
        root_seed16 = bytes(buf[off : off + 16])
        off += 16
        root_t = int(buf[off] & 1)
        off += 1
        cw_seed_L: List[bytes] = []
        cw_seed_R: List[bytes] = []
        cw_tL: List[int] = []
        cw_tR: List[int] = []
        for _ in range(W):
            cw_seed_L.append(bytes(buf[off : off + 16]))
            off += 16
            cw_seed_R.append(bytes(buf[off : off + 16]))
            off += 16
            cw_tL.append(int(buf[off] & 1))
            off += 1
            cw_tR.append(int(buf[off] & 1))
            off += 1
        cw_last = int.from_bytes(buf[off : off + 8], "little", signed=False)
        return DPFKeyAES128AR64V1(
            w=W,
            root_seed16=root_seed16,
            root_t=root_t,
            cw_seed_L=tuple(cw_seed_L),
            cw_seed_R=tuple(cw_seed_R),
            cw_tL=tuple(cw_tL),
            cw_tR=tuple(cw_tR),
            cw_last_u64=int(cw_last),
        )


def keygen_dpf_aes128_ar64_v1(
    *,
    K_master32: bytes,
    sid32: bytes,
    fss_id: int,
    w: int,
    alpha: int,
    beta_u64: int,
) -> Tuple[bytes, bytes]:
    """
    Deterministic DPF keygen for arithmetic u64 output shares (privacy_new.txt §1.4).
    Returns (k0_bytes, k1_bytes) for parties b=0 and b=1.
    """
    _require_len(K_master32, 32, "K_master32")
    _require_len(sid32, 32, "sid32")
    W = int(w)
    if W not in (8, 16):
        raise ValueError("w must be 8 or 16")
    if int(alpha) < 0 or int(alpha) >= (1 << W):
        raise ValueError("alpha out of range")

    ctx = (
        DS_DPF_KEYGEN
        + bytes(sid32)
        + struct.pack("<Q", int(fss_id) & 0xFFFFFFFFFFFFFFFF)
        + struct.pack("<H", W & 0xFFFF)
        + struct.pack("<Q", int(alpha) & 0xFFFFFFFFFFFFFFFF)
        + struct.pack("<Q", _u64(int(beta_u64)))
    )
    R = _sha256(bytes(K_master32) + ctx)
    s_root_0 = bytes(R[0:16])
    s_root_1 = bytes(R[16:32])
    t0 = 0
    t1 = 1

    cw_seed_L: List[bytes] = []
    cw_seed_R: List[bytes] = []
    cw_tL: List[int] = []
    cw_tR: List[int] = []

    s0 = s_root_0
    s1 = s_root_1
    t0_cur = t0
    t1_cur = t1

    for i in range(W):
        s0L, t0L, s0R, t0R = g_expand_aes128_v1(s0)
        s1L, t1L, s1R, t1R = g_expand_aes128_v1(s1)
        abit = (int(alpha) >> (W - 1 - i)) & 1  # MSB-first
        if abit == 0:
            # keep L, lose R
            cws = bytes(x ^ y for x, y in zip(s0R, s1R))
            cwtR = t0R ^ t1R
            cwtL = t0L ^ t1L ^ 1
        else:
            # keep R, lose L
            cws = bytes(x ^ y for x, y in zip(s0L, s1L))
            cwtL = t0L ^ t1L
            cwtR = t0R ^ t1R ^ 1

        # Simulate eval: apply correction if parent t==1
        if t0_cur & 1:
            s0L = bytes(x ^ y for x, y in zip(s0L, cws))
            s0R = bytes(x ^ y for x, y in zip(s0R, cws))
            t0L ^= cwtL & 1
            t0R ^= cwtR & 1
        if t1_cur & 1:
            s1L = bytes(x ^ y for x, y in zip(s1L, cws))
            s1R = bytes(x ^ y for x, y in zip(s1R, cws))
            t1L ^= cwtL & 1
            t1R ^= cwtR & 1

        if abit == 0:
            s0, t0_cur = s0L, t0L & 1
            s1, t1_cur = s1L, t1L & 1
        else:
            s0, t0_cur = s0R, t0R & 1
            s1, t1_cur = s1R, t1R & 1

        cw_seed_L.append(cws)
        cw_seed_R.append(cws)
        cw_tL.append(int(cwtL) & 1)
        cw_tR.append(int(cwtR) & 1)

    v0 = V_aes128_u64_v1(s0)
    v1 = V_aes128_u64_v1(s1)
    diff = _u64(v0 - v1)
    dt = int(t0_cur) - int(t1_cur)  # +1 or -1
    beta = _u64(int(beta_u64))
    if dt == 1:
        cw_last = _u64(beta - diff)
    elif dt == -1:
        cw_last = _u64(diff - beta)
    else:
        raise ValueError("bad leaf dt (should be ±1)")

    k0 = DPFKeyAES128AR64V1(
        w=W,
        root_seed16=s_root_0,
        root_t=0,
        cw_seed_L=tuple(cw_seed_L),
        cw_seed_R=tuple(cw_seed_R),
        cw_tL=tuple(cw_tL),
        cw_tR=tuple(cw_tR),
        cw_last_u64=cw_last,
    )
    k1 = DPFKeyAES128AR64V1(
        w=W,
        root_seed16=s_root_1,
        root_t=1,
        cw_seed_L=tuple(cw_seed_L),
        cw_seed_R=tuple(cw_seed_R),
        cw_tL=tuple(cw_tL),
        cw_tR=tuple(cw_tR),
        cw_last_u64=cw_last,
    )
    return k0.to_bytes(), k1.to_bytes()


def eval_dpf_point_aes128_ar64_v1(*, key_bytes: bytes, w: int, x: int, party_b: int) -> int:
    """
    Evaluate arithmetic DPF at a single public x. party_b is 0 or 1 (sign handling).
    """
    W = int(w)
    if int(party_b) not in (0, 1):
        raise ValueError("party_b must be 0 or 1")
    k = DPFKeyAES128AR64V1.from_bytes(key_bytes, w=W)
    s = bytes(k.root_seed16)
    t = int(k.root_t) & 1
    xi = int(x) & ((1 << W) - 1)
    for i in range(W):
        sL, tL, sR, tR = g_expand_aes128_v1(s)
        if t:
            cwsL = k.cw_seed_L[i]
            cwsR = k.cw_seed_R[i]
            sL = bytes(a ^ b for a, b in zip(sL, cwsL))
            sR = bytes(a ^ b for a, b in zip(sR, cwsR))
            tL ^= int(k.cw_tL[i]) & 1
            tR ^= int(k.cw_tR[i]) & 1
        bit = (xi >> (W - 1 - i)) & 1
        if bit == 0:
            s, t = sL, tL & 1
        else:
            s, t = sR, tR & 1
    v = V_aes128_u64_v1(s)
    val = _u64(v + (k.cw_last_u64 if t else 0))
    if int(party_b) == 1:
        return _u64(0 - val)
    return val


