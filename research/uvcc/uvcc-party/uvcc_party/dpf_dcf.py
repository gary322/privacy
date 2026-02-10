from __future__ import annotations

# pyright: reportMissingImports=false

# UVCC_REQ_GROUP: uvcc_group_e5ab5bae0f6f1790,uvcc_group_2c46210f0240cf5f,uvcc_group_c66222b84339eca1,uvcc_group_22930b2f16ac685e,uvcc_group_acaeca358a192ce8,uvcc_group_65ed671ff3fd69f5,uvcc_group_e6c2716a57857b5f,uvcc_group_131735f7a41346a4,uvcc_group_b2a809ccbb581fc9,uvcc_group_e63d36190f1cb57e,uvcc_group_f1817a0260a2d9bb

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Tuple

import torch

try:
    from Crypto.Cipher import AES as _AES  # type: ignore
except Exception:  # pragma: no cover
    _AES = None  # type: ignore


MAGIC_KEYREC = b"UVCCFSS1"
VERSION_KEYREC = 1

PRIM_DPF = 0x21
PRIM_DCF = 0x22

EDGE_01 = 0
EDGE_12 = 1
EDGE_20 = 2

PRG_AES128 = 1
PRG_CHACHA12 = 2

FLAG_DCF_INVERT = 1 << 0
FLAG_DCF_HAS_PAYLOAD_MASK = 1 << 1


def _u64_to_i64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    return x if x < (1 << 63) else x - (1 << 64)


def _i64_to_u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


_HDR = struct.Struct("<8sHBBBBHQ32sHHI")  # 64 bytes


@dataclass(frozen=True)
class CW17V1:
    sigma16: bytes
    tau_mask: int  # bit0=tauL, bit1=tauR

    def __post_init__(self) -> None:
        if len(self.sigma16) != 16:
            raise ValueError("sigma16 must be 16 bytes")
        if int(self.tau_mask) & ~0x03:
            raise ValueError("tau_mask must fit 2 bits")


@dataclass(frozen=True)
class KeyrecV1:
    prim_type: int
    party_edge: int
    w: int
    prg_id: int
    flags: int
    fss_id: int
    sid_hash32: bytes
    root_seed16: bytes
    root_t: int  # 0/1
    cws: Tuple[CW17V1, ...]
    payload_mask_u64: int  # only for DCF; else 0

    def __post_init__(self) -> None:
        if len(self.sid_hash32) != 32:
            raise ValueError("sid_hash32 must be 32 bytes")
        if len(self.root_seed16) != 16:
            raise ValueError("root_seed16 must be 16 bytes")
        if int(self.root_t) not in (0, 1):
            raise ValueError("root_t must be 0/1")
        if int(self.w) not in (8, 16):
            raise ValueError("w must be 8 or 16")
        if len(self.cws) != int(self.w):
            raise ValueError("cws length must equal w")

    def to_bytes(self) -> bytes:
        cw_stride = 17
        hdr = _HDR.pack(
            MAGIC_KEYREC,
            VERSION_KEYREC,
            int(self.prim_type) & 0xFF,
            int(self.party_edge) & 0xFF,
            int(self.w) & 0xFF,
            int(self.prg_id) & 0xFF,
            int(self.flags) & 0xFFFF,
            int(self.fss_id) & 0xFFFFFFFFFFFFFFFF,
            self.sid_hash32,
            int(self.w) & 0xFFFF,
            cw_stride & 0xFFFF,
            0,
        )
        out = bytearray(hdr)
        out += self.root_seed16
        out += bytes([int(self.root_t) & 1])
        for cw in self.cws:
            out += cw.sigma16
            out += bytes([int(cw.tau_mask) & 0xFF])
        if int(self.prim_type) == PRIM_DCF:
            out += int(self.payload_mask_u64 & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
        return bytes(out)

    @staticmethod
    def from_bytes(buf: bytes) -> "KeyrecV1":
        if len(buf) < 64 + 17:
            raise ValueError("buffer too small for keyrec")
        (
            magic,
            ver,
            prim_type,
            party_edge,
            w,
            prg_id,
            flags,
            fss_id,
            sid_hash32,
            cw_count,
            cw_stride,
            reserved0,
        ) = _HDR.unpack_from(buf, 0)
        if magic != MAGIC_KEYREC:
            raise ValueError("bad magic")
        if int(ver) != VERSION_KEYREC:
            raise ValueError("bad version")
        if int(reserved0) != 0:
            raise ValueError("reserved0 must be 0")
        if int(cw_count) != int(w) or int(cw_stride) != 17:
            raise ValueError("cw_count/cw_stride mismatch")
        off = 64
        root_seed16 = bytes(buf[off : off + 16])
        off += 16
        root_t = int(buf[off] & 1)
        off += 1
        cws: List[CW17V1] = []
        for _ in range(int(w)):
            sigma16 = bytes(buf[off : off + 16])
            tau_mask = int(buf[off + 16])
            off += 17
            cws.append(CW17V1(sigma16=sigma16, tau_mask=tau_mask & 0x03))
        payload_mask_u64 = 0
        if int(prim_type) == PRIM_DCF:
            if len(buf) != off + 8:
                raise ValueError("bad DCF keyrec length")
            payload_mask_u64 = int.from_bytes(buf[off : off + 8], "little", signed=False)
        else:
            if len(buf) != off:
                raise ValueError("bad DPF keyrec length")
        return KeyrecV1(
            prim_type=int(prim_type),
            party_edge=int(party_edge),
            w=int(w),
            prg_id=int(prg_id),
            flags=int(flags),
            fss_id=int(fss_id),
            sid_hash32=bytes(sid_hash32),
            root_seed16=root_seed16,
            root_t=root_t,
            cws=tuple(cws),
            payload_mask_u64=int(payload_mask_u64),
        )


# ChaCha12 PRG expansion (torch-friendly, runs on CPU or CUDA).
_CHACHA_CONST = torch.tensor([0x61707865, 0x3320646E, 0x79622D32, 0x6B206574], dtype=torch.int64)


def _rotl32(x: torch.Tensor, r: int) -> torch.Tensor:
    x = x & 0xFFFFFFFF
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def _add32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x + y) & 0xFFFFFFFF


def _qr(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = _add32(a, b)
    d = _rotl32(d ^ a, 16)
    c = _add32(c, d)
    b = _rotl32(b ^ c, 12)
    a = _add32(a, b)
    d = _rotl32(d ^ a, 8)
    c = _add32(c, d)
    b = _rotl32(b ^ c, 7)
    return a, b, c, d


def _chacha_block_u32(key_words8: torch.Tensor, nonce_words3: torch.Tensor, *, counter32: int, rounds: int) -> torch.Tensor:
    """
    key_words8: int64 tensor shape (N,8) with u32 values
    nonce_words3: int64 tensor shape (N,3) with u32 values
    Returns: int64 tensor shape (N,16) with u32 values (little-endian words).
    """
    if key_words8.ndim != 2 or key_words8.shape[1] != 8:
        raise ValueError("key_words8 must be (N,8)")
    if nonce_words3.ndim != 2 or nonce_words3.shape[1] != 3:
        raise ValueError("nonce_words3 must be (N,3)")
    rr = int(rounds)
    if rr <= 0 or (rr % 2) != 0:
        raise ValueError("rounds must be a positive even integer")
    N = int(key_words8.shape[0])
    device = key_words8.device
    const = _CHACHA_CONST.to(device=device)
    state0 = torch.empty((N, 16), dtype=torch.int64, device=device)
    state0[:, 0:4] = const.view(1, 4).expand(N, 4)
    state0[:, 4:12] = key_words8 & 0xFFFFFFFF
    state0[:, 12] = int(counter32) & 0xFFFFFFFF
    state0[:, 13:16] = nonce_words3 & 0xFFFFFFFF

    x = state0.clone()
    # rounds = (rounds/2) double-rounds
    for _ in range(rr // 2):
        # column rounds
        x0, x4, x8, x12 = _qr(x[:, 0], x[:, 4], x[:, 8], x[:, 12])
        x1, x5, x9, x13 = _qr(x[:, 1], x[:, 5], x[:, 9], x[:, 13])
        x2, x6, x10, x14 = _qr(x[:, 2], x[:, 6], x[:, 10], x[:, 14])
        x3, x7, x11, x15 = _qr(x[:, 3], x[:, 7], x[:, 11], x[:, 15])
        x[:, 0], x[:, 4], x[:, 8], x[:, 12] = x0, x4, x8, x12
        x[:, 1], x[:, 5], x[:, 9], x[:, 13] = x1, x5, x9, x13
        x[:, 2], x[:, 6], x[:, 10], x[:, 14] = x2, x6, x10, x14
        x[:, 3], x[:, 7], x[:, 11], x[:, 15] = x3, x7, x11, x15
        # diagonal rounds
        x0, x5, x10, x15 = _qr(x[:, 0], x[:, 5], x[:, 10], x[:, 15])
        x1, x6, x11, x12 = _qr(x[:, 1], x[:, 6], x[:, 11], x[:, 12])
        x2, x7, x8, x13 = _qr(x[:, 2], x[:, 7], x[:, 8], x[:, 13])
        x3, x4, x9, x14 = _qr(x[:, 3], x[:, 4], x[:, 9], x[:, 14])
        x[:, 0], x[:, 5], x[:, 10], x[:, 15] = x0, x5, x10, x15
        x[:, 1], x[:, 6], x[:, 11], x[:, 12] = x1, x6, x11, x12
        x[:, 2], x[:, 7], x[:, 8], x[:, 13] = x2, x7, x8, x13
        x[:, 3], x[:, 4], x[:, 9], x[:, 14] = x3, x4, x9, x14
    out = _add32(x, state0)
    return out & 0xFFFFFFFF


def _chacha12_block_u32(key_words8: torch.Tensor, nonce_words3: torch.Tensor, *, counter32: int = 0) -> torch.Tensor:
    return _chacha_block_u32(key_words8, nonce_words3, counter32=int(counter32), rounds=12)


def _chacha20_block_u32(key_words8: torch.Tensor, nonce_words3: torch.Tensor, *, counter32: int = 0) -> torch.Tensor:
    return _chacha_block_u32(key_words8, nonce_words3, counter32=int(counter32), rounds=20)


def chacha20_block_bytes_v1(*, key32: bytes, nonce12: bytes, counter32: int) -> bytes:
    """
    Deterministic ChaCha20 block (RFC mapping) for test/validation.
    Returns 64 bytes.
    """
    if len(key32) != 32:
        raise ValueError("key32 must be 32 bytes")
    if len(nonce12) != 12:
        raise ValueError("nonce12 must be 12 bytes")
    key_words = [int.from_bytes(key32[4 * i : 4 * i + 4], "little", signed=False) for i in range(8)]
    nonce_words = [int.from_bytes(nonce12[4 * i : 4 * i + 4], "little", signed=False) for i in range(3)]
    key8 = torch.tensor([key_words], dtype=torch.int64)
    nonce3 = torch.tensor([nonce_words], dtype=torch.int64)
    out16 = _chacha20_block_u32(key8, nonce3, counter32=int(counter32))[0].tolist()
    out = bytearray()
    for w in out16:
        out += int(w & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
    return bytes(out)


def g_expand_chacha12_v1(seed_lo: torch.Tensor, seed_hi: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    G_expand_u128_to_2children (ChaCha12 mode) for a batch of seeds.

    Inputs are int64 tensors holding u64 bit-patterns:
    - seed_lo = bytes[0..7] as u64
    - seed_hi = bytes[8..15] as u64

    Returns:
    - SL_lo, SL_hi, SR_lo, SR_hi (int64 u64 bit-patterns)
    - tmask (int64) with bit0=tL, bit1=tR
    """
    if seed_lo.dtype != torch.int64 or seed_hi.dtype != torch.int64:
        raise TypeError("seed_lo/seed_hi must be int64 tensors")
    if seed_lo.shape != seed_hi.shape:
        raise ValueError("seed_lo/seed_hi shape mismatch")
    d = int(depth) & 0xFFFFFFFF
    device = seed_lo.device

    # Extract 4 u32 words from 128-bit seed (little-endian).
    w0 = seed_lo & 0xFFFFFFFF
    w1 = (seed_lo >> 32) & 0xFFFFFFFF
    w2 = seed_hi & 0xFFFFFFFF
    w3 = (seed_hi >> 32) & 0xFFFFFFFF
    key8 = torch.stack([w0, w1, w2, w3, w0, w1, w2, w3], dim=1).to(dtype=torch.int64)

    # nonce12 = b"G_SG2_v1" (8B) || LE32(d)
    # bytes: 47 5f 53 47 32 5f 76 31  (ASCII "G_SG2_v1")
    n0 = int.from_bytes(b"G_SG", "little") & 0xFFFFFFFF  # bytes0..3
    n1 = int.from_bytes(b"2_v1", "little") & 0xFFFFFFFF  # bytes4..7
    nonce3 = torch.empty((int(seed_lo.numel()), 3), dtype=torch.int64, device=device)
    nonce3[:, 0] = int(n0)
    nonce3[:, 1] = int(n1)
    nonce3[:, 2] = int(d)

    out16 = _chacha12_block_u32(key8.to(device=device), nonce3, counter32=0)
    # SL bytes[0..15] => words 0..3
    SL0 = out16[:, 0]
    SL1 = out16[:, 1]
    SL2 = out16[:, 2]
    SL3 = out16[:, 3]
    # SR bytes[16..31] => words 4..7
    SR0 = out16[:, 4]
    SR1 = out16[:, 5]
    SR2 = out16[:, 6]
    SR3 = out16[:, 7]
    # pack u32 words into u64 (little-endian): lo = word0 | (word1<<32)
    SL_lo = (SL0 & 0xFFFFFFFF) | ((SL1 & 0xFFFFFFFF) << 32)
    SL_hi = (SL2 & 0xFFFFFFFF) | ((SL3 & 0xFFFFFFFF) << 32)
    SR_lo = (SR0 & 0xFFFFFFFF) | ((SR1 & 0xFFFFFFFF) << 32)
    SR_hi = (SR2 & 0xFFFFFFFF) | ((SR3 & 0xFFFFFFFF) << 32)
    # Convert to signed int64 preserving bit-pattern.
    SL_lo = SL_lo.to(torch.int64)
    SL_hi = SL_hi.to(torch.int64)
    SR_lo = SR_lo.to(torch.int64)
    SR_hi = SR_hi.to(torch.int64)

    # tL = bytes[32]&1, tR = bytes[33]&1 => from word8 low two bytes.
    w8 = out16[:, 8]
    tL = (w8 & 1).to(torch.int64)
    tR = ((w8 >> 8) & 1).to(torch.int64)
    tmask = (tL & 1) | ((tR & 1) << 1)
    return SL_lo, SL_hi, SR_lo, SR_hi, tmask


def _aes128_enc_fk_block_v1(*, key16: bytes, block16: bytes) -> bytes:
    """
    AES-128 encrypt in ECB mode (foreign-key): key16 is AES key, block16 is plaintext.
    """
    if _AES is None:
        raise RuntimeError("Crypto.Cipher.AES not available (install pycryptodome)")
    if len(key16) != 16 or len(block16) != 16:
        raise ValueError("key16/block16 must be 16 bytes")
    cipher = _AES.new(bytes(key16), _AES.MODE_ECB)
    return cipher.encrypt(bytes(block16))


def g_expand_aes128_v1(seed_lo: torch.Tensor, seed_hi: torch.Tensor, depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    G_expand_u128_to_2children (AES128 mode) for a batch of seeds.

    Implements privacy_new.txt §3.1:
      Pk = (k as u64 little-endian) || LE64(d), k=0..3, treated as 16-byte plaintext blocks.
      E0 -> SL, E1 -> SR, tL = E2[0]&1, tR = E3[0]&1
    """
    if seed_lo.dtype != torch.int64 or seed_hi.dtype != torch.int64:
        raise TypeError("seed_lo/seed_hi must be int64 tensors")
    if seed_lo.shape != seed_hi.shape:
        raise ValueError("seed_lo/seed_hi shape mismatch")
    device = seed_lo.device
    # Work on CPU (AES implementation is CPU).
    lo_list = seed_lo.contiguous().view(-1).cpu().tolist()
    hi_list = seed_hi.contiguous().view(-1).cpu().tolist()

    d64 = int(depth) & 0xFFFFFFFFFFFFFFFF
    suffix8 = struct.pack("<Q", d64)

    SL_lo_out: List[int] = []
    SL_hi_out: List[int] = []
    SR_lo_out: List[int] = []
    SR_hi_out: List[int] = []
    tmask_out: List[int] = []

    for lo_i, hi_i in zip(lo_list, hi_list):
        key16 = (int(lo_i) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False) + (int(hi_i) & 0xFFFFFFFFFFFFFFFF).to_bytes(
            8, "little", signed=False
        )
        # Four fixed plaintext blocks per spec.
        p0 = bytes([0, 0, 0, 0, 0, 0, 0, 0]) + suffix8
        p1 = bytes([1, 0, 0, 0, 0, 0, 0, 0]) + suffix8
        p2 = bytes([2, 0, 0, 0, 0, 0, 0, 0]) + suffix8
        p3 = bytes([3, 0, 0, 0, 0, 0, 0, 0]) + suffix8
        e0 = _aes128_enc_fk_block_v1(key16=key16, block16=p0)
        e1 = _aes128_enc_fk_block_v1(key16=key16, block16=p1)
        e2 = _aes128_enc_fk_block_v1(key16=key16, block16=p2)
        e3 = _aes128_enc_fk_block_v1(key16=key16, block16=p3)
        sl_lo_i, sl_hi_i = _seed16_to_u64pair(e0)
        sr_lo_i, sr_hi_i = _seed16_to_u64pair(e1)
        tL = int(e2[0] & 1)
        tR = int(e3[0] & 1)
        SL_lo_out.append(int(sl_lo_i))
        SL_hi_out.append(int(sl_hi_i))
        SR_lo_out.append(int(sr_lo_i))
        SR_hi_out.append(int(sr_hi_i))
        tmask_out.append((tL & 1) | ((tR & 1) << 1))

    shape = seed_lo.shape
    SL_lo = torch.tensor(SL_lo_out, dtype=torch.int64).view(shape).to(device=device)
    SL_hi = torch.tensor(SL_hi_out, dtype=torch.int64).view(shape).to(device=device)
    SR_lo = torch.tensor(SR_lo_out, dtype=torch.int64).view(shape).to(device=device)
    SR_hi = torch.tensor(SR_hi_out, dtype=torch.int64).view(shape).to(device=device)
    tmask = torch.tensor(tmask_out, dtype=torch.int64).view(shape).to(device=device)
    return SL_lo, SL_hi, SR_lo, SR_hi, tmask


def g_expand_v1(seed_lo: torch.Tensor, seed_hi: torch.Tensor, depth: int, *, prg_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(prg_id) == PRG_AES128:
        return g_expand_aes128_v1(seed_lo, seed_hi, depth)
    if int(prg_id) == PRG_CHACHA12:
        return g_expand_chacha12_v1(seed_lo, seed_hi, depth)
    raise ValueError("unsupported prg_id")


def _sigma_to_u64pair(sigma16: bytes) -> Tuple[int, int]:
    lo = int.from_bytes(sigma16[0:8], "little", signed=False)
    hi = int.from_bytes(sigma16[8:16], "little", signed=False)
    return _u64_to_i64(lo), _u64_to_i64(hi)


def _seed16_to_u64pair(seed16: bytes) -> Tuple[int, int]:
    lo = int.from_bytes(seed16[0:8], "little", signed=False)
    hi = int.from_bytes(seed16[8:16], "little", signed=False)
    return _u64_to_i64(lo), _u64_to_i64(hi)


def dpf_stage1_w16_v1(keyrec_bytes: bytes, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU/GPU reference for uvcc_dpf_stage1_w16_v1.
    Returns frontier_seed_lo, frontier_seed_hi (int64), frontier_t (int64 0/1), frontier_acc (int64 0/1).
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.w) != 16:
        raise ValueError("stage1 only for w=16")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")

    seed_lo0, seed_hi0 = _seed16_to_u64pair(kr.root_seed16)
    seeds_lo = torch.tensor([seed_lo0], dtype=torch.int64, device=device)
    seeds_hi = torch.tensor([seed_hi0], dtype=torch.int64, device=device)
    t = torch.tensor([int(kr.root_t) & 1], dtype=torch.int64, device=device)

    for d in range(0, 8):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        # Child arrays are length 2*parents, interleaved [L0,R0,L1,R1,...].
        child_lo = torch.empty((SL_lo.numel() * 2,), dtype=torch.int64, device=device)
        child_hi = torch.empty((SL_hi.numel() * 2,), dtype=torch.int64, device=device)
        child_t = torch.empty((t.numel() * 2,), dtype=torch.int64, device=device)
        child_lo[0::2] = SL_lo
        child_hi[0::2] = SL_hi
        child_lo[1::2] = SR_lo
        child_hi[1::2] = SR_hi
        child_t[0::2] = (tmask & 1)
        child_t[1::2] = ((tmask >> 1) & 1)

        # Apply correction if parent t==1
        sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
        tau = int(kr.cws[d].tau_mask)
        tauL = tau & 1
        tauR = (tau >> 1) & 1
        parent_t = t.repeat_interleave(2)
        child_lo = child_lo ^ (parent_t * int(sigma_lo))
        child_hi = child_hi ^ (parent_t * int(sigma_hi))
        tau_child = torch.empty_like(child_t)
        tau_child[0::2] = int(tauL)
        tau_child[1::2] = int(tauR)
        child_t = child_t ^ (parent_t * tau_child)

        seeds_lo, seeds_hi, t = child_lo, child_hi, child_t

    frontier_seed_lo = seeds_lo.contiguous()
    frontier_seed_hi = seeds_hi.contiguous()
    frontier_t = t.contiguous()
    # frontier_acc[i] = XOR_{k < i} frontier_t[k] (exclusive) == cumsum mod2 shifted.
    inc = (torch.cumsum(frontier_t, dim=0) & 1).to(torch.int64)
    frontier_acc = torch.empty_like(frontier_t)
    frontier_acc[0] = 0
    frontier_acc[1:] = inc[:-1]
    return frontier_seed_lo, frontier_seed_hi, frontier_t, frontier_acc


def dcf_stage2_w16_v1(
    keyrec_bytes: bytes,
    *,
    frontier_seed_lo: torch.Tensor,
    frontier_seed_hi: torch.Tensor,
    frontier_t: torch.Tensor,
    frontier_acc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    CPU/GPU reference for uvcc_dcf_stage2_w16_v1.
    Returns out_word_u64 int64 tensor shape (65536,) with u64 bit-patterns.
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.prim_type) != PRIM_DCF:
        raise ValueError("dcf_stage2 requires DCF keyrec")
    if int(kr.w) != 16:
        raise ValueError("dcf_stage2 only for w=16")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")
    if frontier_seed_lo.numel() != 256 or frontier_seed_hi.numel() != 256 or frontier_t.numel() != 256 or frontier_acc.numel() != 256:
        raise ValueError("frontier buffers must be length 256")

    seeds_lo = frontier_seed_lo.to(device=device, dtype=torch.int64).contiguous()
    seeds_hi = frontier_seed_hi.to(device=device, dtype=torch.int64).contiguous()
    t = frontier_t.to(device=device, dtype=torch.int64).contiguous()

    for d in range(8, 16):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        child_lo = torch.empty((SL_lo.numel() * 2,), dtype=torch.int64, device=device)
        child_hi = torch.empty((SL_hi.numel() * 2,), dtype=torch.int64, device=device)
        child_t = torch.empty((t.numel() * 2,), dtype=torch.int64, device=device)
        child_lo[0::2] = SL_lo
        child_hi[0::2] = SL_hi
        child_lo[1::2] = SR_lo
        child_hi[1::2] = SR_hi
        child_t[0::2] = (tmask & 1)
        child_t[1::2] = ((tmask >> 1) & 1)

        sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
        tau = int(kr.cws[d].tau_mask)
        tauL = tau & 1
        tauR = (tau >> 1) & 1
        parent_t = t.repeat_interleave(2)
        child_lo = child_lo ^ (parent_t * int(sigma_lo))
        child_hi = child_hi ^ (parent_t * int(sigma_hi))
        tau_child = torch.empty_like(child_t)
        tau_child[0::2] = int(tauL)
        tau_child[1::2] = int(tauR)
        child_t = child_t ^ (parent_t * tau_child)

        seeds_lo, seeds_hi, t = child_lo, child_hi, child_t

    # Leaf t shares for x=0..65535 (lexicographic/MSB-first order).
    t_leaf = t.view(256, 256)
    # Inclusive prefix XOR within each block: XOR == sum mod2 for bits.
    P_share = (torch.cumsum(t_leaf, dim=1) & 1).to(torch.int64)
    carry = frontier_acc.to(device=device, dtype=torch.int64).view(256, 1)
    P_global = P_share ^ carry
    invert = 1 if (int(kr.flags) & FLAG_DCF_INVERT) else 0
    root_t = int(kr.root_t) & 1
    if invert:
        dcf_bit_share = P_global ^ int(root_t)
    else:
        dcf_bit_share = P_global

    mask = int(kr.payload_mask_u64) & 0xFFFFFFFFFFFFFFFF
    out_u64 = (dcf_bit_share.view(-1) * int(mask)).to(torch.int64)
    return out_u64


def dpf_stage2_w16_v1(
    keyrec_bytes: bytes,
    *,
    frontier_seed_lo: torch.Tensor,
    frontier_seed_hi: torch.Tensor,
    frontier_t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    CPU reference for DPF leaf t-bit shares over the full domain (w=16).
    Returns out_bit_i64 tensor shape (65536,) with values in {0,1}.
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.prim_type) != PRIM_DPF:
        raise ValueError("dpf_stage2 requires DPF keyrec")
    if int(kr.w) != 16:
        raise ValueError("dpf_stage2 only for w=16")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")
    if frontier_seed_lo.numel() != 256 or frontier_seed_hi.numel() != 256 or frontier_t.numel() != 256:
        raise ValueError("frontier buffers must be length 256")

    seeds_lo = frontier_seed_lo.to(device=device, dtype=torch.int64).contiguous()
    seeds_hi = frontier_seed_hi.to(device=device, dtype=torch.int64).contiguous()
    t = frontier_t.to(device=device, dtype=torch.int64).contiguous() & 1

    for d in range(8, 16):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        child_lo = torch.empty((SL_lo.numel() * 2,), dtype=torch.int64, device=device)
        child_hi = torch.empty((SL_hi.numel() * 2,), dtype=torch.int64, device=device)
        child_t = torch.empty((t.numel() * 2,), dtype=torch.int64, device=device)
        child_lo[0::2] = SL_lo
        child_hi[0::2] = SL_hi
        child_lo[1::2] = SR_lo
        child_hi[1::2] = SR_hi
        child_t[0::2] = (tmask & 1)
        child_t[1::2] = ((tmask >> 1) & 1)

        sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
        tau = int(kr.cws[d].tau_mask)
        tauL = tau & 1
        tauR = (tau >> 1) & 1
        parent_t = t.repeat_interleave(2)
        child_lo = child_lo ^ (parent_t * int(sigma_lo))
        child_hi = child_hi ^ (parent_t * int(sigma_hi))
        tau_child = torch.empty_like(child_t)
        tau_child[0::2] = int(tauL)
        tau_child[1::2] = int(tauR)
        child_t = child_t ^ (parent_t * tau_child)

        seeds_lo, seeds_hi, t = child_lo, child_hi, child_t & 1

    return (t.view(-1) & 1).to(torch.int64)


def dcf_full_w8_v1(keyrec_bytes: bytes, *, device: torch.device) -> torch.Tensor:
    """
    CPU/GPU reference for uvcc_dcf_full_w8_v1 (domain 256).
    Returns out_word_u64 int64 tensor shape (256,).
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.prim_type) != PRIM_DCF:
        raise ValueError("dcf_full_w8 requires DCF keyrec")
    if int(kr.w) != 8:
        raise ValueError("dcf_full_w8 only for w=8")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")

    seed_lo0, seed_hi0 = _seed16_to_u64pair(kr.root_seed16)
    seeds_lo = torch.tensor([seed_lo0], dtype=torch.int64, device=device)
    seeds_hi = torch.tensor([seed_hi0], dtype=torch.int64, device=device)
    t = torch.tensor([int(kr.root_t) & 1], dtype=torch.int64, device=device)

    for d in range(0, 8):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        child_lo = torch.empty((SL_lo.numel() * 2,), dtype=torch.int64, device=device)
        child_hi = torch.empty((SL_hi.numel() * 2,), dtype=torch.int64, device=device)
        child_t = torch.empty((t.numel() * 2,), dtype=torch.int64, device=device)
        child_lo[0::2] = SL_lo
        child_hi[0::2] = SL_hi
        child_lo[1::2] = SR_lo
        child_hi[1::2] = SR_hi
        child_t[0::2] = (tmask & 1)
        child_t[1::2] = ((tmask >> 1) & 1)

        sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
        tau = int(kr.cws[d].tau_mask)
        tauL = tau & 1
        tauR = (tau >> 1) & 1
        parent_t = t.repeat_interleave(2)
        child_lo = child_lo ^ (parent_t * int(sigma_lo))
        child_hi = child_hi ^ (parent_t * int(sigma_hi))
        tau_child = torch.empty_like(child_t)
        tau_child[0::2] = int(tauL)
        tau_child[1::2] = int(tauR)
        child_t = child_t ^ (parent_t * tau_child)

        seeds_lo, seeds_hi, t = child_lo, child_hi, child_t

    t_leaf = t.view(1, 256)
    P_share = (torch.cumsum(t_leaf, dim=1) & 1).to(torch.int64)
    invert = 1 if (int(kr.flags) & FLAG_DCF_INVERT) else 0
    root_t = int(kr.root_t) & 1
    if invert:
        dcf_bit_share = P_share ^ int(root_t)
    else:
        dcf_bit_share = P_share
    mask = int(kr.payload_mask_u64) & 0xFFFFFFFFFFFFFFFF
    out_u64 = (dcf_bit_share.view(-1) * int(mask)).to(torch.int64)
    return out_u64


def dpf_full_w8_v1(keyrec_bytes: bytes, *, device: torch.device) -> torch.Tensor:
    """
    CPU reference for DPF leaf t-bit shares over the full domain (w=8).
    Returns out_bit_i64 tensor shape (256,) with values in {0,1}.
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.prim_type) != PRIM_DPF:
        raise ValueError("dpf_full_w8 requires DPF keyrec")
    if int(kr.w) != 8:
        raise ValueError("dpf_full_w8 only for w=8")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")

    seed_lo0, seed_hi0 = _seed16_to_u64pair(kr.root_seed16)
    seeds_lo = torch.tensor([seed_lo0], dtype=torch.int64, device=device)
    seeds_hi = torch.tensor([seed_hi0], dtype=torch.int64, device=device)
    t = torch.tensor([int(kr.root_t) & 1], dtype=torch.int64, device=device)

    for d in range(0, 8):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        child_lo = torch.empty((SL_lo.numel() * 2,), dtype=torch.int64, device=device)
        child_hi = torch.empty((SL_hi.numel() * 2,), dtype=torch.int64, device=device)
        child_t = torch.empty((t.numel() * 2,), dtype=torch.int64, device=device)
        child_lo[0::2] = SL_lo
        child_hi[0::2] = SL_hi
        child_lo[1::2] = SR_lo
        child_hi[1::2] = SR_hi
        child_t[0::2] = (tmask & 1)
        child_t[1::2] = ((tmask >> 1) & 1)

        sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
        tau = int(kr.cws[d].tau_mask)
        tauL = tau & 1
        tauR = (tau >> 1) & 1
        parent_t = t.repeat_interleave(2)
        child_lo = child_lo ^ (parent_t * int(sigma_lo))
        child_hi = child_hi ^ (parent_t * int(sigma_hi))
        tau_child = torch.empty_like(child_t)
        tau_child[0::2] = int(tauL)
        tau_child[1::2] = int(tauR)
        child_t = child_t ^ (parent_t * tau_child)

        seeds_lo, seeds_hi, t = child_lo, child_hi, child_t & 1

    return (t.view(-1) & 1).to(torch.int64)


def dpf_eval_point_bit_v1(keyrec_bytes: bytes, *, u: int, device: torch.device) -> int:
    """
    Point-eval for bit-output DPF: returns this party's XOR-share bit (0/1) at public index u.
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.prim_type) != PRIM_DPF:
        raise ValueError("dpf_eval_point_bit_v1 requires DPF keyrec")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")
    W = int(kr.w)
    if W not in (8, 16):
        raise ValueError("w must be 8 or 16")
    x = int(u) & ((1 << W) - 1)

    seed_lo0, seed_hi0 = _seed16_to_u64pair(kr.root_seed16)
    seeds_lo = torch.tensor([seed_lo0], dtype=torch.int64, device=device)
    seeds_hi = torch.tensor([seed_hi0], dtype=torch.int64, device=device)
    t = int(kr.root_t) & 1

    for d in range(W):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        # scalar tensors
        SL_lo_i = int(SL_lo.item())
        SL_hi_i = int(SL_hi.item())
        SR_lo_i = int(SR_lo.item())
        SR_hi_i = int(SR_hi.item())
        tL = int(tmask.item()) & 1
        tR = (int(tmask.item()) >> 1) & 1
        if t & 1:
            sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
            SL_lo_i ^= int(sigma_lo)
            SL_hi_i ^= int(sigma_hi)
            SR_lo_i ^= int(sigma_lo)
            SR_hi_i ^= int(sigma_hi)
            tau = int(kr.cws[d].tau_mask) & 0x03
            tL ^= tau & 1
            tR ^= (tau >> 1) & 1
        bit = (x >> (W - 1 - d)) & 1
        if bit == 0:
            seeds_lo = torch.tensor([SL_lo_i], dtype=torch.int64, device=device)
            seeds_hi = torch.tensor([SL_hi_i], dtype=torch.int64, device=device)
            t = tL & 1
        else:
            seeds_lo = torch.tensor([SR_lo_i], dtype=torch.int64, device=device)
            seeds_hi = torch.tensor([SR_hi_i], dtype=torch.int64, device=device)
            t = tR & 1
    return int(t) & 1


def dcf_eval_point_bit_w8_v1(keyrec_bytes: bytes, *, u: int, device: torch.device) -> int:
    """
    Point-eval for bit-output DCF (w=8): returns this party's XOR-share bit (0/1) at public index u.

    v1 python reference uses full-domain expansion (256) and indexes at u.
    """
    vec = dcf_full_w8_v1(keyrec_bytes, device=device)
    return int(vec[int(u) & 0xFF].item() != 0)


def dcf_eval_point_bit_w16_v1(keyrec_bytes: bytes, *, u: int, device: torch.device) -> int:
    """
    Point-eval for bit-output DCF (w=16): returns this party's XOR-share bit (0/1) at public index u.

    This avoids full-domain 65536 output by:
    - stage1: expand to 256 frontier nodes (depth 8) to get carry for the selected prefix
    - stage2: expand only the selected prefix subtree (256 leaves) and scan within it
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    if int(kr.prim_type) != PRIM_DCF:
        raise ValueError("dcf_eval_point_bit_w16_v1 requires DCF keyrec")
    if int(kr.w) != 16:
        raise ValueError("dcf_eval_point_bit_w16_v1 only for w=16")
    if int(kr.prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")

    # Stage-1 frontier expansion (depth 8).
    frontier_seed_lo, frontier_seed_hi, frontier_t, frontier_acc = dpf_stage1_w16_v1(keyrec_bytes, device=device)
    x = int(u) & 0xFFFF
    prefix = (x >> 8) & 0xFF
    lane = x & 0xFF
    carry = int(frontier_acc[int(prefix)].item()) & 1

    # Stage-2 expand ONLY this prefix subtree (depth 8: global levels 8..15).
    seeds_lo = frontier_seed_lo[int(prefix) : int(prefix) + 1].to(device=device, dtype=torch.int64).contiguous()
    seeds_hi = frontier_seed_hi[int(prefix) : int(prefix) + 1].to(device=device, dtype=torch.int64).contiguous()
    t = frontier_t[int(prefix) : int(prefix) + 1].to(device=device, dtype=torch.int64).contiguous() & 1

    for d in range(8, 16):
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(seeds_lo, seeds_hi, d, prg_id=int(kr.prg_id))
        child_lo = torch.empty((SL_lo.numel() * 2,), dtype=torch.int64, device=device)
        child_hi = torch.empty((SL_hi.numel() * 2,), dtype=torch.int64, device=device)
        child_t = torch.empty((t.numel() * 2,), dtype=torch.int64, device=device)
        child_lo[0::2] = SL_lo
        child_hi[0::2] = SL_hi
        child_lo[1::2] = SR_lo
        child_hi[1::2] = SR_hi
        child_t[0::2] = (tmask & 1)
        child_t[1::2] = ((tmask >> 1) & 1)

        sigma_lo, sigma_hi = _sigma_to_u64pair(kr.cws[d].sigma16)
        tau = int(kr.cws[d].tau_mask)
        tauL = tau & 1
        tauR = (tau >> 1) & 1
        parent_t = t.repeat_interleave(2)
        child_lo = child_lo ^ (parent_t * int(sigma_lo))
        child_hi = child_hi ^ (parent_t * int(sigma_hi))
        tau_child = torch.empty_like(child_t)
        tau_child[0::2] = int(tauL)
        tau_child[1::2] = int(tauR)
        child_t = child_t ^ (parent_t * tau_child)

        seeds_lo, seeds_hi, t = child_lo, child_hi, child_t & 1

    # Inclusive prefix XOR within the 256-leaf block, then select `lane`.
    t_leaf = t.view(-1) & 1  # (256,)
    P_share = (torch.cumsum(t_leaf, dim=0) & 1).to(torch.int64)
    P_global = int(P_share[int(lane)].item()) ^ int(carry)

    invert = 1 if (int(kr.flags) & FLAG_DCF_INVERT) else 0
    root_t = int(kr.root_t) & 1
    dcf_bit_share = (P_global ^ root_t) & 1 if invert else (P_global & 1)
    return int(dcf_bit_share) & 1

def keygen_dpf_dcf_keyrecs_v1(
    *,
    sid: bytes,
    sid_hash32: bytes,
    fss_id: int,
    alpha: int,
    w: int,
    prg_id: int,
    party_edge: int,
    master_seed32: bytes,
    prim_type: int,
    dcf_invert: bool = True,
    payload_mask_u64: int = 1,
) -> Tuple[bytes, bytes]:
    """
    Deterministic keygen for one 2-party edge.

    Returns (keyrec_for_party_b0, keyrec_for_party_b1).
    """
    if len(sid_hash32) != 32:
        raise ValueError("sid_hash32 must be 32 bytes")
    if len(master_seed32) != 32:
        raise ValueError("master_seed32 must be 32 bytes")
    if int(w) not in (8, 16):
        raise ValueError("w must be 8 or 16")
    if int(prg_id) not in (PRG_AES128, PRG_CHACHA12):
        raise ValueError("unsupported prg_id")
    if int(alpha) < 0 or int(alpha) >= (1 << int(w)):
        raise ValueError("alpha out of range")
    if int(prim_type) not in (PRIM_DPF, PRIM_DCF):
        raise ValueError("prim_type must be DPF or DCF")

    # Derive per-edge key32' (privacy_new.txt §7.1).
    h = hashlib.sha256()
    h.update(b"UVCC.FSS.keygen.v1\0")
    h.update(master_seed32)
    h.update(bytes(sid))
    h.update(sid_hash32)
    h.update(struct.pack("<QIBBB", int(fss_id) & 0xFFFFFFFFFFFFFFFF, int(alpha) & 0xFFFFFFFF, int(w) & 0xFF, int(prg_id) & 0xFF, int(party_edge) & 0xFF))
    h.update(struct.pack("<B", int(prim_type) & 0xFF))
    h.update(struct.pack("<Q", int(payload_mask_u64) & 0xFFFFFFFFFFFFFFFF))
    key32p = h.digest()

    S0 = _sha256(key32p + b"root0")[:16]
    S1 = _sha256(key32p + b"root1")[:16]
    t0 = 0
    t1 = 1

    def g_expand_seed(seed16: bytes, d: int) -> Tuple[bytes, bytes, int, int]:
        lo = _u64_to_i64(int.from_bytes(seed16[0:8], "little", signed=False))
        hi = _u64_to_i64(int.from_bytes(seed16[8:16], "little", signed=False))
        SL_lo, SL_hi, SR_lo, SR_hi, tmask = g_expand_v1(
            torch.tensor([lo], dtype=torch.int64),
            torch.tensor([hi], dtype=torch.int64),
            d,
            prg_id=int(prg_id),
        )
        sl = _i64_to_u64(int(SL_lo.item())).to_bytes(8, "little", signed=False) + _i64_to_u64(int(SL_hi.item())).to_bytes(8, "little", signed=False)
        sr = _i64_to_u64(int(SR_lo.item())).to_bytes(8, "little", signed=False) + _i64_to_u64(int(SR_hi.item())).to_bytes(8, "little", signed=False)
        tL = int(tmask.item()) & 1
        tR = (int(tmask.item()) >> 1) & 1
        return sl, sr, tL, tR

    cws: List[CW17V1] = []
    S0_cur, S1_cur = S0, S1
    t0_cur, t1_cur = t0, t1
    for d in range(int(w)):
        SL0, SR0, tL0, tR0 = g_expand_seed(S0_cur, d)
        SL1, SR1, tL1, tR1 = g_expand_seed(S1_cur, d)
        a = (int(alpha) >> (int(w) - 1 - d)) & 1  # MSB-first
        if a == 0:
            sigma = bytes(x ^ y for x, y in zip(SR0, SR1))
            tauL = tL0 ^ tL1 ^ 1
            tauR = tR0 ^ tR1
        else:
            sigma = bytes(x ^ y for x, y in zip(SL0, SL1))
            tauL = tL0 ^ tL1
            tauR = tR0 ^ tR1 ^ 1

        # IMPORTANT: keygen must update the simulated keep-child states *as they will appear during eval*,
        # i.e. after applying the CW when parent t==1 (privacy_new.txt §3.4 eval semantics).
        if int(t0_cur) & 1:
            SL0 = bytes(x ^ y for x, y in zip(SL0, sigma))
            SR0 = bytes(x ^ y for x, y in zip(SR0, sigma))
            tL0 ^= int(tauL) & 1
            tR0 ^= int(tauR) & 1
        if int(t1_cur) & 1:
            SL1 = bytes(x ^ y for x, y in zip(SL1, sigma))
            SR1 = bytes(x ^ y for x, y in zip(SR1, sigma))
            tL1 ^= int(tauL) & 1
            tR1 ^= int(tauR) & 1

        if a == 0:
            S0_cur, S1_cur = SL0, SL1
            t0_cur, t1_cur = tL0, tL1
        else:
            S0_cur, S1_cur = SR0, SR1
            t0_cur, t1_cur = tR0, tR1
        tau_mask = (tauL & 1) | ((tauR & 1) << 1)
        cws.append(CW17V1(sigma16=sigma, tau_mask=tau_mask))

    def build_party(root_seed16: bytes, root_t: int) -> bytes:
        flags = 0
        pm = 0
        if int(prim_type) == PRIM_DCF:
            flags |= FLAG_DCF_HAS_PAYLOAD_MASK
            if bool(dcf_invert):
                flags |= FLAG_DCF_INVERT
            pm = int(payload_mask_u64) & 0xFFFFFFFFFFFFFFFF
        kr = KeyrecV1(
            prim_type=int(prim_type),
            party_edge=int(party_edge),
            w=int(w),
            prg_id=int(prg_id),
            flags=int(flags),
            fss_id=int(fss_id),
            sid_hash32=bytes(sid_hash32),
            root_seed16=bytes(root_seed16),
            root_t=int(root_t) & 1,
            cws=tuple(cws),
            payload_mask_u64=pm,
        )
        return kr.to_bytes()

    return build_party(S0, 0), build_party(S1, 1)


