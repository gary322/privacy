from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_25803b2067d6fd77

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .open import OpenBoolItemWords, open_bool_words_round_v1
from .rss import RSSArithU64, RSSBoolU64Words


MAGIC_B2A = b"UVCCB2A1"
VERSION_B2A = 1

_HDR = struct.Struct("<8sHHIIQ32sI")  # 64 bytes


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _u64_to_i64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    return x if x < (1 << 63) else x - (1 << 64)


def _i64_to_u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _pack_bits01_to_u64_words(bits01: torch.Tensor, n_bits: int) -> torch.Tensor:
    if bits01.dtype != torch.int64 or bits01.ndim != 1:
        raise TypeError("bits01 must be 1-D int64")
    if int(bits01.shape[0]) != int(n_bits):
        raise ValueError("bits01 length mismatch")
    n = int(n_bits)
    n_words = (n + 63) // 64
    out = torch.zeros((n_words,), dtype=torch.int64, device=bits01.device)
    for i in range(n):
        w = i >> 6
        b = i & 63
        if int(bits01[i].item()) & 1:
            out[w] = int(out[w].item()) ^ (1 << b)
    return out


def _u64_words_to_bits01(words: torch.Tensor, n_bits: int) -> torch.Tensor:
    if words.dtype != torch.int64 or words.ndim != 1:
        raise TypeError("words must be 1-D int64")
    n = int(n_bits)
    out = torch.empty((n,), dtype=torch.int64, device=words.device)
    for i in range(n):
        w = i >> 6
        b = i & 63
        out[i] = (int(words[w].item()) >> b) & 1
    return out


@dataclass(frozen=True)
class B2APackV1:
    count_bits: int
    sgir_op_id: int
    base_stream_id: int
    sid_hash32: bytes
    r_bool_lo: torch.Tensor  # int64 shape (N,) bits 0/1
    r_bool_hi: torch.Tensor  # int64 shape (N,) bits 0/1
    r_arith_lo: torch.Tensor  # int64 shape (N,) u64 bit-patterns (0/1)
    r_arith_hi: torch.Tensor  # int64 shape (N,) u64 bit-patterns (0/1)

    def __post_init__(self) -> None:
        if len(self.sid_hash32) != 32:
            raise ValueError("sid_hash32 must be 32 bytes")
        n = int(self.count_bits)
        if n < 0:
            raise ValueError("count_bits must be >= 0")
        for t in (self.r_bool_lo, self.r_bool_hi, self.r_arith_lo, self.r_arith_hi):
            if not isinstance(t, torch.Tensor) or t.dtype != torch.int64 or t.ndim != 1:
                raise TypeError("pack tensors must be 1-D int64")
            if int(t.shape[0]) != n:
                raise ValueError("pack tensor length mismatch")

    def to_bytes(self) -> bytes:
        hdr = _HDR.pack(
            MAGIC_B2A,
            int(VERSION_B2A) & 0xFFFF,
            0,
            int(self.count_bits) & 0xFFFFFFFF,
            int(self.sgir_op_id) & 0xFFFFFFFF,
            int(self.base_stream_id) & 0xFFFFFFFFFFFFFFFF,
            bytes(self.sid_hash32),
            0,
        )
        out = bytearray(hdr)
        n = int(self.count_bits)
        for i in range(n):
            out += bytes([int(self.r_bool_lo[i].item()) & 1, int(self.r_bool_hi[i].item()) & 1])
            out += int(_i64_to_u64(int(self.r_arith_lo[i].item()))).to_bytes(8, "little", signed=False)
            out += int(_i64_to_u64(int(self.r_arith_hi[i].item()))).to_bytes(8, "little", signed=False)
        return bytes(out)

    @staticmethod
    def from_bytes(buf: bytes) -> "B2APackV1":
        if len(buf) < _HDR.size:
            raise ValueError("buffer too small")
        magic, ver, flags, count_bits, sgir_op_id, base_stream_id, sid_hash32, reserved0 = _HDR.unpack_from(buf, 0)
        if magic != MAGIC_B2A:
            raise ValueError("bad magic")
        if int(ver) != VERSION_B2A:
            raise ValueError("bad version")
        if int(flags) != 0 or int(reserved0) != 0:
            raise ValueError("reserved fields must be 0")
        n = int(count_bits)
        need = _HDR.size + 18 * n
        if len(buf) != need:
            raise ValueError("bad length for B2A pack")
        off = _HDR.size
        rbl = torch.empty((n,), dtype=torch.int64)
        rbh = torch.empty((n,), dtype=torch.int64)
        ral = torch.empty((n,), dtype=torch.int64)
        rah = torch.empty((n,), dtype=torch.int64)
        for i in range(n):
            rbl[i] = int(buf[off] & 1)
            rbh[i] = int(buf[off + 1] & 1)
            off += 2
            lo = int.from_bytes(buf[off : off + 8], "little", signed=False)
            hi = int.from_bytes(buf[off + 8 : off + 16], "little", signed=False)
            off += 16
            ral[i] = int(_u64_to_i64(lo))
            rah[i] = int(_u64_to_i64(hi))
        return B2APackV1(
            count_bits=n,
            sgir_op_id=int(sgir_op_id),
            base_stream_id=int(base_stream_id),
            sid_hash32=bytes(sid_hash32),
            r_bool_lo=rbl,
            r_bool_hi=rbh,
            r_arith_lo=ral,
            r_arith_hi=rah,
        )

    def r_bool_words(self, *, device: torch.device) -> RSSBoolU64Words:
        n = int(self.count_bits)
        lo = _pack_bits01_to_u64_words(self.r_bool_lo.to(device=device, dtype=torch.int64), n)
        hi = _pack_bits01_to_u64_words(self.r_bool_hi.to(device=device, dtype=torch.int64), n)
        return RSSBoolU64Words(lo_words=lo, hi_words=hi, n_bits=n)

    def r_arith(self, *, device: torch.device) -> RSSArithU64:
        return RSSArithU64(
            lo=self.r_arith_lo.to(device=device, dtype=torch.int64).contiguous(),
            hi=self.r_arith_hi.to(device=device, dtype=torch.int64).contiguous(),
            fxp_frac_bits=0,
        )


def build_b2a_packs_det_v1(
    *,
    sid: bytes,
    sgir_op_id: int,
    base_stream_id: int,
    count_bits: int,
    seed32: bytes,
) -> Tuple[bytes, bytes, bytes]:
    """
    Deterministic B2A pack builder for tests/demos.

    Produces three per-party pack blobs for the same public rb bit-vector, shared consistently
    in both boolean and arithmetic domains.
    """
    if len(seed32) != 32:
        raise ValueError("seed32 must be 32 bytes")
    sid_hash32 = _sha256(bytes(sid))
    n = int(count_bits)
    if n < 0:
        raise ValueError("count_bits must be >= 0")

    # Public rb bits (N).
    rb = torch.empty((n,), dtype=torch.int64)
    for i in range(n):
        h = _sha256(b"UVCC.b2a.rb.v1\0" + seed32 + sid_hash32 + struct.pack("<II", int(sgir_op_id) & 0xFFFFFFFF, i & 0xFFFFFFFF))
        rb[i] = int(h[0] & 1)

    # Boolean RSS components (b0,b1,b2) and arithmetic RSS components (a0,a1,a2), per bit.
    b0 = torch.empty((n,), dtype=torch.int64)
    b1 = torch.empty((n,), dtype=torch.int64)
    b2 = torch.empty((n,), dtype=torch.int64)
    a0 = torch.empty((n,), dtype=torch.int64)
    a1 = torch.empty((n,), dtype=torch.int64)
    a2 = torch.empty((n,), dtype=torch.int64)
    for i in range(n):
        h0 = _sha256(b"UVCC.b2a.b0.v1\0" + seed32 + sid_hash32 + struct.pack("<II", int(sgir_op_id) & 0xFFFFFFFF, i & 0xFFFFFFFF))
        h1 = _sha256(b"UVCC.b2a.b1.v1\0" + seed32 + sid_hash32 + struct.pack("<II", int(sgir_op_id) & 0xFFFFFFFF, i & 0xFFFFFFFF))
        b0[i] = int(h0[0] & 1)
        b1[i] = int(h1[0] & 1)
        b2[i] = int(rb[i].item()) ^ int(b0[i].item()) ^ int(b1[i].item())

        ha0 = _sha256(b"UVCC.b2a.a0.v1\0" + seed32 + sid_hash32 + struct.pack("<II", int(sgir_op_id) & 0xFFFFFFFF, i & 0xFFFFFFFF))
        ha1 = _sha256(b"UVCC.b2a.a1.v1\0" + seed32 + sid_hash32 + struct.pack("<II", int(sgir_op_id) & 0xFFFFFFFF, i & 0xFFFFFFFF))
        aa0 = int.from_bytes(ha0[0:8], "little", signed=False)
        aa1 = int.from_bytes(ha1[0:8], "little", signed=False)
        aa2 = (_i64_to_u64(int(rb[i].item())) - aa0 - aa1) & 0xFFFFFFFFFFFFFFFF
        a0[i] = int(_u64_to_i64(aa0))
        a1[i] = int(_u64_to_i64(aa1))
        a2[i] = int(_u64_to_i64(aa2))

    # Build per-party (lo,hi) pairs.
    def pack_for_party(pid: int) -> bytes:
        if pid == 0:
            rbl, rbh = b0, b1
            ral, rah = a0, a1
        elif pid == 1:
            rbl, rbh = b1, b2
            ral, rah = a1, a2
        else:
            rbl, rbh = b2, b0
            ral, rah = a2, a0
        p = B2APackV1(
            count_bits=n,
            sgir_op_id=int(sgir_op_id),
            base_stream_id=int(base_stream_id),
            sid_hash32=sid_hash32,
            r_bool_lo=rbl.clone(),
            r_bool_hi=rbh.clone(),
            r_arith_lo=ral.clone(),
            r_arith_hi=rah.clone(),
        )
        return p.to_bytes()

    return pack_for_party(0), pack_for_party(1), pack_for_party(2)


def b2a_convert_batch_v1(
    party,
    *,
    open_id: int,
    items: Sequence[Tuple[int, RSSBoolU64Words, B2APackV1]],
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
) -> Dict[int, RSSArithU64]:
    """
    Batch B2A conversions:
      - For each item: open d = b XOR rb (public), then map to arithmetic.
    Uses exactly one OPEN_BOOL round.
    """
    if not items:
        return {}

    # Build OPEN_BOOL items.
    open_items: List[OpenBoolItemWords] = []
    rb_words: Dict[int, RSSBoolU64Words] = {}
    ra_arith: Dict[int, RSSArithU64] = {}
    n_bits0: int | None = None
    for sub_id, b, pack in items:
        if n_bits0 is None:
            n_bits0 = int(b.n_bits)
        if int(b.n_bits) != int(n_bits0):
            raise ValueError("all B2A items must share the same n_bits")
        if int(pack.count_bits) != int(b.n_bits):
            raise ValueError("B2A pack count_bits must equal b.n_bits")
        if pack.sid_hash32 != party.sid_hash32():
            raise ValueError("sid_hash32 mismatch")
        rb = pack.r_bool_words(device=b.lo_words.device)
        ra = pack.r_arith(device=b.lo_words.device)
        d = RSSBoolU64Words(lo_words=(b.lo_words ^ rb.lo_words), hi_words=(b.hi_words ^ rb.hi_words), n_bits=b.n_bits)
        open_items.append(OpenBoolItemWords(open_id=int(open_id), sub_id=int(sub_id), x=d))
        rb_words[int(sub_id)] = rb
        ra_arith[int(sub_id)] = ra

    pub = open_bool_words_round_v1(party, items=open_items, epoch=int(epoch), step=int(step), round=int(round), sgir_op_id=int(sgir_op_id))

    out: Dict[int, RSSArithU64] = {}
    for sub_id, _b, _pack in items:
        d_pub_words = pub[(int(open_id), int(sub_id))]
        n_bits = int(_b.n_bits)
        d_bits01 = _u64_words_to_bits01(d_pub_words, n_bits)  # int64 0/1
        ra = ra_arith[int(sub_id)]

        # Correct 3PC B2A mapping (v1):
        # b = rb XOR d, where rb is shared both in bool and arith domains.
        # If d=0 -> b = rb (arith shares = ra).
        # If d=1 -> b = 1 - rb = (-rb) + 1. The "+1" must be applied to exactly ONE additive component (share-0),
        # using the same placement rule as other public constants: share-0 is held by (P0.lo, P2.hi).
        if int(party.party_id) == 0:
            add0_lo = d_bits01
            add0_hi = torch.zeros_like(d_bits01)
        elif int(party.party_id) == 2:
            add0_lo = torch.zeros_like(d_bits01)
            add0_hi = d_bits01
        else:
            add0_lo = torch.zeros_like(d_bits01)
            add0_hi = torch.zeros_like(d_bits01)

        # Conditional negation by public d: if d=1 => negate share; else keep.
        lo = torch.where(d_bits01 != 0, -ra.lo, ra.lo) + add0_lo
        hi = torch.where(d_bits01 != 0, -ra.hi, ra.hi) + add0_hi
        out[int(sub_id)] = RSSArithU64(lo=lo, hi=hi, fxp_frac_bits=0)

    return out


