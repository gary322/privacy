from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Tuple

import torch

from .rss import RSSBoolU64Words


MAGIC_GF2_TRIPLES_V1 = b"UVCCG2T1"
VERSION_GF2_TRIPLES_V1 = 1

_HDR = struct.Struct("<8sHHIQI32sI")  # 64 bytes


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _prg_bytes(seed32: bytes, label: bytes, n_bytes: int) -> bytes:
    if len(seed32) != 32:
        raise ValueError("seed32 must be 32 bytes")
    out = bytearray()
    ctr = 0
    while len(out) < n_bytes:
        out += _sha256(seed32 + label + struct.pack("<I", ctr))
        ctr += 1
    return bytes(out[:n_bytes])


def _prg_bits(seed32: bytes, label: bytes, n_bits: int) -> List[int]:
    n_bytes = (n_bits + 7) // 8
    b = _prg_bytes(seed32, label, n_bytes)
    out: List[int] = []
    for byte in b:
        for k in range(8):
            out.append((byte >> k) & 1)
            if len(out) >= n_bits:
                return out
    return out[:n_bits]


def _pack_bits_to_words(bits01: torch.Tensor, n_bits: int) -> torch.Tensor:
    if bits01.dtype != torch.int64 or bits01.ndim != 1:
        raise TypeError("bits01 must be 1-D int64")
    if int(bits01.shape[0]) != int(n_bits):
        raise ValueError("bits01 length mismatch")
    n_words = (int(n_bits) + 63) // 64
    if n_words == 0:
        return torch.zeros((0,), dtype=torch.int64, device=bits01.device)
    pow2 = torch.tensor([(1 << k) if k < 63 else -(1 << 63) for k in range(64)], dtype=torch.int64, device=bits01.device)
    words = torch.empty((n_words,), dtype=torch.int64, device=bits01.device)
    for wi in range(n_words):
        start = wi * 64
        end = min(int(n_bits), start + 64)
        chunk = bits01[start:end]
        words[wi] = (chunk * pow2[: (end - start)]).sum(dtype=torch.int64)
    return words


@dataclass(frozen=True)
class GF2TriplesPackV1:
    count_triples: int
    triple_id_base: int
    sgir_op_id: int
    sid_hash32: bytes

    a_lo: torch.Tensor
    a_hi: torch.Tensor
    b_lo: torch.Tensor
    b_hi: torch.Tensor
    c_lo: torch.Tensor
    c_hi: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.sid_hash32) != 32:
            raise ValueError("sid_hash32 must be 32 bytes")
        m = int(self.count_triples)
        for name in ("a_lo", "a_hi", "b_lo", "b_hi", "c_lo", "c_hi"):
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor) or t.dtype != torch.int64 or t.shape != (m,):
                raise ValueError(f"{name} must be int64 tensor shape ({m},)")
        # ensure 0/1
        for name in ("a_lo", "a_hi", "b_lo", "b_hi", "c_lo", "c_hi"):
            t = getattr(self, name)
            if torch.any((t != 0) & (t != 1)).item():
                raise ValueError(f"{name} must contain 0/1")

    def to_bytes(self) -> bytes:
        hdr = _HDR.pack(
            MAGIC_GF2_TRIPLES_V1,
            VERSION_GF2_TRIPLES_V1,
            0,
            int(self.count_triples) & 0xFFFFFFFF,
            int(self.triple_id_base) & 0xFFFFFFFFFFFFFFFF,
            int(self.sgir_op_id) & 0xFFFFFFFF,
            self.sid_hash32,
            0,
        )
        out = bytearray(hdr)
        m = int(self.count_triples)
        for i in range(m):
            out += struct.pack(
                "<BBBBBB",
                int(self.a_lo[i].item()) & 1,
                int(self.a_hi[i].item()) & 1,
                int(self.b_lo[i].item()) & 1,
                int(self.b_hi[i].item()) & 1,
                int(self.c_lo[i].item()) & 1,
                int(self.c_hi[i].item()) & 1,
            )
        return bytes(out)

    @staticmethod
    def from_bytes(buf: bytes, *, device: torch.device = torch.device("cpu")) -> "GF2TriplesPackV1":
        if len(buf) < 64:
            raise ValueError("buffer too small")
        magic, ver, flags, count_triples, triple_id_base, sgir_op_id, sid_hash32, reserved0 = _HDR.unpack_from(buf, 0)
        if magic != MAGIC_GF2_TRIPLES_V1:
            raise ValueError("bad magic")
        if int(ver) != VERSION_GF2_TRIPLES_V1:
            raise ValueError("bad version")
        if int(flags) != 0 or int(reserved0) != 0:
            raise ValueError("reserved fields must be 0")
        m = int(count_triples)
        want = 64 + 6 * m
        if len(buf) != want:
            raise ValueError("bad length")

        a_lo = torch.empty((m,), dtype=torch.int64, device=device)
        a_hi = torch.empty((m,), dtype=torch.int64, device=device)
        b_lo = torch.empty((m,), dtype=torch.int64, device=device)
        b_hi = torch.empty((m,), dtype=torch.int64, device=device)
        c_lo = torch.empty((m,), dtype=torch.int64, device=device)
        c_hi = torch.empty((m,), dtype=torch.int64, device=device)

        off = 64
        for i in range(m):
            al, ah, bl, bh, cl, ch = struct.unpack_from("<BBBBBB", buf, off)
            off += 6
            a_lo[i] = int(al) & 1
            a_hi[i] = int(ah) & 1
            b_lo[i] = int(bl) & 1
            b_hi[i] = int(bh) & 1
            c_lo[i] = int(cl) & 1
            c_hi[i] = int(ch) & 1

        return GF2TriplesPackV1(
            count_triples=m,
            triple_id_base=int(triple_id_base),
            sgir_op_id=int(sgir_op_id),
            sid_hash32=bytes(sid_hash32),
            a_lo=a_lo,
            a_hi=a_hi,
            b_lo=b_lo,
            b_hi=b_hi,
            c_lo=c_lo,
            c_hi=c_hi,
        )

    def vector_at(self, *, triple_id_start: int, n_bits: int) -> Tuple[RSSBoolU64Words, RSSBoolU64Words, RSSBoolU64Words]:
        """Return (a,b,c) as packed RSSBoolU64Words over `n_bits` lanes using triples [start .. start+n_bits)."""

        start = int(triple_id_start) - int(self.triple_id_base)
        if start < 0:
            raise ValueError("triple_id_start < triple_id_base")
        n = int(n_bits)
        end = start + n
        if end > int(self.count_triples):
            raise ValueError("triple slice out of range")

        a = RSSBoolU64Words(
            lo_words=_pack_bits_to_words(self.a_lo[start:end], n),
            hi_words=_pack_bits_to_words(self.a_hi[start:end], n),
            n_bits=n,
        )
        b = RSSBoolU64Words(
            lo_words=_pack_bits_to_words(self.b_lo[start:end], n),
            hi_words=_pack_bits_to_words(self.b_hi[start:end], n),
            n_bits=n,
        )
        c = RSSBoolU64Words
        c = RSSBoolU64Words(
            lo_words=_pack_bits_to_words(self.c_lo[start:end], n),
            hi_words=_pack_bits_to_words(self.c_hi[start:end], n),
            n_bits=n,
        )
        return a, b, c


def generate_gf2_triples_packs_v1(
    *,
    sid_hash32: bytes,
    triple_id_base: int,
    count_triples: int,
    seed32: bytes,
    sgir_op_id: int = 0,
) -> Tuple[bytes, bytes, bytes]:
    """Generate three per-party GF(2) triple pool packs deterministically from a master seed."""

    if len(sid_hash32) != 32:
        raise ValueError("sid_hash32 must be 32 bytes")
    m = int(count_triples)

    A = _prg_bits(seed32, b"gf2.A", m)
    B = _prg_bits(seed32, b"gf2.B", m)
    C = [int(A[i] & B[i]) for i in range(m)]

    def share_bits(val_bits: List[int], label: bytes) -> Tuple[List[int], List[int], List[int]]:
        s0 = _prg_bits(seed32, label + b".s0", m)
        s1 = _prg_bits(seed32, label + b".s1", m)
        s2 = [int(val_bits[i] ^ s0[i] ^ s1[i]) for i in range(m)]
        return s0, s1, s2

    A0, A1, A2 = share_bits(A, b"gf2.A")
    B0, B1, B2 = share_bits(B, b"gf2.B")
    C0, C1, C2 = share_bits(C, b"gf2.C")

    def build_party(party_id: int) -> bytes:
        if party_id == 0:
            a_lo, a_hi = A0, A1
            b_lo, b_hi = B0, B1
            c_lo, c_hi = C0, C1
        elif party_id == 1:
            a_lo, a_hi = A1, A2
            b_lo, b_hi = B1, B2
            c_lo, c_hi = C1, C2
        elif party_id == 2:
            a_lo, a_hi = A2, A0
            b_lo, b_hi = B2, B0
            c_lo, c_hi = C2, C0
        else:
            raise ValueError("party_id must be 0..2")

        hdr = _HDR.pack(
            MAGIC_GF2_TRIPLES_V1,
            VERSION_GF2_TRIPLES_V1,
            0,
            m & 0xFFFFFFFF,
            int(triple_id_base) & 0xFFFFFFFFFFFFFFFF,
            int(sgir_op_id) & 0xFFFFFFFF,
            sid_hash32,
            0,
        )
        out = bytearray(hdr)
        for i in range(m):
            out += struct.pack("<BBBBBB", a_lo[i] & 1, a_hi[i] & 1, b_lo[i] & 1, b_hi[i] & 1, c_lo[i] & 1, c_hi[i] & 1)
        return bytes(out)

    return (build_party(0), build_party(1), build_party(2))


