from __future__ import annotations

# pyright: reportMissingImports=false

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .dpf_dcf import chacha20_block_bytes_v1
from .netframe import DT_U64, SegmentPayloadV1, build_netframe_v1
from .open import _le_bytes_to_u64_tensor  # reuse canonical bytes->u64 tensor helper
from .party import DEFAULT_NET_TIMEOUT_S, DEFAULT_RELAY_TTL_S, Party
from .rss import RSSArithU64


# NetFrame msg_kind for TCF replication (privacy_new.txt §C.5.0).
MSG_TCF_REPL_V1 = 200

# Segment kinds for TCF replication (privacy_new.txt §C.2).
SEG_TCF_C0 = 20
SEG_TCF_C1 = 21
SEG_TCF_C2 = 22


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _u64_to_i64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    return x if x < (1 << 63) else x - (1 << 64)


def _u64_tensor_to_le_bytes(x_u64_i64: torch.Tensor) -> bytes:
    if not isinstance(x_u64_i64, torch.Tensor) or x_u64_i64.dtype != torch.int64:
        raise TypeError("expected torch.int64 tensor of u64 bit-patterns")
    xs = x_u64_i64.contiguous().view(-1).cpu().tolist()
    out = bytearray()
    for v in xs:
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _le_bytes_to_u64_matrix(buf: bytes, *, d: int, device: torch.device) -> torch.Tensor:
    n = int(d) * int(d)
    v = _le_bytes_to_u64_tensor(buf, n, device=device).view(int(d), int(d))
    return v


_TCF_V0B_CACHE_MAGIC = b"TCFCACH1"
_TCF_V0B_CACHE_HDR = struct.Struct("<8sIIII")  # magic, d, fxp_frac_bits, tile_id32, reserved


def _tcf_cache_pack_pair_v1(*, tile_id32: int, d: int, fxp_frac_bits: int, c_lo: torch.Tensor, c_hi: torch.Tensor) -> bytes:
    d = int(d)
    if c_lo.shape != (d, d) or c_hi.shape != (d, d) or c_lo.dtype != torch.int64 or c_hi.dtype != torch.int64:
        raise ValueError("bad C share shapes/dtypes for cache")
    hdr = _TCF_V0B_CACHE_HDR.pack(
        _TCF_V0B_CACHE_MAGIC,
        int(d) & 0xFFFFFFFF,
        int(fxp_frac_bits) & 0xFFFFFFFF,
        int(tile_id32) & 0xFFFFFFFF,
        0,
    )
    return bytes(hdr) + _u64_tensor_to_le_bytes(c_lo) + _u64_tensor_to_le_bytes(c_hi)


def _tcf_cache_unpack_pair_v1(blob: bytes, *, tile_id32: int, d: int, fxp_frac_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    d = int(d)
    if len(blob) < _TCF_V0B_CACHE_HDR.size:
        raise ValueError("bad cache blob (too small)")
    (magic, d_le, fxp_le, tile_le, reserved) = _TCF_V0B_CACHE_HDR.unpack_from(blob, 0)
    if magic != _TCF_V0B_CACHE_MAGIC:
        raise ValueError("bad cache magic")
    if int(reserved) != 0:
        raise ValueError("bad cache reserved")
    if int(d_le) != int(d):
        raise ValueError("cache d mismatch")
    if int(fxp_le) != (int(fxp_frac_bits) & 0xFFFFFFFF):
        raise ValueError("cache fxp_frac_bits mismatch")
    if int(tile_le) != (int(tile_id32) & 0xFFFFFFFF):
        raise ValueError("cache tile_id32 mismatch")
    need = _TCF_V0B_CACHE_HDR.size + 2 * (8 * d * d)
    if len(blob) != need:
        raise ValueError("bad cache blob length")
    off = _TCF_V0B_CACHE_HDR.size
    c0 = _le_bytes_to_u64_matrix(blob[off : off + 8 * d * d], d=d, device=torch.device("cpu"))
    off += 8 * d * d
    c1 = _le_bytes_to_u64_matrix(blob[off : off + 8 * d * d], d=d, device=torch.device("cpu"))
    return c0, c1


def tcf_tile_id32_v1(*, sid_hash32: bytes, op_id: int, i: int, j: int, p: int) -> int:
    """
    Deterministic tile id for wire object_id (privacy_new.txt §C.5.1).
    tile_id32 = Trunc32(SHA256(sid||op_id||i||j||p))
    """
    if len(sid_hash32) != 32:
        raise ValueError("sid_hash32 must be 32 bytes")
    h = hashlib.sha256()
    h.update(b"UVCC.TCF.tile_id.v1\0")
    h.update(sid_hash32)
    h.update(struct.pack("<IIII", int(op_id) & 0xFFFFFFFF, int(i) & 0xFFFFFFFF, int(j) & 0xFFFFFFFF, int(p) & 0xFFFFFFFF))
    return int.from_bytes(h.digest()[:4], "little", signed=False)


@dataclass(frozen=True)
class TCFKeyV1:
    """
    Per-party TCF key schedule aligned to RSS replicated shares (privacy_new.txt §10.3).

    P0 holds (s01,s02), P1 holds (s01,s12), P2 holds (s02,s12).
    """

    sid_hash32: bytes
    s01: bytes
    s02: bytes
    s12: bytes

    def __post_init__(self) -> None:
        if len(self.sid_hash32) != 32:
            raise ValueError("sid_hash32 must be 32 bytes")
        if len(self.s01) != 32 or len(self.s02) != 32 or len(self.s12) != 32:
            raise ValueError("TCF seeds must be 32 bytes each")


def tcf_gen_v1(*, master_seed32: bytes, sid: bytes) -> Tuple[TCFKeyV1, TCFKeyV1, TCFKeyV1]:
    """
    Deterministic keygen for TCF-v0a/v0b.
    Returns (K0,K1,K2) matching privacy_new.txt §10.1–§10.3.
    """
    if len(master_seed32) != 32:
        raise ValueError("master_seed32 must be 32 bytes")
    sid_hash32 = _sha256(bytes(sid))

    def derive(tag: bytes) -> bytes:
        return _sha256(b"UVCC.TCF.Gen.v1\0" + master_seed32 + sid_hash32 + tag)

    s01 = derive(b"s01")
    s02 = derive(b"s02")
    s12 = derive(b"s12")
    z = b"\x00" * 32
    # Enforce replicated-share key schedule: each party holds only two pairwise seeds (privacy_new.txt §10.3).
    k0 = TCFKeyV1(sid_hash32=sid_hash32, s01=s01, s02=s02, s12=z)
    k1 = TCFKeyV1(sid_hash32=sid_hash32, s01=s01, s02=z, s12=s12)
    k2 = TCFKeyV1(sid_hash32=sid_hash32, s01=z, s02=s02, s12=s12)
    return k0, k1, k2


def _tcf_prg_u64_tile_v1(*, seed32: bytes, label: bytes, tile_id32: int, d: int, device: torch.device) -> torch.Tensor:
    """
    PRG(seed,label,tile_id)-> u64[d,d] using ChaCha20 blocks (deterministic).
    """
    if len(seed32) != 32:
        raise ValueError("seed32 must be 32 bytes")
    d = int(d)
    if d <= 0:
        raise ValueError("d must be > 0")
    n_u64 = d * d
    # nonce12 derived deterministically from (label,tile_id32).
    nonce12 = _sha256(b"UVCC.TCF.PRG.nonce.v1\0" + bytes(label) + struct.pack("<I", int(tile_id32) & 0xFFFFFFFF))[:12]
    need = n_u64 * 8
    buf = bytearray()
    ctr = 0
    while len(buf) < need:
        buf += chacha20_block_bytes_v1(key32=seed32, nonce12=nonce12, counter32=int(ctr))
        ctr += 1
    out = torch.empty((n_u64,), dtype=torch.int64, device=device)
    mv = memoryview(buf)
    for i in range(n_u64):
        u = int.from_bytes(mv[8 * i : 8 * i + 8], "little", signed=False)
        out[i] = int(_u64_to_i64(u))
    return out.view(d, d)


def tcf_eval_v0a_tile_u64_v1(
    party: Party,
    *,
    key: TCFKeyV1,
    op_id: int,
    i: int,
    j: int,
    p: int,
    epoch: int,
    step: int,
    round: int,
    fxp_frac_bits: int = 0,
    d: int = 16,
    relay_domain: bytes = b"uvcc.tcf.repl.v1",
) -> Tuple[RSSArithU64, RSSArithU64, RSSArithU64]:
    """
    TCF-v0a tile triple generation (privacy_new.txt §10.4 / §C.5).

    Returns RSS shares of (A,B,C) for one dxd tile, such that:
      C = A @ B  (ring Z/2^64, u64 bit-patterns)
    """
    if key.sid_hash32 != party.sid_hash32():
        raise ValueError("sid_hash32 mismatch")

    d = int(d)
    tile_id32 = tcf_tile_id32_v1(sid_hash32=key.sid_hash32, op_id=int(op_id), i=int(i), j=int(j), p=int(p))
    pid = int(party.party_id)

    # Generate the two replicated A/B share components held at this party.
    if pid == 0:
        a0 = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        a1 = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b0 = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b1 = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        A = RSSArithU64(lo=a0, hi=a1, fxp_frac_bits=int(fxp_frac_bits))
        B = RSSArithU64(lo=b0, hi=b1, fxp_frac_bits=int(fxp_frac_bits))
        c0 = (a0 @ b0) + (a0 @ b1) + (a1 @ b0)
        # Replicate C0 to P2.
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(MSG_TCF_REPL_V1),
            flags=0,
            sender=0,
            receiver=2,
            seq_no=0,
            segments=[
                SegmentPayloadV1(
                    seg_kind=SEG_TCF_C0,
                    object_id=int(tile_id32) & 0xFFFFFFFF,
                    sub_id=0,
                    dtype=DT_U64,
                    fxp_frac_bits=int(fxp_frac_bits),
                    payload=_u64_tensor_to_le_bytes(c0),
                )
            ],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)
        # Receive C1 from P1.
        f_in = party.recv_netframe_expect(
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(MSG_TCF_REPL_V1),
            sender=1,
            receiver=0,
            seq_no=0,
            timeout_s=float(DEFAULT_NET_TIMEOUT_S),
            relay_domain=relay_domain,
        )
        seg = next((s for s in f_in.segments if int(s.seg_kind) == SEG_TCF_C1 and int(s.object_id) == int(tile_id32)), None)
        if seg is None:
            raise ValueError("missing TCF_C1 segment")
        c1 = _le_bytes_to_u64_matrix(f_in.payload[int(seg.offset) : int(seg.offset) + int(seg.length)], d=d, device=torch.device("cpu"))
        C = RSSArithU64(lo=c0, hi=c1, fxp_frac_bits=int(fxp_frac_bits))
        return A, B, C

    if pid == 1:
        a1 = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        a2 = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b1 = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b2 = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        A = RSSArithU64(lo=a1, hi=a2, fxp_frac_bits=int(fxp_frac_bits))
        B = RSSArithU64(lo=b1, hi=b2, fxp_frac_bits=int(fxp_frac_bits))
        c1 = (a1 @ b1) + (a1 @ b2) + (a2 @ b1)
        # Replicate C1 to P0.
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(MSG_TCF_REPL_V1),
            flags=0,
            sender=1,
            receiver=0,
            seq_no=0,
            segments=[
                SegmentPayloadV1(
                    seg_kind=SEG_TCF_C1,
                    object_id=int(tile_id32) & 0xFFFFFFFF,
                    sub_id=0,
                    dtype=DT_U64,
                    fxp_frac_bits=int(fxp_frac_bits),
                    payload=_u64_tensor_to_le_bytes(c1),
                )
            ],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)
        # Receive C2 from P2.
        f_in = party.recv_netframe_expect(
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(MSG_TCF_REPL_V1),
            sender=2,
            receiver=1,
            seq_no=0,
            timeout_s=float(DEFAULT_NET_TIMEOUT_S),
            relay_domain=relay_domain,
        )
        seg = next((s for s in f_in.segments if int(s.seg_kind) == SEG_TCF_C2 and int(s.object_id) == int(tile_id32)), None)
        if seg is None:
            raise ValueError("missing TCF_C2 segment")
        c2 = _le_bytes_to_u64_matrix(f_in.payload[int(seg.offset) : int(seg.offset) + int(seg.length)], d=d, device=torch.device("cpu"))
        C = RSSArithU64(lo=c1, hi=c2, fxp_frac_bits=int(fxp_frac_bits))
        return A, B, C

    if pid == 2:
        a2 = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        a0 = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b2 = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b0 = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        A = RSSArithU64(lo=a2, hi=a0, fxp_frac_bits=int(fxp_frac_bits))
        B = RSSArithU64(lo=b2, hi=b0, fxp_frac_bits=int(fxp_frac_bits))
        c2 = (a2 @ b2) + (a2 @ b0) + (a0 @ b2)
        # Replicate C2 to P1.
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(MSG_TCF_REPL_V1),
            flags=0,
            sender=2,
            receiver=1,
            seq_no=0,
            segments=[
                SegmentPayloadV1(
                    seg_kind=SEG_TCF_C2,
                    object_id=int(tile_id32) & 0xFFFFFFFF,
                    sub_id=0,
                    dtype=DT_U64,
                    fxp_frac_bits=int(fxp_frac_bits),
                    payload=_u64_tensor_to_le_bytes(c2),
                )
            ],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=relay_domain)
        # Receive C0 from P0.
        f_in = party.recv_netframe_expect(
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(MSG_TCF_REPL_V1),
            sender=0,
            receiver=2,
            seq_no=0,
            timeout_s=float(DEFAULT_NET_TIMEOUT_S),
            relay_domain=relay_domain,
        )
        seg = next((s for s in f_in.segments if int(s.seg_kind) == SEG_TCF_C0 and int(s.object_id) == int(tile_id32)), None)
        if seg is None:
            raise ValueError("missing TCF_C0 segment")
        c0 = _le_bytes_to_u64_matrix(f_in.payload[int(seg.offset) : int(seg.offset) + int(seg.length)], d=d, device=torch.device("cpu"))
        C = RSSArithU64(lo=c2, hi=c0, fxp_frac_bits=int(fxp_frac_bits))
        return A, B, C

    raise ValueError("party_id must be 0..2")


def tcf_eval_v0b_tile_u64_v1(
    party: Party,
    *,
    key: TCFKeyV1,
    op_id: int,
    i: int,
    j: int,
    p: int,
    epoch: int,
    step: int,
    round: int,
    fxp_frac_bits: int = 0,
    d: int = 16,
    relay_domain: bytes = b"uvcc.tcf.repl.v1",
    cache: Optional[Dict[int, bytes]] = None,
    allow_repl_on_miss: bool = True,
) -> Tuple[RSSArithU64, RSSArithU64, RSSArithU64]:
    """
    TCF-v0b: cached local expansion of (A,B,C) tiles by id.

    - A and B are always derived locally from the party's pairwise seeds (same as v0a).
    - C is loaded from a local cache keyed by tile_id32 when available.
    - On cache miss, if allow_repl_on_miss=True, this function deterministically falls back to v0a
      (1-round replication) and populates the cache.
    """
    if key.sid_hash32 != party.sid_hash32():
        raise ValueError("sid_hash32 mismatch")
    d = int(d)
    tile_id32 = tcf_tile_id32_v1(sid_hash32=key.sid_hash32, op_id=int(op_id), i=int(i), j=int(j), p=int(p))

    # Local A/B derivation (no communication).
    pid = int(party.party_id)
    if pid == 0:
        a_lo = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        a_hi = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b_lo = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b_hi = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    elif pid == 1:
        a_lo = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        a_hi = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b_lo = _tcf_prg_u64_tile_v1(seed32=key.s01, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b_hi = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    elif pid == 2:
        a_lo = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        a_hi = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b_lo = _tcf_prg_u64_tile_v1(seed32=key.s12, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
        b_hi = _tcf_prg_u64_tile_v1(seed32=key.s02, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    else:
        raise ValueError("party_id must be 0..2")

    A = RSSArithU64(lo=a_lo, hi=a_hi, fxp_frac_bits=int(fxp_frac_bits))
    B = RSSArithU64(lo=b_lo, hi=b_hi, fxp_frac_bits=int(fxp_frac_bits))

    # Cached C pair (per-party) if available.
    if cache is not None:
        blob = cache.get(int(tile_id32), None)
        if blob is not None:
            c_lo, c_hi = _tcf_cache_unpack_pair_v1(blob, tile_id32=int(tile_id32), d=d, fxp_frac_bits=int(fxp_frac_bits))
            C = RSSArithU64(lo=c_lo, hi=c_hi, fxp_frac_bits=int(fxp_frac_bits))
            return A, B, C

    if not bool(allow_repl_on_miss):
        raise ValueError("TCF-v0b cache miss (allow_repl_on_miss=False)")

    # Deterministic fallback to v0a, then populate cache.
    A2, B2, C2 = tcf_eval_v0a_tile_u64_v1(
        party,
        key=key,
        op_id=op_id,
        i=i,
        j=j,
        p=p,
        epoch=epoch,
        step=step,
        round=round,
        fxp_frac_bits=fxp_frac_bits,
        d=d,
        relay_domain=relay_domain,
    )
    if cache is not None:
        cache[int(tile_id32)] = _tcf_cache_pack_pair_v1(
            tile_id32=int(tile_id32),
            d=d,
            fxp_frac_bits=int(fxp_frac_bits),
            c_lo=C2.lo.to(torch.int64).contiguous(),
            c_hi=C2.hi.to(torch.int64).contiguous(),
        )
    return A2, B2, C2


