from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_4bf28fa973d7eb82,uvcc_group_f4494bb8307a6da6

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .b2a import B2APackV1, b2a_convert_batch_v1
from .dpf_dcf import (
    EDGE_01,
    EDGE_12,
    EDGE_20,
    KeyrecV1,
    PRIM_DCF,
    PRIM_DPF,
    dcf_eval_point_bit_w16_v1,
    dcf_eval_point_bit_w8_v1,
    dpf_eval_point_bit_v1,
)
from .gf2_and import GF2AndGate, gf2_and_batch_v1
from .gf2_triples import GF2TriplesPackV1
from .netframe import DT_U64, SegmentPayloadV1, build_netframe_v1
from .open import OpenArithItemU64, open_arith_u64_round_v1
from .party import DEFAULT_NET_TIMEOUT_S, DEFAULT_RELAY_TTL_S, Party
from .rss import RSSArithU64, RSSBoolU64Words
from .transcript import SegmentDescV1, sha256


MAGIC_TRUNC = b"UVCCTRN1"
VERSION_TRUNC = 1

# NetFrame msg_kind for TRUNC bit-share REPL (local v1; internal protocol traffic).
MSG_TRUNC_REPL = 0x0320

# TRUNC transcript leaf types (privacy_new.txt §6.3).
MSG_TRUNC_OPEN_ARITH_SEND = 0x6001
MSG_TRUNC_OPEN_ARITH_RESULT = 0x6002
MSG_TRUNC_CARRY_RESULT = 0x6003
MSG_TRUNC_OUTPUT_COMMIT = 0x6004

DS_TRUNC_OPEN_SEND = b"UVCC.trunc.openarith.send.v1\0"
DS_TRUNC_OPEN_RESULT = b"UVCC.trunc.openarith.result.v1\0"
DS_TRUNC_CARRY = b"UVCC.trunc.carry.result.v1\0"
DS_TRUNC_OUTPUT = b"UVCC.trunc.output.commit.v1\0"

# Transcript segment kinds (local v1).
SEG_TRUNC_OPEN_SHARE_LO = 10  # matches SEG_OPEN_SHARE_LO
SEG_TRUNC_OPEN_RESULT_PUB = 12  # matches OPEN result seg kind in open.py
SEG_TRUNC_CARRY_LO = 1001
SEG_TRUNC_CARRY_HI = 1002
SEG_TRUNC_OV_LO = 1003
SEG_TRUNC_OV_HI = 1004
SEG_TRUNC_Y_LO = 1010
SEG_TRUNC_Y_HI = 1011

_HDR = struct.Struct("<8sHHBBBB32sIQI")  # 64 bytes


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _u64_to_i64(x: int) -> int:
    x &= 0xFFFFFFFFFFFFFFFF
    return x if x < (1 << 63) else x - (1 << 64)


def _i64_to_u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _u64_tensor_to_le_bytes(x: torch.Tensor) -> bytes:
    if x.dtype != torch.int64:
        raise TypeError("expected int64 tensor of u64 bit-patterns")
    x = x.contiguous().cpu().view(-1).tolist()
    out = bytearray()
    for v in x:
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _le_bytes_to_u64_tensor(buf: bytes, n: int, device: torch.device) -> torch.Tensor:
    if len(buf) != 8 * int(n):
        raise ValueError("bad u64 bytes length")
    out = torch.empty((int(n),), dtype=torch.int64, device=device)
    for i in range(int(n)):
        out[i] = int.from_bytes(buf[8 * i : 8 * i + 8], "little", signed=True)
    return out


def _pack_bits01_to_u64_words(bits01: torch.Tensor, n_bits: int) -> torch.Tensor:
    if bits01.dtype != torch.int64 or bits01.ndim != 1:
        raise TypeError("bits01 must be 1-D int64")
    n = int(n_bits)
    if int(bits01.shape[0]) != n:
        raise ValueError("length mismatch")
    n_words = (n + 63) // 64
    out = torch.zeros((n_words,), dtype=torch.int64, device=bits01.device)
    for i in range(n):
        if int(bits01[i].item()) & 1:
            w = i >> 6
            b = i & 63
            out[w] = int(out[w].item()) ^ (1 << b)
    # Mask unused bits in last word.
    rem = n % 64
    if rem != 0 and n_words > 0:
        mask = (1 << rem) - 1
        out[-1] = int(out[-1].item()) & int(mask)
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


def _record_trunc_open_send_v1(
    party: Party,
    *,
    open_id: int,
    sub_id: int,
    send_to: int,
    share_lo_u64_i64: torch.Tensor,
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
    f_bits: int,
) -> None:
    if party.transcript is None:
        return
    pb = _u64_tensor_to_le_bytes(share_lo_u64_i64)
    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_TRUNC_OPEN_ARITH_SEND,
        sender=int(party.party_id),
        receiver=int(send_to),
        dir=0,
        seq_no=0,
        payload_bytes=len(pb),
        payload_hash32=sha256(pb),
        header_hash32=sha256(
            DS_TRUNC_OPEN_SEND
            + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
            + struct.pack("<QII", int(open_id) & 0xFFFFFFFFFFFFFFFF, int(sgir_op_id) & 0xFFFFFFFF, int(f_bits) & 0xFFFFFFFF)
        ),
        segments=[
            SegmentDescV1(
                seg_kind=SEG_TRUNC_OPEN_SHARE_LO,
                object_id=int(open_id) & 0xFFFFFFFF,
                sub_id=int(sub_id) & 0xFFFFFFFF,
                dtype=DT_U64,
                offset=0,
                length=len(pb),
                fxp_frac_bits=int(f_bits),
            )
        ],
    )


def _record_trunc_open_result_v1(
    party: Party,
    *,
    open_id: int,
    sub_id: int,
    c_pub_u64_i64: torch.Tensor,
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
    f_bits: int,
) -> None:
    if party.transcript is None:
        return
    pb = _u64_tensor_to_le_bytes(c_pub_u64_i64)
    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_TRUNC_OPEN_ARITH_RESULT,
        sender=int(party.party_id),
        receiver=int(party.party_id),
        dir=0,
        seq_no=0,
        payload_bytes=len(pb),
        payload_hash32=sha256(pb),
        header_hash32=sha256(
            DS_TRUNC_OPEN_RESULT
            + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
            + struct.pack("<QII", int(open_id) & 0xFFFFFFFFFFFFFFFF, int(sgir_op_id) & 0xFFFFFFFF, int(f_bits) & 0xFFFFFFFF)
        ),
        segments=[
            SegmentDescV1(
                seg_kind=SEG_TRUNC_OPEN_RESULT_PUB,
                object_id=int(open_id) & 0xFFFFFFFF,
                sub_id=int(sub_id) & 0xFFFFFFFF,
                dtype=DT_U64,
                offset=0,
                length=len(pb),
                fxp_frac_bits=int(f_bits),
            )
        ],
    )


def _record_trunc_carry_result_v1(
    party: Party,
    *,
    open_id: int,
    carry: RSSBoolU64Words,
    ov: RSSBoolU64Words,
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
    f_bits: int,
) -> None:
    if party.transcript is None:
        return
    c_lo = _u64_tensor_to_le_bytes(carry.lo_words)
    c_hi = _u64_tensor_to_le_bytes(carry.hi_words)
    o_lo = _u64_tensor_to_le_bytes(ov.lo_words)
    o_hi = _u64_tensor_to_le_bytes(ov.hi_words)
    payload = c_lo + c_hi + o_lo + o_hi
    descs: List[SegmentDescV1] = [
        SegmentDescV1(seg_kind=SEG_TRUNC_CARRY_LO, object_id=int(open_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, offset=0, length=len(c_lo), fxp_frac_bits=0),
        SegmentDescV1(seg_kind=SEG_TRUNC_CARRY_HI, object_id=int(open_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, offset=len(c_lo), length=len(c_hi), fxp_frac_bits=0),
        SegmentDescV1(seg_kind=SEG_TRUNC_OV_LO, object_id=int(open_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, offset=len(c_lo) + len(c_hi), length=len(o_lo), fxp_frac_bits=0),
        SegmentDescV1(
            seg_kind=SEG_TRUNC_OV_HI,
            object_id=int(open_id) & 0xFFFFFFFF,
            sub_id=0,
            dtype=DT_U64,
            offset=len(c_lo) + len(c_hi) + len(o_lo),
            length=len(o_hi),
            fxp_frac_bits=0,
        ),
    ]

    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_TRUNC_CARRY_RESULT,
        sender=int(party.party_id),
        receiver=int(party.party_id),
        dir=0,
        seq_no=0,
        payload_bytes=len(payload),
        payload_hash32=sha256(payload),
        header_hash32=sha256(
            DS_TRUNC_CARRY
            + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
            + struct.pack("<QII", int(open_id) & 0xFFFFFFFFFFFFFFFF, int(sgir_op_id) & 0xFFFFFFFF, int(f_bits) & 0xFFFFFFFF)
            + struct.pack("<II", int(carry.n_bits) & 0xFFFFFFFF, int(ov.n_bits) & 0xFFFFFFFF)
        ),
        segments=descs,
    )


def _record_trunc_output_commit_v1(
    party: Party,
    *,
    open_id: int,
    y: RSSArithU64,
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
    f_bits: int,
) -> None:
    if party.transcript is None:
        return
    y_lo = _u64_tensor_to_le_bytes(y.lo)
    y_hi = _u64_tensor_to_le_bytes(y.hi)
    payload = y_lo + y_hi
    segs = [
        SegmentDescV1(seg_kind=SEG_TRUNC_Y_LO, object_id=int(open_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, offset=0, length=len(y_lo), fxp_frac_bits=int(f_bits)),
        SegmentDescV1(seg_kind=SEG_TRUNC_Y_HI, object_id=int(open_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, offset=len(y_lo), length=len(y_hi), fxp_frac_bits=int(f_bits)),
    ]
    party.transcript.record_frame(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_TRUNC_OUTPUT_COMMIT,
        sender=int(party.party_id),
        receiver=int(party.party_id),
        dir=0,
        seq_no=0,
        payload_bytes=len(payload),
        payload_hash32=sha256(payload),
        header_hash32=sha256(
            DS_TRUNC_OUTPUT
            + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
            + struct.pack("<QII", int(open_id) & 0xFFFFFFFFFFFFFFFF, int(sgir_op_id) & 0xFFFFFFFF, int(f_bits) & 0xFFFFFFFF)
        ),
        segments=segs,
    )

def _xor(a: RSSBoolU64Words, b: RSSBoolU64Words) -> RSSBoolU64Words:
    if int(a.n_bits) != int(b.n_bits):
        raise ValueError("n_bits mismatch")
    return RSSBoolU64Words(lo_words=(a.lo_words ^ b.lo_words), hi_words=(a.hi_words ^ b.hi_words), n_bits=int(a.n_bits))


def _rss_arith_add_public_vec_into_share0(party_id: int, x: RSSArithU64, pub_u64_i64: torch.Tensor) -> RSSArithU64:
    if pub_u64_i64.dtype != torch.int64 or pub_u64_i64.shape != x.lo.shape:
        raise TypeError("pub_u64_i64 must be int64 tensor matching x shape")
    lo = x.lo
    hi = x.hi
    if int(party_id) == 0:
        lo = lo + pub_u64_i64
    if int(party_id) == 2:
        hi = hi + pub_u64_i64
    return RSSArithU64(lo=lo, hi=hi, fxp_frac_bits=x.fxp_frac_bits)


def _rss_arith_add_public_const_into_share0(party_id: int, x: RSSArithU64, c_pub_u64: int) -> RSSArithU64:
    c = int(c_pub_u64) & 0xFFFFFFFFFFFFFFFF
    c_i64 = _u64_to_i64(c)
    lo = x.lo
    hi = x.hi
    if int(party_id) == 0:
        lo = lo + int(c_i64)
    if int(party_id) == 2:
        hi = hi + int(c_i64)
    return RSSArithU64(lo=lo, hi=hi, fxp_frac_bits=x.fxp_frac_bits)


def _edge_parties(edge: int) -> Tuple[int, int, int]:
    e = int(edge)
    if e == EDGE_01:
        return 0, 1, 2
    if e == EDGE_12:
        return 1, 2, 0
    if e == EDGE_20:
        return 2, 0, 1
    raise ValueError("edge must be EDGE_01/EDGE_12/EDGE_20")


def _prg_edge_bits_v1(*, edge_key32: bytes, job_id32: bytes, sgir_op_id: int, fss_id: int, label: bytes, n_bits: int) -> torch.Tensor:
    """
    Expand an edge secret into a deterministic bitstream (0/1 int64).

    Security note: edge_key32 MUST be secret to the two parties on that edge; the lifter must not have it.
    """
    if len(edge_key32) != 32:
        raise ValueError("edge_key32 must be 32 bytes")
    if len(job_id32) != 32:
        raise ValueError("job_id32 must be 32 bytes")
    if not isinstance(label, (bytes, bytearray)):
        raise TypeError("label must be bytes")
    n = int(n_bits)
    if n < 0:
        raise ValueError("n_bits must be >= 0")
    out = torch.empty((n,), dtype=torch.int64, device=torch.device("cpu"))
    dom = b"UVCC.TRUNC.REPLBITS.v1\0"
    ctr = 0
    i = 0
    while i < n:
        h = hashlib.sha256(
            dom
            + edge_key32
            + job_id32
            + struct.pack("<I", int(sgir_op_id) & 0xFFFFFFFF)
            + struct.pack("<Q", int(fss_id) & 0xFFFFFFFFFFFFFFFF)
            + bytes(label)
            + struct.pack("<I", int(ctr) & 0xFFFFFFFF)
        ).digest()
        ctr += 1
        for byte in h:
            for k in range(8):
                if i >= n:
                    break
                out[i] = (byte >> k) & 1
                i += 1
            if i >= n:
                break
    return out


def _lift_2pc_xor_bits_to_rss_v1(
    party: Party,
    *,
    edge: int,
    fss_id: int,
    sgir_op_id: int,
    bits_share: torch.Tensor,
    edge_key32: Optional[bytes],
    epoch: int,
    step: int,
    round: int,
    label: bytes,
) -> RSSBoolU64Words:
    """
    Lift a 2PC XOR-share bit-vector on one edge into 3PC replicated XOR sharing (RSSBoolU64Words).

    This is the same construction used by the CMP fast-path:
      comp_edge0 = share_p
      comp_edge1 = m (edge-shared secret mask)
      comp_lifter = share_q XOR m
    """
    if bits_share.dtype != torch.int64 or bits_share.ndim != 1:
        raise TypeError("bits_share must be 1-D int64 0/1")
    n = int(bits_share.numel())
    p, q, l = _edge_parties(edge)
    pid = int(party.party_id)
    seq_no = int(fss_id) & 0xFFFFFFFF

    # Build the per-bitvector edge mask m (only p and q can compute).
    if pid in (p, q):
        if edge_key32 is None:
            raise ValueError("edge_key32 is required for edge parties")
        m_bits = _prg_edge_bits_v1(edge_key32=edge_key32, job_id32=party.job_id32, sgir_op_id=int(sgir_op_id), fss_id=int(fss_id), label=bytes(label), n_bits=n)
        m_words = _pack_bits01_to_u64_words(m_bits, n)
    else:
        m_words = torch.empty((0,), dtype=torch.int64)

    share_words = _pack_bits01_to_u64_words(bits_share, n)
    n_words = int(share_words.numel())

    def words_to_bytes(words: torch.Tensor) -> bytes:
        out = bytearray()
        for v in words.tolist():
            out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
        return bytes(out)

    # Edge party p sends comp_p (share_p) to lifter.
    if pid == p:
        comp_p = share_words
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=MSG_TRUNC_REPL,
            flags=0,
            sender=int(p),
            receiver=int(l),
            seq_no=int(seq_no),
            segments=[SegmentPayloadV1(seg_kind=20, object_id=int(sgir_op_id) & 0xFFFFFFFF, sub_id=0, dtype=DT_U64, fxp_frac_bits=0, payload=words_to_bytes(comp_p))],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=b"uvcc.trunc.repl.v1")
        return RSSBoolU64Words(lo_words=comp_p, hi_words=m_words, n_bits=n)

    # Edge party q sends comp_l (share_q XOR m) to lifter.
    if pid == q:
        comp_l = share_words ^ m_words
        frame = build_netframe_v1(
            job_id32=party.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=MSG_TRUNC_REPL,
            flags=0,
            sender=int(q),
            receiver=int(l),
            seq_no=int(seq_no),
            segments=[SegmentPayloadV1(seg_kind=20, object_id=int(sgir_op_id) & 0xFFFFFFFF, sub_id=1, dtype=DT_U64, fxp_frac_bits=0, payload=words_to_bytes(comp_l))],
        )
        party.send_netframe(frame=frame, ttl_s=int(DEFAULT_RELAY_TTL_S), relay_domain=b"uvcc.trunc.repl.v1")
        return RSSBoolU64Words(lo_words=m_words, hi_words=comp_l, n_bits=n)

    # Lifter receives comp_p from p and comp_l from q.
    f_p = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_TRUNC_REPL,
        sender=int(p),
        receiver=int(l),
        seq_no=int(seq_no),
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
        relay_domain=b"uvcc.trunc.repl.v1",
    )
    f_q = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_TRUNC_REPL,
        sender=int(q),
        receiver=int(l),
        seq_no=int(seq_no),
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
        relay_domain=b"uvcc.trunc.repl.v1",
    )

    def get_one_payload(frame, want_sub: int) -> bytes:
        seg = next((s for s in frame.segments if int(s.seg_kind) != 1 and int(s.sub_id) == int(want_sub)), None)
        if seg is None:
            raise ValueError("missing segment in TRUNC REPL frame")
        return frame.payload[int(seg.offset) : int(seg.offset) + int(seg.length)]

    bp = get_one_payload(f_p, 0)
    bl = get_one_payload(f_q, 1)
    comp_p = _le_bytes_to_u64_tensor(bp, n_words, device=torch.device("cpu"))
    comp_l = _le_bytes_to_u64_tensor(bl, n_words, device=torch.device("cpu"))
    return RSSBoolU64Words(lo_words=comp_l, hi_words=comp_p, n_bits=n)


@dataclass(frozen=True)
class TruncPackV1:
    flags: int
    f_bits: int
    sid_hash32: bytes
    sgir_op_id: int
    base_fss_id: int
    R: RSSArithU64
    R1: RSSArithU64
    R0: RSSArithU64

    @property
    def signed_mode(self) -> bool:
        return bool(int(self.flags) & 0x0001)


def parse_trunc_pack_v1(buf: bytes, *, lanes: Optional[int] = None, device: torch.device = torch.device("cpu")) -> TruncPackV1:
    if len(buf) < _HDR.size:
        raise ValueError("buffer too small for trunc pack")
    magic, ver, flags, k_bits, f_bits, chunk_bits, reserved0, sid_hash32, sgir_op_id, base_fss_id, reserved1 = _HDR.unpack_from(buf, 0)
    if magic != MAGIC_TRUNC:
        raise ValueError("bad magic")
    if int(ver) != VERSION_TRUNC:
        raise ValueError("bad version")
    if int(reserved0) != 0 or int(reserved1) != 0:
        raise ValueError("reserved fields must be 0")
    if int(k_bits) != 64:
        raise ValueError("k_bits must be 64")
    if int(chunk_bits) != 16:
        raise ValueError("chunk_bits must be 16")
    F = int(f_bits)
    if F < 0 or F > 63:
        raise ValueError("f_bits must be 0..63")

    body = memoryview(buf)[_HDR.size :]
    if lanes is None:
        if len(body) % 48 != 0:
            raise ValueError("bad trunc pack body length")
        lanes = len(body) // 48
    n = int(lanes)
    if len(body) != 48 * n:
        raise ValueError("bad trunc pack body length for lanes")

    r_lo = torch.empty((n,), dtype=torch.int64, device=device)
    r_hi = torch.empty((n,), dtype=torch.int64, device=device)
    r1_lo = torch.empty((n,), dtype=torch.int64, device=device)
    r1_hi = torch.empty((n,), dtype=torch.int64, device=device)
    r0_lo = torch.empty((n,), dtype=torch.int64, device=device)
    r0_hi = torch.empty((n,), dtype=torch.int64, device=device)

    off = 0
    for i in range(n):
        # R_pair
        lo = int.from_bytes(body[off : off + 8], "little", signed=False)
        hi = int.from_bytes(body[off + 8 : off + 16], "little", signed=False)
        r_lo[i] = int(_u64_to_i64(lo))
        r_hi[i] = int(_u64_to_i64(hi))
        off += 16
        # R1_pair
        lo = int.from_bytes(body[off : off + 8], "little", signed=False)
        hi = int.from_bytes(body[off + 8 : off + 16], "little", signed=False)
        r1_lo[i] = int(_u64_to_i64(lo))
        r1_hi[i] = int(_u64_to_i64(hi))
        off += 16
        # R0_pair
        lo = int.from_bytes(body[off : off + 8], "little", signed=False)
        hi = int.from_bytes(body[off + 8 : off + 16], "little", signed=False)
        r0_lo[i] = int(_u64_to_i64(lo))
        r0_hi[i] = int(_u64_to_i64(hi))
        off += 16

    return TruncPackV1(
        flags=int(flags),
        f_bits=F,
        sid_hash32=bytes(sid_hash32),
        sgir_op_id=int(sgir_op_id),
        base_fss_id=int(base_fss_id),
        R=RSSArithU64(lo=r_lo, hi=r_hi, fxp_frac_bits=0),
        R1=RSSArithU64(lo=r1_lo, hi=r1_hi, fxp_frac_bits=0),
        R0=RSSArithU64(lo=r0_lo, hi=r0_hi, fxp_frac_bits=0),
    )


def trunc_fss_id_carry_v1(base_fss_id: int) -> int:
    return int(base_fss_id) ^ 0x0001


def trunc_fss_id_lt_chunk_v1(base_fss_id: int, j: int) -> int:
    return int(base_fss_id) ^ (0x0100 + 2 * int(j))


def trunc_fss_id_eq_chunk_v1(base_fss_id: int, j: int) -> int:
    return int(base_fss_id) ^ (0x0101 + 2 * int(j))


@dataclass(frozen=True)
class TruncFSSKeysV1:
    edge: int
    # Per-lane keyrecs for this party. Lifter passes empty lists.
    carry_keyrecs: List[bytes]
    lt_keyrecs: Tuple[List[bytes], List[bytes], List[bytes], List[bytes]]  # j=0..3
    eq_keyrecs: Tuple[List[bytes], List[bytes], List[bytes], List[bytes]]  # j=0..3


def _eval_dcf_bits(
    party: Party,
    *,
    keyrecs: List[bytes],
    u16_list: List[int],
    w: int,
) -> torch.Tensor:
    n = len(u16_list)
    if int(party.party_id) in (0, 1, 2) and len(keyrecs) not in (0, n):
        raise ValueError("keyrecs length mismatch")
    bits = torch.zeros((n,), dtype=torch.int64, device=torch.device("cpu"))
    if len(keyrecs) == 0:
        return bits
    # Optional CUDA fast path (batch point-eval).
    if torch.cuda.is_available():
        cuda_ok = False
        try:
            from .cuda_ext import dcf_eval_point_w16_batch, dcf_eval_point_w8_batch

            key_bytes = len(keyrecs[0])
            if key_bytes > 0 and all(len(k) == key_bytes for k in keyrecs):
                blob = torch.tensor(list(b"".join(keyrecs)), dtype=torch.uint8, device="cuda")

                def u16_to_i16(u: int) -> int:
                    u &= 0xFFFF
                    return u - 0x10000 if u >= 0x8000 else u

                u_i16 = torch.tensor([u16_to_i16(u) for u in u16_list], dtype=torch.int16, device="cuda")
                if int(w) == 8:
                    out_u8 = dcf_eval_point_w8_batch(keyrecs_blob_u8=blob, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16)
                else:
                    out_u8 = dcf_eval_point_w16_batch(keyrecs_blob_u8=blob, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16)
                cuda_ok = True
                return out_u8.to(torch.int64).cpu().view(-1) & 1
        except Exception:
            cuda_ok = False

    for i in range(n):
        if int(w) == 8:
            bits[i] = int(dcf_eval_point_bit_w8_v1(keyrecs[i], u=u16_list[i], device=torch.device("cpu"))) & 1
        else:
            bits[i] = int(dcf_eval_point_bit_w16_v1(keyrecs[i], u=u16_list[i], device=torch.device("cpu"))) & 1
    return bits


def _eval_dpf_bits(party: Party, *, keyrecs: List[bytes], u16_list: List[int], w: int) -> torch.Tensor:
    n = len(u16_list)
    if len(keyrecs) not in (0, n):
        raise ValueError("keyrecs length mismatch")
    bits = torch.zeros((n,), dtype=torch.int64, device=torch.device("cpu"))
    if len(keyrecs) == 0:
        return bits
    # Optional CUDA fast path (batch point-eval).
    if torch.cuda.is_available():
        cuda_ok = False
        try:
            from .cuda_ext import dpf_eval_point_w16_batch, dpf_eval_point_w8_batch

            key_bytes = len(keyrecs[0])
            if key_bytes > 0 and all(len(k) == key_bytes for k in keyrecs):
                blob = torch.tensor(list(b"".join(keyrecs)), dtype=torch.uint8, device="cuda")

                def u16_to_i16(u: int) -> int:
                    u &= 0xFFFF
                    return u - 0x10000 if u >= 0x8000 else u

                u_i16 = torch.tensor([u16_to_i16(u) for u in u16_list], dtype=torch.int16, device="cuda")
                if int(w) == 8:
                    out_u8 = dpf_eval_point_w8_batch(keyrecs_blob_u8=blob, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16)
                else:
                    out_u8 = dpf_eval_point_w16_batch(keyrecs_blob_u8=blob, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16)
                cuda_ok = True
                return out_u8.to(torch.int64).cpu().view(-1) & 1
        except Exception:
            cuda_ok = False

    for i in range(n):
        bits[i] = int(dpf_eval_point_bit_v1(keyrecs[i], u=u16_list[i], device=torch.device("cpu"))) & 1
    return bits


def _u64_tensor_to_py_u64_list(x: torch.Tensor) -> List[int]:
    if x.dtype != torch.int64:
        raise TypeError("expected int64")
    xs = x.contiguous().view(-1).tolist()
    return [int(v) & 0xFFFFFFFFFFFFFFFF for v in xs]


def _u64_list_to_tensor(xs_u64: List[int]) -> torch.Tensor:
    vals = [int(_u64_to_i64(int(x))) for x in xs_u64]
    return torch.tensor(vals, dtype=torch.int64, device=torch.device("cpu"))


def op_trunc_prob_v1(
    party: Party,
    *,
    x: RSSArithU64,
    trunc_pack_blob: bytes,
    epoch: int,
    step: int,
    sgir_op_id: int,
    f_bits: int,
    signedness: int,
) -> RSSArithU64:
    """
    Probabilistic truncation (TRUNC_STOCH / TRUNC_PROB):
      OPEN(C = X + R), return Y = (C>>F) - R1 (and signed bias if requested).
    Uses only the TRUNC pack masks (R, R1).
    """
    pack = parse_trunc_pack_v1(trunc_pack_blob, lanes=int(x.lo.numel()), device=x.lo.device)
    F = int(f_bits)
    if F != int(pack.f_bits):
        raise ValueError("f_bits mismatch with pack")
    if int(signedness) not in (0, 1):
        raise ValueError("signedness must be 0/1")
    if bool(int(signedness)) != bool(pack.signed_mode):
        raise ValueError("signedness mismatch with pack flags")
    if pack.sid_hash32 != party.sid_hash32():
        raise ValueError("sid_hash32 mismatch")

    # Step 0: signed bias to unsigned.
    if int(signedness) == 1:
        x = _rss_arith_add_public_const_into_share0(int(party.party_id), x, 1 << 63)

    # Step 1: mask and open C = X + R.
    S = x.add(pack.R)
    open_id = int(sgir_op_id) & 0xFFFFFFFFFFFFFFFF
    _record_trunc_open_send_v1(
        party,
        open_id=int(open_id),
        sub_id=0,
        send_to=(int(party.party_id) + 1) % 3,
        share_lo_u64_i64=S.lo,
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )
    C = open_arith_u64_round_v1(
        party,
        items=[OpenArithItemU64(open_id=open_id, sub_id=0, x=S)],
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
    )[(open_id, 0)]
    _record_trunc_open_result_v1(
        party,
        open_id=int(open_id),
        sub_id=0,
        c_pub_u64_i64=C,
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )

    c_u64 = _u64_tensor_to_py_u64_list(C)
    c1_u64 = [(v >> F) & 0xFFFFFFFFFFFFFFFF for v in c_u64]
    C1 = _u64_list_to_tensor(c1_u64).to(device=x.lo.device)

    # Y = C1 - R1, with C1 placed into share-0.
    y = pack.R1
    y = RSSArithU64(lo=(-y.lo), hi=(-y.hi), fxp_frac_bits=x.fxp_frac_bits)
    y = _rss_arith_add_public_vec_into_share0(int(party.party_id), y, C1)

    # Undo signed bias.
    if int(signedness) == 1:
        bias_trunc = 1 << (63 - F) if F <= 63 else 0
        y = _rss_arith_add_public_const_into_share0(int(party.party_id), y, (0 - bias_trunc) & 0xFFFFFFFFFFFFFFFF)

    _record_trunc_output_commit_v1(
        party,
        open_id=int(open_id),
        y=y,
        epoch=int(epoch),
        step=int(step),
        round=1,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )
    return y


def op_trunc_exact_v1(
    party: Party,
    *,
    x: RSSArithU64,
    trunc_pack_blob: bytes,
    fss_keys: TruncFSSKeysV1,
    gf2_triples_blob: bytes,
    b2a_carry_blob: bytes,
    b2a_ov_blob: bytes,
    edge_key32: Optional[bytes],
    epoch: int,
    step: int,
    sgir_op_id: int,
    f_bits: int,
    signedness: int,
) -> RSSArithU64:
    """
    Exact truncation (TRUNC_FLOOR_EXACT):
      Implements `privacy_new.txt` §1.2–§1.6 (TRUNC pack + DPF/DCF compares + GF2 AND + B2A).
    """
    if int(signedness) not in (0, 1):
        raise ValueError("signedness must be 0/1")
    pack = parse_trunc_pack_v1(trunc_pack_blob, lanes=int(x.lo.numel()), device=x.lo.device)
    F = int(f_bits)
    if F != int(pack.f_bits):
        raise ValueError("f_bits mismatch with pack")
    if bool(int(signedness)) != bool(pack.signed_mode):
        raise ValueError("signedness mismatch with pack flags")
    if pack.sid_hash32 != party.sid_hash32():
        raise ValueError("sid_hash32 mismatch")

    n = int(x.lo.numel())
    if n <= 0:
        raise ValueError("empty vector")

    # Step 0: signed bias.
    if int(signedness) == 1:
        x = _rss_arith_add_public_const_into_share0(int(party.party_id), x, 1 << 63)

    # Step 1: mask and open C = X + R.
    S = x.add(pack.R)
    open_id = int(sgir_op_id) & 0xFFFFFFFFFFFFFFFF
    _record_trunc_open_send_v1(
        party,
        open_id=int(open_id),
        sub_id=0,
        send_to=(int(party.party_id) + 1) % 3,
        share_lo_u64_i64=S.lo,
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )
    C = open_arith_u64_round_v1(
        party,
        items=[OpenArithItemU64(open_id=open_id, sub_id=0, x=S)],
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
    )[(open_id, 0)]
    _record_trunc_open_result_v1(
        party,
        open_id=int(open_id),
        sub_id=0,
        c_pub_u64_i64=C,
        epoch=int(epoch),
        step=int(step),
        round=0,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )

    c_u64 = _u64_tensor_to_py_u64_list(C)
    c0_u16 = [int(v & ((1 << F) - 1)) if F > 0 else 0 for v in c_u64]
    c1_u64 = [(v >> F) & 0xFFFFFFFFFFFFFFFF for v in c_u64]
    C1 = _u64_list_to_tensor(c1_u64).to(device=x.lo.device)

    c_chunks = []
    for v in c_u64:
        c_chunks.append([(v >> 0) & 0xFFFF, (v >> 16) & 0xFFFF, (v >> 32) & 0xFFFF, (v >> 48) & 0xFFFF])

    # Step 2: carry_low = [C0 < R0] using DCF at width 8/16 (on fss_id_carry).
    fss_id_carry = trunc_fss_id_carry_v1(pack.base_fss_id)
    w_carry = 8 if F <= 8 else 16
    carry_share_bits = _eval_dcf_bits(party, keyrecs=fss_keys.carry_keyrecs, u16_list=c0_u16, w=w_carry)
    carry = _lift_2pc_xor_bits_to_rss_v1(
        party,
        edge=int(fss_keys.edge),
        fss_id=int(fss_id_carry),
        sgir_op_id=int(sgir_op_id),
        bits_share=carry_share_bits,
        edge_key32=edge_key32,
        epoch=int(epoch),
        step=int(step),
        round=1,
        label=b"carry_low",
    )

    # Step 3: overflow ov = [C < R] using 4 chunks of 16 bits.
    # Evaluate lt_j, eq_j for each chunk j=0..3. (We combine MSB-first 3..0.)
    lt: Dict[int, RSSBoolU64Words] = {}
    eq: Dict[int, RSSBoolU64Words] = {}
    for j in range(4):
        fss_id_lt = trunc_fss_id_lt_chunk_v1(pack.base_fss_id, j)
        fss_id_eq = trunc_fss_id_eq_chunk_v1(pack.base_fss_id, j)
        u_list = [int(c_chunks[i][j]) for i in range(n)]

        lt_bits = _eval_dcf_bits(party, keyrecs=fss_keys.lt_keyrecs[j], u16_list=u_list, w=16)
        eq_bits = _eval_dpf_bits(party, keyrecs=fss_keys.eq_keyrecs[j], u16_list=u_list, w=16)

        lt[j] = _lift_2pc_xor_bits_to_rss_v1(
            party,
            edge=int(fss_keys.edge),
            fss_id=int(fss_id_lt),
            sgir_op_id=int(sgir_op_id),
            bits_share=lt_bits,
            edge_key32=edge_key32,
            epoch=int(epoch),
            step=int(step),
            round=1,
            label=(b"lt" + bytes([j & 0xFF])),
        )
        eq[j] = _lift_2pc_xor_bits_to_rss_v1(
            party,
            edge=int(fss_keys.edge),
            fss_id=int(fss_id_eq),
            sgir_op_id=int(sgir_op_id),
            bits_share=eq_bits,
            edge_key32=edge_key32,
            epoch=int(epoch),
            step=int(step),
            round=1,
            label=(b"eq" + bytes([j & 0xFF])),
        )

    # Combine MSB-first:
    # ov = lt3 OR (eq3&lt2) OR (eq3&eq2&lt1) OR (eq3&eq2&eq1&lt0)
    lt3, lt2, lt1, lt0 = lt[3], lt[2], lt[1], lt[0]
    eq3, eq2, eq1 = eq[3], eq[2], eq[1]

    triples = GF2TriplesPackV1.from_bytes(gf2_triples_blob, device=torch.device("cpu"))
    if triples.sid_hash32 != party.sid_hash32():
        raise ValueError("gf2 triples sid_hash32 mismatch")
    if int(triples.count_triples) < 8 * n:
        raise ValueError("gf2 triples pack too small for TRUNC")

    # Build 8 AND gates in one OPEN_BOOL round.
    gates: List[GF2AndGate] = []
    cursor = int(triples.triple_id_base)
    gid = 0

    def take_triple() -> Tuple[RSSBoolU64Words, RSSBoolU64Words, RSSBoolU64Words]:
        nonlocal cursor
        a, b, c = triples.vector_at(triple_id_start=cursor, n_bits=n)
        cursor += n
        return a, b, c

    # g0 = eq3 & lt2
    a, b, c = take_triple()
    gates.append(GF2AndGate(gate_index=gid, x=eq3, y=lt2, triple_a=a, triple_b=b, triple_c=c))
    gid += 1
    # g1 = eq3 & eq2
    a, b, c = take_triple()
    gates.append(GF2AndGate(gate_index=gid, x=eq3, y=eq2, triple_a=a, triple_b=b, triple_c=c))
    gid += 1
    # Dependencies require multiple AND rounds: compute g0 and g1 first, then the dependent gates.
    outs01 = gf2_and_batch_v1(
        party,
        open_id=int(open_id),
        gates=gates,
        epoch=int(epoch),
        step=int(step),
        round=2,
        sub_id_base=0x1000,
        sgir_op_id=int(sgir_op_id),
    )
    g0 = outs01[0]
    g1 = outs01[1]

    # Second round for remaining dependent gates.
    gates2: List[GF2AndGate] = []
    gid2 = 0
    # h0 = g1 & lt1
    a, b, c = take_triple()
    gates2.append(GF2AndGate(gate_index=gid2, x=g1, y=lt1, triple_a=a, triple_b=b, triple_c=c))
    gid2 += 1
    # h1 = g1 & eq1
    a, b, c = take_triple()
    gates2.append(GF2AndGate(gate_index=gid2, x=g1, y=eq1, triple_a=a, triple_b=b, triple_c=c))
    gid2 += 1
    # h2 = h1 & lt0
    # (dependent; computed in a third round below)
    outs2 = gf2_and_batch_v1(
        party,
        open_id=int(open_id),
        gates=gates2,
        epoch=int(epoch),
        step=int(step),
        round=3,
        sub_id_base=0x1100,
        sgir_op_id=int(sgir_op_id),
    )
    h0 = outs2[0]
    h1 = outs2[1]

    # Third round: h2 and OR-chain ANDs.
    gates3: List[GF2AndGate] = []
    gid3 = 0
    # h2 = h1 & lt0
    a, b, c = take_triple()
    gates3.append(GF2AndGate(gate_index=gid3, x=h1, y=lt0, triple_a=a, triple_b=b, triple_c=c))
    gid3 += 1
    # or1_and = lt3 & g0
    a, b, c = take_triple()
    gates3.append(GF2AndGate(gate_index=gid3, x=lt3, y=g0, triple_a=a, triple_b=b, triple_c=c))
    gid3 += 1
    outs3 = gf2_and_batch_v1(
        party,
        open_id=int(open_id),
        gates=gates3,
        epoch=int(epoch),
        step=int(step),
        round=4,
        sub_id_base=0x1200,
        sgir_op_id=int(sgir_op_id),
    )
    h2 = outs3[0]
    and_or1 = outs3[1]

    # or1 = lt3 OR g0
    or1 = _xor(_xor(lt3, g0), and_or1)

    # Fourth round: finish OR chain.
    gates4: List[GF2AndGate] = []
    gid4 = 0
    # or2_and = or1 & h0
    a, b, c = take_triple()
    gates4.append(GF2AndGate(gate_index=gid4, x=or1, y=h0, triple_a=a, triple_b=b, triple_c=c))
    gid4 += 1
    outs4 = gf2_and_batch_v1(
        party,
        open_id=int(open_id),
        gates=gates4,
        epoch=int(epoch),
        step=int(step),
        round=5,
        sub_id_base=0x1300,
        sgir_op_id=int(sgir_op_id),
    )
    and_or2 = outs4[0]
    or2 = _xor(_xor(or1, h0), and_or2)

    gates5: List[GF2AndGate] = []
    a, b, c = take_triple()
    gates5.append(GF2AndGate(gate_index=0, x=or2, y=h2, triple_a=a, triple_b=b, triple_c=c))
    outs5 = gf2_and_batch_v1(
        party,
        open_id=int(open_id),
        gates=gates5,
        epoch=int(epoch),
        step=int(step),
        round=6,
        sub_id_base=0x1400,
        sgir_op_id=int(sgir_op_id),
    )
    and_or3 = outs5[0]
    ov = _xor(_xor(or2, h2), and_or3)

    _record_trunc_carry_result_v1(
        party,
        open_id=int(open_id),
        carry=carry,
        ov=ov,
        epoch=int(epoch),
        step=int(step),
        round=8,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )

    # Step 4: B2A conversions for carry and ov (batch into one OPEN_BOOL round).
    b2a_carry = B2APackV1.from_bytes(b2a_carry_blob)
    b2a_ov = B2APackV1.from_bytes(b2a_ov_blob)
    if int(b2a_carry.count_bits) != n or int(b2a_ov.count_bits) != n:
        raise ValueError("B2A pack size mismatch")
    if b2a_carry.sid_hash32 != party.sid_hash32() or b2a_ov.sid_hash32 != party.sid_hash32():
        raise ValueError("B2A sid_hash32 mismatch")

    b2a_out = b2a_convert_batch_v1(
        party,
        open_id=int(open_id),
        items=[
            (0x2000, carry, b2a_carry),
            (0x2001, ov, b2a_ov),
        ],
        epoch=int(epoch),
        step=int(step),
        round=7,
        sgir_op_id=int(sgir_op_id),
    )
    carry_a = b2a_out[0x2000]
    ov_a = b2a_out[0x2001]

    # Step 5: output arithmetic Y = C1 - R1 - carry + ov*2^(64-F)
    add_const = (1 << (64 - F)) & 0xFFFFFFFFFFFFFFFF if 0 < F < 64 else 0
    term_lo = ov_a.lo * int(_u64_to_i64(add_const))
    term_hi = ov_a.hi * int(_u64_to_i64(add_const))
    y = RSSArithU64(lo=term_lo - pack.R1.lo - carry_a.lo, hi=term_hi - pack.R1.hi - carry_a.hi, fxp_frac_bits=x.fxp_frac_bits)
    y = _rss_arith_add_public_vec_into_share0(int(party.party_id), y, C1)

    # Step 6: undo signed bias.
    if int(signedness) == 1:
        bias_trunc = 1 << (63 - F) if F <= 63 else 0
        y = _rss_arith_add_public_const_into_share0(int(party.party_id), y, (0 - bias_trunc) & 0xFFFFFFFFFFFFFFFF)

    _record_trunc_output_commit_v1(
        party,
        open_id=int(open_id),
        y=y,
        epoch=int(epoch),
        step=int(step),
        round=9,
        sgir_op_id=int(sgir_op_id),
        f_bits=int(F),
    )

    return y


