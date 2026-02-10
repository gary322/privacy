from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_516fb18a3bfe557c,uvcc_group_20bb5431060336ca

import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .netframe import SEG_OPEN_SHARE_LO, SegmentPayloadV1, build_netframe_v1
from .party import DEFAULT_NET_TIMEOUT_S, DEFAULT_RELAY_TTL_S, Party
from .rss import RSSArithU64, RSSBoolU64Words
from .transcript import SegmentDescV1, sha256


# Message kind codes used in NetFrame headers (v1 uses the same numeric values as OPEN_* message types).
MSG_OPEN_BOOL = 0x0101
MSG_OPEN_ARITH = 0x0102

# Local transcript-only msg kinds for OPEN completion records (do not go on the wire).
MSG_OPEN_ARITH_RESULT = 0x0202
MSG_OPEN_BOOL_RESULT = 0x0203


# SGIR dtype codes (privacy_new.txt §A.6.1).
DT_U32 = 8
DT_U64 = 9

# Transcript segment kind for reconstructed public OPEN results (local record).
SEG_OPEN_RESULT_PUB = 12


def _u64_tensor_to_le_bytes(x: torch.Tensor) -> bytes:
    # x is int64 bit-patterns representing u64.
    if x.dtype != torch.int64:
        raise TypeError("u64 tensor must be int64")
    x = x.contiguous().cpu()
    out = bytearray()
    for v in x.view(-1).tolist():
        out += int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)
    return bytes(out)


def _le_bytes_to_u64_tensor(buf: bytes, n: int, device: torch.device) -> torch.Tensor:
    if len(buf) != 8 * int(n):
        raise ValueError("bad u64 bytes length")
    out = torch.empty((int(n),), dtype=torch.int64, device=device)
    for i in range(int(n)):
        # Interpret as signed int64 to preserve raw u64 bit-patterns in two's complement.
        out[i] = int.from_bytes(buf[8 * i : 8 * i + 8], "little", signed=True)
    return out


def _u64words_to_u32words(words64: torch.Tensor, n_bits: int) -> torch.Tensor:
    # Convert u64 bitset words -> u32 words (as int64) for OPEN_BOOL payload.
    if words64.dtype != torch.int64 or words64.ndim != 1:
        raise TypeError("words64 must be 1-D int64")
    n = int(n_bits)
    n_words32 = (n + 31) // 32
    out = torch.empty((n_words32,), dtype=torch.int64, device=words64.device)
    oi = 0
    for w in words64.tolist():
        if oi >= n_words32:
            break
        w_u = int(w) & 0xFFFFFFFFFFFFFFFF
        out[oi] = w_u & 0xFFFFFFFF
        oi += 1
        if oi >= n_words32:
            break
        out[oi] = (w_u >> 32) & 0xFFFFFFFF
        oi += 1
    # If n_words32 is odd and we didn't fill last due to 0 words64 (n_bits=0), handle.
    if oi < n_words32:
        out[oi:] = 0
    # Mask unused bits in last u32 word.
    rem = n % 32
    if rem != 0 and n_words32 > 0:
        mask = (1 << rem) - 1
        out[-1] = int(out[-1].item()) & mask
    return out


def _u32words_to_u64words(words32: torch.Tensor, n_bits: int) -> torch.Tensor:
    if words32.dtype != torch.int64 or words32.ndim != 1:
        raise TypeError("words32 must be 1-D int64")
    n = int(n_bits)
    n_words64 = (n + 63) // 64
    out = torch.zeros((n_words64,), dtype=torch.int64, device=words32.device)
    for wi in range(n_words64):
        lo = int(words32[2 * wi].item()) if (2 * wi) < int(words32.shape[0]) else 0
        hi = int(words32[2 * wi + 1].item()) if (2 * wi + 1) < int(words32.shape[0]) else 0
        u = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32)
        # Convert to signed int64 preserving bit-pattern.
        if u >= (1 << 63):
            u -= (1 << 64)
        out[wi] = int(u)
    # Mask unused bits in last u64 word.
    rem = n % 64
    if rem != 0 and n_words64 > 0:
        mask = (1 << rem) - 1
        out[-1] = int(out[-1].item()) & mask
    return out


@dataclass(frozen=True)
class OpenArithItemU64:
    open_id: int
    sub_id: int
    x: RSSArithU64


@dataclass(frozen=True)
class OpenBoolItemWords:
    open_id: int
    sub_id: int
    x: RSSBoolU64Words


def open_arith_u64_round_v1(
    party: Party,
    *,
    items: List[OpenArithItemU64],
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """OPEN_ARITH for u64 vectors. Returns {(open_id,sub_id): pub_u64_tensor_int64}."""

    if not items:
        return {}
    # Deterministic ordering by (open_id, sub_id) within a round.
    items_sorted = sorted(items, key=lambda it: (int(it.open_id), int(it.sub_id)))
    dst = party.next_party()
    src = party.prev_party()

    segs: List[SegmentPayloadV1] = []
    for it in items_sorted:
        xlo = it.x.lo.view(-1)
        segs.append(
            SegmentPayloadV1(
                seg_kind=SEG_OPEN_SHARE_LO,
                object_id=int(it.open_id),
                sub_id=int(it.sub_id),
                dtype=DT_U64,
                fxp_frac_bits=int(it.x.fxp_frac_bits),
                payload=_u64_tensor_to_le_bytes(xlo),
            )
        )

    # v1 MVP: one frame per OPEN round; seq_no is 0 (deterministic).
    frame_out = build_netframe_v1(
        job_id32=party.job_id32,
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_OPEN_ARITH,
        flags=0,
        sender=int(party.party_id),
        receiver=int(dst),
        seq_no=0,
        segments=segs,
    )
    party.send_netframe(frame=frame_out, ttl_s=int(DEFAULT_RELAY_TTL_S))

    frame_in = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_OPEN_ARITH,
        sender=int(src),
        receiver=int(party.party_id),
        seq_no=0,
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
    )

    recv_share: Dict[Tuple[int, int], bytes] = {}
    for sh in frame_in.segments:
        if int(sh.seg_kind) != SEG_OPEN_SHARE_LO:
            continue
        recv_share[(int(sh.object_id), int(sh.sub_id))] = frame_in.payload[int(sh.offset) : int(sh.offset) + int(sh.length)]

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for it in items_sorted:
        key = (int(it.open_id), int(it.sub_id))
        b = recv_share.get(key, None)
        if b is None:
            raise ValueError("missing OPEN_SHARE_LO segment for OPEN_ARITH operand")
        n = int(it.x.lo.numel())
        recv_prev = _le_bytes_to_u64_tensor(b, n, device=it.x.lo.device)
        out[key] = recv_prev + it.x.lo.view(-1) + it.x.hi.view(-1)

    # Record deterministic OPEN_ARITH_RESULT leaf committing to reconstructed public bytes.
    if party.transcript is not None:
        payload_parts: List[bytes] = []
        segs_result: List[SegmentDescV1] = []
        off0 = 0
        for it in items_sorted:
            key = (int(it.open_id), int(it.sub_id))
            pub_u64 = out[key].contiguous()
            pb = _u64_tensor_to_le_bytes(pub_u64)
            payload_parts.append(pb)
            segs_result.append(
                SegmentDescV1(
                    seg_kind=SEG_OPEN_RESULT_PUB,
                    object_id=int(it.open_id),
                    sub_id=int(it.sub_id),
                    dtype=DT_U64,
                    offset=int(off0),
                    length=len(pb),
                    fxp_frac_bits=int(it.x.fxp_frac_bits),
                )
            )
            off0 += len(pb)
        payload_all = b"".join(payload_parts)
        party.transcript.record_frame(
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=MSG_OPEN_ARITH_RESULT,
            sender=int(party.party_id),
            receiver=int(party.party_id),
            dir=0,
            seq_no=0,
            payload_bytes=len(payload_all),
            payload_hash32=sha256(payload_all),
            header_hash32=sha256(
                b"UVCC.openarith.result.v1\0"
                + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
                + struct.pack("<I", len(items_sorted) & 0xFFFFFFFF)
            ),
            segments=segs_result,
        )
    return out


def open_bool_words_round_v1(
    party: Party,
    *,
    items: List[OpenBoolItemWords],
    epoch: int,
    step: int,
    round: int,
    sgir_op_id: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """OPEN_BOOL for packed bit vectors (u64 internally; NetFrame payload uses packed u32 words)."""

    if not items:
        return {}
    items_sorted = sorted(items, key=lambda it: (int(it.open_id), int(it.sub_id)))
    dst = party.next_party()
    src = party.prev_party()

    n_bits = int(items_sorted[0].x.n_bits)
    for it in items_sorted:
        if int(it.x.n_bits) != n_bits:
            raise ValueError("OPEN_BOOL requires equal n_bits across vectors in a round")

    # Convert each vector lo_words (u64) to packed u32 words for wire payload.
    n_words32 = (n_bits + 31) // 32
    segs: List[SegmentPayloadV1] = []
    for it in items_sorted:
        w32 = _u64words_to_u32words(it.x.lo_words, n_bits)
        words_bytes = bytearray()
        for v in w32.tolist():
            words_bytes += int(v & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
        segs.append(
            SegmentPayloadV1(
                seg_kind=SEG_OPEN_SHARE_LO,
                object_id=int(it.open_id),
                sub_id=int(it.sub_id),
                dtype=DT_U32,
                fxp_frac_bits=0,
                payload=bytes(words_bytes),
            )
        )

    frame_out = build_netframe_v1(
        job_id32=party.job_id32,
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_OPEN_BOOL,
        flags=0,
        sender=int(party.party_id),
        receiver=int(dst),
        seq_no=0,
        segments=segs,
    )
    party.send_netframe(frame=frame_out, ttl_s=int(DEFAULT_RELAY_TTL_S))

    frame_in = party.recv_netframe_expect(
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        msg_kind=MSG_OPEN_BOOL,
        sender=int(src),
        receiver=int(party.party_id),
        seq_no=0,
        timeout_s=float(DEFAULT_NET_TIMEOUT_S),
    )

    recv_share: Dict[Tuple[int, int], bytes] = {}
    for sh in frame_in.segments:
        if int(sh.seg_kind) != SEG_OPEN_SHARE_LO:
            continue
        recv_share[(int(sh.object_id), int(sh.sub_id))] = frame_in.payload[int(sh.offset) : int(sh.offset) + int(sh.length)]

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for it in items_sorted:
        key = (int(it.open_id), int(it.sub_id))
        b = recv_share.get(key, None)
        if b is None:
            raise ValueError("missing OPEN_SHARE_LO segment for OPEN_BOOL operand")
        if len(b) != 4 * n_words32:
            raise ValueError("bad OPEN_BOOL segment length")
        w32 = torch.empty((n_words32,), dtype=torch.int64, device=it.x.lo_words.device)
        for wi in range(n_words32):
            w32[wi] = int.from_bytes(b[4 * wi : 4 * wi + 4], "little", signed=False)
        recv_prev64 = _u32words_to_u64words(w32, n_bits)
        out[key] = recv_prev64 ^ it.x.lo_words ^ it.x.hi_words

    # Record deterministic OPEN_BOOL_RESULT leaf committing to reconstructed public bytes (BITPACK_LE64).
    if party.transcript is not None:
        payload_parts: List[bytes] = []
        segs_result: List[SegmentDescV1] = []
        off0 = 0
        for it in items_sorted:
            key = (int(it.open_id), int(it.sub_id))
            pub_words64 = out[key].contiguous()
            pb = _u64_tensor_to_le_bytes(pub_words64)
            payload_parts.append(pb)
            segs_result.append(
                SegmentDescV1(
                    seg_kind=SEG_OPEN_RESULT_PUB,
                    object_id=int(it.open_id),
                    sub_id=int(it.sub_id),
                    dtype=DT_U64,
                    offset=int(off0),
                    length=len(pb),
                    fxp_frac_bits=0,
                )
            )
            off0 += len(pb)
        payload_all = b"".join(payload_parts)
        party.transcript.record_frame(
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=MSG_OPEN_BOOL_RESULT,
            sender=int(party.party_id),
            receiver=int(party.party_id),
            dir=0,
            seq_no=0,
            payload_bytes=len(payload_all),
            payload_hash32=sha256(payload_all),
            header_hash32=sha256(
                b"UVCC.openbool.result.v1\0"
                + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
                + struct.pack("<I", len(items_sorted) & 0xFFFFFFFF)
                + struct.pack("<I", int(n_bits) & 0xFFFFFFFF)
            ),
            segments=segs_result,
        )
    return out


