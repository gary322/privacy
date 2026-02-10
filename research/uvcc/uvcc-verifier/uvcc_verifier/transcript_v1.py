from __future__ import annotations

import base64
import dataclasses
import json
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DS_LEAF = b"UVCC.leaf.v1\0"
DS_NODE = b"UVCC.node.v1\0"
DS_EMPTY_EPOCH = b"UVCC.emptyepoch.v1\0"
DS_FINAL = b"UVCC.final.v1\0"


def sha256(b: bytes) -> bytes:
    import hashlib

    return hashlib.sha256(b).digest()


_LEAF_PREFIX = struct.Struct("<32sIIHHBBBBIQ32s32s")  # 124 bytes (matches uvcc_party.transcript.LeafBodyPrefixV1)
_SEG_DESC = struct.Struct("<IIIIQQIi")  # 40 bytes (matches uvcc_party.transcript.SegmentDescV1)


@dataclass(frozen=True)
class SegmentDescV1:
    seg_kind: int
    object_id: int
    sub_id: int
    dtype: int
    offset: int
    length: int
    fxp_frac_bits: int


@dataclass(frozen=True)
class LeafBodyPrefixV1:
    job_id32: bytes
    epoch: int
    step: int
    round: int
    msg_kind: int
    sender: int
    receiver: int
    dir: int
    seq_no: int
    payload_bytes: int
    payload_hash32: bytes
    header_hash32: bytes

    @property
    def ordering_key(self) -> Tuple[int, int, int, int, int, int, int, int]:
        # Matches uvcc_party.transcript.TranscriptLeafV1.ordering_key
        return (
            int(self.epoch),
            int(self.step),
            int(self.round),
            int(self.msg_kind),
            int(self.sender),
            int(self.receiver),
            int(self.seq_no),
            int(self.dir),
        )


@dataclass(frozen=True)
class TranscriptLeafParsedV1:
    prefix: LeafBodyPrefixV1
    segments: Tuple[SegmentDescV1, ...]
    body_bytes: bytes
    leaf_hash32: bytes


def parse_transcript_leaf_body_v1(body_bytes: bytes) -> TranscriptLeafParsedV1:
    if len(body_bytes) < _LEAF_PREFIX.size + 4:
        raise ValueError("leaf body too small")
    (
        job_id32,
        epoch,
        step,
        round_u16,
        msg_kind_u16,
        sender_u8,
        receiver_u8,
        dir_u8,
        reserved0_u8,
        seq_no_u32,
        payload_bytes_u64,
        payload_hash32,
        header_hash32,
    ) = _LEAF_PREFIX.unpack_from(body_bytes, 0)
    if int(reserved0_u8) != 0:
        raise ValueError("leaf reserved0 must be 0")
    if int(dir_u8) not in (0, 1):
        raise ValueError("leaf dir must be 0/1")
    if len(payload_hash32) != 32 or len(header_hash32) != 32:
        raise ValueError("bad hash32 length")

    seg_count = int.from_bytes(body_bytes[_LEAF_PREFIX.size : _LEAF_PREFIX.size + 4], "little", signed=False)
    off = _LEAF_PREFIX.size + 4
    need = off + seg_count * _SEG_DESC.size
    if len(body_bytes) != need:
        raise ValueError("leaf body size mismatch with seg_count")

    segs: List[SegmentDescV1] = []
    for _ in range(seg_count):
        (
            seg_kind,
            object_id,
            sub_id,
            dtype,
            offset_u64,
            length_u64,
            reserved_seg_u32,
            fxp_frac_bits_i32,
        ) = _SEG_DESC.unpack_from(body_bytes, off)
        off += _SEG_DESC.size
        if int(reserved_seg_u32) != 0:
            raise ValueError("segment reserved must be 0")
        segs.append(
            SegmentDescV1(
                seg_kind=int(seg_kind),
                object_id=int(object_id),
                sub_id=int(sub_id),
                dtype=int(dtype),
                offset=int(offset_u64),
                length=int(length_u64),
                fxp_frac_bits=int(fxp_frac_bits_i32),
            )
        )

    # Basic bounds: segments are offsets into the payload region.
    payload_bytes = int(payload_bytes_u64)
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be >=0")
    for s in segs:
        if int(s.offset) < 0 or int(s.length) < 0:
            raise ValueError("negative segment offset/length")
        if int(s.offset) + int(s.length) > payload_bytes:
            raise ValueError("segment out of payload bounds")

    prefix = LeafBodyPrefixV1(
        job_id32=bytes(job_id32),
        epoch=int(epoch),
        step=int(step),
        round=int(round_u16),
        msg_kind=int(msg_kind_u16),
        sender=int(sender_u8),
        receiver=int(receiver_u8),
        dir=int(dir_u8),
        seq_no=int(seq_no_u32),
        payload_bytes=int(payload_bytes_u64),
        payload_hash32=bytes(payload_hash32),
        header_hash32=bytes(header_hash32),
    )
    leaf_hash32 = sha256(DS_LEAF + bytes(body_bytes))
    return TranscriptLeafParsedV1(prefix=prefix, segments=tuple(segs), body_bytes=bytes(body_bytes), leaf_hash32=leaf_hash32)


def merkle_root_v1(leaf_hashes: Sequence[bytes]) -> bytes:
    if not leaf_hashes:
        return sha256(DS_EMPTY_EPOCH)
    level = list(leaf_hashes)
    if len(level) == 1:
        h0 = level[0]
        return sha256(DS_NODE + h0 + h0)
    while len(level) > 1:
        nxt: List[bytes] = []
        i = 0
        while i < len(level):
            left = level[i]
            right = level[i + 1] if (i + 1) < len(level) else left
            nxt.append(sha256(DS_NODE + left + right))
            i += 2
        level = nxt
    return level[0]


def compute_epoch_roots_v1(leaves: Sequence[TranscriptLeafParsedV1]) -> Dict[int, bytes]:
    by_epoch: Dict[int, List[TranscriptLeafParsedV1]] = {}
    for leaf in leaves:
        by_epoch.setdefault(int(leaf.prefix.epoch), []).append(leaf)
    out: Dict[int, bytes] = {}
    for e, ls in by_epoch.items():
        ls_sorted = sorted(ls, key=lambda l: l.prefix.ordering_key)
        out[int(e)] = merkle_root_v1([l.leaf_hash32 for l in ls_sorted])
    return out


def compute_final_root_v1(*, epoch_roots: Sequence[bytes]) -> bytes:
    E = int(len(epoch_roots))
    return sha256(DS_FINAL + struct.pack("<I", E & 0xFFFFFFFF) + b"".join(epoch_roots))


def parse_transcript_jsonl_v1(path: str) -> List[TranscriptLeafParsedV1]:
    leaves: List[TranscriptLeafParsedV1] = []
    with open(path, "rb") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            if not isinstance(obj, dict):
                raise ValueError(f"bad jsonl line {line_no}: expected object")
            body_b64 = obj.get("body_b64", None)
            if not isinstance(body_b64, str):
                raise ValueError(f"bad jsonl line {line_no}: missing body_b64")
            body = base64.b64decode(body_b64.encode("ascii"))
            leaf = parse_transcript_leaf_body_v1(body)

            want_hex = obj.get("leaf_hash_hex", None)
            if want_hex is not None:
                if not isinstance(want_hex, str) or not want_hex.startswith("0x") or len(want_hex) != 66:
                    raise ValueError(f"bad jsonl line {line_no}: leaf_hash_hex must be 0x + 64 hex")
                if bytes.fromhex(want_hex[2:]) != leaf.leaf_hash32:
                    raise ValueError(f"leaf_hash_hex mismatch on line {line_no}")
            leaves.append(leaf)
    return leaves


def is_netframe_leaf_v1(leaf: TranscriptLeafParsedV1) -> bool:
    """
    Best-effort classifier: checks if header_hash32 matches a NetFrame v1 header hash
    reconstructed from the transcript leaf fields under v1 assumptions.
    """
    # Frame header struct (128B) and segment header struct (48B) per uvcc_party.netframe.
    frame_hdr = struct.Struct("<4sHHHH32sIIHHBBHIIIIIQQ32s")
    seg_hdr = struct.Struct("<IIIIiIQQQ")

    p = leaf.prefix
    segs = leaf.segments

    header_bytes = frame_hdr.size + seg_hdr.size * len(segs)

    # Rebuild segment headers in the same order as recorded.
    seg_bytes = bytearray()
    for s in segs:
        seg_bytes += seg_hdr.pack(
            int(s.seg_kind) & 0xFFFFFFFF,
            int(s.object_id) & 0xFFFFFFFF,
            int(s.sub_id) & 0xFFFFFFFF,
            int(s.dtype) & 0xFFFFFFFF,
            int(s.fxp_frac_bits),
            0,
            int(s.offset) & 0xFFFFFFFFFFFFFFFF,
            int(s.length) & 0xFFFFFFFFFFFFFFFF,
            0,
        )

    hdr_zero = frame_hdr.pack(
        b"UVCC",
        1,
        0,
        int(p.msg_kind) & 0xFFFF,
        0,  # flags (v1 runtime uses 0)
        bytes(p.job_id32),
        int(p.epoch) & 0xFFFFFFFF,
        int(p.step) & 0xFFFFFFFF,
        int(p.round) & 0xFFFF,
        0,
        int(p.sender) & 0xFF,
        int(p.receiver) & 0xFF,
        0,
        int(p.seq_no) & 0xFFFFFFFF,
        0,  # frame_no
        1,  # frame_count
        int(len(segs)) & 0xFFFFFFFF,
        0,
        int(header_bytes) & 0xFFFFFFFFFFFFFFFF,
        int(p.payload_bytes) & 0xFFFFFFFFFFFFFFFF,
        b"\x00" * 32,
    )
    want = sha256(bytes(hdr_zero) + bytes(seg_bytes))
    return want == p.header_hash32


def _hdrhash_sks_local_leaf_v1(*, epoch: int, step: int, round: int, msg_kind: int) -> bytes:
    # Matches uvcc_party.sks._record_sks_leaf_v1 header_hash32 construction.
    return sha256(
        b"UVCC.sks.leaf.v1\0"
        + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
        + struct.pack("<H", int(msg_kind) & 0xFFFF)
    )


def _hdrhash_openarith_result_v1(*, epoch: int, step: int, round: int, item_count: int) -> bytes:
    # Matches uvcc_party.open.open_arith_u64_round_v1 header_hash32.
    return sha256(
        b"UVCC.openarith.result.v1\0"
        + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
        + struct.pack("<I", int(item_count) & 0xFFFFFFFF)
    )


def _hdrhash_fssblock_commit_v1(*, epoch: int, step: int, round: int) -> bytes:
    return sha256(b"UVCC.fssblock.commit.hdr.v1\0" + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF))


def _hdrhash_trunc_leaf_v1(*, ds: bytes, epoch: int, step: int, round: int, open_id_u64: int, sgir_op_id_u32: int, f_bits_u32: int) -> bytes:
    return sha256(
        bytes(ds)
        + struct.pack("<IIH", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF, int(round) & 0xFFFF)
        + struct.pack("<QII", int(open_id_u64) & 0xFFFFFFFFFFFFFFFF, int(sgir_op_id_u32) & 0xFFFFFFFF, int(f_bits_u32) & 0xFFFFFFFF)
    )


def validate_transcript_leaves_v1(
    leaves: Sequence[TranscriptLeafParsedV1],
    *,
    strict_unknown_msg_kind: bool = False,
    strict_netframe_header_hash: bool = True,
) -> None:
    """
    Deterministic structural validation for the v1 transcript leaf model used by uvcc_party.

    This validator is intentionally strict about:
    - job_id consistency
    - leaf hash correctness (already enforced by parsing)
    - payload bounds vs segment offsets
    - NetFrame SEND/RECV leaves having a reconstructible NetFrame header hash (when strict_netframe_header_hash=True)

    And best-effort about:
    - local-only leaf header hashing (when reconstructible from leaf bytes)
    """
    if not leaves:
        return
    job_id32 = leaves[0].prefix.job_id32
    for lf in leaves:
        if lf.prefix.job_id32 != job_id32:
            raise ValueError("mixed job_id32 in transcript")

    # Known msg_kind sets in the current v1 runtime.
    MSG_OPEN_BOOL = 0x0101
    MSG_OPEN_ARITH = 0x0102
    MSG_OPEN_ARITH_RESULT = 0x0202
    MSG_OPEN_BOOL_RESULT = 0x0203
    MSG_CMP_REPL = 0x0201
    MSG_OPLUT_REPL = 0x0310
    MSG_TRUNC_REPL = 0x0320
    MSG_TCF_REPL = 200

    # Local-only commit types used by the runtime.
    MSG_FSSBLOCK_COMMIT = 0x0001
    MSG_FSSEVAL_COMMIT = 0x0002
    MSG_OPLUT_COMMIT = 0x0003

    # TRUNC leaf types (privacy_new.txt §6.3).
    MSG_TRUNC_OPEN_ARITH_SEND = 0x6001
    MSG_TRUNC_OPEN_ARITH_RESULT = 0x6002
    MSG_TRUNC_CARRY_RESULT = 0x6003
    MSG_TRUNC_OUTPUT_COMMIT = 0x6004

    # SKS leaf types.
    SKS_LOCAL_OR_WIRE = {0x70, 0x71, 0x73}
    SKS_LOCAL_ONLY = {0x72, 0x74}

    local_only = {
        MSG_OPEN_ARITH_RESULT,
        MSG_OPEN_BOOL_RESULT,
        MSG_FSSBLOCK_COMMIT,
        MSG_FSSEVAL_COMMIT,
        MSG_OPLUT_COMMIT,
        MSG_TRUNC_OPEN_ARITH_SEND,
        MSG_TRUNC_OPEN_ARITH_RESULT,
        MSG_TRUNC_CARRY_RESULT,
        MSG_TRUNC_OUTPUT_COMMIT,
        *SKS_LOCAL_ONLY,
    }
    wire_only = {MSG_OPEN_BOOL, MSG_OPEN_ARITH, MSG_CMP_REPL, MSG_OPLUT_REPL, MSG_TRUNC_REPL, MSG_TCF_REPL}

    for lf in leaves:
        mk = int(lf.prefix.msg_kind)
        if mk in wire_only:
            if strict_netframe_header_hash and not is_netframe_leaf_v1(lf):
                raise ValueError(f"msg_kind=0x{mk:04x} expected NetFrame header hash match")
            continue

        # SKS types can be either local leaf or on-wire NetFrame.
        if mk in SKS_LOCAL_OR_WIRE:
            if is_netframe_leaf_v1(lf):
                continue
            want = _hdrhash_sks_local_leaf_v1(epoch=lf.prefix.epoch, step=lf.prefix.step, round=lf.prefix.round, msg_kind=mk)
            if lf.prefix.header_hash32 != want:
                raise ValueError(f"SKS local leaf header_hash32 mismatch for msg_kind=0x{mk:02x}")
            continue

        if mk == MSG_OPEN_ARITH_RESULT:
            # Can recompute header hash from leaf bytes.
            item_count = sum(1 for s in lf.segments if int(s.seg_kind) == 12)
            want = _hdrhash_openarith_result_v1(epoch=lf.prefix.epoch, step=lf.prefix.step, round=lf.prefix.round, item_count=item_count)
            if lf.prefix.header_hash32 != want:
                raise ValueError("OPEN_ARITH_RESULT header_hash32 mismatch")
            continue

        if mk == MSG_FSSBLOCK_COMMIT:
            want = _hdrhash_fssblock_commit_v1(epoch=lf.prefix.epoch, step=lf.prefix.step, round=lf.prefix.round)
            if lf.prefix.header_hash32 != want:
                raise ValueError("FSSBLOCK_COMMIT header_hash32 mismatch")
            continue

        if mk in (MSG_TRUNC_OPEN_ARITH_SEND, MSG_TRUNC_OPEN_ARITH_RESULT, MSG_TRUNC_OUTPUT_COMMIT):
            if not lf.segments:
                raise ValueError("TRUNC leaf must have segments")
            open_id_u64 = int(lf.segments[0].object_id) & 0xFFFFFFFFFFFFFFFF
            sgir_op_id_u32 = int(lf.segments[0].object_id) & 0xFFFFFFFF
            f_bits_u32 = int(lf.segments[0].fxp_frac_bits) & 0xFFFFFFFF
            if mk == MSG_TRUNC_OPEN_ARITH_SEND:
                ds = b"UVCC.trunc.openarith.send.v1\0"
            elif mk == MSG_TRUNC_OPEN_ARITH_RESULT:
                ds = b"UVCC.trunc.openarith.result.v1\0"
            else:
                ds = b"UVCC.trunc.output.commit.v1\0"
            want = _hdrhash_trunc_leaf_v1(
                ds=ds,
                epoch=lf.prefix.epoch,
                step=lf.prefix.step,
                round=lf.prefix.round,
                open_id_u64=open_id_u64,
                sgir_op_id_u32=sgir_op_id_u32,
                f_bits_u32=f_bits_u32,
            )
            if lf.prefix.header_hash32 != want:
                raise ValueError(f"TRUNC leaf header_hash32 mismatch for msg_kind=0x{mk:04x}")
            continue

        # For other local-only leaves, we validate only that they are NOT misclassified as wire-only.
        if mk in local_only:
            continue

        if strict_unknown_msg_kind:
            raise ValueError(f"unknown msg_kind=0x{mk:04x}")


