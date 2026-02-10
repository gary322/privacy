from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_df382033ede3f858

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


# NetFrame v1 constants (privacy_new.txt §C.1).
MAGIC_UVCC = b"UVCC"
VER_MAJOR = 1
VER_MINOR = 0

# Flags (privacy_new.txt §C.1.1).
FLAG_FRAG = 1 << 0
FLAG_COMPRESSED = 1 << 1  # MUST be 0 in v1
FLAG_GPU_BUFFER = 1 << 2

# SegmentKind (privacy_new.txt §C.2; v1 subset).
SEG_PAD = 1
SEG_OPEN_SHARE_LO = 10
SEG_OPEN_SHARE_META = 11

# SGIR_DType codes (privacy_new.txt §A.6.1) as u32 values.
DT_I1 = 1
DT_I8 = 2
DT_I16 = 3
DT_I32 = 4
DT_I64 = 5
DT_U8 = 6
DT_U16 = 7
DT_U32 = 8
DT_U64 = 9

# Alignment (privacy_new.txt §E).
ALIGN_BYTES_V1 = 128

# Canonical maximum payload size (privacy_new.txt §C.3.2).
MAX_PAYLOAD_BYTES_V1 = 64 * 1024 * 1024


# Packed structs (byte-exact, little-endian).
_FRAME_HDR = struct.Struct("<4sHHHH32sIIHHBBHIIIIIQQ32s")  # 128 bytes
_SEG_HDR = struct.Struct("<IIIIiIQQQ")  # 48 bytes


@dataclass(frozen=True)
class SegmentPayloadV1:
    seg_kind: int
    object_id: int
    sub_id: int
    dtype: int
    fxp_frac_bits: int
    payload: bytes


@dataclass(frozen=True)
class SegmentHeaderV1:
    seg_kind: int
    object_id: int
    sub_id: int
    dtype: int
    fxp_frac_bits: int
    offset: int
    length: int

    def to_bytes(self) -> bytes:
        return _SEG_HDR.pack(
            int(self.seg_kind) & 0xFFFFFFFF,
            int(self.object_id) & 0xFFFFFFFF,
            int(self.sub_id) & 0xFFFFFFFF,
            int(self.dtype) & 0xFFFFFFFF,
            int(self.fxp_frac_bits),
            0,
            int(self.offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.length) & 0xFFFFFFFFFFFFFFFF,
            0,
        )


@dataclass(frozen=True)
class FrameHeaderV1:
    msg_kind: int
    flags: int
    job_id32: bytes
    epoch: int
    step: int
    round: int
    sender: int
    receiver: int
    seq_no: int
    frame_no: int
    frame_count: int
    segment_count: int
    header_bytes: int
    payload_bytes: int
    header_hash32: bytes

    def __post_init__(self) -> None:
        if len(self.job_id32) != 32:
            raise ValueError("job_id32 must be 32 bytes")
        if len(self.header_hash32) != 32:
            raise ValueError("header_hash32 must be 32 bytes")

    def with_zero_hash(self) -> "FrameHeaderV1":
        return FrameHeaderV1(
            msg_kind=self.msg_kind,
            flags=self.flags,
            job_id32=self.job_id32,
            epoch=self.epoch,
            step=self.step,
            round=self.round,
            sender=self.sender,
            receiver=self.receiver,
            seq_no=self.seq_no,
            frame_no=self.frame_no,
            frame_count=self.frame_count,
            segment_count=self.segment_count,
            header_bytes=self.header_bytes,
            payload_bytes=self.payload_bytes,
            header_hash32=b"\x00" * 32,
        )

    def to_bytes(self) -> bytes:
        return _FRAME_HDR.pack(
            MAGIC_UVCC,
            int(VER_MAJOR) & 0xFFFF,
            int(VER_MINOR) & 0xFFFF,
            int(self.msg_kind) & 0xFFFF,
            int(self.flags) & 0xFFFF,
            self.job_id32,
            int(self.epoch) & 0xFFFFFFFF,
            int(self.step) & 0xFFFFFFFF,
            int(self.round) & 0xFFFF,
            0,
            int(self.sender) & 0xFF,
            int(self.receiver) & 0xFF,
            0,
            int(self.seq_no) & 0xFFFFFFFF,
            int(self.frame_no) & 0xFFFFFFFF,
            int(self.frame_count) & 0xFFFFFFFF,
            int(self.segment_count) & 0xFFFFFFFF,
            0,
            int(self.header_bytes) & 0xFFFFFFFFFFFFFFFF,
            int(self.payload_bytes) & 0xFFFFFFFFFFFFFFFF,
            self.header_hash32,
        )

    @staticmethod
    def from_bytes(buf: bytes) -> "FrameHeaderV1":
        if len(buf) < _FRAME_HDR.size:
            raise ValueError("buffer too small for FrameHeader")
        (
            magic,
            ver_major,
            ver_minor,
            msg_kind,
            flags,
            job_id32,
            epoch,
            step,
            round16,
            reserved0,
            sender,
            receiver,
            reserved1,
            seq_no,
            frame_no,
            frame_count,
            segment_count,
            reserved2,
            header_bytes,
            payload_bytes,
            header_hash32,
        ) = _FRAME_HDR.unpack_from(buf, 0)
        if magic != MAGIC_UVCC:
            raise ValueError("bad magic")
        if int(ver_major) != VER_MAJOR or int(ver_minor) != VER_MINOR:
            raise ValueError("bad version")
        if int(reserved0) != 0 or int(reserved1) != 0 or int(reserved2) != 0:
            raise ValueError("reserved fields must be 0")
        if int(frame_count) <= 0:
            raise ValueError("frame_count must be >= 1")
        if int(frame_no) >= int(frame_count):
            raise ValueError("frame_no out of range")
        return FrameHeaderV1(
            msg_kind=int(msg_kind),
            flags=int(flags),
            job_id32=bytes(job_id32),
            epoch=int(epoch),
            step=int(step),
            round=int(round16),
            sender=int(sender),
            receiver=int(receiver),
            seq_no=int(seq_no),
            frame_no=int(frame_no),
            frame_count=int(frame_count),
            segment_count=int(segment_count),
            header_bytes=int(header_bytes),
            payload_bytes=int(payload_bytes),
            header_hash32=bytes(header_hash32),
        )


@dataclass(frozen=True)
class NetFrameV1:
    header: FrameHeaderV1
    segments: Tuple[SegmentHeaderV1, ...]
    payload: bytes

    def to_bytes(self) -> bytes:
        hdr = self.header.to_bytes()
        seg_bytes = b"".join(s.to_bytes() for s in self.segments)
        return hdr + seg_bytes + self.payload


def _align_up(n: int, a: int) -> int:
    if a <= 0:
        raise ValueError("alignment must be > 0")
    r = n % a
    return n if r == 0 else (n + (a - r))


def build_netframe_v1(
    *,
    job_id32: bytes,
    epoch: int,
    step: int,
    round: int,
    msg_kind: int,
    flags: int,
    sender: int,
    receiver: int,
    seq_no: int,
    frame_no: int = 0,
    frame_count: int = 1,
    segments: Sequence[SegmentPayloadV1],
    max_payload_bytes: int = MAX_PAYLOAD_BYTES_V1,
) -> NetFrameV1:
    """
    Build a single NetFrame (no automatic fragmentation).

    Segments are sorted lexicographically by (seg_kind, object_id, sub_id) as required,
    and PAD segments are inserted deterministically to ensure 128B alignment for non-PAD
    segment starts.
    """
    if len(job_id32) != 32:
        raise ValueError("job_id32 must be 32 bytes")
    if int(frame_count) != 1 or int(frame_no) != 0:
        # v1 runtime currently does not emit multi-fragment frames; callers can implement
        # fragmentation by emitting multiple frames explicitly.
        raise ValueError("multi-fragment frames not supported in build_netframe_v1")

    # Sort non-PAD segments.
    segs_sorted = sorted(segments, key=lambda s: (int(s.seg_kind), int(s.object_id), int(s.sub_id)))

    headers: List[SegmentHeaderV1] = []
    payload_parts: List[bytes] = []
    cur_off = 0

    for seg in segs_sorted:
        if int(seg.seg_kind) == SEG_PAD:
            raise ValueError("caller must not supply SEG_PAD; it is inserted automatically")
        if cur_off % ALIGN_BYTES_V1 != 0:
            pad_len = (ALIGN_BYTES_V1 - (cur_off % ALIGN_BYTES_V1)) % ALIGN_BYTES_V1
            if pad_len:
                headers.append(
                    SegmentHeaderV1(
                        seg_kind=SEG_PAD,
                        object_id=0,
                        sub_id=0,
                        dtype=0,
                        fxp_frac_bits=0,
                        offset=cur_off,
                        length=pad_len,
                    )
                )
                payload_parts.append(b"\x00" * pad_len)
                cur_off += pad_len

        p = bytes(seg.payload)
        headers.append(
            SegmentHeaderV1(
                seg_kind=int(seg.seg_kind),
                object_id=int(seg.object_id),
                sub_id=int(seg.sub_id),
                dtype=int(seg.dtype),
                fxp_frac_bits=int(seg.fxp_frac_bits),
                offset=cur_off,
                length=len(p),
            )
        )
        payload_parts.append(p)
        cur_off += len(p)

    payload = b"".join(payload_parts)
    if len(payload) > int(max_payload_bytes):
        raise ValueError("payload exceeds max_payload_bytes (fragmentation required)")

    header_bytes = _FRAME_HDR.size + _SEG_HDR.size * len(headers)
    hdr0 = FrameHeaderV1(
        msg_kind=int(msg_kind),
        flags=int(flags),
        job_id32=bytes(job_id32),
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        sender=int(sender),
        receiver=int(receiver),
        seq_no=int(seq_no),
        frame_no=int(frame_no),
        frame_count=int(frame_count),
        segment_count=len(headers),
        header_bytes=header_bytes,
        payload_bytes=len(payload),
        header_hash32=b"\x00" * 32,
    )

    hdr_bytes_zero = hdr0.with_zero_hash().to_bytes()
    seg_bytes = b"".join(h.to_bytes() for h in headers)
    header_hash32 = sha256(hdr_bytes_zero + seg_bytes)
    hdr = FrameHeaderV1(
        msg_kind=int(msg_kind),
        flags=int(flags),
        job_id32=bytes(job_id32),
        epoch=int(epoch),
        step=int(step),
        round=int(round),
        sender=int(sender),
        receiver=int(receiver),
        seq_no=int(seq_no),
        frame_no=int(frame_no),
        frame_count=int(frame_count),
        segment_count=len(headers),
        header_bytes=header_bytes,
        payload_bytes=len(payload),
        header_hash32=header_hash32,
    )
    return NetFrameV1(header=hdr, segments=tuple(headers), payload=payload)


def parse_netframe_v1(buf: bytes) -> NetFrameV1:
    hdr = FrameHeaderV1.from_bytes(buf)
    if len(buf) < int(hdr.header_bytes) + int(hdr.payload_bytes):
        raise ValueError("truncated frame bytes")
    if int(hdr.header_bytes) != _FRAME_HDR.size + _SEG_HDR.size * int(hdr.segment_count):
        raise ValueError("header_bytes mismatch")
    segs: List[SegmentHeaderV1] = []
    off = _FRAME_HDR.size
    for _ in range(int(hdr.segment_count)):
        (
            seg_kind,
            object_id,
            sub_id,
            dtype,
            fxp_frac_bits,
            reserved0,
            offset_le,
            length_le,
            reserved1,
        ) = _SEG_HDR.unpack_from(buf, off)
        off += _SEG_HDR.size
        if int(reserved0) != 0 or int(reserved1) != 0:
            raise ValueError("segment reserved fields must be 0")
        segs.append(
            SegmentHeaderV1(
                seg_kind=int(seg_kind),
                object_id=int(object_id),
                sub_id=int(sub_id),
                dtype=int(dtype),
                fxp_frac_bits=int(fxp_frac_bits),
                offset=int(offset_le),
                length=int(length_le),
            )
        )
    payload = bytes(buf[int(hdr.header_bytes) : int(hdr.header_bytes) + int(hdr.payload_bytes)])

    # Verify header hash.
    hdr_zero = hdr.with_zero_hash().to_bytes()
    seg_bytes = b"".join(s.to_bytes() for s in segs)
    want = sha256(hdr_zero + seg_bytes)
    if want != hdr.header_hash32:
        raise ValueError("header_hash32 mismatch")

    # Basic payload bounds checks.
    for s in segs:
        if int(s.offset) < 0 or int(s.length) < 0:
            raise ValueError("negative segment offset/length")
        if int(s.offset) + int(s.length) > len(payload):
            raise ValueError("segment out of payload bounds")

    return NetFrameV1(header=hdr, segments=tuple(segs), payload=payload)


def payload_hash32_v1(frame: NetFrameV1) -> bytes:
    return sha256(frame.payload)


def relay_msg_id_v1(
    *,
    domain: bytes,
    sid_hash32: bytes,
    job_id32: bytes,
    epoch: int,
    step: int,
    round: int,
    msg_kind: int,
    sender: int,
    receiver: int,
    seq_no: int,
    frame_no: int = 0,
) -> str:
    """
    Deterministic relay message id (string) for a NetFrame.

    MUST be derivable by both sides without seeing payload bytes.
    """
    if len(sid_hash32) != 32:
        raise ValueError("sid_hash32 must be 32 bytes")
    if len(job_id32) != 32:
        raise ValueError("job_id32 must be 32 bytes")
    h = hashlib.sha256()
    h.update(domain)
    h.update(sid_hash32)
    h.update(job_id32)
    h.update(
        struct.pack(
            "<IIHHBBII",
            int(epoch) & 0xFFFFFFFF,
            int(step) & 0xFFFFFFFF,
            int(round) & 0xFFFF,
            int(msg_kind) & 0xFFFF,
            int(sender) & 0xFF,
            int(receiver) & 0xFF,
            int(seq_no) & 0xFFFFFFFF,
            int(frame_no) & 0xFFFFFFFF,
        )
    )
    return h.hexdigest()


