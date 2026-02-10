from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_0bad2ad63695a9fd,uvcc_group_91b08a9d0e68235e,uvcc_group_805dc0ce43c13243,uvcc_group_a215f6bfd7a10303,uvcc_group_35ac5da9edf180e2

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


DS_LEAF = b"UVCC.leaf.v1\0"
DS_NODE = b"UVCC.node.v1\0"
DS_EMPTY_EPOCH = b"UVCC.emptyepoch.v1\0"
DS_FINAL = b"UVCC.final.v1\0"
DS_SIG = b"UVCC.sig.v1\0"


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


@dataclass(frozen=True)
class SegmentDescV1:
    seg_kind: int
    object_id: int
    sub_id: int
    dtype: int
    offset: int
    length: int
    fxp_frac_bits: int

    def to_bytes(self) -> bytes:
        # struct SegmentDesc (40 bytes), `privacy_new.txt` §B.2.2
        return struct.pack(
            "<IIIIQQIi",
            int(self.seg_kind) & 0xFFFFFFFF,
            int(self.object_id) & 0xFFFFFFFF,
            int(self.sub_id) & 0xFFFFFFFF,
            int(self.dtype) & 0xFFFFFFFF,
            int(self.offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.length) & 0xFFFFFFFFFFFFFFFF,
            0,
            int(self.fxp_frac_bits),
        )


@dataclass(frozen=True)
class LeafBodyPrefixV1:
    job_id32: bytes
    epoch: int
    step: int
    round: int
    msg_kind: int
    sender: int
    receiver: int
    dir: int  # 0=SEND, 1=RECV
    seq_no: int
    payload_bytes: int
    payload_hash32: bytes
    header_hash32: bytes

    def __post_init__(self) -> None:
        if len(self.job_id32) != 32:
            raise ValueError("job_id32 must be 32 bytes")
        if len(self.payload_hash32) != 32:
            raise ValueError("payload_hash32 must be 32 bytes")
        if len(self.header_hash32) != 32:
            raise ValueError("header_hash32 must be 32 bytes")
        if int(self.dir) not in (0, 1):
            raise ValueError("dir must be 0 (SEND) or 1 (RECV)")

    def to_bytes(self) -> bytes:
        # LeafBodyPrefix (124 bytes), `privacy_new.txt` §B.2.1
        return struct.pack(
            "<32sIIHHBBBBIQ32s32s",
            self.job_id32,
            int(self.epoch) & 0xFFFFFFFF,
            int(self.step) & 0xFFFFFFFF,
            int(self.round) & 0xFFFF,
            int(self.msg_kind) & 0xFFFF,
            int(self.sender) & 0xFF,
            int(self.receiver) & 0xFF,
            int(self.dir) & 0xFF,
            0,
            int(self.seq_no) & 0xFFFFFFFF,
            int(self.payload_bytes) & 0xFFFFFFFFFFFFFFFF,
            self.payload_hash32,
            self.header_hash32,
        )


@dataclass(frozen=True)
class TranscriptLeafV1:
    prefix: LeafBodyPrefixV1
    segments: Tuple[SegmentDescV1, ...]
    leaf_hash32: bytes
    body_bytes: bytes

    @property
    def ordering_key(self) -> Tuple[int, int, int, int, int, int, int, int]:
        # `privacy_new.txt` §B.4 leaf ordering
        return (
            int(self.prefix.epoch),
            int(self.prefix.step),
            int(self.prefix.round),
            int(self.prefix.msg_kind),
            int(self.prefix.sender),
            int(self.prefix.receiver),
            int(self.prefix.seq_no),
            int(self.prefix.dir),
        )


def merkle_root_v1(leaf_hashes: List[bytes]) -> bytes:
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


class TranscriptStoreV1:
    """
    Minimal transcript store for UVCC v1 canonicalization.

    Stores leaves as (prefix + segment descriptors) and exposes epoch/final roots.
    """

    def __init__(self, *, job_id32: bytes, hash_alg: str = "sha256"):
        if len(job_id32) != 32:
            raise ValueError("job_id32 must be 32 bytes")
        if str(hash_alg).lower() != "sha256":
            raise ValueError("v1 profile requires sha256 transcript hash_alg")
        self.job_id32 = bytes(job_id32)
        self.hash_alg = "sha256"
        self._leaves: List[TranscriptLeafV1] = []
        self._by_key: Dict[Tuple[int, int, int, int, int, int, int, int], TranscriptLeafV1] = {}

    def record_frame(
        self,
        *,
        epoch: int,
        step: int,
        round: int,
        msg_kind: int,
        sender: int,
        receiver: int,
        dir: int,
        seq_no: int,
        payload_bytes: int,
        payload_hash32: bytes,
        header_hash32: bytes,
        segments: Iterable[SegmentDescV1],
    ) -> TranscriptLeafV1:
        segs = tuple(segments)
        prefix = LeafBodyPrefixV1(
            job_id32=self.job_id32,
            epoch=int(epoch),
            step=int(step),
            round=int(round),
            msg_kind=int(msg_kind),
            sender=int(sender),
            receiver=int(receiver),
            dir=int(dir),
            seq_no=int(seq_no),
            payload_bytes=int(payload_bytes),
            payload_hash32=bytes(payload_hash32),
            header_hash32=bytes(header_hash32),
        )

        body = bytearray(prefix.to_bytes())
        body += struct.pack("<I", len(segs) & 0xFFFFFFFF)
        for sd in segs:
            body += sd.to_bytes()
        body_bytes = bytes(body)
        leaf_hash32 = sha256(DS_LEAF + body_bytes)
        leaf = TranscriptLeafV1(prefix=prefix, segments=segs, leaf_hash32=leaf_hash32, body_bytes=body_bytes)
        k = leaf.ordering_key
        prev = self._by_key.get(k)
        if prev is not None:
            # Exactly-once acceptance: duplicates must match the first accepted leaf for this key.
            if prev.leaf_hash32 != leaf.leaf_hash32:
                raise ValueError("duplicate transcript leaf key with different hash")
            return prev
        self._by_key[k] = leaf
        self._leaves.append(leaf)
        return leaf

    def leaves(self) -> List[TranscriptLeafV1]:
        return list(self._leaves)

    def leaves_for_epoch(self, epoch: int) -> List[TranscriptLeafV1]:
        e = int(epoch)
        return [l for l in self._leaves if int(l.prefix.epoch) == e]

    def epoch_root(self, *, epoch: int) -> bytes:
        ls = self.leaves_for_epoch(epoch)
        ls_sorted = sorted(ls, key=lambda l: l.ordering_key)
        leaf_hashes = [l.leaf_hash32 for l in ls_sorted]
        return merkle_root_v1(leaf_hashes)

    def final_root(self, *, epoch_count: int) -> bytes:
        E = int(epoch_count)
        if E < 0:
            raise ValueError("epoch_count must be >=0")
        roots = [self.epoch_root(epoch=e) for e in range(E)]
        return sha256(DS_FINAL + struct.pack("<I", E & 0xFFFFFFFF) + b"".join(roots))


def signature_preimage_hash_v1(*, policy_hash32: bytes, final_root32: bytes, job_id32: bytes) -> bytes:
    if len(policy_hash32) != 32:
        raise ValueError("policy_hash32 must be 32 bytes")
    if len(final_root32) != 32:
        raise ValueError("final_root32 must be 32 bytes")
    if len(job_id32) != 32:
        raise ValueError("job_id32 must be 32 bytes")
    return sha256(DS_SIG + policy_hash32 + final_root32 + job_id32)


