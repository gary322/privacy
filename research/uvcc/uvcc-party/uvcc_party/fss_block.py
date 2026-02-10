from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


MAGIC_FSB1_LE = 0x31425346  # b"FSB1" little-endian
VERSION_FSB1 = 1
HDR_BYTES_FSB1 = 88
RECORD_BYTES_FSB1 = 52

DS_FSS_ID = b"UVCC.FSS.v1\0"
DS_FSSBLOCK = b"UVCC.FSSBLOCK.v1\0"


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def trunc64_le(h32: bytes) -> int:
    if len(h32) != 32:
        raise ValueError("expected 32-byte hash")
    return int.from_bytes(h32[:8], "little")


def fss_id_v1(*, job_nonce32: bytes, epoch: int, step: int, sgir_op_id: int, lane: int, fss_kind: int) -> int:
    """
    Deterministic fss_id derivation (privacy_new.txt §5.1).
    """
    if len(job_nonce32) != 32:
        raise ValueError("job_nonce32 must be 32 bytes")
    pre = (
        DS_FSS_ID
        + bytes(job_nonce32)
        + struct.pack("<II", int(epoch) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF)
        + struct.pack("<I", int(sgir_op_id) & 0xFFFFFFFF)
        + struct.pack("<H", int(lane) & 0xFFFF)
        + struct.pack("<B", int(fss_kind) & 0xFF)
    )
    return trunc64_le(sha256(pre))


def fssblock_hash32_v1(block_bytes: bytes) -> bytes:
    return sha256(DS_FSSBLOCK + bytes(block_bytes))


_HDR = struct.Struct("<IHHQIIIIQQQQQQQ")  # 88 bytes
_REC = struct.Struct("<QIHBBHHBBBBIQQQ")  # 52 bytes


@dataclass(frozen=True)
class FSSBlockHeaderV1:
    block_id: int
    epoch: int
    step: int
    record_count: int
    records_offset: int
    payload_offset: int
    payload_bytes: int


@dataclass(frozen=True)
class FSSRecordV1:
    fss_id: int
    sgir_op_id: int
    lane: int
    fss_kind: int
    domain_bits: int
    range_bits: int
    share_mode: int
    edge_mode: int
    edge_id: int
    flags: int
    key_bytes: int
    key_offset: int  # from payload_offset
    key_hash_hi: int
    key_hash_lo: int

    @property
    def ordering_key(self) -> Tuple[int, int, int, int]:
        return (int(self.fss_id), int(self.fss_kind), int(self.sgir_op_id), int(self.lane))


class FSSBlockV1:
    def __init__(self, buf: bytes):
        self._buf = bytes(buf)
        self.header, self.records, self.payload = parse_fss_block_v1(self._buf)

    @property
    def buf(self) -> bytes:
        return self._buf

    def find_record(self, *, fss_id: int) -> Optional[FSSRecordV1]:
        target = int(fss_id) & 0xFFFFFFFFFFFFFFFF
        lo = 0
        hi = len(self.records) - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            v = int(self.records[mid].fss_id)
            if v == target:
                return self.records[mid]
            if v < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def key_blob(self, *, fss_id: int) -> bytes:
        r = self.find_record(fss_id=int(fss_id))
        if r is None:
            raise KeyError("fss_id not found in block")
        off_abs = int(self.header.payload_offset) + int(r.key_offset)
        ln = int(r.key_bytes)
        if ln == 0:
            return b""
        if off_abs < 0 or ln < 0 or off_abs + ln > len(self._buf):
            raise ValueError("key slice out of bounds")
        return bytes(self._buf[off_abs : off_abs + ln])


def parse_fss_block_v1(buf: bytes) -> Tuple[FSSBlockHeaderV1, List[FSSRecordV1], bytes]:
    if len(buf) < HDR_BYTES_FSB1:
        raise ValueError("buffer too small for FSB1 header")
    (
        magic,
        ver,
        header_bytes,
        block_id,
        epoch,
        step,
        record_count,
        reserved0,
        records_offset,
        payload_offset,
        payload_bytes,
        records_hash_hi,
        records_hash_lo,
        payload_hash_hi,
        payload_hash_lo,
    ) = _HDR.unpack_from(buf, 0)
    if int(magic) != int(MAGIC_FSB1_LE):
        raise ValueError("bad magic")
    if int(ver) != int(VERSION_FSB1):
        raise ValueError("bad version")
    if int(header_bytes) != int(HDR_BYTES_FSB1):
        raise ValueError("bad header_bytes")
    if int(reserved0) != 0:
        raise ValueError("reserved0 must be 0")
    exp_block_id = ((int(epoch) & 0xFFFFFFFF) << 32) | (int(step) & 0xFFFFFFFF)
    if int(block_id) != int(exp_block_id):
        raise ValueError("block_id mismatch")
    ro = int(records_offset)
    po = int(payload_offset)
    pb = int(payload_bytes)
    if ro < HDR_BYTES_FSB1 or po < HDR_BYTES_FSB1 or ro >= len(buf) or po > len(buf):
        raise ValueError("bad records_offset/payload_offset")
    if ro >= po:
        raise ValueError("records_offset must be < payload_offset")
    if po + pb > len(buf):
        raise ValueError("payload out of bounds")

    n = int(record_count)
    rec_bytes = n * RECORD_BYTES_FSB1
    if ro + rec_bytes > len(buf):
        raise ValueError("record table out of bounds")

    records: List[FSSRecordV1] = []
    prev_key = None
    for i in range(n):
        off = ro + i * RECORD_BYTES_FSB1
        (
            fss_id,
            sgir_op_id,
            lane,
            fss_kind,
            rec_reserved0,
            domain_bits,
            range_bits,
            share_mode,
            edge_mode,
            edge_id,
            flags,
            key_bytes,
            key_offset,
            key_hash_hi,
            key_hash_lo,
        ) = _REC.unpack_from(buf, off)
        if int(rec_reserved0) != 0:
            raise ValueError("record reserved0 must be 0")
        r = FSSRecordV1(
            fss_id=int(fss_id),
            sgir_op_id=int(sgir_op_id) & 0xFFFFFFFF,
            lane=int(lane) & 0xFFFF,
            fss_kind=int(fss_kind) & 0xFF,
            domain_bits=int(domain_bits) & 0xFFFF,
            range_bits=int(range_bits) & 0xFFFF,
            share_mode=int(share_mode) & 0xFF,
            edge_mode=int(edge_mode) & 0xFF,
            edge_id=int(edge_id) & 0xFF,
            flags=int(flags) & 0xFF,
            key_bytes=int(key_bytes) & 0xFFFFFFFF,
            key_offset=int(key_offset) & 0xFFFFFFFFFFFFFFFF,
            key_hash_hi=int(key_hash_hi) & 0xFFFFFFFFFFFFFFFF,
            key_hash_lo=int(key_hash_lo) & 0xFFFFFFFFFFFFFFFF,
        )
        if prev_key is not None and r.ordering_key < prev_key:
            raise ValueError("records must be sorted by (fss_id,fss_kind,sgir_op_id,lane)")
        prev_key = r.ordering_key
        # Bounds check key blob within payload.
        if int(r.key_bytes) == 0:
            if int(r.key_offset) != 0 or int(r.key_hash_hi) != 0 or int(r.key_hash_lo) != 0:
                raise ValueError("zero key_bytes requires key_offset/hash==0")
        else:
            if int(r.key_offset) + int(r.key_bytes) > pb:
                raise ValueError("key blob out of payload bounds")
        records.append(r)

    payload = bytes(buf[po : po + pb])
    # Optional hashes (ignored unless non-zero in this runtime; verifier can enforce later).
    _ = (records_hash_hi, records_hash_lo, payload_hash_hi, payload_hash_lo)

    hdr = FSSBlockHeaderV1(
        block_id=int(block_id),
        epoch=int(epoch),
        step=int(step),
        record_count=n,
        records_offset=ro,
        payload_offset=po,
        payload_bytes=pb,
    )
    return hdr, records, payload


def build_fss_block_v1(
    *,
    epoch: int,
    step: int,
    records: List[FSSRecordV1],
    key_blobs: List[bytes],
) -> bytes:
    if len(records) != len(key_blobs):
        raise ValueError("records/key_blobs length mismatch")
    # Canonical record sort (privacy_new.txt §4.1).
    pairs = sorted(zip(records, key_blobs), key=lambda t: t[0].ordering_key)
    payload = bytearray()
    encoded_records: List[FSSRecordV1] = []
    for r0, kb in pairs:
        off = len(payload)
        kb_b = bytes(kb)
        payload += kb_b
        encoded_records.append(
            FSSRecordV1(
                fss_id=int(r0.fss_id) & 0xFFFFFFFFFFFFFFFF,
                sgir_op_id=int(r0.sgir_op_id) & 0xFFFFFFFF,
                lane=int(r0.lane) & 0xFFFF,
                fss_kind=int(r0.fss_kind) & 0xFF,
                domain_bits=int(r0.domain_bits) & 0xFFFF,
                range_bits=int(r0.range_bits) & 0xFFFF,
                share_mode=int(r0.share_mode) & 0xFF,
                edge_mode=int(r0.edge_mode) & 0xFF,
                edge_id=int(r0.edge_id) & 0xFF,
                flags=int(r0.flags) & 0xFF,
                key_bytes=len(kb_b),
                key_offset=int(off) & 0xFFFFFFFFFFFFFFFF,
                key_hash_hi=0,
                key_hash_lo=0,
            )
        )
    n = len(encoded_records)
    rec_table = bytearray()
    for r in encoded_records:
        rec_table += _REC.pack(
            int(r.fss_id) & 0xFFFFFFFFFFFFFFFF,
            int(r.sgir_op_id) & 0xFFFFFFFF,
            int(r.lane) & 0xFFFF,
            int(r.fss_kind) & 0xFF,
            0,
            int(r.domain_bits) & 0xFFFF,
            int(r.range_bits) & 0xFFFF,
            int(r.share_mode) & 0xFF,
            int(r.edge_mode) & 0xFF,
            int(r.edge_id) & 0xFF,
            int(r.flags) & 0xFF,
            int(r.key_bytes) & 0xFFFFFFFF,
            int(r.key_offset) & 0xFFFFFFFFFFFFFFFF,
            0,
            0,
        )
    records_offset = HDR_BYTES_FSB1
    payload_offset = records_offset + len(rec_table)
    block_id = ((int(epoch) & 0xFFFFFFFF) << 32) | (int(step) & 0xFFFFFFFF)
    hdr = _HDR.pack(
        MAGIC_FSB1_LE,
        VERSION_FSB1,
        HDR_BYTES_FSB1,
        int(block_id) & 0xFFFFFFFFFFFFFFFF,
        int(epoch) & 0xFFFFFFFF,
        int(step) & 0xFFFFFFFF,
        int(n) & 0xFFFFFFFF,
        0,
        int(records_offset) & 0xFFFFFFFFFFFFFFFF,
        int(payload_offset) & 0xFFFFFFFFFFFFFFFF,
        int(len(payload)) & 0xFFFFFFFFFFFFFFFF,
        0,
        0,
        0,
        0,
    )
    return bytes(hdr + rec_table + payload)


