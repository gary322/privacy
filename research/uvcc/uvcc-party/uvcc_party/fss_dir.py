from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


MAGIC_FSS_DIR_V1 = b"UVCCFSSD"
VERSION_FSS_DIR_V1 = 1

_HDR = struct.Struct("<8sHBB32sQIH6s")  # 64 bytes
_ENTRY = struct.Struct("<QBBBBIIII I")  # 32 bytes (note: trailing u32 stream_id32)


@dataclass(frozen=True)
class FSSDirHeaderV1:
    party_id: int
    sid_hash32: bytes
    epoch: int
    entry_count: int
    entry_stride: int

    def __post_init__(self) -> None:
        if int(self.party_id) not in (0, 1, 2):
            raise ValueError("party_id must be 0..2")
        if len(self.sid_hash32) != 32:
            raise ValueError("sid_hash32 must be 32 bytes")
        if int(self.entry_stride) != 32:
            raise ValueError("entry_stride must be 32 (v1)")


@dataclass(frozen=True)
class FSSDirEntryV1:
    fss_id: int
    prim_type: int  # 0x21=DPF, 0x22=DCF, 0x32=OP_LUT
    w: int
    out_kind: int
    prg_id: int
    keyrec_off: int
    keyrec_len: int
    aux_off: int
    aux_len: int
    stream_id32: int


class FSSDirectoryV1:
    """
    Minimal per-party unified FSS directory reader.

    Layout is the per-party local blob from `privacy_new.txt` §1.2:
      [header64][entry*count][raw keyrecs][raw aux blocks]
    """

    def __init__(self, buf: bytes):
        self._buf = bytes(buf)
        self.header, self.entries = parse_fss_dir_v1(self._buf)

    @property
    def buf(self) -> bytes:
        return self._buf

    def find_entry(self, *, fss_id: int) -> Optional[FSSDirEntryV1]:
        # Entries MUST be sorted by fss_id (binary search).
        lo = 0
        hi = len(self.entries) - 1
        target = int(fss_id) & 0xFFFFFFFFFFFFFFFF
        while lo <= hi:
            mid = (lo + hi) >> 1
            v = int(self.entries[mid].fss_id)
            if v == target:
                return self.entries[mid]
            if v < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def keyrec_bytes(self, *, fss_id: int) -> bytes:
        e = self.find_entry(fss_id=int(fss_id))
        if e is None:
            raise KeyError("fss_id not found")
        off = int(e.keyrec_off)
        ln = int(e.keyrec_len)
        if off < 0 or ln < 0 or off + ln > len(self._buf):
            raise ValueError("keyrec slice out of bounds")
        return bytes(self._buf[off : off + ln])

    def aux_bytes(self, *, fss_id: int) -> Optional[bytes]:
        e = self.find_entry(fss_id=int(fss_id))
        if e is None:
            raise KeyError("fss_id not found")
        if int(e.aux_len) == 0:
            return None
        off = int(e.aux_off)
        ln = int(e.aux_len)
        if off < 0 or ln < 0 or off + ln > len(self._buf):
            raise ValueError("aux slice out of bounds")
        return bytes(self._buf[off : off + ln])


def parse_fss_dir_v1(buf: bytes) -> Tuple[FSSDirHeaderV1, List[FSSDirEntryV1]]:
    if len(buf) < 64:
        raise ValueError("buffer too small for fss dir header")
    magic, ver, party_id, flags, sid_hash32, epoch, entry_count, entry_stride, reserved = _HDR.unpack_from(buf, 0)
    if magic != MAGIC_FSS_DIR_V1:
        raise ValueError("bad magic")
    if int(ver) != VERSION_FSS_DIR_V1:
        raise ValueError("bad version")
    if int(flags) != 0:
        raise ValueError("flags must be 0 in v1")
    if bytes(reserved) != b"\x00" * 6:
        raise ValueError("reserved must be zero")
    if int(entry_stride) != 32:
        raise ValueError("entry_stride must be 32")
    n = int(entry_count)
    if n < 0:
        raise ValueError("bad entry_count")
    entries_off = 64
    need = entries_off + n * 32
    if len(buf) < need:
        raise ValueError("truncated entries")
    entries: List[FSSDirEntryV1] = []
    prev_id = None
    for i in range(n):
        off = entries_off + i * 32
        (
            fss_id,
            prim_type,
            w,
            out_kind,
            prg_id,
            keyrec_off,
            keyrec_len,
            aux_off,
            aux_len,
            stream_id32,
        ) = struct.unpack_from("<QBBBBIIII I", buf, off)
        if prev_id is not None and int(fss_id) < int(prev_id):
            raise ValueError("entries must be sorted by fss_id ascending")
        prev_id = int(fss_id)
        entries.append(
            FSSDirEntryV1(
                fss_id=int(fss_id),
                prim_type=int(prim_type) & 0xFF,
                w=int(w) & 0xFF,
                out_kind=int(out_kind) & 0xFF,
                prg_id=int(prg_id) & 0xFF,
                keyrec_off=int(keyrec_off) & 0xFFFFFFFF,
                keyrec_len=int(keyrec_len) & 0xFFFFFFFF,
                aux_off=int(aux_off) & 0xFFFFFFFF,
                aux_len=int(aux_len) & 0xFFFFFFFF,
                stream_id32=int(stream_id32) & 0xFFFFFFFF,
            )
        )

    hdr = FSSDirHeaderV1(
        party_id=int(party_id),
        sid_hash32=bytes(sid_hash32),
        epoch=int(epoch),
        entry_count=n,
        entry_stride=int(entry_stride),
    )
    return hdr, entries


def build_fss_dir_v1(
    *,
    party_id: int,
    sid_hash32: bytes,
    epoch: int,
    entries: List[Tuple[FSSDirEntryV1, bytes, Optional[bytes]]],
) -> bytes:
    """
    Build a per-party directory blob.

    Args:
      entries: list of (entry_template, keyrec_bytes, aux_bytes_or_None).
              Offsets/lengths in entry_template are ignored and filled deterministically.
    """
    if len(sid_hash32) != 32:
        raise ValueError("sid_hash32 must be 32 bytes")
    # Sort by fss_id (canonical).
    ent_sorted = sorted(entries, key=lambda t: int(t[0].fss_id))
    n = len(ent_sorted)
    header = _HDR.pack(
        MAGIC_FSS_DIR_V1,
        VERSION_FSS_DIR_V1,
        int(party_id) & 0xFF,
        0,
        bytes(sid_hash32),
        int(epoch) & 0xFFFFFFFFFFFFFFFF,
        int(n) & 0xFFFFFFFF,
        32,
        b"\x00" * 6,
    )
    out = bytearray(header)
    # Reserve entries region.
    out += b"\x00" * (n * 32)
    cursor = 64 + n * 32

    encoded_entries = []
    for e, keyrec, aux in ent_sorted:
        keyrec_b = bytes(keyrec)
        key_off = cursor
        key_len = len(keyrec_b)
        out += keyrec_b
        cursor += key_len
        aux_off = 0
        aux_len = 0
        if aux is not None:
            aux_b = bytes(aux)
            aux_off = cursor
            aux_len = len(aux_b)
            out += aux_b
            cursor += aux_len
        encoded_entries.append((e, key_off, key_len, aux_off, aux_len))

    # Fill entry table.
    for i, (e, key_off, key_len, aux_off, aux_len) in enumerate(encoded_entries):
        ent_bytes = struct.pack(
            "<QBBBBIIII I",
            int(e.fss_id) & 0xFFFFFFFFFFFFFFFF,
            int(e.prim_type) & 0xFF,
            int(e.w) & 0xFF,
            int(e.out_kind) & 0xFF,
            int(e.prg_id) & 0xFF,
            int(key_off) & 0xFFFFFFFF,
            int(key_len) & 0xFFFFFFFF,
            int(aux_off) & 0xFFFFFFFF,
            int(aux_len) & 0xFFFFFFFF,
            int(e.stream_id32) & 0xFFFFFFFF,
        )
        off = 64 + i * 32
        out[off : off + 32] = ent_bytes

    return bytes(out)


