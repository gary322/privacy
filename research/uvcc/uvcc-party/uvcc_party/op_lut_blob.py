from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_30225604e2990a9f,uvcc_group_e6b8d87d4097bede,uvcc_group_48a6f9c1656f1342,uvcc_group_509a5eba42fcc1ce,uvcc_group_b69668db9263f95f

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .dpf_aes128 import keygen_dpf_aes128_ar64_v1
from .dpf_dcf import chacha20_block_bytes_v1


MAGIC_FSSREC_V1 = b"UVFSSv1\0"
FSSREC_VERSION_V1 = 1
FSS_KIND_OP_LUT = 0x00000020

SEC_META = 0x00000001
SEC_MASK_RSS = 0x00000002
SEC_DPF_2PC = 0x00000003
SEC_REFRESH = 0x00000004

P0 = 0
P1 = 1
P2 = 2

DPF_NONE = 0
DPF_LEFT = 1
DPF_RIGHT = 2

RING_Z2_64 = 1
PRG_CHACHA20 = 1

EDGE_01 = 1
EDGE_12 = 2
EDGE_20 = 3

DPFALG_GGM_AES128 = 1


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _require_len(b: bytes, n: int, name: str) -> None:
    if not isinstance(b, (bytes, bytearray)) or len(b) != n:
        raise ValueError(f"{name} must be {n} bytes")


def _u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _trunc16(h32: bytes) -> bytes:
    return bytes(h32[:16])


def _trunc12(h32: bytes) -> bytes:
    return bytes(h32[:12])


def oplut_salt16_v1(*, sid: bytes, fss_id: int, sgir_op_id: int) -> bytes:
    return _trunc16(sha256(b"OPLUTv1" + bytes(sid) + struct.pack("<QI", int(fss_id) & 0xFFFFFFFFFFFFFFFF, int(sgir_op_id) & 0xFFFFFFFF)))


def oplut_nonce_r12_v1(*, salt16: bytes) -> bytes:
    _require_len(salt16, 16, "salt16")
    return _trunc12(sha256(b"OPLUTv1.R" + bytes(salt16)))


def oplut_nonce_m12_v1(*, salt16: bytes) -> bytes:
    _require_len(salt16, 16, "salt16")
    return _trunc12(sha256(b"OPLUTv1.M" + bytes(salt16)))


_HDR = struct.Struct("<8sIIQIIQ")  # 40
_DIRENT = struct.Struct("<IIQQ")  # 24
_META = struct.Struct("<BBBBIIB3s16sQHH")  # 44
_MASK = struct.Struct("<BB2s32s32s12sQQ")  # 96
_DPF2PC = struct.Struct("<BBBBHHI")  # 12
_REFRESH = struct.Struct("<BB2s12sQQ")  # 32


@dataclass(frozen=True)
class FSSRecordHeaderV1:
    kind: int
    fss_id: int
    header_bytes: int
    section_count: int
    payload_bytes: int


@dataclass(frozen=True)
class FSSDirEntV1:
    type: int
    flags: int
    offset: int
    bytes: int


@dataclass(frozen=True)
class OPLUTMetaV1:
    party_id: int
    dpf_role: int
    domain_w: int
    lanes: int
    ring_id: int
    prg_id: int
    salt16: bytes
    lane_base: int
    dpf_key_bytes_per_lane: int
    flags: int


@dataclass(frozen=True)
class OPLUTMaskRSSV1:
    edge_a: int
    edge_b: int
    seed_a32: bytes
    seed_b32: bytes
    nonce_r12: bytes
    counter0: int
    lane_stride: int


@dataclass(frozen=True)
class OPLUTDPF2PCV1:
    dpf_alg: int
    domain_w: int
    out_mode: int
    key_bytes_per_lane: int
    lanes: int
    keys: bytes  # lanes*key_bytes_per_lane

    def lane_key(self, lane_index: int) -> bytes:
        i = int(lane_index)
        if i < 0 or i >= int(self.lanes):
            raise IndexError("lane_index out of range")
        off = i * int(self.key_bytes_per_lane)
        return bytes(self.keys[off : off + int(self.key_bytes_per_lane)])


@dataclass(frozen=True)
class OPLUTRefreshV1:
    uses_edge20: int
    uses_edge12: int
    nonce_m12: bytes
    counter0: int
    lane_stride: int


@dataclass(frozen=True)
class OPLUTRecordV1:
    header: FSSRecordHeaderV1
    dirents: Tuple[FSSDirEntV1, ...]
    meta: OPLUTMetaV1
    mask: OPLUTMaskRSSV1
    refresh: OPLUTRefreshV1
    dpf2pc: Optional[OPLUTDPF2PCV1]


def parse_oplut_record_v1(buf: bytes) -> OPLUTRecordV1:
    if len(buf) < _HDR.size:
        raise ValueError("buffer too small for FSSRecordHeader_v1")
    magic, ver, kind, fss_id, header_bytes, section_count, payload_bytes = _HDR.unpack_from(buf, 0)
    if magic != MAGIC_FSSREC_V1:
        raise ValueError("bad magic")
    if int(ver) != FSSREC_VERSION_V1:
        raise ValueError("bad version")
    if int(kind) != FSS_KIND_OP_LUT:
        raise ValueError("not an OP_LUT record")
    hb = int(header_bytes)
    sc = int(section_count)
    if hb < _HDR.size + sc * _DIRENT.size:
        raise ValueError("header_bytes too small")
    if len(buf) != hb + int(payload_bytes):
        raise ValueError("payload_bytes mismatch")
    # Parse dirents
    dirents = []
    off = _HDR.size
    for _ in range(sc):
        t, fl, o, n = _DIRENT.unpack_from(buf, off)
        dirents.append(FSSDirEntV1(type=int(t), flags=int(fl), offset=int(o), bytes=int(n)))
        off += _DIRENT.size
    payload0 = hb

    sec_map: Dict[int, bytes] = {}
    for de in dirents:
        so = payload0 + int(de.offset)
        sn = int(de.bytes)
        if so < payload0 or so + sn > len(buf):
            raise ValueError("section out of bounds")
        sec_map[int(de.type)] = bytes(buf[so : so + sn])

    if SEC_META not in sec_map or SEC_MASK_RSS not in sec_map or SEC_REFRESH not in sec_map:
        raise ValueError("missing required sections")

    # META
    (
        party_id,
        dpf_role,
        domain_w,
        reserved0,
        lanes,
        ring_id,
        prg_id,
        reserved1,
        salt16,
        lane_base,
        dpf_key_bytes_per_lane,
        meta_flags,
    ) = _META.unpack_from(sec_map[SEC_META], 0)
    if int(reserved0) != 0 or reserved1 != b"\x00" * 3:
        raise ValueError("META reserved fields must be 0")
    meta = OPLUTMetaV1(
        party_id=int(party_id),
        dpf_role=int(dpf_role),
        domain_w=int(domain_w),
        lanes=int(lanes),
        ring_id=int(ring_id),
        prg_id=int(prg_id),
        salt16=bytes(salt16),
        lane_base=int(lane_base),
        dpf_key_bytes_per_lane=int(dpf_key_bytes_per_lane),
        flags=int(meta_flags),
    )

    # MASK
    edge_a, edge_b, _r0, seed_a, seed_b, nonce_r, counter0, lane_stride = _MASK.unpack_from(sec_map[SEC_MASK_RSS], 0)
    mask = OPLUTMaskRSSV1(
        edge_a=int(edge_a),
        edge_b=int(edge_b),
        seed_a32=bytes(seed_a),
        seed_b32=bytes(seed_b),
        nonce_r12=bytes(nonce_r),
        counter0=int(counter0),
        lane_stride=int(lane_stride),
    )

    # REFRESH
    uses_edge20, uses_edge12, _r1, nonce_m, counter0m, lane_stridem = _REFRESH.unpack_from(sec_map[SEC_REFRESH], 0)
    refresh = OPLUTRefreshV1(
        uses_edge20=int(uses_edge20),
        uses_edge12=int(uses_edge12),
        nonce_m12=bytes(nonce_m),
        counter0=int(counter0m),
        lane_stride=int(lane_stridem),
    )

    dpf2 = None
    if SEC_DPF_2PC in sec_map:
        head = sec_map[SEC_DPF_2PC]
        if len(head) < _DPF2PC.size:
            raise ValueError("DPF_2PC section too small")
        dpf_alg, dw, out_mode, _r, key_bytes_per_lane, _rsv, lanes2 = _DPF2PC.unpack_from(head, 0)
        kb = int(key_bytes_per_lane)
        lanes2i = int(lanes2)
        keys = bytes(head[_DPF2PC.size :])
        if len(keys) != kb * lanes2i:
            raise ValueError("DPF_2PC keys length mismatch")
        dpf2 = OPLUTDPF2PCV1(
            dpf_alg=int(dpf_alg),
            domain_w=int(dw),
            out_mode=int(out_mode),
            key_bytes_per_lane=kb,
            lanes=lanes2i,
            keys=keys,
        )

    return OPLUTRecordV1(
        header=FSSRecordHeaderV1(kind=int(kind), fss_id=int(fss_id), header_bytes=hb, section_count=sc, payload_bytes=int(payload_bytes)),
        dirents=tuple(dirents),
        meta=meta,
        mask=mask,
        refresh=refresh,
        dpf2pc=dpf2,
    )


def _derive_edge_component_u16_u8(
    *,
    seed32: bytes,
    nonce12: bytes,
    ctr_u64: int,
    w: int,
) -> int:
    _require_len(seed32, 32, "seed32")
    _require_len(nonce12, 12, "nonce12")
    W = int(w)
    if W not in (8, 16):
        raise ValueError("w must be 8 or 16")
    block = chacha20_block_bytes_v1(key32=seed32, nonce12=nonce12, counter32=int(ctr_u64) & 0xFFFFFFFF)
    if W == 8:
        return int(block[0]) & 0xFF
    return int.from_bytes(block[0:2], "little", signed=False) & 0xFFFF


def build_oplut_record_blobs_v1(
    *,
    sid: bytes,
    fss_id: int,
    sgir_op_id: int,
    domain_w: int,
    lanes: int,
    lane_base: int,
    K_master32: bytes,
    seed_edge01_32: bytes,
    seed_edge12_32: bytes,
    seed_edge20_32: bytes,
    counter0: int = 0,
    lane_stride: int = 1,
) -> Tuple[bytes, bytes, bytes]:
    """
    Deterministic OP_LUT record builder for tests/demo.
    Produces three per-party blobs matching the spec layout (META + MASK_RSS + [DPF_2PC] + REFRESH).
    """
    _require_len(K_master32, 32, "K_master32")
    _require_len(seed_edge01_32, 32, "seed_edge01_32")
    _require_len(seed_edge12_32, 32, "seed_edge12_32")
    _require_len(seed_edge20_32, 32, "seed_edge20_32")
    W = int(domain_w)
    if W not in (8, 16):
        raise ValueError("domain_w must be 8 or 16")
    L = int(lanes)
    if L <= 0:
        raise ValueError("lanes must be >0")

    salt16 = oplut_salt16_v1(sid=sid, fss_id=int(fss_id), sgir_op_id=int(sgir_op_id))
    nonce_r = oplut_nonce_r12_v1(salt16=salt16)
    nonce_m = oplut_nonce_m12_v1(salt16=salt16)

    # Derive full r_lane values so we can keygen per-lane DPF at alpha=r_lane.
    mask = (1 << W) - 1
    r_lane: list[int] = []
    for ell in range(L):
        ctr = int(counter0) + (int(lane_base) + ell) * int(lane_stride)
        c0 = _derive_edge_component_u16_u8(seed32=seed_edge20_32, nonce12=nonce_r, ctr_u64=ctr, w=W) & mask
        c1 = _derive_edge_component_u16_u8(seed32=seed_edge01_32, nonce12=nonce_r, ctr_u64=ctr, w=W) & mask
        c2 = _derive_edge_component_u16_u8(seed32=seed_edge12_32, nonce12=nonce_r, ctr_u64=ctr, w=W) & mask
        r_lane.append(int((c0 + c1 + c2) & mask))

    # Key size per lane for DPFKey_GGM_AES128_AR64_v1:
    # root_seed16(16) + root_t(1) + w*(cw_seed_L16+cw_seed_R16+cw_tL+cw_tR) + cw_last_u64(8)
    key_bytes_per_lane = 16 + 1 + int(W) * (16 + 16 + 1 + 1) + 8
    sid32 = sha256(sid) if len(sid) != 32 else bytes(sid)

    # Precompute lane keys for P0 and P1.
    keys_p0 = bytearray()
    keys_p1 = bytearray()
    for ell in range(L):
        k0, k1 = keygen_dpf_aes128_ar64_v1(
            K_master32=K_master32,
            sid32=sid32,
            fss_id=int(fss_id),
            w=W,
            alpha=int(r_lane[ell]),
            beta_u64=1,
        )
        keys_p0 += k0
        keys_p1 += k1

    def pack_meta(*, party_id: int, dpf_role: int) -> bytes:
        return _META.pack(
            int(party_id) & 0xFF,
            int(dpf_role) & 0xFF,
            W & 0xFF,
            0,
            L & 0xFFFFFFFF,
            RING_Z2_64,
            PRG_CHACHA20,
            b"\x00" * 3,
            salt16,
            int(lane_base) & 0xFFFFFFFFFFFFFFFF,
            key_bytes_per_lane & 0xFFFF,
            0,
        )

    def pack_mask(*, edge_a: int, edge_b: int, seed_a32: bytes, seed_b32: bytes) -> bytes:
        return _MASK.pack(
            int(edge_a) & 0xFF,
            int(edge_b) & 0xFF,
            b"\x00\x00",
            bytes(seed_a32),
            bytes(seed_b32),
            nonce_r,
            int(counter0) & 0xFFFFFFFFFFFFFFFF,
            int(lane_stride) & 0xFFFFFFFFFFFFFFFF,
        )

    def pack_refresh(*, uses_edge20: int, uses_edge12: int) -> bytes:
        return _REFRESH.pack(
            int(uses_edge20) & 0xFF,
            int(uses_edge12) & 0xFF,
            b"\x00\x00",
            nonce_m,
            int(counter0) & 0xFFFFFFFFFFFFFFFF,
            int(lane_stride) & 0xFFFFFFFFFFFFFFFF,
        )

    def pack_dpf2pc(keys: bytes) -> bytes:
        hdr = _DPF2PC.pack(
            DPFALG_GGM_AES128,
            W & 0xFF,
            1,
            0,
            key_bytes_per_lane & 0xFFFF,
            0,
            L & 0xFFFFFFFF,
        )
        return bytes(hdr + bytes(keys))

    def build_blob(*, party_id: int) -> bytes:
        if int(party_id) == P0:
            meta = pack_meta(party_id=P0, dpf_role=DPF_LEFT)
            masksec = pack_mask(edge_a=EDGE_20, edge_b=EDGE_01, seed_a32=seed_edge20_32, seed_b32=seed_edge01_32)
            refresh = pack_refresh(uses_edge20=1, uses_edge12=0)
            dpfsec = pack_dpf2pc(keys_p0)
            secs = [(SEC_META, meta), (SEC_MASK_RSS, masksec), (SEC_DPF_2PC, dpfsec), (SEC_REFRESH, refresh)]
        elif int(party_id) == P1:
            meta = pack_meta(party_id=P1, dpf_role=DPF_RIGHT)
            masksec = pack_mask(edge_a=EDGE_01, edge_b=EDGE_12, seed_a32=seed_edge01_32, seed_b32=seed_edge12_32)
            refresh = pack_refresh(uses_edge20=0, uses_edge12=1)
            dpfsec = pack_dpf2pc(keys_p1)
            secs = [(SEC_META, meta), (SEC_MASK_RSS, masksec), (SEC_DPF_2PC, dpfsec), (SEC_REFRESH, refresh)]
        elif int(party_id) == P2:
            meta = pack_meta(party_id=P2, dpf_role=DPF_NONE)
            masksec = pack_mask(edge_a=EDGE_12, edge_b=EDGE_20, seed_a32=seed_edge12_32, seed_b32=seed_edge20_32)
            refresh = pack_refresh(uses_edge20=1, uses_edge12=1)
            secs = [(SEC_META, meta), (SEC_MASK_RSS, masksec), (SEC_REFRESH, refresh)]
        else:
            raise ValueError("party_id must be 0..2")

        payload = bytearray()
        dirents = []
        for t, b in secs:
            off = len(payload)
            payload += bytes(b)
            dirents.append((int(t), 0, int(off), len(b)))

        section_count = len(dirents)
        header_bytes = _HDR.size + section_count * _DIRENT.size
        hdr = _HDR.pack(MAGIC_FSSREC_V1, FSSREC_VERSION_V1, FSS_KIND_OP_LUT, int(fss_id) & 0xFFFFFFFFFFFFFFFF, header_bytes, section_count, len(payload))
        out = bytearray(hdr)
        for t, fl, off, n in dirents:
            out += _DIRENT.pack(int(t) & 0xFFFFFFFF, int(fl) & 0xFFFFFFFF, int(off) & 0xFFFFFFFFFFFFFFFF, int(n) & 0xFFFFFFFFFFFFFFFF)
        out += payload
        return bytes(out)

    return build_blob(party_id=P0), build_blob(party_id=P1), build_blob(party_id=P2)


