from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b50b9410e18bf310,uvcc_group_c3c058cd62b15ae3

import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MAGIC_SGIR = b"SGIR"

# SGIR module hash alg identifiers (uvcc_sgir_v1.h).
HASH_SHA256 = 1
HASH_BLAKE3_256 = 2
HASH_KECCAK256 = 3

ENDIAN_LITTLE = 1


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


_MOD_HDR = struct.Struct("<4sHHIIIIBBHQ16s16s8s")  # 76 bytes (packed)
_SECT_ENTRY = struct.Struct("<IIQQ32s")  # 56 bytes


@dataclass(frozen=True)
class SGIRModuleHeaderV1:
    ver_major: int
    ver_minor: int
    header_bytes: int
    flags: int
    section_count: int
    section_table_off: int
    hash_alg: int
    endianness: int
    file_bytes: int
    module_uuid16: bytes

    @staticmethod
    def from_bytes(buf: bytes) -> "SGIRModuleHeaderV1":
        if len(buf) < _MOD_HDR.size:
            raise ValueError("buffer too small for SGIR header")
        (
            magic,
            ver_major,
            ver_minor,
            header_bytes,
            flags,
            section_count,
            section_table_off,
            hash_alg,
            endianness,
            reserved0,
            file_bytes,
            module_uuid16,
            reserved1_16,
            header_checksum8,
        ) = _MOD_HDR.unpack_from(buf, 0)
        if magic != MAGIC_SGIR:
            raise ValueError("bad SGIR magic")
        if int(reserved0) != 0:
            raise ValueError("reserved0 must be 0")
        if reserved1_16 != b"\x00" * 16:
            raise ValueError("reserved1 must be 0")
        if header_checksum8 != b"\x00" * 8:
            raise ValueError("header_checksum must be 0 in v1")
        if int(endianness) != ENDIAN_LITTLE:
            raise ValueError("only little-endian SGIR supported in v1")
        if int(header_bytes) != _MOD_HDR.size:
            raise ValueError("header_bytes mismatch")
        return SGIRModuleHeaderV1(
            ver_major=int(ver_major),
            ver_minor=int(ver_minor),
            header_bytes=int(header_bytes),
            flags=int(flags),
            section_count=int(section_count),
            section_table_off=int(section_table_off),
            hash_alg=int(hash_alg),
            endianness=int(endianness),
            file_bytes=int(file_bytes),
            module_uuid16=bytes(module_uuid16),
        )


@dataclass(frozen=True)
class SGIRSectionEntryV1:
    kind: int
    flags: int
    offset: int
    length: int
    sha256_32: bytes  # either all-zero or SHA256(payload)

    @staticmethod
    def from_bytes(buf: bytes, off: int) -> "SGIRSectionEntryV1":
        kind, flags, offset, length, sha256_32 = _SECT_ENTRY.unpack_from(buf, off)
        return SGIRSectionEntryV1(
            kind=int(kind),
            flags=int(flags),
            offset=int(offset),
            length=int(length),
            sha256_32=bytes(sha256_32),
        )


# Section kinds (uvcc_sgir_v1.h subset)
SECT_STRTAB = 1
SECT_SYMTAB = 2
SECT_TYPETAB = 3
SECT_VALTAB = 4
SECT_FUNTAB = 5
SECT_CODE = 6
SECT_CONST = 7
SECT_ATTR = 8


# SGIR core enums from privacy_new.txt §A.6.1.
SEC_PUB = 0
SEC_SEC = 1

DT_I1 = 1
DT_I8 = 2
DT_I16 = 3
DT_I32 = 4
DT_I64 = 5
DT_U8 = 6
DT_U16 = 7
DT_U32 = 8
DT_U64 = 9

LYT_ROW_MAJOR = 1

DIM_LIT = 0
DIM_SYM = 1


@dataclass(frozen=True)
class SGIRDimV1:
    kind: int
    sym_id: int
    lit: int


@dataclass(frozen=True)
class SGIRTypeRecordV1:
    secrecy: int
    dtype: int
    layout: int
    rank: int
    fxp_frac_bits: int
    dims: Tuple[SGIRDimV1, ...]


@dataclass(frozen=True)
class SGIRValueRecordV1:
    type_id: int
    name_str_id: int
    storage: int
    flags: int


@dataclass(frozen=True)
class SGIRFunctionV1:
    name_str_id: int
    entry_block_id: int
    block_count: int
    param_count: int
    result_count: int
    blocks_off: int  # within FUNTAB payload
    code_off: int  # within CODE section payload
    code_len: int


@dataclass(frozen=True)
class SGIRBlockV1:
    block_id: int
    inst_start_off: int
    inst_end_off: int


_TYPETAB_HDR = struct.Struct("<II")
_VALTAB_HDR = struct.Struct("<II")
_VALREC = struct.Struct("<IIII")

_TYPEREC_BASE = struct.Struct("<BBHHhI")  # 12 bytes packed
_DIMREC = struct.Struct("<BBHIQ")  # 16 bytes packed

_FUNTAB_HDR = struct.Struct("<II")
_FUNREC = struct.Struct("<IIIIIQQQ")  # 44 bytes packed
_BLKREC = struct.Struct("<IIQQ")  # 24 bytes packed


@dataclass(frozen=True)
class SGIRInstructionV1:
    opcode: int
    flags: int
    dst_ids: Tuple[int, ...]
    src_ids: Tuple[int, ...]
    imm: bytes


_INST_HDR = struct.Struct("<HHHHHH")  # 12 bytes (packed, per struct definition)


@dataclass(frozen=True)
class SGIRModuleV1:
    raw_bytes: bytes
    header: SGIRModuleHeaderV1
    sections: Tuple[SGIRSectionEntryV1, ...]
    section_payloads: Dict[int, bytes]  # kind -> bytes (first occurrence)
    types: Tuple[SGIRTypeRecordV1, ...]
    values: Tuple[SGIRValueRecordV1, ...]
    functions: Tuple[SGIRFunctionV1, ...]
    blocks: Dict[int, Tuple[SGIRBlockV1, ...]]  # fun_index -> blocks
    code_bytes: bytes

    def module_hash32(self) -> bytes:
        # Normative: module hash is over the entire module byte stream as stored.
        if int(self.header.hash_alg) == HASH_SHA256:
            return sha256(self.raw_bytes)
        raise ValueError("unsupported SGIR hash_alg in v1")

    def iter_function_instructions(self, fun_index: int) -> List[SGIRInstructionV1]:
        f = self.functions[int(fun_index)]
        code = self.code_bytes[int(f.code_off) : int(f.code_off) + int(f.code_len)]
        out: List[SGIRInstructionV1] = []
        off = 0
        while off < len(code):
            if (len(code) - off) < _INST_HDR.size:
                raise ValueError("truncated instruction header")
            opcode, flags, dst_count, src_count, imm_bytes, reserved = _INST_HDR.unpack_from(code, off)
            off += _INST_HDR.size
            if int(reserved) != 0:
                raise ValueError("inst reserved must be 0")
            dst_ids = []
            for _ in range(int(dst_count)):
                if (len(code) - off) < 4:
                    raise ValueError("truncated dst_ids")
                (vid,) = struct.unpack_from("<I", code, off)
                off += 4
                dst_ids.append(int(vid))
            src_ids = []
            for _ in range(int(src_count)):
                if (len(code) - off) < 4:
                    raise ValueError("truncated src_ids")
                (vid,) = struct.unpack_from("<I", code, off)
                off += 4
                src_ids.append(int(vid))
            if (len(code) - off) < int(imm_bytes):
                raise ValueError("truncated imm bytes")
            imm = bytes(code[off : off + int(imm_bytes)])
            off += int(imm_bytes)
            out.append(SGIRInstructionV1(opcode=int(opcode), flags=int(flags), dst_ids=tuple(dst_ids), src_ids=tuple(src_ids), imm=imm))
        return out


def parse_sgir_module_v1(buf: bytes) -> SGIRModuleV1:
    hdr = SGIRModuleHeaderV1.from_bytes(buf)
    if int(hdr.file_bytes) != len(buf):
        raise ValueError("file_bytes mismatch")
    if int(hdr.section_table_off) < int(hdr.header_bytes):
        raise ValueError("section_table_off must be >= header_bytes")
    table_off = int(hdr.section_table_off)
    if table_off + int(hdr.section_count) * _SECT_ENTRY.size > len(buf):
        raise ValueError("section table out of bounds")

    sections: List[SGIRSectionEntryV1] = []
    for i in range(int(hdr.section_count)):
        sections.append(SGIRSectionEntryV1.from_bytes(buf, table_off + i * _SECT_ENTRY.size))

    payloads: Dict[int, bytes] = {}
    for s in sections:
        if int(s.offset) + int(s.length) > len(buf):
            raise ValueError("section out of bounds")
        pl = bytes(buf[int(s.offset) : int(s.offset) + int(s.length)])
        if s.sha256_32 != b"\x00" * 32:
            if sha256(pl) != s.sha256_32:
                raise ValueError("section sha256 mismatch")
        if int(s.kind) not in payloads:
            payloads[int(s.kind)] = pl

    # Parse TYPETAB
    if SECT_TYPETAB not in payloads:
        raise ValueError("missing TYPETAB section")
    typ_pl = payloads[SECT_TYPETAB]
    if len(typ_pl) < _TYPETAB_HDR.size:
        raise ValueError("bad TYPETAB header")
    type_count, reserved = _TYPETAB_HDR.unpack_from(typ_pl, 0)
    if int(reserved) != 0:
        raise ValueError("TYPETAB reserved must be 0")
    types: List[SGIRTypeRecordV1] = []
    off = _TYPETAB_HDR.size
    for _ in range(int(type_count)):
        if (len(typ_pl) - off) < _TYPEREC_BASE.size:
            raise ValueError("truncated type record")
        secrecy, dtype, layout, rank, fxp_frac_bits, reserved0 = _TYPEREC_BASE.unpack_from(typ_pl, off)
        off += _TYPEREC_BASE.size
        if int(reserved0) != 0:
            raise ValueError("type record reserved must be 0")
        dims: List[SGIRDimV1] = []
        for _d in range(int(rank)):
            if (len(typ_pl) - off) < _DIMREC.size:
                raise ValueError("truncated dim record")
            kind, r0, r1, sym_id, lit = _DIMREC.unpack_from(typ_pl, off)
            off += _DIMREC.size
            if int(r0) != 0 or int(r1) != 0:
                raise ValueError("dim reserved must be 0")
            dims.append(SGIRDimV1(kind=int(kind), sym_id=int(sym_id), lit=int(lit)))
        types.append(
            SGIRTypeRecordV1(
                secrecy=int(secrecy),
                dtype=int(dtype),
                layout=int(layout),
                rank=int(rank),
                fxp_frac_bits=int(fxp_frac_bits),
                dims=tuple(dims),
            )
        )
    if off != len(typ_pl):
        # Allow trailing padding only if it's zero; v1 expects exact.
        if any(b != 0 for b in typ_pl[off:]):
            raise ValueError("extra nonzero bytes in TYPETAB")

    # Parse VALTAB
    if SECT_VALTAB not in payloads:
        raise ValueError("missing VALTAB section")
    val_pl = payloads[SECT_VALTAB]
    if len(val_pl) < _VALTAB_HDR.size:
        raise ValueError("bad VALTAB header")
    value_count, reserved = _VALTAB_HDR.unpack_from(val_pl, 0)
    if int(reserved) != 0:
        raise ValueError("VALTAB reserved must be 0")
    values: List[SGIRValueRecordV1] = []
    off = _VALTAB_HDR.size
    for _ in range(int(value_count)):
        if (len(val_pl) - off) < _VALREC.size:
            raise ValueError("truncated value record")
        type_id, name_str_id, storage, flags = _VALREC.unpack_from(val_pl, off)
        off += _VALREC.size
        values.append(SGIRValueRecordV1(type_id=int(type_id), name_str_id=int(name_str_id), storage=int(storage), flags=int(flags)))
    if off != len(val_pl):
        if any(b != 0 for b in val_pl[off:]):
            raise ValueError("extra nonzero bytes in VALTAB")

    # Parse FUNTAB
    if SECT_FUNTAB not in payloads:
        raise ValueError("missing FUNTAB section")
    fun_pl = payloads[SECT_FUNTAB]
    if len(fun_pl) < _FUNTAB_HDR.size:
        raise ValueError("bad FUNTAB header")
    fun_count, reserved = _FUNTAB_HDR.unpack_from(fun_pl, 0)
    if int(reserved) != 0:
        raise ValueError("FUNTAB reserved must be 0")
    funs: List[SGIRFunctionV1] = []
    off = _FUNTAB_HDR.size
    for _ in range(int(fun_count)):
        if (len(fun_pl) - off) < _FUNREC.size:
            raise ValueError("truncated function record")
        name_str_id, entry_block_id, block_count, param_count, result_count, blocks_off, code_off, code_len = _FUNREC.unpack_from(fun_pl, off)
        off += _FUNREC.size
        funs.append(
            SGIRFunctionV1(
                name_str_id=int(name_str_id),
                entry_block_id=int(entry_block_id),
                block_count=int(block_count),
                param_count=int(param_count),
                result_count=int(result_count),
                blocks_off=int(blocks_off),
                code_off=int(code_off),
                code_len=int(code_len),
            )
        )

    blocks_by_fun: Dict[int, Tuple[SGIRBlockV1, ...]] = {}
    for fi, f in enumerate(funs):
        boff = int(f.blocks_off)
        if boff < 0 or boff > len(fun_pl):
            raise ValueError("blocks_off out of bounds")
        blk_off = boff
        blks: List[SGIRBlockV1] = []
        for _ in range(int(f.block_count)):
            if (len(fun_pl) - blk_off) < _BLKREC.size:
                raise ValueError("truncated block record")
            block_id, reserved_b, inst_start_off, inst_end_off = _BLKREC.unpack_from(fun_pl, blk_off)
            blk_off += _BLKREC.size
            if int(reserved_b) != 0:
                raise ValueError("block reserved must be 0")
            blks.append(SGIRBlockV1(block_id=int(block_id), inst_start_off=int(inst_start_off), inst_end_off=int(inst_end_off)))
        blocks_by_fun[fi] = tuple(blks)

    # Parse CODE payload (raw bytes)
    if SECT_CODE not in payloads:
        raise ValueError("missing CODE section")
    code_bytes = payloads[SECT_CODE]

    return SGIRModuleV1(
        raw_bytes=bytes(buf),
        header=hdr,
        sections=tuple(sections),
        section_payloads=payloads,
        types=tuple(types),
        values=tuple(values),
        functions=tuple(funs),
        blocks=blocks_by_fun,
        code_bytes=code_bytes,
    )


