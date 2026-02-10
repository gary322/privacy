from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b50b9410e18bf310,uvcc_group_c3c058cd62b15ae3

import struct

from uvcc_party.sgir import (
    DIM_LIT,
    DT_U64,
    ENDIAN_LITTLE,
    HASH_SHA256,
    LYT_ROW_MAJOR,
    SEC_PUB,
    SEC_SEC,
    SECT_CODE,
    SECT_FUNTAB,
    SECT_TYPETAB,
    SECT_VALTAB,
    parse_sgir_module_v1,
)


def _align8(x: int) -> int:
    r = x % 8
    return x if r == 0 else (x + (8 - r))


def _build_min_sgir_add_open_module(*, n: int) -> bytes:
    # Build minimal SGIR module bytes with TYPETAB/VALTAB/FUNTAB/CODE.
    hdr_size = 76
    sect_entry_size = 56

    # TYPETAB: two types (sec u64[N], pub u64[N])
    typ = bytearray()
    typ += struct.pack("<II", 2, 0)
    for secrecy in (SEC_SEC, SEC_PUB):
        typ += struct.pack("<BBHHhI", secrecy, DT_U64, LYT_ROW_MAJOR, 1, 0, 0)
        typ += struct.pack("<BBHIQ", DIM_LIT, 0, 0, 0, int(n))
    typ_pl = bytes(typ)

    # VALTAB: 4 values: x,y,z,z_pub
    # storage: 1 INPUT, 0 TMP, 2 OUTPUT (from privacy_new).
    val = bytearray()
    val += struct.pack("<II", 4, 0)
    # x: type0
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 1, 0)
    # y: type0
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 1, 0)
    # z: type0 tmp
    val += struct.pack("<IIII", 0, 0xFFFFFFFF, 0, 0)
    # z_pub: type1 output
    val += struct.pack("<IIII", 1, 0xFFFFFFFF, 2, 0)
    val_pl = bytes(val)

    # CODE: OP_ADD dst=2 src=0,1; OP_OPEN dst=3 src=2; OP_RET
    inst = bytearray()
    inst += struct.pack("<HHHHHH", 10, 0, 1, 2, 0, 0)  # OP_ADD
    inst += struct.pack("<I", 2)
    inst += struct.pack("<II", 0, 1)
    inst += struct.pack("<HHHHHH", 60, 0, 1, 1, 0, 0)  # OP_OPEN
    inst += struct.pack("<I", 3)
    inst += struct.pack("<I", 2)
    inst += struct.pack("<HHHHHH", 52, 0, 0, 0, 0, 0)  # OP_RET
    code_pl = bytes(inst)

    # FUNTAB: one function, one block.
    fun = bytearray()
    fun += struct.pack("<II", 1, 0)
    blocks_off = 8 + 44  # header + one function record
    fun += struct.pack("<IIIIIQQQ", 0, 0, 1, 2, 1, blocks_off, 0, len(code_pl))
    fun += struct.pack("<IIQQ", 0, 0, 0, len(code_pl))
    fun_pl = bytes(fun)

    payloads = [
        (SECT_TYPETAB, typ_pl),
        (SECT_VALTAB, val_pl),
        (SECT_FUNTAB, fun_pl),
        (SECT_CODE, code_pl),
    ]

    # Place payloads after header, 8-byte aligned.
    cur = _align8(hdr_size)
    sect_meta = []
    blob = bytearray(b"\x00" * hdr_size)
    for kind, pl in payloads:
        cur = _align8(cur)
        if len(blob) < cur:
            blob += b"\x00" * (cur - len(blob))
        off = cur
        blob += pl
        cur = off + len(pl)
        sect_meta.append((kind, off, len(pl)))

    section_table_off = _align8(cur)
    if len(blob) < section_table_off:
        blob += b"\x00" * (section_table_off - len(blob))

    # Section table entries
    for kind, off, ln in sect_meta:
        blob += struct.pack("<IIQQ32s", int(kind), 0, int(off), int(ln), b"\x00" * 32)

    file_bytes = len(blob)

    # Now fill module header (76 bytes).
    hdr = struct.pack(
        "<4sHHIIIIBBHQ16s16s8s",
        b"SGIR",
        0,
        1,
        hdr_size,
        0,
        len(sect_meta),
        section_table_off,
        HASH_SHA256,
        ENDIAN_LITTLE,
        0,
        file_bytes,
        b"\x00" * 16,
        b"\x00" * 16,
        b"\x00" * 8,
    )
    assert len(hdr) == hdr_size
    blob[0:hdr_size] = hdr
    return bytes(blob)


def test_parse_min_sgir_module_v1() -> None:
    b = _build_min_sgir_add_open_module(n=17)
    m = parse_sgir_module_v1(b)
    assert m.header.file_bytes == len(b)
    assert len(m.types) == 2
    assert len(m.values) == 4
    assert len(m.functions) == 1
    insts = m.iter_function_instructions(0)
    assert [i.opcode for i in insts] == [10, 60, 52]


