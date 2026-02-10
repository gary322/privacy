from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable


DS_OP_LUT = b"UVCC.OP_LUT.v1\0"


# UVCC_LUTPlanDevice_v1 (packed, pointers treated as u64).
_PLAN = struct.Struct("<II" + "Q" * 11)  # 96 bytes

# UVCC_LUTTask_v1 (packed) — 84 bytes.
_TASK = struct.Struct("<QIBBBB" "I" "QII" "QII" "QII" "QII")


@dataclass(frozen=True)
class OPLUTPlanPrimeV1:
    task_count: int
    key_arena_bytes: int
    const_arena_bytes: int
    u_pub_bytes: int
    out_arena_bytes: int
    scratch_bytes: int = 0

    def to_bytes(self) -> bytes:
        # Pointers are zeroed to avoid nondeterminism.
        return _PLAN.pack(
            1,
            int(self.task_count) & 0xFFFFFFFF,
            0,
            int(self.key_arena_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
            int(self.const_arena_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
            int(self.u_pub_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
            int(self.out_arena_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
            int(self.scratch_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
        )


@dataclass(frozen=True)
class OPLUTTaskV1:
    fss_id: int
    sgir_op_id: int
    domain_w: int
    elem_fmt: int
    dpf_mode: int
    flags: int
    lanes: int
    u_pub_offset: int
    u_pub_stride: int
    table_offset: int
    table_bytes: int
    out_offset: int
    out_stride: int
    key_offset: int
    key_bytes: int

    def to_bytes(self) -> bytes:
        return _TASK.pack(
            int(self.fss_id) & 0xFFFFFFFFFFFFFFFF,
            int(self.sgir_op_id) & 0xFFFFFFFF,
            int(self.domain_w) & 0xFF,
            int(self.elem_fmt) & 0xFF,
            int(self.dpf_mode) & 0xFF,
            int(self.flags) & 0xFF,
            int(self.lanes) & 0xFFFFFFFF,
            int(self.u_pub_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.u_pub_stride) & 0xFFFFFFFF,
            0,
            int(self.table_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.table_bytes) & 0xFFFFFFFF,
            0,
            int(self.out_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.out_stride) & 0xFFFFFFFF,
            0,
            int(self.key_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.key_bytes) & 0xFFFFFFFF,
            0,
        )


def oplut_tasks_bytes_v1(tasks: Iterable[OPLUTTaskV1]) -> bytes:
    out = bytearray()
    for t in tasks:
        out += t.to_bytes()
    return bytes(out)


