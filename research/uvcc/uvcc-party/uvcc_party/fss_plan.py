from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List

from .transcript import sha256


DS_FSSEVAL = b"UVCC.FSSEVAL.v1\0"


_PLAN = struct.Struct("<IIQQQQQQQQQ")  # 80 bytes
_TASK = struct.Struct("<QIHBBHHBBHIQIIQIIQII")  # 76 bytes


@dataclass(frozen=True)
class FSSPlanPrimeV1:
    task_count: int
    key_arena_bytes: int
    in_arena_bytes: int
    out_arena_bytes: int
    scratch_bytes: int = 0

    def to_bytes(self) -> bytes:
        # UVCC_FSSPlanDevice_v1 with pointer fields replaced by 0 (privacy_new.txt §8.1).
        return _PLAN.pack(
            1,  # version
            int(self.task_count) & 0xFFFFFFFF,
            int(self.key_arena_bytes) & 0xFFFFFFFFFFFFFFFF,
            int(self.in_arena_bytes) & 0xFFFFFFFFFFFFFFFF,
            int(self.out_arena_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
            0,
            0,
            0,
            int(self.scratch_bytes) & 0xFFFFFFFFFFFFFFFF,
            0,
        )


@dataclass(frozen=True)
class FSSExecTaskV1:
    fss_id: int
    sgir_op_id: int
    lane: int
    kind: int
    domain_bits: int
    range_bits: int
    in_type: int
    out_type: int
    flags: int
    lanes: int
    in_offset: int
    in_stride: int
    out_offset: int
    out_stride: int
    key_offset: int
    key_bytes: int

    def to_bytes(self) -> bytes:
        return _TASK.pack(
            int(self.fss_id) & 0xFFFFFFFFFFFFFFFF,
            int(self.sgir_op_id) & 0xFFFFFFFF,
            int(self.lane) & 0xFFFF,
            int(self.kind) & 0xFF,
            0,
            int(self.domain_bits) & 0xFFFF,
            int(self.range_bits) & 0xFFFF,
            int(self.in_type) & 0xFF,
            int(self.out_type) & 0xFF,
            int(self.flags) & 0xFFFF,
            int(self.lanes) & 0xFFFFFFFF,
            int(self.in_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.in_stride) & 0xFFFFFFFF,
            0,
            int(self.out_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.out_stride) & 0xFFFFFFFF,
            0,
            int(self.key_offset) & 0xFFFFFFFFFFFFFFFF,
            int(self.key_bytes) & 0xFFFFFFFF,
            0,
        )


def fss_tasks_bytes_v1(tasks: Iterable[FSSExecTaskV1]) -> bytes:
    out = bytearray()
    for t in tasks:
        out += t.to_bytes()
    return bytes(out)


def fsseval_hash32_v1(*, plan_prime_bytes: bytes, tasks_bytes: bytes, fssblock_hash32: bytes) -> bytes:
    if len(fssblock_hash32) != 32:
        raise ValueError("fssblock_hash32 must be 32 bytes")
    return sha256(DS_FSSEVAL + bytes(plan_prime_bytes) + bytes(tasks_bytes) + bytes(fssblock_hash32))


def build_fsseval_hash32_from_parts_v1(*, plan: FSSPlanPrimeV1, tasks: List[FSSExecTaskV1], fssblock_hash32: bytes) -> bytes:
    plan_b = plan.to_bytes()
    tasks_b = fss_tasks_bytes_v1(tasks)
    return fsseval_hash32_v1(plan_prime_bytes=plan_b, tasks_bytes=tasks_b, fssblock_hash32=fssblock_hash32)


