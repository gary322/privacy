from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_4fc95d18c2004924

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .dpf_dcf import PRIM_DCF, PRIM_DPF, KeyrecV1, dcf_full_w8_v1, dcf_stage2_w16_v1, dpf_full_w8_v1, dpf_stage1_w16_v1, dpf_stage2_w16_v1
from .fss_plan import FSSExecTaskV1, FSSPlanPrimeV1


def _pack_bits_to_u32_words(bits01_i64: torch.Tensor) -> torch.Tensor:
    bits = bits01_i64.to(torch.int64).view(-1)
    n = int(bits.numel())
    if (n % 32) != 0:
        raise ValueError("n_bits must be a multiple of 32 for BITPACK32")
    out = torch.zeros((n // 32,), dtype=torch.int32, device=torch.device("cpu"))
    for w in range(n // 32):
        v = 0
        for i in range(32):
            b = int(bits[w * 32 + i].item()) & 1
            v |= b << i
        # Store as signed int32 holding an unsigned u32 bit-pattern.
        if v >= (1 << 31):
            v -= 1 << 32
        out[w] = int(v)
    return out


def fss_eval_full_domain_bitpack32_cpu_v1(*, keyrec_bytes: bytes) -> torch.Tensor:
    """
    CPU reference: evaluate a single DPF/DCF keyrec over the full domain and return BITPACK32 words.
    """
    kr = KeyrecV1.from_bytes(keyrec_bytes)
    w = int(kr.w)
    if w == 8:
        if int(kr.prim_type) == PRIM_DPF:
            bits = dpf_full_w8_v1(keyrec_bytes, device=torch.device("cpu"))
        elif int(kr.prim_type) == PRIM_DCF:
            bits = (dcf_full_w8_v1(keyrec_bytes, device=torch.device("cpu")) & 1).to(torch.int64)
        else:
            raise ValueError("unsupported prim_type")
        return _pack_bits_to_u32_words(bits)
    if w == 16:
        front = dpf_stage1_w16_v1(keyrec_bytes, device=torch.device("cpu"))
        if int(kr.prim_type) == PRIM_DPF:
            bits = dpf_stage2_w16_v1(keyrec_bytes, frontier_seed_lo=front[0], frontier_seed_hi=front[1], frontier_t=front[2], device=torch.device("cpu"))
        elif int(kr.prim_type) == PRIM_DCF:
            bits = (dcf_stage2_w16_v1(keyrec_bytes, frontier_seed_lo=front[0], frontier_seed_hi=front[1], frontier_t=front[2], frontier_acc=front[3], device=torch.device("cpu")) & 1).to(torch.int64)
        else:
            raise ValueError("unsupported prim_type")
        return _pack_bits_to_u32_words(bits)
    raise ValueError("unsupported w")


@dataclass(frozen=True)
class FSSStepResultV1:
    out_arena_bytes: bytes
    out_words_i32: List[torch.Tensor]  # per task, CPU tensors


def fss_eval_step_cpu_v1(
    *,
    plan: FSSPlanPrimeV1,
    tasks: List[FSSExecTaskV1],
    key_arena: bytes,
) -> FSSStepResultV1:
    """
    Deterministic CPU reference for a step-wide FSS plan execution.

    This implements the full-domain mode used by the v1 DPF/DCF kernels:
    - ignores in_arena (tasks must be full-domain compatible)
    - writes BITPACK32 outputs at each task's out_offset.
    """
    out_arena = bytearray(b"\x00" * int(plan.out_arena_bytes))
    outs: List[torch.Tensor] = []
    for t in tasks:
        key_blob = bytes(key_arena[int(t.key_offset) : int(t.key_offset) + int(t.key_bytes)])
        words = fss_eval_full_domain_bitpack32_cpu_v1(keyrec_bytes=key_blob)
        outs.append(words)
        out_off = int(t.out_offset)
        out_bytes = b"".join(int(x.item() & 0xFFFFFFFF).to_bytes(4, "little", signed=False) for x in words)
        if out_off + len(out_bytes) > len(out_arena):
            raise ValueError("out_arena overflow")
        out_arena[out_off : out_off + len(out_bytes)] = out_bytes
    return FSSStepResultV1(out_arena_bytes=bytes(out_arena), out_words_i32=outs)


def fss_eval_full_domain_bitpack32_cuda_v1(*, keyrec_bytes: bytes) -> torch.Tensor:
    """
    GPU (CUDA) full-domain evaluator returning BITPACK32 int32 words.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    from .cuda_ext import dcf_full_w8_bitpack32, dpf_full_w8_bitpack32, dpf_stage1_w16, dpf_stage2_w16_bitpack32, dcf_stage2_w16_bitpack32

    kr = KeyrecV1.from_bytes(keyrec_bytes)
    w = int(kr.w)
    key_t = torch.tensor(list(keyrec_bytes), dtype=torch.uint8, device="cuda")
    if w == 8:
        if int(kr.prim_type) == PRIM_DPF:
            return dpf_full_w8_bitpack32(key_t)
        if int(kr.prim_type) == PRIM_DCF:
            return dcf_full_w8_bitpack32(key_t)
        raise ValueError("unsupported prim_type")
    if w == 16:
        seed_lo, seed_hi, t_u8, acc_u8 = dpf_stage1_w16(key_t)
        if int(kr.prim_type) == PRIM_DPF:
            return dpf_stage2_w16_bitpack32(key_t, seed_lo, seed_hi, t_u8)
        if int(kr.prim_type) == PRIM_DCF:
            return dcf_stage2_w16_bitpack32(key_t, seed_lo, seed_hi, t_u8, acc_u8)
        raise ValueError("unsupported prim_type")
    raise ValueError("unsupported w")


def fss_eval_step_cuda_v1(
    *,
    plan: FSSPlanPrimeV1,
    tasks: List[FSSExecTaskV1],
    key_arena: bytes,
) -> FSSStepResultV1:
    """
    Minimal GPU host wrapper for v1 FSS step evaluation.

    This matches the plan/task ABI expectations and produces the canonical BITPACK32 outputs.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    out_arena = bytearray(b"\x00" * int(plan.out_arena_bytes))
    outs: List[torch.Tensor] = []
    for t in tasks:
        key_blob = bytes(key_arena[int(t.key_offset) : int(t.key_offset) + int(t.key_bytes)])
        words_cuda = fss_eval_full_domain_bitpack32_cuda_v1(keyrec_bytes=key_blob)
        words = words_cuda.cpu()
        outs.append(words)
        out_off = int(t.out_offset)
        out_bytes = b"".join(int(x.item() & 0xFFFFFFFF).to_bytes(4, "little", signed=False) for x in words)
        if out_off + len(out_bytes) > len(out_arena):
            raise ValueError("out_arena overflow")
        out_arena[out_off : out_off + len(out_bytes)] = out_bytes
    return FSSStepResultV1(out_arena_bytes=bytes(out_arena), out_words_i32=outs)


