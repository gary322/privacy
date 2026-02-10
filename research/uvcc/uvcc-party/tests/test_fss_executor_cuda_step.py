from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_4fc95d18c2004924

import pytest
import torch

from uvcc_party.dpf_dcf import PRG_AES128, PRG_CHACHA12, PRIM_DCF, PRIM_DPF, keygen_dpf_dcf_keyrecs_v1
from uvcc_party.fss_executor import fss_eval_step_cpu_v1, fss_eval_step_cuda_v1
from uvcc_party.fss_plan import FSSExecTaskV1, FSSPlanPrimeV1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fss_eval_step_cuda_matches_cpu_for_bitpack32_tasks() -> None:
    sid = b"sid-fss-step"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()

    # Build two keyrecs: one DPF(w=8), one DCF(w=16). Exercise both PRGs.
    k0_dpf, _k1_dpf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1010,
        alpha=77,
        w=8,
        prg_id=PRG_AES128,
        party_edge=0,
        master_seed32=b"\x11" * 32,
        prim_type=PRIM_DPF,
    )
    k0_dcf, _k1_dcf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x2020,
        alpha=1337,
        w=16,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x22" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    key_arena = k0_dpf + k0_dcf
    tasks = [
        FSSExecTaskV1(
            fss_id=0x1010,
            sgir_op_id=1,
            lane=0xFFFF,
            kind=1,  # UVCC_FSS_DPF_POINT
            domain_bits=8,
            range_bits=1,
            in_type=1,  # UVCC_IN_U16
            out_type=1,  # UVCC_OUT_BITPACK32
            flags=0,
            lanes=1 << 8,
            in_offset=0,
            in_stride=0,
            out_offset=0,
            out_stride=0,
            key_offset=0,
            key_bytes=len(k0_dpf),
        ),
        FSSExecTaskV1(
            fss_id=0x2020,
            sgir_op_id=2,
            lane=0xFFFF,
            kind=2,  # UVCC_FSS_DCF_LT
            domain_bits=16,
            range_bits=1,
            in_type=1,
            out_type=1,
            flags=0,
            lanes=1 << 16,
            in_offset=0,
            in_stride=0,
            out_offset=64,  # arbitrary byte offset
            out_stride=0,
            key_offset=len(k0_dpf),
            key_bytes=len(k0_dcf),
        ),
    ]
    plan = FSSPlanPrimeV1(task_count=len(tasks), key_arena_bytes=len(key_arena), in_arena_bytes=0, out_arena_bytes=64 + 2048 * 4)

    cpu = fss_eval_step_cpu_v1(plan=plan, tasks=tasks, key_arena=key_arena)
    gpu = fss_eval_step_cuda_v1(plan=plan, tasks=tasks, key_arena=key_arena)

    assert gpu.out_arena_bytes == cpu.out_arena_bytes
    assert len(gpu.out_words_i32) == len(cpu.out_words_i32) == 2
    assert torch.equal(gpu.out_words_i32[0], cpu.out_words_i32[0])
    assert torch.equal(gpu.out_words_i32[1], cpu.out_words_i32[1])


