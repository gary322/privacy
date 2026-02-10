from __future__ import annotations

import hashlib

import torch

from uvcc_party.dpf_dcf import PRG_CHACHA12, PRIM_DCF, PRIM_DPF, keygen_dpf_dcf_keyrecs_v1
from uvcc_party.fss_executor import fss_eval_step_cpu_v1
from uvcc_party.fss_plan import FSSExecTaskV1, FSSPlanPrimeV1


def _pack_bits_to_u32_words(bits01_i64: torch.Tensor) -> torch.Tensor:
    bits = bits01_i64.to(torch.int64).view(-1)
    n = int(bits.numel())
    assert (n % 32) == 0
    out = torch.zeros((n // 32,), dtype=torch.int32)
    for w in range(n // 32):
        v = 0
        for i in range(32):
            v |= (int(bits[w * 32 + i].item()) & 1) << i
        if v >= (1 << 31):
            v -= 1 << 32
        out[w] = int(v)
    return out


def test_fss_eval_step_cpu_v1_two_tasks_w8_dpf_and_dcf() -> None:
    sid = b"sid-fss-exec-cpu"
    sid_hash32 = hashlib.sha256(sid).digest()
    alpha_dpf = 77
    alpha_dcf = 123

    k0_dpf, k1_dpf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0xAAAABBBBCCCCDDDD,
        alpha=alpha_dpf,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x10" * 32,
        prim_type=PRIM_DPF,
    )
    k0_dcf, k1_dcf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1111222233334444,
        alpha=alpha_dcf,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x20" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    # Two tasks: DPF then DCF.
    key_arena0 = k0_dpf + k0_dcf
    key_arena1 = k1_dpf + k1_dcf
    off_dpf = 0
    off_dcf = len(k0_dpf)
    out0 = 0
    out1 = 32  # 8 u32 words * 4 bytes
    tasks = [
        FSSExecTaskV1(
            fss_id=0,
            sgir_op_id=1,
            lane=0xFFFF,
            kind=1,
            domain_bits=8,
            range_bits=1,
            in_type=1,
            out_type=1,
            flags=0,
            lanes=256,
            in_offset=0,
            in_stride=0,
            out_offset=out0,
            out_stride=4,
            key_offset=off_dpf,
            key_bytes=len(k0_dpf),
        ),
        FSSExecTaskV1(
            fss_id=0,
            sgir_op_id=2,
            lane=0xFFFF,
            kind=2,
            domain_bits=8,
            range_bits=1,
            in_type=1,
            out_type=1,
            flags=0,
            lanes=256,
            in_offset=0,
            in_stride=0,
            out_offset=out1,
            out_stride=4,
            key_offset=off_dcf,
            key_bytes=len(k0_dcf),
        ),
    ]
    plan = FSSPlanPrimeV1(task_count=2, key_arena_bytes=len(key_arena0), in_arena_bytes=0, out_arena_bytes=64, scratch_bytes=0)

    r0 = fss_eval_step_cpu_v1(plan=plan, tasks=tasks, key_arena=key_arena0)
    r1 = fss_eval_step_cpu_v1(plan=plan, tasks=tasks, key_arena=key_arena1)

    # DPF public one-hot
    pub_dpf = r0.out_words_i32[0] ^ r1.out_words_i32[0]
    expect_bits = torch.zeros((256,), dtype=torch.int64)
    expect_bits[alpha_dpf] = 1
    assert torch.equal(pub_dpf, _pack_bits_to_u32_words(expect_bits))

    # DCF public threshold
    pub_dcf = r0.out_words_i32[1] ^ r1.out_words_i32[1]
    expect_bits = (torch.arange(256, dtype=torch.int64) < alpha_dcf).to(torch.int64)
    assert torch.equal(pub_dcf, _pack_bits_to_u32_words(expect_bits))


