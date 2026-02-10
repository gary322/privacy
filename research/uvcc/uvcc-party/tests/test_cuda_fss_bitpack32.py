from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_e4c33796e7d46452,uvcc_group_bfa55eababdc7ee8,uvcc_group_37bb78042819626f

import pytest
import torch

from uvcc_party.dpf_dcf import (
    PRG_AES128,
    PRG_CHACHA12,
    PRIM_DCF,
    PRIM_DPF,
    dcf_full_w8_v1,
    dcf_stage2_w16_v1,
    dpf_full_w8_v1,
    dpf_stage1_w16_v1,
    dpf_stage2_w16_v1,
    keygen_dpf_dcf_keyrecs_v1,
)


def _pack_bits_to_u32_words(bits01_i64: torch.Tensor) -> torch.Tensor:
    bits = bits01_i64.to(torch.int64).view(-1)
    n = int(bits.numel())
    if (n % 32) != 0:
        raise ValueError("n must be multiple of 32")
    words = torch.zeros((n // 32,), dtype=torch.int32)
    for w in range(n // 32):
        v = 0
        for i in range(32):
            b = int(bits[w * 32 + i].item()) & 1
            v |= b << i
        if v >= (1 << 31):
            v -= 1 << 32
        words[w] = int(v)
    return words


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_dcf_full_w8_bitpack32_matches_cpu() -> None:
    from uvcc_party.cuda_ext import dcf_full_w8_bitpack32

    sid = b"sid-cuda-bitpack-w8"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    alpha = 123
    key0, key1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x9999000011112222,
        alpha=alpha,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x12" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    cpu0 = dcf_full_w8_v1(key0, device=torch.device("cpu"))
    cpu1 = dcf_full_w8_v1(key1, device=torch.device("cpu"))
    cpu0_bits = (cpu0 & 1).to(torch.int64)
    cpu1_bits = (cpu1 & 1).to(torch.int64)
    cpu0_words = _pack_bits_to_u32_words(cpu0_bits)
    cpu1_words = _pack_bits_to_u32_words(cpu1_bits)

    key0_t = torch.tensor(list(key0), dtype=torch.uint8, device="cuda")
    key1_t = torch.tensor(list(key1), dtype=torch.uint8, device="cuda")
    gpu0_words = dcf_full_w8_bitpack32(key0_t).cpu()
    gpu1_words = dcf_full_w8_bitpack32(key1_t).cpu()

    assert torch.equal(gpu0_words, cpu0_words)
    assert torch.equal(gpu1_words, cpu1_words)

    pub_words = gpu0_words ^ gpu1_words
    expect_bits = (torch.arange(256, dtype=torch.int64) < alpha).to(torch.int64)
    expect_words = _pack_bits_to_u32_words(expect_bits)
    assert torch.equal(pub_words, expect_words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_dpf_full_w8_bitpack32_matches_cpu() -> None:
    from uvcc_party.cuda_ext import dpf_full_w8_bitpack32

    sid = b"sid-cuda-bitpack-dpf-w8"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    alpha = 77
    key0, key1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x8888000011112222,
        alpha=alpha,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x34" * 32,
        prim_type=PRIM_DPF,
    )

    cpu0_bits = dpf_full_w8_v1(key0, device=torch.device("cpu"))
    cpu1_bits = dpf_full_w8_v1(key1, device=torch.device("cpu"))
    cpu0_words = _pack_bits_to_u32_words(cpu0_bits)
    cpu1_words = _pack_bits_to_u32_words(cpu1_bits)

    key0_t = torch.tensor(list(key0), dtype=torch.uint8, device="cuda")
    key1_t = torch.tensor(list(key1), dtype=torch.uint8, device="cuda")
    gpu0_words = dpf_full_w8_bitpack32(key0_t).cpu()
    gpu1_words = dpf_full_w8_bitpack32(key1_t).cpu()

    assert torch.equal(gpu0_words, cpu0_words)
    assert torch.equal(gpu1_words, cpu1_words)

    pub_words = gpu0_words ^ gpu1_words
    expect_bits = torch.zeros((256,), dtype=torch.int64)
    expect_bits[alpha] = 1
    expect_words = _pack_bits_to_u32_words(expect_bits)
    assert torch.equal(pub_words, expect_words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_bitpack32_aes_matches_cpu_for_w8_and_w16() -> None:
    """
    Exercise AES PRG path (prg_id=1) through the same full-domain BITPACK32 entrypoints.
    """
    from uvcc_party.cuda_ext import dcf_full_w8_bitpack32, dpf_full_w8_bitpack32, dpf_stage1_w16, dpf_stage2_w16_bitpack32

    sid = b"sid-cuda-bitpack-aes"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    alpha = 200

    # w=8 DPF
    k0_dpf, k1_dpf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1111000011112222,
        alpha=alpha,
        w=8,
        prg_id=PRG_AES128,
        party_edge=0,
        master_seed32=b"\xAB" * 32,
        prim_type=PRIM_DPF,
    )
    cpu0_bits = dpf_full_w8_v1(k0_dpf, device=torch.device("cpu"))
    cpu1_bits = dpf_full_w8_v1(k1_dpf, device=torch.device("cpu"))
    cpu0_words = _pack_bits_to_u32_words(cpu0_bits)
    cpu1_words = _pack_bits_to_u32_words(cpu1_bits)
    k0_t = torch.tensor(list(k0_dpf), dtype=torch.uint8, device="cuda")
    k1_t = torch.tensor(list(k1_dpf), dtype=torch.uint8, device="cuda")
    g0 = dpf_full_w8_bitpack32(k0_t).cpu()
    g1 = dpf_full_w8_bitpack32(k1_t).cpu()
    assert torch.equal(g0, cpu0_words)
    assert torch.equal(g1, cpu1_words)

    # w=8 DCF
    k0_dcf, k1_dcf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1111000011113333,
        alpha=alpha,
        w=8,
        prg_id=PRG_AES128,
        party_edge=0,
        master_seed32=b"\xCD" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )
    cpu0_u64 = dcf_full_w8_v1(k0_dcf, device=torch.device("cpu"))
    cpu1_u64 = dcf_full_w8_v1(k1_dcf, device=torch.device("cpu"))
    cpu0_words = _pack_bits_to_u32_words((cpu0_u64 & 1).to(torch.int64))
    cpu1_words = _pack_bits_to_u32_words((cpu1_u64 & 1).to(torch.int64))
    k0_t = torch.tensor(list(k0_dcf), dtype=torch.uint8, device="cuda")
    k1_t = torch.tensor(list(k1_dcf), dtype=torch.uint8, device="cuda")
    g0 = dcf_full_w8_bitpack32(k0_t).cpu()
    g1 = dcf_full_w8_bitpack32(k1_t).cpu()
    assert torch.equal(g0, cpu0_words)
    assert torch.equal(g1, cpu1_words)

    # w=16 DPF (bitpack32 stage2)
    k0, k1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1111000011114444,
        alpha=1337,
        w=16,
        prg_id=PRG_AES128,
        party_edge=0,
        master_seed32=b"\xEF" * 32,
        prim_type=PRIM_DPF,
    )
    cpu0_front = dpf_stage1_w16_v1(k0, device=torch.device("cpu"))
    cpu1_front = dpf_stage1_w16_v1(k1, device=torch.device("cpu"))
    cpu0_bits = dpf_stage2_w16_v1(k0, frontier_seed_lo=cpu0_front[0], frontier_seed_hi=cpu0_front[1], frontier_t=cpu0_front[2], device=torch.device("cpu"))
    cpu1_bits = dpf_stage2_w16_v1(k1, frontier_seed_lo=cpu1_front[0], frontier_seed_hi=cpu1_front[1], frontier_t=cpu1_front[2], device=torch.device("cpu"))
    cpu0_words = _pack_bits_to_u32_words(cpu0_bits)
    cpu1_words = _pack_bits_to_u32_words(cpu1_bits)
    k0_t = torch.tensor(list(k0), dtype=torch.uint8, device="cuda")
    k1_t = torch.tensor(list(k1), dtype=torch.uint8, device="cuda")
    g0_seed_lo, g0_seed_hi, g0_t, _ = dpf_stage1_w16(k0_t)
    g1_seed_lo, g1_seed_hi, g1_t, _ = dpf_stage1_w16(k1_t)
    g0_words = dpf_stage2_w16_bitpack32(k0_t, g0_seed_lo, g0_seed_hi, g0_t).cpu()
    g1_words = dpf_stage2_w16_bitpack32(k1_t, g1_seed_lo, g1_seed_hi, g1_t).cpu()
    assert torch.equal(g0_words, cpu0_words)
    assert torch.equal(g1_words, cpu1_words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_stage2_w16_bitpack32_matches_cpu_for_dpf_and_dcf() -> None:
    from uvcc_party.cuda_ext import (
        dcf_stage2_w16_bitpack32,
        dpf_stage1_w16,
        dpf_stage2_w16_bitpack32,
    )

    sid = b"sid-cuda-bitpack-w16"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    alpha = 1337

    # DPF keys
    k0_dpf, k1_dpf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1234000011112222,
        alpha=alpha,
        w=16,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x56" * 32,
        prim_type=PRIM_DPF,
    )
    # DCF keys
    k0_dcf, k1_dcf = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1235000011112222,
        alpha=alpha,
        w=16,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x78" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    # CPU DPF
    cpu0_front = dpf_stage1_w16_v1(k0_dpf, device=torch.device("cpu"))
    cpu1_front = dpf_stage1_w16_v1(k1_dpf, device=torch.device("cpu"))
    cpu0_bits = dpf_stage2_w16_v1(k0_dpf, frontier_seed_lo=cpu0_front[0], frontier_seed_hi=cpu0_front[1], frontier_t=cpu0_front[2], device=torch.device("cpu"))
    cpu1_bits = dpf_stage2_w16_v1(k1_dpf, frontier_seed_lo=cpu1_front[0], frontier_seed_hi=cpu1_front[1], frontier_t=cpu1_front[2], device=torch.device("cpu"))
    cpu0_words = _pack_bits_to_u32_words(cpu0_bits)
    cpu1_words = _pack_bits_to_u32_words(cpu1_bits)

    # GPU DPF
    k0_t = torch.tensor(list(k0_dpf), dtype=torch.uint8, device="cuda")
    k1_t = torch.tensor(list(k1_dpf), dtype=torch.uint8, device="cuda")
    g0_seed_lo, g0_seed_hi, g0_t, _ = dpf_stage1_w16(k0_t)
    g1_seed_lo, g1_seed_hi, g1_t, _ = dpf_stage1_w16(k1_t)
    gpu0_words = dpf_stage2_w16_bitpack32(k0_t, g0_seed_lo, g0_seed_hi, g0_t).cpu()
    gpu1_words = dpf_stage2_w16_bitpack32(k1_t, g1_seed_lo, g1_seed_hi, g1_t).cpu()
    assert torch.equal(gpu0_words, cpu0_words)
    assert torch.equal(gpu1_words, cpu1_words)
    pub_words = gpu0_words ^ gpu1_words
    expect_bits = torch.zeros((65536,), dtype=torch.int64)
    expect_bits[alpha] = 1
    expect_words = _pack_bits_to_u32_words(expect_bits)
    assert torch.equal(pub_words, expect_words)

    # CPU DCF
    cpu0_front = dpf_stage1_w16_v1(k0_dcf, device=torch.device("cpu"))
    cpu1_front = dpf_stage1_w16_v1(k1_dcf, device=torch.device("cpu"))
    cpu0_u64 = dcf_stage2_w16_v1(
        k0_dcf,
        frontier_seed_lo=cpu0_front[0],
        frontier_seed_hi=cpu0_front[1],
        frontier_t=cpu0_front[2],
        frontier_acc=cpu0_front[3],
        device=torch.device("cpu"),
    )
    cpu1_u64 = dcf_stage2_w16_v1(
        k1_dcf,
        frontier_seed_lo=cpu1_front[0],
        frontier_seed_hi=cpu1_front[1],
        frontier_t=cpu1_front[2],
        frontier_acc=cpu1_front[3],
        device=torch.device("cpu"),
    )
    cpu0_words = _pack_bits_to_u32_words((cpu0_u64 & 1).to(torch.int64))
    cpu1_words = _pack_bits_to_u32_words((cpu1_u64 & 1).to(torch.int64))

    # GPU DCF
    k0_t = torch.tensor(list(k0_dcf), dtype=torch.uint8, device="cuda")
    k1_t = torch.tensor(list(k1_dcf), dtype=torch.uint8, device="cuda")
    g0_seed_lo, g0_seed_hi, g0_t, g0_acc = dpf_stage1_w16(k0_t)
    g1_seed_lo, g1_seed_hi, g1_t, g1_acc = dpf_stage1_w16(k1_t)
    gpu0_words = dcf_stage2_w16_bitpack32(k0_t, g0_seed_lo, g0_seed_hi, g0_t, g0_acc).cpu()
    gpu1_words = dcf_stage2_w16_bitpack32(k1_t, g1_seed_lo, g1_seed_hi, g1_t, g1_acc).cpu()
    assert torch.equal(gpu0_words, cpu0_words)
    assert torch.equal(gpu1_words, cpu1_words)
    pub_words = gpu0_words ^ gpu1_words
    expect_bits = (torch.arange(65536, dtype=torch.int64) < alpha).to(torch.int64)
    expect_words = _pack_bits_to_u32_words(expect_bits)
    assert torch.equal(pub_words, expect_words)


