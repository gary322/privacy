from __future__ import annotations

import pytest
import torch


def _cpu_cpub_to_cjmask_u8(c_pub_u8: torch.Tensor) -> torch.Tensor:
    if c_pub_u8.dtype != torch.uint8 or c_pub_u8.ndim != 1:
        raise TypeError("c_pub_u8 must be 1D uint8")
    L = int(c_pub_u8.numel())
    n_words = (L + 31) // 32
    pad = n_words * 32 - L
    if pad:
        c = torch.cat([c_pub_u8, torch.zeros((pad,), dtype=torch.uint8)], dim=0)
    else:
        c = c_pub_u8
    c2 = c.view(n_words, 32).to(torch.int64)
    weights = (1 << torch.arange(32, dtype=torch.int64)).view(1, 32)
    out = torch.empty((8 * n_words,), dtype=torch.int32)
    for j in range(8):
        bits = (c2 >> j) & 1
        words = (bits * weights).sum(dim=1)
        out[j * n_words : (j + 1) * n_words] = words.to(torch.int32)
    return out


def _cpu_cpub_to_cjmask_u16(c_pub_i16: torch.Tensor) -> torch.Tensor:
    if c_pub_i16.dtype != torch.int16 or c_pub_i16.ndim != 1:
        raise TypeError("c_pub_i16 must be 1D int16 (u16 bit-patterns)")
    L = int(c_pub_i16.numel())
    n_words = (L + 31) // 32
    pad = n_words * 32 - L
    if pad:
        c = torch.cat([c_pub_i16, torch.zeros((pad,), dtype=torch.int16)], dim=0)
    else:
        c = c_pub_i16
    vals = (c.view(n_words, 32).to(torch.int32) & 0xFFFF).to(torch.int64)
    weights = (1 << torch.arange(32, dtype=torch.int64)).view(1, 32)
    out = torch.empty((16 * n_words,), dtype=torch.int32)
    for j in range(16):
        bits = (vals >> j) & 1
        words = (bits * weights).sum(dim=1)
        out[j * n_words : (j + 1) * n_words] = words.to(torch.int32)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_a2b_pack_c_lo_u8_u16_matches_cpu() -> None:
    from uvcc_party.cuda_ext import a2b_pack_c_lo_u16, a2b_pack_c_lo_u8

    gen = torch.Generator(device="cpu").manual_seed(2025)
    n = 4097  # cover multiple blocks and tail
    # Generate arbitrary u64 bit-patterns by sampling signed int64.
    x = torch.randint(-(2**63), 2**63 - 1, (n,), dtype=torch.int64, generator=gen)
    r = torch.randint(-(2**63), 2**63 - 1, (n,), dtype=torch.int64, generator=gen)

    # CPU expected
    exp_u8 = bytearray(n)
    exp_u16 = []
    for i in range(n):
        xu = int(x[i].item()) & 0xFFFFFFFFFFFFFFFF
        ru = int(r[i].item()) & 0xFFFFFFFFFFFFFFFF
        cu = (xu + ru) & 0xFFFFFFFFFFFFFFFF
        exp_u8[i] = cu & 0xFF
        exp_u16.append(cu & 0xFFFF)
    exp_u8_t = torch.tensor(list(exp_u8), dtype=torch.uint8)
    exp_u16_t = torch.tensor(exp_u16, dtype=torch.int64).to(torch.int16)

    out_u8 = a2b_pack_c_lo_u8(x_lo_u64_i64=x.cuda(), r_lo_u64_i64=r.cuda()).cpu()
    out_u16 = a2b_pack_c_lo_u16(x_lo_u64_i64=x.cuda(), r_lo_u64_i64=r.cuda()).cpu()
    assert torch.equal(out_u8, exp_u8_t)
    assert torch.equal(out_u16, exp_u16_t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_a2b_cpub_to_cjmask_u8_u16_matches_cpu() -> None:
    from uvcc_party.cuda_ext import a2b_cpub_to_cjmask_u16, a2b_cpub_to_cjmask_u8

    gen = torch.Generator(device="cpu").manual_seed(42)

    # U8
    L8 = 77
    c8 = torch.randint(0, 256, (L8,), dtype=torch.int64, generator=gen).to(torch.uint8)
    exp8 = _cpu_cpub_to_cjmask_u8(c8)
    out8 = a2b_cpub_to_cjmask_u8(c_pub_u8=c8.cuda()).cpu()
    assert torch.equal(out8, exp8)

    # U16 (stored as int16 bit-patterns)
    L16 = 123
    c16_u = torch.randint(0, 65536, (L16,), dtype=torch.int64, generator=gen)
    c16 = c16_u.to(torch.int16)
    exp16 = _cpu_cpub_to_cjmask_u16(c16)
    out16 = a2b_cpub_to_cjmask_u16(c_pub_u16_i16=c16.cuda()).cpu()
    assert torch.equal(out16, exp16)


