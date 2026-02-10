from __future__ import annotations

import pytest
import torch

from uvcc_party.cuda_ext import trunc_apply_u64


def _u64_to_i64(v: int) -> int:
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v >= (1 << 63) else v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_trunc_apply_u64_matches_python_u64() -> None:
    # Choose an F where add_const fits comfortably in signed range, but still exercises modular arithmetic.
    F = 16
    add_const = 1 << (64 - F)
    n = 257  # includes a remainder block

    gen = torch.Generator(device="cpu").manual_seed(999)

    # Public C1 values (u64).
    C1 = [int(torch.randint(0, 1 << 32, (1,), generator=gen).item()) | (int(torch.randint(0, 1 << 32, (1,), generator=gen).item()) << 32) for _ in range(n)]
    # Secret share components (u64 bit-patterns).
    R1_lo = [int(torch.randint(0, 1 << 32, (1,), generator=gen).item()) | (int(torch.randint(0, 1 << 32, (1,), generator=gen).item()) << 32) for _ in range(n)]
    R1_hi = [int(torch.randint(0, 1 << 32, (1,), generator=gen).item()) | (int(torch.randint(0, 1 << 32, (1,), generator=gen).item()) << 32) for _ in range(n)]

    carry_lo = [int(torch.randint(0, 2, (1,), generator=gen).item()) for _ in range(n)]
    carry_hi = [int(torch.randint(0, 2, (1,), generator=gen).item()) for _ in range(n)]
    ov_lo = [int(torch.randint(0, 2, (1,), generator=gen).item()) for _ in range(n)]
    ov_hi = [int(torch.randint(0, 2, (1,), generator=gen).item()) for _ in range(n)]

    # Upload as int64 with u64 bit-patterns.
    C1_t = torch.tensor([_u64_to_i64(v) for v in C1], dtype=torch.int64, device="cuda")
    R1_lo_t = torch.tensor([_u64_to_i64(v) for v in R1_lo], dtype=torch.int64, device="cuda")
    R1_hi_t = torch.tensor([_u64_to_i64(v) for v in R1_hi], dtype=torch.int64, device="cuda")
    carry_lo_t = torch.tensor([_u64_to_i64(v) for v in carry_lo], dtype=torch.int64, device="cuda")
    carry_hi_t = torch.tensor([_u64_to_i64(v) for v in carry_hi], dtype=torch.int64, device="cuda")
    ov_lo_t = torch.tensor([_u64_to_i64(v) for v in ov_lo], dtype=torch.int64, device="cuda")
    ov_hi_t = torch.tensor([_u64_to_i64(v) for v in ov_hi], dtype=torch.int64, device="cuda")

    for pid in (0, 1, 2):
        y_lo_cuda, y_hi_cuda = trunc_apply_u64(
            C1_pub_u64_i64=C1_t,
            R1_lo_u64_i64=R1_lo_t,
            R1_hi_u64_i64=R1_hi_t,
            carry_lo_u64_i64=carry_lo_t,
            carry_hi_u64_i64=carry_hi_t,
            ov_lo_u64_i64=ov_lo_t,
            ov_hi_u64_i64=ov_hi_t,
            add_const_u64=add_const,
            party_id=pid,
        )
        y_lo = [_u64_to_i64(int(v) & 0xFFFFFFFFFFFFFFFF) for v in y_lo_cuda.cpu().tolist()]
        y_hi = [_u64_to_i64(int(v) & 0xFFFFFFFFFFFFFFFF) for v in y_hi_cuda.cpu().tolist()]

        exp_lo: list[int] = []
        exp_hi: list[int] = []
        for i in range(n):
            y0 = (-R1_lo[i] - (carry_lo[i] & 1) + ((ov_lo[i] & 1) * add_const)) & 0xFFFFFFFFFFFFFFFF
            y1 = (-R1_hi[i] - (carry_hi[i] & 1) + ((ov_hi[i] & 1) * add_const)) & 0xFFFFFFFFFFFFFFFF
            if pid == 0:
                y0 = (y0 + C1[i]) & 0xFFFFFFFFFFFFFFFF
            if pid == 2:
                y1 = (y1 + C1[i]) & 0xFFFFFFFFFFFFFFFF
            exp_lo.append(_u64_to_i64(y0))
            exp_hi.append(_u64_to_i64(y1))

        assert y_lo == exp_lo
        assert y_hi == exp_hi


