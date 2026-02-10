from __future__ import annotations

# pyright: reportMissingImports=false

import torch

from uvcc_party.cuda_ext import matmul_u64


_U64_MASK = (1 << 64) - 1


def _u64_to_i64(v_u64: int) -> int:
    v = int(v_u64) & _U64_MASK
    if v >= (1 << 63):
        v -= 1 << 64
    return int(v)


def _matmul_u64_ref_cpu(A_u64_i64: torch.Tensor, B_u64_i64: torch.Tensor) -> torch.Tensor:
    if A_u64_i64.dtype != torch.int64 or B_u64_i64.dtype != torch.int64:
        raise TypeError("ref expects int64 tensors")
    if A_u64_i64.dim() != 2 or B_u64_i64.dim() != 2:
        raise ValueError("ref expects 2D tensors")
    if A_u64_i64.shape[1] != B_u64_i64.shape[0]:
        raise ValueError("shape mismatch")
    A = A_u64_i64.cpu()
    B = B_u64_i64.cpu()
    m, k = int(A.shape[0]), int(A.shape[1])
    n = int(B.shape[1])
    out = torch.empty((m, n), dtype=torch.int64, device="cpu")
    for i in range(m):
        for j in range(n):
            acc = 0
            for t in range(k):
                a = int(A[i, t].item()) & _U64_MASK
                b = int(B[t, j].item()) & _U64_MASK
                acc = (acc + (a * b)) & _U64_MASK
            out[i, j] = _u64_to_i64(acc)
    return out


def test_cuda_matmul_u64_matches_cpu_ref_small() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    # Use small shapes; CPU ref is O(n^3).
    A = torch.randint(-(1 << 63), (1 << 63) - 1, (16, 16), dtype=torch.int64, device="cpu")
    B = torch.randint(-(1 << 63), (1 << 63) - 1, (16, 16), dtype=torch.int64, device="cpu")

    exp = _matmul_u64_ref_cpu(A, B)
    got = matmul_u64(A.cuda(), B.cuda()).cpu()

    assert got.dtype == torch.int64
    assert got.shape == exp.shape
    assert torch.equal(got, exp)


def test_cuda_matmul_u64_non_square() -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(1)
    A = torch.randint(-(1 << 63), (1 << 63) - 1, (8, 5), dtype=torch.int64, device="cpu")
    B = torch.randint(-(1 << 63), (1 << 63) - 1, (5, 7), dtype=torch.int64, device="cpu")

    exp = _matmul_u64_ref_cpu(A, B)
    got = matmul_u64(A.cuda(), B.cuda()).cpu()
    assert torch.equal(got, exp)


