from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class A2BPackV1:
    """A2B/EDABIT carrier in-memory form for the v1 runtime (torch tensors)."""

    w_bits: int
    count_vals: int
    sgir_op_id: int
    base_triple_id: int
    sid_hash32: bytes

    r_arith_lo: torch.Tensor
    r_arith_hi: torch.Tensor
    r_bits_lo: torch.Tensor  # shape [N, w] int64 0/1
    r_bits_hi: torch.Tensor  # shape [N, w] int64 0/1

    def __post_init__(self) -> None:
        if int(self.w_bits) not in (8, 16):
            raise ValueError("w_bits must be 8 or 16")
        if len(self.sid_hash32) != 32:
            raise ValueError("sid_hash32 must be 32 bytes")
        n = int(self.count_vals)
        w = int(self.w_bits)
        if self.r_arith_lo.dtype != torch.int64 or self.r_arith_hi.dtype != torch.int64:
            raise TypeError("r_arith_lo/hi must be int64 tensors")
        if self.r_arith_lo.shape != (n,) or self.r_arith_hi.shape != (n,):
            raise ValueError("r_arith_lo/hi must be shape (N,)")
        if self.r_bits_lo.shape != (n, w) or self.r_bits_hi.shape != (n, w):
            raise ValueError("r_bits_lo/hi must be shape (N,w)")
        if self.r_bits_lo.dtype != torch.int64 or self.r_bits_hi.dtype != torch.int64:
            raise TypeError("r_bits_lo/hi must be int64 tensors")


