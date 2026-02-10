from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_fef1ba5ecf03ce43

from dataclasses import dataclass
from typing import Tuple

import torch


# Edge ids for 2-party primitives in 3PC RSS (matches uvcc_fss_keyrec_v1.h).
EDGE_01 = 0  # share1 known to (P0,P1)
EDGE_12 = 1  # share2 known to (P1,P2)
EDGE_20 = 2  # share0 known to (P2,P0)


def rss_pair_share_indices_for_party_v1(party_id: int) -> Tuple[int, int]:
    """
    Party Pi holds the pair (share_i, share_{i+1}) in v1 RSS.
    """
    pid = int(party_id)
    if pid not in (0, 1, 2):
        raise ValueError("party_id must be 0..2")
    return pid, (pid + 1) % 3


def rss_share_index_for_edge_v1(edge: int) -> int:
    """
    Map a pairwise edge id to the corresponding RSS component share index:
      EDGE_20 -> share0, EDGE_01 -> share1, EDGE_12 -> share2
    """
    e = int(edge)
    if e == EDGE_20:
        return 0
    if e == EDGE_01:
        return 1
    if e == EDGE_12:
        return 2
    raise ValueError("edge must be 0..2")


@dataclass(frozen=True)
class RSSArithU64:
    """Replicated secret share (RSS) of u64 values over Z/2^64.

    Party Pi holds the pair (x_i, x_{i+1}). We store those as `lo` and `hi` tensors of dtype int64
    representing the raw 64-bit ring element bit-patterns.
    """

    lo: torch.Tensor
    hi: torch.Tensor
    fxp_frac_bits: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.lo, torch.Tensor) or not isinstance(self.hi, torch.Tensor):
            raise TypeError("lo/hi must be torch tensors")
        if self.lo.dtype != torch.int64 or self.hi.dtype != torch.int64:
            raise TypeError("RSSArithU64 requires int64 tensors (u64 bit-patterns)")
        if self.lo.shape != self.hi.shape:
            raise ValueError("lo/hi shape mismatch")

    def add(self, other: "RSSArithU64") -> "RSSArithU64":
        if self.lo.shape != other.lo.shape:
            raise ValueError("shape mismatch")
        return RSSArithU64(lo=self.lo + other.lo, hi=self.hi + other.hi, fxp_frac_bits=self.fxp_frac_bits)

    def sub(self, other: "RSSArithU64") -> "RSSArithU64":
        if self.lo.shape != other.lo.shape:
            raise ValueError("shape mismatch")
        return RSSArithU64(lo=self.lo - other.lo, hi=self.hi - other.hi, fxp_frac_bits=self.fxp_frac_bits)


@dataclass(frozen=True)
class RSSBoolU64Words:
    """Boolean RSS over GF(2) with packed u64 word storage.

    Each lane is one bit; bits are packed LSB-first into u64 words (stored as int64 bit-patterns).
    """

    lo_words: torch.Tensor  # int64 words
    hi_words: torch.Tensor  # int64 words
    n_bits: int

    def __post_init__(self) -> None:
        if self.lo_words.dtype != torch.int64 or self.hi_words.dtype != torch.int64:
            raise TypeError("RSSBoolU64Words requires int64 tensors")
        if self.lo_words.shape != self.hi_words.shape:
            raise ValueError("lo_words/hi_words shape mismatch")
        if int(self.n_bits) < 0:
            raise ValueError("n_bits must be >= 0")


def _rand_i64_bits(shape, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    lo = torch.randint(0, 2**32, shape, dtype=torch.int64, generator=gen, device=device)
    hi = torch.randint(0, 2**32, shape, dtype=torch.int64, generator=gen, device=device)
    return (hi << 32) | lo


def make_rss_arith_u64_triple(
    *,
    x_pub: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[RSSArithU64, RSSArithU64, RSSArithU64]:
    """Create 3-party replicated shares for a public u64 vector `x_pub`."""

    if not isinstance(x_pub, torch.Tensor) or x_pub.dtype != torch.int64:
        raise TypeError("x_pub must be torch.int64 tensor of u64 bit-patterns")
    a = _rand_i64_bits(x_pub.shape, generator, device)
    b = _rand_i64_bits(x_pub.shape, generator, device)
    c = x_pub.to(device) - a - b

    p0 = RSSArithU64(lo=a, hi=b)
    p1 = RSSArithU64(lo=b, hi=c)
    p2 = RSSArithU64(lo=c, hi=a)
    return p0, p1, p2


