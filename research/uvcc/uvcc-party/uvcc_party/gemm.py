from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .open import OpenArithItemU64, open_arith_u64_round_v1
from .party import Party
from .rss import RSSArithU64
from .tcf import TCFKeyV1, tcf_eval_v0a_tile_u64_v1


_U64_MASK = (1 << 64) - 1


def _u64_to_i64(x: int) -> int:
    v = int(x) & _U64_MASK
    if v >= (1 << 63):
        v -= 1 << 64
    return int(v)


def _matmul_u64_ref_cpu(A_u64_i64: torch.Tensor, B_u64_i64: torch.Tensor) -> torch.Tensor:
    """
    CPU reference for u64 ring matmul (mod 2^64) where tensors are int64 carrying u64 bit-patterns.
    This is intentionally simple and used only when CUDA kernels aren't applicable.
    """
    if A_u64_i64.dtype != torch.int64 or B_u64_i64.dtype != torch.int64:
        raise TypeError("matmul_u64_ref_cpu expects int64 tensors")
    if A_u64_i64.dim() != 2 or B_u64_i64.dim() != 2:
        raise ValueError("matmul_u64_ref_cpu expects 2D tensors")
    if A_u64_i64.shape[1] != B_u64_i64.shape[0]:
        raise ValueError("matmul_u64_ref_cpu shape mismatch")
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
    return out.to(device=A_u64_i64.device)


def _matmul_u64(A_u64_i64: torch.Tensor, B_u64_i64: torch.Tensor) -> torch.Tensor:
    """
    u64 ring matmul (mod 2^64) for int64 tensors carrying u64 bit-patterns.
    Uses CUDA kernel when tensors are CUDA; otherwise uses a strict CPU reference.
    """
    if A_u64_i64.is_cuda or B_u64_i64.is_cuda:
        if (A_u64_i64.is_cuda != B_u64_i64.is_cuda) or (A_u64_i64.device != B_u64_i64.device):
            raise ValueError("matmul_u64 requires both operands on the same device")
        from .cuda_ext import matmul_u64 as _cuda_matmul_u64

        return _cuda_matmul_u64(A_u64_i64.contiguous(), B_u64_i64.contiguous())
    return _matmul_u64_ref_cpu(A_u64_i64, B_u64_i64)


@dataclass(frozen=True)
class BeaverGEMMResultV1:
    """
    Result of one Beaver-style GEMM tile multiply.

    Exposes E/F opens and the triple used so SKS/Freivalds can reuse them.
    """

    Z: RSSArithU64
    E_pub: torch.Tensor
    F_pub: torch.Tensor
    triple_A: RSSArithU64
    triple_B: RSSArithU64
    triple_C: RSSArithU64


def _rss_add_public_into_share0(party_id: int, x: RSSArithU64, pub_u64_i64: torch.Tensor) -> RSSArithU64:
    if pub_u64_i64.dtype != torch.int64 or pub_u64_i64.shape != x.lo.shape:
        raise TypeError("pub_u64_i64 must match RSSArithU64 shape/dtype")
    pid = int(party_id)
    lo = x.lo
    hi = x.hi
    if pid == 0:
        lo = lo + pub_u64_i64
    elif pid == 2:
        hi = hi + pub_u64_i64
    return RSSArithU64(lo=lo, hi=hi, fxp_frac_bits=x.fxp_frac_bits)


def _matmul_public_left(E_pub: torch.Tensor, B: RSSArithU64) -> RSSArithU64:
    # (public matrix) @ (secret matrix) => secret matrix
    return RSSArithU64(lo=_matmul_u64(E_pub, B.lo), hi=_matmul_u64(E_pub, B.hi), fxp_frac_bits=B.fxp_frac_bits)


def _matmul_public_right(A: RSSArithU64, F_pub: torch.Tensor) -> RSSArithU64:
    # (secret matrix) @ (public matrix) => secret matrix
    return RSSArithU64(lo=_matmul_u64(A.lo, F_pub), hi=_matmul_u64(A.hi, F_pub), fxp_frac_bits=A.fxp_frac_bits)


def op_gemm_tile_beaver_tcf_v0a_u64_v1(
    party: Party,
    *,
    X: RSSArithU64,
    Y: RSSArithU64,
    tcf_key: TCFKeyV1,
    op_id: int,
    tile_i: int,
    tile_j: int,
    tile_p: int,
    epoch: int,
    step: int,
    sgir_op_id: int,
    fxp_frac_bits: int = 0,
    d: int = 16,
) -> BeaverGEMMResultV1:
    """
    Secure GEMM for a single dxd tile using Beaver triples from TCF-v0a.

    This is the standard RSS Beaver recipe:
      E = OPEN(X - A)
      F = OPEN(Y - B)
      Z = C + E*B + A*F + E*F
    """
    if X.lo.shape != (int(d), int(d)) or Y.lo.shape != (int(d), int(d)):
        raise ValueError("X and Y must be (d,d) tiles")
    if int(X.fxp_frac_bits) != int(fxp_frac_bits) or int(Y.fxp_frac_bits) != int(fxp_frac_bits):
        # Keep profile strict; callers should pass consistent fxp bits.
        raise ValueError("fxp_frac_bits mismatch")

    # Offline-ish: generate tile triple via TCF-v0a (1 round replication of Ck).
    A, B, C = tcf_eval_v0a_tile_u64_v1(
        party,
        key=tcf_key,
        op_id=int(op_id),
        i=int(tile_i),
        j=int(tile_j),
        p=int(tile_p),
        epoch=int(epoch),
        step=int(step),
        round=0,
        fxp_frac_bits=int(fxp_frac_bits),
        d=int(d),
    )
    # Allow callers to keep secret shares on GPU: TCF key expansion is CPU-based but the
    # resulting share tensors can be moved to the operand device for compute.
    dev = X.lo.device
    if A.lo.device != dev:
        A = RSSArithU64(lo=A.lo.to(dev), hi=A.hi.to(dev), fxp_frac_bits=A.fxp_frac_bits)
    if B.lo.device != dev:
        B = RSSArithU64(lo=B.lo.to(dev), hi=B.hi.to(dev), fxp_frac_bits=B.fxp_frac_bits)
    if C.lo.device != dev:
        C = RSSArithU64(lo=C.lo.to(dev), hi=C.hi.to(dev), fxp_frac_bits=C.fxp_frac_bits)

    # Online: OPEN E and F in a single OPEN_ARITH round.
    E = X.sub(A)
    F = Y.sub(B)
    open_id = int(sgir_op_id) & 0xFFFFFFFFFFFFFFFF
    out: Dict[Tuple[int, int], torch.Tensor] = open_arith_u64_round_v1(
        party,
        items=[
            OpenArithItemU64(open_id=open_id, sub_id=0, x=E),
            OpenArithItemU64(open_id=open_id, sub_id=1, x=F),
        ],
        epoch=int(epoch),
        step=int(step),
        round=1,
        sgir_op_id=int(sgir_op_id),
    )
    E_pub = out[(open_id, 0)].view(int(d), int(d))
    F_pub = out[(open_id, 1)].view(int(d), int(d))

    term1 = _matmul_public_left(E_pub, B)
    term2 = _matmul_public_right(A, F_pub)
    term3_pub = _matmul_u64(E_pub, F_pub).to(dtype=torch.int64)

    Z = C.add(term1).add(term2)
    Z = _rss_add_public_into_share0(int(party.party_id), Z, term3_pub)

    return BeaverGEMMResultV1(Z=Z, E_pub=E_pub, F_pub=F_pub, triple_A=A, triple_B=B, triple_C=C)


