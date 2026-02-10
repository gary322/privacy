from __future__ import annotations

# pyright: reportMissingImports=false

import hashlib

import torch

from uvcc_party.party import Party
from uvcc_party.tcf import TCFKeyV1, _tcf_prg_u64_tile_v1, tcf_eval_v0b_tile_u64_v1, tcf_gen_v1, tcf_tile_id32_v1


class _NullRelay:
    def enqueue(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("relay enqueue should not be called for v0b cache-hit")

    def poll(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("relay poll should not be called for v0b cache-hit")

    def ack(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("relay ack should not be called for v0b cache-hit")

    def healthz(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("relay healthz should not be called for v0b cache-hit")


def _u64(x: torch.Tensor) -> torch.Tensor:
    # Keep int64 u64 bit-patterns as-is (wraparound semantics come from int64 arithmetic).
    if x.dtype != torch.int64:
        raise TypeError("expected int64")
    return x


def test_tcf_v0b_cache_hit_has_no_repl_and_is_correct() -> None:
    sid = b"sid-tcf-v0b-cache"
    sid_hash32 = hashlib.sha256(sid).digest()
    master_seed32 = b"\x11" * 32
    k0, k1, k2 = tcf_gen_v1(master_seed32=master_seed32, sid=sid)

    # Recover full seeds (dealer view) to build cache blobs deterministically.
    s01 = k0.s01
    s02 = k0.s02
    s12 = k1.s12
    d = 16
    fxp = 0
    op_id, i, j, p = 7, 0, 0, 0
    tile_id32 = tcf_tile_id32_v1(sid_hash32=sid_hash32, op_id=op_id, i=i, j=j, p=p)

    A0 = _tcf_prg_u64_tile_v1(seed32=s02, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    A1 = _tcf_prg_u64_tile_v1(seed32=s01, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    A2 = _tcf_prg_u64_tile_v1(seed32=s12, label=b"A", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    B0 = _tcf_prg_u64_tile_v1(seed32=s02, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    B1 = _tcf_prg_u64_tile_v1(seed32=s01, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))
    B2 = _tcf_prg_u64_tile_v1(seed32=s12, label=b"B", tile_id32=tile_id32, d=d, device=torch.device("cpu"))

    C0 = (A0 @ B0) + (A0 @ B1) + (A1 @ B0)
    C1 = (A1 @ B1) + (A1 @ B2) + (A2 @ B1)
    C2 = (A2 @ B2) + (A2 @ B0) + (A0 @ B2)

    # Cache is per-party: store the exact RSS pair that party will hold.
    from uvcc_party.tcf import _tcf_cache_pack_pair_v1

    cache0 = {tile_id32: _tcf_cache_pack_pair_v1(tile_id32=tile_id32, d=d, fxp_frac_bits=fxp, c_lo=C0, c_hi=C1)}
    cache1 = {tile_id32: _tcf_cache_pack_pair_v1(tile_id32=tile_id32, d=d, fxp_frac_bits=fxp, c_lo=C1, c_hi=C2)}
    cache2 = {tile_id32: _tcf_cache_pack_pair_v1(tile_id32=tile_id32, d=d, fxp_frac_bits=fxp, c_lo=C2, c_hi=C0)}

    job_id32 = b"\x22" * 32
    p0 = Party(party_id=0, job_id32=job_id32, sid=sid, relay=_NullRelay())  # type: ignore[arg-type]
    p1 = Party(party_id=1, job_id32=job_id32, sid=sid, relay=_NullRelay())  # type: ignore[arg-type]
    p2 = Party(party_id=2, job_id32=job_id32, sid=sid, relay=_NullRelay())  # type: ignore[arg-type]

    A0p, B0p, C0p = tcf_eval_v0b_tile_u64_v1(
        p0, key=k0, op_id=op_id, i=i, j=j, p=p, epoch=0, step=0, round=0, fxp_frac_bits=fxp, d=d, cache=cache0, allow_repl_on_miss=False
    )
    A1p, B1p, C1p = tcf_eval_v0b_tile_u64_v1(
        p1, key=k1, op_id=op_id, i=i, j=j, p=p, epoch=0, step=0, round=0, fxp_frac_bits=fxp, d=d, cache=cache1, allow_repl_on_miss=False
    )
    A2p, B2p, C2p = tcf_eval_v0b_tile_u64_v1(
        p2, key=k2, op_id=op_id, i=i, j=j, p=p, epoch=0, step=0, round=0, fxp_frac_bits=fxp, d=d, cache=cache2, allow_repl_on_miss=False
    )

    # Reconstruct publics and check C == A@B in Z2^64 (wraparound int64).
    A_pub = _u64(A0p.lo + A0p.hi + A1p.hi)
    B_pub = _u64(B0p.lo + B0p.hi + B1p.hi)
    C_pub = _u64(C0p.lo + C0p.hi + C1p.hi)
    assert torch.equal(C_pub, _u64(A_pub @ B_pub))


