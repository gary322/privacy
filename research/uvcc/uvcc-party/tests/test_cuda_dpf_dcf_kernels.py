from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_28df78cb5ca5e330,uvcc_group_b2a809ccbb581fc9,uvcc_group_e63d36190f1cb57e
# UVCC_REQ_GROUP: uvcc_group_e4c33796e7d46452,uvcc_group_bfa55eababdc7ee8,uvcc_group_37bb78042819626f

import pytest
import torch

from uvcc_party.dpf_dcf import (
    PRG_AES128,
    PRG_CHACHA12,
    PRIM_DCF,
    dcf_full_w8_v1,
    dcf_stage2_w16_v1,
    dpf_stage1_w16_v1,
    keygen_dpf_dcf_keyrecs_v1,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_dcf_full_w8_matches_cpu_reference() -> None:
    from uvcc_party.cuda_ext import dcf_full_w8

    sid = b"sid-cuda"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    fss_id = 0x1111222233334444
    w = 8
    alpha = 123
    key0, key1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=fss_id,
        alpha=alpha,
        w=w,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x22" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    cpu0 = dcf_full_w8_v1(key0, device=torch.device("cpu"))
    cpu1 = dcf_full_w8_v1(key1, device=torch.device("cpu"))
    key0_t = torch.tensor(list(key0), dtype=torch.uint8, device="cuda")
    key1_t = torch.tensor(list(key1), dtype=torch.uint8, device="cuda")
    gpu0 = dcf_full_w8(key0_t).cpu()
    gpu1 = dcf_full_w8(key1_t).cpu()

    assert torch.equal(gpu0, cpu0)
    assert torch.equal(gpu1, cpu1)

    # Functional check: XOR the two party outputs to get the public DCF bit vector.
    pub = gpu0 ^ gpu1
    expect = (torch.arange(256, dtype=torch.int64) < alpha).to(torch.int64)
    assert torch.equal(pub, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_dcf_full_w8_matches_cpu_reference_aes() -> None:
    from uvcc_party.cuda_ext import dcf_full_w8

    sid = b"sid-cuda-aes"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    fss_id = 0x1111222233334444
    w = 8
    alpha = 123
    key0, key1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=fss_id,
        alpha=alpha,
        w=w,
        prg_id=PRG_AES128,
        party_edge=0,
        master_seed32=b"\x22" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    cpu0 = dcf_full_w8_v1(key0, device=torch.device("cpu"))
    cpu1 = dcf_full_w8_v1(key1, device=torch.device("cpu"))
    key0_t = torch.tensor(list(key0), dtype=torch.uint8, device="cuda")
    key1_t = torch.tensor(list(key1), dtype=torch.uint8, device="cuda")
    gpu0 = dcf_full_w8(key0_t).cpu()
    gpu1 = dcf_full_w8(key1_t).cpu()

    assert torch.equal(gpu0, cpu0)
    assert torch.equal(gpu1, cpu1)

    pub = gpu0 ^ gpu1
    expect = (torch.arange(256, dtype=torch.int64) < alpha).to(torch.int64)
    assert torch.equal(pub, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_stage1_stage2_w16_matches_cpu_reference() -> None:
    from uvcc_party.cuda_ext import dcf_stage2_w16, dpf_stage1_w16

    sid = b"sid-cuda-w16"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    fss_id = 0xABCDEF0123456789
    w = 16
    alpha = 1337
    key0, key1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=fss_id,
        alpha=alpha,
        w=w,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x33" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    cpu0_front = dpf_stage1_w16_v1(key0, device=torch.device("cpu"))
    cpu1_front = dpf_stage1_w16_v1(key1, device=torch.device("cpu"))

    key0_t = torch.tensor(list(key0), dtype=torch.uint8, device="cuda")
    key1_t = torch.tensor(list(key1), dtype=torch.uint8, device="cuda")

    g0_seed_lo, g0_seed_hi, g0_t, g0_acc = dpf_stage1_w16(key0_t)
    g1_seed_lo, g1_seed_hi, g1_t, g1_acc = dpf_stage1_w16(key1_t)

    # Compare stage1 outputs (cast u8 -> i64 where needed)
    assert torch.equal(g0_seed_lo.cpu(), cpu0_front[0])
    assert torch.equal(g0_seed_hi.cpu(), cpu0_front[1])
    assert torch.equal(g0_t.cpu().to(torch.int64), cpu0_front[2])
    assert torch.equal(g0_acc.cpu().to(torch.int64), cpu0_front[3])

    assert torch.equal(g1_seed_lo.cpu(), cpu1_front[0])
    assert torch.equal(g1_seed_hi.cpu(), cpu1_front[1])
    assert torch.equal(g1_t.cpu().to(torch.int64), cpu1_front[2])
    assert torch.equal(g1_acc.cpu().to(torch.int64), cpu1_front[3])

    # Stage2 outputs
    cpu0_out = dcf_stage2_w16_v1(
        key0,
        frontier_seed_lo=cpu0_front[0],
        frontier_seed_hi=cpu0_front[1],
        frontier_t=cpu0_front[2],
        frontier_acc=cpu0_front[3],
        device=torch.device("cpu"),
    )
    cpu1_out = dcf_stage2_w16_v1(
        key1,
        frontier_seed_lo=cpu1_front[0],
        frontier_seed_hi=cpu1_front[1],
        frontier_t=cpu1_front[2],
        frontier_acc=cpu1_front[3],
        device=torch.device("cpu"),
    )

    gpu0_out = dcf_stage2_w16(key0_t, g0_seed_lo, g0_seed_hi, g0_t, g0_acc).cpu()
    gpu1_out = dcf_stage2_w16(key1_t, g1_seed_lo, g1_seed_hi, g1_t, g1_acc).cpu()

    assert torch.equal(gpu0_out, cpu0_out)
    assert torch.equal(gpu1_out, cpu1_out)

    # Spot functional check via XOR of party outputs at a few points.
    pub = gpu0_out ^ gpu1_out
    for u in (0, 1, alpha - 1, alpha, alpha + 1, 65535):
        u = int(u) & 0xFFFF
        expect = 1 if u < alpha else 0
        assert int(pub[u].item()) == expect


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_stage1_stage2_w16_matches_cpu_reference_aes() -> None:
    from uvcc_party.cuda_ext import dcf_stage2_w16, dpf_stage1_w16

    sid = b"sid-cuda-w16-aes"
    sid_hash32 = __import__("hashlib").sha256(sid).digest()
    fss_id = 0xABCDEF0123456789
    w = 16
    alpha = 1337
    key0, key1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=fss_id,
        alpha=alpha,
        w=w,
        prg_id=PRG_AES128,
        party_edge=0,
        master_seed32=b"\x33" * 32,
        prim_type=PRIM_DCF,
        dcf_invert=True,
        payload_mask_u64=1,
    )

    cpu0_front = dpf_stage1_w16_v1(key0, device=torch.device("cpu"))
    cpu1_front = dpf_stage1_w16_v1(key1, device=torch.device("cpu"))

    key0_t = torch.tensor(list(key0), dtype=torch.uint8, device="cuda")
    key1_t = torch.tensor(list(key1), dtype=torch.uint8, device="cuda")

    g0_seed_lo, g0_seed_hi, g0_t, g0_acc = dpf_stage1_w16(key0_t)
    g1_seed_lo, g1_seed_hi, g1_t, g1_acc = dpf_stage1_w16(key1_t)

    assert torch.equal(g0_seed_lo.cpu(), cpu0_front[0])
    assert torch.equal(g0_seed_hi.cpu(), cpu0_front[1])
    assert torch.equal(g0_t.cpu().to(torch.int64), cpu0_front[2])
    assert torch.equal(g0_acc.cpu().to(torch.int64), cpu0_front[3])

    assert torch.equal(g1_seed_lo.cpu(), cpu1_front[0])
    assert torch.equal(g1_seed_hi.cpu(), cpu1_front[1])
    assert torch.equal(g1_t.cpu().to(torch.int64), cpu1_front[2])
    assert torch.equal(g1_acc.cpu().to(torch.int64), cpu1_front[3])

    cpu0_out = dcf_stage2_w16_v1(
        key0,
        frontier_seed_lo=cpu0_front[0],
        frontier_seed_hi=cpu0_front[1],
        frontier_t=cpu0_front[2],
        frontier_acc=cpu0_front[3],
        device=torch.device("cpu"),
    )
    cpu1_out = dcf_stage2_w16_v1(
        key1,
        frontier_seed_lo=cpu1_front[0],
        frontier_seed_hi=cpu1_front[1],
        frontier_t=cpu1_front[2],
        frontier_acc=cpu1_front[3],
        device=torch.device("cpu"),
    )

    gpu0_out = dcf_stage2_w16(key0_t, g0_seed_lo, g0_seed_hi, g0_t, g0_acc).cpu()
    gpu1_out = dcf_stage2_w16(key1_t, g1_seed_lo, g1_seed_hi, g1_t, g1_acc).cpu()

    assert torch.equal(gpu0_out, cpu0_out)
    assert torch.equal(gpu1_out, cpu1_out)

    pub = gpu0_out ^ gpu1_out
    for u in (0, 1, alpha - 1, alpha, alpha + 1, 65535):
        u = int(u) & 0xFFFF
        expect = 1 if u < alpha else 0
        assert int(pub[u].item()) == expect


