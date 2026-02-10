from __future__ import annotations

# pyright: reportMissingImports=false
# UVCC_REQ_GROUP: uvcc_group_e5ab5bae0f6f1790,uvcc_group_2c46210f0240cf5f,uvcc_group_c66222b84339eca1,uvcc_group_22930b2f16ac685e,uvcc_group_acaeca358a192ce8,uvcc_group_65ed671ff3fd69f5,uvcc_group_e6c2716a57857b5f,uvcc_group_131735f7a41346a4,uvcc_group_b2a809ccbb581fc9,uvcc_group_f1817a0260a2d9bb

import hashlib
import random

import torch

from uvcc_party.dpf_dcf import (
    PRG_CHACHA12,
    PRIM_DCF,
    PRIM_DPF,
    dcf_eval_point_bit_w16_v1,
    dcf_eval_point_bit_w8_v1,
    dcf_full_w8_v1,
    dcf_stage2_w16_v1,
    dpf_eval_point_bit_v1,
    dpf_full_w8_v1,
    dpf_stage1_w16_v1,
    dpf_stage2_w16_v1,
    keygen_dpf_dcf_keyrecs_v1,
)


def _sid_hash32(sid: bytes) -> bytes:
    return hashlib.sha256(sid).digest()


def test_point_eval_matches_full_domain_w8() -> None:
    sid = b"sid-point-eval-w8"
    sid_hash32 = _sid_hash32(sid)
    rng = random.Random(123)

    # DPF bit
    k0, k1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x1111,
        alpha=77,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x11" * 32,
        prim_type=PRIM_DPF,
    )
    full0 = dpf_full_w8_v1(k0, device=torch.device("cpu"))
    full1 = dpf_full_w8_v1(k1, device=torch.device("cpu"))
    for _ in range(25):
        u = rng.randrange(0, 256)
        assert dpf_eval_point_bit_v1(k0, u=u, device=torch.device("cpu")) == (int(full0[u].item()) & 1)
        assert dpf_eval_point_bit_v1(k1, u=u, device=torch.device("cpu")) == (int(full1[u].item()) & 1)

    # DCF bit (payload mask is 1 by default)
    k0c, k1c = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x2222,
        alpha=91,
        w=8,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x22" * 32,
        prim_type=PRIM_DCF,
    )
    full0c = dcf_full_w8_v1(k0c, device=torch.device("cpu"))
    full1c = dcf_full_w8_v1(k1c, device=torch.device("cpu"))
    for _ in range(25):
        u = rng.randrange(0, 256)
        assert dcf_eval_point_bit_w8_v1(k0c, u=u, device=torch.device("cpu")) == int(full0c[u].item() != 0)
        assert dcf_eval_point_bit_w8_v1(k1c, u=u, device=torch.device("cpu")) == int(full1c[u].item() != 0)


def test_point_eval_matches_full_domain_w16() -> None:
    sid = b"sid-point-eval-w16"
    sid_hash32 = _sid_hash32(sid)
    rng = random.Random(456)

    # DPF bit
    k0, k1 = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x3333,
        alpha=0x1234,
        w=16,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x33" * 32,
        prim_type=PRIM_DPF,
    )
    front0 = dpf_stage1_w16_v1(k0, device=torch.device("cpu"))
    full0 = dpf_stage2_w16_v1(k0, frontier_seed_lo=front0[0], frontier_seed_hi=front0[1], frontier_t=front0[2], device=torch.device("cpu"))
    front1 = dpf_stage1_w16_v1(k1, device=torch.device("cpu"))
    full1 = dpf_stage2_w16_v1(k1, frontier_seed_lo=front1[0], frontier_seed_hi=front1[1], frontier_t=front1[2], device=torch.device("cpu"))
    for _ in range(20):
        u = rng.randrange(0, 65536)
        assert dpf_eval_point_bit_v1(k0, u=u, device=torch.device("cpu")) == (int(full0[u].item()) & 1)
        assert dpf_eval_point_bit_v1(k1, u=u, device=torch.device("cpu")) == (int(full1[u].item()) & 1)

    # DCF bit
    k0c, k1c = keygen_dpf_dcf_keyrecs_v1(
        sid=sid,
        sid_hash32=sid_hash32,
        fss_id=0x4444,
        alpha=0xBEEF,
        w=16,
        prg_id=PRG_CHACHA12,
        party_edge=0,
        master_seed32=b"\x44" * 32,
        prim_type=PRIM_DCF,
    )
    front0c = dpf_stage1_w16_v1(k0c, device=torch.device("cpu"))
    full0c = dcf_stage2_w16_v1(
        k0c,
        frontier_seed_lo=front0c[0],
        frontier_seed_hi=front0c[1],
        frontier_t=front0c[2],
        frontier_acc=front0c[3],
        device=torch.device("cpu"),
    )
    front1c = dpf_stage1_w16_v1(k1c, device=torch.device("cpu"))
    full1c = dcf_stage2_w16_v1(
        k1c,
        frontier_seed_lo=front1c[0],
        frontier_seed_hi=front1c[1],
        frontier_t=front1c[2],
        frontier_acc=front1c[3],
        device=torch.device("cpu"),
    )
    for _ in range(20):
        u = rng.randrange(0, 65536)
        assert dcf_eval_point_bit_w16_v1(k0c, u=u, device=torch.device("cpu")) == int(full0c[u].item() != 0)
        assert dcf_eval_point_bit_w16_v1(k1c, u=u, device=torch.device("cpu")) == int(full1c[u].item() != 0)


