from __future__ import annotations

import hashlib

import pytest
import torch

from uvcc_party import dpf_dcf
from uvcc_party.cuda_ext import (
    dcf_eval_point_w16_batch,
    dcf_eval_point_w8_batch,
    dpf_eval_point_w16_batch,
    dpf_eval_point_w8_batch,
)


def _u16_to_i16_bitpattern(u_u16: torch.Tensor) -> torch.Tensor:
    u = u_u16.to(torch.int32)
    u = torch.where(u >= (1 << 15), u - (1 << 16), u)
    return u.to(torch.int16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_point_eval_dpf_dcf_matches_cpu() -> None:
    sid = b"sid-point-eval"
    sid_hash32 = hashlib.sha256(sid).digest()
    master_seed32 = b"\x11" * 32

    # Keep this small; CUDA extension compilation dominates anyway.
    N = 8
    gen = torch.Generator(device="cpu").manual_seed(1234)

    # ---- DPF w=8 ----
    keyrecs0: list[bytes] = []
    keyrecs1: list[bytes] = []
    u_list: list[int] = []
    for i in range(N):
        alpha = int(torch.randint(0, 256, (1,), generator=gen).item())
        u = int(torch.randint(0, 256, (1,), generator=gen).item())
        k0, k1 = dpf_dcf.keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=0x1000 + i,
            alpha=alpha,
            w=8,
            prg_id=dpf_dcf.PRG_CHACHA12,
            party_edge=dpf_dcf.EDGE_01,
            master_seed32=master_seed32,
            prim_type=dpf_dcf.PRIM_DPF,
        )
        keyrecs0.append(k0)
        keyrecs1.append(k1)
        u_list.append(u)

    key_bytes = len(keyrecs0[0])
    assert all(len(k) == key_bytes for k in keyrecs0)
    blob0 = torch.tensor(list(b"".join(keyrecs0)), dtype=torch.uint8, device="cuda")
    blob1 = torch.tensor(list(b"".join(keyrecs1)), dtype=torch.uint8, device="cuda")
    u_u16 = torch.tensor(u_list, dtype=torch.int64)
    u_i16 = _u16_to_i16_bitpattern(u_u16).to(device="cuda")

    out0 = dpf_eval_point_w8_batch(keyrecs_blob_u8=blob0, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    out1 = dpf_eval_point_w8_batch(keyrecs_blob_u8=blob1, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    for i in range(N):
        exp0 = dpf_dcf.dpf_eval_point_bit_v1(keyrecs0[i], int(u_list[i]))
        exp1 = dpf_dcf.dpf_eval_point_bit_v1(keyrecs1[i], int(u_list[i]))
        assert int(out0[i].item()) == exp0
        assert int(out1[i].item()) == exp1

    # ---- DPF w=16 ----
    keyrecs0 = []
    keyrecs1 = []
    u_list = []
    for i in range(N):
        alpha = int(torch.randint(0, 1 << 16, (1,), generator=gen).item())
        u = int(torch.randint(0, 1 << 16, (1,), generator=gen).item())
        k0, k1 = dpf_dcf.keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=0x2000 + i,
            alpha=alpha,
            w=16,
            prg_id=dpf_dcf.PRG_CHACHA12,
            party_edge=dpf_dcf.EDGE_01,
            master_seed32=master_seed32,
            prim_type=dpf_dcf.PRIM_DPF,
        )
        keyrecs0.append(k0)
        keyrecs1.append(k1)
        u_list.append(u)

    key_bytes = len(keyrecs0[0])
    assert all(len(k) == key_bytes for k in keyrecs0)
    blob0 = torch.tensor(list(b"".join(keyrecs0)), dtype=torch.uint8, device="cuda")
    blob1 = torch.tensor(list(b"".join(keyrecs1)), dtype=torch.uint8, device="cuda")
    u_u16 = torch.tensor(u_list, dtype=torch.int64)
    u_i16 = _u16_to_i16_bitpattern(u_u16).to(device="cuda")

    out0 = dpf_eval_point_w16_batch(keyrecs_blob_u8=blob0, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    out1 = dpf_eval_point_w16_batch(keyrecs_blob_u8=blob1, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    for i in range(N):
        exp0 = dpf_dcf.dpf_eval_point_bit_v1(keyrecs0[i], int(u_list[i]))
        exp1 = dpf_dcf.dpf_eval_point_bit_v1(keyrecs1[i], int(u_list[i]))
        assert int(out0[i].item()) == exp0
        assert int(out1[i].item()) == exp1

    # ---- DCF w=8 ----
    keyrecs0 = []
    keyrecs1 = []
    u_list = []
    for i in range(N):
        alpha = int(torch.randint(0, 256, (1,), generator=gen).item())
        u = int(torch.randint(0, 256, (1,), generator=gen).item())
        k0, k1 = dpf_dcf.keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=0x3000 + i,
            alpha=alpha,
            w=8,
            prg_id=dpf_dcf.PRG_CHACHA12,
            party_edge=dpf_dcf.EDGE_01,
            master_seed32=master_seed32,
            prim_type=dpf_dcf.PRIM_DCF,
            dcf_invert=True,
            payload_mask_u64=1,
        )
        keyrecs0.append(k0)
        keyrecs1.append(k1)
        u_list.append(u)

    key_bytes = len(keyrecs0[0])
    blob0 = torch.tensor(list(b"".join(keyrecs0)), dtype=torch.uint8, device="cuda")
    blob1 = torch.tensor(list(b"".join(keyrecs1)), dtype=torch.uint8, device="cuda")
    u_u16 = torch.tensor(u_list, dtype=torch.int64)
    u_i16 = _u16_to_i16_bitpattern(u_u16).to(device="cuda")

    out0 = dcf_eval_point_w8_batch(keyrecs_blob_u8=blob0, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    out1 = dcf_eval_point_w8_batch(keyrecs_blob_u8=blob1, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    for i in range(N):
        exp0 = dpf_dcf.dcf_eval_point_bit_w8_v1(keyrecs0[i], int(u_list[i]))
        exp1 = dpf_dcf.dcf_eval_point_bit_w8_v1(keyrecs1[i], int(u_list[i]))
        assert int(out0[i].item()) == exp0
        assert int(out1[i].item()) == exp1

    # ---- DCF w=16 ----
    keyrecs0 = []
    keyrecs1 = []
    u_list = []
    for i in range(N):
        alpha = int(torch.randint(0, 1 << 16, (1,), generator=gen).item())
        u = int(torch.randint(0, 1 << 16, (1,), generator=gen).item())
        k0, k1 = dpf_dcf.keygen_dpf_dcf_keyrecs_v1(
            sid=sid,
            sid_hash32=sid_hash32,
            fss_id=0x4000 + i,
            alpha=alpha,
            w=16,
            prg_id=dpf_dcf.PRG_CHACHA12,
            party_edge=dpf_dcf.EDGE_01,
            master_seed32=master_seed32,
            prim_type=dpf_dcf.PRIM_DCF,
            dcf_invert=True,
            payload_mask_u64=1,
        )
        keyrecs0.append(k0)
        keyrecs1.append(k1)
        u_list.append(u)

    key_bytes = len(keyrecs0[0])
    blob0 = torch.tensor(list(b"".join(keyrecs0)), dtype=torch.uint8, device="cuda")
    blob1 = torch.tensor(list(b"".join(keyrecs1)), dtype=torch.uint8, device="cuda")
    u_u16 = torch.tensor(u_list, dtype=torch.int64)
    u_i16 = _u16_to_i16_bitpattern(u_u16).to(device="cuda")

    out0 = dcf_eval_point_w16_batch(keyrecs_blob_u8=blob0, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    out1 = dcf_eval_point_w16_batch(keyrecs_blob_u8=blob1, key_stride_bytes=key_bytes, x_pub_u16_i16=u_i16).cpu()
    for i in range(N):
        exp0 = dpf_dcf.dcf_eval_point_bit_w16_v1(keyrecs0[i], int(u_list[i]))
        exp1 = dpf_dcf.dcf_eval_point_bit_w16_v1(keyrecs1[i], int(u_list[i]))
        assert int(out0[i].item()) == exp0
        assert int(out1[i].item()) == exp1


