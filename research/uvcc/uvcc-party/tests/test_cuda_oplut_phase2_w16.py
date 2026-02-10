from __future__ import annotations

import pytest
import torch

from uvcc_party.dpf_dcf import chacha20_block_bytes_v1
from uvcc_party.op_lut_blob import oplut_nonce_r12_v1, oplut_salt16_v1
from uvcc_party.op_lut_blob import build_oplut_record_blobs_v1


def _u16_to_i16(u: torch.Tensor) -> torch.Tensor:
    x = u.to(torch.int64).clone()
    x[x >= 32768] -= 65536
    return x.to(torch.int16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_oplut_phase2_w16_reconstructs_table_lookup() -> None:
    from uvcc_party.cuda_ext import oplut_phase2_w16_record

    sid = b"sid-oplut-cuda-w16"
    fss_id = 0xABCDEF0123456789
    sgir_op_id = 7
    lanes = 2
    lane_base = 0
    counter0 = 0
    lane_stride = 1

    K_master32 = b"\x11" * 32
    seed_edge01_32 = b"\x22" * 32
    seed_edge12_32 = b"\x33" * 32
    seed_edge20_32 = b"\x44" * 32

    p0_blob, p1_blob, p2_blob = build_oplut_record_blobs_v1(
        sid=sid,
        fss_id=fss_id,
        sgir_op_id=sgir_op_id,
        domain_w=16,
        lanes=lanes,
        lane_base=lane_base,
        K_master32=K_master32,
        seed_edge01_32=seed_edge01_32,
        seed_edge12_32=seed_edge12_32,
        seed_edge20_32=seed_edge20_32,
        counter0=counter0,
        lane_stride=lane_stride,
    )
    blobs = [p0_blob, p1_blob, p2_blob]

    gen = torch.Generator(device="cpu").manual_seed(2037)
    u_pub_u16 = torch.randint(0, 65536, (lanes,), dtype=torch.int64, generator=gen).to(torch.int64)
    u_pub_i16 = _u16_to_i16(u_pub_u16)

    # Public table (ring u64 values stored as int64 bit-patterns).
    table = (torch.arange(65536, dtype=torch.int64) * 3 + 1).to(torch.int64)
    table_cuda = table.to(device="cuda", dtype=torch.int64).contiguous()
    u_cuda_i16 = u_pub_i16.to(device="cuda", dtype=torch.int16).contiguous()

    outs = []
    for blob in blobs:
        rec_cuda = torch.tensor(list(blob), dtype=torch.uint8, device="cuda").contiguous()
        outs.append(oplut_phase2_w16_record(rec_cuda, u_cuda_i16, table_cuda).cpu())

    # Reconstruct: sum across parties (refresh masks cancel).
    y_pub = (outs[0] + outs[1] + outs[2]).to(torch.int64)

    # Compute expected y = T[x] where x = u - r mod 2^16, and r is derived from the three edge seeds.
    salt16 = oplut_salt16_v1(sid=sid, fss_id=fss_id, sgir_op_id=sgir_op_id)
    nonce_r12 = oplut_nonce_r12_v1(salt16=salt16)
    maskN = 0xFFFF
    r_vals = []
    for ell in range(lanes):
        ctr = int(counter0) + (int(lane_base) + ell) * int(lane_stride)
        blk20 = chacha20_block_bytes_v1(key32=seed_edge20_32, nonce12=nonce_r12, counter32=int(ctr) & 0xFFFFFFFF)
        blk01 = chacha20_block_bytes_v1(key32=seed_edge01_32, nonce12=nonce_r12, counter32=int(ctr) & 0xFFFFFFFF)
        blk12 = chacha20_block_bytes_v1(key32=seed_edge12_32, nonce12=nonce_r12, counter32=int(ctr) & 0xFFFFFFFF)
        c0 = int.from_bytes(blk20[0:2], "little", signed=False) & maskN
        c1 = int.from_bytes(blk01[0:2], "little", signed=False) & maskN
        c2 = int.from_bytes(blk12[0:2], "little", signed=False) & maskN
        r_vals.append(int((c0 + c1 + c2) & maskN))
    r = torch.tensor(r_vals, dtype=torch.int64)
    x = (u_pub_u16 - r) & maskN
    expect = table[x]
    assert torch.equal(y_pub, expect)


