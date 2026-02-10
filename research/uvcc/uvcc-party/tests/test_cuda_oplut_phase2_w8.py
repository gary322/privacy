from __future__ import annotations

import pytest
import torch

from uvcc_party.op_lut import op_lut_phase2_local_cpu_v1
from uvcc_party.op_lut_blob import build_oplut_record_blobs_v1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_oplut_phase2_w8_record_matches_cpu() -> None:
    from uvcc_party.cuda_ext import oplut_phase2_w8_record

    sid = b"sid-oplut-cuda"
    fss_id = 0xABCDEF0123456789
    sgir_op_id = 7
    lanes = 4

    p0_blob, p1_blob, p2_blob = build_oplut_record_blobs_v1(
        sid=sid,
        fss_id=fss_id,
        sgir_op_id=sgir_op_id,
        domain_w=8,
        lanes=lanes,
        lane_base=0,
        K_master32=b"\x11" * 32,
        seed_edge01_32=b"\x22" * 32,
        seed_edge12_32=b"\x33" * 32,
        seed_edge20_32=b"\x44" * 32,
    )
    blobs = [p0_blob, p1_blob, p2_blob]

    gen = torch.Generator(device="cpu").manual_seed(2031)
    u_pub = torch.randint(0, 256, (lanes,), dtype=torch.int64, generator=gen).to(torch.uint8)
    table = torch.randint(0, 2**16, (256,), dtype=torch.int64, generator=gen)

    table_cuda = table.to(device="cuda", dtype=torch.int64).contiguous()
    u_cuda = u_pub.to(device="cuda", dtype=torch.uint8).contiguous()

    for blob in blobs:
        cpu = op_lut_phase2_local_cpu_v1(fss_blob=blob, u_pub_u8=u_pub, table_u64=table).contiguous()
        rec_cuda = torch.tensor(list(blob), dtype=torch.uint8, device="cuda").contiguous()
        gpu = oplut_phase2_w8_record(rec_cuda, u_cuda, table_cuda).cpu()
        assert torch.equal(gpu, cpu)


