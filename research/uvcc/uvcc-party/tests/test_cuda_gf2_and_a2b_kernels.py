from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b4712f8a9200c638,uvcc_group_edc7705141457666,uvcc_group_cb38774080e4a4d3,uvcc_group_10df23d70db0500b

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_gf2_and_prepare_finish_pack32_matches_cpu() -> None:
    from uvcc_party.cuda_ext import gf2_and_finish_pack32, gf2_and_prepare_pack32

    gen = torch.Generator(device="cpu").manual_seed(123)
    W = 257  # covers full+rem launch path

    def r_u32():
        return torch.randint(0, 2**32, (W,), dtype=torch.int64, generator=gen).to(torch.int32)

    x_lo = r_u32()
    x_hi = r_u32()
    y_lo = r_u32()
    y_hi = r_u32()
    a_lo = r_u32()
    a_hi = r_u32()
    b_lo = r_u32()
    b_hi = r_u32()
    c_lo = r_u32()
    c_hi = r_u32()
    e_pub = r_u32()
    f_pub = r_u32()

    # Prepare CPU
    e_lo_exp = x_lo ^ a_lo
    f_lo_exp = y_lo ^ b_lo
    e_hi_exp = x_hi ^ a_hi
    f_hi_exp = y_hi ^ b_hi

    # GPU prepare
    e_lo_g, f_lo_g, e_hi_g, f_hi_g = gf2_and_prepare_pack32(
        x_lo_i32=x_lo.cuda(),
        x_hi_i32=x_hi.cuda(),
        y_lo_i32=y_lo.cuda(),
        y_hi_i32=y_hi.cuda(),
        a_lo_i32=a_lo.cuda(),
        a_hi_i32=a_hi.cuda(),
        b_lo_i32=b_lo.cuda(),
        b_hi_i32=b_hi.cuda(),
    )
    assert torch.equal(e_lo_g.cpu(), e_lo_exp)
    assert torch.equal(f_lo_g.cpu(), f_lo_exp)
    assert torch.equal(e_hi_g.cpu(), e_hi_exp)
    assert torch.equal(f_hi_g.cpu(), f_hi_exp)

    for pid in (0, 1, 2):
        # Finish CPU
        term_eb_lo = b_lo & e_pub
        term_eb_hi = b_hi & e_pub
        term_fa_lo = a_lo & f_pub
        term_fa_hi = a_hi & f_pub
        ef = e_pub & f_pub
        z_lo = c_lo ^ term_eb_lo ^ term_fa_lo
        z_hi = c_hi ^ term_eb_hi ^ term_fa_hi
        if pid == 0:
            z_lo = z_lo ^ ef
        if pid == 2:
            z_hi = z_hi ^ ef

        z_lo_g, z_hi_g = gf2_and_finish_pack32(
            a_lo_i32=a_lo.cuda(),
            a_hi_i32=a_hi.cuda(),
            b_lo_i32=b_lo.cuda(),
            b_hi_i32=b_hi.cuda(),
            c_lo_i32=c_lo.cuda(),
            c_hi_i32=c_hi.cuda(),
            e_pub_i32=e_pub.cuda(),
            f_pub_i32=f_pub.cuda(),
            party_id=pid,
        )
        assert torch.equal(z_lo_g.cpu(), z_lo)
        assert torch.equal(z_hi_g.cpu(), z_hi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_a2b_sub_prepare_finish_pack32_matches_cpu() -> None:
    from uvcc_party.cuda_ext import a2b_sub_finish_pack32, a2b_sub_prepare_pack32

    gen = torch.Generator(device="cpu").manual_seed(999)
    W = 513  # covers multi-block + remainder

    def r_u32():
        return torch.randint(0, 2**32, (W,), dtype=torch.int64, generator=gen).to(torch.int32)

    rj_lo = r_u32()
    rj_hi = r_u32()
    bj_lo = r_u32()
    bj_hi = r_u32()
    aj_lo = r_u32()
    aj_hi = r_u32()
    bjT_lo = r_u32()
    bjT_hi = r_u32()
    cj_lo = r_u32()
    cj_hi = r_u32()
    e_pub = r_u32()
    f_pub = r_u32()
    cj_mask = r_u32()

    # Prepare CPU
    e_lo = rj_lo ^ aj_lo
    e_hi = rj_hi ^ aj_hi
    f_lo = bj_lo ^ bjT_lo
    f_hi = bj_hi ^ bjT_hi

    e_lo_g, f_lo_g, e_hi_g, f_hi_g = a2b_sub_prepare_pack32(
        rj_lo_i32=rj_lo.cuda(),
        rj_hi_i32=rj_hi.cuda(),
        bj_lo_i32=bj_lo.cuda(),
        bj_hi_i32=bj_hi.cuda(),
        aj_lo_i32=aj_lo.cuda(),
        aj_hi_i32=aj_hi.cuda(),
        bjT_lo_i32=bjT_lo.cuda(),
        bjT_hi_i32=bjT_hi.cuda(),
    )
    assert torch.equal(e_lo_g.cpu(), e_lo)
    assert torch.equal(f_lo_g.cpu(), f_lo)
    assert torch.equal(e_hi_g.cpu(), e_hi)
    assert torch.equal(f_hi_g.cpu(), f_hi)

    for pid in (0, 1, 2):
        # Finish CPU
        g_lo = cj_lo ^ (bjT_lo & e_pub) ^ (aj_lo & f_pub)
        g_hi = cj_hi ^ (bjT_hi & e_pub) ^ (aj_hi & f_pub)
        ef = e_pub & f_pub
        if pid == 0:
            g_lo = g_lo ^ ef
        if pid == 2:
            g_hi = g_hi ^ ef

        t_lo = rj_lo ^ bj_lo
        t_hi = rj_hi ^ bj_hi

        x_lo = t_lo.clone()
        x_hi = t_hi.clone()
        if pid == 0:
            x_lo = x_lo ^ cj_mask
        if pid == 2:
            x_hi = x_hi ^ cj_mask

        mask0 = ~cj_mask
        bnext_lo = g_lo ^ (mask0 & t_lo)
        bnext_hi = g_hi ^ (mask0 & t_hi)

        x_lo_g, x_hi_g, bnext_lo_g, bnext_hi_g = a2b_sub_finish_pack32(
            rj_lo_i32=rj_lo.cuda(),
            rj_hi_i32=rj_hi.cuda(),
            bj_lo_i32=bj_lo.cuda(),
            bj_hi_i32=bj_hi.cuda(),
            aj_lo_i32=aj_lo.cuda(),
            aj_hi_i32=aj_hi.cuda(),
            bjT_lo_i32=bjT_lo.cuda(),
            bjT_hi_i32=bjT_hi.cuda(),
            cj_lo_i32=cj_lo.cuda(),
            cj_hi_i32=cj_hi.cuda(),
            e_pub_i32=e_pub.cuda(),
            f_pub_i32=f_pub.cuda(),
            cj_public_mask_i32=cj_mask.cuda(),
            party_id=pid,
        )

        assert torch.equal(x_lo_g.cpu(), x_lo)
        assert torch.equal(x_hi_g.cpu(), x_hi)
        assert torch.equal(bnext_lo_g.cpu(), bnext_lo)
        assert torch.equal(bnext_hi_g.cpu(), bnext_hi)


