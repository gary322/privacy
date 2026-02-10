from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_f1817a0260a2d9bb

from uvcc_party.dpf_aes128 import eval_dpf_point_aes128_ar64_v1, keygen_dpf_aes128_ar64_v1


def test_dpf_aes128_ar64_v1_point_function_w8_beta1() -> None:
    K_master32 = b"\x01" * 32
    sid32 = b"\x02" * 32
    fss_id = 0x1111222233334444
    w = 8
    alpha = 77
    beta = 1
    k0, k1 = keygen_dpf_aes128_ar64_v1(K_master32=K_master32, sid32=sid32, fss_id=fss_id, w=w, alpha=alpha, beta_u64=beta)
    for x in range(256):
        y0 = eval_dpf_point_aes128_ar64_v1(key_bytes=k0, w=w, x=x, party_b=0)
        y1 = eval_dpf_point_aes128_ar64_v1(key_bytes=k1, w=w, x=x, party_b=1)
        s = (int(y0) + int(y1)) & 0xFFFFFFFFFFFFFFFF
        exp = beta if x == alpha else 0
        assert s == exp


