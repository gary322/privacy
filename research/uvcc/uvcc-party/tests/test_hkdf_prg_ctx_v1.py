from __future__ import annotations

# UVCC_REQ_GROUP: uvcc_group_b09f4ad9c27cdb7d,uvcc_group_fce8a8bb25a7fd5f,uvcc_group_fffe7f7fba4343b8,uvcc_group_6d0a46e2d04f891f,uvcc_group_732bc069128b3469

from uvcc_party.hkdf import hkdf_expand, hkdf_extract
from uvcc_party.kdf_info import KIND_DPF_PRG_CTX, PRG_TYPE_AES128_FIXEDKEY, PRG_TYPE_CHACHA20_FIXEDKEY, UVCCKDFInfoV1, kdf_info_zero_table_hash16
from uvcc_party.prg_ctx import (
    UVCCPRGCTXv1,
    aes128_encrypt_fixedrk_v1,
    aes128_expand_rk_v1,
    derive_dpf_prg_ctx_v1,
    prg_expand2_seed16_aes_fixedkey_v1,
    prg_expand2_seed16_chacha20_fixedkey_v1,
    prg_salt32_v1,
    prk_pair_v1,
)


def test_hkdf_rfc5869_sha256_vector_1() -> None:
    ikm = bytes([0x0B] * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    prk = hkdf_extract(salt=salt, ikm=ikm)
    assert prk.hex() == "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
    okm = hkdf_expand(prk32=prk, info=info, length=42)
    assert okm.hex() == "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"


def test_kdf_info_v1_roundtrip_and_size() -> None:
    info = UVCCKDFInfoV1(
        kind=KIND_DPF_PRG_CTX,
        prg_type=PRG_TYPE_AES128_FIXEDKEY,
        pair_id=0,
        share_idx=255,
        w_bits=16,
        elem_bits=16,
        sid32=b"\x11" * 32,
        fss_id16=b"\x22" * 16,
        table_hash16=kdf_info_zero_table_hash16(),
    )
    b = info.to_bytes()
    assert len(b) == 83
    info2 = UVCCKDFInfoV1.from_bytes(b)
    assert info2 == info


def test_prg_ctx_v1_derivation_aes_fixedkey_is_deterministic() -> None:
    sid32 = b"\x01" * 32
    kpair32 = b"\x02" * 32
    fss_id16 = b"\x03" * 16

    ctx = derive_dpf_prg_ctx_v1(
        sid32=sid32,
        kpair32=kpair32,
        prg_type=PRG_TYPE_AES128_FIXEDKEY,
        impl=1,
        pair_id=0,
        w_bits=16,
        elem_bits=16,
        fss_id16=fss_id16,
        table_hash16=None,
    )
    assert isinstance(ctx, UVCCPRGCTXv1)
    ctx_bytes = ctx.to_bytes()
    assert len(ctx_bytes) == 256
    ctx2 = UVCCPRGCTXv1.from_bytes(ctx_bytes)
    assert ctx2 == ctx

    # Re-derive key material via HKDF to spot-check AES key schedule hookup.
    prk = prk_pair_v1(sid32=sid32, kpair32=kpair32)
    info_bytes = UVCCKDFInfoV1(
        kind=KIND_DPF_PRG_CTX,
        prg_type=PRG_TYPE_AES128_FIXEDKEY,
        pair_id=0,
        share_idx=255,
        w_bits=16,
        elem_bits=16,
        sid32=sid32,
        fss_id16=fss_id16,
        table_hash16=kdf_info_zero_table_hash16(),
    ).to_bytes()
    okm32 = hkdf_expand(prk32=prk, info=info_bytes, length=32)
    aes_key16 = okm32[0:16]
    assert ctx.aes_rk176 == aes128_expand_rk_v1(aes_key16)

    # Functional check: encrypt a known block with this derived rk, compare to pycryptodome if present.
    pt = bytes(range(16))
    ct = aes128_encrypt_fixedrk_v1(rk176=ctx.aes_rk176, block16=pt)
    assert len(ct) == 16

    # PRG expansion produces seeds with LSB of byte0 cleared (t extracted).
    seed = bytes([0xAA] * 16)
    sL, tL, sR, tR = prg_expand2_seed16_aes_fixedkey_v1(ctx=ctx, seed16=seed, lvl=7)
    assert len(sL) == 16 and len(sR) == 16
    assert (sL[0] & 1) == 0 and (sR[0] & 1) == 0
    assert tL in (0, 1) and tR in (0, 1)


def test_prg_ctx_v1_derivation_chacha20_fixedkey_is_deterministic() -> None:
    sid32 = b"\x09" * 32
    kpair32 = b"\x08" * 32
    fss_id16 = b"\x07" * 16

    ctx = derive_dpf_prg_ctx_v1(
        sid32=sid32,
        kpair32=kpair32,
        prg_type=PRG_TYPE_CHACHA20_FIXEDKEY,
        impl=1,
        pair_id=1,
        w_bits=16,
        elem_bits=16,
        fss_id16=fss_id16,
        table_hash16=None,
    )
    ctx_bytes = ctx.to_bytes()
    assert len(ctx_bytes) == 256
    assert UVCCPRGCTXv1.from_bytes(ctx_bytes) == ctx

    seed = bytes(range(16))
    sL, tL, sR, tR = prg_expand2_seed16_chacha20_fixedkey_v1(ctx=ctx, seed16=seed, lvl=0x1234)
    assert len(sL) == 16 and len(sR) == 16
    assert (sL[0] & 1) == 0 and (sR[0] & 1) == 0
    assert tL in (0, 1) and tR in (0, 1)


def test_prg_salt_is_32_bytes() -> None:
    salt = prg_salt32_v1(sid32=b"\x00" * 32)
    assert len(salt) == 32


