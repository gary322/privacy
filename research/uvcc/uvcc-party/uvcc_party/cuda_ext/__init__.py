from __future__ import annotations

from typing import Tuple

import torch

from .loader import load_uvcc_cuda_ext


def dpf_stage1_w16(keyrec_cuda_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA: uvcc_dpf_stage1_w16_v1.

    Args:
      keyrec_cuda_u8: CUDA uint8 tensor containing the keyrec bytes.
    Returns:
      (frontier_seed_lo_i64[256], frontier_seed_hi_i64[256], frontier_t_u8[256], frontier_acc_u8[256])
    """
    if not isinstance(keyrec_cuda_u8, torch.Tensor) or not keyrec_cuda_u8.is_cuda or keyrec_cuda_u8.dtype != torch.uint8:
        raise TypeError("keyrec_cuda_u8 must be a CUDA torch.uint8 tensor")
    ext = load_uvcc_cuda_ext()
    return ext.dpf_stage1_w16(keyrec_cuda_u8)


def dcf_stage2_w16(
    keyrec_cuda_u8: torch.Tensor,
    frontier_seed_lo_i64: torch.Tensor,
    frontier_seed_hi_i64: torch.Tensor,
    frontier_t_u8: torch.Tensor,
    frontier_acc_u8: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA: uvcc_dcf_stage2_w16_v1.

    Returns:
      out_word_u64_i64[65536] (u64 bit-patterns stored in int64)
    """
    ext = load_uvcc_cuda_ext()
    return ext.dcf_stage2_w16(keyrec_cuda_u8, frontier_seed_lo_i64, frontier_seed_hi_i64, frontier_t_u8, frontier_acc_u8)


def dcf_full_w8(keyrec_cuda_u8: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dcf_full_w8_v1.

    Returns:
      out_word_u64_i64[256]
    """
    ext = load_uvcc_cuda_ext()
    return ext.dcf_full_w8(keyrec_cuda_u8)


def dpf_eval_point_w8_batch(*, keyrecs_blob_u8: torch.Tensor, key_stride_bytes: int, x_pub_u16_i16: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dpf_eval_point_w8_batch_v1.

    Args:
      keyrecs_blob_u8: CUDA uint8 tensor with concatenated keyrecs (N * key_stride_bytes).
      key_stride_bytes: keyrec length in bytes.
      x_pub_u16_i16: CUDA int16 tensor length N (u16 bit-patterns).

    Returns:
      out_bits_u8: CUDA uint8 tensor length N (0/1).
    """
    ext = load_uvcc_cuda_ext()
    return ext.dpf_eval_point_w8_batch(keyrecs_blob_u8, int(key_stride_bytes), x_pub_u16_i16)


def dpf_eval_point_w16_batch(*, keyrecs_blob_u8: torch.Tensor, key_stride_bytes: int, x_pub_u16_i16: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dpf_eval_point_w16_batch_v1.
    Returns out_bits_u8 length N (0/1).
    """
    ext = load_uvcc_cuda_ext()
    return ext.dpf_eval_point_w16_batch(keyrecs_blob_u8, int(key_stride_bytes), x_pub_u16_i16)


def dcf_eval_point_w8_batch(*, keyrecs_blob_u8: torch.Tensor, key_stride_bytes: int, x_pub_u16_i16: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dcf_eval_point_w8_batch_v1.
    Returns out_bits_u8 length N (0/1).
    """
    ext = load_uvcc_cuda_ext()
    return ext.dcf_eval_point_w8_batch(keyrecs_blob_u8, int(key_stride_bytes), x_pub_u16_i16)


def dcf_eval_point_w16_batch(*, keyrecs_blob_u8: torch.Tensor, key_stride_bytes: int, x_pub_u16_i16: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dcf_eval_point_w16_batch_v1.
    Returns out_bits_u8 length N (0/1).
    """
    ext = load_uvcc_cuda_ext()
    return ext.dcf_eval_point_w16_batch(keyrecs_blob_u8, int(key_stride_bytes), x_pub_u16_i16)


def trunc_apply_u64(
    *,
    C1_pub_u64_i64: torch.Tensor,
    R1_lo_u64_i64: torch.Tensor,
    R1_hi_u64_i64: torch.Tensor,
    carry_lo_u64_i64: torch.Tensor,
    carry_hi_u64_i64: torch.Tensor,
    ov_lo_u64_i64: torch.Tensor,
    ov_hi_u64_i64: torch.Tensor,
    add_const_u64: int,
    party_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA: uvcc_trunc_apply_u64_v1.

    Notes:
      - Inputs/outputs are int64 tensors containing u64 bit-patterns.
      - The kernel applies the share-0 placement rule for the public C1 term using party_id:
        P0 adds into lo; P2 adds into hi; P1 adds none.

    Returns:
      (y_lo_u64_i64, y_hi_u64_i64)
    """
    ext = load_uvcc_cuda_ext()
    y_lo, y_hi = ext.trunc_apply_u64(
        C1_pub_u64_i64,
        R1_lo_u64_i64,
        R1_hi_u64_i64,
        carry_lo_u64_i64,
        carry_hi_u64_i64,
        ov_lo_u64_i64,
        ov_hi_u64_i64,
        int(add_const_u64),
        int(party_id),
    )
    return y_lo, y_hi


def matmul_u64(A_u64_i64: torch.Tensor, B_u64_i64: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_matmul_u64_v1.

    Computes C = A @ B over the u64 ring (mod 2^64), with tensors represented as
    torch.int64 carrying u64 bit-patterns (two's complement).
    """
    if not isinstance(A_u64_i64, torch.Tensor) or not isinstance(B_u64_i64, torch.Tensor):
        raise TypeError("A_u64_i64 and B_u64_i64 must be torch.Tensors")
    if not A_u64_i64.is_cuda or not B_u64_i64.is_cuda:
        raise TypeError("A_u64_i64 and B_u64_i64 must be CUDA tensors")
    if A_u64_i64.dtype != torch.int64 or B_u64_i64.dtype != torch.int64:
        raise TypeError("A_u64_i64 and B_u64_i64 must be torch.int64 (u64 bit-patterns)")
    ext = load_uvcc_cuda_ext()
    return ext.matmul_u64(A_u64_i64, B_u64_i64)


def dpf_full_w8_bitpack32(keyrec_cuda_u8: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dpf_full_w8_bitpack32_v1.

    Returns:
      out_words_i32[8] (each word is a u32 bitmask for 32 lanes; total 256 bits)
    """
    ext = load_uvcc_cuda_ext()
    return ext.dpf_full_w8_bitpack32(keyrec_cuda_u8)


def dcf_full_w8_bitpack32(keyrec_cuda_u8: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_dcf_full_w8_bitpack32_v1.

    Returns:
      out_words_i32[8] (each word is a u32 bitmask for 32 lanes; total 256 bits)
    """
    ext = load_uvcc_cuda_ext()
    return ext.dcf_full_w8_bitpack32(keyrec_cuda_u8)


def dpf_stage2_w16_bitpack32(
    keyrec_cuda_u8: torch.Tensor,
    frontier_seed_lo_i64: torch.Tensor,
    frontier_seed_hi_i64: torch.Tensor,
    frontier_t_u8: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA: uvcc_dpf_stage2_w16_bitpack32_v1.

    Returns:
      out_words_i32[2048] (65536 bits packed into u32 masks)
    """
    ext = load_uvcc_cuda_ext()
    return ext.dpf_stage2_w16_bitpack32(keyrec_cuda_u8, frontier_seed_lo_i64, frontier_seed_hi_i64, frontier_t_u8)


def dcf_stage2_w16_bitpack32(
    keyrec_cuda_u8: torch.Tensor,
    frontier_seed_lo_i64: torch.Tensor,
    frontier_seed_hi_i64: torch.Tensor,
    frontier_t_u8: torch.Tensor,
    frontier_acc_u8: torch.Tensor,
) -> torch.Tensor:
    """
    CUDA: uvcc_dcf_stage2_w16_bitpack32_v1.

    Returns:
      out_words_i32[2048] (65536 bits packed into u32 masks)
    """
    ext = load_uvcc_cuda_ext()
    return ext.dcf_stage2_w16_bitpack32(keyrec_cuda_u8, frontier_seed_lo_i64, frontier_seed_hi_i64, frontier_t_u8, frontier_acc_u8)


def oplut_phase2_w8_record(record_cuda_u8: torch.Tensor, u_pub_cuda_u8: torch.Tensor, table_cuda_i64: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_oplut_phase2_w8_record_v1.

    Args:
      record_cuda_u8: CUDA uint8 tensor containing the per-party OP_LUT record blob bytes.
      u_pub_cuda_u8:  CUDA uint8 tensor length LANES (opened masked indices).
      table_cuda_i64: CUDA int64 tensor length 256 (public table lifted into ring u64 bit-patterns).
    Returns:
      y_prime_i64[LANES] additive shares (after refresh) as int64 u64 bit-patterns.
    """
    ext = load_uvcc_cuda_ext()
    return ext.oplut_phase2_w8_record(record_cuda_u8, u_pub_cuda_u8, table_cuda_i64)


def oplut_phase2_w16_record(record_cuda_u8: torch.Tensor, u_pub_cuda_i16: torch.Tensor, table_cuda_i64: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_oplut_phase2_w16_record_v1.

    Args:
      record_cuda_u8: CUDA uint8 tensor containing the per-party OP_LUT record blob bytes.
      u_pub_cuda_i16: CUDA int16 tensor length LANES (opened masked indices as u16 bit-patterns).
      table_cuda_i64: CUDA int64 tensor length 65536 (public table lifted into ring u64 bit-patterns).
    Returns:
      y_prime_i64[LANES] additive shares (after refresh) as int64 u64 bit-patterns.
    """
    ext = load_uvcc_cuda_ext()
    return ext.oplut_phase2_w16_record(record_cuda_u8, u_pub_cuda_i16, table_cuda_i64)


def gf2_and_prepare_pack32(
    *,
    x_lo_i32: torch.Tensor,
    x_hi_i32: torch.Tensor,
    y_lo_i32: torch.Tensor,
    y_hi_i32: torch.Tensor,
    a_lo_i32: torch.Tensor,
    a_hi_i32: torch.Tensor,
    b_lo_i32: torch.Tensor,
    b_hi_i32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA: uvcc_gf2_and_prepare_pack32_v1.
    Returns: (e_lo_out, f_lo_out, e_hi_scratch, f_hi_scratch) all int32 CUDA tensors.
    """
    ext = load_uvcc_cuda_ext()
    e_lo, f_lo, e_hi, f_hi = ext.gf2_and_prepare_pack32(x_lo_i32, x_hi_i32, y_lo_i32, y_hi_i32, a_lo_i32, a_hi_i32, b_lo_i32, b_hi_i32)
    return e_lo, f_lo, e_hi, f_hi


def gf2_and_finish_pack32(
    *,
    a_lo_i32: torch.Tensor,
    a_hi_i32: torch.Tensor,
    b_lo_i32: torch.Tensor,
    b_hi_i32: torch.Tensor,
    c_lo_i32: torch.Tensor,
    c_hi_i32: torch.Tensor,
    e_pub_i32: torch.Tensor,
    f_pub_i32: torch.Tensor,
    party_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA: uvcc_gf2_and_finish_pack32_v1.
    Returns: (z_lo_out, z_hi_out) int32 CUDA tensors.
    """
    ext = load_uvcc_cuda_ext()
    z_lo, z_hi = ext.gf2_and_finish_pack32(a_lo_i32, a_hi_i32, b_lo_i32, b_hi_i32, c_lo_i32, c_hi_i32, e_pub_i32, f_pub_i32, int(party_id))
    return z_lo, z_hi


def a2b_sub_prepare_pack32(
    *,
    rj_lo_i32: torch.Tensor,
    rj_hi_i32: torch.Tensor,
    bj_lo_i32: torch.Tensor,
    bj_hi_i32: torch.Tensor,
    aj_lo_i32: torch.Tensor,
    aj_hi_i32: torch.Tensor,
    bjT_lo_i32: torch.Tensor,
    bjT_hi_i32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA: uvcc_a2b_sub_prepare_and_bit_pack32_v1.
    Returns: (e_lo_out, f_lo_out, e_hi_scratch, f_hi_scratch) int32 CUDA tensors.
    """
    ext = load_uvcc_cuda_ext()
    e_lo, f_lo, e_hi, f_hi = ext.a2b_sub_prepare_pack32(rj_lo_i32, rj_hi_i32, bj_lo_i32, bj_hi_i32, aj_lo_i32, aj_hi_i32, bjT_lo_i32, bjT_hi_i32)
    return e_lo, f_lo, e_hi, f_hi


def a2b_sub_finish_pack32(
    *,
    rj_lo_i32: torch.Tensor,
    rj_hi_i32: torch.Tensor,
    bj_lo_i32: torch.Tensor,
    bj_hi_i32: torch.Tensor,
    aj_lo_i32: torch.Tensor,
    aj_hi_i32: torch.Tensor,
    bjT_lo_i32: torch.Tensor,
    bjT_hi_i32: torch.Tensor,
    cj_lo_i32: torch.Tensor,
    cj_hi_i32: torch.Tensor,
    e_pub_i32: torch.Tensor,
    f_pub_i32: torch.Tensor,
    cj_public_mask_i32: torch.Tensor,
    party_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA: uvcc_a2b_sub_finish_and_bit_pack32_v1.
    Returns: (xj_lo_out, xj_hi_out, bnext_lo_out, bnext_hi_out) int32 CUDA tensors.
    """
    ext = load_uvcc_cuda_ext()
    x_lo, x_hi, b_lo, b_hi = ext.a2b_sub_finish_pack32(
        rj_lo_i32,
        rj_hi_i32,
        bj_lo_i32,
        bj_hi_i32,
        aj_lo_i32,
        aj_hi_i32,
        bjT_lo_i32,
        bjT_hi_i32,
        cj_lo_i32,
        cj_hi_i32,
        e_pub_i32,
        f_pub_i32,
        cj_public_mask_i32,
        int(party_id),
    )
    return x_lo, x_hi, b_lo, b_hi


def a2b_pack_c_lo_u8(*, x_lo_u64_i64: torch.Tensor, r_lo_u64_i64: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_a2b_pack_c_lo_u8_v1.
    Returns torch.uint8 CUDA tensor length n_elems.
    """
    ext = load_uvcc_cuda_ext()
    return ext.a2b_pack_c_lo_u8(x_lo_u64_i64, r_lo_u64_i64)


def a2b_pack_c_lo_u16(*, x_lo_u64_i64: torch.Tensor, r_lo_u64_i64: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_a2b_pack_c_lo_u16_v1.
    Returns torch.int16 CUDA tensor length n_elems (u16 bit-patterns).
    """
    ext = load_uvcc_cuda_ext()
    return ext.a2b_pack_c_lo_u16(x_lo_u64_i64, r_lo_u64_i64)


def a2b_cpub_to_cjmask_u8(*, c_pub_u8: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_a2b_cpub_to_cjmask_u8_v1.
    Returns torch.int32 CUDA tensor of length (8 * n_words), SoA layout: out[j*n_words + word_idx].
    """
    ext = load_uvcc_cuda_ext()
    return ext.a2b_cpub_to_cjmask_u8(c_pub_u8)


def a2b_cpub_to_cjmask_u16(*, c_pub_u16_i16: torch.Tensor) -> torch.Tensor:
    """
    CUDA: uvcc_a2b_cpub_to_cjmask_u16_v1.
    Returns torch.int32 CUDA tensor of length (16 * n_words), SoA layout: out[j*n_words + word_idx].
    """
    ext = load_uvcc_cuda_ext()
    return ext.a2b_cpub_to_cjmask_u16(c_pub_u16_i16)


