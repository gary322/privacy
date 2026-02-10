#ifndef UVCC_BOOL_GPU_ABI_V1_H
#define UVCC_BOOL_GPU_ABI_V1_H
// UVCC_REQ_GROUP: uvcc_group_10df23d70db0500b
/*
UVCC v1 — CUDA ABIs for boolean/GF(2) kernels and A2B public packing helpers.

Source of truth: `research/privacy_new.txt`
  - GF(2) AND kernels: §2 (GF(2) AND — kernel scratch layout)
  - A2B subtract kernels: §3 (A2B subtract — GPU ABIs)
  - A2B packing helpers: §4.4 and §4 (cpub→cjmask)

This header declares canonical CUDA entrypoints. When compiled as plain C/C++, the CUDA
attributes are elided; when compiled with nvcc, the prototypes match the canonical ABI.
*/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__
#define UVCC_CUDA_GLOBAL __global__
#define UVCC_RESTRICT __restrict__
#else
#define UVCC_CUDA_GLOBAL
#define UVCC_RESTRICT
#endif

// GF(2) AND — phase A (prepare OPEN_BOOL payloads for e,f), word-packed u32.
UVCC_CUDA_GLOBAL void uvcc_gf2_and_prepare_pack32_v1(
    const uint32_t* UVCC_RESTRICT x_lo,
    const uint32_t* UVCC_RESTRICT x_hi,
    const uint32_t* UVCC_RESTRICT y_lo,
    const uint32_t* UVCC_RESTRICT y_hi,
    const uint32_t* UVCC_RESTRICT a_lo,
    const uint32_t* UVCC_RESTRICT a_hi,
    const uint32_t* UVCC_RESTRICT b_lo,
    const uint32_t* UVCC_RESTRICT b_hi,
    uint32_t* UVCC_RESTRICT e_lo_out,
    uint32_t* UVCC_RESTRICT f_lo_out,
    uint32_t* UVCC_RESTRICT e_hi_scratch,
    uint32_t* UVCC_RESTRICT f_hi_scratch);

// GF(2) AND — phase B (finish), word-packed u32.
UVCC_CUDA_GLOBAL void uvcc_gf2_and_finish_pack32_v1(
    const uint32_t* UVCC_RESTRICT a_lo,
    const uint32_t* UVCC_RESTRICT a_hi,
    const uint32_t* UVCC_RESTRICT b_lo,
    const uint32_t* UVCC_RESTRICT b_hi,
    const uint32_t* UVCC_RESTRICT c_lo,
    const uint32_t* UVCC_RESTRICT c_hi,
    const uint32_t* UVCC_RESTRICT e_pub,
    const uint32_t* UVCC_RESTRICT f_pub,
    uint32_t* UVCC_RESTRICT z_lo_out,
    uint32_t* UVCC_RESTRICT z_hi_out,
    uint32_t party_id);

// A2B subtract — per-bit prepare (build OPEN_BOOL payloads for AND(r_j, b_j)).
UVCC_CUDA_GLOBAL void uvcc_a2b_sub_prepare_and_bit_pack32_v1(
    const uint32_t* UVCC_RESTRICT rj_lo,
    const uint32_t* UVCC_RESTRICT rj_hi,
    const uint32_t* UVCC_RESTRICT bj_lo,
    const uint32_t* UVCC_RESTRICT bj_hi,
    const uint32_t* UVCC_RESTRICT aj_lo,
    const uint32_t* UVCC_RESTRICT aj_hi,
    const uint32_t* UVCC_RESTRICT bjT_lo,
    const uint32_t* UVCC_RESTRICT bjT_hi,
    uint32_t* UVCC_RESTRICT e_lo_out,
    uint32_t* UVCC_RESTRICT f_lo_out,
    uint32_t* UVCC_RESTRICT e_hi_scratch,
    uint32_t* UVCC_RESTRICT f_hi_scratch);

// A2B subtract — per-bit finish (compute g_j, output x_j, update borrow).
UVCC_CUDA_GLOBAL void uvcc_a2b_sub_finish_and_bit_pack32_v1(
    const uint32_t* UVCC_RESTRICT rj_lo,
    const uint32_t* UVCC_RESTRICT rj_hi,
    const uint32_t* UVCC_RESTRICT bj_lo,
    const uint32_t* UVCC_RESTRICT bj_hi,
    const uint32_t* UVCC_RESTRICT aj_lo,
    const uint32_t* UVCC_RESTRICT aj_hi,
    const uint32_t* UVCC_RESTRICT bjT_lo,
    const uint32_t* UVCC_RESTRICT bjT_hi,
    const uint32_t* UVCC_RESTRICT cj_lo,
    const uint32_t* UVCC_RESTRICT cj_hi,
    const uint32_t* UVCC_RESTRICT e_pub,
    const uint32_t* UVCC_RESTRICT f_pub,
    const uint32_t* UVCC_RESTRICT cj_public_mask,
    uint32_t* UVCC_RESTRICT xj_lo_out,
    uint32_t* UVCC_RESTRICT xj_hi_out,
    uint32_t* UVCC_RESTRICT bnext_lo_out,
    uint32_t* UVCC_RESTRICT bnext_hi_out,
    uint32_t party_id);

// A2B subtract — scratch layout helper for a per-bit iteration (pack32).
// The scratch is a flat u32 array of length (store_g ? 8W : 6W), where W = ceil(N/32).
// Offsets in u32 units:
//   0..W-1      e_lo
//   W..2W-1     f_lo
//   2W..3W-1    e_hi
//   3W..4W-1    f_hi
//   4W..5W-1    e_pub
//   5W..6W-1    f_pub
//   6W..7W-1    g_lo (optional)
//   7W..8W-1    g_hi (optional)
static inline uint32_t uvcc_a2b_sub_bit_scratch_words_pack32_v1(uint32_t W, uint32_t store_g) {
  return store_g ? (8u * W) : (6u * W);
}

static inline uint32_t* uvcc_a2b_sub_bit_scratch_e_lo_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 0u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_f_lo_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 1u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_e_hi_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 2u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_f_hi_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 3u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_e_pub_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 4u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_f_pub_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 5u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_g_lo_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 6u * W; }
static inline uint32_t* uvcc_a2b_sub_bit_scratch_g_hi_pack32_v1(uint32_t* base_u32, uint32_t W) { return base_u32 + 7u * W; }

// A2B optional helper: pack c_lo payload from u64 → u8 (w=8).
UVCC_CUDA_GLOBAL void uvcc_a2b_pack_c_lo_u8_v1(
    const uint64_t* UVCC_RESTRICT x_lo_u64,
    const uint64_t* UVCC_RESTRICT r_lo_u64,
    uint8_t* UVCC_RESTRICT c_lo_u8_out,
    uint32_t n_elems);

// A2B optional helper: pack c_lo payload from u64 → u16 (w=16).
UVCC_CUDA_GLOBAL void uvcc_a2b_pack_c_lo_u16_v1(
    const uint64_t* UVCC_RESTRICT x_lo_u64,
    const uint64_t* UVCC_RESTRICT r_lo_u64,
    uint16_t* UVCC_RESTRICT c_lo_u16_out,
    uint32_t n_elems);

// A2B deterministic public pack: c_pub (u8) → cj_public_mask (SoA, u32 words).
UVCC_CUDA_GLOBAL void uvcc_a2b_cpub_to_cjmask_u8_v1(
    const uint8_t* UVCC_RESTRICT c_pub_u8,
    uint32_t* UVCC_RESTRICT out_u32,
    uint32_t L);

// A2B deterministic public pack: c_pub (u16) → cj_public_mask (SoA, u32 words).
UVCC_CUDA_GLOBAL void uvcc_a2b_cpub_to_cjmask_u16_v1(
    const uint16_t* UVCC_RESTRICT c_pub_u16,
    uint32_t* UVCC_RESTRICT out_u32,
    uint32_t L);

#undef UVCC_CUDA_GLOBAL
#undef UVCC_RESTRICT

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // UVCC_BOOL_GPU_ABI_V1_H


