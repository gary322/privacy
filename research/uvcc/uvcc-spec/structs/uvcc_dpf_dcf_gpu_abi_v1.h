#ifndef UVCC_DPF_DCF_GPU_ABI_V1_H
#define UVCC_DPF_DCF_GPU_ABI_V1_H
// UVCC_REQ_GROUP: uvcc_group_b2a809ccbb581fc9
/*
UVCC v1 — CUDA ABIs for DPF/DCF evaluation kernels.

Source of truth: `research/privacy_new.txt` §6 (GPU ABIs for DPF/DCF).

This header is a compile-safe declaration of the expected CUDA entrypoints. When compiled
as plain C/C++, the CUDA attributes are elided; when compiled with nvcc, the prototypes
match the canonical ABI.
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

// Stage-1 for w=16: expand to 8-bit frontier (256 nodes).
UVCC_CUDA_GLOBAL void uvcc_dpf_stage1_w16_v1(
    const uint8_t* UVCC_RESTRICT keyrec_bytes,
    uint64_t* UVCC_RESTRICT frontier_seed_lo,
    uint64_t* UVCC_RESTRICT frontier_seed_hi,
    uint8_t*  UVCC_RESTRICT frontier_t,
    uint8_t*  UVCC_RESTRICT frontier_acc
);

// Stage-2 for w=16: DCF full-domain (65536 words).
UVCC_CUDA_GLOBAL void uvcc_dcf_stage2_w16_v1(
    const uint8_t* UVCC_RESTRICT keyrec_bytes,
    const uint64_t* UVCC_RESTRICT frontier_seed_lo,
    const uint64_t* UVCC_RESTRICT frontier_seed_hi,
    const uint8_t*  UVCC_RESTRICT frontier_t,
    const uint8_t*  UVCC_RESTRICT frontier_acc,
    uint64_t* UVCC_RESTRICT out_word_u64
);

// w=8 single-kernel full-domain DCF (256 words).
UVCC_CUDA_GLOBAL void uvcc_dcf_full_w8_v1(
    const uint8_t* UVCC_RESTRICT keyrec_bytes,
    uint64_t* UVCC_RESTRICT out_word_u64
);

#undef UVCC_CUDA_GLOBAL
#undef UVCC_RESTRICT

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // UVCC_DPF_DCF_GPU_ABI_V1_H


