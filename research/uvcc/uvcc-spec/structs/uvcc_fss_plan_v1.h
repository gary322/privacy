#ifndef UVCC_FSS_PLAN_V1_H
#define UVCC_FSS_PLAN_V1_H
/*
UVCC v1 — unified GPU evaluator plan ABI (device-side) for FSS (DPF/DCF) evaluation.

Source of truth: `research/privacy_new.txt` (OP_LUT ABI appendix) §3.1–§4 (UVCC_FSSPlanDevice_v1 / tasks).
*/

#include <stdint.h>

#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define UVCC_PACKED __attribute__((packed))
#else
#define UVCC_PACKED
#endif

// Task kinds (v1).
typedef enum {
  UVCC_FSS_DPF_POINT = 1,  // output: [u == alpha] share
  UVCC_FSS_DCF_LT    = 2,  // output: [u <  alpha] share
  UVCC_FSS_DCF_LE    = 3,  // optional
  UVCC_FSS_DCF_GT    = 4,  // optional
  UVCC_FSS_DCF_GE    = 5,  // optional
} UVCC_FSSKind_v1;

typedef enum {
  UVCC_IN_U16 = 1,
  UVCC_IN_U32 = 2,
} UVCC_FSSInType_v1;

typedef enum {
  UVCC_OUT_BITPACK32 = 1,  // canonical output
  UVCC_OUT_U8        = 2,  // debug/interop only
} UVCC_FSSOutType_v1;

// Packed task descriptor (76 bytes).
typedef struct UVCC_PACKED {
  uint64_t fss_id;        // lookup key (already resolved)
  uint32_t sgir_op_id;    // for debugging / transcript binding
  uint16_t lane;          // 0.. or 0xFFFF broadcast; usually 0xFFFF
  uint8_t  kind;          // UVCC_FSSKind_v1
  uint8_t  reserved0;

  uint16_t domain_bits;   // w (<= 32 for this ABI)
  uint16_t range_bits;    // 1 (boolean)
  uint8_t  in_type;       // UVCC_FSSInType_v1
  uint8_t  out_type;      // UVCC_FSSOutType_v1
  uint16_t flags;         // bit0 LOAD_CW_TO_SHMEM, bit1 INPUT_MASK_DOMAIN, bit2 SIGNED_INPUT, bit3 ZERO_OUTPUT_FIRST

  uint32_t lanes;         // number of u's to evaluate

  uint64_t in_offset;     // byte offset into d_in_arena
  uint32_t in_stride;     // byte stride between consecutive u's
  uint32_t in_pad0;

  uint64_t out_offset;    // byte offset into d_out_arena
  uint32_t out_stride;    // byte stride between consecutive output vectors
  uint32_t out_pad0;

  uint64_t key_offset;    // byte offset into d_key_arena
  uint32_t key_bytes;     // key blob length
  uint32_t key_pad0;
} UVCC_FSSExecTask_v1;

_Static_assert(sizeof(UVCC_FSSExecTask_v1) == 76, "UVCC_FSSExecTask_v1 must be 76 bytes");

// Packed device plan header (80 bytes on 64-bit platforms).
typedef struct UVCC_PACKED {
  uint32_t version;       // = 1
  uint32_t task_count;    // number of tasks

  uint64_t key_arena_bytes;
  uint64_t in_arena_bytes;
  uint64_t out_arena_bytes;

  const uint8_t* d_key_arena;
  const uint8_t* d_in_arena;
  uint8_t*       d_out_arena;

  // Optional scratch (may be NULL)
  uint8_t*       d_scratch;
  uint64_t       scratch_bytes;

  const void*    d_tasks;  // points to UVCC_FSSExecTask_v1[task_count]
} UVCC_FSSPlanDevice_v1;

_Static_assert(sizeof(UVCC_FSSPlanDevice_v1) == 80, "UVCC_FSSPlanDevice_v1 must be 80 bytes on 64-bit");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_FSS_PLAN_V1_H


