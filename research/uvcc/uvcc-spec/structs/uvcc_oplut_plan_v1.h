#ifndef UVCC_OPLUT_PLAN_V1_H
#define UVCC_OPLUT_PLAN_V1_H
/*
UVCC v1 — OP_LUT (public table) device plan ABI (packed, byte-exact).

Source of truth: `research/privacy_new.txt` §6 (UVCC_LUTPlanDevice_v1 / UVCC_LUTTask_v1).
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

// Plan header (96 bytes on 64-bit platforms).
typedef struct UVCC_PACKED {
  uint32_t version;      // = 1
  uint32_t task_count;

  const uint8_t* d_key_arena;    uint64_t key_arena_bytes;
  const uint8_t* d_const_arena;  uint64_t const_arena_bytes;
  const uint8_t* d_u_pub_arena;  uint64_t u_pub_bytes;
  uint8_t*       d_out_arena;    uint64_t out_arena_bytes;

  uint8_t*       d_scratch;      uint64_t scratch_bytes;

  const void*    d_tasks;        // UVCC_LUTTask_v1[task_count]
} UVCC_LUTPlanDevice_v1;

_Static_assert(sizeof(UVCC_LUTPlanDevice_v1) == 96, "UVCC_LUTPlanDevice_v1 must be 96 bytes on 64-bit");

enum UVCC_LUTDomain_v1 : uint8_t { UVCC_LUT_W8 = 8, UVCC_LUT_W16 = 16 };

enum UVCC_LUTElemFmt_v1 : uint8_t {
  UVCC_LUT_ELEM_U8  = 1,
  UVCC_LUT_ELEM_I8  = 2,
  UVCC_LUT_ELEM_U16 = 3,
  UVCC_LUT_ELEM_I16 = 4,
  UVCC_LUT_ELEM_U32 = 5,
  UVCC_LUT_ELEM_I32 = 6,
  UVCC_LUT_ELEM_R64 = 7,
};

enum UVCC_LUTDPFMode_v1 : uint8_t {
  UVCC_LUT_DPF_ARITH_R64 = 1,
  UVCC_LUT_DPF_BOOL_BIT  = 2,
};

// Task descriptor (84 bytes).
typedef struct UVCC_PACKED {
  uint64_t fss_id;
  uint32_t sgir_op_id;
  uint8_t  domain_w;
  uint8_t  elem_fmt;
  uint8_t  dpf_mode;
  uint8_t  flags;

  uint32_t lanes;

  uint64_t u_pub_offset;
  uint32_t u_pub_stride;
  uint32_t u_pad0;

  uint64_t table_offset;
  uint32_t table_bytes;
  uint32_t table_pad0;

  uint64_t out_offset;
  uint32_t out_stride;
  uint32_t out_pad0;

  uint64_t key_offset;
  uint32_t key_bytes;
  uint32_t key_pad0;
} UVCC_LUTTask_v1;

_Static_assert(sizeof(UVCC_LUTTask_v1) == 84, "UVCC_LUTTask_v1 must be 84 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_OPLUT_PLAN_V1_H


