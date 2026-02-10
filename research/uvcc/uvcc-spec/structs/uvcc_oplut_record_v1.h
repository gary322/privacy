#ifndef UVCC_OPLUT_RECORD_V1_H
#define UVCC_OPLUT_RECORD_V1_H
/*
UVCC v1 — OP_LUT transcript payload schema (byte-exact).

Source of truth: `research/privacy_new.txt` §1.6.2 (OP_LUT payload v1).
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

// Leaf flags (outer leaf header flags, where applicable).
enum uvcc_leaf_flags_v1 {
  UVCC_LEAF_FLAG_NONE        = 0x0000,
  UVCC_LEAF_FLAG_HAS_DELTA_H = 0x0001, // payload includes delta_hash[32]
  UVCC_LEAF_FLAG_PUBLIC_U    = 0x0002, // OP_LUT ran in public-table mode
};

// semantics_flags inside payload
#define OPLUT_SEM_HAS_DELTA_HASH 0x0001
#define OPLUT_SEM_HAS_YPAIR_HASH 0x0002  // MUST be set in v1
#define OPLUT_SEM_MASKED_TABLE   0x0004
#define OPLUT_SEM_PUBLIC_TABLE   0x0008

// OP_LUT payload v1 (216 bytes) + optional tail delta_hash[32].
typedef struct UVCC_PACKED {
  uint32_t sgir_op_id_le;      // u32
  uint8_t  w;                 // 8 or 16
  uint8_t  elem_bits;         // 8 or 16
  uint16_t semantics_flags_le; // bitfield
  uint32_t q_count_le;         // number of queries Q
  uint32_t reserved0_le;       // = 0

  uint8_t  fss_id32[32];      // unified namespace id (bytes32)
  uint64_t table_epoch_id_le; // u64
  uint8_t  table_id32[32];    // bytes32
  uint8_t  u_pub_root32[32];  // H(U_pub_bytes) as installed

  uint8_t  dpf_blob_hash32[32];
  uint8_t  outmask_hash32[32];
  uint8_t  y_pair_hash32[32];

  // Optional tail if (semantics_flags & OPLUT_SEM_HAS_DELTA_HASH):
  //   uint8_t delta_hash32[32];
} uvcc_oplut_payload_v1;

_Static_assert(sizeof(uvcc_oplut_payload_v1) == 216, "uvcc_oplut_payload_v1 must be 216 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_OPLUT_RECORD_V1_H


