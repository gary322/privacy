#ifndef UVCC_TRUNC_PACK_V1_H
#define UVCC_TRUNC_PACK_V1_H
/*
UVCC v1 — TRUNC preprocessing pack record, byte-exact.

Source of truth: `research/privacy_new.txt` §1.4 (TRUNC pack) and §1.5 (fss_id mapping).
*/

#include <stdint.h>

#include "uvcc_rss_pairs_v1.h"

#if defined(_MSC_VER)
#pragma pack(push, 1)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define UVCC_PACKED __attribute__((packed))
#else
#define UVCC_PACKED
#endif

// Header (64 bytes).
typedef struct UVCC_PACKED {
  uint8_t  magic[8];          // "UVCCTRN1"
  uint16_t version_le;        // = 1
  uint16_t flags_le;          // bit0 = signed_mode
  uint8_t  k_bits;            // = 64
  uint8_t  f_bits;            // F
  uint8_t  chunk_bits;        // = 16
  uint8_t  reserved0;         // = 0
  uint8_t  sid_hash32[32];    // must match job
  uint32_t sgir_op_id_le;     // this TRUNC op
  uint64_t base_fss_id_le;    // base namespace for all compares
  uint32_t reserved1_le;      // = 0
} uvcc_trunc_pack_hdr_v1;

_Static_assert(sizeof(uvcc_trunc_pack_hdr_v1) == 64, "uvcc_trunc_pack_hdr_v1 must be 64 bytes");

// Following the header (per party): for each value k (vector lane), store:
// - R_pair[k]  : uvcc_rss_u64_pair_v1
// - R1_pair[k] : uvcc_rss_u64_pair_v1
// - R0_pair[k] : uvcc_rss_u64_pair_v1
//
// The count of values is known from SGIR tensor metadata for the consuming op.

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_TRUNC_PACK_V1_H


