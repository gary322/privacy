#ifndef UVCC_B2A_PACK_V1_H
#define UVCC_B2A_PACK_V1_H
/*
UVCC v1 — B2A pack wire format, byte-exact.

Source of truth: `research/privacy_new.txt` §1.2 (B2A pack wire format v1).
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
  uint8_t  magic[8];        // "UVCCB2A1"
  uint16_t version_le;      // = 1
  uint16_t flags_le;        // reserved (0)
  uint32_t count_bits_le;   // N conversions
  uint32_t sgir_op_id_le;   // consuming op id (or 0 for pool)
  uint64_t base_stream_id_le; // optional namespace base (0 allowed)
  uint8_t  sid_hash32[32];  // must match job sid_hash
  uint32_t reserved0_le;    // = 0
} uvcc_b2a_pack_hdr_v1;

_Static_assert(sizeof(uvcc_b2a_pack_hdr_v1) == 64, "uvcc_b2a_pack_hdr_v1 must be 64 bytes");

// Body (per party), repeated k=0..N-1:
// - r_bool_pair[k]  : uvcc_rss_u1_pair_v1  (2 bytes)
// - r_arith_pair[k] : uvcc_rss_u64_pair_v1 (16 bytes)
// Total body bytes: 18*N.

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_B2A_PACK_V1_H


