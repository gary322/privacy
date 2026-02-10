#ifndef UVCC_A2B_PACK_V1_H
#define UVCC_A2B_PACK_V1_H
/*
UVCC v1 — A2B pack wire format, byte-exact (also used as EDABIT carrier for CMP).

Source of truth: `research/privacy_new.txt` §2.3 (A2B pack wire format v1).
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
  uint8_t  magic[8];           // "UVCCA2B1"
  uint16_t version_le;         // = 1
  uint8_t  w_bits;             // 8 or 16
  uint8_t  flags;              // bit0 = mod-2^w (else require bounded)
  uint32_t count_vals_le;      // N values
  uint32_t sgir_op_id_le;      // consuming op id (or 0 for pool)
  uint64_t base_triple_id_le;  // GF(2) triple pool base index
  uint8_t  sid_hash32[32];     // must match job sid_hash
  uint32_t reserved0_le;       // = 0
} uvcc_a2b_pack_hdr_v1;

_Static_assert(sizeof(uvcc_a2b_pack_hdr_v1) == 64, "uvcc_a2b_pack_hdr_v1 must be 64 bytes");

// Body (per party), repeated k=0..N-1:
// 1) Arithmetic mask r_arith_pair[k] as uvcc_rss_u64_pair_v1 (16 bytes)
// 2) Boolean bits r_bits_pair[k][j] for j=0..w-1 as uvcc_rss_u1_pair_v1 (2*w bytes)
//
// Total per value:
// - w=8  : 16 + 16 = 32 bytes
// - w=16 : 16 + 32 = 48 bytes

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_A2B_PACK_V1_H


