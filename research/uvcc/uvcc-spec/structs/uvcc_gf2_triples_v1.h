#ifndef UVCC_GF2_TRIPLES_V1_H
#define UVCC_GF2_TRIPLES_V1_H
/*
UVCC v1 — GF(2) AND triple pool page, byte-exact.

Source of truth: `research/privacy_new.txt` §3.2 (Triple pool page wire format v1).
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
  uint8_t  magic[8];            // "UVCCG2T1"
  uint16_t version_le;          // = 1
  uint16_t flags_le;            // reserved (0)
  uint32_t count_triples_le;    // M triples
  uint64_t triple_id_base_le;   // global index base
  uint32_t sgir_op_id_le;       // 0 if shared pool
  uint8_t  sid_hash32[32];      // must match job sid_hash
  uint32_t reserved0_le;        // = 0
} uvcc_gf2_triples_hdr_v1;

_Static_assert(sizeof(uvcc_gf2_triples_hdr_v1) == 64, "uvcc_gf2_triples_hdr_v1 must be 64 bytes");

// Body (per party), per triple t=0..M-1:
// - a_pair: uvcc_rss_u1_pair_v1 (2 bytes)
// - b_pair: uvcc_rss_u1_pair_v1 (2 bytes)
// - c_pair: uvcc_rss_u1_pair_v1 (2 bytes)
//
// Total per party file size: 64 + 6*M bytes.

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_GF2_TRIPLES_V1_H


