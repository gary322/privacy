#ifndef UVCC_RSS_PAIRS_V1_H
#define UVCC_RSS_PAIRS_V1_H
/*
UVCC v1 — canonical replicated-share pair encodings (byte-exact).

Source of truth: `research/privacy_new.txt` (B2A/A2B pack definitions) and shared conventions.
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

// Boolean RSS pair for one bit (stored as bytes 0/1).
// Size: 2 bytes.
typedef struct UVCC_PACKED {
  uint8_t lo;  // share component held by this party (x_i), in {0,1}
  uint8_t hi;  // next share component (x_{i+1}), in {0,1}
} uvcc_rss_u1_pair_v1;

_Static_assert(sizeof(uvcc_rss_u1_pair_v1) == 2, "uvcc_rss_u1_pair_v1 must be 2 bytes");

// Arithmetic RSS pair in Z/2^64 (little-endian on wire).
// Size: 16 bytes.
typedef struct UVCC_PACKED {
  uint64_t lo_le;  // x_i
  uint64_t hi_le;  // x_{i+1}
} uvcc_rss_u64_pair_v1;

_Static_assert(sizeof(uvcc_rss_u64_pair_v1) == 16, "uvcc_rss_u64_pair_v1 must be 16 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_RSS_PAIRS_V1_H


