#ifndef UVCC_KDF_INFO_V1_H
#define UVCC_KDF_INFO_V1_H
// UVCC_REQ_GROUP: uvcc_group_fce8a8bb25a7fd5f
/*
UVCC v1 — byte-exact KDF info struct for HKDF-Expand inputs.

Source of truth: `research/privacy_new.txt` §D (UVCC_KDF_INFO_v1).
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

// UVCC_KDF_INFO_v1 is exactly 83 bytes packed.
typedef struct UVCC_PACKED {
  uint8_t magic[8];         // "UVCCKDF\0"
  uint8_t version;          // = 1
  uint8_t kind;             // kind selector (0x01/0x02/0x10/...)
  uint8_t prg_type;         // 1=AES128_FIXEDKEY, 2=CHACHA20_FIXEDKEY
  uint8_t pair_id;          // 0,1,2 or 255
  uint8_t share_idx;        // 0,1,2 or 255
  uint8_t w_bits;           // 8 or 16
  uint8_t elem_bits;        // 8 or 16
  uint8_t reserved[4];      // must be 0
  uint8_t sid32[32];        // session id (32 bytes)
  uint8_t fss_id16[16];     // unified fss_id namespace (16 bytes)
  uint8_t table_hash16[16]; // optional binding; else 0
} UVCC_KDF_INFO_v1;

_Static_assert(sizeof(UVCC_KDF_INFO_v1) == 83, "UVCC_KDF_INFO_v1 must be 83 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_KDF_INFO_V1_H


