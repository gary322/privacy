#ifndef UVCC_PRG_CTX_V1_H
#define UVCC_PRG_CTX_V1_H
// UVCC_REQ_GROUP: uvcc_group_6efdfddadfd25e55
/*
UVCC v1 — device-visible PRG context struct (byte-exact) and seed type.

Source of truth: `research/privacy_new.txt` §3.1–§3.3.
*/

#include <stdint.h>

#if defined(_MSC_VER)
#define UVCC_ALIGN16 __declspec(align(16))
#define UVCC_ALIGNOF(T) __alignof(T)
#elif defined(__GNUC__) || defined(__clang__)
#define UVCC_ALIGN16 __attribute__((aligned(16)))
#define UVCC_ALIGNOF(T) __alignof__(T)
#else
#define UVCC_ALIGN16
#define UVCC_ALIGNOF(T) _Alignof(T)
#endif

// PRG type selectors (v1). Numeric values are u32.
enum UVCC_PRG_TYPE_v1 {
  UVCC_PRG_AES128_FIXEDKEY_v1   = 1,
  UVCC_PRG_CHACHA20_FIXEDKEY_v1 = 2,
};

// PRG implementation selectors (v1). Numeric values are u32.
enum UVCC_PRG_IMPL_v1 {
  UVCC_PRG_IMPL_PER_THREAD_v1 = 1,
  UVCC_PRG_IMPL_WARP_BATCH_v1 = 2,
  UVCC_PRG_IMPL_BITSLICE32_v1 = 3,
};

// 16-byte seed type (aligned to 16).
typedef struct UVCC_ALIGN16 {
  uint8_t b[16];
} DPFSeed16_v1;

_Static_assert(sizeof(DPFSeed16_v1) == 16, "DPFSeed16_v1 must be 16 bytes");
_Static_assert(UVCC_ALIGNOF(DPFSeed16_v1) == 16, "DPFSeed16_v1 must be 16-byte aligned");

// PRG context (aligned to 16, size=256 bytes).
typedef struct UVCC_ALIGN16 {
  uint32_t prg_type;   // UVCC_PRG_TYPE_v1
  uint32_t impl;       // UVCC_PRG_IMPL_v1
  uint32_t flags;      // reserved v1, must be 0
  uint32_t reserved0;  // must be 0

  uint8_t domain_tag16[16];  // public constant: SHA256("UVCC_G_V1")[0..15]

  uint8_t aes_rk[176];       // AES-128 round keys, expanded on host
  uint32_t chacha_k[8];      // ChaCha20 key words (LE u32)

  uint8_t pad[16];           // reserved for future extensions (must be 0 in v1)
} UVCC_PRG_CTX_v1;

_Static_assert(sizeof(UVCC_PRG_CTX_v1) == 256, "UVCC_PRG_CTX_v1 must be 256 bytes");
_Static_assert(UVCC_ALIGNOF(UVCC_PRG_CTX_v1) == 16, "UVCC_PRG_CTX_v1 must be 16-byte aligned");

#undef UVCC_ALIGNOF
#undef UVCC_ALIGN16

#endif  // UVCC_PRG_CTX_V1_H


