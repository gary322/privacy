#ifndef UVCC_POLICY_WIRE_V1_H
#define UVCC_POLICY_WIRE_V1_H
/*
UVCC v1 — canonical binary policy wire format (byte-exact structs).

Source of truth: `research/privacy_new.txt` §2.2 (Canonical binary policy wire format v1).

Notes:
- This header freezes only the fixed-size blocks. Variable/optional blocks are defined
  as fixed structs but may be omitted/present according to the policy layout rules.
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

// Policy header (64 bytes).
typedef struct UVCC_PACKED {
  uint8_t  magic[8];        // "UVCCPOL1"
  uint16_t version_le;      // = 1
  uint8_t  backend;         // 0=CRYPTO_CC_3PC, 1=GPU_TEE
  uint8_t  party_count;     // = 3
  uint32_t flags_le;        // bit0=require_transcript, bit1=require_epoch_roots, bit2=require_sks
  uint8_t  sid_hash32[32];  // v1 profile: keccak256(sid bytes)
  uint64_t job_id_le;       // optional u64 job number
  uint64_t reserved0_le;    // = 0
} uvcc_policy_hdr_v1;

_Static_assert(sizeof(uvcc_policy_hdr_v1) == 64, "uvcc_policy_hdr_v1 must be 64 bytes");

// Fixed digests block (128 bytes).
typedef struct UVCC_PACKED {
  uint8_t sgir_hash32[32];     // keccak256(SGIR wire module)
  uint8_t runtime_hash32[32];  // keccak256(runtime/compiler/container manifest)
  uint8_t fss_dir_hash32[32];  // keccak256(FSS directory manifest)
  uint8_t preproc_hash32[32];  // keccak256(preprocessing manifest)
} uvcc_policy_digests_v1;

_Static_assert(sizeof(uvcc_policy_digests_v1) == 128, "uvcc_policy_digests_v1 must be 128 bytes");

// Party record (64 bytes), repeated for party_id order 0,1,2.
typedef struct UVCC_PACKED {
  uint8_t party_id;            // 0..2
  uint8_t sig_scheme;          // 1=ECDSA_secp256k1
  uint8_t addr20[20];          // EVM address
  uint8_t attn_type;           // 0=none, 1=NVIDIA_GPU_TEE, 2=TDX, 3=SNP
  uint8_t attn_policy_hash32[32]; // keccak256(attestation policy)
  uint8_t reserved0[9];        // = 0
} uvcc_policy_party_v1;

_Static_assert(sizeof(uvcc_policy_party_v1) == 64, "uvcc_policy_party_v1 must be 64 bytes");

// Limits block (40 bytes), optional but canonical when present.
typedef struct UVCC_PACKED {
  uint64_t max_steps_le;
  uint32_t max_epochs_le;
  uint64_t max_wallclock_sec_le;
  uint64_t max_bytes_in_le;
  uint64_t max_bytes_out_le;
  uint32_t reserved0_le;
} uvcc_policy_limits_v1;

_Static_assert(sizeof(uvcc_policy_limits_v1) == 40, "uvcc_policy_limits_v1 must be 40 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_POLICY_WIRE_V1_H


