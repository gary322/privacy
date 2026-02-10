#ifndef UVCC_FSS_BLOCK_V1_H
#define UVCC_FSS_BLOCK_V1_H
/*
UVCC v1 — unified per-step FSS block ("FSB1") wire structs.

Source of truth: `research/privacy_new.txt` §3–§9 (FSB1, records, hashing rules).
All fields are little-endian on the wire. Structs are packed.
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

// UVCC_FSSBlockHeader_v1 (88 bytes).
typedef struct UVCC_PACKED {
  uint32_t magic_le;         // 'FSB1' = 0x31425346 (LE)
  uint16_t version_le;       // 1
  uint16_t header_bytes_le;  // sizeof(UVCC_FSSBlockHeader_v1)

  uint64_t block_id_le;      // (epoch<<32 | step)
  uint32_t epoch_le;
  uint32_t step_le;

  uint32_t record_count_le;  // total DPF+DCF records
  uint32_t reserved0_le;     // 0

  uint64_t records_offset_le; // from start of block
  uint64_t payload_offset_le; // from start of block
  uint64_t payload_bytes_le;  // payload size

  // Optional record/payload hashes (implementation-defined; may be 0).
  uint64_t records_hash_hi_le;
  uint64_t records_hash_lo_le;
  uint64_t payload_hash_hi_le;
  uint64_t payload_hash_lo_le;
} uvcc_fss_block_hdr_v1;

_Static_assert(sizeof(uvcc_fss_block_hdr_v1) == 88, "uvcc_fss_block_hdr_v1 must be 88 bytes");

// UVCC_FSSRecord_v1 (52 bytes).
typedef struct UVCC_PACKED {
  uint64_t fss_id_le;       // global namespace key (DPF+DCF share this space)
  uint32_t sgir_op_id_le;   // SGIR instruction consuming this key
  uint16_t lane_le;         // SIMD lane (0..), or 0xFFFF = broadcast
  uint8_t  fss_kind;        // 1=DPF_POINT, 2=DCF_LT
  uint8_t  reserved0;       // 0

  uint16_t domain_bits_le;  // w
  uint16_t range_bits_le;   // 1 for bool

  uint8_t  share_mode;      // 1=BOOL_XOR_OUTPUT (recommended)
  uint8_t  edge_mode;       // 0=3-party keys, 1=edge-only keys
  uint8_t  edge_id;         // if edge_mode=1: 0=01,1=12,2=20 ; else 0
  uint8_t  flags;           // bit0=bitpacked_out, bit1=wire_has_hash

  uint32_t key_bytes_le;    // size of key blob bytes (party-local)
  uint64_t key_offset_le;   // offset from payload_offset to key blob

  uint64_t key_hash_hi_le;  // optional if flags.bit1=1
  uint64_t key_hash_lo_le;
} uvcc_fss_record_v1;

_Static_assert(sizeof(uvcc_fss_record_v1) == 52, "uvcc_fss_record_v1 must be 52 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_FSS_BLOCK_V1_H


