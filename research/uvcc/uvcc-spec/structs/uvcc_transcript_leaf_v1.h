#ifndef UVCC_TRANSCRIPT_LEAF_V1_H
#define UVCC_TRANSCRIPT_LEAF_V1_H
/*
UVCC v1 — transcript leaf bytes for reorder-independent relay acceptance.

Source of truth: `research/privacy_new.txt` §6.3 (Leaf bytes, canonical 128 bytes).

Hashing (v1 profile):
- `leaf_key` and `leaf_digest` use SHA256 with the domain-sep strings specified in the doc,
  per `research/uvcc/uvcc-spec/profiles/uvcc_profile_v1.md`.
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

// Canonical leaf bytes (128 bytes), stored keyed by leaf_key for order-independent Merkle roots.
typedef struct UVCC_PACKED {
  uint16_t leaf_type_le;     // v1: see doc §6.2 (e.g., 0x4101..0x4106)
  uint16_t version_le;       // = 1
  uint32_t epoch_id32_le;    // epoch identifier
  uint64_t stream_id64_le;   // stream identifier
  uint32_t msg_id32_le;      // message id within stream/epoch
  uint32_t op_id32_le;       // sgir op id (or 0)
  uint8_t  src_party;        // 0..2
  uint8_t  dst_party;        // 0..2
  uint8_t  msg_class;        // 0x21/0x22/0x23 (per doc)
  uint8_t  payload_kind;     // 0x01/0x02; 0 for ACK/NACK
  uint32_t chunk_idx_le;     // 0..chunk_count-1
  uint32_t chunk_count_le;   // >=1
  uint32_t payload_bytes_le; // payload bytes for this chunk/frame
  uint64_t sid_hash64_le;    // truncated sid hash (wire binding)
  uint8_t  frame_hash32[32];   // for DATA or zeros (or H(ack_bytes) if used)
  uint8_t  control_hash32[32]; // for ACK/NACK (or zero for DATA)
  uint8_t  reserved[16];       // = 0
} uvcc_leaf_v1;

_Static_assert(sizeof(uvcc_leaf_v1) == 128, "uvcc_leaf_v1 must be 128 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_TRANSCRIPT_LEAF_V1_H


