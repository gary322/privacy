#ifndef UVCC_NETFRAME_V1_H
#define UVCC_NETFRAME_V1_H
// UVCC_REQ_GROUP: uvcc_group_df382033ede3f858
/*
UVCC v1 — canonical NetFrame transport header + segment header (byte-exact).

Source of truth: `research/privacy_new.txt` §C.1 (canonical NetFrame format).

NOTE (v1 profile): The spec text includes both `header_hash` and `payload_hash` descriptions.
To satisfy the stated fixed 128-byte FrameHeader requirement, v1 freezes a 128-byte header
containing `header_hash` (hash of FrameHeader+SegmentHeaders) and binds the payload via
transcript leaves (and/or higher-level frame hashing) rather than embedding a second 32-byte
payload hash into the fixed header.
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

// Frame header (128 bytes).
typedef struct UVCC_PACKED {
  uint8_t  magic[4];        // "UVCC"
  uint16_t ver_major_le;    // = 1
  uint16_t ver_minor_le;    // = 0

  uint16_t msg_kind_le;     // u16, matches transcript leaf kind
  uint16_t flags_le;        // fragmentation/compression/etc.

  uint8_t  job_id32[32];    // canonical job id bytes32

  uint32_t epoch_le;        // u32
  uint32_t step_le;         // u32
  uint16_t round_le;        // u16
  uint16_t reserved0_le;    // = 0

  uint8_t  sender;          // 0..2
  uint8_t  receiver;        // 0..2
  uint8_t  reserved1[2];    // = 0

  uint32_t seq_no_le;       // frame sequence number
  uint32_t frame_no_le;     // 0..frame_count-1
  uint32_t frame_count_le;  // >=1

  uint32_t segment_count_le; // number of segment headers
  uint32_t reserved2_le;     // = 0

  uint64_t header_bytes_le; // = 128 + segment_count*sizeof(SegmentHeader)
  uint64_t payload_bytes_le;// bytes after all headers

  uint8_t  header_hash32[32]; // H(FrameHeader || SegmentHeaders)
} uvcc_netframe_hdr_v1;

_Static_assert(sizeof(uvcc_netframe_hdr_v1) == 128, "uvcc_netframe_hdr_v1 must be 128 bytes");

// Segment header (48 bytes).
typedef struct UVCC_PACKED {
  uint32_t seg_kind_le;   // SegmentKind enum
  uint32_t object_id_le;  // open_id / tile_id / tensor_id depending on kind
  uint32_t sub_id_le;     // open_sub / tile_sub / 0
  uint32_t dtype_le;      // SGIR_DType as u32
  int32_t  fxp_frac_bits_le;
  uint32_t reserved0_le;  // = 0

  uint64_t offset_le;     // offset from start of payload region
  uint64_t length_le;     // bytes
  uint64_t reserved1_le;  // = 0
} uvcc_segment_hdr_v1;

_Static_assert(sizeof(uvcc_segment_hdr_v1) == 48, "uvcc_segment_hdr_v1 must be 48 bytes");

// SegmentKind enum values (v1 subset).
enum uvcc_segment_kind_v1 {
  UVCC_SEG_PAD = 1,
  UVCC_SEG_OPEN_SHARE_LO = 10,
  UVCC_SEG_OPEN_SHARE_META = 11,
  // 20.. reserved for TCF tiles
};

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_NETFRAME_V1_H


