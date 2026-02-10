#ifndef UVCC_NCCL_FRAMES_V1_H
#define UVCC_NCCL_FRAMES_V1_H
// UVCC_REQ_GROUP: uvcc_group_516fb18a3bfe557c
/*
UVCC v1 — OPEN_BOOL / OPEN_ARITH NCCL-style frame headers + payload headers (byte-exact).

Source of truth:
- `research/privacy_new.txt` §1.1 (OPEN_BOOL frame header) and §1.2 (OPEN_BOOL payload)
- `research/privacy_new.txt` §1.1 (OPEN_ARITH frame header) and §1.2 (OPEN_ARITH payload + chunkdesc)

These structs freeze the **wire bytes** for the specialized OPEN_* frames; transport may be relay or NCCL,
but the bytes are canonical for transcript purposes.
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

// Message type codes (v1).
enum uvcc_nccl_msg_type_v1 {
  UVCC_MSG_OPEN_BOOL  = 0x0101,
  UVCC_MSG_OPEN_ARITH = 0x0102,
};

// Frame header (80 bytes).
typedef struct UVCC_PACKED {
  uint8_t  magic[8];           // "UVCCFRM1"
  uint16_t version_le;         // = 1
  uint16_t msg_type_le;        // uvcc_nccl_msg_type_v1
  uint32_t flags_le;           // bit0=SEND, bit1=RECV, bit2=RETRANSMIT, else 0

  uint8_t  sid_hash32[32];     // H(sid) = 32 bytes (binding)
  uint64_t stream_id_le;       // deterministic
  uint32_t sgir_op_id_le;      // operation id
  uint16_t src_party_le;       // 0..2
  uint16_t dst_party_le;       // 0..2

  uint32_t seq_le;             // per-(stream_id,dir) monotonically increasing
  uint32_t payload_len_le;     // bytes after header
  uint32_t payload_crc32c_le;  // optional; 0 if disabled
  uint32_t reserved0_le;       // = 0
} uvcc_nccl_frame_hdr_v1;

_Static_assert(sizeof(uvcc_nccl_frame_hdr_v1) == 80, "uvcc_nccl_frame_hdr_v1 must be 80 bytes");

// OPEN_BOOL payload header (16 bytes), followed by packed u32 words[V][W].
typedef struct UVCC_PACKED {
  uint16_t vec_count_le;   // V vectors in this frame (>=1)
  uint16_t word_bits_le;   // = 32 for v1
  uint32_t n_bits_le;      // N bits per vector
  uint32_t n_words_le;     // W = ceil(N/32)
  uint32_t reserved0_le;   // = 0
} uvcc_open_bool_payload_v1;

_Static_assert(sizeof(uvcc_open_bool_payload_v1) == 16, "uvcc_open_bool_payload_v1 must be 16 bytes");

// OPEN_ARITH payload header (32 bytes).
typedef struct UVCC_PACKED {
  uint16_t vec_count_le;       // V vectors in this frame (>=1)
  uint16_t dtype_code_le;      // dtype code (see doc §0.2)
  uint32_t payload_flags_le;   // bit0 MOD_2POW, bit1 PACKED, bit2 CHUNKED

  uint32_t elem_bytes_le;      // 8/4/2/1
  uint32_t elem_bits_le;       // 64/32/16/8

  uint32_t reserved0_le;       // = 0
  uint32_t reserved1_le;       // = 0
  uint32_t reserved2_le;       // = 0
  uint32_t reserved3_le;       // = 0  (padding to 32 bytes; v1 fixed)
} uvcc_open_arith_payload_v1;

_Static_assert(sizeof(uvcc_open_arith_payload_v1) == 32, "uvcc_open_arith_payload_v1 must be 32 bytes");

// OPEN_ARITH chunk descriptor (16 bytes), present iff FLAG_CHUNKED.
typedef struct UVCC_PACKED {
  uint32_t vec_id_le;
  uint32_t chunk_start_le;
  uint32_t chunk_elems_le;
  uint32_t total_elems_le;
} uvcc_open_arith_chunkdesc_v1;

_Static_assert(sizeof(uvcc_open_arith_chunkdesc_v1) == 16, "uvcc_open_arith_chunkdesc_v1 must be 16 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_NCCL_FRAMES_V1_H


