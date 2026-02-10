#ifndef UVCC_FSS_DIR_V1_H
#define UVCC_FSS_DIR_V1_H
/*
UVCC v1 — unified per-party FSS directory (DPF/DCF/OP_LUT) layout.

Source of truth: `research/privacy_new.txt` §1.2 (Directory blob layout) / §8.1.
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

// Directory header (64 bytes).
typedef struct UVCC_PACKED {
  uint8_t  magic[8];         // "UVCCFSSD"
  uint16_t version_le;       // = 1
  uint8_t  party_id;         // 0/1/2
  uint8_t  flags;            // reserved, =0
  uint8_t  sid_hash32[32];   // chosen 32-byte sid hash (profile-v1: SHA256(sid))
  uint64_t epoch_le;         // monotonically increasing (local)
  uint32_t entry_count_le;   // number of entries
  uint16_t entry_stride_le;  // = 32
  uint8_t  reserved[6];      // =0
} uvcc_fss_dir_header_v1;

_Static_assert(sizeof(uvcc_fss_dir_header_v1) == 64, "uvcc_fss_dir_header_v1 must be 64 bytes");

// Directory entry (32 bytes).
typedef struct UVCC_PACKED {
  uint64_t fss_id_le;        // lookup key
  uint8_t  prim_type;        // 0x21=DPF, 0x22=DCF, 0x32=OP_LUT
  uint8_t  w;                // 8 or 16
  uint8_t  out_kind;         // 0=bit, 1=bitvec, 2=u64mask
  uint8_t  prg_id;           // 1=AES128, 2=ChaCha12
  uint32_t keyrec_off_le;    // offset from blob start
  uint32_t keyrec_len_le;    // bytes
  uint32_t aux_off_le;       // optional aux offset (0 if none)
  uint32_t aux_len_le;       // optional aux length
  // NOTE: `privacy_new.txt` lists stream_id as 8 bytes but totals the entry as 32 bytes.
  // v1 profile interprets this as Trunc32(stream_id64).
  uint32_t stream_id32_le;   // derived stream id (optional)
} uvcc_fss_dir_entry_v1;

_Static_assert(sizeof(uvcc_fss_dir_entry_v1) == 32, "uvcc_fss_dir_entry_v1 must be 32 bytes");

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_FSS_DIR_V1_H


