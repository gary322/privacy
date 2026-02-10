#ifndef UVCC_FSS_KEYREC_V1_H
#define UVCC_FSS_KEYREC_V1_H
// UVCC_REQ_GROUP: uvcc_group_2c46210f0240cf5f,uvcc_group_c66222b84339eca1,uvcc_group_22930b2f16ac685e
/*
UVCC v1 — byte-exact containers for FSS key records (DPF/DCF).

Source of truth: `research/privacy_new.txt` §5 (Containers): keyrec header + DPF/DCF bodies.
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

enum uvcc_fss_prim_type_v1 {
  UVCC_FSS_PRIM_DPF = 0x21,
  UVCC_FSS_PRIM_DCF = 0x22,
};

enum uvcc_fss_party_edge_v1 {
  UVCC_EDGE_01 = 0,  // P0-P1
  UVCC_EDGE_12 = 1,  // P1-P2
  UVCC_EDGE_20 = 2,  // P2-P0
};

enum uvcc_fss_prg_id_v1 {
  UVCC_PRG_AES128  = 1,
  UVCC_PRG_CHACHA12 = 2,
};

// Common key record header (64 bytes).
typedef struct UVCC_PACKED {
  uint8_t  magic[8];        // "UVCCFSS1"
  uint16_t version_le;      // = 1
  uint8_t  prim_type;       // uvcc_fss_prim_type_v1
  uint8_t  party_edge;      // uvcc_fss_party_edge_v1
  uint8_t  w;               // 8 or 16
  uint8_t  prg_id;          // uvcc_fss_prg_id_v1
  uint16_t flags_le;        // bit0=invert(DCF), bit1=payload_mask_enabled
  uint64_t fss_id_le;       // namespace id
  uint8_t  sid_hash32[32];  // must match directory header
  uint16_t cw_count_le;     // = w
  uint16_t cw_stride_le;    // bytes per CW record (= 17)
  uint32_t reserved0_le;    // = 0
} uvcc_fss_keyrec_hdr_v1;

_Static_assert(sizeof(uvcc_fss_keyrec_hdr_v1) == 64, "uvcc_fss_keyrec_hdr_v1 must be 64 bytes");

// DPF/DCF correction word record (17 bytes): sigma[16] + tau_mask.
typedef struct UVCC_PACKED {
  uint8_t sigma[16];
  uint8_t tau_mask;  // bit0=tauL, bit1=tauR
} uvcc_dpf_cw_v1;

_Static_assert(sizeof(uvcc_dpf_cw_v1) == 17, "uvcc_dpf_cw_v1 must be 17 bytes");

// Immediately after uvcc_fss_keyrec_hdr_v1 in keyrec bytes:
// - root_seed[16]
// - root_t[1] (bit0 used)
// - uvcc_dpf_cw_v1 CW[w]
// - for DCF only: payload_mask_u64 (8 bytes LE)

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_FSS_KEYREC_V1_H


