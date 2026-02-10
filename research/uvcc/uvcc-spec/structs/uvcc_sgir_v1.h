#ifndef UVCC_SGIR_V1_H
#define UVCC_SGIR_V1_H
// UVCC_REQ_GROUP: uvcc_group_b50b9410e18bf310,uvcc_group_c3c058cd62b15ae3
/*
UVCC v1 — SGIR module container structs (byte-exact).

Source of truth: `research/privacy_new.txt` §A (SGIR v0 wire format).

NOTE: The source doc contains inconsistent size annotations for some SGIR structs.
v1 freezes the field layout exactly as written in the struct definitions and treats the
size annotations as non-normative comments. The `header_bytes` field MUST equal
`sizeof(SGIR_ModuleHeader_v1)` for canonical hashing and parsing.
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

// Hash algorithm identifiers for SGIR modules.
enum uvcc_sgir_hash_alg_v1 {
  UVCC_SGIR_HASH_SHA256   = 1,
  UVCC_SGIR_HASH_BLAKE3_256 = 2,
  UVCC_SGIR_HASH_KECCAK256  = 3,
};

// Endianness identifiers (v1 only supports little endian).
enum uvcc_sgir_endianness_v1 {
  UVCC_SGIR_ENDIAN_LITTLE = 1,
};

// SGIR module header (packed).
typedef struct UVCC_PACKED {
  uint8_t  magic[4];          // 'S','G','I','R'
  uint16_t ver_major_le;      // = 0
  uint16_t ver_minor_le;      // = 1

  uint32_t header_bytes_le;   // = sizeof(SGIR_ModuleHeader_v1)
  uint32_t flags_le;          // bitset

  uint32_t section_count_le;    // N
  uint32_t section_table_off_le;// offset to section table

  uint8_t  hash_alg;          // uvcc_sgir_hash_alg_v1
  uint8_t  endianness;        // uvcc_sgir_endianness_v1
  uint16_t reserved0_le;      // = 0

  uint64_t file_bytes_le;     // total length in bytes

  uint8_t  module_uuid[16];   // UUIDv4 bytes (not a hash)
  uint8_t  reserved1[16];     // = 0

  uint8_t  header_checksum[8];// v1: set to 0 (reserved)
} SGIR_ModuleHeader_v1;

_Static_assert(sizeof(SGIR_ModuleHeader_v1) == 76, "SGIR_ModuleHeader_v1 size mismatch");

// Section table entry (packed).
typedef struct UVCC_PACKED {
  uint32_t kind_le;       // enum SGIR_SectionKind (u32)
  uint32_t flags_le;      // per-section flags (usually 0)
  uint64_t offset_le;     // file offset to section payload (8-byte aligned)
  uint64_t length_le;     // payload length in bytes
  uint8_t  sha256[32];    // OPTIONAL: either all-zero OR SHA256(payload)
} SGIR_SectionEntry_v1;

_Static_assert(sizeof(SGIR_SectionEntry_v1) == 56, "SGIR_SectionEntry_v1 size mismatch");

// Minimal required section kinds (u32) for execution (see doc §A.3.4).
enum uvcc_sgir_section_kind_v1 {
  SGIR_SECT_STRTAB = 1,
  SGIR_SECT_SYMTAB = 2,
  SGIR_SECT_TYPETAB = 3,
  SGIR_SECT_VALTAB = 4,
  SGIR_SECT_FUNTAB = 5,
  SGIR_SECT_CODE = 6,
  SGIR_SECT_CONST = 7,
  SGIR_SECT_ATTR = 8,
  SGIR_SECT_SIG = 9,
  SGIR_SECT_DEBUG = 10,
};

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // UVCC_SGIR_V1_H


