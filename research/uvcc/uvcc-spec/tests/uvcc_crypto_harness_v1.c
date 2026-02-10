// uvcc_crypto_harness_v1.c
// Canonical CPU test harness for UVCC crypto primitives v1.
//
// What it does:
//  1) Expands an AES-128 key into rk[11] as uint4[11] in the exact UVCC u128 layout.
//  2) Runs uvcc_aes128_enc_fk_v1_cpu(plaintext, rk) and prints ciphertext bytes + uint4 words.
//  3) Runs uvcc_chacha12_block_v1(...) and prints 64-byte keystream bytes.
//
// Compile:
//   cc -O2 -std=c99 -Wall -Wextra -o uvcc_crypto_harness_v1 uvcc_crypto_harness_v1.c
//
// Run:
//   ./uvcc_crypto_harness_v1
//
// Notes:
// - uint4 layout matches the earlier UVCC u128 rule:
//     x=LE32(b[0..3]), y=LE32(b[4..7]), z=LE32(b[8..11]), w=LE32(b[12..15]).
// - AES state mapping is standard: S[r][c] = in[4*c + r].
// - AES key schedule is byte-based (no endian ambiguity) and produces round keys in the same
//   byte index space as the AES state vector (b[0..15]) used by this harness.
// - ChaCha is ChaCha12: 12 rounds = 6 double-rounds, standard RFC word/byte mapping.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// -------------------------
// "uint4" (CUDA-compatible shape)
// -------------------------
typedef struct uint4_s {
  uint32_t x, y, z, w;
} uint4;

// -------------------------
// Little-endian helpers
// -------------------------
static inline uint32_t le32_load(const uint8_t *p) {
  return ((uint32_t)p[0]) |
         ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
static inline void le32_store(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)((v >> 0) & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF);
  p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static inline uint4 encode_u128_le(const uint8_t b16[16]) {
  uint4 u;
  u.x = le32_load(&b16[0]);
  u.y = le32_load(&b16[4]);
  u.z = le32_load(&b16[8]);
  u.w = le32_load(&b16[12]);
  return u;
}
static inline void decode_u128_le(uint8_t b16[16], uint4 u) {
  le32_store(&b16[0],  u.x);
  le32_store(&b16[4],  u.y);
  le32_store(&b16[8],  u.z);
  le32_store(&b16[12], u.w);
}

// -------------------------
// Printing utilities
// -------------------------
static void print_hex(const char *label, const uint8_t *buf, size_t n) {
  printf("%s (%zu bytes):\n", label, n);
  for (size_t i = 0; i < n; i++) {
    printf("%02x", buf[i]);
    if ((i + 1) % 16 == 0) printf("\n");
    else if ((i + 1) % 4 == 0) printf(" ");
  }
  if (n % 16 != 0) printf("\n");
}

static void print_uint4_words(const char *label, const uint4 u) {
  printf("%s uint4 = { x=%08x, y=%08x, z=%08x, w=%08x }\n",
         label, u.x, u.y, u.z, u.w);
}

static void print_round_keys_as_bytes_and_words(const uint4 rk[11]) {
  for (int r = 0; r < 11; r++) {
    uint8_t kb[16];
    decode_u128_le(kb, rk[r]);
    char lbl[64];
    snprintf(lbl, sizeof(lbl), "AES rk[%d]", r);
    print_uint4_words(lbl, rk[r]);
    print_hex(lbl, kb, 16);
  }
}

// ============================================================
// AES-128 (fixed-key encrypt with pre-expanded round keys)
// ============================================================

// AES S-box (standard)
static const uint8_t AES_SBOX[256] = {
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
  0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
  0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
  0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
  0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
  0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
  0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
  0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
  0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
  0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
  0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// AES-128 rcon values (for rounds 1..10)
static const uint8_t AES_RCON[11] = {
  0x00, // unused (index 0)
  0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
};

static inline uint8_t xtime(uint8_t a) {
  return (uint8_t)((a << 1) ^ ((a & 0x80) ? 0x1b : 0x00));
}
static inline uint8_t mul2(uint8_t a) { return xtime(a); }
static inline uint8_t mul3(uint8_t a) { return (uint8_t)(xtime(a) ^ a); }

static void aes_sub_bytes(uint8_t st[16]) {
  for (int i = 0; i < 16; i++) st[i] = AES_SBOX[st[i]];
}

static void aes_shift_rows(uint8_t st[16]) {
  // Permutation as specified in the ABI v1 description
  uint8_t in[16];
  memcpy(in, st, 16);

  st[ 0]=in[ 0]; st[ 4]=in[ 4]; st[ 8]=in[ 8]; st[12]=in[12];
  st[ 1]=in[ 5]; st[ 5]=in[ 9]; st[ 9]=in[13]; st[13]=in[ 1];
  st[ 2]=in[10]; st[ 6]=in[14]; st[10]=in[ 2]; st[14]=in[ 6];
  st[ 3]=in[15]; st[ 7]=in[ 3]; st[11]=in[ 7]; st[15]=in[11];
}

static void aes_mix_columns(uint8_t st[16]) {
  for (int c = 0; c < 4; c++) {
    uint8_t a0 = st[4*c + 0];
    uint8_t a1 = st[4*c + 1];
    uint8_t a2 = st[4*c + 2];
    uint8_t a3 = st[4*c + 3];

    uint8_t b0 = (uint8_t)(mul2(a0) ^ mul3(a1) ^ a2 ^ a3);
    uint8_t b1 = (uint8_t)(a0 ^ mul2(a1) ^ mul3(a2) ^ a3);
    uint8_t b2 = (uint8_t)(a0 ^ a1 ^ mul2(a2) ^ mul3(a3));
    uint8_t b3 = (uint8_t)(mul3(a0) ^ a1 ^ a2 ^ mul2(a3));

    st[4*c + 0] = b0;
    st[4*c + 1] = b1;
    st[4*c + 2] = b2;
    st[4*c + 3] = b3;
  }
}

static void aes_add_round_key(uint8_t st[16], const uint8_t rk_bytes[16]) {
  for (int i = 0; i < 16; i++) st[i] ^= rk_bytes[i];
}

// AES-128 key expansion into rk[11] as uint4[11] in UVCC u128 layout.
static void uvcc_aes128_expand_key_to_rk11_u128_v1(const uint8_t key16[16], uint4 rk_u128_11[11]) {
  // Word array w[44], each word is 4 bytes.
  // AES-128: 4 initial words + 40 derived words = 44 words total.
  uint8_t w[44][4];

  // Initialize w[0..3] from the key bytes (word i = key[4i..4i+3])
  for (int i = 0; i < 4; i++) {
    w[i][0] = key16[4*i + 0];
    w[i][1] = key16[4*i + 1];
    w[i][2] = key16[4*i + 2];
    w[i][3] = key16[4*i + 3];
  }

  for (int i = 4; i < 44; i++) {
    uint8_t temp[4];
    temp[0] = w[i-1][0];
    temp[1] = w[i-1][1];
    temp[2] = w[i-1][2];
    temp[3] = w[i-1][3];

    if ((i % 4) == 0) {
      // RotWord
      uint8_t t0 = temp[0];
      temp[0] = temp[1];
      temp[1] = temp[2];
      temp[2] = temp[3];
      temp[3] = t0;

      // SubWord
      temp[0] = AES_SBOX[temp[0]];
      temp[1] = AES_SBOX[temp[1]];
      temp[2] = AES_SBOX[temp[2]];
      temp[3] = AES_SBOX[temp[3]];

      // Rcon
      temp[0] ^= AES_RCON[i/4];
    }

    // w[i] = w[i-4] XOR temp
    w[i][0] = (uint8_t)(w[i-4][0] ^ temp[0]);
    w[i][1] = (uint8_t)(w[i-4][1] ^ temp[1]);
    w[i][2] = (uint8_t)(w[i-4][2] ^ temp[2]);
    w[i][3] = (uint8_t)(w[i-4][3] ^ temp[3]);
  }

  // Round key r uses words w[4r..4r+3], laid out in column-major state order:
  // roundKey[4*c + row] = w[4r + c][row]
  for (int r = 0; r < 11; r++) {
    uint8_t rk_bytes[16];
    for (int c = 0; c < 4; c++) {
      rk_bytes[4*c + 0] = w[4*r + c][0];
      rk_bytes[4*c + 1] = w[4*r + c][1];
      rk_bytes[4*c + 2] = w[4*r + c][2];
      rk_bytes[4*c + 3] = w[4*r + c][3];
    }
    rk_u128_11[r] = encode_u128_le(rk_bytes);
  }
}

// AES-128 encrypt with fixed pre-expanded keys (rk[0..10]) in uint4 layout.
static uint4 uvcc_aes128_enc_fk_v1_cpu(uint4 in_u128, const uint4 rk_u128_11[11]) {
  uint8_t st[16];
  uint8_t rk[16];

  decode_u128_le(st, in_u128);

  // round 0
  decode_u128_le(rk, rk_u128_11[0]);
  aes_add_round_key(st, rk);

  // rounds 1..9
  for (int r = 1; r <= 9; r++) {
    aes_sub_bytes(st);
    aes_shift_rows(st);
    aes_mix_columns(st);
    decode_u128_le(rk, rk_u128_11[r]);
    aes_add_round_key(st, rk);
  }

  // round 10 (no MixColumns)
  aes_sub_bytes(st);
  aes_shift_rows(st);
  decode_u128_le(rk, rk_u128_11[10]);
  aes_add_round_key(st, rk);

  return encode_u128_le(st);
}

// ============================================================
// ChaCha12 block (RFC mapping, but 12 rounds)
// ============================================================

static inline uint32_t rotl32(uint32_t v, int n) {
  return (v << n) | (v >> (32 - n));
}

#define QR(a,b,c,d) do {           \
  x[a] += x[b]; x[d] ^= x[a]; x[d] = rotl32(x[d], 16); \
  x[c] += x[d]; x[b] ^= x[c]; x[b] = rotl32(x[b], 12); \
  x[a] += x[b]; x[d] ^= x[a]; x[d] = rotl32(x[d],  8); \
  x[c] += x[d]; x[b] ^= x[c]; x[b] = rotl32(x[b],  7); \
} while(0)

static void uvcc_chacha12_block_v1(uint8_t out64[64],
                                   const uint32_t key32[8],
                                   uint32_t counter32,
                                   const uint8_t nonce12[12]) {
  uint32_t state[16];
  uint32_t x[16];

  // constants "expand 32-byte k"
  state[0] = 0x61707865;
  state[1] = 0x3320646e;
  state[2] = 0x79622d32;
  state[3] = 0x6b206574;

  // key words (already little-endian word values)
  state[4]  = key32[0];
  state[5]  = key32[1];
  state[6]  = key32[2];
  state[7]  = key32[3];
  state[8]  = key32[4];
  state[9]  = key32[5];
  state[10] = key32[6];
  state[11] = key32[7];

  // counter
  state[12] = counter32;

  // nonce (12 bytes -> 3 u32 LE words)
  state[13] = le32_load(&nonce12[0]);
  state[14] = le32_load(&nonce12[4]);
  state[15] = le32_load(&nonce12[8]);

  // working state
  for (int i = 0; i < 16; i++) x[i] = state[i];

  // 12 rounds = 6 double-rounds
  for (int i = 0; i < 6; i++) {
    // column rounds
    QR(0, 4,  8, 12);
    QR(1, 5,  9, 13);
    QR(2, 6, 10, 14);
    QR(3, 7, 11, 15);

    // diagonal rounds
    QR(0, 5, 10, 15);
    QR(1, 6, 11, 12);
    QR(2, 7,  8, 13);
    QR(3, 4,  9, 14);
  }

  // add original state and serialize LE words
  for (int i = 0; i < 16; i++) {
    uint32_t v = x[i] + state[i];
    out64[4*i + 0] = (uint8_t)((v >> 0) & 0xFF);
    out64[4*i + 1] = (uint8_t)((v >> 8) & 0xFF);
    out64[4*i + 2] = (uint8_t)((v >> 16) & 0xFF);
    out64[4*i + 3] = (uint8_t)((v >> 24) & 0xFF);
  }
}

#undef QR

// ============================================================
// Main: Known-answer AES test + deterministic ChaCha12 print
// ============================================================

int main(void) {
  // --- AES-128 known-answer test (NIST SP 800-38A / FIPS-197 common vector) ---
  const uint8_t aes_key[16] = {
    0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07,
    0x08,0x09,0x0a,0x0b, 0x0c,0x0d,0x0e,0x0f
  };
  const uint8_t aes_pt[16] = {
    0x00,0x11,0x22,0x33, 0x44,0x55,0x66,0x77,
    0x88,0x99,0xaa,0xbb, 0xcc,0xdd,0xee,0xff
  };
  const uint8_t aes_expected_ct[16] = {
    0x69,0xc4,0xe0,0xd8, 0x6a,0x7b,0x04,0x30,
    0xd8,0xcd,0xb7,0x80, 0x70,0xb4,0xc5,0x5a
  };

  uint4 rk[11];
  uvcc_aes128_expand_key_to_rk11_u128_v1(aes_key, rk);

  printf("=== UVCC AES-128 / ChaCha12 CPU Harness v1 ===\n\n");

  print_hex("AES key", aes_key, 16);
  print_hex("AES plaintext", aes_pt, 16);

  // Print rk[0..10] (both uint4 words and bytes) so GPU can be validated exactly.
  printf("\n--- AES round keys (rk[0..10]) in uint4 + bytes (UVCC u128 layout) ---\n");
  print_round_keys_as_bytes_and_words(rk);

  // Run AES encrypt
  uint4 pt_u = encode_u128_le(aes_pt);
  uint4 ct_u = uvcc_aes128_enc_fk_v1_cpu(pt_u, rk);

  uint8_t aes_ct[16];
  decode_u128_le(aes_ct, ct_u);

  printf("\n--- AES encrypt result ---\n");
  print_uint4_words("AES plaintext", pt_u);
  print_uint4_words("AES ciphertext", ct_u);
  print_hex("AES ciphertext", aes_ct, 16);

  // Print expected CT and a simple compare
  print_hex("AES expected ciphertext", aes_expected_ct, 16);
  int ok = (memcmp(aes_ct, aes_expected_ct, 16) == 0);
  printf("AES KAT match: %s\n", ok ? "OK" : "FAIL");
  if (!ok) return 2;

  // --- ChaCha12 deterministic block output ---
  // Use a simple deterministic key/nonce/counter (RFC-style key bytes 0..31).
  uint8_t chacha_key_bytes[32];
  for (int i = 0; i < 32; i++) chacha_key_bytes[i] = (uint8_t)i;

  uint32_t chacha_key32[8];
  for (int i = 0; i < 8; i++) chacha_key32[i] = le32_load(&chacha_key_bytes[4*i]);

  // RFC 8439 example nonce (12 bytes): 000000090000004a00000000 (just a common choice)
  const uint8_t chacha_nonce12[12] = {
    0x00,0x00,0x00,0x09, 0x00,0x00,0x00,0x4a, 0x00,0x00,0x00,0x00
  };
  const uint32_t chacha_counter = 1;

  uint8_t chacha_out[64];
  uvcc_chacha12_block_v1(chacha_out, chacha_key32, chacha_counter, chacha_nonce12);

  printf("\n--- ChaCha12 block ---\n");
  print_hex("ChaCha key (bytes)", chacha_key_bytes, 32);
  print_hex("ChaCha nonce12", chacha_nonce12, 12);
  printf("ChaCha counter32: %u (0x%08x)\n", chacha_counter, chacha_counter);
  print_hex("ChaCha12 out64", chacha_out, 64);

  // Also show as four uint4 blocks in UVCC u128 layout (bytes 0..15, 16..31, 32..47, 48..63)
  printf("\nChaCha12 out64 as 4x uint4 (UVCC u128 layout):\n");
  for (int i = 0; i < 4; i++) {
    uint4 u = encode_u128_le(&chacha_out[16*i]);
    char lbl[64];
    snprintf(lbl, sizeof(lbl), "out_u4[%d]", i);
    print_uint4_words(lbl, u);
  }

  printf("\nDONE. Match these bytes/words on GPU for parity.\n");
  return 0;
}
