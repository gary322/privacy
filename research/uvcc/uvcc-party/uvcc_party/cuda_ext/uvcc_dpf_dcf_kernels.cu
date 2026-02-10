#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

// UVCC_REQ_GROUP: uvcc_group_acaeca358a192ce8,uvcc_group_b4712f8a9200c638,uvcc_group_28df78cb5ca5e330,uvcc_group_edc7705141457666,uvcc_group_b2a809ccbb581fc9,uvcc_group_e4c33796e7d46452,uvcc_group_732bc069128b3469,uvcc_group_d82b686532784bbf,uvcc_group_ced608f7f7149280,uvcc_group_37bb78042819626f

// -------------------------
// Small helpers
// -------------------------

static inline __device__ uint32_t rotl32(uint32_t x, int r) {
  return (x << r) | (x >> (32 - r));
}

static inline __device__ void qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  a += b;
  d ^= a;
  d = rotl32(d, 16);
  c += d;
  b ^= c;
  b = rotl32(b, 12);
  a += b;
  d ^= a;
  d = rotl32(d, 8);
  c += d;
  b ^= c;
  b = rotl32(b, 7);
}

static inline __device__ uint64_t load_le_u64(const uint8_t* p) {
  uint64_t v = 0;
  // Unaligned-safe load
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    v |= (uint64_t)p[i] << (8 * i);
  }
  return v;
}

static inline __device__ uint16_t load_le_u16(const uint8_t* p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

// -------------------------
// ChaCha12 PRG (per-spec mapping)
// -------------------------

static inline __device__ void chacha12_block(uint32_t out16[16], const uint32_t key8[8], uint32_t counter, const uint32_t nonce3[3]) {
  // constants "expand 32-byte k"
  uint32_t x0 = 0x61707865u;
  uint32_t x1 = 0x3320646eu;
  uint32_t x2 = 0x79622d32u;
  uint32_t x3 = 0x6b206574u;
  uint32_t x4 = key8[0];
  uint32_t x5 = key8[1];
  uint32_t x6 = key8[2];
  uint32_t x7 = key8[3];
  uint32_t x8 = key8[4];
  uint32_t x9 = key8[5];
  uint32_t x10 = key8[6];
  uint32_t x11 = key8[7];
  uint32_t x12 = counter;
  uint32_t x13 = nonce3[0];
  uint32_t x14 = nonce3[1];
  uint32_t x15 = nonce3[2];

  const uint32_t s0 = x0, s1 = x1, s2 = x2, s3 = x3;
  const uint32_t s4 = x4, s5 = x5, s6 = x6, s7 = x7;
  const uint32_t s8 = x8, s9 = x9, s10 = x10, s11 = x11;
  const uint32_t s12 = x12, s13 = x13, s14 = x14, s15 = x15;

  // 12 rounds = 6 double-rounds
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    // column rounds
    qr(x0, x4, x8, x12);
    qr(x1, x5, x9, x13);
    qr(x2, x6, x10, x14);
    qr(x3, x7, x11, x15);
    // diagonal rounds
    qr(x0, x5, x10, x15);
    qr(x1, x6, x11, x12);
    qr(x2, x7, x8, x13);
    qr(x3, x4, x9, x14);
  }

  out16[0] = x0 + s0;
  out16[1] = x1 + s1;
  out16[2] = x2 + s2;
  out16[3] = x3 + s3;
  out16[4] = x4 + s4;
  out16[5] = x5 + s5;
  out16[6] = x6 + s6;
  out16[7] = x7 + s7;
  out16[8] = x8 + s8;
  out16[9] = x9 + s9;
  out16[10] = x10 + s10;
  out16[11] = x11 + s11;
  out16[12] = x12 + s12;
  out16[13] = x13 + s13;
  out16[14] = x14 + s14;
  out16[15] = x15 + s15;
}

static inline __device__ void g_expand_chacha12_seed16(uint64_t seed_lo, uint64_t seed_hi, uint32_t depth, uint64_t& SL_lo, uint64_t& SL_hi, uint64_t& SR_lo, uint64_t& SR_hi, uint8_t& tL, uint8_t& tR) {
  // key32 = seed||seed (little-endian word mapping)
  uint32_t w0 = (uint32_t)(seed_lo & 0xffffffffull);
  uint32_t w1 = (uint32_t)(seed_lo >> 32);
  uint32_t w2 = (uint32_t)(seed_hi & 0xffffffffull);
  uint32_t w3 = (uint32_t)(seed_hi >> 32);
  uint32_t key8[8] = {w0, w1, w2, w3, w0, w1, w2, w3};

  // nonce12 = "G_SG2_v1" || LE32(depth)
  // "G_SG" -> 0x47535F47, "2_v1" -> 0x31765F32 in little-endian u32
  uint32_t nonce3[3] = {0x47535F47u, 0x31765F32u, depth};
  uint32_t out16[16];
  chacha12_block(out16, key8, 0u, nonce3);

  SL_lo = (uint64_t)out16[0] | ((uint64_t)out16[1] << 32);
  SL_hi = (uint64_t)out16[2] | ((uint64_t)out16[3] << 32);
  SR_lo = (uint64_t)out16[4] | ((uint64_t)out16[5] << 32);
  SR_hi = (uint64_t)out16[6] | ((uint64_t)out16[7] << 32);
  uint32_t w8 = out16[8];
  tL = (uint8_t)(w8 & 1u);
  tR = (uint8_t)((w8 >> 8) & 1u);
}

// -------------------------
// AES-128 PRG (foreign-key mode): seed16 is AES key, per privacy_new.txt §3.1 (prg_id=1)
// -------------------------

__device__ __constant__ uint8_t kAesSbox[256] = {
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

__device__ __constant__ uint8_t kAesRcon[10] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36};

static inline __device__ uint8_t xtime(uint8_t x) {
  return (uint8_t)((x << 1) ^ ((x & 0x80u) ? 0x1Bu : 0x00u));
}

static inline __device__ void aes128_key_expand(const uint8_t key16[16], uint8_t rk[176]) {
  #pragma unroll
  for (int i = 0; i < 16; i++) rk[i] = key16[i];
  int rcon_i = 0;
  for (int i = 16; i < 176; i += 4) {
    uint8_t t0 = rk[i - 4 + 0];
    uint8_t t1 = rk[i - 4 + 1];
    uint8_t t2 = rk[i - 4 + 2];
    uint8_t t3 = rk[i - 4 + 3];
    if ((i % 16) == 0) {
      // RotWord
      const uint8_t tmp = t0;
      t0 = t1; t1 = t2; t2 = t3; t3 = tmp;
      // SubWord
      t0 = kAesSbox[t0];
      t1 = kAesSbox[t1];
      t2 = kAesSbox[t2];
      t3 = kAesSbox[t3];
      // Rcon
      t0 ^= kAesRcon[rcon_i++];
    }
    rk[i + 0] = rk[i - 16 + 0] ^ t0;
    rk[i + 1] = rk[i - 16 + 1] ^ t1;
    rk[i + 2] = rk[i - 16 + 2] ^ t2;
    rk[i + 3] = rk[i - 16 + 3] ^ t3;
  }
}

static inline __device__ void aes_shift_rows(uint8_t s[16]) {
  // State is column-major: indices [0,4,8,12] row0; [1,5,9,13] row1; etc.
  uint8_t t[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) t[i] = s[i];
  // row0 unchanged
  s[0]  = t[0];  s[4]  = t[4];  s[8]  = t[8];  s[12] = t[12];
  // row1 left shift 1
  s[1]  = t[5];  s[5]  = t[9];  s[9]  = t[13]; s[13] = t[1];
  // row2 left shift 2
  s[2]  = t[10]; s[6]  = t[14]; s[10] = t[2];  s[14] = t[6];
  // row3 left shift 3
  s[3]  = t[15]; s[7]  = t[3];  s[11] = t[7];  s[15] = t[11];
}

static inline __device__ void aes_mix_columns(uint8_t s[16]) {
  #pragma unroll
  for (int c = 0; c < 4; c++) {
    const int o = c * 4;
    const uint8_t a0 = s[o + 0];
    const uint8_t a1 = s[o + 1];
    const uint8_t a2 = s[o + 2];
    const uint8_t a3 = s[o + 3];
    const uint8_t r0 = (uint8_t)(xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3);
    const uint8_t r1 = (uint8_t)(a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3);
    const uint8_t r2 = (uint8_t)(a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3));
    const uint8_t r3 = (uint8_t)((xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3));
    s[o + 0] = r0;
    s[o + 1] = r1;
    s[o + 2] = r2;
    s[o + 3] = r3;
  }
}

static inline __device__ void aes128_encrypt_block(const uint8_t rk[176], const uint8_t in[16], uint8_t out[16]) {
  uint8_t s[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) s[i] = in[i] ^ rk[i];

  // rounds 1..9
  #pragma unroll
  for (int round = 1; round <= 9; round++) {
    #pragma unroll
    for (int i = 0; i < 16; i++) s[i] = kAesSbox[s[i]];
    aes_shift_rows(s);
    aes_mix_columns(s);
    const int rk_off = round * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++) s[i] ^= rk[rk_off + i];
  }

  // final round 10
  #pragma unroll
  for (int i = 0; i < 16; i++) s[i] = kAesSbox[s[i]];
  aes_shift_rows(s);
  #pragma unroll
  for (int i = 0; i < 16; i++) out[i] = (uint8_t)(s[i] ^ rk[160 + i]);
}

static inline __device__ void g_expand_aes128_seed16(uint64_t seed_lo, uint64_t seed_hi, uint32_t depth, uint64_t& SL_lo, uint64_t& SL_hi, uint64_t& SR_lo, uint64_t& SR_hi, uint8_t& tL, uint8_t& tR) {
  uint8_t key16[16];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    key16[i] = (uint8_t)((seed_lo >> (8 * i)) & 0xFFu);
    key16[8 + i] = (uint8_t)((seed_hi >> (8 * i)) & 0xFFu);
  }
  uint8_t rk[176];
  aes128_key_expand(key16, rk);

  const uint64_t d64 = (uint64_t)depth;
  uint8_t p0[16] = {0};
  uint8_t p1[16] = {0};
  uint8_t p2[16] = {0};
  uint8_t p3[16] = {0};
  p1[0] = 1;
  p2[0] = 2;
  p3[0] = 3;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint8_t di = (uint8_t)((d64 >> (8 * i)) & 0xFFu);
    p0[8 + i] = di;
    p1[8 + i] = di;
    p2[8 + i] = di;
    p3[8 + i] = di;
  }
  uint8_t e0[16], e1[16], e2[16], e3[16];
  aes128_encrypt_block(rk, p0, e0);
  aes128_encrypt_block(rk, p1, e1);
  aes128_encrypt_block(rk, p2, e2);
  aes128_encrypt_block(rk, p3, e3);
  SL_lo = load_le_u64(e0 + 0);
  SL_hi = load_le_u64(e0 + 8);
  SR_lo = load_le_u64(e1 + 0);
  SR_hi = load_le_u64(e1 + 8);
  tL = (uint8_t)(e2[0] & 1u);
  tR = (uint8_t)(e3[0] & 1u);
}

static inline __device__ void g_expand_seed16(uint64_t seed_lo, uint64_t seed_hi, uint32_t depth, uint8_t prg_id, uint64_t& SL_lo, uint64_t& SL_hi, uint64_t& SR_lo, uint64_t& SR_hi, uint8_t& tL, uint8_t& tR) {
  if (prg_id == 1u) {
    g_expand_aes128_seed16(seed_lo, seed_hi, depth, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
  } else {
    g_expand_chacha12_seed16(seed_lo, seed_hi, depth, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
  }
}

// -------------------------
// Kernels (ABI: uvcc_dpf_dcf_gpu_abi_v1.h)
// -------------------------

__global__ void uvcc_dpf_stage1_w16_v1(
    const uint8_t* __restrict__ keyrec_bytes,
    uint64_t* __restrict__ frontier_seed_lo,
    uint64_t* __restrict__ frontier_seed_hi,
    uint8_t* __restrict__ frontier_t,
    uint8_t* __restrict__ frontier_acc) {
  const int tid = (int)threadIdx.x;
  if (tid >= 256) return;

  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (w != 16 || (prg_id != 1u && prg_id != 2u)) {
    frontier_seed_lo[tid] = 0;
    frontier_seed_hi[tid] = 0;
    frontier_t[tid] = 0;
    frontier_acc[tid] = 0;
    return;
  }

  uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
  uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
  uint8_t t = keyrec_bytes[80] & 1;

  // CW base offset: header(64) + root_seed(16) + root_t(1)
  const int cw_base = 64 + 16 + 1;

  // Traverse to depth 8 for this prefix tid (MSB-first bits).
  #pragma unroll
  for (int d = 0; d < 8; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1;
    const uint8_t tauR = (tau_mask >> 1) & 1;

    if (t) {
      SL_lo ^= sigma_lo;
      SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo;
      SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }

    const int bit = (tid >> (7 - d)) & 1;
    if (bit == 0) {
      seed_lo = SL_lo;
      seed_hi = SL_hi;
      t = tL & 1;
    } else {
      seed_lo = SR_lo;
      seed_hi = SR_hi;
      t = tR & 1;
    }
  }

  frontier_seed_lo[tid] = seed_lo;
  frontier_seed_hi[tid] = seed_hi;
  frontier_t[tid] = t;

  __shared__ uint8_t scan[256];
  scan[tid] = t;
  __syncthreads();

  // Inclusive prefix XOR scan in-place.
  uint8_t v = scan[tid];
  for (int offset = 1; offset < 256; offset <<= 1) {
    __syncthreads();
    const uint8_t prev = (tid >= offset) ? scan[tid - offset] : 0;
    __syncthreads();
    v ^= prev;
    scan[tid] = v;
    __syncthreads();
  }

  frontier_acc[tid] = (tid == 0) ? 0 : scan[tid - 1];
}

__global__ void uvcc_dcf_stage2_w16_v1(
    const uint8_t* __restrict__ keyrec_bytes,
    const uint64_t* __restrict__ frontier_seed_lo,
    const uint64_t* __restrict__ frontier_seed_hi,
    const uint8_t* __restrict__ frontier_t,
    const uint8_t* __restrict__ frontier_acc,
    uint64_t* __restrict__ out_word_u64) {
  const int block = (int)blockIdx.x;
  const int lane = (int)threadIdx.x;
  if (block >= 256 || lane >= 256) return;

  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (w != 16 || (prg_id != 1u && prg_id != 2u)) {
    out_word_u64[block * 256 + lane] = 0;
    return;
  }

  const uint16_t flags = load_le_u16(keyrec_bytes + 14);
  const uint8_t invert = (uint8_t)(flags & 0x0001u);
  const uint8_t root_t = keyrec_bytes[80] & 1;

  // DCF payload mask (present in DCF keyrecs; offset = 64 + 17 + 17*w)
  const int cw_base = 64 + 16 + 1;
  const int payload_mask_off = cw_base + 17 * 16;
  const uint64_t payload_mask = load_le_u64(keyrec_bytes + payload_mask_off);

  uint64_t seed_lo = frontier_seed_lo[block];
  uint64_t seed_hi = frontier_seed_hi[block];
  uint8_t t = frontier_t[block] & 1;

  // Expand depths 8..15 following lane bits (MSB-first within the 8-bit suffix).
  #pragma unroll
  for (int d = 8; d < 16; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1;
    const uint8_t tauR = (tau_mask >> 1) & 1;

    if (t) {
      SL_lo ^= sigma_lo;
      SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo;
      SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }

    const int bit = (lane >> (15 - d)) & 1;
    if (bit == 0) {
      seed_lo = SL_lo;
      seed_hi = SL_hi;
      t = tL & 1;
    } else {
      seed_lo = SR_lo;
      seed_hi = SR_hi;
      t = tR & 1;
    }
  }

  __shared__ uint8_t scan[256];
  scan[lane] = t;
  __syncthreads();
  uint8_t v = scan[lane];
  for (int offset = 1; offset < 256; offset <<= 1) {
    __syncthreads();
    const uint8_t prev = (lane >= offset) ? scan[lane - offset] : 0;
    __syncthreads();
    v ^= prev;
    scan[lane] = v;
    __syncthreads();
  }

  const uint8_t carry = frontier_acc[block] & 1;
  const uint8_t P_global = (scan[lane] ^ carry) & 1;
  const uint8_t dcf_bit = invert ? ((P_global ^ root_t) & 1) : P_global;
  const uint64_t out = dcf_bit ? payload_mask : 0;
  out_word_u64[block * 256 + lane] = out;
}

__global__ void uvcc_dcf_full_w8_v1(const uint8_t* __restrict__ keyrec_bytes, uint64_t* __restrict__ out_word_u64) {
  const int lane = (int)threadIdx.x;
  if (lane >= 256) return;

  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (w != 8 || (prg_id != 1u && prg_id != 2u)) {
    out_word_u64[lane] = 0;
    return;
  }
  const uint16_t flags = load_le_u16(keyrec_bytes + 14);
  const uint8_t invert = (uint8_t)(flags & 0x0001u);
  const uint8_t root_t = keyrec_bytes[80] & 1;

  const int cw_base = 64 + 16 + 1;
  const int payload_mask_off = cw_base + 17 * 8;
  const uint64_t payload_mask = load_le_u64(keyrec_bytes + payload_mask_off);

  uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
  uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
  uint8_t t = keyrec_bytes[80] & 1;

  #pragma unroll
  for (int d = 0; d < 8; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1;
    const uint8_t tauR = (tau_mask >> 1) & 1;
    if (t) {
      SL_lo ^= sigma_lo;
      SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo;
      SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (lane >> (7 - d)) & 1;
    if (bit == 0) {
      seed_lo = SL_lo;
      seed_hi = SL_hi;
      t = tL & 1;
    } else {
      seed_lo = SR_lo;
      seed_hi = SR_hi;
      t = tR & 1;
    }
  }

  __shared__ uint8_t scan[256];
  scan[lane] = t;
  __syncthreads();
  uint8_t v = scan[lane];
  for (int offset = 1; offset < 256; offset <<= 1) {
    __syncthreads();
    const uint8_t prev = (lane >= offset) ? scan[lane - offset] : 0;
    __syncthreads();
    v ^= prev;
    scan[lane] = v;
    __syncthreads();
  }
  const uint8_t P = scan[lane] & 1;
  const uint8_t dcf_bit = invert ? ((P ^ root_t) & 1) : P;
  out_word_u64[lane] = dcf_bit ? payload_mask : 0;
}

// Bitpacked outputs (UVCC_OUT_BITPACK32): pack 32 lanes per uint32 using __ballot_sync.
__global__ void uvcc_dpf_full_w8_bitpack32_v1(const uint8_t* __restrict__ keyrec_bytes, uint32_t* __restrict__ out_words_u32) {
  const int lane = (int)threadIdx.x;
  if (lane >= 256) return;

  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  const bool ok = (prim == 0x21u) && (w == 8u) && ((prg_id == 1u) || (prg_id == 2u));

  uint8_t t = 0;
  if (ok) {
    const int cw_base = 64 + 16 + 1;
    uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
    uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
    t = keyrec_bytes[80] & 1;

    #pragma unroll
    for (int d = 0; d < 8; d++) {
      uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
      uint8_t tL, tR;
      g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

      const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
      const uint64_t sigma_lo = load_le_u64(cw + 0);
      const uint64_t sigma_hi = load_le_u64(cw + 8);
      const uint8_t tau_mask = cw[16] & 0x03;
      const uint8_t tauL = tau_mask & 1;
      const uint8_t tauR = (tau_mask >> 1) & 1;
      if (t) {
        SL_lo ^= sigma_lo;
        SL_hi ^= sigma_hi;
        SR_lo ^= sigma_lo;
        SR_hi ^= sigma_hi;
        tL ^= tauL;
        tR ^= tauR;
      }
      const int bit = (lane >> (7 - d)) & 1;
      if (bit == 0) {
        seed_lo = SL_lo;
        seed_hi = SL_hi;
        t = tL & 1;
      } else {
        seed_lo = SR_lo;
        seed_hi = SR_hi;
        t = tR & 1;
      }
    }
  }

  const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (int)(t & 1u));
  if ((lane & 31) == 0) out_words_u32[lane >> 5] = mask;
}

__global__ void uvcc_dcf_full_w8_bitpack32_v1(const uint8_t* __restrict__ keyrec_bytes, uint32_t* __restrict__ out_words_u32) {
  const int lane = (int)threadIdx.x;
  if (lane >= 256) return;

  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  const bool ok = (prim == 0x22u) && (w == 8u) && ((prg_id == 1u) || (prg_id == 2u));

  uint8_t dcf_bit = 0;
  if (ok) {
    const uint16_t flags = load_le_u16(keyrec_bytes + 14);
    const uint8_t invert = (uint8_t)(flags & 0x0001u);
    const uint8_t root_t = keyrec_bytes[80] & 1;
    const int cw_base = 64 + 16 + 1;

    uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
    uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
    uint8_t t = keyrec_bytes[80] & 1;

    #pragma unroll
    for (int d = 0; d < 8; d++) {
      uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
      uint8_t tL, tR;
      g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

      const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
      const uint64_t sigma_lo = load_le_u64(cw + 0);
      const uint64_t sigma_hi = load_le_u64(cw + 8);
      const uint8_t tau_mask = cw[16] & 0x03;
      const uint8_t tauL = tau_mask & 1;
      const uint8_t tauR = (tau_mask >> 1) & 1;
      if (t) {
        SL_lo ^= sigma_lo;
        SL_hi ^= sigma_hi;
        SR_lo ^= sigma_lo;
        SR_hi ^= sigma_hi;
        tL ^= tauL;
        tR ^= tauR;
      }
      const int bit = (lane >> (7 - d)) & 1;
      if (bit == 0) {
        seed_lo = SL_lo;
        seed_hi = SL_hi;
        t = tL & 1;
      } else {
        seed_lo = SR_lo;
        seed_hi = SR_hi;
        t = tR & 1;
      }
    }

    __shared__ uint8_t scan[256];
    scan[lane] = t;
    __syncthreads();
    uint8_t v = scan[lane];
    for (int offset = 1; offset < 256; offset <<= 1) {
      __syncthreads();
      const uint8_t prev = (lane >= offset) ? scan[lane - offset] : 0;
      __syncthreads();
      v ^= prev;
      scan[lane] = v;
      __syncthreads();
    }
    const uint8_t P = scan[lane] & 1;
    dcf_bit = invert ? ((P ^ root_t) & 1) : P;
  }

  const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (int)(dcf_bit & 1u));
  if ((lane & 31) == 0) out_words_u32[lane >> 5] = mask;
}

__global__ void uvcc_dpf_stage2_w16_bitpack32_v1(
    const uint8_t* __restrict__ keyrec_bytes,
    const uint64_t* __restrict__ frontier_seed_lo,
    const uint64_t* __restrict__ frontier_seed_hi,
    const uint8_t* __restrict__ frontier_t,
    uint32_t* __restrict__ out_words_u32) {
  const int block = (int)blockIdx.x;
  const int lane = (int)threadIdx.x;
  if (block >= 256 || lane >= 256) return;

  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  const bool ok = (prim == 0x21u) && (w == 16u) && ((prg_id == 1u) || (prg_id == 2u));

  uint8_t t = 0;
  if (ok) {
    const int cw_base = 64 + 16 + 1;
    uint64_t seed_lo = frontier_seed_lo[block];
    uint64_t seed_hi = frontier_seed_hi[block];
    t = frontier_t[block] & 1;

    #pragma unroll
    for (int d = 8; d < 16; d++) {
      uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
      uint8_t tL, tR;
      g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

      const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
      const uint64_t sigma_lo = load_le_u64(cw + 0);
      const uint64_t sigma_hi = load_le_u64(cw + 8);
      const uint8_t tau_mask = cw[16] & 0x03;
      const uint8_t tauL = tau_mask & 1;
      const uint8_t tauR = (tau_mask >> 1) & 1;
      if (t) {
        SL_lo ^= sigma_lo;
        SL_hi ^= sigma_hi;
        SR_lo ^= sigma_lo;
        SR_hi ^= sigma_hi;
        tL ^= tauL;
        tR ^= tauR;
      }
      const int bit = (lane >> (15 - d)) & 1;
      if (bit == 0) {
        seed_lo = SL_lo;
        seed_hi = SL_hi;
        t = tL & 1;
      } else {
        seed_lo = SR_lo;
        seed_hi = SR_hi;
        t = tR & 1;
      }
    }
  }

  const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (int)(t & 1u));
  const uint32_t warp = (uint32_t)(lane >> 5);
  if ((lane & 31) == 0) out_words_u32[block * 8 + warp] = mask;
}

__global__ void uvcc_dcf_stage2_w16_bitpack32_v1(
    const uint8_t* __restrict__ keyrec_bytes,
    const uint64_t* __restrict__ frontier_seed_lo,
    const uint64_t* __restrict__ frontier_seed_hi,
    const uint8_t* __restrict__ frontier_t,
    const uint8_t* __restrict__ frontier_acc,
    uint32_t* __restrict__ out_words_u32) {
  const int block = (int)blockIdx.x;
  const int lane = (int)threadIdx.x;
  if (block >= 256 || lane >= 256) return;

  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  const bool ok = (prim == 0x22u) && (w == 16u) && ((prg_id == 1u) || (prg_id == 2u));

  uint8_t dcf_bit = 0;
  if (ok) {
    const uint16_t flags = load_le_u16(keyrec_bytes + 14);
    const uint8_t invert = (uint8_t)(flags & 0x0001u);
    const uint8_t root_t = keyrec_bytes[80] & 1;
    const int cw_base = 64 + 16 + 1;

    uint64_t seed_lo = frontier_seed_lo[block];
    uint64_t seed_hi = frontier_seed_hi[block];
    uint8_t t = frontier_t[block] & 1;

    #pragma unroll
    for (int d = 8; d < 16; d++) {
      uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
      uint8_t tL, tR;
      g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);

      const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
      const uint64_t sigma_lo = load_le_u64(cw + 0);
      const uint64_t sigma_hi = load_le_u64(cw + 8);
      const uint8_t tau_mask = cw[16] & 0x03;
      const uint8_t tauL = tau_mask & 1;
      const uint8_t tauR = (tau_mask >> 1) & 1;
      if (t) {
        SL_lo ^= sigma_lo;
        SL_hi ^= sigma_hi;
        SR_lo ^= sigma_lo;
        SR_hi ^= sigma_hi;
        tL ^= tauL;
        tR ^= tauR;
      }
      const int bit = (lane >> (15 - d)) & 1;
      if (bit == 0) {
        seed_lo = SL_lo;
        seed_hi = SL_hi;
        t = tL & 1;
      } else {
        seed_lo = SR_lo;
        seed_hi = SR_hi;
        t = tR & 1;
      }
    }

    __shared__ uint8_t scan[256];
    scan[lane] = t;
    __syncthreads();
    uint8_t v = scan[lane];
    for (int offset = 1; offset < 256; offset <<= 1) {
      __syncthreads();
      const uint8_t prev = (lane >= offset) ? scan[lane - offset] : 0;
      __syncthreads();
      v ^= prev;
      scan[lane] = v;
      __syncthreads();
    }
    const uint8_t carry = frontier_acc[block] & 1;
    const uint8_t P_global = (scan[lane] ^ carry) & 1;
    dcf_bit = invert ? ((P_global ^ root_t) & 1) : P_global;
  }

  const uint32_t mask = __ballot_sync(0xFFFFFFFFu, (int)(dcf_bit & 1u));
  const uint32_t warp = (uint32_t)(lane >> 5);
  if ((lane & 31) == 0) out_words_u32[block * 8 + warp] = mask;
}

// -------------------------
// GF(2) AND (pack32) kernels
// -------------------------

__global__ void uvcc_gf2_and_prepare_pack32_v1(
    const uint32_t* __restrict__ x_lo,
    const uint32_t* __restrict__ x_hi,
    const uint32_t* __restrict__ y_lo,
    const uint32_t* __restrict__ y_hi,
    const uint32_t* __restrict__ a_lo,
    const uint32_t* __restrict__ a_hi,
    const uint32_t* __restrict__ b_lo,
    const uint32_t* __restrict__ b_hi,
    uint32_t* __restrict__ e_lo_out,
    uint32_t* __restrict__ f_lo_out,
    uint32_t* __restrict__ e_hi_scratch,
    uint32_t* __restrict__ f_hi_scratch) {
  const int w = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  // Caller must launch exactly W threads.
  const uint32_t e_lo = x_lo[w] ^ a_lo[w];
  const uint32_t e_hi = x_hi[w] ^ a_hi[w];
  const uint32_t f_lo = y_lo[w] ^ b_lo[w];
  const uint32_t f_hi = y_hi[w] ^ b_hi[w];
  e_lo_out[w] = e_lo;
  f_lo_out[w] = f_lo;
  e_hi_scratch[w] = e_hi;
  f_hi_scratch[w] = f_hi;
}

__global__ void uvcc_gf2_and_finish_pack32_v1(
    const uint32_t* __restrict__ a_lo,
    const uint32_t* __restrict__ a_hi,
    const uint32_t* __restrict__ b_lo,
    const uint32_t* __restrict__ b_hi,
    const uint32_t* __restrict__ c_lo,
    const uint32_t* __restrict__ c_hi,
    const uint32_t* __restrict__ e_pub,
    const uint32_t* __restrict__ f_pub,
    uint32_t* __restrict__ z_lo_out,
    uint32_t* __restrict__ z_hi_out,
    uint32_t party_id) {
  const int w = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const uint32_t term_eb_lo = b_lo[w] & e_pub[w];
  const uint32_t term_eb_hi = b_hi[w] & e_pub[w];
  const uint32_t term_fa_lo = a_lo[w] & f_pub[w];
  const uint32_t term_fa_hi = a_hi[w] & f_pub[w];
  const uint32_t ef_pub = e_pub[w] & f_pub[w];
  uint32_t z_lo = c_lo[w] ^ term_eb_lo ^ term_fa_lo;
  uint32_t z_hi = c_hi[w] ^ term_eb_hi ^ term_fa_hi;
  if (party_id == 0) z_lo ^= ef_pub;
  if (party_id == 2) z_hi ^= ef_pub;
  z_lo_out[w] = z_lo;
  z_hi_out[w] = z_hi;
}

// -------------------------
// A2B subtract (pack32) kernels
// -------------------------

__global__ void uvcc_a2b_sub_prepare_and_bit_pack32_v1(
    const uint32_t* __restrict__ rj_lo,
    const uint32_t* __restrict__ rj_hi,
    const uint32_t* __restrict__ bj_lo,
    const uint32_t* __restrict__ bj_hi,
    const uint32_t* __restrict__ aj_lo,
    const uint32_t* __restrict__ aj_hi,
    const uint32_t* __restrict__ bjT_lo,
    const uint32_t* __restrict__ bjT_hi,
    uint32_t* __restrict__ e_lo_out,
    uint32_t* __restrict__ f_lo_out,
    uint32_t* __restrict__ e_hi_scratch,
    uint32_t* __restrict__ f_hi_scratch) {
  const int w = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const uint32_t e_lo = rj_lo[w] ^ aj_lo[w];
  const uint32_t e_hi = rj_hi[w] ^ aj_hi[w];
  const uint32_t f_lo = bj_lo[w] ^ bjT_lo[w];
  const uint32_t f_hi = bj_hi[w] ^ bjT_hi[w];
  e_lo_out[w] = e_lo;
  f_lo_out[w] = f_lo;
  e_hi_scratch[w] = e_hi;
  f_hi_scratch[w] = f_hi;
}

__global__ void uvcc_a2b_sub_finish_and_bit_pack32_v1(
    const uint32_t* __restrict__ rj_lo,
    const uint32_t* __restrict__ rj_hi,
    const uint32_t* __restrict__ bj_lo,
    const uint32_t* __restrict__ bj_hi,
    const uint32_t* __restrict__ aj_lo,
    const uint32_t* __restrict__ aj_hi,
    const uint32_t* __restrict__ bjT_lo,
    const uint32_t* __restrict__ bjT_hi,
    const uint32_t* __restrict__ cj_lo,
    const uint32_t* __restrict__ cj_hi,
    const uint32_t* __restrict__ e_pub,
    const uint32_t* __restrict__ f_pub,
    const uint32_t* __restrict__ cj_public_mask,
    uint32_t* __restrict__ xj_lo_out,
    uint32_t* __restrict__ xj_hi_out,
    uint32_t* __restrict__ bnext_lo_out,
    uint32_t* __restrict__ bnext_hi_out,
    uint32_t party_id) {
  const int w = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

  // g = AND(r_j, b_j) via Beaver triple
  uint32_t g_lo = cj_lo[w] ^ (bjT_lo[w] & e_pub[w]) ^ (aj_lo[w] & f_pub[w]);
  uint32_t g_hi = cj_hi[w] ^ (bjT_hi[w] & e_pub[w]) ^ (aj_hi[w] & f_pub[w]);
  const uint32_t ef_pub = e_pub[w] & f_pub[w];
  if (party_id == 0) g_lo ^= ef_pub;
  if (party_id == 2) g_hi ^= ef_pub;

  // t = r xor b
  const uint32_t t_lo = rj_lo[w] ^ bj_lo[w];
  const uint32_t t_hi = rj_hi[w] ^ bj_hi[w];

  // x_j = t with public c_j injected into share-0 only
  uint32_t x_lo = t_lo;
  uint32_t x_hi = t_hi;
  if (party_id == 0) x_lo ^= cj_public_mask[w];
  if (party_id == 2) x_hi ^= cj_public_mask[w];

  // bnext = g xor (~c_j & t)
  const uint32_t mask0 = ~cj_public_mask[w];
  const uint32_t bnext_lo = g_lo ^ (mask0 & t_lo);
  const uint32_t bnext_hi = g_hi ^ (mask0 & t_hi);

  xj_lo_out[w] = x_lo;
  xj_hi_out[w] = x_hi;
  bnext_lo_out[w] = bnext_lo;
  bnext_hi_out[w] = bnext_hi;
}

// -------------------------
// A2B packing helpers (optional but recommended)
// -------------------------

__global__ void uvcc_a2b_pack_c_lo_u8_v1(
    const uint64_t* __restrict__ x_lo_u64,
    const uint64_t* __restrict__ r_lo_u64,
    uint8_t* __restrict__ c_lo_u8_out,
    uint32_t n_elems) {
  const uint32_t i = (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
  if (i >= n_elems) return;
  const uint64_t c = x_lo_u64[i] + r_lo_u64[i];
  c_lo_u8_out[i] = (uint8_t)c;
}

__global__ void uvcc_a2b_pack_c_lo_u16_v1(
    const uint64_t* __restrict__ x_lo_u64,
    const uint64_t* __restrict__ r_lo_u64,
    uint16_t* __restrict__ c_lo_u16_out,
    uint32_t n_elems) {
  const uint32_t i = (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
  if (i >= n_elems) return;
  const uint64_t c = x_lo_u64[i] + r_lo_u64[i];
  c_lo_u16_out[i] = (uint16_t)c;
}

extern "C" __global__ void uvcc_a2b_cpub_to_cjmask_u8_v1(
    const uint8_t* __restrict__ c_pub_u8,
    uint32_t* __restrict__ out_u32,
    uint32_t L) {
  const uint32_t lane = (uint32_t)(threadIdx.x & 31);
  const uint32_t warps_per_block = (uint32_t)(blockDim.x >> 5);
  const uint32_t warp_in_block = (uint32_t)(threadIdx.x >> 5);
  const uint32_t warp_id = (uint32_t)blockIdx.x * warps_per_block + warp_in_block;
  const uint32_t n_words = (L + 31u) >> 5;
  if (warp_id >= n_words) return;
  const uint32_t base = warp_id << 5;
  const uint32_t k = base + lane;
  const uint8_t v = (k < L) ? c_pub_u8[k] : 0;

  #pragma unroll
  for (uint32_t j = 0; j < 8; j++) {
    const int pred = (int)((v >> j) & 1u);
    const uint32_t mask = __ballot_sync(0xFFFFFFFFu, pred);
    if (lane == 0) out_u32[j * n_words + warp_id] = mask;
  }
}

extern "C" __global__ void uvcc_a2b_cpub_to_cjmask_u16_v1(
    const uint16_t* __restrict__ c_pub_u16,
    uint32_t* __restrict__ out_u32,
    uint32_t L) {
  const uint32_t lane = (uint32_t)(threadIdx.x & 31);
  const uint32_t warps_per_block = (uint32_t)(blockDim.x >> 5);
  const uint32_t warp_in_block = (uint32_t)(threadIdx.x >> 5);
  const uint32_t warp_id = (uint32_t)blockIdx.x * warps_per_block + warp_in_block;
  const uint32_t n_words = (L + 31u) >> 5;
  if (warp_id >= n_words) return;
  const uint32_t base = warp_id << 5;
  const uint32_t k = base + lane;
  const uint16_t v = (k < L) ? c_pub_u16[k] : 0;

  #pragma unroll
  for (uint32_t j = 0; j < 16; j++) {
    const int pred = (int)((v >> j) & 1u);
    const uint32_t mask = __ballot_sync(0xFFFFFFFFu, pred);
    if (lane == 0) out_u32[j * n_words + warp_id] = mask;
  }
}

// -------------------------
// Point-eval kernels (DPF/DCF) for TRUNC and other protocols
// -------------------------

// DPF point-eval (bit output) — batch, 1 thread per query.
extern "C" __global__ void uvcc_dpf_eval_point_w8_batch_v1(
    const uint8_t* __restrict__ keyrecs_blob,
    uint32_t key_stride_bytes,
    const uint16_t* __restrict__ x_pub_u16,
    uint8_t* __restrict__ out_bit_u8,
    uint32_t N) {
  const uint32_t q = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= N) return;
  const uint8_t* keyrec_bytes = keyrecs_blob + (uint64_t)q * (uint64_t)key_stride_bytes;
  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (prim != 0x21u || w != 8u || (prg_id != 1u && prg_id != 2u)) {
    out_bit_u8[q] = 0;
    return;
  }
  const uint16_t u = x_pub_u16[q];

  uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
  uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
  uint8_t t = keyrec_bytes[80] & 1u;
  const int cw_base = 64 + 16 + 1;

  #pragma unroll
  for (int d = 0; d < 8; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1u;
    const uint8_t tauR = (tau_mask >> 1) & 1u;
    if (t) {
      SL_lo ^= sigma_lo; SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo; SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (int)((u >> (7 - d)) & 1u);
    if (bit == 0) {
      seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u;
    } else {
      seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u;
    }
  }
  out_bit_u8[q] = (uint8_t)(t & 1u);
}

extern "C" __global__ void uvcc_dpf_eval_point_w16_batch_v1(
    const uint8_t* __restrict__ keyrecs_blob,
    uint32_t key_stride_bytes,
    const uint16_t* __restrict__ x_pub_u16,
    uint8_t* __restrict__ out_bit_u8,
    uint32_t N) {
  const uint32_t q = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (q >= N) return;
  const uint8_t* keyrec_bytes = keyrecs_blob + (uint64_t)q * (uint64_t)key_stride_bytes;
  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (prim != 0x21u || w != 16u || (prg_id != 1u && prg_id != 2u)) {
    out_bit_u8[q] = 0;
    return;
  }
  const uint16_t u = x_pub_u16[q];

  uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
  uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
  uint8_t t = keyrec_bytes[80] & 1u;
  const int cw_base = 64 + 16 + 1;

  #pragma unroll
  for (int d = 0; d < 16; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1u;
    const uint8_t tauR = (tau_mask >> 1) & 1u;
    if (t) {
      SL_lo ^= sigma_lo; SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo; SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (int)((u >> (15 - d)) & 1u);
    if (bit == 0) {
      seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u;
    } else {
      seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u;
    }
  }
  out_bit_u8[q] = (uint8_t)(t & 1u);
}

// DCF point-eval — batch, 1 block per query (256 threads) to compute the prefix XOR inside the block.
extern "C" __global__ void uvcc_dcf_eval_point_w8_batch_v1(
    const uint8_t* __restrict__ keyrecs_blob,
    uint32_t key_stride_bytes,
    const uint16_t* __restrict__ x_pub_u16,
    uint8_t* __restrict__ out_bit_u8) {
  const uint32_t q = (uint32_t)blockIdx.x;
  const int lane = (int)threadIdx.x;
  if (lane >= 256) return;
  const uint8_t* keyrec_bytes = keyrecs_blob + (uint64_t)q * (uint64_t)key_stride_bytes;
  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (prim != 0x22u || w != 8u || (prg_id != 1u && prg_id != 2u)) {
    if (lane == 0) out_bit_u8[q] = 0;
    return;
  }
  const uint16_t u = x_pub_u16[q];
  const uint16_t flags = load_le_u16(keyrec_bytes + 14);
  const uint8_t invert = (uint8_t)(flags & 0x0001u);
  const uint8_t root_t = keyrec_bytes[80] & 1u;

  uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
  uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
  uint8_t t = keyrec_bytes[80] & 1u;
  const int cw_base = 64 + 16 + 1;

  #pragma unroll
  for (int d = 0; d < 8; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1u;
    const uint8_t tauR = (tau_mask >> 1) & 1u;
    if (t) {
      SL_lo ^= sigma_lo;
      SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo;
      SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (lane >> (7 - d)) & 1;
    if (bit == 0) {
      seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u;
    } else {
      seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u;
    }
  }

  __shared__ uint8_t scan[256];
  scan[lane] = t;
  __syncthreads();
  uint8_t v = scan[lane];
  for (int offset = 1; offset < 256; offset <<= 1) {
    __syncthreads();
    const uint8_t prev = (lane >= offset) ? scan[lane - offset] : 0;
    __syncthreads();
    v ^= prev;
    scan[lane] = v;
    __syncthreads();
  }
  const uint8_t P = scan[lane] & 1u;
  const uint8_t dcf_bit = invert ? ((P ^ root_t) & 1u) : P;
  if (lane == (int)(u & 0xFFu)) out_bit_u8[q] = dcf_bit;
}

extern "C" __global__ void uvcc_dcf_eval_point_w16_batch_v1(
    const uint8_t* __restrict__ keyrecs_blob,
    uint32_t key_stride_bytes,
    const uint16_t* __restrict__ x_pub_u16,
    uint8_t* __restrict__ out_bit_u8) {
  const uint32_t q = (uint32_t)blockIdx.x;
  const int tid = (int)threadIdx.x;
  if (tid >= 256) return;
  const uint8_t* keyrec_bytes = keyrecs_blob + (uint64_t)q * (uint64_t)key_stride_bytes;
  const uint8_t prim = keyrec_bytes[10];
  const uint8_t w = keyrec_bytes[12];
  const uint8_t prg_id = keyrec_bytes[13];
  if (prim != 0x22u || w != 16u || (prg_id != 1u && prg_id != 2u)) {
    if (tid == 0) out_bit_u8[q] = 0;
    return;
  }
  const uint16_t u = x_pub_u16[q];
  const uint16_t flags = load_le_u16(keyrec_bytes + 14);
  const uint8_t invert = (uint8_t)(flags & 0x0001u);
  const uint8_t root_t = keyrec_bytes[80] & 1u;

  // Shared frontier (256).
  __shared__ uint64_t frontier_seed_lo[256];
  __shared__ uint64_t frontier_seed_hi[256];
  __shared__ uint8_t frontier_t[256];
  __shared__ uint8_t frontier_acc[256];
  __shared__ uint8_t scan[256];

  // Stage-1: compute frontier node for prefix=tid (depth 8).
  uint64_t seed_lo = load_le_u64(keyrec_bytes + 64);
  uint64_t seed_hi = load_le_u64(keyrec_bytes + 72);
  uint8_t t = keyrec_bytes[80] & 1u;
  const int cw_base = 64 + 16 + 1;

  #pragma unroll
  for (int d = 0; d < 8; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_seed16(seed_lo, seed_hi, (uint32_t)d, prg_id, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1u;
    const uint8_t tauR = (tau_mask >> 1) & 1u;
    if (t) {
      SL_lo ^= sigma_lo; SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo; SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (tid >> (7 - d)) & 1;
    if (bit == 0) {
      seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u;
    } else {
      seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u;
    }
  }

  frontier_seed_lo[tid] = seed_lo;
  frontier_seed_hi[tid] = seed_hi;
  frontier_t[tid] = t & 1u;
  __syncthreads();

  // Stage-1 scan: frontier_acc[p] = XOR_{k<p} frontier_t[k]
  scan[tid] = frontier_t[tid] & 1u;
  __syncthreads();
  uint8_t v = scan[tid];
  for (int offset = 1; offset < 256; offset <<= 1) {
    __syncthreads();
    const uint8_t prev = (tid >= offset) ? scan[tid - offset] : 0;
    __syncthreads();
    v ^= prev;
    scan[tid] = v;
    __syncthreads();
  }
  frontier_acc[tid] = (tid == 0) ? 0 : (scan[tid - 1] & 1u);
  __syncthreads();

  // Stage-2: expand selected prefix block and scan within it.
  const uint32_t prefix = (uint32_t)((u >> 8) & 0xFFu);
  const uint32_t lane = (uint32_t)(tid & 0xFF);
  seed_lo = frontier_seed_lo[prefix];
  seed_hi = frontier_seed_hi[prefix];
  t = frontier_t[prefix] & 1u;

  #pragma unroll
  for (int d = 8; d < 16; d++) {
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_chacha12_seed16(seed_lo, seed_hi, (uint32_t)d, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
    const uint8_t* cw = keyrec_bytes + cw_base + d * 17;
    const uint64_t sigma_lo = load_le_u64(cw + 0);
    const uint64_t sigma_hi = load_le_u64(cw + 8);
    const uint8_t tau_mask = cw[16] & 0x03;
    const uint8_t tauL = tau_mask & 1u;
    const uint8_t tauR = (tau_mask >> 1) & 1u;
    if (t) {
      SL_lo ^= sigma_lo; SL_hi ^= sigma_hi;
      SR_lo ^= sigma_lo; SR_hi ^= sigma_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (int)((lane >> (15 - d)) & 1u);  // lane bits MSB-first for depth 8 subtree
    if (bit == 0) {
      seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u;
    } else {
      seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u;
    }
  }

  // Inclusive prefix XOR over lane index.
  scan[lane] = t & 1u;
  __syncthreads();
  uint8_t pv = scan[lane];
  for (int offset = 1; offset < 256; offset <<= 1) {
    __syncthreads();
    const uint8_t prev = (lane >= (uint32_t)offset) ? scan[lane - offset] : 0;
    __syncthreads();
    pv ^= prev;
    scan[lane] = pv;
    __syncthreads();
  }

  const uint8_t carry = frontier_acc[prefix] & 1u;
  const uint8_t P_global = (scan[lane] ^ carry) & 1u;
  const uint8_t dcf_bit = invert ? ((P_global ^ root_t) & 1u) : P_global;
  if (lane == (uint32_t)(u & 0xFFu)) out_bit_u8[q] = dcf_bit;
}

// TRUNC apply (vector arithmetic) — no length parameter; wrapper must launch exact thread counts.
extern "C" __global__ void uvcc_trunc_apply_u64_v1(
    const uint64_t* __restrict__ C1_pub,
    const uint64_t* __restrict__ R1_lo,
    const uint64_t* __restrict__ R1_hi,
    const uint64_t* __restrict__ carry_lo,
    const uint64_t* __restrict__ carry_hi,
    const uint64_t* __restrict__ ov_lo,
    const uint64_t* __restrict__ ov_hi,
    uint64_t add_const,
    uint8_t party_id,
    uint64_t* __restrict__ Y_lo,
    uint64_t* __restrict__ Y_hi) {
  const uint32_t idx = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  uint64_t y0 = (uint64_t)0 - (R1_lo[idx] + (carry_lo[idx] & 1u));
  uint64_t y1 = (uint64_t)0 - (R1_hi[idx] + (carry_hi[idx] & 1u));
  y0 += (ov_lo[idx] & 1u) ? add_const : 0ull;
  y1 += (ov_hi[idx] & 1u) ? add_const : 0ull;

  const uint64_t c1 = C1_pub[idx];
  if (party_id == 0) y0 += c1;
  if (party_id == 2) y1 += c1;
  Y_lo[idx] = y0;
  Y_hi[idx] = y1;
}

// -------------------------
// C++/PyTorch wrappers
// -------------------------

static void check_u8_cuda_contig(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_i32_cuda_contig(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32 (u32 bit-patterns)");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_i16_cuda_contig(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kInt16, name, " must be int16 (u16 bit-patterns)");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_i64_cuda_contig(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> uvcc_dpf_stage1_w16_cuda(torch::Tensor keyrec_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 16, "keyrec_u8 too small");
  auto dev = keyrec_u8.device();
  auto opts_i64 = torch::TensorOptions().device(dev).dtype(torch::kInt64);
  auto opts_u8 = torch::TensorOptions().device(dev).dtype(torch::kUInt8);
  auto seed_lo = torch::empty({256}, opts_i64);
  auto seed_hi = torch::empty({256}, opts_i64);
  auto t = torch::empty({256}, opts_u8);
  auto acc = torch::empty({256}, opts_u8);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  uvcc_dpf_stage1_w16_v1<<<1, 256, 0, stream>>>(
      (const uint8_t*)keyrec_u8.data_ptr<uint8_t>(),
      (uint64_t*)seed_lo.data_ptr<int64_t>(),
      (uint64_t*)seed_hi.data_ptr<int64_t>(),
      (uint8_t*)t.data_ptr<uint8_t>(),
      (uint8_t*)acc.data_ptr<uint8_t>());
  return {seed_lo, seed_hi, t, acc};
}

torch::Tensor uvcc_dcf_stage2_w16_cuda(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8,
    torch::Tensor frontier_acc_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  check_i64_cuda_contig(frontier_seed_lo_i64, "frontier_seed_lo_i64");
  check_i64_cuda_contig(frontier_seed_hi_i64, "frontier_seed_hi_i64");
  check_u8_cuda_contig(frontier_t_u8, "frontier_t_u8");
  check_u8_cuda_contig(frontier_acc_u8, "frontier_acc_u8");
  TORCH_CHECK(frontier_seed_lo_i64.numel() == 256, "frontier_seed_lo_i64 must be len 256");
  TORCH_CHECK(frontier_seed_hi_i64.numel() == 256, "frontier_seed_hi_i64 must be len 256");
  TORCH_CHECK(frontier_t_u8.numel() == 256, "frontier_t_u8 must be len 256");
  TORCH_CHECK(frontier_acc_u8.numel() == 256, "frontier_acc_u8 must be len 256");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 16 + 8, "keyrec_u8 too small for DCF w16");

  auto dev = keyrec_u8.device();
  auto out = torch::empty({65536}, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  dim3 grid(256);
  dim3 block(256);
  uvcc_dcf_stage2_w16_v1<<<grid, block, 0, stream>>>(
      (const uint8_t*)keyrec_u8.data_ptr<uint8_t>(),
      (const uint64_t*)frontier_seed_lo_i64.data_ptr<int64_t>(),
      (const uint64_t*)frontier_seed_hi_i64.data_ptr<int64_t>(),
      (const uint8_t*)frontier_t_u8.data_ptr<uint8_t>(),
      (const uint8_t*)frontier_acc_u8.data_ptr<uint8_t>(),
      (uint64_t*)out.data_ptr<int64_t>());
  return out;
}

torch::Tensor uvcc_dcf_full_w8_cuda(torch::Tensor keyrec_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 8 + 8, "keyrec_u8 too small for DCF w8");
  auto dev = keyrec_u8.device();
  auto out = torch::empty({256}, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  uvcc_dcf_full_w8_v1<<<1, 256, 0, stream>>>((const uint8_t*)keyrec_u8.data_ptr<uint8_t>(), (uint64_t*)out.data_ptr<int64_t>());
  return out;
}

torch::Tensor uvcc_dpf_eval_point_w8_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  check_u8_cuda_contig(keyrecs_blob_u8, "keyrecs_blob_u8");
  check_i16_cuda_contig(x_pub_u16_i16, "x_pub_u16_i16");
  TORCH_CHECK(key_stride_bytes > 0, "key_stride_bytes must be > 0");
  const int64_t N = x_pub_u16_i16.numel();
  TORCH_CHECK(N >= 0, "N must be >= 0");
  TORCH_CHECK(keyrecs_blob_u8.numel() >= N * key_stride_bytes, "keyrecs_blob_u8 too small for N*key_stride_bytes");
  auto dev = keyrecs_blob_u8.device();
  auto out = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kUInt8));
  if (N == 0) return out;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 256;
  const int blocks = (int)((N + threads - 1) / threads);
  uvcc_dpf_eval_point_w8_batch_v1<<<blocks, threads, 0, stream>>>(
      (const uint8_t*)keyrecs_blob_u8.data_ptr<uint8_t>(),
      (uint32_t)(key_stride_bytes & 0xFFFFFFFFu),
      (const uint16_t*)x_pub_u16_i16.data_ptr<int16_t>(),
      (uint8_t*)out.data_ptr<uint8_t>(),
      (uint32_t)(N & 0xFFFFFFFFu));
  return out;
}

torch::Tensor uvcc_dpf_eval_point_w16_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  check_u8_cuda_contig(keyrecs_blob_u8, "keyrecs_blob_u8");
  check_i16_cuda_contig(x_pub_u16_i16, "x_pub_u16_i16");
  TORCH_CHECK(key_stride_bytes > 0, "key_stride_bytes must be > 0");
  const int64_t N = x_pub_u16_i16.numel();
  TORCH_CHECK(N >= 0, "N must be >= 0");
  TORCH_CHECK(keyrecs_blob_u8.numel() >= N * key_stride_bytes, "keyrecs_blob_u8 too small for N*key_stride_bytes");
  auto dev = keyrecs_blob_u8.device();
  auto out = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kUInt8));
  if (N == 0) return out;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 256;
  const int blocks = (int)((N + threads - 1) / threads);
  uvcc_dpf_eval_point_w16_batch_v1<<<blocks, threads, 0, stream>>>(
      (const uint8_t*)keyrecs_blob_u8.data_ptr<uint8_t>(),
      (uint32_t)(key_stride_bytes & 0xFFFFFFFFu),
      (const uint16_t*)x_pub_u16_i16.data_ptr<int16_t>(),
      (uint8_t*)out.data_ptr<uint8_t>(),
      (uint32_t)(N & 0xFFFFFFFFu));
  return out;
}

torch::Tensor uvcc_dcf_eval_point_w8_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  check_u8_cuda_contig(keyrecs_blob_u8, "keyrecs_blob_u8");
  check_i16_cuda_contig(x_pub_u16_i16, "x_pub_u16_i16");
  TORCH_CHECK(key_stride_bytes > 0, "key_stride_bytes must be > 0");
  const int64_t N = x_pub_u16_i16.numel();
  TORCH_CHECK(N >= 0, "N must be >= 0");
  TORCH_CHECK(keyrecs_blob_u8.numel() >= N * key_stride_bytes, "keyrecs_blob_u8 too small for N*key_stride_bytes");
  auto dev = keyrecs_blob_u8.device();
  auto out = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kUInt8));
  if (N == 0) return out;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  dim3 grid((unsigned)N);
  dim3 block(256);
  uvcc_dcf_eval_point_w8_batch_v1<<<grid, block, 0, stream>>>(
      (const uint8_t*)keyrecs_blob_u8.data_ptr<uint8_t>(),
      (uint32_t)(key_stride_bytes & 0xFFFFFFFFu),
      (const uint16_t*)x_pub_u16_i16.data_ptr<int16_t>(),
      (uint8_t*)out.data_ptr<uint8_t>());
  return out;
}

torch::Tensor uvcc_dcf_eval_point_w16_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  check_u8_cuda_contig(keyrecs_blob_u8, "keyrecs_blob_u8");
  check_i16_cuda_contig(x_pub_u16_i16, "x_pub_u16_i16");
  TORCH_CHECK(key_stride_bytes > 0, "key_stride_bytes must be > 0");
  const int64_t N = x_pub_u16_i16.numel();
  TORCH_CHECK(N >= 0, "N must be >= 0");
  TORCH_CHECK(keyrecs_blob_u8.numel() >= N * key_stride_bytes, "keyrecs_blob_u8 too small for N*key_stride_bytes");
  auto dev = keyrecs_blob_u8.device();
  auto out = torch::empty({N}, torch::TensorOptions().device(dev).dtype(torch::kUInt8));
  if (N == 0) return out;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  dim3 grid((unsigned)N);
  dim3 block(256);
  uvcc_dcf_eval_point_w16_batch_v1<<<grid, block, 0, stream>>>(
      (const uint8_t*)keyrecs_blob_u8.data_ptr<uint8_t>(),
      (uint32_t)(key_stride_bytes & 0xFFFFFFFFu),
      (const uint16_t*)x_pub_u16_i16.data_ptr<int16_t>(),
      (uint8_t*)out.data_ptr<uint8_t>());
  return out;
}

std::vector<torch::Tensor> uvcc_trunc_apply_u64_cuda(
    torch::Tensor C1_pub_i64,
    torch::Tensor R1_lo_i64,
    torch::Tensor R1_hi_i64,
    torch::Tensor carry_lo_i64,
    torch::Tensor carry_hi_i64,
    torch::Tensor ov_lo_i64,
    torch::Tensor ov_hi_i64,
    int64_t add_const_u64,
    int64_t party_id) {
  check_i64_cuda_contig(C1_pub_i64, "C1_pub_i64");
  check_i64_cuda_contig(R1_lo_i64, "R1_lo_i64");
  check_i64_cuda_contig(R1_hi_i64, "R1_hi_i64");
  check_i64_cuda_contig(carry_lo_i64, "carry_lo_i64");
  check_i64_cuda_contig(carry_hi_i64, "carry_hi_i64");
  check_i64_cuda_contig(ov_lo_i64, "ov_lo_i64");
  check_i64_cuda_contig(ov_hi_i64, "ov_hi_i64");
  TORCH_CHECK(party_id == 0 || party_id == 1 || party_id == 2, "party_id must be 0/1/2");
  TORCH_CHECK(C1_pub_i64.numel() == R1_lo_i64.numel(), "C1_pub_i64 len mismatch");
  TORCH_CHECK(C1_pub_i64.numel() == R1_hi_i64.numel(), "C1_pub_i64 len mismatch");
  TORCH_CHECK(C1_pub_i64.numel() == carry_lo_i64.numel(), "carry_lo_i64 len mismatch");
  TORCH_CHECK(C1_pub_i64.numel() == carry_hi_i64.numel(), "carry_hi_i64 len mismatch");
  TORCH_CHECK(C1_pub_i64.numel() == ov_lo_i64.numel(), "ov_lo_i64 len mismatch");
  TORCH_CHECK(C1_pub_i64.numel() == ov_hi_i64.numel(), "ov_hi_i64 len mismatch");
  const int64_t n = C1_pub_i64.numel();
  auto dev = C1_pub_i64.device();
  auto y_lo = torch::empty({n}, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  auto y_hi = torch::empty({n}, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  if (n == 0) return {y_lo, y_hi};
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  const int64_t full = n / 256;
  const int64_t rem = n % 256;
  const uint64_t add_const = (uint64_t)add_const_u64;
  const uint8_t pid = (uint8_t)(party_id & 0xFF);

  if (full > 0) {
    uvcc_trunc_apply_u64_v1<<<(unsigned)full, 256, 0, stream>>>(
        (const uint64_t*)C1_pub_i64.data_ptr<int64_t>(),
        (const uint64_t*)R1_lo_i64.data_ptr<int64_t>(),
        (const uint64_t*)R1_hi_i64.data_ptr<int64_t>(),
        (const uint64_t*)carry_lo_i64.data_ptr<int64_t>(),
        (const uint64_t*)carry_hi_i64.data_ptr<int64_t>(),
        (const uint64_t*)ov_lo_i64.data_ptr<int64_t>(),
        (const uint64_t*)ov_hi_i64.data_ptr<int64_t>(),
        add_const,
        pid,
        (uint64_t*)y_lo.data_ptr<int64_t>(),
        (uint64_t*)y_hi.data_ptr<int64_t>());
  }
  if (rem > 0) {
    const int64_t off = full * 256;
    uvcc_trunc_apply_u64_v1<<<1, (unsigned)rem, 0, stream>>>(
        (const uint64_t*)C1_pub_i64.data_ptr<int64_t>() + off,
        (const uint64_t*)R1_lo_i64.data_ptr<int64_t>() + off,
        (const uint64_t*)R1_hi_i64.data_ptr<int64_t>() + off,
        (const uint64_t*)carry_lo_i64.data_ptr<int64_t>() + off,
        (const uint64_t*)carry_hi_i64.data_ptr<int64_t>() + off,
        (const uint64_t*)ov_lo_i64.data_ptr<int64_t>() + off,
        (const uint64_t*)ov_hi_i64.data_ptr<int64_t>() + off,
        add_const,
        pid,
        (uint64_t*)y_lo.data_ptr<int64_t>() + off,
        (uint64_t*)y_hi.data_ptr<int64_t>() + off);
  }
  return {y_lo, y_hi};
}

torch::Tensor uvcc_dpf_full_w8_bitpack32_cuda(torch::Tensor keyrec_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 8, "keyrec_u8 too small for DPF w8");
  auto dev = keyrec_u8.device();
  auto out = torch::empty({8}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  uvcc_dpf_full_w8_bitpack32_v1<<<1, 256, 0, stream>>>(
      (const uint8_t*)keyrec_u8.data_ptr<uint8_t>(),
      (uint32_t*)out.data_ptr<int32_t>());
  return out;
}

torch::Tensor uvcc_dcf_full_w8_bitpack32_cuda(torch::Tensor keyrec_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 8 + 8, "keyrec_u8 too small for DCF w8");
  auto dev = keyrec_u8.device();
  auto out = torch::empty({8}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  uvcc_dcf_full_w8_bitpack32_v1<<<1, 256, 0, stream>>>(
      (const uint8_t*)keyrec_u8.data_ptr<uint8_t>(),
      (uint32_t*)out.data_ptr<int32_t>());
  return out;
}

torch::Tensor uvcc_dpf_stage2_w16_bitpack32_cuda(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  check_i64_cuda_contig(frontier_seed_lo_i64, "frontier_seed_lo_i64");
  check_i64_cuda_contig(frontier_seed_hi_i64, "frontier_seed_hi_i64");
  check_u8_cuda_contig(frontier_t_u8, "frontier_t_u8");
  TORCH_CHECK(frontier_seed_lo_i64.numel() == 256, "frontier_seed_lo_i64 must be len 256");
  TORCH_CHECK(frontier_seed_hi_i64.numel() == 256, "frontier_seed_hi_i64 must be len 256");
  TORCH_CHECK(frontier_t_u8.numel() == 256, "frontier_t_u8 must be len 256");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 16, "keyrec_u8 too small for DPF w16");
  auto dev = keyrec_u8.device();
  auto out = torch::empty({2048}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  dim3 grid(256);
  dim3 block(256);
  uvcc_dpf_stage2_w16_bitpack32_v1<<<grid, block, 0, stream>>>(
      (const uint8_t*)keyrec_u8.data_ptr<uint8_t>(),
      (const uint64_t*)frontier_seed_lo_i64.data_ptr<int64_t>(),
      (const uint64_t*)frontier_seed_hi_i64.data_ptr<int64_t>(),
      (const uint8_t*)frontier_t_u8.data_ptr<uint8_t>(),
      (uint32_t*)out.data_ptr<int32_t>());
  return out;
}

torch::Tensor uvcc_dcf_stage2_w16_bitpack32_cuda(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8,
    torch::Tensor frontier_acc_u8) {
  check_u8_cuda_contig(keyrec_u8, "keyrec_u8");
  check_i64_cuda_contig(frontier_seed_lo_i64, "frontier_seed_lo_i64");
  check_i64_cuda_contig(frontier_seed_hi_i64, "frontier_seed_hi_i64");
  check_u8_cuda_contig(frontier_t_u8, "frontier_t_u8");
  check_u8_cuda_contig(frontier_acc_u8, "frontier_acc_u8");
  TORCH_CHECK(frontier_seed_lo_i64.numel() == 256, "frontier_seed_lo_i64 must be len 256");
  TORCH_CHECK(frontier_seed_hi_i64.numel() == 256, "frontier_seed_hi_i64 must be len 256");
  TORCH_CHECK(frontier_t_u8.numel() == 256, "frontier_t_u8 must be len 256");
  TORCH_CHECK(frontier_acc_u8.numel() == 256, "frontier_acc_u8 must be len 256");
  TORCH_CHECK(keyrec_u8.numel() >= 64 + 16 + 1 + 17 * 16 + 8, "keyrec_u8 too small for DCF w16");
  auto dev = keyrec_u8.device();
  auto out = torch::empty({2048}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  dim3 grid(256);
  dim3 block(256);
  uvcc_dcf_stage2_w16_bitpack32_v1<<<grid, block, 0, stream>>>(
      (const uint8_t*)keyrec_u8.data_ptr<uint8_t>(),
      (const uint64_t*)frontier_seed_lo_i64.data_ptr<int64_t>(),
      (const uint64_t*)frontier_seed_hi_i64.data_ptr<int64_t>(),
      (const uint8_t*)frontier_t_u8.data_ptr<uint8_t>(),
      (const uint8_t*)frontier_acc_u8.data_ptr<uint8_t>(),
      (uint32_t*)out.data_ptr<int32_t>());
  return out;
}

static inline void launch_words_exact_1d(cudaStream_t stream, int64_t words, void* kernel, void** args) {
  // Launch the kernel in chunks so total threads == words (avoids passing length into ABI).
  const int64_t full = words / 256;
  const int64_t rem = words % 256;
  if (full > 0) {
    cudaLaunchKernel(kernel, dim3((unsigned)full), dim3(256), args, 0, stream);
  }
  if (rem > 0) {
    cudaLaunchKernel(kernel, dim3(1), dim3((unsigned)rem), args, 0, stream);
  }
}

std::vector<torch::Tensor> uvcc_gf2_and_prepare_pack32_cuda(
    torch::Tensor x_lo,
    torch::Tensor x_hi,
    torch::Tensor y_lo,
    torch::Tensor y_hi,
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi) {
  check_i32_cuda_contig(x_lo, "x_lo");
  check_i32_cuda_contig(x_hi, "x_hi");
  check_i32_cuda_contig(y_lo, "y_lo");
  check_i32_cuda_contig(y_hi, "y_hi");
  check_i32_cuda_contig(a_lo, "a_lo");
  check_i32_cuda_contig(a_hi, "a_hi");
  check_i32_cuda_contig(b_lo, "b_lo");
  check_i32_cuda_contig(b_hi, "b_hi");
  TORCH_CHECK(x_lo.numel() == x_hi.numel(), "x_lo/x_hi size mismatch");
  TORCH_CHECK(x_lo.numel() == y_lo.numel(), "x/y size mismatch");
  TORCH_CHECK(x_lo.numel() == a_lo.numel(), "x/a size mismatch");
  TORCH_CHECK(x_lo.numel() == b_lo.numel(), "x/b size mismatch");

  auto dev = x_lo.device();
  auto opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);
  auto e_lo = torch::empty_like(x_lo, opts);
  auto f_lo = torch::empty_like(x_lo, opts);
  auto e_hi = torch::empty_like(x_lo, opts);
  auto f_hi = torch::empty_like(x_lo, opts);

  const int64_t W = x_lo.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // full blocks
  const int64_t full = W / 256;
  const int64_t rem = W % 256;
  if (full > 0) {
    uvcc_gf2_and_prepare_pack32_v1<<<(unsigned)full, 256, 0, stream>>>(
        (const uint32_t*)x_lo.data_ptr<int32_t>(),
        (const uint32_t*)x_hi.data_ptr<int32_t>(),
        (const uint32_t*)y_lo.data_ptr<int32_t>(),
        (const uint32_t*)y_hi.data_ptr<int32_t>(),
        (const uint32_t*)a_lo.data_ptr<int32_t>(),
        (const uint32_t*)a_hi.data_ptr<int32_t>(),
        (const uint32_t*)b_lo.data_ptr<int32_t>(),
        (const uint32_t*)b_hi.data_ptr<int32_t>(),
        (uint32_t*)e_lo.data_ptr<int32_t>(),
        (uint32_t*)f_lo.data_ptr<int32_t>(),
        (uint32_t*)e_hi.data_ptr<int32_t>(),
        (uint32_t*)f_hi.data_ptr<int32_t>());
  }
  if (rem > 0) {
    const int64_t base = full * 256;
    uvcc_gf2_and_prepare_pack32_v1<<<1, (unsigned)rem, 0, stream>>>(
        (const uint32_t*)(x_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(x_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(y_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(y_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(a_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(a_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(b_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(b_hi.data_ptr<int32_t>() + base),
        (uint32_t*)(e_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(f_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(e_hi.data_ptr<int32_t>() + base),
        (uint32_t*)(f_hi.data_ptr<int32_t>() + base));
  }

  return {e_lo, f_lo, e_hi, f_hi};
}

std::vector<torch::Tensor> uvcc_gf2_and_finish_pack32_cuda(
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi,
    torch::Tensor c_lo,
    torch::Tensor c_hi,
    torch::Tensor e_pub,
    torch::Tensor f_pub,
    int64_t party_id) {
  check_i32_cuda_contig(a_lo, "a_lo");
  check_i32_cuda_contig(a_hi, "a_hi");
  check_i32_cuda_contig(b_lo, "b_lo");
  check_i32_cuda_contig(b_hi, "b_hi");
  check_i32_cuda_contig(c_lo, "c_lo");
  check_i32_cuda_contig(c_hi, "c_hi");
  check_i32_cuda_contig(e_pub, "e_pub");
  check_i32_cuda_contig(f_pub, "f_pub");
  TORCH_CHECK(a_lo.numel() == a_hi.numel(), "a size mismatch");
  TORCH_CHECK(a_lo.numel() == b_lo.numel(), "b size mismatch");
  TORCH_CHECK(a_lo.numel() == c_lo.numel(), "c size mismatch");
  TORCH_CHECK(a_lo.numel() == e_pub.numel(), "e_pub size mismatch");
  TORCH_CHECK(a_lo.numel() == f_pub.numel(), "f_pub size mismatch");
  TORCH_CHECK(party_id >= 0 && party_id <= 2, "party_id must be 0..2");

  auto dev = a_lo.device();
  auto opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);
  auto z_lo = torch::empty_like(a_lo, opts);
  auto z_hi = torch::empty_like(a_lo, opts);
  const int64_t W = a_lo.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int64_t full = W / 256;
  const int64_t rem = W % 256;
  if (full > 0) {
    uvcc_gf2_and_finish_pack32_v1<<<(unsigned)full, 256, 0, stream>>>(
        (const uint32_t*)a_lo.data_ptr<int32_t>(),
        (const uint32_t*)a_hi.data_ptr<int32_t>(),
        (const uint32_t*)b_lo.data_ptr<int32_t>(),
        (const uint32_t*)b_hi.data_ptr<int32_t>(),
        (const uint32_t*)c_lo.data_ptr<int32_t>(),
        (const uint32_t*)c_hi.data_ptr<int32_t>(),
        (const uint32_t*)e_pub.data_ptr<int32_t>(),
        (const uint32_t*)f_pub.data_ptr<int32_t>(),
        (uint32_t*)z_lo.data_ptr<int32_t>(),
        (uint32_t*)z_hi.data_ptr<int32_t>(),
        (uint32_t)party_id);
  }
  if (rem > 0) {
    const int64_t base = full * 256;
    uvcc_gf2_and_finish_pack32_v1<<<1, (unsigned)rem, 0, stream>>>(
        (const uint32_t*)(a_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(a_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(b_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(b_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(c_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(c_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(e_pub.data_ptr<int32_t>() + base),
        (const uint32_t*)(f_pub.data_ptr<int32_t>() + base),
        (uint32_t*)(z_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(z_hi.data_ptr<int32_t>() + base),
        (uint32_t)party_id);
  }
  return {z_lo, z_hi};
}

std::vector<torch::Tensor> uvcc_a2b_sub_prepare_pack32_cuda(
    torch::Tensor rj_lo,
    torch::Tensor rj_hi,
    torch::Tensor bj_lo,
    torch::Tensor bj_hi,
    torch::Tensor aj_lo,
    torch::Tensor aj_hi,
    torch::Tensor bjT_lo,
    torch::Tensor bjT_hi) {
  check_i32_cuda_contig(rj_lo, "rj_lo");
  check_i32_cuda_contig(rj_hi, "rj_hi");
  check_i32_cuda_contig(bj_lo, "bj_lo");
  check_i32_cuda_contig(bj_hi, "bj_hi");
  check_i32_cuda_contig(aj_lo, "aj_lo");
  check_i32_cuda_contig(aj_hi, "aj_hi");
  check_i32_cuda_contig(bjT_lo, "bjT_lo");
  check_i32_cuda_contig(bjT_hi, "bjT_hi");
  TORCH_CHECK(rj_lo.numel() == rj_hi.numel(), "rj size mismatch");
  TORCH_CHECK(rj_lo.numel() == bj_lo.numel(), "bj size mismatch");
  TORCH_CHECK(rj_lo.numel() == aj_lo.numel(), "aj size mismatch");
  TORCH_CHECK(rj_lo.numel() == bjT_lo.numel(), "bjT size mismatch");

  auto dev = rj_lo.device();
  auto opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);
  auto e_lo = torch::empty_like(rj_lo, opts);
  auto f_lo = torch::empty_like(rj_lo, opts);
  auto e_hi = torch::empty_like(rj_lo, opts);
  auto f_hi = torch::empty_like(rj_lo, opts);
  const int64_t W = rj_lo.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int64_t full = W / 256;
  const int64_t rem = W % 256;
  if (full > 0) {
    uvcc_a2b_sub_prepare_and_bit_pack32_v1<<<(unsigned)full, 256, 0, stream>>>(
        (const uint32_t*)rj_lo.data_ptr<int32_t>(),
        (const uint32_t*)rj_hi.data_ptr<int32_t>(),
        (const uint32_t*)bj_lo.data_ptr<int32_t>(),
        (const uint32_t*)bj_hi.data_ptr<int32_t>(),
        (const uint32_t*)aj_lo.data_ptr<int32_t>(),
        (const uint32_t*)aj_hi.data_ptr<int32_t>(),
        (const uint32_t*)bjT_lo.data_ptr<int32_t>(),
        (const uint32_t*)bjT_hi.data_ptr<int32_t>(),
        (uint32_t*)e_lo.data_ptr<int32_t>(),
        (uint32_t*)f_lo.data_ptr<int32_t>(),
        (uint32_t*)e_hi.data_ptr<int32_t>(),
        (uint32_t*)f_hi.data_ptr<int32_t>());
  }
  if (rem > 0) {
    const int64_t base = full * 256;
    uvcc_a2b_sub_prepare_and_bit_pack32_v1<<<1, (unsigned)rem, 0, stream>>>(
        (const uint32_t*)(rj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(rj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(bj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(bj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(aj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(aj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(bjT_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(bjT_hi.data_ptr<int32_t>() + base),
        (uint32_t*)(e_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(f_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(e_hi.data_ptr<int32_t>() + base),
        (uint32_t*)(f_hi.data_ptr<int32_t>() + base));
  }
  return {e_lo, f_lo, e_hi, f_hi};
}

std::vector<torch::Tensor> uvcc_a2b_sub_finish_pack32_cuda(
    torch::Tensor rj_lo,
    torch::Tensor rj_hi,
    torch::Tensor bj_lo,
    torch::Tensor bj_hi,
    torch::Tensor aj_lo,
    torch::Tensor aj_hi,
    torch::Tensor bjT_lo,
    torch::Tensor bjT_hi,
    torch::Tensor cj_lo,
    torch::Tensor cj_hi,
    torch::Tensor e_pub,
    torch::Tensor f_pub,
    torch::Tensor cj_public_mask,
    int64_t party_id) {
  check_i32_cuda_contig(rj_lo, "rj_lo");
  check_i32_cuda_contig(rj_hi, "rj_hi");
  check_i32_cuda_contig(bj_lo, "bj_lo");
  check_i32_cuda_contig(bj_hi, "bj_hi");
  check_i32_cuda_contig(aj_lo, "aj_lo");
  check_i32_cuda_contig(aj_hi, "aj_hi");
  check_i32_cuda_contig(bjT_lo, "bjT_lo");
  check_i32_cuda_contig(bjT_hi, "bjT_hi");
  check_i32_cuda_contig(cj_lo, "cj_lo");
  check_i32_cuda_contig(cj_hi, "cj_hi");
  check_i32_cuda_contig(e_pub, "e_pub");
  check_i32_cuda_contig(f_pub, "f_pub");
  check_i32_cuda_contig(cj_public_mask, "cj_public_mask");
  TORCH_CHECK(rj_lo.numel() == rj_hi.numel(), "rj size mismatch");
  TORCH_CHECK(rj_lo.numel() == bj_lo.numel(), "bj size mismatch");
  TORCH_CHECK(rj_lo.numel() == aj_lo.numel(), "aj size mismatch");
  TORCH_CHECK(rj_lo.numel() == bjT_lo.numel(), "bjT size mismatch");
  TORCH_CHECK(rj_lo.numel() == cj_lo.numel(), "cj size mismatch");
  TORCH_CHECK(rj_lo.numel() == e_pub.numel(), "e_pub size mismatch");
  TORCH_CHECK(rj_lo.numel() == cj_public_mask.numel(), "cj_public_mask size mismatch");
  TORCH_CHECK(party_id >= 0 && party_id <= 2, "party_id must be 0..2");

  auto dev = rj_lo.device();
  auto opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);
  auto x_lo = torch::empty_like(rj_lo, opts);
  auto x_hi = torch::empty_like(rj_lo, opts);
  auto bnext_lo = torch::empty_like(rj_lo, opts);
  auto bnext_hi = torch::empty_like(rj_lo, opts);
  const int64_t W = rj_lo.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int64_t full = W / 256;
  const int64_t rem = W % 256;
  if (full > 0) {
    uvcc_a2b_sub_finish_and_bit_pack32_v1<<<(unsigned)full, 256, 0, stream>>>(
        (const uint32_t*)rj_lo.data_ptr<int32_t>(),
        (const uint32_t*)rj_hi.data_ptr<int32_t>(),
        (const uint32_t*)bj_lo.data_ptr<int32_t>(),
        (const uint32_t*)bj_hi.data_ptr<int32_t>(),
        (const uint32_t*)aj_lo.data_ptr<int32_t>(),
        (const uint32_t*)aj_hi.data_ptr<int32_t>(),
        (const uint32_t*)bjT_lo.data_ptr<int32_t>(),
        (const uint32_t*)bjT_hi.data_ptr<int32_t>(),
        (const uint32_t*)cj_lo.data_ptr<int32_t>(),
        (const uint32_t*)cj_hi.data_ptr<int32_t>(),
        (const uint32_t*)e_pub.data_ptr<int32_t>(),
        (const uint32_t*)f_pub.data_ptr<int32_t>(),
        (const uint32_t*)cj_public_mask.data_ptr<int32_t>(),
        (uint32_t*)x_lo.data_ptr<int32_t>(),
        (uint32_t*)x_hi.data_ptr<int32_t>(),
        (uint32_t*)bnext_lo.data_ptr<int32_t>(),
        (uint32_t*)bnext_hi.data_ptr<int32_t>(),
        (uint32_t)party_id);
  }
  if (rem > 0) {
    const int64_t base = full * 256;
    uvcc_a2b_sub_finish_and_bit_pack32_v1<<<1, (unsigned)rem, 0, stream>>>(
        (const uint32_t*)(rj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(rj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(bj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(bj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(aj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(aj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(bjT_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(bjT_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(cj_lo.data_ptr<int32_t>() + base),
        (const uint32_t*)(cj_hi.data_ptr<int32_t>() + base),
        (const uint32_t*)(e_pub.data_ptr<int32_t>() + base),
        (const uint32_t*)(f_pub.data_ptr<int32_t>() + base),
        (const uint32_t*)(cj_public_mask.data_ptr<int32_t>() + base),
        (uint32_t*)(x_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(x_hi.data_ptr<int32_t>() + base),
        (uint32_t*)(bnext_lo.data_ptr<int32_t>() + base),
        (uint32_t*)(bnext_hi.data_ptr<int32_t>() + base),
        (uint32_t)party_id);
  }

  return {x_lo, x_hi, bnext_lo, bnext_hi};
}

torch::Tensor uvcc_a2b_pack_c_lo_u8_cuda(torch::Tensor x_lo_i64, torch::Tensor r_lo_i64) {
  check_i64_cuda_contig(x_lo_i64, "x_lo_i64");
  check_i64_cuda_contig(r_lo_i64, "r_lo_i64");
  TORCH_CHECK(x_lo_i64.numel() == r_lo_i64.numel(), "x/r size mismatch");
  auto dev = x_lo_i64.device();
  auto out = torch::empty({x_lo_i64.numel()}, torch::TensorOptions().device(dev).dtype(torch::kUInt8));
  const uint32_t n = (uint32_t)x_lo_i64.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  uvcc_a2b_pack_c_lo_u8_v1<<<blocks, threads, 0, stream>>>(
      (const uint64_t*)x_lo_i64.data_ptr<int64_t>(),
      (const uint64_t*)r_lo_i64.data_ptr<int64_t>(),
      (uint8_t*)out.data_ptr<uint8_t>(),
      n);
  return out;
}

torch::Tensor uvcc_a2b_pack_c_lo_u16_cuda(torch::Tensor x_lo_i64, torch::Tensor r_lo_i64) {
  check_i64_cuda_contig(x_lo_i64, "x_lo_i64");
  check_i64_cuda_contig(r_lo_i64, "r_lo_i64");
  TORCH_CHECK(x_lo_i64.numel() == r_lo_i64.numel(), "x/r size mismatch");
  auto dev = x_lo_i64.device();
  auto out = torch::empty({x_lo_i64.numel()}, torch::TensorOptions().device(dev).dtype(torch::kInt16));
  const uint32_t n = (uint32_t)x_lo_i64.numel();
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  uvcc_a2b_pack_c_lo_u16_v1<<<blocks, threads, 0, stream>>>(
      (const uint64_t*)x_lo_i64.data_ptr<int64_t>(),
      (const uint64_t*)r_lo_i64.data_ptr<int64_t>(),
      (uint16_t*)out.data_ptr<int16_t>(),
      n);
  return out;
}

torch::Tensor uvcc_a2b_cpub_to_cjmask_u8_cuda(torch::Tensor c_pub_u8) {
  check_u8_cuda_contig(c_pub_u8, "c_pub_u8");
  const uint32_t L = (uint32_t)c_pub_u8.numel();
  const uint32_t n_words = (L + 31u) >> 5;
  auto dev = c_pub_u8.device();
  auto out = torch::empty({(int64_t)8 * (int64_t)n_words}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 256; // 8 warps
  const int warps_per_block = threads / 32;
  const int blocks = (int)((n_words + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
  uvcc_a2b_cpub_to_cjmask_u8_v1<<<blocks, threads, 0, stream>>>(
      (const uint8_t*)c_pub_u8.data_ptr<uint8_t>(),
      (uint32_t*)out.data_ptr<int32_t>(),
      L);
  return out;
}

torch::Tensor uvcc_a2b_cpub_to_cjmask_u16_cuda(torch::Tensor c_pub_i16) {
  check_i16_cuda_contig(c_pub_i16, "c_pub_i16");
  const uint32_t L = (uint32_t)c_pub_i16.numel();
  const uint32_t n_words = (L + 31u) >> 5;
  auto dev = c_pub_i16.device();
  auto out = torch::empty({(int64_t)16 * (int64_t)n_words}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 256; // 8 warps
  const int warps_per_block = threads / 32;
  const int blocks = (int)((n_words + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
  uvcc_a2b_cpub_to_cjmask_u16_v1<<<blocks, threads, 0, stream>>>(
      (const uint16_t*)c_pub_i16.data_ptr<int16_t>(),
      (uint32_t*)out.data_ptr<int32_t>(),
      L);
  return out;
}


