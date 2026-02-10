#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

// -------------------------
// Unaligned little-endian loads
// -------------------------
static inline __device__ uint32_t load_le_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline __device__ uint64_t load_le_u64(const uint8_t* p) {
  uint64_t v = 0;
  #pragma unroll
  for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
  return v;
}
static inline __device__ void store_le_u64(uint8_t* p, uint64_t v) {
  #pragma unroll
  for (int i = 0; i < 8; i++) p[i] = (uint8_t)((v >> (8 * i)) & 0xFF);
}

// -------------------------
// AES-128 (tiny, reference) — ECB one-block
// -------------------------
__device__ __constant__ uint8_t UVCC_AES_SBOX[256] = {
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

static inline __device__ uint8_t xtime(uint8_t x) { return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b)); }

static inline __device__ void subbytes(uint8_t st[16]) {
  #pragma unroll
  for (int i = 0; i < 16; i++) st[i] = UVCC_AES_SBOX[st[i]];
}
static inline __device__ void shiftrows(uint8_t st[16]) {
  uint8_t t[16];
  t[0]=st[0]; t[1]=st[5]; t[2]=st[10]; t[3]=st[15];
  t[4]=st[4]; t[5]=st[9]; t[6]=st[14]; t[7]=st[3];
  t[8]=st[8]; t[9]=st[13]; t[10]=st[2]; t[11]=st[7];
  t[12]=st[12]; t[13]=st[1]; t[14]=st[6]; t[15]=st[11];
  #pragma unroll
  for (int i=0;i<16;i++) st[i]=t[i];
}
static inline __device__ void mixcolumns(uint8_t st[16]) {
  #pragma unroll
  for (int c = 0; c < 4; c++) {
    uint8_t* a = &st[4 * c];
    uint8_t t = a[0] ^ a[1] ^ a[2] ^ a[3];
    uint8_t u = a[0];
    a[0] ^= t ^ xtime(a[0] ^ a[1]);
    a[1] ^= t ^ xtime(a[1] ^ a[2]);
    a[2] ^= t ^ xtime(a[2] ^ a[3]);
    a[3] ^= t ^ xtime(a[3] ^ u);
  }
}
static inline __device__ void addroundkey(uint8_t st[16], const uint8_t rk[16]) {
  #pragma unroll
  for (int i=0;i<16;i++) st[i] ^= rk[i];
}
static inline __device__ void aes128_keyexp(uint8_t rk[11][16], const uint8_t key[16]) {
  static const uint8_t rcon[10] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};
  #pragma unroll
  for (int i=0;i<16;i++) rk[0][i] = key[i];
  #pragma unroll
  for (int i=1;i<=10;i++) {
    uint8_t* prev = rk[i-1];
    uint8_t* cur = rk[i];
    uint8_t t0 = prev[13], t1 = prev[14], t2 = prev[15], t3 = prev[12];
    t0 = UVCC_AES_SBOX[t0]; t1 = UVCC_AES_SBOX[t1]; t2 = UVCC_AES_SBOX[t2]; t3 = UVCC_AES_SBOX[t3];
    t0 ^= rcon[i-1];
    cur[0] = prev[0] ^ t0; cur[1] = prev[1] ^ t1; cur[2] = prev[2] ^ t2; cur[3] = prev[3] ^ t3;
    #pragma unroll
    for (int j=4;j<16;j++) cur[j] = prev[j] ^ cur[j-4];
  }
}
static inline __device__ void aes128_enc_block(uint8_t out[16], const uint8_t in[16], const uint8_t rk[11][16]) {
  uint8_t st[16];
  #pragma unroll
  for (int i=0;i<16;i++) st[i] = in[i];
  addroundkey(st, rk[0]);
  #pragma unroll
  for (int r=1;r<=9;r++) {
    subbytes(st); shiftrows(st); mixcolumns(st); addroundkey(st, rk[r]);
  }
  subbytes(st); shiftrows(st); addroundkey(st, rk[10]);
  #pragma unroll
  for (int i=0;i<16;i++) out[i] = st[i];
}

// AES constants C0..C4 (16-byte blocks with last byte set).
static inline __device__ void aes_const_block(uint8_t out[16], uint8_t last) {
  #pragma unroll
  for (int i=0;i<16;i++) out[i] = 0;
  out[15] = last;
}

static inline __device__ void g_expand_aes_v1(
    uint64_t seed_lo, uint64_t seed_hi,
    uint64_t& SL_lo, uint64_t& SL_hi, uint64_t& SR_lo, uint64_t& SR_hi,
    uint8_t& tL, uint8_t& tR) {
  uint8_t seed[16];
  store_le_u64(seed + 0, seed_lo);
  store_le_u64(seed + 8, seed_hi);
  uint8_t rk[11][16];
  aes128_keyexp(rk, seed);
  uint8_t c0[16],c1[16],c2[16],c3[16];
  aes_const_block(c0, 0);
  aes_const_block(c1, 1);
  aes_const_block(c2, 2);
  aes_const_block(c3, 3);
  uint8_t b0[16],b1[16],b2[16],b3[16];
  aes128_enc_block(b0,c0,rk);
  aes128_enc_block(b1,c1,rk);
  aes128_enc_block(b2,c2,rk);
  aes128_enc_block(b3,c3,rk);
  SL_lo = load_le_u64(b0 + 0);
  SL_hi = load_le_u64(b0 + 8);
  SR_lo = load_le_u64(b1 + 0);
  SR_hi = load_le_u64(b1 + 8);
  tL = (uint8_t)(b2[0] & 1u);
  tR = (uint8_t)(b3[0] & 1u);
}

static inline __device__ uint64_t V_aes_u64_v1(uint64_t seed_lo, uint64_t seed_hi) {
  uint8_t seed[16];
  store_le_u64(seed + 0, seed_lo);
  store_le_u64(seed + 8, seed_hi);
  uint8_t rk[11][16];
  aes128_keyexp(rk, seed);
  uint8_t c4[16], b4[16];
  aes_const_block(c4, 4);
  aes128_enc_block(b4, c4, rk);
  return load_le_u64(b4 + 0);
}

// -------------------------
// ChaCha20 block (for refresh masks): returns first 8 bytes as u64.
// -------------------------
static inline __device__ uint32_t rotl32(uint32_t x, int r){ return (x<<r) | (x>>(32-r)); }
static inline __device__ void qr(uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d){
  a += b; d ^= a; d = rotl32(d,16);
  c += d; b ^= c; b = rotl32(b,12);
  a += b; d ^= a; d = rotl32(d, 8);
  c += d; b ^= c; b = rotl32(b, 7);
}
static inline __device__ uint32_t load_u32_le(const uint8_t* p){
  return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static inline __device__ void store_u32_le(uint8_t* p, uint32_t x){
  p[0]=(uint8_t)x; p[1]=(uint8_t)(x>>8); p[2]=(uint8_t)(x>>16); p[3]=(uint8_t)(x>>24);
}

static inline __device__ void chacha20_block(uint8_t out64[64], const uint8_t key32[32], const uint8_t nonce12[12], uint32_t counter){
  // constants "expand 32-byte k"
  uint32_t st[16], x[16];
  st[0]=0x61707865u; st[1]=0x3320646eu; st[2]=0x79622d32u; st[3]=0x6b206574u;
  #pragma unroll
  for(int i=0;i<8;i++) st[4+i]=load_u32_le(key32 + 4*i);
  st[12]=counter;
  st[13]=load_u32_le(nonce12+0);
  st[14]=load_u32_le(nonce12+4);
  st[15]=load_u32_le(nonce12+8);
  #pragma unroll
  for(int i=0;i<16;i++) x[i]=st[i];
  // 20 rounds = 10 double-rounds
  #pragma unroll
  for(int r=0;r<20;r+=2){
    qr(x[0],x[4],x[8], x[12]);
    qr(x[1],x[5],x[9], x[13]);
    qr(x[2],x[6],x[10],x[14]);
    qr(x[3],x[7],x[11],x[15]);
    qr(x[0],x[5],x[10],x[15]);
    qr(x[1],x[6],x[11],x[12]);
    qr(x[2],x[7],x[8], x[13]);
    qr(x[3],x[4],x[9], x[14]);
  }
  #pragma unroll
  for(int i=0;i<16;i++) x[i] += st[i];
  #pragma unroll
  for(int i=0;i<16;i++) store_u32_le(out64 + 4*i, x[i]);
}

static inline __device__ uint64_t chacha20_first_u64(const uint8_t key32[32], const uint8_t nonce12[12], uint32_t counter){
  uint8_t out64[64];
  chacha20_block(out64, key32, nonce12, counter);
  return load_le_u64(out64);
}

// -------------------------
// OP_LUT record parser helpers (device-side)
// -------------------------
static inline __device__ const uint8_t* find_section(
    const uint8_t* rec,
    uint32_t header_bytes,
    uint32_t section_count,
    uint32_t want_type,
    uint64_t& out_bytes) {
  const int hdr_size = 40;  // UVCC_FSSRecordHeader_v1 in op_lut_blob.py
  const int dirent_size = 24;
  const uint8_t* dir0 = rec + hdr_size;
  const uint8_t* payload0 = rec + header_bytes;
  #pragma unroll
  for (int i = 0; i < 64; i++) {  // section_count is tiny; cap for safety
    if ((uint32_t)i >= section_count) break;
    const uint8_t* de = dir0 + i * dirent_size;
    uint32_t type = load_le_u32(de + 0);
    // uint32 flags = load_le_u32(de + 4);
    uint64_t off = load_le_u64(de + 8);
    uint64_t bytes = load_le_u64(de + 16);
    if (type == want_type) {
      out_bytes = bytes;
      return payload0 + off;
    }
  }
  out_bytes = 0;
  return nullptr;
}

// -------------------------
// Kernel: OP_LUT Phase-2 (w=8) local eval (DPF dot product + refresh) for one record blob
// -------------------------
__global__ void uvcc_oplut_phase2_w8_record_v1(
    const uint8_t* __restrict__ rec_bytes,
    uint32_t rec_len,
    const uint8_t* __restrict__ u_pub_u8,
    const int64_t* __restrict__ table_i64,
    int64_t* __restrict__ out_yprime_i64) {
  const int lane = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;

  // Parse header (40 bytes)
  if (rec_len < 40) return;
  // magic[8] = "UVFSSv1\0"
  if (!(rec_bytes[0]=='U' && rec_bytes[1]=='V' && rec_bytes[2]=='F' && rec_bytes[3]=='S')) return;
  const uint32_t ver = load_le_u32(rec_bytes + 8);
  if (ver != 1) return;
  const uint32_t kind = load_le_u32(rec_bytes + 12);
  if (kind != 0x00000020u) return;  // OP_LUT
  const uint32_t header_bytes = load_le_u32(rec_bytes + 24);
  const uint32_t section_count = load_le_u32(rec_bytes + 28);
  const uint64_t payload_bytes = load_le_u64(rec_bytes + 32);
  if (header_bytes < 40u) return;
  if ((uint64_t)header_bytes + payload_bytes != (uint64_t)rec_len) return;

  // META section (type=1)
  uint64_t meta_len = 0;
  const uint8_t* meta = find_section(rec_bytes, header_bytes, section_count, 0x00000001u, meta_len);
  if (!meta || meta_len < 44) return;
  const uint8_t party_id = meta[0];
  const uint8_t dpf_role = meta[1];
  const uint8_t domain_w = meta[2];
  const uint32_t lanes = load_le_u32(meta + 4);
  const uint32_t ring_id = load_le_u32(meta + 8);
  const uint8_t prg_id = meta[12];
  const uint64_t lane_base = load_le_u64(meta + 32);
  const uint16_t dpf_key_bytes_per_lane = (uint16_t)(meta[40] | ((uint16_t)meta[41] << 8));
  if (domain_w != 8) return;
  if (ring_id != 1u) return;
  if (prg_id != 1u) return;  // ChaCha20 for masks
  if (lane >= (int)lanes) return;

  // MASK_RSS (type=2) and REFRESH (type=4)
  uint64_t mask_len = 0;
  const uint8_t* mask = find_section(rec_bytes, header_bytes, section_count, 0x00000002u, mask_len);
  if (!mask || mask_len < 96) return;
  const uint8_t edge_a = mask[0];
  const uint8_t edge_b = mask[1];
  const uint8_t* seed_a32 = mask + 4;
  const uint8_t* seed_b32 = mask + 36;

  uint64_t ref_len = 0;
  const uint8_t* ref = find_section(rec_bytes, header_bytes, section_count, 0x00000004u, ref_len);
  if (!ref || ref_len < 32) return;
  const uint8_t uses_edge20 = ref[0];
  const uint8_t uses_edge12 = ref[1];
  const uint8_t* nonce_m12 = ref + 4;
  const uint64_t counter0 = load_le_u64(ref + 16);
  const uint64_t lane_stride = load_le_u64(ref + 24);

  // Resolve edge20/edge12 seeds from the two seeds in MASK_RSS.
  const uint8_t* seed_edge20 = nullptr;
  const uint8_t* seed_edge12 = nullptr;
  if (edge_a == 3) seed_edge20 = seed_a32;
  if (edge_b == 3) seed_edge20 = seed_b32;
  if (edge_a == 2) seed_edge12 = seed_a32;
  if (edge_b == 2) seed_edge12 = seed_b32;

  const uint32_t ctr32 = (uint32_t)((counter0 + (lane_base + (uint64_t)lane) * lane_stride) & 0xFFFFFFFFu);
  const uint64_t m0 = ((uses_edge20 & 1u) && seed_edge20) ? chacha20_first_u64(seed_edge20, nonce_m12, ctr32) : 0ull;
  const uint64_t m1 = ((uses_edge12 & 1u) && seed_edge12) ? chacha20_first_u64(seed_edge12, nonce_m12, ctr32) : 0ull;

  // DPF_2PC (type=3) is optional (P2 has none).
  uint64_t y_share = 0ull;
  if (dpf_role == 1 || dpf_role == 2) {
    uint64_t dpf_len = 0;
    const uint8_t* dpf = find_section(rec_bytes, header_bytes, section_count, 0x00000003u, dpf_len);
    if (!dpf || dpf_len < 12) return;
    const uint8_t key_dw = dpf[1];
    const uint16_t key_bytes_per_lane = (uint16_t)(dpf[4] | ((uint16_t)dpf[5] << 8));
    const uint32_t key_lanes = load_le_u32(dpf + 8);
    if (key_dw != 8) return;
    if (key_lanes != lanes) return;
    if (key_bytes_per_lane != dpf_key_bytes_per_lane) return;
    const uint8_t* keys = dpf + 12;
    const uint8_t* key_lane = keys + (uint64_t)lane * (uint64_t)key_bytes_per_lane;

    const uint8_t root_t = key_lane[16] & 1u;
    uint64_t seeds_lo[256];
    uint64_t seeds_hi[256];
    uint8_t tbits[256];
    seeds_lo[0] = load_le_u64(key_lane + 0);
    seeds_hi[0] = load_le_u64(key_lane + 8);
    tbits[0] = root_t;
    int nodes = 1;
    const int w = 8;
    const int cw_off0 = 17;
    const int cw_stride = 34;  // seed_L16 + seed_R16 + tL + tR
    const int cw_last_off = cw_off0 + cw_stride * w;
    const uint64_t cw_last = load_le_u64(key_lane + cw_last_off);

    // Expand in-place per level (parents processed high->low).
    #pragma unroll
    for (int d = 0; d < w; d++) {
      const int cw_off = cw_off0 + d * cw_stride;
      const uint64_t seedL_lo = load_le_u64(key_lane + cw_off + 0);
      const uint64_t seedL_hi = load_le_u64(key_lane + cw_off + 8);
      const uint64_t seedR_lo = load_le_u64(key_lane + cw_off + 16);
      const uint64_t seedR_hi = load_le_u64(key_lane + cw_off + 24);
      const uint8_t tauL = key_lane[cw_off + 32] & 1u;
      const uint8_t tauR = key_lane[cw_off + 33] & 1u;
      for (int i = nodes - 1; i >= 0; i--) {
        uint64_t seed_lo = seeds_lo[i];
        uint64_t seed_hi = seeds_hi[i];
        uint8_t t = tbits[i] & 1u;
        uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
        uint8_t tL, tR;
        g_expand_aes_v1(seed_lo, seed_hi, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
        if (t) {
          SL_lo ^= seedL_lo; SL_hi ^= seedL_hi;
          SR_lo ^= seedR_lo; SR_hi ^= seedR_hi;
          tL ^= tauL;
          tR ^= tauR;
        }
        const int li = i << 1;
        seeds_lo[li + 0] = SL_lo;
        seeds_hi[li + 0] = SL_hi;
        tbits[li + 0] = tL & 1u;
        seeds_lo[li + 1] = SR_lo;
        seeds_hi[li + 1] = SR_hi;
        tbits[li + 1] = tR & 1u;
      }
      nodes <<= 1;
    }

    const uint8_t u = u_pub_u8[lane];
    uint64_t acc = 0ull;
    #pragma unroll
    for (int v = 0; v < 256; v++) {
      uint64_t val = V_aes_u64_v1(seeds_lo[v], seeds_hi[v]);
      if (tbits[v] & 1u) val = val + cw_last;
      if (dpf_role == 2) val = (uint64_t)0 - val;  // right party has negative sign
      const uint8_t j = (uint8_t)(u - (uint8_t)v);
      const uint64_t tword = (uint64_t)table_i64[(int)j];
      acc = acc + (val * tword);
    }
    y_share = acc;
  }

  uint64_t y_prime_u64 = 0ull;
  if (party_id == 0) {
    y_prime_u64 = y_share + m0;
  } else if (party_id == 1) {
    y_prime_u64 = y_share + m1;
  } else {
    y_prime_u64 = (uint64_t)0 - (m0 + m1);
  }
  out_yprime_i64[lane] = (int64_t)y_prime_u64;
}

// -------------------------
// C++/PyTorch wrapper
// -------------------------
static void check_u8_cuda_contig(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kUInt8, name, " must be uint8");
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

torch::Tensor uvcc_oplut_phase2_w8_record_cuda(torch::Tensor record_u8, torch::Tensor u_pub_u8, torch::Tensor table_i64) {
  check_u8_cuda_contig(record_u8, "record_u8");
  check_u8_cuda_contig(u_pub_u8, "u_pub_u8");
  check_i64_cuda_contig(table_i64, "table_i64");
  TORCH_CHECK(table_i64.numel() == 256, "table_i64 must be length 256 for w=8");

  auto dev = record_u8.device();
  auto lanes = (int64_t)u_pub_u8.numel();
  auto out = torch::empty({lanes}, torch::TensorOptions().device(dev).dtype(torch::kInt64));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int threads = 32;
  const int blocks = (int)((lanes + threads - 1) / threads);
  uvcc_oplut_phase2_w8_record_v1<<<blocks, threads, 0, stream>>>(
      (const uint8_t*)record_u8.data_ptr<uint8_t>(),
      (uint32_t)record_u8.numel(),
      (const uint8_t*)u_pub_u8.data_ptr<uint8_t>(),
      (const int64_t*)table_i64.data_ptr<int64_t>(),
      (int64_t*)out.data_ptr<int64_t>());
  return out;
}

// -------------------------
// w=16 kernels: stage2 partials + reduce
// -------------------------
__global__ void uvcc_oplut_phase2_w16_record_partials_v1(
    const uint8_t* __restrict__ rec_bytes,
    uint32_t rec_len,
    const uint16_t* __restrict__ u_pub_u16,
    const int64_t* __restrict__ table_i64,
    uint64_t* __restrict__ partial_u64,  // [lanes*256]
    uint32_t lanes) {
  const int prefix = (int)blockIdx.x;      // 0..255
  const int lane = (int)blockIdx.y;        // 0..lanes-1
  const int j = (int)threadIdx.x;          // 0..255
  if (prefix >= 256 || lane >= (int)lanes || j >= 256) return;

  // Parse header/meta (same as w8) — minimal, assumes validated by wrapper.
  const uint32_t header_bytes = load_le_u32(rec_bytes + 24);
  const uint32_t section_count = load_le_u32(rec_bytes + 28);

  uint64_t meta_len = 0;
  const uint8_t* meta = find_section(rec_bytes, header_bytes, section_count, 0x00000001u, meta_len);
  const uint8_t dpf_role = meta ? meta[1] : 0;
  if (!(dpf_role == 1 || dpf_role == 2)) {
    // No DPF section (P2): partials are zero.
    if (j == 0) partial_u64[lane * 256 + prefix] = 0;
    return;
  }

  uint64_t dpf_len = 0;
  const uint8_t* dpf = find_section(rec_bytes, header_bytes, section_count, 0x00000003u, dpf_len);
  if (!dpf) return;
  const uint16_t key_bytes_per_lane = (uint16_t)(dpf[4] | ((uint16_t)dpf[5] << 8));
  const uint8_t* keys = dpf + 12;
  const uint8_t* key_lane = keys + (uint64_t)lane * (uint64_t)key_bytes_per_lane;

  // Compute frontier seed/t for this prefix in thread0, broadcast via shared.
  __shared__ uint64_t s_seed_lo;
  __shared__ uint64_t s_seed_hi;
  __shared__ uint8_t s_t;
  if (j == 0) {
    uint64_t seed_lo = load_le_u64(key_lane + 0);
    uint64_t seed_hi = load_le_u64(key_lane + 8);
    uint8_t t = key_lane[16] & 1u;
    const int w = 16;
    const int cw_off0 = 17;
    const int cw_stride = 34;
    // Traverse 8 levels using prefix bits (MSB-first).
    #pragma unroll
    for (int d = 0; d < 8; d++) {
      const int cw_off = cw_off0 + d * cw_stride;
      const uint64_t seedL_lo = load_le_u64(key_lane + cw_off + 0);
      const uint64_t seedL_hi = load_le_u64(key_lane + cw_off + 8);
      const uint64_t seedR_lo = load_le_u64(key_lane + cw_off + 16);
      const uint64_t seedR_hi = load_le_u64(key_lane + cw_off + 24);
      const uint8_t tauL = key_lane[cw_off + 32] & 1u;
      const uint8_t tauR = key_lane[cw_off + 33] & 1u;
      uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
      uint8_t tL, tR;
      g_expand_aes_v1(seed_lo, seed_hi, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
      if (t) {
        SL_lo ^= seedL_lo; SL_hi ^= seedL_hi;
        SR_lo ^= seedR_lo; SR_hi ^= seedR_hi;
        tL ^= tauL;
        tR ^= tauR;
      }
      const int bit = (prefix >> (7 - d)) & 1;
      if (bit == 0) { seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u; }
      else          { seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u; }
    }
    s_seed_lo = seed_lo;
    s_seed_hi = seed_hi;
    s_t = t & 1u;
  }
  __syncthreads();

  // Now evaluate the 8-bit suffix path (j) from the frontier state using levels 8..15.
  uint64_t seed_lo = s_seed_lo;
  uint64_t seed_hi = s_seed_hi;
  uint8_t t = s_t & 1u;
  const int w = 16;
  const int cw_off0 = 17;
  const int cw_stride = 34;
  const int cw_last_off = cw_off0 + cw_stride * w;
  const uint64_t cw_last = load_le_u64(key_lane + cw_last_off);

  #pragma unroll
  for (int d = 8; d < 16; d++) {
    const int cw_off = cw_off0 + d * cw_stride;
    const uint64_t seedL_lo = load_le_u64(key_lane + cw_off + 0);
    const uint64_t seedL_hi = load_le_u64(key_lane + cw_off + 8);
    const uint64_t seedR_lo = load_le_u64(key_lane + cw_off + 16);
    const uint64_t seedR_hi = load_le_u64(key_lane + cw_off + 24);
    const uint8_t tauL = key_lane[cw_off + 32] & 1u;
    const uint8_t tauR = key_lane[cw_off + 33] & 1u;
    uint64_t SL_lo, SL_hi, SR_lo, SR_hi;
    uint8_t tL, tR;
    g_expand_aes_v1(seed_lo, seed_hi, SL_lo, SL_hi, SR_lo, SR_hi, tL, tR);
    if (t) {
      SL_lo ^= seedL_lo; SL_hi ^= seedL_hi;
      SR_lo ^= seedR_lo; SR_hi ^= seedR_hi;
      tL ^= tauL;
      tR ^= tauR;
    }
    const int bit = (j >> (15 - d)) & 1;  // j bits MSB-first for suffix
    if (bit == 0) { seed_lo = SL_lo; seed_hi = SL_hi; t = tL & 1u; }
    else          { seed_lo = SR_lo; seed_hi = SR_hi; t = tR & 1u; }
  }

  uint64_t val = V_aes_u64_v1(seed_lo, seed_hi);
  if (t) val = val + cw_last;
  if (dpf_role == 2) val = (uint64_t)0 - val;

  const uint16_t u = u_pub_u16[lane];
  const uint16_t v = (uint16_t)((prefix << 8) | j);
  const uint16_t idx = (uint16_t)(u - v);
  const uint64_t tword = (uint64_t)table_i64[(int)idx];
  const uint64_t prod = val * tword;

  // Reduce 256 products to one partial sum for this (lane,prefix).
  __shared__ uint64_t sh[256];
  sh[j] = prod;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (j < offset) sh[j] += sh[j + offset];
    __syncthreads();
  }
  if (j == 0) partial_u64[lane * 256 + prefix] = sh[0];
}

__global__ void uvcc_oplut_phase2_w16_record_reduce_v1(
    const uint8_t* __restrict__ rec_bytes,
    uint32_t rec_len,
    const uint64_t* __restrict__ partial_u64,  // [lanes*256]
    int64_t* __restrict__ out_yprime_i64,
    uint32_t lanes) {
  const int lane = (int)blockIdx.x;
  const int tid = (int)threadIdx.x;
  if (lane >= (int)lanes) return;

  // Sum partials across prefixes.
  __shared__ uint64_t sh[256];
  if (tid < 256) sh[tid] = partial_u64[lane * 256 + tid];
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (tid < offset) sh[tid] += sh[tid + offset];
    __syncthreads();
  }
  const uint64_t y_share = sh[0];

  // Parse header/meta/mask/refresh to compute refresh masks.
  if (tid != 0) return;
  if (rec_len < 40) return;
  const uint32_t header_bytes = load_le_u32(rec_bytes + 24);
  const uint32_t section_count = load_le_u32(rec_bytes + 28);

  uint64_t meta_len = 0;
  const uint8_t* meta = find_section(rec_bytes, header_bytes, section_count, 0x00000001u, meta_len);
  if (!meta || meta_len < 44) return;
  const uint8_t party_id = meta[0];
  const uint8_t domain_w = meta[2];
  const uint8_t prg_id = meta[12];
  const uint64_t lane_base = load_le_u64(meta + 32);
  if (domain_w != 16) return;
  if (prg_id != 1u) return;

  uint64_t mask_len = 0;
  const uint8_t* mask = find_section(rec_bytes, header_bytes, section_count, 0x00000002u, mask_len);
  if (!mask || mask_len < 96) return;
  const uint8_t edge_a = mask[0];
  const uint8_t edge_b = mask[1];
  const uint8_t* seed_a32 = mask + 4;
  const uint8_t* seed_b32 = mask + 36;

  uint64_t ref_len = 0;
  const uint8_t* ref = find_section(rec_bytes, header_bytes, section_count, 0x00000004u, ref_len);
  if (!ref || ref_len < 32) return;
  const uint8_t uses_edge20 = ref[0];
  const uint8_t uses_edge12 = ref[1];
  const uint8_t* nonce_m12 = ref + 4;
  const uint64_t counter0 = load_le_u64(ref + 16);
  const uint64_t lane_stride = load_le_u64(ref + 24);

  const uint8_t* seed_edge20 = nullptr;
  const uint8_t* seed_edge12 = nullptr;
  if (edge_a == 3) seed_edge20 = seed_a32;
  if (edge_b == 3) seed_edge20 = seed_b32;
  if (edge_a == 2) seed_edge12 = seed_a32;
  if (edge_b == 2) seed_edge12 = seed_b32;

  const uint32_t ctr32 = (uint32_t)((counter0 + (lane_base + (uint64_t)lane) * lane_stride) & 0xFFFFFFFFu);
  const uint64_t m0 = ((uses_edge20 & 1u) && seed_edge20) ? chacha20_first_u64(seed_edge20, nonce_m12, ctr32) : 0ull;
  const uint64_t m1 = ((uses_edge12 & 1u) && seed_edge12) ? chacha20_first_u64(seed_edge12, nonce_m12, ctr32) : 0ull;

  uint64_t y_prime_u64 = 0ull;
  if (party_id == 0) y_prime_u64 = y_share + m0;
  else if (party_id == 1) y_prime_u64 = y_share + m1;
  else y_prime_u64 = (uint64_t)0 - (m0 + m1);
  out_yprime_i64[lane] = (int64_t)y_prime_u64;
}

torch::Tensor uvcc_oplut_phase2_w16_record_cuda(torch::Tensor record_u8, torch::Tensor u_pub_i16, torch::Tensor table_i64) {
  check_u8_cuda_contig(record_u8, "record_u8");
  check_i16_cuda_contig(u_pub_i16, "u_pub_i16");
  check_i64_cuda_contig(table_i64, "table_i64");
  TORCH_CHECK(table_i64.numel() == 65536, "table_i64 must be length 65536 for w=16");

  auto dev = record_u8.device();
  auto lanes = (uint32_t)u_pub_i16.numel();
  auto out = torch::empty({(int64_t)lanes}, torch::TensorOptions().device(dev).dtype(torch::kInt64));

  // temp partials [lanes*256] u64 stored in int64 tensor
  auto partial = torch::empty({(int64_t)lanes * 256}, torch::TensorOptions().device(dev).dtype(torch::kInt64));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  dim3 grid(256, lanes, 1);
  dim3 block(256, 1, 1);
  uvcc_oplut_phase2_w16_record_partials_v1<<<grid, block, 0, stream>>>(
      (const uint8_t*)record_u8.data_ptr<uint8_t>(),
      (uint32_t)record_u8.numel(),
      (const uint16_t*)u_pub_i16.data_ptr<int16_t>(),
      (const int64_t*)table_i64.data_ptr<int64_t>(),
      (uint64_t*)partial.data_ptr<int64_t>(),
      lanes);
  uvcc_oplut_phase2_w16_record_reduce_v1<<<lanes, 256, 0, stream>>>(
      (const uint8_t*)record_u8.data_ptr<uint8_t>(),
      (uint32_t)record_u8.numel(),
      (const uint64_t*)partial.data_ptr<int64_t>(),
      (int64_t*)out.data_ptr<int64_t>(),
      lanes);
  return out;
}


