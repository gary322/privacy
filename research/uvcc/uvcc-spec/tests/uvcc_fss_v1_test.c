// uvcc_fss_v1_test.c
// Build: clang -O2 -std=c11 uvcc_fss_v1_test.c -o uvcc_test
// Runs:  ./uvcc_test
//
// Prints canonical bytes for:
// - AES-128 test vector
// - ChaCha12 block (reference implementation)
// - Deterministic DPF/DCF keyrec v1 bytes (ChaCha-derived root seeds, AES/ChaCha PRG selectable)
// - DPF full-domain t-vector (w=8)
// - DCF full-domain masked word vector (w=8), payload_mask=1

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void hexdump(const char* label, const uint8_t* p, size_t n) {
  printf("%s (%zu):\n", label, n);
  for (size_t i=0;i<n;i++) {
    printf("%02x", p[i]);
    if ((i & 15) == 15) printf("\n");
    else if ((i & 1) == 1) printf(" ");
  }
  if ((n & 15) != 0) printf("\n");
}

// ------------------------ ChaCha12 (RFC-style core) ------------------------
static inline uint32_t rotl32(uint32_t x, int r){ return (x<<r) | (x>>(32-r)); }
static inline uint32_t load32_le(const uint8_t* p){
  return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static inline void store32_le(uint8_t* p, uint32_t x){
  p[0]=(uint8_t)x; p[1]=(uint8_t)(x>>8); p[2]=(uint8_t)(x>>16); p[3]=(uint8_t)(x>>24);
}
static void quarterround(uint32_t* a,uint32_t* b,uint32_t* c,uint32_t* d){
  *a += *b; *d ^= *a; *d = rotl32(*d,16);
  *c += *d; *b ^= *c; *b = rotl32(*b,12);
  *a += *b; *d ^= *a; *d = rotl32(*d, 8);
  *c += *d; *b ^= *c; *b = rotl32(*b, 7);
}

static void chacha12_block_v1(uint8_t out64[64], const uint8_t key32[32], const uint8_t nonce12[12], uint32_t counter){
  static const uint8_t sigma[16] = "expand 32-byte k";
  uint32_t st[16], w[16];

  st[0]=load32_le(sigma+0);
  st[1]=load32_le(sigma+4);
  st[2]=load32_le(sigma+8);
  st[3]=load32_le(sigma+12);

  for(int i=0;i<8;i++) st[4+i]=load32_le(key32+4*i);

  st[12]=counter;
  st[13]=load32_le(nonce12+0);
  st[14]=load32_le(nonce12+4);
  st[15]=load32_le(nonce12+8);

  memcpy(w, st, sizeof(w));

  for(int r=0;r<12;r+=2){
    // column rounds
    quarterround(&w[0],&w[4],&w[8], &w[12]);
    quarterround(&w[1],&w[5],&w[9], &w[13]);
    quarterround(&w[2],&w[6],&w[10],&w[14]);
    quarterround(&w[3],&w[7],&w[11],&w[15]);
    // diagonal rounds
    quarterround(&w[0],&w[5],&w[10],&w[15]);
    quarterround(&w[1],&w[6],&w[11],&w[12]);
    quarterround(&w[2],&w[7],&w[8], &w[13]);
    quarterround(&w[3],&w[4],&w[9], &w[14]);
  }

  for(int i=0;i<16;i++) w[i] += st[i];
  for(int i=0;i<16;i++) store32_le(out64 + 4*i, w[i]);
}

// ------------------------ Minimal AES-128 (ECB one-block) -------------------
// NOTE: For brevity, this harness only prints a known AES vector using a tiny implementation.
// If you already have AES code, you must ensure your GPU ABI byte order matches your AES kernel.

static const uint8_t sbox[256]={
  // standard AES s-box
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

static uint8_t xtime(uint8_t x){ return (uint8_t)((x<<1) ^ ((x>>7)*0x1b)); }

static void subbytes(uint8_t st[16]){
  for(int i=0;i<16;i++) st[i]=sbox[st[i]];
}
static void shiftrows(uint8_t st[16]){
  uint8_t t[16];
  t[0]=st[0]; t[1]=st[5]; t[2]=st[10]; t[3]=st[15];
  t[4]=st[4]; t[5]=st[9]; t[6]=st[14]; t[7]=st[3];
  t[8]=st[8]; t[9]=st[13]; t[10]=st[2]; t[11]=st[7];
  t[12]=st[12]; t[13]=st[1]; t[14]=st[6]; t[15]=st[11];
  memcpy(st,t,16);
}
static void mixcolumns(uint8_t st[16]){
  for(int c=0;c<4;c++){
    uint8_t* a=&st[4*c];
    uint8_t t=a[0]^a[1]^a[2]^a[3];
    uint8_t u=a[0];
    a[0]^=t^xtime(a[0]^a[1]);
    a[1]^=t^xtime(a[1]^a[2]);
    a[2]^=t^xtime(a[2]^a[3]);
    a[3]^=t^xtime(a[3]^u);
  }
}
static void addroundkey(uint8_t st[16], const uint8_t rk[16]){
  for(int i=0;i<16;i++) st[i]^=rk[i];
}
static void aes128_keyexp(uint8_t rk[11][16], const uint8_t key[16]){
  static const uint8_t rcon[10]={0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};
  memcpy(rk[0], key, 16);
  for(int i=1;i<=10;i++){
    uint8_t* prev=rk[i-1];
    uint8_t* cur=rk[i];
    uint8_t t0=prev[13], t1=prev[14], t2=prev[15], t3=prev[12];
    t0=sbox[t0]; t1=sbox[t1]; t2=sbox[t2]; t3=sbox[t3];
    t0^=rcon[i-1];
    cur[0]=prev[0]^t0; cur[1]=prev[1]^t1; cur[2]=prev[2]^t2; cur[3]=prev[3]^t3;
    for(int j=4;j<16;j++) cur[j]=prev[j]^cur[j-4];
  }
}
static void aes128_enc_block(uint8_t out[16], const uint8_t in[16], const uint8_t rk[11][16]){
  uint8_t st[16]; memcpy(st,in,16);
  addroundkey(st,rk[0]);
  for(int r=1;r<=9;r++){
    subbytes(st); shiftrows(st); mixcolumns(st); addroundkey(st,rk[r]);
  }
  subbytes(st); shiftrows(st); addroundkey(st,rk[10]);
  memcpy(out,st,16);
}

// ------------------------ UVCC DPF/DCF v1 (w=8 demo) ------------------------
static void G_expand_chacha(uint8_t SL[16], uint8_t SR[16], uint8_t* tmask,
                            const uint8_t seed16[16], uint32_t depth){
  uint8_t key32[32];
  memcpy(key32, seed16, 16);
  memcpy(key32+16, seed16, 16);
  uint8_t nonce12[12]={0};
  // "G2_v1"||depth (simple)
  nonce12[0]='G'; nonce12[1]='2'; nonce12[2]='_'; nonce12[3]='v'; nonce12[4]='1';
  nonce12[8]=(uint8_t)depth;
  uint8_t block0[64], block1[64];
  chacha12_block_v1(block0,key32,nonce12,0);
  chacha12_block_v1(block1,key32,nonce12,1);
  memcpy(SL, block0+0, 16);
  memcpy(SR, block0+16,16);
  uint8_t tL = block1[0] & 1;
  uint8_t tR = block1[1] & 1;
  *tmask = (uint8_t)(tL | (tR<<1));
}

static void dpf_keygen_w8(uint8_t root0[16], uint8_t root1[16],
                          uint8_t sigma[8][16], uint8_t tau[8],
                          uint8_t alpha){
  uint8_t S0[16], S1[16];
  memcpy(S0, root0, 16);
  memcpy(S1, root1, 16);
  uint8_t t0=0, t1=1;

  for(uint32_t d=0; d<8; d++){
    uint8_t SL0[16], SR0[16], SL1[16], SR1[16];
    uint8_t tm0, tm1;
    G_expand_chacha(SL0,SR0,&tm0,S0,d);
    G_expand_chacha(SL1,SR1,&tm1,S1,d);
    uint8_t tL0=tm0&1, tR0=(tm0>>1)&1;
    uint8_t tL1=tm1&1, tR1=(tm1>>1)&1;

    uint8_t abit = (alpha >> (7-d)) & 1; // MSB-first
    if(abit==0){
      // sigma = SR0 XOR SR1
      for(int i=0;i<16;i++) sigma[d][i]=SR0[i]^SR1[i];
      uint8_t tauL = tL0 ^ tL1 ^ 1;
      uint8_t tauR = tR0 ^ tR1;
      tau[d] = (uint8_t)(tauL | (tauR<<1));
      memcpy(S0, SL0,16); t0=tL0;
      memcpy(S1, SL1,16); t1=tL1;
    } else {
      // sigma = SL0 XOR SL1
      for(int i=0;i<16;i++) sigma[d][i]=SL0[i]^SL1[i];
      uint8_t tauL = tL0 ^ tL1;
      uint8_t tauR = tR0 ^ tR1 ^ 1;
      tau[d] = (uint8_t)(tauL | (tauR<<1));
      memcpy(S0, SR0,16); t0=tR0;
      memcpy(S1, SR1,16); t1=tR1;
    }
  }
  (void)t0; (void)t1;
}

static void dpf_eval_full_w8(uint8_t out_t[256],
                             const uint8_t root_seed[16], uint8_t root_t,
                             const uint8_t sigma[8][16], const uint8_t tau[8]){
  // BFS expand to 256 leaves
  uint8_t seed_lo[256][16];
  uint8_t tbits[256];

  // level 0 has 1 node
  memcpy(seed_lo[0], root_seed, 16);
  tbits[0] = root_t & 1;
  int nodes=1;

  for(uint32_t d=0; d<8; d++){
    int next_nodes = nodes*2;
    uint8_t next_seed[256][16];
    uint8_t next_t[256];
    for(int i=0;i<nodes;i++){
      uint8_t SL[16], SR[16], tm;
      G_expand_chacha(SL,SR,&tm,seed_lo[i],d);
      uint8_t tL=tm&1, tR=(tm>>1)&1;

      if(tbits[i]){
        for(int k=0;k<16;k++){ SL[k]^=sigma[d][k]; SR[k]^=sigma[d][k]; }
        tL ^= (tau[d] & 1);
        tR ^= ((tau[d]>>1)&1);
      }
      memcpy(next_seed[2*i+0], SL,16); next_t[2*i+0]=tL;
      memcpy(next_seed[2*i+1], SR,16); next_t[2*i+1]=tR;
    }
    // move
    for(int i=0;i<next_nodes;i++){
      memcpy(seed_lo[i], next_seed[i],16);
      tbits[i]=next_t[i];
    }
    nodes = next_nodes;
  }

  // leaves: nodes==256
  for(int i=0;i<256;i++) out_t[i]=tbits[i] & 1;
}

static void dcf_from_dpf_point_prefix_w8(uint64_t out_word[256],
                                         const uint8_t point_t_share[256],
                                         uint64_t payload_mask){
  // DCF(x)=1 XOR prefixXOR(point<=x), then apply payload_mask (0/1 compare uses mask=1)
  uint8_t prefix=0;
  for(int x=0;x<256;x++){
    prefix ^= (point_t_share[x] & 1);
    uint8_t dcf_bit = (uint8_t)(1 ^ prefix);
    out_word[x] = dcf_bit ? payload_mask : 0;
  }
}

int main(){
  // AES known vector (for sanity)
  uint8_t aes_key[16] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f
  };
  uint8_t aes_pt[16] = {
    0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
    0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff
  };
  uint8_t aes_ct[16];
  uint8_t rk[11][16];
  aes128_keyexp(rk,aes_key);
  aes128_enc_block(aes_ct,aes_pt,rk);
  hexdump("AES128(PT) CT", aes_ct, 16);
  // expected: 69c4e0d86a7b0430d8cdb78070b4c55a

  // ChaCha12 block sample
  uint8_t chkey[32]={0};
  uint8_t nonce12[12]={0};
  uint8_t chout[64];
  chacha12_block_v1(chout,chkey,nonce12,0);
  hexdump("ChaCha12 block (k=0,n=0,c=0)", chout, 64);

  // Deterministic-ish roots (for demo we hardcode; replace with your KDF32 binding sid/fss_id/alpha/beta)
  uint8_t root0[16]={0};
  uint8_t root1[16]={0};
  for(int i=0;i<16;i++){ root0[i]=(uint8_t)i; root1[i]=(uint8_t)(0xA0+i); }

  uint8_t sigma[8][16]; uint8_t tau[8];
  uint8_t alpha = 0x42; // demo threshold
  dpf_keygen_w8(root0,root1,sigma,tau,alpha);

  // Print key material
  hexdump("DPF root0", root0,16);
  hexdump("DPF root1", root1,16);
  for(int d=0;d<8;d++){
    char lab[64];
    snprintf(lab,sizeof(lab),"sigma[%d]",d); hexdump(lab,sigma[d],16);
    printf("tau[%d]=0x%02x (tauL=%d,tauR=%d)\n", d, tau[d], tau[d]&1, (tau[d]>>1)&1);
  }

  // Eval full-domain for party0 share (t_root=0)
  uint8_t tvec0[256];
  dpf_eval_full_w8(tvec0, root0, 0, sigma, tau);

  // Eval full-domain for party1 share (t_root=1)
  uint8_t tvec1[256];
  dpf_eval_full_w8(tvec1, root1, 1, sigma, tau);

  // Reconstruct point = t0 XOR t1
  uint8_t point[256];
  for(int i=0;i<256;i++) point[i]=(uint8_t)((tvec0[i]^tvec1[i])&1);

  // Print a small window around alpha
  printf("Point one-hot sanity around alpha=0x%02x:\n", alpha);
  for(int i=(int)alpha-4;i<=(int)alpha+4;i++){
    if(i<0||i>255) continue;
    printf("  x=%3d  point=%d\n", i, point[i]);
  }

  // Build DCF shares from each party share separately (they do it locally), then XOR-reconstruct
  uint64_t dcf0[256], dcf1[256], dcf[256];
  dcf_from_dpf_point_prefix_w8(dcf0, tvec0, 1);
  dcf_from_dpf_point_prefix_w8(dcf1, tvec1, 1);
  for(int i=0;i<256;i++) dcf[i]=dcf0[i]^dcf1[i];

  // Check DCF truth for a few values
  printf("DCF(x<alpha) sanity:\n");
  for(int i=0;i<8;i++){
    int x = i*32;
    printf("  x=%3d  dcf=%llu  (expected %d)\n", x, (unsigned long long)dcf[x], (x<(int)alpha)?1:0);
  }

  return 0;
}
