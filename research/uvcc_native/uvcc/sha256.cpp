#include "uvcc/sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace uvcc {
namespace {

inline u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

inline u32 ch(u32 x, u32 y, u32 z) { return (x & y) ^ (~x & z); }
inline u32 maj(u32 x, u32 y, u32 z) { return (x & y) ^ (x & z) ^ (y & z); }
inline u32 big_sigma0(u32 x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline u32 big_sigma1(u32 x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline u32 small_sigma0(u32 x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline u32 small_sigma1(u32 x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

constexpr std::array<u32, 64> K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline u32 load_be_u32(const u8* p) {
    return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) | (static_cast<u32>(p[2]) << 8) | (static_cast<u32>(p[3]) << 0);
}

inline void store_be_u32(u8* p, u32 x) {
    p[0] = static_cast<u8>((x >> 24) & 0xFF);
    p[1] = static_cast<u8>((x >> 16) & 0xFF);
    p[2] = static_cast<u8>((x >> 8) & 0xFF);
    p[3] = static_cast<u8>((x >> 0) & 0xFF);
}

}  // namespace

Hash32 sha256(const void* data, std::size_t len) {
    const auto* in = static_cast<const u8*>(data);

    // Preprocess: pad to 512-bit blocks with 64-bit big-endian bit length.
    std::vector<u8> msg;
    msg.reserve(len + 1 + 64);
    msg.insert(msg.end(), in, in + len);
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0x00);
    const std::uint64_t bit_len = static_cast<std::uint64_t>(len) * 8ULL;
    for (int i = 7; i >= 0; i--) msg.push_back(static_cast<u8>((bit_len >> (8 * i)) & 0xFF));

    u32 h0 = 0x6a09e667u;
    u32 h1 = 0xbb67ae85u;
    u32 h2 = 0x3c6ef372u;
    u32 h3 = 0xa54ff53au;
    u32 h4 = 0x510e527fu;
    u32 h5 = 0x9b05688cu;
    u32 h6 = 0x1f83d9abu;
    u32 h7 = 0x5be0cd19u;

    std::array<u32, 64> w{};
    for (std::size_t off = 0; off < msg.size(); off += 64) {
        const u8* chunk = msg.data() + off;
        for (int i = 0; i < 16; i++) w[i] = load_be_u32(chunk + (4 * i));
        for (int i = 16; i < 64; i++) w[i] = small_sigma1(w[i - 2]) + w[i - 7] + small_sigma0(w[i - 15]) + w[i - 16];

        u32 a = h0;
        u32 b = h1;
        u32 c = h2;
        u32 d = h3;
        u32 e = h4;
        u32 f = h5;
        u32 g = h6;
        u32 h = h7;

        for (int i = 0; i < 64; i++) {
            const u32 t1 = h + big_sigma1(e) + ch(e, f, g) + K[i] + w[i];
            const u32 t2 = big_sigma0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }

    Hash32 out;
    store_be_u32(out.v.data() + 0, h0);
    store_be_u32(out.v.data() + 4, h1);
    store_be_u32(out.v.data() + 8, h2);
    store_be_u32(out.v.data() + 12, h3);
    store_be_u32(out.v.data() + 16, h4);
    store_be_u32(out.v.data() + 20, h5);
    store_be_u32(out.v.data() + 24, h6);
    store_be_u32(out.v.data() + 28, h7);
    return out;
}

}  // namespace uvcc


