#pragma once

#include "uvcc/types.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uvcc {

inline std::string hex_lower(const u8* data, std::size_t len) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (std::size_t i = 0; i < len; i++) {
        const u8 b = data[i];
        out.push_back(kHex[(b >> 4) & 0x0F]);
        out.push_back(kHex[b & 0x0F]);
    }
    return out;
}

template <std::size_t N>
inline std::string hex_lower(const BytesN<N>& x) {
    return hex_lower(x.v.data(), x.v.size());
}

inline u8 _hex_nibble(char c) {
    if (c >= '0' && c <= '9') return static_cast<u8>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<u8>(10 + (c - 'a'));
    if (c >= 'A' && c <= 'F') return static_cast<u8>(10 + (c - 'A'));
    throw std::runtime_error("invalid hex nibble");
}

inline std::vector<u8> parse_hex_bytes(std::string s) {
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    if (s.size() % 2 != 0) throw std::runtime_error("hex must have even length");
    std::vector<u8> out;
    out.resize(s.size() / 2);
    for (std::size_t i = 0; i < out.size(); i++) {
        const u8 hi = _hex_nibble(s[2 * i + 0]);
        const u8 lo = _hex_nibble(s[2 * i + 1]);
        out[i] = static_cast<u8>((hi << 4) | lo);
    }
    return out;
}

inline Sid32 parse_hex_sid32(std::string s) {
    const auto b = parse_hex_bytes(std::move(s));
    if (b.size() != 32) throw std::runtime_error("expected 32 bytes");
    Sid32 out;
    std::copy(b.begin(), b.end(), out.v.begin());
    return out;
}

}  // namespace uvcc


