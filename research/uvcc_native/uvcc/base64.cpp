#include "uvcc/base64.h"

#include <cctype>
#include <stdexcept>

namespace uvcc {

static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const std::vector<u8>& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= data.size()) {
        const u32 b0 = static_cast<u32>(data[i + 0]);
        const u32 b1 = static_cast<u32>(data[i + 1]);
        const u32 b2 = static_cast<u32>(data[i + 2]);
        const u32 v = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(B64[(v >> 18) & 0x3F]);
        out.push_back(B64[(v >> 12) & 0x3F]);
        out.push_back(B64[(v >> 6) & 0x3F]);
        out.push_back(B64[(v >> 0) & 0x3F]);
        i += 3;
    }
    const std::size_t rem = data.size() - i;
    if (rem == 1) {
        const u32 b0 = static_cast<u32>(data[i + 0]);
        const u32 v = (b0 << 16);
        out.push_back(B64[(v >> 18) & 0x3F]);
        out.push_back(B64[(v >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        const u32 b0 = static_cast<u32>(data[i + 0]);
        const u32 b1 = static_cast<u32>(data[i + 1]);
        const u32 v = (b0 << 16) | (b1 << 8);
        out.push_back(B64[(v >> 18) & 0x3F]);
        out.push_back(B64[(v >> 12) & 0x3F]);
        out.push_back(B64[(v >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

static int b64_index(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    if (c == '=') return -2;
    return -1;
}

std::vector<u8> base64_decode(const std::string& s) {
    // Strip whitespace first.
    std::string t;
    t.reserve(s.size());
    for (char ch : s) {
        if (std::isspace(static_cast<unsigned char>(ch))) continue;
        t.push_back(ch);
    }
    if (t.empty()) return {};
    if (t.size() % 4 != 0) throw std::runtime_error("base64_decode: length not multiple of 4");

    std::vector<u8> out;
    out.reserve((t.size() / 4) * 3);

    for (std::size_t i = 0; i < t.size(); i += 4) {
        const char c0 = t[i + 0];
        const char c1 = t[i + 1];
        const char c2 = t[i + 2];
        const char c3 = t[i + 3];

        const int v0 = b64_index(c0);
        const int v1 = b64_index(c1);
        const int v2 = b64_index(c2);
        const int v3 = b64_index(c3);
        if (v0 < 0 || v1 < 0) throw std::runtime_error("base64_decode: invalid leading chars");

        const bool pad2 = (c2 == '=');
        const bool pad3 = (c3 == '=');
        if (pad2 && !pad3) throw std::runtime_error("base64_decode: invalid padding");

        const u32 x0 = static_cast<u32>(v0);
        const u32 x1 = static_cast<u32>(v1);
        const u32 x2 = pad2 ? 0u : static_cast<u32>(v2);
        const u32 x3 = pad3 ? 0u : static_cast<u32>(v3);
        if (!pad2 && v2 < 0) throw std::runtime_error("base64_decode: invalid c2");
        if (!pad3 && v3 < 0) throw std::runtime_error("base64_decode: invalid c3");

        const u32 triple = (x0 << 18) | (x1 << 12) | (x2 << 6) | (x3);
        out.push_back(static_cast<u8>((triple >> 16) & 0xFF));
        if (!pad2) out.push_back(static_cast<u8>((triple >> 8) & 0xFF));
        if (!pad3) out.push_back(static_cast<u8>((triple >> 0) & 0xFF));
    }
    return out;
}

}  // namespace uvcc


