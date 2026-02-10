#include "uvcc/crc32c.h"

#include <array>

namespace uvcc {
namespace {

// Reflected CRC32C polynomial (Castagnoli) 0x82F63B78.
constexpr std::uint32_t POLY = 0x82F63B78u;

constexpr std::array<std::uint32_t, 256> make_table() {
    std::array<std::uint32_t, 256> tbl{};
    for (std::uint32_t i = 0; i < 256; i++) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1u) ? (POLY ^ (c >> 1)) : (c >> 1);
        }
        tbl[i] = c;
    }
    return tbl;
}

constexpr auto TABLE = make_table();

}  // namespace

std::uint32_t crc32c(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; i++) {
        crc = TABLE[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

}  // namespace uvcc


