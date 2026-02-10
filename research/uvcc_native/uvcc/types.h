#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace uvcc {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

template <std::size_t N>
struct BytesN {
    std::array<u8, N> v{};

    bool operator==(const BytesN<N>& o) const { return v == o.v; }
    bool operator!=(const BytesN<N>& o) const { return !(*this == o); }
};

using Sid32 = BytesN<32>;
using Hash32 = BytesN<32>;

}  // namespace uvcc


