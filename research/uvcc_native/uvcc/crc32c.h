#pragma once

#include <cstddef>
#include <cstdint>

namespace uvcc {

// CRC32C (Castagnoli) as required by privacy_new.txt (deterministic across impls).
std::uint32_t crc32c(const void* data, std::size_t len);

}  // namespace uvcc


