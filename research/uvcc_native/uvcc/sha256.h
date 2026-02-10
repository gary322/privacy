#pragma once

#include "uvcc/types.h"

#include <cstddef>

namespace uvcc {

Hash32 sha256(const void* data, std::size_t len);

}  // namespace uvcc


