#pragma once

#include "uvcc/types.h"

namespace uvcc {

inline Sid32 sid32_seq_00_1f() {
    Sid32 s{};
    for (int i = 0; i < 32; i++) s.v[static_cast<std::size_t>(i)] = static_cast<u8>(i);
    return s;
}

}  // namespace uvcc


