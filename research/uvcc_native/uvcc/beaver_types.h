#pragma once

#include "uvcc/types.h"

#include <cstdint>
#include <vector>

namespace uvcc {

// Shared types used by Beaver GEMM and preprocessing (TCF/W-VOLE).

struct RSSU64MatV1 {
    u32 rows = 0;
    u32 cols = 0;
    std::vector<u64> lo;  // length rows*cols
    std::vector<u64> hi;  // length rows*cols
};

struct BeaverTripleU64MatV1 {
    u32 m = 0;
    u32 k = 0;
    u32 n = 0;
    RSSU64MatV1 A;  // (m,k)
    RSSU64MatV1 B;  // (k,n)
    RSSU64MatV1 C;  // (m,n) with C = A@B (mod 2^64) in RSS form
};

}  // namespace uvcc


