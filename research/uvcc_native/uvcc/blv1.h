#pragma once

#include "uvcc/types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uvcc {

struct TLV {
    u8 type = 0;
    u8 flags = 0;
    std::vector<u8> value;
};

struct BLV1Batch {
    u8 version = 1;
    u8 flags = 0;
    std::vector<TLV> tlvs;
};

std::vector<u8> blv1_encode(const BLV1Batch& b);
BLV1Batch blv1_decode(const std::vector<u8>& payload);

}  // namespace uvcc


