#pragma once

#include <cstdint>

namespace uvcc {

struct ClockV1 {
    virtual ~ClockV1() = default;
    virtual std::uint64_t now_ms() = 0;
};

struct RealClockV1 final : public ClockV1 {
    std::uint64_t now_ms() override;
};

}  // namespace uvcc


