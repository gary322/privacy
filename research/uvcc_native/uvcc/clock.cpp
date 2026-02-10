#include "uvcc/clock.h"

#include <chrono>

namespace uvcc {

std::uint64_t RealClockV1::now_ms() {
    using namespace std::chrono;
    const auto tp = steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(duration_cast<milliseconds>(tp).count());
}

}  // namespace uvcc


