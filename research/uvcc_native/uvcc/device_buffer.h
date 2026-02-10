#pragma once

#include <cstddef>
#include <cstdint>

namespace uvcc {

// Minimal device/host buffer view used by intra-party collective wrappers.
// For now this is just a (ptr, bytes) span; Phase 6+ will bind to CUDA device pointers.
struct DeviceBufferV1 {
    void* ptr = nullptr;
    std::size_t n_bytes = 0;
};

}  // namespace uvcc


