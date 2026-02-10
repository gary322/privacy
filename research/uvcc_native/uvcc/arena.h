#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace uvcc {

enum class ArenaLife : std::uint8_t {
    LIFE_STEP = 1,
    LIFE_EPOCH = 2,
    LIFE_PERSIST = 3,
};

// Simple bump allocator arena (Phase 4). Deterministic given deterministic allocation order.
class Arena {
   public:
    explicit Arena(std::size_t reserve_bytes = 0);

    void reset();
    void* alloc(std::size_t n_bytes, std::size_t align_bytes = 16);
    std::size_t used() const { return used_bytes_; }

   private:
    struct Block {
        std::unique_ptr<std::uint8_t[]> data;
        std::size_t size = 0;
    };
    std::vector<Block> blocks_;
    std::size_t block_idx_ = 0;
    std::size_t off_in_block_ = 0;
    std::size_t used_bytes_ = 0;
    std::size_t default_block_size_ = 1 << 20;  // 1 MiB
};

}  // namespace uvcc


