#include "uvcc/arena.h"

#include <stdexcept>

namespace uvcc {
namespace {

inline std::size_t align_up(std::size_t n, std::size_t a) {
    if (a == 0) return n;
    const std::size_t r = n % a;
    return (r == 0) ? n : (n + (a - r));
}

}  // namespace

Arena::Arena(std::size_t reserve_bytes) {
    if (reserve_bytes) default_block_size_ = reserve_bytes;
}

void Arena::reset() {
    block_idx_ = 0;
    off_in_block_ = 0;
    used_bytes_ = 0;
}

void* Arena::alloc(std::size_t n_bytes, std::size_t align_bytes) {
    if (align_bytes == 0) align_bytes = 1;
    if ((align_bytes & (align_bytes - 1)) != 0) throw std::runtime_error("Arena::alloc align must be power of two");
    if (n_bytes == 0) n_bytes = 1;

    // Ensure we have a block.
    if (blocks_.empty()) {
        Block b;
        b.size = default_block_size_;
        b.data.reset(new std::uint8_t[b.size]);
        blocks_.push_back(std::move(b));
    }

    // Move to an existing block if possible.
    if (block_idx_ >= blocks_.size()) {
        block_idx_ = blocks_.size() - 1;
        off_in_block_ = blocks_[block_idx_].size;
    }

    auto ensure_block = [&]() {
        // Allocate a new block.
        Block b;
        const std::size_t need = n_bytes + align_bytes;
        b.size = (need > default_block_size_) ? need : default_block_size_;
        b.data.reset(new std::uint8_t[b.size]);
        blocks_.push_back(std::move(b));
        block_idx_ = blocks_.size() - 1;
        off_in_block_ = 0;
    };

    // Find space in current block; otherwise create a new one.
    std::size_t start = align_up(off_in_block_, align_bytes);
    if (start + n_bytes > blocks_[block_idx_].size) {
        ensure_block();
        start = align_up(off_in_block_, align_bytes);
        if (start + n_bytes > blocks_[block_idx_].size) throw std::runtime_error("Arena block too small");
    }
    off_in_block_ = start + n_bytes;
    used_bytes_ += n_bytes;
    return blocks_[block_idx_].data.get() + start;
}

}  // namespace uvcc


