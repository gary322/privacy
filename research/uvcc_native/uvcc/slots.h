#pragma once

#include "uvcc/arena.h"
#include "uvcc/types.h"

#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace uvcc {

// Minimal RSS slot storage for u64 tensors (Phase 4).
// Each party stores two components locally; we treat them as (lo, hi) buffers.
struct RSSU64SlotV1 {
    u32 slot_id32 = 0;  // deterministic id (e.g., fss_id truncated or op-local value id)
    u32 n_words = 0;
    u64* lo = nullptr;
    u64* hi = nullptr;
};

class SlotMapV1 {
   public:
    explicit SlotMapV1(Arena* arena) : arena_(arena) {
        if (arena_ == nullptr) throw std::runtime_error("SlotMapV1 requires arena");
    }

    RSSU64SlotV1& get_or_create_rss_u64(u32 slot_id32, u32 n_words);

   private:
    Arena* arena_;
    struct U32Hash {
        std::size_t operator()(u32 x) const noexcept { return static_cast<std::size_t>(x); }
    };
    std::unordered_map<u32, RSSU64SlotV1, U32Hash> rss_u64_;
};

}  // namespace uvcc


