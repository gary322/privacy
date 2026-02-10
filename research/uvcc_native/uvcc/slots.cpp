#include "uvcc/slots.h"

#include <cstring>

namespace uvcc {

RSSU64SlotV1& SlotMapV1::get_or_create_rss_u64(u32 slot_id32, u32 n_words) {
    auto it = rss_u64_.find(slot_id32);
    if (it != rss_u64_.end()) {
        if (it->second.n_words != n_words) throw std::runtime_error("slot size mismatch");
        return it->second;
    }
    RSSU64SlotV1 s;
    s.slot_id32 = slot_id32;
    s.n_words = n_words;
    s.lo = static_cast<u64*>(arena_->alloc(static_cast<std::size_t>(n_words) * sizeof(u64), 16));
    s.hi = static_cast<u64*>(arena_->alloc(static_cast<std::size_t>(n_words) * sizeof(u64), 16));
    std::memset(s.lo, 0, static_cast<std::size_t>(n_words) * sizeof(u64));
    std::memset(s.hi, 0, static_cast<std::size_t>(n_words) * sizeof(u64));
    auto [jt, ok] = rss_u64_.emplace(slot_id32, s);
    (void)ok;
    return jt->second;
}

}  // namespace uvcc


