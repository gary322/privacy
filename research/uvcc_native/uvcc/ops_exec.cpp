#include "uvcc/ops_exec.h"

#include <stdexcept>

namespace uvcc {

void rss_u64_add_inplace(RSSU64SlotV1& dst, const RSSU64SlotV1& src) {
    if (dst.n_words != src.n_words) throw std::runtime_error("rss_u64_add_inplace size mismatch");
    for (std::size_t i = 0; i < static_cast<std::size_t>(dst.n_words); i++) {
        dst.lo[i] = static_cast<u64>(dst.lo[i] + src.lo[i]);
        dst.hi[i] = static_cast<u64>(dst.hi[i] + src.hi[i]);
    }
}

void rss_u64_add(RSSU64SlotV1& out, const RSSU64SlotV1& a, const RSSU64SlotV1& b) {
    if (a.n_words != b.n_words) throw std::runtime_error("rss_u64_add size mismatch");
    if (out.n_words != a.n_words) throw std::runtime_error("rss_u64_add out size mismatch");
    for (std::size_t i = 0; i < static_cast<std::size_t>(a.n_words); i++) {
        out.lo[i] = static_cast<u64>(a.lo[i] + b.lo[i]);
        out.hi[i] = static_cast<u64>(a.hi[i] + b.hi[i]);
    }
}

}  // namespace uvcc


