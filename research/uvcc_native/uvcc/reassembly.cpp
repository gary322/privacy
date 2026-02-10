#include "uvcc/reassembly.h"

namespace uvcc {

bool ReassemblyV1::put_chunk(u64 logical_msg_id64, u32 chunk_idx, u32 chunk_count, const std::vector<u8>& payload) {
    if (chunk_count == 0) throw std::runtime_error("chunk_count must be >= 1");
    if (chunk_idx >= chunk_count) throw std::runtime_error("chunk_idx out of range");
    auto& m = msgs_[logical_msg_id64];
    if (m.chunk_count == 0) m.chunk_count = chunk_count;
    if (m.chunk_count != chunk_count) throw std::runtime_error("chunk_count mismatch for logical_msg_id64");
    // idempotent insert: keep first
    if (m.chunks.find(chunk_idx) == m.chunks.end()) {
        m.chunks.emplace(chunk_idx, payload);
    }
    return m.chunks.size() == static_cast<std::size_t>(m.chunk_count);
}

bool ReassemblyV1::is_complete(u64 logical_msg_id64) const {
    auto it = msgs_.find(logical_msg_id64);
    if (it == msgs_.end()) return false;
    if (it->second.chunk_count == 0) return false;
    return it->second.chunks.size() == static_cast<std::size_t>(it->second.chunk_count);
}

std::vector<u8> ReassemblyV1::take_message(u64 logical_msg_id64) {
    auto it = msgs_.find(logical_msg_id64);
    if (it == msgs_.end()) throw std::runtime_error("message not found");
    if (it->second.chunk_count == 0) throw std::runtime_error("bad message state");
    if (it->second.chunks.size() != static_cast<std::size_t>(it->second.chunk_count)) throw std::runtime_error("message not complete");

    std::vector<u8> out;
    for (u32 i = 0; i < it->second.chunk_count; i++) {
        auto jt = it->second.chunks.find(i);
        if (jt == it->second.chunks.end()) throw std::runtime_error("missing chunk during take_message");
        out.insert(out.end(), jt->second.begin(), jt->second.end());
    }
    msgs_.erase(it);
    return out;
}

}  // namespace uvcc


