#include "uvcc/transcript.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace uvcc {
namespace {

inline Hash32 sha256_bytes(const std::vector<u8>& v) { return sha256(v.data(), v.size()); }

inline void write_ascii(ByteWriter& w, const char* s) { w.write_bytes(s, std::strlen(s)); }

}  // namespace

std::array<u8, 128> leaf_bytes_v1(const LeafV1& l) {
    ByteWriter w;
    w.write_u16_le(l.leaf_type);
    w.write_u16_le(1);  // version
    w.write_u32_le(l.epoch_id32);
    w.write_u64_le(l.stream_id64);
    w.write_u32_le(l.msg_id32);
    w.write_u32_le(l.op_id32);
    w.write_u8(l.src_party);
    w.write_u8(l.dst_party);
    w.write_u8(l.msg_class);
    w.write_u8(l.payload_kind);
    w.write_u32_le(l.chunk_idx);
    w.write_u32_le(l.chunk_count);
    w.write_u32_le(l.payload_bytes);
    w.write_u64_le(l.sid_hash64);
    w.write_bytes(l.frame_hash32);
    w.write_bytes(l.control_hash32);
    // reserved (16 bytes)
    for (int i = 0; i < 16; i++) w.write_u8(0);
    if (w.size() != 128) throw std::runtime_error("leaf_bytes_v1 size mismatch");
    std::array<u8, 128> out{};
    std::memcpy(out.data(), w.bytes().data(), 128);
    return out;
}

Hash32 leaf_key_v1(const Sid32& sid32, const LeafV1& l) {
    ByteWriter w;
    write_ascii(w, "uvcc.leafkey.v1");
    w.write_bytes(sid32);
    w.write_u32_le(l.epoch_id32);
    w.write_u64_le(l.stream_id64);
    w.write_u32_le(l.msg_id32);
    w.write_u16_le(l.leaf_type);
    return sha256(w.bytes().data(), w.bytes().size());
}

Hash32 leaf_digest_v1(const std::array<u8, 128>& leaf_bytes) {
    ByteWriter w;
    write_ascii(w, "uvcc.leafhash.v1");
    w.write_bytes(leaf_bytes.data(), leaf_bytes.size());
    return sha256(w.bytes().data(), w.bytes().size());
}

Hash32 epoch_root_v1(const std::vector<std::pair<Hash32, Hash32>>& leaf_key_and_digest) {
    if (leaf_key_and_digest.empty()) {
        const char* empty = "UVCC.emptyepoch.v1\0";
        return sha256(empty, std::strlen(empty) + 1);
    }
    std::vector<std::pair<Hash32, Hash32>> v = leaf_key_and_digest;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first.v < b.first.v; });
    std::vector<Hash32> level;
    level.reserve(v.size());
    for (const auto& kv : v) level.push_back(kv.second);

    while (level.size() > 1) {
        std::vector<Hash32> nxt;
        nxt.reserve((level.size() + 1) / 2);
        for (std::size_t i = 0; i < level.size(); i += 2) {
            const Hash32& left = level[i];
            const Hash32& right = (i + 1 < level.size()) ? level[i + 1] : left;
            ByteWriter w;
            write_ascii(w, "uvcc.merkle.v1");
            w.write_bytes(left);
            w.write_bytes(right);
            nxt.push_back(sha256(w.bytes().data(), w.bytes().size()));
        }
        level = std::move(nxt);
    }
    return level[0];
}

void TranscriptStoreV1::record_leaf(const LeafV1& l) {
    auto leaf_b = leaf_bytes_v1(l);
    const Hash32 k = leaf_key_v1(sid32_, l);
    const Hash32 d = leaf_digest_v1(leaf_b);
    auto& ep = epochs_[l.epoch_id32];
    auto it = ep.find(k);
    if (it != ep.end()) {
        if (it->second != d) throw std::runtime_error("duplicate leaf_key with different digest");
        return;
    }
    ep.emplace(k, d);
}

Hash32 TranscriptStoreV1::epoch_root(u32 epoch_id32) const {
    auto it = epochs_.find(epoch_id32);
    if (it == epochs_.end()) return epoch_root_v1({});
    std::vector<std::pair<Hash32, Hash32>> v;
    v.reserve(it->second.size());
    for (const auto& kv : it->second) v.push_back(kv);
    return epoch_root_v1(v);
}

}  // namespace uvcc


