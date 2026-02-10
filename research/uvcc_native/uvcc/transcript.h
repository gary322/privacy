#pragma once

#include "uvcc/types.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace uvcc {

// Canonical leaf (privacy_new.txt §6.3) for lift/open transport events.
struct LeafV1 {
    u16 leaf_type = 0;
    u32 epoch_id32 = 0;
    u64 stream_id64 = 0;
    u32 msg_id32 = 0;
    u32 op_id32 = 0;
    u8 src_party = 0;
    u8 dst_party = 0;
    u8 msg_class = 0;
    u8 payload_kind = 0;
    u32 chunk_idx = 0;
    u32 chunk_count = 1;
    u32 payload_bytes = 0;
    u64 sid_hash64 = 0;
    Hash32 frame_hash32{};    // for DATA leaves, else zeros
    Hash32 control_hash32{};  // for ACK/NACK leaves, else zeros
};

std::array<u8, 128> leaf_bytes_v1(const LeafV1& l);

// leaf_key := H256("uvcc.leafkey.v1" || sid || LE32(epoch_id32) || LE64(stream_id64) || LE32(msg_id32) || LE16(leaf_type))
Hash32 leaf_key_v1(const Sid32& sid32, const LeafV1& l);

// leaf_digest := H256("uvcc.leafhash.v1" || leaf_bytes)
Hash32 leaf_digest_v1(const std::array<u8, 128>& leaf_bytes);

// Epoch root from keyed leaves:
// 1) sort by leaf_key ascending (byte lexicographic)
// 2) merkle hash: H256("uvcc.merkle.v1" || left || right)
// 3) if empty: H256("UVCC.emptyepoch.v1\\0")
Hash32 epoch_root_v1(const std::vector<std::pair<Hash32, Hash32>>& leaf_key_and_digest);

// In-memory transcript store for one sid_sub. Stores (leaf_key -> leaf_digest) with exactly-once semantics.
class TranscriptStoreV1 {
   public:
    explicit TranscriptStoreV1(Sid32 sid32) : sid32_(sid32) {}

    void record_leaf(const LeafV1& l);
    Hash32 epoch_root(u32 epoch_id32) const;

   private:
    struct Hash32Hash {
        std::size_t operator()(const Hash32& h) const noexcept {
            // Use first 8 bytes as a cheap hash seed.
            std::size_t x = 0;
            for (int i = 0; i < 8; i++) x = (x << 8) ^ static_cast<std::size_t>(h.v[static_cast<std::size_t>(i)]);
            return x;
        }
    };

    Sid32 sid32_;
    // Map by epoch -> (leaf_key -> leaf_digest)
    std::unordered_map<u32, std::unordered_map<Hash32, Hash32, Hash32Hash>> epochs_;
};

}  // namespace uvcc


