#pragma once

#include "uvcc/types.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace uvcc {

// Phase 2 reassembly helper (privacy_new.txt §7.3).
// Stores accepted chunks by logical_msg_id64 and concatenates once complete.
class ReassemblyV1 {
   public:
    // Returns true if the message became complete after inserting this chunk.
    bool put_chunk(u64 logical_msg_id64, u32 chunk_idx, u32 chunk_count, const std::vector<u8>& payload);

    bool is_complete(u64 logical_msg_id64) const;
    std::vector<u8> take_message(u64 logical_msg_id64);

   private:
    struct Msg {
        u32 chunk_count = 0;
        std::unordered_map<u32, std::vector<u8>> chunks;
    };
    std::unordered_map<u64, Msg> msgs_;
};

}  // namespace uvcc


