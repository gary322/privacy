#pragma once

#include "uvcc/frame.h"
#include "uvcc/ids.h"
#include "uvcc/types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace uvcc {

// OPEN_ARITH send (payload_kind) from privacy_new.txt §1.1.
constexpr u8 PAYLOAD_KIND_OPEN_ARITH_SEND = 0x11;

// Minimal OpenEngine for OPEN_ARITH over u64 arrays (Phase 4).
// Protocol:
// - each party sends its lo component to next party as one DATA frame
// - each party receives missing component from prev party and reconstructs pub = lo+hi+missing (mod 2^64)
class OpenEngineV1 {
   public:
    OpenEngineV1(Sid32 sid_sub, u8 self_party) : sid_sub_(sid_sub), self_(self_party) {}

    // Enqueue an OPEN of one u64-array secret share.
    //
    // - `lo_local` and `hi_local` are this party's replicated shares (same length).
    // - Result is retrievable with `is_done()` / `take_result_u64()`.
    void enqueue_open_u64(
        u32 op_id32,
        u32 epoch_id32,
        u64 stream_id64,
        const std::vector<u64>& lo_local,
        const std::vector<u64>& hi_local,
        FrameV1* out_frame_to_next);

    // Handle a delivered DATA payload for OPEN_ARITH_SEND from Transport.
    void on_deliver(const FrameHdrV1& hdr, const std::vector<u8>& full_payload);

    bool is_done(u32 op_id32) const;
    std::vector<u64> take_result_u64(u32 op_id32);

   private:
    struct Task {
        u32 op_id32 = 0;
        u32 epoch_id32 = 0;
        u64 stream_id64 = 0;
        u32 recv_msg_id32 = 0;
        std::vector<u64> lo_local;
        std::vector<u64> hi_local;
        std::vector<u64> pub_out;
        bool done = false;
    };

    void apply_open_recv_(Task& t, const FrameHdrV1& hdr, const std::vector<u8>& full_payload);

    Sid32 sid_sub_;
    u8 self_;
    std::unordered_map<u32, Task> tasks_by_op_;
    // If an OPEN payload arrives before the local task is registered (enqueue_open_u64),
    // we stash it here keyed by op_id32 and apply it when the task is created.
    std::unordered_map<u32, std::pair<FrameHdrV1, std::vector<u8>>> pending_by_op_;
};

}  // namespace uvcc


