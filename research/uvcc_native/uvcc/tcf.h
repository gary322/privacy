#pragma once

#include "uvcc/frame.h"
#include "uvcc/beaver_types.h"
#include "uvcc/status.h"
#include "uvcc/transport.h"
#include "uvcc/types.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uvcc {

// Phase 7+: TCF-v0a (dealerless tile triple generation) CPU reference.
//
// This implements privacy_new.txt §10.4:
// - A,B additive shares are generated from pairwise seeds via a PRG
// - C_p is computed locally as:
//     C_p = A_p B_p + A_p B_{p+1} + A_{p+1} B_p
//   where indices are mod 3
// - C_p is replicated to the other party that stores component p (ring send to prev)
//   so that each party ends with RSS components (C_p, C_{p+1}) in its (lo,hi) buffers.

constexpr u8 PAYLOAD_KIND_TCF_C_REPL_V0A = 200;  // aligns with privacy_new.txt suggestion msg_kind=200

// Seeds for the two RSS components stored locally by a party.
// Party p holds components:
//   lo_comp = p
//   hi_comp = (p+1)%3
// so these seeds correspond to those components, respectively.
struct TcfSeedsV1 {
    Hash32 seed_lo{};
    Hash32 seed_hi{};
};

class TcfV0aEngineV1 {
   public:
    TcfV0aEngineV1(Sid32 sid_sub, u8 self_party, TcfSeedsV1 seeds, TransportV1* transport)
        : sid_sub_(sid_sub), self_(self_party), seeds_(seeds), transport_(transport) {}

    // Begin generating a Beaver triple for GEMM X(m,k) @ Y(k,n).
    void start(u32 op_id32, u32 epoch_id32, u32 m, u32 k, u32 n);

    // Called from TransportCallbacksV1.on_deliver to accept a replicated C component.
    void on_deliver(const FrameHdrV1& hdr, const std::vector<u8>& full_payload);

    bool is_done(u32 op_id32) const;
    BeaverTripleU64MatV1 take_triple(u32 op_id32);

   private:
    struct Task {
        u32 op_id32 = 0;
        u32 epoch_id32 = 0;
        u32 m = 0;
        u32 k = 0;
        u32 n = 0;

        BeaverTripleU64MatV1 triple;
        u32 recv_c_msg_id32 = 0;  // expected msg_id32 for incoming replicated C_{p+1}
        bool sent_c = false;
        bool have_c_hi = false;
        bool done = false;
    };

    static u64 sid_hash64_(const Sid32& sid32);
    static u64 logical_msg_id64_(const Sid32& sid32, u64 stream_id64, u8 payload_kind, u32 op_id32);
    static u64 stream_id64_(const Sid32& sid32, u32 op_id32);
    static u64 prg_u64_(const Hash32& seed, const Sid32& sid32, u32 op_id32, u8 role, u32 word_idx);
    static std::vector<u64> prg_u64_vec_(const Hash32& seed, const Sid32& sid32, u32 op_id32, u8 role, u32 n_words);
    static std::vector<u8> encode_u64_vec_le_(const std::vector<u64>& v);
    static std::vector<u64> decode_u64_vec_le_(const std::vector<u8>& b);
    static void matmul_u64_(const u64* A, const u64* B, u64* C, u32 m, u32 k, u32 n);

    void maybe_apply_pending_(u32 op_id32);

    Sid32 sid_sub_;
    u8 self_;
    TcfSeedsV1 seeds_;
    TransportV1* transport_;

    std::unordered_map<u32, Task> tasks_;
    // (op_id32 -> (hdr,payload)) if C replication arrives before start().
    std::unordered_map<u32, std::pair<FrameHdrV1, std::vector<u8>>> pending_by_op_;
};

}  // namespace uvcc


