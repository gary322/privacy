#pragma once

#include "uvcc/beaver_types.h"
#include "uvcc/open.h"
#include "uvcc/transport.h"
#include "uvcc/tcf.h"
#include "uvcc/types.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace uvcc {

// Minimal Beaver GEMM (u64 ring) for square matrices dxd.
// This is a Phase 4 CPU implementation used for correctness/determinism and early end-to-end tests.

struct BeaverGemmTaskV1 {
    u32 op_id32 = 0;
    u32 epoch_id32 = 0;
    u32 m = 0;
    u32 k = 0;
    u32 n = 0;
    RSSU64MatV1 X;  // (m,k)
    RSSU64MatV1 Y;  // (k,n)
    BeaverTripleU64MatV1 triple;
    bool done = false;
    RSSU64MatV1 Z;  // (m,n)
};

class BeaverGemmEngineV1 {
   public:
    BeaverGemmEngineV1(
        Sid32 sid_sub,
        u8 self_party,
        TransportV1* transport,
        OpenEngineV1* open,
        // Optional preprocessing engine. If provided, BeaverGemmEngine can generate
        // its own (A,B,C) triple via TCF-v0a when the task omits `task.triple`.
        TcfV0aEngineV1* tcf_v0a = nullptr)
        : sid_sub_(sid_sub), self_(self_party), transport_(transport), open_(open), tcf_v0a_(tcf_v0a) {}

    void start(BeaverGemmTaskV1 task);
    void tick();
    bool is_done(u32 op_id32) const;
    RSSU64MatV1 take_result(u32 op_id32);

   private:
    struct State {
        BeaverGemmTaskV1 task;
        bool use_tcf_v0a = false;
        u32 tcf_op_id32 = 0;
        u32 open_e_op_id32 = 0;
        u32 open_f_op_id32 = 0;
        u64 open_e_stream_id64 = 0;
        u64 open_f_stream_id64 = 0;
        RSSU64MatV1 E;
        RSSU64MatV1 F;
        std::vector<u64> E_pub;
        std::vector<u64> F_pub;
        int phase = 0;  // 0 init, 1 waitE, 2 waitF, 3 done
    };

    static u32 derive_subop_id32_(const Sid32& sid_sub, u32 parent_op_id32, const char* tag);
    static u64 derive_stream_id64_(const Sid32& sid_sub, u32 op_id32);
    static void matmul_u64_(const u64* A, const u64* B, u64* C, u32 m, u32 k, u32 n);

    Sid32 sid_sub_;
    u8 self_;
    TransportV1* transport_;
    OpenEngineV1* open_;
    TcfV0aEngineV1* tcf_v0a_;
    std::unordered_map<u32, State> st_;
};

}  // namespace uvcc


