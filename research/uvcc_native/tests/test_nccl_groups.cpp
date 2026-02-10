#include "uvcc/nccl.h"

#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

static bool eq_u32(const std::vector<uvcc::u32>& a, const std::vector<uvcc::u32>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

int main() {
    // Match PARALLEL.txt example: R=8, S=4, T=2 => 64 local ranks per party.
    uvcc::TopologyV1 topo;
    topo.replicas = 8;
    topo.stages = 4;
    topo.tp_ranks = 2;

    // TP(P0,0,0): ranks {0,1}
    const auto tp00 = uvcc::make_tp_group_v1(topo, /*replica=*/0, /*stage=*/0);
    if (!eq_u32(tp00.ranks, std::vector<uvcc::u32>{0, 1})) return fail("TP(0,0) ranks mismatch");

    // PP(P0, r=0, t=0): {0,2,4,6}
    const auto pp0t0 = uvcc::make_pp_group_v1(topo, /*replica=*/0, /*tp_rank=*/0);
    if (!eq_u32(pp0t0.ranks, std::vector<uvcc::u32>{0, 2, 4, 6})) return fail("PP(r=0,t=0) ranks mismatch");

    // DP(P0, s=0, t=0): {0,8,16,24,32,40,48,56}
    const auto dp_s0t0 = uvcc::make_dp_group_v1(topo, /*stage=*/0, /*tp_rank=*/0);
    if (!eq_u32(dp_s0t0.ranks, std::vector<uvcc::u32>{0, 8, 16, 24, 32, 40, 48, 56})) return fail("DP(s=0,t=0) ranks mismatch");

    // Spot-check: local_rank formula r*(S*T)+s*T+t
    const auto lr = uvcc::local_rank_v1(topo, /*r=*/5, /*s=*/1, /*t=*/0);
    if (lr != 42) return fail("local_rank_v1 mismatch for (r=5,s=1,t=0): expected 42");

    return 0;
}


