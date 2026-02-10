#pragma once

#include "uvcc/status.h"
#include "uvcc/types.h"

#include <cstdint>
#include <vector>

namespace uvcc {

// Phase 7+ (research stub): Warp-VOLE expander interface.
//
// This is a placeholder API that lets us swap a future GPU-native correlation expander
// (Warp-VOLE) underneath preprocessing without perturbing transcript determinism.
//
// For now, the CPU implementation is a deterministic PRG keyed by per-party seeds.

// Domain-separated roles for derived PRG streams.
enum class WVoleRoleV1 : u8 {
    U = 1,
    V = 2,
    DELTA = 3,
};

struct WVoleSeedsV1 {
    // Two local seeds (mirroring RSS's two local components). These are NOT a real VOLE key schedule;
    // they exist so we can bind outputs deterministically to sid_sub + op_id + block_id.
    Hash32 seed_lo{};
    Hash32 seed_hi{};
};

struct WVoleBlockV1 {
    u32 block_id32 = 0;
    u32 n_words = 0;
    // Packed "correlation block" outputs.
    // In the stub we just output pseudorandom u64 vectors.
    std::vector<u64> u_lo;
    std::vector<u64> u_hi;
    std::vector<u64> v_lo;
    std::vector<u64> v_hi;
};

class WarpVoleStubV1 {
   public:
    WarpVoleStubV1(Sid32 sid_sub, u8 self_party, WVoleSeedsV1 seeds) : sid_sub_(sid_sub), self_(self_party), seeds_(seeds) {}

    // Deterministically expand one block.
    WVoleBlockV1 expand_block(u32 op_id32, u32 block_id32, u32 n_words) const;

   private:
    static u64 prg_u64_(const Hash32& seed, const Sid32& sid32, u32 op_id32, u32 block_id32, WVoleRoleV1 role, u32 word_idx);

    Sid32 sid_sub_;
    u8 self_;
    WVoleSeedsV1 seeds_;
};

}  // namespace uvcc


