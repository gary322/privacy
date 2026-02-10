#pragma once

#include "uvcc/nccl.h"
#include "uvcc/slots.h"
#include "uvcc/status.h"

#include <vector>

namespace uvcc {

// Phase 5: data-parallel gradient reduction (within a party, across replicas).
//
// On GPU this is an NCCL allreduce across replicas for fixed (s,t). In this repo
// we provide:
// - a stub NCCL-based API (returns error until Phase 6 wires real NCCL), and
// - a CPU in-process simulator used by unit tests.

struct DPContextV1 {
    NcclCommV1 comm;  // group across replicas for fixed (s,t)
    u32 R = 1;
};

// NCCL path (stub until NCCL binding exists).
StatusV1 dp_allreduce_gradients_v1(DPContextV1* dp, RSSU64SlotV1* grad);

// CPU simulator: sum gradients across all provided replicas and write result back into each.
void dp_allreduce_gradients_sim_v1(const std::vector<RSSU64SlotV1*>& grads);

}  // namespace uvcc


