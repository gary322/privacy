#pragma once

#include "uvcc/types.h"

#include <stdexcept>

namespace uvcc {

struct TopologyV1 {
    // Number of replicas (R), pipeline stages (S), tensor ranks (T).
    u32 replicas = 1;
    u16 stages = 1;
    u16 tp_ranks = 1;
};

struct CoordV1 {
    // Party id in {0,1,2}.
    u8 party = 0;
    // Replica id in [0..R-1]
    u32 replica = 0;
    // Stage id in [0..S-1]
    u16 stage = 0;
    // Tensor rank in [0..T-1]
    u16 tp_rank = 0;
};

inline void validate_topology(const TopologyV1& t) {
    if (t.replicas == 0) throw std::runtime_error("topology.replicas must be >0");
    if (t.stages == 0) throw std::runtime_error("topology.stages must be >0");
    if (t.tp_ranks == 0) throw std::runtime_error("topology.tp_ranks must be >0");
    if (t.stages > 255) throw std::runtime_error("topology.stages must be <=255 in v1 (sid_sub uses U8(stage))");
}

inline void validate_coord(const TopologyV1& t, const CoordV1& c) {
    validate_topology(t);
    if (c.party > 2) throw std::runtime_error("coord.party must be in {0,1,2}");
    if (c.replica >= t.replicas) throw std::runtime_error("coord.replica out of range");
    if (c.stage >= t.stages) throw std::runtime_error("coord.stage out of range");
    if (c.tp_rank >= t.tp_ranks) throw std::runtime_error("coord.tp_rank out of range");
}

}  // namespace uvcc


