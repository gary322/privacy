#pragma once

#include "uvcc/ids.h"
#include "uvcc/topology.h"
#include "uvcc/types.h"

namespace uvcc {

struct SubsessionV1 {
    TopologyV1 topo;
    CoordV1 coord;
    Sid32 sid_job{};
    Sid32 sid_rep{};
    Sid32 sid_sub{};
};

inline SubsessionV1 make_subsession_v1(const TopologyV1& topo, const CoordV1& coord, const Sid32& sid_job) {
    validate_coord(topo, coord);
    SubsessionV1 s;
    s.topo = topo;
    s.coord = coord;
    s.sid_job = sid_job;
    s.sid_rep = derive_sid_replica_v1(sid_job, static_cast<u32>(coord.replica));
    s.sid_sub = derive_sid_sub_v1(s.sid_rep, static_cast<u16>(coord.stage), static_cast<u16>(coord.tp_rank));
    return s;
}

}  // namespace uvcc


