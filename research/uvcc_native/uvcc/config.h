#pragma once

#include "uvcc/subsession.h"

#include <stdexcept>
#include <string>

namespace uvcc {

// Minimal Phase 1 worker config: enough to deterministically derive (coord, sid_rep, sid_sub).
struct WorkerConfigV1 {
    TopologyV1 topo;
    CoordV1 coord;
    Sid32 sid_job{};
};

// Parse the uvcc_worker CLI arguments (Phase 0–1 bring-up).
WorkerConfigV1 parse_worker_args_v1(int argc, char** argv);

inline SubsessionV1 make_subsession_from_cfg_v1(const WorkerConfigV1& cfg) {
    return make_subsession_v1(cfg.topo, cfg.coord, cfg.sid_job);
}

}  // namespace uvcc


