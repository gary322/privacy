#pragma once

#include "uvcc/runtime.h"
#include "uvcc/status.h"

#include <cstddef>
#include <vector>

namespace uvcc {

// Phase 6: per-(party) step orchestrator.
//
// In production each party runs its own StepRunner over its local workers. In tests
// we may also run a single StepRunner over all workers (across parties) to drive
// an in-process simulation.
struct StepRunnerV1 {
    u32 step_id = 0;

    StatusV1 run_one_step(std::vector<WorkerRuntimeV1*>& workers, std::size_t max_ticks = 100000);
};

}  // namespace uvcc


