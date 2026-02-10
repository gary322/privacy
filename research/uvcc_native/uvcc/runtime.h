#pragma once

#include "uvcc/config.h"
#include "uvcc/dp.h"
#include "uvcc/open.h"
#include "uvcc/optim.h"
#include "uvcc/pp.h"
#include "uvcc/sgir.h"
#include "uvcc/slots.h"
#include "uvcc/status.h"
#include "uvcc/tp.h"
#include "uvcc/transport.h"
#include "uvcc/transcript.h"
#include "uvcc/types.h"

// STL
#include <vector>

namespace uvcc {

// Phase 5: per-worker runtime glue.
//
// For now this is a minimal skeleton to wire together:
// - transport polling
// - deterministic PP scheduling
// - TP/DP contexts
// - stage program execution (placeholder)
//
// Phase 6 will add a StepRunner that coordinates all workers in a party.

struct WorkerRuntimeV1 {
    WorkerConfigV1 cfg;
    CoordV1 coord;
    Sid32 sid_sub{};

    TransportV1* transport = nullptr;
    TranscriptStoreV1* transcript = nullptr;
    SlotMapV1* slots = nullptr;
    OpenEngineV1* open = nullptr;

    PPSchedulerV1 pp_sched = PPSchedulerV1(/*S=*/1, /*M=*/1, /*stage_id=*/0);

    TPContextV1 tp;
    DPContextV1 dp;
    SGDParamsV1 sgd;

    StageProgramV1 program;

    // Current training step id (also used as transcript epoch_id32 in v1).
    u32 step_id = 0;
    // Number of microbatches for this worker.
    u16 microbatches = 1;

    // Toy OpenEngine integration state (Phase 6 bring-up):
    // tracks which (phase,mb) currently has an in-flight OPEN.
    std::vector<bool> open_inflight_fwd;
    std::vector<bool> open_inflight_bwd;

    // One tick: pump network, then run at most one runnable task (placeholder compute).
    StatusV1 tick_one();

    bool is_done() const;
};

}  // namespace uvcc


