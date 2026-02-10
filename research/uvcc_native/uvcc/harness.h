#pragma once

#include "uvcc/status.h"
#include "uvcc/types.h"

namespace uvcc {

// Phase 6: test harness entrypoints (stub; tests live under /tests for now).
struct FaultConfigV1 {
    u32 seed = 0;
    double drop_prob = 0.0;
    double dup_prob = 0.0;
    double reorder_prob = 0.0;
};

StatusV1 run_unit_tests_v1();
StatusV1 run_golden_tests_v1();
StatusV1 run_determinism_test_v1(u32 steps);
StatusV1 run_fault_injection_test_v1(const FaultConfigV1& cfg);
StatusV1 run_perf_smoke_test_v1();

}  // namespace uvcc


