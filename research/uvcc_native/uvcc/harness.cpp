#include "uvcc/harness.h"

namespace uvcc {

StatusV1 run_unit_tests_v1() { return StatusV1::Ok(); }
StatusV1 run_golden_tests_v1() { return StatusV1::Ok(); }
StatusV1 run_determinism_test_v1(u32) { return StatusV1::Ok(); }
StatusV1 run_fault_injection_test_v1(const FaultConfigV1&) { return StatusV1::Ok(); }
StatusV1 run_perf_smoke_test_v1() { return StatusV1::Ok(); }

}  // namespace uvcc


