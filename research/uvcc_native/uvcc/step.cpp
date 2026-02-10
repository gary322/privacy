#include "uvcc/step.h"

namespace uvcc {

StatusV1 StepRunnerV1::run_one_step(std::vector<WorkerRuntimeV1*>& workers, std::size_t max_ticks) {
    if (workers.empty()) return StatusV1::Error("StepRunnerV1::run_one_step: workers empty");
    if (max_ticks == 0) return StatusV1::Error("StepRunnerV1::run_one_step: max_ticks must be >0");

    for (std::size_t tick = 0; tick < max_ticks; tick++) {
        bool all_done = true;
        for (auto* w : workers) {
            if (w == nullptr) return StatusV1::Error("StepRunnerV1::run_one_step: null worker");
            auto st = w->tick_one();
            if (!st.ok()) return st;
            if (!w->is_done()) all_done = false;
        }
        if (all_done) return StatusV1::Ok();
    }
    return StatusV1::Error("StepRunnerV1::run_one_step: timeout (max_ticks exceeded)");
}

}  // namespace uvcc


