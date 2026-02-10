#include "uvcc/dp.h"

#include <stdexcept>
#include <vector>

namespace uvcc {

StatusV1 dp_allreduce_gradients_v1(DPContextV1* dp, RSSU64SlotV1* grad) {
    if (dp == nullptr) return StatusV1::Error("dp_allreduce_gradients_v1: dp is null");
    if (grad == nullptr) return StatusV1::Error("dp_allreduce_gradients_v1: grad is null");
    // In the current C++ core, RSSU64SlotV1 stores both local components (lo/hi) together.
    // A real NCCL binding would allreduce both buffers across replicas.
    if (!dp->comm.enabled) return StatusV1::Error("dp_allreduce_gradients_v1: NCCL not enabled (stub)");
    // If enabled, we'd call NCCL allreduce for both buffers.
    return StatusV1::Error("dp_allreduce_gradients_v1: NCCL path unimplemented");
}

void dp_allreduce_gradients_sim_v1(const std::vector<RSSU64SlotV1*>& grads) {
    if (grads.empty()) throw std::runtime_error("dp_allreduce_gradients_sim_v1: grads empty");
    const RSSU64SlotV1* first = grads[0];
    if (first == nullptr) throw std::runtime_error("dp_allreduce_gradients_sim_v1: null grad slot");
    const u32 n = first->n_words;
    for (auto* g : grads) {
        if (g == nullptr) throw std::runtime_error("dp_allreduce_gradients_sim_v1: null grad slot");
        if (g->n_words != n) throw std::runtime_error("dp_allreduce_gradients_sim_v1: n_words mismatch");
        if (g->lo == nullptr || g->hi == nullptr) throw std::runtime_error("dp_allreduce_gradients_sim_v1: null buffers");
    }

    std::vector<u64> sum_lo(static_cast<std::size_t>(n), 0);
    std::vector<u64> sum_hi(static_cast<std::size_t>(n), 0);
    for (auto* g : grads) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); i++) {
            sum_lo[i] = static_cast<u64>(sum_lo[i] + g->lo[i]);
            sum_hi[i] = static_cast<u64>(sum_hi[i] + g->hi[i]);
        }
    }

    for (auto* g : grads) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); i++) {
            g->lo[i] = sum_lo[i];
            g->hi[i] = sum_hi[i];
        }
    }
}

}  // namespace uvcc


