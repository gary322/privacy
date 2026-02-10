#include "uvcc/optim.h"

#include <cmath>
#include <limits>

namespace uvcc {

StatusV1 sgd_update_v1(const SGDParamsV1& p, RSSU64SlotV1* weight, const RSSU64SlotV1* grad) {
    if (weight == nullptr) return StatusV1::Error("sgd_update_v1: weight is null");
    if (grad == nullptr) return StatusV1::Error("sgd_update_v1: grad is null");
    if (weight->n_words != grad->n_words) return StatusV1::Error("sgd_update_v1: n_words mismatch");
    if (weight->lo == nullptr || weight->hi == nullptr) return StatusV1::Error("sgd_update_v1: weight buffers null");
    if (grad->lo == nullptr || grad->hi == nullptr) return StatusV1::Error("sgd_update_v1: grad buffers null");

    if (!std::isfinite(p.lr)) return StatusV1::Error("sgd_update_v1: lr must be finite");
    const double r = std::round(p.lr);
    if (std::abs(p.lr - r) > 1e-9) return StatusV1::Error("sgd_update_v1: lr must be integer-valued in v1");
    if (r < 0.0) return StatusV1::Error("sgd_update_v1: lr must be >=0 in v1");
    if (r > static_cast<double>(std::numeric_limits<u64>::max())) return StatusV1::Error("sgd_update_v1: lr too large");
    const u64 lr = static_cast<u64>(r);

    for (std::size_t i = 0; i < static_cast<std::size_t>(weight->n_words); i++) {
        weight->lo[i] = static_cast<u64>(weight->lo[i] - lr * grad->lo[i]);
        weight->hi[i] = static_cast<u64>(weight->hi[i] - lr * grad->hi[i]);
    }
    return StatusV1::Ok();
}

}  // namespace uvcc


