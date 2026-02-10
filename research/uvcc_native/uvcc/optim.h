#pragma once

#include "uvcc/slots.h"
#include "uvcc/status.h"
#include "uvcc/types.h"

namespace uvcc {

// Phase 5: optimizer on RSS shares.
//
// For now we implement deterministic SGD on u64 ring values.
// (Adam and fixed-point scaling are Phase 7+ territory.)

struct SGDParamsV1 {
    double lr = 0.0;  // must be an integer-valued scalar in v1 implementation
};

StatusV1 sgd_update_v1(const SGDParamsV1& p, RSSU64SlotV1* weight, const RSSU64SlotV1* grad);

}  // namespace uvcc


