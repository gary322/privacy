#pragma once

#include "uvcc/ops.h"

#include <vector>

namespace uvcc {

// Phase 4 scaffold: StageProgram is the per-(stage,tp_rank) op list executed deterministically.
struct StageProgramV1 {
    std::vector<OpV1> ops;
};

}  // namespace uvcc


