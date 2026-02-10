#pragma once

#include "uvcc/device_buffer.h"
#include "uvcc/nccl.h"
#include "uvcc/status.h"
#include "uvcc/types.h"

namespace uvcc {

// Phase 5: tensor-parallel collective wrappers (intra-party).
struct TPContextV1 {
    NcclCommV1 comm;
    u16 tp_rank = 0;
    u16 T = 1;
};

// Row-parallel partial output sum (SUM allreduce on shares).
StatusV1 tp_allreduce_partial_v1(TPContextV1* tp, DeviceBufferV1 partial);

// Column-parallel input allgather (replicate inputs across TP ranks).
StatusV1 tp_allgather_inputs_v1(TPContextV1* tp, DeviceBufferV1 local, DeviceBufferV1 gathered);

}  // namespace uvcc


