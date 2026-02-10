#include "uvcc/tp.h"

namespace uvcc {

StatusV1 tp_allreduce_partial_v1(TPContextV1* tp, DeviceBufferV1 partial) {
    if (tp == nullptr) return StatusV1::Error("tp_allreduce_partial_v1: tp is null");
    return nccl_allreduce_sum_v1(&tp->comm, partial);
}

StatusV1 tp_allgather_inputs_v1(TPContextV1* tp, DeviceBufferV1 local, DeviceBufferV1 gathered) {
    if (tp == nullptr) return StatusV1::Error("tp_allgather_inputs_v1: tp is null");
    return nccl_allgather_v1(&tp->comm, local, gathered);
}

}  // namespace uvcc


