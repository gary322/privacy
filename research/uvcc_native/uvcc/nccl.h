#pragma once

#include "uvcc/device_buffer.h"
#include "uvcc/status.h"
#include "uvcc/topology.h"
#include "uvcc/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef UVCC_WITH_CUDA_NCCL
// NOTE: CUDA/NCCL headers must be included in the global namespace.
// Including them inside `namespace uvcc {}` breaks CUDA's inline wrappers
// that reference global `::cuda*` symbols.
#include <cuda_runtime_api.h>
#include <nccl.h>
#endif

namespace uvcc {

// Phase 5: intra-party communicator groups and (optional) NCCL wrappers.
//
// IMPORTANT:
// - Group membership logic is implemented and unit-tested.
// - CUDA/NCCL bindings are optional and enabled only when built with UVCC_WITH_CUDA_NCCL.

enum class NcclGroupKindV1 : u8 { TP = 1, PP = 2, DP = 3 };

struct NcclGroupV1 {
    NcclGroupKindV1 kind = NcclGroupKindV1::TP;
    // Parameters for debugging / deterministic naming.
    u32 replica = 0;
    u16 stage = 0;
    u16 tp_rank = 0;
    std::vector<u32> ranks;  // local ranks within a party
};

// Map (replica, stage, tp_rank) -> local rank within a party.
//
// Layout matches PARALLEL.txt examples:
//   local_rank = r*(S*T) + s*T + t
inline u32 local_rank_v1(const TopologyV1& topo, u32 replica, u16 stage, u16 tp_rank) {
    validate_topology(topo);
    if (replica >= topo.replicas) throw std::runtime_error("local_rank_v1: replica out of range");
    if (stage >= topo.stages) throw std::runtime_error("local_rank_v1: stage out of range");
    if (tp_rank >= topo.tp_ranks) throw std::runtime_error("local_rank_v1: tp_rank out of range");
    const u32 S = static_cast<u32>(topo.stages);
    const u32 T = static_cast<u32>(topo.tp_ranks);
    return static_cast<u32>(replica) * (S * T) + static_cast<u32>(stage) * T + static_cast<u32>(tp_rank);
}

inline u32 local_worker_count_v1(const TopologyV1& topo) {
    validate_topology(topo);
    return static_cast<u32>(topo.replicas) * static_cast<u32>(topo.stages) * static_cast<u32>(topo.tp_ranks);
}

// TP group: fixed (r,s), varying t in [0..T-1], size T
NcclGroupV1 make_tp_group_v1(const TopologyV1& topo, u32 replica, u16 stage);

// PP group: fixed (r,t), varying s in [0..S-1], size S
NcclGroupV1 make_pp_group_v1(const TopologyV1& topo, u32 replica, u16 tp_rank);

// DP group: fixed (s,t), varying r in [0..R-1], size R
NcclGroupV1 make_dp_group_v1(const TopologyV1& topo, u16 stage, u16 tp_rank);

struct NcclCommV1 {
    NcclGroupV1 group;
    u32 my_rank_in_party = 0;
    u32 my_rank_in_group = 0;
    u32 n_ranks = 0;
    bool enabled = false;       // true iff built with CUDA+NCCL
    bool initialized = false;   // true after ncclCommInitRank succeeds

#ifdef UVCC_WITH_CUDA_NCCL
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
#endif
};

ResultV1<NcclCommV1> create_nccl_comm_v1(const NcclGroupV1& g, u32 my_rank_in_party);

// NCCL unique id is 128 bytes in current NCCL versions.
constexpr std::size_t NCCL_UNIQUE_ID_BYTES_V1 = 128;
using NcclUniqueIdV1 = std::array<u8, NCCL_UNIQUE_ID_BYTES_V1>;

ResultV1<NcclUniqueIdV1> nccl_get_unique_id_v1();
StatusV1 nccl_init_rank_v1(NcclCommV1* comm, const NcclUniqueIdV1& uid);
StatusV1 nccl_destroy_v1(NcclCommV1* comm);

StatusV1 nccl_allreduce_sum_v1(NcclCommV1* comm, DeviceBufferV1 buf);
// Async variants: enqueue NCCL op on comm->stream but do NOT synchronize.
// Caller is responsible for stream synchronization (or event-based waiting).
StatusV1 nccl_allreduce_sum_async_v1(NcclCommV1* comm, DeviceBufferV1 buf);
StatusV1 nccl_allgather_v1(NcclCommV1* comm, DeviceBufferV1 in, DeviceBufferV1 out);
StatusV1 nccl_allgather_async_v1(NcclCommV1* comm, DeviceBufferV1 in, DeviceBufferV1 out);
StatusV1 nccl_send_v1(NcclCommV1* comm, DeviceBufferV1 buf, u32 peer_rank);
StatusV1 nccl_recv_v1(NcclCommV1* comm, DeviceBufferV1 buf, u32 peer_rank);

}  // namespace uvcc


