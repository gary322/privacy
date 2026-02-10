#include "uvcc/nccl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace uvcc {

NcclGroupV1 make_tp_group_v1(const TopologyV1& topo, u32 replica, u16 stage) {
    validate_topology(topo);
    NcclGroupV1 g;
    g.kind = NcclGroupKindV1::TP;
    g.replica = replica;
    g.stage = stage;
    g.tp_rank = 0;
    g.ranks.reserve(static_cast<std::size_t>(topo.tp_ranks));
    for (u16 t = 0; t < topo.tp_ranks; t++) {
        g.ranks.push_back(local_rank_v1(topo, replica, stage, t));
    }
    return g;
}

NcclGroupV1 make_pp_group_v1(const TopologyV1& topo, u32 replica, u16 tp_rank) {
    validate_topology(topo);
    NcclGroupV1 g;
    g.kind = NcclGroupKindV1::PP;
    g.replica = replica;
    g.stage = 0;
    g.tp_rank = tp_rank;
    g.ranks.reserve(static_cast<std::size_t>(topo.stages));
    for (u16 s = 0; s < topo.stages; s++) {
        g.ranks.push_back(local_rank_v1(topo, replica, s, tp_rank));
    }
    return g;
}

NcclGroupV1 make_dp_group_v1(const TopologyV1& topo, u16 stage, u16 tp_rank) {
    validate_topology(topo);
    NcclGroupV1 g;
    g.kind = NcclGroupKindV1::DP;
    g.replica = 0;
    g.stage = stage;
    g.tp_rank = tp_rank;
    g.ranks.reserve(static_cast<std::size_t>(topo.replicas));
    for (u32 r = 0; r < topo.replicas; r++) {
        g.ranks.push_back(local_rank_v1(topo, r, stage, tp_rank));
    }
    return g;
}

ResultV1<NcclCommV1> create_nccl_comm_v1(const NcclGroupV1& g, u32 my_rank_in_party) {
    const bool member = std::find(g.ranks.begin(), g.ranks.end(), my_rank_in_party) != g.ranks.end();
    if (!member) return ResultV1<NcclCommV1>(StatusV1::Error("create_nccl_comm_v1: rank is not a member of group"));
    NcclCommV1 c;
    c.group = g;
    c.my_rank_in_party = my_rank_in_party;
    // Rank index within this group (0..n_ranks-1).
    for (std::size_t i = 0; i < g.ranks.size(); i++) {
        if (g.ranks[i] == my_rank_in_party) {
            c.my_rank_in_group = static_cast<u32>(i);
            break;
        }
    }
    c.n_ranks = static_cast<u32>(g.ranks.size());

#ifdef UVCC_WITH_CUDA_NCCL
    c.enabled = true;
#else
    c.enabled = false;
#endif
    return ResultV1<NcclCommV1>(c);
}

#ifndef UVCC_WITH_CUDA_NCCL
static StatusV1 _no_nccl() { return StatusV1::Error("NCCL/CUDA not enabled (build with -DUVCC_WITH_CUDA_NCCL=ON)"); }
#endif

ResultV1<NcclUniqueIdV1> nccl_get_unique_id_v1() {
#ifndef UVCC_WITH_CUDA_NCCL
    return ResultV1<NcclUniqueIdV1>(_no_nccl());
#else
    ncclUniqueId id;
    const ncclResult_t rc = ncclGetUniqueId(&id);
    if (rc != ncclSuccess) {
        return ResultV1<NcclUniqueIdV1>(StatusV1::Error(std::string("ncclGetUniqueId failed: ") + ncclGetErrorString(rc)));
    }
    NcclUniqueIdV1 out{};
    static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES_V1, "ncclUniqueId size unexpected");
    std::memcpy(out.data(), &id, NCCL_UNIQUE_ID_BYTES_V1);
    return ResultV1<NcclUniqueIdV1>(out);
#endif
}

StatusV1 nccl_init_rank_v1(NcclCommV1* comm, const NcclUniqueIdV1& uid) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)uid;
    return _no_nccl();
#else
    if (comm == nullptr) return StatusV1::Error("nccl_init_rank_v1: comm is null");
    if (!comm->enabled) return StatusV1::Error("nccl_init_rank_v1: comm not enabled");
    if (comm->initialized) return StatusV1::Ok();
    if (comm->n_ranks == 0) return StatusV1::Error("nccl_init_rank_v1: n_ranks=0");
    if (comm->my_rank_in_group >= comm->n_ranks) return StatusV1::Error("nccl_init_rank_v1: rank_in_group out of range");

    // Ensure CUDA is usable on this process (CUDA_VISIBLE_DEVICES is typically set by the runner).
    const cudaError_t d0 = cudaSetDevice(0);
    if (d0 != cudaSuccess) {
        return StatusV1::Error(std::string("cudaSetDevice(0) failed: ") + cudaGetErrorString(d0));
    }

    ncclUniqueId id;
    static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES_V1, "ncclUniqueId size unexpected");
    std::memcpy(&id, uid.data(), NCCL_UNIQUE_ID_BYTES_V1);

    ncclComm_t c = nullptr;
    const ncclResult_t rc = ncclCommInitRank(&c, static_cast<int>(comm->n_ranks), id, static_cast<int>(comm->my_rank_in_group));
    if (rc != ncclSuccess) {
        return StatusV1::Error(std::string("ncclCommInitRank failed: ") + ncclGetErrorString(rc));
    }
    comm->comm = c;

    cudaStream_t s = nullptr;
    const cudaError_t sc = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (sc != cudaSuccess) {
        (void)ncclCommDestroy(comm->comm);
        comm->comm = nullptr;
        return StatusV1::Error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(sc));
    }
    comm->stream = s;
    comm->initialized = true;
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_destroy_v1(NcclCommV1* comm) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    return StatusV1::Ok();
#else
    if (comm == nullptr) return StatusV1::Ok();
    if (comm->stream != nullptr) {
        (void)cudaStreamDestroy(comm->stream);
        comm->stream = nullptr;
    }
    if (comm->comm != nullptr) {
        (void)ncclCommDestroy(comm->comm);
        comm->comm = nullptr;
    }
    comm->initialized = false;
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_allreduce_sum_v1(NcclCommV1* comm, DeviceBufferV1 buf) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)buf;
    return _no_nccl();
#else
    auto st = nccl_allreduce_sum_async_v1(comm, buf);
    if (!st.ok()) return st;
    const cudaError_t sc = cudaStreamSynchronize(comm->stream);
    if (sc != cudaSuccess) return StatusV1::Error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(sc));
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_allreduce_sum_async_v1(NcclCommV1* comm, DeviceBufferV1 buf) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)buf;
    return _no_nccl();
#else
    if (comm == nullptr) return StatusV1::Error("nccl_allreduce_sum_v1: comm is null");
    if (!comm->enabled || !comm->initialized || comm->comm == nullptr) return StatusV1::Error("nccl_allreduce_sum_v1: comm not initialized");
    if (buf.ptr == nullptr || buf.n_bytes == 0) return StatusV1::Error("nccl_allreduce_sum_v1: empty buffer");
    if ((buf.n_bytes % 8) != 0) return StatusV1::Error("nccl_allreduce_sum_v1: n_bytes must be multiple of 8 (u64)");
    const std::size_t count = buf.n_bytes / 8;

    const ncclResult_t rc = ncclAllReduce(buf.ptr, buf.ptr, static_cast<std::size_t>(count), ncclUint64, ncclSum, comm->comm, comm->stream);
    if (rc != ncclSuccess) return StatusV1::Error(std::string("ncclAllReduce failed: ") + ncclGetErrorString(rc));
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_allgather_v1(NcclCommV1* comm, DeviceBufferV1 in, DeviceBufferV1 out) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)in;
    (void)out;
    return _no_nccl();
#else
    auto st = nccl_allgather_async_v1(comm, in, out);
    if (!st.ok()) return st;
    const cudaError_t sc = cudaStreamSynchronize(comm->stream);
    if (sc != cudaSuccess) return StatusV1::Error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(sc));
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_allgather_async_v1(NcclCommV1* comm, DeviceBufferV1 in, DeviceBufferV1 out) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)in;
    (void)out;
    return _no_nccl();
#else
    if (comm == nullptr) return StatusV1::Error("nccl_allgather_v1: comm is null");
    if (!comm->enabled || !comm->initialized || comm->comm == nullptr) return StatusV1::Error("nccl_allgather_v1: comm not initialized");
    if (in.ptr == nullptr || in.n_bytes == 0) return StatusV1::Error("nccl_allgather_v1: empty in buffer");
    if (out.ptr == nullptr || out.n_bytes == 0) return StatusV1::Error("nccl_allgather_v1: empty out buffer");
    if ((in.n_bytes % 8) != 0 || (out.n_bytes % 8) != 0) return StatusV1::Error("nccl_allgather_v1: buffers must be multiple of 8 (u64)");
    const std::size_t in_count = in.n_bytes / 8;
    const std::size_t want_out = in.n_bytes * static_cast<std::size_t>(comm->n_ranks);
    if (out.n_bytes != want_out) return StatusV1::Error("nccl_allgather_v1: out.n_bytes mismatch (want in.n_bytes*n_ranks)");

    const ncclResult_t rc = ncclAllGather(in.ptr, out.ptr, static_cast<std::size_t>(in_count), ncclUint64, comm->comm, comm->stream);
    if (rc != ncclSuccess) return StatusV1::Error(std::string("ncclAllGather failed: ") + ncclGetErrorString(rc));
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_send_v1(NcclCommV1* comm, DeviceBufferV1 buf, u32 peer_rank) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)buf;
    (void)peer_rank;
    return _no_nccl();
#else
    if (comm == nullptr) return StatusV1::Error("nccl_send_v1: comm is null");
    if (!comm->enabled || !comm->initialized || comm->comm == nullptr) return StatusV1::Error("nccl_send_v1: comm not initialized");
    if (buf.ptr == nullptr || buf.n_bytes == 0) return StatusV1::Error("nccl_send_v1: empty buffer");
    if ((buf.n_bytes % 8) != 0) return StatusV1::Error("nccl_send_v1: n_bytes must be multiple of 8 (u64)");
    if (peer_rank >= comm->n_ranks) return StatusV1::Error("nccl_send_v1: peer_rank out of range");
    const std::size_t count = buf.n_bytes / 8;
    const ncclResult_t rc = ncclSend(buf.ptr, static_cast<std::size_t>(count), ncclUint64, static_cast<int>(peer_rank), comm->comm, comm->stream);
    if (rc != ncclSuccess) return StatusV1::Error(std::string("ncclSend failed: ") + ncclGetErrorString(rc));
    return StatusV1::Ok();
#endif
}

StatusV1 nccl_recv_v1(NcclCommV1* comm, DeviceBufferV1 buf, u32 peer_rank) {
#ifndef UVCC_WITH_CUDA_NCCL
    (void)comm;
    (void)buf;
    (void)peer_rank;
    return _no_nccl();
#else
    if (comm == nullptr) return StatusV1::Error("nccl_recv_v1: comm is null");
    if (!comm->enabled || !comm->initialized || comm->comm == nullptr) return StatusV1::Error("nccl_recv_v1: comm not initialized");
    if (buf.ptr == nullptr || buf.n_bytes == 0) return StatusV1::Error("nccl_recv_v1: empty buffer");
    if ((buf.n_bytes % 8) != 0) return StatusV1::Error("nccl_recv_v1: n_bytes must be multiple of 8 (u64)");
    if (peer_rank >= comm->n_ranks) return StatusV1::Error("nccl_recv_v1: peer_rank out of range");
    const std::size_t count = buf.n_bytes / 8;
    const ncclResult_t rc = ncclRecv(buf.ptr, static_cast<std::size_t>(count), ncclUint64, static_cast<int>(peer_rank), comm->comm, comm->stream);
    if (rc != ncclSuccess) return StatusV1::Error(std::string("ncclRecv failed: ") + ncclGetErrorString(rc));
    return StatusV1::Ok();
#endif
}

}  // namespace uvcc


