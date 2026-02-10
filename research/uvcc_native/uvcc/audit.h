#pragma once

#include "uvcc/status.h"
#include "uvcc/topology.h"
#include "uvcc/types.h"

#include <cstdint>
#include <vector>

namespace uvcc {

// Phase 6: transcript-of-transcripts root packaging.

struct SubsessionRootV1 {
    CoordV1 coord;
    Hash32 merkle_root{};
};

struct ReplicaRootV1 {
    u32 replica = 0;
    Hash32 root{};
};

struct GlobalAuditBundleV1 {
    Sid32 sid_job{};
    u32 step_id = 0;
    Hash32 global_root{};
    std::vector<ReplicaRootV1> replica_roots;
    // Signatures per party are Phase 7+ (keys/signing).
};

Hash32 compute_replica_root_v1(const Sid32& sid_job, u32 step_id, u32 replica, const std::vector<SubsessionRootV1>& all_roots);
Hash32 compute_global_root_v1(const Sid32& sid_job, u32 step_id, const std::vector<ReplicaRootV1>& reps);

ResultV1<GlobalAuditBundleV1> build_audit_bundle_v1(const Sid32& sid_job, u32 step_id, const std::vector<SubsessionRootV1>& all_roots);

}  // namespace uvcc


