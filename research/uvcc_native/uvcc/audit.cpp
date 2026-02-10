#include "uvcc/audit.h"

#include "uvcc/bytes.h"
#include "uvcc/ids.h"
#include "uvcc/sha256.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

namespace uvcc {

static void _write_dom(ByteWriter& w, const char* dom) {
    w.write_bytes(dom, std::strlen(dom));
}

Hash32 compute_replica_root_v1(const Sid32& sid_job, u32 step_id, u32 replica, const std::vector<SubsessionRootV1>& all_roots) {
    // replica_root = SHA256("UVCC_REPLICA_ROOT_V1" || sid_rep[r] || LE32(step) || concat(subsession_roots_for_replica_sorted))
    const Sid32 sid_rep = derive_sid_replica_v1(sid_job, replica);

    std::vector<SubsessionRootV1> rs;
    for (const auto& x : all_roots) {
        if (static_cast<u32>(x.coord.replica) == replica) rs.push_back(x);
    }
    if (rs.empty()) throw std::runtime_error("compute_replica_root_v1: no subsession roots for replica");

    std::sort(rs.begin(), rs.end(), [](const SubsessionRootV1& a, const SubsessionRootV1& b) {
        if (a.coord.party != b.coord.party) return a.coord.party < b.coord.party;
        if (a.coord.stage != b.coord.stage) return a.coord.stage < b.coord.stage;
        return a.coord.tp_rank < b.coord.tp_rank;
    });

    ByteWriter w;
    _write_dom(w, "UVCC_REPLICA_ROOT_V1");
    w.write_bytes(sid_rep);
    w.write_u32_le(step_id);
    for (const auto& x : rs) {
        w.write_bytes(x.merkle_root);
    }
    return sha256(w.bytes().data(), w.bytes().size());
}

Hash32 compute_global_root_v1(const Sid32& sid_job, u32 step_id, const std::vector<ReplicaRootV1>& reps) {
    // global_root = SHA256("UVCC_GLOBAL_ROOT_V1" || sid_job || LE32(step) || concat(replica_root[r] for r in increasing r))
    if (reps.empty()) throw std::runtime_error("compute_global_root_v1: empty reps");
    std::vector<ReplicaRootV1> rs = reps;
    std::sort(rs.begin(), rs.end(), [](const ReplicaRootV1& a, const ReplicaRootV1& b) { return a.replica < b.replica; });

    ByteWriter w;
    _write_dom(w, "UVCC_GLOBAL_ROOT_V1");
    w.write_bytes(sid_job);
    w.write_u32_le(step_id);
    for (const auto& r : rs) w.write_bytes(r.root);
    return sha256(w.bytes().data(), w.bytes().size());
}

ResultV1<GlobalAuditBundleV1> build_audit_bundle_v1(const Sid32& sid_job, u32 step_id, const std::vector<SubsessionRootV1>& all_roots) {
    try {
        if (all_roots.empty()) return ResultV1<GlobalAuditBundleV1>(StatusV1::Error("build_audit_bundle_v1: all_roots empty"));

        std::set<u32> replicas;
        for (const auto& x : all_roots) replicas.insert(static_cast<u32>(x.coord.replica));
        if (replicas.empty()) return ResultV1<GlobalAuditBundleV1>(StatusV1::Error("build_audit_bundle_v1: no replicas found"));

        std::vector<ReplicaRootV1> rep_roots;
        rep_roots.reserve(replicas.size());
        for (u32 r : replicas) {
            ReplicaRootV1 rr;
            rr.replica = r;
            rr.root = compute_replica_root_v1(sid_job, step_id, r, all_roots);
            rep_roots.push_back(rr);
        }

        GlobalAuditBundleV1 b;
        b.sid_job = sid_job;
        b.step_id = step_id;
        b.replica_roots = rep_roots;
        b.global_root = compute_global_root_v1(sid_job, step_id, rep_roots);
        return ResultV1<GlobalAuditBundleV1>(b);
    } catch (const std::exception& e) {
        return ResultV1<GlobalAuditBundleV1>(StatusV1::Error(std::string("build_audit_bundle_v1: ") + e.what()));
    }
}

}  // namespace uvcc


