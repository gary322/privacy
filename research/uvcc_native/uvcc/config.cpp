#include "uvcc/config.h"

#include "uvcc/hex.h"
#include "uvcc/topology.h"

#include <cstdlib>
#include <string>

namespace uvcc {
namespace {

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == key) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return std::string(argv[i + 1]);
        }
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == key) return true;
    }
    return false;
}

}  // namespace

WorkerConfigV1 parse_worker_args_v1(int argc, char** argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        throw std::runtime_error("help_requested");
    }
    const std::string sid_job_hex = get_arg(argc, argv, "--sid-job-hex");
    if (sid_job_hex.empty()) throw std::runtime_error("missing --sid-job-hex");

    WorkerConfigV1 cfg;
    cfg.sid_job = parse_hex_sid32(sid_job_hex);

    cfg.topo.replicas = static_cast<u32>(std::stoul(get_arg(argc, argv, "--replicas", "1")));
    cfg.topo.stages = static_cast<u16>(std::stoul(get_arg(argc, argv, "--stages", "1")));
    cfg.topo.tp_ranks = static_cast<u16>(std::stoul(get_arg(argc, argv, "--tp-ranks", "1")));

    cfg.coord.party = static_cast<u8>(std::stoul(get_arg(argc, argv, "--party", "0")));
    cfg.coord.replica = static_cast<u32>(std::stoul(get_arg(argc, argv, "--replica", "0")));
    cfg.coord.stage = static_cast<u16>(std::stoul(get_arg(argc, argv, "--stage", "0")));
    cfg.coord.tp_rank = static_cast<u16>(std::stoul(get_arg(argc, argv, "--tp", "0")));

    // Validate everything early.
    validate_coord(cfg.topo, cfg.coord);
    return cfg;
}

}  // namespace uvcc


