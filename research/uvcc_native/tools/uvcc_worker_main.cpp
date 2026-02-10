#include "uvcc/config.h"
#include "uvcc/bytes.h"
#include "uvcc/ids.h"
#include "uvcc/hex.h"
#include "uvcc/nccl.h"
#include "uvcc/relay_rawconn.h"
#include "uvcc/runtime.h"
#include "uvcc/sha256.h"
#include "uvcc/transcript_hooks.h"

#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == key) return true;
    }
    return false;
}

std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == key) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return std::string(argv[i + 1]);
        }
    }
    return def;
}

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("failed to read file: " + path);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // Trim whitespace.
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static uvcc::u64 hash64_open_base(const uvcc::Sid32& sid_sub, uvcc::u32 step_id, uvcc::u8 phase_u8, uvcc::u16 mb) {
    uvcc::ByteWriter w;
    const char* dom = "uvcc.phase6.open.base.v1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid_sub);
    w.write_u32_le(step_id);
    w.write_u8(phase_u8);
    w.write_u16_le(mb);
    const uvcc::Hash32 h = uvcc::sha256(w.bytes().data(), w.bytes().size());
    uvcc::u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<uvcc::u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

static void toy_open_vectors_for_party(uvcc::u8 party, uvcc::u64 base, std::vector<uvcc::u64>* lo, std::vector<uvcc::u64>* hi) {
    if (!lo || !hi) throw std::runtime_error("toy_open_vectors_for_party: null out");
    std::vector<uvcc::u64> c0{static_cast<uvcc::u64>(base + 0), static_cast<uvcc::u64>(base + 1)};
    std::vector<uvcc::u64> c1{static_cast<uvcc::u64>(base + 10), static_cast<uvcc::u64>(base + 11)};
    std::vector<uvcc::u64> c2{static_cast<uvcc::u64>(base + 20), static_cast<uvcc::u64>(base + 21)};

    if (party == 0) {
        *lo = std::move(c0);
        *hi = std::move(c1);
    } else if (party == 1) {
        *lo = std::move(c1);
        *hi = std::move(c2);
    } else if (party == 2) {
        *lo = std::move(c2);
        *hi = std::move(c0);
    } else {
        throw std::runtime_error("toy_open_vectors_for_party: party out of range");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
            std::cout << "uvcc_worker (Phase 0–1 bring-up)\n\n"
                      << "Args:\n"
                      << "  --sid-job-hex <0x..32bytes>\n"
                      << "  --party <0|1|2>\n"
                      << "  --replica <u32>\n"
                      << "  --stage <u16>\n"
                      << "  --tp <u16>\n"
                      << "  --replicas <u32>   (topology R)\n"
                      << "  --stages <u16>     (topology S)\n"
                      << "  --tp-ranks <u16>   (topology T)\n";
            return 0;
        }

        const std::string mode = get_arg(argc, argv, "--mode", "print_ids");
        const uvcc::WorkerConfigV1 cfg = uvcc::parse_worker_args_v1(argc, argv);
        const auto sub = uvcc::make_subsession_from_cfg_v1(cfg);

        if (mode == "print_ids") {
            std::cout << "coord=p" << int(sub.coord.party) << " r" << sub.coord.replica << " s" << sub.coord.stage << " t" << sub.coord.tp_rank
                      << "\n";
            std::cout << "sid_job=0x" << uvcc::hex_lower(sub.sid_job) << "\n";
            std::cout << "sid_rep=0x" << uvcc::hex_lower(sub.sid_rep) << "\n";
            std::cout << "sid_sub=0x" << uvcc::hex_lower(sub.sid_sub) << "\n";
            return 0;
        }

        if (mode == "toy_open") {
            const std::string relay_url = get_arg(argc, argv, "--relay-url");
            if (relay_url.empty()) throw std::runtime_error("missing --relay-url for --mode toy_open");
            const std::string group_id = get_arg(argc, argv, "--relay-group-id");
            if (group_id.empty()) throw std::runtime_error("missing --relay-group-id for --mode toy_open");
            std::string token = get_arg(argc, argv, "--relay-token", "");
            const std::string token_file = get_arg(argc, argv, "--relay-token-file", "");
            if (token.empty() && !token_file.empty()) token = read_text_file(token_file);
            const std::string tls_ca = get_arg(argc, argv, "--tls-ca-pem", "");
            const int ttl_s = static_cast<int>(std::stol(get_arg(argc, argv, "--relay-ttl-s", "3600")));
            const double timeout_s = std::stod(get_arg(argc, argv, "--relay-timeout-s", "10.0"));

            uvcc::RelayHttpConfigV1 rcfg;
            rcfg.base_url = relay_url;
            rcfg.group_id = group_id;
            rcfg.token = token;
            rcfg.tls_ca_pem_path = tls_ca;
            rcfg.timeout_s = timeout_s;
            rcfg.default_ttl_s = ttl_s;

            auto http = std::make_shared<uvcc::RelayHttpClientV1>(rcfg);
            // Relay can take a moment to come up; retry healthz briefly to avoid flaking in multi-host bring-up.
            std::string last_hz_err;
            bool hz_ok = false;
            for (int i = 0; i < 50; i++) {
                const auto hz = http->healthz();
                if (hz.ok()) {
                    hz_ok = true;
                    break;
                }
                last_hz_err = hz.message();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            if (!hz_ok) throw std::runtime_error(std::string("relay healthz failed: ") + last_hz_err);

            auto mailbox = std::make_shared<uvcc::RelayMailboxV1>(http, group_id, int(sub.coord.party));
            auto to_prev = std::make_unique<uvcc::RelayRawConnV1>(
                http, mailbox, group_id, int(sub.coord.party), int((int(sub.coord.party) + 2) % 3), ttl_s);
            auto to_next = std::make_unique<uvcc::RelayRawConnV1>(
                http, mailbox, group_id, int(sub.coord.party), int((int(sub.coord.party) + 1) % 3), ttl_s);

            uvcc::TranscriptStoreV1 ts(sub.sid_sub);
            uvcc::OpenEngineV1 open(sub.sid_sub, sub.coord.party);
            uvcc::TransportCallbacksV1 cbs = uvcc::make_lift_transcript_callbacks_v1(&ts, sub.sid_sub);
            cbs.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& payload) { open.on_deliver(hdr, payload); };
            uvcc::TransportV1 transport(sub.sid_sub, sub.coord.party, to_prev.get(), to_next.get(), cbs, nullptr);

            uvcc::WorkerRuntimeV1 w;
            w.cfg = cfg;
            w.coord = sub.coord;
            w.sid_sub = sub.sid_sub;
            w.transport = &transport;
            w.transcript = &ts;
            w.open = &open;
            w.step_id = static_cast<uvcc::u32>(std::stoul(get_arg(argc, argv, "--step-id", "0")));
            w.microbatches = static_cast<uvcc::u16>(std::stoul(get_arg(argc, argv, "--microbatches", "1")));
            w.pp_sched = uvcc::PPSchedulerV1(/*S=*/1, /*M=*/w.microbatches, /*stage_id=*/0, /*kmax_fwd=*/1, /*kmax_bwd=*/1);

            // Drive until done (local). In a real deployment all parties run concurrently.
            for (int it = 0; it < 100000; it++) {
                const auto st = w.tick_one();
                if (!st.ok()) throw std::runtime_error(st.message());
                if (w.is_done()) break;
            }
            // Drain acks/messages a bit so transcript is stable.
            for (int i = 0; i < 100; i++) {
                transport.poll();
            }
            const auto root = ts.epoch_root(w.step_id);
            std::cout << "epoch_root=0x" << uvcc::hex_lower(root) << "\n";
            return 0;
        }

        if (mode == "nccl_smoke") {
#ifndef UVCC_WITH_CUDA_NCCL
            throw std::runtime_error("nccl_smoke requires UVCC_WITH_CUDA_NCCL (build with -DUVCC_WITH_CUDA_NCCL=ON)");
#else
            const std::string relay_url = get_arg(argc, argv, "--relay-url");
            if (relay_url.empty()) throw std::runtime_error("missing --relay-url for --mode nccl_smoke");
            const std::string relay_group_base = get_arg(argc, argv, "--relay-group-id");
            if (relay_group_base.empty()) throw std::runtime_error("missing --relay-group-id for --mode nccl_smoke");
            std::string token = get_arg(argc, argv, "--relay-token", "");
            const std::string token_file = get_arg(argc, argv, "--relay-token-file", "");
            if (token.empty() && !token_file.empty()) token = read_text_file(token_file);
            const std::string tls_ca = get_arg(argc, argv, "--tls-ca-pem", "");
            const int ttl_s = static_cast<int>(std::stol(get_arg(argc, argv, "--relay-ttl-s", "3600")));
            const double timeout_s = std::stod(get_arg(argc, argv, "--relay-timeout-s", "10.0"));

            // Build a base Relay config; we create per-group clients by overriding group_id.
            uvcc::RelayHttpConfigV1 base_cfg;
            base_cfg.base_url = relay_url;
            base_cfg.group_id = relay_group_base;
            base_cfg.token = token;
            base_cfg.tls_ca_pem_path = tls_ca;
            base_cfg.timeout_s = timeout_s;
            base_cfg.default_ttl_s = ttl_s;

            // Health check once (best-effort).
            {
                auto http0 = std::make_shared<uvcc::RelayHttpClientV1>(base_cfg);
                std::string last_hz_err;
                bool hz_ok = false;
                for (int i = 0; i < 50; i++) {
                    const auto hz = http0->healthz();
                    if (hz.ok()) {
                        hz_ok = true;
                        break;
                    }
                    last_hz_err = hz.message();
                    sleep_ms(200);
                }
                if (!hz_ok) throw std::runtime_error(std::string("relay healthz failed: ") + last_hz_err);
            }

            // Compute this worker's local rank within its party (0..R*S*T-1).
            const uvcc::TopologyV1 topo = cfg.topo;
            const uvcc::CoordV1 coord = cfg.coord;
            const uvcc::u32 my_rank_in_party = uvcc::local_rank_v1(topo, static_cast<uvcc::u32>(coord.replica), coord.stage, coord.tp_rank);

            auto init_group = [&](const std::string& gid, const uvcc::NcclGroupV1& g) -> uvcc::NcclCommV1 {
                auto cr = uvcc::create_nccl_comm_v1(g, my_rank_in_party);
                if (!cr.ok()) throw std::runtime_error(cr.status().message());
                uvcc::NcclCommV1 c = cr.value();
                if (!c.enabled) throw std::runtime_error("NCCL not enabled");

                // Create a per-group HTTP client.
                uvcc::RelayHttpConfigV1 cfg2 = base_cfg;
                cfg2.group_id = gid;
                auto http = std::make_shared<uvcc::RelayHttpClientV1>(cfg2);

                uvcc::NcclUniqueIdV1 uid{};
                if (c.my_rank_in_group == 0) {
                    auto ur = uvcc::nccl_get_unique_id_v1();
                    if (!ur.ok()) throw std::runtime_error(ur.status().message());
                    uid = ur.value();
                    // Broadcast to all other group ranks.
                    for (uvcc::u32 dst = 1; dst < c.n_ranks; dst++) {
                        std::vector<uvcc::u8> payload(uid.begin(), uid.end());
                        const std::string msg_id = std::string("uid-") + std::to_string(dst);
                        auto st = http->enqueue(/*sender=*/0, /*receiver=*/static_cast<int>(dst), msg_id, payload, ttl_s);
                        if (!st.ok()) throw std::runtime_error(st.message());
                    }
                } else {
                    const double deadline = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() + 30.0;
                    while (true) {
                        const double dl = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() + 1.0;
                        auto pr = http->poll(static_cast<int>(c.my_rank_in_group), dl);
                        if (!pr.ok()) throw std::runtime_error(pr.status().message());
                        if (!pr.value().has_value()) {
                            if (std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() > deadline) {
                                throw std::runtime_error("timeout waiting for nccl unique id");
                            }
                            continue;
                        }
                        auto m = std::move(*pr.value());
                        auto st = http->ack(static_cast<int>(c.my_rank_in_group), m.msg_id, m.lease_token);
                        if (!st.ok()) throw std::runtime_error(st.message());
                        if (m.payload.size() != uvcc::NCCL_UNIQUE_ID_BYTES_V1) throw std::runtime_error("bad nccl uid payload size");
                        std::copy(m.payload.begin(), m.payload.end(), uid.begin());
                        break;
                    }
                }

                auto st = uvcc::nccl_init_rank_v1(&c, uid);
                if (!st.ok()) throw std::runtime_error(st.message());
                return c;
            };

            // Derive relay group ids for each NCCL group.
            const std::string job8 = uvcc::hex_lower(sub.sid_job).substr(0, 8);
            const int party = static_cast<int>(coord.party);
            const int r = static_cast<int>(coord.replica);
            const int s = static_cast<int>(coord.stage);
            const int t = static_cast<int>(coord.tp_rank);

            // Create TP/PP/DP comms.
            const auto g_tp = uvcc::make_tp_group_v1(topo, static_cast<uvcc::u32>(coord.replica), coord.stage);
            const auto g_pp = uvcc::make_pp_group_v1(topo, static_cast<uvcc::u32>(coord.replica), coord.tp_rank);
            const auto g_dp = uvcc::make_dp_group_v1(topo, coord.stage, coord.tp_rank);

            auto tp_comm = init_group(relay_group_base + "-nccl-tp-p" + std::to_string(party) + "-r" + std::to_string(r) + "-s" + std::to_string(s) + "-j" + job8, g_tp);
            auto pp_comm = init_group(relay_group_base + "-nccl-pp-p" + std::to_string(party) + "-r" + std::to_string(r) + "-t" + std::to_string(t) + "-j" + job8, g_pp);
            auto dp_comm = init_group(relay_group_base + "-nccl-dp-p" + std::to_string(party) + "-s" + std::to_string(s) + "-t" + std::to_string(t) + "-j" + job8, g_dp);

            // TP allreduce: sum(u64(rank_in_group+1)).
            auto allreduce_check = [&](uvcc::NcclCommV1* c, const char* label) {
                uvcc::u64* d = nullptr;
                const cudaError_t a = cudaMalloc(&d, sizeof(uvcc::u64));
                if (a != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(a));
                const uvcc::u64 x = static_cast<uvcc::u64>(c->my_rank_in_group + 1);
                const cudaError_t b = cudaMemcpy(d, &x, sizeof(uvcc::u64), cudaMemcpyHostToDevice);
                if (b != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(b));
                uvcc::DeviceBufferV1 buf;
                buf.ptr = d;
                buf.n_bytes = sizeof(uvcc::u64);
                auto st = uvcc::nccl_allreduce_sum_v1(c, buf);
                if (!st.ok()) throw std::runtime_error(std::string(label) + ": " + st.message());
                uvcc::u64 y = 0;
                const cudaError_t cpy = cudaMemcpy(&y, d, sizeof(uvcc::u64), cudaMemcpyDeviceToHost);
                if (cpy != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(cpy));
                cudaFree(d);
                const uvcc::u64 want = static_cast<uvcc::u64>(c->n_ranks) * static_cast<uvcc::u64>(c->n_ranks + 1) / 2;
                if (y != want) throw std::runtime_error(std::string(label) + ": allreduce mismatch got=" + std::to_string(y) + " want=" + std::to_string(want));
            };

            allreduce_check(&tp_comm, "tp_allreduce");
            allreduce_check(&dp_comm, "dp_allreduce");

            // PP ping-pong (only if S==2).
            if (topo.stages == 2) {
                uvcc::u64* d_send = nullptr;
                uvcc::u64* d_recv = nullptr;
                if (cudaMalloc(&d_send, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc send failed");
                if (cudaMalloc(&d_recv, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc recv failed");
                const uvcc::u64 v_send = (pp_comm.my_rank_in_group == 0) ? 1111ULL : 2222ULL;
                {
                    const cudaError_t c0 = cudaMemcpy(d_send, &v_send, sizeof(uvcc::u64), cudaMemcpyHostToDevice);
                    if (c0 != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(c0));
                }
                uvcc::DeviceBufferV1 bs{d_send, sizeof(uvcc::u64)};
                uvcc::DeviceBufferV1 br{d_recv, sizeof(uvcc::u64)};
                // Two-rank ping-pong using ncclSend/ncclRecv in a group.
                {
                    const ncclResult_t grc = ncclGroupStart();
                    if (grc != ncclSuccess) throw std::runtime_error(std::string("ncclGroupStart failed: ") + ncclGetErrorString(grc));
                }
                if (pp_comm.my_rank_in_group == 0) {
                    auto st0 = uvcc::nccl_send_v1(&pp_comm, bs, 1);
                    if (!st0.ok()) throw std::runtime_error(std::string("pp send failed: ") + st0.message());
                    auto st1 = uvcc::nccl_recv_v1(&pp_comm, br, 1);
                    if (!st1.ok()) throw std::runtime_error(std::string("pp recv failed: ") + st1.message());
                } else {
                    auto st0 = uvcc::nccl_recv_v1(&pp_comm, br, 0);
                    if (!st0.ok()) throw std::runtime_error(std::string("pp recv failed: ") + st0.message());
                    auto st1 = uvcc::nccl_send_v1(&pp_comm, bs, 0);
                    if (!st1.ok()) throw std::runtime_error(std::string("pp send failed: ") + st1.message());
                }
                {
                    const ncclResult_t grc = ncclGroupEnd();
                    if (grc != ncclSuccess) throw std::runtime_error(std::string("ncclGroupEnd failed: ") + ncclGetErrorString(grc));
                }
                {
                    const cudaError_t sc = cudaStreamSynchronize(pp_comm.stream);
                    if (sc != cudaSuccess) throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(sc));
                }
                uvcc::u64 v_out = 0;
                {
                    const cudaError_t c0 = cudaMemcpy(&v_out, d_recv, sizeof(uvcc::u64), cudaMemcpyDeviceToHost);
                    if (c0 != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(c0));
                }
                cudaFree(d_send);
                cudaFree(d_recv);
                const uvcc::u64 want = (pp_comm.my_rank_in_group == 0) ? 2222ULL : 1111ULL;
                if (v_out != want) throw std::runtime_error("pp_sendrecv mismatch");
            }

            // Cleanup.
            (void)uvcc::nccl_destroy_v1(&tp_comm);
            (void)uvcc::nccl_destroy_v1(&pp_comm);
            (void)uvcc::nccl_destroy_v1(&dp_comm);

            std::cout << "nccl_smoke_ok coord=p" << int(coord.party) << " r" << coord.replica << " s" << coord.stage << " t" << coord.tp_rank
                      << "\n";
            return 0;
#endif
        }

        if (mode == "phase6_step") {
#ifndef UVCC_WITH_CUDA_NCCL
            throw std::runtime_error("phase6_step requires UVCC_WITH_CUDA_NCCL (build with -DUVCC_WITH_CUDA_NCCL=ON)");
#else
            // This mode is a Phase-6 bring-up that combines:
            // - cross-party toy OPENs (transport + transcript)
            // - intra-party NCCL TP/PP/DP communication
            //
            // It is intentionally minimal: we do not run real SGIR/Transformer kernels yet,
            // but we exercise the scheduling + networking skeleton end-to-end on real GPUs.

            const std::string relay_url = get_arg(argc, argv, "--relay-url");
            if (relay_url.empty()) throw std::runtime_error("missing --relay-url for --mode phase6_step");
            const std::string relay_group_base = get_arg(argc, argv, "--relay-group-id");
            if (relay_group_base.empty()) throw std::runtime_error("missing --relay-group-id for --mode phase6_step");
            std::string token = get_arg(argc, argv, "--relay-token", "");
            const std::string token_file = get_arg(argc, argv, "--relay-token-file", "");
            if (token.empty() && !token_file.empty()) token = read_text_file(token_file);
            const std::string tls_ca = get_arg(argc, argv, "--tls-ca-pem", "");
            const int ttl_s = static_cast<int>(std::stol(get_arg(argc, argv, "--relay-ttl-s", "3600")));
            const double timeout_s = std::stod(get_arg(argc, argv, "--relay-timeout-s", "10.0"));
            const bool skip_dp = has_flag(argc, argv, "--phase6-skip-dp");

            // Base relay config (group_id overridden per use).
            uvcc::RelayHttpConfigV1 base_cfg;
            base_cfg.base_url = relay_url;
            base_cfg.group_id = relay_group_base;
            base_cfg.token = token;
            base_cfg.tls_ca_pem_path = tls_ca;
            base_cfg.timeout_s = timeout_s;
            base_cfg.default_ttl_s = ttl_s;

            // Relay health check once.
            {
                auto http0 = std::make_shared<uvcc::RelayHttpClientV1>(base_cfg);
                std::string last_hz_err;
                bool hz_ok = false;
                for (int i = 0; i < 50; i++) {
                    const auto hz = http0->healthz();
                    if (hz.ok()) {
                        hz_ok = true;
                        break;
                    }
                    last_hz_err = hz.message();
                    sleep_ms(200);
                }
                if (!hz_ok) throw std::runtime_error(std::string("relay healthz failed: ") + last_hz_err);
            }

            const uvcc::TopologyV1 topo = cfg.topo;
            const uvcc::CoordV1 coord = cfg.coord;
            const uvcc::u32 my_rank_in_party = uvcc::local_rank_v1(topo, static_cast<uvcc::u32>(coord.replica), coord.stage, coord.tp_rank);
            const int phase6_timeout_s = static_cast<int>(std::stoul(get_arg(argc, argv, "--phase6-timeout-s", "120")));

            // NCCL init helper (same as nccl_smoke).
            auto init_group = [&](const std::string& gid, const uvcc::NcclGroupV1& g) -> uvcc::NcclCommV1 {
                auto cr = uvcc::create_nccl_comm_v1(g, my_rank_in_party);
                if (!cr.ok()) throw std::runtime_error(cr.status().message());
                uvcc::NcclCommV1 c = cr.value();
                if (!c.enabled) throw std::runtime_error("NCCL not enabled");
                std::cout << "phase6_init_group gid=" << gid << " kind=" << int(g.kind) << " my_rank_in_group=" << c.my_rank_in_group
                          << " n_ranks=" << c.n_ranks << "\n"
                          << std::flush;

                uvcc::RelayHttpConfigV1 cfg2 = base_cfg;
                cfg2.group_id = gid;
                auto http = std::make_shared<uvcc::RelayHttpClientV1>(cfg2);

                uvcc::NcclUniqueIdV1 uid{};
                if (c.my_rank_in_group == 0) {
                    std::cout << "phase6_uid_gen gid=" << gid << "\n" << std::flush;
                    auto ur = uvcc::nccl_get_unique_id_v1();
                    if (!ur.ok()) throw std::runtime_error(ur.status().message());
                    uid = ur.value();
                    for (uvcc::u32 dst = 1; dst < c.n_ranks; dst++) {
                        std::cout << "phase6_uid_enqueue gid=" << gid << " dst=" << dst << "\n" << std::flush;
                        std::vector<uvcc::u8> payload(uid.begin(), uid.end());
                        const std::string msg_id = std::string("uid-") + std::to_string(dst);
                        auto st = http->enqueue(/*sender=*/0, /*receiver=*/static_cast<int>(dst), msg_id, payload, ttl_s);
                        if (!st.ok()) throw std::runtime_error(st.message());
                    }
                    std::cout << "phase6_uid_enqueue_done gid=" << gid << "\n" << std::flush;
                } else {
                    std::cout << "phase6_uid_wait gid=" << gid << " rank=" << c.my_rank_in_group << "\n" << std::flush;
                    // Important at scale: rank0 (the uid generator) can be delayed relative to other ranks
                    // due to oversubscription and PP/OPEN skew. Use the Phase-6 timeout (typically minutes)
                    // rather than a hard-coded 60s, otherwise non-zero ranks can time out *before* uid is enqueued.
                    const double deadline =
                        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() + double(std::max(60, phase6_timeout_s));
                    while (true) {
                        const double now_s = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
                        // Long-poll up to 5s at a time to reduce relay load.
                        const double dl = std::min(deadline, now_s + 5.0);
                        auto pr = http->poll(static_cast<int>(c.my_rank_in_group), dl);
                        if (!pr.ok()) throw std::runtime_error(pr.status().message());
                        if (!pr.value().has_value()) {
                            if (std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() > deadline) {
                                throw std::runtime_error("timeout waiting for nccl unique id");
                            }
                            continue;
                        }
                        auto m = std::move(*pr.value());
                        std::cout << "phase6_uid_recv gid=" << gid << " msg_id=" << m.msg_id << " payload=" << m.payload.size() << "\n" << std::flush;
                        auto st = http->ack(static_cast<int>(c.my_rank_in_group), m.msg_id, m.lease_token);
                        if (!st.ok()) throw std::runtime_error(st.message());
                        std::cout << "phase6_uid_acked gid=" << gid << " msg_id=" << m.msg_id << "\n" << std::flush;
                        if (m.payload.size() != uvcc::NCCL_UNIQUE_ID_BYTES_V1) throw std::runtime_error("bad nccl uid payload size");
                        std::copy(m.payload.begin(), m.payload.end(), uid.begin());
                        break;
                    }
                }

                std::cout << "phase6_nccl_init_start gid=" << gid << "\n" << std::flush;
                auto st = uvcc::nccl_init_rank_v1(&c, uid);
                if (!st.ok()) throw std::runtime_error(st.message());
                std::cout << "phase6_nccl_init_done gid=" << gid << "\n" << std::flush;
                return c;
            };

            const std::string job8 = uvcc::hex_lower(sub.sid_job).substr(0, 8);
            const int party = static_cast<int>(coord.party);
            const int r = static_cast<int>(coord.replica);
            const int s = static_cast<int>(coord.stage);
            const int t = static_cast<int>(coord.tp_rank);
            std::cout << "phase6_begin coord=p" << party << " r" << r << " s" << s << " t" << t << " job8=" << job8 << " skip_dp=" << (skip_dp ? 1 : 0)
                      << "\n"
                      << std::flush;

            const auto g_tp = uvcc::make_tp_group_v1(topo, static_cast<uvcc::u32>(coord.replica), coord.stage);
            const auto g_pp = uvcc::make_pp_group_v1(topo, static_cast<uvcc::u32>(coord.replica), coord.tp_rank);
            const auto g_dp = uvcc::make_dp_group_v1(topo, coord.stage, coord.tp_rank);

            auto tp_comm = init_group(relay_group_base + "-nccl-tp-p" + std::to_string(party) + "-r" + std::to_string(r) + "-s" + std::to_string(s) + "-j" + job8, g_tp);
            std::cout << "phase6_tp_ready\n" << std::flush;
            auto pp_comm = init_group(relay_group_base + "-nccl-pp-p" + std::to_string(party) + "-r" + std::to_string(r) + "-t" + std::to_string(t) + "-j" + job8, g_pp);
            std::cout << "phase6_pp_ready\n" << std::flush;
            // NOTE: We intentionally defer DP NCCL init until after backward.
            //
            // Motivation:
            // - Cross-party OPEN depends on all three parties reaching the OPEN loop.
            // - Large oversubscribed runs (R=8,S=4,T=2) spend a long time in intra-party NCCL init,
            //   and skew across parties can cause OPEN timeouts even though nothing is "wrong".
            // - DP reduction (in real training) happens after backward anyway.
            //
            // So we bring up TP+PP first, run forward/backward OPEN, and then init DP + run a single DP allreduce check.
            std::optional<uvcc::NcclCommV1> dp_comm;

            // Cross-party OPEN transport uses a per-(r,s,t) group id so parties don't interfere.
            const std::string open_group_id =
                std::string("g-native-sub-r") + std::to_string(r) + "-s" + std::to_string(s) + "-t" + std::to_string(t) + "-j" + job8;

            uvcc::RelayHttpConfigV1 open_cfg = base_cfg;
            open_cfg.group_id = open_group_id;
            auto http_open = std::make_shared<uvcc::RelayHttpClientV1>(open_cfg);
            auto mailbox = std::make_shared<uvcc::RelayMailboxV1>(http_open, open_group_id, int(sub.coord.party));
            auto to_prev = std::make_unique<uvcc::RelayRawConnV1>(
                http_open, mailbox, open_group_id, int(sub.coord.party), int((int(sub.coord.party) + 2) % 3), ttl_s);
            auto to_next = std::make_unique<uvcc::RelayRawConnV1>(
                http_open, mailbox, open_group_id, int(sub.coord.party), int((int(sub.coord.party) + 1) % 3), ttl_s);

            uvcc::TranscriptStoreV1 ts(sub.sid_sub);
            uvcc::OpenEngineV1 open(sub.sid_sub, sub.coord.party);
            uvcc::TransportCallbacksV1 cbs = uvcc::make_lift_transcript_callbacks_v1(&ts, sub.sid_sub);
            cbs.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& payload) { open.on_deliver(hdr, payload); };
            uvcc::TransportV1 transport(sub.sid_sub, sub.coord.party, to_prev.get(), to_next.get(), cbs, nullptr);

            const uvcc::u32 step_id = static_cast<uvcc::u32>(std::stoul(get_arg(argc, argv, "--step-id", "0")));
            const uvcc::u16 M = static_cast<uvcc::u16>(std::stoul(get_arg(argc, argv, "--microbatches", "1")));
            const uvcc::u16 S = topo.stages;

            // NOTE: The initial Phase-6 bring-up used a per-(p,r,s,t) scheduler loop plus
            // pre-posted PP recvs for both forward activations and backward gradients.
            // That deadlocks on a single CUDA stream because backward recvs enqueued early
            // block later forward sends in stream order. It also risks TP collective ordering
            // divergence across tp ranks.
            //
            // For v1 correctness/determinism, we run a simple lockstep schedule:
            // - Post *only* activation recvs upfront (stage>0).
            // - Run all forwards mb=0..M-1 (OPEN + TP allreduce + optional PP send act).
            // - Then post grad recvs (stage<S-1), run all backwards mb=0..M-1 (wait grad + OPEN + TP allreduce + optional PP send grad).
            //
            // This still exercises: relay transport, transcript, TP allreduce, PP send/recv, and deterministic IDs.

            auto enqueue_open = [&](uvcc::u8 phase_u8, uvcc::u16 mb) {
                const uvcc::u32 op_id32 = uvcc::derive_sgir_op_id32_v1(sub.sid_sub, step_id, phase_u8, mb, /*k=*/0);
                const uvcc::u64 stream_id64 = 0x2222000000000000ULL ^ (static_cast<uvcc::u64>(phase_u8) << 32) ^ static_cast<uvcc::u64>(mb);
                const uvcc::u64 base = hash64_open_base(sub.sid_sub, step_id, phase_u8, mb);
                std::vector<uvcc::u64> lo, hi;
                toy_open_vectors_for_party(static_cast<uvcc::u8>(coord.party), base, &lo, &hi);
                uvcc::FrameV1 f;
                open.enqueue_open_u64(op_id32, /*epoch_id32=*/step_id, stream_id64, lo, hi, &f);
                transport.send_frame_reliable(std::move(f));
            };

            auto wait_open_done = [&](uvcc::u8 phase_u8, uvcc::u16 mb) {
                const uvcc::u32 op_id32 = uvcc::derive_sgir_op_id32_v1(sub.sid_sub, step_id, phase_u8, mb, /*k=*/0);
                const double deadline =
                    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() + double(std::max(1, phase6_timeout_s));
                while (!open.is_done(op_id32)) {
                    transport.poll();
                    if (std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() > deadline) {
                        throw std::runtime_error("timeout waiting for OPEN completion");
                    }
                    sleep_ms(1);
                }
                (void)open.take_result_u64(op_id32);
            };

            auto wait_cuda_event = [&](cudaEvent_t ev, const char* what) {
                if (ev == nullptr) throw std::runtime_error(std::string("wait_cuda_event null: ") + what);
                const double deadline =
                    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() + double(std::max(1, phase6_timeout_s));
                while (true) {
                    const cudaError_t q = cudaEventQuery(ev);
                    if (q == cudaSuccess) return;
                    if (q != cudaErrorNotReady) throw std::runtime_error(std::string("cudaEventQuery failed: ") + cudaGetErrorString(q));
                    if (std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() > deadline) {
                        throw std::runtime_error(std::string("timeout waiting for PP event: ") + what);
                    }
                    transport.poll();
                    sleep_ms(1);
                }
            };

            // TP allreduce check using pinned host memory and the TP stream (avoid default-stream global sync).
            uvcc::u64* d_tp = nullptr;
            if (cudaMalloc(&d_tp, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc d_tp failed");
            uvcc::DeviceBufferV1 tp_buf{d_tp, sizeof(uvcc::u64)};
            uvcc::u64* h_tp = nullptr;
            if (cudaMallocHost(&h_tp, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMallocHost h_tp failed");
            cudaEvent_t tp_done_ev = nullptr;
            if (cudaEventCreateWithFlags(&tp_done_ev, cudaEventDisableTiming) != cudaSuccess) throw std::runtime_error("cudaEventCreate tp_done_ev failed");

            auto tp_allreduce_barrier = [&](const char* label) {
                // Use value (rank+1) and check sum == n(n+1)/2 to validate TP collective correctness.
                *h_tp = static_cast<uvcc::u64>(tp_comm.my_rank_in_group + 1);
                const cudaError_t h2d = cudaMemcpyAsync(d_tp, h_tp, sizeof(uvcc::u64), cudaMemcpyHostToDevice, tp_comm.stream);
                if (h2d != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync H2D failed: ") + cudaGetErrorString(h2d));
                auto st = uvcc::nccl_allreduce_sum_async_v1(&tp_comm, tp_buf);
                if (!st.ok()) throw std::runtime_error(std::string(label) + ": " + st.message());
                const cudaError_t d2h = cudaMemcpyAsync(h_tp, d_tp, sizeof(uvcc::u64), cudaMemcpyDeviceToHost, tp_comm.stream);
                if (d2h != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync D2H failed: ") + cudaGetErrorString(d2h));
                if (cudaEventRecord(tp_done_ev, tp_comm.stream) != cudaSuccess) throw std::runtime_error("cudaEventRecord(tp_done_ev) failed");
                // IMPORTANT: poll transport while waiting for NCCL to complete so we can ACK/receive OPEN traffic.
                wait_cuda_event(tp_done_ev, label);
                const uvcc::u64 out = *h_tp;
                const uvcc::u64 want = static_cast<uvcc::u64>(tp_comm.n_ranks) * static_cast<uvcc::u64>(tp_comm.n_ranks + 1) / 2;
                if (out != want) throw std::runtime_error(std::string(label) + ": allreduce mismatch got=" + std::to_string(out) + " want=" + std::to_string(want));
            };

            // DP is initialized + executed after backward (see below). Keep these declared for cleanup.
            uvcc::u64* d_dp = nullptr;
            uvcc::u64* h_dp = nullptr;

            // PP bookkeeping (separate recv vs send buffers so intermediate stages won't overwrite pointers).
            std::vector<uvcc::u64*> act_recv_buf(static_cast<std::size_t>(M), nullptr);
            std::vector<cudaEvent_t> act_recv_ev(static_cast<std::size_t>(M), nullptr);
            std::vector<uvcc::u64*> act_send_buf(static_cast<std::size_t>(M), nullptr);

            std::vector<uvcc::u64*> grad_recv_buf(static_cast<std::size_t>(M), nullptr);
            std::vector<cudaEvent_t> grad_recv_ev(static_cast<std::size_t>(M), nullptr);
            std::vector<uvcc::u64*> grad_send_buf(static_cast<std::size_t>(M), nullptr);
            cudaEvent_t pp_sync_ev = nullptr;
            if (cudaEventCreateWithFlags(&pp_sync_ev, cudaEventDisableTiming) != cudaSuccess) throw std::runtime_error("cudaEventCreate pp_sync_ev failed");

            auto post_act_recv = [&](uvcc::u16 mb) {
                const std::size_t i = static_cast<std::size_t>(mb);
                if (act_recv_buf[i] != nullptr || act_recv_ev[i] != nullptr) throw std::runtime_error("duplicate act recv post");
                if (cudaMalloc(&act_recv_buf[i], sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc act failed");
                if (cudaEventCreateWithFlags(&act_recv_ev[i], cudaEventDisableTiming) != cudaSuccess) throw std::runtime_error("cudaEventCreate act failed");
                uvcc::DeviceBufferV1 br{act_recv_buf[i], sizeof(uvcc::u64)};
                auto st = uvcc::nccl_recv_v1(&pp_comm, br, /*peer_rank=*/static_cast<uvcc::u32>(coord.stage - 1));
                if (!st.ok()) throw std::runtime_error(std::string("pp recv act failed: ") + st.message());
                if (cudaEventRecord(act_recv_ev[i], pp_comm.stream) != cudaSuccess) throw std::runtime_error("cudaEventRecord act failed");
            };

            // Post activation recv for mb=0 only (stage>0); post subsequent mb as we advance.
            // This avoids enqueueing all recvs up front, which can block forward PP sends in CUDA stream order.
            if (coord.stage > 0 && M > 0) {
                post_act_recv(0);
            }

            // Forward pass: mb=0..M-1
            for (uvcc::u16 mb = 0; mb < M; mb++) {
                const std::size_t i = static_cast<std::size_t>(mb);
                if (coord.stage > 0) {
                    wait_cuda_event(act_recv_ev[i], "act_recv");
                }
                std::cout << "phase6_fwd mb=" << mb << "\n" << std::flush;
                enqueue_open(/*phase_u8=*/0, mb);
                tp_allreduce_barrier("tp_allreduce_fwd");
                std::cout << "phase6_fwd_tp_ok mb=" << mb << "\n" << std::flush;
                wait_open_done(/*phase_u8=*/0, mb);
                std::cout << "phase6_fwd_open_ok mb=" << mb << "\n" << std::flush;

                // Send activation to next stage (if not last stage).
                if (coord.stage + 1 < S) {
                    const uvcc::u64 act = 0xA100000000000000ULL ^ (static_cast<uvcc::u64>(r) << 32) ^ (static_cast<uvcc::u64>(mb) << 8) ^
                                         static_cast<uvcc::u64>(t);
                    // Use a per-microbatch send buffer so we never reuse in-flight device memory.
                    uvcc::u64* d_send = nullptr;
                    if (cudaMalloc(&d_send, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc act_send failed");
                    act_send_buf[i] = d_send;
                    const cudaError_t h2d = cudaMemcpyAsync(act_send_buf[i], &act, sizeof(uvcc::u64), cudaMemcpyHostToDevice, pp_comm.stream);
                    if (h2d != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync act H2D failed: ") + cudaGetErrorString(h2d));
                    uvcc::DeviceBufferV1 bs{act_send_buf[i], sizeof(uvcc::u64)};
                    auto st = uvcc::nccl_send_v1(&pp_comm, bs, /*peer_rank=*/static_cast<uvcc::u32>(coord.stage + 1));
                    if (!st.ok()) throw std::runtime_error(std::string("pp send act failed: ") + st.message());
                }

                // Post next activation recv (stage>0) for mb+1.
                if (coord.stage > 0 && (mb + 1) < M) {
                    post_act_recv(static_cast<uvcc::u16>(mb + 1));
                }
            }

            // Ensure all forward PP ops are complete before posting backward grad recvs on the same stream.
            {
                if (cudaEventRecord(pp_sync_ev, pp_comm.stream) != cudaSuccess) throw std::runtime_error("cudaEventRecord(pp_sync after fwd) failed");
                wait_cuda_event(pp_sync_ev, "pp_after_fwd");
            }

            auto post_grad_recv = [&](uvcc::u16 mb) {
                const std::size_t i = static_cast<std::size_t>(mb);
                if (grad_recv_buf[i] != nullptr || grad_recv_ev[i] != nullptr) throw std::runtime_error("duplicate grad recv post");
                if (cudaMalloc(&grad_recv_buf[i], sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc grad failed");
                if (cudaEventCreateWithFlags(&grad_recv_ev[i], cudaEventDisableTiming) != cudaSuccess) throw std::runtime_error("cudaEventCreate grad failed");
                uvcc::DeviceBufferV1 br{grad_recv_buf[i], sizeof(uvcc::u64)};
                auto st = uvcc::nccl_recv_v1(&pp_comm, br, /*peer_rank=*/static_cast<uvcc::u32>(coord.stage + 1));
                if (!st.ok()) throw std::runtime_error(std::string("pp recv grad failed: ") + st.message());
                if (cudaEventRecord(grad_recv_ev[i], pp_comm.stream) != cudaSuccess) throw std::runtime_error("cudaEventRecord grad failed");
            };

            // Post gradient recv for mb=0 only (stage<S-1); post subsequent mb as we advance.
            // This prevents a subtle CUDA stream ordering issue where early gradient sends can be queued
            // behind later gradient recvs (starving stage0 of grads and causing grad_recv timeouts).
            if (coord.stage + 1 < S && M > 0) {
                post_grad_recv(0);
            }

            // Backward pass: mb=0..M-1
            for (uvcc::u16 mb = 0; mb < M; mb++) {
                const std::size_t i = static_cast<std::size_t>(mb);
                if (coord.stage + 1 < S) {
                    wait_cuda_event(grad_recv_ev[i], "grad_recv");
                }
                std::cout << "phase6_bwd mb=" << mb << "\n" << std::flush;
                enqueue_open(/*phase_u8=*/1, mb);
                tp_allreduce_barrier("tp_allreduce_bwd");
                std::cout << "phase6_bwd_tp_ok mb=" << mb << "\n" << std::flush;
                wait_open_done(/*phase_u8=*/1, mb);
                std::cout << "phase6_bwd_open_ok mb=" << mb << "\n" << std::flush;

                // Send grad to prev stage (if not first stage).
                if (coord.stage > 0) {
                    const uvcc::u64 g = 0xB200000000000000ULL ^ (static_cast<uvcc::u64>(r) << 32) ^ (static_cast<uvcc::u64>(mb) << 8) ^
                                        static_cast<uvcc::u64>(t);
                    uvcc::u64* d_send = nullptr;
                    if (cudaMalloc(&d_send, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc grad_send failed");
                    grad_send_buf[i] = d_send;
                    const cudaError_t h2d = cudaMemcpyAsync(grad_send_buf[i], &g, sizeof(uvcc::u64), cudaMemcpyHostToDevice, pp_comm.stream);
                    if (h2d != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync grad H2D failed: ") + cudaGetErrorString(h2d));
                    uvcc::DeviceBufferV1 bs{grad_send_buf[i], sizeof(uvcc::u64)};
                    auto st = uvcc::nccl_send_v1(&pp_comm, bs, /*peer_rank=*/static_cast<uvcc::u32>(coord.stage - 1));
                    if (!st.ok()) throw std::runtime_error(std::string("pp send grad failed: ") + st.message());
                }

                // Post next gradient recv (stage<S-1) for mb+1.
                if (coord.stage + 1 < S && (mb + 1) < M) {
                    post_grad_recv(static_cast<uvcc::u16>(mb + 1));
                }
            }

            // DP reduction in a real training step happens after backward (reduce gradients across replicas).
            // Bring up DP only now (after OPEN path has run), then do a single DP allreduce sanity check.
            if (!skip_dp) {
                dp_comm = init_group(relay_group_base + "-nccl-dp-p" + std::to_string(party) + "-s" + std::to_string(s) + "-t" + std::to_string(t) + "-j" + job8, g_dp);
                std::cout << "phase6_dp_ready\n" << std::flush;
                if (cudaMalloc(&d_dp, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMalloc d_dp failed");
                if (cudaMallocHost(&h_dp, sizeof(uvcc::u64)) != cudaSuccess) throw std::runtime_error("cudaMallocHost h_dp failed");
                uvcc::DeviceBufferV1 dp_buf{d_dp, sizeof(uvcc::u64)};
                cudaEvent_t dp_done_ev = nullptr;
                if (cudaEventCreateWithFlags(&dp_done_ev, cudaEventDisableTiming) != cudaSuccess) throw std::runtime_error("cudaEventCreate dp_done_ev failed");

                auto dp_allreduce_barrier = [&](const char* label) {
                    // Use value (rank+1) and check sum == n(n+1)/2 to validate DP collective correctness.
                    *h_dp = static_cast<uvcc::u64>((*dp_comm).my_rank_in_group + 1);
                    const cudaError_t h2d = cudaMemcpyAsync(d_dp, h_dp, sizeof(uvcc::u64), cudaMemcpyHostToDevice, (*dp_comm).stream);
                    if (h2d != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(dp) H2D failed: ") + cudaGetErrorString(h2d));
                    auto st = uvcc::nccl_allreduce_sum_async_v1(&(*dp_comm), dp_buf);
                    if (!st.ok()) throw std::runtime_error(std::string(label) + ": " + st.message());
                    const cudaError_t d2h = cudaMemcpyAsync(h_dp, d_dp, sizeof(uvcc::u64), cudaMemcpyDeviceToHost, (*dp_comm).stream);
                    if (d2h != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(dp) D2H failed: ") + cudaGetErrorString(d2h));
                    if (cudaEventRecord(dp_done_ev, (*dp_comm).stream) != cudaSuccess) throw std::runtime_error("cudaEventRecord(dp_done_ev) failed");
                    wait_cuda_event(dp_done_ev, label);
                    const uvcc::u64 out = *h_dp;
                    const uvcc::u64 want = static_cast<uvcc::u64>((*dp_comm).n_ranks) * static_cast<uvcc::u64>((*dp_comm).n_ranks + 1) / 2;
                    if (out != want) throw std::runtime_error(std::string(label) + ": allreduce mismatch got=" + std::to_string(out) + " want=" + std::to_string(want));
                };

                dp_allreduce_barrier("dp_allreduce_after_bwd");
                std::cout << "phase6_dp_after_bwd_ok\n" << std::flush;
                if (dp_done_ev) cudaEventDestroy(dp_done_ev);
            }

            // Drain some acks so transcript is stable.
            for (int i = 0; i < 200; i++) transport.poll();

            // Ensure all PP operations complete before freeing device buffers.
            {
                if (cudaEventRecord(pp_sync_ev, pp_comm.stream) != cudaSuccess) throw std::runtime_error("cudaEventRecord(pp_sync end) failed");
                wait_cuda_event(pp_sync_ev, "pp_end");
            }

            const auto root = ts.epoch_root(step_id);
            std::cout << "epoch_root=0x" << uvcc::hex_lower(root) << "\n";

            // Cleanup.
            if (h_tp) cudaFreeHost(h_tp);
            if (d_tp) cudaFree(d_tp);
            if (tp_done_ev) cudaEventDestroy(tp_done_ev);
            if (h_dp) cudaFreeHost(h_dp);
            if (d_dp) cudaFree(d_dp);
            if (pp_sync_ev) cudaEventDestroy(pp_sync_ev);
            for (std::size_t i = 0; i < act_recv_ev.size(); i++) {
                if (act_recv_ev[i]) cudaEventDestroy(act_recv_ev[i]);
            }
            for (std::size_t i = 0; i < grad_recv_ev.size(); i++) {
                if (grad_recv_ev[i]) cudaEventDestroy(grad_recv_ev[i]);
            }
            for (std::size_t i = 0; i < act_recv_buf.size(); i++) {
                if (act_recv_buf[i]) cudaFree(act_recv_buf[i]);
            }
            for (std::size_t i = 0; i < grad_recv_buf.size(); i++) {
                if (grad_recv_buf[i]) cudaFree(grad_recv_buf[i]);
            }
            for (std::size_t i = 0; i < act_send_buf.size(); i++) {
                if (act_send_buf[i]) cudaFree(act_send_buf[i]);
            }
            for (std::size_t i = 0; i < grad_send_buf.size(); i++) {
                if (grad_send_buf[i]) cudaFree(grad_send_buf[i]);
            }
            (void)uvcc::nccl_destroy_v1(&tp_comm);
            (void)uvcc::nccl_destroy_v1(&pp_comm);
            if (!skip_dp && dp_comm.has_value()) {
                (void)uvcc::nccl_destroy_v1(&(*dp_comm));
            }
            return 0;
#endif
        }

        throw std::runtime_error("unknown --mode " + mode);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}


