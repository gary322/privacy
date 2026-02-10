#include "uvcc/audit.h"
#include "uvcc/open.h"
#include "uvcc/runtime.h"
#include "uvcc/subsession.h"
#include "uvcc/testkit.h"
#include "uvcc/transcript_hooks.h"

#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

struct FakeClock final : public uvcc::ClockV1 {
    std::uint64_t t_ms = 0;
    std::uint64_t now_ms() override { return t_ms; }
};

struct MemDuplex {
    std::deque<std::vector<uvcc::u8>> a_to_b;
    std::deque<std::vector<uvcc::u8>> b_to_a;
};

struct MemEndpoint final : public uvcc::RawConn {
    MemDuplex* d = nullptr;
    bool is_a = true;

    void send_bytes(const std::vector<uvcc::u8>& bytes) override {
        if (!d) return;
        if (is_a)
            d->a_to_b.push_back(bytes);
        else
            d->b_to_a.push_back(bytes);
    }

    bool poll_recv(std::vector<uvcc::u8>* out) override {
        if (!d || !out) return false;
        auto& q = is_a ? d->b_to_a : d->a_to_b;
        if (q.empty()) return false;
        *out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

struct SubgroupHarness {
    // One subgroup (r,s,t) has its own isolated 3-party raw links.
    MemDuplex d01;
    MemDuplex d12;
    MemDuplex d20;  // a=2, b=0

    // Endpoints for party0
    MemEndpoint p0_to1;
    MemEndpoint p0_to2;
    // party1
    MemEndpoint p1_to0;
    MemEndpoint p1_to2;
    // party2
    MemEndpoint p2_to1;
    MemEndpoint p2_to0;

    SubgroupHarness() {
        p0_to1.d = &d01;
        p0_to1.is_a = true;
        p1_to0.d = &d01;
        p1_to0.is_a = false;

        p1_to2.d = &d12;
        p1_to2.is_a = true;
        p2_to1.d = &d12;
        p2_to1.is_a = false;

        p2_to0.d = &d20;
        p2_to0.is_a = true;
        p0_to2.d = &d20;
        p0_to2.is_a = false;
    }
};

struct PartyWorker {
    uvcc::TranscriptStoreV1 ts;
    uvcc::OpenEngineV1 open;
    uvcc::TransportCallbacksV1 cbs;
    std::unique_ptr<uvcc::TransportV1> transport;
    uvcc::WorkerRuntimeV1 worker;

    PartyWorker(const uvcc::SubsessionV1& sub, uvcc::RawConn* to_prev, uvcc::RawConn* to_next, FakeClock* clk)
        : ts(sub.sid_sub),
          open(sub.sid_sub, sub.coord.party),
          cbs(uvcc::make_lift_transcript_callbacks_v1(&ts, sub.sid_sub)),
          transport(nullptr),
          worker() {
        cbs.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& payload) { open.on_deliver(hdr, payload); };
        transport = std::make_unique<uvcc::TransportV1>(sub.sid_sub, sub.coord.party, to_prev, to_next, cbs, clk);

        worker.cfg.topo = sub.topo;
        worker.cfg.coord = sub.coord;
        worker.cfg.sid_job = sub.sid_job;
        worker.coord = sub.coord;
        worker.sid_sub = sub.sid_sub;
        worker.transport = transport.get();
        worker.transcript = &ts;
        worker.open = &open;
        worker.step_id = 0;
        worker.microbatches = 1;
        worker.pp_sched = uvcc::PPSchedulerV1(/*S=*/1, /*M=*/1, /*stage_id=*/0, /*kmax_fwd=*/1, /*kmax_bwd=*/1);
    }
};

static uvcc::Hash32 run_matrix_once() {
    const uvcc::Sid32 sid_job = uvcc::sid32_seq_00_1f();
    uvcc::TopologyV1 topo;
    topo.replicas = 2;
    topo.stages = 2;
    topo.tp_ranks = 2;

    FakeClock clk;

    // Build all subgroups and all 3-party workers.
    std::vector<std::unique_ptr<SubgroupHarness>> nets;
    std::vector<std::unique_ptr<PartyWorker>> workers;
    nets.reserve(static_cast<std::size_t>(topo.replicas) * topo.stages * topo.tp_ranks);
    workers.reserve(static_cast<std::size_t>(3) * topo.replicas * topo.stages * topo.tp_ranks);

    for (uvcc::u32 r = 0; r < topo.replicas; r++) {
        for (uvcc::u16 s = 0; s < topo.stages; s++) {
            for (uvcc::u16 t = 0; t < topo.tp_ranks; t++) {
                auto net = std::make_unique<SubgroupHarness>();
                // Party0 coord
                uvcc::CoordV1 c0;
                c0.party = 0;
                c0.replica = r;
                c0.stage = s;
                c0.tp_rank = t;
                uvcc::CoordV1 c1 = c0;
                c1.party = 1;
                uvcc::CoordV1 c2 = c0;
                c2.party = 2;

                const auto sub0 = uvcc::make_subsession_v1(topo, c0, sid_job);
                const auto sub1 = uvcc::make_subsession_v1(topo, c1, sid_job);
                const auto sub2 = uvcc::make_subsession_v1(topo, c2, sid_job);

                // Wire ring:
                // p0: prev=2 (to_prev = p0_to2), next=1 (to_next=p0_to1)
                // p1: prev=0 (p1_to0), next=2 (p1_to2)
                // p2: prev=1 (p2_to1), next=0 (p2_to0)
                auto w0 = std::make_unique<PartyWorker>(sub0, &net->p0_to2, &net->p0_to1, &clk);
                auto w1 = std::make_unique<PartyWorker>(sub1, &net->p1_to0, &net->p1_to2, &clk);
                auto w2 = std::make_unique<PartyWorker>(sub2, &net->p2_to1, &net->p2_to0, &clk);

                nets.push_back(std::move(net));
                workers.push_back(std::move(w0));
                workers.push_back(std::move(w1));
                workers.push_back(std::move(w2));
            }
        }
    }

    // Drive until all workers done.
    for (int it = 0; it < 20000; it++) {
        bool all_done = true;
        for (auto& w : workers) {
            const auto st = w->worker.tick_one();
            if (!st.ok()) throw std::runtime_error(st.message());
            if (!w->worker.is_done()) all_done = false;
        }
        clk.t_ms += 20;
        if (all_done) break;
        if (it == 19999) throw std::runtime_error("matrix did not complete");
    }

    // Drain some extra polls.
    for (int k = 0; k < 200; k++) {
        for (auto& w : workers) w->transport->poll();
        clk.t_ms += 20;
    }

    std::vector<uvcc::SubsessionRootV1> roots;
    roots.reserve(workers.size());
    for (auto& w : workers) {
        uvcc::SubsessionRootV1 sr;
        sr.coord = w->worker.coord;
        sr.merkle_root = w->ts.epoch_root(/*epoch_id32=*/0);
        roots.push_back(sr);
    }

    auto bundle = uvcc::build_audit_bundle_v1(sid_job, /*step_id=*/0, roots);
    if (!bundle.ok()) throw std::runtime_error(bundle.status().message());
    return bundle.value().global_root;
}

int main() {
    try {
        const auto a = run_matrix_once();
        const auto b = run_matrix_once();
        if (a != b) return fail("matrix determinism: global_root mismatch across runs");
        return 0;
    } catch (const std::exception& e) {
        return fail(std::string("exception: ") + e.what());
    }
}


