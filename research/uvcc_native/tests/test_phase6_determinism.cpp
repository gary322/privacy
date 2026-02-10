#include "uvcc/audit.h"
#include "uvcc/config.h"
#include "uvcc/open.h"
#include "uvcc/runtime.h"
#include "uvcc/subsession.h"
#include "uvcc/testkit.h"
#include "uvcc/transcript_hooks.h"

#include <cstring>
#include <deque>
#include <iostream>
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
    bool drop_first_ack_a_to_b = false;
    bool drop_first_ack_b_to_a = false;
};

struct MemEndpoint final : public uvcc::RawConn {
    MemDuplex* d = nullptr;
    bool is_a = true;

    void send_bytes(const std::vector<uvcc::u8>& bytes) override {
        if (!d) return;
        const bool is_ack = (bytes.size() >= 4 && std::memcmp(bytes.data(), "ACK1", 4) == 0);
        if (is_ack) {
            if (is_a && d->drop_first_ack_a_to_b) {
                d->drop_first_ack_a_to_b = false;
                return;
            }
            if (!is_a && d->drop_first_ack_b_to_a) {
                d->drop_first_ack_b_to_a = false;
                return;
            }
        }
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

static uvcc::Hash32 run_toy_step_once(bool drop_first_ack_on_one_link) {
    const uvcc::Sid32 sid_job = uvcc::sid32_seq_00_1f();
    const uvcc::u32 step_id = 0;

    uvcc::TopologyV1 topo;
    topo.replicas = 1;
    topo.stages = 1;
    topo.tp_ranks = 1;

    // Connections: (0<->1), (1<->2), (2<->0)
    MemDuplex d01, d12, d20;
    if (drop_first_ack_on_one_link) {
        // Drop the first ACK from party2 -> party1 on the (1<->2) link.
        d12.drop_first_ack_b_to_a = true;
    }
    MemEndpoint ep0_to1;
    ep0_to1.d = &d01;
    ep0_to1.is_a = true;
    MemEndpoint ep1_to0;
    ep1_to0.d = &d01;
    ep1_to0.is_a = false;

    MemEndpoint ep1_to2;
    ep1_to2.d = &d12;
    ep1_to2.is_a = true;
    MemEndpoint ep2_to1;
    ep2_to1.d = &d12;
    ep2_to1.is_a = false;

    // Define a=2, b=0 for (2<->0)
    MemEndpoint ep2_to0;
    ep2_to0.d = &d20;
    ep2_to0.is_a = true;
    MemEndpoint ep0_to2;
    ep0_to2.d = &d20;
    ep0_to2.is_a = false;

    FakeClock clk;

    // Build per-party subsessions (same sid_sub across parties for same (r,s,t)).
    uvcc::CoordV1 c0{0, 0, 0, 0};
    uvcc::CoordV1 c1{1, 0, 0, 0};
    uvcc::CoordV1 c2{2, 0, 0, 0};

    const auto ss0 = uvcc::make_subsession_v1(topo, c0, sid_job);
    const auto ss1 = uvcc::make_subsession_v1(topo, c1, sid_job);
    const auto ss2 = uvcc::make_subsession_v1(topo, c2, sid_job);
    if (ss0.sid_sub != ss1.sid_sub || ss0.sid_sub != ss2.sid_sub) throw std::runtime_error("sid_sub mismatch across parties");

    uvcc::TranscriptStoreV1 ts0(ss0.sid_sub);
    uvcc::TranscriptStoreV1 ts1(ss1.sid_sub);
    uvcc::TranscriptStoreV1 ts2(ss2.sid_sub);

    uvcc::OpenEngineV1 open0(ss0.sid_sub, /*self_party=*/0);
    uvcc::OpenEngineV1 open1(ss1.sid_sub, /*self_party=*/1);
    uvcc::OpenEngineV1 open2(ss2.sid_sub, /*self_party=*/2);

    // Transport callbacks: record transcript leaves and deliver OPEN payloads to OpenEngine.
    uvcc::TransportCallbacksV1 cb0 = uvcc::make_lift_transcript_callbacks_v1(&ts0, ss0.sid_sub);
    cb0.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& payload) { open0.on_deliver(hdr, payload); };
    uvcc::TransportCallbacksV1 cb1 = uvcc::make_lift_transcript_callbacks_v1(&ts1, ss1.sid_sub);
    cb1.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& payload) { open1.on_deliver(hdr, payload); };
    uvcc::TransportCallbacksV1 cb2 = uvcc::make_lift_transcript_callbacks_v1(&ts2, ss2.sid_sub);
    cb2.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& payload) { open2.on_deliver(hdr, payload); };

    // Party0: next=1, prev=2
    uvcc::TransportV1 t0(ss0.sid_sub, /*self_party=*/0, /*to_prev=*/&ep0_to2, /*to_next=*/&ep0_to1, cb0, &clk);
    // Party1: next=2, prev=0
    uvcc::TransportV1 t1(ss1.sid_sub, /*self_party=*/1, /*to_prev=*/&ep1_to0, /*to_next=*/&ep1_to2, cb1, &clk);
    // Party2: next=0, prev=1
    uvcc::TransportV1 t2(ss2.sid_sub, /*self_party=*/2, /*to_prev=*/&ep2_to1, /*to_next=*/&ep2_to0, cb2, &clk);

    // Build one worker per party (toy program: FWD open then BWD open).
    uvcc::WorkerRuntimeV1 w0;
    w0.cfg.topo = topo;
    w0.cfg.coord = c0;
    w0.cfg.sid_job = sid_job;
    w0.coord = c0;
    w0.sid_sub = ss0.sid_sub;
    w0.transport = &t0;
    w0.transcript = &ts0;
    w0.open = &open0;
    w0.step_id = step_id;
    w0.microbatches = 1;
    w0.pp_sched = uvcc::PPSchedulerV1(/*S=*/1, /*M=*/1, /*stage_id=*/0, /*kmax_fwd=*/1, /*kmax_bwd=*/1);

    uvcc::WorkerRuntimeV1 w1;
    w1.cfg.topo = topo;
    w1.cfg.coord = c1;
    w1.cfg.sid_job = sid_job;
    w1.coord = c1;
    w1.sid_sub = ss1.sid_sub;
    w1.transport = &t1;
    w1.transcript = &ts1;
    w1.open = &open1;
    w1.step_id = step_id;
    w1.microbatches = 1;
    w1.pp_sched = uvcc::PPSchedulerV1(/*S=*/1, /*M=*/1, /*stage_id=*/0, /*kmax_fwd=*/1, /*kmax_bwd=*/1);

    uvcc::WorkerRuntimeV1 w2;
    w2.cfg.topo = topo;
    w2.cfg.coord = c2;
    w2.cfg.sid_job = sid_job;
    w2.coord = c2;
    w2.sid_sub = ss2.sid_sub;
    w2.transport = &t2;
    w2.transcript = &ts2;
    w2.open = &open2;
    w2.step_id = step_id;
    w2.microbatches = 1;
    w2.pp_sched = uvcc::PPSchedulerV1(/*S=*/1, /*M=*/1, /*stage_id=*/0, /*kmax_fwd=*/1, /*kmax_bwd=*/1);

    // Drive until all workers report done.
    const std::size_t max_iters = 2000;
    for (std::size_t it = 0; it < max_iters; it++) {
        auto s0 = w0.tick_one();
        auto s1 = w1.tick_one();
        auto s2 = w2.tick_one();
        if (!s0.ok()) throw std::runtime_error(s0.message());
        if (!s1.ok()) throw std::runtime_error(s1.message());
        if (!s2.ok()) throw std::runtime_error(s2.message());
        // Advance time to allow retransmits.
        clk.t_ms += 60;
        if (w0.is_done() && w1.is_done() && w2.is_done()) break;
        if (it + 1 == max_iters) throw std::runtime_error("toy step did not complete");
    }

    // Drain: ensure ACKs/retransmits settle so transcript roots include ACK_ACCEPT leaves.
    for (int i = 0; i < 20; i++) {
        t0.poll();
        t1.poll();
        t2.poll();
        clk.t_ms += 60;
    }

    const auto r0 = ts0.epoch_root(step_id);
    const auto r1 = ts1.epoch_root(step_id);
    const auto r2 = ts2.epoch_root(step_id);

    std::vector<uvcc::SubsessionRootV1> roots;
    roots.push_back(uvcc::SubsessionRootV1{c0, r0});
    roots.push_back(uvcc::SubsessionRootV1{c1, r1});
    roots.push_back(uvcc::SubsessionRootV1{c2, r2});

    auto bundle = uvcc::build_audit_bundle_v1(sid_job, step_id, roots);
    if (!bundle.ok()) throw std::runtime_error(bundle.status().message());
    return bundle.value().global_root;
}

int main() {
    try {
        const auto root0 = run_toy_step_once(false);
        const auto root1 = run_toy_step_once(false);
        if (root0 != root1) return fail("determinism: global_root mismatch across identical runs");

        const auto root_fault = run_toy_step_once(true);
        if (root_fault != root0) return fail("fault injection: global_root mismatch vs baseline");
        return 0;
    } catch (const std::exception& e) {
        return fail(std::string("exception: ") + e.what());
    }
}


