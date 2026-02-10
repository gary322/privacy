#include "uvcc/open.h"
#include "uvcc/testkit.h"
#include "uvcc/transport.h"

#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

struct MemLink {
    std::deque<std::vector<uvcc::u8>> a_to_b;
    std::deque<std::vector<uvcc::u8>> b_to_a;
};

struct MemEndpoint final : public uvcc::RawConn {
    MemLink* link = nullptr;
    bool is_a = true;

    void send_bytes(const std::vector<uvcc::u8>& bytes) override {
        if (!link) return;
        if (is_a)
            link->a_to_b.push_back(bytes);
        else
            link->b_to_a.push_back(bytes);
    }

    bool poll_recv(std::vector<uvcc::u8>* out) override {
        if (!link || !out) return false;
        auto& q = is_a ? link->b_to_a : link->a_to_b;
        if (q.empty()) return false;
        *out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

int main() {
    const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
    const uvcc::u64 stream_id64 = 0x0123456789ABCDEFULL;
    const uvcc::u32 op_id32 = 0x11223344u;
    const uvcc::u32 epoch_id32 = 0;

    // Build 3-party ring links: 0<->1, 1<->2, 0<->2.
    MemLink link01;
    MemLink link12;
    MemLink link02;
    // Endpoints per party:
    // P0: next->1 via link01(a), prev->2 via link02(a)
    MemEndpoint p0_next;
    p0_next.link = &link01;
    p0_next.is_a = true;
    MemEndpoint p0_prev;
    p0_prev.link = &link02;
    p0_prev.is_a = true;
    // P1: next->2 via link12(a), prev->0 via link01(b)
    MemEndpoint p1_next;
    p1_next.link = &link12;
    p1_next.is_a = true;
    MemEndpoint p1_prev;
    p1_prev.link = &link01;
    p1_prev.is_a = false;
    // P2: next->0 via link02(b), prev->1 via link12(b)
    MemEndpoint p2_next;
    p2_next.link = &link02;
    p2_next.is_a = false;
    MemEndpoint p2_prev;
    p2_prev.link = &link12;
    p2_prev.is_a = false;

    // Open engines.
    uvcc::OpenEngineV1 o0(sid, 0);
    uvcc::OpenEngineV1 o1(sid, 1);
    uvcc::OpenEngineV1 o2(sid, 2);

    // Transports with on_deliver wiring to OpenEngine.
    uvcc::TransportCallbacksV1 c0;
    uvcc::TransportCallbacksV1 c1;
    uvcc::TransportCallbacksV1 c2;
    c0.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) { o0.on_deliver(hdr, full); };
    c1.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) { o1.on_deliver(hdr, full); };
    c2.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) { o2.on_deliver(hdr, full); };

    uvcc::TransportV1 t0(sid, 0, &p0_prev, &p0_next, c0);
    uvcc::TransportV1 t1(sid, 1, &p1_prev, &p1_next, c1);
    uvcc::TransportV1 t2(sid, 2, &p2_prev, &p2_next, c2);

    // Build a random secret x as additive shares x0+x1+x2 (mod 2^64).
    constexpr std::size_t N = 32;
    std::mt19937_64 rng(123);
    std::vector<uvcc::u64> x_pub(N);
    std::vector<uvcc::u64> x0(N), x1(N), x2(N);
    for (std::size_t i = 0; i < N; i++) {
        x_pub[i] = static_cast<uvcc::u64>(rng());
        x0[i] = static_cast<uvcc::u64>(rng());
        x1[i] = static_cast<uvcc::u64>(rng());
        x2[i] = static_cast<uvcc::u64>(x_pub[i] - x0[i] - x1[i]);  // wraps mod 2^64
    }

    const std::vector<uvcc::u64> p0_lo = x0, p0_hi = x1;
    const std::vector<uvcc::u64> p1_lo = x1, p1_hi = x2;
    const std::vector<uvcc::u64> p2_lo = x2, p2_hi = x0;

    uvcc::FrameV1 f0, f1, f2;
    o0.enqueue_open_u64(op_id32, epoch_id32, stream_id64, p0_lo, p0_hi, &f0);
    o1.enqueue_open_u64(op_id32, epoch_id32, stream_id64, p1_lo, p1_hi, &f1);
    o2.enqueue_open_u64(op_id32, epoch_id32, stream_id64, p2_lo, p2_hi, &f2);

    t0.send_frame_reliable(std::move(f0));
    t1.send_frame_reliable(std::move(f1));
    t2.send_frame_reliable(std::move(f2));

    // Pump until all done (or fail).
    for (int iter = 0; iter < 1000; iter++) {
        t0.poll();
        t1.poll();
        t2.poll();
        if (o0.is_done(op_id32) && o1.is_done(op_id32) && o2.is_done(op_id32)) break;
    }
    if (!o0.is_done(op_id32) || !o1.is_done(op_id32) || !o2.is_done(op_id32)) return fail("open did not complete");

    const auto r0 = o0.take_result_u64(op_id32);
    const auto r1 = o1.take_result_u64(op_id32);
    const auto r2 = o2.take_result_u64(op_id32);
    if (r0 != x_pub) return fail("party0 open mismatch");
    if (r1 != x_pub) return fail("party1 open mismatch");
    if (r2 != x_pub) return fail("party2 open mismatch");
    return 0;
}


