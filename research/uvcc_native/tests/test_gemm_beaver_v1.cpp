#include "uvcc/gemm_beaver.h"
#include "uvcc/tcf.h"
#include "uvcc/testkit.h"
#include "uvcc/transport.h"
#include "uvcc/sha256.h"

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

static void matmul_u64(const std::vector<uvcc::u64>& A, const std::vector<uvcc::u64>& B, std::vector<uvcc::u64>& C, uvcc::u32 d) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
    C.assign(n, 0);
    for (uvcc::u32 i = 0; i < d; i++) {
        for (uvcc::u32 j = 0; j < d; j++) {
            uvcc::u64 acc = 0;
            for (uvcc::u32 t = 0; t < d; t++) {
                acc = static_cast<uvcc::u64>(acc + (A[static_cast<std::size_t>(i) * d + t] * B[static_cast<std::size_t>(t) * d + j]));
            }
            C[static_cast<std::size_t>(i) * d + j] = acc;
        }
    }
}

static void share_additive(const std::vector<uvcc::u64>& x_pub, std::vector<uvcc::u64>& x0, std::vector<uvcc::u64>& x1, std::vector<uvcc::u64>& x2, std::mt19937_64& rng) {
    const std::size_t n = x_pub.size();
    x0.resize(n);
    x1.resize(n);
    x2.resize(n);
    for (std::size_t i = 0; i < n; i++) {
        x0[i] = static_cast<uvcc::u64>(rng());
        x1[i] = static_cast<uvcc::u64>(rng());
        x2[i] = static_cast<uvcc::u64>(x_pub[i] - x0[i] - x1[i]);  // wraps mod 2^64
    }
}

static uvcc::RSSU64MatV1 make_party_rss_mat(uvcc::u32 rows, uvcc::u32 cols, int party_id, const std::vector<uvcc::u64>& s0, const std::vector<uvcc::u64>& s1, const std::vector<uvcc::u64>& s2) {
    uvcc::RSSU64MatV1 m;
    m.rows = rows;
    m.cols = cols;
    if (party_id == 0) {
        m.lo = s0;
        m.hi = s1;
    } else if (party_id == 1) {
        m.lo = s1;
        m.hi = s2;
    } else if (party_id == 2) {
        m.lo = s2;
        m.hi = s0;
    } else {
        throw std::runtime_error("bad party_id");
    }
    return m;
}

static uvcc::Hash32 seed_from_dom(const char* dom) {
    return uvcc::sha256(dom, std::strlen(dom));
}

int main() {
    const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
    constexpr uvcc::u32 d = 4;
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
    const uvcc::u32 epoch_id32 = 0;
    const uvcc::u32 gemm_op_id32 = 0x42424242u;

    // Build 3-party ring links: 0<->1, 1<->2, 0<->2.
    MemLink link01;
    MemLink link12;
    MemLink link02;

    MemEndpoint p0_next;
    p0_next.link = &link01;
    p0_next.is_a = true;
    MemEndpoint p0_prev;
    p0_prev.link = &link02;
    p0_prev.is_a = true;

    MemEndpoint p1_next;
    p1_next.link = &link12;
    p1_next.is_a = true;
    MemEndpoint p1_prev;
    p1_prev.link = &link01;
    p1_prev.is_a = false;

    MemEndpoint p2_next;
    p2_next.link = &link02;
    p2_next.is_a = false;
    MemEndpoint p2_prev;
    p2_prev.link = &link12;
    p2_prev.is_a = false;

    uvcc::OpenEngineV1 o0(sid, 0);
    uvcc::OpenEngineV1 o1(sid, 1);
    uvcc::OpenEngineV1 o2(sid, 2);

    // TCF-v0a seeds (pairwise by component):
    const uvcc::Hash32 seed0 = seed_from_dom("seed_comp0_02");
    const uvcc::Hash32 seed1 = seed_from_dom("seed_comp1_01");
    const uvcc::Hash32 seed2 = seed_from_dom("seed_comp2_12");

    // Avoid assigning TransportV1 (it holds an internal clock pointer).
    uvcc::TcfV0aEngineV1* tcf0p = nullptr;
    uvcc::TcfV0aEngineV1* tcf1p = nullptr;
    uvcc::TcfV0aEngineV1* tcf2p = nullptr;

    uvcc::TransportCallbacksV1 c0;
    uvcc::TransportCallbacksV1 c1;
    uvcc::TransportCallbacksV1 c2;
    c0.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) {
        o0.on_deliver(hdr, full);
        if (tcf0p) tcf0p->on_deliver(hdr, full);
    };
    c1.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) {
        o1.on_deliver(hdr, full);
        if (tcf1p) tcf1p->on_deliver(hdr, full);
    };
    c2.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) {
        o2.on_deliver(hdr, full);
        if (tcf2p) tcf2p->on_deliver(hdr, full);
    };

    // Transports.
    uvcc::TransportV1 t0(sid, 0, &p0_prev, &p0_next, c0);
    uvcc::TransportV1 t1(sid, 1, &p1_prev, &p1_next, c1);
    uvcc::TransportV1 t2(sid, 2, &p2_prev, &p2_next, c2);

    // TCF engines bound to those transports.
    uvcc::TcfV0aEngineV1 tcf0(sid, 0, uvcc::TcfSeedsV1{seed0, seed1}, &t0);
    uvcc::TcfV0aEngineV1 tcf1(sid, 1, uvcc::TcfSeedsV1{seed1, seed2}, &t1);
    uvcc::TcfV0aEngineV1 tcf2(sid, 2, uvcc::TcfSeedsV1{seed2, seed0}, &t2);
    tcf0p = &tcf0;
    tcf1p = &tcf1;
    tcf2p = &tcf2;

    uvcc::BeaverGemmEngineV1 g0(sid, 0, &t0, &o0, &tcf0);
    uvcc::BeaverGemmEngineV1 g1(sid, 1, &t1, &o1, &tcf1);
    uvcc::BeaverGemmEngineV1 g2(sid, 2, &t2, &o2, &tcf2);

    std::mt19937_64 rng(999);

    // Public X, Y.
    std::vector<uvcc::u64> X_pub(n), Y_pub(n);
    for (std::size_t i = 0; i < n; i++) {
        X_pub[i] = static_cast<uvcc::u64>(rng());
        Y_pub[i] = static_cast<uvcc::u64>(rng());
    }

    // Additive shares for X and Y.
    std::vector<uvcc::u64> X0, X1, X2, Y0, Y1, Y2;
    share_additive(X_pub, X0, X1, X2, rng);
    share_additive(Y_pub, Y0, Y1, Y2, rng);

    auto make_task = [&](int pid) -> uvcc::BeaverGemmTaskV1 {
        uvcc::BeaverGemmTaskV1 task;
        task.op_id32 = gemm_op_id32;
        task.epoch_id32 = epoch_id32;
        task.m = d;
        task.k = d;
        task.n = d;
        task.X = make_party_rss_mat(d, d, pid, X0, X1, X2);
        task.Y = make_party_rss_mat(d, d, pid, Y0, Y1, Y2);
        // Leave task.triple empty: BeaverGemmEngine will generate it via TCF-v0a.
        return task;
    };

    g0.start(make_task(0));
    g1.start(make_task(1));
    g2.start(make_task(2));

    for (int iter = 0; iter < 5000; iter++) {
        g0.tick();
        g1.tick();
        g2.tick();
        t0.poll();
        t1.poll();
        t2.poll();
        if (g0.is_done(gemm_op_id32) && g1.is_done(gemm_op_id32) && g2.is_done(gemm_op_id32)) break;
    }
    if (!g0.is_done(gemm_op_id32) || !g1.is_done(gemm_op_id32) || !g2.is_done(gemm_op_id32)) return fail("gemm did not complete");

    const auto Z0 = g0.take_result(gemm_op_id32);
    const auto Z1 = g1.take_result(gemm_op_id32);
    const auto Z2 = g2.take_result(gemm_op_id32);

    // Reconstruct public Z from replicated shares:
    // share0 = P0.lo, share1 = P0.hi, share2 = P1.hi
    std::vector<uvcc::u64> Z_pub_rec(n);
    for (std::size_t i = 0; i < n; i++) {
        const uvcc::u64 z0 = Z0.lo[i];
        const uvcc::u64 z1 = Z0.hi[i];
        const uvcc::u64 z2 = Z1.hi[i];
        Z_pub_rec[i] = static_cast<uvcc::u64>(z0 + z1 + z2);
        // Sanity: overlaps match.
        if (Z1.lo[i] != z1) return fail("replicated overlap mismatch (share1)");
        if (Z2.lo[i] != z2) return fail("replicated overlap mismatch (share2)");
        if (Z2.hi[i] != z0) return fail("replicated overlap mismatch (share0)");
    }

    // Expected = X_pub @ Y_pub.
    std::vector<uvcc::u64> expect(n);
    matmul_u64(X_pub, Y_pub, expect, d);
    if (Z_pub_rec != expect) return fail("gemm output mismatch");
    (void)Z2;
    return 0;
}


