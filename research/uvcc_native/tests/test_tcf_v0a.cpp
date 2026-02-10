#include "uvcc/sha256.h"
#include "uvcc/tcf.h"
#include "uvcc/testkit.h"
#include "uvcc/transport.h"

#include <cstring>
#include <deque>
#include <iostream>
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

static void matmul_u64(const std::vector<uvcc::u64>& A, const std::vector<uvcc::u64>& B, std::vector<uvcc::u64>& C, uvcc::u32 m, uvcc::u32 k, uvcc::u32 n) {
    C.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0);
    for (uvcc::u32 i = 0; i < m; i++) {
        for (uvcc::u32 j = 0; j < n; j++) {
            uvcc::u64 acc = 0;
            for (uvcc::u32 t = 0; t < k; t++) {
                acc = static_cast<uvcc::u64>(acc + A[static_cast<std::size_t>(i) * k + t] * B[static_cast<std::size_t>(t) * n + j]);
            }
            C[static_cast<std::size_t>(i) * n + j] = acc;
        }
    }
}

static uvcc::Hash32 seed_from_dom(const char* dom) {
    return uvcc::sha256(dom, std::strlen(dom));
}

static void reconstruct_pub_mat(
    const uvcc::RSSU64MatV1& p0,
    const uvcc::RSSU64MatV1& p1,
    const uvcc::RSSU64MatV1& p2,
    std::vector<uvcc::u64>* out_pub) {
    // share0 = P0.lo
    // share1 = P0.hi (= P1.lo)
    // share2 = P1.hi (= P2.lo)
    const std::size_t n = p0.lo.size();
    out_pub->resize(n);
    for (std::size_t i = 0; i < n; i++) {
        const uvcc::u64 s0 = p0.lo[i];
        const uvcc::u64 s1 = p0.hi[i];
        const uvcc::u64 s2 = p1.hi[i];
        // overlap checks
        if (p1.lo[i] != s1) throw std::runtime_error("RSS overlap mismatch share1");
        if (p2.lo[i] != s2) throw std::runtime_error("RSS overlap mismatch share2");
        if (p2.hi[i] != s0) throw std::runtime_error("RSS overlap mismatch share0");
        (*out_pub)[i] = static_cast<uvcc::u64>(s0 + s1 + s2);
    }
}

int main() {
    try {
        const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
        const uvcc::u32 epoch = 0;
        const uvcc::u32 op_id32 = 0x77777777u;
        const uvcc::u32 m = 4, k = 4, n = 4;

        // Pairwise component seeds:
        // component0 shared by (0,2)
        // component1 shared by (0,1)
        // component2 shared by (1,2)
        const uvcc::Hash32 seed0 = seed_from_dom("seed_comp0_02");
        const uvcc::Hash32 seed1 = seed_from_dom("seed_comp1_01");
        const uvcc::Hash32 seed2 = seed_from_dom("seed_comp2_12");

        // Ring links: 0<->1, 1<->2, 0<->2.
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

        // Avoid assigning TransportV1 (it holds an internal clock pointer).
        uvcc::TcfV0aEngineV1* tcf0p = nullptr;
        uvcc::TcfV0aEngineV1* tcf1p = nullptr;
        uvcc::TcfV0aEngineV1* tcf2p = nullptr;

        uvcc::TransportCallbacksV1 c0;
        uvcc::TransportCallbacksV1 c1;
        uvcc::TransportCallbacksV1 c2;
        c0.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) {
            if (tcf0p) tcf0p->on_deliver(hdr, full);
        };
        c1.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) {
            if (tcf1p) tcf1p->on_deliver(hdr, full);
        };
        c2.on_deliver = [&](const uvcc::FrameHdrV1& hdr, const std::vector<uvcc::u8>& full) {
            if (tcf2p) tcf2p->on_deliver(hdr, full);
        };

        uvcc::TransportV1 tr0(sid, 0, &p0_prev, &p0_next, c0);
        uvcc::TransportV1 tr1(sid, 1, &p1_prev, &p1_next, c1);
        uvcc::TransportV1 tr2(sid, 2, &p2_prev, &p2_next, c2);

        uvcc::TcfV0aEngineV1 tcf0(sid, 0, uvcc::TcfSeedsV1{seed0, seed1}, &tr0);
        uvcc::TcfV0aEngineV1 tcf1(sid, 1, uvcc::TcfSeedsV1{seed1, seed2}, &tr1);
        uvcc::TcfV0aEngineV1 tcf2(sid, 2, uvcc::TcfSeedsV1{seed2, seed0}, &tr2);
        tcf0p = &tcf0;
        tcf1p = &tcf1;
        tcf2p = &tcf2;

        // Start on all parties.
        tcf0.start(op_id32, epoch, m, k, n);
        tcf1.start(op_id32, epoch, m, k, n);
        tcf2.start(op_id32, epoch, m, k, n);

        for (int iter = 0; iter < 5000; iter++) {
            tr0.poll();
            tr1.poll();
            tr2.poll();
            if (tcf0.is_done(op_id32) && tcf1.is_done(op_id32) && tcf2.is_done(op_id32)) break;
        }
        if (!tcf0.is_done(op_id32) || !tcf1.is_done(op_id32) || !tcf2.is_done(op_id32)) return fail("TCF did not complete");

        const auto tri0 = tcf0.take_triple(op_id32);
        const auto tri1 = tcf1.take_triple(op_id32);
        const auto tri2 = tcf2.take_triple(op_id32);

        std::vector<uvcc::u64> A_pub, B_pub, C_pub, expect;
        reconstruct_pub_mat(tri0.A, tri1.A, tri2.A, &A_pub);
        reconstruct_pub_mat(tri0.B, tri1.B, tri2.B, &B_pub);
        reconstruct_pub_mat(tri0.C, tri1.C, tri2.C, &C_pub);
        matmul_u64(A_pub, B_pub, expect, m, k, n);
        if (C_pub != expect) return fail("TCF triple C != A@B");

        return 0;
    } catch (const std::exception& e) {
        return fail(std::string("exception: ") + e.what());
    }
}


