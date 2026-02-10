#include "uvcc/sha256.h"
#include "uvcc/testkit.h"
#include "uvcc/wvole.h"

#include <cstring>
#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

static uvcc::Hash32 seed_from_dom(const char* dom) {
    return uvcc::sha256(dom, std::strlen(dom));
}

int main() {
    try {
        const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
        const uvcc::Hash32 s0 = seed_from_dom("seed_lo");
        const uvcc::Hash32 s1 = seed_from_dom("seed_hi");
        uvcc::WarpVoleStubV1 w0(sid, 0, uvcc::WVoleSeedsV1{s0, s1});
        uvcc::WarpVoleStubV1 w1(sid, 0, uvcc::WVoleSeedsV1{s0, s1});

        const auto b0 = w0.expand_block(/*op_id32=*/0x1111, /*block_id32=*/7, /*n_words=*/16);
        const auto b1 = w1.expand_block(/*op_id32=*/0x1111, /*block_id32=*/7, /*n_words=*/16);
        if (b0.n_words != 16 || b1.n_words != 16) return fail("n_words mismatch");
        if (b0.u_lo != b1.u_lo) return fail("determinism mismatch u_lo");
        if (b0.u_hi != b1.u_hi) return fail("determinism mismatch u_hi");
        if (b0.v_lo != b1.v_lo) return fail("determinism mismatch v_lo");
        if (b0.v_hi != b1.v_hi) return fail("determinism mismatch v_hi");

        // Changing op_id changes output.
        const auto b2 = w0.expand_block(/*op_id32=*/0x2222, /*block_id32=*/7, /*n_words=*/16);
        if (b2.u_lo == b0.u_lo) return fail("expected op_id to domain-separate output");

        return 0;
    } catch (const std::exception& e) {
        return fail(std::string("exception: ") + e.what());
    }
}


