#include "uvcc/arena.h"
#include "uvcc/dp.h"
#include "uvcc/optim.h"
#include "uvcc/slots.h"

#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // Build three replica-local gradient slots and validate DP sum across replicas.
    uvcc::Arena a0, a1, a2;
    uvcc::SlotMapV1 s0(&a0), s1(&a1), s2(&a2);

    auto& g0 = s0.get_or_create_rss_u64(0x100, 4);
    auto& g1 = s1.get_or_create_rss_u64(0x100, 4);
    auto& g2 = s2.get_or_create_rss_u64(0x100, 4);

    for (int i = 0; i < 4; i++) {
        g0.lo[i] = static_cast<uvcc::u64>(1 + i);
        g0.hi[i] = static_cast<uvcc::u64>(10 + i);
        g1.lo[i] = static_cast<uvcc::u64>(100 + i);
        g1.hi[i] = static_cast<uvcc::u64>(200 + i);
        g2.lo[i] = static_cast<uvcc::u64>(1000 + i);
        g2.hi[i] = static_cast<uvcc::u64>(3000 + i);
    }

    uvcc::dp_allreduce_gradients_sim_v1(std::vector<uvcc::RSSU64SlotV1*>{&g0, &g1, &g2});

    for (int i = 0; i < 4; i++) {
        const uvcc::u64 want_lo = static_cast<uvcc::u64>((1 + i) + (100 + i) + (1000 + i));
        const uvcc::u64 want_hi = static_cast<uvcc::u64>((10 + i) + (200 + i) + (3000 + i));
        if (g0.lo[i] != want_lo || g1.lo[i] != want_lo || g2.lo[i] != want_lo) return fail("DP lo mismatch");
        if (g0.hi[i] != want_hi || g1.hi[i] != want_hi || g2.hi[i] != want_hi) return fail("DP hi mismatch");
    }

    // Now test deterministic SGD update on RSS shares: w := w - lr*g (mod 2^64).
    uvcc::Arena aw;
    uvcc::SlotMapV1 sw(&aw);
    auto& w = sw.get_or_create_rss_u64(0x200, 4);
    for (int i = 0; i < 4; i++) {
        w.lo[i] = static_cast<uvcc::u64>(5000 + i);
        w.hi[i] = static_cast<uvcc::u64>(6000 + i);
    }

    uvcc::SGDParamsV1 p;
    p.lr = 1.0;
    auto st = uvcc::sgd_update_v1(p, &w, &g0);
    if (!st.ok()) return fail("sgd_update_v1 returned error: " + st.message());

    for (int i = 0; i < 4; i++) {
        const uvcc::u64 want_lo = static_cast<uvcc::u64>((5000 + i) - g0.lo[i]);
        const uvcc::u64 want_hi = static_cast<uvcc::u64>((6000 + i) - g0.hi[i]);
        if (w.lo[i] != want_lo) return fail("SGD lo mismatch");
        if (w.hi[i] != want_hi) return fail("SGD hi mismatch");
    }
    return 0;
}


