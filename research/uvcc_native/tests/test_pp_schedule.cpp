#include "uvcc/pp.h"

#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

static std::string fmt_action(const uvcc::PPTickActionV1& a) {
    if (a.idle) return "-";
    const char* p = (a.phase == uvcc::PhaseV1::FWD) ? "F" : (a.phase == uvcc::PhaseV1::BWD ? "B" : "U");
    return std::string(p) + "(" + std::to_string(a.microbatch) + ")";
}

static bool eq_action(const uvcc::PPTickActionV1& a, bool idle, uvcc::PhaseV1 ph, uvcc::u16 mb) {
    if (a.idle != idle) return false;
    if (idle) return true;
    return a.phase == ph && a.microbatch == mb;
}

int main() {
    // Validate schedule generator against the canonical 1F1B-flush behavior for S=4.
    // We use M=8 so the schedule is small enough for a unit test.
    const uvcc::u16 S = 4;
    const uvcc::u16 M = 8;
    const auto ticks = uvcc::generate_1f1b_ticks_flush_v1(S, M);

    // For flush 1F1B: total ticks = 2*M + 2*(S-1) = 22
    const std::size_t expected_ticks = static_cast<std::size_t>(2 * M + 2 * (S - 1));
    if (ticks.size() != expected_ticks) {
        return fail("tick count mismatch: got " + std::to_string(ticks.size()) + " expected " + std::to_string(expected_ticks));
    }

    // Check first 9 ticks (0..8) for S=4 (matches the warmup then steady-state alternation).
    // tick  Stage0  Stage1  Stage2  Stage3
    // 0     F(0)    -       -       -
    // 1     F(1)    F(0)    -       -
    // 2     F(2)    F(1)    F(0)    -
    // 3     F(3)    F(2)    F(1)    F(0)
    // 4     F(4)    F(3)    F(2)    B(0)
    // 5     F(5)    F(4)    B(0)    F(1)
    // 6     F(6)    B(0)    F(3)    B(1)
    // 7     B(0)    F(5)    B(1)    F(2)
    // 8     F(7)    B(1)    F(4)    B(2)
    struct ExpRow {
        bool idle;
        uvcc::PhaseV1 ph;
        uvcc::u16 mb;
    };
    const std::vector<std::vector<ExpRow>> exp = {
        {{false, uvcc::PhaseV1::FWD, 0}, {true, uvcc::PhaseV1::FWD, 0}, {true, uvcc::PhaseV1::FWD, 0}, {true, uvcc::PhaseV1::FWD, 0}},
        {{false, uvcc::PhaseV1::FWD, 1}, {false, uvcc::PhaseV1::FWD, 0}, {true, uvcc::PhaseV1::FWD, 0}, {true, uvcc::PhaseV1::FWD, 0}},
        {{false, uvcc::PhaseV1::FWD, 2}, {false, uvcc::PhaseV1::FWD, 1}, {false, uvcc::PhaseV1::FWD, 0}, {true, uvcc::PhaseV1::FWD, 0}},
        {{false, uvcc::PhaseV1::FWD, 3}, {false, uvcc::PhaseV1::FWD, 2}, {false, uvcc::PhaseV1::FWD, 1}, {false, uvcc::PhaseV1::FWD, 0}},
        {{false, uvcc::PhaseV1::FWD, 4}, {false, uvcc::PhaseV1::FWD, 3}, {false, uvcc::PhaseV1::FWD, 2}, {false, uvcc::PhaseV1::BWD, 0}},
        {{false, uvcc::PhaseV1::FWD, 5}, {false, uvcc::PhaseV1::FWD, 4}, {false, uvcc::PhaseV1::BWD, 0}, {false, uvcc::PhaseV1::FWD, 1}},
        {{false, uvcc::PhaseV1::FWD, 6}, {false, uvcc::PhaseV1::BWD, 0}, {false, uvcc::PhaseV1::FWD, 3}, {false, uvcc::PhaseV1::BWD, 1}},
        {{false, uvcc::PhaseV1::BWD, 0}, {false, uvcc::PhaseV1::FWD, 5}, {false, uvcc::PhaseV1::BWD, 1}, {false, uvcc::PhaseV1::FWD, 2}},
        {{false, uvcc::PhaseV1::FWD, 7}, {false, uvcc::PhaseV1::BWD, 1}, {false, uvcc::PhaseV1::FWD, 4}, {false, uvcc::PhaseV1::BWD, 2}},
    };
    for (std::size_t t = 0; t < exp.size(); t++) {
        if (ticks[t].size() != S) return fail("ticks[t] stage count mismatch");
        for (std::size_t s = 0; s < S; s++) {
            const auto& a = ticks[t][s];
            const auto& e = exp[t][s];
            if (!eq_action(a, e.idle, e.ph, e.mb)) {
                return fail("tick " + std::to_string(t) + " stage " + std::to_string(s) + " mismatch: got " + fmt_action(a));
            }
        }
    }

    // Sanity: each stage executes exactly M forward and M backward.
    for (std::size_t s = 0; s < S; s++) {
        std::size_t nf = 0, nb = 0;
        for (const auto& row : ticks) {
            const auto& a = row[s];
            if (a.idle) continue;
            if (a.phase == uvcc::PhaseV1::FWD) nf++;
            if (a.phase == uvcc::PhaseV1::BWD) nb++;
        }
        if (nf != M) return fail("stage " + std::to_string(s) + " forward count mismatch");
        if (nb != M) return fail("stage " + std::to_string(s) + " backward count mismatch");
    }

    // Also validate PPSchedulerV1 "smallest mb first" behavior on last stage.
    {
        uvcc::PPSchedulerV1 sched(/*S=*/4, /*M=*/4, /*stage_id=*/3, /*kmax_fwd=*/1, /*kmax_bwd=*/1);
        // stage3 sees activations for mb0 and mb1 arrive.
        sched.notify_activation_arrived(0);
        sched.notify_activation_arrived(1);
        auto k0 = sched.pick_next_runnable();
        if (!k0.has_value() || k0->phase != uvcc::PhaseV1::FWD || k0->microbatch != 0) return fail("sched expected FWD(0) first");
        sched.mark_done(*k0);  // completes FWD(0) => BWD(0) becomes runnable on last stage
        auto k1 = sched.pick_next_runnable();
        if (!k1.has_value() || k1->phase != uvcc::PhaseV1::BWD || k1->microbatch != 0) return fail("sched expected BWD(0) before FWD(1)");
    }

    return 0;
}


