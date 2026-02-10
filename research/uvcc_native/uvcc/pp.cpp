#include "uvcc/pp.h"

#include <algorithm>
#include <stdexcept>

namespace uvcc {

PPSchedulerV1::PPSchedulerV1(u16 S, u16 M, u16 stage_id, u16 k_max_fwd, u16 k_max_bwd)
    : S_(S), M_(M), stage_(stage_id), kmax_fwd_(k_max_fwd), kmax_bwd_(k_max_bwd) {
    if (S_ == 0) throw std::runtime_error("PPSchedulerV1: S must be >0");
    if (M_ == 0) throw std::runtime_error("PPSchedulerV1: M must be >0");
    if (stage_ >= S_) throw std::runtime_error("PPSchedulerV1: stage_id out of range");
    if (kmax_fwd_ == 0) throw std::runtime_error("PPSchedulerV1: k_max_fwd must be >0");
    if (kmax_bwd_ == 0) throw std::runtime_error("PPSchedulerV1: k_max_bwd must be >0");

    fwd_ready_.assign(static_cast<std::size_t>(M_), false);
    grad_ready_.assign(static_cast<std::size_t>(M_), false);
    fwd_nextk_.assign(static_cast<std::size_t>(M_), 0);
    bwd_nextk_.assign(static_cast<std::size_t>(M_), 0);
    fwd_wait_.assign(static_cast<std::size_t>(M_), false);
    bwd_wait_.assign(static_cast<std::size_t>(M_), false);
    fwd_done_.assign(static_cast<std::size_t>(M_), false);
    bwd_done_.assign(static_cast<std::size_t>(M_), false);

    // Stage 0 always has input activations available for all microbatches.
    if (stage_ == 0) {
        for (u16 mb = 0; mb < M_; mb++) fwd_ready_[static_cast<std::size_t>(mb)] = true;
    }
}

void PPSchedulerV1::notify_activation_arrived(u16 microbatch) {
    if (microbatch >= M_) throw std::runtime_error("notify_activation_arrived: microbatch out of range");
    fwd_ready_[static_cast<std::size_t>(microbatch)] = true;
}

void PPSchedulerV1::notify_grad_arrived(u16 microbatch) {
    if (microbatch >= M_) throw std::runtime_error("notify_grad_arrived: microbatch out of range");
    grad_ready_[static_cast<std::size_t>(microbatch)] = true;
}

void PPSchedulerV1::set_waiting(const TaskKeyV1& key, bool waiting) {
    if (key.microbatch >= M_) throw std::runtime_error("set_waiting: microbatch out of range");
    const std::size_t i = static_cast<std::size_t>(key.microbatch);
    if (key.phase == PhaseV1::FWD) {
        if (key.k != fwd_nextk_[i]) throw std::runtime_error("set_waiting: FWD key.k mismatch (not current next_k)");
        fwd_wait_[i] = waiting;
    } else if (key.phase == PhaseV1::BWD) {
        if (key.k != bwd_nextk_[i]) throw std::runtime_error("set_waiting: BWD key.k mismatch (not current next_k)");
        bwd_wait_[i] = waiting;
    } else {
        throw std::runtime_error("set_waiting: UPD not implemented in v1 scheduler");
    }
}

bool PPSchedulerV1::fwd_runnable_(u16 mb) const {
    const std::size_t i = static_cast<std::size_t>(mb);
    if (mb >= M_) return false;
    if (!fwd_ready_[i]) return false;
    if (fwd_done_[i]) return false;
    if (fwd_wait_[i]) return false;
    if (fwd_nextk_[i] >= kmax_fwd_) return false;
    return true;
}

bool PPSchedulerV1::bwd_runnable_(u16 mb) const {
    const std::size_t i = static_cast<std::size_t>(mb);
    if (mb >= M_) return false;
    if (!fwd_done_[i]) return false;  // cannot backprop before local forward done
    // For last stage, loss grad is local; we treat it as ready when forward completes.
    const bool grad_ok = (stage_ == static_cast<u16>(S_ - 1)) ? true : grad_ready_[i];
    if (!grad_ok) return false;
    if (bwd_done_[i]) return false;
    if (bwd_wait_[i]) return false;
    if (bwd_nextk_[i] >= kmax_bwd_) return false;
    return true;
}

std::optional<TaskKeyV1> PPSchedulerV1::pick_next_runnable() const {
    for (u16 mb = 0; mb < M_; mb++) {
        if (fwd_runnable_(mb)) {
            return TaskKeyV1{PhaseV1::FWD, mb, fwd_nextk_[static_cast<std::size_t>(mb)]};
        }
        if (bwd_runnable_(mb)) {
            return TaskKeyV1{PhaseV1::BWD, mb, bwd_nextk_[static_cast<std::size_t>(mb)]};
        }
    }
    return std::nullopt;
}

void PPSchedulerV1::mark_done(const TaskKeyV1& key) {
    if (key.microbatch >= M_) throw std::runtime_error("mark_done: microbatch out of range");
    const std::size_t i = static_cast<std::size_t>(key.microbatch);
    if (key.phase == PhaseV1::FWD) {
        if (key.k != fwd_nextk_[i]) throw std::runtime_error("mark_done: FWD key.k mismatch");
        if (fwd_wait_[i]) throw std::runtime_error("mark_done: FWD called while waiting");
        fwd_nextk_[i] = static_cast<u16>(fwd_nextk_[i] + 1);
        if (fwd_nextk_[i] >= kmax_fwd_) {
            fwd_done_[i] = true;
            // For last stage, forward completion implies loss grad is now available.
            if (stage_ == static_cast<u16>(S_ - 1)) grad_ready_[i] = true;
        }
    } else if (key.phase == PhaseV1::BWD) {
        if (key.k != bwd_nextk_[i]) throw std::runtime_error("mark_done: BWD key.k mismatch");
        if (bwd_wait_[i]) throw std::runtime_error("mark_done: BWD called while waiting");
        bwd_nextk_[i] = static_cast<u16>(bwd_nextk_[i] + 1);
        if (bwd_nextk_[i] >= kmax_bwd_) {
            bwd_done_[i] = true;
        }
    } else {
        throw std::runtime_error("mark_done: UPD not implemented in v1 scheduler");
    }
}

bool PPSchedulerV1::is_phase_done(PhaseV1 phase, u16 microbatch) const {
    if (microbatch >= M_) throw std::runtime_error("is_phase_done: microbatch out of range");
    const std::size_t i = static_cast<std::size_t>(microbatch);
    if (phase == PhaseV1::FWD) return fwd_done_[i];
    if (phase == PhaseV1::BWD) return bwd_done_[i];
    return false;
}

std::vector<std::vector<PPTickActionV1>> generate_1f1b_ticks_flush_v1(u16 S, u16 M) {
    if (S == 0) throw std::runtime_error("generate_1f1b_ticks_flush_v1: S must be >0");
    if (M == 0) throw std::runtime_error("generate_1f1b_ticks_flush_v1: M must be >0");

    // State: fwd_done[s][mb], bwd_done[s][mb].
    std::vector<std::vector<bool>> fwd_done(static_cast<std::size_t>(S), std::vector<bool>(static_cast<std::size_t>(M), false));
    std::vector<std::vector<bool>> bwd_done(static_cast<std::size_t>(S), std::vector<bool>(static_cast<std::size_t>(M), false));
    std::vector<u16> fwd_next(static_cast<std::size_t>(S), 0);
    std::vector<u16> bwd_next(static_cast<std::size_t>(S), 0);
    std::vector<u16> fwd_count(static_cast<std::size_t>(S), 0);

    std::vector<std::vector<PPTickActionV1>> out;
    out.reserve(static_cast<std::size_t>(2 * M + 2 * (S - 1) + 4));

    const auto all_done = [&]() -> bool {
        for (u16 s = 0; s < S; s++) {
            if (bwd_next[static_cast<std::size_t>(s)] < M) return false;
        }
        return true;
    };

    // Defensive upper bound: flush schedule is <= 2*M + 2*(S-1) ticks for M>=1.
    const std::size_t max_ticks = static_cast<std::size_t>(2 * M + 2 * (S - 1) + 8);

    for (std::size_t tick = 0; tick < max_ticks && !all_done(); tick++) {
        std::vector<PPTickActionV1> actions(static_cast<std::size_t>(S));

        // Decide actions for all stages based on *previous* tick state (synchronous pipeline).
        for (u16 s = 0; s < S; s++) {
            const std::size_t si = static_cast<std::size_t>(s);

            const u16 mb_fwd = fwd_next[si];
            const u16 mb_bwd = bwd_next[si];

            const bool fwd_has = mb_fwd < M;
            const bool bwd_has = mb_bwd < M;

            bool fwd_avail = false;
            if (fwd_has) {
                if (s == 0) {
                    fwd_avail = true;
                } else {
                    fwd_avail = fwd_done[static_cast<std::size_t>(s - 1)][static_cast<std::size_t>(mb_fwd)];
                }
            }

            bool bwd_avail = false;
            if (bwd_has) {
                const bool local_fwd_done = fwd_done[si][static_cast<std::size_t>(mb_bwd)];
                if (local_fwd_done) {
                    if (s == static_cast<u16>(S - 1)) {
                        // Loss grad is local; backward can start one tick after forward.
                        bwd_avail = true;
                    } else {
                        bwd_avail = bwd_done[static_cast<std::size_t>(s + 1)][static_cast<std::size_t>(mb_bwd)];
                    }
                }
            }

            const u16 warmup_target = static_cast<u16>((S - 1) - s);  // PARALLEL.txt

            // 1F1B-flush policy:
            // - after local warmup target, prefer backward when available,
            // - else do forward when available,
            // - else do backward (cooldown drain),
            // - else idle.
            if (bwd_avail && fwd_count[si] >= warmup_target) {
                actions[si] = PPTickActionV1{false, PhaseV1::BWD, mb_bwd};
            } else if (fwd_avail) {
                actions[si] = PPTickActionV1{false, PhaseV1::FWD, mb_fwd};
            } else if (bwd_avail) {
                actions[si] = PPTickActionV1{false, PhaseV1::BWD, mb_bwd};
            } else {
                actions[si] = PPTickActionV1{true, PhaseV1::FWD, 0};
            }
        }

        // Apply actions (advance state).
        bool progressed = false;
        for (u16 s = 0; s < S; s++) {
            const std::size_t si = static_cast<std::size_t>(s);
            const auto a = actions[si];
            if (a.idle) continue;
            progressed = true;
            if (a.phase == PhaseV1::FWD) {
                const u16 mb = a.microbatch;
                if (mb != fwd_next[si]) throw std::runtime_error("schedule bug: fwd microbatch mismatch");
                fwd_done[si][static_cast<std::size_t>(mb)] = true;
                fwd_next[si] = static_cast<u16>(fwd_next[si] + 1);
                fwd_count[si] = static_cast<u16>(fwd_count[si] + 1);
            } else if (a.phase == PhaseV1::BWD) {
                const u16 mb = a.microbatch;
                if (mb != bwd_next[si]) throw std::runtime_error("schedule bug: bwd microbatch mismatch");
                bwd_done[si][static_cast<std::size_t>(mb)] = true;
                bwd_next[si] = static_cast<u16>(bwd_next[si] + 1);
            } else {
                throw std::runtime_error("UPD not part of tick schedule in v1");
            }
        }

        if (!progressed) throw std::runtime_error("generate_1f1b_ticks_flush_v1: deadlock (no stage progressed)");
        out.push_back(std::move(actions));
    }

    if (!all_done()) throw std::runtime_error("generate_1f1b_ticks_flush_v1: exceeded max_ticks (bug)");
    return out;
}

}  // namespace uvcc


