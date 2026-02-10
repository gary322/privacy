#pragma once

#include "uvcc/status.h"
#include "uvcc/types.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace uvcc {

// Phase 5: deterministic microbatch scheduling (PP).
//
// This scheduler is intentionally:
// - dependency-aware (activation/grad readiness + network wait blocks), and
// - deterministic (always picks smallest runnable microbatch first).
//
// NOTE: PARALLEL.txt has a small inconsistency in one place: it says "smallest mb first"
// but later writes a lexicographic tuple starting with phase. We implement "smallest mb first"
// as the primary key, because it is what yields canonical 1F1B behavior once backward becomes
// runnable.

enum class PhaseV1 : u8 { FWD = 0, BWD = 1, UPD = 2 };

struct TaskKeyV1 {
    PhaseV1 phase = PhaseV1::FWD;
    u16 microbatch = 0;
    u16 k = 0;  // op index within that (phase, microbatch) for this stage
};

inline bool operator==(const TaskKeyV1& a, const TaskKeyV1& b) {
    return a.phase == b.phase && a.microbatch == b.microbatch && a.k == b.k;
}

// Ordering: (microbatch, phase, k) so microbatch is the primary deterministic key.
inline bool operator<(const TaskKeyV1& a, const TaskKeyV1& b) {
    if (a.microbatch != b.microbatch) return a.microbatch < b.microbatch;
    if (static_cast<u8>(a.phase) != static_cast<u8>(b.phase)) return static_cast<u8>(a.phase) < static_cast<u8>(b.phase);
    return a.k < b.k;
}

struct PPTickActionV1 {
    bool idle = true;
    PhaseV1 phase = PhaseV1::FWD;
    u16 microbatch = 0;
};

// Generate a global 1F1B "flush" schedule for validation / debugging.
//
// Returns matrix[tick][stage] = action.
std::vector<std::vector<PPTickActionV1>> generate_1f1b_ticks_flush_v1(u16 S, u16 M);

class PPSchedulerV1 {
   public:
    PPSchedulerV1(u16 S, u16 M, u16 stage_id, u16 k_max_fwd = 1, u16 k_max_bwd = 1);

    void notify_activation_arrived(u16 microbatch);  // activation from prev stage
    void notify_grad_arrived(u16 microbatch);        // gradient from next stage

    // Block/unblock a task at its current k (used when an op yields on network).
    void set_waiting(const TaskKeyV1& key, bool waiting);

    // Returns the next runnable task for this stage, or nullopt if nothing runnable.
    std::optional<TaskKeyV1> pick_next_runnable() const;

    // Mark one k as completed for this task. Advances next_k; marks phase done when k_max reached.
    void mark_done(const TaskKeyV1& key);

    bool is_phase_done(PhaseV1 phase, u16 microbatch) const;

   private:
    u16 S_ = 1;
    u16 M_ = 1;
    u16 stage_ = 0;
    u16 kmax_fwd_ = 1;
    u16 kmax_bwd_ = 1;

    std::vector<bool> fwd_ready_;
    std::vector<bool> grad_ready_;

    std::vector<u16> fwd_nextk_;
    std::vector<u16> bwd_nextk_;
    std::vector<bool> fwd_wait_;
    std::vector<bool> bwd_wait_;
    std::vector<bool> fwd_done_;
    std::vector<bool> bwd_done_;

    bool fwd_runnable_(u16 mb) const;
    bool bwd_runnable_(u16 mb) const;
};

}  // namespace uvcc


