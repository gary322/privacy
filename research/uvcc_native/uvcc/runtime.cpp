#include "uvcc/runtime.h"

#include "uvcc/bytes.h"
#include "uvcc/ids.h"
#include "uvcc/sha256.h"

#include <cstring>

namespace uvcc {

static u64 _hash64_le(const Sid32& sid_sub, u32 step_id, u8 phase_u8, u16 mb) {
    ByteWriter w;
    const char* dom = "uvcc.toy.open.v1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid_sub);
    w.write_u32_le(step_id);
    w.write_u8(phase_u8);
    w.write_u16_le(mb);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

static void _toy_open_vectors_for_party(u8 party, u64 base, std::vector<u64>* lo, std::vector<u64>* hi) {
    if (!lo || !hi) throw std::runtime_error("_toy_open_vectors_for_party: null out");
    // Component vectors (deterministic, shared across parties):
    // comp0 = [base+0, base+1]
    // comp1 = [base+10, base+11]
    // comp2 = [base+20, base+21]
    std::vector<u64> c0{static_cast<u64>(base + 0), static_cast<u64>(base + 1)};
    std::vector<u64> c1{static_cast<u64>(base + 10), static_cast<u64>(base + 11)};
    std::vector<u64> c2{static_cast<u64>(base + 20), static_cast<u64>(base + 21)};

    if (party == 0) {
        *lo = std::move(c0);
        *hi = std::move(c1);
    } else if (party == 1) {
        *lo = std::move(c1);
        *hi = std::move(c2);
    } else if (party == 2) {
        *lo = std::move(c2);
        *hi = std::move(c0);
    } else {
        throw std::runtime_error("_toy_open_vectors_for_party: party out of range");
    }
}

bool WorkerRuntimeV1::is_done() const {
    for (u16 mb = 0; mb < microbatches; mb++) {
        if (!pp_sched.is_phase_done(PhaseV1::FWD, mb)) return false;
        if (!pp_sched.is_phase_done(PhaseV1::BWD, mb)) return false;
    }
    return true;
}

StatusV1 WorkerRuntimeV1::tick_one() {
    if (transport != nullptr) {
        transport->poll();
    }

    // Ensure inflight vectors match microbatch count.
    if (open_inflight_fwd.size() != static_cast<std::size_t>(microbatches)) open_inflight_fwd.assign(static_cast<std::size_t>(microbatches), false);
    if (open_inflight_bwd.size() != static_cast<std::size_t>(microbatches)) open_inflight_bwd.assign(static_cast<std::size_t>(microbatches), false);

    // Resolve any completed OPENs and advance the scheduler.
    if (open != nullptr) {
        for (u16 mb = 0; mb < microbatches; mb++) {
            const std::size_t i = static_cast<std::size_t>(mb);

            if (open_inflight_fwd[i]) {
                const u32 op_id32 = derive_sgir_op_id32_v1(sid_sub, step_id, /*phase=*/0, mb, /*k=*/0);
                if (open->is_done(op_id32)) {
                    (void)open->take_result_u64(op_id32);
                    const TaskKeyV1 key{PhaseV1::FWD, mb, 0};
                    try {
                        pp_sched.set_waiting(key, false);
                        pp_sched.mark_done(key);
                    } catch (const std::exception& e) {
                        return StatusV1::Error(std::string("tick_one: fwd complete: ") + e.what());
                    }
                    open_inflight_fwd[i] = false;
                }
            }

            if (open_inflight_bwd[i]) {
                const u32 op_id32 = derive_sgir_op_id32_v1(sid_sub, step_id, /*phase=*/1, mb, /*k=*/0);
                if (open->is_done(op_id32)) {
                    (void)open->take_result_u64(op_id32);
                    const TaskKeyV1 key{PhaseV1::BWD, mb, 0};
                    try {
                        pp_sched.set_waiting(key, false);
                        pp_sched.mark_done(key);
                    } catch (const std::exception& e) {
                        return StatusV1::Error(std::string("tick_one: bwd complete: ") + e.what());
                    }
                    open_inflight_bwd[i] = false;
                }
            }
        }
    }

    auto task = pp_sched.pick_next_runnable();
    if (!task.has_value()) return StatusV1::Ok();

    // If we have a transport+open engine, run a toy OPEN at k=0 that yields until completion.
    if (transport != nullptr && open != nullptr) {
        if (task->k != 0) return StatusV1::Error("tick_one: toy runtime only supports k=0");
        const u8 phase_u8 = static_cast<u8>(task->phase == PhaseV1::FWD ? 0 : (task->phase == PhaseV1::BWD ? 1 : 2));
        if (phase_u8 > 1) return StatusV1::Error("tick_one: toy runtime only supports FWD/BWD");

        const std::size_t i = static_cast<std::size_t>(task->microbatch);
        const u32 op_id32 = derive_sgir_op_id32_v1(sid_sub, step_id, phase_u8, task->microbatch, /*k=*/0);
        const u64 stream_id64 = 0x1111000000000000ULL ^ (static_cast<u64>(phase_u8) << 32) ^ static_cast<u64>(task->microbatch);

        const u64 base = _hash64_le(sid_sub, step_id, phase_u8, task->microbatch);
        std::vector<u64> lo, hi;
        try {
            _toy_open_vectors_for_party(coord.party, base, &lo, &hi);
            FrameV1 f;
            open->enqueue_open_u64(op_id32, /*epoch_id32=*/step_id, stream_id64, lo, hi, &f);
            transport->send_frame_reliable(std::move(f));
            pp_sched.set_waiting(*task, true);
        } catch (const std::exception& e) {
            return StatusV1::Error(std::string("tick_one: enqueue_open_u64: ") + e.what());
        }

        if (task->phase == PhaseV1::FWD) {
            open_inflight_fwd[i] = true;
        } else {
            open_inflight_bwd[i] = true;
        }
        return StatusV1::Ok();
    }

    // Fallback: no protocol engines wired, just advance deterministically.
    try {
        pp_sched.mark_done(*task);
    } catch (const std::exception& e) {
        return StatusV1::Error(std::string("tick_one: ") + e.what());
    }
    return StatusV1::Ok();
}

}  // namespace uvcc


