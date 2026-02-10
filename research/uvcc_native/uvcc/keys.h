#pragma once

#include "uvcc/types.h"

namespace uvcc {

// Phase 1 placeholder: key schedules will be implemented once transport+protocol engines are wired.
// We keep the type+API stable so later phases can plug in the real KDF that matches the existing UVCC formulas.
struct KeyScheduleV1 {
    Sid32 sid_sub{};
};

inline KeyScheduleV1 make_keys_v1(const Sid32& sid_sub) {
    KeyScheduleV1 ks;
    ks.sid_sub = sid_sub;
    return ks;
}

}  // namespace uvcc


