#pragma once

#include "uvcc/slots.h"

namespace uvcc {

// Minimal CPU implementations of some slot ops for early bring-up/determinism tests.
void rss_u64_add_inplace(RSSU64SlotV1& dst, const RSSU64SlotV1& src);
void rss_u64_add(RSSU64SlotV1& out, const RSSU64SlotV1& a, const RSSU64SlotV1& b);

}  // namespace uvcc


