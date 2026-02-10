#pragma once

#include "uvcc/transcript.h"
#include "uvcc/transport.h"

namespace uvcc {

// Leaf type codes for lift transport events (privacy_new.txt §6.2).
constexpr u16 LEAF_LIFT_DATA_SEND = 0x4101;
constexpr u16 LEAF_LIFT_DATA_ACCEPT = 0x4102;
constexpr u16 LEAF_LIFT_ACK_SEND = 0x4103;
constexpr u16 LEAF_LIFT_ACK_ACCEPT = 0x4104;

// Build TransportCallbacksV1 that records deterministic transcript leaves into `ts`.
TransportCallbacksV1 make_lift_transcript_callbacks_v1(TranscriptStoreV1* ts, const Sid32& sid32);

}  // namespace uvcc


