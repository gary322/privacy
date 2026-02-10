#pragma once

#include "uvcc/types.h"

namespace uvcc {

// Minimal opcode set for early bring-up (Phase 4). Expanded in Phase 7+ for full transformer.
enum class OpCode : u16 {
    NOP = 0,
    GEMM_SS = 1,      // secret-secret GEMM (placeholder)
    ADD_SS = 2,       // secret add
    SEND_PP = 10,     // pipeline send (placeholder)
    RECV_PP = 11,     // pipeline recv (placeholder)
};

struct OpV1 {
    OpCode opcode = OpCode::NOP;
    u32 op_id32 = 0;  // deterministic SGIR op id
};

}  // namespace uvcc


