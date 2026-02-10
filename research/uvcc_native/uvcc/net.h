#pragma once

#include "uvcc/types.h"

#include <string>

namespace uvcc {

// Phase 1 placeholder: later phases will implement RawConn + exactly-once frame transport.
struct PartyEndpointV1 {
    std::string host;
    u16 port = 0;
};

}  // namespace uvcc


