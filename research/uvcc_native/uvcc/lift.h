#pragma once

#include "uvcc/types.h"

#include <cstdint>
#include <vector>

namespace uvcc {

// LIFT batch TLV types (privacy_new.txt §"UVCC_LIFT_BATCH_TLV_V1").
constexpr u8 TLV_LIFT_M_U64VEC_V1 = 0x01;
constexpr u8 TLV_LIFT_COMP_U64VEC_V1 = 0x02;
constexpr u8 TLV_ACK_PIGGYBACK_V1 = 0xF0;

struct LiftMU64VecV1 {
    u64 fss_id = 0;
    u32 q_count = 0;
    u8 producer_edge_id = 0;  // 0=edge01, 1=edge12, 2=edge20
    std::vector<u64> m_u64;
};

struct LiftCompU64VecV1 {
    u64 fss_id = 0;
    u32 q_count = 0;
    u8 component_id = 0;  // 0,1,2
    std::vector<u64> comp_u64;
};

LiftMU64VecV1 parse_tlv_lift_m_u64vec_v1(const std::vector<u8>& value);
LiftCompU64VecV1 parse_tlv_lift_comp_u64vec_v1(const std::vector<u8>& value);

}  // namespace uvcc


