#include "uvcc/lift.h"

#include "uvcc/bytes.h"

#include <stdexcept>

namespace uvcc {

LiftMU64VecV1 parse_tlv_lift_m_u64vec_v1(const std::vector<u8>& value) {
    if (value.size() < 16) throw std::runtime_error("TLV_LIFT_M_U64VEC_V1 too small");
    ByteReader r(value.data(), value.size());
    LiftMU64VecV1 out;
    out.fss_id = r.read_u64_le();
    out.q_count = r.read_u32_le();
    out.producer_edge_id = r.read_u8();
    const u8 reserved1 = r.read_u8();
    const u16 reserved2 = r.read_u16_le();
    if (reserved1 != 0 || reserved2 != 0) throw std::runtime_error("TLV_LIFT_M_U64VEC_V1 reserved fields must be 0");
    if (out.q_count == 0) throw std::runtime_error("TLV_LIFT_M_U64VEC_V1 q_count must be >=1");
    if (out.producer_edge_id > 2) throw std::runtime_error("TLV_LIFT_M_U64VEC_V1 producer_edge_id out of range");
    const std::size_t want_len = 16u + 8u * static_cast<std::size_t>(out.q_count);
    if (value.size() != want_len) throw std::runtime_error("TLV_LIFT_M_U64VEC_V1 length mismatch");
    out.m_u64.resize(out.q_count);
    for (std::size_t i = 0; i < out.m_u64.size(); i++) out.m_u64[i] = r.read_u64_le();
    return out;
}

LiftCompU64VecV1 parse_tlv_lift_comp_u64vec_v1(const std::vector<u8>& value) {
    if (value.size() < 16) throw std::runtime_error("TLV_LIFT_COMP_U64VEC_V1 too small");
    ByteReader r(value.data(), value.size());
    LiftCompU64VecV1 out;
    out.fss_id = r.read_u64_le();
    out.q_count = r.read_u32_le();
    out.component_id = r.read_u8();
    const u8 reserved1 = r.read_u8();
    const u16 reserved2 = r.read_u16_le();
    if (reserved1 != 0 || reserved2 != 0) throw std::runtime_error("TLV_LIFT_COMP_U64VEC_V1 reserved fields must be 0");
    if (out.q_count == 0) throw std::runtime_error("TLV_LIFT_COMP_U64VEC_V1 q_count must be >=1");
    if (out.component_id > 2) throw std::runtime_error("TLV_LIFT_COMP_U64VEC_V1 component_id out of range");
    const std::size_t want_len = 16u + 8u * static_cast<std::size_t>(out.q_count);
    if (value.size() != want_len) throw std::runtime_error("TLV_LIFT_COMP_U64VEC_V1 length mismatch");
    out.comp_u64.resize(out.q_count);
    for (std::size_t i = 0; i < out.comp_u64.size(); i++) out.comp_u64[i] = r.read_u64_le();
    return out;
}

}  // namespace uvcc


