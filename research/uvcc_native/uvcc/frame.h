#pragma once

#include "uvcc/types.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace uvcc {

// Canonical FRAME := HDR || PAYLOAD || TRL (privacy_new.txt §"Canonical DATA frame v1").
struct FrameHdrV1 {
    u8 msg_class = 0x21;     // 0x21 DATA, 0x22 ACK, 0x23 NACK
    u8 payload_kind = 0;     // protocol-specific (e.g., 0x66 for LIFT_BATCH_SEND)
    u64 sid_hash64 = 0;      // deterministic from sid
    u64 stream_id64 = 0;     // deterministic per op/stream
    u32 msg_id32 = 0;        // derived
    u32 op_id32 = 0;         // sgir_op_id32 or derived open_op_id32
    u32 epoch_id32 = 0;      // epoch
    u8 src_party = 0;
    u8 dst_party = 0;
    u16 flags = 0;           // bit0 HAS_TRAILER_HASH must be 1 in v1
    u32 chunk_idx = 0;
    u32 chunk_count = 1;
    u64 logical_msg_id64 = 0;
    u32 payload_codec = 1;
    u32 payload_words_u64 = 0;
    u32 payload_bytes = 0;
};

struct FrameTrlV1 {
    Hash32 frame_hash32{};
    u32 crc32 = 0;  // optional; v1 sets 0 unless enabled
};

struct FrameV1 {
    FrameHdrV1 hdr;
    std::vector<u8> payload;
    FrameTrlV1 trl;
};

constexpr std::size_t FRAME_HDR_V1_BYTES = 96;
constexpr std::size_t FRAME_TRL_V1_BYTES = 48;

std::vector<u8> frame_hdr_encode_v1(const FrameHdrV1& h);
FrameHdrV1 frame_hdr_decode_v1(const u8* bytes, std::size_t len);

std::vector<u8> frame_trl_encode_v1(const FrameTrlV1& t);
FrameTrlV1 frame_trl_decode_v1(const u8* bytes, std::size_t len);

Hash32 frame_hash32_v1(const std::vector<u8>& hdr_bytes, const std::vector<u8>& payload_bytes);

std::vector<u8> frame_encode_v1(const FrameV1& f);
FrameV1 frame_decode_v1(const std::vector<u8>& bytes);

}  // namespace uvcc


