#include "uvcc/blv1.h"

#include "uvcc/bytes.h"

#include <cstring>
#include <stdexcept>

namespace uvcc {
namespace {

constexpr u32 BLV1_MAGIC_LE = 0x31564C42u;  // "BLV1" bytes: 42 4C 56 31

inline std::size_t pad4(std::size_t n) { return (4 - (n & 3u)) & 3u; }

}  // namespace

std::vector<u8> blv1_encode(const BLV1Batch& b) {
    if (b.version != 1) throw std::runtime_error("BLV1 version must be 1");

    ByteWriter tlvs_w;
    for (const auto& tlv : b.tlvs) {
        if (tlv.value.size() > 0xFFFFu) throw std::runtime_error("TLV value too large");
        tlvs_w.write_u8(tlv.type);
        tlvs_w.write_u8(tlv.flags);
        tlvs_w.write_u16_le(static_cast<u16>(tlv.value.size()));
        tlvs_w.write_bytes(tlv.value);
        const std::size_t p = pad4(tlv.value.size());
        for (std::size_t i = 0; i < p; i++) tlvs_w.write_u8(0);
    }

    const u32 tlv_count = static_cast<u32>(b.tlvs.size());
    const u32 tlv_bytes = static_cast<u32>(tlvs_w.size());

    ByteWriter out;
    out.write_u32_le(BLV1_MAGIC_LE);
    out.write_u8(b.version);
    out.write_u8(b.flags);
    out.write_u16_le(0);  // reserved0
    out.write_u32_le(tlv_count);
    out.write_u32_le(tlv_bytes);
    out.write_bytes(tlvs_w.bytes());
    return out.bytes();
}

BLV1Batch blv1_decode(const std::vector<u8>& payload) {
    if (payload.size() < 16) throw std::runtime_error("BLV1 payload too small");
    ByteReader r(payload.data(), payload.size());
    const u32 magic = r.read_u32_le();
    if (magic != BLV1_MAGIC_LE) throw std::runtime_error("bad BLV1 magic");
    const u8 version = r.read_u8();
    const u8 flags = r.read_u8();
    const u16 reserved0 = r.read_u16_le();
    if (version != 1) throw std::runtime_error("bad BLV1 version");
    if (reserved0 != 0) throw std::runtime_error("BLV1 reserved0 must be 0");
    const u32 tlv_count = r.read_u32_le();
    const u32 tlv_bytes = r.read_u32_le();
    if (payload.size() != 16u + static_cast<std::size_t>(tlv_bytes)) throw std::runtime_error("BLV1 tlv_bytes mismatch");

    BLV1Batch b;
    b.version = version;
    b.flags = flags;
    b.tlvs.reserve(tlv_count);

    const std::size_t tlv_end_off = 16u + static_cast<std::size_t>(tlv_bytes);
    std::size_t off = 16;
    for (u32 i = 0; i < tlv_count; i++) {
        if (off + 4 > tlv_end_off) throw std::runtime_error("BLV1 truncated TLV header");
        const u8 t = payload[off + 0];
        const u8 f = payload[off + 1];
        const u16 len = static_cast<u16>(payload[off + 2] | (static_cast<u16>(payload[off + 3]) << 8));
        off += 4;
        if (off + static_cast<std::size_t>(len) > tlv_end_off) throw std::runtime_error("BLV1 truncated TLV value");
        TLV tlv;
        tlv.type = t;
        tlv.flags = f;
        tlv.value.assign(payload.begin() + static_cast<std::ptrdiff_t>(off), payload.begin() + static_cast<std::ptrdiff_t>(off + len));
        off += len;
        const std::size_t p = pad4(len);
        if (off + p > tlv_end_off) throw std::runtime_error("BLV1 truncated TLV padding");
        for (std::size_t j = 0; j < p; j++) {
            if (payload[off + j] != 0) throw std::runtime_error("BLV1 padding must be 0");
        }
        off += p;
        b.tlvs.push_back(std::move(tlv));
    }
    if (off != tlv_end_off) throw std::runtime_error("BLV1 parse did not consume tlv_bytes");
    return b;
}

}  // namespace uvcc


