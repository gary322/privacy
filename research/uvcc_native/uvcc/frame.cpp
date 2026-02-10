#include "uvcc/frame.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <cstring>

namespace uvcc {
namespace {

inline void write_magic(ByteWriter& w, const char* s4) { w.write_bytes(s4, 4); }

inline void require_len(std::size_t got, std::size_t want, const char* what) {
    if (got < want) throw std::runtime_error(std::string("short ") + what);
}

}  // namespace

std::vector<u8> frame_hdr_encode_v1(const FrameHdrV1& h) {
    ByteWriter w;
    write_magic(w, "UVCC");
    w.write_u16_le(1);  // version
    w.write_u8(h.msg_class);
    w.write_u8(h.payload_kind);
    w.write_u64_le(h.sid_hash64);
    w.write_u64_le(h.stream_id64);
    w.write_u32_le(h.msg_id32);
    w.write_u32_le(h.op_id32);
    w.write_u32_le(h.epoch_id32);
    w.write_u8(h.src_party);
    w.write_u8(h.dst_party);
    w.write_u16_le(h.flags);
    w.write_u32_le(h.chunk_idx);
    w.write_u32_le(h.chunk_count);
    w.write_u64_le(h.logical_msg_id64);
    w.write_u32_le(h.payload_codec);
    w.write_u32_le(h.payload_words_u64);
    w.write_u32_le(h.payload_bytes);
    w.write_u32_le(static_cast<u32>(FRAME_HDR_V1_BYTES));
    w.write_u32_le(static_cast<u32>(FRAME_TRL_V1_BYTES));
    w.write_u32_le(0);          // reserved0
    w.write_u64_le(0);          // reserved1
    w.write_u64_le(0);          // reserved2
    if (w.size() != FRAME_HDR_V1_BYTES) throw std::runtime_error("frame_hdr_encode_v1 size mismatch");
    return w.bytes();
}

FrameHdrV1 frame_hdr_decode_v1(const u8* bytes, std::size_t len) {
    require_len(len, FRAME_HDR_V1_BYTES, "frame hdr");
    ByteReader r(bytes, FRAME_HDR_V1_BYTES);
    char magic[4];
    r.read_bytes(magic, 4);
    if (std::memcmp(magic, "UVCC", 4) != 0) throw std::runtime_error("bad frame hdr magic");
    const u16 version = r.read_u16_le();
    if (version != 1) throw std::runtime_error("bad frame hdr version");

    FrameHdrV1 h;
    h.msg_class = r.read_u8();
    h.payload_kind = r.read_u8();
    h.sid_hash64 = r.read_u64_le();
    h.stream_id64 = r.read_u64_le();
    h.msg_id32 = r.read_u32_le();
    h.op_id32 = r.read_u32_le();
    h.epoch_id32 = r.read_u32_le();
    h.src_party = r.read_u8();
    h.dst_party = r.read_u8();
    h.flags = r.read_u16_le();
    h.chunk_idx = r.read_u32_le();
    h.chunk_count = r.read_u32_le();
    h.logical_msg_id64 = r.read_u64_le();
    h.payload_codec = r.read_u32_le();
    h.payload_words_u64 = r.read_u32_le();
    h.payload_bytes = r.read_u32_le();
    const u32 hdr_bytes = r.read_u32_le();
    const u32 trl_bytes = r.read_u32_le();
    const u32 reserved0 = r.read_u32_le();
    const u64 reserved1 = r.read_u64_le();
    const u64 reserved2 = r.read_u64_le();

    if (hdr_bytes != FRAME_HDR_V1_BYTES) throw std::runtime_error("hdr_bytes mismatch");
    if (trl_bytes != FRAME_TRL_V1_BYTES) throw std::runtime_error("trl_bytes mismatch");
    if (reserved0 != 0 || reserved1 != 0 || reserved2 != 0) throw std::runtime_error("reserved fields must be 0");
    if ((h.flags & 0x0001u) == 0) throw std::runtime_error("HAS_TRAILER_HASH flag must be set in v1");
    if (h.chunk_count == 0) throw std::runtime_error("chunk_count must be >= 1");
    if (h.chunk_idx >= h.chunk_count) throw std::runtime_error("chunk_idx out of range");
    return h;
}

std::vector<u8> frame_trl_encode_v1(const FrameTrlV1& t) {
    ByteWriter w;
    write_magic(w, "TLR1");
    w.write_u32_le(static_cast<u32>(FRAME_TRL_V1_BYTES));
    w.write_bytes(t.frame_hash32);
    w.write_u32_le(t.crc32);
    w.write_u32_le(0);  // reserved
    if (w.size() != FRAME_TRL_V1_BYTES) throw std::runtime_error("frame_trl_encode_v1 size mismatch");
    return w.bytes();
}

FrameTrlV1 frame_trl_decode_v1(const u8* bytes, std::size_t len) {
    require_len(len, FRAME_TRL_V1_BYTES, "frame trl");
    ByteReader r(bytes, FRAME_TRL_V1_BYTES);
    char magic[4];
    r.read_bytes(magic, 4);
    if (std::memcmp(magic, "TLR1", 4) != 0) throw std::runtime_error("bad frame trl magic");
    const u32 trl_bytes = r.read_u32_le();
    if (trl_bytes != FRAME_TRL_V1_BYTES) throw std::runtime_error("trl_bytes mismatch");
    FrameTrlV1 t;
    r.read_bytes(t.frame_hash32.v.data(), t.frame_hash32.v.size());
    t.crc32 = r.read_u32_le();
    const u32 reserved = r.read_u32_le();
    if (reserved != 0) throw std::runtime_error("trl reserved must be 0");
    return t;
}

Hash32 frame_hash32_v1(const std::vector<u8>& hdr_bytes, const std::vector<u8>& payload_bytes) {
    ByteWriter w;
    const char* dom = "uvcc.framehash.v1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(hdr_bytes);
    w.write_bytes(payload_bytes);
    return sha256(w.bytes().data(), w.bytes().size());
}

std::vector<u8> frame_encode_v1(const FrameV1& f) {
    FrameHdrV1 h = f.hdr;
    h.payload_bytes = static_cast<u32>(f.payload.size());
    // For codec BYTES32, payload_words_u64 is required to be 0; otherwise leave as provided.
    const std::vector<u8> hdr_b = frame_hdr_encode_v1(h);
    const Hash32 fh = frame_hash32_v1(hdr_b, f.payload);
    FrameTrlV1 t = f.trl;
    t.frame_hash32 = fh;
    const std::vector<u8> trl_b = frame_trl_encode_v1(t);
    std::vector<u8> out;
    out.reserve(hdr_b.size() + f.payload.size() + trl_b.size());
    out.insert(out.end(), hdr_b.begin(), hdr_b.end());
    out.insert(out.end(), f.payload.begin(), f.payload.end());
    out.insert(out.end(), trl_b.begin(), trl_b.end());
    return out;
}

FrameV1 frame_decode_v1(const std::vector<u8>& bytes) {
    if (bytes.size() < FRAME_HDR_V1_BYTES + FRAME_TRL_V1_BYTES) throw std::runtime_error("frame too small");
    const FrameHdrV1 h = frame_hdr_decode_v1(bytes.data(), bytes.size());
    const std::size_t want_total = FRAME_HDR_V1_BYTES + static_cast<std::size_t>(h.payload_bytes) + FRAME_TRL_V1_BYTES;
    if (bytes.size() != want_total) throw std::runtime_error("frame size mismatch");

    const std::vector<u8> hdr_b(bytes.begin(), bytes.begin() + FRAME_HDR_V1_BYTES);
    const std::vector<u8> payload(bytes.begin() + FRAME_HDR_V1_BYTES, bytes.begin() + FRAME_HDR_V1_BYTES + h.payload_bytes);
    const FrameTrlV1 t = frame_trl_decode_v1(bytes.data() + FRAME_HDR_V1_BYTES + h.payload_bytes, bytes.size() - (FRAME_HDR_V1_BYTES + h.payload_bytes));

    const Hash32 fh = frame_hash32_v1(hdr_b, payload);
    if (fh != t.frame_hash32) throw std::runtime_error("frame_hash32 mismatch");

    FrameV1 f;
    f.hdr = h;
    f.payload = payload;
    f.trl = t;
    return f;
}

}  // namespace uvcc


