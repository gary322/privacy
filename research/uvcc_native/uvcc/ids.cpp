#include "uvcc/ids.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <cstring>
#include <stdexcept>

namespace uvcc {
namespace {

inline void write_ascii(ByteWriter& w, const char* s) { w.write_bytes(s, std::strlen(s)); }

inline u32 u32_from_le_bytes(const u8 b0, const u8 b1, const u8 b2, const u8 b3) {
    return (static_cast<u32>(b0) << 0) | (static_cast<u32>(b1) << 8) | (static_cast<u32>(b2) << 16) | (static_cast<u32>(b3) << 24);
}

}  // namespace

Sid32 derive_sid_replica_v1(const Sid32& sid_job, u32 replica_id) {
    ByteWriter w;
    write_ascii(w, "UVCC_SID_REPLICA_V1");
    w.write_bytes(sid_job);
    w.write_u32_le(replica_id);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    Sid32 out;
    out.v = h.v;
    return out;
}

Sid32 derive_sid_sub_v1(const Sid32& sid_rep, u16 stage, u16 tp_rank) {
    if (static_cast<u32>(stage) > 0xFFu) throw std::runtime_error("stage must fit in U8 for sid_sub_v1");
    ByteWriter w;
    write_ascii(w, "UVCC_SID_SUB_V1");
    w.write_bytes(sid_rep);
    w.write_u8(static_cast<u8>(stage & 0xFFu));
    w.write_u16_le(tp_rank);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    Sid32 out;
    out.v = h.v;
    return out;
}

u32 derive_msg_id32_liftbatch_v1(const Sid32& sid32, u32 sgir_op_id32, u8 src_party, u8 dst_party, u16 chunk_idx16, u16 chunk_cnt16) {
    ByteWriter w;
    write_ascii(w, "UVCC_MSGID_LIFTBATCH_V1");
    w.write_bytes(sid32);
    w.write_u32_le(sgir_op_id32);
    w.write_u8(src_party);
    w.write_u8(dst_party);
    w.write_u16_le(chunk_idx16);
    w.write_u16_le(chunk_cnt16);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    // Interpret SHA256(preimage)[0..3] as little-endian u32.
    return u32_from_le_bytes(h.v[0], h.v[1], h.v[2], h.v[3]);
}

u32 derive_msg_id32_v1(const Sid32& sid32, u64 stream_id64, u8 src_party, u8 dst_party, u8 msg_class, u8 payload_kind, u32 op_id32, u32 chunk_idx, u32 chunk_count) {
    ByteWriter w;
    write_ascii(w, "uvcc.msgid.v1");
    w.write_bytes(sid32);
    w.write_u64_le(stream_id64);
    w.write_u8(src_party);
    w.write_u8(dst_party);
    w.write_u8(msg_class);
    w.write_u8(payload_kind);
    w.write_u32_le(op_id32);
    w.write_u32_le(chunk_idx);
    w.write_u32_le(chunk_count);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return u32_from_le_bytes(h.v[0], h.v[1], h.v[2], h.v[3]);
}

u32 derive_sgir_op_id32_v1(const Sid32& sid_sub, u32 global_step_idx, u8 phase, u16 mb, u16 k) {
    ByteWriter w;
    write_ascii(w, "UVCC_OPID_V1");
    w.write_bytes(sid_sub);
    w.write_u32_le(global_step_idx);
    w.write_u8(phase);
    w.write_u16_le(mb);
    w.write_u16_le(k);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return u32_from_le_bytes(h.v[0], h.v[1], h.v[2], h.v[3]);
}

u64 derive_fss_id64_v1(const Sid32& sid_sub, u32 global_step_idx, u16 mb, u16 op_kind_u16, u16 call_idx_u16, u32 tensor_shard) {
    ByteWriter w;
    write_ascii(w, "UVCC_FSSID_V1");
    w.write_bytes(sid_sub);
    w.write_u32_le(global_step_idx);
    w.write_u16_le(mb);
    w.write_u16_le(op_kind_u16);
    w.write_u16_le(call_idx_u16);
    w.write_u32_le(tensor_shard);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    // Interpret SHA256(preimage)[0..7] as little-endian u64.
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

}  // namespace uvcc


