#include "uvcc/open.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <cstring>
#include <stdexcept>

namespace uvcc {
namespace {

inline u64 sid_hash64_v1(const Sid32& sid32) {
    const Hash32 h = sha256(sid32.v.data(), sid32.v.size());
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

inline u64 logical_msg_id64_v1(const Sid32& sid32, u64 stream_id64, u8 payload_kind, u32 op_id32) {
    // As in privacy_new.txt: LE64(H64("uvcc.lmsg.v1"||sid||stream_id||payload_kind||op_id32))
    ByteWriter w;
    const char* dom = "uvcc.lmsg.v1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid32);
    w.write_u64_le(stream_id64);
    w.write_u8(payload_kind);
    w.write_u32_le(op_id32);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

inline std::vector<u8> encode_u64_vec_le(const std::vector<u64>& v) {
    ByteWriter w;
    for (u64 x : v) w.write_u64_le(x);
    return w.bytes();
}

inline std::vector<u64> decode_u64_vec_le(const std::vector<u8>& b) {
    if (b.size() % 8 != 0) throw std::runtime_error("u64 vec payload must be multiple of 8 bytes");
    const std::size_t n = b.size() / 8;
    ByteReader r(b.data(), b.size());
    std::vector<u64> out;
    out.resize(n);
    for (std::size_t i = 0; i < n; i++) out[i] = r.read_u64_le();
    return out;
}

}  // namespace

void OpenEngineV1::apply_open_recv_(Task& t, const FrameHdrV1& hdr, const std::vector<u8>& full_payload) {
    // Validate expected recv msg_id32 (strong bind to deterministic identity).
    if (hdr.msg_id32 != t.recv_msg_id32) throw std::runtime_error("open recv msg_id32 mismatch");
    if (hdr.stream_id64 != t.stream_id64) throw std::runtime_error("open recv stream_id64 mismatch");
    if (hdr.epoch_id32 != t.epoch_id32) throw std::runtime_error("open recv epoch mismatch");

    const auto miss = decode_u64_vec_le(full_payload);
    if (miss.size() != t.lo_local.size()) throw std::runtime_error("open recv length mismatch");
    t.pub_out.resize(miss.size());
    for (std::size_t i = 0; i < miss.size(); i++) {
        // u64 wraparound is desired.
        t.pub_out[i] = static_cast<u64>(t.lo_local[i] + t.hi_local[i] + miss[i]);
    }
    t.done = true;
}

void OpenEngineV1::enqueue_open_u64(
    u32 op_id32,
    u32 epoch_id32,
    u64 stream_id64,
    const std::vector<u64>& lo_local,
    const std::vector<u64>& hi_local,
    FrameV1* out_frame_to_next) {
    if (lo_local.size() != hi_local.size()) throw std::runtime_error("open_u64 lo/hi size mismatch");
    if (out_frame_to_next == nullptr) throw std::runtime_error("out_frame_to_next must not be null");
    if (tasks_by_op_.find(op_id32) != tasks_by_op_.end()) throw std::runtime_error("duplicate open op_id32");

    const u8 next_party = static_cast<u8>((static_cast<int>(self_) + 1) % 3);
    const u8 prev_party = static_cast<u8>((static_cast<int>(self_) + 2) % 3);

    // msg_id32 for our send to next.
    const u32 msg_id_send = derive_msg_id32_v1(
        sid_sub_,
        stream_id64,
        /*src_party=*/self_,
        /*dst_party=*/next_party,
        /*msg_class=*/0x21,
        /*payload_kind=*/PAYLOAD_KIND_OPEN_ARITH_SEND,
        /*op_id32=*/op_id32,
        /*chunk_idx=*/0,
        /*chunk_count=*/1);

    // Expected msg_id32 for recv from prev (same params, swapped src/dst).
    const u32 msg_id_recv = derive_msg_id32_v1(
        sid_sub_,
        stream_id64,
        /*src_party=*/prev_party,
        /*dst_party=*/self_,
        /*msg_class=*/0x21,
        /*payload_kind=*/PAYLOAD_KIND_OPEN_ARITH_SEND,
        /*op_id32=*/op_id32,
        /*chunk_idx=*/0,
        /*chunk_count=*/1);

    Task t;
    t.op_id32 = op_id32;
    t.epoch_id32 = epoch_id32;
    t.stream_id64 = stream_id64;
    t.recv_msg_id32 = msg_id_recv;
    t.lo_local = lo_local;
    t.hi_local = hi_local;
    tasks_by_op_.emplace(op_id32, std::move(t));

    FrameV1 f;
    f.hdr.msg_class = 0x21;
    f.hdr.payload_kind = PAYLOAD_KIND_OPEN_ARITH_SEND;
    f.hdr.sid_hash64 = sid_hash64_v1(sid_sub_);
    f.hdr.stream_id64 = stream_id64;
    f.hdr.msg_id32 = msg_id_send;
    f.hdr.op_id32 = op_id32;
    f.hdr.epoch_id32 = epoch_id32;
    f.hdr.src_party = self_;
    f.hdr.dst_party = next_party;
    f.hdr.flags = 0x0001;
    f.hdr.chunk_idx = 0;
    f.hdr.chunk_count = 1;
    f.hdr.logical_msg_id64 = logical_msg_id64_v1(sid_sub_, stream_id64, PAYLOAD_KIND_OPEN_ARITH_SEND, op_id32);
    f.hdr.payload_codec = 0x00000001u;  // U64_LE_ARRAY
    f.hdr.payload_words_u64 = static_cast<u32>(lo_local.size());
    f.payload = encode_u64_vec_le(lo_local);
    *out_frame_to_next = std::move(f);

    // If the peer payload arrived early (before enqueue), apply it now.
    auto pit = pending_by_op_.find(op_id32);
    if (pit != pending_by_op_.end()) {
        auto it2 = tasks_by_op_.find(op_id32);
        if (it2 != tasks_by_op_.end() && !it2->second.done) {
            apply_open_recv_(it2->second, pit->second.first, pit->second.second);
        }
        pending_by_op_.erase(pit);
    }
}

void OpenEngineV1::on_deliver(const FrameHdrV1& hdr, const std::vector<u8>& full_payload) {
    if (hdr.msg_class != 0x21) return;
    if (hdr.payload_kind != PAYLOAD_KIND_OPEN_ARITH_SEND) return;
    // OPEN_ARITH send must be single chunk in v1 usage here.
    if (hdr.chunk_count != 1 || hdr.chunk_idx != 0) throw std::runtime_error("unexpected open chunking");

    // Find matching task by op_id; if not yet registered, stash the message until enqueue.
    auto it = tasks_by_op_.find(hdr.op_id32);
    if (it == tasks_by_op_.end()) {
        // Store only the first arrival; duplicates are idempotent at transport level anyway.
        if (pending_by_op_.find(hdr.op_id32) == pending_by_op_.end()) {
            pending_by_op_.emplace(hdr.op_id32, std::make_pair(hdr, full_payload));
        }
        return;
    }
    Task& t = it->second;
    if (t.done) return;
    apply_open_recv_(t, hdr, full_payload);
}

bool OpenEngineV1::is_done(u32 op_id32) const {
    auto it = tasks_by_op_.find(op_id32);
    if (it == tasks_by_op_.end()) return false;
    return it->second.done;
}

std::vector<u64> OpenEngineV1::take_result_u64(u32 op_id32) {
    auto it = tasks_by_op_.find(op_id32);
    if (it == tasks_by_op_.end()) throw std::runtime_error("open result not found");
    if (!it->second.done) throw std::runtime_error("open not done");
    auto out = std::move(it->second.pub_out);
    tasks_by_op_.erase(it);
    return out;
}

}  // namespace uvcc


