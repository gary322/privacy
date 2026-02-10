#include "uvcc/transport.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <cstring>
#include <stdexcept>

namespace uvcc {

namespace {

inline void write_magic(ByteWriter& w, const char* s4) { w.write_bytes(s4, 4); }

inline u64 u64_from_le_bytes(const u8* p8) {
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(p8[i]) << (8 * i));
    return x;
}

inline u64 h64(const std::vector<u8>& preimage) {
    const Hash32 h = sha256(preimage.data(), preimage.size());
    return u64_from_le_bytes(h.v.data());
}

}  // namespace

std::vector<u8> encode_ack_v1(const AckMsgV1& a, const Sid32& sid32) {
    ByteWriter w;
    write_magic(w, "ACK1");
    w.write_u16_le(1);
    w.write_u8(0x22);  // msg_class ACK
    w.write_u8(0);
    w.write_u64_le(a.sid_hash64);
    w.write_u64_le(a.stream_id64);
    w.write_u32_le(a.msg_id32);
    w.write_u32_le(a.op_id32);
    w.write_u32_le(a.epoch_id32);
    w.write_u8(a.src_party);
    w.write_u8(a.dst_party);
    w.write_u16_le(0);
    w.write_bytes(a.frame_hash16.data(), a.frame_hash16.size());

    // ack_hash64 = H64("uvcc.ack.v1"||sid||stream_id||msg_id||frame_hash16)
    ByteWriter pre;
    const char* dom = "uvcc.ack.v1";
    pre.write_bytes(dom, std::strlen(dom));
    pre.write_bytes(sid32);
    pre.write_u64_le(a.stream_id64);
    pre.write_u32_le(a.msg_id32);
    pre.write_bytes(a.frame_hash16.data(), a.frame_hash16.size());
    const u64 ah = h64(pre.bytes());
    w.write_u64_le(ah);

    if (w.size() != 64) throw std::runtime_error("encode_ack_v1 size mismatch");
    return w.bytes();
}

AckMsgV1 decode_ack_v1(const std::vector<u8>& bytes, const Sid32& /*sid32*/) {
    if (bytes.size() != 64) throw std::runtime_error("bad ACK size");
    ByteReader r(bytes.data(), bytes.size());
    char magic[4];
    r.read_bytes(magic, 4);
    if (std::memcmp(magic, "ACK1", 4) != 0) throw std::runtime_error("bad ACK magic");
    const u16 ver = r.read_u16_le();
    if (ver != 1) throw std::runtime_error("bad ACK version");
    const u8 msg_class = r.read_u8();
    (void)r.read_u8();  // reserved
    if (msg_class != 0x22) throw std::runtime_error("bad ACK msg_class");

    AckMsgV1 a;
    a.sid_hash64 = r.read_u64_le();
    a.stream_id64 = r.read_u64_le();
    a.msg_id32 = r.read_u32_le();
    a.op_id32 = r.read_u32_le();
    a.epoch_id32 = r.read_u32_le();
    a.src_party = r.read_u8();
    a.dst_party = r.read_u8();
    (void)r.read_u16_le();  // flags
    r.read_bytes(a.frame_hash16.data(), a.frame_hash16.size());
    (void)r.read_u64_le();  // ack_hash64 (optional to validate)
    return a;
}

std::vector<u8> encode_nack_v1(const NackMsgV1& n) {
    ByteWriter w;
    write_magic(w, "NAK1");
    w.write_u16_le(1);
    w.write_u8(0x23);  // msg_class NACK
    w.write_u8(0);
    w.write_u64_le(n.sid_hash64);
    w.write_u64_le(n.stream_id64);
    w.write_u32_le(n.msg_id32);
    w.write_u32_le(n.op_id32);
    w.write_u32_le(n.epoch_id32);
    w.write_u8(n.src_party);
    w.write_u8(n.dst_party);
    w.write_u16_le(0);
    w.write_bytes(n.frame_hash16.data(), n.frame_hash16.size());
    w.write_u32_le(n.reason_code32);
    w.write_u32_le(n.reason_detail32);
    if (w.size() != 64) throw std::runtime_error("encode_nack_v1 size mismatch");
    return w.bytes();
}

NackMsgV1 decode_nack_v1(const std::vector<u8>& bytes) {
    if (bytes.size() != 64) throw std::runtime_error("bad NACK size");
    ByteReader r(bytes.data(), bytes.size());
    char magic[4];
    r.read_bytes(magic, 4);
    if (std::memcmp(magic, "NAK1", 4) != 0) throw std::runtime_error("bad NACK magic");
    const u16 ver = r.read_u16_le();
    if (ver != 1) throw std::runtime_error("bad NACK version");
    const u8 msg_class = r.read_u8();
    (void)r.read_u8();  // reserved
    if (msg_class != 0x23) throw std::runtime_error("bad NACK msg_class");

    NackMsgV1 n;
    n.sid_hash64 = r.read_u64_le();
    n.stream_id64 = r.read_u64_le();
    n.msg_id32 = r.read_u32_le();
    n.op_id32 = r.read_u32_le();
    n.epoch_id32 = r.read_u32_le();
    n.src_party = r.read_u8();
    n.dst_party = r.read_u8();
    (void)r.read_u16_le();  // flags
    r.read_bytes(n.frame_hash16.data(), n.frame_hash16.size());
    n.reason_code32 = r.read_u32_le();
    n.reason_detail32 = r.read_u32_le();
    return n;
}

TransportV1::TransportV1(Sid32 sid_sub, u8 self_party, RawConn* to_prev, RawConn* to_next, TransportCallbacksV1 cbs, ClockV1* clock)
    : sid_sub_(sid_sub), self_(self_party), prev_(to_prev), next_(to_next), cbs_(std::move(cbs)), clock_(clock) {
    if (clock_ == nullptr) clock_ = &real_clock_;
}

RawConn* TransportV1::conn_for_dst_(u8 dst_party) const {
    // In a 3-party ring, self sends to next or prev.
    const u8 next_party = static_cast<u8>((static_cast<int>(self_) + 1) % 3);
    const u8 prev_party = static_cast<u8>((static_cast<int>(self_) + 2) % 3);
    if (dst_party == next_party) return next_;
    if (dst_party == prev_party) return prev_;
    throw std::runtime_error("invalid dst_party for conn_for_dst_");
}

void TransportV1::send_frame_reliable(FrameV1 frame) {
    const u32 msg_id32 = frame.hdr.msg_id32;
    if (msg_id32 == 0) throw std::runtime_error("msg_id32 must be set by caller");
    if (pending_.find(msg_id32) != pending_.end()) throw std::runtime_error("duplicate send_frame_reliable msg_id32");

    // Encode deterministically and remember bytes for retransmit.
    FrameHdrV1 h = frame.hdr;
    h.payload_bytes = static_cast<u32>(frame.payload.size());
    const std::vector<u8> hdr_b = frame_hdr_encode_v1(h);
    const Hash32 fh = frame_hash32_v1(hdr_b, frame.payload);
    FrameTrlV1 trl;
    trl.frame_hash32 = fh;
    trl.crc32 = 0;
    const std::vector<u8> trl_b = frame_trl_encode_v1(trl);

    std::vector<u8> bytes;
    bytes.reserve(hdr_b.size() + frame.payload.size() + trl_b.size());
    bytes.insert(bytes.end(), hdr_b.begin(), hdr_b.end());
    bytes.insert(bytes.end(), frame.payload.begin(), frame.payload.end());
    bytes.insert(bytes.end(), trl_b.begin(), trl_b.end());

    PendingTx p;
    p.bytes = bytes;
    p.frame_hash32 = fh;
    p.dst_party = frame.hdr.dst_party;
    p.last_send_ms = clock_->now_ms();
    p.tries = 0;
    pending_.emplace(msg_id32, p);

    if (cbs_.on_send_first) cbs_.on_send_first(h, fh);

    RawConn* c = conn_for_dst_(frame.hdr.dst_party);
    c->send_bytes(bytes);
}

void TransportV1::handle_control_(const std::vector<u8>& bytes) {
    if (bytes.size() < 4) return;
    if (std::memcmp(bytes.data(), "ACK1", 4) == 0) {
        const AckMsgV1 a = decode_ack_v1(bytes, sid_sub_);
        if (a.dst_party != self_) return;  // not for us
        if (cbs_.on_ack_accept) cbs_.on_ack_accept(a);
        auto it = pending_.find(a.msg_id32);
        if (it == pending_.end()) return;
        // Validate frame_hash16 matches our pending hash.
        for (int i = 0; i < 16; i++) {
            if (it->second.frame_hash32.v[static_cast<std::size_t>(i)] != a.frame_hash16[static_cast<std::size_t>(i)]) {
                throw std::runtime_error("ACK frame_hash16 mismatch");
            }
        }
        pending_.erase(it);
        return;
    }
    if (std::memcmp(bytes.data(), "NAK1", 4) == 0) {
        const NackMsgV1 n = decode_nack_v1(bytes);
        if (n.dst_party != self_) return;
        throw std::runtime_error("received NACK");
    }
}

void TransportV1::retransmit_(std::uint64_t now_ms) {
    // Remote multi-host bring-up can take seconds (process scheduling, first poll, relay jitter).
    // At scale, parties can be skewed for minutes (oversubscription, PP/TP/NCCL init, OPEN progress).
    // Use conservative retransmit parameters so we don't abort *before* higher-level phase timeouts fire.
    constexpr std::uint64_t RTO0_MS = 50;
    constexpr std::uint64_t RTOMAX_MS = 10000;
    constexpr std::uint32_t TRIES_MAX = 400;

    for (auto& kv : pending_) {
        PendingTx& p = kv.second;
        std::uint64_t rto = RTO0_MS;
        if (p.tries < 31) {
            rto = RTO0_MS * (1ULL << p.tries);
        }
        if (rto > RTOMAX_MS) rto = RTOMAX_MS;
        if (now_ms < p.last_send_ms) continue;
        if ((now_ms - p.last_send_ms) < rto) continue;
        if (p.tries >= TRIES_MAX) throw std::runtime_error("retransmit tries_max exceeded");
        RawConn* c = conn_for_dst_(p.dst_party);
        c->send_bytes(p.bytes);
        p.tries += 1;
        p.last_send_ms = now_ms;
    }
}

void TransportV1::poll() {
    // Receive from both peers (if connected).
    auto pump = [&](RawConn* c) {
        if (c == nullptr) return;
        std::vector<u8> in;
        while (c->poll_recv(&in)) {
            if (in.size() >= 4 && (std::memcmp(in.data(), "ACK1", 4) == 0 || std::memcmp(in.data(), "NAK1", 4) == 0)) {
                handle_control_(in);
                continue;
            }

            FrameV1 f = frame_decode_v1(in);
            const u32 msg_id32 = f.hdr.msg_id32;
            if (msg_id32 == 0) throw std::runtime_error("received msg_id32=0");
            auto it = accepted_.find(msg_id32);
            if (it == accepted_.end()) {
                accepted_.emplace(msg_id32, f.trl.frame_hash32);
                if (cbs_.on_accept) cbs_.on_accept(f.hdr, f.trl.frame_hash32);

                // ACK idempotently.
                AckMsgV1 ack;
                ack.sid_hash64 = f.hdr.sid_hash64;
                ack.stream_id64 = f.hdr.stream_id64;
                ack.msg_id32 = f.hdr.msg_id32;
                ack.op_id32 = f.hdr.op_id32;
                ack.epoch_id32 = f.hdr.epoch_id32;
                ack.src_party = self_;
                ack.dst_party = f.hdr.src_party;
                for (int j = 0; j < 16; j++) ack.frame_hash16[static_cast<std::size_t>(j)] = f.trl.frame_hash32.v[static_cast<std::size_t>(j)];
                const auto ack_bytes = encode_ack_v1(ack, sid_sub_);
                if (cbs_.on_ack_send) cbs_.on_ack_send(ack);
                conn_for_dst_(ack.dst_party)->send_bytes(ack_bytes);

                // Reassembly: accept chunks independently; deliver only when full logical message complete.
                const bool done = reass_.put_chunk(f.hdr.logical_msg_id64, f.hdr.chunk_idx, f.hdr.chunk_count, f.payload);
                if (done) {
                    std::vector<u8> full = reass_.take_message(f.hdr.logical_msg_id64);
                    if (cbs_.on_deliver) cbs_.on_deliver(f.hdr, full);
                }
            } else {
                // Duplicate: require identical hash, then re-ACK.
                if (it->second != f.trl.frame_hash32) throw std::runtime_error("msg_id32 collision with different frame_hash32");
                AckMsgV1 ack;
                ack.sid_hash64 = f.hdr.sid_hash64;
                ack.stream_id64 = f.hdr.stream_id64;
                ack.msg_id32 = f.hdr.msg_id32;
                ack.op_id32 = f.hdr.op_id32;
                ack.epoch_id32 = f.hdr.epoch_id32;
                ack.src_party = self_;
                ack.dst_party = f.hdr.src_party;
                for (int j = 0; j < 16; j++) ack.frame_hash16[static_cast<std::size_t>(j)] = f.trl.frame_hash32.v[static_cast<std::size_t>(j)];
                conn_for_dst_(ack.dst_party)->send_bytes(encode_ack_v1(ack, sid_sub_));
            }
        }
    };
    pump(prev_);
    pump(next_);
    retransmit_(clock_->now_ms());
}

}  // namespace uvcc


