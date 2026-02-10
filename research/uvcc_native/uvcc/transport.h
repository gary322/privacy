#pragma once

#include "uvcc/clock.h"
#include "uvcc/frame.h"
#include "uvcc/reassembly.h"
#include "uvcc/types.h"

#include <cstdint>
#include <array>
#include <functional>
#include <unordered_map>
#include <vector>

namespace uvcc {

// RawConn is the underlying byte pipe. In Phase 2 we keep it abstract so we can:
// - test with in-memory connections, and
// - later bind it to relay HTTP (Prime-friendly) or direct TCP/QUIC.
struct RawConn {
    virtual ~RawConn() = default;
    virtual void send_bytes(const std::vector<u8>& bytes) = 0;
    // Returns true and fills `out` if a message was received.
    virtual bool poll_recv(std::vector<u8>* out) = 0;
};

// ACK control message (privacy_new.txt §5.1).
struct AckMsgV1 {
    u64 sid_hash64 = 0;
    u64 stream_id64 = 0;
    u32 msg_id32 = 0;
    u32 op_id32 = 0;
    u32 epoch_id32 = 0;
    u8 src_party = 0;  // ACK sender
    u8 dst_party = 0;  // ACK receiver
    std::array<u8, 16> frame_hash16{};
};

// NACK control message (privacy_new.txt §5.2).
struct NackMsgV1 {
    u64 sid_hash64 = 0;
    u64 stream_id64 = 0;
    u32 msg_id32 = 0;
    u32 op_id32 = 0;
    u32 epoch_id32 = 0;
    u8 src_party = 0;
    u8 dst_party = 0;
    std::array<u8, 16> frame_hash16{};
    u32 reason_code32 = 0;
    u32 reason_detail32 = 0;
};

std::vector<u8> encode_ack_v1(const AckMsgV1& a, const Sid32& sid32);
AckMsgV1 decode_ack_v1(const std::vector<u8>& bytes, const Sid32& sid32);
std::vector<u8> encode_nack_v1(const NackMsgV1& n);
NackMsgV1 decode_nack_v1(const std::vector<u8>& bytes);

struct TransportCallbacksV1 {
    // Called when a logical message (possibly reassembled) is delivered.
    // NOTE: transcript hooks will be added in Phase 4; Phase 2 just provides the place to hang them.
    std::function<void(const FrameHdrV1& hdr, const std::vector<u8>& full_payload)> on_deliver;
    std::function<void(const FrameHdrV1& hdr, const Hash32& frame_hash32)> on_accept;
    std::function<void(const FrameHdrV1& hdr, const Hash32& frame_hash32)> on_send_first;
    std::function<void(const AckMsgV1& ack)> on_ack_send;
    std::function<void(const AckMsgV1& ack)> on_ack_accept;
};

// Phase 2 transport: exactly-once accept, tx-cache, retransmit timers (partial; expanded in Phase 3/4).
class TransportV1 {
   public:
    TransportV1(Sid32 sid_sub, u8 self_party, RawConn* to_prev, RawConn* to_next, TransportCallbacksV1 cbs, ClockV1* clock = nullptr);

    // Transport owns internal state with self-referential pointers (clock_ may point to real_clock_).
    // To avoid dangling pointers, we make it non-copyable and non-movable.
    TransportV1(const TransportV1&) = delete;
    TransportV1& operator=(const TransportV1&) = delete;
    TransportV1(TransportV1&&) = delete;
    TransportV1& operator=(TransportV1&&) = delete;

    void send_frame_reliable(FrameV1 frame);
    void poll();

   private:
    RawConn* conn_for_dst_(u8 dst_party) const;
    void handle_control_(const std::vector<u8>& bytes);
    void retransmit_(std::uint64_t now_ms);

    Sid32 sid_sub_;
    u8 self_;
    RawConn* prev_;
    RawConn* next_;
    TransportCallbacksV1 cbs_;
    RealClockV1 real_clock_;
    ClockV1* clock_;

    // accepted[msg_id32] = frame_hash32
    std::unordered_map<u32, Hash32> accepted_;

    struct PendingTx {
        std::vector<u8> bytes;
        Hash32 frame_hash32{};
        u8 dst_party = 0;
        std::uint64_t last_send_ms = 0;
        std::uint32_t tries = 0;
    };
    std::unordered_map<u32, PendingTx> pending_;

    // chunk reassembly by logical_msg_id64
    ReassemblyV1 reass_;
};

}  // namespace uvcc


