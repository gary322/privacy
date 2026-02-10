#include "uvcc/ids.h"
#include "uvcc/sha256.h"
#include "uvcc/testkit.h"
#include "uvcc/transport.h"

#include <cstring>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

struct FakeClock final : public uvcc::ClockV1 {
    std::uint64_t t_ms = 0;
    std::uint64_t now_ms() override { return t_ms; }
};

struct MemDuplex {
    std::deque<std::vector<uvcc::u8>> a_to_b;
    std::deque<std::vector<uvcc::u8>> b_to_a;
    bool drop_first_ack_b_to_a = false;
};

struct MemEndpoint final : public uvcc::RawConn {
    MemDuplex* d = nullptr;
    bool is_a = true;

    void send_bytes(const std::vector<uvcc::u8>& bytes) override {
        if (!d) return;
        if (!is_a && d->drop_first_ack_b_to_a) {
            if (bytes.size() >= 4 && std::memcmp(bytes.data(), "ACK1", 4) == 0) {
                d->drop_first_ack_b_to_a = false;
                return;  // drop
            }
        }
        if (is_a)
            d->a_to_b.push_back(bytes);
        else
            d->b_to_a.push_back(bytes);
    }

    bool poll_recv(std::vector<uvcc::u8>* out) override {
        if (!d || !out) return false;
        auto& q = is_a ? d->b_to_a : d->a_to_b;
        if (q.empty()) return false;
        *out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

static uvcc::u64 sid_hash64(const uvcc::Sid32& sid) {
    const uvcc::Hash32 h = uvcc::sha256(sid.v.data(), sid.v.size());
    // Interpret first 8 bytes as LE64.
    uvcc::u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<uvcc::u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

static uvcc::FrameV1 make_test_frame(
    const uvcc::Sid32& sid,
    uvcc::u8 src,
    uvcc::u8 dst,
    uvcc::u64 stream_id64,
    uvcc::u32 op_id32,
    uvcc::u32 epoch_id32,
    uvcc::u32 chunk_idx,
    uvcc::u32 chunk_count,
    uvcc::u64 logical_msg_id64,
    std::vector<uvcc::u8> payload) {
    uvcc::FrameV1 f;
    f.hdr.msg_class = 0x21;
    f.hdr.payload_kind = 0x66;
    f.hdr.sid_hash64 = sid_hash64(sid);
    f.hdr.stream_id64 = stream_id64;
    f.hdr.op_id32 = op_id32;
    f.hdr.epoch_id32 = epoch_id32;
    f.hdr.src_party = src;
    f.hdr.dst_party = dst;
    f.hdr.flags = 0x0001;  // HAS_TRAILER_HASH
    f.hdr.chunk_idx = chunk_idx;
    f.hdr.chunk_count = chunk_count;
    f.hdr.logical_msg_id64 = logical_msg_id64;
    f.hdr.payload_codec = 0x00000099u;
    f.hdr.payload_words_u64 = 0;
    f.payload = std::move(payload);
    f.hdr.msg_id32 = uvcc::derive_msg_id32_v1(
        sid, stream_id64, src, dst, /*msg_class=*/0x21, /*payload_kind=*/0x66, op_id32, chunk_idx, chunk_count);
    return f;
}

int main() {
    const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
    FakeClock clk;
    MemDuplex d;
    MemEndpoint ep0;
    ep0.d = &d;
    ep0.is_a = true;
    MemEndpoint ep1;
    ep1.d = &d;
    ep1.is_a = false;

    int p1_accept = 0;
    int p1_ack_send = 0;
    int p0_ack_accept = 0;
    int p0_send_first = 0;

    uvcc::TransportCallbacksV1 c0;
    c0.on_send_first = [&](const uvcc::FrameHdrV1&, const uvcc::Hash32&) { p0_send_first++; };
    c0.on_ack_accept = [&](const uvcc::AckMsgV1&) { p0_ack_accept++; };
    uvcc::TransportCallbacksV1 c1;
    c1.on_accept = [&](const uvcc::FrameHdrV1&, const uvcc::Hash32&) { p1_accept++; };
    c1.on_ack_send = [&](const uvcc::AckMsgV1&) { p1_ack_send++; };

    // Party0: next=1, prev=2 (unused)
    uvcc::TransportV1 t0(sid, /*self_party=*/0, /*to_prev=*/nullptr, /*to_next=*/&ep0, c0, &clk);
    // Party1: prev=0, next=2 (unused)
    uvcc::TransportV1 t1(sid, /*self_party=*/1, /*to_prev=*/&ep1, /*to_next=*/nullptr, c1, &clk);

    // Case 1: normal send -> accept -> ACK -> sender clears pending.
    {
        const auto f = make_test_frame(sid, 0, 1, /*stream*/ 0x1111, /*op*/ 0x2222, /*epoch*/ 0, 0, 1, /*lmsg*/ 0xABC, {1, 2, 3});
        t0.send_frame_reliable(f);
        t1.poll();
        t0.poll();
        if (p1_accept != 1) return fail("p1_accept should be 1");
        if (p1_ack_send != 1) return fail("p1_ack_send should be 1");
        if (p0_ack_accept != 1) return fail("p0_ack_accept should be 1");
        if (p0_send_first != 1) return fail("p0_send_first should be 1");
    }

    // Case 2: drop first ACK; sender retransmits; receiver dedups and re-ACKs.
    {
        d.drop_first_ack_b_to_a = true;
        const auto f = make_test_frame(sid, 0, 1, /*stream*/ 0x3333, /*op*/ 0x4444, /*epoch*/ 0, 0, 1, /*lmsg*/ 0xDEF, {9, 9, 9});
        t0.send_frame_reliable(f);
        t1.poll();  // accept + (dropped) ACK
        // Advance past RTO0 (50ms)
        clk.t_ms += 60;
        t0.poll();  // retransmit
        t1.poll();  // duplicate accept -> re-ACK
        t0.poll();  // accept ACK
        if (p1_accept != 2) return fail("p1_accept should be 2 (one per unique msg_id32)");
        // Note: receiver resends ACK on duplicates but MUST NOT emit another ACK_SEND leaf; our callback tracks the leaf.
        if (p1_ack_send != 2) return fail("p1_ack_send should be 2 (one per unique msg_id32)");
        if (p0_ack_accept != 2) return fail("p0_ack_accept should be 2");
        if (p0_send_first != 2) return fail("p0_send_first should be 2 (retransmit must not increment)");
    }

    return 0;
}


