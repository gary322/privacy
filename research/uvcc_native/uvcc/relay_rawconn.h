#pragma once

#include "uvcc/relay_http.h"
#include "uvcc/transport.h"

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>

namespace uvcc {

// Shared mailbox for a given (group_id, receiver_party). It demuxes relay messages by sender.
class RelayMailboxV1 {
   public:
    RelayMailboxV1(std::shared_ptr<RelayHttpClientV1> http, std::string group_id, int receiver_party);

    // Pop next message payload from `sender_party` if available. If empty, performs a short poll
    // to fetch at most one relay message, acks it immediately, and stashes it by sender.
    bool pop_from_sender(int sender_party, std::vector<u8>* out);

    // Pop next message payload from *any* sender (0/1/2). This is useful for transports that
    // don't want to pre-partition receive by peer connection.
    bool pop_any(std::vector<u8>* out);

   private:
    bool debug_ = false;
    std::uint64_t poll_calls_ = 0;
    std::uint64_t recv_msgs_ = 0;

    std::shared_ptr<RelayHttpClientV1> http_;
    std::string group_id_;
    int receiver_ = 0;
    std::mutex mu_;
    std::deque<std::vector<u8>> q_by_sender_[3];
};

// RawConn backed by the UVCC relay HTTP API.
//
// It supports bidirectional traffic:
// - send_bytes enqueues a message (self -> peer)
// - poll_recv pulls messages from peer -> self by reading from RelayMailboxV1.
class RelayRawConnV1 final : public RawConn {
   public:
    RelayRawConnV1(
        std::shared_ptr<RelayHttpClientV1> http,
        std::shared_ptr<RelayMailboxV1> mailbox,
        std::string group_id,
        int self_party,
        int peer_party,
        int ttl_s);

    void send_bytes(const std::vector<u8>& bytes) override;
    bool poll_recv(std::vector<u8>* out) override;

   private:
    std::string make_msg_id_(const std::vector<u8>& bytes);

    std::shared_ptr<RelayHttpClientV1> http_;
    std::shared_ptr<RelayMailboxV1> mailbox_;
    std::string group_id_;
    int self_ = 0;
    int peer_ = 0;
    int ttl_s_ = 3600;
    u64 counter_ = 0;
    std::string nonce_;
};

}  // namespace uvcc


