#include "uvcc/relay_rawconn.h"

#include "uvcc/hex.h"
#include "uvcc/sha256.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

namespace uvcc {

static double now_s() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now().time_since_epoch();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    return static_cast<double>(ns) / 1e9;
}

RelayMailboxV1::RelayMailboxV1(std::shared_ptr<RelayHttpClientV1> http, std::string group_id, int receiver_party)
    : http_(std::move(http)), group_id_(std::move(group_id)), receiver_(receiver_party) {
    if (!http_) throw std::runtime_error("RelayMailboxV1: http is null");
    if (group_id_.empty()) throw std::runtime_error("RelayMailboxV1: group_id empty");
    if (receiver_ < 0 || receiver_ > 2) throw std::runtime_error("RelayMailboxV1: receiver out of range");
    const char* dbg = std::getenv("UVCC_DEBUG_RELAY");
    debug_ = (dbg != nullptr && std::string(dbg) == "1");
    if (debug_) {
        std::cerr << "[relay] mailbox init receiver=" << receiver_ << " group_id=" << group_id_ << " base_url=" << http_->cfg().base_url << "\n";
    }
}

bool RelayMailboxV1::pop_from_sender(int sender_party, std::vector<u8>* out) {
    if (sender_party < 0 || sender_party > 2) throw std::runtime_error("RelayMailboxV1::pop_from_sender: sender out of range");
    if (!out) return false;

    {
        std::lock_guard<std::mutex> g(mu_);
        auto& q = q_by_sender_[sender_party];
        if (!q.empty()) {
            *out = std::move(q.front());
            q.pop_front();
            return true;
        }
    }

    // Fetch at most one message and stash it.
    // Use a short deadline so this behaves like non-blocking poll.
    const double dl = now_s() + 0.2;
    auto r = http_->poll(receiver_, dl);
    if (!r.ok()) throw std::runtime_error(r.status().message());
    if (!r.value().has_value()) return false;
    RelayPolledMsgV1 m = std::move(*r.value());
    // Ack immediately (mirrors uvcc_party/party.py behavior).
    auto st = http_->ack(receiver_, m.msg_id, m.lease_token);
    if (!st.ok()) throw std::runtime_error(st.message());

    if (m.sender < 0 || m.sender > 2) return false;
    {
        std::lock_guard<std::mutex> g(mu_);
        q_by_sender_[m.sender].push_back(std::move(m.payload));
        auto& q = q_by_sender_[sender_party];
        if (!q.empty()) {
            *out = std::move(q.front());
            q.pop_front();
            return true;
        }
    }
    return false;
}

bool RelayMailboxV1::pop_any(std::vector<u8>* out) {
    if (!out) return false;
    // First, drain any already-stashed payload (any sender).
    {
        std::lock_guard<std::mutex> g(mu_);
        for (int s = 0; s < 3; s++) {
            auto& q = q_by_sender_[s];
            if (!q.empty()) {
                *out = std::move(q.front());
                q.pop_front();
                return true;
            }
        }
    }

    // Otherwise, poll once and stash by sender.
    const double dl = now_s() + 0.2;
    poll_calls_ += 1;
    if (debug_ && (poll_calls_ % 50 == 1)) {
        std::cerr << "[relay] poll receiver=" << receiver_ << " group_id=" << http_->cfg().group_id << " calls=" << poll_calls_ << "\n";
    }
    auto r = http_->poll(receiver_, dl);
    if (!r.ok()) throw std::runtime_error(r.status().message());
    if (!r.value().has_value()) return false;
    RelayPolledMsgV1 m = std::move(*r.value());
    recv_msgs_ += 1;
    if (debug_) {
        std::cerr << "[relay] recv receiver=" << receiver_ << " sender=" << m.sender << " msg_id=" << m.msg_id << " bytes=" << m.payload.size()
                  << " recv_msgs=" << recv_msgs_ << "\n";
    }
    auto st = http_->ack(receiver_, m.msg_id, m.lease_token);
    if (!st.ok()) throw std::runtime_error(st.message());
    if (m.sender < 0 || m.sender > 2) return false;
    {
        std::lock_guard<std::mutex> g(mu_);
        q_by_sender_[m.sender].push_back(std::move(m.payload));
        for (int s = 0; s < 3; s++) {
            auto& q = q_by_sender_[s];
            if (!q.empty()) {
                *out = std::move(q.front());
                q.pop_front();
                return true;
            }
        }
    }
    return false;
}

RelayRawConnV1::RelayRawConnV1(
    std::shared_ptr<RelayHttpClientV1> http,
    std::shared_ptr<RelayMailboxV1> mailbox,
    std::string group_id,
    int self_party,
    int peer_party,
    int ttl_s)
    : http_(std::move(http)),
      mailbox_(std::move(mailbox)),
      group_id_(std::move(group_id)),
      self_(self_party),
      peer_(peer_party),
      ttl_s_(ttl_s),
      counter_(0) {
    if (!http_) throw std::runtime_error("RelayRawConnV1: http is null");
    if (!mailbox_) throw std::runtime_error("RelayRawConnV1: mailbox is null");
    if (group_id_.empty()) throw std::runtime_error("RelayRawConnV1: group_id empty");
    if (self_ < 0 || self_ > 2) throw std::runtime_error("RelayRawConnV1: self out of range");
    if (peer_ < 0 || peer_ > 2) throw std::runtime_error("RelayRawConnV1: peer out of range");
    if (ttl_s_ <= 0) ttl_s_ = 3600;

    // Random nonce to avoid relay msg_id collisions across process restarts (relay msg_id is the DB primary key).
    // This does NOT affect transcript determinism (transport uses its own msg_id32 inside frame bytes).
    try {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        const u64 x = gen();
        nonce_ = hex_lower(reinterpret_cast<const u8*>(&x), sizeof(u64));
    } catch (...) {
        const double t = now_s();
        const u64 x = static_cast<u64>(t * 1e9);
        nonce_ = hex_lower(reinterpret_cast<const u8*>(&x), sizeof(u64));
    }
}

std::string RelayRawConnV1::make_msg_id_(const std::vector<u8>& bytes) {
    const Hash32 h = sha256(bytes.data(), bytes.size());
    // Short digest to help debugging collisions.
    const std::string h8 = hex_lower(h.v.data(), 4);
    counter_ += 1;
    return "raw-" + std::to_string(self_) + "to" + std::to_string(peer_) + "-" + nonce_ + "-" + std::to_string(counter_) + "-" + h8;
}

void RelayRawConnV1::send_bytes(const std::vector<u8>& bytes) {
    const std::string mid = make_msg_id_(bytes);
    auto st = http_->enqueue(self_, peer_, mid, bytes, ttl_s_);
    if (!st.ok()) throw std::runtime_error(st.message());
}

bool RelayRawConnV1::poll_recv(std::vector<u8>* out) {
    // We intentionally receive from *any* sender. Transport-level headers include src_party,
    // and this avoids peer-specific mailbox starvation when acks/data arrive out-of-order.
    return mailbox_->pop_any(out);
}

}  // namespace uvcc


