#pragma once

#include "uvcc/status.h"
#include "uvcc/types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace uvcc {

struct RelayPolledMsgV1 {
    std::string msg_id;
    int sender = 0;
    int receiver = 0;
    std::string lease_token;
    std::vector<u8> payload;
};

struct RelayHttpConfigV1 {
    std::string base_url;         // e.g. "http://host:1234"
    std::string group_id;         // relay group partition key
    std::string token;            // optional Bearer token
    std::string tls_ca_pem_path;  // optional CA bundle path for https/self-signed
    double timeout_s = 10.0;
    int default_ttl_s = 3600;
};

class RelayHttpClientV1 {
   public:
    explicit RelayHttpClientV1(RelayHttpConfigV1 cfg);

    StatusV1 healthz() const;
    StatusV1 enqueue(int sender, int receiver, const std::string& msg_id, const std::vector<u8>& payload, std::optional<int> ttl_s = std::nullopt) const;
    ResultV1<std::optional<RelayPolledMsgV1>> poll(int receiver, double deadline_s) const;
    StatusV1 ack(int receiver, const std::string& msg_id, const std::string& lease_token) const;

    const RelayHttpConfigV1& cfg() const { return cfg_; }

   private:
    RelayHttpConfigV1 cfg_;
};

}  // namespace uvcc


