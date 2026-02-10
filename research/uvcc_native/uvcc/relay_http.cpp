#include "uvcc/relay_http.h"

#include "uvcc/base64.h"

#include <curl/curl.h>

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace uvcc {

static void ensure_curl_global_init() {
    static std::once_flag once;
    std::call_once(once, []() {
        // libcurl requires global init before any easy handles are created.
        // Without this, behavior is undefined on some systems and can crash.
        (void)curl_global_init(CURL_GLOBAL_DEFAULT);
    });
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(ch);
        }
    }
    return out;
}

static size_t curl_write_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    const size_t n = size * nmemb;
    auto* s = static_cast<std::string*>(userp);
    s->append(static_cast<const char*>(contents), n);
    return n;
}

static std::string http_post_json(const RelayHttpConfigV1& cfg, const std::string& path, const std::string& body_json, long* http_code_out) {
    if (http_code_out) *http_code_out = 0;
    ensure_curl_global_init();
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string resp;
    resp.reserve(1024);

    const std::string url = cfg.base_url + path;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!cfg.token.empty()) {
        const std::string h = "Authorization: Bearer " + cfg.token;
        headers = curl_slist_append(headers, h.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_json.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body_json.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(cfg.timeout_s * 1000.0));

    // TLS CA bundle (optional).
    if (!cfg.tls_ca_pem_path.empty()) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, cfg.tls_ca_pem_path.c_str());
    }

    const CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        std::string msg = "curl_easy_perform failed: ";
        msg += curl_easy_strerror(rc);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        throw std::runtime_error(msg);
    }
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (http_code_out) *http_code_out = code;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

static std::string http_get(const RelayHttpConfigV1& cfg, const std::string& path, long* http_code_out) {
    if (http_code_out) *http_code_out = 0;
    ensure_curl_global_init();
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");
    std::string resp;
    resp.reserve(256);
    const std::string url = cfg.base_url + path;

    struct curl_slist* headers = nullptr;
    if (!cfg.token.empty()) {
        const std::string h = "Authorization: Bearer " + cfg.token;
        headers = curl_slist_append(headers, h.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(cfg.timeout_s * 1000.0));
    if (!cfg.tls_ca_pem_path.empty()) curl_easy_setopt(curl, CURLOPT_CAINFO, cfg.tls_ca_pem_path.c_str());

    const CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        std::string msg = "curl_easy_perform failed: ";
        msg += curl_easy_strerror(rc);
        if (headers) curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        throw std::runtime_error(msg);
    }
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (http_code_out) *http_code_out = code;
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

static bool json_get_bool(const std::string& s, const std::string& key, bool* out) {
    const std::string pat = "\"" + key + "\":";
    const std::size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    std::size_t i = p + pat.size();
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    if (s.compare(i, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (s.compare(i, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

static bool json_get_string(const std::string& s, const std::string& key, std::string* out) {
    const std::string pat = "\"" + key + "\":";
    const std::size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    std::size_t i = p + pat.size();
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    if (i >= s.size() || s[i] != '"') return false;
    i++;
    std::string v;
    while (i < s.size()) {
        const char ch = s[i++];
        if (ch == '"') break;
        if (ch == '\\') {
            if (i >= s.size()) return false;
            const char esc = s[i++];
            if (esc == '"' || esc == '\\' || esc == '/') v.push_back(esc);
            else if (esc == 'n') v.push_back('\n');
            else if (esc == 'r') v.push_back('\r');
            else if (esc == 't') v.push_back('\t');
            else return false;
        } else {
            v.push_back(ch);
        }
    }
    *out = v;
    return true;
}

static bool json_get_int(const std::string& s, const std::string& key, int* out) {
    const std::string pat = "\"" + key + "\":";
    const std::size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    std::size_t i = p + pat.size();
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) i++;
    bool neg = false;
    if (i < s.size() && s[i] == '-') {
        neg = true;
        i++;
    }
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    long v = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        v = v * 10 + (s[i] - '0');
        i++;
    }
    if (neg) v = -v;
    *out = static_cast<int>(v);
    return true;
}

RelayHttpClientV1::RelayHttpClientV1(RelayHttpConfigV1 cfg) : cfg_(std::move(cfg)) {
    if (cfg_.base_url.empty()) throw std::runtime_error("RelayHttpClientV1: base_url empty");
    if (cfg_.group_id.empty()) throw std::runtime_error("RelayHttpClientV1: group_id empty");
    if (cfg_.timeout_s <= 0.0) throw std::runtime_error("RelayHttpClientV1: timeout_s must be >0");
}

StatusV1 RelayHttpClientV1::healthz() const {
    try {
        long code = 0;
        (void)http_get(cfg_, "/healthz", &code);
        if (code != 200) return StatusV1::Error("relay healthz bad status " + std::to_string(code));
        return StatusV1::Ok();
    } catch (const std::exception& e) {
        return StatusV1::Error(std::string("relay healthz failed: ") + e.what());
    }
}

StatusV1 RelayHttpClientV1::enqueue(
    int sender, int receiver, const std::string& msg_id, const std::vector<u8>& payload, std::optional<int> ttl_s) const {
    try {
        const int ttl = ttl_s.has_value() ? int(*ttl_s) : cfg_.default_ttl_s;
        const std::string payload_b64 = base64_encode(payload);
        std::ostringstream oss;
        oss << "{";
        oss << "\"group_id\":\"" << json_escape(cfg_.group_id) << "\",";
        oss << "\"msg_id\":\"" << json_escape(msg_id) << "\",";
        oss << "\"sender\":" << int(sender) << ",";
        oss << "\"receiver\":" << int(receiver) << ",";
        oss << "\"payload_b64\":\"" << json_escape(payload_b64) << "\",";
        oss << "\"ttl_s\":" << ttl;
        oss << "}";

        long code = 0;
        const std::string resp = http_post_json(cfg_, "/enqueue", oss.str(), &code);
        if (code != 200) {
            return StatusV1::Error("relay enqueue http " + std::to_string(code) + " resp=" + resp);
        }
        bool ok = false;
        if (!json_get_bool(resp, "ok", &ok) || !ok) return StatusV1::Error("relay enqueue not ok resp=" + resp);
        return StatusV1::Ok();
    } catch (const std::exception& e) {
        return StatusV1::Error(std::string("relay enqueue failed: ") + e.what());
    }
}

ResultV1<std::optional<RelayPolledMsgV1>> RelayHttpClientV1::poll(int receiver, double deadline_s) const {
    try {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(6);
        oss << "{";
        oss << "\"group_id\":\"" << json_escape(cfg_.group_id) << "\",";
        oss << "\"receiver\":" << int(receiver) << ",";
        oss << "\"deadline_s\":" << deadline_s;
        oss << "}";

        long code = 0;
        const std::string resp = http_post_json(cfg_, "/poll", oss.str(), &code);
        if (code != 200) {
            return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll http " + std::to_string(code) + " resp=" + resp));
        }
        bool ok = false;
        if (!json_get_bool(resp, "ok", &ok) || !ok) {
            return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll not ok resp=" + resp));
        }
        // Fast path: "msg":null
        if (resp.find("\"msg\":null") != std::string::npos) {
            return ResultV1<std::optional<RelayPolledMsgV1>>(std::optional<RelayPolledMsgV1>{});
        }

        RelayPolledMsgV1 m;
        if (!json_get_string(resp, "msg_id", &m.msg_id)) return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll missing msg_id"));
        if (!json_get_int(resp, "sender", &m.sender)) return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll missing sender"));
        if (!json_get_int(resp, "receiver", &m.receiver)) return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll missing receiver"));
        if (!json_get_string(resp, "lease_token", &m.lease_token)) return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll missing lease_token"));
        std::string payload_b64;
        if (!json_get_string(resp, "payload_b64", &payload_b64)) return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error("relay poll missing payload_b64"));
        m.payload = base64_decode(payload_b64);
        return ResultV1<std::optional<RelayPolledMsgV1>>(std::optional<RelayPolledMsgV1>(std::move(m)));
    } catch (const std::exception& e) {
        return ResultV1<std::optional<RelayPolledMsgV1>>(StatusV1::Error(std::string("relay poll failed: ") + e.what()));
    }
}

StatusV1 RelayHttpClientV1::ack(int receiver, const std::string& msg_id, const std::string& lease_token) const {
    try {
        std::ostringstream oss;
        oss << "{";
        oss << "\"group_id\":\"" << json_escape(cfg_.group_id) << "\",";
        oss << "\"receiver\":" << int(receiver) << ",";
        oss << "\"msg_id\":\"" << json_escape(msg_id) << "\",";
        oss << "\"lease_token\":\"" << json_escape(lease_token) << "\"";
        oss << "}";
        long code = 0;
        const std::string resp = http_post_json(cfg_, "/ack", oss.str(), &code);
        if (code != 200) return StatusV1::Error("relay ack http " + std::to_string(code) + " resp=" + resp);
        bool ok = false;
        if (!json_get_bool(resp, "ok", &ok) || !ok) return StatusV1::Error("relay ack not ok resp=" + resp);
        return StatusV1::Ok();
    } catch (const std::exception& e) {
        return StatusV1::Error(std::string("relay ack failed: ") + e.what());
    }
}

}  // namespace uvcc


