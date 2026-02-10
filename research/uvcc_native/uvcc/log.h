#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

namespace uvcc {

enum class LogLevel : std::uint8_t { kInfo = 1, kWarn = 2, kError = 3 };

// Minimal logger for Phase 0–1 bring-up.
// (Later phases will emit structured transcript leaves; this is just operator-friendly output.)
class Logger {
   public:
    explicit Logger(std::ostream& os = std::cout) : os_(os) {}

    void log(LogLevel lvl, const std::string& msg) {
        std::lock_guard<std::mutex> g(mu_);
        os_ << msg << "\n";
        os_.flush();
    }

    void info(const std::string& msg) { log(LogLevel::kInfo, msg); }
    void warn(const std::string& msg) { log(LogLevel::kWarn, msg); }
    void error(const std::string& msg) { log(LogLevel::kError, msg); }

   private:
    std::mutex mu_;
    std::ostream& os_;
};

}  // namespace uvcc


