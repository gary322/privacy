#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace uvcc {

// Minimal Status/Result helpers for Phase 5+ modules.
// Earlier phases often use exceptions directly; these helpers let us implement
// "Status-returning" APIs without forcing a refactor.
class StatusV1 {
   public:
    static StatusV1 Ok() { return StatusV1(true, ""); }
    static StatusV1 Error(std::string msg) { return StatusV1(false, std::move(msg)); }

    bool ok() const { return ok_; }
    const std::string& message() const { return msg_; }

    void throw_if_error() const {
        if (!ok_) throw std::runtime_error(msg_.empty() ? "StatusV1 error" : msg_);
    }

   private:
    StatusV1(bool ok, std::string msg) : ok_(ok), msg_(std::move(msg)) {}

    bool ok_ = true;
    std::string msg_;
};

template <typename T>
class ResultV1 {
   public:
    ResultV1(T v) : ok_(true), value_(std::move(v)), status_(StatusV1::Ok()) {}
    ResultV1(StatusV1 s) : ok_(false), value_(), status_(std::move(s)) {
        if (status_.ok()) status_ = StatusV1::Error("ResultV1 constructed with Ok() status but no value");
    }

    bool ok() const { return ok_; }
    const StatusV1& status() const { return status_; }

    const T& value() const {
        if (!ok_) status_.throw_if_error();
        return value_;
    }
    T& value() {
        if (!ok_) status_.throw_if_error();
        return value_;
    }

   private:
    bool ok_ = false;
    T value_{};
    StatusV1 status_ = StatusV1::Error("uninitialized");
};

}  // namespace uvcc


