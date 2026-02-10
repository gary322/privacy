#pragma once

#include "uvcc/types.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace uvcc {

class ByteWriter {
   public:
    void write_u8(u8 x) { buf_.push_back(x); }

    void write_u16_le(u16 x) {
        buf_.push_back(static_cast<u8>(x & 0xFF));
        buf_.push_back(static_cast<u8>((x >> 8) & 0xFF));
    }

    void write_u32_le(u32 x) {
        buf_.push_back(static_cast<u8>(x & 0xFF));
        buf_.push_back(static_cast<u8>((x >> 8) & 0xFF));
        buf_.push_back(static_cast<u8>((x >> 16) & 0xFF));
        buf_.push_back(static_cast<u8>((x >> 24) & 0xFF));
    }

    void write_u64_le(u64 x) {
        for (int i = 0; i < 8; i++) buf_.push_back(static_cast<u8>((x >> (8 * i)) & 0xFF));
    }

    void write_bytes(const void* p, std::size_t n) {
        const auto* b = static_cast<const u8*>(p);
        buf_.insert(buf_.end(), b, b + n);
    }

    void write_bytes(const std::vector<u8>& v) { write_bytes(v.data(), v.size()); }

    template <std::size_t N>
    void write_bytes(const BytesN<N>& v) {
        write_bytes(v.v.data(), v.v.size());
    }

    const std::vector<u8>& bytes() const { return buf_; }
    std::size_t size() const { return buf_.size(); }

   private:
    std::vector<u8> buf_;
};

class ByteReader {
   public:
    ByteReader(const void* p, std::size_t n) : p_(static_cast<const u8*>(p)), n_(n), off_(0) {}

    u8 read_u8() {
        require_(1);
        return p_[off_++];
    }

    u16 read_u16_le() {
        require_(2);
        const u16 a = static_cast<u16>(p_[off_ + 0]);
        const u16 b = static_cast<u16>(p_[off_ + 1]);
        off_ += 2;
        return static_cast<u16>(a | (b << 8));
    }

    u32 read_u32_le() {
        require_(4);
        const u32 b0 = static_cast<u32>(p_[off_ + 0]);
        const u32 b1 = static_cast<u32>(p_[off_ + 1]);
        const u32 b2 = static_cast<u32>(p_[off_ + 2]);
        const u32 b3 = static_cast<u32>(p_[off_ + 3]);
        off_ += 4;
        return (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    u64 read_u64_le() {
        require_(8);
        u64 x = 0;
        for (int i = 0; i < 8; i++) x |= (static_cast<u64>(p_[off_ + static_cast<std::size_t>(i)]) << (8 * i));
        off_ += 8;
        return x;
    }

    void read_bytes(void* dst, std::size_t n) {
        require_(n);
        auto* d = static_cast<u8*>(dst);
        for (std::size_t i = 0; i < n; i++) d[i] = p_[off_ + i];
        off_ += n;
    }

    std::size_t remaining() const { return n_ - off_; }

   private:
    void require_(std::size_t need) const {
        if (off_ + need > n_) throw std::runtime_error("ByteReader: out of range");
    }

    const u8* p_;
    std::size_t n_;
    std::size_t off_;
};

}  // namespace uvcc


