#include "uvcc/wvole.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <cstring>

namespace uvcc {

static u64 le64_0_7(const Hash32& h) {
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

u64 WarpVoleStubV1::prg_u64_(
    const Hash32& seed, const Sid32& sid32, u32 op_id32, u32 block_id32, WVoleRoleV1 role, u32 word_idx) {
    // u64 := LE64(SHA256("UVCC_WVOLE_PRG_V1"||seed||sid||LE32(op)||LE32(block)||U8(role)||LE32(word_idx))[0..7])
    ByteWriter w;
    const char* dom = "UVCC_WVOLE_PRG_V1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(seed);
    w.write_bytes(sid32);
    w.write_u32_le(op_id32);
    w.write_u32_le(block_id32);
    w.write_u8(static_cast<u8>(role));
    w.write_u32_le(word_idx);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return le64_0_7(h);
}

WVoleBlockV1 WarpVoleStubV1::expand_block(u32 op_id32, u32 block_id32, u32 n_words) const {
    if (self_ > 2) throw std::runtime_error("WarpVoleStubV1: bad self_party");
    if (n_words == 0) throw std::runtime_error("WarpVoleStubV1: n_words must be >0");
    WVoleBlockV1 b;
    b.block_id32 = block_id32;
    b.n_words = n_words;
    b.u_lo.resize(static_cast<std::size_t>(n_words));
    b.u_hi.resize(static_cast<std::size_t>(n_words));
    b.v_lo.resize(static_cast<std::size_t>(n_words));
    b.v_hi.resize(static_cast<std::size_t>(n_words));
    for (u32 i = 0; i < n_words; i++) {
        b.u_lo[static_cast<std::size_t>(i)] = prg_u64_(seeds_.seed_lo, sid_sub_, op_id32, block_id32, WVoleRoleV1::U, i);
        b.u_hi[static_cast<std::size_t>(i)] = prg_u64_(seeds_.seed_hi, sid_sub_, op_id32, block_id32, WVoleRoleV1::U, i);
        b.v_lo[static_cast<std::size_t>(i)] = prg_u64_(seeds_.seed_lo, sid_sub_, op_id32, block_id32, WVoleRoleV1::V, i);
        b.v_hi[static_cast<std::size_t>(i)] = prg_u64_(seeds_.seed_hi, sid_sub_, op_id32, block_id32, WVoleRoleV1::V, i);
    }
    return b;
}

}  // namespace uvcc


