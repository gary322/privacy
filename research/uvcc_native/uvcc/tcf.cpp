#include "uvcc/tcf.h"

#include "uvcc/bytes.h"
#include "uvcc/ids.h"
#include "uvcc/sha256.h"

#include <cstring>
#include <stdexcept>

namespace uvcc {
namespace {

inline u64 le64_0_7(const Hash32& h) {
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

}  // namespace

u64 TcfV0aEngineV1::sid_hash64_(const Sid32& sid32) {
    const Hash32 h = sha256(sid32.v.data(), sid32.v.size());
    return le64_0_7(h);
}

u64 TcfV0aEngineV1::logical_msg_id64_(const Sid32& sid32, u64 stream_id64, u8 payload_kind, u32 op_id32) {
    // As in privacy_new.txt: LE64(H64("uvcc.lmsg.v1"||sid||stream_id||payload_kind||op_id32))
    ByteWriter w;
    const char* dom = "uvcc.lmsg.v1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid32);
    w.write_u64_le(stream_id64);
    w.write_u8(payload_kind);
    w.write_u32_le(op_id32);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return le64_0_7(h);
}

u64 TcfV0aEngineV1::stream_id64_(const Sid32& sid32, u32 op_id32) {
    // stream_id64 := H64("uvcc.stream.tcf.v0a"||sid||LE32(op_id32))
    ByteWriter w;
    const char* dom = "uvcc.stream.tcf.v0a";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid32);
    w.write_u32_le(op_id32);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return le64_0_7(h);
}

u64 TcfV0aEngineV1::prg_u64_(const Hash32& seed, const Sid32& sid32, u32 op_id32, u8 role, u32 word_idx) {
    // Deterministic PRG:
    // u64 := LE64(SHA256("UVCC_TCF_PRG_V1"||seed||sid||LE32(op)||U8(role)||LE32(word_idx))[0..7])
    ByteWriter w;
    const char* dom = "UVCC_TCF_PRG_V1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(seed);
    w.write_bytes(sid32);
    w.write_u32_le(op_id32);
    w.write_u8(role);
    w.write_u32_le(word_idx);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return le64_0_7(h);
}

std::vector<u64> TcfV0aEngineV1::prg_u64_vec_(const Hash32& seed, const Sid32& sid32, u32 op_id32, u8 role, u32 n_words) {
    std::vector<u64> out;
    out.resize(static_cast<std::size_t>(n_words));
    for (u32 i = 0; i < n_words; i++) out[static_cast<std::size_t>(i)] = prg_u64_(seed, sid32, op_id32, role, i);
    return out;
}

std::vector<u8> TcfV0aEngineV1::encode_u64_vec_le_(const std::vector<u64>& v) {
    ByteWriter w;
    for (u64 x : v) w.write_u64_le(x);
    return w.bytes();
}

std::vector<u64> TcfV0aEngineV1::decode_u64_vec_le_(const std::vector<u8>& b) {
    if (b.size() % 8 != 0) throw std::runtime_error("u64 vec payload must be multiple of 8 bytes");
    const std::size_t n = b.size() / 8;
    ByteReader r(b.data(), b.size());
    std::vector<u64> out;
    out.resize(n);
    for (std::size_t i = 0; i < n; i++) out[i] = r.read_u64_le();
    return out;
}

void TcfV0aEngineV1::matmul_u64_(const u64* A, const u64* B, u64* C, u32 m, u32 k, u32 n) {
    for (u32 i = 0; i < m; i++) {
        for (u32 j = 0; j < n; j++) {
            u64 acc = 0;
            for (u32 t = 0; t < k; t++) acc = static_cast<u64>(acc + (A[i * k + t] * B[t * n + j]));
            C[i * n + j] = acc;
        }
    }
}

void TcfV0aEngineV1::maybe_apply_pending_(u32 op_id32) {
    auto pit = pending_by_op_.find(op_id32);
    if (pit == pending_by_op_.end()) return;
    auto it = tasks_.find(op_id32);
    if (it == tasks_.end()) return;
    Task& t = it->second;
    if (t.done || t.have_c_hi) {
        pending_by_op_.erase(pit);
        return;
    }
    const FrameHdrV1& hdr = pit->second.first;
    const std::vector<u8>& payload = pit->second.second;
    // Reuse normal receive path.
    on_deliver(hdr, payload);
    pending_by_op_.erase(pit);
}

void TcfV0aEngineV1::start(u32 op_id32, u32 epoch_id32, u32 m, u32 k, u32 n) {
    if (transport_ == nullptr) throw std::runtime_error("TCF requires transport");
    if (self_ > 2) throw std::runtime_error("bad self_party");
    if (tasks_.find(op_id32) != tasks_.end()) throw std::runtime_error("duplicate TCF start op_id32");
    if (m == 0 || k == 0 || n == 0) throw std::runtime_error("TCF dims must be >0");

    const u32 lo_comp = static_cast<u32>(self_);
    const u32 hi_comp = static_cast<u32>((static_cast<u32>(self_) + 1u) % 3u);
    (void)lo_comp;
    (void)hi_comp;

    Task t;
    t.op_id32 = op_id32;
    t.epoch_id32 = epoch_id32;
    t.m = m;
    t.k = k;
    t.n = n;

    // Generate additive shares for A and B for our two local components (lo,hi).
    const u32 a_words = static_cast<u32>(m * k);
    const u32 b_words = static_cast<u32>(k * n);
    const u32 c_words = static_cast<u32>(m * n);

    t.triple.m = m;
    t.triple.k = k;
    t.triple.n = n;
    t.triple.A.rows = m;
    t.triple.A.cols = k;
    t.triple.B.rows = k;
    t.triple.B.cols = n;
    t.triple.C.rows = m;
    t.triple.C.cols = n;

    // role bytes: 'A' and 'B' domain separation.
    t.triple.A.lo = prg_u64_vec_(seeds_.seed_lo, sid_sub_, op_id32, static_cast<u8>('A'), a_words);
    t.triple.A.hi = prg_u64_vec_(seeds_.seed_hi, sid_sub_, op_id32, static_cast<u8>('A'), a_words);
    t.triple.B.lo = prg_u64_vec_(seeds_.seed_lo, sid_sub_, op_id32, static_cast<u8>('B'), b_words);
    t.triple.B.hi = prg_u64_vec_(seeds_.seed_hi, sid_sub_, op_id32, static_cast<u8>('B'), b_words);

    // Compute C_lo (component self_) as:
    //   A_lo*B_lo + A_lo*B_hi + A_hi*B_lo
    std::vector<u64> term0(c_words), term1(c_words), term2(c_words);
    matmul_u64_(t.triple.A.lo.data(), t.triple.B.lo.data(), term0.data(), m, k, n);
    matmul_u64_(t.triple.A.lo.data(), t.triple.B.hi.data(), term1.data(), m, k, n);
    matmul_u64_(t.triple.A.hi.data(), t.triple.B.lo.data(), term2.data(), m, k, n);

    t.triple.C.lo.resize(c_words);
    t.triple.C.hi.resize(c_words);
    for (std::size_t i = 0; i < static_cast<std::size_t>(c_words); i++) {
        t.triple.C.lo[i] = static_cast<u64>(term0[i] + term1[i] + term2[i]);
        t.triple.C.hi[i] = 0;
    }

    // Send C_lo to prev party to replicate component self_.
    const u8 prev_party = static_cast<u8>((static_cast<int>(self_) + 2) % 3);
    const u8 next_party = static_cast<u8>((static_cast<int>(self_) + 1) % 3);
    const u64 stream = stream_id64_(sid_sub_, op_id32);
    const u32 msg_send = derive_msg_id32_v1(
        sid_sub_,
        stream,
        /*src_party=*/self_,
        /*dst_party=*/prev_party,
        /*msg_class=*/0x21,
        /*payload_kind=*/PAYLOAD_KIND_TCF_C_REPL_V0A,
        /*op_id32=*/op_id32,
        /*chunk_idx=*/0,
        /*chunk_count=*/1);
    const u32 msg_recv = derive_msg_id32_v1(
        sid_sub_,
        stream,
        /*src_party=*/next_party,
        /*dst_party=*/self_,
        /*msg_class=*/0x21,
        /*payload_kind=*/PAYLOAD_KIND_TCF_C_REPL_V0A,
        /*op_id32=*/op_id32,
        /*chunk_idx=*/0,
        /*chunk_count=*/1);
    t.recv_c_msg_id32 = msg_recv;

    FrameV1 f;
    f.hdr.msg_class = 0x21;
    f.hdr.payload_kind = PAYLOAD_KIND_TCF_C_REPL_V0A;
    f.hdr.sid_hash64 = sid_hash64_(sid_sub_);
    f.hdr.stream_id64 = stream;
    f.hdr.msg_id32 = msg_send;
    f.hdr.op_id32 = op_id32;
    f.hdr.epoch_id32 = epoch_id32;
    f.hdr.src_party = self_;
    f.hdr.dst_party = prev_party;
    f.hdr.flags = 0x0001;
    f.hdr.chunk_idx = 0;
    f.hdr.chunk_count = 1;
    f.hdr.logical_msg_id64 = logical_msg_id64_(sid_sub_, stream, PAYLOAD_KIND_TCF_C_REPL_V0A, op_id32);
    f.hdr.payload_codec = 0x00000001u;  // U64_LE_ARRAY
    f.hdr.payload_words_u64 = c_words;
    f.payload = encode_u64_vec_le_(t.triple.C.lo);
    transport_->send_frame_reliable(std::move(f));
    t.sent_c = true;

    tasks_.emplace(op_id32, std::move(t));
    maybe_apply_pending_(op_id32);
}

void TcfV0aEngineV1::on_deliver(const FrameHdrV1& hdr, const std::vector<u8>& full_payload) {
    if (hdr.msg_class != 0x21) return;
    if (hdr.payload_kind != PAYLOAD_KIND_TCF_C_REPL_V0A) return;
    if (hdr.chunk_count != 1 || hdr.chunk_idx != 0) throw std::runtime_error("TCF recv unexpected chunking");

    auto it = tasks_.find(hdr.op_id32);
    if (it == tasks_.end()) {
        if (pending_by_op_.find(hdr.op_id32) == pending_by_op_.end()) {
            pending_by_op_.emplace(hdr.op_id32, std::make_pair(hdr, full_payload));
        }
        return;
    }
    Task& t = it->second;
    if (t.done) return;
    if (hdr.msg_id32 != t.recv_c_msg_id32) throw std::runtime_error("TCF recv msg_id32 mismatch");
    if (hdr.epoch_id32 != t.epoch_id32) throw std::runtime_error("TCF recv epoch mismatch");

    const auto v = decode_u64_vec_le_(full_payload);
    const std::size_t c_words = static_cast<std::size_t>(t.m) * static_cast<std::size_t>(t.n);
    if (v.size() != c_words) throw std::runtime_error("TCF recv length mismatch");
    t.triple.C.hi = v;
    t.have_c_hi = true;
    t.done = true;
}

bool TcfV0aEngineV1::is_done(u32 op_id32) const {
    auto it = tasks_.find(op_id32);
    if (it == tasks_.end()) return false;
    return it->second.done;
}

BeaverTripleU64MatV1 TcfV0aEngineV1::take_triple(u32 op_id32) {
    auto it = tasks_.find(op_id32);
    if (it == tasks_.end()) throw std::runtime_error("TCF triple not found");
    if (!it->second.done) throw std::runtime_error("TCF triple not done");
    auto out = std::move(it->second.triple);
    tasks_.erase(it);
    return out;
}

}  // namespace uvcc


