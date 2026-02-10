#include "uvcc/gemm_beaver.h"

#include "uvcc/bytes.h"
#include "uvcc/sha256.h"

#include <cstring>
#include <stdexcept>

namespace uvcc {
namespace {

inline u64 sid_hash64_v1(const Sid32& sid32) {
    const Hash32 h = sha256(sid32.v.data(), sid32.v.size());
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

}  // namespace

u32 BeaverGemmEngineV1::derive_subop_id32_(const Sid32& sid_sub, u32 parent_op_id32, const char* tag) {
    ByteWriter w;
    const char* dom = "UVCC_SUBOP_V1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid_sub);
    w.write_u32_le(parent_op_id32);
    w.write_bytes(tag, std::strlen(tag));
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    return static_cast<u32>(static_cast<u32>(h.v[0]) | (static_cast<u32>(h.v[1]) << 8) | (static_cast<u32>(h.v[2]) << 16) |
                            (static_cast<u32>(h.v[3]) << 24));
}

u64 BeaverGemmEngineV1::derive_stream_id64_(const Sid32& sid_sub, u32 op_id32) {
    // stream_id64 := H64("uvcc.stream.open.v1"||sid||LE32(op_id32))
    ByteWriter w;
    const char* dom = "uvcc.stream.open.v1";
    w.write_bytes(dom, std::strlen(dom));
    w.write_bytes(sid_sub);
    w.write_u32_le(op_id32);
    const Hash32 h = sha256(w.bytes().data(), w.bytes().size());
    u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

void BeaverGemmEngineV1::matmul_u64_(const u64* A, const u64* B, u64* C, u32 m, u32 k, u32 n) {
    for (u32 i = 0; i < m; i++) {
        for (u32 j = 0; j < n; j++) {
            u64 acc = 0;
            for (u32 t = 0; t < k; t++) {
                acc = static_cast<u64>(acc + (A[i * k + t] * B[t * n + j]));
            }
            C[i * n + j] = acc;
        }
    }
}

void BeaverGemmEngineV1::start(BeaverGemmTaskV1 task) {
    if (st_.find(task.op_id32) != st_.end()) throw std::runtime_error("duplicate gemm task op_id32");
    if (transport_ == nullptr || open_ == nullptr) throw std::runtime_error("gemm engine requires transport+open");
    if (task.m == 0 || task.k == 0 || task.n == 0) throw std::runtime_error("gemm dims must be >0");
    if (task.X.rows != task.m || task.X.cols != task.k) throw std::runtime_error("X shape mismatch");
    if (task.Y.rows != task.k || task.Y.cols != task.n) throw std::runtime_error("Y shape mismatch");
    const std::size_t x_words = static_cast<std::size_t>(task.m) * static_cast<std::size_t>(task.k);
    const std::size_t y_words = static_cast<std::size_t>(task.k) * static_cast<std::size_t>(task.n);
    const std::size_t z_words = static_cast<std::size_t>(task.m) * static_cast<std::size_t>(task.n);
    if (task.X.lo.size() != x_words || task.X.hi.size() != x_words) throw std::runtime_error("X size mismatch");
    if (task.Y.lo.size() != y_words || task.Y.hi.size() != y_words) throw std::runtime_error("Y size mismatch");

    const bool triple_missing =
        task.triple.A.lo.empty() && task.triple.A.hi.empty() && task.triple.B.lo.empty() && task.triple.B.hi.empty() && task.triple.C.lo.empty() &&
        task.triple.C.hi.empty();
    if (!triple_missing) {
        if (task.triple.m != task.m || task.triple.k != task.k || task.triple.n != task.n) throw std::runtime_error("triple dims mismatch");
        if (task.triple.A.lo.size() != x_words || task.triple.A.hi.size() != x_words) throw std::runtime_error("A size mismatch");
        if (task.triple.B.lo.size() != y_words || task.triple.B.hi.size() != y_words) throw std::runtime_error("B size mismatch");
        if (task.triple.C.lo.size() != z_words || task.triple.C.hi.size() != z_words) throw std::runtime_error("C size mismatch");
    } else {
        if (tcf_v0a_ == nullptr) throw std::runtime_error("missing beaver triple and no TCF-v0a engine configured");
    }

    State s;
    s.task = std::move(task);
    s.use_tcf_v0a = triple_missing;
    if (s.use_tcf_v0a) {
        // Use a derived sub-op id so callers can still use op_id32 as the GEMM id.
        s.tcf_op_id32 = derive_subop_id32_(sid_sub_, s.task.op_id32, "tcf_v0a");
        tcf_v0a_->start(s.tcf_op_id32, s.task.epoch_id32, s.task.m, s.task.k, s.task.n);
    }
    s.open_e_op_id32 = derive_subop_id32_(sid_sub_, s.task.op_id32, "openE");
    s.open_f_op_id32 = derive_subop_id32_(sid_sub_, s.task.op_id32, "openF");
    s.open_e_stream_id64 = derive_stream_id64_(sid_sub_, s.open_e_op_id32);
    s.open_f_stream_id64 = derive_stream_id64_(sid_sub_, s.open_f_op_id32);
    s.phase = 0;
    st_.emplace(s.task.op_id32, std::move(s));
}

void BeaverGemmEngineV1::tick() {
    for (auto& kv : st_) {
        State& s = kv.second;
        if (s.task.done) continue;
        const u32 m = s.task.m;
        const u32 k = s.task.k;
        const u32 n = s.task.n;
        const std::size_t x_words = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
        const std::size_t y_words = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
        const std::size_t z_words = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

        if (s.use_tcf_v0a) {
            if (tcf_v0a_ == nullptr) throw std::runtime_error("tick: use_tcf_v0a but tcf_v0a_ is null");
            if (!tcf_v0a_->is_done(s.tcf_op_id32)) continue;
            s.task.triple = tcf_v0a_->take_triple(s.tcf_op_id32);
            s.use_tcf_v0a = false;
            // Validate triple sizes now.
            if (s.task.triple.A.lo.size() != x_words || s.task.triple.A.hi.size() != x_words) throw std::runtime_error("TCF A size mismatch");
            if (s.task.triple.B.lo.size() != y_words || s.task.triple.B.hi.size() != y_words) throw std::runtime_error("TCF B size mismatch");
            if (s.task.triple.C.lo.size() != z_words || s.task.triple.C.hi.size() != z_words) throw std::runtime_error("TCF C size mismatch");
        }

        if (s.phase == 0) {
            // E = X - A, F = Y - B
            s.E.rows = m;
            s.E.cols = k;
            s.F.rows = k;
            s.F.cols = n;
            s.E.lo.resize(x_words);
            s.E.hi.resize(x_words);
            s.F.lo.resize(y_words);
            s.F.hi.resize(y_words);
            for (std::size_t i = 0; i < x_words; i++) {
                s.E.lo[i] = static_cast<u64>(s.task.X.lo[i] - s.task.triple.A.lo[i]);
                s.E.hi[i] = static_cast<u64>(s.task.X.hi[i] - s.task.triple.A.hi[i]);
            }
            for (std::size_t i = 0; i < y_words; i++) {
                s.F.lo[i] = static_cast<u64>(s.task.Y.lo[i] - s.task.triple.B.lo[i]);
                s.F.hi[i] = static_cast<u64>(s.task.Y.hi[i] - s.task.triple.B.hi[i]);
            }

            // Enqueue OPEN(E) and send.
            FrameV1 f_e;
            open_->enqueue_open_u64(s.open_e_op_id32, s.task.epoch_id32, s.open_e_stream_id64, s.E.lo, s.E.hi, &f_e);
            transport_->send_frame_reliable(std::move(f_e));
            s.phase = 1;
        } else if (s.phase == 1) {
            if (!open_->is_done(s.open_e_op_id32)) continue;
            s.E_pub = open_->take_result_u64(s.open_e_op_id32);
            // Enqueue OPEN(F) and send.
            FrameV1 f_f;
            open_->enqueue_open_u64(s.open_f_op_id32, s.task.epoch_id32, s.open_f_stream_id64, s.F.lo, s.F.hi, &f_f);
            transport_->send_frame_reliable(std::move(f_f));
            s.phase = 2;
        } else if (s.phase == 2) {
            if (!open_->is_done(s.open_f_op_id32)) continue;
            s.F_pub = open_->take_result_u64(s.open_f_op_id32);

            // Compute Z = C + E*B + A*F + E*F (public term added into share0 only).
            std::vector<u64> term1_lo(z_words), term1_hi(z_words), term2_lo(z_words), term2_hi(z_words), term3_pub(z_words);

            // term1 = E_pub @ B
            matmul_u64_(s.E_pub.data(), s.task.triple.B.lo.data(), term1_lo.data(), m, k, n);
            matmul_u64_(s.E_pub.data(), s.task.triple.B.hi.data(), term1_hi.data(), m, k, n);

            // term2 = A @ F_pub
            matmul_u64_(s.task.triple.A.lo.data(), s.F_pub.data(), term2_lo.data(), m, k, n);
            matmul_u64_(s.task.triple.A.hi.data(), s.F_pub.data(), term2_hi.data(), m, k, n);

            // term3_pub = E_pub @ F_pub
            matmul_u64_(s.E_pub.data(), s.F_pub.data(), term3_pub.data(), m, k, n);

            s.task.Z.rows = m;
            s.task.Z.cols = n;
            s.task.Z.lo.resize(z_words);
            s.task.Z.hi.resize(z_words);
            for (std::size_t i = 0; i < z_words; i++) {
                s.task.Z.lo[i] = static_cast<u64>(s.task.triple.C.lo[i] + term1_lo[i] + term2_lo[i]);
                s.task.Z.hi[i] = static_cast<u64>(s.task.triple.C.hi[i] + term1_hi[i] + term2_hi[i]);
            }

            // Add public term3 into share0 component only (matches replicated-share mapping).
            if (self_ == 0) {
                for (std::size_t i = 0; i < z_words; i++) s.task.Z.lo[i] = static_cast<u64>(s.task.Z.lo[i] + term3_pub[i]);
            } else if (self_ == 2) {
                for (std::size_t i = 0; i < z_words; i++) s.task.Z.hi[i] = static_cast<u64>(s.task.Z.hi[i] + term3_pub[i]);
            }

            s.task.done = true;
            s.phase = 3;
        }
    }
}

bool BeaverGemmEngineV1::is_done(u32 op_id32) const {
    auto it = st_.find(op_id32);
    if (it == st_.end()) return false;
    return it->second.task.done;
}

RSSU64MatV1 BeaverGemmEngineV1::take_result(u32 op_id32) {
    auto it = st_.find(op_id32);
    if (it == st_.end()) throw std::runtime_error("gemm result not found");
    if (!it->second.task.done) throw std::runtime_error("gemm not done");
    auto out = std::move(it->second.task.Z);
    st_.erase(it);
    return out;
}

}  // namespace uvcc


