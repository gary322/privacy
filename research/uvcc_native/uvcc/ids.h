#pragma once

#include "uvcc/types.h"

namespace uvcc {

// sid_rep[r] = SHA256("UVCC_SID_REPLICA_V1" || sid_job || LE32(r))[0..31]
Sid32 derive_sid_replica_v1(const Sid32& sid_job, u32 replica_id);

// sid_sub[r,s,t] = SHA256("UVCC_SID_SUB_V1" || sid_rep || U8(stage) || LE16(tp_rank))[0..31]
Sid32 derive_sid_sub_v1(const Sid32& sid_rep, u16 stage, u16 tp_rank);

// msg_id32 = LE32( SHA256(
//   "UVCC_MSGID_LIFTBATCH_V1" ||
//   sid32 ||
//   LE32(sgir_op_id32) ||
//   U8(src_party) || U8(dst_party) ||
//   LE16(chunk_idx16) || LE16(chunk_cnt16)
// )[0..3] )
u32 derive_msg_id32_liftbatch_v1(const Sid32& sid32, u32 sgir_op_id32, u8 src_party, u8 dst_party, u16 chunk_idx16, u16 chunk_cnt16);

// Canonical `msg_id32` derivation from research/privacy_new.txt §"Canonical msg_id32 and chunking":
//
// msg_id32 :=
//   H32("uvcc.msgid.v1" ||
//       sid ||
//       LE64(stream_id64) ||
//       LE8(src_party) || LE8(dst_party) ||
//       LE8(msg_class) || LE8(payload_kind) ||
//       LE32(op_id32) ||
//       LE32(chunk_idx) || LE32(chunk_count))
//
// Where H32 is the first 4 bytes of SHA256(preimage), interpreted as LE32.
u32 derive_msg_id32_v1(const Sid32& sid32, u64 stream_id64, u8 src_party, u8 dst_party, u8 msg_class, u8 payload_kind, u32 op_id32, u32 chunk_idx, u32 chunk_count);

// sgir_op_id32 = Trunc32LE( SHA256(
//   "UVCC_OPID_V1" ||
//   sid_sub ||
//   LE32(global_step_idx) ||
//   U8(phase) ||
//   LE16(mb) ||
//   LE16(k)
// )[0..3] )
u32 derive_sgir_op_id32_v1(const Sid32& sid_sub, u32 global_step_idx, u8 phase, u16 mb, u16 k);

// fss_id64 = Trunc64LE( SHA256(
//   "UVCC_FSSID_V1" ||
//   sid_sub ||
//   LE32(global_step_idx) ||
//   LE16(mb) ||
//   LE16(op_kind_u16) ||
//   LE16(call_idx_u16) ||
//   LE32(tensor_shard)
// )[0..7] )
u64 derive_fss_id64_v1(const Sid32& sid_sub, u32 global_step_idx, u16 mb, u16 op_kind_u16, u16 call_idx_u16, u32 tensor_shard);

}  // namespace uvcc


