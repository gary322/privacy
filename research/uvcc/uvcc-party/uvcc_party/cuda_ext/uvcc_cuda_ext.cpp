#include <torch/extension.h>

#include <vector>

// CUDA implementations live in the .cu TU.
std::vector<torch::Tensor> uvcc_dpf_stage1_w16_cuda(torch::Tensor keyrec_u8);
torch::Tensor uvcc_dcf_stage2_w16_cuda(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8,
    torch::Tensor frontier_acc_u8);
torch::Tensor uvcc_dcf_full_w8_cuda(torch::Tensor keyrec_u8);
torch::Tensor uvcc_dpf_full_w8_bitpack32_cuda(torch::Tensor keyrec_u8);
torch::Tensor uvcc_dcf_full_w8_bitpack32_cuda(torch::Tensor keyrec_u8);
torch::Tensor uvcc_dpf_stage2_w16_bitpack32_cuda(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8);
torch::Tensor uvcc_dcf_stage2_w16_bitpack32_cuda(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8,
    torch::Tensor frontier_acc_u8);
torch::Tensor uvcc_dpf_eval_point_w8_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16);
torch::Tensor uvcc_dpf_eval_point_w16_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16);
torch::Tensor uvcc_dcf_eval_point_w8_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16);
torch::Tensor uvcc_dcf_eval_point_w16_batch_cuda(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16);
std::vector<torch::Tensor> uvcc_trunc_apply_u64_cuda(
    torch::Tensor C1_pub_i64,
    torch::Tensor R1_lo_i64,
    torch::Tensor R1_hi_i64,
    torch::Tensor carry_lo_i64,
    torch::Tensor carry_hi_i64,
    torch::Tensor ov_lo_i64,
    torch::Tensor ov_hi_i64,
    int64_t add_const_u64,
    int64_t party_id);
torch::Tensor uvcc_oplut_phase2_w8_record_cuda(torch::Tensor record_u8, torch::Tensor u_pub_u8, torch::Tensor table_i64);
torch::Tensor uvcc_oplut_phase2_w16_record_cuda(torch::Tensor record_u8, torch::Tensor u_pub_i16, torch::Tensor table_i64);
std::vector<torch::Tensor> uvcc_gf2_and_prepare_pack32_cuda(
    torch::Tensor x_lo,
    torch::Tensor x_hi,
    torch::Tensor y_lo,
    torch::Tensor y_hi,
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi);
std::vector<torch::Tensor> uvcc_gf2_and_finish_pack32_cuda(
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi,
    torch::Tensor c_lo,
    torch::Tensor c_hi,
    torch::Tensor e_pub,
    torch::Tensor f_pub,
    int64_t party_id);
std::vector<torch::Tensor> uvcc_a2b_sub_prepare_pack32_cuda(
    torch::Tensor rj_lo,
    torch::Tensor rj_hi,
    torch::Tensor bj_lo,
    torch::Tensor bj_hi,
    torch::Tensor aj_lo,
    torch::Tensor aj_hi,
    torch::Tensor bjT_lo,
    torch::Tensor bjT_hi);
std::vector<torch::Tensor> uvcc_a2b_sub_finish_pack32_cuda(
    torch::Tensor rj_lo,
    torch::Tensor rj_hi,
    torch::Tensor bj_lo,
    torch::Tensor bj_hi,
    torch::Tensor aj_lo,
    torch::Tensor aj_hi,
    torch::Tensor bjT_lo,
    torch::Tensor bjT_hi,
    torch::Tensor cj_lo,
    torch::Tensor cj_hi,
    torch::Tensor e_pub,
    torch::Tensor f_pub,
    torch::Tensor cj_public_mask,
    int64_t party_id);
torch::Tensor uvcc_a2b_pack_c_lo_u8_cuda(torch::Tensor x_lo_i64, torch::Tensor r_lo_i64);
torch::Tensor uvcc_a2b_pack_c_lo_u16_cuda(torch::Tensor x_lo_i64, torch::Tensor r_lo_i64);
torch::Tensor uvcc_a2b_cpub_to_cjmask_u8_cuda(torch::Tensor c_pub_u8);
torch::Tensor uvcc_a2b_cpub_to_cjmask_u16_cuda(torch::Tensor c_pub_i16);
torch::Tensor uvcc_matmul_u64_cuda(torch::Tensor A_i64, torch::Tensor B_i64);

static std::vector<torch::Tensor> dpf_stage1_w16(torch::Tensor keyrec_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dpf_stage1_w16_cuda(keyrec_u8);
}

static torch::Tensor dcf_stage2_w16(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8,
    torch::Tensor frontier_acc_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dcf_stage2_w16_cuda(keyrec_u8, frontier_seed_lo_i64, frontier_seed_hi_i64, frontier_t_u8, frontier_acc_u8);
}

static torch::Tensor dcf_full_w8(torch::Tensor keyrec_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dcf_full_w8_cuda(keyrec_u8);
}

static torch::Tensor dpf_eval_point_w8_batch(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  TORCH_CHECK(keyrecs_blob_u8.is_cuda(), "keyrecs_blob_u8 must be CUDA");
  TORCH_CHECK(x_pub_u16_i16.is_cuda(), "x_pub_u16_i16 must be CUDA");
  return uvcc_dpf_eval_point_w8_batch_cuda(keyrecs_blob_u8, key_stride_bytes, x_pub_u16_i16);
}

static torch::Tensor dpf_eval_point_w16_batch(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  TORCH_CHECK(keyrecs_blob_u8.is_cuda(), "keyrecs_blob_u8 must be CUDA");
  TORCH_CHECK(x_pub_u16_i16.is_cuda(), "x_pub_u16_i16 must be CUDA");
  return uvcc_dpf_eval_point_w16_batch_cuda(keyrecs_blob_u8, key_stride_bytes, x_pub_u16_i16);
}

static torch::Tensor dcf_eval_point_w8_batch(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  TORCH_CHECK(keyrecs_blob_u8.is_cuda(), "keyrecs_blob_u8 must be CUDA");
  TORCH_CHECK(x_pub_u16_i16.is_cuda(), "x_pub_u16_i16 must be CUDA");
  return uvcc_dcf_eval_point_w8_batch_cuda(keyrecs_blob_u8, key_stride_bytes, x_pub_u16_i16);
}

static torch::Tensor dcf_eval_point_w16_batch(torch::Tensor keyrecs_blob_u8, int64_t key_stride_bytes, torch::Tensor x_pub_u16_i16) {
  TORCH_CHECK(keyrecs_blob_u8.is_cuda(), "keyrecs_blob_u8 must be CUDA");
  TORCH_CHECK(x_pub_u16_i16.is_cuda(), "x_pub_u16_i16 must be CUDA");
  return uvcc_dcf_eval_point_w16_batch_cuda(keyrecs_blob_u8, key_stride_bytes, x_pub_u16_i16);
}

static std::vector<torch::Tensor> trunc_apply_u64(
    torch::Tensor C1_pub_i64,
    torch::Tensor R1_lo_i64,
    torch::Tensor R1_hi_i64,
    torch::Tensor carry_lo_i64,
    torch::Tensor carry_hi_i64,
    torch::Tensor ov_lo_i64,
    torch::Tensor ov_hi_i64,
    int64_t add_const_u64,
    int64_t party_id) {
  TORCH_CHECK(C1_pub_i64.is_cuda(), "C1_pub_i64 must be CUDA");
  TORCH_CHECK(R1_lo_i64.is_cuda(), "R1_lo_i64 must be CUDA");
  TORCH_CHECK(R1_hi_i64.is_cuda(), "R1_hi_i64 must be CUDA");
  TORCH_CHECK(carry_lo_i64.is_cuda(), "carry_lo_i64 must be CUDA");
  TORCH_CHECK(carry_hi_i64.is_cuda(), "carry_hi_i64 must be CUDA");
  TORCH_CHECK(ov_lo_i64.is_cuda(), "ov_lo_i64 must be CUDA");
  TORCH_CHECK(ov_hi_i64.is_cuda(), "ov_hi_i64 must be CUDA");
  return uvcc_trunc_apply_u64_cuda(C1_pub_i64, R1_lo_i64, R1_hi_i64, carry_lo_i64, carry_hi_i64, ov_lo_i64, ov_hi_i64, add_const_u64, party_id);
}

static torch::Tensor dpf_full_w8_bitpack32(torch::Tensor keyrec_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dpf_full_w8_bitpack32_cuda(keyrec_u8);
}

static torch::Tensor dcf_full_w8_bitpack32(torch::Tensor keyrec_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dcf_full_w8_bitpack32_cuda(keyrec_u8);
}

static torch::Tensor dpf_stage2_w16_bitpack32(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dpf_stage2_w16_bitpack32_cuda(keyrec_u8, frontier_seed_lo_i64, frontier_seed_hi_i64, frontier_t_u8);
}

static torch::Tensor dcf_stage2_w16_bitpack32(
    torch::Tensor keyrec_u8,
    torch::Tensor frontier_seed_lo_i64,
    torch::Tensor frontier_seed_hi_i64,
    torch::Tensor frontier_t_u8,
    torch::Tensor frontier_acc_u8) {
  TORCH_CHECK(keyrec_u8.is_cuda(), "keyrec_u8 must be CUDA");
  return uvcc_dcf_stage2_w16_bitpack32_cuda(keyrec_u8, frontier_seed_lo_i64, frontier_seed_hi_i64, frontier_t_u8, frontier_acc_u8);
}

static torch::Tensor oplut_phase2_w8_record(torch::Tensor record_u8, torch::Tensor u_pub_u8, torch::Tensor table_i64) {
  TORCH_CHECK(record_u8.is_cuda(), "record_u8 must be CUDA");
  TORCH_CHECK(u_pub_u8.is_cuda(), "u_pub_u8 must be CUDA");
  TORCH_CHECK(table_i64.is_cuda(), "table_i64 must be CUDA");
  return uvcc_oplut_phase2_w8_record_cuda(record_u8, u_pub_u8, table_i64);
}

static torch::Tensor oplut_phase2_w16_record(torch::Tensor record_u8, torch::Tensor u_pub_i16, torch::Tensor table_i64) {
  TORCH_CHECK(record_u8.is_cuda(), "record_u8 must be CUDA");
  TORCH_CHECK(u_pub_i16.is_cuda(), "u_pub_i16 must be CUDA");
  TORCH_CHECK(table_i64.is_cuda(), "table_i64 must be CUDA");
  return uvcc_oplut_phase2_w16_record_cuda(record_u8, u_pub_i16, table_i64);
}

static std::vector<torch::Tensor> gf2_and_prepare_pack32(
    torch::Tensor x_lo,
    torch::Tensor x_hi,
    torch::Tensor y_lo,
    torch::Tensor y_hi,
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi) {
  return uvcc_gf2_and_prepare_pack32_cuda(x_lo, x_hi, y_lo, y_hi, a_lo, a_hi, b_lo, b_hi);
}

static std::vector<torch::Tensor> gf2_and_finish_pack32(
    torch::Tensor a_lo,
    torch::Tensor a_hi,
    torch::Tensor b_lo,
    torch::Tensor b_hi,
    torch::Tensor c_lo,
    torch::Tensor c_hi,
    torch::Tensor e_pub,
    torch::Tensor f_pub,
    int64_t party_id) {
  return uvcc_gf2_and_finish_pack32_cuda(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, e_pub, f_pub, party_id);
}

static std::vector<torch::Tensor> a2b_sub_prepare_pack32(
    torch::Tensor rj_lo,
    torch::Tensor rj_hi,
    torch::Tensor bj_lo,
    torch::Tensor bj_hi,
    torch::Tensor aj_lo,
    torch::Tensor aj_hi,
    torch::Tensor bjT_lo,
    torch::Tensor bjT_hi) {
  return uvcc_a2b_sub_prepare_pack32_cuda(rj_lo, rj_hi, bj_lo, bj_hi, aj_lo, aj_hi, bjT_lo, bjT_hi);
}

static std::vector<torch::Tensor> a2b_sub_finish_pack32(
    torch::Tensor rj_lo,
    torch::Tensor rj_hi,
    torch::Tensor bj_lo,
    torch::Tensor bj_hi,
    torch::Tensor aj_lo,
    torch::Tensor aj_hi,
    torch::Tensor bjT_lo,
    torch::Tensor bjT_hi,
    torch::Tensor cj_lo,
    torch::Tensor cj_hi,
    torch::Tensor e_pub,
    torch::Tensor f_pub,
    torch::Tensor cj_public_mask,
    int64_t party_id) {
  return uvcc_a2b_sub_finish_pack32_cuda(rj_lo, rj_hi, bj_lo, bj_hi, aj_lo, aj_hi, bjT_lo, bjT_hi, cj_lo, cj_hi, e_pub, f_pub, cj_public_mask, party_id);
}

static torch::Tensor a2b_pack_c_lo_u8(torch::Tensor x_lo_i64, torch::Tensor r_lo_i64) {
  return uvcc_a2b_pack_c_lo_u8_cuda(x_lo_i64, r_lo_i64);
}

static torch::Tensor a2b_pack_c_lo_u16(torch::Tensor x_lo_i64, torch::Tensor r_lo_i64) {
  return uvcc_a2b_pack_c_lo_u16_cuda(x_lo_i64, r_lo_i64);
}

static torch::Tensor a2b_cpub_to_cjmask_u8(torch::Tensor c_pub_u8) {
  return uvcc_a2b_cpub_to_cjmask_u8_cuda(c_pub_u8);
}

static torch::Tensor a2b_cpub_to_cjmask_u16(torch::Tensor c_pub_i16) {
  return uvcc_a2b_cpub_to_cjmask_u16_cuda(c_pub_i16);
}

static torch::Tensor matmul_u64(torch::Tensor A_i64, torch::Tensor B_i64) {
  TORCH_CHECK(A_i64.is_cuda(), "A_i64 must be CUDA");
  TORCH_CHECK(B_i64.is_cuda(), "B_i64 must be CUDA");
  TORCH_CHECK(A_i64.scalar_type() == torch::kInt64, "A_i64 must be int64");
  TORCH_CHECK(B_i64.scalar_type() == torch::kInt64, "B_i64 must be int64");
  return uvcc_matmul_u64_cuda(A_i64, B_i64);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dpf_stage1_w16", &dpf_stage1_w16, "UVCC DPF stage1 w16 (CUDA)");
  m.def("dcf_stage2_w16", &dcf_stage2_w16, "UVCC DCF stage2 w16 (CUDA)");
  m.def("dcf_full_w8", &dcf_full_w8, "UVCC DCF full-domain w8 (CUDA)");
  m.def("dpf_eval_point_w8_batch", &dpf_eval_point_w8_batch, "UVCC DPF point-eval w8 (batch, CUDA)");
  m.def("dpf_eval_point_w16_batch", &dpf_eval_point_w16_batch, "UVCC DPF point-eval w16 (batch, CUDA)");
  m.def("dcf_eval_point_w8_batch", &dcf_eval_point_w8_batch, "UVCC DCF point-eval w8 (batch, CUDA)");
  m.def("dcf_eval_point_w16_batch", &dcf_eval_point_w16_batch, "UVCC DCF point-eval w16 (batch, CUDA)");
  m.def("trunc_apply_u64", &trunc_apply_u64, "UVCC TRUNC apply u64 (CUDA)");
  m.def("dpf_full_w8_bitpack32", &dpf_full_w8_bitpack32, "UVCC DPF full-domain w8 -> BITPACK32 (CUDA)");
  m.def("dcf_full_w8_bitpack32", &dcf_full_w8_bitpack32, "UVCC DCF full-domain w8 -> BITPACK32 (CUDA)");
  m.def("dpf_stage2_w16_bitpack32", &dpf_stage2_w16_bitpack32, "UVCC DPF stage2 w16 -> BITPACK32 (CUDA)");
  m.def("dcf_stage2_w16_bitpack32", &dcf_stage2_w16_bitpack32, "UVCC DCF stage2 w16 -> BITPACK32 (CUDA)");
  m.def("oplut_phase2_w8_record", &oplut_phase2_w8_record, "UVCC OP_LUT phase2 (w8) from record blob (CUDA)");
  m.def("oplut_phase2_w16_record", &oplut_phase2_w16_record, "UVCC OP_LUT phase2 (w16) from record blob (CUDA)");
  m.def("gf2_and_prepare_pack32", &gf2_and_prepare_pack32, "UVCC GF2 AND prepare (pack32)");
  m.def("gf2_and_finish_pack32", &gf2_and_finish_pack32, "UVCC GF2 AND finish (pack32)");
  m.def("a2b_sub_prepare_pack32", &a2b_sub_prepare_pack32, "UVCC A2B subtract prepare (pack32)");
  m.def("a2b_sub_finish_pack32", &a2b_sub_finish_pack32, "UVCC A2B subtract finish (pack32)");
  m.def("a2b_pack_c_lo_u8", &a2b_pack_c_lo_u8, "UVCC A2B pack c_lo u8 (CUDA)");
  m.def("a2b_pack_c_lo_u16", &a2b_pack_c_lo_u16, "UVCC A2B pack c_lo u16 (CUDA)");
  m.def("a2b_cpub_to_cjmask_u8", &a2b_cpub_to_cjmask_u8, "UVCC A2B pack c_pub->cjmask u8 (CUDA)");
  m.def("a2b_cpub_to_cjmask_u16", &a2b_cpub_to_cjmask_u16, "UVCC A2B pack c_pub->cjmask u16 (CUDA)");
  m.def("matmul_u64", &matmul_u64, "UVCC u64 matmul (mod 2^64) over int64 bit-patterns (CUDA)");
}


