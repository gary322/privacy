// structs_compile.c
// Build (example): clang -O2 -std=c11 structs_compile.c -o uvcc_structs_compile
// This file exists purely to validate size/layout invariants of frozen structs.
// UVCC_REQ_GROUP: uvcc_group_b50b9410e18bf310,uvcc_group_516fb18a3bfe557c,uvcc_group_df382033ede3f858,uvcc_group_2c46210f0240cf5f

#include <stdint.h>

#include "../structs/uvcc_rss_pairs_v1.h"
#include "../structs/uvcc_b2a_pack_v1.h"
#include "../structs/uvcc_a2b_pack_v1.h"
#include "../structs/uvcc_gf2_triples_v1.h"
#include "../structs/uvcc_policy_wire_v1.h"
#include "../structs/uvcc_netframe_v1.h"
#include "../structs/uvcc_nccl_frames_v1.h"
#include "../structs/uvcc_fss_keyrec_v1.h"
#include "../structs/uvcc_fss_plan_v1.h"
#include "../structs/uvcc_fss_dir_v1.h"
#include "../structs/uvcc_fss_block_v1.h"
#include "../structs/uvcc_dpf_dcf_gpu_abi_v1.h"
#include "../structs/uvcc_bool_gpu_abi_v1.h"
#include "../structs/uvcc_oplut_record_v1.h"
#include "../structs/uvcc_oplut_plan_v1.h"
#include "../structs/uvcc_kdf_info_v1.h"
#include "../structs/uvcc_prg_ctx_v1.h"
#include "../structs/uvcc_trunc_pack_v1.h"
#include "../structs/uvcc_sgir_v1.h"
#include "../structs/uvcc_transcript_leaf_v1.h"

int main(void) {
  // If this compiles, _Static_assert checks succeeded.
  return 0;
}


