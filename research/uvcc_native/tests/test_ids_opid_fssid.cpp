#include "uvcc/ids.h"
#include "uvcc/testkit.h"

#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // Golden vectors computed from PARALLEL.txt formulas using Python hashlib.
    //
    // sid_sub = 00..1f
    // global_step_idx = 25
    // phase = 1
    // mb = 7
    // k = 3
    // => sgir_op_id32 = 0x9869D092
    const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
    const uvcc::u32 op_id = uvcc::derive_sgir_op_id32_v1(sid, 25, 1, 7, 3);
    if (op_id != 0x9869D092u) return fail("sgir_op_id32 mismatch");

    // fss_id64 = Trunc64LE(SHA256("UVCC_FSSID_V1"||sid||LE32(step)||LE16(mb)||LE16(op_kind)||LE16(call_idx)||LE32(shard))[0..7])
    // mb=7 op_kind=0x0201 call_idx=5 shard=2
    // => fss_id64 = 0x38A76CF873018E62
    const uvcc::u64 fss = uvcc::derive_fss_id64_v1(sid, 25, 7, 0x0201u, 5, 2);
    if (fss != 0x38A76CF873018E62ULL) return fail("fss_id64 mismatch");
    return 0;
}


