#include "uvcc/ids.h"

#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // Golden vector from research/PARALLEL.txt §2.2:
    // sid32[i]=i (00..1f), sgir_op_id32=0x11223344, src=1 dst=2, chunk_idx=0 chunk_cnt=1
    // msg_id32 = 0xED23AD38
    uvcc::Sid32 sid{};
    for (int i = 0; i < 32; i++) sid.v[static_cast<std::size_t>(i)] = static_cast<uvcc::u8>(i);

    const uvcc::u32 msg_id = uvcc::derive_msg_id32_liftbatch_v1(
        sid, /*sgir_op_id32=*/0x11223344u, /*src_party=*/1, /*dst_party=*/2, /*chunk_idx16=*/0, /*chunk_cnt16=*/1);
    if (msg_id != 0xED23AD38u) return fail("msg_id32 mismatch: got=0x" + std::to_string(msg_id));
    return 0;
}


