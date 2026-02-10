#include "uvcc/ids.h"
#include "uvcc/testkit.h"

#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // Golden vector computed from research/privacy_new.txt §"Canonical msg_id32 and chunking".
    //
    // Inputs:
    //   sid = 00..1f
    //   stream_id64 = 0x0123456789ABCDEF
    //   src=1 dst=2 msg_class=0x21 payload_kind=0x66 op_id32=0x11223344 chunk_idx=0 chunk_count=1
    // Expected (computed via Python hashlib):
    //   msg_id32 = 0x8138FC3A
    const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
    const uvcc::u32 msg_id = uvcc::derive_msg_id32_v1(
        sid,
        /*stream_id64=*/0x0123456789ABCDEFULL,
        /*src_party=*/1,
        /*dst_party=*/2,
        /*msg_class=*/0x21,
        /*payload_kind=*/0x66,
        /*op_id32=*/0x11223344u,
        /*chunk_idx=*/0u,
        /*chunk_count=*/1u);
    if (msg_id != 0x8138FC3Au) return fail("msg_id32 mismatch");
    return 0;
}


