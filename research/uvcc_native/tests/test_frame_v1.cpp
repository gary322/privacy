#include "uvcc/frame.h"

#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    uvcc::FrameV1 f;
    f.hdr.msg_class = 0x21;
    f.hdr.payload_kind = 0x66;
    f.hdr.sid_hash64 = 0x0706050403020100ULL;
    f.hdr.stream_id64 = 0x0123456789ABCDEFULL;
    f.hdr.msg_id32 = 0x8138FC3Au;
    f.hdr.op_id32 = 0x11223344u;
    f.hdr.epoch_id32 = 0u;
    f.hdr.src_party = 1;
    f.hdr.dst_party = 2;
    f.hdr.flags = 0x0001;  // HAS_TRAILER_HASH
    f.hdr.chunk_idx = 0;
    f.hdr.chunk_count = 1;
    f.hdr.logical_msg_id64 = 0xAABBCCDD11223344ULL;
    f.hdr.payload_codec = 0x00000099u;
    f.hdr.payload_words_u64 = 0;
    f.payload = {0x00, 0x01, 0x02, 0x03};

    const auto bytes = uvcc::frame_encode_v1(f);
    const auto g = uvcc::frame_decode_v1(bytes);
    if (g.hdr.msg_class != f.hdr.msg_class) return fail("hdr.msg_class mismatch");
    if (g.hdr.payload_kind != f.hdr.payload_kind) return fail("hdr.payload_kind mismatch");
    if (g.hdr.msg_id32 != f.hdr.msg_id32) return fail("hdr.msg_id32 mismatch");
    if (g.hdr.chunk_count != 1 || g.hdr.chunk_idx != 0) return fail("hdr.chunk mismatch");
    if (g.payload != f.payload) return fail("payload mismatch");

    // Corrupt payload: must fail frame_hash check.
    std::vector<uvcc::u8> bad = bytes;
    bad[uvcc::FRAME_HDR_V1_BYTES + 1] ^= 0xFF;
    try {
        (void)uvcc::frame_decode_v1(bad);
        return fail("expected frame_decode_v1 to fail on corrupted payload");
    } catch (const std::exception&) {
        // ok
    }
    return 0;
}


