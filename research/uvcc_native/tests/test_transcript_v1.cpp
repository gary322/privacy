#include "uvcc/hex.h"
#include "uvcc/sha256.h"
#include "uvcc/testkit.h"
#include "uvcc/transcript.h"

#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

static uvcc::u64 sid_hash64(const uvcc::Sid32& sid) {
    const uvcc::Hash32 h = uvcc::sha256(sid.v.data(), sid.v.size());
    uvcc::u64 x = 0;
    for (int i = 0; i < 8; i++) x |= (static_cast<uvcc::u64>(h.v[static_cast<std::size_t>(i)]) << (8 * i));
    return x;
}

int main() {
    // Golden vectors computed from an independent Python implementation (see repo history in this session).
    const uvcc::Sid32 sid = uvcc::sid32_seq_00_1f();
    const uvcc::u64 sh64 = sid_hash64(sid);
    if (sh64 != 0x6633c46629cd0d63ULL) return fail("sid_hash64 mismatch");

    uvcc::LeafV1 l1;
    l1.leaf_type = 0x4101;
    l1.epoch_id32 = 0;
    l1.stream_id64 = 0x1111;
    l1.msg_id32 = 0x2222;
    l1.op_id32 = 0x7777;
    l1.src_party = 1;
    l1.dst_party = 2;
    l1.msg_class = 0x21;
    l1.payload_kind = 0x66;
    l1.chunk_idx = 0;
    l1.chunk_count = 1;
    l1.payload_bytes = 123;
    l1.sid_hash64 = sh64;
    l1.frame_hash32 = uvcc::sha256("frame1", 6);

    uvcc::LeafV1 l2 = l1;
    l2.leaf_type = 0x4102;
    l2.msg_id32 = 0x3333;
    l2.payload_bytes = 456;
    l2.frame_hash32 = uvcc::sha256("frame2", 6);

    const auto b1 = uvcc::leaf_bytes_v1(l1);
    const auto b2 = uvcc::leaf_bytes_v1(l2);
    const uvcc::Hash32 k1 = uvcc::leaf_key_v1(sid, l1);
    const uvcc::Hash32 d1 = uvcc::leaf_digest_v1(b1);
    const uvcc::Hash32 k2 = uvcc::leaf_key_v1(sid, l2);
    const uvcc::Hash32 d2 = uvcc::leaf_digest_v1(b2);

    if (uvcc::hex_lower(k1) != "bc29b3573fe3ddbc1995f94c76d13d193a78d9f992b46073bec52283e51ece6e") return fail("leaf_key1 mismatch");
    if (uvcc::hex_lower(d1) != "e05a997cadc8ab3fdd1efe4fe2ba3ea7443516118d05dcc0f6849f7e43d41684") return fail("leaf_digest1 mismatch");
    if (uvcc::hex_lower(k2) != "e458dcf08c5d9f40a2c1fb9903f25f868f1e03bb83492681f0038b176f3d7137") return fail("leaf_key2 mismatch");
    if (uvcc::hex_lower(d2) != "9ca09bfdd940dd49bd9f9c8c812df8038281ef00f7f98cf3fd222814b6d21129") return fail("leaf_digest2 mismatch");

    const uvcc::Hash32 root = uvcc::epoch_root_v1({{k1, d1}, {k2, d2}});
    if (uvcc::hex_lower(root) != "7cc2d0e90bcf18b81a36f21281d829ec573c72329df0be3dbbdee1b38bfe3be3") return fail("epoch_root mismatch");

    // TranscriptStore: order-independence + exactly-once.
    uvcc::TranscriptStoreV1 ts(sid);
    ts.record_leaf(l2);
    ts.record_leaf(l1);
    ts.record_leaf(l1);  // duplicate ok
    const uvcc::Hash32 root2 = ts.epoch_root(0);
    if (root2 != root) return fail("TranscriptStore epoch_root mismatch");
    return 0;
}


