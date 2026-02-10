#include "uvcc/lift.h"

#include <iostream>
#include <string>
#include <vector>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // Minimal sanity test for TLV_LIFT_M_U64VEC_V1 parsing (privacy_new.txt layout).
    //
    // fss_id=0x1122334455667788, q_count=2, producer_edge_id=1, m=[0,1]
    std::vector<uvcc::u8> v;
    // fss_id LE
    v.insert(v.end(), {0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11});
    // q_count=2
    v.insert(v.end(), {0x02, 0x00, 0x00, 0x00});
    // producer_edge_id=1, reserved1=0, reserved2=0
    v.insert(v.end(), {0x01, 0x00, 0x00, 0x00});
    // m[0]=0
    v.insert(v.end(), {0, 0, 0, 0, 0, 0, 0, 0});
    // m[1]=1
    v.insert(v.end(), {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});

    const auto m = uvcc::parse_tlv_lift_m_u64vec_v1(v);
    if (m.fss_id != 0x1122334455667788ULL) return fail("fss_id mismatch");
    if (m.q_count != 2) return fail("q_count mismatch");
    if (m.producer_edge_id != 1) return fail("producer_edge_id mismatch");
    if (m.m_u64.size() != 2) return fail("m_u64 size mismatch");
    if (m.m_u64[0] != 0 || m.m_u64[1] != 1) return fail("m_u64 values mismatch");
    return 0;
}


