#include "uvcc/hex.h"
#include "uvcc/ids.h"

#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    // Golden vectors computed from PARALLEL.txt derivation formulas.
    // Inputs:
    //   sid_job = 00..1f (sid_job[i]=i)
    //   replica_id=7
    //   stage=3
    //   tp_rank=5
    // Expected values computed with Python hashlib:
    //   sid_rep = 54350d2e1cb398d2e53e87dc6a409e402124a1443344f0a9ea3d44c7c066756c
    //   sid_sub = 33e65a985dd53846381106824026b6af1a14a92ff90926251d99ddbdbcb84fa4
    uvcc::Sid32 sid_job{};
    for (int i = 0; i < 32; i++) sid_job.v[static_cast<std::size_t>(i)] = static_cast<uvcc::u8>(i);

    const uvcc::Sid32 sid_rep = uvcc::derive_sid_replica_v1(sid_job, /*replica_id=*/7);
    const uvcc::Sid32 sid_sub = uvcc::derive_sid_sub_v1(sid_rep, /*stage=*/3, /*tp_rank=*/5);

    const std::string sid_rep_hex = uvcc::hex_lower(sid_rep);
    const std::string sid_sub_hex = uvcc::hex_lower(sid_sub);
    const std::string expect_rep = "54350d2e1cb398d2e53e87dc6a409e402124a1443344f0a9ea3d44c7c066756c";
    const std::string expect_sub = "33e65a985dd53846381106824026b6af1a14a92ff90926251d99ddbdbcb84fa4";

    if (sid_rep_hex != expect_rep) return fail("sid_rep mismatch: got=" + sid_rep_hex + " want=" + expect_rep);
    if (sid_sub_hex != expect_sub) return fail("sid_sub mismatch: got=" + sid_sub_hex + " want=" + expect_sub);
    return 0;
}


