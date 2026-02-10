#include "uvcc/arena.h"
#include "uvcc/ops_exec.h"
#include "uvcc/slots.h"

#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    uvcc::Arena a;
    uvcc::SlotMapV1 slots(&a);
    auto& x = slots.get_or_create_rss_u64(0x10, 4);
    auto& y = slots.get_or_create_rss_u64(0x11, 4);
    auto& z = slots.get_or_create_rss_u64(0x12, 4);

    for (int i = 0; i < 4; i++) {
        x.lo[i] = static_cast<uvcc::u64>(i);
        x.hi[i] = static_cast<uvcc::u64>(10 + i);
        y.lo[i] = static_cast<uvcc::u64>(100 + i);
        y.hi[i] = static_cast<uvcc::u64>(200 + i);
    }

    uvcc::rss_u64_add(z, x, y);
    if (z.lo[0] != 100 || z.lo[1] != 102 || z.lo[2] != 104 || z.lo[3] != 106) return fail("rss_u64_add lo mismatch");
    if (z.hi[0] != 210 || z.hi[1] != 212 || z.hi[2] != 214 || z.hi[3] != 216) return fail("rss_u64_add hi mismatch");

    uvcc::rss_u64_add_inplace(x, y);
    if (x.lo[0] != 100 || x.lo[1] != 102 || x.lo[2] != 104 || x.lo[3] != 106) return fail("rss_u64_add_inplace lo mismatch");
    if (x.hi[0] != 210 || x.hi[1] != 212 || x.hi[2] != 214 || x.hi[3] != 216) return fail("rss_u64_add_inplace hi mismatch");
    return 0;
}


