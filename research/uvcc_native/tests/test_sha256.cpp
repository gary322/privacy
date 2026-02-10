#include "uvcc/hex.h"
#include "uvcc/sha256.h"

#include <iostream>
#include <string>

static int fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << "\n";
    return 1;
}

int main() {
    const char* s = "abc";
    const uvcc::Hash32 h = uvcc::sha256(s, 3);
    const std::string hex = uvcc::hex_lower(h);
    const std::string expect = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    if (hex != expect) return fail("sha256('abc') mismatch: got=" + hex + " want=" + expect);
    return 0;
}


