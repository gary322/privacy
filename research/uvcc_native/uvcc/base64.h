#pragma once

#include "uvcc/types.h"

#include <string>
#include <vector>

namespace uvcc {

std::string base64_encode(const std::vector<u8>& data);
std::vector<u8> base64_decode(const std::string& s);

}  // namespace uvcc


