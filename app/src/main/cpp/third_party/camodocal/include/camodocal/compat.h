#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace camodocal {

inline bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    return std::equal(
        a.begin(),
        a.end(),
        b.begin(),
        [](char ca, char cb) {
            return std::tolower(static_cast<unsigned char>(ca)) ==
                   std::tolower(static_cast<unsigned char>(cb));
        });
}

}  // namespace camodocal
