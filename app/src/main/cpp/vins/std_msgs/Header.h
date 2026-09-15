#pragma once

#include <cstdint>
#include <string>

namespace std_msgs {

struct Time {
    double sec = 0.0;
    double toSec() const { return sec; }
};

struct Header {
    std::uint32_t seq = 0;
    Time stamp;
    std::string frame_id;
};

}  // namespace std_msgs
