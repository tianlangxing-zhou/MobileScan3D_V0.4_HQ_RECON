#pragma once

#include <cmath>
#include <limits>

// A finite float may still lie outside int range. Reserve a block-sized margin
// for mesh extraction's neighboring lattice accesses before performing any cast.
inline bool checkedGridIndex(float coordinate, int& index) {
    constexpr double low = static_cast<double>(std::numeric_limits<int>::min()) + 16;
    constexpr double high = static_cast<double>(std::numeric_limits<int>::max()) - 16;
    if (!std::isfinite(coordinate) || coordinate < low || coordinate > high) return false;
    index = static_cast<int>(std::floor(coordinate));
    return true;
}
