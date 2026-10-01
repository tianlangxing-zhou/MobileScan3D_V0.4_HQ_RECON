#pragma once
#include <algorithm>
#include <cstdint>

namespace scan_policy {
// Sparse VINS observations are intermittent. Missing samples are not a failed
// geometric test; retain confirmations briefly, but never resume on elapsed time
// or count the same exposure twice. An actual contradiction cancels the window.
class FusionEvidence {
    uint64_t first_ = 0, last_ = 0, seen_ = 0;
    int count_ = 0;
public:
    static constexpr uint64_t maxGapNs = 2'000'000'000ULL;
    static constexpr uint64_t maxWindowNs = 10'000'000'000ULL;
    void reset() { first_ = last_ = seen_ = 0; count_ = 0; }
    int count() const { return count_; }
    bool observe(uint64_t ts, bool supported, bool contradicted, int required = 6) {
        if (!ts || (seen_ && ts <= seen_)) return false;
        seen_ = ts;
        if (contradicted || (last_ && (ts - last_ > maxGapNs || ts - first_ > maxWindowNs))) {
            first_ = last_ = 0; count_ = 0;
        }
        if (!supported || contradicted) return false;
        if (!count_) first_ = ts;
        last_ = ts;
        count_ = std::min(std::max(1, required), count_ + 1);
        return count_ >= std::max(1, required);
    }
};
}
