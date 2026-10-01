#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "depth_calib.h"

namespace scan_policy {
// Cached calibration validity is not a new observation. During warmup only
// accepted fits count; an active epoch can also use independent current evidence.
inline bool hasCurrentCalibrationEvidence(bool updated, bool active, bool frozenGood) {
    return updated || (active && frozenGood);
}
// Test the actual working domain, not only one reference q where two different
// scale/shift lines may cross. Bounded sampling keeps this independent of size.
inline bool mappingsAgree(const DepthCalibration& a, const DepthCalibration& b,
                          const float* raw, int count, float relativeTolerance) {
    if (!a.valid || !b.valid || a.inverseDepthModel != b.inverseDepthModel ||
        !raw || count <= 0 || !std::isfinite(relativeTolerance) || relativeTolerance < 0) return false;
    int tested = 0;
    const int step = std::max(1, count / 128);
    for (int i = 0; i < count; i += step) {
        if (!std::isfinite(raw[i])) continue;
        const float za = a.toMetric(raw[i], 0.f), zb = b.toMetric(raw[i], 0.f);
        if (!(za > .08f) || !(zb > .08f)) return false;
        if (std::fabs(za-zb) > relativeTolerance * za) return false;
        ++tested;
    }
    return tested >= 16;
}
// Euclidean camera-to-surface distance, not optical-axis Z or world origin.
inline bool inRange(float z, float rayX, float rayY, float maxMeters,
                    float worldPerMeter = 1.f) {
    if (!std::isfinite(z) || z <= 0.08f || !std::isfinite(rayX) ||
        !std::isfinite(rayY) || !std::isfinite(maxMeters) || maxMeters <= 0.f ||
        !std::isfinite(worldPerMeter) || worldPerMeter <= 0.f) return false;
    const double distanceSquared = double(z) * z *
        (1.0 + double(rayX) * rayX + double(rayY) * rayY);
    const double limit = double(maxMeters) * worldPerMeter;
    return distanceSquared <= limit * limit;
}

// Independent current sparse observations can vindicate a frozen mapping even
// when a newer fit drifted at an old reference value. Never resume on time alone.
inline bool validatesFrozen(const DepthCalibration& frozen,
                            const std::vector<float>& raw,
                            const std::vector<float>& world) {
    if (!frozen.valid || raw.size() != world.size()) return false;
    int valid = 0, good = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (!std::isfinite(raw[i]) || !std::isfinite(world[i]) || world[i] <= .08f) continue;
        ++valid;
        const float z = frozen.toMetric(raw[i], 0.f);
        if (z > 0.f && std::fabs(z - world[i]) <= .08f * world[i]) ++good;
    }
    return valid >= 20 && good >= 20 && good * 5 >= valid * 4;
}
} // namespace scan_policy
