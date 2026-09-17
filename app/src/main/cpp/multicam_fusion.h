#pragma once

namespace mobilescan3d::multicam {

constexpr int kStatsSlots = 24;

/** u, v, zMetricM, reprojectionPx, confidence, parallaxDeg */
constexpr int kAnchorStride = 6;
constexpr int kMaxStereoAnchors = 96;

}  // namespace mobilescan3d::multicam
