#include "object_tracker.h"

#include <algorithm>
#include <cmath>

void ObjectTracker::reset()
{
    enabled_ = false;
    info_ = TargetTrackInfo{};
}

void ObjectTracker::clear()
{
    reset();
}

void ObjectTracker::select(float u, float v, float medianDepth, uint64_t timestamp)
{
    u = std::max(0.0f, std::min(1.0f, u));
    v = std::max(0.0f, std::min(1.0f, v));

    const float half = 0.08f;
    info_.state = TargetState::TRACKING;
    info_.x0 = std::max(0.0f, u - half);
    info_.y0 = std::max(0.0f, v - half);
    info_.x1 = std::min(1.0f, u + half);
    info_.y1 = std::min(1.0f, v + half);
    info_.confidence = 1.0f;
    info_.trackedPoints = 0;
    info_.inlierRatio = 1.0f;
    info_.medianDepth = medianDepth;
    info_.roiSharpness = 0.0f;
    info_.timestamp = timestamp;
    enabled_ = true;
}

void ObjectTracker::update(float medianDepth, float sharpness, uint64_t timestamp)
{
    if (!enabled_) {
        return;
    }

    if (medianDepth > 0.0f) {
        info_.medianDepth = medianDepth;
    }

    info_.roiSharpness = sharpness;
    info_.timestamp = timestamp;
    info_.state = TargetState::TRACKING;
}

bool ObjectTracker::isEnabled() const
{
    return enabled_;
}

bool ObjectTracker::isTracking() const
{
    return enabled_ && info_.state == TargetState::TRACKING;
}

void ObjectTracker::filterDepth(float* depth, int width, int height, uint64_t timestamp)
{
    if (!depth || width <= 0 || height <= 0 || !isTracking()) {
        return;
    }

    const float target = info_.medianDepth;
    if (target <= 0.0f) {
        return;
    }

    const float margin = std::max(0.15f, target * 0.08f);
    const int x0 = static_cast<int>(info_.x0 * width);
    const int y0 = static_cast<int>(info_.y0 * height);
    const int x1 = static_cast<int>(info_.x1 * width);
    const int y1 = static_cast<int>(info_.y1 * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            if (x < x0 || x > x1 || y < y0 || y > y1) {
                depth[idx] = 0.0f;
                continue;
            }

            const float d = depth[idx];
            if (d < target - margin || d > target + margin) {
                depth[idx] = 0.0f;
            }
        }
    }

    info_.timestamp = timestamp;
}

TargetTrackInfo ObjectTracker::info() const
{
    return info_;
}
