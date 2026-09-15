#include "object_tracker.h"

#include <algorithm>
#include <cmath>

void ObjectTracker::reset()
{
    enabled_ = false;
    info_ = TargetTrackInfo{};
    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;
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
    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;
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

void ObjectTracker::updateFromDepth(const float* depth, int width, int height, uint64_t timestamp)
{
    if (!depth || width <= 0 || height <= 0 || !isTracking()) {
        return;
    }

    const int x0 = static_cast<int>(info_.x0 * width);
    const int y0 = static_cast<int>(info_.y0 * height);
    const int x1 = static_cast<int>(info_.x1 * width);
    const int y1 = static_cast<int>(info_.y1 * height);

    std::vector<float> values;
    values.reserve(static_cast<size_t>((x1 - x0 + 1) * (y1 - y0 + 1)));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float d = depth[static_cast<size_t>(y) * width + x];
            if (std::isfinite(d) && d > 0.0f) {
                values.push_back(d);
            }
        }
    }

    if (!values.empty()) {
        std::sort(values.begin(), values.end());
        info_.medianDepth = values[values.size() / 2];
    }
    info_.timestamp = timestamp;
}

void ObjectTracker::track(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp)
{
    if (!gray || width <= 0 || height <= 0 || !enabled_) {
        return;
    }

    const int x0 = std::max(0, static_cast<int>(info_.x0 * width));
    const int y0 = std::max(0, static_cast<int>(info_.y0 * height));
    const int x1 = std::min(width - 1, static_cast<int>(info_.x1 * width));
    const int y1 = std::min(height - 1, static_cast<int>(info_.y1 * height));
    if (x1 <= x0 || y1 <= y0) {
        info_.state = TargetState::LOST;
        return;
    }

    cv::Mat cur(height, width, CV_8UC1, const_cast<uint8_t*>(gray), static_cast<size_t>(stride));
    const cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);

    if (!havePrev_) {
        prevGray_ = cur(roi).clone();
        cv::goodFeaturesToTrack(prevGray_, prevPoints_, 120, 0.01, 10.0);
        havePrev_ = true;
        info_.trackedPoints = static_cast<int>(prevPoints_.size());
        info_.inlierRatio = 1.0f;
        info_.timestamp = timestamp;
        return;
    }

    cv::Mat curRoi = cur(roi).clone();
    std::vector<cv::Point2f> next;
    std::vector<uint8_t> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray_, curRoi, prevPoints_, next, status, err, cv::Size(21, 21), 3);

    std::vector<cv::Point2f> goodPrev, goodNext;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && next[i].x >= 0 && next[i].y >= 0 && next[i].x < curRoi.cols && next[i].y < curRoi.rows) {
            goodPrev.push_back(prevPoints_[i]);
            goodNext.push_back(next[i]);
        }
    }

    if (goodPrev.size() < 20) {
        info_.state = TargetState::LOST;
        info_.trackedPoints = static_cast<int>(goodPrev.size());
        info_.inlierRatio = 0.0f;
        return;
    }

    cv::Mat affine = cv::estimateAffinePartial2D(goodPrev, goodNext);
    if (!affine.empty()) {
        const double* a = affine.ptr<double>(0);
        const double dx = a[2];
        const double dy = a[5];
        const float scaleX = static_cast<float>(std::hypot(a[0], a[1]));
        const float scaleY = static_cast<float>(std::hypot(a[3], a[4]));
        const float halfW = (x1 - x0) * 0.5f;
        const float halfH = (y1 - y0) * 0.5f;
        const float cx = x0 + halfW;
        const float cy = y0 + halfH;
        const float newHalfW = std::max(10.0f, halfW * scaleX);
        const float newHalfH = std::max(10.0f, halfH * scaleY);
        info_.x0 = std::max(0.0, (cx + dx - newHalfW) / width);
        info_.y0 = std::max(0.0, (cy + dy - newHalfH) / height);
        info_.x1 = std::min(1.0, (cx + dx + newHalfW) / width);
        info_.y1 = std::min(1.0, (cy + dy + newHalfH) / height);
    }

    info_.trackedPoints = static_cast<int>(goodPrev.size());
    info_.inlierRatio = goodPrev.empty() ? 0.0f : static_cast<float>(goodPrev.size()) / prevPoints_.size();
    info_.timestamp = timestamp;

    prevGray_ = curRoi.clone();
    prevPoints_ = goodNext;
}

TargetTrackInfo ObjectTracker::info() const
{
    return info_;
}
