#include "object_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

void ObjectTracker::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = false;
    info_ = TargetTrackInfo{};
    lastGray_.release();
    targetTemplate_.release();
    mask_.release();
    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;
}

void ObjectTracker::setEnabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
    if (enabled) {
        if (info_.state == TargetState::OFF) {
            info_.state = TargetState::ARMED;
        }
    } else {
        info_.state = TargetState::OFF;
        targetTemplate_.release();
        mask_.release();
        prevGray_.release();
        prevPoints_.clear();
        havePrev_ = false;
    }
}

void ObjectTracker::setError(const std::string& msg)
{
    info_.lastError = msg;
}

bool ObjectTracker::selectTarget(float u, float v)
{
    std::lock_guard<std::mutex> lock(mutex_);
    info_.selectTargetCalls++;
    info_.lastError.clear();

    if (!enabled_) {
        info_.selectTargetFail++;
        setError("selectTarget: not enabled");
        return false;
    }
    if (!info_.haveCameraFrame || lastGray_.empty()) {
        info_.selectTargetFail++;
        setError("selectTarget: no camera frame");
        return false;
    }
    if (lastGray_.type() != CV_8UC1) {
        info_.selectTargetFail++;
        setError("selectTarget: bad frame type");
        return false;
    }

    const int w = lastGray_.cols;
    const int h = lastGray_.rows;
    if (w < 64 || h < 64) {
        info_.selectTargetFail++;
        setError("selectTarget: frame too small");
        return false;
    }

    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    const int cx = std::clamp(static_cast<int>(u * w), 0, w - 1);
    const int cy = std::clamp(static_cast<int>(v * h), 0, h - 1);
    const int roiW = std::min(240, w / 3);
    const int roiH = std::min(240, h / 3);

    cv::Rect wanted(cx - roiW / 2, cy - roiH / 2, roiW, roiH);
    const cv::Rect imageRect(0, 0, w, h);
    cv::Rect roi = wanted & imageRect;
    if (roi.width < 48 || roi.height < 48) {
        info_.selectTargetFail++;
        setError("selectTarget: roi too small");
        return false;
    }

    targetTemplate_ = lastGray_(roi).clone();
    if (targetTemplate_.empty()) {
        info_.selectTargetFail++;
        setError("selectTarget: template empty");
        return false;
    }

    info_.x0 = static_cast<float>(roi.x) / w;
    info_.y0 = static_cast<float>(roi.y) / h;
    info_.x1 = static_cast<float>(roi.x + roi.width) / w;
    info_.y1 = static_cast<float>(roi.y + roi.height) / h;
    info_.targetTemplateAllocated = true;
    info_.templateWidth = roi.width;
    info_.templateHeight = roi.height;
    info_.state = TargetState::ACQUIRING;
    info_.selectTargetSuccess++;

    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;
    return true;
}

void ObjectTracker::updateFrame(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp)
{
    if (!gray || width <= 0 || height <= 0 || stride <= 0) {
        return;
    }

    cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(gray), static_cast<size_t>(stride));
    cv::Mat owned = src.clone();
    if (owned.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    lastGray_ = owned;
    info_.haveCameraFrame = true;
    info_.frameWidth = width;
    info_.frameHeight = height;
    info_.lastFrameTs = timestamp;
    info_.cameraUpdateCalls++;
}

void ObjectTracker::track(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp)
{
    if (!gray || width <= 0 || height <= 0 || stride <= 0) {
        return;
    }

    cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(gray), static_cast<size_t>(stride));
    cv::Mat owned = src.clone();
    if (owned.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    info_.trackerUpdateCalls++;
    if (!enabled_ || info_.state != TargetState::TRACKING) {
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

    cv::Mat roi = owned(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)).clone();
    if (!havePrev_) {
        prevGray_ = roi;
        cv::goodFeaturesToTrack(prevGray_, prevPoints_, 120, 0.01, 10.0);
        havePrev_ = true;
        info_.trackedPoints = static_cast<int>(prevPoints_.size());
        info_.inlierRatio = 1.0f;
        info_.timestamp = timestamp;
        return;
    }

    std::vector<cv::Point2f> next;
    std::vector<uint8_t> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray_, roi, prevPoints_, next, status, err, cv::Size(21, 21), 3);

    std::vector<cv::Point2f> goodPrev, goodNext;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i] && next[i].x >= 0 && next[i].y >= 0 && next[i].x < roi.cols && next[i].y < roi.rows) {
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
    prevGray_ = roi.clone();
    prevPoints_ = goodNext;
}

void ObjectTracker::updateFromDepth(const float* depth, int width, int height, uint64_t timestamp)
{
    if (!depth || width <= 0 || height <= 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (info_.state != TargetState::TRACKING) {
        return;
    }

    const int x0 = std::max(0, static_cast<int>(info_.x0 * width));
    const int y0 = std::max(0, static_cast<int>(info_.y0 * height));
    const int x1 = std::min(width - 1, static_cast<int>(info_.x1 * width));
    const int y1 = std::min(height - 1, static_cast<int>(info_.y1 * height));
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

bool ObjectTracker::filterDepth(float* depth, int width, int height, uint64_t timestamp)
{
    std::lock_guard<std::mutex> lock(mutex_);
    info_.depthFilterCalls++;
    if (!depth) {
        info_.depthFilterSkipped++;
        setError("filterDepth: null depth");
        return false;
    }
    if (info_.state != TargetState::TRACKING) {
        info_.depthFilterSkipped++;
        setError("filterDepth: not tracking");
        return false;
    }

    const float target = info_.medianDepth;
    if (target <= 0.0f) {
        info_.depthFilterSkipped++;
        setError("filterDepth: no target depth");
        return false;
    }

    const float margin = std::max(0.15f, target * 0.08f);
    const int x0 = std::max(0, static_cast<int>(info_.x0 * width));
    const int y0 = std::max(0, static_cast<int>(info_.y0 * height));
    const int x1 = std::min(width - 1, static_cast<int>(info_.x1 * width));
    const int y1 = std::min(height - 1, static_cast<int>(info_.y1 * height));

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
    info_.maskAllocated = true;
    info_.maskWidth = x1 - x0 + 1;
    info_.maskHeight = y1 - y0 + 1;
    info_.timestamp = timestamp;
    return true;
}

bool ObjectTracker::isEnabled() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

bool ObjectTracker::isTracking() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ && info_.state == TargetState::TRACKING;
}

TargetTrackInfo ObjectTracker::info() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return info_;
}
