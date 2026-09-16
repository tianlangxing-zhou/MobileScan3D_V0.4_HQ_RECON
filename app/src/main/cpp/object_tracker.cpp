#include "object_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

void ObjectTracker::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pendingSelect_.store(false, std::memory_order_release);
    pendingU_.store(0.5f);
    pendingV_.store(0.5f);
    enabled_ = false;
    info_ = TargetTrackInfo{};
    lastGray_.release();
    targetTemplate_.release();
    mask_.release();
    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;
}

void ObjectTracker::clearTarget()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pendingSelect_.store(false, std::memory_order_release);

    targetTemplate_.release();
    mask_.release();
    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;

    info_.x0 = 0.f;
    info_.y0 = 0.f;
    info_.x1 = 0.f;
    info_.y1 = 0.f;
    info_.confidence = 0.f;
    info_.trackedPoints = 0;
    info_.inlierRatio = 0.f;
    info_.medianDepth = 0.f;
    info_.roiSharpness = 0.f;
    info_.targetTemplateAllocated = false;
    info_.templateWidth = 0;
    info_.templateHeight = 0;
    info_.maskAllocated = false;
    info_.maskWidth = 0;
    info_.maskHeight = 0;
    info_.prevGrayValid = false;
    info_.prevGrayWidth = 0;
    info_.prevGrayHeight = 0;
    info_.prevPointCount = 0;
    info_.state = enabled_ ? TargetState::ARMED : TargetState::OFF;
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
        pendingSelect_.store(false, std::memory_order_release);
        info_.state = TargetState::OFF;
        targetTemplate_.release();
        mask_.release();
        prevGray_.release();
        prevPoints_.clear();
        havePrev_ = false;
    }
}

bool ObjectTracker::configureNano(const std::string& backbone, const std::string& head)
{
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        cv::TrackerNano::Params p;
        p.backbone = backbone;
        p.neckhead = head;
        nano_ = cv::TrackerNano::create(p);
        nanoLoaded_ = !nano_.empty();
        info_.nanoLoaded = nanoLoaded_;
        return nanoLoaded_;
    } catch (const cv::Exception& e) {
        info_.lastError = std::string("NanoTrack load: ") + e.what();
        nanoLoaded_ = false;
        info_.nanoLoaded = false;
        return false;
    }
}

bool ObjectTracker::requestTarget(float u, float v)
{
    std::lock_guard<std::mutex> lock(mutex_);
    info_.targetRequestCalls++;

    if (!std::isfinite(u) || !std::isfinite(v)) {
        info_.targetRequestRejected++;
        info_.lastError = "requestTarget: invalid coordinate";
        return false;
    }

    if (!enabled_) {
        info_.targetRequestRejected++;
        info_.lastError = "requestTarget: not enabled";
        return false;
    }

    pendingU_.store(std::clamp(u, 0.0f, 1.0f));
    pendingV_.store(std::clamp(v, 0.0f, 1.0f));
    pendingSelect_.store(true, std::memory_order_release);
    info_.targetRequestAccepted++;
    return true;
}

void ObjectTracker::setError(const std::string& msg)
{
    info_.lastError = msg;
}

void ObjectTracker::markLost(const std::string& reason)
{
    if (info_.state != TargetState::LOST) {
        info_.trackLost++;
    }
    info_.state = TargetState::LOST;
    info_.confidence = 0.f;
    info_.inlierRatio = 0.f;
    info_.lastError = reason;
}

bool ObjectTracker::selectTarget(float u, float v)
{
    (void)u;
    (void)v;
    return false;
}

void ObjectTracker::updateFrame(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp)
{
    if (!gray || width <= 0 || height <= 0 || stride < width) {
        return;
    }

    cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(gray), static_cast<size_t>(stride));
    cv::Mat owned = src.clone();
    if (owned.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) {
        pendingSelect_.store(false, std::memory_order_release);
        return;
    }

    lastGray_ = owned;
    info_.haveCameraFrame = true;
    info_.frameWidth = width;
    info_.frameHeight = height;
    info_.lastFrameTs = timestamp;
    info_.cameraUpdateCalls++;

    if (pendingSelect_.exchange(false, std::memory_order_acq_rel)) {
        const float u = pendingU_.load();
        const float v = pendingV_.load();
        const int cx = std::clamp(static_cast<int>(u * (width - 1)), 0, width - 1);
        const int cy = std::clamp(static_cast<int>(v * (height - 1)), 0, height - 1);
        constexpr int ROI_W = 240;
        constexpr int ROI_H = 240;
        cv::Rect wanted(cx - ROI_W / 2, cy - ROI_H / 2, ROI_W, ROI_H);
        const cv::Rect bounds(0, 0, width, height);
        cv::Rect roi = wanted & bounds;
        if (roi.width >= 64 && roi.height >= 64) {
            trackCx_ = static_cast<float>(cx);
            trackCy_ = static_cast<float>(cy);
            trackHalfW_ = ROI_W * 0.5f;
            trackHalfH_ = ROI_H * 0.5f;
            info_.x0 = static_cast<float>(roi.x) / width;
            info_.y0 = static_cast<float>(roi.y) / height;
            info_.x1 = static_cast<float>(roi.x + roi.width) / width;
            info_.y1 = static_cast<float>(roi.y + roi.height) / height;
            info_.state = TargetState::ACQUIRING;
            targetTemplate_.release();
            prevGray_.release();
            prevPoints_.clear();
            havePrev_ = false;
        }
    }

    if (info_.state == TargetState::ACQUIRING) {
        info_.acquireCalls++;
        info_.lastError.clear();
        const int x0 = std::max(0, static_cast<int>(info_.x0 * width));
        const int y0 = std::max(0, static_cast<int>(info_.y0 * height));
        const int x1 = std::min(width - 1, static_cast<int>(info_.x1 * width));
        const int y1 = std::min(height - 1, static_cast<int>(info_.y1 * height));
        cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        roi &= cv::Rect(0, 0, width, height);
        if (roi.width < 48 || roi.height < 48) {
            info_.acquireFail++;
            info_.lastError = "acquire: ROI too small";
            return;
        }

        cv::Mat roiGray = owned(roi).clone();
        cv::Mat lap;
        cv::Laplacian(roiGray, lap, CV_32F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(lap, mean, stddev);
        info_.roiSharpness = static_cast<float>(stddev[0] * stddev[0]);

        std::vector<cv::Point2f> localPts;
        cv::goodFeaturesToTrack(roiGray, localPts, 120, 0.01, 5.0);
        if (localPts.size() < 15) {
            info_.acquireFail++;
            info_.trackedPoints = static_cast<int>(localPts.size());
            info_.confidence = 0.0f;
            info_.lastError = "acquire: too few features";
            return;
        }

        prevPoints_.clear();
        prevPoints_.reserve(localPts.size());
        for (auto p : localPts) {
            p.x += roi.x;
            p.y += roi.y;
            prevPoints_.push_back(p);
        }
        prevGray_ = owned.clone();
        info_.trackedPoints = static_cast<int>(prevPoints_.size());
        info_.inlierRatio = 1.0f;
        info_.confidence = std::min(1.0f, info_.trackedPoints / 80.0f);
        info_.prevGrayValid = !prevGray_.empty();
        info_.prevGrayWidth = prevGray_.cols;
        info_.prevGrayHeight = prevGray_.rows;
        info_.prevPointCount = static_cast<int>(prevPoints_.size());
        info_.bboxWidthPx = roi.width;
        info_.bboxHeightPx = roi.height;
        info_.lastAffineScale = 1.0f;
        info_.affineScaleEMA = 1.0f;
        info_.timestamp = timestamp;
        info_.acquireSuccess++;
        info_.state = TargetState::TRACKING;
        info_.lastError.clear();
        havePrev_ = true;
        nanoNeedInit_ = true;
    }
}

void ObjectTracker::track(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp)
{
    if (!gray || width <= 0 || height <= 0 || stride < width) {
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

    if (timestamp <= info_.timestamp) {
        return;
    }

    if (!havePrev_ || prevGray_.empty() || prevGray_.size() != owned.size() || prevGray_.type() != owned.type() || prevPoints_.empty()) {
        markLost("track: invalid previous frame");
        return;
    }

    std::vector<cv::Point2f> next;
    std::vector<cv::Point2f> back;
    std::vector<uint8_t> statusForward;
    std::vector<uint8_t> statusBackward;
    std::vector<float> errorForward;
    std::vector<float> errorBackward;

    try {
        cv::calcOpticalFlowPyrLK(prevGray_, owned, prevPoints_, next, statusForward, errorForward, cv::Size(21, 21), 3);
        cv::calcOpticalFlowPyrLK(owned, prevGray_, next, back, statusBackward, errorBackward, cv::Size(21, 21), 3);
    } catch (const cv::Exception& e) {
        markLost(std::string("LK exception: ") + e.what());
        return;
    }

    std::vector<cv::Point2f> goodPrev, goodNext;
    for (size_t i = 0; i < statusForward.size() && i < statusBackward.size() && i < next.size() && i < prevPoints_.size(); ++i) {
        if (!statusForward[i] || !statusBackward[i]) continue;
        const float fb = cv::norm(prevPoints_[i] - back[i]);
        if (fb > 1.5f) continue;
        if (errorForward[i] > 20.f) continue;
        const cv::Point2f& p = next[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
        if (p.x < 0 || p.y < 0 || p.x >= width || p.y >= height) continue;
        goodPrev.push_back(prevPoints_[i]);
        goodNext.push_back(p);
    }

    if (goodPrev.size() < 15) {
        info_.trackedPoints = static_cast<int>(goodPrev.size());
        info_.inlierRatio = 0.0f;
        markLost("track: too few LK points");
        return;
    }

    cv::Mat inliers;
    cv::Mat affine;
    try {
        affine = cv::estimateAffinePartial2D(goodPrev, goodNext, inliers, cv::RANSAC, 3.0);
    } catch (const cv::Exception& e) {
        markLost(std::string("affine exception: ") + e.what());
        return;
    }

    if (affine.empty()) {
        markLost("track: affine failed");
        return;
    }

    affine.convertTo(affine, CV_64F);
    const double* a = affine.ptr<double>(0);
    const double dx = a[2];
    const double dy = a[5];
    const float scale = static_cast<float>(std::hypot(a[0], a[1]));

    if (!std::isfinite(dx) ||
        !std::isfinite(dy) ||
        !std::isfinite(scale) ||
        scale < 0.5f ||
        scale > 2.0f ||
        std::abs(dx) > width * 0.5 ||
        std::abs(dy) > height * 0.5)
    {
        markLost("track: insane affine");
        return;
    }

    trackCx_ += static_cast<float>(dx);
    trackCy_ += static_cast<float>(dy);

    const int left = static_cast<int>(std::floor(trackCx_ - trackHalfW_));
    const int top = static_cast<int>(std::floor(trackCy_ - trackHalfH_));
    const int right = static_cast<int>(std::ceil(trackCx_ + trackHalfW_));
    const int bottom = static_cast<int>(std::ceil(trackCy_ + trackHalfH_));
    cv::Rect fullBox(left, top, right - left, bottom - top);
    cv::Rect visibleBox = fullBox & cv::Rect(0, 0, width, height);

    const float fullArea = static_cast<float>(fullBox.area());
    const float visibleArea = static_cast<float>(visibleBox.area());
    info_.visibleFraction = fullArea > 0.f ? visibleArea / fullArea : 0.f;

    if (info_.visibleFraction < 0.12f) {
        edgeLostFrames_++;
    } else {
        edgeLostFrames_ = 0;
    }

    if (edgeLostFrames_ >= 3) {
        markLost("track: target left frame");
        return;
    }

    if (info_.visibleFraction < 0.40f) {
        info_.lastEvent = "target near frame edge";
    }

    info_.x0 = std::clamp(static_cast<float>(visibleBox.x) / width, 0.0f, 1.0f);
    info_.y0 = std::clamp(static_cast<float>(visibleBox.y) / height, 0.0f, 1.0f);
    info_.x1 = std::clamp(static_cast<float>(visibleBox.x + visibleBox.width) / width, 0.0f, 1.0f);
    info_.y1 = std::clamp(static_cast<float>(visibleBox.y + visibleBox.height) / height, 0.0f, 1.0f);

    const int bboxW = visibleBox.width;
    const int bboxH = visibleBox.height;
    info_.bboxWidthPx = bboxW;
    info_.bboxHeightPx = bboxH;
    info_.fullBBoxWidthPx = fullBox.width;
    info_.fullBBoxHeightPx = fullBox.height;
    info_.visibleBBoxWidthPx = visibleBox.width;
    info_.visibleBBoxHeightPx = visibleBox.height;
    info_.lastAffineScale = scale;
    info_.affineScaleEMA = 0.7f * info_.affineScaleEMA + 0.3f * scale;

    int inlierCount = 0;
    std::vector<cv::Point2f> inlierNext;
    if (!inliers.empty()) {
        for (int i = 0; i < inliers.rows; ++i) {
            if (inliers.at<uchar>(i)) {
                inlierCount++;
                if (i < static_cast<int>(goodNext.size())) {
                    inlierNext.push_back(goodNext[i]);
                }
            }
        }
    }

    info_.trackedPoints = static_cast<int>(goodNext.size());
    info_.inlierRatio = goodNext.empty() ? 0.0f : static_cast<float>(inlierCount) / static_cast<float>(goodNext.size());
    info_.confidence = std::clamp(info_.inlierRatio * std::min(1.0f, info_.trackedPoints / 60.0f), 0.0f, 1.0f);
    info_.timestamp = timestamp;
    info_.lastError.clear();
    info_.trackSuccess++;

    prevGray_ = owned.clone();
    prevPoints_ = inlierNext.empty() ? std::move(goodNext) : std::move(inlierNext);
    info_.prevGrayValid = !prevGray_.empty();
    info_.prevGrayWidth = prevGray_.cols;
    info_.prevGrayHeight = prevGray_.rows;
    info_.prevPointCount = static_cast<int>(prevPoints_.size());

    if (nanoLoaded_ && info_.state == TargetState::TRACKING) {
        nanoFrameCounter_++;
        cv::Mat bgr;
        try {
            cv::cvtColor(owned, bgr, cv::COLOR_GRAY2BGR);
            if (nanoNeedInit_) {
                cv::Rect initRect(
                    static_cast<int>(info_.x0 * width),
                    static_cast<int>(info_.y0 * height),
                    std::max(1, static_cast<int>((info_.x1 - info_.x0) * width)),
                    std::max(1, static_cast<int>((info_.y1 - info_.y0) * height))
                );
                nano_->init(bgr, initRect);
                nanoNeedInit_ = false;
                nanoInitCalls_++;
                info_.nanoInitCalls = nanoInitCalls_;
            } else if ((nanoFrameCounter_ % 5) == 0) {
                const auto start = std::chrono::steady_clock::now();
                cv::Rect nanoRect;
                const bool ok = nano_->update(bgr, nanoRect);
                const auto end = std::chrono::steady_clock::now();
                nanoLastMs_ = std::chrono::duration<double, std::milli>(end - start).count();
                nanoUpdateCalls_++;
                info_.nanoUpdateCalls = nanoUpdateCalls_;
                info_.nanoLastMs = nanoLastMs_;
                if (ok) {
                    nanoScore_ = nano_->getTrackingScore();
                    nanoBoxFull_ = nanoRect;
                    info_.nanoScore = nanoScore_;
                    const float nanoCx = nanoRect.x + nanoRect.width * 0.5f;
                    const float nanoCy = nanoRect.y + nanoRect.height * 0.5f;
                    if (nanoScore_ > 0.45f && info_.confidence > 0.45f) {
                        trackCx_ = trackCx_ * 0.75f + nanoCx * 0.25f;
                        trackCy_ = trackCy_ * 0.75f + nanoCy * 0.25f;
                    } else if (nanoScore_ > 0.60f && prevPoints_.size() < 20) {
                        trackCx_ = nanoCx;
                        trackCy_ = nanoCy;
                        nanoRecoveries_++;
                        info_.nanoRecoveries = nanoRecoveries_;
                    }
                } else {
                    nanoFailures_++;
                    info_.nanoFailures = nanoFailures_;
                }
            }
        } catch (const cv::Exception& e) {
            nanoFailures_++;
            info_.nanoFailures = nanoFailures_;
            info_.lastError = std::string("NanoTrack: ") + e.what();
            nanoNeedInit_ = false;
        } catch (const std::exception& e) {
            nanoFailures_++;
            info_.nanoFailures = nanoFailures_;
            info_.lastError = std::string("NanoTrack: ") + e.what();
            nanoNeedInit_ = false;
        }
    }

    const bool lowFeatures = goodNext.size() < 35;
    const bool enoughVisible = info_.visibleFraction >= 0.40f;
    const bool cooldownOk = framesSinceLastReseed_ >= 10;

    if (lowFeatures && enoughVisible && cooldownOk) {
        const int rx0 = std::max(0, static_cast<int>(info_.x0 * width));
        const int ry0 = std::max(0, static_cast<int>(info_.y0 * height));
        const int rx1 = std::min(width - 1, static_cast<int>(info_.x1 * width));
        const int ry1 = std::min(height - 1, static_cast<int>(info_.y1 * height));
        cv::Rect reseedRect(rx0, ry0, rx1 - rx0 + 1, ry1 - ry0 + 1);
        reseedRect &= cv::Rect(0, 0, width, height);
        if (reseedRect.width >= 48 && reseedRect.height >= 48) {
            cv::Mat reseedGray = owned(reseedRect).clone();
            std::vector<cv::Point2f> reseedPts;
            cv::goodFeaturesToTrack(reseedGray, reseedPts, 120, 0.01, 5.0);
            if (reseedPts.size() >= 20) {
                prevPoints_.clear();
                for (auto p : reseedPts) {
                    p.x += reseedRect.x;
                    p.y += reseedRect.y;
                    prevPoints_.push_back(p);
                }
                info_.reseedCount++;
                info_.trackedPoints = static_cast<int>(prevPoints_.size());
                info_.prevPointCount = info_.trackedPoints;
                info_.lastEvent = "reseeded " + std::to_string(info_.trackedPoints) + " features";
                info_.lastError.clear();
                framesSinceLastReseed_ = 0;
            }
        } else {
            framesSinceLastReseed_++;
        }
    } else {
        framesSinceLastReseed_++;
    }
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
