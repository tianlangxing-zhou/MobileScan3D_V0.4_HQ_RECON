#include "object_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

namespace {

// KLT 有效点下限：低于该值视为弱跟踪，需要 NanoTrack 介入
constexpr int kMinKltPoints = 15;
// 弱 KLT 连续帧容忍次数：超过且 Nano 也救不回来才判 LOST
constexpr int kWeakKltPatience = 3;
// 弱 KLT 场景下采纳 Nano 结果所需的最低分数
constexpr float kNanoWeakScore = 0.35f;

} // namespace

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
    weakKltFrames_ = 0;
    edgeLostFrames_ = 0;
    framesSinceLastReseed_ = 0;
    nanoNeedInit_ = false;
    nanoFrameCounter_ = 0;
    nanoScore_ = 0.f;
    lastNanoScore_ = 0.f;
    nanoBoxFull_ = cv::Rect2f();
    colorFrame_.release();
    colorFrameValid_ = false;
    colorFrameTs_ = 0;
    colorFrameCalls_ = 0;
    nanoWeakAttempts_ = 0;
    nanoWeakRecoveries_ = 0;
    nanoForcedUpdates_ = 0;
    info_.colorFrameValid = false;
    info_.colorFrameWidth = 0;
    info_.colorFrameHeight = 0;
}

void ObjectTracker::clearWeakKlt()
{
    weakKltFrames_ = 0;
    info_.weakKltFrames = 0;
}

void ObjectTracker::setColorFrame(const uint8_t* bgr, int width, int height, int stride, uint64_t timestamp)
{
    if (!bgr || width <= 0 || height <= 0 || stride < width * 3) {
        return;
    }
    cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(bgr), static_cast<size_t>(stride));
    cv::Mat owned = src.clone();
    if (owned.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    colorFrame_ = std::move(owned);
    colorFrameValid_ = true;
    colorFrameTs_ = timestamp;
    colorFrameCalls_++;
    info_.colorFrameValid = true;
    info_.colorFrameWidth = width;
    info_.colorFrameHeight = height;
    info_.colorFrameCalls = colorFrameCalls_;
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
    info_.depthP10 = 0.f;
    info_.depthP90 = 0.f;
    info_.roiSharpness = 0.f;
    info_.depthValidPixels = 0;
    info_.depthSampleCount = 0;
    info_.depthRoiArea = 0;
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
    weakKltFrames_ = 0;
    info_.weakKltFrames = 0;
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
        clearWeakKlt();
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

    info_.kltGoodPoints = static_cast<int>(goodPrev.size());

    if (goodPrev.size() < kMinKltPoints) {
        info_.trackedPoints = static_cast<int>(goodPrev.size());
        info_.inlierRatio = 0.0f;
        ++weakKltFrames_;
        info_.weakKltFrames = weakKltFrames_;

        // KLT 变弱时立刻强制运行 NanoTrack 抢救，不等 every-5-frame 节奏。
        // 修复点：原实现先 markLost() 再更新 Nano，等于最需要救场时函数已经 return，
        // 导致 nanoUpdateCalls=45 / nanoFailures=0 却 nanoRecoveries=0。
        if (nanoLoaded_) {
            bool usedColor = false;
            const cv::Mat nanoFrame = resolveNanoFrame(owned, &usedColor);
            if (!nanoFrame.empty()) {
                cv::Rect nanoRect;
                ++nanoWeakAttempts_;
                info_.nanoWeakAttempts = nanoWeakAttempts_;
                ++nanoForcedUpdates_;
                info_.nanoForcedUpdates = nanoForcedUpdates_;
                if (runNanoUpdate(nanoFrame, nanoFrame.cols, nanoFrame.rows, nanoRect)) {
                    info_.nanoUsedRealColor = usedColor;
                    if (nanoScore_ >= kNanoWeakScore) {
                        adoptNanoBox(nanoRect, nanoFrame.cols, nanoFrame.rows, owned, timestamp);
                        ++nanoRecoveries_;
                        ++nanoWeakRecoveries_;
                        info_.nanoRecoveries = nanoRecoveries_;
                        info_.nanoWeakRecoveries = nanoWeakRecoveries_;
                        clearWeakKlt();
                        info_.lastError.clear();
                        info_.lastEvent = "Nano recovered weak KLT";
                        info_.trackSuccess++;
                        info_.timestamp = timestamp;
                        return;
                    }
                }
            }
        }

        if (weakKltFrames_ < kWeakKltPatience) {
            info_.lastEvent = "KLT weak, waiting for recovery";
            return;
        }
        markLost("track: KLT+Nano recovery failed");
        return;
    }

    clearWeakKlt();

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

    // 诊断字段必须在「出界 return」之前更新。
    // 旧顺序是：先算 visibleFraction -> 累计边缘丢失 -> 出界就 return，
    // 而 bbox 相关的字段在 return 之后才写。于是报告里会出现
    // 「visibleBBox=241x7（最新一帧）+ visibleFraction=0（也来自最新一帧）」
    // 这种看似矛盾的组合 —— 其实是混用了两个不同帧的数据。
    // 这里只是把同一帧的诊断信息写在一起，不动 tracker 主算法。
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

    if (edgeLostFrames_ >= 3) {
        markLost("track: target left frame");
        return;
    }

    if (info_.visibleFraction < 0.40f) {
        info_.lastEvent = "target near frame edge";
    }
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
    info_.kltGoodPoints = info_.trackedPoints;
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
        if (nanoNeedInit_ || (nanoFrameCounter_ % 5) == 0) {
            bool usedColor = false;
            const cv::Mat nanoFrame = resolveNanoFrame(owned, &usedColor);
            info_.nanoUsedRealColor = usedColor;
            if (!nanoFrame.empty()) {
                cv::Rect nanoRect;
                if (runNanoUpdate(nanoFrame, nanoFrame.cols, nanoFrame.rows, nanoRect)) {
                    // Nano 输入与灰度帧尺寸/长宽比可能不同，统一换算回灰度帧坐标
                    const float nanoCx =
                        (nanoRect.x + nanoRect.width * 0.5f) / nanoFrame.cols * width;
                    const float nanoCy =
                        (nanoRect.y + nanoRect.height * 0.5f) / nanoFrame.rows * height;
                    if (nanoScore_ > 0.45f && info_.confidence > 0.45f) {
                        trackCx_ = trackCx_ * 0.75f + nanoCx * 0.25f;
                        trackCy_ = trackCy_ * 0.75f + nanoCy * 0.25f;
                    } else if (nanoScore_ > 0.60f && prevPoints_.size() < 20) {
                        adoptNanoBox(nanoRect, nanoFrame.cols, nanoFrame.rows, owned, timestamp);
                        ++nanoRecoveries_;
                        info_.nanoRecoveries = nanoRecoveries_;
                        info_.lastEvent = "Nano reseeded KLT";
                    }
                }
            }
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
    const int roiArea = (x1 - x0 + 1) * (y1 - y0 + 1);
    values.reserve(static_cast<size_t>(roiArea));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float d = depth[static_cast<size_t>(y) * width + x];
            if (std::isfinite(d) && d > 0.0f) {
                values.push_back(d);
            }
        }
    }
    // ROI 面积 / 有效深度像素数必须一起上报：只看 P10/P50/P90 无法区分
    // 「目标处深度真的一致」和「ROI 里只剩 3 个有效像素」。
    // 这里不把空的 medianDepth 置 0 —— filterDepth 依赖它，而这一轮
    // 明确不动 Object Lock / Depth Filter 的既有行为。
    info_.depthRoiArea = roiArea;
    info_.depthValidPixels = static_cast<int>(values.size());
    info_.depthSampleCount = static_cast<int>(values.size());
    if (!values.empty()) {
        std::sort(values.begin(), values.end());
        const size_t n = values.size();
        const auto pick = [&values, n](double p) -> float {
            if (n == 0) return 0.f;
            const double pos = p * static_cast<double>(n - 1);
            size_t i = static_cast<size_t>(std::lround(pos));
            if (i >= n) i = n - 1;
            return values[i];
        };
        info_.depthP10 = pick(0.10);
        info_.medianDepth = pick(0.50);
        info_.depthP90 = pick(0.90);
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

cv::Mat ObjectTracker::resolveNanoFrame(const cv::Mat& grayOwned, bool* usedColor) const
{
    // 优先使用由 YUV420 生成的真实彩色帧：NanoTrack 的外观判别依赖真实颜色，
    // 原来的 GRAY2BGR 只是把灰度复制成三通道，等于放弃了这个能力。
    if (colorFrameValid_ && !colorFrame_.empty() && colorFrame_.type() == CV_8UC3) {
        if (usedColor) *usedColor = true;
        return colorFrame_;
    }
    cv::Mat bgr;
    try {
        cv::cvtColor(grayOwned, bgr, cv::COLOR_GRAY2BGR);
    } catch (const cv::Exception&) {
        if (usedColor) *usedColor = false;
        return cv::Mat();
    }
    if (usedColor) *usedColor = false;
    return bgr;
}

cv::Rect ObjectTracker::normToRect(float x0, float y0, float x1, float y1, int w, int h)
{
    if (w <= 0 || h <= 0) {
        return cv::Rect();
    }
    const int px0 = std::clamp(static_cast<int>(std::floor(x0 * w)), 0, w - 1);
    const int py0 = std::clamp(static_cast<int>(std::floor(y0 * h)), 0, h - 1);
    const int px1 = std::clamp(static_cast<int>(std::ceil(x1 * w)), px0 + 1, w);
    const int py1 = std::clamp(static_cast<int>(std::ceil(y1 * h)), py0 + 1, h);
    return cv::Rect(px0, py0, px1 - px0, py1 - py0);
}

bool ObjectTracker::runNanoUpdate(const cv::Mat& frame, int width, int height, cv::Rect& outRect)
{
    if (!nanoLoaded_ || frame.empty() || width <= 0 || height <= 0) {
        return false;
    }
    try {
        if (nanoNeedInit_) {
            const cv::Rect initRect =
                normToRect(info_.x0, info_.y0, info_.x1, info_.y1, width, height);
            if (initRect.width < 8 || initRect.height < 8) {
                return false;
            }
            nano_->init(frame, initRect);
            nanoNeedInit_ = false;
            ++nanoInitCalls_;
            info_.nanoInitCalls = nanoInitCalls_;
            return false; // 初始化帧不产生分数
        }

        const auto start = std::chrono::steady_clock::now();
        cv::Rect rect;
        const bool ok = nano_->update(frame, rect);
        const auto end = std::chrono::steady_clock::now();
        nanoLastMs_ = std::chrono::duration<double, std::milli>(end - start).count();
        ++nanoUpdateCalls_;
        info_.nanoUpdateCalls = nanoUpdateCalls_;
        info_.nanoLastMs = nanoLastMs_;

        if (!ok) {
            ++nanoFailures_;
            info_.nanoFailures = nanoFailures_;
            return false;
        }

        lastNanoScore_ = static_cast<float>(nano_->getTrackingScore());
        nanoScore_ = lastNanoScore_;
        info_.nanoScore = nanoScore_;
        outRect = rect;
        nanoBoxFull_ = cv::Rect2f(
            static_cast<float>(rect.x) / width,
            static_cast<float>(rect.y) / height,
            static_cast<float>(rect.width) / width,
            static_cast<float>(rect.height) / height);
        return true;
    } catch (const cv::Exception& e) {
        ++nanoFailures_;
        info_.nanoFailures = nanoFailures_;
        info_.lastError = std::string("NanoTrack: ") + e.what();
        nanoNeedInit_ = false;
        return false;
    } catch (const std::exception& e) {
        ++nanoFailures_;
        info_.nanoFailures = nanoFailures_;
        info_.lastError = std::string("NanoTrack: ") + e.what();
        nanoNeedInit_ = false;
        return false;
    }
}

void ObjectTracker::adoptNanoBox(const cv::Rect& nanoBox, int nanoW, int nanoH,
                                 const cv::Mat& grayOwned, uint64_t timestamp)
{
    if (nanoW <= 0 || nanoH <= 0 || grayOwned.empty() || nanoBox.width <= 0 ||
        nanoBox.height <= 0) {
        return;
    }

    const int width = grayOwned.cols;
    const int height = grayOwned.rows;

    // Nano 框（Nano 输入坐标系）-> 归一化 -> 灰度帧像素坐标
    const float cxNorm =
        (static_cast<float>(nanoBox.x) + nanoBox.width * 0.5f) / static_cast<float>(nanoW);
    const float cyNorm =
        (static_cast<float>(nanoBox.y) + nanoBox.height * 0.5f) / static_cast<float>(nanoH);
    const float hwNorm = std::max(0.02f, 0.5f * nanoBox.width / static_cast<float>(nanoW));
    const float hhNorm = std::max(0.02f, 0.5f * nanoBox.height / static_cast<float>(nanoH));

    trackCx_ = std::clamp(cxNorm * width, 0.f, static_cast<float>(std::max(0, width - 1)));
    trackCy_ = std::clamp(cyNorm * height, 0.f, static_cast<float>(std::max(0, height - 1)));
    trackHalfW_ = std::max(8.f, std::min(hwNorm * width, width * 0.5f));
    trackHalfH_ = std::max(8.f, std::min(hhNorm * height, height * 0.5f));

    // 用 Nano 框重新播种 KLT 特征：恢复后仍由 KLT 做精细跟踪，
    // Nano 只在 KLT 失效时兜底，避免长期依赖 CNN 导致漂移。
    const int left = std::max(0, static_cast<int>(std::floor(trackCx_ - trackHalfW_)));
    const int top = std::max(0, static_cast<int>(std::floor(trackCy_ - trackHalfH_)));
    const int right = std::min(width, static_cast<int>(std::ceil(trackCx_ + trackHalfW_)));
    const int bottom = std::min(height, static_cast<int>(std::ceil(trackCy_ + trackHalfH_)));
    cv::Rect box(left, top, std::max(1, right - left), std::max(1, bottom - top));
    box &= cv::Rect(0, 0, width, height);

    std::vector<cv::Point2f> pts;
    if (box.width >= 16 && box.height >= 16) {
        try {
            cv::Mat roiGray = grayOwned(box).clone();
            cv::goodFeaturesToTrack(roiGray, pts, 120, 0.01, 5.0);
        } catch (const cv::Exception&) {
            pts.clear();
        }
    }

    prevPoints_.clear();
    if (pts.size() >= 10) {
        prevPoints_.reserve(pts.size());
        for (auto p : pts) {
            p.x += static_cast<float>(box.x);
            p.y += static_cast<float>(box.y);
            prevPoints_.push_back(p);
        }
    }
    prevGray_ = grayOwned.clone();
    havePrev_ = !prevGray_.empty();
    framesSinceLastReseed_ = 0;
    edgeLostFrames_ = 0;
    nanoNeedInit_ = false;

    info_.x0 = std::clamp(static_cast<float>(box.x) / width, 0.0f, 1.0f);
    info_.y0 = std::clamp(static_cast<float>(box.y) / height, 0.0f, 1.0f);
    info_.x1 = std::clamp(static_cast<float>(box.x + box.width) / width, 0.0f, 1.0f);
    info_.y1 = std::clamp(static_cast<float>(box.y + box.height) / height, 0.0f, 1.0f);
    info_.bboxWidthPx = box.width;
    info_.bboxHeightPx = box.height;
    info_.fullBBoxWidthPx = box.width;
    info_.fullBBoxHeightPx = box.height;
    info_.visibleBBoxWidthPx = box.width;
    info_.visibleBBoxHeightPx = box.height;
    info_.visibleFraction = 1.0f;
    info_.trackedPoints = static_cast<int>(prevPoints_.size());
    info_.kltGoodPoints = info_.trackedPoints;
    info_.prevPointCount = info_.trackedPoints;
    info_.prevGrayValid = !prevGray_.empty();
    info_.prevGrayWidth = prevGray_.cols;
    info_.prevGrayHeight = prevGray_.rows;
    info_.inlierRatio = 1.0f;
    info_.confidence = std::clamp(0.5f + 0.5f * lastNanoScore_, 0.f, 1.f);
    info_.lastAffineScale = 1.0f;
    info_.affineScaleEMA = 1.0f;
    info_.lastError.clear();
}
