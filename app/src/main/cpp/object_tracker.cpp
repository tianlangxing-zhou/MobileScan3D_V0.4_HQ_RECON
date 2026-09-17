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

// ---------- REACQUIRING（短暂出屏后自动找回） ----------
// 重捕获期间采纳 Nano 框所需的最低分数。比 kNanoWeakScore 高得多：
// 此时 KLT 已经完全失效，唯一依据就是 Nano 的外观判别，宁可超时判 LOST，
// 也不能把一个错误位置当成目标找回来（假锁定比丢失更难排查）。
constexpr float kNanoReacquireScore = 0.65f;
// 重捕获尝试的帧数上限。track() 每帧调一次，约 1.5s @30fps。
// 用帧数而非时间：掉帧时帧数计数更保守，不会因为卡顿就提前判死。
constexpr int kReacquireTimeoutFrames = 45;
// 重捕获期间每帧跑一次 Nano（正常态是每 4 帧）—— 危险态需要更快的节奏。
constexpr int kReacquireNanoPeriod = 1;

// ---------- NanoTrack 自适应节奏 / 纠偏权重 ----------
// 健康时每 4 帧（约 5Hz @20fps）跑一次省算力；
// confidence / 点数 / 可见比例任一项变差就提到每帧（约 20Hz）。
// V0.12 Fast Presence：危险态从「每 2 帧」提到「每帧」—— 目标出界那几帧
// 必须立刻给出正确框，等下一次 Nano（哪怕只晚 2 帧）KLT 就已经黏到背景上了。
constexpr int kNanoPeriodHealthy = 4;
constexpr int kNanoPeriodStressed = 1;
// V0.12：Nano/KLT 不一致判据。只有 Nano **分数够高**（说明它自己有把握、
// 对背景纹理不敏感）且两者中心/重叠**明显分歧**时才算「KLT 跟丢了」。
// 连续 kNanoMismatchFrames 帧才进 REACQUIRING —— 单帧抖动不判。
constexpr float kNanoMismatchMinScore = 0.70f;
constexpr float kNanoMismatchMaxIou = 0.10f;
constexpr float kNanoMismatchMaxCenterDiag = 0.65f;
// 中度中心偏移：配合「IoU 很低」一起用，捕捉「框还在附近但已经跟错东西」。
// ★ 上游补丁包只用了这个名字、**从未定义它** —— 不补就编译不过。
constexpr float kNanoMismatchModerateCenterDiag = 0.35f;
constexpr int kNanoMismatchFrames = 2;
// 纠偏权重：正常 0.20 的慢融合，避免把 KLT 的精细结果拽向 CNN 的粗框；
// 危险状态最多提到 0.55，让屏幕上的框明显跟得上目标。
constexpr float kNanoWeightBase = 0.20f;
constexpr float kNanoWeightLowConfidence = 0.55f;
constexpr float kNanoWeightNearEdge = 0.50f;
// 参与自适应判定的阈值
constexpr float kNanoStressConfidence = 0.75f;
constexpr int   kNanoStressPoints = 40;
constexpr float kNanoStressVisible = 0.70f;
// bbox 尺寸融合必须比中心慢得多。尺寸更新的权重一旦放大，
// 以前出现过的 bbox collapse 就会回来 —— 那个问题比「框大小略滞后」严重得多。
constexpr float kNanoSizeWeight = 0.15f;

// ---------- V0.13 Immutable Target Identity + BBox growth guard ----------
// 身份锚点模板的高度（像素）。宽度按初始框长宽比推出并夹到 [min,max]，
// 这样匹配时只需一次 resize，不需要多尺度金字塔。
constexpr int kIdentityTemplateH = 48;
constexpr int kIdentityMinW = 24;
constexpr int kIdentityMaxW = 96;
// 取模板时只保留初始框中心这一块：用户点选回退到 240x240 时框里大半是
// 背景，全框当模板会让「另一块相似的背景」也拿到高分，身份门就白设了。
constexpr float kIdentityCenterCrop = 0.70f;
// 候选框外扩比例（位置容差）。matchTemplate 的滑窗负责吸收这点位移。
constexpr float kIdentitySearchScale = 1.30f;
// 通过身份验证的归一化互相关(NCC)下限。实测正确目标 0.6~0.9、
// 不同背景纹理通常 < 0.3，0.45 留了光照/视角变化的余量。
constexpr float kIdentityAcceptMin = 0.45f;
// 允许 Nano **慢融合**纠偏的下限。比采纳门略低：慢融合权重只有
// 0.15~0.55，略微放宽不会立刻把框带跑，但比完全不纠偏好。
constexpr float kIdentitySlowMixMin = 0.42f;
// 模板方差下限：纯色/极糊的模板匹配无意义，宁可如实标记锚点不可用，
// 也不要拿它去门控 —— 那会把正常跟踪全判成「身份不符」。
constexpr float kIdentityMinTemplateVar = 1.0f;
// BBox growth guard：候选框最多涨到初始框的 2.0 倍（长宽各自比较，取大者）。
// 「Nano recovery 把框膨胀成数倍、吞掉整块背景」是 V0.12 截图里的直接形态。
constexpr float kIdentityMaxBBoxScale = 2.00f;
// 长宽比漂移上限：候选框 w/h 相对初始值最多 2.2 倍（双向）。
constexpr float kIdentityMaxAspectDrift = 2.20f;

} // namespace

void ObjectTracker::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pendingSelect_.store(false, std::memory_order_release);
    pendingU_.store(0.5f);
    pendingV_.store(0.5f);
    enabled_ = false;
    info_ = TargetTrackInfo{};
    pendingRectSelect_.store(false, std::memory_order_release);
    lastGray_.release();
    targetTemplate_.release();
    mask_.release();
    prevGray_.release();
    prevPoints_.clear();
    havePrev_ = false;
    weakKltFrames_ = 0;
    edgeLostFrames_ = 0;
    reacquireFrames_ = 0;
    framesSinceLastReseed_ = 0;
    nanoNeedInit_ = false;
    nanoFrameCounter_ = 0;
    nanoScore_ = 0.f;
    lastNanoScore_ = 0.f;
    nanoBoxFull_ = cv::Rect2f();
    nanoBoxAge_ = -1;
    appearanceMismatchFrames_ = 0;
    nanoKltRejects_ = 0;
    releaseIdentityAnchor();
    info_.nanoKltIou = 1.0f;
    info_.nanoKltRejects = 0;
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
    info_.presenceValid = true;
    info_.presenceLostCount = 0;
    appearanceInit_ = false;
    if (appearance_) {
        appearance_->reset();
    }
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
    pendingRectSelect_.store(false, std::memory_order_release);

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
    edgeLostFrames_ = 0;
    reacquireFrames_ = 0;
    info_.edgeLostFrames = 0;
    info_.centerXNorm = 0.f;
    info_.centerYNorm = 0.f;
    info_.visibleFraction = 1.0f;
    // V0.13：清目标 = 身份锚点也作废。下一次框选会重新建立，
    // 绝不能把上一个物体的外观拿来门控新目标。
    releaseIdentityAnchor();
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
        pendingRectSelect_.store(false, std::memory_order_release);
        edgeLostFrames_ = 0;
        appearanceMismatchFrames_ = 0;
        reacquireFrames_ = 0;
        info_.edgeLostFrames = 0;
        info_.state = TargetState::OFF;
        targetTemplate_.release();
        releaseIdentityAnchor();
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
    // 点按与拖框互斥：后设的那个生效
    pendingRectSelect_.store(false, std::memory_order_release);
    pendingSelect_.store(true, std::memory_order_release);
    info_.targetRequestAccepted++;
    return true;
}

bool ObjectTracker::requestTargetRect(float x0, float y0, float x1, float y1)
{
    std::lock_guard<std::mutex> lock(mutex_);
    info_.targetRequestCalls++;

    if (!std::isfinite(x0) || !std::isfinite(y0) ||
        !std::isfinite(x1) || !std::isfinite(y1)) {
        info_.targetRequestRejected++;
        info_.lastError = "requestTargetRect: invalid coordinate";
        return false;
    }
    if (!enabled_) {
        info_.targetRequestRejected++;
        info_.lastError = "requestTargetRect: not enabled";
        return false;
    }

    // 允许任意方向拖动：统一成 min/max，不假设 x0 < x1
    const float lx = std::clamp(std::min(x0, x1), 0.0f, 1.0f);
    const float ly = std::clamp(std::min(y0, y1), 0.0f, 1.0f);
    const float hx = std::clamp(std::max(x0, x1), 0.0f, 1.0f);
    const float hy = std::clamp(std::max(y0, y1), 0.0f, 1.0f);
    if (hx - lx <= 0.0f || hy - ly <= 0.0f) {
        info_.targetRequestRejected++;
        info_.lastError = "requestTargetRect: empty rect";
        return false;
    }

    pendingRectX0_.store(lx);
    pendingRectY0_.store(ly);
    pendingRectX1_.store(hx);
    pendingRectY1_.store(hy);
    pendingSelect_.store(false, std::memory_order_release);
    pendingRectSelect_.store(true, std::memory_order_release);
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
    info_.edgeLostFrames = edgeLostFrames_;
    reacquireFrames_ = 0;
    info_.lastError = reason;
}

/**
 * PresenceGate 判定失败时的入口（由 native 的目标 Mask 引擎驱动）。
 *
 * 与 edgeLostFrames_ 那条路径的区别必须讲清楚，否则以后很容易被误"优化"掉：
 *   - `edgeLostFrames_ >= 3` 判的是**几何**：bbox 已经滑出画面 3 帧以上。
 *     物体走了、tracker 停在墙上时，这条**永远不会触发**，绿框会一直挂在墙上。
 *   - 这里判的是**语义**：外观分数、目标 Mask、KLT 运动、中心一致性四项对不上。
 *     它和 bbox 在不在画面里完全无关，所以才是"物体已经走了"的正确判据。
 *
 * 进 REACQUIRING（而不是直接 LOST）：track() 里的找回分支会拿**出界前那份完整
 * 模板**在整帧上做 update，通常一两秒内能把目标找回来。
 */
void ObjectTracker::markPresenceLost()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (info_.state != TargetState::TRACKING) {
        return;
    }
    info_.state = TargetState::REACQUIRING;
    info_.confidence = 0.f;
    info_.inlierRatio = 0.f;
    reacquireFrames_ = 0;
    info_.presenceValid = false;
    ++info_.presenceLostCount;
    info_.lastEvent = "presence gate: target left";
    info_.lastError = "presence: appearance/mask disagree";
}

void ObjectTracker::setPresenceValid(bool valid)
{
    std::lock_guard<std::mutex> lock(mutex_);
    info_.presenceValid = valid;
}

bool ObjectTracker::configureAppearance(const std::string& lighttrackParam,
                                        const std::string& lighttrackBin,
                                        const std::string& nanoBackbone,
                                        const std::string& nanoHead)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string reason;
    auto tracker = createAppearanceTracker(lighttrackParam, lighttrackBin,
                                           nanoBackbone, nanoHead, &reason);
    info_.appearanceBackend = reason;
    appearanceInit_ = false;
    if (!tracker) {
        appearance_.reset();
        info_.appearanceAvailable = false;
        return false;
    }
    appearance_ = std::move(tracker);
    info_.appearanceAvailable = appearance_->ready();
    return info_.appearanceAvailable;
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

    // 目标初始化区域有两种来源，拖框优先：
    //   1) requestTargetRect —— 用户明确框选，直接采用他给的矩形；
    //   2) requestTarget    —— 单点回退到固定 240x240。
    //
    // 第 2 条只是兜底：固定 240x240 与目标真实形状无关，点一个细长瓶子时
    // 会把周围背景一起放进 ROI，KLT 追的其实是「物体 + 墙 + 桌子」的混合
    // 纹理 —— 这才是「框没有完全跟着物体」的真正原因，不是 KLT 算得慢。
    constexpr int kMinRoiPx = 32;
    bool haveNewRoi = false;
    cv::Rect roi;
    if (pendingRectSelect_.exchange(false, std::memory_order_acq_rel)) {
        const float lx = pendingRectX0_.load();
        const float ly = pendingRectY0_.load();
        const float hx = pendingRectX1_.load();
        const float hy = pendingRectY1_.load();
        roi = normToRect(lx, ly, hx, hy, width, height);
        haveNewRoi = roi.width >= kMinRoiPx && roi.height >= kMinRoiPx;
        if (!haveNewRoi) {
            info_.targetRequestRejected++;
            info_.lastError = "select: drag rect too small";
        }
    } else if (pendingSelect_.exchange(false, std::memory_order_acq_rel)) {
        const float u = pendingU_.load();
        const float v = pendingV_.load();
        const int cx = std::clamp(static_cast<int>(u * (width - 1)), 0, width - 1);
        const int cy = std::clamp(static_cast<int>(v * (height - 1)), 0, height - 1);
        constexpr int ROI_W = 240;
        constexpr int ROI_H = 240;
        roi = cv::Rect(cx - ROI_W / 2, cy - ROI_H / 2, ROI_W, ROI_H) &
              cv::Rect(0, 0, width, height);
        haveNewRoi = roi.width >= kMinRoiPx && roi.height >= kMinRoiPx;
    }

    if (haveNewRoi) {
        trackCx_ = static_cast<float>(roi.x) + static_cast<float>(roi.width) * 0.5f;
        trackCy_ = static_cast<float>(roi.y) + static_cast<float>(roi.height) * 0.5f;
        trackHalfW_ = static_cast<float>(roi.width) * 0.5f;
        trackHalfH_ = static_cast<float>(roi.height) * 0.5f;
        info_.x0 = static_cast<float>(roi.x) / width;
        info_.y0 = static_cast<float>(roi.y) / height;
        info_.x1 = static_cast<float>(roi.x + roi.width) / width;
        info_.y1 = static_cast<float>(roi.y + roi.height) / height;
        info_.centerXNorm = trackCx_ / width;
        info_.centerYNorm = trackCy_ / height;
        info_.visibleFraction = 1.0f;
        info_.edgeLostFrames = 0;
        edgeLostFrames_ = 0;
        appearanceMismatchFrames_ = 0;
        reacquireFrames_ = 0;
        info_.state = TargetState::ACQUIRING;
        targetTemplate_.release();
        prevGray_.release();
        prevPoints_.clear();
        havePrev_ = false;
        // V0.13：把**用户框的这一帧**的外观固化成不可变身份锚点。
        // 必须在这里、必须在 tracker 开始漂之前 —— 之后无论 Nano 还是 KLT
        // 说什么，都要拿它回答「这还是当初那个目标吗」。
        captureIdentityAnchor(owned, roi);
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
        // 由 48 放宽到 32：拖框选择让用户能给出贴合目标的窄框，
        // 太小的 ROI 本来就凑不出 15 个特征点，会走下面的
        // "acquire: too few features" 分支如实报错。
        if (roi.width < 32 || roi.height < 32) {
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
        edgeLostFrames_ = 0;
        appearanceMismatchFrames_ = 0;
        reacquireFrames_ = 0;
        info_.edgeLostFrames = 0;
        info_.visibleFraction = 1.0f;
        info_.centerXNorm = trackCx_ / width;
        info_.centerYNorm = trackCy_ / height;
        info_.lastEvent = "target acquired";
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

    if (!enabled_) {
        return;
    }

    // REACQUIRING：目标短暂出屏后由 NanoTrack 自己找回。
    //
    // 旧实现在这里就 `state != TRACKING -> return` 了，连 Nano 都不会跑，
    // 用户把物体移回画面也毫无反应 —— 必须重新点一次，这对手机扫描体验
    // 明显不够好。这里给约 1.5s 的找回窗口。
    if (info_.state == TargetState::REACQUIRING) {
        reacquireFrames_++;
        // 只有 Nano 模板已经存在时才尝试找回。
        //
        // 刻意不允许这里重新 init：目标刚出界时 info_.x0..y1 是被裁剪过的
        // 残片，拿它当模板初始化等于让 Nano 只记住「物体的一条边」，
        // 之后很可能把别的相似纹理认成目标 —— 比超时判 LOST 更糟。
        // 保留出界前那份完整模板、直接在整帧里 update() 才是我们要的。
        if (nanoLoaded_ && !nanoNeedInit_ &&
            (reacquireFrames_ % kReacquireNanoPeriod) == 0) {
            bool usedColor = false;
            const cv::Mat nanoFrame = resolveNanoFrame(owned, &usedColor);
            if (!nanoFrame.empty()) {
                info_.nanoUsedRealColor = usedColor;
                cv::Rect nanoRect;
                if (runNanoUpdate(nanoFrame, nanoFrame.cols, nanoFrame.rows, nanoRect) &&
                    nanoScore_ >= kNanoReacquireScore) {
                    // V0.13：Nano 的分数只说明「它自己很确信」，不说明「这就是
                    // 当初那个目标」。找回路径必须再过一次不可变身份锚点 ——
                    // V0.12 的实机故障正是 Nano 把背景纹理当成目标找了回来。
                    if (adoptNanoBox(nanoRect, nanoFrame.cols, nanoFrame.rows, owned, timestamp)) {
                        ++nanoRecoveries_;
                        info_.nanoRecoveries = nanoRecoveries_;
                        info_.state = TargetState::TRACKING;
                        info_.lastEvent = "target reacquired";
                        info_.lastError.clear();
                        info_.trackSuccess++;
                        reacquireFrames_ = 0;
                        info_.timestamp = timestamp;
                        return;
                    }
                    info_.lastEvent = "reacquire blocked by identity gate";
                }
            }
        }
        if (reacquireFrames_ > kReacquireTimeoutFrames) {
            markLost("target reacquire timeout");
        } else {
            info_.lastEvent = "target left frame, reacquiring";
        }
        return;
    }

    if (info_.state != TargetState::TRACKING) {
        return;
    }

    if (timestamp <= info_.timestamp) {
        return;
    }

    // V0.12: Nano box 新鲜度。nanoBoxAge_ >= 0 表示「有框」，每帧 +1；
    // runNanoUpdate() 成功时归零。
    if (nanoBoxAge_ >= 0) {
        ++nanoBoxAge_;
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
                        // V0.13：这里的 0.35 分非常低（背景纹理也能刷出来），
                        // 身份锚点是唯一能挡住「Nano 说找到了、其实换了东西」的门。
                        if (adoptNanoBox(nanoRect, nanoFrame.cols, nanoFrame.rows, owned, timestamp)) {
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

    // V0.12 Fast Presence：0.12 -> 0.18，配合下面的 2 帧（原 3 帧），
    // 目标刚压到画面边缘就提前进 REACQUIRING，而不是等它几乎整块滑出去。
    if (info_.visibleFraction < 0.18f) {
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
    // 中心点用**未裁剪**的 trackCx_/trackCy_，并允许跑出 [0,1]：
    // 目标完全出界时 visibleBox 会退化成空矩形，只有中心点还保留
    // 「目标往哪个方向去了」的信息，UI 才能给出正确的找回提示。
    info_.centerXNorm = std::clamp(trackCx_ / static_cast<float>(width), -1.0f, 2.0f);
    info_.centerYNorm = std::clamp(trackCy_ / static_cast<float>(height), -1.0f, 2.0f);
    info_.edgeLostFrames = edgeLostFrames_;

    const int bboxW = visibleBox.width;
    const int bboxH = visibleBox.height;
    info_.bboxWidthPx = bboxW;
    info_.bboxHeightPx = bboxH;
    info_.fullBBoxWidthPx = fullBox.width;
    info_.fullBBoxHeightPx = fullBox.height;
    info_.visibleBBoxWidthPx = visibleBox.width;
    info_.visibleBBoxHeightPx = visibleBox.height;

    // 目标整块滑出画面：不再直接判 LOST，先进 REACQUIRING 给 Nano 一个
    // 约 1.5s 的找回窗口。这样「扫出去再扫回来」不需要用户重新点一次。
    // 真正的终态 LOST 只留给「窗口内没找回来」这一种情况。
    // V0.12: 3 -> 2 帧。目标整块滑出画面时 3 帧太慢（相机 ~30fps 也要 100ms，
    // 加上 mask/depth 那一段更久），用户感受就是「物体都出画了才警告」。
    // 配合 UI 侧 0.65/0.30 两级 camera-rate 预警，慢的那一层只做语义确认。
    if (edgeLostFrames_ >= 2) {
        info_.state = TargetState::REACQUIRING;
        reacquireFrames_ = 0;
        info_.lastEvent = "target left frame, reacquiring";
        info_.lastError = "track: target left frame";
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

        // 自适应节奏：健康时维持原来的每 5 帧（约 4Hz @20fps）省算力；
        // 一旦 confidence / 点数 / 可见比例任一项变差就提到每 2 帧（约 10Hz）。
        // 不是「干脆每帧都跑」—— 那会把算力全花在正常情况下最不需要纠偏的时候。
        const bool stressed =
            info_.confidence < kNanoStressConfidence ||
            info_.trackedPoints < kNanoStressPoints ||
            info_.visibleFraction < kNanoStressVisible;
        const int nanoPeriod = stressed ? kNanoPeriodStressed : kNanoPeriodHealthy;

        if (nanoNeedInit_ || (nanoFrameCounter_ % nanoPeriod) == 0) {
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
                    const float nanoHalfW =
                        nanoRect.width * 0.5f / nanoFrame.cols * width;
                    const float nanoHalfH =
                        nanoRect.height * 0.5f / nanoFrame.rows * height;

                    // ---- V0.12 外观/KLT 一致性门控（camera-rate）----
                    // 目标离开画面后 KLT 常常**继续**在背景纹理上跟出一个「看起来很
                    // 正常的框」，而 Nano（外观判别）会立刻给出完全不同的位置。
                    // 两者长期不一致 = KLT 跟丢了，不必等 PresenceGate（它要 mask +
                    // 深度，慢好几帧）。
                    //
                    // 判据刻意保守：不是「IoU 小就判丢」，而是要求 Nano **分数够高**
                    // 且**明显分歧**，并且**连续 2 帧**成立 —— 正常跟踪时的单帧抖动
                    // 不能把目标判死。
                    //
                    // 坐标系：两个框都是**灰度帧像素**坐标。KLT 框由 trackCx_/
                    // trackHalfW_ 直接给出；Nano 框已在上面的 nanoCx/nanoHalfW 里
                    // 用 nanoFrame.cols -> width 线性换算过，可以直接比。
                    const cv::Rect2f kltBox(
                        trackCx_ - trackHalfW_,
                        trackCy_ - trackHalfH_,
                        trackHalfW_ * 2.f,
                        trackHalfH_ * 2.f);
                    const cv::Rect2f nanoBox(
                        nanoCx - nanoHalfW,
                        nanoCy - nanoHalfH,
                        nanoHalfW * 2.f,
                        nanoHalfH * 2.f);
                    const cv::Rect2f kltNanoInter = kltBox & nanoBox;
                    const float kltNanoUnion =
                        kltBox.area() + nanoBox.area() - kltNanoInter.area();
                    const float nanoKltIouNow =
                        kltNanoUnion > 1.f
                            ? kltNanoInter.area() / kltNanoUnion
                            : 0.f;
                    const float kltDiag = std::max(
                        1.f,
                        std::sqrt(
                            kltBox.width * kltBox.width +
                            kltBox.height * kltBox.height));
                    const float nanoKltCenterDiag =
                        std::hypot(nanoCx - trackCx_, nanoCy - trackCy_) / kltDiag;
                    info_.nanoKltIou = nanoKltIouNow;

                    // V0.13：给 Nano 候选框单独算一次身份分。它有两个用途：
                    //   1) 决定下面那次「慢融合纠偏」要不要执行；
                    //   2) 进报告，让「Nano 说 score=0.94 但身份只有 0.12」
                    //      这种「结构性共谋」一眼可见。
                    const float nanoIdentityNow = computeIdentityScore(
                        cv::Rect(static_cast<int>(std::lround(nanoBox.x)),
                                 static_cast<int>(std::lround(nanoBox.y)),
                                 std::max(1, static_cast<int>(std::lround(nanoBox.width))),
                                 std::max(1, static_cast<int>(std::lround(nanoBox.height)))),
                        owned);
                    info_.identityScore = nanoIdentityNow;

                    const bool hardAppearanceMismatch =
                        nanoScore_ >= kNanoMismatchMinScore &&
                        (nanoKltCenterDiag > kNanoMismatchMaxCenterDiag ||
                         (nanoKltIouNow < kNanoMismatchMaxIou &&
                          nanoKltCenterDiag > kNanoMismatchModerateCenterDiag));

                    if (hardAppearanceMismatch) {
                        appearanceMismatchFrames_++;
                        ++nanoKltRejects_;
                        info_.nanoKltRejects = nanoKltRejects_;
                    } else {
                        appearanceMismatchFrames_ =
                            std::max(0, appearanceMismatchFrames_ - 1);
                    }

                    if (appearanceMismatchFrames_ >= kNanoMismatchFrames) {
                        info_.state = TargetState::REACQUIRING;
                        info_.confidence = 0.f;
                        reacquireFrames_ = 0;
                        info_.lastEvent =
                            "Nano/KLT disagreement, fast reacquiring";
                        info_.lastError = "track: appearance mismatch";
                        return;
                    }

                    // 动态纠偏权重：从 0.20 的慢融合起步，危险状态最多 0.55。
                    // 固定 0.25 在危险的 10Hz 前看起来就是「拖尾」。
                    float nanoWeight = kNanoWeightBase;
                    if (info_.confidence < 0.65f) {
                        nanoWeight = kNanoWeightLowConfidence;
                    }
                    if (info_.visibleFraction < 0.60f) {
                        nanoWeight = std::max(nanoWeight, kNanoWeightNearEdge);
                    }

                    // V0.13：慢融合也必须过身份门。否则「Nano 已经漂到背景、
                    // KLT 还挂在目标上」时，这里的 0.15~0.55 权重会主动把 KLT
                    // 一起拖到背景去 —— 这正是「两个 tracker 一起漂到同一块
                    // 背景纹理」的动力学来源，也是 V0.12 截图里跟错目标的成因。
                    const bool nanoIdentityOk =
                        (nanoIdentityNow < 0.f) || (nanoIdentityNow >= kIdentitySlowMixMin);
                    if (nanoScore_ > 0.45f && info_.confidence > 0.45f && nanoIdentityOk) {
                        trackCx_ = trackCx_ * (1.f - nanoWeight) + nanoCx * nanoWeight;
                        trackCy_ = trackCy_ * (1.f - nanoWeight) + nanoCy * nanoWeight;
                        // 尺寸也要纠，但必须比中心慢得多：0.85/0.15。
                        // 放大尺寸权重会把以前出现过的 bbox collapse 带回来。
                        trackHalfW_ = trackHalfW_ * (1.f - kNanoSizeWeight) +
                                      nanoHalfW * kNanoSizeWeight;
                        trackHalfH_ = trackHalfH_ * (1.f - kNanoSizeWeight) +
                                      nanoHalfH * kNanoSizeWeight;
                        const float minHalf = 8.f;
                        trackHalfW_ = std::clamp(trackHalfW_, minHalf,
                                                 static_cast<float>(width) * 0.5f);
                        trackHalfH_ = std::clamp(trackHalfH_, minHalf,
                                                 static_cast<float>(height) * 0.5f);
                    } else if (nanoScore_ > 0.60f && prevPoints_.size() < 20) {
                        // V0.13：re-seed 会把 KLT 的特征点**整批换到 Nano 框里**，
                        // 是「两个 tracker 一起漂走」最主要的动力学入口，必须过身份门。
                        if (adoptNanoBox(nanoRect, nanoFrame.cols, nanoFrame.rows, owned, timestamp)) {
                            ++nanoRecoveries_;
                            info_.nanoRecoveries = nanoRecoveries_;
                            info_.lastEvent = "Nano reseeded KLT";
                        }
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

// ---------------------------------------------------------------- V0.13 identity
void ObjectTracker::releaseIdentityAnchor()
{
    identityTemplate_.release();
    identityReady_ = false;
    identityAspect_ = 1.f;
    identityHalfWpx_ = 1.f;
    identityHalfHpx_ = 1.f;
    identityRejects_ = 0;
    info_.identityScore = -1.f;
    info_.identityRejects = 0;
    info_.bboxScaleFromInitial = 1.f;
    info_.identityAnchorReady = false;
}

/**
 * V0.13：把用户框选那一帧的外观存成**永不更新**的身份锚点。
 *
 * 只保留框的中心 kIdentityCenterCrop 区域：用户点选时会回退到固定
 * 240x240，框里大半是背景，把整框当模板会让「另一块相似的背景」也拿高分，
 * 身份门就形同虚设。中心裁剪让模板更贴近目标本身。
 */
void ObjectTracker::captureIdentityAnchor(const cv::Mat& gray, const cv::Rect& roi)
{
    releaseIdentityAnchor();
    if (gray.empty() || roi.width < 16 || roi.height < 16) {
        return;
    }
    cv::Rect r = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width < 16 || r.height < 16) {
        return;
    }
    // 初始框尺寸（**未裁剪**）：bbox growth guard 的比较基准。
    identityHalfWpx_ = static_cast<float>(r.width) * 0.5f;
    identityHalfHpx_ = static_cast<float>(r.height) * 0.5f;
    identityAspect_ = static_cast<float>(r.width) / static_cast<float>(r.height);

    const int cw = std::max(16, static_cast<int>(std::lround(r.width * kIdentityCenterCrop)));
    const int ch = std::max(16, static_cast<int>(std::lround(r.height * kIdentityCenterCrop)));
    cv::Rect core(r.x + (r.width - cw) / 2, r.y + (r.height - ch) / 2, cw, ch);
    core &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (core.width < 16 || core.height < 16) {
        return;
    }

    cv::Mat patch = gray(core).clone();
    const float aspect = static_cast<float>(core.width) / static_cast<float>(core.height);
    const int th = kIdentityTemplateH;
    const int tw = std::clamp(static_cast<int>(std::lround(th * aspect)),
                              kIdentityMinW, kIdentityMaxW);
    cv::Mat tmpl;
    cv::resize(patch, tmpl, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
    if (tmpl.empty() || tmpl.type() != CV_8U) {
        return;
    }
    cv::Scalar m, s;
    cv::meanStdDev(tmpl, m, s);
    // 方差太小（纯色/极糊）-> 锚点不可用。如实标记，让身份门自动退出，
    // 而不是拿一个没信息量的模板去把正常跟踪全判成「不像」。
    if (!(s[0] * s[0] >= kIdentityMinTemplateVar)) {
        return;
    }
    identityTemplate_ = tmpl;
    identityReady_ = true;
    info_.identityAnchorReady = true;
    info_.lastEvent = "identity anchor captured " +
                      std::to_string(tmpl.cols) + "x" + std::to_string(tmpl.rows);
}

/**
 * V0.13：候选框与身份锚点的归一化互相关峰值。
 *
 * **这是全工程唯一一个不依赖 KLT 历史、也不依赖 Nano 内部状态的判据** ——
 * 另外两个 tracker 会互相确认（Nano 重新播种 KLT 并把 inlierRatio 写回 1.0），
 * 只有「和最初那个东西长得像不像」是独立证据。
 *
 * 位置容差由 matchTemplate 的滑窗给出，尺度差异由 resize 消除，
 * 所以它只回答「是不是同一个外观」，不回答「框得准不准」——
 * 后者仍然交给 KLT。
 *
 * @return [0,1] 的 NCC 峰值；**-1 表示无法判定**（调用方必须放行而不是拒绝）。
 */
float ObjectTracker::computeIdentityScore(const cv::Rect& box, const cv::Mat& gray) const
{
    if (!identityReady_ || identityTemplate_.empty() || gray.empty()) {
        return -1.f;
    }
    if (box.width <= 0 || box.height <= 0) {
        return -1.f;
    }

    // 候选框大半出界时搜索窗里几乎没有目标，硬算只会得到「不像」的假结论。
    const cv::Rect inside = box & cv::Rect(0, 0, gray.cols, gray.rows);
    const float boxArea = static_cast<float>(box.width) * static_cast<float>(box.height);
    if (boxArea <= 0.f ||
        static_cast<float>(inside.area()) / boxArea < 0.55f) {
        return -1.f;
    }

    const float cx = static_cast<float>(box.x) + static_cast<float>(box.width) * 0.5f;
    const float cy = static_cast<float>(box.y) + static_cast<float>(box.height) * 0.5f;
    const float hw = static_cast<float>(box.width) * 0.5f * kIdentitySearchScale;
    const float hh = static_cast<float>(box.height) * 0.5f * kIdentitySearchScale;
    cv::Rect search(static_cast<int>(std::floor(cx - hw)),
                    static_cast<int>(std::floor(cy - hh)),
                    static_cast<int>(std::ceil(hw * 2.f)),
                    static_cast<int>(std::ceil(hh * 2.f)));
    search &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (search.width < 16 || search.height < 16) {
        return -1.f;
    }

    cv::Mat patch = gray(search).clone();
    if (patch.empty() || patch.type() != CV_8U) {
        return -1.f;
    }
    // 把搜索窗缩放到「模板尺寸 x kIdentitySearchScale」，多出来的那一圈
    // 就是位置容差。
    const int pw = std::max(8, static_cast<int>(std::lround(identityTemplate_.cols * kIdentitySearchScale)));
    const int ph = std::max(8, static_cast<int>(std::lround(identityTemplate_.rows * kIdentitySearchScale)));
    cv::Mat scaled;
    cv::resize(patch, scaled, cv::Size(pw, ph), 0, 0, cv::INTER_AREA);
    if (scaled.empty() || scaled.cols < identityTemplate_.cols ||
        scaled.rows < identityTemplate_.rows) {
        return -1.f;
    }

    cv::Mat resp;
    try {
        cv::matchTemplate(scaled, identityTemplate_, resp, cv::TM_CCOEFF_NORMED);
    } catch (const cv::Exception&) {
        return -1.f;
    }
    if (resp.empty()) {
        return -1.f;
    }
    double minV = 0.0;
    double maxV = 0.0;
    cv::minMaxLoc(resp, &minV, &maxV);
    return static_cast<float>(std::clamp(maxV, -1.0, 1.0));
}

bool ObjectTracker::runNanoUpdate(const cv::Mat& frame, int width, int height, cv::Rect& outRect)
{
    // ---- 外观后端接缝 ----
    // 如果工厂给了一个**真正可用且不是 nanotrack 的**后端（例如将来接入的
    // LightTrack-ncnn），就优先用它做外观纠偏。当前工厂只会返回 nanotrack
    // （LightTrack 需要 ncnn 运行库，本工程未链接），所以这一段今天不会执行，
    // 下面的 NanoTrack 路径行为与改造前**完全一致**。
    if (appearance_ && appearance_->ready() &&
        std::string(appearance_->backendName()) != "nanotrack") {
        if (!appearanceInit_) {
            const cv::Rect2f initBox(static_cast<float>(info_.x0) * width,
                                     static_cast<float>(info_.y0) * height,
                                     static_cast<float>(info_.x1 - info_.x0) * width,
                                     static_cast<float>(info_.y1 - info_.y0) * height);
            appearance_->init(frame, initBox);
            appearanceInit_ = true;
            return false; // 初始化帧不产生分数
        }
        cv::Rect2f box;
        float score = 0.f;
        if (!appearance_->update(frame, box, score)) {
            return false;
        }
        lastNanoScore_ = score;
        nanoScore_ = score;
        info_.nanoScore = score;
        outRect = cv::Rect(static_cast<int>(std::lround(box.x)),
                           static_cast<int>(std::lround(box.y)),
                           static_cast<int>(std::lround(box.width)),
                           static_cast<int>(std::lround(box.height)));
        nanoBoxFull_ = cv::Rect2f(box.x / width, box.y / height,
                                  box.width / width, box.height / height);
        nanoBoxAge_ = 0;
        return true;
    }

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
        nanoBoxAge_ = 0;
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

bool ObjectTracker::adoptNanoBox(const cv::Rect& nanoBox, int nanoW, int nanoH,
                                 const cv::Mat& grayOwned, uint64_t timestamp)
{
    if (nanoW <= 0 || nanoH <= 0 || grayOwned.empty() || nanoBox.width <= 0 ||
        nanoBox.height <= 0) {
        return false;
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

    const float candHalfW = std::max(8.f, std::min(hwNorm * width, width * 0.5f));
    const float candHalfH = std::max(8.f, std::min(hhNorm * height, height * 0.5f));

    // ---- V0.13 Immutable Target Identity + BBox growth guard ----
    // 顺序很重要：这两项判定必须在**任何状态字节被写入之前**完成。
    // 不通过时直接 return false，调用方会保留原来的 KLT 状态继续跟踪 ——
    // 身份门只切断「Nano -> 状态机」，绝不切断 KLT 主链，所以它不会死锁。
    if (identityReady_) {
        const cv::Rect candBox(
            static_cast<int>(std::floor(cxNorm * width - candHalfW)),
            static_cast<int>(std::floor(cyNorm * height - candHalfH)),
            std::max(1, static_cast<int>(std::round(candHalfW * 2.f))),
            std::max(1, static_cast<int>(std::round(candHalfH * 2.f))));
        const float idScore = computeIdentityScore(candBox, grayOwned);
        info_.identityScore = idScore;

        const float sW = candHalfW / std::max(1.f, identityHalfWpx_);
        const float sH = candHalfH / std::max(1.f, identityHalfHpx_);
        const float scaleFromInitial = std::max(sW, sH);
        info_.bboxScaleFromInitial = scaleFromInitial;
        const float candAspect = candHalfW / std::max(1.f, candHalfH);
        const float aspectDrift = (candAspect > identityAspect_)
            ? (candAspect / identityAspect_)
            : (identityAspect_ / std::max(0.0001f, candAspect));

        // idScore < 0 = 无法判定（锚点/搜索窗/出界），**放行**：
        // 拒绝只留给「有明确证据说明不像」的情形，否则目标贴边时会误杀。
        const bool idOk = (idScore < 0.f) || (idScore >= kIdentityAcceptMin);
        // 膨胀/长宽比守卫**不受 -1 影响**：它们是纯几何判据，永远有效。
        const bool sizeOk = scaleFromInitial <= kIdentityMaxBBoxScale;
        const bool aspectOk = aspectDrift <= kIdentityMaxAspectDrift;

        if (!idOk || !sizeOk || !aspectOk) {
            ++identityRejects_;
            info_.identityRejects = identityRejects_;
            info_.lastEvent = !sizeOk ? "identity reject: bbox growth"
                                      : (!aspectOk ? "identity reject: aspect drift"
                                                   : "identity reject: appearance");
            info_.lastError.clear();
            return false;
        }
    }

    trackCx_ = std::clamp(cxNorm * width, 0.f, static_cast<float>(std::max(0, width - 1)));
    trackCy_ = std::clamp(cyNorm * height, 0.f, static_cast<float>(std::max(0, height - 1)));
    trackHalfW_ = candHalfW;
    trackHalfH_ = candHalfH;

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
    reacquireFrames_ = 0;
    nanoNeedInit_ = false;
    info_.edgeLostFrames = 0;

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
    info_.centerXNorm = trackCx_ / static_cast<float>(width);
    info_.centerYNorm = trackCy_ / static_cast<float>(height);
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
    return true;
}
