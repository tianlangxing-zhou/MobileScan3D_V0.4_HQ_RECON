#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

enum class TargetState
{
    OFF = 0,
    ARMED = 1,
    ACQUIRING = 2,
    TRACKING = 3,
    LOST = 4
};

struct TargetTrackInfo
{
    TargetState state = TargetState::OFF;
    float x0 = 0.f;
    float y0 = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    float confidence = 0.f;
    int trackedPoints = 0;
    float inlierRatio = 0.f;
    float medianDepth = 0.f;
    float depthP10 = 0.f;
    float depthP90 = 0.f;
    float roiSharpness = 0.f;
    uint64_t timestamp = 0;

    bool haveCameraFrame = false;
    int frameWidth = 0;
    int frameHeight = 0;
    uint64_t lastFrameTs = 0;
    uint64_t cameraUpdateCalls = 0;
    uint64_t trackerUpdateCalls = 0;
    uint64_t targetRequestCalls = 0;
    uint64_t targetRequestAccepted = 0;
    uint64_t targetRequestRejected = 0;
    uint64_t acquireCalls = 0;
    uint64_t acquireSuccess = 0;
    uint64_t acquireFail = 0;
    uint64_t trackSuccess = 0;
    uint64_t trackLost = 0;
    bool nanoLoaded = false;
    uint64_t nanoInitCalls = 0;
    uint64_t nanoUpdateCalls = 0;
    uint64_t nanoFailures = 0;
    uint64_t nanoRecoveries = 0;
    float nanoScore = 0.f;
    double nanoLastMs = 0.0;
    // KLT 弱帧连续计数；以及 NanoTrack 恢复/彩色输入诊断
    int weakKltFrames = 0;
    uint64_t nanoWeakRecoveries = 0;
    uint64_t nanoWeakAttempts = 0;
    bool nanoUsedRealColor = false;
    bool colorFrameValid = false;
    int colorFrameWidth = 0;
    int colorFrameHeight = 0;
    uint64_t colorFrameCalls = 0;
    uint64_t depthFilterCalls = 0;
    uint64_t depthFilterSkipped = 0;
    bool maskAllocated = false;
    int maskWidth = 0;
    int maskHeight = 0;
    bool targetTemplateAllocated = false;
    int templateWidth = 0;
    int templateHeight = 0;
    bool prevGrayValid = false;
    int prevGrayWidth = 0;
    int prevGrayHeight = 0;
    int prevPointCount = 0;
    int bboxWidthPx = 0;
    int bboxHeightPx = 0;
    int fullBBoxWidthPx = 0;
    int fullBBoxHeightPx = 0;
    int visibleBBoxWidthPx = 0;
    int visibleBBoxHeightPx = 0;
    float visibleFraction = 1.0f;
    float lastAffineScale = 1.0f;
    float affineScaleEMA = 1.0f;
    uint64_t reseedCount = 0;
    std::string lastError;
    std::string lastEvent;
};

class ObjectTracker
{
public:
    ObjectTracker() = default;

    void reset();
    void clearTarget();
    void setEnabled(bool enabled);
    bool configureNano(const std::string& backbone, const std::string& head);
    bool requestTarget(float u, float v);
    bool selectTarget(float u, float v);
    void updateFrame(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp);
    void track(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp);
    // 由 YUV420 生成的真实 BGR 帧（不需要与灰度帧同尺寸）。
    // 供 NanoTrack 做外观判别；调用方可每 N 帧才喂一次以省算力。
    void setColorFrame(const uint8_t* bgr, int width, int height, int stride, uint64_t timestamp);
    void updateFromDepth(const float* depth, int width, int height, uint64_t timestamp);
    bool filterDepth(float* depth, int width, int height, uint64_t timestamp);
    bool isEnabled() const;
    bool isTracking() const;
    TargetTrackInfo info() const;

private:
    mutable std::mutex mutex_;
    TargetTrackInfo info_;
    bool enabled_ = false;
    cv::Mat lastGray_;
    cv::Mat targetTemplate_;
    cv::Mat mask_;
    cv::Mat prevGray_;
    std::vector<cv::Point2f> prevPoints_;
    bool havePrev_ = false;
    float trackCx_ = 0.f;
    float trackCy_ = 0.f;
    float trackHalfW_ = 120.f;
    float trackHalfH_ = 120.f;
    int framesSinceLastReseed_ = 0;
    int edgeLostFrames_ = 0;
    cv::Ptr<cv::TrackerNano> nano_;
    bool nanoLoaded_ = false;
    bool nanoNeedInit_ = false;
    int nanoFrameCounter_ = 0;
    float nanoScore_ = 0.f;
    cv::Rect2f nanoBoxFull_;
    uint64_t nanoInitCalls_ = 0;
    uint64_t nanoUpdateCalls_ = 0;
    uint64_t nanoRecoveries_ = 0;
    uint64_t nanoFailures_ = 0;
    double nanoLastMs_ = 0.0;
    int weakKltFrames_ = 0;
    uint64_t nanoWeakAttempts_ = 0;
    uint64_t nanoWeakRecoveries_ = 0;
    float lastNanoScore_ = 0.f;

    // 真实彩色帧（YUV420 -> BGR），保护在 mutex_ 之下
    cv::Mat colorFrame_;
    bool colorFrameValid_ = false;
    uint64_t colorFrameTs_ = 0;
    uint64_t colorFrameCalls_ = 0;

    std::atomic<bool> pendingSelect_{false};
    std::atomic<float> pendingU_{0.5f};
    std::atomic<float> pendingV_{0.5f};

    void setError(const std::string& msg);
    void markLost(const std::string& reason);

    // NanoTrack 辅助：取输入帧 / 归一化框互转 / 强制更新 / 采纳 Nano 框并重播种
    cv::Mat resolveNanoFrame(const cv::Mat& grayOwned, bool* usedColor) const;
    static cv::Rect normToRect(float x0, float y0, float x1, float y1, int w, int h);
    bool runNanoUpdate(const cv::Mat& frame, int width, int height, cv::Rect& outRect);
    void adoptNanoBox(const cv::Rect& nanoBox, int nanoW, int nanoH,
                      const cv::Mat& grayOwned, uint64_t timestamp);
    void clearWeakKlt();
};
