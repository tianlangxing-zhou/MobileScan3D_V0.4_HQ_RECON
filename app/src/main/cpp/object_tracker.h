#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

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
    float roiSharpness = 0.f;
    uint64_t timestamp = 0;

    bool haveCameraFrame = false;
    int frameWidth = 0;
    int frameHeight = 0;
    uint64_t lastFrameTs = 0;
    uint64_t cameraUpdateCalls = 0;
    uint64_t trackerUpdateCalls = 0;
    uint64_t selectTargetCalls = 0;
    uint64_t selectTargetSuccess = 0;
    uint64_t selectTargetFail = 0;
    uint64_t depthFilterCalls = 0;
    uint64_t depthFilterSkipped = 0;
    bool maskAllocated = false;
    int maskWidth = 0;
    int maskHeight = 0;
    bool targetTemplateAllocated = false;
    int templateWidth = 0;
    int templateHeight = 0;
    std::string lastError;
};

class ObjectTracker
{
public:
    ObjectTracker() = default;

    void reset();
    void setEnabled(bool enabled);
    bool selectTarget(float u, float v);
    void updateFrame(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp);
    void track(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp);
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

    void setError(const std::string& msg);
};
