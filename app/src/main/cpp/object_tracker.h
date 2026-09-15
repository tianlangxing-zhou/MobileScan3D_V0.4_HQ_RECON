#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>

enum class TargetState
{
    OFF = 0,
    ACQUIRING = 1,
    TRACKING = 2,
    LOST = 3
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
};

class ObjectTracker
{
public:
    void reset();
    void select(float u, float v, float medianDepth, uint64_t timestamp);
    void clear();
    void update(float medianDepth, float sharpness, uint64_t timestamp);
    void updateFromDepth(const float* depth, int width, int height, uint64_t timestamp);
    void track(const uint8_t* gray, int width, int height, int stride, uint64_t timestamp);
    bool isEnabled() const;
    bool isTracking() const;
    void filterDepth(float* depth, int width, int height, uint64_t timestamp);
    TargetTrackInfo info() const;

private:
    TargetTrackInfo info_;
    bool enabled_ = false;
    cv::Mat prevGray_;
    std::vector<cv::Point2f> prevPoints_;
    bool havePrev_ = false;
};
