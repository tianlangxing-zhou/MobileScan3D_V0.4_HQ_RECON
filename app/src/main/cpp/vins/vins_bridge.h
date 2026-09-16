#pragma once

#include <cstdint>

struct VinsHealth {
    bool initialized = false;

    float velocity = 0.0f;
    float accBias = 0.0f;
    float gyroBias = 0.0f;
    float gravity = 0.0f;

    int trackedFeatures = 0;

    double lastImuDt = 0.0;
    double timeOffset = 0.0;
    float ric[9] = {};
    float tic[3] = {};
};

void vinsInit(
    float fx,
    float fy,
    float cx,
    float cy,
    int w,
    int h,
    const float* ric,
    const float* tic,
    float accN,
    float accW,
    float gyrN,
    float gyrW);

void vinsInputImu(double t, double ax, double ay, double az, double gx, double gy, double gz);

void vinsInputImage(double t, const std::uint8_t* gray, int w, int h, int stride);

bool vinsGetPose(float* out7);

bool vinsReady();

bool vinsInitialized();

bool vinsGetHealth(VinsHealth* out);

// 当前滑窗内成功三角化的特征深度中位数（VINS 自身尺度）。
// 仅作深度尺度诊断使用；没有任何有效特征时返回 0。
float vinsFeatureDepthMedian();

// 只统计**落在指定归一化 ROI 内**的已三角化特征深度中位数。
//
// 为什么需要"只取目标 ROI"：全局 VINS 中位数包含墙、地面、天花板，
// 而它们的深度分布和目标完全不是一回事。用「全场 VINS median ÷ 目标 raw depth
// median」得到的尺度，等于拿一个混合分布去校准单个物体 —— 这就是实机上
// correction 一会儿 0.057、一会儿 0.0975 跳来跳去的原因之一。
// 目标 ROI 内的 VINS 特征 ≤ 目标 raw depth 中位数，才是这一物体自己的尺度。
//
// 参数是相机归一化坐标 [0,1]（与 tracker 的 bbox 同一套），内部换算到
// VINS 输入图像像素。ROI 内没有有效特征时返回 0，并令 *outSampleCount = 0。
float vinsFeatureDepthMedianInRoi(float nx0, float ny0, float nx1, float ny1,
                                  int* outSampleCount);
