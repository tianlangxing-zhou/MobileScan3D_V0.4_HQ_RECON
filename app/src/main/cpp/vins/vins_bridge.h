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

// 取当前滑窗内已三角化特征的 (u, v, depth) 样本。
//
// 为什么需要"逐样本"而不是又一个中位数：深度标定要拟合
//
//     z = a*d + b        或        1/z = a*d + b
//
// 这是一个需要**配对样本**的回归问题。中位数只能给出一个乘性 ratio，
// 既估不出偏移项，也没法做 MAD 剔除与 Huber 降权（那正是把三角化失败的
// 野值、以及画面里混进来的墙/地面挡在外面的手段）。
//
// u/v 是**归一化**坐标 [0,1]（内部已按 VINS 输入图像尺寸归一化），
// 调用方可以直接用它去采样同尺寸的深度图；depth 是 VINS 自身尺度下的米。
//
// @param out         至少 maxSamples*3 个 float
// @param maxSamples  最多取多少样本
// @return 实际写入的样本数
int vinsFeatureSamples(float* out, int maxSamples);

struct VinsWorldFeature {
    float u = 0.0f;
    float v = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    int featureId = -1;
};

/**
 * Current/latest VINS observations that already have a triangulated 3D point.
 * u/v are VINS-image pixels; x/y/z are in the current VINS world frame.
 */
int vinsGetCurrentWorldFeatures(
    VinsWorldFeature* out,
    int maxFeatures);

/** Camera (not IMU body) pose in the current VINS world. */
bool vinsGetCameraPoseMatrix(
    float outRwc[9],
    float outTwc[3]);
/** SENSOR_TIMESTAMP of the last image that actually reached Estimator::processImage. */
std::uint64_t vinsLastProcessedImageTimestampNs();
