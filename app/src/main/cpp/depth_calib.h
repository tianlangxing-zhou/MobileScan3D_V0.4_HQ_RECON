#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// ============================================================================
//  深度尺度鲁棒标定
// ============================================================================
//
// 旧做法只有一个 median ratio：
//
//      scale = VINS 深度中位数 / 模型深度中位数        -> EMA
//
// 问题有三个：
//   1. 中位数对「同一帧里混进了墙/地面」非常敏感 —— 一个大平面就能把中位数
//      整个带偏，而目标只占画面很小一块；
//   2. 只估一个乘性 scale，估不了偏移。模型输出与 VINS 米制深度之间通常还有
//      常数项（尤其是 1/z 形式），一个 scale 无论如何都压不平；
//   3. 没有任何离群点剔除 —— 三角化失败的 VINS 点（深度巨大或为负）会直接
//      进中位数。
//
// 现在改成标准做法：
//
//   MAD 剔除   ->  Huber IRLS 加权最小二乘  ->  EMA 时序平滑
//
// 并同时支持两种模型（按鲁棒相对残差自动选更合适的那个）：
//
//   线性模型   z      = a * d + b            （模型直接输出深度）
//   逆深度模型 1 / z  = a * d + b            （模型输出 disparity / 相对逆深度）
//
// 逆深度模型是单目深度网络最常见的形式（Depth Anything 的 relative 版本就是
// 输出 disparity），所以这两个都要支持，不能只做一种。
//
// **门控**：拟合不达标（样本太少 / 相对残差太大 / a 不适定）时 valid=false，
// 调用方必须退回未标定的深度 —— 一个坏标定比不标定危险得多。
// ============================================================================

struct DepthCalibration {
    /** 线性模型: z = scale * d + shift。逆深度模型: 1/z = scale * d + shift。 */
    float scale = 1.f;
    float shift = 0.f;
    /** 标定可信度 [0,1]（由样本数与相对残差综合得到）。 */
    float confidence = 0.f;
    int samples = 0;
    int inliers = 0;
    int rejected = 0;
    /** 中位相对残差 median(|z_pred - z_vins| / z_vins)。 */
    float medianRelativeResidual = 0.f;
    bool inverseDepthModel = false;
    bool valid = false;

    /** 模型输出 d -> 米制深度。未标定或结果非正时返回 fallback。 */
    float toMetric(float d, float fallback) const {
        if (!valid || !(scale > 0.f) || !std::isfinite(scale)) {
            return fallback;
        }
        const float v = scale * d + shift;
        if (inverseDepthModel) {
            if (!(v > 1e-6f) || !std::isfinite(v)) {
                return fallback;
            }
            return 1.f / v;
        }
        if (!(v > 1e-4f) || !std::isfinite(v)) {
            return fallback;
        }
        return v;
    }
};

/**
 * 单帧鲁棒拟合。
 *
 * @param d            模型原始输出（逐样本）
 * @param z            对应位置的 VINS 三角化深度（米，> 0）
 * @param inverseDepth true 表示同时尝试 1/z = a*d + b 模型
 * @return 拟合结果；样本不足或退化时 valid=false
 */
DepthCalibration fitDepthRobust(const std::vector<float>& d,
                                const std::vector<float>& z,
                                bool inverseDepth = true,
                                int maxIterations = 10);

/** 中位数（会复制输入排序）。空输入返回 0。 */
float medianOf(std::vector<float> v);

/**
 * 带 EMA 时序平滑的标定器。
 *
 * `update()` 每帧调用一次；只有「本次拟合可信 && 与上一状态差异不过分」时才
 * 接受新值，否则保持旧值并累计 rejected（用于诊断「标定一直上不去」）。
 */
class DepthCalibrator {
public:
    struct Config {
        int minSamples = 20;                 // 少于这个样本数不拟合
        int maxSamples = 512;                // 参与拟合的样本上限
        float emaNew = 0.20f;                // 新观测权重（越大跟随越快、越抖）
        float maxMedianRelativeResidual = 0.30f;  // 拟合相对残差上限
        float minConfidence = 0.25f;         // 低于此置信度视为无效
        float maxJumpRatio = 3.0f;           // 新旧 scale 比超过此值视为野值
        bool allowInverseDepth = true;
    };

    void reset();
    void setConfig(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    /** 用一帧的配对样本更新。返回本次是否被接受。 */
    bool update(const std::vector<float>& d, const std::vector<float>& z);

    const DepthCalibration& calibration() const { return calib_; }
    uint64_t acceptedFrames() const { return acceptedFrames_; }
    uint64_t rejectedFrames() const { return rejectedFrames_; }
    uint64_t totalSamples() const { return totalSamples_; }
    const char* lastRejectReason() const { return lastReject_; }

    /** 把模型输出映射成米制深度（未标定时返回 fallback）。 */
    float toMetric(float d, float fallback) const { return calib_.toMetric(d, fallback); }
    bool usable() const { return calib_.valid; }

private:
    Config cfg_{};
    DepthCalibration calib_{};
    uint64_t acceptedFrames_ = 0;
    uint64_t rejectedFrames_ = 0;
    uint64_t totalSamples_ = 0;
    const char* lastReject_ = "";
};

/**
 * 深度时序一致性比例（用于给融合降权）。
 *
 * 把上一帧深度按「上一帧 -> 当前帧」的相对位姿重投影到当前帧，逐像素比较
 * |z_proj - z_cur| 是否在 max(minAbsTol, z_cur * relTol) 之内，返回一致像素比例。
 *
 * @param Rrel / trel  当前帧相对上一帧的位姿（t_cur = Rrel * t_prev + trel）
 */
float temporalConsistencyRatio(const float* prevDepth, const float* curDepth,
                               int w, int h,
                               float fx, float fy, float cx, float cy,
                               const float Rrel[9], const float trel[3],
                               float minAbsTol = 0.015f,
                               float relTol = 0.025f);
