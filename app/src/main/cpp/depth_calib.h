#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
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

    // ---- V0.13.25 标定健康度诊断 ----
    //
    // 「模型发平、整场景像一块平板」的根因不是网络输出坏，而是**拟合样本的
    // 纵深不足**：VINS 稀疏三角化点全部落在很薄的一层深度里时，
    //   1/z = scale*d + shift
    // 的因变量 1/z 几乎没有变化，最小二乘只能给出 scale~0 —— 深度对输入 d
    // 完全不敏感，整个场景被钉死在 z = 1/shift 附近。
    //
    // refDepthSpanRel  = (p90(z)-p10(z)) / p50(z)  —— 参考深度的相对纵深
    // outputInvZSpan = |1/z(d90)-1/z(d10)|，两种模型都以 1/米计。
    float refDepthSpanRel = 0.f;
    float outputInvZSpan = 0.f;
    // V0.13.35：输出**米制**深度的相对跨度 (p90(ẑ)-p10(ẑ))/p50(ẑ)，ẑ 为
    // 本帧映射对 base 样本算出的预测深度。这是「映射是否保留场景结构」的
    // 直接度量 —— vc175 实机（PLK110 22:30 会话）实锤：outputInvZSpan=0.196
    // 能过 0.15 的旧门（shift~6 时 1/z 绝对值大，小绝对跨度仍是好门），
    // 但米制 z 只剩 0.171~0.187（16mm 厚的墙），整场景深度结构被摧毁。
    float outputDepthSpanRel = 0.f;

    /**
     * V0.13.4：**输入数值域变更时的显式重参数化**。
     *
     * 背景：非米制深度 provider 输出的是网络原始值 q 的仿射
     *   `d = A*q + B`，而 A/B 由会话级 min/max 决定，会随扫描推进变化。
     * 标定拟合的是 `z = a*d + b`（或 `1/z = a*d + b`），一旦 d 的含义变了
     * 而 (a,b) 还冻结着，同一个物体就会随扫描推进整体膨胀/收缩。
     *
     * 设旧映射 `d_old = A_old*q + B_old`、新映射 `d_new = A_new*q + B_new`，
     * 则 `d_old = (A_old/A_new)*(d_new - B_new) + B_old`，代入
     * `z = a_old*d_old + b_old` 得
     *
     *   `a_new = a_old * A_old / A_new`
     *   `b_new = b_old + a_old*B_old - a_new*B_new`
     *
     * 线性模型与逆深度模型都适用（两者对 d 都是仿射）。
     *
     * @return 是否真的改了（返回 false 表示输入不适定，调用方应保持原值）
     */
    bool reparameterizeLinearInput(float aOld, float bOld, float aNew, float bNew) {
        if (!valid || !std::isfinite(scale) || !std::isfinite(shift)) return false;
        if (!std::isfinite(aOld) || !std::isfinite(bOld) ||
            !std::isfinite(aNew) || !std::isfinite(bNew)) return false;
        if (std::fabs(aOld) < 1e-12f || std::fabs(aNew) < 1e-12f) return false;
        const double k = static_cast<double>(aOld) / aNew;
        const double scaled = scale * k;
        if (!std::isfinite(scaled) || std::fabs(scaled) > std::numeric_limits<float>::max()) return false;
        const float newScale = static_cast<float>(scaled);
        const double shifted = static_cast<double>(shift) +
            static_cast<double>(scale) * bOld - static_cast<double>(newScale) * bNew;
        if (!std::isfinite(shifted) || std::fabs(shifted) > std::numeric_limits<float>::max()) return false;
        const float newShift = static_cast<float>(shifted);
        scale = newScale;
        shift = newShift;
        return true;
    }

    /**
     * V0.13.4：把一个「旧数值域下的 d」换算到新数值域。
     * `d_new = (A_new/A_old)*(d_old - B_old) + B_new`
     * 用于 epoch 参考深度等随帧携带的历史标量。
     */
    static float remapRaw(float d, float aOld, float bOld, float aNew, float bNew) {
        if (!std::isfinite(d) || std::fabs(aOld) < 1e-12f) return d;
        return (aNew / aOld) * (d - bOld) + bNew;
    }

    /** 模型输出 d -> 米制深度。未标定或结果非正时返回 fallback。 */
    float toMetric(float d, float fallback) const {
        // V0.10 FIX: inverse-depth 1/z = a*d+b may have a negative slope.
        // Gate the resulting value, not the sign of a.
        if (!valid ||
            !std::isfinite(scale) ||
            !std::isfinite(shift) ||
            !std::isfinite(d)) {
            return fallback;
        }
        // Keep cancellation in double even though the stored model/input are float.
        const double v = static_cast<double>(scale) * d + shift;
        if (inverseDepthModel) {
            if (!(v > 1e-6f) || !std::isfinite(v)) {
                return fallback;
            }
            return 1.f / v;
        }
        if (!(v > 1e-4f) || !std::isfinite(v) || v > std::numeric_limits<float>::max()) {
            return fallback;
        }
        const float metric = static_cast<float>(v);
        return std::isfinite(metric) ? metric : fallback;
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
                                int maxIterations = 10,
                                bool forceInverse = false);

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
        bool forceInverseDepth = false;

        // ---- V0.13.25 标定健康门 ----
        //
        // 参考深度相对纵深下限 (p90(z)-p10(z))/p50(z)。
        // 低于此值说明这一帧的 VINS 参考点没有纵深结构，拟合出的 scale
        // 必然接近 0（场景会被压平）。此时**拒绝该帧**，保持上一版标定。
        //
        // 首次建立（calib_ 尚无效）时用 minRefDepthSpanRel * firstBuildRelax
        // 作为兜底，避免一开始完全拿不到标定而退回未标定的原始深度。
        float minRefDepthSpanRel = 0.30f;
        float firstBuildRelax = 0.5f;
        // 输出逆深度跨度下限（1/米）；首次建立也检查，使用 firstBuildRelax 放宽。
        float minOutputInvZSpan = 0.15f;
        // ---- V0.13.35 输出米制跨度门 ----
        //
        // outputInvZSpan 在 shift 较大时是**弱门**：1/z 的大绝对值让同样小的
        // 米制结构仍能凑出 0.15 的 1/米跨度（vc175 实锤：shift=5.93、
        // invZSpan=0.196 过门，z 却只剩 16mm 的跨度 —— 满屏 17cm 平板墙）。
        // 直接在米制 z 域设门：
        //   绝对门   ẑ 的 spanRel ≥ minOutputDepthSpanRel（首次建立放宽）
        //   保持门   ẑ 的 spanRel ≥ outputSpanPreservation × 参考 spanRel
        // 保持门不放宽：参考跨度本身已被 minRefDepthSpanRel 门住，标定没有
        // 理由把 VINS 看得到的纵深再摧毁一半以上。
        float minOutputDepthSpanRel = 0.20f;
        float outputSpanPreservation = 0.5f;
        // 关掉健康门（用于对照实验 / 回归测试）。
        bool enforceCalibHealthGate = true;
    };

    void reset();
    void setConfig(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    /** 用一帧的配对样本更新。返回本次是否被接受。 */
    bool update(const std::vector<float>& d, const std::vector<float>& z);

    /**
     * V0.13.4：输入数值域变更时重参数化**已收敛的标定**。
     *
     * 与 [DepthCalibration::reparameterizeLinearInput] 同式，作用在内部
     * 持有的 `calib_` 上。之所以必须显式支持：provider 的归一化范围在
     * epoch 冻结之后仍可能扩张，若不重参数化，冻结的 (a,b) 会把新的 d
     * 解释成完全不同的米制深度 —— 表现为几何随扫描推进持续漂移。
     */
    bool reparameterizeInput(float aOld, float bOld, float aNew, float bNew) {
        return calib_.reparameterizeLinearInput(aOld, bOld, aNew, bNew);
    }

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

/**
 * 逐像素时序一致性的三态判定。
 *
 * 0 = 未测试，1 = 一致，2 = 冲突。
 *
 * **为什么必须是三态而不是布尔**：一帧里总有大量像素拿不到可比较的上一帧
 * 观测 —— 刚刚进入视野的新区域、被前景遮挡后重新露出的背景、重投影出界。
 * 这些像素的深度**不是错的**，只是无从对照。把它们和真正的冲突（飞点、
 * 动态物体、估计失败）混成同一类"不可信"去降权，会让模型边缘系统性变薄、
 * 绕行一圈后新露出的表面补不上 —— 属于「过滤过度」，比飞点更难发现。
 */
enum TemporalMask : uint8_t {
    kTemporalUntested = 0,
    kTemporalAgree = 1,
    kTemporalDisagree = 2,
    kTemporalOccluded = 3, // current foreground hides the previous surface
    kTemporalDisoccluded = 4, // previous foreground/edge reveals background
};

/**
 * Reproject previous depth using a nearest-surface z-buffer, then classify current pixels.
 * Conflicts count once per current pixel. Large jumps near silhouettes are visibility
 * changes (neutral weight); interior mismatches are conflicts. This is a heuristic,
 * not a complete multi-view occlusion detector.
 * mask is cleared even for invalid camera parameters. Dimensions are limited to 4096.
 * projectedDepth optionally supplies reusable w*h scratch storage (must not alias inputs).
 * outTested excludes unknown/occluded/disoccluded pixels; outAgree counts agreements.
 */
int temporalConsistencyMask(const float* prevDepth, const float* curDepth,
                            int w, int h,
                            float fx, float fy, float cx, float cy,
                            const float Rrel[9], const float trel[3],
                            uint8_t* mask, int step = 1,
                            float minAbsTol = 0.015f,
                            float relTol = 0.025f,
                            int* outTested = nullptr,
                            int* outAgree = nullptr,
                            std::vector<float>* projectedDepth = nullptr);
