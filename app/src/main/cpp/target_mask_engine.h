#pragma once

#include <opencv2/opencv.hpp>

/**
 * 目标分割 Mask 引擎 —— 「框住目标」和「知道目标是什么」是两件事。
 *
 * ## 为什么不能再围着 bbox 的阈值打转
 *
 * 旧实现里 Object Lock 只提供三样东西：一个矩形框、ROI 内的深度分位数、
 * 以及一层调试点。**正式的重建（Gaussian / TSDF）从来没有被这个框影响过** ——
 * `TARGET_DEPTH_FILTER_ENABLED` 是 0，完整的一帧 depth 照旧进 `fuseDepth()`。
 * 于是「锁定物体 → 只重建这个物体 → 只显示这个物体」在当前架构里根本不存在，
 * 天花板、墙、桌子照样进点云。
 *
 * 而且 bbox 本身也不能当作「物体」：
 *   - 用户拖框总有误差，框里必然混进背景；
 *   - tracker 失配时框会停在墙上（真实物体已经走了，框还在）；
 *   - 目标形状千变万化，矩形框天然不是它的轮廓。
 *
 * ## 这一层的做法
 *
 * 以 seed（用户点 / 框中心）周围 7x7 的**中位深度**为基准，在 searchBox 内做
 * 一次深度带通得到 candidate，然后只保留 **seed 所在的那个连通域**，最后做一次
 * 小 close 把细缝补上。
 *
 * 关键性质：**它只利用同一帧内部的相对深度连续性，不依赖 Depth 的绝对米制准确度。**
 * 这一点非常重要 —— 实机上目标 raw depth 中位数约 5.56m，而 VINS 三角化只有
 * 约 0.54m，两者差了十几倍（全局尺度问题至今没有解决）。任何「按绝对深度门限
 * 切目标」的方案都会被这个尺度差直接打死；而「跟 seed 深度差不多的像素」不受影响。
 */
struct TargetMaskStats {
    /** mask 是否可用（连通域面积达到下限） */
    bool valid = false;
    /** mask 内像素数 */
    int area = 0;
    /** mask 质心（depth 图像像素坐标） */
    float centerX = 0.f;
    float centerY = 0.f;
    /** seed 处 7x7 的深度中位数 */
    float seedDepth = 0.f;
    /** 实际使用的深度容差（米） */
    float tolerance = 0.f;
    /** searchBox 内落入深度带的像素数（在连通域筛选之前） */
    int candidatePixels = 0;
    /** searchBox 的面积 */
    int searchPixels = 0;
    /** seed 周围是否取到了有效深度（false = 这一帧无法建 mask） */
    bool seedValid = false;
    /** 拒绝原因，直接进报告 */
    const char* rejectReason = "";
};

class TargetMaskEngine {
public:
    /**
     * @param depth      深度图（w*h，float，米；非有限值 / <=0 视为无效）
     * @param width      深度图宽
     * @param height     深度图高
     * @param searchBox  搜索范围（一般是 tracker bbox 稍微外扩一点）
     * @param seed       种子点（一般取 bbox 中心）
     * @param mask       [out] 与 depth 同尺寸的 CV_8U，目标内 = 255，其余 = 0
     * @param stats      [out] 诊断数据
     * @return true 表示 mask 可用（等价于 stats.valid）
     */
    bool build(const float* depth, int width, int height,
               const cv::Rect& searchBox, cv::Point seed,
               cv::Mat& mask, TargetMaskStats& stats);

    /** mask 面积下限：低于这个值不足以说明找到了目标（可能只是几个噪点）。 */
    static constexpr int kMinMaskArea = 100;

private:
    /** seed 周围 (2*radius+1)^2 窗口内的深度中位数；没有有效值时返回 0 */
    static float medianAroundSeed(const float* depth, int width, int height,
                                  cv::Point seed, int radius);

    /** 在 seed 周围找最近的、深度有效的像素（seed 本身不落在目标上时用） */
    static bool findValidSeed(const float* depth, int width, int height,
                              cv::Point hint, cv::Point& out);

    // 复用缓冲，避免每帧重新分配（depth 帧可能到 30Hz）
    cv::Mat candidate_;
    cv::Mat labels_;
    cv::Mat statsArr_;
    cv::Mat centroids_;
    cv::Mat kernel_;
};
