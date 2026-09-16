#pragma once

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

/**
 * 外观跟踪后端抽象 —— 给「每帧高速层」之外再挂一层「外观纠偏层」留的接缝。
 *
 * ## 为什么要这层抽象
 *
 * 当前 ObjectTracker 里外观纠偏和 KLT 是**写死在一起**的（`cv::TrackerNano` 直接
 * 作为成员 `nano_`）。而实机报告指出绿框「物体走了还在」，是典型的 tracker
 * failure：真实目标离开画面后，KLT / NanoTrack 都在背景上找到了可以继续跟踪的
 * 纹理，于是仍报 TRACKING。要治它，最终需要更强的外观模型（LightTrack 在
 * 移动端比 NanoTrack 有明显速度优势，且已有 ncnn Android demo）。
 *
 * 但**整工程替换**会破坏已经稳定的 VINS / Camera2 / 追踪主链，所以这里只做两件事：
 *   1. 把「外观跟踪后端」抽成接口，ObjectTracker 只依赖接口；
 *   2. NanoTrack 仍然是**当前唯一真正可用的实现**，LightTrack-ncnn 作为骨架接入，
 *      在 ncnn 未被链接进本工程时**如实报告不可用并回退**，绝不假装成功。
 *
 * 这样下一轮把 ncnn + LightTrack 权重加进来时，只需要换工厂，不动追踪主链。
 */
class AppearanceTracker {
public:
    virtual ~AppearanceTracker() = default;

    /** 后端名，进报告用（"nanotrack" / "lighttrack-ncnn"） */
    virtual const char* backendName() const = 0;

    /** 后端是否真的可用（模型加载成功 / 依赖已链接） */
    virtual bool ready() const = 0;

    /**
     * 用第一帧和初始框初始化外观模型。
     * @param bgr  三通道 BGR 图像（真实彩色，不要拿灰度复制成三通道）
     * @param box  初始框，bgr 图像像素坐标（float）
     */
    virtual bool init(const cv::Mat& bgr, const cv::Rect2f& box) = 0;

    /**
     * 在下一帧上更新外观模型。
     * @param bgr   [in]  三通道 BGR 图像
     * @param box   [out] 更新后的框（bgr 图像像素坐标）
     * @param score [out] 外观置信度（不同后端量纲不同，只做同后端内比较）
     */
    virtual bool update(const cv::Mat& bgr, cv::Rect2f& box, float& score) = 0;

    virtual void reset() = 0;
};

/**
 * NanoTrack 适配（当前唯一真正可用的后端）。
 *
 * 注意：它仍然是「同一个 cv::TrackerNano」，只是换成了接口形态，
 * 行为与改造前完全一致 —— 这一步刻意不改变任何数值行为。
 */
std::unique_ptr<AppearanceTracker> makeNanoTrackAppearance(const std::string& backbone,
                                                           const std::string& head);

/**
 * LightTrack-ncnn 骨架。
 *
 * 本工程当前**没有链接 ncnn**（CMakeLists 里没有 ncnn 目标，本地也没有可用的
 * 预编译库），所以这个工厂会返回 nullptr，并通过 `reason` 说明原因。
 * 保留它是为了下一轮接入时不必再动追踪主链。
 */
std::unique_ptr<AppearanceTracker> makeLightTrackNcnn(const std::string& modelParam,
                                                      const std::string& modelBin,
                                                      std::string* reason);

/**
 * 按可用性选择外观后端：
 *   1. LightTrack-ncnn（当前不可用）；
 *   2. NanoTrack（有模型路径时）；
 *   3. 都没有 -> nullptr。
 * @param reason [out] 人类可读的选择理由，直接进诊断报告
 */
std::unique_ptr<AppearanceTracker> createAppearanceTracker(const std::string& lighttrackParam,
                                                           const std::string& lighttrackBin,
                                                           const std::string& nanoBackbone,
                                                           const std::string& nanoHead,
                                                           std::string* reason);
