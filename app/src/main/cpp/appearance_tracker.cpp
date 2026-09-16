#include "appearance_tracker.h"

#include <cmath>
#include <utility>

namespace {

// ------------------------------------------------------------------ NanoTrack
class NanoTrackAppearance : public AppearanceTracker {
public:
    NanoTrackAppearance(std::string backbone, std::string head)
        : backbone_(std::move(backbone)), head_(std::move(head)) {
        // 与 ObjectTracker::configureNano() 一样，在构造时就尝试加载模型，
        // 这样 ready() 能立刻反映「模型到底有没有加载成功」，而不是等到 init()。
        try {
            cv::TrackerNano::Params p;
            p.backbone = backbone_;
            p.neckhead = head_;
            tracker_ = cv::TrackerNano::create(p);
            ready_ = !tracker_.empty();
        } catch (const cv::Exception&) {
            tracker_.release();
            ready_ = false;
        }
    }

    const char* backendName() const override { return "nanotrack"; }
    bool ready() const override { return ready_; }

    bool init(const cv::Mat& bgr, const cv::Rect2f& box) override {
        if (!ready_ || bgr.empty() || box.width < 2.f || box.height < 2.f) {
            return false;
        }
        try {
            const cv::Rect r(static_cast<int>(std::lround(box.x)),
                             static_cast<int>(std::lround(box.y)),
                             static_cast<int>(std::lround(box.width)),
                             static_cast<int>(std::lround(box.height)));
            tracker_->init(bgr, r);
            initialized_ = true;
            // 初始化帧不产生分数，和改造前的行为一致
            return false;
        } catch (const cv::Exception&) {
            initialized_ = false;
            return false;
        }
    }

    bool update(const cv::Mat& bgr, cv::Rect2f& box, float& score) override {
        if (!ready_ || !initialized_ || bgr.empty()) {
            return false;
        }
        try {
            cv::Rect r;
            if (!tracker_->update(bgr, r)) {
                return false;
            }
            score = static_cast<float>(tracker_->getTrackingScore());
            box = cv::Rect2f(static_cast<float>(r.x), static_cast<float>(r.y),
                             static_cast<float>(r.width), static_cast<float>(r.height));
            return true;
        } catch (const cv::Exception&) {
            return false;
        }
    }

    void reset() override { initialized_ = false; }

private:
    std::string backbone_;
    std::string head_;
    cv::Ptr<cv::TrackerNano> tracker_;
    bool ready_ = false;
    bool initialized_ = false;
};

// ------------------------------------------------------------ LightTrack-ncnn
/**
 * LightTrack-ncnn 骨架（**当前不可用**）。
 *
 * 真实实现需要三样现在都没有的东西：
 *   1. ncnn 运行库（CMake 里没有 ncnn 目标，本地也没有 arm64-v8a 预编译库）；
 *   2. LightTrack 的 param/bin 模型文件（未随工程分发）；
 *   3. 与 NanoTrack 对齐的 crop / 归一化 / 峰值提取后处理。
 *
 * 刻意不做「假装能跑」的桩实现：如果这里返回 true，上层会拿一个恒定的框当成
 * 「外观纠偏成功」，那比不接还糟。所以 ready() 恒为 false，init/update 恒返回
 * false，让上层如实回退到 NanoTrack，并在报告里写清楚原因。
 */
class LightTrackNcnnAppearance : public AppearanceTracker {
public:
    LightTrackNcnnAppearance(std::string param, std::string bin)
        : param_(std::move(param)), bin_(std::move(bin)) {}

    const char* backendName() const override { return "lighttrack-ncnn"; }
    bool ready() const override { return false; }
    bool init(const cv::Mat&, const cv::Rect2f&) override { return false; }
    bool update(const cv::Mat&, cv::Rect2f&, float&) override { return false; }
    void reset() override {}

private:
    std::string param_;
    std::string bin_;
};

} // namespace

std::unique_ptr<AppearanceTracker> makeNanoTrackAppearance(const std::string& backbone,
                                                           const std::string& head) {
    if (backbone.empty() || head.empty()) {
        return nullptr;
    }
    return std::make_unique<NanoTrackAppearance>(backbone, head);
}

std::unique_ptr<AppearanceTracker> makeLightTrackNcnn(const std::string& modelParam,
                                                      const std::string& modelBin,
                                                      std::string* reason) {
    // 目前一律不可用：本工程未链接 ncnn，也没有 LightTrack 权重。
    if (reason) {
        if (modelParam.empty() || modelBin.empty()) {
            *reason = "lighttrack-ncnn: model files not configured";
        } else {
            *reason = "lighttrack-ncnn: ncnn backend not linked in this build";
        }
    }
    (void)modelParam;
    (void)modelBin;
    return nullptr;
}

std::unique_ptr<AppearanceTracker> createAppearanceTracker(const std::string& lighttrackParam,
                                                           const std::string& lighttrackBin,
                                                           const std::string& nanoBackbone,
                                                           const std::string& nanoHead,
                                                           std::string* reason) {
    std::string ltReason;
    auto lt = makeLightTrackNcnn(lighttrackParam, lighttrackBin, &ltReason);
    if (lt && lt->ready()) {
        if (reason) *reason = "appearance backend: lighttrack-ncnn";
        return lt;
    }

    auto nano = makeNanoTrackAppearance(nanoBackbone, nanoHead);
    if (nano) {
        if (reason) {
            *reason = "appearance backend: nanotrack";
            if (!ltReason.empty()) {
                *reason += " (" + ltReason + ")";
            }
        }
        return nano;
    }

    if (reason) {
        *reason = ltReason.empty() ? "appearance backend: none" : ltReason;
    }
    return nullptr;
}
