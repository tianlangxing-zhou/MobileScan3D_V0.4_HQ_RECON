#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

// 外观跟踪后端接缝（抽象接口 + NanoTrack 适配 + LightTrack-ncnn 骨架）。
// 放在这里而不是 cpp 里，是因为 ObjectTracker 持有一个 unique_ptr<AppearanceTracker>。
#include "appearance_tracker.h"

enum class TargetState
{
    OFF = 0,
    ARMED = 1,
    ACQUIRING = 2,
    TRACKING = 3,
    LOST = 4,
    // 目标短暂滑出画面时的中间态。旧行为是直接进 LOST，而 track() 开头
    // 就 `state != TRACKING -> return`，连 NanoTrack 都不会再跑 ——
    // 用户把物体移回画面也必须重新点一次。REACQUIRING 给 Nanotrack 一个
    // 约 1.5s 的找回窗口，窗口内自己找回来就无缝回到 TRACKING，超时才判 LOST。
    REACQUIRING = 5
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
    // 深度样本的「有效程度」。目标出界时 ROI 会被裁到只剩几行，
    // ROI 里真正拿到有效深度的像素会掉到个位数，此时 P10/P50/P90 会退化成
    // 同一个值（实机出现过 P10=P50=P90=5.14128），不能当作可用观测。
    //   depthRoiArea      —— ROI 像素总数
    //   depthValidPixels  —— ROI 内 isfinite(d) && d>0 的像素数
    //   depthSampleCount  —— 实际参与分位统计的样本数（当前等于 validPixels，
    //                        独立保留是为了将来引入 outlier 剔除时不必改报告格式）
    int depthValidPixels = 0;
    int depthSampleCount = 0;
    int depthRoiArea = 0;
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
    int kltGoodPoints = 0;
    uint64_t nanoForcedUpdates = 0;
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
    // 目标中心的相机归一化坐标，**不裁剪**。
    // 为什么不复用 bbox：目标整块滑出画面时 visibleBox 会退化成空矩形，
    // UI 连「目标往哪个方向去了」都无从判断，只能干等。中心点即使跑到
    // [0,1] 之外也仍然保留方向信息，UI 才能给出「向左移动手机找回目标」。
    float centerXNorm = 0.f;
    float centerYNorm = 0.f;
    // 连续「可见比例 < 12%」的帧数，native 用它判定是否进入 REACQUIRING。
    // 上报给 UI 是为了让报告能区分「刚出界」和「出界很久了」。
    int edgeLostFrames = 0;
    // ---- V0.12 外观/KLT 一致性门控（camera-rate）----
    // Nano box 与 KLT box 的 IoU（同一帧、灰度帧归一化坐标）。目标离开画面后
    // KLT 常常继续在背景纹理上跟出一个「看起来很正常的框」，而 Nano（外观判别）
    // 会立刻给出一个完全不同的位置 —— 两者长期不一致就是最直接的「跟丢了」证据，
    // 比等 PresenceGate（要 mask + 深度，慢好几帧）快得多。
    float nanoKltIou = 1.0f;
    // 因一致性门控而直接进入 REACQUIRING 的累计次数。
    uint64_t nanoKltRejects = 0;
    // ---- PresenceGate（目标存在性判定）----
    // **box 还在画面里 != 物体还在**。目标离开后 tracker 常常在背景纹理上继续
    // 找到一个「看起来正常」的框（OpenCV tracking #619 / NanoTrack 均有记录），
    // 所以判定依据必须是「外观 + Mask + 运动 + 中心一致」四项合成，
    // 而不是「bbox 是否还在画面内」。
    //   presenceValid      —— 本帧目标是否真的在（UI 据此决定绿框画不画）
    //   presenceLostCount  —— 累计被判「目标不在」的次数
    bool presenceValid = true;
    uint64_t presenceLostCount = 0;
    // ---- V0.13 Immutable Target Identity ----
    // 首次框选那一刻保存的**永不更新**的目标外观锚点。Nano 和 KLT 会被
    // 设计成互相确认（adoptNanoBox 用 Nano 框重新播种 KLT，并把 inlierRatio
    // 写回 1.0），所以它们的「一致性」完全不能证明目标正确 —— 实机上两个
    // tracker 会一起漂到同一块背景纹理上，IoU 0.78 / inlier 1.0 却已经跟错人。
    // 唯一独立的判据是：候选框的样子还像不像**当初被框住的那个东西**。
    //   identityScore        —— 本帧对候选框的身份匹配分（-1 = 无法判定）
    //   identityRejects      —— 因身份不通过而拒绝采纳 Nano 的累计次数
    //   bboxScaleFromInitial —— 当前候选框相对初始框的膨胀倍数（取长宽较大者）
    //   identityAnchorReady  —— 锚点是否可用（框太小/太糊时为 false，此时不门控）
    float identityScore = -1.f;
    uint64_t identityRejects = 0;
    float bboxScaleFromInitial = 1.f;
    bool identityAnchorReady = false;
    // 外观后端接缝（见 appearance_tracker.h）。当前实际生效的是 nanotrack，
    // lighttrack-ncnn 因未链接 ncnn 而如实回退。
    bool appearanceAvailable = false;
    std::string appearanceBackend;
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
    // 用户手指拖出的矩形（相机归一化坐标，允许任意方向）。
    // 固定 240x240 的点选 ROI 与目标真实形状无关：点一个细长瓶子时会把
    // 周围背景一起塞进 ROI，KLT 追的其实是「物体 + 墙 + 桌子」的混合纹理，
    // 绿色框自然贴不住目标。拖框让用户直接把 bbox 告诉 tracker，
    // 比立刻上 SAM/YOLO 更实用，而且几乎没有额外算力。
    bool requestTargetRect(float x0, float y0, float x1, float y1);
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

    /**
     * 由 native 的目标 Mask / PresenceGate 驱动：判定「目标已经不在画面里」。
     *
     * 与 edgeLostFrames_ 那条路径的区别：那条是「bbox 几何上滑出画面」，
     * 要连续 3 帧才动作，而且物体走了、tracker 停在墙上时它**永远不会触发**。
     * 这条是语义判定，连续两帧就能定，且和 bbox 位置无关。
     * 行为：TRACKING -> REACQUIRING，confidence 清零，并清掉找回计数器，
     * 让 track() 里那段 Nano 找回逻辑立刻接手。
     */
    void markPresenceLost();

    /** 逐帧上报 PresenceGate 的结论（不改状态机，只更新诊断位）。 */
    void setPresenceValid(bool valid);

    /**
     * 配置外观后端（接缝，见 appearance_tracker.h）。
     * 当前工厂只会返回 nanotrack（LightTrack-ncnn 需要 ncnn 运行库，本工程未链接），
     * 失败/不可用时保留 nullptr 并如实记录到 info_.appearanceBackend。
     */
    bool configureAppearance(const std::string& lighttrackParam,
                             const std::string& lighttrackBin,
                             const std::string& nanoBackbone,
                             const std::string& nanoHead);

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
    // V0.12：Nano 与 KLT 框「明显不一致」的**连续**帧数。单帧抖动不判，
    // 连续 kNanoMismatchFrames 帧才进 REACQUIRING。
    int appearanceMismatchFrames_ = 0;
    // REACQUIRING 已经持续的帧数；超过 kReacquireTimeoutFrames 才判 LOST。
    // 刻意不用「时间」，因为 track() 的调用频率由相机帧率决定，
    // 帧数计数在掉帧时反而更保守（不急于判死）。
    int reacquireFrames_ = 0;
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
    // 被 KLT 弱帧分支"强制"触发的 Nano 更新次数（区别于每 5 帧的周期性更新）
    uint64_t nanoForcedUpdates_ = 0;
    uint64_t nanoWeakAttempts_ = 0;
    uint64_t nanoWeakRecoveries_ = 0;
    float lastNanoScore_ = 0.f;
    // V0.12: Nano box 的「新鲜度」（距上次成功更新过了多少帧）。
    // 一致性门控只在 Nano **刚刚**更新过的那一帧可信 —— Nano 正常态每 5 帧才跑
    // 一次，拿一个 5 帧前的框去和当前 KLT 框比对会天天误报。
    int nanoBoxAge_ = -1;
    // V0.12: 一致性门控触发次数。
    uint64_t nanoKltRejects_ = 0;

    // ---- V0.13：不可变身份锚点（只在用户框选那一帧写入，之后只读）----
    cv::Mat identityTemplate_;      // 归一化尺寸的灰度外观锚点（固定高度）
    bool identityReady_ = false;
    float identityAspect_ = 1.f;    // 初始框 w/h
    float identityHalfWpx_ = 1.f;   // 初始框半宽（灰度帧像素坐标）
    float identityHalfHpx_ = 1.f;
    uint64_t identityRejects_ = 0;

    // 外观后端接缝（见 appearance_tracker.h）。
    // 当前工厂只会返回 nanotrack，lighttrack-ncnn 因本工程未链接 ncnn 而回退，
    // 所以这里为 nullptr 时行为与改造前**完全一致**。
    std::unique_ptr<AppearanceTracker> appearance_;
    bool appearanceInit_ = false;

    // 真实彩色帧（YUV420 -> BGR），保护在 mutex_ 之下
    cv::Mat colorFrame_;
    bool colorFrameValid_ = false;
    uint64_t colorFrameTs_ = 0;
    uint64_t colorFrameCalls_ = 0;

    std::atomic<bool> pendingSelect_{false};
    std::atomic<float> pendingU_{0.5f};
    std::atomic<float> pendingV_{0.5f};

    // 拖框选择。与 pendingSelect_ 互相排斥：后设的那一个生效
    // （requestTarget 清 rect 标志，requestTargetRect 清 tap 标志）。
    std::atomic<bool> pendingRectSelect_{false};
    std::atomic<float> pendingRectX0_{0.f};
    std::atomic<float> pendingRectY0_{0.f};
    std::atomic<float> pendingRectX1_{1.f};
    std::atomic<float> pendingRectY1_{1.f};

    void setError(const std::string& msg);
    void markLost(const std::string& reason);

    // NanoTrack 辅助：取输入帧 / 归一化框互转 / 强制更新 / 采纳 Nano 框并重播种
    cv::Mat resolveNanoFrame(const cv::Mat& grayOwned, bool* usedColor) const;
    static cv::Rect normToRect(float x0, float y0, float x1, float y1, int w, int h);
    bool runNanoUpdate(const cv::Mat& frame, int width, int height, cv::Rect& outRect);
    /**
     * 采纳 Nano 框并**用它重新播种 KLT**。
     *
     * V0.13 起返回是否真的采纳：门控不通过时提前 return false、
     * **一个状态字节都不写**，调用方据此保持原来的 KLT 状态继续跟踪。
     * 这是「身份门只切断 Nano -> 状态机这条路径，绝不切断 KLT 主链」的
     * 落地方式 —— 所以身份门不会变成新的死锁（V0.7.0.1 的教训）。
     */
    bool adoptNanoBox(const cv::Rect& nanoBox, int nanoW, int nanoH,
                      const cv::Mat& grayOwned, uint64_t timestamp);
    void clearWeakKlt();

    // ---- V0.13 Immutable Target Identity ----
    // 保存身份锚点。只在 updateFrame() 里「用户给了新 ROI」那一帧调用。
    void captureIdentityAnchor(const cv::Mat& gray, const cv::Rect& roi);
    // 候选框与身份锚点的归一化互相关峰值；返回 -1 表示**无法判定**
    // （锚点不可用 / 搜索窗太小 / 目标大半出界）—— 调用方必须把 -1 与
    // 「分数低」区分对待：前者放行，后者拒绝，否则贴边时会误杀。
    float computeIdentityScore(const cv::Rect& box, const cv::Mat& gray) const;
    // 释放身份锚点（新目标 / 关锁 / 会话重置）。
    void releaseIdentityAnchor();
};
