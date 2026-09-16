#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <vector>
#include <deque>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <utility>

#include "vins/vins_bridge.h"
#include "vins/estimator/feature_manager.h"
#include "surfel_engine.h"
#include "vio_engine.h"
#include "depth_fusion.h"
#include "ai_quality.h"
#include "tsdf_engine.h"
#include "ai_backend.h"
#include "keyframe_engine.h"
#include "object_tracker.h"
#include "target_mask_engine.h"
#include "depth_calib.h"
#include "mesh/mesh_engine.h"
#include "export/gltf_exporter.h"
#include "v06/mesh_postprocess.h"
#include "v06/uv_unwrap.h"
#include "v06/texture_baker.h"
#include "v06/textured_glb_exporter.h"
#include <fstream>
#include <opencv2/imgcodecs.hpp>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MobileScan3D", __VA_ARGS__)
#define TARGET_DEPTH_FILTER_ENABLED 0

// 全局状态被 4 个线程并发访问：IMU 线程（nativeOnImu）、相机线程
// （nativeOnCameraFrame）、深度线程（nativeOnDepthMap）、GL 渲染/UI 线程
// （nativeGetGaussians / nativeGetStats）。原代码完全无锁，snaps 的 deque
// 迭代器失效和 g_ 的 vector 扩容都是未定义行为（崩溃/花屏根因）。
static std::mutex gStateMutex;
static NullAiBackend aiBackend;
static SurfelEngine g;
static VioEngine vio;
static DepthFusion df;
static AiQualityEngine ai;
static TsdfEngine tsdf;
// ---------------------------------------------------------------- 目标专用模型
// 与全场景的 g / tsdf **物理隔离**：天花板、墙、地面根本没有进入这两个容器的
// 代码路径。比「给 g 加一个过滤标志」强得多 —— 后者只要有一个调用点漏掉，
// 背景就又会漏进来，而且很难在实机上发现。
// Renderer 只读 targetG，PLY 只导 targetTsdf。
static SurfelEngine targetG;
static TsdfEngine targetTsdf;
static TargetMaskEngine targetMaskEngine;

// ---------------------------------------------------------------- mesh 管线
// 评审 P0-3：旧导出只有顶点、没有三角面，那不是 AR 模型。这里补一条真正的
//   TSDF -> Marching Tetrahedra -> 去小分量/平滑/简化 -> GLB
// 链路。构建是一次性操作（用户点「生成网格」或停止扫描时），不是每帧。
static MeshEngine meshEngine;
static MeshBuildStats meshStats;
static int meshQuality = 1;            // 0 = preview / 1 = normal / 2 = hq
static bool meshDirty = true;          // 体素场变了置脏，避免重复构建
static uint64_t meshBuilds = 0;
static uint64_t meshDirtyMarks = 0;
/** 顶点交错布局：x,y,z, nx,ny,nz, r,g,b —— 9 个 float 一个顶点（与 Kotlin 契约一致）。 */
static constexpr int MESH_VERTEX_FLOATS = 9;

// ---------------------------------------------------------------- V0.6 纹理管线
// 与 mesh 管线分开：这些东西只在「停扫导出」时用一次（几百毫秒到数秒），
// 每帧路径完全不碰。HQ 关键帧由 HqCaptureController 在 HQ 合成 / 降级单帧
// 成功后注册进来，和网格一样按会话清理。
//
// 锁：**统一用 gStateMutex，不再引入第二把 gTextureMutex**。resetMeshPipeline()
// 是在 gStateMutex 下被调用的，再加一把纹理锁就形成 gStateMutex -> gTextureMutex
// 的嵌套；而烘焙路径若反向取锁就是死锁。共用一把锁的代价可以忽略 ——
// 纹理状态只在会话开始/结束与停扫导出时变化，不在每帧路径上。
static MeshPostProcessStats meshCleanupStats;
static std::vector<TextureKeyframe> gTextureKeyframes;
static TextureBakeStats textureBakeStats;
static UvUnwrapStats uvUnwrapStats;

// --------------------------------------------------------------- 深度标定
// 评审 P0-4：把「单一 median ratio」升级成 MAD 剔除 + Huber IRLS + EMA 的
// 鲁棒回归，并同时支持线性 / 逆深度两种模型。
//
// 标定目标刻意选 **VINS 自身尺度**而不是米制：TSDF 融合用的位姿（s.R / s.t）
// 就是 VINS world 下的。深度与位姿必须处在同一尺度，否则点云会整体膨胀或
// 缩小 —— 这正是评审说的「模型膨胀/缩小」。旧代码里深度是「按会话 min/max
// 映射到 [0.4, 6]m 的假米制」，与 VINS 位姿根本不是一个尺度。
static DepthCalibrator depthCalibrator;
static bool depthCalibrationEnabled = true;
static float lastCalibScale = 1.f;
static float lastCalibShift = 0.f;
static float lastCalibConfidence = 0.f;
static int lastCalibSamples = 0;
static bool lastCalibValid = false;
static bool lastCalibInverse = false;
static uint64_t calibFrames = 0;
static uint64_t calibRejectFrames = 0;
static constexpr int kMaxCalibSamples = 256;
static float gCalibSampleBuf[kMaxCalibSamples * 3];

// 时序一致性：上一帧深度按相对位姿重投影到当前帧，比较逐像素一致性，
// 不一致就压低这一帧的融合权重（消「毛刺 / 浮点 / 重影」）。
static std::vector<float> lastFusedDepth;
static float lastFuseR[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
static float lastFuseT[3] = {0.f, 0.f, 0.f};
static bool haveLastFusePose = false;
static bool lastFuseCalibrated = false;
static float lastTemporalRatio = 1.f;
static uint64_t temporalChecks = 0;
static uint64_t temporalRejects = 0;
static cv::Mat targetMaskMat;        // 与 depth 同尺寸的 CV_8U，目标内 255
static TargetMaskStats targetMaskStats;
// PresenceGate 连续失败帧数（只在 gStateMutex 下访问）
static int presenceFailStreak = 0;
// 目标专属深度尺度：scaleTarget = vinsRoiMedian / rawDepthMedian，EMA 平滑。
// 全局「全场 VINS median ÷ 目标 raw median」会把墙/地面/天花板的深度混进来，
// 只有目标 ROI 内的 VINS 三角化特征才是这个物体自己的尺度。
static float targetDepthScaleEma = 0.f;
static int targetDepthScaleSamples = 0;
static float lastTargetVinsMedian = 0.f;
static float lastTargetRawMedian = 0.f;
static int lastTargetVinsFeatureCount = 0;
static uint64_t targetMaskFrames = 0;
static uint64_t targetFuseFrames = 0;
static uint64_t presenceLostFrames = 0;
static KeyframeEngine kf;
static std::shared_ptr<ObjectTracker> objectTracker;
static bool vk = false;
static int mode = 0;
static uint64_t frames = 0;
static uint32_t lastKF = 0;
static float gFx = 1000.f, gFy = 1000.f, gCx = 0, gCy = 0;
static bool haveExternalDepth = false;
static float vinsT[3] = {0, 0, 0};
static float vinsQ[4] = {0, 0, 0, 1};
static bool vinsPoseOk = false;
static bool haveLastGoodVinsPose = false;
static float lastGoodVinsT[3] = {0.f, 0.f, 0.f};
static bool vinsLostAfterInit = false;
static float rawVinsT[3] = {0.f, 0.f, 0.f};
static float rawVinsQ[4] = {0.f, 0.f, 0.f, 1.f};
static float acceptedVinsT[3] = {0.f, 0.f, 0.f};
static float acceptedVinsQ[4] = {0.f, 0.f, 0.f, 1.f};
static float lastVinsStep = 0.f;
static uint64_t vinsRejectCount = 0;
static const char* vinsRejectReason = "";
static bool haveFirstReject = false;
static float firstRejectStep = 0.f;
static float firstRejectT[3] = {0.f, 0.f, 0.f};
static float firstRejectVelocity = 0.f;
static float firstRejectAccBias = 0.f;
static float firstRejectGyroBias = 0.f;
static double firstRejectTd = 0.0;
static uint64_t vinsFrames = 0;
static uint64_t depthFrames = 0;
static double lastVinsMs = 0.0;

struct FrameSnap {
    uint64_t ts = 0;
    float t[3] = {0, 0, 0};
    float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<uint8_t> rgb;
    int w = 0;
    int h = 0;
};

static std::deque<FrameSnap> snaps;
static const int SNAP_W = 160;
static const int SNAP_H = 120;
static int gVinsW = 640;
static int gVinsH = 480;

// NanoTrack 真实彩色输入的节流计数：每 5 帧生成一张 BGR 缩略图
static uint64_t gTrackerColorFrameCounter = 0;

// tracker 的灰度输入缓冲（640 长边）。nativeOnCameraFrame 只由相机线程串行调用，
// 所以这个缓冲不需要额外加锁。
static std::vector<uint8_t> gTrackerGray;

// ---------------------------------------------------------------- AR pose 历史
/**
 * 一帧「被采纳的 VINS 位姿」样本，带它自己的 SENSOR_TIMESTAMP。
 *
 * 为什么必须带时间戳：Renderer 过去直接拿 nativeGetRenderPose()，
 * 也就是「回调发生这一刻」的最新 pose。但屏幕上正在显示的 Camera frame
 * 有自己的 SENSOR_TIMESTAMP，二者不是同一时刻。手机静止时看不出来，
 * 一转起来点云就会漂/甩/跟不上 —— 因为 AR 拿「现在」的 pose 去画
 * 「过去某一帧」的画面。
 *
 * 正确做法：拿 Preview 的时间戳回来查表，取那一时刻的 pose。
 */
struct RenderPoseSample {
    uint64_t ts = 0;
    float R[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    float t[3] = {0.f, 0.f, 0.f};
};

static std::deque<RenderPoseSample> renderPoseHistory;
static constexpr size_t kRenderPoseHistoryMax = 120;
// 查询允许的最大 pose 年龄；超过就说明这段历史里没有对应时刻的位姿
static constexpr int64_t kRenderPoseMaxAgeNs = 80'000'000LL; // 80ms

// ------------------------------------------------- 当前帧 target depth 调试层
/**
 * AR 调试层：**只取当前一帧 depth + 当前 target ROI + 该帧 pose**，不掺历史积累。
 *
 * 意义在于把「AR 坐标链」和「点云融合质量」这两件容易互相甩锅的事拆开：
 * 如果这层绿色点能贴住现实物体、手机平移有正确视差、转动后仍停在原处，
 * 就证明 VINS pose / 相机光轴 / fx fy cx cy / 竖屏旋转 / TextureView 裁剪 /
 * 时间戳同步 整条链是对的；此后累计点云若仍然散，问题就只剩
 * depth scale / depth 精度 / fusion，而不必再回头怀疑 Renderer。
 */
static constexpr int kTargetDebugMaxPoints = 2000;
static bool targetDebugEnabled = false;
static std::vector<float> targetDebugPoints(kTargetDebugMaxPoints * 6, 0.f);
static int targetDebugPointCount = 0;
static int targetDebugRoiPixels = 0;
static int targetDebugValidPixels = 0;
static int targetDebugRoiX0 = 0;
static int targetDebugRoiY0 = 0;
static int targetDebugRoiX1 = 0;
static int targetDebugRoiY1 = 0;
static uint64_t targetDebugTs = 0;
static bool targetDebugPoseOk = false;
static uint64_t targetDebugBuilds = 0;
static uint64_t targetDebugQueries = 0;

// AR 渲染用的过滤门限与计数器（由 Java 侧下发，报告里回显）
static int gArMinHits = 2;
static size_t gArDrawnPoints = 0;

static void quatToR(float qx, float qy, float qz, float qw, float R[9]) {
    R[0] = 1 - 2 * (qy * qy + qz * qz);
    R[1] = 2 * (qx * qy - qz * qw);
    R[2] = 2 * (qx * qz + qy * qw);
    R[3] = 2 * (qx * qy + qz * qw);
    R[4] = 1 - 2 * (qx * qx + qz * qz);
    R[5] = 2 * (qy * qz - qx * qw);
    R[6] = 2 * (qx * qz - qy * qw);
    R[7] = 2 * (qy * qz + qx * qw);
    R[8] = 1 - 2 * (qx * qx + qy * qy);
}

static void eulerToR(float yaw, float pitch, float roll, float R[9]) {
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll), sr = sinf(roll);
    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;
}

static void makePose(float R[9], float t[3]) {
    if (vinsPoseOk) {
        quatToR(vinsQ[0], vinsQ[1], vinsQ[2], vinsQ[3], R);
        t[0] = vinsT[0];
        t[1] = vinsT[1];
        t[2] = vinsT[2];
    } else {
        eulerToR(vio.yaw(), vio.pitch(), vio.roll(), R);
        t[0] = vio.tx();
        t[1] = vio.ty();
        t[2] = vio.tz();
    }
}

static void rotatePoint(const float R[9], const float t[3],
                        float Xc, float Yc, float Zc,
                        float& Xw, float& Yw, float& Zw) {
    Xw = R[0] * Xc + R[1] * Yc + R[2] * Zc + t[0];
    Yw = R[3] * Xc + R[4] * Yc + R[5] * Zc + t[1];
    Zw = R[6] * Xc + R[7] * Yc + R[8] * Zc + t[2];
}

static void storeSnap(uint64_t ts, int w, int h, int rs, int urs, int ups,
                      const uint8_t* y, const uint8_t* u, const uint8_t* v,
                      const float R[9], const float t[3]) {
    FrameSnap s;
    s.ts = ts;
    for (int i = 0; i < 3; i++) {
        s.t[i] = t[i];
    }
    for (int i = 0; i < 9; i++) {
        s.R[i] = R[i];
    }
    s.w = SNAP_W;
    s.h = SNAP_H;
    s.rgb.resize((size_t)SNAP_W * SNAP_H * 3);
    for (int yy = 0; yy < SNAP_H; yy++) {
        int sy = std::min(h - 1, (yy * h) / SNAP_H);
        for (int xx = 0; xx < SNAP_W; xx++) {
            int sx = std::min(w - 1, (xx * w) / SNAP_W);
            int yi = sy * rs + sx;
            int ui = (sy / 2) * urs + (sx / 2) * ups;
            int Y = y[yi];
            int U = u[ui] - 128;
            int V = v[ui] - 128;
            int C = Y - 16;
            int Rr = std::clamp((298 * C + 409 * V + 128) >> 8, 0, 255);
            int Gg = std::clamp((298 * C - 100 * U - 208 * V + 128) >> 8, 0, 255);
            int Bb = std::clamp((298 * C + 516 * U + 128) >> 8, 0, 255);
            size_t o = ((size_t)yy * SNAP_W + xx) * 3;
            s.rgb[o] = (uint8_t)Rr;
            s.rgb[o + 1] = (uint8_t)Gg;
            s.rgb[o + 2] = (uint8_t)Bb;
        }
    }
    snaps.push_back(std::move(s));
    while (snaps.size() > 10) {
        snaps.pop_front();
    }
}

static void downscaleGray(const uint8_t* src, int w, int h, int stride,
                          uint8_t* dst, int dw, int dh) {
    if (!src || !dst || w <= 0 || h <= 0 || dw <= 0 || dh <= 0) {
        return;
    }
    for (int y = 0; y < dh; y++) {
        int sy = std::min(h - 1, y * h / dh);
        const uint8_t* row = src + (size_t)sy * stride;
        uint8_t* out = dst + (size_t)y * dw;
        for (int x = 0; x < dw; x++) {
            out[x] = row[std::min(w - 1, x * w / dw)];
        }
    }
}

// YUV_420_888 -> 缩小后的 BGR（供 NanoTrack 做真实彩色外观输入）。
// y/u/v 三个平面可能带行填充，uv 的 pixelStride 在 semi-planar 设备上为 2。
static void downscaleYuvToBgr(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                              int w, int h, int rs, int urs, int ups,
                              size_t uSize, size_t vSize,
                              uint8_t* dst, int dw, int dh) {
    if (!y || !u || !v || !dst || w <= 0 || h <= 0 || dw <= 0 || dh <= 0) {
        return;
    }
    if (ups <= 0) {
        ups = 1;
    }
    if (urs <= 0) {
        urs = 1;
    }
    const int cw = std::max(1, w / 2);

    for (int dy = 0; dy < dh; dy++) {
        const int sy = std::min(h - 1, dy * h / dh);
        const uint8_t* yRow = y + (size_t)sy * rs;
        const size_t cRowOff = std::min((size_t)(sy / 2) * urs,
                                        uSize > 0 ? uSize - 1 : (size_t)0);
        uint8_t* out = dst + (size_t)dy * dw * 3;
        for (int dx = 0; dx < dw; dx++) {
            const int sx = std::min(w - 1, dx * w / dw);
            const int cx = std::min(cw - 1, sx / 2);
            const size_t cIdx = cRowOff + (size_t)cx * ups;

            const uint8_t uu = (uSize > 0 && cIdx < uSize) ? u[cIdx] : 128;
            const uint8_t vv = (vSize > 0 && cIdx < vSize) ? v[cIdx] : 128;

            const int Y = std::max(0, (int)yRow[sx] - 16);
            const int U = (int)uu - 128;
            const int V = (int)vv - 128;

            const int r = (298 * Y + 409 * V + 128) >> 8;
            const int g = (298 * Y - 100 * U - 208 * V + 128) >> 8;
            const int b = (298 * Y + 516 * U + 128) >> 8;

            out[dx * 3 + 0] = (uint8_t)std::clamp(b, 0, 255);
            out[dx * 3 + 1] = (uint8_t)std::clamp(g, 0, 255);
            out[dx * 3 + 2] = (uint8_t)std::clamp(r, 0, 255);
        }
    }
}

static void fuseDepth(const float* depth, int w, int h, const FrameSnap& s, float confidence) {    if (!depth || w < 4 || h < 4 || s.rgb.empty()) {
        return;
    }
    tsdf.integrateDepth(depth, w, h, s.rgb.data(), s.w, s.h,
                        gFx, gFy, gCx, gCy, s.R, s.t, confidence);

    for (int yy = 0; yy < s.h; yy++) {
        int y = std::min(h - 1, yy * h / s.h);
        for (int xx = 0; xx < s.w; xx++) {
            int x = std::min(w - 1, xx * w / s.w);
            float z = depth[(size_t)y * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            float Xc = (x - gCx) * z / gFx;
            float Yc = (y - gCy) * z / gFy;
            float Xw, Yw, Zw;
            rotatePoint(s.R, s.t, Xc, Yc, z, Xw, Yw, Zw);
            size_t o = ((size_t)yy * s.w + xx) * 3;
            g.ingestPoint(Xw, Yw, Zw, s.rgb[o], s.rgb[o + 1], s.rgb[o + 2], confidence);
        }
    }
}

// ------------------------------------------------------------- 目标专用融合
/**
 * 清空目标专用模型与它的全部诊断状态。
 * 调用点：新建会话 / 销毁 / 选中新目标 / 清除目标。
 * 必须在 gStateMutex 下调用。
 */
/**
 * 重置 mesh 管线与深度标定。调用点：新建会话 / 销毁 / 换目标。
 * 必须在 gStateMutex 下调用。
 */
static void resetMeshPipeline() {
    meshEngine = MeshEngine();
    meshStats = MeshBuildStats{};
    meshCleanupStats = MeshPostProcessStats{};
    gTextureKeyframes.clear();
    textureBakeStats = TextureBakeStats{};
    uvUnwrapStats = UvUnwrapStats{};
    meshDirty = true;
    depthCalibrator.reset();
    lastCalibScale = 1.f;
    lastCalibShift = 0.f;
    lastCalibConfidence = 0.f;
    lastCalibSamples = 0;
    lastCalibValid = false;
    lastCalibInverse = false;
    calibFrames = 0;
    calibRejectFrames = 0;
    lastFusedDepth.clear();
    haveLastFusePose = false;
    lastFuseCalibrated = false;
    lastTemporalRatio = 1.f;
    temporalChecks = 0;
    temporalRejects = 0;
}

static void resetTargetModel() {
    targetG.reset();
    targetTsdf.reset();
    targetMaskMat.release();
    targetMaskStats = TargetMaskStats{};
    presenceFailStreak = 0;
    targetDepthScaleEma = 0.f;
    targetDepthScaleSamples = 0;
    lastTargetVinsMedian = 0.f;
    lastTargetRawMedian = 0.f;
    lastTargetVinsFeatureCount = 0;
    // 换了目标（或新会话）之后，旧网格是对旧目标/旧体素场提的，必须作废。
    resetMeshPipeline();
}

/** 目标专属深度尺度的最小样本数：样本太少就先用原始深度，不猜。 */
static constexpr int kTargetScaleMinSamples = 10;
/** 目标专属尺度的合理区间，挡住 vins/raw 同时偏小时产生的野值。 */
static constexpr float kTargetScaleMin = 0.01f;
static constexpr float kTargetScaleMax = 20.f;

/** 当前生效的目标深度尺度（vins/raw）。样本不足时返回 1（即不修正）。 */
static float currentTargetDepthScale() {
    // 全局深度标定已经把深度对齐到 VINS 尺度时，不再叠加「目标专属尺度」——
    // 两者目标完全相同，叠加就变成双重缩放。
    if (depthCalibrationEnabled && depthCalibrator.usable()) {
        return 1.f;
    }
    if (targetDepthScaleSamples < kTargetScaleMinSamples) {
        return 1.f;
    }
    if (!std::isfinite(targetDepthScaleEma) || !(targetDepthScaleEma > 0.f)) {
        return 1.f;
    }
    return std::clamp(targetDepthScaleEma, kTargetScaleMin, kTargetScaleMax);
}

/**
 * 只把 **mask 内** 的深度融进目标专用模型。
 *
 * 与 fuseDepth() 的两点区别：
 *   1. mask 外的 depth 一律置 0 —— TSDF 与 Gaussian 都看不到背景，
 *      这是「墙/天花板/桌子物理上不可能进入目标点云」的落地点；
 *   2. mask 内深度先乘**目标专属尺度**（vins/raw）再转 3D。累计模型是米制的，
 *      必须用目标自己的尺度；当前帧 live 层不需要（它直接用相机坐标投影，
 *      完全不依赖 depth 的绝对准确性）。
 */
static void fuseTargetDepth(const float* depth, int w, int h, const FrameSnap& s,
                            const cv::Mat& maskIn, float confidence) {
    if (!depth || w < 4 || h < 4 || s.rgb.empty() || maskIn.empty()) {
        return;
    }
    if (maskIn.cols != w || maskIn.rows != h || maskIn.type() != CV_8U) {
        return;
    }
    cv::Mat mask = maskIn; // 浅拷贝，只为拿到非 const 的 ptr()

    const float scale = currentTargetDepthScale();

    std::vector<float> masked(static_cast<size_t>(w) * h, 0.f);
    for (int y = 0; y < h; y++) {
        const float* srcRow = depth + static_cast<size_t>(y) * w;
        const uint8_t* mRow = mask.ptr<uint8_t>(y);
        float* dstRow = masked.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x++) {
            if (!mRow[x]) {
                continue;
            }
            const float z = srcRow[x];
            if (!std::isfinite(z) || z <= 0.f) {
                continue;
            }
            dstRow[x] = z * scale;
        }
    }

    targetTsdf.integrateDepth(masked.data(), w, h, s.rgb.data(), s.w, s.h,
                              gFx, gFy, gCx, gCy, s.R, s.t, confidence);

    for (int yy = 0; yy < s.h; yy++) {
        const int y = std::min(h - 1, yy * h / s.h);
        const uint8_t* mRow = mask.ptr<uint8_t>(y);
        for (int xx = 0; xx < s.w; xx++) {
            const int x = std::min(w - 1, xx * w / s.w);
            if (!mRow[x]) {
                continue;
            }
            const float z = masked[static_cast<size_t>(y) * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            const float Xc = (x - gCx) * z / gFx;
            const float Yc = (y - gCy) * z / gFy;
            float Xw, Yw, Zw;
            rotatePoint(s.R, s.t, Xc, Yc, z, Xw, Yw, Zw);
            const size_t o = (static_cast<size_t>(yy) * s.w + xx) * 3;
            targetG.ingestPoint(Xw, Yw, Zw, s.rgb[o], s.rgb[o + 1], s.rgb[o + 2], confidence);
        }
    }
    targetFuseFrames++;
}

// ---------------------------------------------------------------- PresenceGate
/** 外观分数下限（NanoTrack score）。 */
static constexpr float kPresenceAppearanceScore = 0.55f;
/** 没有外观后端时的替代下限：用 KLT 侧的 confidence，而不是直接判 false。 */
static constexpr float kPresenceConfidenceFallback = 0.35f;
/** mask 面积下限，与 TargetMaskEngine::kMinMaskArea 一致。 */
static constexpr int kPresenceMinMaskArea = 100;
/** KLT 运动判据：内点率或跟踪点数任一达标即可。 */
static constexpr float kPresenceMinInlierRatio = 0.45f;
static constexpr int kPresenceMinPoints = 25;
/** tracker 中心与 mask 质心的不一致上限（按 bbox 对角线归一化）。 */
static constexpr float kPresenceMaxCenterOffset = 0.20f;
/** 连续多少帧判定失败才认为「目标不在」。1 帧太敏感、3 帧太慢 —— 取 2。 */
static constexpr int kPresenceFailFrames = 2;

struct PresenceDecision {
    bool valid = false;
    bool appearanceOk = false;
    bool motionOk = false;
    bool maskOk = false;
    bool centerOk = false;
    float centerOffsetRatio = 0.f;
};

/**
 * 「目标真的还在吗？」
 *
 * 核心认知转变：**box 还在画面里 != 物体还在**。
 * 真实目标离开后，tracker 往往会在背景纹理上找到一个「看起来正常」的框并
 * 继续输出（OpenCV tracking #619 / NanoTrack issue 都有记录）。所以判定依据
 * 必须是外观 + Mask + 运动 + 中心一致四项合成，而不是 bbox 的几何位置。
 *
 * presenceOk = appearanceOk && maskOk && (motionOk || centerOk)
 */
static PresenceDecision evaluatePresence(const TargetTrackInfo& ti,
                                        const TargetMaskStats& ms,
                                        int depthW, int depthH) {
    PresenceDecision d;
    // 外观：有 Nano 就用它的分数；模型没加载时退回 KLT 置信度，
    // 否则没有 NanoTrack 的设备会永远被判「目标不在」。
    d.appearanceOk = ti.nanoLoaded
        ? (ti.nanoScore >= kPresenceAppearanceScore)
        : (ti.confidence >= kPresenceConfidenceFallback);
    d.motionOk = (ti.inlierRatio >= kPresenceMinInlierRatio) ||
                 (ti.trackedPoints >= kPresenceMinPoints);
    d.maskOk = ms.valid && ms.area >= kPresenceMinMaskArea;
    if (d.maskOk) {
        const float bw = (ti.x1 - ti.x0) * static_cast<float>(depthW);
        const float bh = (ti.y1 - ti.y0) * static_cast<float>(depthH);
        const float diag = std::sqrt(bw * bw + bh * bh);
        const float tcx = 0.5f * (ti.x0 + ti.x1) * static_cast<float>(depthW);
        const float tcy = 0.5f * (ti.y0 + ti.y1) * static_cast<float>(depthH);
        const float dx = ms.centerX - tcx;
        const float dy = ms.centerY - tcy;
        const float dist = std::sqrt(dx * dx + dy * dy);
        d.centerOffsetRatio = diag > 1.f ? dist / diag : 0.f;
        d.centerOk = d.centerOffsetRatio < kPresenceMaxCenterOffset;
    }
    d.valid = d.appearanceOk && d.maskOk && (d.motionOk || d.centerOk);
    return d;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeCreate(JNIEnv*, jobject, jint w, jint h,
                                                jfloat fx, jfloat fy, jfloat cx, jfloat cy) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    gFx = fx;
    gFy = fy;
    gCx = cx;
    gCy = cy;
    g.reset();
    vio.reset();
    df.reset();
    ai.reset();
    tsdf.reset();
    // 体素分辨率：全场景沿用 20mm（与历史行为一致，避免内存暴涨）；
    // 目标专属模型用 8mm —— 它只为单个物体建网格，体素数可控，
    // 而 mesh 的质量直接取决于体素分辨率。
    tsdf.setVoxelSize(0.020f);
    targetTsdf.setVoxelSize(0.008f);
    resetMeshPipeline();
    // 目标专用模型与诊断：新会话必须从头开始，否则会残留上一轮的目标点云
    targetG.reset();
    targetTsdf.reset();
    targetMaskMat.release();
    targetMaskStats = TargetMaskStats{};
    targetMaskEngine = TargetMaskEngine();
    presenceFailStreak = 0;
    targetDepthScaleEma = 0.f;
    targetDepthScaleSamples = 0;
    lastTargetVinsMedian = 0.f;
    lastTargetRawMedian = 0.f;
    lastTargetVinsFeatureCount = 0;
    targetMaskFrames = 0;
    targetFuseFrames = 0;
    presenceLostFrames = 0;
    kf.reset();
    objectTracker = std::make_shared<ObjectTracker>();
    aiBackend.initialize(w, h);
    frames = 0;
    lastKF = 0;
    haveExternalDepth = false;
    vinsPoseOk = false;
    haveLastGoodVinsPose = false;
    vinsLostAfterInit = false;
    lastGoodVinsT[0] = 0.f;
    lastGoodVinsT[1] = 0.f;
    lastGoodVinsT[2] = 0.f;
    rawVinsT[0] = rawVinsT[1] = rawVinsT[2] = 0.f;
    rawVinsQ[0] = rawVinsQ[1] = rawVinsQ[2] = 0.f;
    rawVinsQ[3] = 1.f;
    acceptedVinsT[0] = acceptedVinsT[1] = acceptedVinsT[2] = 0.f;
    acceptedVinsQ[0] = acceptedVinsQ[1] = acceptedVinsQ[2] = 0.f;
    acceptedVinsQ[3] = 1.f;
    lastVinsStep = 0.f;
    vinsRejectCount = 0;
    vinsRejectReason = "";
    haveFirstReject = false;
    firstRejectStep = 0.f;
    firstRejectT[0] = firstRejectT[1] = firstRejectT[2] = 0.f;
    firstRejectVelocity = 0.f;
    firstRejectAccBias = 0.f;
    firstRejectGyroBias = 0.f;
    firstRejectTd = 0.0;
    vinsT[0] = vinsT[1] = vinsT[2] = 0;
    vinsQ[0] = vinsQ[1] = vinsQ[2] = 0;
    vinsQ[3] = 1;
    snaps.clear();
    vinsFrames = depthFrames = 0;
    lastVinsMs = 0.0;

    uint32_t v = 0;
    auto pfn = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(
        VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    VkResult r = pfn ? pfn(&v) : VK_ERROR_INITIALIZATION_FAILED;
    vk = (r == VK_SUCCESS || r == VK_INCOMPLETE);
    LOGI("V0.4 native create Vulkan=%d", vk);

    float ric[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float tic[3] = {0, 0, 0};
    const float vScale = 640.f / (float)std::max(w, h);
    gVinsW = std::max(1, (int)(w * vScale));
    gVinsH = std::max(1, (int)(h * vScale));
    const float vFx = fx * vScale;
    const float vFy = fy * vScale;
    const float vCx = cx * vScale;
    const float vCy = cy * vScale;
    vinsInit(vFx, vFy, vCx, vCy, gVinsW, gVinsH, ric, tic, 0.1f, 0.001f, 0.001f, 0.0001f);
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeDestroy(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    g.reset();
    vio.reset();
    df.reset();
    ai.reset();
    tsdf.reset();
    targetG.reset();
    targetTsdf.reset();
    targetMaskMat.release();
    targetMaskStats = TargetMaskStats{};
    presenceFailStreak = 0;
    targetDepthScaleSamples = 0;
    targetDepthScaleEma = 0.f;
    resetMeshPipeline();
    kf.reset();
    objectTracker.reset();
    snaps.clear();
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnImu(JNIEnv*, jobject, jlong t,
                                               jfloat ax, jfloat ay, jfloat az,
                                               jfloat gx, jfloat gy, jfloat gz) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    vio.imu((uint64_t)t, ax, ay, az, gx, gy, gz);
    vinsInputImu((double)t * 1e-9, ax, ay, az, gx, gy, gz);
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnCameraFrame(
        JNIEnv* e, jobject,
        jbyteArray y, jbyteArray u, jbyteArray v,
        jint w, jint h, jint rs, jint urs, jint ups, jlong frameTs, jlong vinsTs) {
    jsize un = e->GetArrayLength(u);
    std::vector<uint8_t> ubuf((size_t)un);
    e->GetByteArrayRegion(u, 0, un, reinterpret_cast<jbyte*>(ubuf.data()));

    jsize vn = e->GetArrayLength(v);
    std::vector<uint8_t> vbuf((size_t)vn);
    e->GetByteArrayRegion(v, 0, vn, reinterpret_cast<jbyte*>(vbuf.data()));

    jsize n = e->GetArrayLength(y);
    std::vector<uint8_t> ybuf((size_t)n);
    e->GetByteArrayRegion(y, 0, n, reinterpret_cast<jbyte*>(ybuf.data()));

    const uint8_t* yy = ybuf.data();
    const uint8_t* uu = ubuf.data();
    const uint8_t* vv = vbuf.data();

    {
        // 轻量的统计/质量分析（df/ai/vio 为共享状态，需要加锁）
        std::lock_guard<std::mutex> lk(gStateMutex);
        df.ingestLuma(yy, w, h, rs);
        ai.analyze(yy, w, h, rs);
        vio.frame((uint64_t)frameTs, yy, w, h, rs);
    }

    // VINS 可能耗时数十至上百毫秒，持有 gStateMutex 会拖死渲染/UI 线程，
    // 因此放在锁外执行（vins_bridge 内部有自己的互斥锁）。
    std::vector<uint8_t> vinsGray((size_t)gVinsW * gVinsH);
    downscaleGray(yy, w, h, rs, vinsGray.data(), gVinsW, gVinsH);
    const auto vinsStart = std::chrono::steady_clock::now();
    vinsInputImage((double)vinsTs * 1e-9, vinsGray.data(), gVinsW, gVinsH, gVinsW);
    const auto vinsEnd = std::chrono::steady_clock::now();
    const double vinsMs = std::chrono::duration<double, std::milli>(vinsEnd - vinsStart).count();

    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (tracker) {
        try {
            // ---- tracking 主分辨率降到 640 长边 ----
            // 原来 KLT 直接在 1280x960 的 Y 平面上跑（1,228,800 px）；降到 640 长边
            // 之后只剩约 307,200 px（约四分之一），LK 金字塔的搜索压力降一个量级。
            //
            // 注意：这里**不需要**把 tracker 输出的 bbox 再乘以缩放系数 ——
            // ObjectTracker 的 x0/y0/x1/y1 全部是按图像尺寸归一化后输出的，
            // 归一化坐标天然与分辨率无关（depth 侧也是用同一套归一化乘它的 w/h）。
            constexpr int kTrackMaxDim = 640;
            const double tScale = std::min(
                1.0, static_cast<double>(kTrackMaxDim) /
                         static_cast<double>(std::max(1, std::max(w, h))));
            const int tw = std::max(16, static_cast<int>(std::lround(w * tScale)));
            const int th = std::max(16, static_cast<int>(std::lround(h * tScale)));
            gTrackerGray.resize(static_cast<size_t>(tw) * static_cast<size_t>(th));
            downscaleGray(yy, w, h, rs, gTrackerGray.data(), tw, th);
            tracker->updateFrame(gTrackerGray.data(), tw, th, tw,
                                 static_cast<uint64_t>(frameTs));
            tracker->track(gTrackerGray.data(), tw, th, tw,
                           static_cast<uint64_t>(frameTs));

            // 每 5 帧喂一张由 YUV420 直接生成的真实彩色缩略图给 NanoTrack。
            // 原实现把灰度复制成三通道（COLOR_GRAY2BGR），NanoTrack 的外观
            // 判别能力实际上被浪费了；这里让它拿到真正的 RGB 信息。
            if (++gTrackerColorFrameCounter % 5ULL == 0ULL) {
                constexpr int kColorMaxDim = 640;
                const double cScale = std::min(
                    1.0, static_cast<double>(kColorMaxDim) /
                             static_cast<double>(std::max(1, std::max(w, h))));
                const int cw = std::max(16, static_cast<int>(std::lround(w * cScale)));
                const int ch = std::max(16, static_cast<int>(std::lround(h * cScale)));
                std::vector<uint8_t> color((size_t)cw * (size_t)ch * 3u);
                downscaleYuvToBgr(yy, uu, vv, w, h, rs, urs, ups,
                                  ubuf.size(), vbuf.size(),
                                  color.data(), cw, ch);
                tracker->setColorFrame(color.data(), cw, ch, cw * 3,
                                       static_cast<uint64_t>(frameTs));
            }
        } catch (const cv::Exception& ex) {
            __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Target", "OpenCV tracker exception: %s", ex.what());
        } catch (const std::exception& ex) {
            __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Target", "Tracker exception: %s", ex.what());
        } catch (...) {
            __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Target", "Unknown tracker exception");
        }
    }

    float vp[7];
    const bool poseOk = vinsGetPose(vp);
    if (poseOk) {
        rawVinsT[0] = vp[0];
        rawVinsT[1] = vp[1];
        rawVinsT[2] = vp[2];
        rawVinsQ[0] = vp[3];
        rawVinsQ[1] = vp[4];
        rawVinsQ[2] = vp[5];
        rawVinsQ[3] = vp[6];
    }

    float R[9], T[3];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        lastVinsMs = vinsMs;
        vinsFrames++;
        if (poseOk &&
            std::isfinite(vp[0]) &&
            std::isfinite(vp[1]) &&
            std::isfinite(vp[2]) &&
            std::isfinite(vp[3]) &&
            std::isfinite(vp[4]) &&
            std::isfinite(vp[5]) &&
            std::isfinite(vp[6])) {
            bool sane = true;
            if (haveLastGoodVinsPose) {
                const float dx = vp[0] - lastGoodVinsT[0];
                const float dy = vp[1] - lastGoodVinsT[1];
                const float dz = vp[2] - lastGoodVinsT[2];
                const float step = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (step > 0.5f) {
                    sane = false;
                }
            }

            if (sane && !vinsLostAfterInit) {
                vinsT[0] = vp[0];
                vinsT[1] = vp[1];
                vinsT[2] = vp[2];
                vinsQ[0] = vp[3];
                vinsQ[1] = vp[4];
                vinsQ[2] = vp[5];
                vinsQ[3] = vp[6];

                acceptedVinsT[0] = vp[0];
                acceptedVinsT[1] = vp[1];
                acceptedVinsT[2] = vp[2];
                acceptedVinsQ[0] = vp[3];
                acceptedVinsQ[1] = vp[4];
                acceptedVinsQ[2] = vp[5];
                acceptedVinsQ[3] = vp[6];

                if (haveLastGoodVinsPose) {
                    const float dx = vp[0] - lastGoodVinsT[0];
                    const float dy = vp[1] - lastGoodVinsT[1];
                    const float dz = vp[2] - lastGoodVinsT[2];
                    lastVinsStep = std::sqrt(dx * dx + dy * dy + dz * dz);
                }

                lastGoodVinsT[0] = vp[0];
                lastGoodVinsT[1] = vp[1];
                lastGoodVinsT[2] = vp[2];

                // 记录「带时间戳的位姿」：AR 要用 Preview 的 SENSOR_TIMESTAMP
                // 反查这一时刻的相机 pose，而不是拿此刻最新的 pose 硬套。
                {
                    RenderPoseSample rp;
                    rp.ts = static_cast<uint64_t>(frameTs);
                    quatToR(vp[3], vp[4], vp[5], vp[6], rp.R);
                    rp.t[0] = vp[0];
                    rp.t[1] = vp[1];
                    rp.t[2] = vp[2];
                    renderPoseHistory.push_back(rp);
                    while (renderPoseHistory.size() > kRenderPoseHistoryMax) {
                        renderPoseHistory.pop_front();
                    }
                }

                haveLastGoodVinsPose = true;
                vinsPoseOk = true;
            } else {
                vinsRejectCount++;
                vinsRejectReason = "step_or_lost";
                vinsPoseOk = false;
                if (haveLastGoodVinsPose) {
                    if (!haveFirstReject) {
                        const float dx = vp[0] - lastGoodVinsT[0];
                        const float dy = vp[1] - lastGoodVinsT[1];
                        const float dz = vp[2] - lastGoodVinsT[2];
                        firstRejectStep = std::sqrt(dx * dx + dy * dy + dz * dz);
                        firstRejectT[0] = vp[0];
                        firstRejectT[1] = vp[1];
                        firstRejectT[2] = vp[2];
                        VinsHealth health;
                        if (vinsGetHealth(&health)) {
                            firstRejectVelocity = health.velocity;
                            firstRejectAccBias = health.accBias;
                            firstRejectGyroBias = health.gyroBias;
                            firstRejectTd = health.timeOffset;
                        }
                        haveFirstReject = true;
                    }
                    vinsLostAfterInit = true;
                    snaps.clear();
                }
            }
        } else {
            if (poseOk) {
                vinsRejectCount++;
                vinsRejectReason = "non_finite";
            }
            if (haveLastGoodVinsPose) {
                if (!haveFirstReject) {
                    firstRejectStep = 0.f;
                    firstRejectT[0] = vp[0];
                    firstRejectT[1] = vp[1];
                    firstRejectT[2] = vp[2];
                    VinsHealth health;
                    if (vinsGetHealth(&health)) {
                        firstRejectVelocity = health.velocity;
                        firstRejectAccBias = health.accBias;
                        firstRejectGyroBias = health.gyroBias;
                        firstRejectTd = health.timeOffset;
                    }
                    haveFirstReject = true;
                }
                vinsLostAfterInit = true;
                snaps.clear();
            }
            vinsPoseOk = false;
        }

        if (vinsPoseOk && !vinsLostAfterInit) {
            makePose(R, T);

            bool accept = kf.consider((uint64_t)frameTs, ai.q().sharpness, ai.q().exposure,
                                      vio.visualNovelty(), vio.features(),
                                      T[0], T[1], T[2]);
            if (accept) {
                lastKF++;
            }

            storeSnap((uint64_t)frameTs, w, h, rs, urs, ups, yy, uu, vv, R, T);
        }
        frames++;
    }

    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        if (!haveExternalDepth && (frames % 2 == 0)) {
            int sx = std::max(1, w / 96);
            int sy = std::max(1, h / 72);
            float depthBase = 1.15f + 0.25f * std::min(1.f, vio.features() / 1200.f);
            for (int yy0 = sy; yy0 < h; yy0 += sy) {
                for (int x = sx; x < w; x += sx) {
                    float z = depthBase;
                    float Xc = (x - gCx) * z / gFx;
                    float Yc = (yy0 - gCy) * z / gFy;
                    float Xw, Yw, Zw;
                    rotatePoint(R, T, Xc, Yc, z, Xw, Yw, Zw);
                    int yi = yy0 * rs + x;
                    int ui = (yy0 / 2) * urs + (x / 2) * ups;
                    int Y = yy[yi];
                    int U = uu[ui] - 128;
                    int V = vv[ui] - 128;
                    int C = Y - 16;
                    int Rr = std::clamp((298 * C + 409 * V + 128) >> 8, 0, 255);
                    int Gg = std::clamp((298 * C - 100 * U - 208 * V + 128) >> 8, 0, 255);
                    int Bb = std::clamp((298 * C + 516 * U + 128) >> 8, 0, 255);
                    g.ingestPoint(Xw, Yw, Zw, (uint8_t)Rr, (uint8_t)Gg, (uint8_t)Bb,
                                  ai.q().sharpness * ai.q().exposure);
                }
            }
        }
    }
}

/**
 * 当前帧 live 目标点：**只取本次 depth + 本次目标 Mask**，直接存相机坐标。
 *
 * ## 为什么去掉了 world 变换
 *
 * 旧实现走的是
 * `Xc,Yc,Zc -> rotatePoint(R,T) -> Xw,Yw,Zw -> 再用另一个时刻的相机 pose 投回屏幕`。
 * 而实机上 `arPoseFromTimestamp=false`（按 Preview 时间戳压根查不到对应时刻的
 * pose），于是本该贴着当前目标的点，被另一个时刻的 pose 投到了屏幕外面。
 *
 * 而当前帧的 AR **根本没必要经过 world**：这一帧的 depth 就是这台相机在
 * 这一刻拍的，点和像素本来就在同一个相机坐标系里。直接存 (Xc, Yc, Z) 之后，
 * Renderer 用 `pc = aPosition` 就画完了，**完全不依赖 VINS 位姿与时间戳**。
 *
 * 这一步的诊断价值也最大：只要 Depth Mask 是对的，绿色点就一定落在绿色目标
 * 区域里；如果点到别处去了，那问题一定在 Mask 而不在 pose 链上。
 *
 * ROI 用 ObjectTracker 的归一化 bbox（和 depth 帧共用同一套归一化坐标），
 * 再叠加本次的 Mask 过滤：**Mask 外的像素一个都不取**。
 */
static void buildTargetDebugLayer(const float* d, int w, int h,
                                 const TargetTrackInfo& ti,
                                 const cv::Mat& mask) {
    targetDebugPointCount = 0;
    targetDebugRoiPixels = 0;
    targetDebugValidPixels = 0;
    // live 层不再依赖位姿，恒为「可用」；报告里的字段名同步改成 liveSpaceOk
    targetDebugPoseOk = true;
    targetDebugTs = 0;
    if (!targetDebugEnabled) {
        return;
    }
    if (d == nullptr || w < 4 || h < 4) {
        return;
    }
    if (ti.state != TargetState::TRACKING && ti.state != TargetState::ACQUIRING) {
        return;
    }

    const int x0 = std::max(0, static_cast<int>(ti.x0 * w));
    const int y0 = std::max(0, static_cast<int>(ti.y0 * h));
    const int x1 = std::min(w - 1, static_cast<int>(ti.x1 * w));
    const int y1 = std::min(h - 1, static_cast<int>(ti.y1 * h));
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    targetDebugRoiX0 = x0;
    targetDebugRoiY0 = y0;
    targetDebugRoiX1 = x1;
    targetDebugRoiY1 = y1;

    const int roiW = x1 - x0 + 1;
    const int roiH = y1 - y0 + 1;
    targetDebugRoiPixels = roiW * roiH;

    const bool useMask = !mask.empty() && mask.cols == w && mask.rows == h &&
                         mask.type() == CV_8U;

    int step = 1;
    while ((roiW / step) * (roiH / step) > kTargetDebugMaxPoints) {
        step++;
    }

    for (int y = y0; y <= y1; y += step) {
        for (int x = x0; x <= x1; x += step) {
            if (useMask && !mask.at<uint8_t>(y, x)) {
                continue;
            }
            const float z = d[static_cast<size_t>(y) * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            targetDebugValidPixels++;
            if (targetDebugPointCount >= kTargetDebugMaxPoints) {
                continue;
            }
            const float Xc = (x - gCx) * z / gFx;
            const float Yc = (y - gCy) * z / gFy;
            float* p = targetDebugPoints.data() +
                       static_cast<size_t>(targetDebugPointCount) * 6;
            // **相机坐标**：Renderer 侧用 uPointSpace=0 时 pc = aPosition。
            p[0] = Xc;
            p[1] = Yc;
            p[2] = z;
            // 固定绿色：这一层的用途是「和真实画面比对」，颜色必须一眼可辨，
            // 不能和累计点云的真实颜色混在一起。
            p[3] = 0.20f;
            p[4] = 1.00f;
            p[5] = 0.35f;
            targetDebugPointCount++;
        }
    }
    targetDebugBuilds++;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnDepthMap(
        JNIEnv* e, jobject, jfloatArray depth, jint w, jint h,
        jfloat confidence, jlong t) {
    if (!depth || w <= 0 || h <= 0) {
        return;
    }
    jsize n = e->GetArrayLength(depth);
    if (n < (jsize)(w * h)) {
        return;
    }
    std::vector<float> d((size_t)w * h);
    e->GetFloatArrayRegion(depth, 0, w * h, d.data());

    {
        std::shared_ptr<ObjectTracker> tracker;
        {
            std::lock_guard<std::mutex> lk(gStateMutex);
            tracker = objectTracker;
        }
        if (tracker && tracker->isTracking()) {
            tracker->updateFromDepth(d.data(), w, h, (uint64_t)t);
        }
    }

    // ---- 目标专属深度尺度：目标 ROI 内的 VINS 三角化特征中位数 ----
    //
    // 为什么放在**进入 gStateMutex 之前**：vinsFeatureDepthMedianInRoi() 内部持
    // g_vinsMutex。若把它嵌进本函数主段的临界区，就会形成
    // gStateMutex -> g_vinsMutex 的嵌套锁序，而 nativeOnCameraFrame 是
    // 「先拿 gStateMutex 做统计、VINS 计算放到锁外」——两者锁序不一致，
    // 迟早会互相等。这里刻意让它独立成段。
    {
        std::shared_ptr<ObjectTracker> probe;
        {
            std::lock_guard<std::mutex> lk(gStateMutex);
            probe = objectTracker;
        }
        if (probe && probe->isTracking()) {
            const TargetTrackInfo t0 = probe->info();
            if (t0.depthRoiArea >= kPresenceMinMaskArea) {
                int vinsSamples = 0;
                const float vinsRoiMedian =
                    vinsFeatureDepthMedianInRoi(t0.x0, t0.y0, t0.x1, t0.y1, &vinsSamples);
                const float rawMedian = t0.medianDepth;
                if (vinsSamples >= 8 && rawMedian > 0.05f && vinsRoiMedian > 0.05f) {
                    const float s = vinsRoiMedian / rawMedian;
                    if (std::isfinite(s) && s > kTargetScaleMin && s < kTargetScaleMax) {
                        std::lock_guard<std::mutex> lk(gStateMutex);
                        targetDepthScaleEma = (targetDepthScaleSamples == 0)
                            ? s
                            : (0.9f * targetDepthScaleEma + 0.1f * s);
                        targetDepthScaleSamples++;
                        lastTargetVinsMedian = vinsRoiMedian;
                        lastTargetRawMedian = rawMedian;
                        lastTargetVinsFeatureCount = vinsSamples;
                    }
                }
            }
        }
    }

    // ---- 深度标定：用 VINS 稀疏三角化深度做鲁棒回归 ----
    //
    // 必须放在**进入 gStateMutex 之前**：vinsFeatureSamples() 内部持 g_vinsMutex，
    // 嵌进主临界区就会形成 gStateMutex -> g_vinsMutex 的嵌套锁序，而
    // nativeOnCameraFrame 是「先拿 gStateMutex 做统计、VINS 计算放在锁外」，
    // 两者锁序不一致迟早互相等。
    if (depthCalibrationEnabled) {
        const int ns = vinsFeatureSamples(gCalibSampleBuf, kMaxCalibSamples);
        if (ns >= depthCalibrator.config().minSamples) {
            std::vector<float> dPairs;
            std::vector<float> zPairs;
            dPairs.reserve((size_t)ns);
            zPairs.reserve((size_t)ns);
            for (int i = 0; i < ns; ++i) {
                const float nu = gCalibSampleBuf[i * 3 + 0];
                const float nv = gCalibSampleBuf[i * 3 + 1];
                const float vz = gCalibSampleBuf[i * 3 + 2];
                if (!(nu >= 0.f && nu <= 1.f && nv >= 0.f && nv <= 1.f)) {
                    continue;
                }
                if (!std::isfinite(vz) || !(vz > 0.05f)) {
                    continue;
                }
                const int px = std::min(w - 1, std::max(0, (int)(nu * (float)w)));
                const int py = std::min(h - 1, std::max(0, (int)(nv * (float)h)));
                const float dv = d[(size_t)py * w + px];
                if (!std::isfinite(dv)) {
                    continue;
                }
                dPairs.push_back(dv);
                zPairs.push_back(vz);
            }
            if (!dPairs.empty()) {
                if (depthCalibrator.update(dPairs, zPairs)) {
                    calibFrames++;
                } else {
                    calibRejectFrames++;
                }
            }
        }
        lastCalibValid = depthCalibrator.usable();
        const DepthCalibration& c = depthCalibrator.calibration();
        lastCalibScale = c.scale;
        lastCalibShift = c.shift;
        lastCalibConfidence = c.confidence;
        lastCalibSamples = c.samples;
        lastCalibInverse = c.inverseDepthModel;
    }

    // 融合用的深度图：标定可用时逐像素映射到 VINS 尺度，否则原样透传
    // （tracker / mask 仍看未标定的 d，保持第八轮已验收的行为不变）。
    const bool calibratedNow = depthCalibrationEnabled && depthCalibrator.usable();
    std::vector<float> zCal;
    const float* depthForFusion = d.data();
    if (calibratedNow) {
        const DepthCalibration& c = depthCalibrator.calibration();
        zCal.assign((size_t)w * h, 0.f);
        for (size_t i = 0; i < zCal.size(); ++i) {
            const float dv = d[i];
            if (!std::isfinite(dv)) {
                continue;
            }
            const float zz = c.toMetric(dv, 0.f);
            if (zz > 0.f && std::isfinite(zz)) {
                zCal[i] = zz;
            }
        }
        depthForFusion = zCal.data();
    }

    std::lock_guard<std::mutex> lk(gStateMutex);
    df.ingestExternalDepth(d.data(), w, h, confidence, (uint64_t)t);
    haveExternalDepth = true;
    depthFrames++;

    // 先抓一份当前 target 状态：mask / presence / 调试层都要用。
    //
    // 注意：本函数在这一段之前**已经持有 gStateMutex**（见上面的 lock_guard lk），
    // 所以这里只能直接读全局 objectTracker，绝不能再 lock 一次 ——
    // std::mutex 不可重入，再锁一次就是自锁死。ObjectTracker 内部有自己的
    // mutex_，isTracking()/info() 都会自己去拿，不会和外层冲突。
    TargetTrackInfo ti;
    bool haveTi = false;
    if (objectTracker && objectTracker->isTracking()) {
        ti = objectTracker->info();
        haveTi = true;
    }

    // ---- 目标分割 Mask + PresenceGate ----
    bool presenceOk = false;
    bool maskOk = false;
    if (haveTi) {
        const int bx0 = std::max(0, static_cast<int>(ti.x0 * w));
        const int by0 = std::max(0, static_cast<int>(ti.y0 * h));
        const int bx1 = std::min(w - 1, static_cast<int>(ti.x1 * w));
        const int by1 = std::min(h - 1, static_cast<int>(ti.y1 * h));
        if (bx1 > bx0 && by1 > by0) {
            const cv::Rect searchBox(bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1);
            const cv::Point seed((bx0 + bx1) / 2, (by0 + by1) / 2);
            targetMaskEngine.build(d.data(), w, h, searchBox, seed,
                                   targetMaskMat, targetMaskStats);
            targetMaskFrames++;
            maskOk = targetMaskStats.valid &&
                     targetMaskStats.area >= kPresenceMinMaskArea;
        } else {
            // 目标整块出界：searchBox 退化成空矩形，无从建 mask
            targetMaskStats = TargetMaskStats{};
            targetMaskStats.rejectReason = "target bbox off screen";
            targetMaskMat.release();
        }

        const PresenceDecision pd = evaluatePresence(ti, targetMaskStats, w, h);
        presenceOk = pd.valid;
        if (presenceOk) {
            presenceFailStreak = 0;
        } else {
            presenceFailStreak++;
            if (presenceFailStreak >= kPresenceFailFrames) {
                // 连续两帧判「目标不在」：立刻退回 REACQUIRING 并清 confidence。
                // UI 同一帧就会把绿框收掉 —— 绝不能等 bbox 滑出画面才反应，
                // 因为「物体走了但 tracker 停在墙上」时 bbox 永远不会出界。
                objectTracker->markPresenceLost();
                presenceLostFrames++;
            }
        }
        // 只更新诊断位（不参与状态机）：连续 1 帧失败就已经不该再画绿框了。
        objectTracker->setPresenceValid(presenceFailStreak == 0);
    } else {
        targetMaskStats = TargetMaskStats{};
        targetMaskMat.release();
        presenceFailStreak = 0;
    }

    const FrameSnap* match = nullptr;
    for (const auto& s : snaps) {
        int64_t diff = (int64_t)s.ts - (int64_t)t;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff < 50000000LL) {
            match = &s;
            break;
        }
    }

    if (match != nullptr) {
        // ---- 时序一致性：上一帧深度按相对位姿重投影到当前帧 ----
        // 单目深度网络在细节上本来就会抖，动得快时更明显。这里把「同一表面在
        // 两帧里深度不一致」的像素识别出来，用一致性比例给融合降权。
        float conf = confidence;
        if (haveLastFusePose && lastFuseCalibrated == calibratedNow &&
            lastFusedDepth.size() == (size_t)w * h) {
            // Pc_cur = Rcur^T * Rprev * Pc_prev + Rcur^T * (tprev - tcur)
            float Rrel[9];
            float trel[3];
            const float dv[3] = {lastFuseT[0] - match->t[0],
                                 lastFuseT[1] - match->t[1],
                                 lastFuseT[2] - match->t[2]};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float acc = 0.f;
                    for (int k = 0; k < 3; ++k) {
                        acc += match->R[k * 3 + i] * lastFuseR[k * 3 + j];
                    }
                    Rrel[i * 3 + j] = acc;
                }
                float acc = 0.f;
                for (int k = 0; k < 3; ++k) {
                    acc += match->R[k * 3 + i] * dv[k];
                }
                trel[i] = acc;
            }
            lastTemporalRatio = temporalConsistencyRatio(
                lastFusedDepth.data(), depthForFusion, w, h,
                gFx, gFy, gCx, gCy, Rrel, trel);
            temporalChecks++;
            if (lastTemporalRatio < 0.35f) {
                temporalRejects++;
            }
            // 下限 0.25：一致性差就降权，但不至于完全不融合 ——
            // 否则一边走一边扫时根本重建不出来。
            conf = confidence * std::clamp(lastTemporalRatio, 0.25f, 1.f);
        }

        // 全场景地图
        fuseDepth(depthForFusion, w, h, *match, conf);

        // 目标专用模型：**只有 presenceOk 且 mask 有效才允许融合**。
        // 这是「墙/天花板/桌子进不了目标点云」的落地点。
        if (haveTi && presenceOk && maskOk) {
            fuseTargetDepth(depthForFusion, w, h, *match, targetMaskMat, conf);
        }

        // live 层用的是**这一帧**的 depth 与**这一帧**的 mask，
        // 所以放在同一个 snap 匹配分支里，而不是另找一次。
        // 只有 mask 有效才建：验收要求「绿色 live 点只存在于目标轮廓内」，
        // mask 建不出来时宁可一个点都不画，也不要把整个 bbox 里的点放出去。
        if (haveTi && maskOk) {
            buildTargetDebugLayer(depthForFusion, w, h, ti, targetMaskMat);
            targetDebugTs = static_cast<uint64_t>(t);
        }

        // 记录本帧，供下一帧做时序一致性检查
        lastFusedDepth.assign(depthForFusion, depthForFusion + (size_t)w * h);
        for (int i = 0; i < 9; ++i) {
            lastFuseR[i] = match->R[i];
        }
        for (int i = 0; i < 3; ++i) {
            lastFuseT[i] = match->t[i];
        }
        haveLastFusePose = true;
        lastFuseCalibrated = calibratedNow;

        // 体素场变了 -> 网格变脏
        meshDirty = true;
        meshDirtyMarks++;
    }

    if (!targetDebugEnabled) {
        // 关掉时把上一帧的残留清空，否则切换开关后屏幕上会留着旧点
        targetDebugPointCount = 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeExportPly(JNIEnv* e, jobject, jstring path) {
    const char* p = e->GetStringUTFChars(path, nullptr);
    std::lock_guard<std::mutex> lk(gStateMutex);
    // 有目标模型时**只导目标**（验收要求：PLY 只含目标，不再有墙、天花板、桌子）。
    // 还没锁定过目标时退回全场景 TSDF，否则导出会是一个空文件。
    bool ok = targetTsdf.voxels() > 0 ? targetTsdf.exportPly(p) : tsdf.exportPly(p);
    e->ReleaseStringUTFChars(path, p);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetMode(JNIEnv*, jobject, jint m) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    mode = m;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetStats(JNIEnv* e, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    float R[9], T[3];
    makePose(R, T);

    float pminX, pminY, pminZ, pmaxX, pmaxY, pmaxZ;
    float pcx, pcy, pcz;
    g.boundingBox(&pminX, &pminY, &pminZ, &pmaxX, &pmaxY, &pmaxZ);
    g.centroid(&pcx, &pcy, &pcz);

    const float offX = pcx - T[0];
    const float offY = pcy - T[1];
    const float offZ = pcz - T[2];
    const float offDist = std::sqrt(offX * offX + offY * offY + offZ * offZ);

    float dmin = 0.f, dmax = 0.f, dmean = 0.f;
    int dvalid = 0;
    if (!df.depth().empty()) {
        dmin = std::numeric_limits<float>::max();
        dmax = -std::numeric_limits<float>::max();
        double sum = 0.0;
        int cnt = 0;
        for (float z : df.depth()) {
            if (std::isfinite(z) && z > 0.f) {
                dmin = std::min(dmin, z);
                dmax = std::max(dmax, z);
                sum += z;
                cnt++;
            }
        }
        dvalid = cnt;
        dmean = cnt > 0 ? (float)(sum / cnt) : 0.f;
    }

    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(3);
    s << "V0.5.1 HQ Reconstruction\n"
      << "Tracking frames: " << vio.frames() << "  features: " << vio.features() << "\n"
      << "VINS: " << (vinsInitialized() ? "nonlinear initialized" : "initializing")
      << "  poseSource: " << (vinsPoseOk ? "vins" : "vio") << "\n"
      << "VINS raw q: (" << vinsQ[0] << ", " << vinsQ[1] << ", " << vinsQ[2] << ", " << vinsQ[3] << ")\n"
      << "VINS raw t: (" << vinsT[0] << ", " << vinsT[1] << ", " << vinsT[2] << ")\n"
      << "VINS estimator raw t: (" << rawVinsT[0] << ", " << rawVinsT[1] << ", " << rawVinsT[2] << ")\n"
      << "VINS accepted t: (" << acceptedVinsT[0] << ", " << acceptedVinsT[1] << ", " << acceptedVinsT[2] << ")\n"
      << "VINS last step: " << lastVinsStep << "\n"
      << "VINS reject count: " << vinsRejectCount << "\n"
      << "VINS reject reason: " << vinsRejectReason << "\n"
      << "VINS lost latch: " << (vinsLostAfterInit ? "true" : "false") << "\n"
      << "VINS invalid feature depth resets: " << featureManagerInvalidDepthResetCount() << "\n";

    VinsHealth health;
    if (vinsGetHealth(&health)) {
        s << "VINS health: velocity=" << health.velocity
          << " accBias=" << health.accBias
          << " gyroBias=" << health.gyroBias
          << " gravity=" << health.gravity
          << " trackedFeatures=" << health.trackedFeatures
          << " lastImuDt=" << health.lastImuDt
          << " estimatedTD=" << (health.timeOffset * 1000.0) << "ms\n";
        s << "VINS estimated RIC: ["
          << health.ric[0] << " " << health.ric[1] << " " << health.ric[2] << "; "
          << health.ric[3] << " " << health.ric[4] << " " << health.ric[5] << "; "
          << health.ric[6] << " " << health.ric[7] << " " << health.ric[8] << "]\n";
        s << "VINS estimated TIC: ("
          << health.tic[0] << ", " << health.tic[1] << ", " << health.tic[2] << ")\n";
    }

    if (haveFirstReject) {
        s << "First VINS reject: step=" << firstRejectStep
          << " t=(" << firstRejectT[0] << ", " << firstRejectT[1] << ", " << firstRejectT[2] << ")"
          << " velocity=" << firstRejectVelocity
          << " accBias=" << firstRejectAccBias
          << " gyroBias=" << firstRejectGyroBias
          << " td=" << (firstRejectTd * 1000.0) << "ms\n";
    }

    s
      << "Cam pose t: (" << T[0] << ", " << T[1] << ", " << T[2] << ")\n"
      << "Cam pose R: ["
      << R[0] << ", " << R[1] << ", " << R[2] << "; "
      << R[3] << ", " << R[4] << ", " << R[5] << "; "
      << R[6] << ", " << R[7] << ", " << R[8] << "]\n"
      << "VIO yaw/pitch/roll: " << vio.yaw() << " / " << vio.pitch() << " / " << vio.roll() << "\n"
      << "VIO t: (" << vio.tx() << ", " << vio.ty() << ", " << vio.tz() << ")"
      << "  novelty: " << vio.visualNovelty() << "\n"
      << "Depth: " << df.width() << "x" << df.height() << " ts=" << df.timestamp()
      << " samples=" << df.stats().samples << " valid=" << df.stats().valid
      << " refined=" << df.stats().refined << " conf=" << df.stats().confidence << "\n"
      << "Depth range: min=" << dmin << " max=" << dmax << " mean=" << dmean
      << " validPix=" << dvalid << "\n"
      << "AI quality: sharp=" << ai.q().sharpness << " exposure=" << ai.q().exposure
      << " motion=" << ai.q().motion << " geometry=" << ai.q().geometry
      << " completion=" << ai.q().completion << "\n"
      << "Keyframes: " << kf.size() << "  accepted: " << lastKF << "\n";
    if (kf.last()) {
        const Keyframe* k = kf.last();
        s << "  lastKF ts=" << k->ts << " score=" << k->score
          << " sharp=" << k->sharp << " exp=" << k->exposure
          << " novel=" << k->novelty << " feats=" << k->features
          << " t=(" << k->tx << ", " << k->ty << ", " << k->tz << ")\n";
    }
    s << "Gaussian: " << g.count() << " stable=" << g.stableCount()
      << " merged=" << g.mergedCount()
      << " confirmed2=" << g.confirmedCount(2)
      << " confirmed3=" << g.confirmedCount(3) << "\n"
      << "AR: minHits=" << gArMinHits << " drawn=" << gArDrawnPoints
      << " renderPoseHistory=" << renderPoseHistory.size() << "\n"
      << "TargetDepthDebug: enabled=" << (targetDebugEnabled ? 1 : 0)
      << " points=" << targetDebugPointCount
      << " roi=" << targetDebugRoiX0 << "," << targetDebugRoiY0 << "-"
      << targetDebugRoiX1 << "," << targetDebugRoiY1
      << " roiPixels=" << targetDebugRoiPixels
      << " validPixels=" << targetDebugValidPixels
      << " ts=" << targetDebugTs
      // live 层现在直接存相机坐标，不再经过 world pose，所以这里恒为 1；
      // 字段名同步改成 liveSpaceOk，免得以后被误读成「pose 有效」
      << " liveSpaceOk=" << (targetDebugPoseOk ? 1 : 0)
      << " space=camera"
      << " builds=" << targetDebugBuilds
      << " queries=" << targetDebugQueries << "\n"
      << "TargetMask: frames=" << targetMaskFrames
      << " valid=" << (targetMaskStats.valid ? 1 : 0)
      << " area=" << targetMaskStats.area
      << " candidatePixels=" << targetMaskStats.candidatePixels
      << " searchPixels=" << targetMaskStats.searchPixels
      << " seedDepth=" << targetMaskStats.seedDepth
      << " tolerance=" << targetMaskStats.tolerance
      << " center=(" << targetMaskStats.centerX << ", " << targetMaskStats.centerY << ")"
      << " reason=" << targetMaskStats.rejectReason << "\n"
      << "TargetModel: gaussians=" << targetG.count()
      << " confirmed2=" << targetG.confirmedCount(2)
      << " stable=" << targetG.confirmedCount(3)
      << " tsdfVoxels=" << targetTsdf.voxels()
      << " fuseFrames=" << targetFuseFrames << "\n"
      << "TargetPresence: failStreak=" << presenceFailStreak
      << " lostFrames=" << presenceLostFrames << "\n"
      << "TargetDepthScale: samples=" << targetDepthScaleSamples
      << " ema=" << targetDepthScaleEma
      << " applied=" << currentTargetDepthScale()
      << " vinsRoiMedian=" << lastTargetVinsMedian
      << " rawMedian=" << lastTargetRawMedian
      << " vinsFeatures=" << lastTargetVinsFeatureCount << "\n"
      << "PointCloud bbox min=(" << pminX << ", " << pminY << ", " << pminZ << ")"
      << " max=(" << pmaxX << ", " << pmaxY << ", " << pmaxZ << ")\n"
      << "PointCloud centroid=(" << pcx << ", " << pcy << ", " << pcz << ")\n"
      << "PointCloud vs Cam offset=(" << offX << ", " << offY << ", " << offZ
      << ")  dist=" << offDist << "\n"
      << "TSDF: voxels=" << tsdf.voxels()
      << " blocks=" << tsdf.blocks()
      << " voxelSize=" << tsdf.voxelSize()
      << " memMB=" << (tsdf.memoryBytes() / (1024 * 1024))
      << " targetVoxels=" << targetTsdf.voxels()
      << " targetVoxelSize=" << targetTsdf.voxelSize() << "\n"
      << "DepthCalib: enabled=" << (depthCalibrationEnabled ? 1 : 0)
      << " valid=" << (lastCalibValid ? 1 : 0)
      << " model=" << (lastCalibInverse ? "inverse" : "linear")
      << " scale=" << lastCalibScale
      << " shift=" << lastCalibShift
      << " confidence=" << lastCalibConfidence
      << " samples=" << lastCalibSamples
      << " accepted=" << calibFrames
      << " rejected=" << calibRejectFrames << "\n"
      << "Temporal: ratio=" << lastTemporalRatio
      << " checks=" << temporalChecks
      << " rejects=" << temporalRejects << "\n"
      << "Mesh: dirty=" << (meshDirty ? 1 : 0)
      << " builds=" << meshBuilds
      << " quality=" << meshQuality
      << " vertices=" << meshStats.outVertices
      << " triangles=" << meshStats.outTriangles
      << " rawTriangles=" << meshStats.rawTriangles
      << " componentsRemoved=" << meshStats.componentsRemoved
      << " decimated=" << (meshStats.decimated ? 1 : 0)
      << " buildMs=" << meshStats.totalMs
      << " note=" << meshStats.note << "\n"
      << "Vulkan: " << (vk ? "available" : "unavailable") << "\n"
      << "Next view: " << kf.guidance();
    return e->NewStringUTF(s.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetGuidance(JNIEnv* e, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return e->NewStringUTF(kf.guidance().c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetGaussians(JNIEnv* e, jobject,
                                                      jfloatArray out, jint maxPoints,
                                                      jint minHits) {
    jfloat* dst = e->GetFloatArrayElements(out, nullptr);
    if (!dst) {
        return 0;
    }
    // 长度契约：native 写 out + written*6，written <= maxPoints。
    // 调用方必须给够 maxPoints*6 个槽，这里再兜一次底，避免越界写 Java 数组。
    if (maxPoints <= 0 ||
        e->GetArrayLength(out) < static_cast<jsize>(maxPoints) * 6) {
        e->ReleaseFloatArrayElements(out, dst, 0);
        return 0;
    }
    size_t n;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const int need = minHits > 0 ? minHits : 1;
        gArMinHits = need;
        // Renderer 只读**目标模型**：背景点云根本没有进入 targetG 的代码路径。
        // 还没有目标（用户没锁定过）时退回全场景 g，避免一上来屏幕全空。
        SurfelEngine& src = (targetG.count() > 0) ? targetG : g;
        n = src.copyPoints(dst, (size_t)maxPoints, need);
        gArDrawnPoints = n;
    }
    e->ReleaseFloatArrayElements(out, dst, 0);
    return (jint)n;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsInitialized(JNIEnv*, jobject) {
    return vinsInitialized() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetHudMetrics(JNIEnv* e, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(1);
    s << "VINS " << vinsFrames << " 帧" << (vinsInitialized() ? "" : ", 初始化中")
      << " / " << lastVinsMs << " ms"
      << " · 特征 " << vio.features()
      << " · 深度 " << depthFrames << " 帧";
    // 只报一个「总量」没有意义：实机出现过 600000 总点里 stable 只有 28，
    // 那种情况下屏幕上几乎全是一次性噪声，而总数看起来却很壮观。
    if (targetG.count() > 0) {
        s << " · 目标 " << targetG.count()
          << " 确认 " << targetG.confirmedCount(2)
          << " 绘制 " << gArDrawnPoints;
    } else {
        s << " · 地图 " << g.count()
          << " 确认 " << g.confirmedCount(2)
          << " 稳定 " << g.confirmedCount(3)
          << " 绘制 " << gArDrawnPoints;
    }
    return e->NewStringUTF(s.str().c_str());
}

// nativeGetDepthDiagnostics 的槽数。
// **必须与 Kotlin 侧 NativeBridge.DEPTH_DIAGNOSTIC_SLOTS 保持一致。**
// 这里做了长度检查、长度不足就安全返回 0；但如果调用方按旧槽数建数组、
// 又按新槽数索引，越界会发生在 Kotlin 侧（「导出反馈报告」闪退那次就是这个形态）。
static constexpr int kDepthDiagSlots = 7;

// nativeGetTargetState 的槽数。
// **必须与 Kotlin 侧 NativeBridge.TARGET_STATE_SLOTS 保持一致。**
// 0..9 是原有字段，10..13 是本轮新增：
//   10 visibleFraction  目标可见比例（<0.12 连续 3 帧 -> REACQUIRING）
//   11 centerXNorm      目标中心 X（相机归一化，未裁剪，可越界）
//   12 centerYNorm      目标中心 Y
//   13 edgeLostFrames   连续「可见比例 < 12%」的帧数
//   14 presenceValid    PresenceGate 结论（0/1）：UI 据此决定绿框画不画
//   15 appearanceOk     外观后端是否可用（0/1，仅诊断）
// UI 靠 10..13 才能给出「目标接近边缘 / 已离开画面」的持续提示和方向箭头；
// 旧协议只给裁剪后的 bbox，目标完全出界时 bbox 退化成空矩形，UI 只能干等。
// 14/15 是这一轮新增：**box 还在画面里 != 物体还在**，绿框的可见性必须由
// presenceValid 决定，而不是 bbox 的几何位置。
static constexpr int kTargetStateSlots = 16;

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetPointCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    // 目标模型优先：UI 上的「点数」应该反映将要导出的东西
    return (jint)((targetG.count() > 0) ? targetG.count() : g.count());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSelectTarget(JNIEnv*, jobject, jfloat u, jfloat v) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (!objectTracker) return JNI_FALSE;
    try {
        const bool ok = objectTracker->requestTarget(u, v);
        if (ok) {
            // 新目标：目标专用模型必须清零重建，否则新旧目标会混在同一个点云里
            resetTargetModel();
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

// 用户手指拖出的矩形（相机归一化坐标，允许任意方向）。
// 走 gStateMutex 只是为了安全拿到 shared_ptr；真正的工作在 ObjectTracker
// 自己的 mutex_ 里完成，所以这里持锁时间很短，不会挡住相机/深度线程。
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSelectTargetRect(
        JNIEnv*, jobject, jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) return JNI_FALSE;
    try {
        const bool ok = tracker->requestTargetRect(x0, y0, x1, y1);
        if (ok) {
            // 新目标：目标专用模型清零重建
            std::lock_guard<std::mutex> lk(gStateMutex);
            resetTargetModel();
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeClearTarget(JNIEnv*, jobject) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (tracker) tracker->clearTarget();
    if (tracker) {
        // 清目标 = 清目标模型：否则下一次锁定会继承上一轮的点云
        std::lock_guard<std::mutex> lk(gStateMutex);
        resetTargetModel();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetObjectLockEnabled(JNIEnv*, jobject, jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (objectTracker) objectTracker->setEnabled(enabled);
}

// UI 侧的绿色目标框现在以 30Hz 拉取本接口（原来是 1Hz，肉眼看起来
// 「追踪很慢」其实就是刷新率问题，不是 tracker 算得慢）。
// 所以这里**不能**像以前那样整段持有 gStateMutex：
// gStateMutex 同时保护 VINS / Depth / Camera 主链，长期占用会直接拖慢它们。
// 正确做法是只在取 shared_ptr 的瞬间持锁，info() 靠 ObjectTracker 自己的 mutex_。
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetState(JNIEnv* env, jobject, jfloatArray out) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) {
        return 0;
    }
    if (out == nullptr || env->GetArrayLength(out) < kTargetStateSlots) {
        return static_cast<jint>(tracker->info().state);
    }

    const TargetTrackInfo info = tracker->info();
    jfloat* dst = env->GetFloatArrayElements(out, nullptr);
    if (dst != nullptr) {
        dst[0] = static_cast<float>(static_cast<int>(info.state));
        dst[1] = info.x0;
        dst[2] = info.y0;
        dst[3] = info.x1;
        dst[4] = info.y1;
        dst[5] = info.confidence;
        dst[6] = info.medianDepth;
        dst[7] = info.roiSharpness;
        dst[8] = static_cast<float>(info.trackedPoints);
        dst[9] = info.inlierRatio;
        dst[10] = info.visibleFraction;
        dst[11] = info.centerXNorm;
        dst[12] = info.centerYNorm;
        dst[13] = static_cast<float>(info.edgeLostFrames);
        dst[14] = info.presenceValid ? 1.f : 0.f;
        dst[15] = info.appearanceAvailable ? 1.f : 0.f;
        env->ReleaseFloatArrayElements(out, dst, 0);
    }
    return static_cast<jint>(info.state);
}

// 深度尺度诊断：只上报可观测量，不自动施加任何 scale 修正。
// out[0]=targetDepthP10 out[1]=targetDepthMedian out[2]=targetDepthP90
// out[3]=vinsTriangulatedDepthMedian
// out[4]=targetDepthValidPixels out[5]=targetDepthSampleCount out[6]=targetDepthRoiArea
//
// 后三个是「这个中位数到底可不可信」的前提：目标出界时 ROI 会被裁到只剩几行，
// 有效像素掉到个位数，P10/P50/P90 就会退化成同一个值（实机 P10=P50=P90=5.14128）。
// 上层据此拒绝这类样本，而不是把它算进 depth scale 的统计里。
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetDepthDiagnostics(JNIEnv* env, jobject, jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < kDepthDiagSlots) {
        return 0;
    }

    float vals[kDepthDiagSlots] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (tracker) {
        const TargetTrackInfo info = tracker->info();
        vals[0] = info.depthP10;
        vals[1] = info.medianDepth;
        vals[2] = info.depthP90;
        vals[4] = static_cast<float>(info.depthValidPixels);
        vals[5] = static_cast<float>(info.depthSampleCount);
        vals[6] = static_cast<float>(info.depthRoiArea);
    }

    // vinsFeatureDepthMedian 内部持有自己的 vins 锁，必须在 gStateMutex 之外调用
    vals[3] = vinsFeatureDepthMedian();

    env->SetFloatArrayRegion(out, 0, kDepthDiagSlots, vals);
    return 1;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetDiagnostics(JNIEnv* env, jobject) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) {
        return env->NewStringUTF("tracker=null");
    }

    const TargetTrackInfo i = tracker->info();
    std::ostringstream s;
    s << "haveCameraFrame=" << (i.haveCameraFrame ? "true" : "false") << "\n"
      << "frame=" << i.frameWidth << "x" << i.frameHeight << "\n"
      << "lastFrameTs=" << i.lastFrameTs << "\n"
      << "cameraUpdateCalls=" << i.cameraUpdateCalls << "\n"
      << "trackerUpdateCalls=" << i.trackerUpdateCalls << "\n"
      << "targetRequestCalls=" << i.targetRequestCalls << "\n"
      << "targetRequestAccepted=" << i.targetRequestAccepted << "\n"
      << "targetRequestRejected=" << i.targetRequestRejected << "\n"
      << "acquireCalls=" << i.acquireCalls << "\n"
      << "acquireSuccess=" << i.acquireSuccess << "\n"
      << "acquireFail=" << i.acquireFail << "\n"
      << "trackSuccess=" << i.trackSuccess << "\n"
      << "trackLost=" << i.trackLost << "\n"
      << "nanoLoaded=" << (i.nanoLoaded ? "true" : "false") << "\n"
      << "nanoInitCalls=" << i.nanoInitCalls << "\n"
      << "nanoUpdateCalls=" << i.nanoUpdateCalls << "\n"
      << "nanoFailures=" << i.nanoFailures << "\n"
      << "nanoRecoveries=" << i.nanoRecoveries << "\n"
      << "nanoScore=" << i.nanoScore << "\n"
      << "nanoLastMs=" << i.nanoLastMs << "\n"
      << "weakKltFrames=" << i.weakKltFrames << "\n"
      << "kltGoodPoints=" << i.kltGoodPoints << "\n"
      << "nanoForcedUpdates=" << i.nanoForcedUpdates << "\n"
      << "nanoWeakAttempts=" << i.nanoWeakAttempts << "\n"
      << "nanoWeakRecoveries=" << i.nanoWeakRecoveries << "\n"
      << "nanoUsedRealColor=" << (i.nanoUsedRealColor ? "true" : "false") << "\n"
      << "colorFrameValid=" << (i.colorFrameValid ? "true" : "false") << "\n"
      << "colorFrameSize=" << i.colorFrameWidth << "x" << i.colorFrameHeight << "\n"
      << "colorFrameCalls=" << i.colorFrameCalls << "\n"
      << "depthP10=" << i.depthP10 << "\n"
      << "depthMedian=" << i.medianDepth << "\n"
      << "depthP90=" << i.depthP90 << "\n"
      << "depthValidPixels=" << i.depthValidPixels << "\n"
      << "depthSampleCount=" << i.depthSampleCount << "\n"
      << "depthRoiArea=" << i.depthRoiArea << "\n"
      << "templateAllocated=" << (i.targetTemplateAllocated ? "true" : "false") << "\n"
      << "templateSize=" << i.templateWidth << "x" << i.templateHeight << "\n"
      << "prevGrayValid=" << (i.prevGrayValid ? "true" : "false") << "\n"
      << "prevGraySize=" << i.prevGrayWidth << "x" << i.prevGrayHeight << "\n"
      << "prevPointCount=" << i.prevPointCount << "\n"
      << "bboxWidthPx=" << i.bboxWidthPx << "\n"
      << "bboxHeightPx=" << i.bboxHeightPx << "\n"
      << "fullBBox=" << i.fullBBoxWidthPx << "x" << i.fullBBoxHeightPx << "\n"
      << "visibleBBox=" << i.visibleBBoxWidthPx << "x" << i.visibleBBoxHeightPx << "\n"
      << "visibleFraction=" << i.visibleFraction << "\n"
      << "lastAffineScale=" << i.lastAffineScale << "\n"
      << "affineScaleEMA=" << i.affineScaleEMA << "\n"
      << "reseedCount=" << i.reseedCount << "\n"
      << "lastEvent=" << i.lastEvent << "\n"
      << "depthFilterCalls=" << i.depthFilterCalls << "\n"
      << "depthFilterSkipped=" << i.depthFilterSkipped << "\n"
      << "presenceValid=" << (i.presenceValid ? "true" : "false") << "\n"
      << "presenceLostCount=" << i.presenceLostCount << "\n"
      << "appearanceAvailable=" << (i.appearanceAvailable ? "true" : "false") << "\n"
      << "appearanceBackend=" << i.appearanceBackend << "\n"
      << "lastError=" << i.lastError;
    return env->NewStringUTF(s.str().c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRenderPose(JNIEnv* env, jobject, jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 12) {
        return JNI_FALSE;
    }

    float pose[12];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        if (!vinsPoseOk || vinsLostAfterInit || !haveLastGoodVinsPose) {
            return JNI_FALSE;
        }
        quatToR(acceptedVinsQ[0], acceptedVinsQ[1], acceptedVinsQ[2], acceptedVinsQ[3], pose);
        pose[9] = acceptedVinsT[0];
        pose[10] = acceptedVinsT[1];
        pose[11] = acceptedVinsT[2];
    }

    env->SetFloatArrayRegion(out, 0, 12, pose);
    return JNI_TRUE;
}

// 当前 depth 帧 + 当前 target ROI 的调试层点数（AR 坐标链验证用）
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetDepthDebug(JNIEnv* e, jobject,
                                                             jfloatArray out,
                                                             jint maxPoints) {
    if (out == nullptr || maxPoints <= 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(gStateMutex);
    targetDebugQueries++;
    const int n = std::min(static_cast<int>(maxPoints), targetDebugPointCount);
    if (n <= 0) {
        return 0;
    }
    if (e->GetArrayLength(out) < static_cast<jsize>(n) * 6) {
        return 0;
    }
    e->SetFloatArrayRegion(out, 0, static_cast<jsize>(n) * 6,
                           targetDebugPoints.data());
    return (jint)n;
}

// 打开/关闭调试层的构建。关掉时不再做任何像素级的提取与转换。
extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetTargetDebugEnabled(JNIEnv*, jobject,
                                                               jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    targetDebugEnabled = (enabled == JNI_TRUE);
    if (!targetDebugEnabled) {
        targetDebugPointCount = 0;
    }
}

/**
 * 按 SENSOR_TIMESTAMP 查历史位姿（AR 正确跟随的前提）。
 *
 * 第一版只做最近邻：|sample.ts - ts| 最小且 < 80ms 就返回。
 * 以后若发现快速运动下仍有偏差，再升级成 translation 线性插值 +
 * 四元数 slerp —— 但现在先把「用对时刻」这件事做出来。
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRenderPoseAt(JNIEnv* env, jobject,
                                                         jlong timestampNs,
                                                         jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 12) {
        return JNI_FALSE;
    }
    float pose[12];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        if (renderPoseHistory.empty()) {
            return JNI_FALSE;
        }
        const int64_t want = static_cast<int64_t>(timestampNs);
        const RenderPoseSample* best = nullptr;
        int64_t bestDelta = 0;
        for (const RenderPoseSample& rp : renderPoseHistory) {
            int64_t d = static_cast<int64_t>(rp.ts) - want;
            if (d < 0) {
                d = -d;
            }
            if (best == nullptr || d < bestDelta) {
                best = &rp;
                bestDelta = d;
            }
        }
        // 找到的样本离目标时刻太远 = 这段历史里没有对应时刻的位姿。
        // 此时宁可返回 false 让上层回退，也不要拿错时刻的 pose 去画 AR。
        if (best == nullptr || bestDelta > kRenderPoseMaxAgeNs) {
            return JNI_FALSE;
        }
        for (int i = 0; i < 9; ++i) {
            pose[i] = best->R[i];
        }
        pose[9] = best->t[0];
        pose[10] = best->t[1];
        pose[11] = best->t[2];
    }
    env->SetFloatArrayRegion(out, 0, 12, pose);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeConfigureTrackerModels(JNIEnv* env, jobject, jstring backbone, jstring head) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) return JNI_FALSE;

    const char* b = env->GetStringUTFChars(backbone, nullptr);
    const char* h = env->GetStringUTFChars(head, nullptr);
    if (!b || !h) {
        if (b) env->ReleaseStringUTFChars(backbone, b);
        if (h) env->ReleaseStringUTFChars(head, h);
        return JNI_FALSE;
    }

    const bool ok = tracker->configureNano(b, h);
    // 外观后端接缝：工厂当前只会返回 nanotrack（LightTrack-ncnn 需要 ncnn 运行库，
    // 本工程没有链接），这里如实配置并把后端名写进诊断报告。
    tracker->configureAppearance(std::string(), std::string(), b, h);
    env->ReleaseStringUTFChars(backbone, b);
    env->ReleaseStringUTFChars(head, h);
    return ok ? JNI_TRUE : JNI_FALSE;
}


// ============================================================================
//  Mesh / GLB / 深度标定 JNI
// ============================================================================

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetVoxelSizes(JNIEnv*, jobject,
                                                       jfloat sceneMeters,
                                                       jfloat targetMeters) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (sceneMeters > 0.f) {
        tsdf.setVoxelSize(sceneMeters);
    }
    if (targetMeters > 0.f) {
        targetTsdf.setVoxelSize(targetMeters);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetVoxelBudget(JNIEnv*, jobject,
                                                        jint sceneBlocks,
                                                        jint targetBlocks) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (sceneBlocks > 0) {
        tsdf.setMaxBlocks((size_t)sceneBlocks);
    }
    if (targetBlocks > 0) {
        targetTsdf.setMaxBlocks((size_t)targetBlocks);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetDepthCalibrationEnabled(JNIEnv*, jobject,
                                                                   jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    depthCalibrationEnabled = (enabled == JNI_TRUE);
}

/**
 * 深度标定状态（12 槽）。
 *   0 scale   1 shift   2 confidence   3 samples       4 valid
 *   5 inverseModel  6 enabled  7 acceptedFrames  8 rejectedFrames
 *   9 temporalRatio 10 temporalRejects 11 usable
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetDepthCalibration(JNIEnv* e, jobject,
                                                             jfloatArray out) {
    if (out == nullptr) {
        return 0;
    }
    const jsize cap = e->GetArrayLength(out);
    if (cap < 12) {
        return 0;
    }
    jfloat v[12];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        v[0] = lastCalibScale;
        v[1] = lastCalibShift;
        v[2] = lastCalibConfidence;
        v[3] = (jfloat)lastCalibSamples;
        v[4] = lastCalibValid ? 1.f : 0.f;
        v[5] = lastCalibInverse ? 1.f : 0.f;
        v[6] = depthCalibrationEnabled ? 1.f : 0.f;
        v[7] = (jfloat)calibFrames;
        v[8] = (jfloat)calibRejectFrames;
        v[9] = lastTemporalRatio;
        v[10] = (jfloat)temporalRejects;
        v[11] = depthCalibrator.usable() ? 1.f : 0.f;
    }
    e->SetFloatArrayRegion(out, 0, 12, v);
    return 12;
}

/**
 * 从当前体素场构建三角网格（Marching Tetrahedra + 清理 + 简化）。
 *
 * 有目标模型时优先用 targetTsdf —— 与「有目标就只导目标」的 PLY / 点云口径一致。
 * 这是一次性操作，可能耗时几百毫秒，不要在帧回调里调。
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeBuildMesh(JNIEnv*, jobject, jint quality) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    meshQuality = std::clamp((int)quality, 0, 2);

    MeshOptions opt;
    opt.quality = meshQuality;
    const TsdfEngine& src = (targetTsdf.voxels() > 0) ? targetTsdf : tsdf;

    const bool ok = meshEngine.build(src, opt, meshStats);
    meshBuilds++;
    // 构建成功后体素场与网格一致；失败则保持脏，允许用户重试。
    meshDirty = !ok;

    LOGI("nativeBuildMesh quality=%d ok=%d src=%s verts=%zu tris=%zu rawTris=%zu "
         "compRemoved=%zu/%zu decimated=%d ms=%.1f note=%s",
         meshQuality, ok ? 1 : 0,
         (&src == &targetTsdf) ? "target" : "scene",
         meshStats.outVertices, meshStats.outTriangles, meshStats.rawTriangles,
         meshStats.componentsRemoved, meshStats.componentsBefore,
         meshStats.decimated ? 1 : 0, meshStats.totalMs, meshStats.note.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

/** mesh 统计（16 槽，全部为整型）。 */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshStats(JNIEnv* e, jobject) {
    jint v[16];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        v[0] = (jint)meshStats.outVertices;
        v[1] = (jint)meshStats.outTriangles;
        v[2] = (jint)meshStats.rawVertices;
        v[3] = (jint)meshStats.rawTriangles;
        v[4] = (jint)meshStats.componentsBefore;
        v[5] = (jint)meshStats.componentsRemoved;
        v[6] = (jint)meshStats.trianglesRemovedRaw;
        v[7] = (jint)meshStats.trianglesRemovedComp;
        v[8] = (jint)meshStats.blocksScanned;
        v[9] = meshStats.decimated ? 1 : 0;
        v[10] = meshStats.ok ? 1 : 0;
        v[11] = (jint)meshQuality;
        v[12] = (jint)meshStats.totalMs;
        v[13] = (jint)meshBuilds;
        v[14] = (jint)(meshStats.voxelSize * 1000000.f);  // 微米
        v[15] = meshStats.smoothIterations;
    }
    jintArray out = e->NewIntArray(16);
    if (out == nullptr) {
        return nullptr;
    }
    e->SetIntArrayRegion(out, 0, 16, v);
    return out;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshVertexCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return (jint)meshEngine.mesh().vertexCount();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshIndexCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return (jint)meshEngine.mesh().indices.size();
}

/** 顶点交错布局 x,y,z,nx,ny,nz,r,g,b（9 float/顶点）。返回写入的顶点数。 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshVertices(JNIEnv* e, jobject,
                                                         jfloatArray out,
                                                         jint maxVertices) {
    if (out == nullptr || maxVertices <= 0) {
        return 0;
    }
    std::vector<float> buf;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& m = meshEngine.mesh();
        const size_t nv = m.vertexCount();
        if (nv == 0) {
            return 0;
        }
        const jsize cap = e->GetArrayLength(out) / MESH_VERTEX_FLOATS;
        n = std::min((size_t)(maxVertices > 0 ? maxVertices : 0), (size_t)cap);
        if (n > nv) {
            n = nv;
        }
        buf.resize(n * MESH_VERTEX_FLOATS);
        const bool hasN = m.normals.size() >= nv * 3;
        const bool hasC = m.colors.size() >= nv * 3;
        for (size_t i = 0; i < n; ++i) {
            float* p = buf.data() + i * MESH_VERTEX_FLOATS;
            p[0] = m.positions[i * 3 + 0];
            p[1] = m.positions[i * 3 + 1];
            p[2] = m.positions[i * 3 + 2];
            if (hasN) {
                p[3] = m.normals[i * 3 + 0];
                p[4] = m.normals[i * 3 + 1];
                p[5] = m.normals[i * 3 + 2];
            } else {
                p[3] = 0.f;
                p[4] = 0.f;
                p[5] = 1.f;
            }
            if (hasC) {
                p[6] = m.colors[i * 3 + 0];
                p[7] = m.colors[i * 3 + 1];
                p[8] = m.colors[i * 3 + 2];
            } else {
                p[6] = p[7] = p[8] = 0.6f;
            }
        }
    }
    if (n == 0) {
        return 0;
    }
    e->SetFloatArrayRegion(out, 0, (jsize)(n * MESH_VERTEX_FLOATS), buf.data());
    return (jint)n;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshIndices(JNIEnv* e, jobject,
                                                        jintArray out,
                                                        jint maxIndices) {
    if (out == nullptr || maxIndices <= 0) {
        return 0;
    }
    std::vector<jint> buf;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& m = meshEngine.mesh();
        if (m.indices.empty()) {
            return 0;
        }
        const jsize cap = e->GetArrayLength(out);
        n = std::min((size_t)maxIndices, (size_t)cap);
        if (n > m.indices.size()) {
            n = m.indices.size();
        }
        buf.resize(n);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = (jint)m.indices[i];
        }
    }
    if (n == 0) {
        return 0;
    }
    e->SetIntArrayRegion(out, 0, (jsize)n, buf.data());
    return (jint)n;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeResetMesh(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    meshEngine = MeshEngine();
    meshStats = MeshBuildStats{};
    meshDirty = true;
}

/** 导出 glTF 2.0 二进制 GLB（vertex color）。返回是否成功。 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeExportGlb(JNIEnv* e, jobject, jstring path) {
    if (path == nullptr) {
        return JNI_FALSE;
    }
    const char* p = e->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return JNI_FALSE;
    }
    bool ok = false;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        GlbExportStats gs;
        ok = exportGlb(meshEngine.mesh(), std::string(p), "MobileScan3D_scan", &gs);
        LOGI("nativeExportGlb ok=%d verts=%zu tris=%zu json=%zu bin=%zu bytes=%zu color=%d note=%s",
             ok ? 1 : 0, gs.vertices, gs.triangles, gs.jsonBytes, gs.binBytes,
             gs.fileBytes, gs.hasVertexColor ? 1 : 0, gs.note.c_str());
    }
    e->ReleaseStringUTFChars(path, p);
    return ok ? JNI_TRUE : JNI_FALSE;
}

// ===========================================================================
//  V0.6：HQ 多视角纹理 -> 自包含 textured GLB
// ===========================================================================
//
// 数据流（只在「停扫导出」时跑一次，每帧路径零开销）：
//
//   meshEngine.mesh()（本工程 SoA 布局）
//     -> toAosMesh()                 边界转换（见 v06/aos_mesh.h 的说明）
//     -> MeshPostProcessor::run()    亚毫米 weld / 去漂浮分量 / ear-clipping 补小洞 / QEM
//     -> UvUnwrapper::unwrap()       xatlas（已 vendor）或 triangle-atlas 回退
//     -> TextureBaker::bake()        Z-buffer 可见性 + 视角评分 + 每 texel 前 3 视角混合
//     -> cv::imencode(".jpg")        atlas 压成 JPEG
//     -> TexturedGlbExporter::write()  内嵌 JPEG 的 glTF 2.0 二进制
//
// 任何一步失败都返回 JNI_FALSE，调用方（MainActivity）回退到 V0.5 的
// vertex-color GLB —— 几何已经算好，不因为纹理失败就一起判废。
//
// 注意：**本工程不在 nativeBuildMesh 里跑 MeshPostProcessor**。nativeBuildMesh
// 用的 MeshEngine 已自带去小分量 / 去孤立面 / Taubin / QEM 是一条已验收的链路，
// 不动它；V0.6 的清理只作用在导出资产这份 AoS 副本上。因此屏幕上 AR overlay
// 显示的仍是 V0.5 的 vertex-color 网格，而 GLB 是清理过、带纹理的资产。
// ===========================================================================

/** 本工程 SoA Mesh -> V0.6 模块使用的 AoS Mesh。 */
static AosMesh toAosMesh(const Mesh& m) {
    AosMesh out;
    const std::size_t n = m.vertexCount();
    if (n == 0 || m.indices.size() < 3) {
        return out;
    }
    out.vertices.resize(n);
    const bool haveNormals = (m.normals.size() >= n * 3);
    const bool haveColors = (m.colors.size() >= n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        AosVertex& v = out.vertices[i];
        v.px = m.positions[i * 3 + 0];
        v.py = m.positions[i * 3 + 1];
        v.pz = m.positions[i * 3 + 2];
        if (haveNormals) {
            v.nx = m.normals[i * 3 + 0];
            v.ny = m.normals[i * 3 + 1];
            v.nz = m.normals[i * 3 + 2];
        }
        if (haveColors) {
            v.r = m.colors[i * 3 + 0];
            v.g = m.colors[i * 3 + 1];
            v.b = m.colors[i * 3 + 2];
        }
    }
    out.indices = m.indices;
    return out;
}

/** 开始新一轮扫描时清空 HQ 关键帧登记表。 */
extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeClearTextureKeyframes(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    gTextureKeyframes.clear();
    textureBakeStats = TextureBakeStats{};
    uvUnwrapStats = UvUnwrapStats{};
}

/**
 * 登记一张 HQ still 作为纹理候选视角。
 *
 * pose12 = camera->world 的 12 个 float（R 行主序 9 个 + t 3 个），由 Kotlin
 * 侧用 nativeGetRenderPoseAt(frameSensorTs) 拿到 —— 也就是**拍摄那一刻**的
 * VINS 位姿，不是当前帧位姿。
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeRegisterTextureKeyframe(
        JNIEnv* env, jobject,
        jstring path, jint width, jint height,
        jfloat fx, jfloat fy, jfloat cx, jfloat cy,
        jfloatArray pose12, jfloat quality, jlong timestampNs) {
    if (!path || !pose12 || width <= 0 || height <= 0 ||
        fx <= 1.0f || fy <= 1.0f ||
        env->GetArrayLength(pose12) < 12) {
        return JNI_FALSE;
    }

    const char* p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return JNI_FALSE;
    }
    std::string imagePath(p);
    env->ReleaseStringUTFChars(path, p);

    {
        std::ifstream test(imagePath, std::ios::binary);
        if (!test.good()) {
            return JNI_FALSE;
        }
    }

    jfloat values[12];
    env->GetFloatArrayRegion(pose12, 0, 12, values);

    TextureKeyframe k;
    k.imagePath = std::move(imagePath);
    k.width = width;
    k.height = height;
    k.fx = fx;
    k.fy = fy;
    k.cx = cx;
    k.cy = cy;
    for (int i = 0; i < 9; ++i) {
        k.Rwc[i] = values[i];
    }
    k.twc[0] = values[9];
    k.twc[1] = values[10];
    k.twc[2] = values[11];
    k.quality = std::clamp(static_cast<float>(quality), 0.05f, 3.0f);
    k.timestampNs = timestampNs > 0 ? static_cast<std::uint64_t>(timestampNs) : 0u;

    std::lock_guard<std::mutex> lk(gStateMutex);

    // 同一个文件被回调重复投递时去重。
    for (const auto& existing : gTextureKeyframes) {
        if (existing.imagePath == k.imagePath) {
            return JNI_TRUE;
        }
    }
    gTextureKeyframes.push_back(std::move(k));

    // 硬上限：元数据不能无界增长。烘焙时会另按 quality + 视角多样性挑选。
    constexpr std::size_t kRegistryLimit = 96;
    if (gTextureKeyframes.size() > kRegistryLimit) {
        auto worst = std::min_element(
            gTextureKeyframes.begin(), gTextureKeyframes.end(),
            [](const TextureKeyframe& a, const TextureKeyframe& b) {
                return a.quality < b.quality;
            });
        if (worst != gTextureKeyframes.end()) {
            gTextureKeyframes.erase(worst);
        }
    }
    return JNI_TRUE;
}

/** 清理几何 + 展开 UV + 多视角烘焙 + 写出内嵌 JPEG 的 textured GLB。 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeBakeTexturedGlb(
        JNIEnv* env, jobject,
        jstring path, jint atlasResolution, jint maxKeyframes) {
    if (path == nullptr) {
        return JNI_FALSE;
    }

    AosMesh meshCopy;
    std::vector<TextureKeyframe> keyframes;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& src = meshEngine.mesh();
        if (src.empty()) {
            return JNI_FALSE;
        }
        meshCopy = toAosMesh(src);
        keyframes = gTextureKeyframes;
    }
    if (meshCopy.empty() || keyframes.empty()) {
        LOGI("nativeBakeTexturedGlb skip: mesh=%zu keyframes=%zu",
             meshCopy.triangleCount(), keyframes.size());
        return JNI_FALSE;
    }

    const int resolution = std::clamp(static_cast<int>(atlasResolution), 512, 4096);

    // ---- 几何清理：weld / 去漂浮分量 / ear-clipping 补小洞 / QEM ----
    MeshPostProcessOptions cleanup;
    cleanup.weldEpsilon = std::clamp(meshStats.voxelSize * 0.08f, 0.00030f, 0.00120f);
    cleanup.minComponentTriangles = meshQuality >= 2 ? 24 : (meshQuality <= 0 ? 72 : 48);
    cleanup.minComponentAreaRatio = meshQuality >= 2 ? 0.0020f : 0.0035f;
    cleanup.maxHoleEdges = meshQuality >= 2 ? 72 : 56;
    cleanup.maxHoleDiameterMeters = meshQuality >= 2 ? 0.060f : 0.075f;
    cleanup.maxHoleDiameterBBoxRatio = meshQuality >= 2 ? 0.10f : 0.12f;
    cleanup.targetTriangles = meshQuality >= 2 ? 180000 : (meshQuality <= 0 ? 30000 : 80000);
    cleanup.qemMaxPasses = 10;
    cleanup.qemMaxNormalFlipDeg = meshQuality >= 2 ? 65.0f : 72.0f;
    cleanup.preserveBoundary = true;

    MeshPostProcessStats cleanupStats;
    if (!MeshPostProcessor::run(meshCopy, cleanup, &cleanupStats)) {
        LOGI("nativeBakeTexturedGlb FAILED: mesh cleanup");
        return JNI_FALSE;
    }

    UvMesh uvMesh;
    UvUnwrapStats uvStats;
    if (!UvUnwrapper::unwrap(meshCopy, resolution, 8, uvMesh, &uvStats)) {
        LOGI("nativeBakeTexturedGlb FAILED: uv unwrap");
        return JNI_FALSE;
    }

    TextureBakeOptions options;
    options.atlasResolution = resolution;
    options.maxKeyframes = std::clamp(static_cast<int>(maxKeyframes), 4, 24);
    options.maxBlendFrames = 3;
    options.sourceMaxSide = resolution >= 4096 ? 2200 : (resolution >= 2048 ? 1600 : 1280);
    options.visibilityMaxSide = resolution >= 4096 ? 512 : 384;
    options.gutterDilationPixels = resolution >= 4096 ? 10 : 6;
    options.jpegQuality = resolution >= 4096 ? 94 : 92;

    cv::Mat atlas;
    TextureBakeStats bakeStats;
    if (!TextureBaker::bake(uvMesh, keyframes, options, atlas, &bakeStats)) {
        LOGI("nativeBakeTexturedGlb FAILED: texture bake");
        return JNI_FALSE;
    }

    std::vector<std::uint8_t> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, options.jpegQuality};
    if (!cv::imencode(".jpg", atlas, jpeg, params) || jpeg.empty()) {
        LOGI("nativeBakeTexturedGlb FAILED: jpeg encode");
        return JNI_FALSE;
    }

    const char* p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return JNI_FALSE;
    }
    const bool ok = TexturedGlbExporter::write(p, uvMesh, jpeg);
    env->ReleaseStringUTFChars(path, p);

    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        meshCleanupStats = cleanupStats;
        if (ok) {
            textureBakeStats = bakeStats;
            uvUnwrapStats = uvStats;
        }
    }

    LOGI("nativeBakeTexturedGlb ok=%d cleanTris=%zu->%zu weld=%zu compsRemoved=%zu "
         "holes=%zu(+%zu tris) uv=%s atlas=%dx%d frames=%d/%d cov=%.1f%% jpeg=%zu",
         ok ? 1 : 0,
         cleanupStats.inputTriangles, cleanupStats.outputTriangles,
         cleanupStats.weldedVertices, cleanupStats.removedComponents,
         cleanupStats.filledHoles, cleanupStats.addedHoleTriangles,
         uvStats.usedXatlas ? "xatlas" : "fallback",
         uvStats.atlasWidth, uvStats.atlasHeight,
         bakeStats.usedKeyframes, bakeStats.requestedKeyframes,
         bakeStats.coveragePercent, jpeg.size());
    return ok ? JNI_TRUE : JNI_FALSE;
}

/**
 * 纹理统计（9 槽）。
 *   0 已登记关键帧数 / 1 实际加载 / 2 实际使用 / 3 atlasW / 4 atlasH
 *   5 有纹理的三角形 / 6 回退 vertex color 的三角形 / 7 覆盖率×10 / 8 是否 xatlas
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTextureStats(
        JNIEnv* env, jobject, jintArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 9) {
        return JNI_FALSE;
    }
    jint values[9] = {0};
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        values[0] = static_cast<jint>(std::min<std::size_t>(
            gTextureKeyframes.size(), static_cast<std::size_t>(std::numeric_limits<jint>::max())));
        values[1] = textureBakeStats.loadedKeyframes;
        values[2] = textureBakeStats.usedKeyframes;
        values[3] = uvUnwrapStats.atlasWidth;
        values[4] = uvUnwrapStats.atlasHeight;
        values[5] = static_cast<jint>(std::min<std::size_t>(
            textureBakeStats.texturedTriangles, static_cast<std::size_t>(std::numeric_limits<jint>::max())));
        values[6] = static_cast<jint>(std::min<std::size_t>(
            textureBakeStats.fallbackTriangles, static_cast<std::size_t>(std::numeric_limits<jint>::max())));
        values[7] = static_cast<jint>(std::clamp(textureBakeStats.coveragePercent * 10.0f, 0.0f, 1000.0f));
        values[8] = uvUnwrapStats.usedXatlas ? 1 : 0;
    }
    env->SetIntArrayRegion(out, 0, 9, values);
    return JNI_TRUE;
}

/**
 * 导出前几何清理统计（10 槽）。
 *   0 输入三角形 / 1 输出三角形 / 2 weld 后顶点 / 3 去掉的分量
 *   4 去掉的分量三角形 / 5 边界环 / 6 补的洞 / 7 补洞新增三角形
 *   8 QEM 塌缩边 / 9 输出顶点
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshCleanupStats(
        JNIEnv* env, jobject, jintArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 10) {
        return JNI_FALSE;
    }
    jint values[10] = {0};
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        values[0] = static_cast<jint>(meshCleanupStats.inputTriangles);
        values[1] = static_cast<jint>(meshCleanupStats.outputTriangles);
        values[2] = static_cast<jint>(meshCleanupStats.weldedVertices);
        values[3] = static_cast<jint>(meshCleanupStats.removedComponents);
        values[4] = static_cast<jint>(meshCleanupStats.removedComponentTriangles);
        values[5] = static_cast<jint>(meshCleanupStats.boundaryLoops);
        values[6] = static_cast<jint>(meshCleanupStats.filledHoles);
        values[7] = static_cast<jint>(meshCleanupStats.addedHoleTriangles);
        values[8] = static_cast<jint>(meshCleanupStats.qemCollapsedEdges);
        values[9] = static_cast<jint>(meshCleanupStats.outputVertices);
    }
    env->SetIntArrayRegion(out, 0, 10, values);
    return JNI_TRUE;
}
