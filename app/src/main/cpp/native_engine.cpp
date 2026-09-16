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
#include "gaussian_engine.h"
#include "vio_engine.h"
#include "depth_fusion.h"
#include "ai_quality.h"
#include "tsdf_engine.h"
#include "ai_backend.h"
#include "keyframe_engine.h"
#include "object_tracker.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MobileScan3D", __VA_ARGS__)
#define TARGET_DEPTH_FILTER_ENABLED 0

// 全局状态被 4 个线程并发访问：IMU 线程（nativeOnImu）、相机线程
// （nativeOnCameraFrame）、深度线程（nativeOnDepthMap）、GL 渲染/UI 线程
// （nativeGetGaussians / nativeGetStats）。原代码完全无锁，snaps 的 deque
// 迭代器失效和 g_ 的 vector 扩容都是未定义行为（崩溃/花屏根因）。
static std::mutex gStateMutex;
static NullAiBackend aiBackend;
static GaussianEngine g;
static VioEngine vio;
static DepthFusion df;
static AiQualityEngine ai;
static TsdfEngine tsdf;
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
            tracker->updateFrame(yy, w, h, rs, static_cast<uint64_t>(frameTs));
            tracker->track(yy, w, h, rs, static_cast<uint64_t>(frameTs));

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
 * 用「当前 depth 帧 + 当前 target ROI + 该帧 pose」重建一小片世界点。
 *
 * ROI 用 ObjectTracker 的归一化 bbox，它和 depth 帧、相机帧共用同一套
 * 归一化坐标（updateFromDepth 就是按 depth 的 w/h 归一化的），
 * 所以这里直接乘 w/h 即可，不需要再做任何旋转/裁剪。
 *
 * 必须放在 rotatePoint() 之后 —— 它依赖那个函数。
 */
static void buildTargetDebugLayer(const float* d, int w, int h,
                                 const TargetTrackInfo& ti,
                                 const float R[9], const float T[3],
                                 bool poseOk) {
    targetDebugPointCount = 0;
    targetDebugRoiPixels = 0;
    targetDebugValidPixels = 0;
    targetDebugPoseOk = poseOk;
    targetDebugTs = 0;
    if (!targetDebugEnabled) {
        return;
    }
    if (!poseOk || d == nullptr || w < 4 || h < 4) {
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

    int step = 1;
    while ((roiW / step) * (roiH / step) > kTargetDebugMaxPoints) {
        step++;
    }

    for (int y = y0; y <= y1; y += step) {
        for (int x = x0; x <= x1; x += step) {
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
            float Xw, Yw, Zw;
            rotatePoint(R, T, Xc, Yc, z, Xw, Yw, Zw);
            float* p = targetDebugPoints.data() +
                       static_cast<size_t>(targetDebugPointCount) * 6;
            p[0] = Xw;
            p[1] = Yw;
            p[2] = Zw;
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

#if TARGET_DEPTH_FILTER_ENABLED
    {
        std::shared_ptr<ObjectTracker> tracker;
        {
            std::lock_guard<std::mutex> lk(gStateMutex);
            tracker = objectTracker;
        }
        if (!tracker || !tracker->isEnabled() || !tracker->isTracking()) {
            return;
        }
        if (!tracker->filterDepth(d.data(), w, h, (uint64_t)t)) {
            return;
        }
    }
#endif

    std::lock_guard<std::mutex> lk(gStateMutex);
    df.ingestExternalDepth(d.data(), w, h, confidence, (uint64_t)t);
    haveExternalDepth = true;
    depthFrames++;

    // 先抓一份当前 target 状态：调试层和 fuseDepth 都要用。
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

    for (const auto& s : snaps) {
        int64_t diff = (int64_t)s.ts - (int64_t)t;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff < 50000000LL) {
            fuseDepth(d.data(), w, h, s, confidence);
            // 调试层用的必须是**这一帧**的 depth 与**这一帧的** pose，
            // 所以放在同一个 snap 匹配分支里，而不是另找一次。
            if (haveTi) {
                buildTargetDebugLayer(d.data(), w, h, ti, s.R, s.t, true);
                targetDebugTs = static_cast<uint64_t>(t);
            }
            break;
        }
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
    bool ok = tsdf.exportPly(p);
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
      << " poseOk=" << (targetDebugPoseOk ? 1 : 0)
      << " builds=" << targetDebugBuilds
      << " queries=" << targetDebugQueries << "\n"
      << "PointCloud bbox min=(" << pminX << ", " << pminY << ", " << pminZ << ")"
      << " max=(" << pmaxX << ", " << pmaxY << ", " << pmaxZ << ")\n"
      << "PointCloud centroid=(" << pcx << ", " << pcy << ", " << pcz << ")\n"
      << "PointCloud vs Cam offset=(" << offX << ", " << offY << ", " << offZ
      << ")  dist=" << offDist << "\n"
      << "TSDF voxels: " << tsdf.voxels() << "\n"
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
        n = g.copyPoints(dst, (size_t)maxPoints, need);
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
    s << " · 地图 " << g.count()
      << " 确认 " << g.confirmedCount(2)
      << " 稳定 " << g.confirmedCount(3)
      << " 绘制 " << gArDrawnPoints;
    return e->NewStringUTF(s.str().c_str());
}

// nativeGetDepthDiagnostics 的槽数。
// **必须与 Kotlin 侧 NativeBridge.DEPTH_DIAGNOSTIC_SLOTS 保持一致。**
// 这里做了长度检查、长度不足就安全返回 0；但如果调用方按旧槽数建数组、
// 又按新槽数索引，越界会发生在 Kotlin 侧（「导出反馈报告」闪退那次就是这个形态）。
static constexpr int kDepthDiagSlots = 7;

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetPointCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return (jint)g.count();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSelectTarget(JNIEnv*, jobject, jfloat u, jfloat v) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (!objectTracker) return JNI_FALSE;
    try {
        return objectTracker->requestTarget(u, v) ? JNI_TRUE : JNI_FALSE;
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
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetObjectLockEnabled(JNIEnv*, jobject, jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (objectTracker) objectTracker->setEnabled(enabled);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetState(JNIEnv* env, jobject, jfloatArray out) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (!objectTracker) return 0;
    const TargetTrackInfo info = objectTracker->info();
    if (out != nullptr) {
        jfloat* dst = env->GetFloatArrayElements(out, nullptr);
        if (dst != nullptr) {
            const jsize n = env->GetArrayLength(out);
            if (n >= 10) {
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
            }
            env->ReleaseFloatArrayElements(out, dst, 0);
        }
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
    env->ReleaseStringUTFChars(backbone, b);
    env->ReleaseStringUTFChars(head, h);
    return ok ? JNI_TRUE : JNI_FALSE;
}
