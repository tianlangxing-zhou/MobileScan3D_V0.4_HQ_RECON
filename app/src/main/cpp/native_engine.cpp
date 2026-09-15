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
static std::unique_ptr<ObjectTracker> objectTracker;
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

static void fuseDepth(const float* depth, int w, int h, const FrameSnap& s, float confidence) {
    if (!depth || w < 4 || h < 4 || s.rgb.empty()) {
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
    objectTracker = std::make_unique<ObjectTracker>();
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

    if (objectTracker) {
        objectTracker->updateFrame(yy, w, h, rs, (uint64_t)frameTs);
        objectTracker->track(yy, w, h, rs, (uint64_t)frameTs);
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

    std::lock_guard<std::mutex> lk(gStateMutex);
#if TARGET_DEPTH_FILTER_ENABLED
    if (objectTracker && objectTracker->isEnabled()) {
        if (!objectTracker->isTracking()) {
            return;
        }
        objectTracker->updateFromDepth(d.data(), w, h, (uint64_t)t);
        if (!objectTracker->filterDepth(d.data(), w, h, (uint64_t)t)) {
            return;
        }
    }
#endif

    df.ingestExternalDepth(d.data(), w, h, confidence, (uint64_t)t);
    haveExternalDepth = true;
    depthFrames++;

    for (const auto& s : snaps) {
        int64_t diff = (int64_t)s.ts - (int64_t)t;
        if (diff < 0) {
            diff = -diff;
        }
        if (diff < 50000000LL) {
            fuseDepth(d.data(), w, h, s, confidence);
            break;
        }
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
      << " merged=" << g.mergedCount() << "\n"
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
                                                      jfloatArray out, jint maxPoints) {
    jfloat* dst = e->GetFloatArrayElements(out, nullptr);
    if (!dst) {
        return 0;
    }
    size_t n;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        n = g.copyPoints(dst, (size_t)maxPoints);
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
      << " · 深度 " << depthFrames << " 帧"
      << " · 点云 " << g.count();
    return e->NewStringUTF(s.str().c_str());
}

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
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (objectTracker) objectTracker->reset();
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
