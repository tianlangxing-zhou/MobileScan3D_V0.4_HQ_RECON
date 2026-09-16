#include <jni.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "vins/vins_bridge.h"
#include "vins/vins_params.h"
#include "vins/feature_tracker/feature_tracker.h"
#include "vins/estimator/estimator.h"
#include "camodocal/camera_models/PinholeCamera.h"

using namespace std;
using namespace Eigen;

static camodocal::CameraPtr g_camera;
static FeatureTracker g_tracker[NUM_OF_CAM];
static Estimator g_estimator;
static bool g_vinsReady = false;
static double g_lastImuTimestamp = -1.0;
static double g_lastImuDt = 0.0;
static double g_lastFeaturePubTimestamp = -1.0;
// V0.7: exact visual frame that most recently reached processImage().
static std::uint64_t g_lastEstimatorImageTimestampNs = 0;
static bool g_firstFeaturePublish = true;

// VINS 输入图像的尺寸（nativeVinsInit 里由 Kotlin 下发，通常是 640 长边）。
// vinsFeatureDepthMedianInRoi() 需要它把归一化 ROI 换算成像素坐标 ——
// FeaturePerFrame::uv 存的是 VINS 图像像素，不是归一化坐标。
static int g_vinsImageW = 0;
static int g_vinsImageH = 0;

struct ImuSample {
    double t = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
};

static std::deque<ImuSample> g_imuBuffer;
static std::mutex g_imuBufferMutex;

static double g_estimatorTime = -1.0;
static Eigen::Vector3d g_lastEstimatorAcc = Eigen::Vector3d::Zero();
static Eigen::Vector3d g_lastEstimatorGyr = Eigen::Vector3d::Zero();
static bool g_haveEstimatorImu = false;

// 保护 tracker + estimator。
static std::mutex g_vinsMutex;

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsInit(
        JNIEnv* env,
        jobject,
        jfloat fx,
        jfloat fy,
        jfloat cx,
        jfloat cy,
        jint w,
        jint h,
        jfloatArray ric,
        jfloatArray tic,
        jfloat accN,
        jfloat accW,
        jfloat gyrN,
        jfloat gyrW) {
    setImageSize((double)h, (double)w);
    g_vinsImageW = (int)w;
    g_vinsImageH = (int)h;
    setFeatureTrackerParams(h, w, (int)fx, 0, 1, 30, 150, 1.0);
    setEstimatorParams(accN, accW, gyrN, gyrW, 5.0, 10.0 / 460.0, 8, 0.04);

    jfloat* r = env->GetFloatArrayElements(ric, nullptr);
    jfloat* t = env->GetFloatArrayElements(tic, nullptr);
    if (r != nullptr && t != nullptr) {
        Eigen::Matrix3d Ric;
        Ric << r[0], r[1], r[2],
               r[3], r[4], r[5],
               r[6], r[7], r[8];
        Eigen::Vector3d Tic(t[0], t[1], t[2]);
        setExtrinsic(Ric, Tic);
    }
    if (r != nullptr) {
        env->ReleaseFloatArrayElements(ric, r, JNI_ABORT);
    }
    if (t != nullptr) {
        env->ReleaseFloatArrayElements(tic, t, JNI_ABORT);
    }

    g_camera = camodocal::CameraPtr(
        new camodocal::PinholeCamera("cam0", w, h, 0.0, 0.0, 0.0, 0.0, fx, fy, cx, cy));
    g_tracker[0].setCamera(g_camera);
    setExtrinsicEstimateMode(0);
    setTemporalParams(0.0, 1, 0, 0.0);
    g_estimator.setParameter();
    g_vinsReady = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsImu(
        JNIEnv*,
        jobject,
        jlong t,
        jfloat ax,
        jfloat ay,
        jfloat az,
        jfloat gx,
        jfloat gy,
        jfloat gz) {
    const double timestamp = static_cast<double>(t) * 1e-9;
    vinsInputImu(timestamp, ax, ay, az, gx, gy, gz);
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsImage(
        JNIEnv* env,
        jobject,
        jdouble t,
        jbyteArray gray,
        jint w,
        jint h,
        jint stride) {
    if (!g_vinsReady) {
        return;
    }
    jsize n = env->GetArrayLength(gray);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    env->GetByteArrayRegion(gray, 0, n, reinterpret_cast<jbyte*>(buf.data()));

    cv::Mat img(h, w, CV_8UC1, buf.data(), static_cast<size_t>(stride));
    g_tracker[0].readImage(img, t);

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
    for (unsigned int j = 0; j < g_tracker[0].ids.size(); j++) {
        if (g_tracker[0].track_cnt[j] > 1) {
            const int feature_id = g_tracker[0].ids[j];
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << g_tracker[0].cur_un_pts[j].x,
                               g_tracker[0].cur_un_pts[j].y,
                               1.0,
                               g_tracker[0].cur_pts[j].x,
                               g_tracker[0].cur_pts[j].y,
                               g_tracker[0].pts_velocity[j].x,
                               g_tracker[0].pts_velocity[j].y;
            image[feature_id].emplace_back(0, xyz_uv_velocity);
        }
    }

    std_msgs::Header header;
    header.stamp.sec = t;
    g_estimator.processImage(image, header);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsGetPose(JNIEnv* env, jobject, jfloatArray out) {
    if (!vinsInitialized()) {
        return JNI_FALSE;
    }
    jfloat* o = env->GetFloatArrayElements(out, nullptr);
    if (o == nullptr) {
        return JNI_FALSE;
    }
    const Eigen::Vector3d p = g_estimator.Ps[WINDOW_SIZE];
    const Eigen::Matrix3d R = g_estimator.Rs[WINDOW_SIZE];
    const Eigen::Quaterniond q(R);
    o[0] = static_cast<float>(p.x());
    o[1] = static_cast<float>(p.y());
    o[2] = static_cast<float>(p.z());
    o[3] = static_cast<float>(q.x());
    o[4] = static_cast<float>(q.y());
    o[5] = static_cast<float>(q.z());
    o[6] = static_cast<float>(q.w());
    env->ReleaseFloatArrayElements(out, o, 0);
    return JNI_TRUE;
}

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
        float gyrW) {
    setImageSize((double)h, (double)w);
    g_vinsImageW = (int)w;
    g_vinsImageH = (int)h;
    setFeatureTrackerParams(h, w, (int)fx, 0, 1, 30, 150, 1.0);
    setEstimatorParams(accN, accW, gyrN, gyrW, 5.0, 10.0 / 460.0, 8, 0.04);

    Eigen::Matrix3d Ric = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Tic = Eigen::Vector3d::Zero();
    if (ric != nullptr && tic != nullptr) {
        Ric << ric[0], ric[1], ric[2],
               ric[3], ric[4], ric[5],
               ric[6], ric[7], ric[8];
        Tic = Eigen::Vector3d(tic[0], tic[1], tic[2]);
    }
    setExtrinsic(Ric, Tic);

    g_camera = camodocal::CameraPtr(
        new camodocal::PinholeCamera("cam0", w, h, 0.0, 0.0, 0.0, 0.0, fx, fy, cx, cy));
    std::lock_guard<std::mutex> lk(g_vinsMutex);
    {
        std::lock_guard<std::mutex> blk(g_imuBufferMutex);
        g_imuBuffer.clear();
    }

    g_tracker[0] = FeatureTracker();
    FeatureTracker::n_id = 0;
    g_tracker[0].setCamera(g_camera);
    // 重复初始化（如重开相机/二次扫描）时清掉滑窗与预积分的残留状态，
    // 否则旧内参/旧时刻的滑窗会与新会话混跑。clearState 后必须重新 setParameter。
    g_estimator.clearState();
    setExtrinsicEstimateMode(0);
    setTemporalParams(0.0, 1, 0, 0.0);
    g_estimator.setParameter();
    g_estimatorTime = -1.0;
    g_lastEstimatorAcc.setZero();
    g_lastEstimatorGyr.setZero();
    g_haveEstimatorImu = false;
    g_lastImuDt = 0.0;
    g_lastImuTimestamp = -1.0;
    g_lastFeaturePubTimestamp = -1.0;
    g_lastEstimatorImageTimestampNs = 0;
    g_firstFeaturePublish = true;
    g_vinsReady = true;
}

void vinsInputImu(double timestamp, double ax, double ay, double az, double gx, double gy, double gz) {
    if (!g_vinsReady) {
        return;
    }

    ImuSample sample;
    sample.t = timestamp;
    sample.acc = Eigen::Vector3d(ax, ay, az);
    sample.gyr = Eigen::Vector3d(gx, gy, gz);

    std::lock_guard<std::mutex> lk(g_imuBufferMutex);

    if (!g_imuBuffer.empty() && timestamp <= g_imuBuffer.back().t) {
        return;
    }

    g_imuBuffer.push_back(sample);

    while (g_imuBuffer.size() > 4000) {
        g_imuBuffer.pop_front();
    }
}

static bool processImuUntil(double imageTime) {
    std::vector<ImuSample> samples;
    ImuSample futureSample;
    bool haveFuture = false;

    {
        std::lock_guard<std::mutex> lk(g_imuBufferMutex);

        if (g_imuBuffer.empty()) {
            return false;
        }

        if (g_imuBuffer.back().t <= imageTime) {
            return false;
        }

        while (!g_imuBuffer.empty() && g_imuBuffer.front().t <= imageTime) {
            samples.push_back(g_imuBuffer.front());
            g_imuBuffer.pop_front();
        }

        if (!g_imuBuffer.empty()) {
            futureSample = g_imuBuffer.front();
            haveFuture = true;
        }
    }

    if (!haveFuture) {
        return false;
    }

    size_t index = 0;

    if (!g_haveEstimatorImu) {
        if (samples.empty()) {
            return false;
        }

        const auto& s = samples[0];
        g_estimatorTime = s.t;
        g_lastEstimatorAcc = s.acc;
        g_lastEstimatorGyr = s.gyr;
        g_estimator.processIMU(0.0, s.acc, s.gyr);
        g_haveEstimatorImu = true;
        index = 1;
    }

    for (; index < samples.size(); ++index) {
        const auto& s = samples[index];
        const double dt = s.t - g_estimatorTime;

        if (dt <= 0.0) {
            continue;
        }

        if (dt > 0.1) {
            return false;
        }

        g_estimator.processIMU(dt, s.acc, s.gyr);
        g_lastImuDt = dt;
        g_estimatorTime = s.t;
        g_lastEstimatorAcc = s.acc;
        g_lastEstimatorGyr = s.gyr;
    }

    if (g_estimatorTime < imageTime) {
        const double dt1 = imageTime - g_estimatorTime;
        const double dt2 = futureSample.t - imageTime;

        if (dt1 < 0.0 || dt2 < 0.0 || dt1 + dt2 <= 0.0) {
            return false;
        }

        const double w1 = dt2 / (dt1 + dt2);
        const double w2 = dt1 / (dt1 + dt2);

        const Eigen::Vector3d acc =
            w1 * g_lastEstimatorAcc + w2 * futureSample.acc;
        const Eigen::Vector3d gyr =
            w1 * g_lastEstimatorGyr + w2 * futureSample.gyr;

        g_estimator.processIMU(dt1, acc, gyr);
        g_lastImuDt = dt1;
        g_estimatorTime = imageTime;
        g_lastEstimatorAcc = acc;
        g_lastEstimatorGyr = gyr;
    }

    return true;
}

void vinsInputImage(double t, const std::uint8_t* gray, int w, int h, int stride) {
    if (!g_vinsReady || gray == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);

    constexpr double FEATURE_FREQ = 10.0;
    constexpr double FEATURE_PERIOD = 1.0 / FEATURE_FREQ;

    if (g_lastFeaturePubTimestamp < 0.0 ||
        t - g_lastFeaturePubTimestamp >= FEATURE_PERIOD) {
        PUB_THIS_FRAME = true;
    } else {
        PUB_THIS_FRAME = false;
    }

    cv::Mat img(h, w, CV_8UC1, const_cast<std::uint8_t*>(gray), static_cast<size_t>(stride));
    g_tracker[0].readImage(img, t);

    for (unsigned int i = 0;; ++i) {
        if (!g_tracker[0].updateID(i)) {
            break;
        }
    }

    if (!PUB_THIS_FRAME) {
        return;
    }

    g_lastFeaturePubTimestamp = t;

    if (g_firstFeaturePublish) {
        g_firstFeaturePublish = false;
        return;
    }

    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
    for (unsigned int j = 0; j < g_tracker[0].ids.size(); j++) {
        if (g_tracker[0].track_cnt[j] <= 1) {
            continue;
        }

        const int feature_id = g_tracker[0].ids[j];
        if (feature_id < 0) {
            continue;
        }

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << g_tracker[0].cur_un_pts[j].x,
                           g_tracker[0].cur_un_pts[j].y,
                           1.0,
                           g_tracker[0].cur_pts[j].x,
                           g_tracker[0].cur_pts[j].y,
                           g_tracker[0].pts_velocity[j].x,
                           g_tracker[0].pts_velocity[j].y;
        image[feature_id].emplace_back(0, xyz_uv_velocity);
    }

    if (image.size() < 20) {
        return;
    }

    double syncTime = t;
    if (ESTIMATE_TD) {
        syncTime += g_estimator.td;
    }

    if (!processImuUntil(syncTime)) {
        return;
    }

    std_msgs::Header header;
    header.stamp.sec = t;
    g_estimator.processImage(image, header);
    g_lastEstimatorImageTimestampNs =
        static_cast<std::uint64_t>(
            std::llround(
                std::max(0.0, t) * 1e9));
}

bool vinsGetPose(float* out7) {
    if (!vinsInitialized() || out7 == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lk(g_vinsMutex);
    const Eigen::Vector3d p = g_estimator.Ps[WINDOW_SIZE];
    const Eigen::Matrix3d R = g_estimator.Rs[WINDOW_SIZE];
    const Eigen::Quaterniond q(R);
    out7[0] = static_cast<float>(p.x());
    out7[1] = static_cast<float>(p.y());
    out7[2] = static_cast<float>(p.z());
    out7[3] = static_cast<float>(q.x());
    out7[4] = static_cast<float>(q.y());
    out7[5] = static_cast<float>(q.z());
    out7[6] = static_cast<float>(q.w());
    return true;
}

bool vinsReady() {
    return g_vinsReady;
}

bool vinsInitialized() {
    return g_vinsReady && g_estimator.solver_flag == Estimator::NON_LINEAR;
}

bool vinsGetHealth(VinsHealth* out) {
    if (out == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);

    const int i = WINDOW_SIZE;

    out->initialized =
        g_estimator.solver_flag == Estimator::NON_LINEAR;

    out->velocity =
        static_cast<float>(g_estimator.Vs[i].norm());

    out->accBias =
        static_cast<float>(g_estimator.Bas[i].norm());

    out->gyroBias =
        static_cast<float>(g_estimator.Bgs[i].norm());

    out->gravity =
        static_cast<float>(g_estimator.g.norm());

    out->trackedFeatures =
        g_estimator.f_manager.last_track_num;

    out->lastImuDt = g_lastImuDt;
    out->timeOffset = g_estimator.td;

    const Eigen::Matrix3d R = g_estimator.ric[0];
    const Eigen::Vector3d T = g_estimator.tic[0];
    int n = 0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out->ric[n++] = static_cast<float>(R(r, c));
        }
    }
    out->tic[0] = static_cast<float>(T.x());
    out->tic[1] = static_cast<float>(T.y());
    out->tic[2] = static_cast<float>(T.z());

    return true;
}

float vinsFeatureDepthMedian() {
    std::lock_guard<std::mutex> lk(g_vinsMutex);

    if (g_estimator.solver_flag != Estimator::NON_LINEAR) {
        return 0.0f;
    }

    std::vector<double> depths;
    depths.reserve(g_estimator.f_manager.feature.size());
    for (const auto &it_per_id : g_estimator.f_manager.feature) {
        // 只看三角化成功的特征，初始化占位值(INIT_DEPTH)会污染中位数
        if (it_per_id.solve_flag != 1) {
            continue;
        }
        const double d = it_per_id.estimated_depth;
        if (std::isfinite(d) && d > 0.1) {
            depths.push_back(d);
        }
    }

    if (depths.empty()) {
        return 0.0f;
    }

    const size_t n = depths.size();
    std::nth_element(depths.begin(), depths.begin() + n / 2, depths.end());
    return static_cast<float>(depths[n / 2]);
}

// ------------------------------------------------------- V0.5 anchor-frame 修正
//
// FeaturePerId::estimated_depth 由 FeatureManager::triangulate() 解出，而它第一个
// 投影矩阵是 P0 = [I | 0] —— 也就是说这个深度是 **start_frame（anchor）那一帧的
// camera 坐标系**下的 z。但 feature_per_frame.back().uv 是**最新观测帧**的像素
// 坐标。相机在 anchor 与观测帧之间发生了平移/旋转，同一个 3D 点的 z 会变，所以
// 两者**不能直接配对**。
//
// 正确做法：先在 anchor camera 系还原 3D 点（FeaturePerFrame::point 的 z 分量恒
// 为 1，见 vins_bridge 里 xyz_uv_velocity 的构造），经 world 变换到观测帧 camera
// 系，再取 z 作为该 uv 的米制深度。
//
// 注意：这里不用 FeaturePerId::endFrame() —— 它是非 const 成员函数，而循环变量是
// const 引用，直接调用会因丢掉 const 限定而编译失败。等价地手算索引即可。
//
// 调用方必须已持有 g_vinsMutex。
static bool vinsAnchorDepthPair(const FeaturePerId& f, Eigen::Vector2d* outUv,
                                double* outDepth) {
    if (f.solve_flag != 1 || f.feature_per_frame.empty()) {
        return false;
    }
    const double anchorDepth = f.estimated_depth;
    if (!std::isfinite(anchorDepth) || anchorDepth <= 0.1) {
        return false;
    }

    const int anchorFrame = f.start_frame;
    const int obsFrame = anchorFrame + (int)f.feature_per_frame.size() - 1;
    if (anchorFrame < 0 || anchorFrame > WINDOW_SIZE ||
        obsFrame < 0 || obsFrame > WINDOW_SIZE) {
        return false;
    }

    const Eigen::Vector2d& uv = f.feature_per_frame.back().uv;
    if (!std::isfinite(uv.x()) || !std::isfinite(uv.y())) {
        return false;
    }

    // anchor 帧 camera 系 -> world -> 观测帧 camera 系
    const Eigen::Vector3d pAnchor =
        f.feature_per_frame.front().point * anchorDepth;
    const Eigen::Matrix3d RwcAnchor =
        g_estimator.Rs[anchorFrame] * g_estimator.ric[0];
    const Eigen::Vector3d twcAnchor =
        g_estimator.Ps[anchorFrame] +
        g_estimator.Rs[anchorFrame] * g_estimator.tic[0];
    const Eigen::Vector3d pWorld = RwcAnchor * pAnchor + twcAnchor;

    const Eigen::Matrix3d RwcObs =
        g_estimator.Rs[obsFrame] * g_estimator.ric[0];
    const Eigen::Vector3d twcObs =
        g_estimator.Ps[obsFrame] +
        g_estimator.Rs[obsFrame] * g_estimator.tic[0];
    const Eigen::Vector3d pObs = RwcObs.transpose() * (pWorld - twcObs);

    const double z = pObs.z();
    if (!std::isfinite(z) || z <= 0.1) {
        return false;
    }

    if (outUv != nullptr) {
        *outUv = uv;
    }
    if (outDepth != nullptr) {
        *outDepth = z;
    }
    return true;
}

float vinsFeatureDepthMedianInRoi(float nx0, float ny0, float nx1, float ny1,
                                  int* outSampleCount) {
    if (outSampleCount != nullptr) {
        *outSampleCount = 0;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);

    if (g_estimator.solver_flag != Estimator::NON_LINEAR) {
        return 0.0f;
    }
    if (g_vinsImageW <= 0 || g_vinsImageH <= 0) {
        return 0.0f;
    }

    // 归一化 ROI -> VINS 输入图像像素。范围统一成 min/max，允许调用方传任意方向。
    const float lx = std::min(nx0, nx1);
    const float ly = std::min(ny0, ny1);
    const float hx = std::max(nx0, nx1);
    const float hy = std::max(ny0, ny1);
    const float px0 = lx * static_cast<float>(g_vinsImageW);
    const float py0 = ly * static_cast<float>(g_vinsImageH);
    const float px1 = hx * static_cast<float>(g_vinsImageW);
    const float py1 = hy * static_cast<float>(g_vinsImageH);

    std::vector<double> depths;
    depths.reserve(g_estimator.f_manager.feature.size());
    for (const auto &it_per_id : g_estimator.f_manager.feature) {
        if (it_per_id.solve_flag != 1) {
            continue;
        }
        // V0.5：estimated_depth 在 anchor camera 系，必须先投到观测帧再取 z。
        Eigen::Vector2d uv;
        double d = 0.0;
        if (!vinsAnchorDepthPair(it_per_id, &uv, &d)) {
            continue;
        }
        // 用最近一次观测的像素位置判断它是否落在目标 ROI 内。
        // uv 是 VINS 输入图像像素（feature_tracker 里 cur_pts 直接写入），
        // 不是归一化坐标 —— 这也是必须保存 g_vinsImageW/H 的原因。
        if (uv.x() < px0 || uv.x() > px1 || uv.y() < py0 || uv.y() > py1) {
            continue;
        }
        depths.push_back(d);
    }

    if (depths.empty()) {
        return 0.0f;
    }

    const size_t n = depths.size();
    std::nth_element(depths.begin(), depths.begin() + n / 2, depths.end());
    if (outSampleCount != nullptr) {
        *outSampleCount = static_cast<int>(n);
    }
    return static_cast<float>(depths[n / 2]);
}

int vinsFeatureSamples(float* out, int maxSamples) {
    if (out == nullptr || maxSamples <= 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);

    if (g_estimator.solver_flag != Estimator::NON_LINEAR) {
        return 0;
    }
    if (g_vinsImageW <= 0 || g_vinsImageH <= 0) {
        return 0;
    }

    const float invW = 1.0f / static_cast<float>(g_vinsImageW);
    const float invH = 1.0f / static_cast<float>(g_vinsImageH);

    int n = 0;
    for (const auto &it_per_id : g_estimator.f_manager.feature) {
        if (n >= maxSamples) {
            break;
        }
        if (it_per_id.solve_flag != 1) {
            continue;
        }
        // V0.5：estimated_depth 在 anchor camera 系，必须先投到观测帧再取 z。
        // 否则 vinsFeatureSamples() 交给深度标定器的 (uv, z) 对是错配的。
        Eigen::Vector2d uv;
        double d = 0.0;
        if (!vinsAnchorDepthPair(it_per_id, &uv, &d)) {
            continue;
        }
        const float nu = static_cast<float>(uv.x()) * invW;
        const float nv = static_cast<float>(uv.y()) * invH;
        if (!(nu >= 0.0f && nu <= 1.0f && nv >= 0.0f && nv <= 1.0f)) {
            continue;
        }
        out[n * 3 + 0] = nu;
        out[n * 3 + 1] = nv;
        out[n * 3 + 2] = static_cast<float>(d);
        ++n;
    }
    return n;
}

int vinsGetCurrentWorldFeatures(
        VinsWorldFeature* out,
        int maxFeatures) {
    if (!out || maxFeatures <= 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);

    if (!g_vinsReady ||
        g_estimator.solver_flag !=
            Estimator::NON_LINEAR) {
        return 0;
    }

    int written = 0;

    for (const auto& feature :
         g_estimator.f_manager.feature) {
        if (written >= maxFeatures) break;

        if (feature.solve_flag != 1 ||
            feature.feature_per_frame.empty() ||
            feature.feature_id < 0) {
            continue;
        }

        const int anchorFrame =
            feature.start_frame;

        const int latestFrame =
            feature.start_frame +
            static_cast<int>(
                feature.feature_per_frame.size()) -
            1;

        // processImage() may have just slid the window. Accept only features
        // seen in the newest one/two slots so the back().uv truly belongs to
        // the gray frame currently being described by ORB.
        if (anchorFrame < 0 ||
            anchorFrame > WINDOW_SIZE ||
            latestFrame != WINDOW_SIZE - 1) {
            continue;
        }

        const double depth =
            feature.estimated_depth;

        if (!std::isfinite(depth) ||
            depth <= 0.08 ||
            depth > 20.0) {
            continue;
        }

        const Eigen::Vector3d pAnchor =
            feature.feature_per_frame.front()
                .point *
            depth;

        const Eigen::Matrix3d RwcAnchor =
            g_estimator.Rs[anchorFrame] *
            g_estimator.ric[0];

        const Eigen::Vector3d twcAnchor =
            g_estimator.Ps[anchorFrame] +
            g_estimator.Rs[anchorFrame] *
                g_estimator.tic[0];

        const Eigen::Vector3d pWorld =
            RwcAnchor * pAnchor +
            twcAnchor;

        const Eigen::Vector2d& uv =
            feature.feature_per_frame.back()
                .uv;

        if (!pWorld.allFinite() ||
            !std::isfinite(uv.x()) ||
            !std::isfinite(uv.y())) {
            continue;
        }

        if (uv.x() < 0.0 ||
            uv.y() < 0.0 ||
            uv.x() >=
                static_cast<double>(
                    g_vinsImageW) ||
            uv.y() >=
                static_cast<double>(
                    g_vinsImageH)) {
            continue;
        }

        VinsWorldFeature& dst =
            out[written++];

        dst.u =
            static_cast<float>(uv.x());
        dst.v =
            static_cast<float>(uv.y());

        dst.x =
            static_cast<float>(
                pWorld.x());
        dst.y =
            static_cast<float>(
                pWorld.y());
        dst.z =
            static_cast<float>(
                pWorld.z());

        dst.featureId =
            feature.feature_id;
    }

    return written;
}

bool vinsGetCameraPoseMatrix(
        float outRwc[9],
        float outTwc[3]) {
    if (!outRwc || !outTwc) {
        return false;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);

    if (!g_vinsReady ||
        g_estimator.solver_flag !=
            Estimator::NON_LINEAR) {
        return false;
    }

    const int i = WINDOW_SIZE;

    const Eigen::Matrix3d Rwc =
        g_estimator.Rs[i] *
        g_estimator.ric[0];

    const Eigen::Vector3d twc =
        g_estimator.Ps[i] +
        g_estimator.Rs[i] *
            g_estimator.tic[0];

    int n = 0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            outRwc[n++] =
                static_cast<float>(
                    Rwc(r,c));
        }
    }

    outTwc[0] =
        static_cast<float>(twc.x());
    outTwc[1] =
        static_cast<float>(twc.y());
    outTwc[2] =
        static_cast<float>(twc.z());

    return true;
}
std::uint64_t vinsLastProcessedImageTimestampNs() {
    std::lock_guard<std::mutex> lk(g_vinsMutex);
    return g_lastEstimatorImageTimestampNs;
}
