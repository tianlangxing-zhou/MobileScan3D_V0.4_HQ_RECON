#include <jni.h>

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
// IMU（传感器线程）与图像（相机线程）并发进入 estimator，
// VINS-Mono 原版靠带锁缓冲队列串行化，移植时被删掉了，这里用互斥锁补上。
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
    setFeatureTrackerParams(h, w, (int)fx, 0, 1, 30, 150, 1.0);
    setEstimatorParams(accN, accW, gyrN, gyrW, 0.0, 10.0 / 460.0, 8, 0.04);

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
    setFeatureTrackerParams(h, w, (int)fx, 0, 1, 30, 150, 1.0);
    setEstimatorParams(accN, accW, gyrN, gyrW, 0.0, 10.0 / 460.0, 8, 0.04);

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
    g_tracker[0].setCamera(g_camera);
    // 重复初始化（如重开相机/二次扫描）时清掉滑窗与预积分的残留状态，
    // 否则旧内参/旧时刻的滑窗会与新会话混跑。clearState 后必须重新 setParameter。
    g_estimator.clearState();
    g_estimator.setParameter();
    g_lastImuTimestamp = -1.0;
    g_vinsReady = true;
}

void vinsInputImu(double timestamp, double ax, double ay, double az, double gx, double gy, double gz) {
    if (!g_vinsReady) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_vinsMutex);
    Eigen::Vector3d acc(ax, ay, az);
    Eigen::Vector3d gyr(gx, gy, gz);

    if (g_lastImuTimestamp < 0.0) {
        g_lastImuTimestamp = timestamp;
        g_estimator.processIMU(0.0, acc, gyr);
        return;
    }

    const double dt = timestamp - g_lastImuTimestamp;
    g_lastImuTimestamp = timestamp;

    if (dt <= 0.0) {
        return;
    }

    if (dt > 0.1) {
        return;
    }

    g_estimator.processIMU(dt, acc, gyr);
}

void vinsInputImage(double t, const std::uint8_t* gray, int w, int h, int stride) {
    if (!g_vinsReady || gray == nullptr) {
        return;
    }
    // VINS-Mono 原版由 feature_tracker 节点按发布频率设置该标志；
    // 移植时这段逻辑丢失，导致 goodFeaturesToTrack 永远不执行、
    // 特征集恒为空、初始化永远失败。这里按 10Hz 重新补上门控。
    static double lastPubT = -1.0;
    const double pubPeriod = 1.0 / 10.0;
    PUB_THIS_FRAME = (lastPubT < 0.0) || (t - lastPubT >= pubPeriod);
    if (PUB_THIS_FRAME) {
        lastPubT = t;
    }

    std::lock_guard<std::mutex> lk(g_vinsMutex);
    cv::Mat img(h, w, CV_8UC1, const_cast<std::uint8_t*>(gray), static_cast<size_t>(stride));
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
