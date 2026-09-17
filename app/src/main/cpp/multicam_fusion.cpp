#include "multicam_fusion.h"

#include <jni.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace {

using mobilescan3d::multicam::kStatsSlots;

struct State {
    bool configured = false;
    bool calibratedSync = false;
    int width = 0;
    int height = 0;

    cv::Matx33d K1 = cv::Matx33d::eye();
    cv::Matx33d K2 = cv::Matx33d::eye();
    cv::Matx33d R21 = cv::Matx33d::eye();  // primary coords -> secondary coords
    cv::Vec3d t21{0.0, 0.0, 0.0};

    double baselineMeters = 0.0;
    double focalRatio = 1.0;

    std::uint64_t pairCount = 0;
    std::uint64_t goodPairCount = 0;

    float lastMatches = 0.0f;
    float lastInliers = 0.0f;
    float lastTriangulated = 0.0f;
    float lastMedianDepth = 0.0f;
    float lastP10Depth = 0.0f;
    float lastP90Depth = 0.0f;
    float lastDeltaMs = 0.0f;
    float lastProcessingMs = 0.0f;
    float lastMedianReprojectionPx = 0.0f;
};

std::mutex gMutex;
State gState;

bool readFloats(
        JNIEnv* env,
        jfloatArray arr,
        int count,
        std::vector<float>* out) {
    if (!arr || !out || env->GetArrayLength(arr) < count) return false;
    out->assign(static_cast<std::size_t>(count), 0.0f);
    env->GetFloatArrayRegion(arr, 0, count, out->data());
    return !env->ExceptionCheck();
}

cv::Matx33d quaternionToR(const std::vector<float>& q) {
    double x = q[0];
    double y = q[1];
    double z = q[2];
    double w = q[3];

    const double n = std::sqrt(x*x + y*y + z*z + w*w);
    if (n > 1e-12) {
        x /= n;
        y /= n;
        z /= n;
        w /= n;
    }

    return cv::Matx33d(
        1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w),
        2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w),
        2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)
    );
}

cv::Matx33d skew(const cv::Vec3d& t) {
    return cv::Matx33d(
         0.0, -t[2],  t[1],
         t[2], 0.0,  -t[0],
        -t[1], t[0],  0.0
    );
}

float percentile(std::vector<float> values, double q) {
    if (values.empty()) return 0.0f;
    q = std::clamp(q, 0.0, 1.0);

    const std::size_t i = static_cast<std::size_t>(
        std::llround(q * static_cast<double>(values.size() - 1))
    );
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(i),
        values.end()
    );
    return values[i];
}

double epipolarDistancePx(
        const cv::Matx33d& F,
        const cv::Point2f& p1,
        const cv::Point2f& p2) {
    const cv::Vec3d x1(p1.x, p1.y, 1.0);
    const cv::Vec3d x2(p2.x, p2.y, 1.0);
    const cv::Vec3d l2 = F * x1;

    const double denom = std::sqrt(l2[0]*l2[0] + l2[1]*l2[1]);
    if (denom < 1e-12) return 1e9;
    return std::abs(x2.dot(l2)) / denom;
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeMultiCamConfigure(
        JNIEnv* env,
        jobject,
        jfloatArray primaryKArray,
        jfloatArray secondaryKArray,
        jfloatArray primaryQuatArray,
        jfloatArray primaryTranslationArray,
        jfloatArray secondaryQuatArray,
        jfloatArray secondaryTranslationArray,
        jint width,
        jint height,
        jboolean calibratedSync) {

    std::vector<float> k1;
    std::vector<float> k2;
    std::vector<float> q1;
    std::vector<float> c1;
    std::vector<float> q2;
    std::vector<float> c2;

    if (!readFloats(env, primaryKArray, 4, &k1) ||
        !readFloats(env, secondaryKArray, 4, &k2) ||
        !readFloats(env, primaryQuatArray, 4, &q1) ||
        !readFloats(env, primaryTranslationArray, 3, &c1) ||
        !readFloats(env, secondaryQuatArray, 4, &q2) ||
        !readFloats(env, secondaryTranslationArray, 3, &c2) ||
        width <= 0 ||
        height <= 0) {
        return JNI_FALSE;
    }

    const cv::Matx33d R1 = quaternionToR(q1);
    const cv::Matx33d R2 = quaternionToR(q2);
    const cv::Vec3d C1(c1[0], c1[1], c1[2]);
    const cv::Vec3d C2(c2[0], c2[1], c2[2]);

    // Camera2:
    // x1 = R1 * (X - C1)
    // x2 = R2 * (X - C2)
    // => x2 = R2*R1^T*x1 + R2*(C1-C2)
    const cv::Matx33d R21 = R2 * R1.t();
    const cv::Vec3d t21 = R2 * (C1 - C2);
    const double baseline = cv::norm(C1 - C2);

    if (!std::isfinite(baseline) || baseline < 0.003 || baseline > 0.30) {
        return JNI_FALSE;
    }

    State next;
    next.configured = true;
    next.calibratedSync = calibratedSync == JNI_TRUE;
    next.width = width;
    next.height = height;

    next.K1 = cv::Matx33d(
        k1[0], 0.0, k1[2],
        0.0, k1[1], k1[3],
        0.0, 0.0, 1.0
    );
    next.K2 = cv::Matx33d(
        k2[0], 0.0, k2[2],
        0.0, k2[1], k2[3],
        0.0, 0.0, 1.0
    );
    next.R21 = R21;
    next.t21 = t21;
    next.baselineMeters = baseline;
    next.focalRatio = k1[0] > 1e-6 ? k2[0] / k1[0] : 1.0;

    std::lock_guard<std::mutex> lock(gMutex);
    gState = next;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeMultiCamOnPair(
        JNIEnv* env,
        jobject,
        jbyteArray primaryGrayArray,
        jbyteArray secondaryGrayArray,
        jint width,
        jint height,
        jlong primaryTimestampNs,
        jlong secondaryTimestampNs) {

    if (!primaryGrayArray ||
        !secondaryGrayArray ||
        width <= 0 ||
        height <= 0) {
        return JNI_FALSE;
    }

    const jsize expected = width * height;
    if (env->GetArrayLength(primaryGrayArray) < expected ||
        env->GetArrayLength(secondaryGrayArray) < expected) {
        return JNI_FALSE;
    }

    std::vector<std::uint8_t> g1(static_cast<std::size_t>(expected));
    std::vector<std::uint8_t> g2(static_cast<std::size_t>(expected));

    env->GetByteArrayRegion(
        primaryGrayArray,
        0,
        expected,
        reinterpret_cast<jbyte*>(g1.data())
    );
    env->GetByteArrayRegion(
        secondaryGrayArray,
        0,
        expected,
        reinterpret_cast<jbyte*>(g2.data())
    );
    if (env->ExceptionCheck()) return JNI_FALSE;

    const auto start = std::chrono::steady_clock::now();

    // Pair processing is intentionally serialized by the Kotlin fusion thread.
    // Holding this lock keeps reconfigure/reset simple and makes stats coherent.
    std::lock_guard<std::mutex> lock(gMutex);
    if (!gState.configured ||
        gState.width != width ||
        gState.height != height) {
        return JNI_FALSE;
    }

    cv::Mat im1(height, width, CV_8UC1, g1.data());
    cv::Mat im2(height, width, CV_8UC1, g2.data());

    // Eight levels cover the large main -> ~3.5x tele scale change.
    auto orb = cv::ORB::create(
        1400,
        1.20f,
        8,
        19,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        12
    );

    std::vector<cv::KeyPoint> kp1;
    std::vector<cv::KeyPoint> kp2;
    cv::Mat d1;
    cv::Mat d2;
    orb->detectAndCompute(im1, cv::noArray(), kp1, d1);
    orb->detectAndCompute(im2, cv::noArray(), kp2, d2);

    gState.pairCount++;
    gState.lastDeltaMs = static_cast<float>(
        std::abs(
            static_cast<double>(primaryTimestampNs) -
            static_cast<double>(secondaryTimestampNs)
        ) / 1.0e6
    );

    if (d1.empty() || d2.empty() || kp1.size() < 20 || kp2.size() < 20) {
        gState.lastMatches = 0.0f;
        gState.lastInliers = 0.0f;
        gState.lastTriangulated = 0.0f;
        return JNI_FALSE;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    std::vector<cv::DMatch> ratioMatches;
    ratioMatches.reserve(knn.size());
    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < 0.78f * pair[1].distance) {
            ratioMatches.push_back(pair[0]);
        }
    }
    gState.lastMatches = static_cast<float>(ratioMatches.size());

    if (ratioMatches.size() < 12) {
        gState.lastInliers = 0.0f;
        gState.lastTriangulated = 0.0f;
        return JNI_FALSE;
    }

    const cv::Matx33d F =
        gState.K2.inv().t() *
        skew(gState.t21) *
        gState.R21 *
        gState.K1.inv();

    std::vector<cv::Point2f> p1;
    std::vector<cv::Point2f> p2;
    p1.reserve(ratioMatches.size());
    p2.reserve(ratioMatches.size());

    for (const auto& m : ratioMatches) {
        const cv::Point2f a =
            kp1[static_cast<std::size_t>(m.queryIdx)].pt;
        const cv::Point2f b =
            kp2[static_cast<std::size_t>(m.trainIdx)].pt;

        const double e = epipolarDistancePx(F, a, b);
        if (std::isfinite(e) && e <= 3.0) {
            p1.push_back(a);
            p2.push_back(b);
        }
    }
    gState.lastInliers = static_cast<float>(p1.size());

    if (p1.size() < 10) {
        gState.lastTriangulated = 0.0f;
        return JNI_FALSE;
    }

    const cv::Matx34d E1(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
    );

    const cv::Matx34d E2(
        gState.R21(0,0), gState.R21(0,1), gState.R21(0,2), gState.t21[0],
        gState.R21(1,0), gState.R21(1,1), gState.R21(1,2), gState.t21[1],
        gState.R21(2,0), gState.R21(2,1), gState.R21(2,2), gState.t21[2]
    );

    const cv::Matx34d P1 = gState.K1 * E1;
    const cv::Matx34d P2 = gState.K2 * E2;

    cv::Mat points4;
    cv::triangulatePoints(cv::Mat(P1), cv::Mat(P2), p1, p2, points4);

    cv::Mat points64;
    points4.convertTo(points64, CV_64F);

    std::vector<float> depths;
    std::vector<float> reprojection;
    depths.reserve(p1.size());
    reprojection.reserve(p1.size());

    for (int i = 0; i < points64.cols; ++i) {
        const double w = points64.at<double>(3, i);
        if (!std::isfinite(w) || std::abs(w) < 1e-10) continue;

        const cv::Vec3d X(
            points64.at<double>(0, i) / w,
            points64.at<double>(1, i) / w,
            points64.at<double>(2, i) / w
        );

        if (!std::isfinite(X[0]) ||
            !std::isfinite(X[1]) ||
            !std::isfinite(X[2])) {
            continue;
        }

        const cv::Vec3d X2 = gState.R21 * X + gState.t21;
        if (X[2] <= 0.15 ||
            X[2] >= 12.0 ||
            X2[2] <= 0.15) {
            continue;
        }

        const double u1 =
            gState.K1(0,0) * X[0] / X[2] + gState.K1(0,2);
        const double v1 =
            gState.K1(1,1) * X[1] / X[2] + gState.K1(1,2);
        const double u2 =
            gState.K2(0,0) * X2[0] / X2[2] + gState.K2(0,2);
        const double v2 =
            gState.K2(1,1) * X2[1] / X2[2] + gState.K2(1,2);

        const double e1 = std::hypot(
            u1 - p1[static_cast<std::size_t>(i)].x,
            v1 - p1[static_cast<std::size_t>(i)].y
        );
        const double e2 = std::hypot(
            u2 - p2[static_cast<std::size_t>(i)].x,
            v2 - p2[static_cast<std::size_t>(i)].y
        );
        const double e = 0.5 * (e1 + e2);

        if (!std::isfinite(e) || e > 3.0) continue;

        depths.push_back(static_cast<float>(X[2]));
        reprojection.push_back(static_cast<float>(e));
    }

    gState.lastTriangulated = static_cast<float>(depths.size());
    gState.lastMedianDepth = percentile(depths, 0.50);
    gState.lastP10Depth = percentile(depths, 0.10);
    gState.lastP90Depth = percentile(depths, 0.90);
    gState.lastMedianReprojectionPx = percentile(reprojection, 0.50);

    if (depths.size() >= 12 &&
        gState.lastMedianReprojectionPx <= 2.0f) {
        gState.goodPairCount++;
    }

    const auto end = std::chrono::steady_clock::now();
    gState.lastProcessingMs = static_cast<float>(
        std::chrono::duration<double, std::milli>(end - start).count()
    );

    return depths.size() >= 10 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMultiCamStats(
        JNIEnv* env,
        jobject,
        jfloatArray outArray) {

    if (!outArray ||
        env->GetArrayLength(outArray) < kStatsSlots) {
        return JNI_FALSE;
    }

    std::array<jfloat, kStatsSlots> out{};
    {
        std::lock_guard<std::mutex> lock(gMutex);
        out[0] = gState.configured ? 1.0f : 0.0f;
        out[1] = static_cast<float>(gState.pairCount);
        out[2] = gState.lastMatches;
        out[3] = gState.lastInliers;
        out[4] = gState.lastTriangulated;
        out[5] = static_cast<float>(gState.baselineMeters * 1000.0);
        out[6] = gState.lastMedianDepth;
        out[7] = gState.lastP10Depth;
        out[8] = gState.lastP90Depth;
        out[9] = gState.lastDeltaMs;
        out[10] = static_cast<float>(gState.focalRatio);
        out[11] = gState.lastProcessingMs;
        out[12] = static_cast<float>(gState.goodPairCount);
        out[13] = 0.0f; // Kotlin owns timestamp-pair rejection count.
        out[14] = gState.lastMedianReprojectionPx;
        out[15] = gState.calibratedSync ? 1.0f : 0.0f;
    }

    env->SetFloatArrayRegion(
        outArray,
        0,
        kStatsSlots,
        out.data()
    );
    return env->ExceptionCheck() ? JNI_FALSE : JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeMultiCamReset(
        JNIEnv*,
        jobject) {
    std::lock_guard<std::mutex> lock(gMutex);
    gState = State{};
}
