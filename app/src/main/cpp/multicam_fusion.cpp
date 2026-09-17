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
#include <opencv2/imgproc.hpp>

namespace {

using mobilescan3d::multicam::kAnchorStride;
using mobilescan3d::multicam::kMaxStereoAnchors;
using mobilescan3d::multicam::kStatsSlots;

constexpr int kEmpiricalMinInliers = 20;
constexpr int kEmpiricalGoodFramesRequired = 3;
constexpr int kEmpiricalBadFramesToDisable = 2;
constexpr double kEmpiricalMaxRotationDeltaDeg = 5.0;
constexpr double kEmpiricalMinTranslationDot = 0.80;

struct StereoAnchor {
    float u = 0.f;
    float v = 0.f;
    float zMetric = 0.f;
    float reprojectionPx = 0.f;
    float confidence = 0.f;
    float parallaxDeg = 0.f;
};

struct State {
    bool configured = false;
    bool calibratedSync = false;
    bool geometryReady = false;
    int width = 0;
    int height = 0;

    cv::Matx33d K1 = cv::Matx33d::eye();
    cv::Matx33d K2 = cv::Matx33d::eye();

    // Empirical relative geometry: X2 = R21 * X1 + t21.
    cv::Matx33d R21 = cv::Matx33d::eye();
    cv::Vec3d t21{0.0, 0.0, 0.0};

    double baselineMeters = 0.0;
    double focalRatio = 1.0;

    std::uint64_t pairCount = 0;
    std::uint64_t goodPairCount = 0;
    std::uint64_t pairsWithAnchors = 0;
    std::uint64_t anchorCandidates = 0;
    std::uint64_t anchorsExported = 0;
    std::uint64_t anchorsRejectedQuality = 0;

    std::uint64_t empiricalAttempts = 0;
    std::uint64_t empiricalAccepted = 0;
    std::uint64_t intrinsicEpoch = 0;
    int empiricalGoodStreak = 0;
    int empiricalBadStreak = 0;
    int lastEmpiricalInliers = 0;
    float lastEmpiricalRotationDeltaDeg = 0.0f;
    float lastEmpiricalTranslationDot = 0.0f;

    float lastMatches = 0.0f;
    float lastInliers = 0.0f;
    float lastTriangulated = 0.0f;
    float lastMedianDepth = 0.0f;
    float lastP10Depth = 0.0f;
    float lastP90Depth = 0.0f;
    float lastDeltaMs = 0.0f;
    float lastProcessingMs = 0.0f;
    float lastMedianReprojectionPx = 0.0f;
    float lastMedianParallaxDeg = 0.0f;
    float lastMedianAnchorConfidence = 0.0f;

    std::vector<StereoAnchor> lastAnchors;
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

double angleDeg(const cv::Vec3d& a, const cv::Vec3d& b) {
    const double na = cv::norm(a);
    const double nb = cv::norm(b);
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    const double d = std::clamp(a.dot(b) / (na * nb), -1.0, 1.0);
    return std::acos(d) * 180.0 / CV_PI;
}

double rotationDeltaDeg(
        const cv::Matx33d& a,
        const cv::Matx33d& b) {
    const cv::Matx33d d = a * b.t();
    const double c = std::clamp(
        (d(0,0) + d(1,1) + d(2,2) - 1.0) * 0.5,
        -1.0,
        1.0
    );
    return std::acos(c) * 180.0 / CV_PI;
}

cv::Mat downsampleForOrb(const cv::Mat& src, double* scaleOut) {
    constexpr int kMaxDim = 800;
    const int maxDim = std::max(src.cols, src.rows);
    if (maxDim <= kMaxDim) {
        if (scaleOut) *scaleOut = 1.0;
        return src;
    }
    const double scale =
        static_cast<double>(kMaxDim) /
        static_cast<double>(maxDim);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), scale, scale, cv::INTER_AREA);
    if (scaleOut) *scaleOut = scale;
    return dst;
}

cv::Point2f normalizedPoint(
        const cv::Matx33d& K,
        const cv::Point2f& p) {
    return cv::Point2f(
        static_cast<float>((p.x - K(0,2)) / K(0,0)),
        static_cast<float>((p.y - K(1,2)) / K(1,1))
    );
}

cv::Matx33d matToMatx33d(const cv::Mat& src) {
    cv::Mat m;
    src.convertTo(m, CV_64F);
    return cv::Matx33d(
        m.at<double>(0,0), m.at<double>(0,1), m.at<double>(0,2),
        m.at<double>(1,0), m.at<double>(1,1), m.at<double>(1,2),
        m.at<double>(2,0), m.at<double>(2,1), m.at<double>(2,2)
    );
}

cv::Vec3d matToVec3d(const cv::Mat& src) {
    cv::Mat m;
    src.convertTo(m, CV_64F);
    cv::Vec3d v(
        m.at<double>(0,0),
        m.at<double>(1,0),
        m.at<double>(2,0)
    );
    const double n = cv::norm(v);
    if (n > 1e-12) v *= (1.0 / n);
    return v;
}

/**
 * Estimate actual relative orientation from delivered synchronized images.
 * Factory translation supplies only baseline magnitude; R and t direction come
 * from the Essential matrix/recoverPose result.
 */
bool updateEmpiricalGeometry(
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2) {
    gState.empiricalAttempts++;
    gState.lastEmpiricalInliers = 0;

    auto reject = [&]() {
        gState.empiricalBadStreak++;
        if (gState.empiricalBadStreak >= kEmpiricalBadFramesToDisable) {
            gState.geometryReady = false;
        }
        return false;
    };

    if (matches.size() < 24) return reject();

    std::vector<cv::Point2f> n1;
    std::vector<cv::Point2f> n2;
    n1.reserve(matches.size());
    n2.reserve(matches.size());

    for (const auto& m : matches) {
        n1.push_back(normalizedPoint(
            gState.K1,
            kp1[static_cast<std::size_t>(m.queryIdx)].pt));
        n2.push_back(normalizedPoint(
            gState.K2,
            kp2[static_cast<std::size_t>(m.trainIdx)].pt));
    }

    const double minFocal = std::max(
        100.0,
        std::min({
            gState.K1(0,0),
            gState.K1(1,1),
            gState.K2(0,0),
            gState.K2(1,1)
        })
    );
    const double normalizedRansacThreshold = 2.0 / minFocal;

    cv::Mat essentialMask;
    const cv::Mat E = cv::findEssentialMat(
        n1,
        n2,
        1.0,
        cv::Point2d(0.0, 0.0),
        cv::RANSAC,
        0.999,
        normalizedRansacThreshold,
        essentialMask
    );
    if (E.empty()) return reject();

    cv::Mat Rcv;
    cv::Mat tcv;
    const int inliers = cv::recoverPose(
        E,
        n1,
        n2,
        Rcv,
        tcv,
        1.0,
        cv::Point2d(0.0, 0.0),
        essentialMask
    );
    gState.lastEmpiricalInliers = inliers;

    if (inliers < kEmpiricalMinInliers ||
        Rcv.rows != 3 || Rcv.cols != 3 ||
        tcv.total() < 3) {
        return reject();
    }

    const cv::Matx33d rNew = matToMatx33d(Rcv);
    cv::Vec3d tUnit = matToVec3d(tcv);
    if (cv::norm(tUnit) < 0.9) return reject();

    double rotDelta = 0.0;
    double translationDot = 1.0;

    if (gState.empiricalAccepted > 0) {
        cv::Vec3d oldUnit = gState.t21;
        const double oldNorm = cv::norm(oldUnit);
        if (oldNorm > 1e-12) oldUnit *= (1.0 / oldNorm);

        translationDot = oldUnit.dot(tUnit);
        if (translationDot < 0.0) {
            tUnit *= -1.0;
            translationDot = -translationDot;
        }
        rotDelta = rotationDeltaDeg(rNew, gState.R21);
    }

    gState.lastEmpiricalRotationDeltaDeg =
        static_cast<float>(rotDelta);
    gState.lastEmpiricalTranslationDot =
        static_cast<float>(translationDot);

    const bool consistent =
        gState.empiricalAccepted == 0 ||
        (rotDelta <= kEmpiricalMaxRotationDeltaDeg &&
         translationDot >= kEmpiricalMinTranslationDot);

    if (!consistent) {
        gState.empiricalGoodStreak = 0;
        gState.empiricalBadStreak++;

        if (gState.empiricalBadStreak >= kEmpiricalBadFramesToDisable) {
            // Start a new calibration epoch from this image pair.
            gState.geometryReady = false;
            gState.R21 = rNew;
            gState.t21 = tUnit * gState.baselineMeters;
            gState.empiricalAccepted++;
            gState.empiricalGoodStreak = 1;
            gState.empiricalBadStreak = 0;
        }
        return false;
    }

    gState.empiricalAccepted++;
    gState.empiricalBadStreak = 0;
    gState.empiricalGoodStreak++;
    gState.R21 = rNew;
    gState.t21 = tUnit * gState.baselineMeters;

    if (gState.empiricalGoodStreak >= kEmpiricalGoodFramesRequired) {
        gState.geometryReady = true;
    }
    return gState.geometryReady;
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

    std::vector<float> k1, k2, q1, c1, q2, c2;
    if (!readFloats(env, primaryKArray, 4, &k1) ||
        !readFloats(env, secondaryKArray, 4, &k2) ||
        !readFloats(env, primaryQuatArray, 4, &q1) ||
        !readFloats(env, primaryTranslationArray, 3, &c1) ||
        !readFloats(env, secondaryQuatArray, 4, &q2) ||
        !readFloats(env, secondaryTranslationArray, 3, &c2) ||
        width <= 0 || height <= 0) {
        return JNI_FALSE;
    }

    (void)q1;
    (void)q2;

    const cv::Vec3d C1(c1[0], c1[1], c1[2]);
    const cv::Vec3d C2(c2[0], c2[1], c2[2]);
    const double baseline = cv::norm(C2 - C1);

    // PLK110 main->ultrawide is ~19.2mm. The reported main->tele ~2.1mm
    // is rejected for geometry rather than lowering the gate.
    if (!std::isfinite(baseline) ||
        baseline < 0.004 ||
        baseline > 0.080) {
        return JNI_FALSE;
    }

    if (!(k1[0] > 100.f) || !(k1[1] > 100.f) ||
        !(k2[0] > 100.f) || !(k2[1] > 100.f)) {
        return JNI_FALSE;
    }

    State next;
    next.configured = true;
    next.calibratedSync = calibratedSync == JNI_TRUE;
    next.geometryReady = false;
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
    next.baselineMeters = baseline;
    next.focalRatio = k1[0] > 1e-6 ? k2[0] / k1[0] : 1.0;

    std::lock_guard<std::mutex> lock(gMutex);
    gState = std::move(next);
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

    if (!primaryGrayArray || !secondaryGrayArray ||
        width <= 0 || height <= 0) {
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
        primaryGrayArray, 0, expected,
        reinterpret_cast<jbyte*>(g1.data()));
    env->GetByteArrayRegion(
        secondaryGrayArray, 0, expected,
        reinterpret_cast<jbyte*>(g2.data()));
    if (env->ExceptionCheck()) return JNI_FALSE;

    const auto start = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(gMutex);
    if (!gState.configured ||
        gState.width != width ||
        gState.height != height) {
        return JNI_FALSE;
    }

    auto finish = [&](bool ok) -> jboolean {
        const auto end = std::chrono::steady_clock::now();
        gState.lastProcessingMs = static_cast<float>(
            std::chrono::duration<double, std::milli>(end - start).count()
        );
        return ok ? JNI_TRUE : JNI_FALSE;
    };

    gState.lastAnchors.clear();

    cv::Mat full1(height, width, CV_8UC1, g1.data());
    cv::Mat full2(height, width, CV_8UC1, g2.data());

    double orbScale1 = 1.0;
    double orbScale2 = 1.0;
    cv::Mat im1 = downsampleForOrb(full1, &orbScale1);
    cv::Mat im2 = downsampleForOrb(full2, &orbScale2);
    const double invScale1 = 1.0 / std::max(1e-9, orbScale1);
    const double invScale2 = 1.0 / std::max(1e-9, orbScale2);

    auto orb = cv::ORB::create(
        1400, 1.20f, 8, 19, 0, 2,
        cv::ORB::HARRIS_SCORE, 31, 12
    );

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat d1, d2;
    orb->detectAndCompute(im1, cv::noArray(), kp1, d1);
    orb->detectAndCompute(im2, cv::noArray(), kp2, d2);

    if (orbScale1 != 1.0) {
        for (auto& k : kp1) {
            k.pt.x = static_cast<float>(k.pt.x * invScale1);
            k.pt.y = static_cast<float>(k.pt.y * invScale1);
        }
    }
    if (orbScale2 != 1.0) {
        for (auto& k : kp2) {
            k.pt.x = static_cast<float>(k.pt.x * invScale2);
            k.pt.y = static_cast<float>(k.pt.y * invScale2);
        }
    }

    gState.pairCount++;
    gState.lastDeltaMs = static_cast<float>(
        std::abs(
            static_cast<double>(primaryTimestampNs) -
            static_cast<double>(secondaryTimestampNs)
        ) / 1.0e6
    );

    if (d1.empty() || d2.empty() ||
        kp1.size() < 20 || kp2.size() < 20) {
        gState.lastMatches = 0.0f;
        gState.lastInliers = 0.0f;
        gState.lastTriangulated = 0.0f;
        return finish(false);
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(d1, d2, knn, 2);

    std::vector<cv::DMatch> ratioMatches;
    ratioMatches.reserve(knn.size());
    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < 0.80f * pair[1].distance) {
            ratioMatches.push_back(pair[0]);
        }
    }
    gState.lastMatches = static_cast<float>(ratioMatches.size());

    updateEmpiricalGeometry(ratioMatches, kp1, kp2);
    if (!gState.geometryReady) {
        gState.lastInliers = 0.0f;
        gState.lastTriangulated = 0.0f;
        return finish(false);
    }

    const cv::Matx33d F =
        gState.K2.inv().t() *
        skew(gState.t21) *
        gState.R21 *
        gState.K1.inv();

    std::vector<cv::Point2f> p1, p2;
    p1.reserve(ratioMatches.size());
    p2.reserve(ratioMatches.size());

    for (const auto& m : ratioMatches) {
        const cv::Point2f a =
            kp1[static_cast<std::size_t>(m.queryIdx)].pt;
        const cv::Point2f b =
            kp2[static_cast<std::size_t>(m.trainIdx)].pt;
        const double e = epipolarDistancePx(F, a, b);
        if (std::isfinite(e) && e <= 2.5) {
            p1.push_back(a);
            p2.push_back(b);
        }
    }
    gState.lastInliers = static_cast<float>(p1.size());

    if (p1.size() < 10) {
        gState.lastTriangulated = 0.0f;
        return finish(false);
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

    std::vector<float> depths, reprojection, parallaxes, confidences;
    depths.reserve(p1.size());
    reprojection.reserve(p1.size());
    parallaxes.reserve(p1.size());
    confidences.reserve(p1.size());

    std::vector<StereoAnchor> anchors;
    anchors.reserve(std::min<std::size_t>(p1.size(), kMaxStereoAnchors));

    const double syncScore = gState.calibratedSync
        ? 1.0
        : std::clamp(
            1.0 - static_cast<double>(gState.lastDeltaMs) / 12.0,
            0.20,
            1.0);

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
        if (X[2] <= 0.12 ||
            X[2] >= 12.0 ||
            X2[2] <= 0.12) {
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
        const double reproj = 0.5 * (e1 + e2);
        if (!std::isfinite(reproj) || reproj > 3.0) continue;

        const cv::Vec3d ray1 = X;
        const cv::Vec3d ray2Primary = gState.R21.t() * X2;
        const double parallax = angleDeg(ray1, ray2Primary);
        if (!std::isfinite(parallax) || parallax < 0.20) continue;

        depths.push_back(static_cast<float>(X[2]));
        reprojection.push_back(static_cast<float>(reproj));
        parallaxes.push_back(static_cast<float>(parallax));

        const double reprojScore = std::exp(-reproj / 1.25);
        const double parallaxScore =
            std::clamp((parallax - 0.20) / 1.50, 0.0, 1.0);
        const float anchorConfidence = static_cast<float>(
            std::clamp(
                reprojScore * parallaxScore * syncScore,
                0.0,
                1.0)
        );
        confidences.push_back(anchorConfidence);
        gState.anchorCandidates++;

        if (reproj > 2.5 ||
            parallax < 0.30 ||
            anchorConfidence < 0.30) {
            gState.anchorsRejectedQuality++;
            continue;
        }

        StereoAnchor a;
        a.u = std::clamp(
            p1[static_cast<std::size_t>(i)].x /
                static_cast<float>(std::max(1, width - 1)),
            0.f,
            1.f);
        a.v = std::clamp(
            p1[static_cast<std::size_t>(i)].y /
                static_cast<float>(std::max(1, height - 1)),
            0.f,
            1.f);
        a.zMetric = static_cast<float>(X[2]);
        a.reprojectionPx = static_cast<float>(reproj);
        a.confidence = anchorConfidence;
        a.parallaxDeg = static_cast<float>(parallax);
        anchors.push_back(a);
    }

    gState.lastTriangulated = static_cast<float>(depths.size());
    gState.lastMedianDepth = percentile(depths, 0.50);
    gState.lastP10Depth = percentile(depths, 0.10);
    gState.lastP90Depth = percentile(depths, 0.90);
    gState.lastMedianReprojectionPx = percentile(reprojection, 0.50);
    gState.lastMedianParallaxDeg = percentile(parallaxes, 0.50);
    gState.lastMedianAnchorConfidence = percentile(confidences, 0.50);

    if (depths.size() >= 12 &&
        gState.lastMedianReprojectionPx <= 2.0f) {
        gState.goodPairCount++;
    }

    std::sort(
        anchors.begin(),
        anchors.end(),
        [](const StereoAnchor& a, const StereoAnchor& b) {
            if (a.confidence != b.confidence) {
                return a.confidence > b.confidence;
            }
            return a.reprojectionPx < b.reprojectionPx;
        });
    if (anchors.size() > static_cast<std::size_t>(kMaxStereoAnchors)) {
        anchors.resize(kMaxStereoAnchors);
    }

    gState.lastAnchors = std::move(anchors);
    gState.anchorsExported += gState.lastAnchors.size();
    if (!gState.lastAnchors.empty()) {
        gState.pairsWithAnchors++;
    }

    return finish(gState.lastAnchors.size() >= 6);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeMultiCamUpdateIntrinsics(
        JNIEnv* env,
        jobject,
        jfloatArray primaryKArray,
        jfloatArray secondaryKArray) {
    std::vector<float> k1;
    std::vector<float> k2;
    if (!readFloats(env, primaryKArray, 4, &k1) ||
        !readFloats(env, secondaryKArray, 4, &k2)) {
        return JNI_FALSE;
    }

    if (!(k1[0] > 100.f) || !(k1[1] > 100.f) ||
        !(k2[0] > 100.f) || !(k2[1] > 100.f)) {
        return JNI_FALSE;
    }

    std::lock_guard<std::mutex> lock(gMutex);
    if (!gState.configured) return JNI_FALSE;

    gState.K1 = cv::Matx33d(
        k1[0], 0.0, k1[2],
        0.0, k1[1], k1[3],
        0.0, 0.0, 1.0
    );
    gState.K2 = cv::Matx33d(
        k2[0], 0.0, k2[2],
        0.0, k2[1], k2[3],
        0.0, 0.0, 1.0
    );
    gState.focalRatio =
        k1[0] > 1e-6 ? k2[0] / k1[0] : 1.0;

    // New intrinsics = new calibration epoch.
    gState.geometryReady = false;
    gState.empiricalAccepted = 0;
    gState.empiricalGoodStreak = 0;
    gState.empiricalBadStreak = 0;
    gState.lastEmpiricalInliers = 0;
    gState.lastEmpiricalRotationDeltaDeg = 0.f;
    gState.lastEmpiricalTranslationDot = 0.f;
    gState.lastAnchors.clear();
    gState.intrinsicEpoch++;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeMultiCamGetAnchors(
        JNIEnv* env,
        jobject,
        jfloatArray outArray) {
    if (!outArray) return 0;

    std::lock_guard<std::mutex> lock(gMutex);
    const int cap = env->GetArrayLength(outArray) / kAnchorStride;
    const int count = std::min(
        cap,
        static_cast<int>(gState.lastAnchors.size()));
    if (count <= 0) return 0;

    std::vector<jfloat> out(
        static_cast<std::size_t>(count) * kAnchorStride,
        0.f);

    for (int i = 0; i < count; ++i) {
        const StereoAnchor& a =
            gState.lastAnchors[static_cast<std::size_t>(i)];
        const int o = i * kAnchorStride;
        out[static_cast<std::size_t>(o + 0)] = a.u;
        out[static_cast<std::size_t>(o + 1)] = a.v;
        out[static_cast<std::size_t>(o + 2)] = a.zMetric;
        out[static_cast<std::size_t>(o + 3)] = a.reprojectionPx;
        out[static_cast<std::size_t>(o + 4)] = a.confidence;
        out[static_cast<std::size_t>(o + 5)] = a.parallaxDeg;
    }

    env->SetFloatArrayRegion(
        outArray,
        0,
        count * kAnchorStride,
        out.data());
    return env->ExceptionCheck() ? 0 : count;
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
        out[13] = 0.0f;
        out[14] = gState.lastMedianReprojectionPx;
        out[15] = gState.calibratedSync ? 1.0f : 0.0f;
        out[16] = static_cast<float>(gState.anchorCandidates);
        out[17] = static_cast<float>(gState.lastAnchors.size());
        out[18] = gState.lastMedianParallaxDeg;
        out[19] = gState.lastMedianAnchorConfidence;
        out[20] = static_cast<float>(gState.pairsWithAnchors);
        out[21] = static_cast<float>(gState.anchorsExported);
        out[22] = static_cast<float>(gState.anchorsRejectedQuality);
        out[23] = static_cast<float>(gState.lastAnchors.size() >= 6 ? 1 : 0);
        out[24] = gState.geometryReady ? 1.0f : 0.0f;
        out[25] = static_cast<float>(gState.empiricalAttempts);
        out[26] = static_cast<float>(gState.empiricalAccepted);
        out[27] = static_cast<float>(gState.empiricalGoodStreak);
        out[28] = static_cast<float>(gState.empiricalBadStreak);
        out[29] = static_cast<float>(gState.lastEmpiricalInliers);
        out[30] = gState.lastEmpiricalRotationDeltaDeg;
        out[31] = gState.lastEmpiricalTranslationDot;
        out[32] = static_cast<float>(gState.intrinsicEpoch);
        out[33] = static_cast<float>(gState.K2(0,0));
    }

    env->SetFloatArrayRegion(outArray, 0, kStatsSlots, out.data());
    return env->ExceptionCheck() ? JNI_FALSE : JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeMultiCamReset(
        JNIEnv*,
        jobject) {
    std::lock_guard<std::mutex> lock(gMutex);
    gState = State{};
}
