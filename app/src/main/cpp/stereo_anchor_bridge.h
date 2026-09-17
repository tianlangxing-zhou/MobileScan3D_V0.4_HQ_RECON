#pragma once

#include "depth_calib.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mobilescan3d::stereo_anchor {

/**
 * Kotlin/native wire format for one stereo metric anchor.
 *
 *  0 uNorm       primary-image normalized x [0,1]
 *  1 vNorm       primary-image normalized y [0,1]
 *  2 zMetricM    triangulated depth in physical meters
 *  3 reprojPx    mean two-view reprojection error in pixels
 *  4 confidence  stereo geometry confidence [0,1]
 *  5 parallaxDeg triangulation angle in degrees
 */
constexpr int kAnchorStride = 6;
constexpr int kMaxAnchorsPerBatch = 96;

/**
 * NativeBridge.nativeGetStereoAnchorStats() float slots.
 * Keep in sync with NativeBridge.STEREO_ANCHOR_STATS_SLOTS.
 */
constexpr int kStatsSlots = 48;

struct Anchor {
    float u = 0.f;
    float v = 0.f;
    float zMetric = 0.f;
    float reprojectionPx = 0.f;
    float confidence = 0.f;
    float parallaxDeg = 0.f;
};

struct AnchorBatch {
    std::uint64_t timestampNs = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<Anchor> anchors;
};

struct CalibrationBatch {
    std::uint64_t timestampNs = 0;
    std::int64_t matchDeltaNs = -1;
    std::vector<float> rawDepth;
    std::vector<float> metricDepth;
    std::vector<float> quality;
};

struct WorldAnchor {
    float u = 0.f;
    float v = 0.f;
    /** Depth expressed in the same world-scale units as VINS/TSDF. */
    float zWorld = 0.f;
    /** Suggested TSDF observation weight multiplier, normally [1,4]. */
    float weight = 1.f;

    float zMetric = 0.f;
    float reprojectionPx = 0.f;
    float confidence = 0.f;
    float parallaxDeg = 0.f;
};

struct WorldAnchorBatch {
    std::uint64_t timestampNs = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    std::vector<WorldAnchor> anchors;
};

/** Clears per-scan queues, robust scale state and diagnostics. */
void reset();

/**
 * Called from nativeOnDepthMap before DepthCalibrator.update().
 * The frame is retained briefly so a stereo batch that finishes a few milliseconds
 * later can still be paired with the exact source frame.
 */
void onDepthFrame(
    std::uint64_t timestampNs,
    const float* rawDepth,
    int width,
    int height);

/** Pop one stereo/raw-depth correspondence batch, if available. */
bool takeCalibrationBatch(CalibrationBatch* out);

/**
 * Uses the currently VINS-calibrated monocular depth as the world-scale reference
 * to estimate robust (VINS-world units / physical meter). This is deliberately
 * conservative: stereo cannot define the TSDF scale until the existing calibrator
 * is already usable.
 */
bool updateWorldScale(
    const DepthCalibration& current,
    const CalibrationBatch& batch);

/** Whether meter -> VINS-world scale passed the multi-frame consistency gate. */
bool worldScaleUsable();

/** Current robust EMA of VINS-world units per physical meter. */
float worldPerMeter();

/**
 * Pop up to maxBatches recent stereo geometry batches, converted into VINS-world
 * scale. Batches are kept until world scale is stable; stale ones are discarded.
 */
std::vector<WorldAnchorBatch> takeWorldAnchorBatches(
    std::uint64_t nowTimestampNs,
    std::uint64_t maxAgeNs,
    int maxBatches);

/** Diagnostics hooks owned by native_engine.cpp. */
void noteCalibratorContribution(
    int uniqueStereoSamples,
    int weightedCopies,
    bool calibratorAccepted);

void noteSceneTsdf(
    int inputAnchors,
    int acceptedAnchors,
    int passes,
    bool poseMatched);

void noteTargetTsdf(
    int inputAnchors,
    int acceptedAnchors,
    int passes,
    bool maskUsable);

void noteGeometryDropNoPose(int anchors);
void noteGeometryDropNoScale(int anchors);
void noteGeometryDropResidual(int anchors);

/** Copies kStatsSlots floats. */
void fillStats(float out[kStatsSlots]);

/** Human-readable copy for nativeGetStats(). */
std::string summary();

}  // namespace mobilescan3d::stereo_anchor
