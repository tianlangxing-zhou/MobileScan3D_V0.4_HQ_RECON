#include "stereo_anchor_bridge.h"

#include <jni.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>

namespace mobilescan3d::stereo_anchor {
namespace {

constexpr std::int64_t kExactDepthMatchNs = 2'000'000LL;   // same primary frame
constexpr std::size_t kRecentDepthFrames = 3;
constexpr std::size_t kPendingBatches = 24;
constexpr std::size_t kReadyCalibrationBatches = 12;
constexpr std::size_t kGeometryBatches = 24;

constexpr int kScaleMinSamples = 6;
constexpr int kScaleStableFrames = 3;
constexpr float kScaleMaxMedianRel = 0.12f;
constexpr float kScaleStableMedianRel = 0.10f;
constexpr float kScaleMin = 0.25f;
constexpr float kScaleMax = 4.0f;
constexpr float kScaleMaxJumpRatio = 1.35f;
constexpr float kScaleEmaNew = 0.20f;

struct DepthFrame {
    std::uint64_t timestampNs = 0;
    int width = 0;
    int height = 0;
    std::vector<float> depth;
};

struct Diagnostics {
    std::uint64_t submittedBatches = 0;
    std::uint64_t submittedAnchors = 0;
    std::uint64_t inputRejected = 0;
    std::uint64_t pendingDropped = 0;

    std::uint64_t depthFramesSeen = 0;
    std::uint64_t depthMatchedBatches = 0;
    std::uint64_t depthMatchedAnchors = 0;
    std::uint64_t depthMatchMisses = 0;
    std::int64_t lastDepthMatchDeltaNs = -1;

    std::uint64_t scaleAcceptedFrames = 0;
    std::uint64_t scaleRejectedFrames = 0;
    std::uint64_t scaleRejectedNoBaseCalib = 0;
    std::uint64_t scaleRejectedFewSamples = 0;
    std::uint64_t scaleRejectedResidual = 0;
    std::uint64_t scaleRejectedJump = 0;

    float worldPerMeter = 1.f;
    float lastFrameScale = 1.f;
    float lastScaleMedianRel = 1.f;
    float lastDenseStereoMedianRel = 1.f;
    int lastScaleSamples = 0;
    bool worldScaleStable = false;
    int scaleGoodStreak = 0;
    int scaleBadStreak = 0;

    std::uint64_t calibratorStereoFrames = 0;
    std::uint64_t calibratorStereoAccepted = 0;
    std::uint64_t calibratorStereoRejected = 0;
    std::uint64_t calibratorUniqueSamples = 0;
    std::uint64_t calibratorWeightedCopies = 0;

    std::uint64_t geometryBatchesPopped = 0;
    std::uint64_t geometryStaleBatches = 0;
    std::uint64_t geometryDropNoPose = 0;
    std::uint64_t geometryDropNoScale = 0;
    std::uint64_t geometryDropResidual = 0;

    std::uint64_t sceneTsdfBatches = 0;
    std::uint64_t sceneTsdfInputAnchors = 0;
    std::uint64_t sceneTsdfAcceptedAnchors = 0;
    std::uint64_t sceneTsdfPasses = 0;

    std::uint64_t targetTsdfBatches = 0;
    std::uint64_t targetTsdfInputAnchors = 0;
    std::uint64_t targetTsdfAcceptedAnchors = 0;
    std::uint64_t targetTsdfPasses = 0;

    float lastAnchorMedianMetricM = 0.f;
    float lastAnchorMedianQuality = 0.f;
};

std::mutex gMutex;
std::deque<AnchorBatch> gPending;
std::deque<DepthFrame> gDepthFrames;
std::deque<CalibrationBatch> gReadyCalibration;
std::deque<AnchorBatch> gGeometry;
Diagnostics gDiag;

float median(std::vector<float> values) {
    if (values.empty()) return 0.f;
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    float hi = values[mid];
    if (values.size() & 1u) return hi;
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
        values.begin() + static_cast<std::ptrdiff_t>(mid));
    return 0.5f * (values[mid - 1] + hi);
}

bool validAnchor(const Anchor& a) {
    return std::isfinite(a.u) &&
           std::isfinite(a.v) &&
           std::isfinite(a.zMetric) &&
           std::isfinite(a.reprojectionPx) &&
           std::isfinite(a.confidence) &&
           std::isfinite(a.parallaxDeg) &&
           a.u >= 0.f && a.u <= 1.f &&
           a.v >= 0.f && a.v <= 1.f &&
           a.zMetric >= 0.12f && a.zMetric <= 12.f &&
           a.reprojectionPx >= 0.f && a.reprojectionPx <= 3.0f &&
           a.confidence >= 0.25f && a.confidence <= 1.001f &&
           a.parallaxDeg >= 0.20f && a.parallaxDeg <= 90.f;
}

std::int64_t absDelta(std::uint64_t a, std::uint64_t b) {
    const std::int64_t aa = static_cast<std::int64_t>(a);
    const std::int64_t bb = static_cast<std::int64_t>(b);
    const std::int64_t d = aa - bb;
    return d < 0 ? -d : d;
}

void trimQueuesLocked() {
    while (gPending.size() > kPendingBatches) {
        gDiag.pendingDropped += gPending.front().anchors.size();
        gPending.pop_front();
    }
    while (gDepthFrames.size() > kRecentDepthFrames) {
        gDepthFrames.pop_front();
    }
    while (gReadyCalibration.size() > kReadyCalibrationBatches) {
        gReadyCalibration.pop_front();
    }
    while (gGeometry.size() > kGeometryBatches) {
        if (!gDiag.worldScaleStable) {
            gDiag.geometryDropNoScale +=
                static_cast<std::uint64_t>(
                    gGeometry.front().anchors.size());
        } else {
            gDiag.geometryStaleBatches++;
        }
        gGeometry.pop_front();
    }
}

bool buildCalibrationBatchLocked(
        const AnchorBatch& anchors,
        const DepthFrame& depth,
        CalibrationBatch* out) {
    if (!out ||
        depth.width <= 0 ||
        depth.height <= 0 ||
        depth.depth.size() < static_cast<std::size_t>(depth.width) * depth.height) {
        return false;
    }

    CalibrationBatch b;
    b.timestampNs = anchors.timestampNs;
    b.matchDeltaNs = absDelta(anchors.timestampNs, depth.timestampNs);
    b.rawDepth.reserve(anchors.anchors.size());
    b.metricDepth.reserve(anchors.anchors.size());
    b.quality.reserve(anchors.anchors.size());

    for (const Anchor& a : anchors.anchors) {
        const int px = std::clamp(
            static_cast<int>(std::lround(a.u * static_cast<float>(depth.width - 1))),
            0,
            depth.width - 1);
        const int py = std::clamp(
            static_cast<int>(std::lround(a.v * static_cast<float>(depth.height - 1))),
            0,
            depth.height - 1);
        const float raw = depth.depth[
            static_cast<std::size_t>(py) * depth.width + px];

        if (!std::isfinite(raw)) continue;

        // Stereo-side quality already includes sync, reprojection and parallax.
        const float q = std::clamp(a.confidence, 0.f, 1.f);
        if (q < 0.30f) continue;

        b.rawDepth.push_back(raw);
        b.metricDepth.push_back(a.zMetric);
        b.quality.push_back(q);
    }

    if (b.rawDepth.size() < 3) return false;
    *out = std::move(b);
    return true;
}

void tryMatchLocked() {
    for (auto ai = gPending.begin(); ai != gPending.end();) {
        auto best = gDepthFrames.end();
        std::int64_t bestDelta = kExactDepthMatchNs + 1;

        for (auto di = gDepthFrames.begin(); di != gDepthFrames.end(); ++di) {
            const std::int64_t d = absDelta(ai->timestampNs, di->timestampNs);
            if (d < bestDelta) {
                bestDelta = d;
                best = di;
            }
        }

        if (best != gDepthFrames.end() && bestDelta <= kExactDepthMatchNs) {
            CalibrationBatch paired;
            if (buildCalibrationBatchLocked(*ai, *best, &paired)) {
                gDiag.depthMatchedBatches++;
                gDiag.depthMatchedAnchors += paired.rawDepth.size();
                gDiag.lastDepthMatchDeltaNs = paired.matchDeltaNs;
                gReadyCalibration.push_back(std::move(paired));
            } else {
                gDiag.depthMatchMisses++;
            }
            ai = gPending.erase(ai);
        } else {
            ++ai;
        }
    }
    trimQueuesLocked();
}

}  // namespace

void reset() {
    std::lock_guard<std::mutex> lock(gMutex);
    gPending.clear();
    gDepthFrames.clear();
    gReadyCalibration.clear();
    gGeometry.clear();
    gDiag = Diagnostics{};
}

void onDepthFrame(
        std::uint64_t timestampNs,
        const float* rawDepth,
        int width,
        int height) {
    if (!rawDepth || width <= 0 || height <= 0) return;

    DepthFrame frame;
    frame.timestampNs = timestampNs;
    frame.width = width;
    frame.height = height;
    frame.depth.assign(
        rawDepth,
        rawDepth + static_cast<std::size_t>(width) * height);

    std::lock_guard<std::mutex> lock(gMutex);
    gDiag.depthFramesSeen++;
    gDepthFrames.push_back(std::move(frame));
    trimQueuesLocked();
    tryMatchLocked();
}

bool takeCalibrationBatch(CalibrationBatch* out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(gMutex);
    if (gReadyCalibration.empty()) return false;
    *out = std::move(gReadyCalibration.front());
    gReadyCalibration.pop_front();
    return true;
}

bool updateWorldScale(
        const DepthCalibration& current,
        const CalibrationBatch& batch) {
    std::lock_guard<std::mutex> lock(gMutex);

    if (!current.valid || current.confidence < 0.20f) {
        gDiag.scaleRejectedFrames++;
        gDiag.scaleRejectedNoBaseCalib++;
        gDiag.scaleGoodStreak = 0;
        gDiag.scaleBadStreak++;
        if (gDiag.scaleBadStreak >= 2) gDiag.worldScaleStable = false;
        return false;
    }

    const std::size_t n = std::min({
        batch.rawDepth.size(),
        batch.metricDepth.size(),
        batch.quality.size()});

    std::vector<float> ratios;
    std::vector<float> denseRel;
    ratios.reserve(n);
    denseRel.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        if (batch.quality[i] < 0.35f) continue;
        const float metric = batch.metricDepth[i];
        if (!(metric > 0.12f) || !std::isfinite(metric)) continue;

        const float pred = current.toMetric(batch.rawDepth[i], 0.f);
        if (!(pred > 0.05f) || !std::isfinite(pred)) continue;

        const float ratio = pred / metric;
        if (!std::isfinite(ratio) || ratio < kScaleMin || ratio > kScaleMax) {
            continue;
        }
        ratios.push_back(ratio);
    }

    if (ratios.size() < static_cast<std::size_t>(kScaleMinSamples)) {
        gDiag.scaleRejectedFrames++;
        gDiag.scaleRejectedFewSamples++;
        gDiag.lastScaleSamples = static_cast<int>(ratios.size());
        gDiag.scaleGoodStreak = 0;
        gDiag.scaleBadStreak++;
        if (gDiag.scaleBadStreak >= 2) gDiag.worldScaleStable = false;
        return false;
    }

    const float frameScale = median(ratios);
    std::vector<float> rel;
    rel.reserve(ratios.size());
    for (float r : ratios) {
        rel.push_back(std::fabs(r / std::max(1e-6f, frameScale) - 1.f));
    }
    const float medRel = median(rel);

    // Independent "dense vs stereo after one robust scale" metric.
    for (std::size_t i = 0; i < n; ++i) {
        if (batch.quality[i] < 0.35f) continue;
        const float metric = batch.metricDepth[i];
        const float pred = current.toMetric(batch.rawDepth[i], 0.f);
        if (!(metric > 0.12f) || !(pred > 0.05f) ||
            !std::isfinite(metric) || !std::isfinite(pred)) {
            continue;
        }
        const float target = metric * frameScale;
        denseRel.push_back(
            std::fabs(pred - target) / std::max(0.05f, pred));
    }
    const float denseStereoRel = median(denseRel);

    gDiag.lastFrameScale = frameScale;
    gDiag.lastScaleMedianRel = medRel;
    gDiag.lastDenseStereoMedianRel = denseStereoRel;
    gDiag.lastScaleSamples = static_cast<int>(ratios.size());

    if (!std::isfinite(frameScale) ||
        frameScale < kScaleMin ||
        frameScale > kScaleMax ||
        medRel > kScaleMaxMedianRel ||
        denseStereoRel > 0.16f) {
        gDiag.scaleRejectedFrames++;
        gDiag.scaleRejectedResidual++;
        gDiag.scaleGoodStreak = 0;
        gDiag.scaleBadStreak++;
        if (gDiag.scaleBadStreak >= 2) gDiag.worldScaleStable = false;
        return false;
    }

    if (gDiag.scaleAcceptedFrames > 0) {
        const float base = std::max(1e-6f, gDiag.worldPerMeter);
        const float jump = frameScale / base;
        if (!std::isfinite(jump) ||
            jump > kScaleMaxJumpRatio ||
            jump < 1.f / kScaleMaxJumpRatio) {
            gDiag.scaleRejectedFrames++;
            gDiag.scaleRejectedJump++;
            gDiag.scaleGoodStreak = 0;
            gDiag.scaleBadStreak++;
            if (gDiag.scaleBadStreak >= 2) gDiag.worldScaleStable = false;
            return false;
        }
    }

    if (gDiag.scaleAcceptedFrames == 0) {
        gDiag.worldPerMeter = frameScale;
    } else {
        gDiag.worldPerMeter =
            (1.f - kScaleEmaNew) * gDiag.worldPerMeter +
            kScaleEmaNew * frameScale;
    }

    gDiag.scaleAcceptedFrames++;
    gDiag.scaleBadStreak = 0;
    gDiag.scaleGoodStreak++;
    gDiag.worldScaleStable =
        gDiag.scaleGoodStreak >= kScaleStableFrames &&
        medRel <= kScaleStableMedianRel &&
        denseStereoRel <= 0.14f;

    return true;
}

bool worldScaleUsable() {
    std::lock_guard<std::mutex> lock(gMutex);
    return gDiag.worldScaleStable &&
           std::isfinite(gDiag.worldPerMeter) &&
           gDiag.worldPerMeter >= kScaleMin &&
           gDiag.worldPerMeter <= kScaleMax;
}

float worldPerMeter() {
    std::lock_guard<std::mutex> lock(gMutex);
    return gDiag.worldPerMeter;
}

std::vector<WorldAnchorBatch> takeWorldAnchorBatches(
        std::uint64_t nowTimestampNs,
        std::uint64_t maxAgeNs,
        int maxBatches) {
    std::vector<WorldAnchorBatch> out;
    if (maxBatches <= 0) return out;

    std::lock_guard<std::mutex> lock(gMutex);

    if (!gDiag.worldScaleStable) {
        if (!gGeometry.empty()) {
            gDiag.geometryDropNoScale += 0; // queue intentionally retained
        }
        return out;
    }

    while (!gGeometry.empty()) {
        const AnchorBatch& front = gGeometry.front();
        if (nowTimestampNs > front.timestampNs &&
            nowTimestampNs - front.timestampNs > maxAgeNs) {
            gDiag.geometryStaleBatches++;
            gGeometry.pop_front();
        } else {
            break;
        }
    }

    const float scale = gDiag.worldPerMeter;
    while (!gGeometry.empty() &&
           static_cast<int>(out.size()) < maxBatches) {
        const AnchorBatch src = std::move(gGeometry.front());
        gGeometry.pop_front();

        WorldAnchorBatch dst;
        dst.timestampNs = src.timestampNs;
        dst.imageWidth = src.imageWidth;
        dst.imageHeight = src.imageHeight;
        dst.anchors.reserve(src.anchors.size());

        for (const Anchor& a : src.anchors) {
            // Strong gate for geometry injection. Calibration may use looser anchors,
            // TSDF does not.
            if (a.confidence < 0.50f ||
                a.reprojectionPx > 1.75f ||
                a.parallaxDeg < 0.45f) {
                gDiag.geometryDropResidual++;
                continue;
            }
            WorldAnchor w;
            w.u = a.u;
            w.v = a.v;
            w.zMetric = a.zMetric;
            w.zWorld = a.zMetric * scale;
            w.reprojectionPx = a.reprojectionPx;
            w.confidence = a.confidence;
            w.parallaxDeg = a.parallaxDeg;

            const float reprojScore =
                std::exp(-a.reprojectionPx / 1.25f);
            const float parallaxScore =
                std::clamp((a.parallaxDeg - 0.35f) / 1.50f, 0.f, 1.f);
            w.weight = std::clamp(
                1.0f + 3.0f * a.confidence * reprojScore * parallaxScore,
                1.0f,
                4.0f);

            if (std::isfinite(w.zWorld) &&
                w.zWorld > 0.05f &&
                w.zWorld < 12.f) {
                dst.anchors.push_back(w);
            }
        }

        if (!dst.anchors.empty()) {
            out.push_back(std::move(dst));
            gDiag.geometryBatchesPopped++;
        }
    }
    return out;
}

void noteCalibratorContribution(
        int uniqueStereoSamples,
        int weightedCopies,
        bool calibratorAccepted) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (uniqueStereoSamples <= 0) return;
    gDiag.calibratorStereoFrames++;
    gDiag.calibratorUniqueSamples +=
        static_cast<std::uint64_t>(uniqueStereoSamples);
    gDiag.calibratorWeightedCopies +=
        static_cast<std::uint64_t>(std::max(0, weightedCopies));
    if (calibratorAccepted) {
        gDiag.calibratorStereoAccepted++;
    } else {
        gDiag.calibratorStereoRejected++;
    }
}

void noteSceneTsdf(
        int inputAnchors,
        int acceptedAnchors,
        int passes,
        bool poseMatched) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (!poseMatched) {
        gDiag.geometryDropNoPose +=
            static_cast<std::uint64_t>(std::max(0, inputAnchors));
        return;
    }
    gDiag.sceneTsdfBatches++;
    gDiag.sceneTsdfInputAnchors +=
        static_cast<std::uint64_t>(std::max(0, inputAnchors));
    gDiag.sceneTsdfAcceptedAnchors +=
        static_cast<std::uint64_t>(std::max(0, acceptedAnchors));
    gDiag.sceneTsdfPasses +=
        static_cast<std::uint64_t>(std::max(0, passes));
}

void noteTargetTsdf(
        int inputAnchors,
        int acceptedAnchors,
        int passes,
        bool maskUsable) {
    std::lock_guard<std::mutex> lock(gMutex);
    if (!maskUsable) return;
    gDiag.targetTsdfBatches++;
    gDiag.targetTsdfInputAnchors +=
        static_cast<std::uint64_t>(std::max(0, inputAnchors));
    gDiag.targetTsdfAcceptedAnchors +=
        static_cast<std::uint64_t>(std::max(0, acceptedAnchors));
    gDiag.targetTsdfPasses +=
        static_cast<std::uint64_t>(std::max(0, passes));
}

void noteGeometryDropNoPose(int anchors) {
    std::lock_guard<std::mutex> lock(gMutex);
    gDiag.geometryDropNoPose +=
        static_cast<std::uint64_t>(std::max(0, anchors));
}

void noteGeometryDropNoScale(int anchors) {
    std::lock_guard<std::mutex> lock(gMutex);
    gDiag.geometryDropNoScale +=
        static_cast<std::uint64_t>(std::max(0, anchors));
}

void noteGeometryDropResidual(int anchors) {
    std::lock_guard<std::mutex> lock(gMutex);
    gDiag.geometryDropResidual +=
        static_cast<std::uint64_t>(std::max(0, anchors));
}

void fillStats(float out[kStatsSlots]) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(gMutex);

    std::fill(out, out + kStatsSlots, 0.f);
    out[0] = static_cast<float>(gDiag.submittedBatches);
    out[1] = static_cast<float>(gDiag.submittedAnchors);
    out[2] = static_cast<float>(gDiag.inputRejected);
    out[3] = static_cast<float>(gDiag.pendingDropped);

    out[4] = static_cast<float>(gDiag.depthFramesSeen);
    out[5] = static_cast<float>(gDiag.depthMatchedBatches);
    out[6] = static_cast<float>(gDiag.depthMatchedAnchors);
    out[7] = static_cast<float>(gDiag.depthMatchMisses);
    out[8] = gDiag.lastDepthMatchDeltaNs >= 0
        ? static_cast<float>(gDiag.lastDepthMatchDeltaNs / 1.0e6)
        : -1.f;

    out[9] = gDiag.worldPerMeter;
    out[10] = gDiag.lastFrameScale;
    out[11] = gDiag.lastScaleMedianRel;
    out[12] = gDiag.lastDenseStereoMedianRel;
    out[13] = static_cast<float>(gDiag.lastScaleSamples);
    out[14] = gDiag.worldScaleStable ? 1.f : 0.f;
    out[15] = static_cast<float>(gDiag.scaleAcceptedFrames);
    out[16] = static_cast<float>(gDiag.scaleRejectedFrames);
    out[17] = static_cast<float>(gDiag.scaleRejectedNoBaseCalib);
    out[18] = static_cast<float>(gDiag.scaleRejectedFewSamples);
    out[19] = static_cast<float>(gDiag.scaleRejectedResidual);
    out[20] = static_cast<float>(gDiag.scaleRejectedJump);

    out[21] = static_cast<float>(gDiag.calibratorStereoFrames);
    out[22] = static_cast<float>(gDiag.calibratorStereoAccepted);
    out[23] = static_cast<float>(gDiag.calibratorStereoRejected);
    out[24] = static_cast<float>(gDiag.calibratorUniqueSamples);
    out[25] = static_cast<float>(gDiag.calibratorWeightedCopies);

    out[26] = static_cast<float>(gDiag.geometryBatchesPopped);
    out[27] = static_cast<float>(gDiag.geometryStaleBatches);
    out[28] = static_cast<float>(gDiag.geometryDropNoPose);
    out[29] = static_cast<float>(gDiag.geometryDropNoScale);
    out[30] = static_cast<float>(gDiag.geometryDropResidual);

    out[31] = static_cast<float>(gDiag.sceneTsdfBatches);
    out[32] = static_cast<float>(gDiag.sceneTsdfInputAnchors);
    out[33] = static_cast<float>(gDiag.sceneTsdfAcceptedAnchors);
    out[34] = static_cast<float>(gDiag.sceneTsdfPasses);

    out[35] = static_cast<float>(gDiag.targetTsdfBatches);
    out[36] = static_cast<float>(gDiag.targetTsdfInputAnchors);
    out[37] = static_cast<float>(gDiag.targetTsdfAcceptedAnchors);
    out[38] = static_cast<float>(gDiag.targetTsdfPasses);

    out[39] = gDiag.lastAnchorMedianMetricM;
    out[40] = gDiag.lastAnchorMedianQuality;
    out[41] = static_cast<float>(gPending.size());
    out[42] = static_cast<float>(gDepthFrames.size());
    out[43] = static_cast<float>(gReadyCalibration.size());
    out[44] = static_cast<float>(gGeometry.size());
    out[45] = static_cast<float>(kExactDepthMatchNs / 1.0e6);
    out[46] = static_cast<float>(gDiag.scaleGoodStreak);
    out[47] = static_cast<float>(gDiag.scaleBadStreak);
}

std::string summary() {
    std::lock_guard<std::mutex> lock(gMutex);
    std::ostringstream s;
    s << "StereoAnchorFusion:"
      << " submittedBatches=" << gDiag.submittedBatches
      << " submittedAnchors=" << gDiag.submittedAnchors
      << " matchedBatches=" << gDiag.depthMatchedBatches
      << " matchedAnchors=" << gDiag.depthMatchedAnchors
      << " worldPerMeter=" << gDiag.worldPerMeter
      << " scaleStable=" << (gDiag.worldScaleStable ? 1 : 0)
      << " scaleMedianRel=" << gDiag.lastScaleMedianRel
      << " denseStereoMedianRel=" << gDiag.lastDenseStereoMedianRel
      << " calibStereoAccepted=" << gDiag.calibratorStereoAccepted
      << "/" << gDiag.calibratorStereoFrames
      << " sceneTsdfAnchors=" << gDiag.sceneTsdfAcceptedAnchors
      << " targetTsdfAnchors=" << gDiag.targetTsdfAcceptedAnchors;
    return s.str();
}

static bool submitBatch(
        const float* values,
        int count,
        std::uint64_t timestampNs,
        int width,
        int height) {
    if (!values || count <= 0 || width <= 0 || height <= 0) return false;
    count = std::min(count, kMaxAnchorsPerBatch);

    AnchorBatch batch;
    batch.timestampNs = timestampNs;
    batch.imageWidth = width;
    batch.imageHeight = height;
    batch.anchors.reserve(static_cast<std::size_t>(count));

    std::vector<float> metric;
    std::vector<float> quality;

    int rejected = 0;
    for (int i = 0; i < count; ++i) {
        const int o = i * kAnchorStride;
        Anchor a;
        a.u = values[o + 0];
        a.v = values[o + 1];
        a.zMetric = values[o + 2];
        a.reprojectionPx = values[o + 3];
        a.confidence = values[o + 4];
        a.parallaxDeg = values[o + 5];

        if (!validAnchor(a)) {
            rejected++;
            continue;
        }
        batch.anchors.push_back(a);
        metric.push_back(a.zMetric);
        quality.push_back(a.confidence);
    }

    if (batch.anchors.size() < 3) {
        std::lock_guard<std::mutex> lock(gMutex);
        gDiag.inputRejected += static_cast<std::uint64_t>(std::max(rejected, count));
        return false;
    }

    const float metricMedian = median(metric);
    const float qualityMedian = median(quality);

    std::lock_guard<std::mutex> lock(gMutex);
    gDiag.submittedBatches++;
    gDiag.submittedAnchors += batch.anchors.size();
    gDiag.inputRejected += static_cast<std::uint64_t>(rejected);
    gDiag.lastAnchorMedianMetricM = metricMedian;
    gDiag.lastAnchorMedianQuality = qualityMedian;

    gPending.push_back(batch);
    gGeometry.push_back(std::move(batch));
    trimQueuesLocked();
    tryMatchLocked();
    return true;
}

}  // namespace mobilescan3d::stereo_anchor

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSubmitStereoAnchors(
        JNIEnv* env,
        jobject,
        jfloatArray anchors,
        jint count,
        jlong timestampNs,
        jint imageWidth,
        jint imageHeight) {
    using namespace mobilescan3d::stereo_anchor;

    if (!anchors || count <= 0) return JNI_FALSE;

    const int safeCount = std::min(
        static_cast<int>(count),
        kMaxAnchorsPerBatch);
    const jsize need = safeCount * kAnchorStride;
    if (env->GetArrayLength(anchors) < need) return JNI_FALSE;

    std::vector<float> values(static_cast<std::size_t>(need));
    env->GetFloatArrayRegion(
        anchors,
        0,
        need,
        values.data());
    if (env->ExceptionCheck()) return JNI_FALSE;

    return submitBatch(
        values.data(),
        safeCount,
        static_cast<std::uint64_t>(timestampNs),
        static_cast<int>(imageWidth),
        static_cast<int>(imageHeight))
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetStereoAnchorStats(
        JNIEnv* env,
        jobject,
        jfloatArray out) {
    using namespace mobilescan3d::stereo_anchor;

    if (!out || env->GetArrayLength(out) < kStatsSlots) return 0;
    float values[kStatsSlots];
    fillStats(values);
    env->SetFloatArrayRegion(out, 0, kStatsSlots, values);
    return env->ExceptionCheck() ? 0 : kStatsSlots;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeResetStereoAnchors(
        JNIEnv*,
        jobject) {
    mobilescan3d::stereo_anchor::reset();
}
