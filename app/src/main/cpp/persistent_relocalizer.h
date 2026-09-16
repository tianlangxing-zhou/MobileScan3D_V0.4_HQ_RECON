#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct PersistentWorldObservation {
    float u = 0.0f;  // VINS image pixel
    float v = 0.0f;
    float x = 0.0f;  // saved/current VINS world point
    float y = 0.0f;
    float z = 0.0f;
    int featureId = -1;
};

struct PersistentRelocalizationStats {
    // 0 idle, 1 map loaded / waiting, 2 searching, 3 localized
    int state = 0;

    int mapPoints = 0;
    int capturedKeyframes = 0;
    int capturedPoints = 0;

    int lastDetected = 0;
    int lastMatches = 0;
    int lastInliers = 0;

    float lastInlierRatio = 0.0f;
    float lastMedianReprojectionPx = 0.0f;

    std::uint64_t attempts = 0;
    std::uint64_t successes = 0;

    bool captureEnabled = false;
    bool mapLoaded = false;
    bool localized = false;
};

/**
 * Mobile persistent relocalization built around the VINS world frame.
 *
 * Capture:
 *   VINS triangulated 3D world points + current image pixels
 *     -> ORB descriptor AT those VINS pixels
 *     -> persistent 3D+descriptor map
 *
 * Restore:
 *   current ORB frame
 *     -> Hamming match against persisted VINS 3D descriptors
 *     -> solvePnPRansac
 *     -> T_savedWorld_currentCamera
 *     -> T_savedWorld_liveVinsWorld
 *
 * We deliberately DO NOT serialize Estimator's marginalization/preintegration
 * pointers. Those are short-lived optimizer internals. The persisted data is
 * the stable geometric product needed for cross-session localization.
 */
class PersistentRelocalizer {
public:
    PersistentRelocalizer();

    void resetAll();
    void resetLiveAlignment();

    void setCaptureEnabled(bool enabled);
    bool captureEnabled() const;

    bool captureFrame(
        const std::uint8_t* gray,
        int width,
        int height,
        int stride,
        std::uint64_t timestampNs,
        const float liveCameraRwc[9],
        const float liveCameraTwc[3],
        const PersistentWorldObservation* observations,
        int observationCount);

    bool saveMap(const std::string& path) const;
    bool loadMap(const std::string& path);

    /**
     * Run one relocalization attempt.
     * liveCameraRwc/twc are the current camera pose in the NEW VINS world.
     * PnP gives the current camera pose in the SAVED world. Their difference
     * is the rigid savedWorld <- liveWorld alignment used for AR.
     */
    bool tryRelocalize(
        const std::uint8_t* gray,
        int width,
        int height,
        int stride,
        std::uint64_t timestampNs,
        float fx,
        float fy,
        float cx,
        float cy,
        const float liveCameraRwc[9],
        const float liveCameraTwc[3]);

    /**
     * Transform a live VINS-world camera/body pose into the saved scan world.
     *
     * If no map is loaded, input is copied unchanged.
     * If a map is loaded but relocalization is not established, returns false.
     */
    bool transformPose(
        const float inRwc[9],
        const float inTwc[3],
        float outRwc[9],
        float outTwc[3]) const;

    bool saveAlignment(const std::string& path) const;
    bool loadAlignment(const std::string& path);

    bool hasLoadedMap() const;
    bool localized() const;

    PersistentRelocalizationStats stats() const;

private:
    struct MapPoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::array<std::uint8_t, 32> descriptor{};
    };

    bool shouldCaptureKeyframeLocked(
        std::uint64_t timestampNs,
        const float Rwc[9],
        const float twc[3]) const;

    void rebuildDescriptorMatrixLocked();

    mutable std::mutex mutex_;

    std::vector<MapPoint> mapPoints_;
    std::vector<std::uint8_t> descriptorBytes_;

    std::unordered_map<int, std::size_t> capturedFeatureIndex_;

    bool captureEnabled_ = false;
    int capturedKeyframes_ = 0;

    bool haveLastCapturePose_ = false;
    std::uint64_t lastCaptureTimestampNs_ = 0;
    float lastCaptureRwc_[9] = {
        1,0,0, 0,1,0, 0,0,1
    };
    float lastCaptureTwc_[3] = {0,0,0};

    bool mapLoaded_ = false;
    bool localized_ = false;

    // savedWorld <- liveWorld
    float alignR_[9] = {
        1,0,0, 0,1,0, 0,0,1
    };
    float alignT_[3] = {0,0,0};

    // Require two geometrically consistent PnP solutions before the first lock.
    bool havePendingAlignment_ = false;
    float pendingR_[9] = {
        1,0,0, 0,1,0, 0,0,1
    };
    float pendingT_[3] = {0,0,0};
    int pendingHits_ = 0;

    std::uint64_t lastAttemptNs_ = 0;

    PersistentRelocalizationStats stats_;
};
