#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "uv_unwrap.h"

struct TextureKeyframe {
    std::string imagePath;

    int width = 0;
    int height = 0;

    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    // camera -> world
    float Rwc[9] = {
        1,0,0,
        0,1,0,
        0,0,1
    };
    float twc[3] = {0,0,0};

    float quality = 1.0f;
    std::uint64_t timestampNs = 0;
};

struct TextureBakeOptions {
    int atlasResolution = 2048;
    int maxKeyframes = 12;
    int maxBlendFrames = 3;
    // HQ JPEGs can be 12MP+. Keep offline baking within a phone-friendly
    // memory envelope; intrinsics are scaled after resize.
    int sourceMaxSide = 1600;

    // Software visibility buffer. 384 is a good phone compromise.
    int visibilityMaxSide = 384;
    float visibilityAbsToleranceMeters = 0.018f;
    float visibilityRelativeTolerance = 0.018f;

    float minViewCos = 0.18f;
    int imageBorderPixels = 8;
    int gutterDilationPixels = 6;

    bool exposureNormalize = true;
    int jpegQuality = 92;
};

struct TextureBakeStats {
    int requestedKeyframes = 0;
    int loadedKeyframes = 0;
    int usedKeyframes = 0;
    std::size_t texturedTriangles = 0;
    std::size_t fallbackTriangles = 0;
    std::size_t paintedTexels = 0;   // all chart texels, including vertex-color fallback
    std::size_t hqTexels = 0;        // texels supported by >=1 HQ camera view
    float coveragePercent = 0.0f;    // hqTexels / paintedTexels * 100
};

class TextureBaker {
public:
    static bool bake(
        const UvMesh& mesh,
        const std::vector<TextureKeyframe>& keyframes,
        const TextureBakeOptions& options,
        cv::Mat& atlasBgr,
        TextureBakeStats* stats = nullptr);

private:
    TextureBaker() = delete;
};
