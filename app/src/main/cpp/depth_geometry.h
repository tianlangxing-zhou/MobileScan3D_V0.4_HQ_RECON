#pragma once
struct DepthIntrinsics {
    float fx, fy, cx, cy;
};
// Matches DepthPreprocessor's full-frame resize (no crop or rotation).
inline DepthIntrinsics depthIntrinsics(float fx, float fy, float cx, float cy,
                                      int cameraW, int cameraH, int depthW, int depthH) {
    const float sx = static_cast<float>(depthW) / cameraW;
    const float sy = static_cast<float>(depthH) / cameraH;
    return {fx * sx, fy * sy, cx * sx, cy * sy};
}
