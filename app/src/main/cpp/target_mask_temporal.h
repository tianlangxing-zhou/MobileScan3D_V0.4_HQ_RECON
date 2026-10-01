#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdint>

// Compare silhouettes AFTER compensating the source-exposure tracking box's
// translation/scale. This is only a mask gate; pose and depth consistency still
// independently guard all geometry writes.
struct TargetMaskMotionComparison {
    bool comparable = false;
    float iou = 0.f, areaJump = 0.f, centerJump = 0.f;
};
inline TargetMaskMotionComparison compareTargetMasks(
        const cv::Mat& previous, const cv::Mat& current,
        const cv::Rect2f& previousBox, const cv::Rect2f& currentBox,
        cv::Mat& alignedScratch) {
    TargetMaskMotionComparison result;
    if (previous.empty() || current.empty() || previous.size() != current.size() ||
        previous.type() != CV_8UC1 || current.type() != CV_8UC1) return result;
    const auto validBox = [](const cv::Rect2f& r) {
        return std::isfinite(r.x) && std::isfinite(r.y) &&
               std::isfinite(r.width) && std::isfinite(r.height) &&
               r.width >= 1.f && r.height >= 1.f;
    };
    if (!validBox(previousBox) || !validBox(currentBox)) return result;
    const double sx = double(currentBox.width) / previousBox.width;
    const double sy = double(currentBox.height) / previousBox.height;
    // Implausible size changes must not be "explained away" by the warp.
    result.comparable = true;
    if (sx < .5 || sx > 2. || sy < .5 || sy > 2.) {
        result.areaJump = 1.f; result.centerJump = 1.f; return result;
    }
    const cv::Matx23d transform(sx, 0., currentBox.x - sx * previousBox.x,
                                0., sy, currentBox.y - sy * previousBox.y);
    cv::warpAffine(previous, alignedScratch, transform, current.size(),
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    int oldArea = 0, newArea = 0, intersection = 0;
    double oldX = 0, oldY = 0, newX = 0, newY = 0;
    for (int y = 0; y < current.rows; ++y) {
        const auto* a = alignedScratch.ptr<uint8_t>(y);
        const auto* b = current.ptr<uint8_t>(y);
        for (int x = 0; x < current.cols; ++x) {
            if (a[x]) { ++oldArea; oldX += x; oldY += y; }
            if (b[x]) { ++newArea; newX += x; newY += y; }
            if (a[x] && b[x]) ++intersection;
        }
    }
    if (!oldArea || !newArea) {
        result.areaJump = 1.f; result.centerJump = 1.f; return result;
    }
    result.iou = float(intersection) / (oldArea + newArea - intersection);
    result.areaJump = float(std::abs(newArea - oldArea)) / oldArea;
    result.centerJump = float(std::hypot(newX/newArea - oldX/oldArea,
                                         newY/newArea - oldY/oldArea) /
                               std::hypot(currentBox.width, currentBox.height));
    return result;
}
