#include "target_mask_engine.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

/** seed 深度窗口半径：7x7 -> radius 3 */
constexpr int kSeedRadius = 3;
/** 深度容差比例：seed 深度的 8% */
constexpr float kToleranceRatio = 0.08f;
/** 深度容差下限（米）。太小的目标 + 有噪声的 depth 需要留一点余量 */
constexpr float kToleranceMin = 0.08f;
/** 深度容差上限（米）。防止远处目标把整面墙一起圈进来 */
constexpr float kToleranceMax = 0.45f;
/** searchBox 外扩比例：bbox 一般比目标略紧，略微外扩避免切掉边缘 */
constexpr float kSearchMargin = 0.10f;

} // namespace

float TargetMaskEngine::medianAroundSeed(const float* depth, int width, int height,
                                         cv::Point seed, int radius) {
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>((2 * radius + 1) * (2 * radius + 1)));
    for (int y = seed.y - radius; y <= seed.y + radius; ++y) {
        if (y < 0 || y >= height) continue;
        for (int x = seed.x - radius; x <= seed.x + radius; ++x) {
            if (x < 0 || x >= width) continue;
            const float d = depth[static_cast<size_t>(y) * width + x];
            if (std::isfinite(d) && d > 0.f) {
                vals.push_back(d);
            }
        }
    }
    if (vals.empty()) {
        return 0.f;
    }
    const size_t n = vals.size();
    std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
    return vals[n / 2];
}

bool TargetMaskEngine::findValidSeed(const float* depth, int width, int height,
                                     cv::Point hint, cv::Point& out) {
    // 以 hint 为中心做环形扩散，找到第一个深度有效的像素。
    // 用户点选 / 拖框的中心偶尔会落在目标的空洞（反光、透明、超距）上，
    // 此时直接在 hint 上取中位数会得到「没有种子」而整帧不建 mask。
    constexpr int kMaxRing = 24;
    for (int r = 0; r <= kMaxRing; ++r) {
        const int x0 = hint.x - r, x1 = hint.x + r;
        const int y0 = hint.y - r, y1 = hint.y + r;
        for (int y = y0; y <= y1; ++y) {
            if (y < 0 || y >= height) continue;
            for (int x = x0; x <= x1; ++x) {
                if (x < 0 || x >= width) continue;
                // 只看这一圈的边框
                if (r > 0 && x != x0 && x != x1 && y != y0 && y != y1) continue;
                const float d = depth[static_cast<size_t>(y) * width + x];
                if (std::isfinite(d) && d > 0.f) {
                    out = cv::Point(x, y);
                    return true;
                }
            }
        }
    }
    return false;
}

bool TargetMaskEngine::build(const float* depth, int width, int height,
                             const cv::Rect& searchBox, cv::Point seed,
                             cv::Mat& mask, TargetMaskStats& stats) {
    stats = TargetMaskStats{};
    if (mask.empty() || mask.cols != width || mask.rows != height || mask.type() != CV_8U) {
        mask = cv::Mat::zeros(height, width, CV_8U);
    } else {
        mask.setTo(0);
    }
    if (!depth || width < 8 || height < 8) {
        stats.rejectReason = "invalid depth image";
        return false;
    }

    // ---- 搜索范围：bbox 略微外扩，并夹到图像内 ----
    cv::Rect box = searchBox;
    if (box.width <= 0 || box.height <= 0) {
        stats.rejectReason = "empty search box";
        return false;
    }
    const int mx = static_cast<int>(std::lround(box.width * kSearchMargin));
    const int my = static_cast<int>(std::lround(box.height * kSearchMargin));
    box = cv::Rect(box.x - mx, box.y - my, box.width + 2 * mx, box.height + 2 * my) &
          cv::Rect(0, 0, width, height);
    if (box.width < 8 || box.height < 8) {
        stats.rejectReason = "search box too small";
        return false;
    }
    stats.searchPixels = box.area();

    // ---- seed：优先用调用方给的，取不到有效深度就在附近扩散找 ----
    seed.x = std::clamp(seed.x, 0, width - 1);
    seed.y = std::clamp(seed.y, 0, height - 1);
    if (!std::isfinite(depth[static_cast<size_t>(seed.y) * width + seed.x]) ||
        depth[static_cast<size_t>(seed.y) * width + seed.x] <= 0.f) {
        cv::Point better;
        if (findValidSeed(depth, width, height, seed, better)) {
            seed = better;
        }
    }

    const float seedDepth = medianAroundSeed(depth, width, height, seed, kSeedRadius);
    if (!(seedDepth > 0.f)) {
        stats.rejectReason = "no valid seed depth";
        return false;
    }
    stats.seedValid = true;
    stats.seedDepth = seedDepth;

    const float tolerance =
        std::clamp(seedDepth * kToleranceRatio, kToleranceMin, kToleranceMax);
    stats.tolerance = tolerance;

    // ---- 深度带通得到 candidate（只在 searchBox 内） ----
    if (candidate_.rows != height || candidate_.cols != width || candidate_.type() != CV_8U) {
        candidate_ = cv::Mat::zeros(height, width, CV_8U);
    } else {
        candidate_.setTo(0);
    }

    int cand = 0;
    for (int y = box.y; y < box.y + box.height; ++y) {
        const float* row = depth + static_cast<size_t>(y) * width;
        uint8_t* crow = candidate_.ptr<uint8_t>(y);
        for (int x = box.x; x < box.x + box.width; ++x) {
            const float z = row[x];
            if (!std::isfinite(z) || z <= 0.f) continue;
            if (std::abs(z - seedDepth) <= tolerance) {
                crow[x] = 255;
                ++cand;
            }
        }
    }
    stats.candidatePixels = cand;
    if (cand < kMinMaskArea) {
        stats.rejectReason = "candidate too small";
        return false;
    }

    // ---- 只保留 seed 所在的连通域 ----
    cv::Mat candRoi = candidate_(box);
    const int n = cv::connectedComponentsWithStats(candRoi, labels_, statsArr_, centroids_,
                                                   8, CV_32S);
    if (n <= 1) {
        stats.rejectReason = "no connected component";
        return false;
    }

    const int sx = seed.x - box.x;
    const int sy = seed.y - box.y;
    int chosen = -1;
    {
        // seed 周围 7x7 里出现次数最多的 label（seed 本身可能落在候选空洞上）
        std::vector<int> hist(static_cast<size_t>(n), 0);
        for (int y = sy - kSeedRadius; y <= sy + kSeedRadius; ++y) {
            if (y < 0 || y >= labels_.rows) continue;
            const int* lrow = labels_.ptr<int>(y);
            for (int x = sx - kSeedRadius; x <= sx + kSeedRadius; ++x) {
                if (x < 0 || x >= labels_.cols) continue;
                const int l = lrow[x];
                if (l > 0 && l < n) hist[static_cast<size_t>(l)]++;
            }
        }
        int best = 0;
        for (int l = 1; l < n; ++l) {
            if (hist[static_cast<size_t>(l)] > best) {
                best = hist[static_cast<size_t>(l)];
                chosen = l;
            }
        }
    }
    if (chosen < 0) {
        // seed 附近没有候选像素（例如 seed 落在目标的深度空洞里）：
        // 退化成「最大的那个连通域」。这仍然局限在 searchBox 内，
        // 不会把画面上别的物体圈进来。
        int bestArea = 0;
        for (int l = 1; l < n; ++l) {
            const int area = statsArr_.at<int>(l, cv::CC_STAT_AREA);
            if (area > bestArea) {
                bestArea = area;
                chosen = l;
            }
        }
        if (chosen < 0) {
            stats.rejectReason = "no usable component";
            return false;
        }
    }

    int area = statsArr_.at<int>(chosen, cv::CC_STAT_AREA);
    if (area < kMinMaskArea) {
        stats.rejectReason = "component too small";
        return false;
    }

    // ---- 写回 mask（ROI 内）并做一次小 close 补细缝 ----
    cv::Mat maskRoi = mask(box);
    maskRoi.setTo(0);
    maskRoi.setTo(255, labels_ == chosen);

    if (kernel_.empty()) {
        kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    }
    cv::morphologyEx(maskRoi, maskRoi, cv::MORPH_CLOSE, kernel_);

    area = cv::countNonZero(maskRoi);
    if (area < kMinMaskArea) {
        stats.rejectReason = "mask too small after close";
        return false;
    }

    const cv::Vec2d c = centroids_.at<cv::Vec2d>(chosen);
    stats.area = area;
    stats.centerX = static_cast<float>(c[0]) + static_cast<float>(box.x);
    stats.centerY = static_cast<float>(c[1]) + static_cast<float>(box.y);
    stats.valid = true;
    stats.rejectReason = "ok";
    return true;
}
