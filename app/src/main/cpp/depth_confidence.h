#pragma once
#include "depth_calib.h"
#include <algorithm>
#include <cmath>
#include <vector>
static constexpr float kTemporalWeightAgree = 1.0f;
static constexpr float kTemporalWeightUntested = 0.85f;
static constexpr float kTemporalWeightDisagree = 0.f;
static constexpr float kEdgePenalty = 6.0f;   // 相对深度跳变 -> 降权强度
static constexpr float kEdgeWeightMin = 0.25f; // 边缘像素权重下限（防止过滤过度）

/**
 * 生成逐像素融合权重图。
 *
 * @param temporalMask  可为 nullptr（首帧或位姿不可用），此时只用边缘因子。
 * @param out           长度 w*h 的输出缓冲，会被全部重写。
 */
inline void buildPixelWeight(const float* depth, int w, int h,
                             const uint8_t* temporalMask,
                             std::vector<float>& out, const float* sourceConfidence = nullptr) {
    if (w <= 0 || h <= 0 || w > 4096 || h > 4096) { out.clear(); return; }
    const size_t n = static_cast<size_t>(w) * h;
    out.assign(n, 0.f);
    if (!depth) return;
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const float z = depth[row + x];
            if (!(z > 0.05f) || !std::isfinite(z)) {
                out[row + x] = 0.f;
                continue;
            }
            // ---- 边缘因子：与四邻域的相对深度跳变 ----
            float gsum = 0.f;
            int gn = 0;
            const int dxs[4] = {1, -1, 0, 0};
            const int dys[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int nx = x + dxs[k];
                const int ny = y + dys[k];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                const float zn = depth[static_cast<size_t>(ny) * w + nx];
                if (!(zn > 0.05f) || !std::isfinite(zn)) continue;
                gsum += std::fabs(zn - z);
                ++gn;
            }
            float wEdge = 1.f;
            if (gn > 0) {
                const float relJump = (gsum / static_cast<float>(gn)) / z;
                if (std::isfinite(relJump)) {
                    wEdge = 1.f / (1.f + relJump * kEdgePenalty);
                    if (wEdge < kEdgeWeightMin) wEdge = kEdgeWeightMin;
                }
            }
            // ---- 时序因子 ----
            float wT = kTemporalWeightUntested;
            if (temporalMask) {
                switch (temporalMask[row + x]) {
                    case kTemporalAgree:    wT = kTemporalWeightAgree; break;
                    case kTemporalDisagree: wT = kTemporalWeightDisagree; break;
                    default:                wT = kTemporalWeightUntested; break;
                }
            }
            const float source = sourceConfidence ? sourceConfidence[row+x] : 1.f;
            const float wv = std::isfinite(source) ? wT * wEdge * std::clamp(source, 0.f, 1.f) : 0.f;
            out[row + x] = (wv < 0.02f) ? 0.f : wv;
        }
    }
}

