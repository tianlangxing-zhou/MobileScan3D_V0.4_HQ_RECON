#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace depth_refinement {
// Inverse-depth q=0 is valid, so represent absent source evidence with NaN.
// Sanitize before filtering/calibration; final fusion weighting alone is too late.
inline void applySourceConfidence(float* depth, float* confidence, size_t count) {
    if (!depth || !confidence) return;
    for (size_t i=0; i<count; ++i) {
        if (!std::isfinite(confidence[i]) || confidence[i] <= 0.f) {
            depth[i] = std::numeric_limits<float>::quiet_NaN();
            confidence[i] = 0.f;
        } else confidence[i] = std::min(1.f, confidence[i]);
    }
}
inline bool valid(float v, bool inverse) {
    // Relative inverse models may legitimately emit zero/negative q.
    return std::isfinite(v) && (inverse || v > .05f);
}
// Confidence-aware 3x3 bilateral-like filter.
//
// KinectFusion-style depth preprocessing benefits from spatial smoothing while
// preserving discontinuities. Keep the support deliberately small for mobile:
// - no hole filling;
// - no temporal blending/ghosting;
// - range Tukey kernel rejects cross-edge samples;
// - source-confidence downweights weak network pixels;
// - representation/scale is unchanged.
inline void spatial(const float* input, int w, int h, bool inverse,
                    const float* confidence,
                    std::vector<float>& output) {
    if (!input || w <= 0 || h <= 0 || w > 4096 || h > 4096) {
        output.clear();
        return;
    }
    output.assign(input, input + size_t(w) * h);
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const size_t i = size_t(y) * w + x;
            const float center = input[i];
            if (!valid(center, inverse)) continue;

            const float tolerance =
                std::max(inverse ? .01f : .004f, std::fabs(center) * .025f);
            const float centerConf =
                confidence && std::isfinite(confidence[i])
                    ? std::clamp(confidence[i], 0.f, 1.f)
                    : 1.f;

            // Keep the center influential so the filter never behaves like a
            // neighborhood hole filler around weakly supported silhouettes.
            double sum = double(center) * (1.5 + centerConf);
            double weight = 1.5 + centerConf;
            int support = 0;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const size_t ni = size_t(y + dy) * w + (x + dx);
                    const float v = input[ni];
                    if (!valid(v, inverse)) continue;

                    const float diff = std::fabs(v - center);
                    if (diff >= tolerance) continue;
                    const float u = diff / tolerance;
                    // Tukey-like compact range kernel: zero influence at edge.
                    const float oneMinus = 1.f - u * u;
                    const float rangeW = oneMinus * oneMinus;
                    const float spatialW = (dx == 0 || dy == 0) ? 1.f : 0.70710678f;
                    const float confW =
                        confidence && std::isfinite(confidence[ni])
                            ? std::clamp(confidence[ni], 0.f, 1.f)
                            : 1.f;
                    const float k = spatialW * rangeW * confW;
                    if (k <= 0.02f) continue;
                    sum += double(k) * v;
                    weight += k;
                    ++support;
                }
            }
            if (support >= 2 && weight > 1e-6) {
                output[i] = float(sum / weight);
            }
        }
    }
}

inline void spatial(const float* input, int w, int h, bool inverse,
                    std::vector<float>& output) {
    spatial(input, w, h, inverse, nullptr, output);
}
// Fit only spatially coherent sparse observations. Median rejects isolated bad
// pixels; a range gate avoids assigning a foreground edge to a background ray.
inline bool calibrationSample(const float* d,int w,int h,int x,int y,bool inverse,float& out) {
    if (!d || w<=0 || h<=0 || x<0 || y<0 || x>=w || y>=h) return false;
    const float center=d[size_t(y)*w+x];
    if (!valid(center,inverse)) return false;
    float values[9];int n=0;
    for(int yy=std::max(0,y-1);yy<=std::min(h-1,y+1);++yy)
        for(int xx=std::max(0,x-1);xx<=std::min(w-1,x+1);++xx) {
            const float v=d[size_t(yy)*w+xx];if(valid(v,inverse))values[n++]=v;
        }
    if(n<5)return false;
    std::sort(values,values+n);
    const float median=values[n/2];
    const float tolerance=std::max(inverse?.02f:.008f,std::fabs(median)*.06f);
    if(values[n-2]-values[1]>tolerance || std::fabs(center-median)>tolerance)return false;
    out=median;return true;
}
}
