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
// Bounded 3x3 range-aware filter. No hole filling, no temporal ghosting and no
// normalization: calibrated q retains its original representation and scale.
inline void spatial(const float* input, int w, int h, bool inverse,
                    std::vector<float>& output) {
    if (!input || w <= 0 || h <= 0 || w > 4096 || h > 4096) { output.clear(); return; }
    output.assign(input, input + size_t(w)*h);
    for (int y=1; y<h-1; ++y) for (int x=1; x<w-1; ++x) {
        const size_t i=size_t(y)*w+x;
        const float center=input[i];
        if (!valid(center,inverse)) continue;
        // Narrow support preserves depth discontinuities and thin structures.
        const float tolerance=std::max(inverse ? .01f : .004f, std::fabs(center)*.025f);
        double sum=2.*center, weight=2.;
        int support=0;
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
            if (dx==0 && dy==0) continue;
            const float v=input[size_t(y+dy)*w+x+dx];
            if (!valid(v,inverse) || std::fabs(v-center)>tolerance) continue;
            const float range=1.f-std::fabs(v-center)/tolerance;
            const float k=(dx==0 || dy==0 ? 1.f : .5f)*range;
            sum+=k*v;weight+=k;++support;
        }
        if (support>=3) output[i]=float(sum/weight);
    }
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
