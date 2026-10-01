#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace camera_distance {
struct Estimate { float meters = 0, coverage = 0, spread = 0; int samples = 0; };
// Bounded stack storage: never sort/copy a whole depth image for camera selection.
inline Estimate estimate(const float* depth, int w, int h, float worldPerMeter,
                         float x0=.3f, float y0=.3f, float x1=.7f, float y1=.7f,
                         const float* confidence=nullptr) {
    Estimate out;
    if (!depth || w < 2 || h < 2 || !std::isfinite(worldPerMeter) || worldPerMeter <= 0 ||
        !std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1)) return out;
    int left=std::clamp(int(std::clamp(x0,0.f,1.f)*w),0,w-1);
    int top=std::clamp(int(std::clamp(y0,0.f,1.f)*h),0,h-1);
    int right=std::clamp(int(std::clamp(x1,0.f,1.f)*w),0,w-1);
    int bottom=std::clamp(int(std::clamp(y1,0.f,1.f)*h),0,h-1);
    if (right-left < 4 || bottom-top < 4) return out;
    std::array<float,289> samples{};
    int n=0, total=0;
    const int nx=std::min(17,right-left+1), ny=std::min(17,bottom-top+1);
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        int x=left+i*(right-left)/(nx-1), y=top+j*(bottom-top)/(ny-1);
        size_t k=size_t(y)*w+x; ++total;
        if(confidence && (!std::isfinite(confidence[k]) || confidence[k]<.35f)) continue;
        float z=depth[k]/worldPerMeter;
        if(std::isfinite(z) && z>=.08f && z<=20.f) samples[n++]=z;
    }
    out.samples=n; out.coverage=float(n)/total;
    if(n<24 || out.coverage<.35f) return out;
    std::sort(samples.begin(),samples.begin()+n);
    const float median=samples[n/2];
    out.spread=(samples[(n-1)*9/10]-samples[(n-1)/10])/median;
    if(out.spread<=.65f) out.meters=median;
    return out;
}
}
