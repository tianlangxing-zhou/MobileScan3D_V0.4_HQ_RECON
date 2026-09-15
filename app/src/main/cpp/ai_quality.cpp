#include "ai_quality.h"

#include <algorithm>
#include <cmath>

void AiQualityEngine::reset() {
    q_ = {};
}

void AiQualityEngine::analyze(const uint8_t* y, int w, int h, int stride) {
    if (!y || w < 3 || h < 3) {
        return;
    }
    if (stride < w) {
        stride = w; // rowStride 可能带对齐 padding
    }
    double mean = 0;
    double var = 0;
    const int step = 4;
    long n = 0;
    long edges = 0;
    for (int yy = 1; yy < h - 1; yy += step) {
        for (int x = 1; x < w - 1; x += step) {
            int i = yy * stride + x;
            mean += y[i];
            n++;
            int g = std::abs(int(y[i + 1]) - int(y[i - 1])) +
                    std::abs(int(y[i + stride]) - int(y[i - stride]));
            if (g > 60) {
                edges++;
            }
        }
    }
    mean /= std::max(1L, n);
    for (int yy = 1; yy < h - 1; yy += step) {
        for (int x = 1; x < w - 1; x += step) {
            float d = y[yy * stride + x] - mean;
            var += d * d;
        }
    }
    var /= std::max(1L, n);
    // Edge-density based sharpness: textured scenes score high even when most of
    // the frame is a smooth background.
    q_.sharpness = std::min(1.f, float(edges) / (std::max(1L, n) * 0.06f));
    q_.exposure = std::max(0.f, 1.f - std::abs(float(mean) - 128.f) / 128.f);
    q_.motion = 1.f - q_.sharpness * 0.35f;
    q_.geometry = 0.0f;
    q_.completion = 0.0f;
}
