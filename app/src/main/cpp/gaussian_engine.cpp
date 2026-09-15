#include "gaussian_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

void GaussianEngine::reset() {
    g_.clear();
    index_.clear();
    stable_ = 0;
    merged_ = 0;
}

void GaussianEngine::ingestPoint(
        float x,
        float y,
        float z,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        float confidence) {
    if (!std::isfinite(z) || z < 0.05f || z > 8.f || confidence < 0.05f) {
        return;
    }

    const float cell = 0.01f;
    Key k{
        (int)std::floor(x / cell),
        (int)std::floor(y / cell),
        (int)std::floor(z / cell)
    };

    auto it = index_.find(k);
    if (it != index_.end()) {
        Gaussian& a = g_[it->second];
        const float w = std::min(0.5f, confidence);
        const float iw = 1.f / (1.f + w);
        a.px = (a.px + x * w) * iw;
        a.py = (a.py + y * w) * iw;
        a.pz = (a.pz + z * w) * iw;
        a.r = (uint8_t)((a.r + r * w) * iw);
        a.g = (uint8_t)((a.g + g * w) * iw);
        a.b = (uint8_t)((a.b + b * w) * iw);
        if (a.hits < 65535) {
            a.hits++;
        }
        if (a.hits >= 3 && a.state < 2) {
            a.state = 2;
            stable_++;
        }
        merged_++;
        return;
    }

    if (g_.size() >= 600000) {
        return;
    }

    Gaussian a{};
    a.px = x;
    a.py = y;
    a.pz = z;
    a.sx = a.sy = a.sz = cell * 0.8f;
    a.qx = 0.f;
    a.qy = 0.f;
    a.qz = 0.f;
    a.qw = 1.f;
    a.r = r;
    a.g = g;
    a.b = b;
    a.opacity = (uint8_t)(confidence * 255.f);
    a.hits = 1;
    a.state = 0;
    index_[k] = g_.size();
    g_.push_back(a);
}

size_t GaussianEngine::count() const { return g_.size(); }
size_t GaussianEngine::stableCount() const { return stable_; }
size_t GaussianEngine::mergedCount() const { return merged_; }

size_t GaussianEngine::copyPoints(float* out, size_t maxPoints) const {
    const size_t n = g_.size();
    if (n == 0 || out == nullptr || maxPoints == 0) {
        return 0;
    }
    const size_t step = n > maxPoints ? (n + maxPoints - 1) / maxPoints : 1;
    size_t written = 0;
    for (size_t i = 0; i < n && written < maxPoints; i += step) {
        const Gaussian& a = g_[i];
        float* p = out + written * 6;
        p[0] = a.px;
        p[1] = a.py;
        p[2] = a.pz;
        p[3] = a.r / 255.0f;
        p[4] = a.g / 255.0f;
        p[5] = a.b / 255.0f;
        written++;
    }
    return written;
}

void GaussianEngine::boundingBox(float* minX, float* minY, float* minZ,
                                 float* maxX, float* maxY, float* maxZ) const {
    if (!minX || !minY || !minZ || !maxX || !maxY || !maxZ) {
        return;
    }
    *minX = *minY = *minZ = std::numeric_limits<float>::max();
    *maxX = *maxY = *maxZ = -std::numeric_limits<float>::max();
    for (const Gaussian& a : g_) {
        *minX = std::min(*minX, a.px);
        *minY = std::min(*minY, a.py);
        *minZ = std::min(*minZ, a.pz);
        *maxX = std::max(*maxX, a.px);
        *maxY = std::max(*maxY, a.py);
        *maxZ = std::max(*maxZ, a.pz);
    }
    if (g_.empty()) {
        *minX = *minY = *minZ = 0.f;
        *maxX = *maxY = *maxZ = 0.f;
    }
}

void GaussianEngine::centroid(float* x, float* y, float* z) const {
    if (!x || !y || !z) {
        return;
    }
    *x = *y = *z = 0.f;
    if (g_.empty()) {
        return;
    }
    for (const Gaussian& a : g_) {
        *x += a.px;
        *y += a.py;
        *z += a.pz;
    }
    const float inv = 1.f / (float)g_.size();
    *x *= inv;
    *y *= inv;
    *z *= inv;
}
