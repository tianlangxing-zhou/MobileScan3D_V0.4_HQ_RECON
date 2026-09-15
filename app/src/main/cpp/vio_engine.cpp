#include "vio_engine.h"

#include <algorithm>
#include <cmath>

void VioEngine::reset() {
    *this = VioEngine();
}

void VioEngine::imu(uint64_t t, float ax, float ay, float az, float gx, float gy, float gz) {
    if (lastImu_) {
        float dt = std::clamp((t - lastImu_) * 1e-9f, 0.f, 0.05f);
        yaw_ += gz * dt;
        pitch_ += gy * dt;
        roll_ += gx * dt;
    }
    lastImu_ = t;
    ax_ = ax;
    ay_ = ay;
    az_ = az;
}

void VioEngine::frame(uint64_t t, const uint8_t* y, int w, int h, int stride) {
    frames_++;
    if (!y || w < 8 || h < 8) {
        return;
    }
    if (stride < w) {
        stride = w; // rowStride 可能带对齐 padding，默认按 packed 处理
    }
    const int step = 8;
    double sx = 0, sy = 0, sw = 0;
    uint32_t corners = 0;
    for (int yy = step; yy < h - step; yy += step) {
        for (int x = step; x < w - step; x += step) {
            int gx = int(y[yy * stride + x + 1]) - int(y[yy * stride + x - 1]);
            int gy = int(y[(yy + 1) * stride + x]) - int(y[(yy - 1) * stride + x]);
            int mag = std::abs(gx) + std::abs(gy);
            if (mag > 45) {
                corners++;
                float ww = float(mag);
                sx += x * ww;
                sy += yy * ww;
                sw += ww;
            }
        }
    }
    features_ = corners;
    float cx = sw > 0 ? sx / sw : w * .5f;
    float cy = sw > 0 ? sy / sw : h * .5f;

    float dyaw = std::abs(yaw_ - prevYaw_);
    prevYaw_ = yaw_;

    if (havePrev_) {
        float dx = (cx - prevCx_) / float(w);
        float dy = (cy - prevCy_) / float(h);
        novelty_ = std::clamp(std::sqrt(dx * dx + dy * dy) * 12.f + dyaw * 8.f, 0.f, 1.f);
        tx_ += -dx * 0.45f;
        ty_ += dy * 0.45f;
        tz_ = std::clamp(tz_ + (std::abs(dx) + std::abs(dy)) * 0.05f, -1.f, 1.f);
    } else {
        havePrev_ = true;
        novelty_ = 1.f;
    }
    prevCx_ = cx;
    prevCy_ = cy;
    lastFrame_ = t;
}
