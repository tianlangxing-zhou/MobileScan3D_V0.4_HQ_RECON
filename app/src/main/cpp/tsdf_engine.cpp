#include "tsdf_engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>

void TsdfEngine::reset() {
    voxels_.clear();
}

void TsdfEngine::integrateDepth(
        const float* depth,
        int w,
        int h,
        const uint8_t* rgb,
        int rgbW,
        int rgbH,
        float fx,
        float fy,
        float cx,
        float cy,
        const float R[9],
        const float t[3],
        float confidence) {
    if (!depth || !rgb || w < 4 || h < 4 || rgbW <= 0 || rgbH <= 0) {
        return;
    }
    const int step = 4;
    const float trunc = voxel_ * 3.f;
    float wt = std::clamp(confidence, 0.05f, 1.f);
    for (int y = step; y < h; y += step) {
        for (int x = step; x < w; x += step) {
            if (voxels_.size() > 1500000) {
                return;
            }
            float z = depth[y * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            float Xc = (x - cx) * z / fx;
            float Yc = (y - cy) * z / fy;
            float Zc = z;
            float Xw = R[0] * Xc + R[1] * Yc + R[2] * Zc + t[0];
            float Yw = R[3] * Xc + R[4] * Yc + R[5] * Zc + t[1];
            float Zw = R[6] * Xc + R[7] * Yc + R[8] * Zc + t[2];
            int vx = (int)std::floor(Xw / voxel_);
            int vy = (int)std::floor(Yw / voxel_);
            int vz = (int)std::floor(Zw / voxel_);
            int sx = std::min(rgbW - 1, x * rgbW / w);
            int sy = std::min(rgbH - 1, y * rgbH / h);
            size_t ci = ((size_t)sy * rgbW + sx) * 3;
            // 按 RGB 三通道独立加权更新（原来 g/b 直接复制 r，导出的 PLY 只能是灰度）
            const float cr = rgb[ci];
            const float cg = rgb[ci + 1];
            const float cb = rgb[ci + 2];
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        Key k{vx + dx, vy + dy, vz + dz};
                        TsdfVoxel& v = voxels_[k];
                        float sdf = std::clamp((float)std::sqrt(dx * dx + dy * dy + dz * dz) * voxel_ / trunc, 0.f, 1.f);
                        v.sdf = (v.sdf * v.weight + sdf * wt) / (v.weight + wt);
                        v.weight = std::min(32.f, v.weight + wt);
                        float iw = 1.f / (v.weight + 1e-6f);
                        const float wOld = v.weight - wt;
                        v.r = (v.r * wOld + cr * wt) * iw;
                        v.g = (v.g * wOld + cg * wt) * iw;
                        v.b = (v.b * wOld + cb * wt) * iw;
                    }
                }
            }
        }
    }
}

bool TsdfEngine::exportPly(const std::string& path) const {
    std::ofstream f(path);
    if (!f) {
        return false;
    }
    size_t n = 0;
    for (auto& p : voxels_) {
        if (p.second.weight > 1.5f && p.second.sdf < 0.35f) {
            n++;
        }
    }
    f << "ply\nformat ascii 1.0\nelement vertex " << n
      << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (auto& p : voxels_) {
        if (p.second.weight > 1.5f && p.second.sdf < 0.35f) {
            auto k = p.first;
            auto v = p.second;
            f << k.x * voxel_ << ' ' << k.y * voxel_ << ' ' << k.z * voxel_ << ' '
              << (int)std::clamp(v.r, 0.f, 255.f) << ' '
              << (int)std::clamp(v.g, 0.f, 255.f) << ' '
              << (int)std::clamp(v.b, 0.f, 255.f) << '\n';
        }
    }
    return true;
}
