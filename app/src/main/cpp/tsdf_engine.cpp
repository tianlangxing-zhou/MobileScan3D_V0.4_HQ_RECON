#include "tsdf_engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>

// ============================================================================
//  定点量化辅助
// ============================================================================

static inline int16_t quantTsdf(float v) {
    const float q = v * kTsdfValueScale;
    if (q >= 32767.f) return (int16_t)32767;
    if (q <= -32767.f) return (int16_t)-32767;
    return (int16_t)std::lround(q);
}

static inline uint16_t quantWeight(float w) {
    const float q = w * kTsdfWeightScale;
    if (q <= 0.f) return 0;
    if (q >= 65535.f) return (uint16_t)65535;
    return (uint16_t)std::lround(q);
}

static inline float dequantWeight(uint16_t w) {
    return static_cast<float>(w) / kTsdfWeightScale;
}

static inline uint16_t packRgb565(uint8_t r, uint8_t g, uint8_t b) {
    return (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}

static inline void unpackRgb565(uint16_t c, int& r, int& g, int& b) {
    r = (c >> 11) & 0x1F;
    g = (c >> 5) & 0x3F;
    b = c & 0x1F;
}

/** 在 565 空间做加权平均（解码 -> 浮点 -> 再编码）。 */
static inline uint16_t blendRgb565(uint16_t a, uint16_t b, float wa, float wb) {
    const float inv = 1.f / std::max(1e-6f, wa + wb);
    int ar, ag, ab, br, bg, bb;
    unpackRgb565(a, ar, ag, ab);
    unpackRgb565(b, br, bg, bb);
    const int r = (int)std::lround((ar * wa + br * wb) * inv);
    const int g = (int)std::lround((ag * wa + bg * wb) * inv);
    const int bl = (int)std::lround((ab * wa + bb * wb) * inv);
    return (uint16_t)(((r & 0x1F) << 11) | ((g & 0x3F) << 5) | (bl & 0x1F));
}

// ============================================================================
//  生命周期
// ============================================================================

void TsdfEngine::reset() {
    blocks_.clear();
    liveVoxels_ = 0;
    liveColoredVoxels_ = 0;
    hasBounds_ = false;
    minV_[0] = minV_[1] = minV_[2] = 0;
    maxV_[0] = maxV_[1] = maxV_[2] = 0;
}

void TsdfEngine::setVoxelSize(float meters) {
    if (!std::isfinite(meters) || meters < 1e-4f) return;
    voxel_ = std::min(meters, 0.5f);
    trunc_ = voxel_ * kTsdfTruncVoxels;
}

void TsdfEngine::setTruncation(float meters) {
    if (!std::isfinite(meters) || meters < 1e-4f) return;
    trunc_ = std::min(meters, 1.0f);
}

// ============================================================================
//  块 / 体素寻址
// ============================================================================

TsdfBlock* TsdfEngine::blockFor(int bx, int by, int bz, bool create) {
    const BlockKey k{bx, by, bz};
    auto it = blocks_.find(k);
    if (it != blocks_.end()) {
        return &it->second;
    }
    if (!create || blocks_.size() >= maxBlocks_) {
        return nullptr;
    }
    auto res = blocks_.emplace(k, TsdfBlock{});
    return &res.first->second;
}

TsdfVoxel* TsdfEngine::voxelFor(int vx, int vy, int vz, bool create) {
    // C++20 起有符号右移是算术右移（等价于 floor 除以 8）,负数下标也正确。
    const int bx = vx >> kTsdfBlockShift;
    const int by = vy >> kTsdfBlockShift;
    const int bz = vz >> kTsdfBlockShift;
    TsdfBlock* b = blockFor(bx, by, bz, create);
    if (!b) {
        return nullptr;
    }
    const int lx = vx & kTsdfBlockMask;
    const int ly = vy & kTsdfBlockMask;
    const int lz = vz & kTsdfBlockMask;
    return &b->voxels[(lz * kTsdfBlockSize + ly) * kTsdfBlockSize + lx];
}

int16_t TsdfEngine::tsdfAt(int vx, int vy, int vz) const {
    TsdfVoxel* v = const_cast<TsdfEngine*>(this)->voxelFor(vx, vy, vz, false);
    if (!v || v->weight == 0) {
        return kEmptyTsdf;
    }
    return v->tsdf;
}

uint16_t TsdfEngine::weightAt(int vx, int vy, int vz) const {
    TsdfVoxel* v = const_cast<TsdfEngine*>(this)->voxelFor(vx, vy, vz, false);
    return v ? v->weight : 0;
}

uint16_t TsdfEngine::color565At(int vx, int vy, int vz) const {
    TsdfVoxel* v = const_cast<TsdfEngine*>(this)->voxelFor(vx, vy, vz, false);
    return v ? v->color565 : 0;
}
uint16_t TsdfEngine::colorWeightAt(int vx, int vy, int vz) const {
    TsdfVoxel* v = const_cast<TsdfEngine*>(this)->voxelFor(vx, vy, vz, false);
    return v ? v->colorWeight : 0;
}

bool TsdfEngine::allocated(int vx, int vy, int vz) const {
    return const_cast<TsdfEngine*>(this)->voxelFor(vx, vy, vz, false) != nullptr;
}

const TsdfBlock* TsdfEngine::blockAt(int bx, int by, int bz) const {
    const BlockKey k{bx, by, bz};
    auto it = blocks_.find(k);
    return (it == blocks_.end()) ? nullptr : &it->second;
}

void TsdfEngine::forEachBlock(
        const std::function<void(int, int, int, const TsdfBlock*)>& fn) const {
    for (const auto& kv : blocks_) {
        fn(kv.first.x, kv.first.y, kv.first.z, &kv.second);
    }
}

bool TsdfEngine::voxelBounds(int mn[3], int mx[3]) const {
    if (!hasBounds_) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        mn[i] = minV_[i];
        mx[i] = maxV_[i];
    }
    return true;
}

// ============================================================================
//  融合
// ============================================================================

void TsdfEngine::integrateDepth(const float* depth, int w, int h,
                                const uint8_t* rgb, int rgbW, int rgbH,
                                float fx, float fy, float cx, float cy,
                                const float R[9], const float t[3], float confidence,
                                float depthScale, float depthShift) {
    if (!depth || w < 4 || h < 4 || !R || !t) {
        return;
    }
    if (atCapacity()) {
        return;
    }

    const int step = pixelStep_ < 1 ? 1 : pixelStep_;
    const float mu = (trunc_ > 1e-4f) ? trunc_ : (voxel_ * kTsdfTruncVoxels);
    const float invMu = 1.f / mu;
    const float baseWeight = std::clamp(confidence, 0.05f, 1.f);
    const bool haveColor = (rgb != nullptr) && rgbW > 0 && rgbH > 0;
    const float invFx = 1.f / (std::fabs(fx) > 1e-3f ? fx : 1.f);
    const float invFy = 1.f / (std::fabs(fy) > 1e-3f ? fy : 1.f);
    // 颜色带：只有非常接近表面（|sdf| < 0.5）的体素才吃颜色。
    const float colorBand = 0.5f;

    for (int y = 0; y < h; y += step) {
        const float* drow = depth + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x += step) {
            const float raw = drow[x];
            if (!std::isfinite(raw) || raw <= 0.f) {
                continue;
            }
            // 深度标定：z' = z * scale + shift（默认 1/0 即不修正）。
            const float z = raw * depthScale + depthShift;
            if (!std::isfinite(z) || z < minDepth_ || z > maxDepth_) {
                continue;
            }

            const float xn = (static_cast<float>(x) - cx) * invFx;
            const float yn = (static_cast<float>(y) - cy) * invFy;

            // 沿射线推进时把步长按射线方向长度归一化（dzStep = voxel / |ray|），
            // 否则掠射角下相邻采样点会跳过体素，表面上会出现空洞与条纹。
            const float rayLen = std::sqrt(1.f + xn * xn + yn * yn);
            float dzStep = voxel_ / (rayLen > 1e-4f ? rayLen : 1.f);
            if (dzStep < 1e-5f) {
                dzStep = 1e-5f;
            }

            float z0 = z - mu;
            if (z0 < minDepth_) {
                z0 = minDepth_;
            }
            const float z1 = z + mu;

            uint16_t obsColor = 0;
            if (haveColor) {
                const int sx = std::min(rgbW - 1, x * rgbW / w);
                const int sy = std::min(rgbH - 1, y * rgbH / h);
                const size_t ci = (static_cast<size_t>(sy) * rgbW + sx) * 3;
                obsColor = packRgb565(rgb[ci], rgb[ci + 1], rgb[ci + 2]);
            }

            int guard = 0;
            for (float zi = z0; zi <= z1; zi += dzStep) {
                if (++guard > 4096) {
                    break;
                }
                const float xc = xn * zi;
                const float yc = yn * zi;
                const float Xw = R[0] * xc + R[1] * yc + R[2] * zi + t[0];
                const float Yw = R[3] * xc + R[4] * yc + R[5] * zi + t[1];
                const float Zw = R[6] * xc + R[7] * yc + R[8] * zi + t[2];
                if (!std::isfinite(Xw) || !std::isfinite(Yw) || !std::isfinite(Zw)) {
                    continue;
                }
                const int vx = (int)std::floor(Xw / voxel_);
                const int vy = (int)std::floor(Yw / voxel_);
                const int vz = (int)std::floor(Zw / voxel_);

                TsdfVoxel* v = voxelFor(vx, vy, vz, true);
                if (!v) {
                    // 块配额用尽：整帧停止融合，避免一边分配失败一边空转。
                    return;
                }

                // 带符号：zi < z -> 正（外侧）; zi > z -> 负（内侧）。
                float sdf = (z - zi) * invMu;
                sdf = sdf < -1.f ? -1.f : (sdf > 1.f ? 1.f : sdf);

                const float wOld = dequantWeight(v->weight);
                const float wSum = wOld + baseWeight;
                const float merged = (wOld > 0.f)
                    ? (v->tsdf / kTsdfValueScale * wOld + sdf * baseWeight) / wSum
                    : sdf;
                v->tsdf = quantTsdf(merged);
                const float satur = std::min(wSum, kTsdfMaxWeight);
                const bool wasEmpty = (v->weight == 0);
                v->weight = quantWeight(satur);
                if (wasEmpty) {
                    ++liveVoxels_;
                }

                if (haveColor && std::fabs(sdf) < colorBand) {
                    const float colorOld = dequantWeight(v->colorWeight);
                    const bool hadColor = v->colorWeight != 0;
                    if (!hadColor) {
                        v->color565 = obsColor;
                        ++liveColoredVoxels_;
                    } else {
                        v->color565 = blendRgb565(
                            v->color565, obsColor, colorOld, baseWeight);
                    }
                    v->colorWeight = quantWeight(
                        std::min(colorOld + baseWeight, kTsdfMaxWeight));
                }

                if (!hasBounds_) {
                    hasBounds_ = true;
                    minV_[0] = maxV_[0] = vx;
                    minV_[1] = maxV_[1] = vy;
                    minV_[2] = maxV_[2] = vz;
                } else {
                    if (vx < minV_[0]) minV_[0] = vx; else if (vx > maxV_[0]) maxV_[0] = vx;
                    if (vy < minV_[1]) minV_[1] = vy; else if (vy > maxV_[1]) maxV_[1] = vy;
                    if (vz < minV_[2]) minV_[2] = vz; else if (vz > maxV_[2]) maxV_[2] = vz;
                }
            }
        }
    }
}

// ============================================================================
//  兼容旧导出：只有顶点的 PLY（表面附近的体素）
// ============================================================================

bool TsdfEngine::exportPly(const std::string& path) const {
    struct Item {
        int x, y, z;
        int r, g, b;
    };
    std::vector<Item> out;
    out.reserve(static_cast<size_t>(std::min<uint64_t>(liveVoxels_, 300000)));

    for (const auto& kv : blocks_) {
        const int bx = kv.first.x << kTsdfBlockShift;
        const int by = kv.first.y << kTsdfBlockShift;
        const int bz = kv.first.z << kTsdfBlockShift;
        for (int lz = 0; lz < kTsdfBlockSize; ++lz) {
            for (int ly = 0; ly < kTsdfBlockSize; ++ly) {
                for (int lx = 0; lx < kTsdfBlockSize; ++lx) {
                    const TsdfVoxel& v =
                        kv.second.voxels[(lz * kTsdfBlockSize + ly) * kTsdfBlockSize + lx];
                    if (v.weight == 0) {
                        continue;
                    }
                    const float val = v.tsdf / kTsdfValueScale;
                    // 「表面附近」= 归一化 |tsdf| 小；这是 sdf 过零带。
                    if (std::fabs(val) > 0.2f) {
                        continue;
                    }
                    int cr, cg, cb;
                    if (v.colorWeight == 0) {
                        cr = 19;
                        cg = 38;
                        cb = 19;
                    } else {
                        unpackRgb565(v.color565, cr, cg, cb);
                    }
                    const int x = bx + lx;
                    const int y = by + ly;
                    const int z = bz + lz;
                    out.push_back(Item{x, y, z,
                                       (cr * 255 + 15) / 31,
                                       (cg * 255 + 31) / 63,
                                       (cb * 255 + 15) / 31});
                }
            }
        }
    }

    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << "ply\nformat ascii 1.0\nelement vertex " << out.size()
      << "\nproperty float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (const Item& it : out) {
        f << it.x * voxel_ << ' ' << it.y * voxel_ << ' ' << it.z * voxel_ << ' '
          << it.r << ' ' << it.g << ' ' << it.b << '\n';
    }
    return true;
}
