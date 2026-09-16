#include "depth_calib.h"

#include <algorithm>
#include <cmath>

// ============================================================================
//  基础统计
// ============================================================================

float medianOf(std::vector<float> v) {
    if (v.empty()) {
        return 0.f;
    }
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float hi = v[mid];
    if (v.size() % 2 == 1) {
        return hi;
    }
    // 偶数个：再取一次较小的一半里的最大值
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.begin() + mid);
    const float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

namespace {

struct Pair {
    float x;   // 自变量（模型输出 d 或 1/d 归一化前的 d）
    float y;   // 因变量（z 或 1/z）
    float w;   // 稳健权重
};

/** 加权最小二乘 y = a*x + b。返回 false 表示退化（x 方差过小）。 */
bool weightedLineFit(const std::vector<Pair>& pts, float* outA, float* outB) {
    double sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
    for (const Pair& p : pts) {
        const double w = p.w;
        sw += w;
        swx += w * p.x;
        swy += w * p.y;
        swxx += w * static_cast<double>(p.x) * p.x;
        swxy += w * static_cast<double>(p.x) * p.y;
    }
    if (sw <= 1e-12) {
        return false;
    }
    const double det = sw * swxx - swx * swx;
    if (std::fabs(det) <= 1e-12 * std::max(1.0, sw * swxx)) {
        return false;
    }
    const double a = (sw * swxy - swx * swy) / det;
    const double b = (swxx * swy - swx * swxy) / det;
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    *outA = static_cast<float>(a);
    *outB = static_cast<float>(b);
    return true;
}

/** 用一组 (d, z) 与候选模型，算中位相对残差 + 内点数。 */
struct ModelScore {
    bool ok = false;
    float medianRelResidual = 1e9f;
    int inliers = 0;
};

ModelScore scoreModel(const std::vector<float>& d, const std::vector<float>& z,
                      float a, float b, bool inverse) {
    ModelScore s;
    std::vector<float> rel;
    rel.reserve(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        float zp;
        if (inverse) {
            const float inv = a * d[i] + b;
            if (!(inv > 1e-6f)) {
                continue;
            }
            zp = 1.f / inv;
        } else {
            zp = a * d[i] + b;
        }
        if (!(zp > 1e-4f) || !std::isfinite(zp)) {
            continue;
        }
        const float r = std::fabs(zp - z[i]) / std::max(1e-4f, z[i]);
        rel.push_back(r);
    }
    if (rel.size() < 4) {
        return s;
    }
    const float med = medianOf(rel);
    s.ok = true;
    s.medianRelResidual = med;
    const float thr = std::max(0.05f, 3.f * med);
    for (float r : rel) {
        if (r <= thr) {
            ++s.inliers;
        }
    }
    return s;
}

}  // namespace

// ============================================================================
//  单帧鲁棒拟合
// ============================================================================

DepthCalibration fitDepthRobust(const std::vector<float>& d,
                                const std::vector<float>& z,
                                bool inverseDepth,
                                int maxIterations) {
    DepthCalibration out;
    const size_t n = std::min(d.size(), z.size());
    if (n < 6) {
        return out;
    }

    std::vector<Pair> base;
    base.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const float di = d[i];
        const float zi = z[i];
        if (!std::isfinite(di) || !std::isfinite(zi) || !(zi > 0.05f) || zi > 50.f) {
            continue;
        }
        base.push_back(Pair{di, zi, 1.f});
    }
    if (base.size() < 6) {
        return out;
    }

    // ---------------------------------------------------------- 模型 A：线性
    auto fitLinear = [&](bool inverse, float* outA, float* outB) -> bool {
        std::vector<Pair> pts;
        pts.reserve(base.size());
        for (const Pair& p : base) {
            const float y = inverse ? (1.f / p.y) : p.y;
            if (!std::isfinite(y)) {
                continue;
            }
            pts.push_back(Pair{p.x, y, 1.f});
        }
        if (pts.size() < 6) {
            return false;
        }

        // ---- 初值：普通最小二乘 ----
        float a = 0.f, b = 0.f;
        if (!weightedLineFit(pts, &a, &b)) {
            return false;
        }

        // ---- MAD 剔除 + Huber IRLS ----
        for (int it = 0; it < maxIterations; ++it) {
            std::vector<float> absRes;
            absRes.reserve(pts.size());
            for (const Pair& p : pts) {
                absRes.push_back(std::fabs(a * p.x + b - p.y));
            }
            const float med = medianOf(absRes);
            std::vector<float> dev;
            dev.reserve(absRes.size());
            for (float r : absRes) {
                dev.push_back(std::fabs(r - med));
            }
            const float mad = medianOf(dev);
            // 1.4826 * MAD 是正态分布下 sigma 的稳健估计
            const float sigma = std::max(1e-6f, 1.4826f * mad);
            const float huberK = 1.345f * sigma;

            for (Pair& p : pts) {
                const float r = std::fabs(a * p.x + b - p.y);
                // MAD 硬剔除（> 5 sigma）+ Huber 软降权
                if (r > 5.f * sigma) {
                    p.w = 0.f;
                } else if (r <= huberK) {
                    p.w = 1.f;
                } else {
                    p.w = huberK / std::max(1e-9f, r);
                }
            }
            float na = 0.f, nb = 0.f;
            if (!weightedLineFit(pts, &na, &nb)) {
                break;
            }
            const float da = std::fabs(na - a);
            const float db = std::fabs(nb - b);
            a = na;
            b = nb;
            if (da < 1e-7f && db < 1e-7f) {
                break;
            }
        }

        *outA = a;
        *outB = b;
        return true;
    };

    float aLin = 0.f, bLin = 0.f;
    const bool okLin = fitLinear(false, &aLin, &bLin);

    float aInv = 0.f, bInv = 0.f;
    bool okInv = false;
    if (inverseDepth) {
        okInv = fitLinear(true, &aInv, &bInv);
    }

    ModelScore sLin;
    if (okLin) {
        sLin = scoreModel(d, z, aLin, bLin, false);
    }
    ModelScore sInv;
    if (okInv) {
        sInv = scoreModel(d, z, aInv, bInv, true);
    }

    // 选相对残差更小的那个模型（残差相同则优先线性，行为更可预期）
    const bool useInv = sInv.ok && (!sLin.ok || sInv.medianRelResidual < sLin.medianRelResidual * 0.98f);
    if (useInv && sInv.ok) {
        out.scale = aInv;
        out.shift = bInv;
        out.inverseDepthModel = true;
        out.medianRelativeResidual = sInv.medianRelResidual;
        out.inliers = sInv.inliers;
    } else if (sLin.ok) {
        out.scale = aLin;
        out.shift = bLin;
        out.inverseDepthModel = false;
        out.medianRelativeResidual = sLin.medianRelResidual;
        out.inliers = sLin.inliers;
    } else {
        return out;
    }

    out.samples = static_cast<int>(base.size());
    out.rejected = static_cast<int>(base.size()) - out.inliers;

    // 置信度：内点比例 × 残差打分
    const float inlierRatio = static_cast<float>(out.inliers) / static_cast<float>(base.size());
    const float residualScore = 1.f / (1.f + 8.f * out.medianRelativeResidual);
    const float sampleScore = std::min(1.f, static_cast<float>(base.size()) / 60.f);
    out.confidence = std::clamp(inlierRatio * residualScore * sampleScore, 0.f, 1.f);
    out.valid = out.confidence > 0.f;
    return out;
}

// ============================================================================
//  DepthCalibrator
// ============================================================================

void DepthCalibrator::reset() {
    calib_ = DepthCalibration{};
    acceptedFrames_ = 0;
    rejectedFrames_ = 0;
    totalSamples_ = 0;
    lastReject_ = "";
}

bool DepthCalibrator::update(const std::vector<float>& d, const std::vector<float>& z) {
    if (d.size() < static_cast<size_t>(cfg_.minSamples)) {
        lastReject_ = "too few paired samples";
        ++rejectedFrames_;
        return false;
    }
    totalSamples_ += d.size();

    // 超量时均匀抽稀，保证调用耗时可控
    std::vector<float> dd, zz;
    const size_t n = std::min(d.size(), z.size());
    const size_t step = (n > static_cast<size_t>(cfg_.maxSamples))
                            ? (n / static_cast<size_t>(cfg_.maxSamples) + 1)
                            : 1;
    dd.reserve(n / step + 2);
    zz.reserve(n / step + 2);
    for (size_t i = 0; i < n; i += step) {
        dd.push_back(d[i]);
        zz.push_back(z[i]);
    }

    const DepthCalibration fresh =
        fitDepthRobust(dd, zz, cfg_.allowInverseDepth);
    if (!fresh.valid) {
        lastReject_ = "fit degenerate";
        ++rejectedFrames_;
        return false;
    }
    if (fresh.confidence < cfg_.minConfidence) {
        lastReject_ = "confidence below gate";
        ++rejectedFrames_;
        return false;
    }
    if (fresh.medianRelativeResidual > cfg_.maxMedianRelativeResidual) {
        lastReject_ = "median relative residual too large";
        ++rejectedFrames_;
        return false;
    }

    if (!calib_.valid) {
        // 首次接受：直接采用
        calib_ = fresh;
        calib_.valid = true;
    } else {
        // 模型类型切换时不做 EMA（两者参数空间不同，混起来没有意义）
        if (calib_.inverseDepthModel != fresh.inverseDepthModel) {
            calib_.inverseDepthModel = fresh.inverseDepthModel;
            calib_.scale = fresh.scale;
            calib_.shift = fresh.shift;
        } else {
            const float ratio = (std::fabs(calib_.scale) > 1e-9f)
                                    ? std::fabs(fresh.scale / calib_.scale)
                                    : 1.f;
            if (ratio > cfg_.maxJumpRatio || ratio < 1.f / cfg_.maxJumpRatio) {
                lastReject_ = "scale jump rejected";
                ++rejectedFrames_;
                return false;
            }
            const float k = std::clamp(cfg_.emaNew, 0.01f, 1.f);
            calib_.scale = (1.f - k) * calib_.scale + k * fresh.scale;
            calib_.shift = (1.f - k) * calib_.shift + k * fresh.shift;
        }
        calib_.confidence = std::max(calib_.confidence * 0.9f, fresh.confidence);
        calib_.inliers = fresh.inliers;
        calib_.rejected = fresh.rejected;
        calib_.medianRelativeResidual = fresh.medianRelativeResidual;
        calib_.samples += fresh.samples;
    }
    calib_.valid = true;
    lastReject_ = "";
    ++acceptedFrames_;
    return true;
}

// ============================================================================
//  时序一致性
// ============================================================================

float temporalConsistencyRatio(const float* prevDepth, const float* curDepth,
                               int w, int h,
                               float fx, float fy, float cx, float cy,
                               const float Rrel[9], const float trel[3],
                               float minAbsTol, float relTol) {
    if (!prevDepth || !curDepth || w < 4 || h < 4 || !Rrel || !trel) {
        return 1.f;
    }
    const float invFx = 1.f / (std::fabs(fx) > 1e-3f ? fx : 1.f);
    const float invFy = 1.f / (std::fabs(fy) > 1e-3f ? fy : 1.f);

    long long tested = 0;
    long long consistent = 0;
    const int step = 4;
    for (int y = step; y < h; y += step) {
        for (int x = step; x < w; x += step) {
            const float zp = prevDepth[y * w + x];
            if (!(zp > 0.05f) || !std::isfinite(zp)) {
                continue;
            }
            const float xc = (x - cx) * zp * invFx;
            const float yc = (y - cy) * zp * invFy;
            const float nx = Rrel[0] * xc + Rrel[1] * yc + Rrel[2] * zp + trel[0];
            const float ny = Rrel[3] * xc + Rrel[4] * yc + Rrel[5] * zp + trel[1];
            const float nz = Rrel[6] * xc + Rrel[7] * yc + Rrel[8] * zp + trel[2];
            if (!(nz > 0.05f) || !std::isfinite(nz)) {
                continue;
            }
            const int ux = static_cast<int>(std::lround(nx * fx / nz + cx));
            const int uy = static_cast<int>(std::lround(ny * fy / nz + cy));
            if (ux < 1 || uy < 1 || ux >= w - 1 || uy >= h - 1) {
                continue;
            }
            const float zc = curDepth[uy * w + ux];
            if (!(zc > 0.05f) || !std::isfinite(zc)) {
                continue;
            }
            ++tested;
            const float tol = std::max(minAbsTol, zc * relTol);
            if (std::fabs(zc - nz) <= tol) {
                ++consistent;
            }
        }
    }
    if (tested < 16) {
        return 1.f;  // 样本太少不下结论，避免误伤
    }
    return static_cast<float>(consistent) / static_cast<float>(tested);
}
