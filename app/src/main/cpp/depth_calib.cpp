#include "depth_calib.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
//  基础统计
// ============================================================================

namespace {
template<class T> T medianInPlace(std::vector<T>& v) {
    if (v.empty()) {
        return 0.f;
    }
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const T hi = v[mid];
    if (v.size() % 2 == 1) {
        return hi;
    }
    // 偶数个：直接扫描已分区的较小一半，避免第二次 nth_element。
    const T lo = *std::max_element(v.begin(), v.begin() + mid);
    return T(0.5) * lo + T(0.5) * hi;
}
} // namespace

float medianOf(std::vector<float> v) { return medianInPlace(v); }

namespace {

struct Pair {
    float x;   // 自变量（模型输出 d 或 1/d 归一化前的 d）
    float y;   // 因变量（z 或 1/z）
    float w;   // 稳健权重
};

/** 加权最小二乘 y = a*x + b。返回 false 表示退化（x 方差过小）。 */
bool weightedLineFit(const std::vector<Pair>& pts, double* outA, double* outB) {
    double sw = 0, swx = 0, swy = 0;
    for (const Pair& p : pts) {
        const double w = p.w;
        sw += w;
        swx += w * p.x;
        swy += w * p.y;
    }
    if (sw <= 1e-12) {
        return false;
    }
    // Center before multiplying: E[x*x]-E[x]*E[x] loses the small
    // variance when a depth provider supplies values with a large offset.
    const double mx = swx / sw, my = swy / sw;
    double sxx = 0, sxy = 0;
    for (const Pair& p : pts) {
        const double dx = static_cast<double>(p.x) - mx;
        const double dy = static_cast<double>(p.y) - my;
        sxx += p.w * dx * dx;
        sxy += p.w * dx * dy;
    }
    if (!(sxx > std::numeric_limits<double>::min()) || !std::isfinite(sxx)) {
        return false;
    }
    const double a = sxy / sxx;
    const double b = my - a * mx;
    if (!std::isfinite(a) || !std::isfinite(b) ||
        std::fabs(a) > std::numeric_limits<float>::max() ||
        std::fabs(b) > std::numeric_limits<float>::max()) {
        return false;
    }
    *outA = a;
    *outB = b;
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
    for (size_t i = 0; i < std::min(d.size(), z.size()); ++i) {
        if (!std::isfinite(d[i]) || !std::isfinite(z[i]) ||
            z[i] <= 0.05f || z[i] > 50.f) continue;
        float zp;
        if (inverse) {
            const double inv = static_cast<double>(a) * d[i] + b;
            if (!(inv > 1e-6f) || !std::isfinite(inv)) {
                rel.push_back(1e6f);
                continue;
            }
            zp = 1.f / inv;
        } else {
            zp = static_cast<float>(static_cast<double>(a) * d[i] + b);
        }
        if (!(zp > 1e-4f) || !std::isfinite(zp)) {
            rel.push_back(1e6f);
            continue;
        }
        const float r = std::fabs(zp - z[i]) / std::max(1e-4f, z[i]);
        rel.push_back(r);
    }
    if (rel.size() < 4) {
        return s;
    }
    const float med = medianInPlace(rel);
    s.ok = true;
    s.medianRelResidual = med;
    const float thr = std::max(0.05f, 3.f * med);
    for (float r : rel) {
        if (r < 1e6f && r <= thr) {
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
                                int maxIterations, bool forceInverse) {
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
        double a = 0, b = 0;
        if (!weightedLineFit(pts, &a, &b)) {
            return false;
        }

        // ---- MAD 剔除 + Huber IRLS ----
        std::vector<double> residuals(pts.size()), dev(pts.size());
        for (int it = 0; it < std::clamp(maxIterations, 0, 64); ++it) {
            size_t i = 0;
            for (const Pair& p : pts) {
                residuals[i++] = a * p.x + b - p.y;
            }
            // MAD is defined on signed residuals, not on their magnitudes.
            // The latter underestimates sigma and rejects valid noisy samples.
            const double med = medianInPlace(residuals);
            for (i = 0; i < residuals.size(); ++i) {
                dev[i] = std::fabs(residuals[i] - med);
            }
            const double mad = medianInPlace(dev);
            // 1.4826 * MAD 是正态分布下 sigma 的稳健估计
            const double sigma = std::max(1e-6, 1.4826 * mad);
            const double huberK = 1.345 * sigma;

            for (Pair& p : pts) {
                const double signedResidual = a * p.x + b - p.y;
                const double r = std::fabs(signedResidual);
                // MAD 硬剔除（> 5 sigma）+ Huber 软降权
                // Center the rejection gate too: an initial intercept bias
                // must not reject the entire consensus set before refitting.
                if (std::fabs(signedResidual - med) > 5.0 * sigma) {
                    p.w = 0.f;
                } else if (r <= huberK) {
                    p.w = 1.f;
                } else {
                    p.w = static_cast<float>(huberK / std::max(1e-9, r));
                }
            }
            double na = 0, nb = 0;
            if (!weightedLineFit(pts, &na, &nb)) {
                break;
            }
            const double da = std::fabs(na - a);
            const double db = std::fabs(nb - b);
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
    const bool okLin = !forceInverse && fitLinear(false, &aLin, &bLin);

    float aInv = 0.f, bInv = 0.f;
    bool okInv = false;
    if (inverseDepth || forceInverse) {
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

    // ---- V0.13.25 标定健康度诊断 ----
    // 用 base（全部有效配对样本）算两个退化指标，供 update() 门控与日志使用。
    {
        std::vector<float> zs;
        std::vector<float> ds;
        zs.reserve(base.size());
        ds.reserve(base.size());
        for (const Pair& p : base) {
            zs.push_back(p.y);
            ds.push_back(p.x);
        }
        std::sort(zs.begin(), zs.end());
        std::sort(ds.begin(), ds.end());
        auto pctOf = [](const std::vector<float>& v, double q) -> float {
            if (v.empty()) return 0.f;
            size_t i = static_cast<size_t>(q * static_cast<double>(v.size() - 1));
            if (i >= v.size()) i = v.size() - 1;
            return v[i];
        };
        const float z10 = pctOf(zs, 0.10);
        const float z50 = pctOf(zs, 0.50);
        const float z90 = pctOf(zs, 0.90);
        const float d10 = pctOf(ds, 0.10);
        const float d90 = pctOf(ds, 0.90);
        out.refDepthSpanRel = (z50 > 1e-4f) ? (z90 - z10) / z50 : 0.f;
        // Compare the same 1/metre unit for linear and inverse models.
        // For z=a*d+b, |a|*delta(d) is metres, not inverse metres.
        if (out.inverseDepthModel) {
            out.outputInvZSpan = std::fabs(out.scale) * (d90 - d10);
        } else {
            const double za = static_cast<double>(out.scale) * d10 + out.shift;
            const double zb = static_cast<double>(out.scale) * d90 + out.shift;
            out.outputInvZSpan = (za > 1e-6 && zb > 1e-6)
                ? static_cast<float>(std::fabs(1.0 / za - 1.0 / zb)) : 0.f;
        }
    }

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
    if (std::min(d.size(), z.size()) < static_cast<size_t>(std::max(6, cfg_.minSamples)) ||
        cfg_.maxSamples < 6) {
        lastReject_ = "too few paired samples";
        ++rejectedFrames_;
        return false;
    }
    totalSamples_ += std::min(d.size(), z.size());

    // 超量时均匀抽稀，保证调用耗时可控
    std::vector<float> dd, zz;
    const size_t n = std::min(d.size(), z.size());
    const size_t count = std::min(n, static_cast<size_t>(cfg_.maxSamples));
    dd.reserve(count);
    zz.reserve(count);
    // Exactly fill the sample budget even just above its boundary (513/512).
    // Quotient/remainder stepping is floor(j*n/count), without j*n overflow.
    const size_t stride = n / count, remainder = n % count;
    size_t index = 0, phase = 0;
    for (size_t j = 0; j < count; ++j) {
        dd.push_back(d[index]);
        zz.push_back(z[index]);
        index += stride;
        phase += remainder;
        if (phase >= count) { ++index; phase -= count; }
    }

    const DepthCalibration fresh =
        fitDepthRobust(dd, zz, cfg_.allowInverseDepth, 10, cfg_.forceInverseDepth);
    if (!fresh.valid || fresh.samples < std::max(6, cfg_.minSamples)) {
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

    // ---- V0.13.25 标定健康门 ----
    // 参考深度没有纵深结构时，1/z = scale*d + shift 的因变量几乎不变，最小二乘
    // 只能给出 scale~0，深度对输入完全失去敏感性 —— 整场景被压成一块平板
    // （z ≈ 1/shift）。实测 vc164：近距离时 VINS 参考 z 仅 0.106~0.141m
    // （相对纵深 0.30），拟合出 scale=1e-4，fusionDepth 宽仅 1cm。
    //
    // 首次建立时用放宽阈值兜底（避免冷启动完全拿不到标定而退回原始深度）；
    // 已有标定时严格门控，防止好标定被退化样本经 EMA 慢慢带偏。
    if (cfg_.enforceCalibHealthGate) {
        const float spanGate = calib_.valid
            ? cfg_.minRefDepthSpanRel
            : cfg_.minRefDepthSpanRel * cfg_.firstBuildRelax;
        if (!(fresh.refDepthSpanRel >= spanGate)) {
            lastReject_ = "reference depth span too narrow";
            ++rejectedFrames_;
            return false;
        }
        const float outputGate = cfg_.minOutputInvZSpan *
            (calib_.valid ? 1.f : cfg_.firstBuildRelax);
        if (!(fresh.outputInvZSpan >= outputGate)) {
            lastReject_ = "calibrated depth resolution too flat";
            ++rejectedFrames_;
            return false;
        }
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
            if (fresh.scale * calib_.scale <= 0.f ||
                ratio > cfg_.maxJumpRatio || ratio < 1.f / cfg_.maxJumpRatio) {
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
        calib_.refDepthSpanRel = fresh.refDepthSpanRel;
        calib_.outputInvZSpan = fresh.outputInvZSpan;
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
    std::vector<uint8_t> mask;
    if (w <= 0 || h <= 0 || w > 4096 || h > 4096) return 1.f;
    mask.resize(static_cast<size_t>(w) * h);
    int tested = 0, agree = 0;
    temporalConsistencyMask(prevDepth, curDepth, w, h, fx, fy, cx, cy,
                            Rrel, trel, mask.data(), 4, minAbsTol, relTol,
                            &tested, &agree);
    return tested < 16 ? 1.f : float(agree) / tested;
}

int temporalConsistencyMask(const float* prevDepth, const float* curDepth,
                            int w, int h, float fx, float fy, float cx, float cy,
                            const float Rrel[9], const float trel[3],
                            uint8_t* mask, int step, float minAbsTol, float relTol,
                            int* outTested, int* outAgree,
                            std::vector<float>* projectedDepth) {
    if (outTested) *outTested = 0;
    if (outAgree) *outAgree = 0;
    if (!mask || w <= 0 || h <= 0 || w > 4096 || h > 4096) return 0;
    const size_t n = static_cast<size_t>(w) * h;
    std::fill(mask, mask + n, uint8_t(kTemporalUntested));
    if (!prevDepth || !curDepth || !Rrel || !trel ||
        !std::isfinite(fx) || !std::isfinite(fy) || fx <= 0 || fy <= 0 ||
        !std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(minAbsTol) || !std::isfinite(relTol) || minAbsTol < 0 || relTol < 0 ||
        !std::all_of(Rrel, Rrel+9, [](float v){return std::isfinite(v);}) ||
        !std::all_of(trel, trel+3, [](float v){return std::isfinite(v);})) return 0;
    std::vector<float> local;
    auto& zbuf = projectedDepth ? *projectedDepth : local;
    zbuf.assign(n, 0.f);
    step = std::max(1, step);
    // A nearest-surface z-buffer makes collisions independent of traversal order.
    for (int y=0; y<h; y+=step) for (int x=0; x<w; x+=step) {
        float z=prevDepth[size_t(y)*w+x];
        if (!std::isfinite(z) || z<=0.05f) continue;
        float px=(x-cx)*z/fx, py=(y-cy)*z/fy;
        float X=Rrel[0]*px+Rrel[1]*py+Rrel[2]*z+trel[0];
        float Y=Rrel[3]*px+Rrel[4]*py+Rrel[5]*z+trel[1];
        float Z=Rrel[6]*px+Rrel[7]*py+Rrel[8]*z+trel[2];
        if (!std::isfinite(Z) || Z<=0.05f) continue;
        float u=X*fx/Z+cx, v=Y*fy/Z+cy;
        if (!std::isfinite(u) || !std::isfinite(v) || u<0 || v<0 || u>w-1 || v>h-1) continue;
        size_t j=size_t(std::lround(v))*w+std::lround(u);
        if (zbuf[j]==0 || Z<zbuf[j]) zbuf[j]=Z;
    }
    int tested=0, agree=0;
    for (int y=0; y<h; ++y) for (int x=0; x<w; ++x) {
        size_t j=size_t(y)*w+x;
        float a=zbuf[j], b=curDepth[j];
        if (!(a>0.05f) || !(b>0.05f) || !std::isfinite(b)) continue;
        float tol=std::max(minAbsTol,b*relTol);
        if (std::fabs(a-b)<=tol) { mask[j]=kTemporalAgree; ++tested; ++agree; continue; }
        // Large jumps near a depth silhouette are visibility changes, not errors.
        // Interior mismatches (including small jitter) remain genuine conflicts.
        bool edge=false;
        for (int dy=-2;dy<=2 && !edge;++dy) for (int dx=-2;dx<=2;++dx) {
            int xx=x+dx, yy=y+dy;
            if(xx<0 || yy<0 || xx>=w || yy>=h) continue;
            size_t k=size_t(yy)*w+xx;
            float p=zbuf[k], c=curDepth[k];
            if ((p>0.05f && std::fabs(p-a)>3*tol) ||
                (std::isfinite(c) && c>0.05f && std::fabs(c-b)>3*tol)) {edge=true;break;}
        }
        if(edge && std::fabs(a-b)>3*tol) mask[j]=b<a?kTemporalOccluded:kTemporalDisoccluded;
        else {mask[j]=kTemporalDisagree; ++tested;}
    }
    if(outTested) *outTested=tested;
    if(outAgree) *outAgree=agree;
    return tested;
}
