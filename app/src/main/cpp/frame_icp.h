#pragma once

// ============================================================================
//  frame_icp.h — vc184 / Option A：帧到模型 ICP 位姿精修
// ============================================================================
//
// 动机（真机证据，2026-10-02 vc183 实测）：
//   单目 VINS 的位姿有**渐变漂移**（旋转尺度不完全可观）。三道融合门
//   （temporal mask / fusionGuard / stale-ref）全都基于「位姿重投影一致性」，
//   漂移累积后重投影残差超过 1.5/2.5cm 容差 → 整帧全拒
//   （实测 guard checks=20 rejected=20、fused 13/60）→
//   「不重影」以「几乎不再融合」为代价（重复观测不足、开放边 1.2~1.5 万、
//   纹理烘焙 0 关键帧）。
//
// 修法：每帧用 epoch 米制深度反投影成世界点云，对齐已建立的 surfel 稳定
// 模型（粗体素 2cm 代表点），求出本帧 VINS 位姿的**刚体修正量**，再进
// 三道门与 TSDF/surfel 写入。修正后的重投影一致性判据在漂移下依然成立。
//
// 安全设计（三层）：
//   1. **每帧独立**：修正量不回馈 VINS、不累计跨帧 —— 坏修正最坏影响一帧，
//      且下游门控（temporal/guard）仍会拦住写坏的场景。
//   2. **有界**：总修正平移 > kIcpMaxTransM 或转角 > kIcpMaxRotDeg 直接放弃
//      （保持原位姿）。ICP 拉飞不可能写进地图。
//   3. **欠约束放弃**：样本/内点/重叠不足（新区域、模型未建立）时不修正，
//      行为与旧版完全一致。
//
// 平面简并防护：共面点云的点对点 Kabsch 在法向旋转上欠定（最小奇异值
// →0）。这里用「逐迭代旋转限幅 + 总转角上限」夹住；即使偶发选错，下一帧
// 重新独立求解，不放大。
// ============================================================================

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace frame_icp {

// ---- 参数（全部可被调用方覆盖的默认值见 frameToModelIcp 签名）----
// vc185 死锁根治：围绕旋转时 VINS 渐变漂移，首段稳定几何很小（几千 surfel、
// 共面代表点可能才一两百）。原门限（400 采样/250 内点/20% 重叠/3cm 邻域/6次迭代）
// 在模型小 + 漂移较大时全部触发拒绝 → attempts 进入但 applied=0 → 位姿不修 →
// 死锁。全面下调：早期小模型即可收敛出有效修正，同时仍有界（kIcpMaxTransM/
// kIcpMaxRotDeg/逐迭代限幅）兜底，不会把碎模型拉飞。
static constexpr int kIcpSampleStep = 4;        // 256x192 深度 → ~3k 采样点
static constexpr int kIcpMaxIterations = 9;
static constexpr float kIcpMaxCorrM = 0.06f;    // 容漂移达 6cm：漂移 cm 级时 3cm 邻域很可能匹配不上
static constexpr int kIcpMinSamples = 160;
static constexpr int kIcpMinInliers = 60;
static constexpr float kIcpMinOverlap = 0.08f;  // 内点/采样 ≥ 8% 即可（早期重叠稀疏）
static constexpr float kIcpMaxTransM = 0.10f;   // 单帧修正上限 10cm（漂移可达
static constexpr float kIcpMaxRotDeg = 8.f;     // 单帧修正上限 8°
static constexpr float kIcpPerIterRotDeg = 2.f; // 平面简并防护：逐迭代限幅
static constexpr float kIcpConvTransM = 0.0008f;
static constexpr float kIcpConvRotDeg = 0.08f;

enum RejectReason {
    kIcpOk = 0,
    kIcpRejectSamples = 1,   // 帧内有效深度样本不足
    kIcpRejectOverlap = 2,   // 与稳定模型内点/重叠不足（新区域或模型未建立）
    kIcpRejectBound = 3,     // 修正量超界（ICP 可能拉飞，放弃）
    kIcpRejectDegenerate = 4 // 刚体拟合退化（奇异值塌缩，放弃）
};

struct Diag {
    bool attempted = false;  // 进入了 ICP 主流程（模型就绪且深度有效）
    bool applied = false;    // 修正被采纳
    int reason = kIcpOk;
    int iterations = 0;
    int samples = 0;
    int inliers = 0;
    float rmseM = -1.f;      // 采纳时的内点 RMSE（米）
    float transM = 0.f;      // 总修正平移
    float rotDeg = 0.f;      // 总修正转角
};

inline float rotationAngleDeg(const Eigen::Matrix3f& R) {
    const float tr = R(0, 0) + R(1, 1) + R(2, 2);
    const float c = std::clamp((tr - 1.f) * 0.5f, -1.f, 1.f);
    return std::acos(c) * 57.2957795f;
}

/**
 * Kabsch 刚体拟合：求 Rc/tc 使 |Rc*src + tc - dst| 最小。
 * 返回 false 表示退化（协方差有效秩低于 2，典型为近共线输入），调用方应放弃
 * 本次修正。共面输入保留：桌面/墙面是扫描中的主要稳定几何，点到点 Kabsch
 * 仍能约束平面内平移与旋转；真正欠约束的是近共线点集。
 */
inline bool rigidFit(const std::vector<Eigen::Vector3f>& src,
                     const std::vector<Eigen::Vector3f>& dst,
                     Eigen::Matrix3f* outR, Eigen::Vector3f* outT) {
    const size_t n = src.size();
    if (n < 8 || dst.size() != n) return false;
    Eigen::Vector3f cs = Eigen::Vector3f::Zero();
    Eigen::Vector3f cd = Eigen::Vector3f::Zero();
    for (size_t i = 0; i < n; ++i) { cs += src[i]; cd += dst[i]; }
    cs /= float(n);
    cd /= float(n);
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < n; ++i) {
        H += (src[i] - cs) * (dst[i] - cd).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3f& s = svd.singularValues();
    if (!(s(0) > 1e-9f) || s(1) < 1e-6f * s(0)) {
        return false; // 近共线：刚体修正的旋转/平移约束不足
    }
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0.f) {
        Eigen::Matrix3f V = svd.matrixV();
        V.col(2) *= -1.f;
        R = V * svd.matrixU().transpose();
    }
    *outR = R;
    *outT = cd - R * cs;
    return true;
}

/**
 * 帧到模型 ICP 位姿精修。
 *
 * @param depth   epoch 米制深度（与进 TSDF 同域），w×h
 * @param R,t     当前帧 VINS cam→world 位姿（列主序 3x3，world = R*p_cam + t）
 * @param nearest 世界点最近邻查询（典型实现：SurfelEngine::nearestStableSurfel）
 * @param outR,outT 修正后的位姿（仅 applied=true 时有效）
 */
template <class NearestFn>
Diag frameToModelIcp(const float* depth, int w, int h,
                     float fx, float fy, float cx, float cy,
                     const float R[9], const float t[3],
                     NearestFn&& nearest,
                     float outR[9], float outT[3]) {
    Diag d;
    if (!depth || !R || !t || w < 16 || h < 16) { d.reason = kIcpRejectSamples; return d; }

    // ---- 1) 反投影采样（相机系 → 原始位姿世界系）----
    std::vector<Eigen::Vector3f> pts;
    pts.reserve((size_t)(w / kIcpSampleStep + 1) * (h / kIcpSampleStep + 1));
    for (int y = kIcpSampleStep / 2; y < h; y += kIcpSampleStep) {
        for (int x = kIcpSampleStep / 2; x < w; x += kIcpSampleStep) {
            const float z = depth[(size_t)y * w + x];
            if (!(z > 0.08f && z < 8.f)) continue;
            const float xc = (x - cx) * z / fx;
            const float yc = (y - cy) * z / fy;
            Eigen::Vector3f pw(R[0] * xc + R[1] * yc + R[2] * z + t[0],
                               R[3] * xc + R[4] * yc + R[5] * z + t[1],
                               R[6] * xc + R[7] * yc + R[8] * z + t[2]);
            if (!pw.allFinite()) continue;
            pts.push_back(pw);
        }
    }
    d.samples = (int)pts.size();
    if (d.samples < kIcpMinSamples) {
        d.reason = kIcpRejectSamples;
        return d;
    }
    d.attempted = true;

    // ---- 2) 迭代 trimmed point-to-point ICP ----
    const int minInliers = std::max(kIcpMinInliers, (int)(pts.size() * kIcpMinOverlap));
    Eigen::Matrix3f totalR = Eigen::Matrix3f::Identity();
    Eigen::Vector3f totalT = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f> cur = pts;
    std::vector<Eigen::Vector3f> model;
    float rmse = -1.f;
    int lastInl = 0;
    for (int iter = 0; iter < kIcpMaxIterations; ++iter) {
        ++d.iterations;
        model.clear();
        std::vector<Eigen::Vector3f> src;
        src.reserve(cur.size());
        model.reserve(cur.size());
        for (const auto& p : cur) {
            float qx, qy, qz;
            if (nearest(p.x(), p.y(), p.z(), kIcpMaxCorrM, &qx, &qy, &qz)) {
                src.push_back(p);
                model.push_back(Eigen::Vector3f(qx, qy, qz));
            }
        }
        const int inl = (int)src.size();
        lastInl = inl;
        if (inl < minInliers) {
            if (iter == 0) {
                // 首轮就无重叠：新区域或模型未建立，行为退回旧版（不修正）。
                d.reason = kIcpRejectOverlap;
                return d;
            }
            break; // 收敛中失去重叠：保留上一步累计修正
        }
        Eigen::Matrix3f Rc;
        Eigen::Vector3f tc;
        if (!rigidFit(src, model, &Rc, &tc)) {
            if (iter == 0) {
                d.reason = kIcpRejectDegenerate;
                return d;
            }
            break;
        }
        // 平面简并防护：逐迭代旋转限幅（超限只取方向、截断幅度）。
        const float rotDeg = rotationAngleDeg(Rc);
        if (rotDeg > kIcpPerIterRotDeg) {
            // 用轴角重参数化到限幅值；平移同步缩放，避免旋转被截而平移全量。
            const float scale = kIcpPerIterRotDeg / rotDeg;
            Eigen::AngleAxisf aa{Eigen::Quaternionf(Rc)};
            aa.angle() *= scale;
            Rc = aa.toRotationMatrix();
            tc *= scale;
        }
        for (auto& p : cur) p = Rc * p + tc;
        totalR = Rc * totalR;
        totalT = Rc * totalT + tc;
        // 收敛判定
        const float dTrans = tc.norm();
        const float dRot = rotationAngleDeg(Rc);
        double sum = 0.0;
        for (size_t i = 0; i < src.size(); ++i) {
            const Eigen::Vector3f diff = (Rc * src[i] + tc) - model[i];
            sum += double(diff.squaredNorm());
        }
        rmse = float(std::sqrt(sum / double(src.size())));
        if (dTrans < kIcpConvTransM && dRot < kIcpConvRotDeg) break;
    }

    // ---- 3) 有界性检查 ----
    d.transM = totalT.norm();
    d.rotDeg = rotationAngleDeg(totalR);
    if (d.transM > kIcpMaxTransM || d.rotDeg > kIcpMaxRotDeg) {
        d.reason = kIcpRejectBound;
        return d;
    }
    d.rmseM = rmse;
    d.inliers = lastInl; // 最近一次迭代的匹配内点数
    if (d.transM < 0.0002f && d.rotDeg < 0.02f) {
        // 修正量可忽略：不算 applied，避免统计噪声。
        return d;
    }

    // ---- 4) 输出修正后位姿：p' = Rt*(M·p + t) + tt ----
    // native_engine 的 R 是**行主序**（rotatePoint：Xw = R[0]x+R[1]y+R[2]z+...）。
    // M' = Rt·M（行主序写出），t' = Rt·t + tt。
    Eigen::Matrix3f M;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) M(r, c) = R[r * 3 + c];
    }
    const Eigen::Matrix3f Mp = totalR * M;
    const Eigen::Vector3f tIn(t[0], t[1], t[2]);
    const Eigen::Vector3f tp = totalR * tIn + totalT;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) outR[r * 3 + c] = Mp(r, c);
    }
    for (int i = 0; i < 3; ++i) outT[i] = tp[i];
    d.applied = true;
    return d;
}

} // namespace frame_icp
