// V0.13.35 (vc175) 回归：输出米制深度跨度门。
//
// 实机根因（PLK110 2026-10-01 22:30 会话）：首建接受 scale=-0.0003/shift=7.67
// 的退化映射 —— invZSpan=0.196 能过旧 0.15 门（shift 大时 1/z 绝对值大），
// 米制 z 却只剩 0.171~0.187（16mm 墙），epoch 冻结后满屏拉丝、导出碎块。
// 本用例锁死该盲区：旧门（refSpan/invZSpan/residual/confidence）全过的样本，
// 只要输出米制跨度塌掉就必须被拒；健康拟合不受影响。
#include "depth_calib.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // ---- 退化形态：raw d 宽分布，z 参考有大纵深，但与 d 仅弱相关 ----
    // 拟合收敛到 scale≈-6e-5 / shift≈6：invZSpan≈0.24 过旧首建门(0.075)、
    // refSpanRel≈0.45 过旧首建门(0.15)、medianResidual≈0.15 过 0.30 门、
    // confidence≈0.4 过 0.25 门 —— 但 ẑ 的米制跨度仅 ~4%（摧毁 90% 纵深）。
    std::vector<float> d, z;
    const int kN = 200;
    for (int i = 0; i < kN; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(kN - 1);
        const float di = 1000.f + 5000.f * t;
        // 确定性三角噪声 ∈ [-0.10, +0.35]：撑大参考纵深，同时保持中位残差低。
        const float u = std::fmod(t * 7.0f, 1.0f);
        const float noise = -0.10f + 0.45f * u;
        d.push_back(di);
        z.push_back((1.f / (6.f - 0.00006f * di)) * (1.f + noise));
    }

    const auto fit = fitDepthRobust(d, z, true, 10, true);
    assert(fit.valid);
    std::printf("fit: scale=%.6f shift=%.4f refSpanRel=%.3f invZSpan=%.3f "
                "zSpanRel=%.3f resid=%.3f conf=%.3f\n",
                static_cast<double>(fit.scale), static_cast<double>(fit.shift),
                static_cast<double>(fit.refDepthSpanRel),
                static_cast<double>(fit.outputInvZSpan),
                static_cast<double>(fit.outputDepthSpanRel),
                static_cast<double>(fit.medianRelativeResidual),
                static_cast<double>(fit.confidence));
    // 旧门全部放行（证明这是旧门控盲区，不是样本太差）：
    assert(fit.refDepthSpanRel >= 0.15f);
    assert(fit.outputInvZSpan >= 0.075f);
    assert(fit.medianRelativeResidual < 0.30f);
    assert(fit.confidence >= 0.25f);
    // 新指标暴露塌陷：
    assert(fit.outputDepthSpanRel < 0.10f);

    // A/B 对照一：关掉健康门（旧行为）→ 必须接受（复现 vc175 吃垃圾）。
    DepthCalibrator legacy;
    auto cfg = legacy.config();
    cfg.enforceCalibHealthGate = false;
    legacy.setConfig(cfg);
    assert(legacy.update(d, z));
    assert(legacy.usable());

    // A/B 对照二：默认配置（健康门开，含新米制跨度门）→ 必须拒绝。
    DepthCalibrator fixed;
    assert(!fixed.update(d, z));
    assert(!fixed.usable());
    assert(fixed.acceptedFrames() == 0);

    // 健康拟合不误伤：raw d ∝ 1/z 的标准逆深度场景，映射保留全部纵深。
    std::vector<float> dh, zh;
    for (int i = 0; i < 120; ++i) {
        const float zi = 0.35f + 1.2f * static_cast<float>(i) / 119.f;
        zh.push_back(zi);
        dh.push_back(3.0f / zi);
    }
    DepthCalibrator healthy;
    assert(healthy.update(dh, zh));
    assert(healthy.usable());
    const auto good = healthy.calibration();
    assert(good.outputDepthSpanRel >= 0.5f * good.refDepthSpanRel);

    std::printf("PASS vc175: output metric span gate blocks flat-mapping fit "
                "that passes all legacy gates; healthy fits unaffected\n");
    return 0;
}
