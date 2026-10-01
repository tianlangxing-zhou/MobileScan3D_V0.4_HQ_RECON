#include "depth_calib.h"
#include "scan_policy.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // One accepted fit followed by seven rejected/absent fits must never
    // supply eight independent confirmations for opening an epoch.
    int streak = 0;
    for (int i = 0; i < 8; ++i) {
        const bool evidence = scan_policy::hasCurrentCalibrationEvidence(i == 0, false, false);
        streak = evidence ? streak + 1 : 0;
    }
    assert(streak == 0);
    assert(!scan_policy::hasCurrentCalibrationEvidence(false, false, true));
    assert(!scan_policy::hasCurrentCalibrationEvidence(false, true, false));
    assert(scan_policy::hasCurrentCalibrationEvidence(false, true, true));
    assert(scan_policy::hasCurrentCalibrationEvidence(true, false, false));

    // Linear-depth and inverse-depth models describing the same depths must
    // report inverse-depth resolution in the same physical unit.
    std::vector<float> depth, inverse;
    for (int i = 0; i < 100; ++i) {
        depth.push_back(.1f + .001f * i);
        inverse.push_back(1.f / depth.back());
    }
    const auto linear = fitDepthRobust(depth, depth, false);
    const auto inv = fitDepthRobust(inverse, depth, true, 10, true);
    assert(linear.valid && inv.valid);
    const float expectedLinear = std::fabs(1.f / depth[9] - 1.f / depth[89]);
    assert(std::fabs(linear.outputInvZSpan - expectedLinear) < 1e-4f);
    assert(inv.outputInvZSpan > 3.f && linear.outputInvZSpan > 3.f);
    DepthCalibrator cal;
    auto cfg = cal.config(); cfg.allowInverseDepth = false; cal.setConfig(cfg);
    assert(cal.update(depth, depth));
    assert(cal.update(depth, depth)); // Previously rejected on second update: .08m < .15/m.

    // A well-spread reference set alone is insufficient if the fitted output
    // is flat. Symmetric values give a zero-covariance (zero-slope) fit, with
    // small relative residuals that pass the old confidence/residual gates.
    std::vector<float> raw, z;
    for (int i = 0; i < 100; ++i) {
        raw.push_back(float(i));
        z.push_back(1.f + .3f * std::fabs(i - 49.5f) / 49.5f);
    }
    DepthCalibrator flat;
    cfg = flat.config(); cfg.forceInverseDepth = true; flat.setConfig(cfg);
    const auto bad = fitDepthRobust(raw, z, true, 10, true);
    assert(bad.valid && bad.refDepthSpanRel > .15f && bad.outputInvZSpan < 1e-4f);
    assert(!flat.update(raw, z));
    assert(!flat.usable());
    assert(flat.acceptedFrames() == 0);
    std::cout << "PASS vc167: current epoch evidence, linear/inverse units, repeated close-range fits, flat first-fit rejection\n";
}
