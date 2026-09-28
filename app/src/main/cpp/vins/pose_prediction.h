#pragma once
#include <Eigen/Geometry>
#include <cmath>
#include <deque>
#include <cstdint>

namespace scan_pose {
struct ImuSample {
    double t = 0.;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
};
struct State {
    double t = -1.;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero(), v = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero(), bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero(), gyr = Eigen::Vector3d::Zero();
};

// Read-only propagation from the last optimized image state to THIS exposure.
// Do not advance the estimator or consume IMU samples needed by its next image.
inline bool predict(const State& base, const std::deque<ImuSample>& samples,
                    double time, State& result) {
    if (base.t < 0. || !std::isfinite(time) || time < base.t - 1e-7 ||
        time - base.t > .15 || !base.R.allFinite() || !base.p.allFinite() ||
        !base.v.allFinite() || !base.ba.allFinite() || !base.bg.allFinite() ||
        !base.gravity.allFinite() || !base.acc.allFinite() || !base.gyr.allFinite()) return false;
    State s = base;
    if (std::abs(time - base.t) <= 1e-7) { result = s; return true; }
    for (const auto& imu : samples) {
        if (!std::isfinite(imu.t) || !imu.acc.allFinite() || !imu.gyr.allFinite()) return false;
        if (imu.t <= s.t) continue;
        const double gap = imu.t - s.t;
        if (gap > .025) return false;
        const double end = std::min(time, imu.t);
        const double dt = end - s.t;
        const double alpha = dt / gap;
        const Eigen::Vector3d acc = s.acc + alpha * (imu.acc - s.acc);
        const Eigen::Vector3d gyr = s.gyr + alpha * (imu.gyr - s.gyr);
        const Eigen::Vector3d omega = .5 * (s.gyr + gyr) - s.bg;
        const Eigen::Vector3d a0 = s.R * (s.acc - s.ba) - s.gravity;
        const double angle = omega.norm() * dt;
        if (angle > 1e-12) s.R = (s.R * Eigen::AngleAxisd(angle, omega.normalized())).eval();
        const Eigen::Vector3d a = .5 * (a0 + s.R * (acc - s.ba) - s.gravity);
        s.p += dt * s.v + .5 * dt * dt * a;
        s.v += dt * a;
        s.acc = acc; s.gyr = gyr; s.t = end;
        if (end >= time) {
            if (!s.R.allFinite() || !s.p.allFinite() || !s.v.allFinite()) return false;
            result = s;
            return true;
        }
    }
    return false; // No future IMU bracket: never relabel a stale pose as current.
}

inline void cameraPose(const State& s, const Eigen::Matrix3d& ric,
                       const Eigen::Vector3d& tic, Eigen::Matrix3d& Rwc,
                       Eigen::Vector3d& twc) {
    Rwc = s.R * ric;
    twc = s.p + s.R * tic;
}

struct CameraSample {
    uint64_t ts = 0;
    float R[9] = {1,0,0,0,1,0,0,0,1};
    float t[3] = {0,0,0};
};

inline bool lookup(const std::deque<CameraSample>& history, uint64_t ts,
                   uint64_t tolerance, float out[12]) {
    if (!out || !ts || history.empty()) return false;
    const CameraSample* before = nullptr;
    const CameraSample* after = nullptr;
    for (const auto& s : history) {
        if (s.ts <= ts) before = &s;
        if (s.ts >= ts) { after = &s; break; }
    }
    if (before && after && before != after) {
        if (ts - before->ts > tolerance || after->ts - ts > tolerance) return false;
        Eigen::Matrix3d a, b;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
            a(i,j)=before->R[i*3+j]; b(i,j)=after->R[i*3+j];
        }
        const double alpha = double(ts-before->ts)/double(after->ts-before->ts);
        const Eigen::Matrix3d R = Eigen::Quaterniond(a).normalized().slerp(
            alpha, Eigen::Quaterniond(b).normalized()).toRotationMatrix();
        for(int i=0;i<3;++i) {
            for(int j=0;j<3;++j) out[i*3+j]=R(i,j);
            out[9+i]=before->t[i]+alpha*(after->t[i]-before->t[i]);
        }
    } else {
        const auto* s = before ? before : after;
        if (!s || (s->ts > ts ? s->ts-ts : ts-s->ts) > tolerance) return false;
        for(int i=0;i<9;++i) out[i]=s->R[i];
        for(int i=0;i<3;++i) out[9+i]=s->t[i];
    }
    for(int i=0;i<12;++i) if (!std::isfinite(out[i])) return false;
    return true;
}
} // namespace scan_pose
