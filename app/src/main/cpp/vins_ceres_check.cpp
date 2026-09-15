#include <ceres/ceres.h>

namespace gsslam {

struct OneResidual {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

void vins_ceres_check() {
    double x = 0.0;
    ceres::Problem problem;
    ceres::CostFunction* cost =
        new ceres::AutoDiffCostFunction<OneResidual, 1, 1>(new OneResidual);
    problem.AddResidualBlock(cost, nullptr, &x);
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    (void)summary;
}

}  // namespace gsslam
