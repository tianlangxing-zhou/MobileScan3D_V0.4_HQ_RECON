#include <Eigen/Dense>

namespace gsslam {

void vins_prelude() {
    Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d inv = m.inverse();
    (void)inv;
}

}  // namespace gsslam
