#include "parameters.h"

double INIT_DEPTH = 0.0;
double MIN_PARALLAX = 10.0 / 460.0;
// ESTIMATE_EXTRINSIC=2 的在线标定只有在旋转激励足够并成功收敛后才会放行初始化；
// 否则会一直停留在 INITIAL，界面表现为“定位失锁”。当前 ric 暂用单位阵，
// 先固定外参保证链路可跑通；真机稳定后建议正式标定并回填 RIC/TIC。
int ESTIMATE_EXTRINSIC = 0;
double ACC_N = 0.1;
double ACC_W = 0.001;
double GYR_N = 0.001;
double GYR_W = 0.0001;
std::vector<Eigen::Matrix3d> RIC(1, Eigen::Matrix3d::Identity());
std::vector<Eigen::Vector3d> TIC(1, Eigen::Vector3d::Zero());
Eigen::Vector3d G(0.0, 0.0, 9.8);
double BIAS_ACC_THRESHOLD = 0.1;
double BIAS_GYR_THRESHOLD = 0.01;
double SOLVER_TIME = 0.04;
int NUM_ITERATIONS = 8;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC = "/imu0";
double TD = 0.0;
double TR = 0.0;
int ESTIMATE_TD = 0;
int ROLLING_SHUTTER = 0;
double ROW = 0.0;
double COL = 0.0;

void setEstimatorParams(
        double accN,
        double accW,
        double gyrN,
        double gyrW,
        double initDepth,
        double minParallax,
        int numIterations,
        double solverTime) {
    ACC_N = accN;
    ACC_W = accW;
    GYR_N = gyrN;
    GYR_W = gyrW;
    INIT_DEPTH = initDepth;
    MIN_PARALLAX = minParallax;
    NUM_ITERATIONS = numIterations;
    SOLVER_TIME = solverTime;
}

void setTemporalParams(
    double td,
    int estimateTd,
    int rollingShutter,
    double tr)
{
    TD = td;
    ESTIMATE_TD = estimateTd;
    ROLLING_SHUTTER = rollingShutter;
    TR = tr;
}

void setExtrinsicEstimateMode(int mode)
{
    ESTIMATE_EXTRINSIC = mode;
}

void setExtrinsic(const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic) {
    RIC[0] = ric;
    TIC[0] = tic;
}

void setImageSize(double row, double col) {
    ROW = row;
    COL = col;
}
