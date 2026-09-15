#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "utility/utility.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;
extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;
extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;

void setEstimatorParams(
    double accN,
    double accW,
    double gyrN,
    double gyrW,
    double initDepth,
    double minParallax,
    int numIterations,
    double solverTime);
void setExtrinsic(const Eigen::Matrix3d& ric, const Eigen::Vector3d& tic);
void setImageSize(double row, double col);

enum SIZE_PARAMETERIZATION {
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder {
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder {
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
