#pragma once

#include <string>

extern int FT_ROW;
extern int FT_COL;
extern int FT_FOCAL_LENGTH;
extern int FISHEYE;
extern int EQUALIZE;
extern int MIN_DIST;
extern int MAX_CNT;
extern double F_THRESHOLD;
extern std::string FISHEYE_MASK;
extern bool PUB_THIS_FRAME;

void setFeatureTrackerParams(
    int row,
    int col,
    int focalLength,
    int fisheye,
    int equalize,
    int minDist,
    int maxCnt,
    double fThreshold);
