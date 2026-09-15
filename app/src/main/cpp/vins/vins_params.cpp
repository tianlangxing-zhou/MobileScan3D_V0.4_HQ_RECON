#include "vins_params.h"

int FT_ROW = 0;
int FT_COL = 0;
int FT_FOCAL_LENGTH = 460;
int FISHEYE = 0;
int EQUALIZE = 1;
int MIN_DIST = 30;
int MAX_CNT = 150;
double F_THRESHOLD = 1.0;
std::string FISHEYE_MASK;
bool PUB_THIS_FRAME = false;

void setFeatureTrackerParams(
        int row,
        int col,
        int focalLength,
        int fisheye,
        int equalize,
        int minDist,
        int maxCnt,
        double fThreshold) {
    FT_ROW = row;
    FT_COL = col;
    FT_FOCAL_LENGTH = focalLength;
    FISHEYE = fisheye;
    EQUALIZE = equalize;
    MIN_DIST = minDist;
    MAX_CNT = maxCnt;
    F_THRESHOLD = fThreshold;
}
