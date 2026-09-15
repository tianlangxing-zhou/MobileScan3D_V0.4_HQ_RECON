#pragma once

#include <cstdint>

struct VinsHealth {
    bool initialized = false;

    float velocity = 0.0f;
    float accBias = 0.0f;
    float gyroBias = 0.0f;
    float gravity = 0.0f;

    int trackedFeatures = 0;

    double lastImuDt = 0.0;
};

void vinsInit(
    float fx,
    float fy,
    float cx,
    float cy,
    int w,
    int h,
    const float* ric,
    const float* tic,
    float accN,
    float accW,
    float gyrN,
    float gyrW);

void vinsInputImu(double t, double ax, double ay, double az, double gx, double gy, double gz);

void vinsInputImage(double t, const std::uint8_t* gray, int w, int h, int stride);

bool vinsGetPose(float* out7);

bool vinsReady();

bool vinsInitialized();

bool vinsGetHealth(VinsHealth* out);
