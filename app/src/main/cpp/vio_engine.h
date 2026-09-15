#pragma once

#include <cstdint>

class VioEngine {
public:
    void reset();
    void imu(uint64_t t, float ax, float ay, float az, float gx, float gy, float gz);
    void frame(uint64_t t, const uint8_t* y, int w, int h, int stride = 0);
    uint64_t frames() const { return frames_; }
    float yaw() const { return yaw_; }
    float pitch() const { return pitch_; }
    float roll() const { return roll_; }
    float tx() const { return tx_; }
    float ty() const { return ty_; }
    float tz() const { return tz_; }
    float visualNovelty() const { return novelty_; }
    uint32_t features() const { return features_; }

private:
    uint64_t lastImu_ = 0;
    uint64_t lastFrame_ = 0;
    uint64_t frames_ = 0;
    float yaw_ = 0;
    float pitch_ = 0;
    float roll_ = 0;
    float prevYaw_ = 0;
    float tx_ = 0;
    float ty_ = 0;
    float tz_ = 0;
    float prevCx_ = 0;
    float prevCy_ = 0;
    bool havePrev_ = false;
    float novelty_ = 1;
    uint32_t features_ = 0;
    float ax_ = 0;
    float ay_ = 0;
    float az_ = 0;
};
