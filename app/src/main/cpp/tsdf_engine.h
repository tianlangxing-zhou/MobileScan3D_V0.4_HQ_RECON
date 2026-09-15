#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TsdfVoxel {
    float sdf = 1.f;
    float weight = 0.f;
    float r = 0, g = 0, b = 0;
};

class TsdfEngine {
public:
    void reset();
    void integrateDepth(const float* depth, int w, int h,
                        const uint8_t* rgb, int rgbW, int rgbH,
                        float fx, float fy, float cx, float cy,
                        const float R[9], const float t[3], float confidence);
    uint64_t voxels() const { return voxels_.size(); }
    bool exportPly(const std::string& path) const;

private:
    struct Key {
        int x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct Hash {
        size_t operator()(const Key& k) const {
            uint64_t a = (uint32_t)k.x * 73856093u;
            uint64_t b = (uint32_t)k.y * 19349663u;
            uint64_t c = (uint32_t)k.z * 83492791u;
            return (size_t)(a ^ b ^ c);
        }
    };

    std::unordered_map<Key, TsdfVoxel, Hash> voxels_;
    float voxel_ = 0.02f;
};
