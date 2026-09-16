#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

struct Gaussian {
    float px, py, pz;
    float sx, sy, sz;
    float qx, qy, qz, qw;
    uint8_t r, g, b, opacity;
    uint16_t hits;
    uint8_t state;
};

class GaussianEngine {
public:
    void reset();
    void ingestPoint(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b, float confidence);
    size_t count() const;
    size_t stableCount() const;
    size_t mergedCount() const;
    /**
     * 只输出 hits >= minHits 的点。
     *
     * 旧实现把 g_ 里所有点均匀抽样后全部画出来，从不看 hits / state / opacity，
     * 于是一次性点（hits=1，占实机统计里的绝大多数）也以实心 3px 画满屏幕 ——
     * 这正是截图上满屏「黑芝麻」的来源。
     *
     *   RAW       minHits = 1
     *   CONFIRMED minHits = 2   <- AR 默认
     *   STABLE    minHits = 3
     */
    size_t copyPoints(float* out, size_t maxPoints, int minHits) const;
    /** 满足 hits >= minHits 的点数（用于 HUD / 报告里的「确认」「稳定」计数） */
    size_t confirmedCount(int minHits) const;
    void boundingBox(float* minX, float* minY, float* minZ,
                     float* maxX, float* maxY, float* maxZ) const;
    void centroid(float* x, float* y, float* z) const;

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

    std::vector<Gaussian> g_;
    std::unordered_map<Key, size_t, Hash> index_;
    size_t stable_ = 0;
    size_t merged_ = 0;
};
