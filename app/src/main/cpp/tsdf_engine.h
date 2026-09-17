#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
//  真正的 signed TSDF（截断符号距离场）
// ============================================================================
//
// 旧实现（已整体替换）有两个致命问题，是「能出点云、但出不了 mesh」的根因：
//
//   1. 体素值用「体素到那个表面点的欧氏距离 / 截断」来算
//      （sqrt(dx*dx+dy*dy+dz*dz) * voxel / trunc），**恒为非负**：
//      既没有符号，也没有沿相机射线做截断；
//   2. 它只在深度表面附近写 3x3x3 个「越远越大」的值，根本没有零交叉面。
//
//   Marching Cubes / Marching Tetrahedra 需要的是 **正负号 + 过零点**，
//   所以旧数据无论怎么后处理都提不出三角面。
//
// 现在是标准形式：沿相机射线在 [z-mu, z+mu] 上采样，体素值取
//
//      sdf = clamp((z - zi) / mu, -1, 1)
//
//   zi < z  （采样点在真实表面前方）  ->  sdf > 0  表面外侧（空）
//   zi > z  （采样点在真实表面后方）  ->  sdf < 0  表面内侧（实体）
//   zi == z （正好落在真实表面）      ->  sdf = 0  零交叉面
//
// 于是零交叉面就落在真实表面处，体素之间的符号变化直接给出三角面。
//
// ---------------------------------------------------------------------------
//  数据结构：Sparse Voxel Block Hashing
// ---------------------------------------------------------------------------
//   旧: std::unordered_map<Key{x,y,z}, TsdfVoxel>     一个体素一次 hash
//   新: std::unordered_map<BlockKey{x,y,z}, TsdfBlock> 8x8x8 = 512 体素一块
//
// 一次 3D 采样只需一次 hash + 块内连续寻址；块内是平铺数组，cache
// locality 好得多，而且这正是将来迁 Vulkan compute 的自然布局
// （一个 workgroup 处理一个 block）。
// ============================================================================

constexpr int kTsdfBlockSize = 8;
constexpr int kTsdfBlockShift = 3;  // log2(kTsdfBlockSize)
constexpr int kTsdfBlockMask = kTsdfBlockSize - 1;
constexpr int kTsdfBlockVoxels = kTsdfBlockSize * kTsdfBlockSize * kTsdfBlockSize;

/** tsdf 定点标度：int16 存 [-1, 1] * kTsdfValueScale。 */
constexpr float kTsdfValueScale = 32767.f;
/** weight 定点标度：uint16 存 weight * kTsdfWeightScale（饱和到 64 时是 32768）。 */
constexpr float kTsdfWeightScale = 512.f;
/** 单个体素的最大权重（饱和，避免老观测永远压不住新观测）。 */
constexpr float kTsdfMaxWeight = 64.f;
/** 截断距离 = 几个体素。4 个体素是 TSDF 的常见取值。 */
constexpr int kTsdfTruncVoxels = 4;

/** V0.11: 颜色权重与几何权重分离；体素从 6 字节增至 8 字节。 */
struct TsdfVoxel {
    int16_t tsdf = 0;
    uint16_t weight = 0;
    uint16_t color565 = 0;
    uint16_t colorWeight = 0;
};

/** 8x8x8 的体素块，一次性分配。 */
struct TsdfBlock {
    TsdfVoxel voxels[kTsdfBlockVoxels];
};

class TsdfEngine {
public:
    /** 未分配 / 权重为 0 的体素一律视为「表面外侧」的默认值。 */
    static constexpr int16_t kEmptyTsdf = 32767;

    TsdfEngine() = default;

    void reset();

    /** 设置体素边长（米），同时把截断距离设为 kTsdfTruncVoxels 个体素。 */
    void setVoxelSize(float meters);
    float voxelSize() const { return voxel_; }

    /** 单独覆盖截断距离（米）。 */
    void setTruncation(float meters);
    float truncation() const { return trunc_; }

    /** 块数上限（控制内存）。每块 sizeof(TsdfBlock) = 4096 字节。 */
    void setMaxBlocks(size_t n) { maxBlocks_ = n < 64 ? 64 : n; }
    size_t maxBlocks() const { return maxBlocks_; }

    /** 融合时的像素步长（1 = 每像素，2 = 隔一个像素）。步长越大越快、越稀。 */
    void setPixelStep(int s) { pixelStep_ = s < 1 ? 1 : (s > 4 ? 4 : s); }
    int pixelStep() const { return pixelStep_; }

    /** 是否已经写满配额（写满后会停止继续分配新块）。 */
    bool atCapacity() const { return blocks_.size() >= maxBlocks_; }

    /**
     * 融合一帧深度。
     *
     * 与旧版本签名**完全兼容**（最后两个参数有默认值），所以 native_engine.cpp
     * 里的两个调用点（fuseDepth / fuseTargetDepth）不需要改动。
     *
     * @param depthScale / depthShift  深度标定后的仿射修正 z' = z*scale + shift。
     *        默认 1/0 表示「不做修正」，行为与旧版本一致。
     */
    void integrateDepth(const float* depth, int w, int h,
                        const uint8_t* rgb, int rgbW, int rgbH,
                        float fx, float fy, float cx, float cy,
                        const float R[9], const float t[3], float confidence,
                        float depthScale = 1.f, float depthShift = 0.f);

    /** 已经分配且 weight > 0 的体素数。 */
    uint64_t voxels() const { return liveVoxels_; }
    uint64_t coloredVoxels() const { return liveColoredVoxels_; }
    uint64_t blocks() const { return static_cast<uint64_t>(blocks_.size()); }
    bool surfacePresent() const { return liveVoxels_ > 0; }
    size_t memoryBytes() const { return blocks_.size() * sizeof(TsdfBlock); }

    /** 仍保留：把表面附近的体素导出成只有顶点的 PLY（兼容旧导出路径）。 */
    bool exportPly(const std::string& path) const;

    // ------------------------------------------------------------- mesh 提取
    /** 整数体素坐标处的 tsdf。未分配或 weight==0 时返回 kEmptyTsdf。 */
    int16_t tsdfAt(int vx, int vy, int vz) const;
    uint16_t weightAt(int vx, int vy, int vz) const;
    /** 体素颜色（RGB565）。未分配时返回 0。 */
    uint16_t color565At(int vx, int vy, int vz) const;
    uint16_t colorWeightAt(int vx, int vy, int vz) const;
    bool allocated(int vx, int vy, int vz) const;
    /** 有内容时把体素坐标闭区间写进 mn/mx 并返回 true。 */
    bool voxelBounds(int mn[3], int mx[3]) const;

    /** 归一化 tsdf（[-1,1]），未分配时返回 1。 */
    float valueAt(int vx, int vy, int vz) const {
        return tsdfAt(vx, vy, vz) / kTsdfValueScale;
    }

    // ---- 给 mesh 提取用的批量读取接口 ----
    // mesh 提取要读几百万个体素，若每个都走一次 hash 查找会慢一个数量级。
    // 这里暴露「按块取指针 + 遍历块」，让提取端能以「块邻域缓存」的方式
    // 全部退化成数组直读。
    const TsdfBlock* blockAt(int bx, int by, int bz) const;
    void forEachBlock(const std::function<void(int, int, int, const TsdfBlock*)>& fn) const;

private:
    struct BlockKey {
        int x, y, z;
        bool operator==(const BlockKey& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct BlockHash {
        size_t operator()(const BlockKey& k) const {
            uint64_t a = (uint32_t)k.x * 73856093u;
            uint64_t b = (uint32_t)k.y * 19349663u;
            uint64_t c = (uint32_t)k.z * 83492791u;
            return (size_t)(a ^ b ^ c);
        }
    };

    TsdfBlock* blockFor(int bx, int by, int bz, bool create);
    TsdfVoxel* voxelFor(int vx, int vy, int vz, bool create);

    std::unordered_map<BlockKey, TsdfBlock, BlockHash> blocks_;
    float voxel_ = 0.020f;     // 20mm：全场景默认（与旧版一致）
    float trunc_ = 0.080f;     // 4 * voxel_
    float minDepth_ = 0.08f;
    float maxDepth_ = 8.0f;
    int pixelStep_ = 4;        // 与旧实现一致：depth 本身就是 256x256 上采样的，密集采样无收益
    size_t maxBlocks_ = 8192;  // V0.11: 8-byte voxel -> ~32MB 上限
    uint64_t liveVoxels_ = 0;
    uint64_t liveColoredVoxels_ = 0;
    bool hasBounds_ = false;
    int minV_[3] = {0, 0, 0};
    int maxV_[3] = {0, 0, 0};
};
