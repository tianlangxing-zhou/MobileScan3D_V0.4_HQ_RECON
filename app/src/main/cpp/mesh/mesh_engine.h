#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../tsdf_engine.h"

// ============================================================================
//  MeshEngine —— 从 signed TSDF 提取三角网格
// ============================================================================
//
// 为什么要有这一层：旧实现（tsdf_engine 的 exportPly）只把「满足权重/阈值的
// 体素」当成顶点导出，**没有任何三角面**，所以导出的 PLY 不是 AR 模型，
// 只是一个点云。AR 渲染、GLB、纹理贴图都要求真正的 indexed triangle mesh。
//
// 提取算法选 **Marching Tetrahedra**（Marching Cubes 家族）：把每个体素立方体
// 沿主对角线 0-6 切成 6 个四面体，每个四面体只有 16 种内外组合，而且三角形
// 的连线可以按「哪些棱发生了符号变化」直接推导出来 —— 不需要把 4096 项的
// MC 三角表硬编码进来（那种大表既难审查也容易抄错），同时天然 watertight。
//
// 后续处理链（与评审建议一致）：
//   提取 -> 去小连通分量 -> 去孤立三角形 -> Taubin 平滑 -> QEM 简化 -> 重算法线
// ============================================================================

/** 三角网格。positions/normals/colors 各 3 个 float 一个顶点，indices 3 个一个面。 */
struct Mesh {
    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> colors;   // 0..1
    std::vector<uint32_t> indices;

    size_t vertexCount() const { return positions.size() / 3; }
    size_t triangleCount() const { return indices.size() / 3; }
    bool empty() const { return indices.empty(); }
    void clear() {
        positions.clear();
        normals.clear();
        colors.clear();
        indices.clear();
    }
};

/** 提取/清理过程的完整统计，直接进 HUD 与反馈报告。 */
struct MeshBuildStats {
    size_t rawVertices = 0;
    size_t rawTriangles = 0;
    size_t outVertices = 0;
    size_t outTriangles = 0;
    size_t componentsBefore = 0;
    size_t componentsRemoved = 0;
    size_t trianglesRemovedRaw = 0;   // 孤立/退化面
    size_t trianglesRemovedComp = 0;  // 小连通分量
    size_t trianglesAfterDecimate = 0;
    size_t blocksScanned = 0;
    int voxelMin[3] = {0, 0, 0};
    int voxelMax[3] = {0, 0, 0};
    float voxelSize = 0.f;
    int smoothIterations = 0;
    int quality = 1;
    bool decimated = false;
    bool decimationSkipped = false;
    bool ok = false;
    std::string note;
    double extractMs = 0.0;
    double cleanupMs = 0.0;
    double totalMs = 0.0;
};

struct MeshOptions {
    /** 0 = preview / 1 = normal / 2 = hq */
    int quality = 1;
    size_t previewTriangles = 20000;
    size_t normalTriangles = 100000;
    size_t hqTriangles = 400000;
    /** 少于这个面数的连通分量整体丢弃（去漂浮垃圾片）。 */
    size_t minComponentTriangles = 50;
    int smoothIterations = 2;
    float taubinLambda = 0.50f;
    float taubinMu = -0.53f;
    /** 原始面数超过这个值就跳过 QEM（保护内存/耗时），只做粗糙化。 */
    size_t decimateHardLimit = 900000;
    bool enableComponentFilter = true;
    bool enableSmoothing = true;
    bool enableDecimation = true;
};

class MeshEngine {
public:
    /**
     * 从 TSDF 体素场提取并清理网格；结果放进 mesh()。
     * 整个过程是一次性操作（不是每帧），允许几百毫秒。
     */
    bool build(const TsdfEngine& tsdf, const MeshOptions& opt, MeshBuildStats& stats);

    const Mesh& mesh() const { return mesh_; }
    const MeshBuildStats& stats() const { return stats_; }

    size_t targetTriangles(const MeshOptions& opt) const;

private:
    Mesh mesh_;
    MeshBuildStats stats_;
};

// ---------------------------------------------------------------- 独立步骤
// 拆成自由函数，方便单独复用与日后替换实现。

/** 去掉面数少于 minTriangles 的连通分量（含顶点压缩）。 */
void meshRemoveSmallComponents(Mesh& mesh, size_t minTriangles, MeshBuildStats* stats);

/** 去掉不与任何面共边的孤立三角形（含顶点压缩）。 */
void meshRemoveIsolatedTriangles(Mesh& mesh, MeshBuildStats* stats);

/** Taubin 平滑（λ/μ 两步，交替进行；不会像 Laplacian 那样收缩体积）。 */
void meshTaubinSmooth(Mesh& mesh, int iterations, float lambda, float mu);

/** 按面积加权重算顶点法线。 */
void meshRecomputeNormals(Mesh& mesh);

/** QEM 边塌缩简化到不超过 targetTriangles 个面。返回是否真的做了简化。 */
bool meshDecimate(Mesh& mesh, size_t targetTriangles, MeshBuildStats* stats);

/** 顶点聚类粗糙化（QEM 的前置保护，O(n)，用于超大网格）。 */
void meshClusterSimplify(Mesh& mesh, float cellMeters);
