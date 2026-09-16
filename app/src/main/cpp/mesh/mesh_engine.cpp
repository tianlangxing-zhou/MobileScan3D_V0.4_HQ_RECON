#include "mesh_engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
//  Marching Tetrahedra
// ============================================================================
//
// 立方体 8 个角按固定编号：
//      0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
//      4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
//
// 沿主对角线 0-6 切成 6 个四面体，恰好无重叠地铺满立方体：
//      {0,5,1,6} {0,1,2,6} {0,2,3,6} {0,3,7,6} {0,7,4,6} {0,4,5,6}
//
// 四面体只有 4 个顶点 / 6 条棱，内外组合一共 16 种。三角形不需要查表 ——
// 直接看「哪几条棱发生了符号变化」：
//   3 条棱变号  -> 1 个三角形
//   4 条棱变号  -> 1 个四边形（拆成 2 个三角形）
//   其它        -> 无面
// 朝向统一用「内侧顶点质心 -> 外侧顶点质心」当参考方向修正，避免依赖
// 手抄的绕序表（那是最容易出错的地方）。
// ============================================================================

namespace {

const int kCornerOff[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

const int kTets[6][4] = {
    {0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
    {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}
};

const int kTetEdge[6][2] = {
    {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}
};

struct EdgeKey {
    int32_t x, y, z, axis;
    bool operator==(const EdgeKey& o) const {
        return x == o.x && y == o.y && z == o.z && axis == o.axis;
    }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const {
        uint64_t a = (uint64_t)(uint32_t)k.x * 73856093ull;
        uint64_t b = (uint64_t)(uint32_t)k.y * 19349663ull;
        uint64_t c = (uint64_t)(uint32_t)k.z * 83492791ull;
        uint64_t d = (uint64_t)(uint32_t)k.axis * 2654435761ull;
        return (size_t)(a ^ b ^ c ^ d);
    }
};

inline double msSince(const std::chrono::steady_clock::time_point& a,
                      const std::chrono::steady_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/** 压缩顶点：只保留仍被三角面引用的顶点，返回 旧下标 -> 新下标（-1 = 丢弃）。 */
std::vector<int32_t> compactVertices(Mesh& mesh) {
    const size_t nv = mesh.vertexCount();
    std::vector<int32_t> remap(nv, -1);
    if (nv == 0) {
        return remap;
    }
    const bool hasNormal = mesh.normals.size() >= nv * 3;
    const bool hasColor = mesh.colors.size() >= nv * 3;

    std::vector<float> np, nn, nc;
    np.reserve(mesh.positions.size());

    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        for (int k = 0; k < 3; ++k) {
            const uint32_t v = mesh.indices[t + k];
            if (v >= nv) {
                continue;
            }
            if (remap[v] < 0) {
                remap[v] = (int32_t)(np.size() / 3);
                np.push_back(mesh.positions[v * 3 + 0]);
                np.push_back(mesh.positions[v * 3 + 1]);
                np.push_back(mesh.positions[v * 3 + 2]);
                if (hasNormal) {
                    nn.push_back(mesh.normals[v * 3 + 0]);
                    nn.push_back(mesh.normals[v * 3 + 1]);
                    nn.push_back(mesh.normals[v * 3 + 2]);
                }
                if (hasColor) {
                    nc.push_back(mesh.colors[v * 3 + 0]);
                    nc.push_back(mesh.colors[v * 3 + 1]);
                    nc.push_back(mesh.colors[v * 3 + 2]);
                }
            }
            mesh.indices[t + k] = (uint32_t)remap[v];
        }
    }

    mesh.positions.swap(np);
    if (hasNormal) {
        mesh.normals.swap(nn);
    } else {
        mesh.normals.clear();
    }
    if (hasColor) {
        mesh.colors.swap(nc);
    } else {
        mesh.colors.clear();
    }
    return remap;
}

/**
 * 去重三角形的顶点对键（无向）。
 */
inline uint64_t edgePairKey(uint32_t a, uint32_t b) {
    if (a > b) {
        std::swap(a, b);
    }
    return ((uint64_t)a << 32) | (uint64_t)b;
}

/**
 * 顶点坐标的整数格子键（用于聚类简化）。
 */
inline uint64_t cellKey(int32_t x, int32_t y, int32_t z) {
    return ((uint64_t)(uint32_t)x * 73856093ull) ^
           ((uint64_t)(uint32_t)y * 19349663ull) ^
           ((uint64_t)(uint32_t)z * 83492791ull);
}

}  // namespace

// ============================================================================
//  提取
// ============================================================================

/**
 * 遍历已分配的块，逐块取「2x2x2 块邻域」缓存，把几百万次体素读取
 * 全部退化成数组直读（而不是每次 hash 查找）。
 */
static void extractMarchingTetrahedra(const TsdfEngine& tsdf, float voxel,
                                      Mesh& mesh, MeshBuildStats& stats) {
    int mn[3], mx[3];
    if (!tsdf.voxelBounds(mn, mx)) {
        stats.note = "tsdf empty";
        return;
    }
    for (int i = 0; i < 3; ++i) {
        stats.voxelMin[i] = mn[i];
        stats.voxelMax[i] = mx[i];
    }
    stats.voxelSize = voxel;

    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> vertOfEdge;
    vertOfEdge.reserve(1 << 16);

    mesh.positions.reserve(1 << 18);
    mesh.indices.reserve(1 << 18);

    // 每个顶点的法线稍后由 TSDF 梯度算；这里先记下顶点落在哪条棱上。
    std::vector<EdgeKey> vertEdge;

    auto vertexForEdge = [&](int ax, int ay, int az, int bx, int by, int bz,
                             float fa, float fb) -> uint32_t {
        // 棱键：取字典序较小的端点 + 轴向
        int lo[3];
        int axis = -1;
        const int a3[3] = {ax, ay, az};
        const int b3[3] = {bx, by, bz};
        for (int i = 0; i < 3; ++i) {
            lo[i] = std::min(a3[i], b3[i]);
            if (a3[i] != b3[i]) {
                axis = i;
            }
        }
        if (axis < 0) {
            axis = 0;
        }
        const EdgeKey key{lo[0], lo[1], lo[2], axis};
        auto it = vertOfEdge.find(key);
        if (it != vertOfEdge.end()) {
            return it->second;
        }

        float denom = fa - fb;
        float t = (std::fabs(denom) > 1e-9f) ? (fa / denom) : 0.5f;
        t = std::clamp(t, 0.f, 1.f);
        const float px = (a3[0] + t * (b3[0] - a3[0])) * voxel;
        const float py = (a3[1] + t * (b3[1] - a3[1])) * voxel;
        const float pz = (a3[2] + t * (b3[2] - a3[2])) * voxel;

        const uint32_t id = (uint32_t)(mesh.positions.size() / 3);
        mesh.positions.push_back(px);
        mesh.positions.push_back(py);
        mesh.positions.push_back(pz);
        vertOfEdge.emplace(key, id);
        vertEdge.push_back(key);
        return id;
    };

    uint64_t cells = 0;
    std::vector<uint32_t> tri;
    tri.reserve(16);

    tsdf.forEachBlock([&](int bx, int by, int bz, const TsdfBlock* self) {
        stats.blocksScanned++;
        const TsdfBlock* nb[8];
        nb[0] = self;
        nb[1] = tsdf.blockAt(bx + 1, by, bz);
        nb[2] = tsdf.blockAt(bx, by + 1, bz);
        nb[3] = tsdf.blockAt(bx + 1, by + 1, bz);
        nb[4] = tsdf.blockAt(bx, by, bz + 1);
        nb[5] = tsdf.blockAt(bx + 1, by, bz + 1);
        nb[6] = tsdf.blockAt(bx, by + 1, bz + 1);
        nb[7] = tsdf.blockAt(bx + 1, by + 1, bz + 1);

        const int gbase[3] = {bx << kTsdfBlockShift,
                              by << kTsdfBlockShift,
                              bz << kTsdfBlockShift};

        for (int lz = 0; lz < kTsdfBlockSize; ++lz) {
            for (int ly = 0; ly < kTsdfBlockSize; ++ly) {
                for (int lx = 0; lx < kTsdfBlockSize; ++lx) {
                    float f[8];
                    int validCount = 0;
                    float fmin = 1e9f;

                    for (int c = 0; c < 8; ++c) {
                        const int gx = lx + kCornerOff[c][0];
                        const int gy = ly + kCornerOff[c][1];
                        const int gz = lz + kCornerOff[c][2];
                        const TsdfBlock* b =
                            nb[(gx >> kTsdfBlockShift) |
                               ((gy >> kTsdfBlockShift) << 1) |
                               ((gz >> kTsdfBlockShift) << 2)];
                        if (!b) {
                            f[c] = 1.f;
                            continue;
                        }
                        const int px = gx & kTsdfBlockMask;
                        const int py = gy & kTsdfBlockMask;
                        const int pz = gz & kTsdfBlockMask;
                        const TsdfVoxel& vv =
                            b->voxels[(pz * kTsdfBlockSize + py) * kTsdfBlockSize + px];
                        if (vv.weight == 0) {
                            f[c] = 1.f;
                        } else {
                            f[c] = vv.tsdf / kTsdfValueScale;
                            ++validCount;
                        }
                        if (f[c] < fmin) {
                            fmin = f[c];
                        }
                    }

                    // 快速跳过：整格都在外侧（没有任何负值）就没有表面。
                    if (fmin >= 0.f) {
                        continue;
                    }
                    // 抗噪：至少 2 个角有真实观测。体积边界上孤立的负值
                    // 不足以构成可信的表面，否则会在包围盒边缘长出一层假面。
                    if (validCount < 2) {
                        continue;
                    }

                    ++cells;
                    const int cx = gbase[0] + lx;
                    const int cy = gbase[1] + ly;
                    const int cz = gbase[2] + lz;

                    for (int ti = 0; ti < 6; ++ti) {
                        const int* tet = kTets[ti];
                        float tf[4];
                        int insideMask = 0;
                        int inIdx[4];
                        int outIdx[4];
                        int nIn = 0, nOut = 0;
                        float inC[3] = {0, 0, 0};
                        float outC[3] = {0, 0, 0};
                        float tp[4][3];

                        for (int k = 0; k < 4; ++k) {
                            const int c = tet[k];
                            tf[k] = f[c];
                            tp[k][0] = (cx + kCornerOff[c][0]) * voxel;
                            tp[k][1] = (cy + kCornerOff[c][1]) * voxel;
                            tp[k][2] = (cz + kCornerOff[c][2]) * voxel;
                            if (tf[k] < 0.f) {
                                insideMask |= (1 << k);
                                inIdx[nIn++] = k;
                                inC[0] += tp[k][0];
                                inC[1] += tp[k][1];
                                inC[2] += tp[k][2];
                            } else {
                                outIdx[nOut++] = k;
                                outC[0] += tp[k][0];
                                outC[1] += tp[k][1];
                                outC[2] += tp[k][2];
                            }
                        }
                        if (insideMask == 0 || insideMask == 15) {
                            continue;
                        }

                        // 找出所有发生符号变化的棱，并算出交点顶点
                        int crossEdges[6];
                        int nCross = 0;
                        uint32_t crossVert[6];
                        for (int e = 0; e < 6; ++e) {
                            const int a = kTetEdge[e][0];
                            const int b = kTetEdge[e][1];
                            if ((tf[a] < 0.f) == (tf[b] < 0.f)) {
                                continue;
                            }
                            const int ca = tet[a];
                            const int cb = tet[b];
                            crossEdges[nCross] = e;
                            crossVert[nCross] = vertexForEdge(
                                cx + kCornerOff[ca][0], cy + kCornerOff[ca][1],
                                cz + kCornerOff[ca][2],
                                cx + kCornerOff[cb][0], cy + kCornerOff[cb][1],
                                cz + kCornerOff[cb][2],
                                tf[a], tf[b]);
                            ++nCross;
                        }
                        if (nCross != 3 && nCross != 4) {
                            continue;
                        }

                        // 参考朝向：内侧质心 -> 外侧质心（即 TSDF 增大的方向）
                        const float invIn = 1.f / (float)std::max(1, nIn);
                        const float invOut = 1.f / (float)std::max(1, nOut);
                        const float ref[3] = {
                            outC[0] * invOut - inC[0] * invIn,
                            outC[1] * invOut - inC[1] * invIn,
                            outC[2] * invOut - inC[2] * invIn};

                        tri.clear();
                        if (nCross == 3) {
                            tri.push_back(crossVert[0]);
                            tri.push_back(crossVert[1]);
                            tri.push_back(crossVert[2]);
                        } else {
                            // 2 内 2 外：4 个交点构成四边形，按
                            // (in0,out0) -> (in0,out1) -> (in1,out1) -> (in1,out0) 成环
                            uint32_t vidx[2][2] = {{0, 0}, {0, 0}};
                            bool have[2][2] = {{false, false}, {false, false}};
                            for (int k = 0; k < nCross; ++k) {
                                const int a = kTetEdge[crossEdges[k]][0];
                                const int b = kTetEdge[crossEdges[k]][1];
                                int ia = -1, ib = -1;
                                if (tf[a] < 0.f) {
                                    ia = (a == inIdx[0]) ? 0 : 1;
                                    ib = (b == outIdx[0]) ? 0 : 1;
                                } else {
                                    ia = (b == inIdx[0]) ? 0 : 1;
                                    ib = (a == outIdx[0]) ? 0 : 1;
                                }
                                if (ia < 0 || ib < 0) {
                                    continue;
                                }
                                have[ia][ib] = true;
                                vidx[ia][ib] = crossVert[k];
                            }
                            if (have[0][0] && have[0][1] && have[1][0] && have[1][1]) {
                                const uint32_t q0 = vidx[0][0];
                                const uint32_t q1 = vidx[0][1];
                                const uint32_t q2 = vidx[1][1];
                                const uint32_t q3 = vidx[1][0];
                                tri.push_back(q0);
                                tri.push_back(q1);
                                tri.push_back(q2);
                                tri.push_back(q0);
                                tri.push_back(q2);
                                tri.push_back(q3);
                            }
                        }

                        for (size_t f0 = 0; f0 + 2 < tri.size(); f0 += 3) {
                            uint32_t a = tri[f0];
                            uint32_t b = tri[f0 + 1];
                            uint32_t c = tri[f0 + 2];
                            if (a == b || b == c || a == c) {
                                continue;
                            }
                            // 朝向修正
                            const float* pa = &mesh.positions[a * 3];
                            const float* pb = &mesh.positions[b * 3];
                            const float* pc = &mesh.positions[c * 3];
                            const float e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
                            const float e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
                            const float nx = e1[1] * e2[2] - e1[2] * e2[1];
                            const float ny = e1[2] * e2[0] - e1[0] * e2[2];
                            const float nz = e1[0] * e2[1] - e1[1] * e2[0];
                            if (nx * ref[0] + ny * ref[1] + nz * ref[2] < 0.f) {
                                std::swap(b, c);
                            }
                            mesh.indices.push_back(a);
                            mesh.indices.push_back(b);
                            mesh.indices.push_back(c);
                        }
                    }
                }
            }
        }
    });

    stats.rawVertices = mesh.vertexCount();
    stats.rawTriangles = mesh.triangleCount();

    // ---- 法线：用 TSDF 梯度的中心差分（比三角形 flat normal 稳得多） ----
    mesh.normals.assign(mesh.positions.size(), 0.f);
    for (size_t i = 0; i < vertEdge.size(); ++i) {
        const EdgeKey& e = vertEdge[i];
        int vx = e.x, vy = e.y, vz = e.z;
        if (e.axis == 0) {
            vx += 1;
        } else if (e.axis == 1) {
            vy += 1;
        } else {
            vz += 1;
        }
        const float gx = tsdf.valueAt(vx + 1, vy, vz) - tsdf.valueAt(vx - 1, vy, vz);
        const float gy = tsdf.valueAt(vx, vy + 1, vz) - tsdf.valueAt(vx, vy - 1, vz);
        const float gz = tsdf.valueAt(vx, vy, vz + 1) - tsdf.valueAt(vx, vy, vz - 1);
        float len = std::sqrt(gx * gx + gy * gy + gz * gz);
        if (!(len > 1e-6f)) {
            continue;
        }
        const float inv = 1.f / len;
        // TSDF 的正方向指向表面外侧，所以梯度方向就是外法线。
        mesh.normals[i * 3 + 0] = gx * inv;
        mesh.normals[i * 3 + 1] = gy * inv;
        mesh.normals[i * 3 + 2] = gz * inv;
    }

    // 颜色：从体素 565 直接取（顶点落在棱上，取两个端点的近似平均）
    mesh.colors.assign(mesh.positions.size(), 0.6f);
    for (size_t i = 0; i < vertEdge.size(); ++i) {
        const EdgeKey& e = vertEdge[i];
        int vx = e.x, vy = e.y, vz = e.z;
        if (e.axis == 0) {
            vx += 1;
        } else if (e.axis == 1) {
            vy += 1;
        } else {
            vz += 1;
        }
        const uint16_t c = tsdf.color565At(vx, vy, vz);
        const int r5 = (c >> 11) & 0x1F;
        const int g6 = (c >> 5) & 0x3F;
        const int b5 = c & 0x1F;
        mesh.colors[i * 3 + 0] = r5 / 31.f;
        mesh.colors[i * 3 + 1] = g6 / 63.f;
        mesh.colors[i * 3 + 2] = b5 / 31.f;
    }
}

// ============================================================================
//  清理步骤
// ============================================================================

void meshRemoveSmallComponents(Mesh& mesh, size_t minTriangles, MeshBuildStats* stats) {
    const size_t nv = mesh.vertexCount();
    const size_t nt = mesh.triangleCount();
    if (nv == 0 || nt == 0) {
        return;
    }
    std::vector<uint32_t> parent(nv);
    for (size_t i = 0; i < nv; ++i) {
        parent[i] = (uint32_t)i;
    }
    auto find = [&parent](uint32_t a) {
        uint32_t root = a;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[a] != root) {
            const uint32_t next = parent[a];
            parent[a] = root;
            a = next;
        }
        return root;
    };
    auto unite = [&find, &parent](uint32_t a, uint32_t b) {
        const uint32_t ra = find(a);
        const uint32_t rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    };

    for (size_t t = 0; t < nt; ++t) {
        const uint32_t* idx = &mesh.indices[t * 3];
        unite(idx[0], idx[1]);
        unite(idx[0], idx[2]);
    }

    std::unordered_map<uint32_t, size_t> triPerRoot;
    triPerRoot.reserve(nt);
    for (size_t t = 0; t < nt; ++t) {
        triPerRoot[find(mesh.indices[t * 3])]++;
    }

    std::vector<uint32_t> kept;
    kept.reserve(mesh.indices.size());
    for (size_t t = 0; t < nt; ++t) {
        const uint32_t* idx = &mesh.indices[t * 3];
        const uint32_t root = find(idx[0]);
        const auto it = triPerRoot.find(root);
        if (it == triPerRoot.end() || it->second < minTriangles) {
            continue;
        }
        kept.push_back(idx[0]);
        kept.push_back(idx[1]);
        kept.push_back(idx[2]);
    }

    if (stats) {
        stats->componentsBefore = triPerRoot.size();
        size_t removedComps = 0;
        for (const auto& kv : triPerRoot) {
            if (kv.second < minTriangles) {
                ++removedComps;
            }
        }
        stats->componentsRemoved = removedComps;
        stats->trianglesRemovedComp += (nt - kept.size() / 3);
    }
    if (kept.size() == mesh.indices.size()) {
        return;
    }
    mesh.indices.swap(kept);
    compactVertices(mesh);
}

void meshRemoveIsolatedTriangles(Mesh& mesh, MeshBuildStats* stats) {
    const size_t nt = mesh.triangleCount();
    if (nt == 0) {
        return;
    }
    std::unordered_map<uint64_t, int> edgeCount;
    edgeCount.reserve(nt * 3);
    for (size_t t = 0; t < nt; ++t) {
        const uint32_t* idx = &mesh.indices[t * 3];
        edgeCount[edgePairKey(idx[0], idx[1])]++;
        edgeCount[edgePairKey(idx[1], idx[2])]++;
        edgeCount[edgePairKey(idx[2], idx[0])]++;
    }

    std::vector<uint32_t> kept;
    kept.reserve(mesh.indices.size());
    for (size_t t = 0; t < nt; ++t) {
        const uint32_t* idx = &mesh.indices[t * 3];
        const bool isolated = edgeCount[edgePairKey(idx[0], idx[1])] <= 1 &&
                              edgeCount[edgePairKey(idx[1], idx[2])] <= 1 &&
                              edgeCount[edgePairKey(idx[2], idx[0])] <= 1;
        if (isolated) {
            continue;
        }
        kept.push_back(idx[0]);
        kept.push_back(idx[1]);
        kept.push_back(idx[2]);
    }
    if (kept.size() == mesh.indices.size()) {
        return;
    }
    if (stats) {
        stats->trianglesRemovedRaw += (nt - kept.size() / 3);
    }
    mesh.indices.swap(kept);
    compactVertices(mesh);
}

// ============================================================================
//  Taubin 平滑
// ============================================================================

static void taubinStep(Mesh& mesh,
                       const std::vector<std::vector<uint32_t>>& adj,
                       float factor,
                       std::vector<float>& tmp) {
    const size_t nv = mesh.vertexCount();
    tmp.resize(mesh.positions.size());
    const std::vector<float>& p = mesh.positions;
    for (size_t i = 0; i < nv; ++i) {
        const std::vector<uint32_t>& n = adj[i];
        if (n.empty()) {
            tmp[i * 3 + 0] = p[i * 3 + 0];
            tmp[i * 3 + 1] = p[i * 3 + 1];
            tmp[i * 3 + 2] = p[i * 3 + 2];
            continue;
        }
        float sx = 0.f, sy = 0.f, sz = 0.f;
        for (uint32_t j : n) {
            sx += p[j * 3 + 0];
            sy += p[j * 3 + 1];
            sz += p[j * 3 + 2];
        }
        const float inv = 1.f / (float)n.size();
        tmp[i * 3 + 0] = p[i * 3 + 0] + factor * (sx * inv - p[i * 3 + 0]);
        tmp[i * 3 + 1] = p[i * 3 + 1] + factor * (sy * inv - p[i * 3 + 1]);
        tmp[i * 3 + 2] = p[i * 3 + 2] + factor * (sz * inv - p[i * 3 + 2]);
    }
    mesh.positions.swap(tmp);
}

void meshTaubinSmooth(Mesh& mesh, int iterations, float lambda, float mu) {
    const size_t nv = mesh.vertexCount();
    if (nv == 0 || iterations <= 0) {
        return;
    }
    std::vector<std::vector<uint32_t>> adj(nv);
    std::unordered_set<uint64_t> seen;
    seen.reserve(mesh.indices.size() * 2);
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t a = mesh.indices[t], b = mesh.indices[t + 1], c = mesh.indices[t + 2];
        const uint32_t pairs[3][2] = {{a, b}, {b, c}, {c, a}};
        for (const auto& pr : pairs) {
            const uint64_t k = edgePairKey(pr[0], pr[1]);
            if (seen.insert(k).second) {
                adj[pr[0]].push_back(pr[1]);
                adj[pr[1]].push_back(pr[0]);
            }
        }
    }
    std::vector<float> tmp;
    for (int it = 0; it < iterations; ++it) {
        taubinStep(mesh, adj, lambda, tmp);
        taubinStep(mesh, adj, mu, tmp);
    }
}

// ============================================================================
//  法线重算
// ============================================================================

void meshRecomputeNormals(Mesh& mesh) {
    const size_t nv = mesh.vertexCount();
    mesh.normals.assign(nv * 3, 0.f);
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t ia = mesh.indices[t], ib = mesh.indices[t + 1], ic = mesh.indices[t + 2];
        if (ia >= nv || ib >= nv || ic >= nv) {
            continue;
        }
        const float* pa = &mesh.positions[ia * 3];
        const float* pb = &mesh.positions[ib * 3];
        const float* pc = &mesh.positions[ic * 3];
        const float e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
        const float e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
        // 不归一化 => 面积加权
        const float nx = e1[1] * e2[2] - e1[2] * e2[1];
        const float ny = e1[2] * e2[0] - e1[0] * e2[2];
        const float nz = e1[0] * e2[1] - e1[1] * e2[0];
        const uint32_t ids[3] = {ia, ib, ic};
        for (uint32_t id : ids) {
            mesh.normals[id * 3 + 0] += nx;
            mesh.normals[id * 3 + 1] += ny;
            mesh.normals[id * 3 + 2] += nz;
        }
    }
    for (size_t i = 0; i < nv; ++i) {
        float* n = &mesh.normals[i * 3];
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 1e-8f) {
            const float inv = 1.f / len;
            n[0] *= inv;
            n[1] *= inv;
            n[2] *= inv;
        } else {
            n[0] = 0.f;
            n[1] = 0.f;
            n[2] = 1.f;
        }
    }
}

// ============================================================================
//  顶点聚类粗糙化（QEM 的前置保护）
// ============================================================================

void meshClusterSimplify(Mesh& mesh, float cellMeters) {
    const size_t nv = mesh.vertexCount();
    if (nv == 0 || cellMeters <= 1e-5f) {
        return;
    }
    const float inv = 1.f / cellMeters;
    struct Acc {
        double x = 0, y = 0, z = 0;
        double r = 0, g = 0, b = 0;
        int n = 0;
    };
    std::vector<Acc> cells;
    cells.reserve(nv / 2 + 8);
    std::unordered_map<uint64_t, uint32_t> cellOf;
    cellOf.reserve(nv * 2);
    std::vector<uint32_t> vcell(nv, 0);
    const bool hasColor = mesh.colors.size() >= nv * 3;

    for (size_t i = 0; i < nv; ++i) {
        const float x = mesh.positions[i * 3 + 0];
        const float y = mesh.positions[i * 3 + 1];
        const float z = mesh.positions[i * 3 + 2];
        const int32_t kx = (int32_t)std::floor(x * inv);
        const int32_t ky = (int32_t)std::floor(y * inv);
        const int32_t kz = (int32_t)std::floor(z * inv);
        const uint64_t key = cellKey(kx, ky, kz);
        auto it = cellOf.find(key);
        uint32_t ci;
        if (it == cellOf.end()) {
            ci = (uint32_t)cells.size();
            cells.push_back(Acc{});
            cellOf.emplace(key, ci);
        } else {
            ci = it->second;
        }
        vcell[i] = ci;
        Acc& a = cells[ci];
        a.x += x;
        a.y += y;
        a.z += z;
        if (hasColor) {
            a.r += mesh.colors[i * 3 + 0];
            a.g += mesh.colors[i * 3 + 1];
            a.b += mesh.colors[i * 3 + 2];
        }
        a.n++;
    }

    std::vector<float> np, nc;
    np.reserve(cells.size() * 3);
    nc.reserve(cells.size() * 3);
    for (const Acc& a : cells) {
        const double invN = 1.0 / std::max(1, a.n);
        np.push_back((float)(a.x * invN));
        np.push_back((float)(a.y * invN));
        np.push_back((float)(a.z * invN));
        if (hasColor) {
            nc.push_back((float)(a.r * invN));
            nc.push_back((float)(a.g * invN));
            nc.push_back((float)(a.b * invN));
        }
    }

    std::vector<uint32_t> ni;
    ni.reserve(mesh.indices.size());
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t a = vcell[mesh.indices[t]];
        const uint32_t b = vcell[mesh.indices[t + 1]];
        const uint32_t c = vcell[mesh.indices[t + 2]];
        if (a == b || b == c || a == c) {
            continue;
        }
        ni.push_back(a);
        ni.push_back(b);
        ni.push_back(c);
    }

    mesh.positions.swap(np);
    mesh.colors.swap(nc);
    mesh.indices.swap(ni);
    meshRecomputeNormals(mesh);
}

// ============================================================================
//  QEM 边塌缩简化
// ============================================================================

namespace {

/** 对称 4x4 二次误差矩阵，压成 10 个 float。 */
struct Quadric {
    float q[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    void addPlane(float a, float b, float c, float d) {
        q[0] += a * a; q[1] += a * b; q[2] += a * c; q[3] += a * d;
        q[4] += b * b; q[5] += b * c; q[6] += b * d;
        q[7] += c * c; q[8] += c * d;
        q[9] += d * d;
    }
    void add(const Quadric& o) {
        for (int i = 0; i < 10; ++i) {
            q[i] += o.q[i];
        }
    }
    float eval(float x, float y, float z) const {
        return q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x +
               q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y +
               q[7] * z * z + 2 * q[8] * z + q[9];
    }
    Quadric operator+(const Quadric& o) const {
        Quadric r = *this;
        r.add(o);
        return r;
    }
};

struct QEntry {
    float cost;
    uint32_t v0;
    uint32_t v1;
    uint32_t ver0;
    uint32_t ver1;
};

struct QGreater {
    bool operator()(const QEntry& a, const QEntry& b) const { return a.cost > b.cost; }
};

}  // namespace

bool meshDecimate(Mesh& mesh, size_t targetTriangles, MeshBuildStats* stats) {
    size_t liveTris = mesh.triangleCount();
    if (targetTriangles == 0 || liveTris <= targetTriangles) {
        return false;
    }
    const size_t nv = mesh.vertexCount();

    std::vector<std::array<uint32_t, 3>> tris(liveTris);
    for (size_t t = 0; t < liveTris; ++t) {
        tris[t] = {mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]};
    }
    std::vector<uint8_t> triValid(liveTris, 1);

    std::vector<uint8_t> vValid(nv, 1);
    std::vector<uint32_t> vVer(nv, 0);
    std::vector<Quadric> quad(nv);
    std::vector<std::vector<uint32_t>> vTris(nv);

    for (size_t t = 0; t < liveTris; ++t) {
        const auto& tr = tris[t];
        if (tr[0] >= nv || tr[1] >= nv || tr[2] >= nv) {
            triValid[t] = 0;
            continue;
        }
        const float* pa = &mesh.positions[tr[0] * 3];
        const float* pb = &mesh.positions[tr[1] * 3];
        const float* pc = &mesh.positions[tr[2] * 3];
        const float e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
        const float e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
        float nx = e1[1] * e2[2] - e1[2] * e2[1];
        float ny = e1[2] * e2[0] - e1[0] * e2[2];
        float nz = e1[0] * e2[1] - e1[1] * e2[0];
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!(len > 1e-12f)) {
            triValid[t] = 0;
            continue;
        }
        const float inv = 1.f / len;
        nx *= inv;
        ny *= inv;
        nz *= inv;
        const float d = -(nx * pa[0] + ny * pa[1] + nz * pa[2]);
        quad[tr[0]].addPlane(nx, ny, nz, d);
        quad[tr[1]].addPlane(nx, ny, nz, d);
        quad[tr[2]].addPlane(nx, ny, nz, d);
        vTris[tr[0]].push_back((uint32_t)t);
        vTris[tr[1]].push_back((uint32_t)t);
        vTris[tr[2]].push_back((uint32_t)t);
    }

    std::priority_queue<QEntry, std::vector<QEntry>, QGreater> pq;
    std::unordered_set<uint64_t> edgeSeen;
    edgeSeen.reserve(liveTris * 3);
    auto tryPush = [&](uint32_t a, uint32_t b) {
        if (a == b || a >= nv || b >= nv) {
            return;
        }
        const uint64_t k = edgePairKey(a, b);
        if (!edgeSeen.insert(k).second) {
            return;
        }
        const float mx = (mesh.positions[a * 3 + 0] + mesh.positions[b * 3 + 0]) * 0.5f;
        const float my = (mesh.positions[a * 3 + 1] + mesh.positions[b * 3 + 1]) * 0.5f;
        const float mz = (mesh.positions[a * 3 + 2] + mesh.positions[b * 3 + 2]) * 0.5f;
        const Quadric q = quad[a] + quad[b];
        pq.push(QEntry{q.eval(mx, my, mz), a, b, vVer[a], vVer[b]});
    };
    for (size_t t = 0; t < liveTris; ++t) {
        if (!triValid[t]) {
            continue;
        }
        const auto& tr = tris[t];
        tryPush(tr[0], tr[1]);
        tryPush(tr[1], tr[2]);
        tryPush(tr[2], tr[0]);
    }

    const size_t maxPops = nv * 40 + 1000;
    size_t pops = 0;
    while (liveTris > targetTriangles && !pq.empty() && pops < maxPops) {
        ++pops;
        const QEntry e = pq.top();
        pq.pop();
        if (!vValid[e.v0] || !vValid[e.v1]) {
            continue;
        }
        if (vVer[e.v0] != e.ver0 || vVer[e.v1] != e.ver1) {
            continue;  // 过期条目
        }
        const uint32_t keep = e.v0;
        const uint32_t drop = e.v1;

        // 放到中点（不做矩阵求解，避免奇异时的数值爆炸）
        mesh.positions[keep * 3 + 0] =
            (mesh.positions[keep * 3 + 0] + mesh.positions[drop * 3 + 0]) * 0.5f;
        mesh.positions[keep * 3 + 1] =
            (mesh.positions[keep * 3 + 1] + mesh.positions[drop * 3 + 1]) * 0.5f;
        mesh.positions[keep * 3 + 2] =
            (mesh.positions[keep * 3 + 2] + mesh.positions[drop * 3 + 2]) * 0.5f;
        if (mesh.colors.size() >= nv * 3) {
            for (int k = 0; k < 3; ++k) {
                mesh.colors[keep * 3 + k] =
                    (mesh.colors[keep * 3 + k] + mesh.colors[drop * 3 + k]) * 0.5f;
            }
        }
        quad[keep] = quad[keep] + quad[drop];
        vValid[drop] = 0;
        ++vVer[keep];
        ++vVer[drop];

        // 只处理「因为 drop 消失而发生变化」的那些三角形。
        // 注意：不能再遍历 vTris[keep] —— 那个列表会随塌缩不断变长，
        // 遍历它会让整体复杂度退化成 O(n^2)。
        std::vector<uint32_t> changed;
        changed.reserve(vTris[drop].size());
        for (uint32_t ti : vTris[drop]) {
            if (!triValid[ti]) {
                continue;
            }
            std::array<uint32_t, 3>& tr = tris[ti];
            for (int k = 0; k < 3; ++k) {
                if (tr[k] == drop) {
                    tr[k] = keep;
                }
            }
            if (tr[0] == tr[1] || tr[1] == tr[2] || tr[0] == tr[2]) {
                triValid[ti] = 0;
                --liveTris;
                continue;
            }
            changed.push_back(ti);
        }
        // 邻接表合并（保持后续塌缩的正确性），但不再整体扫描它。
        vTris[keep].insert(vTris[keep].end(), changed.begin(), changed.end());
        vTris[drop].clear();

        // 以 keep 为中心，把受影响的边重新投递
        for (uint32_t ti : changed) {
            const std::array<uint32_t, 3>& tr = tris[ti];
            for (int k = 0; k < 3; ++k) {
                const uint32_t a = tr[k];
                const uint32_t b = tr[(k + 1) % 3];
                if (a == keep || b == keep) {
                    edgeSeen.erase(edgePairKey(a, b));
                    tryPush(a, b);
                }
            }
        }
    }

    // 回写
    std::vector<uint32_t> outIdx;
    outIdx.reserve(liveTris * 3);
    for (size_t t = 0; t < tris.size(); ++t) {
        if (!triValid[t]) {
            continue;
        }
        outIdx.push_back(tris[t][0]);
        outIdx.push_back(tris[t][1]);
        outIdx.push_back(tris[t][2]);
    }
    mesh.indices.swap(outIdx);
    compactVertices(mesh);
    meshRecomputeNormals(mesh);
    if (stats) {
        stats->trianglesAfterDecimate = mesh.triangleCount();
        stats->decimated = true;
    }
    return true;
}

// ============================================================================
//  MeshEngine
// ============================================================================

size_t MeshEngine::targetTriangles(const MeshOptions& opt) const {
    switch (opt.quality) {
        case 0: return opt.previewTriangles;
        case 2: return opt.hqTriangles;
        default: return opt.normalTriangles;
    }
}

bool MeshEngine::build(const TsdfEngine& tsdf, const MeshOptions& opt,
                       MeshBuildStats& stats) {
    stats = MeshBuildStats{};
    stats.quality = opt.quality;
    const auto t0 = std::chrono::steady_clock::now();

    Mesh m;
    extractMarchingTetrahedra(tsdf, tsdf.voxelSize(), m, stats);

    const auto t1 = std::chrono::steady_clock::now();
    stats.extractMs = msSince(t0, t1);

    if (m.indices.empty()) {
        stats.note = "no triangles extracted";
        stats.ok = false;
        stats.totalMs = msSince(t0, std::chrono::steady_clock::now());
        mesh_ = Mesh{};
        stats_ = stats;
        return false;
    }

    if (opt.enableComponentFilter) {
        meshRemoveSmallComponents(m, opt.minComponentTriangles, &stats);
        meshRemoveIsolatedTriangles(m, &stats);
    }

    // 超大网格先做一次聚类粗糙化，避免 QEM 在几十万面上跑到超时。
    if (opt.enableDecimation && m.triangleCount() > opt.decimateHardLimit) {
        const float cell = std::max(tsdf.voxelSize() * 2.f, 0.004f);
        meshClusterSimplify(m, cell);
        stats.note += "cluster-presimplified;";
    }

    if (opt.enableSmoothing && opt.smoothIterations > 0 && !m.indices.empty()) {
        meshTaubinSmooth(m, opt.smoothIterations, opt.taubinLambda, opt.taubinMu);
        meshRecomputeNormals(m);
        stats.smoothIterations = opt.smoothIterations;
    }

    if (opt.enableDecimation && !m.indices.empty()) {
        const size_t target = targetTriangles(opt);
        if (m.triangleCount() > target) {
            if (!meshDecimate(m, target, &stats)) {
                stats.decimationSkipped = true;
            }
        }
    }

    if (m.normals.size() != m.positions.size()) {
        meshRecomputeNormals(m);
    }
    if (m.colors.size() != m.positions.size()) {
        m.colors.assign(m.positions.size(), 0.6f);
    }

    const auto t2 = std::chrono::steady_clock::now();
    stats.cleanupMs = msSince(t1, t2);
    stats.totalMs = msSince(t0, t2);
    stats.outVertices = m.vertexCount();
    stats.outTriangles = m.triangleCount();
    stats.ok = stats.outTriangles > 0;
    if (stats.note.empty()) {
        stats.note = "ok";
    }

    mesh_ = std::move(m);
    stats_ = stats;
    return stats.ok;
}
