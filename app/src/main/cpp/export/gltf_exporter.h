#pragma once

#include <cstdint>
#include <string>

#include "../mesh/mesh_engine.h"

// ============================================================================
//  GLB（glTF 2.0 二进制）导出
// ============================================================================
//
// 为什么不再让 PLY 当最终资产：PLY 只有顶点、没有三角面，也不是 AR 生态的
// 通用格式。GLB 是单文件二进制，自带 mesh / normal / vertex color /
// material / scene 结构，Filament、three.js、Blender、Unity 都能直接打开。
//
// 第一版刻意只做 **vertex color**，不做纹理贴图（评审建议的开发顺序）：
// 现在已经有 RGB + Depth + Pose，vertex color 立刻可用；纹理 atlas
// （xatlas + HQ keyframes 多视角投影）留到第二阶段，不要一开始就卡在那里。
// ============================================================================

struct GlbExportStats {
    size_t vertices = 0;
    size_t triangles = 0;
    size_t jsonBytes = 0;
    size_t binBytes = 0;
    size_t fileBytes = 0;
    bool hasVertexColor = false;
    bool ok = false;
    std::string note;
};

/**
 * 把网格写成 glTF 2.0 二进制 (.glb)。
 *
 * accessor 布局：
 *   0 POSITION  VEC3 FLOAT        （必带 min/max，规范要求）
 *   1 NORMAL    VEC3 FLOAT
 *   2 COLOR_0   VEC4 UNSIGNED_BYTE normalized（可选）
 *   3 indices   SCALAR UNSIGNED_INT
 */
bool exportGlb(const Mesh& mesh, const std::string& path,
               const std::string& name, GlbExportStats* stats);
