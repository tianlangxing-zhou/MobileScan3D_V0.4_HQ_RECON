#pragma once

// ============================================================================
//  AosMesh —— V0.6 模块使用的 AoS（array-of-structs）网格类型
// ============================================================================
//
// 本工程的核心 `Mesh`（app/src/main/cpp/mesh/mesh_engine.h）是 **SoA**：
//
//     std::vector<float> positions;   // 3 float / 顶点
//     std::vector<float> normals;
//     std::vector<float> colors;
//     std::vector<uint32_t> indices;
//
// 而 V0.6 的四个模块（mesh_postprocess / uv_unwrap / texture_baker /
// textured_glb_exporter）是按 V0.5 那一版的 **AoS** 布局写的：
//
//     std::vector<MeshVertex> vertices;   // 一个顶点一个结构体
//
// 两者同名 `Mesh` 会在这个翻译单元里直接冲突。所以这里另立
// `AosMesh` / `AosVertex`，字段布局与补丁版**逐字节一致**（uv_unwrap 会把
// `sizeof(AosVertex)` 直接当作 xatlas 的顶点 stride），只在
// native_engine.cpp 的边界处与 SoA 做一次显式转换。
//
// 刻意**不改**本工程已验收的 SoA Mesh：它被 MeshEngine、gltf_exporter、
// nativeGetMeshVertices（交错 9 float 的 JNI 契约）和 Kotlin 侧
// MeshRenderer 同时依赖，改布局的风险远大于收益。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

struct AosVertex {
    float px = 0.0f;
    float py = 0.0f;
    float pz = 0.0f;

    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 1.0f;

    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

struct AosMesh {
    std::vector<AosVertex> vertices;
    std::vector<std::uint32_t> indices;

    void clear() {
        vertices.clear();
        indices.clear();
    }

    std::size_t triangleCount() const {
        return indices.size() / 3u;
    }

    bool empty() const {
        return vertices.empty() || indices.empty();
    }
};
