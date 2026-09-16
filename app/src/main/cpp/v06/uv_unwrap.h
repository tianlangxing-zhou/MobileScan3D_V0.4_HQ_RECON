#pragma once

#include <cstdint>
#include <vector>

#include "aos_mesh.h"

struct UvVertex {
    AosVertex base;
    float u = 0.0f;
    float v = 0.0f;
};

struct UvMesh {
    std::vector<UvVertex> vertices;
    std::vector<std::uint32_t> indices;
    int atlasWidth = 0;
    int atlasHeight = 0;

    void clear() {
        vertices.clear();
        indices.clear();
        atlasWidth = 0;
        atlasHeight = 0;
    }

    bool empty() const {
        return vertices.empty() || indices.empty();
    }

    std::size_t triangleCount() const {
        return indices.size() / 3u;
    }
};

struct UvUnwrapStats {
    bool usedXatlas = false;
    int atlasWidth = 0;
    int atlasHeight = 0;
    std::size_t inputVertices = 0;
    std::size_t outputVertices = 0;
    std::size_t chartCount = 0;
};

class UvUnwrapper {
public:
    static bool unwrap(
        const AosMesh& mesh,
        int resolution,
        int padding,
        UvMesh& out,
        UvUnwrapStats* stats = nullptr);

private:
    static bool fallbackTriangleAtlas(
        const AosMesh& mesh,
        int resolution,
        int padding,
        UvMesh& out,
        UvUnwrapStats* stats);
};
