#pragma once

#include <cstddef>

#include "aos_mesh.h"

struct MeshPostProcessOptions {
    float weldEpsilon = 0.00075f;

    // Components smaller than BOTH thresholds are removed.
    int minComponentTriangles = 48;
    float minComponentAreaRatio = 0.0035f;

    // Only small boundary loops are capped. Large missing sides are intentionally
    // left open because a centroid fan across a large non-planar gap is worse
    // than a truthful hole.
    int maxHoleEdges = 56;
    float maxHoleDiameterMeters = 0.075f;
    float maxHoleDiameterBBoxRatio = 0.12f;

    // QEM target. <= 0 means "do not simplify".
    int targetTriangles = 80000;
    int qemMaxPasses = 10;
    float qemMaxNormalFlipDeg = 72.0f;
    bool preserveBoundary = true;

    // vc183.3: conservative support/table-plane removal.
    //
    // VINS world +Z is physical UP. We only remove nearly-horizontal triangles
    // near the lowest part of the model AND outside the XY footprint estimated
    // from geometry above the support plane. This intentionally prefers leaving
    // some table behind over deleting a real flat bottom from the object.
    bool removeSupportPlane = false;
    float supportPlaneMaxTiltDeg = 18.0f;
    float supportPlaneBottomRatio = 0.30f;
    float supportPlaneMinAreaRatio = 0.055f;
    float supportPlaneCorePaddingRatio = 0.055f;
    float supportPlaneBandMeters = 0.006f;
    float supportPlaneMaxRemovalRatio = 0.45f;

    float minTriangleArea = 1e-10f;
};

struct MeshPostProcessStats {
    std::size_t inputVertices = 0;
    std::size_t inputTriangles = 0;

    std::size_t weldedVertices = 0;
    std::size_t removedComponents = 0;
    std::size_t removedComponentTriangles = 0;

    std::size_t boundaryLoops = 0;
    std::size_t filledHoles = 0;
    std::size_t addedHoleTriangles = 0;

    std::size_t qemCollapsedEdges = 0;
    std::size_t qemPasses = 0;

    bool supportPlaneDetected = false;
    std::size_t supportPlaneTrianglesRemoved = 0;

    std::size_t outputVertices = 0;
    std::size_t outputTriangles = 0;
};

class MeshPostProcessor {
public:
    static bool run(
        AosMesh& mesh,
        const MeshPostProcessOptions& options,
        MeshPostProcessStats* stats = nullptr);

    static bool weld(
        AosMesh& mesh,
        float epsilon,
        float minTriangleArea = 1e-10f);

    static bool removeSmallComponents(
        AosMesh& mesh,
        int minTriangles,
        float minAreaRatio,
        std::size_t* removedComponents = nullptr,
        std::size_t* removedTriangles = nullptr);

    static bool fillSmallHoles(
        AosMesh& mesh,
        int maxHoleEdges,
        float maxDiameterMeters,
        float maxDiameterBBoxRatio,
        std::size_t* boundaryLoops = nullptr,
        std::size_t* filledHoles = nullptr,
        std::size_t* addedTriangles = nullptr);

    static bool removeSupportPlane(
        AosMesh& mesh,
        float maxTiltDeg,
        float bottomRatio,
        float minAreaRatio,
        float corePaddingRatio,
        float bandMeters,
        float maxRemovalRatio,
        bool* detected = nullptr,
        std::size_t* removedTriangles = nullptr);

    static bool simplifyQem(
        AosMesh& mesh,
        int targetTriangles,
        int maxPasses,
        float maxNormalFlipDeg,
        bool preserveBoundary,
        float minTriangleArea,
        std::size_t* collapsedEdges = nullptr,
        std::size_t* passes = nullptr);
};
