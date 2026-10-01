#pragma once

#include <array>
#include <cstddef>

#include "mesh_engine.h"

struct MeshRegistrationStats {
    bool ok = false;
    float rmseMeters = 0.0f;
    float overlap = 0.0f;
    int inliers = 0;
    int iterations = 0;
    int yawHypothesisDeg = 0;
    std::size_t referenceVertices = 0;
    std::size_t movingVertices = 0;
    std::size_t outputVertices = 0;
    std::size_t outputTriangles = 0;
};

/**
 * Align moving mesh into reference-mesh coordinates and build a cumulative mesh.
 *
 * Assumptions intentionally match MobileScan3D's scan sessions:
 * - both sessions are metric (same depth calibration family)
 * - world +Z is gravity-aligned by VINS
 * - independent sessions may disagree strongly in yaw and translation
 *
 * The solver searches several yaw hypotheses around gravity, then runs robust
 * point-to-point ICP. It refuses low-overlap solutions rather than returning an
 * attractive-looking but wrong merge.
 */
bool alignAndMergeMeshes(
    const Mesh& reference,
    const Mesh& moving,
    Mesh* output,
    std::array<float, 16>* outReferenceFromMoving,
    MeshRegistrationStats* stats = nullptr,
    int forcedYawDeg = -1);
