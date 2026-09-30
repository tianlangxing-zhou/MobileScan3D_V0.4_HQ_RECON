#pragma once
#include "mesh_engine.h"
#include <string>

// Original implementation of sampled RANSAC plane detection and orthogonal box fitting.
// No CGAL/PolyFit code or runtime dependency. Input/output use the scan world coordinates.
enum class HardSurfaceMode : int { Scan=0, Planar=1, Cuboid=2, Cube=3 };
struct HardSurfaceStats {
    bool applied=false;
    int planes=0, observedFaces=0, inferredFaces=0;
    float rms=0, support=0;
    std::string message;
};
// Failure leaves output unchanged. Never modifies input or the TSDF map.
bool fitHardSurface(const Mesh& input, HardSurfaceMode mode, float voxelSize,
                    Mesh& output, HardSurfaceStats& stats);
