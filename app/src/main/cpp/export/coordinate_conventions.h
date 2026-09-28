#pragma once

namespace scan_coordinates {
// VINS world is right-handed +Z up; glTF is right-handed +Y up.
// Column-major Rx(-90 degrees): (x,y,z) -> (x,z,-y), determinant +1.
// Apply ONLY to the external GLB root node. Keeping mesh-local positions,
// normals and the AR cache in VINS world preserves saved-map relocalization.
inline constexpr const char* kVinsToGltfMatrixJson =
    "[1,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,1]";
}
