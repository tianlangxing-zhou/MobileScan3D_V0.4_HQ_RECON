#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "v06/uv_unwrap.h"

/**
 * CPU-side copy of the V0.6 baked AR asset.
 *
 * Layout:
 *   vertex = x y z, nx ny nz, u v    (8 floats)
 *   index  = uint32
 *   atlas  = complete JPEG byte stream
 *
 * The GLB remains the external interchange asset. This compact binary exists
 * only so the Android renderer can restore the exact mesh+atlas without
 * implementing a full glTF parser on the camera render path.
 */
class ArTexturedAsset {
public:
    static constexpr int kVertexFloats = 8;

    void clear();

    bool set(
        const UvMesh& mesh,
        const std::vector<std::uint8_t>& jpegBytes);

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    bool ready() const;

    std::size_t vertexCount() const;
    std::size_t indexCount() const;
    std::size_t triangleCount() const;

    const std::vector<float>& vertices() const { return vertices_; }
    const std::vector<std::uint32_t>& indices() const { return indices_; }
    const std::vector<std::uint8_t>& jpeg() const { return jpeg_; }

private:
    std::vector<float> vertices_;
    std::vector<std::uint32_t> indices_;
    std::vector<std::uint8_t> jpeg_;
};
