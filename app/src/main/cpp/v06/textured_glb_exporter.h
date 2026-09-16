#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "uv_unwrap.h"

class TexturedGlbExporter {
public:
    // jpegBytes must contain a complete JPEG file. It is embedded into GLB,
    // so the exported asset is a single self-contained file.
    static bool write(
        const std::string& path,
        const UvMesh& mesh,
        const std::vector<std::uint8_t>& jpegBytes);
};
