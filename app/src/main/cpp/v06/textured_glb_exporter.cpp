#include "textured_glb_exporter.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

void append(
        std::vector<std::uint8_t>& out,
        const void* data,
        std::size_t n) {
    const auto* p =
        static_cast<const std::uint8_t*>(data);
    out.insert(out.end(), p, p + n);
}

void appendFloat(
        std::vector<std::uint8_t>& out,
        float v) {
    append(out, &v, sizeof(v));
}

void appendU32(
        std::vector<std::uint8_t>& out,
        std::uint32_t v) {
    append(out, &v, sizeof(v));
}

void pad4(
        std::vector<std::uint8_t>& out,
        std::uint8_t value = 0) {
    while ((out.size() & 3u) != 0u) {
        out.push_back(value);
    }
}

void padJson4(std::string& json) {
    while ((json.size() & 3u) != 0u) {
        json.push_back(' ');
    }
}

}  // namespace

bool TexturedGlbExporter::write(
        const std::string& path,
        const UvMesh& mesh,
        const std::vector<std::uint8_t>& jpegBytes) {
    if (mesh.empty() || jpegBytes.size() < 4) {
        return false;
    }

    // Basic JPEG SOI check. We intentionally do not parse the whole image.
    if (!(jpegBytes[0] == 0xFFu &&
          jpegBytes[1] == 0xD8u)) {
        return false;
    }

    // Interleaved vertex:
    // POSITION xyz, NORMAL xyz, TEXCOORD_0 uv = 8 floats / 32 bytes.
    std::vector<std::uint8_t> bin;
    bin.reserve(
        mesh.vertices.size() * 8u * sizeof(float) +
        mesh.indices.size() * sizeof(std::uint32_t) +
        jpegBytes.size() + 16u);

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    float maxZ = -std::numeric_limits<float>::max();

    float minU = std::numeric_limits<float>::max();
    float minV = std::numeric_limits<float>::max();
    float maxU = -std::numeric_limits<float>::max();
    float maxV = -std::numeric_limits<float>::max();

    for (const auto& v : mesh.vertices) {
        const auto& b = v.base;

        minX = std::min(minX, b.px);
        minY = std::min(minY, b.py);
        minZ = std::min(minZ, b.pz);
        maxX = std::max(maxX, b.px);
        maxY = std::max(maxY, b.py);
        maxZ = std::max(maxZ, b.pz);

        const float u = std::clamp(v.u, 0.0f, 1.0f);
        const float vv = std::clamp(v.v, 0.0f, 1.0f);
        minU = std::min(minU, u);
        minV = std::min(minV, vv);
        maxU = std::max(maxU, u);
        maxV = std::max(maxV, vv);

        appendFloat(bin, b.px);
        appendFloat(bin, b.py);
        appendFloat(bin, b.pz);

        appendFloat(bin, b.nx);
        appendFloat(bin, b.ny);
        appendFloat(bin, b.nz);

        appendFloat(bin, u);
        appendFloat(bin, vv);
    }

    const std::uint32_t vertexOffset = 0;
    const std::uint32_t vertexBytes =
        static_cast<std::uint32_t>(bin.size());
    pad4(bin);

    const std::uint32_t indexOffset =
        static_cast<std::uint32_t>(bin.size());
    for (const auto idx : mesh.indices) {
        appendU32(bin, idx);
    }
    const std::uint32_t indexBytes =
        static_cast<std::uint32_t>(
            mesh.indices.size() * sizeof(std::uint32_t));
    pad4(bin);

    const std::uint32_t imageOffset =
        static_cast<std::uint32_t>(bin.size());
    append(bin, jpegBytes.data(), jpegBytes.size());
    const std::uint32_t imageBytes =
        static_cast<std::uint32_t>(jpegBytes.size());
    pad4(bin);

    std::ostringstream js;
    js.setf(std::ios::fixed);
    js << std::setprecision(7);

    js
        << "{"
        << "\"asset\":{\"version\":\"2.0\","
        << "\"generator\":\"MobileScan3D V0.6\"},"
        << "\"scene\":0,"
        << "\"scenes\":[{\"nodes\":[0]}],"
        << "\"nodes\":[{\"mesh\":0,\"name\":\"MobileScan3D_Textured\"}],"

        << "\"meshes\":[{"
        << "\"name\":\"ScanMesh\","
        << "\"primitives\":[{"
        << "\"attributes\":{"
        << "\"POSITION\":0,"
        << "\"NORMAL\":1,"
        << "\"TEXCOORD_0\":2"
        << "},"
        << "\"indices\":3,"
        << "\"material\":0,"
        << "\"mode\":4"
        << "}]"
        << "}],"

        << "\"materials\":[{"
        << "\"name\":\"HQ_MultiView\","
        << "\"doubleSided\":true,"
        << "\"pbrMetallicRoughness\":{"
        << "\"baseColorTexture\":{\"index\":0,\"texCoord\":0},"
        << "\"baseColorFactor\":[1,1,1,1],"
        << "\"metallicFactor\":0.0,"
        << "\"roughnessFactor\":1.0"
        << "}"
        << "}],"

        << "\"samplers\":[{"
        << "\"magFilter\":9729,"
        << "\"minFilter\":9729,"
        << "\"wrapS\":33071,"
        << "\"wrapT\":33071"
        << "}],"

        << "\"textures\":[{"
        << "\"sampler\":0,"
        << "\"source\":0"
        << "}],"

        << "\"images\":[{"
        << "\"bufferView\":2,"
        << "\"mimeType\":\"image/jpeg\","
        << "\"name\":\"MobileScan3D_Atlas\""
        << "}],"

        << "\"buffers\":[{"
        << "\"byteLength\":" << bin.size()
        << "}],"

        << "\"bufferViews\":["
        << "{"
        << "\"buffer\":0,"
        << "\"byteOffset\":" << vertexOffset << ","
        << "\"byteLength\":" << vertexBytes << ","
        << "\"byteStride\":32,"
        << "\"target\":34962"
        << "},"
        << "{"
        << "\"buffer\":0,"
        << "\"byteOffset\":" << indexOffset << ","
        << "\"byteLength\":" << indexBytes << ","
        << "\"target\":34963"
        << "},"
        << "{"
        << "\"buffer\":0,"
        << "\"byteOffset\":" << imageOffset << ","
        << "\"byteLength\":" << imageBytes
        << "}"
        << "],"

        << "\"accessors\":["
        << "{"
        << "\"bufferView\":0,"
        << "\"byteOffset\":0,"
        << "\"componentType\":5126,"
        << "\"count\":" << mesh.vertices.size() << ","
        << "\"type\":\"VEC3\","
        << "\"min\":[" << minX << "," << minY << "," << minZ << "],"
        << "\"max\":[" << maxX << "," << maxY << "," << maxZ << "]"
        << "},"
        << "{"
        << "\"bufferView\":0,"
        << "\"byteOffset\":12,"
        << "\"componentType\":5126,"
        << "\"count\":" << mesh.vertices.size() << ","
        << "\"type\":\"VEC3\""
        << "},"
        << "{"
        << "\"bufferView\":0,"
        << "\"byteOffset\":24,"
        << "\"componentType\":5126,"
        << "\"count\":" << mesh.vertices.size() << ","
        << "\"type\":\"VEC2\","
        << "\"min\":[" << minU << "," << minV << "],"
        << "\"max\":[" << maxU << "," << maxV << "]"
        << "},"
        << "{"
        << "\"bufferView\":1,"
        << "\"byteOffset\":0,"
        << "\"componentType\":5125,"
        << "\"count\":" << mesh.indices.size() << ","
        << "\"type\":\"SCALAR\""
        << "}"
        << "]"
        << "}";

    std::string json = js.str();
    padJson4(json);

    const std::uint32_t jsonLength =
        static_cast<std::uint32_t>(json.size());
    const std::uint32_t binLength =
        static_cast<std::uint32_t>(bin.size());
    const std::uint32_t totalLength =
        12u + 8u + jsonLength + 8u + binLength;

    std::ofstream out(
        path,
        std::ios::binary | std::ios::trunc);
    if (!out) return false;

    const std::uint32_t magic = 0x46546C67u;
    const std::uint32_t version = 2u;
    const std::uint32_t jsonType = 0x4E4F534Au;
    const std::uint32_t binType = 0x004E4942u;

    out.write(
        reinterpret_cast<const char*>(&magic),
        sizeof(magic));
    out.write(
        reinterpret_cast<const char*>(&version),
        sizeof(version));
    out.write(
        reinterpret_cast<const char*>(&totalLength),
        sizeof(totalLength));

    out.write(
        reinterpret_cast<const char*>(&jsonLength),
        sizeof(jsonLength));
    out.write(
        reinterpret_cast<const char*>(&jsonType),
        sizeof(jsonType));
    out.write(json.data(), json.size());

    out.write(
        reinterpret_cast<const char*>(&binLength),
        sizeof(binLength));
    out.write(
        reinterpret_cast<const char*>(&binType),
        sizeof(binType));
    out.write(
        reinterpret_cast<const char*>(bin.data()),
        static_cast<std::streamsize>(bin.size()));

    return static_cast<bool>(out);
}
