#include "ar_textured_asset.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{
    'M','S','A','R','0','7','0','1'
};
constexpr std::uint32_t kVersion = 1u;
constexpr std::uint32_t kMaxVertices = 800000u;
constexpr std::uint32_t kMaxIndices = 2400000u;
constexpr std::uint32_t kMaxJpegBytes = 64u * 1024u * 1024u;

template <typename T>
bool writePod(std::ofstream& out, const T& value) {
    out.write(
        reinterpret_cast<const char*>(&value),
        static_cast<std::streamsize>(sizeof(T)));
    return static_cast<bool>(out);
}

template <typename T>
bool readPod(std::ifstream& in, T* value) {
    if (!value) return false;
    in.read(
        reinterpret_cast<char*>(value),
        static_cast<std::streamsize>(sizeof(T)));
    return static_cast<bool>(in);
}

}  // namespace

void ArTexturedAsset::clear() {
    vertices_.clear();
    indices_.clear();
    jpeg_.clear();
}

bool ArTexturedAsset::set(
        const UvMesh& mesh,
        const std::vector<std::uint8_t>& jpegBytes) {
    if (mesh.empty() ||
        jpegBytes.size() < 4 ||
        jpegBytes[0] != 0xFFu ||
        jpegBytes[1] != 0xD8u) {
        return false;
    }

    if (mesh.vertices.size() > kMaxVertices ||
        mesh.indices.size() > kMaxIndices ||
        jpegBytes.size() > kMaxJpegBytes) {
        return false;
    }

    std::vector<float> vertices;
    vertices.reserve(
        mesh.vertices.size() *
        static_cast<std::size_t>(kVertexFloats));

    for (const auto& v : mesh.vertices) {
        vertices.push_back(v.base.px);
        vertices.push_back(v.base.py);
        vertices.push_back(v.base.pz);

        vertices.push_back(v.base.nx);
        vertices.push_back(v.base.ny);
        vertices.push_back(v.base.nz);

        vertices.push_back(
            std::clamp(v.u, 0.0f, 1.0f));
        vertices.push_back(
            std::clamp(v.v, 0.0f, 1.0f));
    }

    vertices_.swap(vertices);
    indices_ = mesh.indices;
    jpeg_ = jpegBytes;
    return true;
}

bool ArTexturedAsset::save(
        const std::string& path) const {
    if (!ready()) return false;

    std::ofstream out(
        path,
        std::ios::binary | std::ios::trunc);
    if (!out) return false;

    out.write(kMagic.data(), kMagic.size());

    const std::uint32_t vertexCount =
        static_cast<std::uint32_t>(
            vertices_.size() /
            static_cast<std::size_t>(kVertexFloats));
    const std::uint32_t indexCount =
        static_cast<std::uint32_t>(indices_.size());
    const std::uint32_t jpegBytes =
        static_cast<std::uint32_t>(jpeg_.size());
    const std::uint32_t vertexFloats =
        static_cast<std::uint32_t>(kVertexFloats);

    if (!writePod(out, kVersion) ||
        !writePod(out, vertexFloats) ||
        !writePod(out, vertexCount) ||
        !writePod(out, indexCount) ||
        !writePod(out, jpegBytes)) {
        return false;
    }

    out.write(
        reinterpret_cast<const char*>(vertices_.data()),
        static_cast<std::streamsize>(
            vertices_.size() * sizeof(float)));

    out.write(
        reinterpret_cast<const char*>(indices_.data()),
        static_cast<std::streamsize>(
            indices_.size() * sizeof(std::uint32_t)));

    out.write(
        reinterpret_cast<const char*>(jpeg_.data()),
        static_cast<std::streamsize>(jpeg_.size()));

    return static_cast<bool>(out);
}

bool ArTexturedAsset::load(
        const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::array<char, 8> magic{};
    in.read(magic.data(), magic.size());
    if (!in || magic != kMagic) return false;

    std::uint32_t version = 0;
    std::uint32_t vertexFloats = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t jpegBytes = 0;

    if (!readPod(in, &version) ||
        !readPod(in, &vertexFloats) ||
        !readPod(in, &vertexCount) ||
        !readPod(in, &indexCount) ||
        !readPod(in, &jpegBytes)) {
        return false;
    }

    if (version != kVersion ||
        vertexFloats !=
            static_cast<std::uint32_t>(kVertexFloats) ||
        vertexCount == 0 ||
        indexCount < 3 ||
        (indexCount % 3u) != 0u ||
        vertexCount > kMaxVertices ||
        indexCount > kMaxIndices ||
        jpegBytes < 4 ||
        jpegBytes > kMaxJpegBytes) {
        return false;
    }

    const std::size_t floatCount =
        static_cast<std::size_t>(vertexCount) *
        static_cast<std::size_t>(kVertexFloats);

    std::vector<float> vertices(floatCount);
    std::vector<std::uint32_t> indices(indexCount);
    std::vector<std::uint8_t> jpeg(jpegBytes);

    in.read(
        reinterpret_cast<char*>(vertices.data()),
        static_cast<std::streamsize>(
            vertices.size() * sizeof(float)));

    in.read(
        reinterpret_cast<char*>(indices.data()),
        static_cast<std::streamsize>(
            indices.size() * sizeof(std::uint32_t)));

    in.read(
        reinterpret_cast<char*>(jpeg.data()),
        static_cast<std::streamsize>(jpeg.size()));

    if (!in ||
        jpeg[0] != 0xFFu ||
        jpeg[1] != 0xD8u) {
        return false;
    }

    for (const auto idx : indices) {
        if (idx >= vertexCount) {
            return false;
        }
    }

    vertices_.swap(vertices);
    indices_.swap(indices);
    jpeg_.swap(jpeg);
    return true;
}

bool ArTexturedAsset::ready() const {
    return
        vertices_.size() >=
            static_cast<std::size_t>(
                kVertexFloats * 3) &&
        (vertices_.size() %
            static_cast<std::size_t>(kVertexFloats)) == 0u &&
        indices_.size() >= 3u &&
        (indices_.size() % 3u) == 0u &&
        jpeg_.size() >= 4u;
}

std::size_t ArTexturedAsset::vertexCount() const {
    return
        vertices_.size() /
        static_cast<std::size_t>(kVertexFloats);
}

std::size_t ArTexturedAsset::indexCount() const {
    return indices_.size();
}

std::size_t ArTexturedAsset::triangleCount() const {
    return indices_.size() / 3u;
}
