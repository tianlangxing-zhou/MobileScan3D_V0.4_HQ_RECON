#include "gltf_exporter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace {

constexpr uint32_t kGlbMagic = 0x46546C67u;     // "glTF"
constexpr uint32_t kGlbVersion = 2u;
constexpr uint32_t kChunkJson = 0x4E4F534Au;    // "JSON"
constexpr uint32_t kChunkBin = 0x004E4942u;     // "BIN\0"

constexpr uint32_t kComponentFloat = 5126u;
constexpr uint32_t kComponentUByte = 5121u;
constexpr uint32_t kComponentUInt = 5125u;

constexpr uint32_t kTargetArrayBuffer = 34962u;
constexpr uint32_t kTargetElementArrayBuffer = 34963u;

/** 按 4 字节向上取整（GLB 的 chunk 与 bufferView 都要对齐）。 */
inline size_t align4(size_t n) { return (n + 3u) & ~static_cast<size_t>(3u); }

inline void putU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

inline void putF32(std::vector<uint8_t>& out, float v) {
    uint32_t bits = 0;
    static_assert(sizeof(float) == sizeof(uint32_t), "float 必须是 32 位");
    std::memcpy(&bits, &v, sizeof(bits));
    putU32(out, bits);
}

inline void appendFloats(std::vector<uint8_t>& out, const float* src, size_t count) {
    out.reserve(out.size() + count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        putF32(out, src[i]);
    }
}

/** JSON 数值：不用 std::to_string（它会输出 6 位小数，min/max 会失真）。 */
std::string jnum(double v) {
    if (!std::isfinite(v)) {
        return "0";
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.7g", v);
    return std::string(buf);
}

/** 极简 JSON 字符串转义（只处理必要字符，避免非法 JSON）。 */
std::string jstr(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += ' ';
                } else {
                    out += c;
                }
        }
    }
    return out;
}

}  // namespace

bool exportGlb(const Mesh& mesh, const std::string& path,
               const std::string& name, GlbExportStats* stats) {
    GlbExportStats st;
    st.vertices = mesh.vertexCount();
    st.triangles = mesh.triangleCount();

    if (st.vertices == 0 || st.triangles == 0) {
        st.note = "mesh empty";
        if (stats) {
            *stats = st;
        }
        return false;
    }
    if (mesh.positions.size() < st.vertices * 3) {
        st.note = "positions truncated";
        if (stats) {
            *stats = st;
        }
        return false;
    }

    const bool hasNormal = mesh.normals.size() >= st.vertices * 3;
    const bool hasColor = mesh.colors.size() >= st.vertices * 3;
    st.hasVertexColor = hasColor;

    // ---------------------------------------------------------------- BIN
    std::vector<uint8_t> bin;
    bin.reserve(st.vertices * 12 * 2 + st.vertices * 4 + st.triangles * 12);

    const size_t offPos = 0;
    appendFloats(bin, mesh.positions.data(), st.vertices * 3);

    size_t offNrm = 0;
    if (hasNormal) {
        offNrm = bin.size();
        appendFloats(bin, mesh.normals.data(), st.vertices * 3);
    }

    size_t offCol = 0;
    if (hasColor) {
        // 对齐到 4 字节（ubyte 本身只需 1，但保持整齐便于审查）
        while (bin.size() % 4 != 0) {
            bin.push_back(0);
        }
        offCol = bin.size();
        for (size_t i = 0; i < st.vertices; ++i) {
            const float r = std::clamp(mesh.colors[i * 3 + 0], 0.f, 1.f);
            const float g = std::clamp(mesh.colors[i * 3 + 1], 0.f, 1.f);
            const float b = std::clamp(mesh.colors[i * 3 + 2], 0.f, 1.f);
            bin.push_back(static_cast<uint8_t>(std::lround(r * 255.f)));
            bin.push_back(static_cast<uint8_t>(std::lround(g * 255.f)));
            bin.push_back(static_cast<uint8_t>(std::lround(b * 255.f)));
            bin.push_back(255);
        }
    }

    while (bin.size() % 4 != 0) {
        bin.push_back(0);
    }
    const size_t offIdx = bin.size();
    bin.reserve(bin.size() + st.triangles * 3 * sizeof(uint32_t));
    for (size_t i = 0; i < st.triangles * 3; ++i) {
        const uint32_t idx = mesh.indices[i];
        putU32(bin, (idx < st.vertices) ? idx : 0u);
    }

    // POSITION 的 min/max 是 glTF 规范强制要求的
    float mn[3] = {std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
    float mx[3] = {-std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max(),
                   -std::numeric_limits<float>::max()};
    for (size_t i = 0; i < st.vertices; ++i) {
        for (int k = 0; k < 3; ++k) {
            const float v = mesh.positions[i * 3 + k];
            if (v < mn[k]) mn[k] = v;
            if (v > mx[k]) mx[k] = v;
        }
    }

    // ---------------------------------------------------------------- JSON
    std::ostringstream j;
    j << "{";
    j << "\"asset\":{\"version\":\"2.0\",\"generator\":\"MobileScan3D\"},";
    j << "\"scene\":0,";
    j << "\"scenes\":[{\"nodes\":[0]}],";
    j << "\"nodes\":[{\"mesh\":0,\"name\":\"" << jstr(name) << "\"}],";
    j << "\"meshes\":[{\"name\":\"" << jstr(name) << "\",\"primitives\":[{";
    j << "\"attributes\":{\"POSITION\":0";
    if (hasNormal) {
        j << ",\"NORMAL\":1";
    }
    if (hasColor) {
        j << ",\"COLOR_0\":2";
    }
    j << "}";
    j << ",\"indices\":" << (hasNormal ? (hasColor ? 3 : 2) : (hasColor ? 2 : 1));
    j << ",\"material\":0,\"mode\":4}]}],";

    // material：baseColorFactor 必须是白色 —— 顶点色是「乘」上去的，
    // 这里一旦写成灰色，导出的模型会整体变暗。
    j << "\"materials\":[{\"name\":\"scan\",\"pbrMetallicRoughness\":{"
         "\"baseColorFactor\":[1,1,1,1],\"metallicFactor\":0,"
         "\"roughnessFactor\":1}}],";

    // accessors
    j << "\"accessors\":[";
    j << "{\"bufferView\":0,\"componentType\":" << kComponentFloat
      << ",\"count\":" << st.vertices << ",\"type\":\"VEC3\",\"min\":["
      << jnum(mn[0]) << "," << jnum(mn[1]) << "," << jnum(mn[2]) << "],\"max\":["
      << jnum(mx[0]) << "," << jnum(mx[1]) << "," << jnum(mx[2]) << "]}";

    int nextView = 1;
    int nrmView = -1;
    int colView = -1;
    if (hasNormal) {
        nrmView = nextView++;
        j << ",{\"bufferView\":" << nrmView << ",\"componentType\":" << kComponentFloat
          << ",\"count\":" << st.vertices << ",\"type\":\"VEC3\"}";
    }
    if (hasColor) {
        colView = nextView++;
        j << ",{\"bufferView\":" << colView << ",\"componentType\":" << kComponentUByte
          << ",\"count\":" << st.vertices << ",\"type\":\"VEC4\",\"normalized\":true}";
    }
    const int idxView = nextView++;
    j << ",{\"bufferView\":" << idxView << ",\"componentType\":" << kComponentUInt
      << ",\"count\":" << (st.triangles * 3) << ",\"type\":\"SCALAR\"}";
    j << "],";

    // bufferViews
    j << "\"bufferViews\":[";
    j << "{\"buffer\":0,\"byteOffset\":" << offPos << ",\"byteLength\":"
      << (st.vertices * 12) << ",\"target\":" << kTargetArrayBuffer << "}";
    if (hasNormal) {
        j << ",{\"buffer\":0,\"byteOffset\":" << offNrm << ",\"byteLength\":"
          << (st.vertices * 12) << ",\"target\":" << kTargetArrayBuffer << "}";
    }
    if (hasColor) {
        j << ",{\"buffer\":0,\"byteOffset\":" << offCol << ",\"byteLength\":"
          << (st.vertices * 4) << ",\"target\":" << kTargetArrayBuffer << "}";
    }
    j << ",{\"buffer\":0,\"byteOffset\":" << offIdx << ",\"byteLength\":"
      << (st.triangles * 12) << ",\"target\":" << kTargetElementArrayBuffer << "}";
    j << "],";

    j << "\"buffers\":[{\"byteLength\":" << bin.size() << "}]";
    j << "}";

    const std::string jsonStr = j.str();

    // ---------------------------------------------------------------- GLB
    const size_t jsonPadded = align4(jsonStr.size());
    const size_t binPadded = align4(bin.size());
    const size_t total = 12 + 8 + jsonPadded + 8 + binPadded;
    st.jsonBytes = jsonPadded;
    st.binBytes = binPadded;
    st.fileBytes = total;

    std::vector<uint8_t> out;
    out.reserve(total);
    putU32(out, kGlbMagic);
    putU32(out, kGlbVersion);
    putU32(out, static_cast<uint32_t>(total));

    putU32(out, static_cast<uint32_t>(jsonPadded));
    putU32(out, kChunkJson);
    out.insert(out.end(), jsonStr.begin(), jsonStr.end());
    for (size_t i = jsonStr.size(); i < jsonPadded; ++i) {
        out.push_back(0x20);  // JSON chunk 用空格补齐
    }

    putU32(out, static_cast<uint32_t>(binPadded));
    putU32(out, kChunkBin);
    out.insert(out.end(), bin.begin(), bin.end());
    for (size_t i = bin.size(); i < binPadded; ++i) {
        out.push_back(0x00);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        st.note = "open failed";
        if (stats) {
            *stats = st;
        }
        return false;
    }
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
    if (!f.good()) {
        st.note = "write failed";
        if (stats) {
            *stats = st;
        }
        return false;
    }

    st.ok = true;
    st.note = "ok";
    if (stats) {
        *stats = st;
    }
    return true;
}
