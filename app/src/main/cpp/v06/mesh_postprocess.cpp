#include "mesh_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 operator*(const Vec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}
Vec3 operator/(const Vec3& a, double s) {
    return s == 0.0 ? Vec3{} : Vec3{a.x / s, a.y / s, a.z / s};
}
double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
double norm2(const Vec3& v) { return dot(v, v); }
double norm(const Vec3& v) { return std::sqrt(norm2(v)); }
Vec3 normalize(const Vec3& v) {
    const double n = norm(v);
    if (n <= 1e-20) return {0.0, 0.0, 1.0};
    return v / n;
}
Vec3 pos(const AosVertex& v) {
    return {v.px, v.py, v.pz};
}
Vec3 color(const AosVertex& v) {
    return {v.r, v.g, v.b};
}

double triangleArea(const AosVertex& a, const AosVertex& b, const AosVertex& c) {
    return 0.5 * norm(cross(pos(b) - pos(a), pos(c) - pos(a)));
}

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

double cross2(const Vec2& a, const Vec2& b, const Vec2& c) {
    return (b.x - a.x) * (c.y - a.y) -
           (b.y - a.y) * (c.x - a.x);
}

double polygonArea2(const std::vector<Vec2>& p) {
    double a = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Vec2& x = p[i];
        const Vec2& y = p[(i + 1) % p.size()];
        a += x.x * y.y - y.x * x.y;
    }
    return 0.5 * a;
}

bool pointInTriangle2(
        const Vec2& p,
        const Vec2& a,
        const Vec2& b,
        const Vec2& c,
        double orientation) {
    const double c0 = cross2(a, b, p) * orientation;
    const double c1 = cross2(b, c, p) * orientation;
    const double c2 = cross2(c, a, p) * orientation;
    constexpr double eps = -1e-12;
    return c0 >= eps && c1 >= eps && c2 >= eps;
}

struct EdgeKey {
    std::uint32_t a = 0;
    std::uint32_t b = 0;

    EdgeKey() = default;
    EdgeKey(std::uint32_t x, std::uint32_t y) {
        if (x < y) { a = x; b = y; }
        else { a = y; b = x; }
    }

    bool operator==(const EdgeKey& o) const {
        return a == o.a && b == o.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& e) const noexcept {
        std::uint64_t x =
            (static_cast<std::uint64_t>(e.a) << 32u) |
            static_cast<std::uint64_t>(e.b);
        x ^= x >> 33u;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33u;
        return static_cast<std::size_t>(x);
    }
};

void compactMesh(AosMesh& mesh, float minArea) {
    std::vector<std::uint32_t> cleaned;
    cleaned.reserve(mesh.indices.size());

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const auto a = mesh.indices[i + 0];
        const auto b = mesh.indices[i + 1];
        const auto c = mesh.indices[i + 2];
        if (a >= mesh.vertices.size() ||
            b >= mesh.vertices.size() ||
            c >= mesh.vertices.size() ||
            a == b || b == c || c == a) {
            continue;
        }
        if (triangleArea(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]) < minArea) {
            continue;
        }
        cleaned.push_back(a);
        cleaned.push_back(b);
        cleaned.push_back(c);
    }

    std::vector<std::uint8_t> used(mesh.vertices.size(), 0);
    for (auto i : cleaned) used[i] = 1;

    std::vector<std::uint32_t> remap(
        mesh.vertices.size(),
        std::numeric_limits<std::uint32_t>::max());

    std::vector<AosVertex> vertices;
    vertices.reserve(mesh.vertices.size());

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        if (!used[i]) continue;
        remap[i] = static_cast<std::uint32_t>(vertices.size());
        vertices.push_back(mesh.vertices[i]);
    }

    for (auto& i : cleaned) {
        i = remap[i];
    }

    mesh.vertices.swap(vertices);
    mesh.indices.swap(cleaned);
}

void recomputeNormals(AosMesh& mesh) {
    for (auto& v : mesh.vertices) {
        v.nx = v.ny = v.nz = 0.0f;
    }

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        AosVertex& a = mesh.vertices[mesh.indices[i + 0]];
        AosVertex& b = mesh.vertices[mesh.indices[i + 1]];
        AosVertex& c = mesh.vertices[mesh.indices[i + 2]];

        const Vec3 n = cross(pos(b) - pos(a), pos(c) - pos(a));
        a.nx += static_cast<float>(n.x);
        a.ny += static_cast<float>(n.y);
        a.nz += static_cast<float>(n.z);
        b.nx += static_cast<float>(n.x);
        b.ny += static_cast<float>(n.y);
        b.nz += static_cast<float>(n.z);
        c.nx += static_cast<float>(n.x);
        c.ny += static_cast<float>(n.y);
        c.nz += static_cast<float>(n.z);
    }

    for (auto& v : mesh.vertices) {
        const Vec3 n = normalize({v.nx, v.ny, v.nz});
        v.nx = static_cast<float>(n.x);
        v.ny = static_cast<float>(n.y);
        v.nz = static_cast<float>(n.z);
    }
}

struct QuantKey {
    std::int64_t x = 0, y = 0, z = 0;
    bool operator==(const QuantKey& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};
struct QuantKeyHash {
    std::size_t operator()(const QuantKey& q) const noexcept {
        std::uint64_t h = static_cast<std::uint64_t>(q.x) * 73856093ULL;
        h ^= static_cast<std::uint64_t>(q.y) * 19349663ULL;
        h ^= static_cast<std::uint64_t>(q.z) * 83492791ULL;
        return static_cast<std::size_t>(h);
    }
};

struct DirectedEdge {
    std::uint32_t from = 0;
    std::uint32_t to = 0;
};

struct EdgeRecord {
    int count = 0;
    DirectedEdge directed;
};

struct Quadric {
    // symmetric 4x4:
    // 00 01 02 03
    //    11 12 13
    //       22 23
    //          33
    double q00 = 0, q01 = 0, q02 = 0, q03 = 0;
    double q11 = 0, q12 = 0, q13 = 0;
    double q22 = 0, q23 = 0;
    double q33 = 0;

    Quadric& operator+=(const Quadric& o) {
        q00 += o.q00; q01 += o.q01; q02 += o.q02; q03 += o.q03;
        q11 += o.q11; q12 += o.q12; q13 += o.q13;
        q22 += o.q22; q23 += o.q23;
        q33 += o.q33;
        return *this;
    }

    double evaluate(const Vec3& p) const {
        const double x = p.x, y = p.y, z = p.z;
        return
            q00*x*x + 2*q01*x*y + 2*q02*x*z + 2*q03*x +
            q11*y*y + 2*q12*y*z + 2*q13*y +
            q22*z*z + 2*q23*z +
            q33;
    }
};

Quadric planeQuadric(const Vec3& n, double d, double weight) {
    const double a = n.x;
    const double b = n.y;
    const double c = n.z;
    const double s = std::max(1e-12, weight);

    Quadric q;
    q.q00 = s*a*a; q.q01 = s*a*b; q.q02 = s*a*c; q.q03 = s*a*d;
    q.q11 = s*b*b; q.q12 = s*b*c; q.q13 = s*b*d;
    q.q22 = s*c*c; q.q23 = s*c*d;
    q.q33 = s*d*d;
    return q;
}

bool solve3x3(
        double a00, double a01, double a02,
        double a10, double a11, double a12,
        double a20, double a21, double a22,
        double b0, double b1, double b2,
        Vec3* out) {
    if (!out) return false;

    const double det =
        a00*(a11*a22 - a12*a21) -
        a01*(a10*a22 - a12*a20) +
        a02*(a10*a21 - a11*a20);

    if (std::abs(det) < 1e-14) return false;

    const double dx =
        b0*(a11*a22 - a12*a21) -
        a01*(b1*a22 - a12*b2) +
        a02*(b1*a21 - a11*b2);
    const double dy =
        a00*(b1*a22 - a12*b2) -
        b0*(a10*a22 - a12*a20) +
        a02*(a10*b2 - b1*a20);
    const double dz =
        a00*(a11*b2 - b1*a21) -
        a01*(a10*b2 - b1*a20) +
        b0*(a10*a21 - a11*a20);

    *out = {dx/det, dy/det, dz/det};
    return std::isfinite(out->x) &&
           std::isfinite(out->y) &&
           std::isfinite(out->z);
}

Vec3 optimalCollapsePoint(const Quadric& q, const Vec3& a, const Vec3& b) {
    Vec3 p;
    if (solve3x3(
            q.q00, q.q01, q.q02,
            q.q01, q.q11, q.q12,
            q.q02, q.q12, q.q22,
            -q.q03, -q.q13, -q.q23,
            &p)) {
        return p;
    }

    const Vec3 m = (a + b) * 0.5;
    const double ca = q.evaluate(a);
    const double cb = q.evaluate(b);
    const double cm = q.evaluate(m);
    if (ca <= cb && ca <= cm) return a;
    if (cb <= ca && cb <= cm) return b;
    return m;
}

struct Candidate {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    Vec3 target;
    double cost = 0.0;
    bool boundaryEdge = false;
};

}  // namespace

bool MeshPostProcessor::run(
        AosMesh& mesh,
        const MeshPostProcessOptions& options,
        MeshPostProcessStats* outStats) {
    MeshPostProcessStats stats;
    stats.inputVertices = mesh.vertices.size();
    stats.inputTriangles = mesh.triangleCount();

    if (mesh.empty()) {
        if (outStats) *outStats = stats;
        return false;
    }

    if (options.weldEpsilon > 0.0f) {
        const std::size_t before = mesh.vertices.size();
        weld(mesh, options.weldEpsilon, options.minTriangleArea);
        stats.weldedVertices = before > mesh.vertices.size()
            ? before - mesh.vertices.size()
            : 0;
    }

    removeSmallComponents(
        mesh,
        options.minComponentTriangles,
        options.minComponentAreaRatio,
        &stats.removedComponents,
        &stats.removedComponentTriangles);

    fillSmallHoles(
        mesh,
        options.maxHoleEdges,
        options.maxHoleDiameterMeters,
        options.maxHoleDiameterBBoxRatio,
        &stats.boundaryLoops,
        &stats.filledHoles,
        &stats.addedHoleTriangles);

    if (options.targetTriangles > 0 &&
        mesh.triangleCount() > static_cast<std::size_t>(options.targetTriangles)) {
        simplifyQem(
            mesh,
            options.targetTriangles,
            options.qemMaxPasses,
            options.qemMaxNormalFlipDeg,
            options.preserveBoundary,
            options.minTriangleArea,
            &stats.qemCollapsedEdges,
            &stats.qemPasses);
    }

    compactMesh(mesh, options.minTriangleArea);
    recomputeNormals(mesh);

    stats.outputVertices = mesh.vertices.size();
    stats.outputTriangles = mesh.triangleCount();

    if (outStats) *outStats = stats;
    return !mesh.empty();
}

bool MeshPostProcessor::weld(
        AosMesh& mesh,
        float epsilon,
        float minTriangleArea) {
    if (mesh.vertices.empty() || epsilon <= 0.0f) return !mesh.empty();

    const double inv = 1.0 / static_cast<double>(epsilon);
    std::unordered_map<QuantKey, std::uint32_t, QuantKeyHash> map;
    map.reserve(mesh.vertices.size());

    struct Accum {
        Vec3 p;
        Vec3 c;
        Vec3 n;
        int count = 0;
    };

    std::vector<AosVertex> newVerts;
    std::vector<Accum> accums;
    std::vector<std::uint32_t> remap(mesh.vertices.size());

    newVerts.reserve(mesh.vertices.size());
    accums.reserve(mesh.vertices.size());

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto& v = mesh.vertices[i];
        const QuantKey key{
            static_cast<std::int64_t>(std::llround(v.px * inv)),
            static_cast<std::int64_t>(std::llround(v.py * inv)),
            static_cast<std::int64_t>(std::llround(v.pz * inv))
        };

        auto it = map.find(key);
        if (it == map.end()) {
            const auto idx = static_cast<std::uint32_t>(newVerts.size());
            map.emplace(key, idx);
            newVerts.push_back(v);
            accums.push_back({
                pos(v),
                color(v),
                {v.nx, v.ny, v.nz},
                1
            });
            remap[i] = idx;
        } else {
            const auto idx = it->second;
            auto& a = accums[idx];
            a.p = a.p + pos(v);
            a.c = a.c + color(v);
            a.n = a.n + Vec3{v.nx, v.ny, v.nz};
            ++a.count;
            remap[i] = idx;
        }
    }

    for (std::size_t i = 0; i < newVerts.size(); ++i) {
        const auto& a = accums[i];
        const double invCount = 1.0 / std::max(1, a.count);
        const Vec3 p = a.p * invCount;
        const Vec3 c = a.c * invCount;
        const Vec3 n = normalize(a.n);
        newVerts[i].px = static_cast<float>(p.x);
        newVerts[i].py = static_cast<float>(p.y);
        newVerts[i].pz = static_cast<float>(p.z);
        newVerts[i].r = static_cast<float>(c.x);
        newVerts[i].g = static_cast<float>(c.y);
        newVerts[i].b = static_cast<float>(c.z);
        newVerts[i].nx = static_cast<float>(n.x);
        newVerts[i].ny = static_cast<float>(n.y);
        newVerts[i].nz = static_cast<float>(n.z);
    }

    for (auto& i : mesh.indices) {
        if (i < remap.size()) i = remap[i];
    }

    mesh.vertices.swap(newVerts);
    compactMesh(mesh, minTriangleArea);
    recomputeNormals(mesh);
    return !mesh.empty();
}

bool MeshPostProcessor::removeSmallComponents(
        AosMesh& mesh,
        int minTriangles,
        float minAreaRatio,
        std::size_t* removedComponents,
        std::size_t* removedTriangles) {
    if (removedComponents) *removedComponents = 0;
    if (removedTriangles) *removedTriangles = 0;

    const std::size_t triCount = mesh.triangleCount();
    if (triCount == 0) return false;

    std::unordered_map<EdgeKey, std::vector<std::uint32_t>, EdgeKeyHash> edgeFaces;
    edgeFaces.reserve(triCount * 2u);

    for (std::uint32_t t = 0; t < triCount; ++t) {
        const std::uint32_t a = mesh.indices[t*3u + 0u];
        const std::uint32_t b = mesh.indices[t*3u + 1u];
        const std::uint32_t c = mesh.indices[t*3u + 2u];
        edgeFaces[EdgeKey(a,b)].push_back(t);
        edgeFaces[EdgeKey(b,c)].push_back(t);
        edgeFaces[EdgeKey(c,a)].push_back(t);
    }

    std::vector<std::vector<std::uint32_t>> adjacency(triCount);
    for (const auto& kv : edgeFaces) {
        const auto& faces = kv.second;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            for (std::size_t j = i + 1; j < faces.size(); ++j) {
                adjacency[faces[i]].push_back(faces[j]);
                adjacency[faces[j]].push_back(faces[i]);
            }
        }
    }

    struct Component {
        std::vector<std::uint32_t> tris;
        double area = 0.0;
    };

    std::vector<Component> components;
    std::vector<std::uint8_t> seen(triCount, 0);
    std::queue<std::uint32_t> q;

    for (std::uint32_t seed = 0; seed < triCount; ++seed) {
        if (seen[seed]) continue;
        Component comp;
        seen[seed] = 1;
        q.push(seed);

        while (!q.empty()) {
            const auto t = q.front();
            q.pop();
            comp.tris.push_back(t);

            const auto a = mesh.indices[t*3u + 0u];
            const auto b = mesh.indices[t*3u + 1u];
            const auto c = mesh.indices[t*3u + 2u];
            comp.area += triangleArea(
                mesh.vertices[a],
                mesh.vertices[b],
                mesh.vertices[c]);

            for (auto n : adjacency[t]) {
                if (!seen[n]) {
                    seen[n] = 1;
                    q.push(n);
                }
            }
        }
        components.push_back(std::move(comp));
    }

    if (components.size() <= 1) return true;

    double largestArea = 0.0;
    std::size_t largestCount = 0;
    std::size_t largestIndex = 0;
    for (std::size_t i = 0; i < components.size(); ++i) {
        const auto& c = components[i];
        if (c.area > largestArea ||
            (c.area == largestArea && c.tris.size() > largestCount)) {
            largestArea = c.area;
            largestCount = c.tris.size();
            largestIndex = i;
        }
    }

    std::vector<std::uint8_t> keep(triCount, 0);
    std::size_t remComp = 0;
    std::size_t remTri = 0;

    for (std::size_t i = 0; i < components.size(); ++i) {
        const auto& c = components[i];
        const bool keepThis =
            i == largestIndex ||
            static_cast<int>(c.tris.size()) >= minTriangles ||
            (largestArea > 0.0 &&
             c.area >= largestArea * std::max(0.0f, minAreaRatio));

        if (keepThis) {
            for (auto t : c.tris) keep[t] = 1;
        } else {
            ++remComp;
            remTri += c.tris.size();
        }
    }

    std::vector<std::uint32_t> newIdx;
    newIdx.reserve(mesh.indices.size());
    for (std::uint32_t t = 0; t < triCount; ++t) {
        if (!keep[t]) continue;
        newIdx.push_back(mesh.indices[t*3u + 0u]);
        newIdx.push_back(mesh.indices[t*3u + 1u]);
        newIdx.push_back(mesh.indices[t*3u + 2u]);
    }
    mesh.indices.swap(newIdx);
    compactMesh(mesh, 1e-12f);
    recomputeNormals(mesh);

    if (removedComponents) *removedComponents = remComp;
    if (removedTriangles) *removedTriangles = remTri;
    return !mesh.empty();
}

bool MeshPostProcessor::fillSmallHoles(
        AosMesh& mesh,
        int maxHoleEdges,
        float maxDiameterMeters,
        float maxDiameterBBoxRatio,
        std::size_t* boundaryLoopsOut,
        std::size_t* filledHolesOut,
        std::size_t* addedTrianglesOut) {
    if (boundaryLoopsOut) *boundaryLoopsOut = 0;
    if (filledHolesOut) *filledHolesOut = 0;
    if (addedTrianglesOut) *addedTrianglesOut = 0;

    if (mesh.empty()) return false;

    std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash> records;
    records.reserve(mesh.indices.size());

    auto addEdge = [&](std::uint32_t a, std::uint32_t b) {
        const EdgeKey k(a, b);
        auto& r = records[k];
        if (r.count == 0) r.directed = DirectedEdge{a, b};
        ++r.count;
    };

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const auto a = mesh.indices[i + 0];
        const auto b = mesh.indices[i + 1];
        const auto c = mesh.indices[i + 2];
        addEdge(a, b);
        addEdge(b, c);
        addEdge(c, a);
    }

    std::vector<DirectedEdge> boundary;
    boundary.reserve(records.size() / 8 + 1);
    for (const auto& kv : records) {
        if (kv.second.count == 1) {
            boundary.push_back(kv.second.directed);
        }
    }
    if (boundary.size() < 3) return true;

    Vec3 bbMin{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    };
    Vec3 bbMax{
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max()
    };
    for (const auto& v : mesh.vertices) {
        const Vec3 p = pos(v);
        bbMin.x = std::min(bbMin.x, p.x);
        bbMin.y = std::min(bbMin.y, p.y);
        bbMin.z = std::min(bbMin.z, p.z);
        bbMax.x = std::max(bbMax.x, p.x);
        bbMax.y = std::max(bbMax.y, p.y);
        bbMax.z = std::max(bbMax.z, p.z);
    }
    const double bboxDiag = norm(bbMax - bbMin);
    const double maxDiameter = std::min(
        maxDiameterMeters > 0.0f
            ? static_cast<double>(maxDiameterMeters)
            : std::numeric_limits<double>::max(),
        maxDiameterBBoxRatio > 0.0f
            ? bboxDiag * static_cast<double>(maxDiameterBBoxRatio)
            : std::numeric_limits<double>::max());

    std::unordered_multimap<std::uint32_t, std::size_t> fromMap;
    fromMap.reserve(boundary.size());
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        fromMap.emplace(boundary[i].from, i);
    }

    std::vector<std::uint8_t> used(boundary.size(), 0);
    std::size_t loops = 0;
    std::size_t filled = 0;
    std::size_t added = 0;

    for (std::size_t seed = 0; seed < boundary.size(); ++seed) {
        if (used[seed]) continue;

        std::vector<std::uint32_t> loop;
        loop.reserve(32);

        const auto first = boundary[seed];
        std::uint32_t current = first.from;
        std::uint32_t next = first.to;

        used[seed] = 1;
        loop.push_back(current);

        bool closed = false;
        int guard = 0;

        while (++guard < 4096) {
            loop.push_back(next);
            if (next == loop.front()) {
                closed = true;
                loop.pop_back();
                break;
            }

            auto range = fromMap.equal_range(next);
            std::size_t chosen = boundary.size();
            for (auto it = range.first; it != range.second; ++it) {
                if (!used[it->second]) {
                    chosen = it->second;
                    break;
                }
            }
            if (chosen == boundary.size()) break;

            used[chosen] = 1;
            current = boundary[chosen].from;
            next = boundary[chosen].to;
            (void)current;
        }

        if (!closed || loop.size() < 3) continue;
        ++loops;

        if (static_cast<int>(loop.size()) > maxHoleEdges) continue;

        double diameter = 0.0;
        for (std::size_t i = 0; i < loop.size(); ++i) {
            for (std::size_t j = i + 1; j < loop.size(); ++j) {
                diameter = std::max(
                    diameter,
                    norm(pos(mesh.vertices[loop[i]]) - pos(mesh.vertices[loop[j]])));
            }
        }
        if (std::isfinite(maxDiameter) && diameter > maxDiameter) continue;

        // Newell normal: a degenerate/non-planar loop has almost zero area.
        Vec3 newell{};
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const Vec3 p0 = pos(mesh.vertices[loop[i]]);
            const Vec3 p1 = pos(mesh.vertices[loop[(i + 1) % loop.size()]]);
            newell.x += (p0.y - p1.y) * (p0.z + p1.z);
            newell.y += (p0.z - p1.z) * (p0.x + p1.x);
            newell.z += (p0.x - p1.x) * (p0.y + p1.y);
        }
        if (norm(newell) <= 1e-10) continue;

        // Project to the most stable 2D plane (drop dominant normal axis).
        const Vec3 absN{
            std::abs(newell.x),
            std::abs(newell.y),
            std::abs(newell.z)
        };
        int dropAxis = 2;
        if (absN.x >= absN.y && absN.x >= absN.z) {
            dropAxis = 0;
        } else if (absN.y >= absN.z) {
            dropAxis = 1;
        }

        std::vector<Vec2> projected;
        projected.reserve(loop.size());
        for (auto vi : loop) {
            const Vec3 p = pos(mesh.vertices[vi]);
            if (dropAxis == 0) {
                projected.push_back({p.y, p.z});
            } else if (dropAxis == 1) {
                projected.push_back({p.x, p.z});
            } else {
                projected.push_back({p.x, p.y});
            }
        }

        const double signedArea = polygonArea2(projected);
        if (std::abs(signedArea) <= 1e-14) continue;
        const double orientation = signedArea > 0.0 ? 1.0 : -1.0;

        // Ear clipping handles small concave boundary loops without creating
        // the out-of-polygon triangles that a centroid fan can create.
        std::vector<int> polygon(loop.size());
        std::iota(polygon.begin(), polygon.end(), 0);

        std::vector<std::array<std::uint32_t, 3>> capTriangles;
        capTriangles.reserve(loop.size() - 2u);

        int earGuard = 0;
        while (polygon.size() > 3 && ++earGuard < 4096) {
            bool clipped = false;

            for (std::size_t pi = 0; pi < polygon.size(); ++pi) {
                const int prev =
                    polygon[(pi + polygon.size() - 1) % polygon.size()];
                const int curr = polygon[pi];
                const int next =
                    polygon[(pi + 1) % polygon.size()];

                const double turn =
                    cross2(
                        projected[prev],
                        projected[curr],
                        projected[next]) *
                    orientation;
                if (turn <= 1e-14) {
                    continue;
                }

                bool contains = false;
                for (int candidate : polygon) {
                    if (candidate == prev ||
                        candidate == curr ||
                        candidate == next) {
                        continue;
                    }
                    if (pointInTriangle2(
                            projected[candidate],
                            projected[prev],
                            projected[curr],
                            projected[next],
                            orientation)) {
                        contains = true;
                        break;
                    }
                }
                if (contains) continue;

                capTriangles.push_back({
                    loop[static_cast<std::size_t>(prev)],
                    loop[static_cast<std::size_t>(curr)],
                    loop[static_cast<std::size_t>(next)]
                });
                polygon.erase(
                    polygon.begin() +
                    static_cast<std::ptrdiff_t>(pi));
                clipped = true;
                break;
            }

            if (!clipped) {
                capTriangles.clear();
                break;
            }
        }

        if (polygon.size() == 3) {
            capTriangles.push_back({
                loop[static_cast<std::size_t>(polygon[0])],
                loop[static_cast<std::size_t>(polygon[1])],
                loop[static_cast<std::size_t>(polygon[2])]
            });
        }

        if (capTriangles.size() != loop.size() - 2u) {
            continue;
        }

        // Existing boundary edges use the loop's directed winding. A valid
        // adjacent cap must use the opposite surface normal.
        const Vec3 desiredNormal = normalize(newell * -1.0);

        for (auto tri : capTriangles) {
            const Vec3 a = pos(mesh.vertices[tri[0]]);
            const Vec3 b = pos(mesh.vertices[tri[1]]);
            const Vec3 c = pos(mesh.vertices[tri[2]]);
            if (dot(cross(b-a, c-a), desiredNormal) < 0.0) {
                std::swap(tri[1], tri[2]);
            }

            mesh.indices.push_back(tri[0]);
            mesh.indices.push_back(tri[1]);
            mesh.indices.push_back(tri[2]);
            ++added;
        }

        ++filled;
    }

    compactMesh(mesh, 1e-12f);
    recomputeNormals(mesh);

    if (boundaryLoopsOut) *boundaryLoopsOut = loops;
    if (filledHolesOut) *filledHolesOut = filled;
    if (addedTrianglesOut) *addedTrianglesOut = added;
    return !mesh.empty();
}

bool MeshPostProcessor::simplifyQem(
        AosMesh& mesh,
        int targetTriangles,
        int maxPasses,
        float maxNormalFlipDeg,
        bool preserveBoundary,
        float minTriangleArea,
        std::size_t* collapsedEdgesOut,
        std::size_t* passesOut) {
    if (collapsedEdgesOut) *collapsedEdgesOut = 0;
    if (passesOut) *passesOut = 0;

    if (targetTriangles <= 0 ||
        mesh.triangleCount() <= static_cast<std::size_t>(targetTriangles)) {
        return !mesh.empty();
    }

    std::size_t totalCollapsed = 0;
    std::size_t passCount = 0;
    const double cosLimit =
        std::cos(
            std::clamp(
                static_cast<double>(maxNormalFlipDeg),
                1.0,
                89.0) *
            3.14159265358979323846 / 180.0);

    for (int pass = 0;
         pass < std::max(1, maxPasses) &&
         mesh.triangleCount() > static_cast<std::size_t>(targetTriangles);
         ++pass) {
        ++passCount;

        const std::size_t vertexCount = mesh.vertices.size();
        const std::size_t triCount = mesh.triangleCount();
        if (vertexCount < 4 || triCount < 4) break;

        std::vector<Quadric> quadrics(vertexCount);
        std::vector<std::vector<std::uint32_t>> vertexFaces(vertexCount);

        for (std::uint32_t t = 0; t < triCount; ++t) {
            const auto a = mesh.indices[t*3u + 0u];
            const auto b = mesh.indices[t*3u + 1u];
            const auto c = mesh.indices[t*3u + 2u];
            if (a >= vertexCount || b >= vertexCount || c >= vertexCount) continue;

            const Vec3 pa = pos(mesh.vertices[a]);
            const Vec3 pb = pos(mesh.vertices[b]);
            const Vec3 pc = pos(mesh.vertices[c]);

            const Vec3 cr = cross(pb - pa, pc - pa);
            const double doubleArea = norm(cr);
            if (doubleArea <= 1e-18) continue;

            const Vec3 n = cr / doubleArea;
            const double d = -dot(n, pa);
            const double area = 0.5 * doubleArea;
            const Quadric q = planeQuadric(n, d, area);

            quadrics[a] += q;
            quadrics[b] += q;
            quadrics[c] += q;

            vertexFaces[a].push_back(t);
            vertexFaces[b].push_back(t);
            vertexFaces[c].push_back(t);
        }

        std::unordered_map<EdgeKey, int, EdgeKeyHash> edgeCounts;
        edgeCounts.reserve(triCount * 2u);
        auto addEdge = [&](std::uint32_t a, std::uint32_t b) {
            ++edgeCounts[EdgeKey(a,b)];
        };

        for (std::uint32_t t = 0; t < triCount; ++t) {
            const auto a = mesh.indices[t*3u + 0u];
            const auto b = mesh.indices[t*3u + 1u];
            const auto c = mesh.indices[t*3u + 2u];
            addEdge(a,b); addEdge(b,c); addEdge(c,a);
        }

        std::vector<std::uint8_t> isBoundary(vertexCount, 0);
        std::vector<std::vector<std::uint32_t>> vertexNeighbors(vertexCount);

        for (const auto& kv : edgeCounts) {
            if (kv.second == 1) {
                isBoundary[kv.first.a] = 1;
                isBoundary[kv.first.b] = 1;
            }

            if (kv.first.a < vertexCount &&
                kv.first.b < vertexCount) {
                vertexNeighbors[kv.first.a].push_back(kv.first.b);
                vertexNeighbors[kv.first.b].push_back(kv.first.a);
            }
        }

        for (auto& n : vertexNeighbors) {
            std::sort(n.begin(), n.end());
            n.erase(
                std::unique(n.begin(), n.end()),
                n.end());
        }

        auto commonNeighborCount =
            [&](std::uint32_t a, std::uint32_t b) -> int {
                const auto& na = vertexNeighbors[a];
                const auto& nb = vertexNeighbors[b];

                std::size_t ia = 0;
                std::size_t ib = 0;
                int count = 0;

                while (ia < na.size() && ib < nb.size()) {
                    if (na[ia] == nb[ib]) {
                        if (na[ia] != a &&
                            na[ia] != b) {
                            ++count;
                        }
                        ++ia;
                        ++ib;
                    } else if (na[ia] < nb[ib]) {
                        ++ia;
                    } else {
                        ++ib;
                    }
                }
                return count;
            };

        std::vector<Candidate> candidates;
        candidates.reserve(edgeCounts.size());

        for (const auto& kv : edgeCounts) {
            const auto a = kv.first.a;
            const auto b = kv.first.b;
            if (a >= vertexCount || b >= vertexCount) continue;

            const bool boundaryEdge = kv.second == 1;
            if (preserveBoundary) {
                if (isBoundary[a] != isBoundary[b]) continue;
                if (isBoundary[a] && isBoundary[b] && !boundaryEdge) continue;
            }

            // Link condition: a manifold interior edge has exactly two common
            // one-ring neighbors; a manifold boundary edge has exactly one.
            // Rejecting violations prevents QEM from introducing bow-ties and
            // most non-manifold edge collapses.
            const int common =
                commonNeighborCount(a, b);
            if (boundaryEdge) {
                if (common != 1) continue;
            } else {
                if (common != 2) continue;
            }

            Quadric q = quadrics[a];
            q += quadrics[b];
            const Vec3 pa = pos(mesh.vertices[a]);
            const Vec3 pb = pos(mesh.vertices[b]);
            Vec3 target = optimalCollapsePoint(q, pa, pb);

            double cost = q.evaluate(target);
            if (boundaryEdge) cost *= 4.0;

            candidates.push_back(Candidate{
                a, b, target, cost, boundaryEdge
            });
        }

        if (candidates.empty()) break;

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& x, const Candidate& y) {
                return x.cost < y.cost;
            });

        std::vector<std::uint8_t> locked(vertexCount, 0);
        std::vector<std::uint32_t> parent(vertexCount);
        std::iota(parent.begin(), parent.end(), 0u);
        std::vector<Vec3> newPos(vertexCount);
        std::vector<Vec3> newColor(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i) {
            newPos[i] = pos(mesh.vertices[i]);
            newColor[i] = color(mesh.vertices[i]);
        }

        std::size_t selected = 0;
        const std::size_t needRemove =
            triCount > static_cast<std::size_t>(targetTriangles)
                ? triCount - static_cast<std::size_t>(targetTriangles)
                : 0;

        auto safe = [&](const Candidate& c) -> bool {
            std::vector<std::uint32_t> affected;
            affected.reserve(
                vertexFaces[c.a].size() +
                vertexFaces[c.b].size());
            affected.insert(
                affected.end(),
                vertexFaces[c.a].begin(),
                vertexFaces[c.a].end());
            affected.insert(
                affected.end(),
                vertexFaces[c.b].begin(),
                vertexFaces[c.b].end());
            std::sort(affected.begin(), affected.end());
            affected.erase(
                std::unique(affected.begin(), affected.end()),
                affected.end());

            for (auto t : affected) {
                const auto ia = mesh.indices[t*3u + 0u];
                const auto ib = mesh.indices[t*3u + 1u];
                const auto ic = mesh.indices[t*3u + 2u];

                // Face on the collapsed edge disappears; that is expected.
                const bool hasA =
                    ia == c.a || ib == c.a || ic == c.a;
                const bool hasB =
                    ia == c.b || ib == c.b || ic == c.b;
                if (hasA && hasB) continue;

                Vec3 pa = pos(mesh.vertices[ia]);
                Vec3 pb = pos(mesh.vertices[ib]);
                Vec3 pc = pos(mesh.vertices[ic]);
                const Vec3 oldN = normalize(cross(pb-pa, pc-pa));

                if (ia == c.a || ia == c.b) pa = c.target;
                if (ib == c.a || ib == c.b) pb = c.target;
                if (ic == c.a || ic == c.b) pc = c.target;

                const Vec3 cr = cross(pb-pa, pc-pa);
                const double area = 0.5 * norm(cr);
                if (area < std::max(1e-16f, minTriangleArea)) {
                    return false;
                }

                const Vec3 newN = normalize(cr);
                if (dot(oldN, newN) < cosLimit) {
                    return false;
                }
            }
            return true;
        };

        for (const auto& c : candidates) {
            if (locked[c.a] || locked[c.b]) continue;
            if (!safe(c)) continue;

            const std::uint32_t keep = std::min(c.a, c.b);
            const std::uint32_t remove = std::max(c.a, c.b);

            parent[remove] = keep;
            newPos[keep] = c.target;
            newColor[keep] =
                (color(mesh.vertices[c.a]) +
                 color(mesh.vertices[c.b])) * 0.5;

            locked[c.a] = locked[c.b] = 1;
            ++selected;

            // One interior edge collapse usually removes 2 triangles,
            // a boundary edge usually removes 1. Stop slightly early so the
            // next pass can make a better final decision.
            const std::size_t approxRemoved =
                selected * 2u;
            if (approxRemoved >= needRemove) break;
        }

        if (selected == 0) break;

        auto rootOf = [&](std::uint32_t v) {
            while (parent[v] != v) v = parent[v];
            return v;
        };

        std::vector<AosVertex> vertices = mesh.vertices;
        for (std::size_t i = 0; i < vertexCount; ++i) {
            const auto r = rootOf(static_cast<std::uint32_t>(i));
            if (r != i) continue;
            vertices[i].px = static_cast<float>(newPos[i].x);
            vertices[i].py = static_cast<float>(newPos[i].y);
            vertices[i].pz = static_cast<float>(newPos[i].z);
            vertices[i].r = static_cast<float>(newColor[i].x);
            vertices[i].g = static_cast<float>(newColor[i].y);
            vertices[i].b = static_cast<float>(newColor[i].z);
        }

        std::vector<std::uint32_t> indices;
        indices.reserve(mesh.indices.size());

        for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            const auto a = rootOf(mesh.indices[i + 0]);
            const auto b = rootOf(mesh.indices[i + 1]);
            const auto c = rootOf(mesh.indices[i + 2]);
            if (a == b || b == c || c == a) continue;

            if (triangleArea(vertices[a], vertices[b], vertices[c]) <
                std::max(1e-16f, minTriangleArea)) {
                continue;
            }

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(c);
        }

        mesh.vertices.swap(vertices);
        mesh.indices.swap(indices);
        compactMesh(mesh, minTriangleArea);
        recomputeNormals(mesh);
        totalCollapsed += selected;

        if (selected < 4) break;
    }

    if (collapsedEdgesOut) *collapsedEdgesOut = totalCollapsed;
    if (passesOut) *passesOut = passCount;
    return !mesh.empty();
}
