#include "mesh_registration.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Vec3 = Eigen::Vector3f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;

struct Cell {
    int x = 0;
    int y = 0;
    int z = 0;
    bool operator==(const Cell& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
        // Three independent large odd multipliers. Cast through uint32_t so
        // negative cells hash deterministically too.
        const std::uint64_t x = static_cast<std::uint32_t>(c.x);
        const std::uint64_t y = static_cast<std::uint32_t>(c.y);
        const std::uint64_t z = static_cast<std::uint32_t>(c.z);
        return static_cast<std::size_t>(
            (x * 73856093ull) ^ (y * 19349663ull) ^ (z * 83492791ull));
    }
};

struct Cloud {
    std::vector<Vec3> points;
    Vec3 centroid = Vec3::Zero();
    float diagonal = 0.0f;
};

Cloud sampleMesh(const Mesh& mesh, std::size_t maxPoints) {
    Cloud out;
    const std::size_t n = mesh.vertexCount();
    if (n == 0 || mesh.positions.size() < n * 3) return out;

    const std::size_t step = std::max<std::size_t>(1, (n + maxPoints - 1) / maxPoints);
    out.points.reserve((n + step - 1) / step);

    Vec3 minP(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    Vec3 maxP = -minP;

    for (std::size_t i = 0; i < n; i += step) {
        const Vec3 p(
            mesh.positions[i * 3 + 0],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2]);
        if (!p.allFinite()) continue;
        out.points.push_back(p);
        out.centroid += p;
        minP = minP.cwiseMin(p);
        maxP = maxP.cwiseMax(p);
    }
    if (out.points.empty()) return out;
    out.centroid /= static_cast<float>(out.points.size());
    out.diagonal = (maxP - minP).norm();
    return out;
}

Cell toCell(const Vec3& p, float cell) {
    return {
        static_cast<int>(std::floor(p.x() / cell)),
        static_cast<int>(std::floor(p.y() / cell)),
        static_cast<int>(std::floor(p.z() / cell))
    };
}

class SpatialGrid {
public:
    SpatialGrid(const Cloud& cloud, float cell)
        : points_(cloud.points), cell_(cell) {
        buckets_.reserve(points_.size() * 2 + 1);
        for (std::size_t i = 0; i < points_.size(); ++i) {
            buckets_[toCell(points_[i], cell_)].push_back(static_cast<int>(i));
        }
    }

    bool nearest(const Vec3& p, float maxDistance, Vec3* out, float* outD2) const {
        if (!out || !outD2 || points_.empty()) return false;
        const Cell c = toCell(p, cell_);
        const int radius = std::max(1, static_cast<int>(std::ceil(maxDistance / cell_)));
        float best = maxDistance * maxDistance;
        int bestIdx = -1;
        for (int dz = -radius; dz <= radius; ++dz) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const auto it = buckets_.find(Cell{c.x + dx, c.y + dy, c.z + dz});
                    if (it == buckets_.end()) continue;
                    for (int idx : it->second) {
                        const float d2 = (points_[static_cast<std::size_t>(idx)] - p).squaredNorm();
                        if (d2 < best) {
                            best = d2;
                            bestIdx = idx;
                        }
                    }
                }
            }
        }
        if (bestIdx < 0) return false;
        *out = points_[static_cast<std::size_t>(bestIdx)];
        *outD2 = best;
        return true;
    }

private:
    const std::vector<Vec3>& points_;
    float cell_ = 0.02f;
    std::unordered_map<Cell, std::vector<int>, CellHash> buckets_;
};

Mat4 rigidFit(const std::vector<Vec3>& src, const std::vector<Vec3>& dst) {
    Mat4 out = Mat4::Identity();
    if (src.size() != dst.size() || src.size() < 3) return out;

    Vec3 cs = Vec3::Zero();
    Vec3 cd = Vec3::Zero();
    for (std::size_t i = 0; i < src.size(); ++i) {
        cs += src[i];
        cd += dst[i];
    }
    const float inv = 1.0f / static_cast<float>(src.size());
    cs *= inv;
    cd *= inv;

    Mat3 H = Mat3::Zero();
    for (std::size_t i = 0; i < src.size(); ++i) {
        H += (src[i] - cs) * (dst[i] - cd).transpose();
    }

    Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0.0f) {
        Mat3 V = svd.matrixV();
        V.col(2) *= -1.0f;
        R = V * svd.matrixU().transpose();
    }
    const Vec3 t = cd - R * cs;
    out.block<3, 3>(0, 0) = R;
    out.block<3, 1>(0, 3) = t;
    return out;
}

Vec3 transformPoint(const Mat4& T, const Vec3& p) {
    return T.block<3,3>(0,0) * p + T.block<3,1>(0,3);
}

struct Candidate {
    Mat4 T = Mat4::Identity();
    float rmse = std::numeric_limits<float>::infinity();
    float overlap = 0.0f;
    int inliers = 0;
    int iterations = 0;
    int yawDeg = 0;
    bool ok = false;
};

Candidate refineCandidate(
    const Cloud& moving,
    const SpatialGrid& grid,
    Mat4 T,
    float baseMaxDistance,
    int yawDeg) {

    Candidate result;
    result.T = T;
    result.yawDeg = yawDeg;

    const std::size_t n = moving.points.size();
    if (n < 50) return result;

    std::vector<Vec3> src;
    std::vector<Vec3> dst;
    std::vector<float> distances;
    src.reserve(n);
    dst.reserve(n);
    distances.reserve(n);

    float prevRmse = std::numeric_limits<float>::infinity();
    for (int iter = 0; iter < 18; ++iter) {
        const float alpha = static_cast<float>(iter) / 17.0f;
        const float maxDistance = baseMaxDistance * (1.0f - 0.55f * alpha);
        src.clear();
        dst.clear();
        distances.clear();

        for (const Vec3& p0 : moving.points) {
            const Vec3 p = transformPoint(result.T, p0);
            Vec3 q;
            float d2 = 0.0f;
            if (!grid.nearest(p, maxDistance, &q, &d2)) continue;
            src.push_back(p);
            dst.push_back(q);
            distances.push_back(std::sqrt(std::max(0.0f, d2)));
        }
        if (src.size() < 80) break;

        // Trim the longest residual tail. This keeps table/ground leakage or a
        // partial extra component from steering the rigid fit.
        std::vector<float> sorted = distances;
        const std::size_t keep = std::max<std::size_t>(60, sorted.size() * 4 / 5);
        std::nth_element(sorted.begin(), sorted.begin() + (keep - 1), sorted.end());
        const float trim = sorted[keep - 1];

        std::vector<Vec3> srcTrim;
        std::vector<Vec3> dstTrim;
        srcTrim.reserve(keep);
        dstTrim.reserve(keep);
        double sq = 0.0;
        for (std::size_t i = 0; i < src.size(); ++i) {
            if (distances[i] > trim) continue;
            srcTrim.push_back(src[i]);
            dstTrim.push_back(dst[i]);
            sq += static_cast<double>(distances[i]) * distances[i];
        }
        if (srcTrim.size() < 60) break;

        const float rmse = static_cast<float>(std::sqrt(sq / srcTrim.size()));
        const Mat4 delta = rigidFit(srcTrim, dstTrim);
        result.T = delta * result.T;
        result.iterations = iter + 1;
        result.inliers = static_cast<int>(srcTrim.size());
        result.overlap = static_cast<float>(srcTrim.size()) / static_cast<float>(n);
        result.rmse = rmse;

        const Vec3 dt = delta.block<3,1>(0,3);
        const Mat3 dR = delta.block<3,3>(0,0);
        const float cosTheta = std::clamp((dR.trace() - 1.0f) * 0.5f, -1.0f, 1.0f);
        const float angle = std::acos(cosTheta);
        if (std::abs(prevRmse - rmse) < 0.00015f && dt.norm() < 0.0006f && angle < 0.0025f) {
            break;
        }
        prevRmse = rmse;
    }

    result.ok =
        result.inliers >= 120 &&
        result.overlap >= 0.20f &&
        std::isfinite(result.rmse) &&
        result.rmse <= std::clamp(moving.diagonal * 0.020f, 0.008f, 0.030f);
    return result;
}

struct TriKey {
    std::uint32_t a = 0, b = 0, c = 0;
    bool operator==(const TriKey& o) const { return a == o.a && b == o.b && c == o.c; }
};
struct TriKeyHash {
    std::size_t operator()(const TriKey& k) const noexcept {
        const std::uint64_t x = (static_cast<std::uint64_t>(k.a) << 32) | k.b;
        return static_cast<std::size_t>((x * 11400714819323198485ull) ^ (k.c * 2654435761u));
    }
};

void removeDuplicateTriangles(Mesh& mesh) {
    if (mesh.indices.size() < 6) return;
    std::unordered_map<TriKey, std::uint8_t, TriKeyHash> seen;
    seen.reserve(mesh.indices.size() / 2);
    std::vector<std::uint32_t> unique;
    unique.reserve(mesh.indices.size());
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const std::uint32_t a0 = mesh.indices[i];
        const std::uint32_t b0 = mesh.indices[i + 1];
        const std::uint32_t c0 = mesh.indices[i + 2];
        std::array<std::uint32_t,3> sorted{a0,b0,c0};
        std::sort(sorted.begin(), sorted.end());
        const TriKey key{sorted[0], sorted[1], sorted[2]};
        if (!seen.emplace(key, 1).second) continue;
        unique.push_back(a0); unique.push_back(b0); unique.push_back(c0);
    }
    mesh.indices.swap(unique);
}

Mesh transformAndConcatenate(const Mesh& reference, const Mesh& moving, const Mat4& T) {
    Mesh out = reference;
    const std::size_t refN = reference.vertexCount();
    const std::size_t movN = moving.vertexCount();
    if (movN == 0) return out;

    out.positions.reserve((refN + movN) * 3);
    out.normals.reserve((refN + movN) * 3);
    out.colors.reserve((refN + movN) * 3);
    out.indices.reserve(reference.indices.size() + moving.indices.size());

    const Mat3 R = T.block<3,3>(0,0);
    const Vec3 t = T.block<3,1>(0,3);
    const bool movNrm = moving.normals.size() >= movN * 3;
    const bool movCol = moving.colors.size() >= movN * 3;

    for (std::size_t i = 0; i < movN; ++i) {
        const Vec3 p(
            moving.positions[i*3+0],
            moving.positions[i*3+1],
            moving.positions[i*3+2]);
        const Vec3 pt = R * p + t;
        out.positions.push_back(pt.x());
        out.positions.push_back(pt.y());
        out.positions.push_back(pt.z());

        Vec3 n(0,0,1);
        if (movNrm) {
            n = R * Vec3(
                moving.normals[i*3+0],
                moving.normals[i*3+1],
                moving.normals[i*3+2]);
            const float nn = n.norm();
            if (nn > 1e-8f) n /= nn;
        }
        out.normals.push_back(n.x());
        out.normals.push_back(n.y());
        out.normals.push_back(n.z());

        if (movCol) {
            out.colors.push_back(moving.colors[i*3+0]);
            out.colors.push_back(moving.colors[i*3+1]);
            out.colors.push_back(moving.colors[i*3+2]);
        } else {
            out.colors.push_back(0.72f);
            out.colors.push_back(0.72f);
            out.colors.push_back(0.72f);
        }
    }

    const std::uint32_t offset = static_cast<std::uint32_t>(refN);
    for (std::uint32_t idx : moving.indices) out.indices.push_back(offset + idx);

    meshRecomputeNormals(out);
    // Keep a mobile-safe cumulative model. QEM preserves actual merged geometry
    // much better than silently refusing later segments because the mesh grew.
    if (out.triangleCount() > 520000) {
        MeshBuildStats tmp;
        meshDecimate(out, 420000, &tmp);
        meshRecomputeNormals(out);
    }
    return out;
}

std::array<float, 16> toArray(const Mat4& T) {
    std::array<float, 16> out{};
    // Row-major contract for JNI / diagnostics.
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[static_cast<std::size_t>(r*4+c)] = T(r,c);
    return out;
}

} // namespace

bool alignAndMergeMeshes(
    const Mesh& reference,
    const Mesh& moving,
    Mesh* output,
    std::array<float, 16>* outReferenceFromMoving,
    MeshRegistrationStats* stats,
    int forcedYawDeg) {

    MeshRegistrationStats local;
    local.referenceVertices = reference.vertexCount();
    local.movingVertices = moving.vertexCount();

    if (!output || reference.triangleCount() < 100 || moving.triangleCount() < 100) {
        if (stats) *stats = local;
        return false;
    }

    const Cloud ref = sampleMesh(reference, 18000);
    const Cloud mov = sampleMesh(moving, 14000);
    if (ref.points.size() < 150 || mov.points.size() < 150) {
        if (stats) *stats = local;
        return false;
    }

    const float scale = std::max(0.05f, std::min(ref.diagonal, mov.diagonal));
    const float cell = std::clamp(scale * 0.018f, 0.008f, 0.045f);
    const float maxDistance = std::clamp(scale * 0.075f, 0.025f, 0.12f);
    const SpatialGrid grid(ref, cell);

    Candidate best;
    float bestScore = std::numeric_limits<float>::infinity();

    // VINS keeps gravity consistent but independent sessions can start at any yaw.
    // Automatic mode searches every 45 degrees. Manual-assisted mode supplies one
    // approximate yaw and still lets full 6-DoF ICP refine the residual.
    std::vector<int> yawHypotheses;
    if (forcedYawDeg >= 0) {
        int y = forcedYawDeg % 360;
        if (y < 0) y += 360;
        yawHypotheses.push_back(y);
    } else {
        for (int y = 0; y < 360; y += 45) yawHypotheses.push_back(y);
    }
    for (int yaw : yawHypotheses) {
        constexpr float kPi = 3.14159265358979323846f;
        const float a = static_cast<float>(yaw) * (kPi / 180.0f);
        Mat3 Rz = Mat3::Identity();
        Rz(0,0) = std::cos(a); Rz(0,1) = -std::sin(a);
        Rz(1,0) = std::sin(a); Rz(1,1) =  std::cos(a);
        Mat4 initial = Mat4::Identity();
        initial.block<3,3>(0,0) = Rz;
        initial.block<3,1>(0,3) = ref.centroid - Rz * mov.centroid;

        Candidate c = refineCandidate(mov, grid, initial, maxDistance, yaw);
        if (!std::isfinite(c.rmse) || c.inliers <= 0) continue;
        const float score = c.rmse / std::max(0.08f, c.overlap);
        if (score < bestScore) {
            bestScore = score;
            best = c;
        }
    }

    if (!best.ok) {
        local.rmseMeters = std::isfinite(best.rmse) ? best.rmse : 0.0f;
        local.overlap = best.overlap;
        local.inliers = best.inliers;
        local.iterations = best.iterations;
        local.yawHypothesisDeg = best.yawDeg;
        if (stats) *stats = local;
        return false;
    }

    Mesh merged = transformAndConcatenate(reference, moving, best.T);
    // Light spatial weld collapses duplicate overlap vertices after ICP without
    // erasing the scan's native geometric detail. The cell scales with object size.
    meshClusterSimplify(merged, std::clamp(scale * 0.0025f, 0.0010f, 0.0040f));
    removeDuplicateTriangles(merged);
    meshRecomputeNormals(merged);
    if (merged.triangleCount() <= 0 || merged.vertexCount() == 0) {
        if (stats) *stats = local;
        return false;
    }

    local.ok = true;
    local.rmseMeters = best.rmse;
    local.overlap = best.overlap;
    local.inliers = best.inliers;
    local.iterations = best.iterations;
    local.yawHypothesisDeg = best.yawDeg;
    local.outputVertices = merged.vertexCount();
    local.outputTriangles = merged.triangleCount();

    *output = std::move(merged);
    if (outReferenceFromMoving) *outReferenceFromMoving = toArray(best.T);
    if (stats) *stats = local;
    return true;
}
