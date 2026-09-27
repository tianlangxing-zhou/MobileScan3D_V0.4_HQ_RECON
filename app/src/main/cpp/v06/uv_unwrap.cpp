#include "uv_unwrap.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <android/log.h>

// V0.13.8：xatlas 失败退 fallback 时必须留下原因。实测（190525 会话）出现
// uv=fallback，但旧代码在 AddMesh 失败 / atlas 维度异常 / 顶点校验失败三条
// 路径上都静默直落 fallback，无法区分是网格质量还是库内问题。
// V0.13.10：根因是 xatlas 会丢弃退化三角形、留下 atlasIndex=-1 的孤儿顶点；
// 旧代码遍历整个 vertexArray 遇任一孤儿即整批 fallback。改为按 indexArray
// （仅含被打包的 charted 面）构建输出，逐面跳过孤儿，保留 xatlas 紧凑 UV。
#define UW_LOGW(...) \
    __android_log_print(ANDROID_LOG_WARN, "MobileScan3D", __VA_ARGS__)

#if defined(MOBILESCAN_HAS_XATLAS)
#include "xatlas.h"
#endif

bool UvUnwrapper::unwrap(
        const AosMesh& mesh,
        int resolution,
        int padding,
        UvMesh& out,
        UvUnwrapStats* stats) {
    out.clear();
    UvUnwrapStats localStats;
    localStats.inputVertices = mesh.vertices.size();

    if (mesh.empty() || resolution < 128) {
        if (stats) *stats = localStats;
        return false;
    }

#if defined(MOBILESCAN_HAS_XATLAS)
    xatlas::Atlas* atlas = xatlas::Create();
    if (atlas) {
        xatlas::MeshDecl decl;
        decl.vertexPositionData =
            static_cast<const void*>(&mesh.vertices[0].px);
        decl.vertexPositionStride = sizeof(AosVertex);
        decl.vertexCount =
            static_cast<std::uint32_t>(mesh.vertices.size());

        decl.vertexNormalData =
            static_cast<const void*>(&mesh.vertices[0].nx);
        decl.vertexNormalStride = sizeof(AosVertex);

        decl.indexData = mesh.indices.data();
        decl.indexCount =
            static_cast<std::uint32_t>(mesh.indices.size());
        decl.indexFormat = xatlas::IndexFormat::UInt32;

        const auto addResult =
            xatlas::AddMesh(atlas, decl, 1);

        if (addResult != xatlas::AddMeshError::Success) {
            // V0.13.8：AddMesh 失败是 fallback 的一条独立路径，必须留下原因
            UW_LOGW("uv_unwrap: AddMesh failed err=%d verts=%u idx=%u",
                    static_cast<int>(addResult),
                    static_cast<std::uint32_t>(mesh.vertices.size()),
                    static_cast<std::uint32_t>(mesh.indices.size()));
        }

        if (addResult == xatlas::AddMeshError::Success) {
            xatlas::ChartOptions chart;
            chart.maxIterations = 2;
            chart.fixWinding = true;
            chart.normalDeviationWeight = 2.0f;
            chart.normalSeamWeight = 4.0f;
            chart.maxCost = 2.0f;

            xatlas::PackOptions pack;
            pack.resolution =
                static_cast<std::uint32_t>(resolution);
            pack.padding =
                static_cast<std::uint32_t>(
                    std::clamp(padding, 1, 32));
            pack.bilinear = true;
            pack.blockAlign = false;
            pack.bruteForce = false;
            pack.createImage = false;
            pack.rotateChartsToAxis = true;
            pack.rotateCharts = true;

            xatlas::Generate(atlas, chart, pack);

            if (atlas->meshCount != 1 ||
                atlas->atlasCount != 1 ||
                atlas->width <= 0 ||
                atlas->height <= 0) {
                UW_LOGW("uv_unwrap: Generate unusable meshCount=%u "
                        "atlasCount=%u %dx%d charts=%u",
                        static_cast<std::uint32_t>(atlas->meshCount),
                        static_cast<std::uint32_t>(atlas->atlasCount),
                        atlas->width, atlas->height,
                        static_cast<std::uint32_t>(atlas->chartCount));
            }

            if (atlas->meshCount == 1 &&
                atlas->atlasCount == 1 &&
                atlas->width > 0 &&
                atlas->height > 0) {
                const xatlas::Mesh& xm = atlas->meshes[0];

                if (xm.indexCount == 0 || (xm.indexCount % 3u) != 0) {
                    UW_LOGW("uv_unwrap: xatlas produced no valid index "
                            "buffer indexCount=%u", xm.indexCount);
                } else {
                    const float invW =
                        1.0f / static_cast<float>(atlas->width);
                    const float invH =
                        1.0f / static_cast<float>(atlas->height);

                    out.vertices.clear();
                    out.indices.clear();
                    out.vertices.reserve(xm.indexCount);
                    out.indices.reserve(xm.indexCount);

                    std::uint32_t dropped = 0;
                    const std::uint32_t triCount = xm.indexCount / 3u;
                    for (std::uint32_t t = 0; t < triCount; ++t) {
                        // indexArray holds only charted faces. Orphan vertices
                        // (atlasIndex == -1, from degenerate faces xatlas drops)
                        // are skipped per-triangle instead of failing the whole
                        // unwrap — previously a single orphan forced the entire
                        // mesh into the low-res triangle-soup fallback.
                        bool triOk = true;
                        for (int k = 0; k < 3; ++k) {
                            const std::uint32_t vi =
                                xm.indexArray[t * 3u +
                                    static_cast<std::uint32_t>(k)];
                            if (vi >= xm.vertexCount) { triOk = false; break; }
                            const auto& xv = xm.vertexArray[vi];
                            if (xv.atlasIndex < 0 ||
                                xv.xref >= mesh.vertices.size()) {
                                triOk = false; break;
                            }
                        }
                        if (!triOk) { ++dropped; continue; }

                        for (int k = 0; k < 3; ++k) {
                            const std::uint32_t vi =
                                xm.indexArray[t * 3u +
                                    static_cast<std::uint32_t>(k)];
                            const auto& xv = xm.vertexArray[vi];
                            UvVertex uvv;
                            uvv.base = mesh.vertices[xv.xref];

                            // xatlas UVs are in atlas texel coordinates. We
                            // bake the image using the same coordinates, so no
                            // hidden Y flip is needed. glTF 2.0 defines (0,0)
                            // as image upper-left.
                            uvv.u = std::clamp(
                                xv.uv[0] * invW, 0.0f, 1.0f);
                            uvv.v = std::clamp(
                                xv.uv[1] * invH, 0.0f, 1.0f);
                            out.vertices.push_back(uvv);
                            out.indices.push_back(
                                static_cast<std::uint32_t>(
                                    out.indices.size()));
                        }
                    }

                    if (dropped > 0) {
                        UW_LOGW("uv_unwrap: xatlas dropped %u degenerate/"
                                "orphan tris of %u (kept %zu)",
                                dropped, triCount, out.vertices.size() / 3u);
                    }

                    if (out.empty()) {
                        UW_LOGW("uv_unwrap: xatlas kept 0 tris, "
                                "falling back to triangle atlas");
                    } else {
                        out.atlasWidth =
                            static_cast<int>(atlas->width);
                        out.atlasHeight =
                            static_cast<int>(atlas->height);

                        localStats.usedXatlas = true;
                        localStats.atlasWidth = out.atlasWidth;
                        localStats.atlasHeight = out.atlasHeight;
                        localStats.outputVertices = out.vertices.size();
                        localStats.chartCount = atlas->chartCount;

                        if (stats) *stats = localStats;
                        xatlas::Destroy(atlas);
                        return true;
                    }
                }
            }
        }

        xatlas::Destroy(atlas);
    }
#endif

    // Always retain a dependency-free fallback. It is intentionally less
    // efficient than xatlas (one UV chart per triangle), but guarantees a
    // standards-compliant textured asset if the third-party source was not
    // fetched on a developer machine.
    const bool ok = fallbackTriangleAtlas(
        mesh,
        resolution,
        padding,
        out,
        &localStats);

    if (stats) *stats = localStats;
    return ok;
}

bool UvUnwrapper::fallbackTriangleAtlas(
        const AosMesh& mesh,
        int resolution,
        int padding,
        UvMesh& out,
        UvUnwrapStats* stats) {
    const std::size_t triangles = mesh.triangleCount();
    if (triangles == 0) return false;

    const int grid =
        static_cast<int>(
            std::ceil(
                std::sqrt(
                    static_cast<double>(triangles))));
    const int cell =
        std::max(4, resolution / std::max(1, grid));
    const int actual =
        cell * std::max(1, grid);

    if (actual > 8192) return false;

    out.clear();
    out.vertices.reserve(triangles * 3u);
    out.indices.reserve(triangles * 3u);
    out.atlasWidth = actual;
    out.atlasHeight = actual;

    const float pad =
        static_cast<float>(
            std::clamp(padding, 1, std::max(1, cell / 4)));
    const float inv =
        1.0f / static_cast<float>(actual);

    for (std::size_t t = 0; t < triangles; ++t) {
        const int gx =
            static_cast<int>(t % static_cast<std::size_t>(grid));
        const int gy =
            static_cast<int>(t / static_cast<std::size_t>(grid));

        const float x0 =
            static_cast<float>(gx * cell) + pad;
        const float y0 =
            static_cast<float>(gy * cell) + pad;
        const float x1 =
            static_cast<float>((gx + 1) * cell) - pad;
        const float y1 =
            static_cast<float>((gy + 1) * cell) - pad;

        const std::uint32_t base =
            static_cast<std::uint32_t>(out.vertices.size());

        for (int k = 0; k < 3; ++k) {
            const auto src =
                mesh.indices[t*3u + static_cast<std::size_t>(k)];
            if (src >= mesh.vertices.size()) return false;

            UvVertex v;
            v.base = mesh.vertices[src];
            if (k == 0) {
                v.u = x0 * inv;
                v.v = y0 * inv;
            } else if (k == 1) {
                v.u = x1 * inv;
                v.v = y0 * inv;
            } else {
                v.u = x0 * inv;
                v.v = y1 * inv;
            }
            out.vertices.push_back(v);
            out.indices.push_back(base + static_cast<std::uint32_t>(k));
        }
    }

    if (stats) {
        stats->usedXatlas = false;
        stats->atlasWidth = out.atlasWidth;
        stats->atlasHeight = out.atlasHeight;
        stats->outputVertices = out.vertices.size();
        stats->chartCount = triangles;
    }

    return !out.empty();
}
