package com.mobilescan3d.mesh

import kotlin.math.max
import kotlin.math.min

/**
 * CPU-side topology inspection for the 3D viewer.
 *
 * A boundary edge belongs to exactly one triangle. Highlighting these edges is
 * materially more useful than a generic "quality %" because it shows actual open
 * topology in the generated mesh.
 *
 * Analysis is capped for very large meshes to avoid a multi-hundred-megabyte
 * temporary edge array. In that case dimensions are still exact, while boundary
 * analysis reports "too large" instead of pretending to be complete.
 */
object MeshInspectionAnalyzer {

    data class Result(
        val width: Float,
        val height: Float,
        val depth: Float,
        val boundaryEdges: IntArray,
        val boundaryEdgeCount: Int,
        val nonManifoldEdgeCount: Int,
        val topologyComplete: Boolean
    )

    private const val VERTEX_FLOATS = 9
    private const val MAX_TOPOLOGY_TRIANGLES = 900_000

    fun analyze(vertices: FloatArray, indices: IntArray): Result {
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var maxZ = -Float.MAX_VALUE

        var i = 0
        while (i + 2 < vertices.size) {
            val x = vertices[i]
            val y = vertices[i + 1]
            val z = vertices[i + 2]
            if (x.isFinite() && y.isFinite() && z.isFinite()) {
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                minZ = min(minZ, z); maxZ = max(maxZ, z)
            }
            i += VERTEX_FLOATS
        }

        val width = if (minX <= maxX) maxX - minX else 0f
        val height = if (minY <= maxY) maxY - minY else 0f
        val depth = if (minZ <= maxZ) maxZ - minZ else 0f

        val triCount = indices.size / 3
        if (triCount <= 0 || triCount > MAX_TOPOLOGY_TRIANGLES) {
            return Result(width, height, depth, IntArray(0), 0, 0, false)
        }

        // Encode undirected edge as (minVertex << 32) | maxVertex.
        val edges = LongArray(triCount * 3)
        var ep = 0
        for (t in 0 until triCount) {
            val a = indices[t * 3]
            val b = indices[t * 3 + 1]
            val c = indices[t * 3 + 2]
            edges[ep++] = edgeKey(a, b)
            edges[ep++] = edgeKey(b, c)
            edges[ep++] = edgeKey(c, a)
        }
        java.util.Arrays.sort(edges)

        var boundaryCount = 0
        var nonManifold = 0
        i = 0
        while (i < edges.size) {
            val key = edges[i]
            var j = i + 1
            while (j < edges.size && edges[j] == key) j++
            val count = j - i
            if (count == 1) boundaryCount++
            if (count > 2) nonManifold++
            i = j
        }

        val boundary = IntArray(boundaryCount * 2)
        var bp = 0
        i = 0
        while (i < edges.size) {
            val key = edges[i]
            var j = i + 1
            while (j < edges.size && edges[j] == key) j++
            if (j - i == 1) {
                boundary[bp++] = (key ushr 32).toInt()
                boundary[bp++] = (key and 0xffffffffL).toInt()
            }
            i = j
        }

        return Result(
            width, height, depth,
            boundary,
            boundaryCount,
            nonManifold,
            true
        )
    }

    private fun edgeKey(a: Int, b: Int): Long {
        val lo = min(a, b)
        val hi = max(a, b)
        return (lo.toLong() shl 32) or (hi.toLong() and 0xffffffffL)
    }
}
