package com.mobilescan3d.multiscan

import android.content.Context
import com.mobilescan3d.NativeBridge
import com.mobilescan3d.export.ExportManager
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/**
 * Round 8 cumulative segment store.
 *
 * One persisted reference mesh is enough for an arbitrary number of sessions:
 * after every successful ICP merge the cumulative result replaces the reference.
 */
class ScanSegmentStore(context: Context) {
    data class Info(
        val sessionId: String,
        val createdAtMs: Long,
        val vertexCount: Int,
        val triangleCount: Int,
        val segmentCount: Int
    )

    data class Base(val mesh: ExportManager.MeshData, val info: Info)

    private val dir = File(context.filesDir, "multiscan_r8").apply { mkdirs() }
    private val bin = File(dir, "cumulative.meshbin")
    private val meta = File(dir, "cumulative.json")

    fun info(): Info? {
        return try {
            if (!bin.isFile || !meta.isFile) return null
            val j = JSONObject(meta.readText(Charsets.UTF_8))
            Info(
                sessionId = j.optString("sessionId", "segment"),
                createdAtMs = j.optLong("createdAtMs", meta.lastModified()),
                vertexCount = j.optInt("vertexCount", 0),
                triangleCount = j.optInt("triangleCount", 0),
                segmentCount = j.optInt("segmentCount", 1).coerceAtLeast(1)
            )
        } catch (_: Throwable) { null }
    }

    fun load(): Base? {
        return try {
            val inf = info() ?: return null
            DataInputStream(BufferedInputStream(bin.inputStream(), 256 * 1024)).use { input ->
                require(input.readInt() == MAGIC) { "bad segment magic" }
                require(input.readInt() == VERSION) { "unsupported segment version" }
                val vc = input.readInt()
                val ic = input.readInt()
                require(vc in 1..2_000_000 && ic in 3..6_000_000 && ic % 3 == 0)
                val verts = FloatArray(vc * NativeBridge.MESH_VERTEX_FLOATS) { input.readFloat() }
                val idx = IntArray(ic) { input.readInt() }
                require(idx.all { it in 0 until vc })
                Base(ExportManager.MeshData(verts, vc, idx, ic), inf)
            }
        } catch (_: Throwable) { null }
    }

    fun save(mesh: ExportManager.MeshData, sessionId: String, segmentCount: Int): Boolean {
        if (mesh.vertexCount !in 1..2_000_000 ||
            mesh.indexCount !in 3..6_000_000 || mesh.indexCount % 3 != 0 ||
            mesh.vertices.size < mesh.vertexCount * NativeBridge.MESH_VERTEX_FLOATS ||
            mesh.indices.size < mesh.indexCount) return false
        val tmp = File(dir, ".cumulative.tmp")
        return try {
            DataOutputStream(BufferedOutputStream(tmp.outputStream(), 256 * 1024)).use { out ->
                out.writeInt(MAGIC)
                out.writeInt(VERSION)
                out.writeInt(mesh.vertexCount)
                out.writeInt(mesh.indexCount)
                mesh.vertices.forEach(out::writeFloat)
                mesh.indices.forEach(out::writeInt)
            }
            if (bin.exists()) bin.delete()
            check(tmp.renameTo(bin))
            val now = System.currentTimeMillis()
            meta.writeText(
                JSONObject()
                    .put("version", VERSION)
                    .put("sessionId", sessionId)
                    .put("createdAtMs", now)
                    .put("vertexCount", mesh.vertexCount)
                    .put("triangleCount", mesh.triangleCount)
                    .put("segmentCount", segmentCount.coerceAtLeast(1))
                    .toString(2),
                Charsets.UTF_8
            )
            true
        } catch (_: Throwable) {
            tmp.delete()
            false
        }
    }

    fun clear() {
        bin.delete()
        meta.delete()
        File(dir, ".cumulative.tmp").delete()
    }

    companion object {
        private const val MAGIC = 0x4D533853 // MS8S
        private const val VERSION = 1
    }
}
