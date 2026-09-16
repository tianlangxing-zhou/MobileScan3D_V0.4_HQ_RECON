package com.mobilescan3d.export

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.mobilescan3d.NativeBridge
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.Executors

/**
 * 导出：PLY（兼容旧路径）与 **GLB（glTF 2.0，vertex color）**，以及网格拉取。
 *
 * ## 为什么 PLY 不再是最终结果
 *
 * 旧链路的终点是 `nativeExportPly()`，而它只写 `element vertex` —— 没有
 * `element face` 的 PLY 在 Blender / Unity / AR 里就是一堆孤立的点，
 * **不是模型**。评审把这一条列为 P0：「PLY 只有 vertex，没有 triangle faces」。
 *
 * 现在最终产物是 `scan_<session>.glb`：带索引三角面 + 逐顶点法线 + 逐顶点颜色。
 * PLY 保留（改名带 `_points` 后缀更诚实，但为了不破坏既有报告字段仍沿用
 * `scan_<session>_debug.ply`），只作为诊断产物。
 *
 * ## 线程
 *
 * 网格构建（Marching Tetrahedra + 清理 + QEM 简化）可能几百毫秒到数秒，
 * **绝不能放在 UI 线程**（ANR）。所以本类持有一个单线程 executor，
 * 所有构建 / 导出都跑在上面，回调再 post 回主线程。
 *
 * 注意：`nativeBuildMesh` 在 native 侧持 `gStateMutex`，构建期间相机帧
 * 融合与 AR pose 查询会一起被挡住 —— 这是为了保证「构建时体素场不被并发
 * 修改」（`TsdfEngine` 的 block 哈希表在插入时会 rehash，并发读会踩空指针）。
 * 表现为预览短暂卡住一下，可以接受；换成无锁快照要改数据结构，风险更高。
 */
class ExportManager(context: Context) {

    /** native 拉下来的网格数据，长度均已裁剪到实际数量。 */
    data class MeshData(
        val vertices: FloatArray,
        val vertexCount: Int,
        val indices: IntArray,
        val indexCount: Int
    ) {
        val triangleCount: Int get() = indexCount / 3
    }

    data class PlyResult(
        val ok: Boolean,
        val file: File,
        val vertexCount: Long,
        val fileBytes: Long,
        val message: String
    )

    data class GlbResult(
        val ok: Boolean,
        val file: File,
        val vertices: Int,
        val triangles: Int,
        val fileBytes: Long,
        val quality: Int,
        val message: String
    )

    private val outputDir: File = context.getExternalFilesDir(null) ?: context.filesDir

    private val io = Executors.newSingleThreadExecutor { r ->
        Thread(r, "MeshExport").apply { isDaemon = true }
    }

    private val main = Handler(Looper.getMainLooper())

    /** 上一次构建得到的网格（供 AR 预览复用，不必重复建）。 */
    @Volatile
    var lastMesh: MeshData? = null
        private set

    val dir: File get() = outputDir

    // ------------------------------------------------------------------ PLY

    /**
     * 导出点云 PLY。native 侧只写顶点，**没有三角面**，所以它只是中间产物。
     */
    fun exportPly(sessionId: String): PlyResult {
        // V0.5：正式产物是 scan_<session>.glb；PLY 只是诊断用的点云 dump。
        val file = File(outputDir, "scan_${sessionId}_debug.ply")
        val ok = try {
            NativeBridge.nativeExportPly(file.absolutePath)
        } catch (t: Throwable) {
            Log.e(TAG, "nativeExportPly failed", t)
            false
        }
        val verts = if (ok) readPlyVertexCount(file) else 0L
        return PlyResult(
            ok = ok,
            file = file,
            vertexCount = verts,
            fileBytes = if (ok) file.length() else 0L,
            message = if (ok) "调试点云已导出：${file.absolutePath}" else "调试点云导出失败（数据不足）"
        )
    }

    /**
     * 只读 PLY 头部解析 `element vertex N`。
     *
     * 用 RandomAccessFile 读头而不是整文件读 —— HQ 点云可以到几十 MB，
     * 为了一个数字把整个文件读进内存不值得。
     */
    private fun readPlyVertexCount(file: File): Long {
        return try {
            RandomAccessFile(file, "r").use { f ->
                var line = 0
                while (line < 64) {
                    val s = f.readLine() ?: break
                    ++line
                    if (s.startsWith("element vertex ")) {
                        return s.removePrefix("element vertex ").trim().toLongOrNull() ?: 0L
                    }
                    if (s.trim() == "end_header") break
                }
                0L
            }
        } catch (t: Throwable) {
            Log.w(TAG, "readPlyVertexCount failed", t)
            0L
        }
    }

    // ------------------------------------------------------------------ 网格

    /**
     * 构建网格（同一线程同步执行）。成功返回 true。
     *
     * @param quality [NativeBridge.MESH_QUALITY_PREVIEW] / `_NORMAL` / `_HQ`
     */
    fun buildMesh(quality: Int): Boolean {
        return try {
            val q = quality.coerceIn(0, 2)
            val ok = NativeBridge.nativeBuildMesh(q)
            if (ok) {
                lastMesh = pullMesh()
            }
            // 必须显式返回；只写 `if (ok) {...}` 会让 try 块的值推断成 Any，
            // 与函数声明的 Boolean 冲突（Kotlin 不把 if-without-else 当表达式）。
            ok
        } catch (t: Throwable) {
            Log.e(TAG, "nativeBuildMesh failed", t)
            false
        }
    }

    /**
     * 把 native 的网格数据拉成 Kotlin 数组。
     *
     * 先查数量再分配，避免「按最大容量分配、返回 0 个」那种每次几 MB 的浪费。
     * 返回的数组按实际数量裁剪。
     */
    fun pullMesh(): MeshData? {
        return try {
            val vcount = NativeBridge.nativeGetMeshVertexCount()
            val icount = NativeBridge.nativeGetMeshIndexCount()
            if (vcount <= 0 || icount < 3) return null

            val vraw = FloatArray(vcount * NativeBridge.MESH_VERTEX_FLOATS)
            val got = NativeBridge.nativeGetMeshVertices(vraw, vcount)
            if (got <= 0) return null

            val iraw = IntArray(icount)
            val gotIdx = NativeBridge.nativeGetMeshIndices(iraw, icount)
            if (gotIdx < 3) return null

            val tris = gotIdx / 3
            val vUsed = got
            MeshData(
                vertices = if (vraw.size == vUsed * NativeBridge.MESH_VERTEX_FLOATS) vraw
                else vraw.copyOf(vUsed * NativeBridge.MESH_VERTEX_FLOATS),
                vertexCount = vUsed,
                indices = if (iraw.size == tris * 3) iraw else iraw.copyOf(tris * 3),
                indexCount = tris * 3
            )
        } catch (t: Throwable) {
            Log.e(TAG, "pullMesh failed", t)
            null
        }
    }

    /** 网格统计（16 槽，下标含义见 [NativeBridge]）。失败返回空数组。 */
    fun meshStats(): IntArray = try {
        NativeBridge.nativeGetMeshStats()
    } catch (t: Throwable) {
        IntArray(0)
    }

    fun resetMesh() {
        try {
            NativeBridge.nativeResetMesh()
        } catch (t: Throwable) {
            Log.w(TAG, "nativeResetMesh failed", t)
        }
        lastMesh = null
    }

    // ------------------------------------------------------------------ GLB

    /**
     * 后台构建网格并导出 GLB（vertex color）。回调在主线程。
     *
     * 这是**扫描结束时的默认动作**：一次会话结束后端到端地给出可用的 AR 模型。
     */
    fun buildAndExportGlb(sessionId: String, quality: Int, onDone: (GlbResult) -> Unit) {
        val q = quality.coerceIn(0, 2)
        io.execute {
            val file = File(outputDir, "scan_$sessionId.glb")
            var verts = 0
            var tris = 0
            var ok = false
            var msg = ""
            try {
                ok = NativeBridge.nativeBuildMesh(q)
                if (ok) {
                    val mesh = pullMesh()
                    lastMesh = mesh
                    verts = mesh?.vertexCount ?: 0
                    tris = mesh?.triangleCount ?: 0
                    ok = verts > 0 && tris > 0
                }
                if (ok) {
                    ok = NativeBridge.nativeExportGlb(file.absolutePath)
                }
                msg = when {
                    !ok && verts == 0 -> "网格构建失败（体素场不足以提取表面）"
                    !ok -> "GLB 写入失败"
                    else -> "网格已导出：${file.absolutePath}"
                }
            } catch (t: Throwable) {
                Log.e(TAG, "buildAndExportGlb failed", t)
                ok = false
                msg = "网格导出异常：${t.javaClass.simpleName}: ${t.message}"
            }
            val res = GlbResult(
                ok = ok && file.exists() && file.length() > 0,
                file = file,
                vertices = verts,
                triangles = tris,
                fileBytes = if (file.exists()) file.length() else 0L,
                quality = q,
                message = msg
            )
            main.post { onDone(res) }
        }
    }

    /**
     * 后台构建网格并回传数据（供 AR 预览），不落盘。
     */
    fun buildMeshAsync(quality: Int, onDone: (MeshData?) -> Unit) {
        val q = quality.coerceIn(0, 2)
        io.execute {
            val mesh = try {
                if (NativeBridge.nativeBuildMesh(q)) pullMesh() else null
            } catch (t: Throwable) {
                Log.e(TAG, "buildMeshAsync failed", t)
                null
            }
            lastMesh = mesh
            main.post { onDone(mesh) }
        }
    }

    fun shutdown() {
        try {
            io.shutdownNow()
        } catch (_: Throwable) {
        }
    }

    companion object {
        private const val TAG = "ExportManager"
    }
}
