package com.mobilescan3d.export

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.mobilescan3d.NativeBridge
import com.mobilescan3d.persistence.ScanPackageManager
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.Executors

/**
 * 导出：PLY（兼容旧路径）与 **GLB（glTF 2.0）**，以及网格拉取。
 *
 * V0.6 起 GLB 是 **textured-first**：优先内嵌 JPEG atlas 的 HQ 纹理模型，
 * 失败回退逐顶点颜色的版本（见 [buildAndExportGlb]）。
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
        val message: String,
        /** V0.6：是否带 HQ 纹理（内嵌 JPEG atlas）。false = 退回 vertex color。 */
        val textured: Boolean = false,
        val persistenceMessage: String? = null
    )

    private val outputDir: File = context.getExternalFilesDir(null) ?: context.filesDir
    private val appContext = context.applicationContext

    private val io = Executors.newSingleThreadExecutor { r ->
        Thread(r, "MeshExport").apply { isDaemon = true }
    }

    private val main = Handler(Looper.getMainLooper())

    /** 上一次构建得到的网格（供 AR 预览复用，不必重复建）。 */
    @Volatile
    var lastMesh: MeshData? = null
        private set

    private val meshGeneration = java.util.concurrent.atomic.AtomicLong(0)
    private val meshCacheLock = Any()

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

    fun exportPlyAsync(sessionId: String, onDone: (PlyResult) -> Unit) {
        val generation = meshGeneration.get()
        io.execute {
            if (generation != meshGeneration.get()) return@execute
            val result = exportPly(sessionId)
            main.post { if (generation == meshGeneration.get()) onDone(result) }
        }
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
            lastMesh = if (ok) pullMesh() else null
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

    /** Round 8: adopt the mesh currently held by native (e.g. after ICP merge). */
    fun refreshCurrentMesh(): MeshData? {
        val mesh = pullMesh()
        synchronized(meshCacheLock) { lastMesh = mesh }
        return mesh
    }

    /**
     * Round 6: pull actual TSDF observation strength for the currently built mesh.
     * Values are accumulated TSDF confidence weights, not camera-view estimates.
     */
    fun meshObservationWeights(vertexCount: Int): FloatArray {
        if (vertexCount <= 0) return FloatArray(0)
        return try {
            val out = FloatArray(vertexCount)
            val got = NativeBridge.nativeGetMeshObservationWeights(out, vertexCount)
            if (got <= 0) FloatArray(0)
            else if (got == out.size) out else out.copyOf(got)
        } catch (t: Throwable) {
            Log.w(TAG, "meshObservationWeights failed", t)
            FloatArray(0)
        }
    }

    /** 网格统计（16 槽，下标含义见 [NativeBridge]）。失败返回空数组。 */
    fun meshStats(): IntArray = try {
        NativeBridge.nativeGetMeshStats()
    } catch (t: Throwable) {
        IntArray(0)
    }

    /** V0.6：导出前几何清理统计（[NativeBridge.MESH_CLEANUP_STATS_SLOTS] 槽）。 */
    fun meshCleanupStats(): IntArray = try {
        val buf = IntArray(NativeBridge.MESH_CLEANUP_STATS_SLOTS)
        if (NativeBridge.nativeGetMeshCleanupStats(buf)) buf else IntArray(0)
    } catch (t: Throwable) {
        IntArray(0)
    }

    /** V0.6：HQ 纹理烘焙统计（[NativeBridge.TEXTURE_STATS_SLOTS] 槽）。 */
    fun textureStats(): IntArray = try {
        val buf = IntArray(NativeBridge.TEXTURE_STATS_SLOTS)
        if (NativeBridge.nativeGetTextureStats(buf)) buf else IntArray(0)
    } catch (t: Throwable) {
        IntArray(0)
    }

    fun resetMesh() {
        synchronized(meshCacheLock) {
            meshGeneration.incrementAndGet()
            lastMesh = null
        }
        try {
            NativeBridge.nativeResetMesh()
        } catch (t: Throwable) {
            Log.w(TAG, "nativeResetMesh failed", t)
        }
        lastMesh = null
    }

    // ------------------------------------------------------------------ GLB

    /**
     * Round 8: export the current native mesh WITHOUT rebuilding from the current
     * TSDF. Required for cumulative multi-session meshes, because rebuilding would
     * silently throw away already aligned earlier segments.
     *
     * Merged sessions currently export vertex-color GLB. Cross-session texture
     * rebaking needs all source keyframes in one coordinate frame and is deliberately
     * not faked here.
     */
    fun exportCurrentMeshGlbAsync(sessionId: String, onDone: (GlbResult) -> Unit) {
        val generation = meshGeneration.get()
        io.execute {
            val file = File(outputDir, "scan_${sessionId}_merged.glb")
            val staging = File(outputDir, "scan_${sessionId}_merged.pending.glb")
            var ok = false
            var mesh: MeshData? = null
            var message = ""
            try {
                if (generation != meshGeneration.get()) return@execute
                staging.delete()
                mesh = pullMesh()
                ok = mesh != null && mesh!!.triangleCount > 0 &&
                    NativeBridge.nativeExportGlb(staging.absolutePath)
                if (generation != meshGeneration.get()) return@execute
                ok = ok && staging.isFile && staging.length() > 0L && staging.renameTo(file)
                if (ok) {
                    synchronized(meshCacheLock) { lastMesh = mesh }
                    message = "分段拼接 GLB 已导出：${file.absolutePath}"
                } else {
                    message = "分段拼接 GLB 导出失败"
                }
            } catch (t: Throwable) {
                Log.e(TAG, "exportCurrentMeshGlbAsync failed", t)
                ok = false
                message = "分段拼接导出异常：${t.message ?: t.javaClass.simpleName}"
            } finally {
                staging.delete()
            }
            val m = mesh
            val result = GlbResult(
                ok = ok,
                file = file,
                vertices = m?.vertexCount ?: 0,
                triangles = m?.triangleCount ?: 0,
                fileBytes = if (ok && file.isFile) file.length() else 0L,
                quality = NativeBridge.MESH_QUALITY_HQ,
                message = message,
                textured = false,
                persistenceMessage = "分段模型使用 vertex color；未伪造跨会话纹理重投影"
            )
            main.post { if (generation == meshGeneration.get()) onDone(result) }
        }
    }

    /**
     * 后台构建网格并导出 GLB。回调在主线程。
     *
     * **V0.6 起是 textured-first**：先试 nativeBakeTexturedGlb（几何清理 + UV 展开
     * + HQ 多视角纹理 + 内嵌 JPEG atlas 的自包含 GLB），失败才退回 nativeExportGlb
     * 的逐顶点颜色版本。
     *
     * 这是**扫描结束时的默认动作**：一次会话结束后端到端地给出可用的 AR 模型。
     */
    fun buildAndExportGlb(sessionId: String, quality: Int, shape: Int = 0, onDone: (GlbResult) -> Unit) {
        val q = quality.coerceIn(0, 2)
        val shapeMode = shape.coerceIn(0, 3)
        val suffix = arrayOf("", "_planar", "_cuboid", "_cube")[shapeMode]
        val exportId = sessionId + suffix
        val generation = meshGeneration.get()
        io.execute {
            fun checkGeneration() {
                if (generation != meshGeneration.get()) {
                    throw java.util.concurrent.CancellationException("Scan session ended")
                }
            }
            val file = File(outputDir, "scan_$exportId.glb")
            val staging = File(outputDir, "scan_$exportId.pending.glb")
            var verts = 0
            var tris = 0
            var ok = false
            var textured = false
            var persistenceMessage: String? = null
            var msg = ""
            var shapeReport = ""
            try {
                checkGeneration()
                staging.delete()
                lastMesh = null
                // Never reuse an earlier bake when this export falls back to vertex color.
                NativeBridge.nativeClearTexturedArAsset()
                ok = NativeBridge.nativeBuildMeshWithShape(q, shapeMode)
                checkGeneration()
                shapeReport = if (shapeMode > 0) NativeBridge.nativeGetHardSurfaceReport() else ""
                if (ok) {
                    val mesh = pullMesh()
                    lastMesh = mesh
                    verts = mesh?.vertexCount ?: 0
                    tris = mesh?.triangleCount ?: 0
                    ok = verts > 0 && tris > 0
                }
                if (ok) {
                    // V0.6：优先烘焙 HQ 多视角纹理，产出内嵌 JPEG atlas 的自包含
                    // GLB。清理（weld / 去漂浮分量 / ear-clipping 补小洞 / QEM）
                    // **只作用在这份导出副本上** —— 屏幕上的 AR overlay 仍是 V0.5
                    // 的 vertex color 网格，由 MeshEngine 自己那条链产生。
                    textured = try {
                        NativeBridge.nativeBakeTexturedGlb(
                            staging.absolutePath,
                            ATLAS_RESOLUTION,
                            MAX_TEXTURE_KEYFRAMES
                        )
                    } catch (t: Throwable) {
                        Log.w(TAG, "nativeBakeTexturedGlb failed", t)
                        false
                    }
                    // JNI work is not interrupted by shutdownNow(). Do not let
                    // a late bake fall back to/export another session's mesh.
                    checkGeneration()
                    // 纹理失败不能连几何一起判废 —— 退回 V0.5 的 vertex color GLB。
                    ok = textured ||
                        try {
                            NativeBridge.nativeExportGlb(staging.absolutePath)
                        } catch (t: Throwable) {
                            Log.w(TAG, "nativeExportGlb failed", t)
                            false
                        }
                }
                // V0.13.7：纹理未生效时把「已登记关键帧数」打进日志，
                // 下一次真机扫描的 logcat 就能直接看出是「没采到关键帧」还是「烘焙失败」。
                if (ok && !textured) {
                    val ts = textureStats()
                    val rk = if (ts.size > NativeBridge.TEXTURE_STATS_INDEX_REGISTERED_KEYFRAMES)
                        ts[NativeBridge.TEXTURE_STATS_INDEX_REGISTERED_KEYFRAMES] else -1
                    Log.w(TAG, "buildAndExportGlb: vertex-color fallback — " +
                        "registeredKeyframes=$rk (texture bake skipped/failed). " +
                        "registeredKeyframes=0 说明扫描时未采到关键帧（VINS 未初始化或 burst 全失败）；" +
                        ">0 说明关键帧已登记但烘焙未产出可用视角。")
                }
                checkGeneration()
                ok = ok && staging.isFile && staging.length() > 0L && staging.renameTo(file)
                if (ok && textured) {
                    val baked = NativeBridge.nativeGetTexturedArAssetStats()
                    if (baked.size >= 2 && baked[0] > 0 && baked[1] >= 3) {
                        verts = baked[0]
                        tris = baked[1] / 3
                    }
                    // Save before the completion callback releases the UI operation guard.
                    persistenceMessage = ScanPackageManager.saveCurrent(
                        appContext, exportId, file
                    ).message
                }
                msg = when {
                    !ok && verts == 0 -> "网格构建失败（体素场不足以提取表面）"
                    !ok -> "GLB 写入失败"
                    textured -> "HQ 纹理模型已导出：${file.absolutePath}"
                    else -> "网格已导出（vertex color；纹理烘焙未生效）：${file.absolutePath}"
                }
                if (shapeReport.isNotBlank()) msg += "\n$shapeReport"
                // V0.13.9：烘焙诊断持久化。logcat 主缓冲区被相机 HAL 噪音冲刷极快，
                // bakeDiag 行常在取证前丢失；改为落盘到 GLB 同目录，adb pull 随时可取。
                if (ok) {
                    try {
                        val ts = textureStats()
                        val cs = meshCleanupStats()
                        val diag = org.json.JSONObject().apply {
                            put("sessionId", sessionId)
                            put("shapeMode", shapeMode)
                            put("shapeReport", shapeReport)
                            put("ts", System.currentTimeMillis())
                            put("textured", textured)
                            put("vertices", verts)
                            put("triangles", tris)
                            put("fileBytes", file.length())
                            put("message", msg)
                            put("registeredKeyframes", if (ts.size > 0) ts[0] else -1)
                            put("loadedKeyframes", if (ts.size > 1) ts[1] else -1)
                            put("usedKeyframes", if (ts.size > 2) ts[2] else -1)
                            put("atlasW", if (ts.size > 3) ts[3] else -1)
                            put("atlasH", if (ts.size > 4) ts[4] else -1)
                            put("texturedTriangles", if (ts.size > 5) ts[5] else -1)
                            put("fallbackTriangles", if (ts.size > 6) ts[6] else -1)
                            put("coveragePercent", if (ts.size > 7) ts[7] / 10f else -1f)
                            put("usedXatlas", if (ts.size > 8) ts[8] == 1 else false)
                            put("meshCleanupStats", org.json.JSONArray(cs.toList()))
                        }
                        val diagFile = File(file.parent, file.nameWithoutExtension + ".bakeDiag.json")
                        diagFile.writeText(diag.toString(2))
                    } catch (t: Throwable) {
                        Log.w(TAG, "write bakeDiag.json failed", t)
                    }
                }
            } catch (_: java.util.concurrent.CancellationException) {
                return@execute
            } catch (t: Throwable) {
                Log.e(TAG, "buildAndExportGlb failed", t)
                ok = false
                msg = "网格导出异常：${t.javaClass.simpleName}: ${t.message}"
            } finally {
                staging.delete()
            }
            val res = GlbResult(
                ok = ok && file.exists() && file.length() > 0,
                file = file,
                vertices = verts,
                triangles = tris,
                fileBytes = if (file.exists()) file.length() else 0L,
                quality = q,
                message = msg,
                textured = ok && textured,
                persistenceMessage = persistenceMessage
            )
            main.post { if (generation == meshGeneration.get()) onDone(res) }
        }
    }

    /**
     * 后台构建网格并回传数据（供 AR 预览），不落盘。
     */
    fun buildMeshAsync(quality: Int, onDone: (MeshData?) -> Unit) {
        val q = quality.coerceIn(0, 2)
        val generation = meshGeneration.get()
        io.execute {
            if (generation != meshGeneration.get()) return@execute
            val mesh = try {
                if (NativeBridge.nativeBuildMesh(q)) pullMesh() else null
            } catch (t: Throwable) {
                Log.e(TAG, "buildMeshAsync failed", t)
                null
            }
            synchronized(meshCacheLock) {
                if (generation == meshGeneration.get()) lastMesh = mesh
            }
            main.post {
                if (generation == meshGeneration.get()) onDone(mesh)
            }
        }
    }

    fun shutdown() {
        synchronized(meshCacheLock) {
            meshGeneration.incrementAndGet()
            lastMesh = null
        }
        try {
            io.shutdownNow()
        } catch (_: Throwable) {
        }
    }

    companion object {
        private const val TAG = "ExportManager"

        /** V0.13.19.5：纹理 atlas 边长 2K->4K。V0.13.19.4 修复 epoch 挂起后几何密度涨 ~5x，2K atlas 像素预算被撑爆（覆盖掉到 44.5%）；4K 面积 x4，且刚好命中 native 的 resolution>=4096 高质量档（sourceMaxSide 2200 / gutter 10 / q94）。4096 是 native 上限（clamp 512..4096），内存/画质折中安全。 */
        private const val ATLAS_RESOLUTION = 4096

        /** V0.6：最多挑多少个 HQ 视角做烘焙（native 侧还会按质量+视角多样性再筛）。 */
        private const val MAX_TEXTURE_KEYFRAMES = 12
    }
}
