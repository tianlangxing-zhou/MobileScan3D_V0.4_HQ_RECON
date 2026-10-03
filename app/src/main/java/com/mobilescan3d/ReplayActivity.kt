package com.mobilescan3d

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.widget.TextView
import com.mobilescan3d.depth.DepthRepresentation
import com.mobilescan3d.depth.MonoDepthProvider
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 离线视频重放桥（vc18520 调试入口）。
 *
 * 输入目录: getExternalFilesDir(null)/replay/
 *   frames/f0000.jpg..       解码后的相机帧 (1280x720, 30fps)
 *   imu.txt                  合成 IMU (t_ns ax ay az gx gy gz, ~100Hz)
 *   trajectory.txt           SfM 轨迹 (仅诊断对比, 引擎位姿由 VINS 自己估计)
 *   config.json              {"fx":914,"fy":914,"cx":640,"cy":360,
 *                             "w":1280,"h":720,"depthEvery":1,"meshQuality":1}
 *
 * 流程完全镜像 MainActivity 实时链路:
 *   nativeCreate -> 每帧 [nativeOnImu 批量] -> nativeOnCameraFrame ->
 *   nativeRetainDepthFrame -> MonoDepthProvider.submitFrame ->
 *   nativeOnDepthMapWeighted -> 结束 nativeBuildMesh -> nativeExportPly
 *
 * 输出: replay/replay_out.ply + replay/replay_report.txt
 *
 * 运行方式（宿主）:
 *   adb push frames/ /sdcard/Android/data/com.mobilescan3d/files/replay/frames/
 *   adb push imu.txt config.json ...
 *   adb shell am start -n com.mobilescan3d/.ReplayActivity
 *
 * 注意：不需要相机权限/真实相机；全链路单线程串行（VINS 时序要求）。
 */
class ReplayActivity : Activity() {

    private lateinit var status: TextView
    private val worker = HandlerThread("ReplayWorker")
    private lateinit var workerHandler: Handler

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this)
        status.setPadding(48, 96, 48, 48)
        status.text = "Replay: idle"
        setContentView(status)
        worker.start()
        workerHandler = Handler(worker.looper)
        workerHandler.post { runReplay() }
    }

    override fun onDestroy() {
        worker.quitSafely()
        super.onDestroy()
    }

    private fun ui(text: String) {
        runOnUiThread { status.text = text }
        android.util.Log.i(TAG, text)
    }

    private fun runReplay() {
        val started = System.currentTimeMillis()
        try {
            val dir = File(getExternalFilesDir(null), "replay")
            val framesDir = File(dir, "frames")
            val cfg: JSONObject = File(dir, "config.json").let {
                if (it.exists()) JSONObject(it.readText()) else JSONObject()
            }
            val imuFile = File(dir, "imu.txt")
            if (!framesDir.isDirectory || !imuFile.exists()) {
                ui("MISSING inputs under $dir (frames/, imu.txt)")
                return
            }
            val frameFiles = framesDir.listFiles { f -> f.name.endsWith(".jpg") }
                ?.sortedBy { it.name } ?: emptyList()
            if (frameFiles.isEmpty()) {
                ui("MISSING frames under ${framesDir.path}")
                return
            }
            val first = BitmapFactory.decodeFile(frameFiles[0].absolutePath)
            val w = cfg.optInt("w", first.width)
            val h = cfg.optInt("h", first.height)
            val fx = cfg.optDouble("fx", w * 0.85).toFloat()
            val fy = cfg.optDouble("fy", h * 0.85).toFloat()
            val cx = cfg.optDouble("cx", w / 2.0).toFloat()
            val cy = cfg.optDouble("cy", h / 2.0).toFloat()
            val depthEvery = cfg.optInt("depthEvery", 1)
            val meshQuality = cfg.optInt("meshQuality", 1)
            first.recycle()

            // ---- IMU 全量读入 ----
            val imuTs = ArrayList<Long>(4096)
            val imuData = ArrayList<FloatArray>(4096)
            imuFile.readLines().forEach { line ->
                val p = line.trim().split(Regex("\\s+"))
                if (p.size >= 7) {
                    imuTs.add(p[0].toLong())
                    imuData.add(floatArrayOf(
                        p[1].toFloat(), p[2].toFloat(), p[3].toFloat(),
                        p[4].toFloat(), p[5].toFloat(), p[6].toFloat()))
                }
            }
            ui("replay start frames=${frameFiles.size} imu=${imuTs.size} K=[$fx,$fy,$cx,$cy]")

            // ---- native 引擎初始化（镜像 nativeCreate 实时路径）----
            if (!NativeBridge.nativeCreate(w, h, fx, fy, cx, cy)) {
                ui("nativeCreate FAILED")
                return
            }
            val depthProvider = MonoDepthProvider(assets)
            if (!depthProvider.available) ui("WARN depth model unavailable -> depth fusion disabled")

            val y = ByteArray(w * h)
            val u = ByteArray(w * h / 4)
            val v = ByteArray(w * h / 4)
            var imuIdx = 0
            var framesFed = 0
            var depthFed = 0
            var retainOk = 0
            var depthRejected = 0
            val t0 = System.nanoTime()

            for ((k, file) in frameFiles.withIndex()) {
                val bmp = BitmapFactory.decodeFile(file.absolutePath) ?: continue
                val bw = bmp.width
                val bh = bmp.height
                if (bw != w || bh != h) {
                    val scaled = Bitmap.createScaledBitmap(bmp, w, h, true)
                    if (scaled !== bmp) bmp.recycle()
                    rgbToI420(scaled, y, u, v)
                    scaled.recycle()
                } else {
                    rgbToI420(bmp, y, u, v)
                    bmp.recycle()
                }
                val ts = k * 1_000_000_000L / 30L + 1_000_000_000_000L
                // 1) 该帧之前的所有 IMU 样本
                while (imuIdx < imuTs.size && imuTs[imuIdx] <= ts) {
                    val d = imuData[imuIdx]
                    NativeBridge.nativeOnImu(imuTs[imuIdx], d[0], d[1], d[2], d[3], d[4], d[5])
                    imuIdx++
                }
                // 2) 相机帧 (I420: rowStride=w, uRowStride=w/2, uPixelStride=1)
                NativeBridge.nativeOnCameraFrame(y, u, v, w, h, w, w / 2, 1, ts, ts)
                framesFed++
                // 3) 深度推理 + 融合
                if (depthEvery > 0 && k % depthEvery == 0 && depthProvider.available) {
                    if (NativeBridge.nativeRetainDepthFrame(ts)) {
                        retainOk++
                        if (depthProvider.submitFrame(y, u, v, w, h, w, w / 2, 1, ts)) {
                            val res = depthProvider.latest()
                            if (res != null && res.depth.isNotEmpty()) {
                                NativeBridge.nativeOnDepthMapWeighted(
                                    res.depth, res.width, res.height,
                                    0.5f, res.timestampNs,
                                    res.representation.nativeCode, res.confidence)
                                depthFed++
                            }
                        }
                    } else {
                        depthRejected++
                    }
                }
                if (k % 20 == 0) {
                    val secs = (System.nanoTime() - t0) / 1_000_000_000.0
                    ui("frame $k/${frameFiles.size} (${("%.1f".format(secs))}s) "
                        + "vinsOk=${NativeBridge.nativeVinsInitialized()} "
                        + "pts=${NativeBridge.nativeGetPointCount()}")
                }
            }

            // ---- 建网格 + 导出 ----
            ui("building mesh (quality=$meshQuality) ...")
            val meshOk = NativeBridge.nativeBuildMesh(meshQuality)
            val meshStats = NativeBridge.nativeGetMeshStats()
            val ply = File(dir, "replay_out.ply")
            val exportOk = NativeBridge.nativeExportPly(ply.absolutePath)
            val stats = try { NativeBridge.nativeGetStats() } catch (t: Throwable) { "stats err: $t" }
            val vinsInit = NativeBridge.nativeVinsInitialized()
            val vinsEver = NativeBridge.nativeVinsEverInitialized()
            val pts = NativeBridge.nativeGetPointCount()

            val rep = buildString {
                appendLine("replay_report ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())}")
                appendLine("frames_fed=$framesFed total=${frameFiles.size}")
                appendLine("imu_fed=$imuIdx/${imuTs.size}")
                appendLine("depth_retained=$retainOk depth_fused=$depthFed depth_noSnap=$depthRejected")
                appendLine("vinsInitialized=$vinsInit vinsEverInitialized=$vinsEver")
                appendLine("pointCount=$pts")
                appendLine("meshOk=$meshOk meshStats=${meshStats?.contentToString()}")
                appendLine("exportPly=$exportOk path=${ply.absolutePath}")
                appendLine("wallMs=${System.currentTimeMillis() - started}")
                appendLine("---- nativeGetStats ----")
                appendLine(stats)
            }
            File(dir, "replay_report.txt").writeText(rep)
            ui("DONE pts=$pts meshOk=$meshOk export=$exportOk -> ${ply.absolutePath}")
            depthProvider.close()
        } catch (t: Throwable) {
            android.util.Log.e(TAG, "replay failed", t)
            ui("REPLAY FAILED: $t")
        }
    }

    /** RGB(A) Bitmap -> I420 (BT.601 full range), 2x2 均值色度。 */
    private fun rgbToI420(bmp: Bitmap, y: ByteArray, u: ByteArray, v: ByteArray) {
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        var yi = 0
        var ui = 0
        var vi = 0
        var j = 0
        while (j < h) {
            var i = 0
            while (i < w) {
                val p00 = px[j * w + i]
                val y00 = yOf(p00)
                y[yi++] = y00.toByte()
                if (j + 1 < h) {
                    val p10 = px[(j + 1) * w + i]
                    y[yi++] = yOf(p10).toByte()
                }
                if (i + 1 < w) {
                    val p01 = px[j * w + i + 1]
                    y[yi++] = yOf(p01).toByte()
                    if (j + 1 < h) {
                        val p11 = px[(j + 1) * w + i + 1]
                        y[yi++] = yOf(p11).toByte()
                    }
                }
                if (j % 2 == 0 && i % 2 == 0) {
                    // 2x2 均值
                    val p0 = p00
                    val p1 = if (i + 1 < w) px[j * w + i + 1] else p00
                    val p2 = if (j + 1 < h) px[(j + 1) * w + i] else p00
                    val p3 = if (i + 1 < w && j + 1 < h) px[(j + 1) * w + i + 1] else p00
                    var r = ((p0 shr 16) + (p1 shr 16) + (p2 shr 16) + (p3 shr 16)) and 0x3FC
                    var g = ((p0 shr 8) + (p1 shr 8) + (p2 shr 8) + (p3 shr 8)) and 0x3FC
                    var b = ((p0 and 0xFF) + (p1 and 0xFF) + (p2 and 0xFF) + (p3 and 0xFF))
                    r = r shr 2; g = g shr 2; b = b shr 2
                    u[ui++] = (((-112 * r - 74 * g + 186 * b) shr 8) + 128).coerceIn(0, 255).toByte()
                    v[vi++] = (((186 * r - 154 * g - 32 * b) shr 8) + 128).coerceIn(0, 255).toByte()
                }
                i += 2
            }
            j += 2
        }
    }

    private fun yOf(p: Int): Int {
        val r = (p shr 16) and 0xFF
        val g = (p shr 8) and 0xFF
        val b = p and 0xFF
        return (77 * r + 150 * g + 29 * b) shr 8
    }

    companion object {
        private const val TAG = "ReplayBridge"
    }
}
