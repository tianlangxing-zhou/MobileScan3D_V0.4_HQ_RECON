package com.mobilescan3d.diagnostics

import android.content.Context
import android.os.Build
import com.mobilescan3d.BuildConfig
import com.mobilescan3d.NativeBridge
import com.mobilescan3d.export.ExportManager
import com.mobilescan3d.mesh.MeshInspectionAnalyzer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Round 7 structured scan-quality evidence package. */
object ScanQualityReporter {
    data class UiSnapshot(
        val captureSufficiency: Int,
        val viewpointCoverage: Int,
        val surfaceCoverage: Int,
        val surfaceRobust: Int,
        val geometryQuality: Int,
        val textureQuality: Int,
        val angularSpeedDps: Float,
        val latestDistanceMeters: Float,
        val cumulativeMesh: Boolean,
        val cumulativeSegments: Int
    )

    data class Result(val json: File, val text: File, val summary: String)

    fun generate(
        context: Context,
        exportManager: ExportManager,
        sessionId: String,
        ui: UiSnapshot
    ): Result {
        val vins = FloatArray(NativeBridge.VINS_HEALTH_SLOTS)
        val vinsCount = runCatching { NativeBridge.nativeGetVinsHealth(vins) }.getOrDefault(0)
        val depth = FloatArray(NativeBridge.DEPTH_CALIBRATION_SLOTS)
        val depthCount = runCatching { NativeBridge.nativeGetDepthCalibration(depth) }.getOrDefault(0)
        val epoch = FloatArray(NativeBridge.FUSION_EPOCH_STATS_SLOTS)
        val epochCount = runCatching { NativeBridge.nativeGetFusionEpochStats(epoch) }.getOrDefault(0)
        val scan = FloatArray(NativeBridge.SCAN_UI_METRICS_SLOTS)
        val scanCount = runCatching { NativeBridge.nativeGetScanUiMetrics(scan) }.getOrDefault(0)
        val meshStats = runCatching { exportManager.meshStats() }.getOrDefault(IntArray(0))
        val cleanup = runCatching { exportManager.meshCleanupStats() }.getOrDefault(IntArray(0))
        val texture = runCatching { exportManager.textureStats() }.getOrDefault(IntArray(0))
        val mesh = exportManager.lastMesh ?: runCatching { exportManager.pullMesh() }.getOrNull()
        val topology = mesh?.let { runCatching { MeshInspectionAnalyzer.analyze(it.vertices, it.indices) }.getOrNull() }
        val observedWeights = if (!ui.cumulativeMesh && mesh != null) {
            runCatching { exportManager.meshObservationWeights(mesh.vertexCount) }.getOrDefault(FloatArray(0))
        } else FloatArray(0)
        var observedValid = 0
        var observedUsable = 0
        var observedRobust = 0
        for (w in observedWeights) {
            if (!w.isFinite() || w <= 0f) continue
            observedValid++
            if (w >= 4f) observedUsable++
            if (w >= 8f) observedRobust++
        }
        val measuredSurfaceCoverage = if (observedValid > 0) observedUsable * 100 / observedValid else ui.surfaceCoverage
        val measuredSurfaceRobust = if (observedValid > 0) observedRobust * 100 / observedValid else ui.surfaceRobust
        val calibration = com.mobilescan3d.calibration.DeviceCalibrationManager.load(context)

        val issues = ArrayList<String>()
        if (vinsCount == vins.size) {
            if (vins[NativeBridge.VINS_HEALTH_INITIALIZED] < 0.5f) issues += "VINS 未稳定初始化"
            if (vins[NativeBridge.VINS_HEALTH_FEATURES] < 45f) issues += "跟踪特征偏少"
            if (vins[NativeBridge.VINS_HEALTH_GRAVITY] !in 8.5f..11.0f) issues += "重力估计异常"
            if (vins[NativeBridge.VINS_HEALTH_GYRO_BIAS] > 0.08f) issues += "陀螺零偏偏高"
        }
        if (depthCount == depth.size && depth[NativeBridge.DEPTH_CALIBRATION_SLOTS - 1] < 0.5f) {
            issues += "深度标定未达到可用状态"
        }
        if (!ui.cumulativeMesh && measuredSurfaceCoverage in 1..64) issues += "真实表面重复观测不足"
        if (ui.viewpointCoverage < 55) issues += "视角覆盖不足"
        if (ui.textureQuality < 55) issues += "纹理画质偏弱"
        if (topology != null && topology.topologyComplete && topology.boundaryEdgeCount > 0) {
            issues += "模型存在 ${topology.boundaryEdgeCount} 条开放边"
        }

        val numericScore = if (ui.cumulativeMesh) {
            (ui.captureSufficiency * 0.22f +
                ui.viewpointCoverage * 0.20f +
                ui.geometryQuality * 0.36f +
                ui.textureQuality * 0.22f).toInt().coerceIn(0, 100)
        } else {
            (ui.captureSufficiency * 0.15f +
                ui.viewpointCoverage * 0.15f +
                measuredSurfaceCoverage * 0.30f +
                ui.geometryQuality * 0.25f +
                ui.textureQuality * 0.15f).toInt().coerceIn(0, 100)
        }
        val grade = when {
            numericScore >= 82 && issues.size <= 1 -> "良好"
            numericScore >= 62 -> "建议补扫"
            else -> "质量较弱"
        }

        val root = JSONObject()
            .put("schema", 1)
            .put("generatedAtMs", System.currentTimeMillis())
            .put("sessionId", sessionId)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("build", BuildConfig.GIT_COMMIT)
            .put("deviceCalibration", calibration.toJson())
            .put("score", numericScore)
            .put("grade", grade)
            .put("issues", JSONArray(issues))
            .put("ui", JSONObject()
                .put("captureSufficiency", ui.captureSufficiency)
                .put("viewpointCoverage", ui.viewpointCoverage)
                .put("surfaceCoverage", measuredSurfaceCoverage)
                .put("surfaceRobust", measuredSurfaceRobust)
                .put("surfaceWeightSamples", observedValid)
                .put("geometryQuality", ui.geometryQuality)
                .put("textureQuality", ui.textureQuality)
                .put("angularSpeedDps", ui.angularSpeedDps.toDouble())
                .put("latestDistanceMeters", ui.latestDistanceMeters.toDouble())
                .put("cumulativeMesh", ui.cumulativeMesh)
                .put("cumulativeSegments", ui.cumulativeSegments))
            .put("vinsHealth", floatProtocol(vins, vinsCount))
            .put("depthCalibration", floatProtocol(depth, depthCount))
            .put("fusionEpoch", floatProtocol(epoch, epochCount))
            .put("scanTelemetry", floatProtocol(scan, scanCount))
            .put("meshStats", JSONArray(meshStats.toList()))
            .put("meshCleanupStats", JSONArray(cleanup.toList()))
            .put("textureStats", JSONArray(texture.toList()))

        if (topology != null) {
            root.put("topology", JSONObject()
                .put("widthMeters", topology.width.toDouble())
                .put("heightMeters", topology.height.toDouble())
                .put("depthMeters", topology.depth.toDouble())
                .put("boundaryEdges", topology.boundaryEdgeCount)
                .put("nonManifoldEdges", topology.nonManifoldEdgeCount)
                .put("analysisComplete", topology.topologyComplete))
        }

        val dir = File(context.getExternalFilesDir(null) ?: context.filesDir, "scan_reports").apply { mkdirs() }
        val safe = sessionId.replace(Regex("[^A-Za-z0-9_.-]"), "_").take(80)
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val jsonFile = File(dir, "quality_${safe}_$stamp.json")
        val txtFile = File(dir, "quality_${safe}_$stamp.txt")
        jsonFile.writeText(root.toString(2), Charsets.UTF_8)
        txtFile.writeText(buildText(root, issues, grade, numericScore), Charsets.UTF_8)
        return Result(jsonFile, txtFile, "$grade · $numericScore/100" + if (issues.isEmpty()) "" else " · ${issues.first()}")
    }

    private fun floatProtocol(values: FloatArray, count: Int): JSONArray {
        val n = count.coerceIn(0, values.size)
        val out = JSONArray()
        for (i in 0 until n) out.put(values[i].toDouble())
        return out
    }

    private fun buildText(root: JSONObject, issues: List<String>, grade: String, score: Int): String = buildString {
        appendLine("MobileScan 3D 扫描质量报告")
        appendLine("会话: ${root.optString("sessionId")}")
        appendLine("结论: $grade ($score/100)")
        appendLine("设备: ${root.optString("device")}")
        appendLine()
        appendLine("问题/建议:")
        if (issues.isEmpty()) appendLine("- 未发现明显质量短板") else issues.forEach { appendLine("- $it") }
        appendLine()
        appendLine("完整机器可读数据见同名 JSON 文件。")
    }
}
