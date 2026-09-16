package com.mobilescan3d

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.DngCreator
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.util.Size
import android.view.Surface
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayDeque
import java.util.LinkedHashMap
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

data class HqCameraCaps(
    val rawSupported: Boolean,
    val manualSensor: Boolean,
    val manualPostProcessing: Boolean,
    val hardwareLevel: Int,
    val rawSizes: List<Size>,
    val jpegSizes: List<Size>,
    val yuvSizes: List<Size>,
    val aeLock: Boolean,
    val awbLock: Boolean,
    val manualFocus: Boolean,
    val hotPixelMapSupported: Boolean,
    val lensShadingMapSupported: Boolean
)

data class HqCaptureStats(
    var captureState: String = "IDLE",
    var sessionCombinationSupported: Boolean = false,
    var previewSize: String = "unknown",
    var analysisYuvSize: String = "unknown",
    var jpegSize: String = "unknown",
    var rawSize: String = "unknown",
    var aeState: Int? = null,
    var aeLocked: Boolean = false,
    var awbState: Int? = null,
    var awbLocked: Boolean = false,
    var afState: Int? = null,
    var afLocked: Boolean = false,
    var exposureNs: Long = 0L,
    var iso: Int = 0,
    var focusDiopters: Float = 0f,
    var burstRequested: Int = 0,
    var burstCompleted: Int = 0,
    var burstDropped: Int = 0,
    var imageResultMatched: Long = 0L,
    var imageResultLate: Long = 0L,
    var imageResultExpired: Long = 0L,
    var gyroRms: Float = 0f,
    var translationDuringBurst: Float = 0f,
    var alignmentInliers: Float = 0f,
    var alignmentRmsePx: Float = 0f,
    var eccScore: Float = 0f,
    var inputFrames: Int = 0,
    var acceptedFrames: Int = 0,
    var rejectedFrames: Int = 0,
    var sharpness: Float = 0f,
    var clippedPercent: Float = 0f,
    var underexposedPercent: Float = 0f,
    var noiseSigma: Float = 0f,
    var confidenceMean: Float = 0f,
    var confidenceLowPercent: Float = 0f,
    var rawDngSaved: Long = 0L,
    var jpegSaved: Long = 0L,
    var fusedImageSaved: Long = 0L,
    var lastCaptureRejectReason: String = ""
)

private data class HqImuSample(
    val timestampNs: Long,
    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,
    val accelX: Float,
    val accelY: Float,
    val accelZ: Float
)

class HqCaptureController(
    private val context: Context,
    private val characteristics: CameraCharacteristics,
    private val handler: Handler,
    private val onHint: (String) -> Unit
) {
    val caps: HqCameraCaps
    val stats = HqCaptureStats()

    private var jpegReader: ImageReader? = null
    private var rawReader: ImageReader? = null
    private var jpegSize: Size? = null
    private var rawSize: Size? = null
    private var active = false

    private val resultLock = Any()
    private val pendingResults = LinkedHashMap<Long, TotalCaptureResult>()
    private val imuLock = Any()
    private val imuSamples = ArrayDeque<HqImuSample>()

    private var burstInFlight = false
    private var burstToken = 0L
    private var burstExpectedFrames = 0
    private var burstCompletedThisRound = 0
    private var burstFailedThisRound = 0
    private var nextBurstAtNs = 0L
    private var currentSessionId = "unknown"
    private var currentBurstJpegs = mutableListOf<String>()
    private var currentBurstToken = 0L
    private var lastPoseNs = 0L
    private val lastPose = FloatArray(12)
    private var lastVinsTranslationSpeed = 0f

    init {
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            ?: intArrayOf()
        val raw = capabilities.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
        val manualSensor = capabilities.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
        val manualPost = capabilities.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING)
        val level = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            ?: CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
        val rawSizes = map?.getOutputSizes(ImageFormat.RAW_SENSOR)?.toList() ?: emptyList()
        val jpegSizes = map?.getOutputSizes(ImageFormat.JPEG)?.toList() ?: emptyList()
        val yuvSizes = map?.getOutputSizes(ImageFormat.YUV_420_888)?.toList() ?: emptyList()
        val ae = characteristics.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE) ?: false
        val awb = characteristics.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE) ?: false
        val minFocus = characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val hot = characteristics.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_HOT_PIXEL_MAP_MODES)?.isNotEmpty() == true
        val lens = characteristics.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_LENS_SHADING_MAP_MODES)?.isNotEmpty() == true

        caps = HqCameraCaps(
            rawSupported = raw,
            manualSensor = manualSensor,
            manualPostProcessing = manualPost,
            hardwareLevel = level,
            rawSizes = rawSizes,
            jpegSizes = jpegSizes,
            yuvSizes = yuvSizes,
            aeLock = ae,
            awbLock = awb,
            manualFocus = minFocus > 0f,
            hotPixelMapSupported = hot,
            lensShadingMapSupported = lens
        )
        stats.sessionCombinationSupported = raw || jpegSizes.isNotEmpty()
        stats.rawSize = chooseRawSize(rawSizes)?.let { "${it.width}x${it.height}" } ?: "unsupported"
        stats.jpegSize = chooseJpegSize(jpegSizes)?.let { "${it.width}x${it.height}" } ?: "unsupported"
        stats.analysisYuvSize = chooseYuvSize(yuvSizes)?.let { "${it.width}x${it.height}" } ?: "unsupported"
    }

    fun prepareSurfaces(previewSize: Size, yuvSize: Size): List<Surface> {
        stats.previewSize = "${previewSize.width}x${previewSize.height}"
        stats.analysisYuvSize = "${yuvSize.width}x${yuvSize.height}"
        val surfaces = mutableListOf<Surface>()

        jpegSize = chooseJpegSize(caps.jpegSizes)
        if (jpegSize != null) {
            val reader = ImageReader.newInstance(jpegSize!!.width, jpegSize!!.height, ImageFormat.JPEG, 8).apply {
                setOnImageAvailableListener({ r -> onJpegAvailable(r) }, handler)
            }
            jpegReader = reader
            surfaces += reader.surface
        }

        rawSize = chooseRawSize(caps.rawSizes)
        if (caps.rawSupported && rawSize != null) {
            val reader = ImageReader.newInstance(rawSize!!.width, rawSize!!.height, ImageFormat.RAW_SENSOR, 5).apply {
                setOnImageAvailableListener({ r -> onRawAvailable(r) }, handler)
            }
            rawReader = reader
            surfaces += reader.surface
        }
        return surfaces
    }

    fun onSessionConfigured(isActive: Boolean) {
        active = isActive
        if (!isActive) {
            detachSurfaces()
        }
    }

    fun detachSurfaces() {
        active = false
        jpegReader?.close(); jpegReader = null
        rawReader?.close(); rawReader = null
    }

    fun beginScan(sessionId: String) {
        currentSessionId = sessionId
        stats.captureState = "CONVERGING"
        stats.burstRequested = 0
        stats.burstCompleted = 0
        stats.burstDropped = 0
        stats.imageResultMatched = 0L
        stats.imageResultLate = 0L
        stats.imageResultExpired = 0L
        stats.rawDngSaved = 0L
        stats.jpegSaved = 0L
        stats.fusedImageSaved = 0L
        stats.lastCaptureRejectReason = ""
        nextBurstAtNs = System.nanoTime() + 1_000_000_000L
        burstInFlight = false
        synchronized(imuLock) { imuSamples.clear() }
        synchronized(resultLock) { pendingResults.clear() }
        burstExpectedFrames = 0
    }

    fun endScan() {
        stats.captureState = "IDLE"
        burstInFlight = false
    }

    fun close() {
        detachSurfaces()
    }

    fun onCaptureResult(result: TotalCaptureResult) {
        stats.aeState = result.get(CaptureResult.CONTROL_AE_STATE)
        stats.awbState = result.get(CaptureResult.CONTROL_AWB_STATE)
        stats.afState = result.get(CaptureResult.CONTROL_AF_STATE)
        stats.aeLocked = result.get(CaptureResult.CONTROL_AE_LOCK) ?: false
        stats.awbLocked = result.get(CaptureResult.CONTROL_AWB_LOCK) ?: false
        stats.afLocked = result.get(CaptureResult.CONTROL_AF_STATE) == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED
        stats.exposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
        stats.iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: 0
        stats.focusDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f

        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        synchronized(resultLock) {
            pendingResults[ts] = result
            while (pendingResults.size > 128) {
                val first = pendingResults.entries.iterator().next().key
                pendingResults.remove(first)
            }
        }
    }

    fun onImuSample(
        timestampNs: Long,
        gyroX: Float,
        gyroY: Float,
        gyroZ: Float,
        accelX: Float,
        accelY: Float,
        accelZ: Float
    ) {
        val sample = HqImuSample(timestampNs, gyroX, gyroY, gyroZ, accelX, accelY, accelZ)
        synchronized(imuLock) {
            imuSamples.addLast(sample)
            while (imuSamples.size > 120) imuSamples.removeFirst()
        }
    }

    fun onFrameTick(
        nowNs: Long,
        scanning: Boolean,
        captureLocked: Boolean,
        camera: CameraDevice?,
        session: CameraCaptureSession?
    ) {
        stats.gyroRms = computeMotionRms(nowNs)
        lastVinsTranslationSpeed = computeTranslationSpeed(nowNs)
        if (burstInFlight) return
        if (!scanning || !captureLocked || camera == null || session == null || !active) return
        if (nowNs < nextBurstAtNs) return
        if (stats.gyroRms > 0.08f || lastVinsTranslationSpeed > 0.12f) return

        startBurst(camera, session)
    }

    fun report(sb: StringBuilder) {
        sb.appendLine("[HQ CAPTURE]")
        sb.appendLine("rawSupported=${caps.rawSupported}")
        sb.appendLine("manualSensorSupported=${caps.manualSensor}")
        sb.appendLine("manualPostProcessingSupported=${caps.manualPostProcessing}")
        sb.appendLine("hardwareLevel=${caps.hardwareLevel}")
        sb.appendLine("sessionCombinationSupported=${stats.sessionCombinationSupported}")
        sb.appendLine("previewSize=${stats.previewSize}")
        sb.appendLine("analysisYuvSize=${stats.analysisYuvSize}")
        sb.appendLine("jpegSize=${stats.jpegSize}")
        sb.appendLine("rawSize=${stats.rawSize}")
        sb.appendLine("captureState=${stats.captureState}")
        sb.appendLine("AE state=${stats.aeState}")
        sb.appendLine("AE locked=${stats.aeLocked}")
        sb.appendLine("AWB state=${stats.awbState}")
        sb.appendLine("AWB locked=${stats.awbLocked}")
        sb.appendLine("AF state=${stats.afState}")
        sb.appendLine("AF locked=${stats.afLocked}")
        sb.appendLine("exposureNs=${stats.exposureNs}")
        sb.appendLine("ISO=${stats.iso}")
        sb.appendLine("focusDiopters=${stats.focusDiopters}")
        sb.appendLine("burstRequested=${stats.burstRequested}")
        sb.appendLine("burstCompleted=${stats.burstCompleted}")
        sb.appendLine("burstDropped=${stats.burstDropped}")
        sb.appendLine("imageResultMatched=${stats.imageResultMatched}")
        sb.appendLine("imageResultLate=${stats.imageResultLate}")
        sb.appendLine("imageResultExpired=${stats.imageResultExpired}")
        sb.appendLine("gyroRms=${stats.gyroRms}")
        sb.appendLine("translationDuringBurst=${stats.translationDuringBurst}")
        sb.appendLine("alignmentInliers=${stats.alignmentInliers}")
        sb.appendLine("alignmentRmsePx=${stats.alignmentRmsePx}")
        sb.appendLine("eccScore=${stats.eccScore}")
        sb.appendLine("inputFrames=${stats.inputFrames}")
        sb.appendLine("acceptedFrames=${stats.acceptedFrames}")
        sb.appendLine("rejectedFrames=${stats.rejectedFrames}")
        sb.appendLine("sharpness=${stats.sharpness}")
        sb.appendLine("clippedPercent=${stats.clippedPercent}")
        sb.appendLine("underexposedPercent=${stats.underexposedPercent}")
        sb.appendLine("noiseSigma=${stats.noiseSigma}")
        sb.appendLine("confidenceMean=${stats.confidenceMean}")
        sb.appendLine("confidenceLowPercent=${stats.confidenceLowPercent}")
        sb.appendLine("rawDngSaved=${stats.rawDngSaved}")
        sb.appendLine("jpegSaved=${stats.jpegSaved}")
        sb.appendLine("fusedImageSaved=${stats.fusedImageSaved}")
        sb.appendLine("lastCaptureRejectReason=${stats.lastCaptureRejectReason}")
        sb.appendLine("hotPixelMapSupported=${caps.hotPixelMapSupported}")
        sb.appendLine("lensShadingMapSupported=${caps.lensShadingMapSupported}")
        sb.appendLine("rawSizes=${caps.rawSizes.joinToString(",") { "${it.width}x${it.height}" }}")
        sb.appendLine("jpegSizes=${caps.jpegSizes.take(8).joinToString(",") { "${it.width}x${it.height}" }}")
        sb.appendLine("yuvSizes=${caps.yuvSizes.take(8).joinToString(",") { "${it.width}x${it.height}" }}")
    }

    private fun startBurst(camera: CameraDevice, session: CameraCaptureSession) {
        val count = if (caps.rawSupported && rawReader != null) 5 else 6
        stats.burstRequested += count
        stats.captureState = "CAPTURE_BURST"
        currentBurstJpegs = mutableListOf()
        currentBurstToken = System.nanoTime()
        burstToken = currentBurstToken
        burstExpectedFrames = count
        burstCompletedThisRound = 0
        burstFailedThisRound = 0

        try {
            val requests = (0 until count).map { _ ->
                camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                    jpegReader?.surface?.let { addTarget(it) }
                    rawReader?.surface?.let { addTarget(it) }
                    set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                    if (caps.manualSensor && stats.exposureNs > 0L) {
                        set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                        set(CaptureRequest.SENSOR_EXPOSURE_TIME, stats.exposureNs)
                        set(CaptureRequest.SENSOR_SENSITIVITY, stats.iso.coerceAtLeast(100))
                        set(CaptureRequest.SENSOR_FRAME_DURATION, stats.exposureNs + 16_000_000L)
                    } else {
                        set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                        if (caps.aeLock) set(CaptureRequest.CONTROL_AE_LOCK, true)
                    }
                    if (caps.awbLock) set(CaptureRequest.CONTROL_AWB_LOCK, true)
                    if (caps.manualFocus && stats.focusDiopters > 0f) {
                        set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                        set(CaptureRequest.LENS_FOCUS_DISTANCE, stats.focusDiopters)
                    }
                    set(CaptureRequest.JPEG_QUALITY, 100.toByte())
                    set(CaptureRequest.JPEG_ORIENTATION, characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90)
                }.build()
            }
            burstInFlight = true
            session.captureBurst(requests, burstCallback, handler)
        } catch (e: Exception) {
            burstInFlight = false
            stats.burstDropped += count
            stats.captureState = "SCAN_LOCKED"
            stats.lastCaptureRejectReason = "burst: ${e.message}"
        }
    }

    private val burstCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            onCaptureResult(result)
            burstCompletedThisRound++
            stats.burstCompleted = burstCompletedThisRound
            if (burstCompletedThisRound >= burstExpectedFrames) {
                burstInFlight = false
                stats.captureState = "SCAN_LOCKED"
                handler.postDelayed({ fuseCurrentBurst() }, 350L)
            }
        }

        override fun onCaptureFailed(
            session: CameraCaptureSession,
            request: CaptureRequest,
            failure: android.hardware.camera2.CaptureFailure
        ) {
            stats.burstDropped++
            burstFailedThisRound++
            if (burstCompletedThisRound + burstFailedThisRound >= burstExpectedFrames) {
                burstInFlight = false
                stats.captureState = "SCAN_LOCKED"
            }
        }
    }

    private fun onJpegAvailable(reader: ImageReader) {
        val image = reader.acquireLatestImage() ?: return
        try {
            if (!active) {
                stats.imageResultExpired++
                return
            }
            val ts = image.timestamp
            val result = matchResult(ts)
            val bytes = ByteArray(image.planes[0].buffer.remaining())
            image.planes[0].buffer.get(bytes)
            val dir = captureDir(currentSessionId)
            val file = File(dir, "hq_${ts}_${System.nanoTime()}.jpg")
            FileOutputStream(file).use { it.write(bytes) }
            stats.jpegSaved++
            synchronized(currentBurstJpegs) {
                currentBurstJpegs.add(file.absolutePath)
            }
            if (result == null) stats.imageResultLate++
        } catch (e: Exception) {
            stats.imageResultLate++
            stats.lastCaptureRejectReason = "jpeg: ${e.message}"
        } finally {
            image.close()
        }
    }

    private fun onRawAvailable(reader: ImageReader) {
        val image = reader.acquireLatestImage() ?: return
        try {
            if (!active) {
                stats.imageResultExpired++
                return
            }
            val ts = image.timestamp
            val result = matchResult(ts)
            if (result == null) {
                stats.imageResultExpired++
                return
            }
            val dir = captureDir(currentSessionId)
            val file = File(dir, "raw_${ts}.dng")
            val dng = DngCreator(characteristics, result)
            try {
                dng.setDescription("MobileScan3D HQ RAW keyframe")
                FileOutputStream(file).use { out -> dng.writeImage(out, image) }
                stats.rawDngSaved++
            } finally {
                dng.close()
            }
        } catch (e: Exception) {
            stats.imageResultExpired++
            stats.lastCaptureRejectReason = "raw: ${e.message}"
        } finally {
            image.close()
        }
    }

    private fun matchResult(timestampNs: Long): TotalCaptureResult? {
        synchronized(resultLock) {
            val exact = pendingResults.remove(timestampNs)
            if (exact != null) {
                stats.imageResultMatched++
                return exact
            }
            val nearest = pendingResults.entries.minByOrNull { abs(it.key - timestampNs) }
            if (nearest != null && abs(nearest.key - timestampNs) <= 10_000_000L) {
                pendingResults.remove(nearest.key)
                stats.imageResultMatched++
                return nearest.value
            }
        }
        return null
    }

    private fun fuseCurrentBurst() {
        val paths = synchronized(currentBurstJpegs) { currentBurstJpegs.toList() }
        if (paths.size < 2) {
            stats.lastCaptureRejectReason = "burst images < 2"
            scheduleRetake()
            return
        }
        val outDir = captureDir(currentSessionId)
        val out = File(outDir, "fused_${currentBurstToken}.jpg")
        val statsOut = FloatArray(12)
        val exposureMs = stats.exposureNs / 1_000_000f
        val iso = stats.iso
        Thread {
            val ok = try {
                NativeBridge.nativeFuseBurst(
                    paths.toTypedArray(),
                    out.absolutePath,
                    exposureMs,
                    iso,
                    statsOut
                )
            } catch (e: Throwable) {
                false
            }
            handler.post {
                applyFusionStats(ok, out, statsOut, paths.size)
            }
        }.start()
    }

    private fun applyFusionStats(ok: Boolean, out: File, statsOut: FloatArray, pathCount: Int) {
        stats.inputFrames = pathCount
        stats.sharpness = statsOut[0]
        stats.clippedPercent = statsOut[1]
        stats.underexposedPercent = statsOut[2]
        stats.noiseSigma = statsOut[3]
        stats.confidenceMean = statsOut[4]
        stats.confidenceLowPercent = statsOut[5]
        stats.alignmentInliers = statsOut[6]
        stats.alignmentRmsePx = statsOut[7]
        stats.eccScore = statsOut[8]
        stats.acceptedFrames = statsOut[10].toInt()
        stats.rejectedFrames = statsOut[11].toInt()
        if (ok && out.exists()) stats.fusedImageSaved++

        val reject = if (!ok) "fusion failed" else {
            when {
                stats.underexposedPercent > 20f -> "underexposed"
                stats.clippedPercent > 5f -> "overexposed"
                stats.alignmentInliers < 10f -> "alignment too weak"
                stats.acceptedFrames < 3 -> "too few accepted frames"
                stats.confidenceMean < 0.52f -> "low confidence"
                else -> ""
            }
        }
        if (reject.isNotEmpty()) {
            stats.lastCaptureRejectReason = reject
            scheduleRetake()
        } else {
            stats.lastCaptureRejectReason = ""
        }
    }

    private fun scheduleRetake() {
        nextBurstAtNs = System.nanoTime() + 1_500_000_000L
        val msg = when (stats.lastCaptureRejectReason) {
            "underexposed" -> "光线不足，请增加照明后保持稳定"
            "overexposed" -> "高光过曝，请调整拍摄角度"
            "alignment too weak" -> "关键帧对齐失败，请更慢地移动手机"
            "low confidence" -> "关键帧质量不足，请保持稳定后自动补拍"
            else -> "关键帧补拍：请保持手机稳定"
        }
        onHint(msg)
    }

    private fun computeMotionRms(nowNs: Long): Float {
        val cutoff = nowNs - 250_000_000L
        val recent = synchronized(imuLock) { imuSamples.filter { it.timestampNs >= cutoff } }
        if (recent.size < 6) return 999f
        val gx = recent.map { it.gyroX }.average().toFloat()
        val gy = recent.map { it.gyroY }.average().toFloat()
        val gz = recent.map { it.gyroZ }.average().toFloat()
        val g = sqrt(recent.map { val dx = it.gyroX - gx; val dy = it.gyroY - gy; val dz = it.gyroZ - gz; dx * dx + dy * dy + dz * dz }.average().toFloat())
        val ax = recent.map { it.accelX }.average().toFloat()
        val ay = recent.map { it.accelY }.average().toFloat()
        val az = recent.map { it.accelZ }.average().toFloat()
        val a = sqrt(recent.map { val dx = it.accelX - ax; val dy = it.accelY - ay; val dz = it.accelZ - az; dx * dx + dy * dy + dz * dz }.average().toFloat())
        return maxOf(g, a * 0.1f)
    }

    private fun computeTranslationSpeed(nowNs: Long): Float {
        val pose = FloatArray(12)
        if (!NativeBridge.nativeGetRenderPose(pose)) return 999f
        if (lastPoseNs == 0L) {
            lastPoseNs = nowNs
            System.arraycopy(pose, 0, lastPose, 0, 12)
            return 0f
        }
        val dt = (nowNs - lastPoseNs) / 1_000_000_000f
        if (dt <= 0f || dt > 1f) {
            lastPoseNs = nowNs
            System.arraycopy(pose, 0, lastPose, 0, 12)
            return lastVinsTranslationSpeed
        }
        val dx = pose[9] - lastPose[9]
        val dy = pose[10] - lastPose[10]
        val dz = pose[11] - lastPose[11]
        val speed = hypot(hypot(dx, dy), dz) / dt
        lastPoseNs = nowNs
        System.arraycopy(pose, 0, lastPose, 0, 12)
        return speed
    }

    private fun captureDir(sessionId: String): File {
        val dir = File(context.filesDir, "hq_capture/$sessionId")
        dir.mkdirs()
        return dir
    }

    private fun chooseRawSize(sizes: List<Size>): Size? =
        sizes.filter { it.width >= it.height && it.width <= 4096 && it.height <= 3072 }
            .maxByOrNull { it.width.toLong() * it.height }

    private fun chooseJpegSize(sizes: List<Size>): Size? {
        val targetRatio = 4f / 3f
        val candidates = sizes.filter { it.width >= it.height && it.width <= 4096 && it.height <= 3072 }
            .ifEmpty { sizes }
        return candidates.sortedWith(
            compareBy<Size> { abs(it.width.toFloat() / it.height - targetRatio) }
                .thenByDescending { it.width.toLong() * it.height }
        ).firstOrNull()
    }

    private fun chooseYuvSize(sizes: List<Size>): Size? =
        sizes.filter { it.width >= it.height && it.width <= 1920 && it.height <= 1440 }
            .maxByOrNull { it.width.toLong() * it.height }
}
