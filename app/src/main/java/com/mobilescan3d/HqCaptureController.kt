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
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
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
    // 状态机：IDLE / WAIT_3A / APPLY_LOCK / VERIFY_LOCK / SCAN_LOCKED / CAPTURE
    var captureState: String = "IDLE",
    var lockRequested: Boolean = false,
    var lockVerifyStreak: Int = 0,
    var lockAttempts: Int = 0,
    var lockFailReason: String = "",
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
    // 逐项锁定校验结果（只有 CaptureResult 真正确认才算 true）
    var aeVerified: Boolean = false,
    var awbVerified: Boolean = false,
    var afVerified: Boolean = false,
    var exposureNs: Long = 0L,
    var iso: Int = 0,
    var focusDiopters: Float = 0f,
    var burstRequested: Int = 0,
    var burstCompleted: Int = 0,
    var burstDropped: Int = 0,
    var captureCycles: Int = 0,
    var burstSkipCount: Long = 0L,
    var burstCooldownMs: Long = 0L,
    var triggerRejectReason: String = "",
    var imageResultMatched: Long = 0L,
    var imageResultLate: Long = 0L,
    var imageResultExpired: Long = 0L,
    var imageDeferredMatched: Long = 0L,
    var rawSkipped: Long = 0L,
    var gyroRms: Float = 0f,
    var vinsDisplacement200ms: Float = 0f,
    var translationDuringBurst: Float = 0f,
    var lastBurstPoseDelta: Float = 0f,
    var lastBurstAngleDeltaDeg: Float = 0f,
    var previewClippedPercent: Float = 0f,
    var previewUnderexposedPercent: Float = 0f,
    var previewLumaSamples: Int = 0,
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
    var referenceJpeg: String = "",
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

private data class PoseSample(val timestampNs: Long, val pose: FloatArray)

/** 等待与 CaptureResult 配对的 Image（双向配对用，Image 直到配对成功或超时才关闭） */
private class PendingImage(val image: Image, val isRaw: Boolean, val arrivedNs: Long)

class HqCaptureController(
    private val context: Context,
    private val characteristics: CameraCharacteristics,
    private val handler: Handler,
    private val onHint: (String) -> Unit
) {
    companion object {
        const val STATE_IDLE = "IDLE"
        const val STATE_WAIT_3A = "WAIT_3A"
        const val STATE_APPLY_LOCK = "APPLY_LOCK"
        const val STATE_VERIFY_LOCK = "VERIFY_LOCK"
        const val STATE_SCAN_LOCKED = "SCAN_LOCKED"
        const val STATE_CAPTURE = "CAPTURE"

        /** 连续多少个 CaptureResult 确认锁定后才算真锁定 */
        private const val LOCK_VERIFY_STREAK = 3
        /** 3A 收敛确认的连续帧数（进入 APPLY_LOCK 的门槛） */
        private const val READY_STREAK = 2
        /** APPLY_LOCK 等待确认的超时 */
        private const val APPLY_LOCK_TIMEOUT_NS = 1_500_000_000L
        /** burst 冷却区间 0.8~1.2s */
        private const val BURST_COOLDOWN_MIN_NS = 800_000_000L
        private const val BURST_COOLDOWN_MAX_NS = 1_200_000_000L
        private const val RETRY_COOLDOWN_NS = 1_500_000_000L
        /** 触发前的手持稳定门限 */
        private const val MAX_GYRO_RMS = 0.05f
        private const val MAX_VINS_DISP_200MS = 0.05f
        /** 触发前的曝光预检门限：高光 clipping 超过即不发 still capture */
        private const val MAX_PREVIEW_CLIP_PERCENT = 4.0f
        /** 已锁定时若高光仍严重过曝，解锁重收敛 */
        private const val OVEREXPOSURE_RECOVER_PERCENT = 6.0f
        private const val MAX_EXPOSURE_RECOVERY_ATTEMPTS = 2
        private const val EXPOSURE_RECOVERY_COOLDOWN_NS = 3_000_000_000L
        /** 新视角判定：平移或转角超过其一才值得再拍 */
        private const val MIN_NEW_VIEW_TRANSLATION = 0.015f
        private const val MIN_NEW_VIEW_ANGLE_DEG = 3.0f
        /** 严格对齐门限 */
        private const val MIN_ALIGN_INLIERS = 30f
        private const val MAX_ALIGN_RMSE_PX = 1.5f
        private const val MIN_ALIGN_ECC = 0.95f
        /** Image 等待 CaptureResult 的最长时间 */
        private const val IMAGE_RESULT_HOLD_NS = 1_200_000_000L
        private const val BURST_TMP_DIR = "tmp_burst"
    }

    val caps: HqCameraCaps
    val stats = HqCaptureStats()

    private var jpegReader: ImageReader? = null
    private var rawReader: ImageReader? = null
    private var jpegSize: Size? = null
    private var rawSize: Size? = null
    private var active = false

    // Image 与 CaptureResult 的双向配对表，共用一把锁，避免锁序反转
    private val matcherLock = Any()
    private val pendingResults = LinkedHashMap<Long, TotalCaptureResult>()
    private val pendingImages = LinkedHashMap<Long, PendingImage>()

    private val imuLock = Any()
    private val imuSamples = ArrayDeque<HqImuSample>()
    private val poseLock = Any()
    private val poseHistory = ArrayDeque<PoseSample>()

    // 预览曝光直方图（粗采样），用于触发前门控
    private val lumaLock = Any()
    private var previewClipEma = 0f
    private var previewUnderEma = 0f
    private var previewLumaInit = false

    // 状态机
    private var currentPhase = STATE_IDLE
    private var lockApplyStartNs = 0L
    /** WAIT_3A 用：3A 处于「已收敛」的连续帧数 */
    private var readyStreak = 0
    /** VERIFY_LOCK 用：CaptureResult 确认「已锁定」的连续帧数 */
    private var lockedStreak = 0
    private var lastManualExpNs = 0L
    private var lastManualIso = 0
    private var exposureRecoveryAttempts = 0
    private var lastExposureRecoveryNs = 0L

    // burst
    private var burstInFlight = false
    private var nextBurstAtNs = 0L
    private var currentSessionId = "unknown"
    private var currentBurstJpegs = mutableListOf<String>()
    private var currentBurstToken = 0L
    private var burstExpectedFrames = 0
    private var burstCompletedThisRound = 0
    private var burstFailedThisRound = 0
    private var burstRawSavedThisRound = 0
    private var fusionScheduled = false

    // 新视角判定
    private var haveLastBurstPose = false
    private val lastBurstPose = FloatArray(12)

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
            // maxImages 要容得下整个 burst，否则 acquireNextImage 会拿不到缓冲
            val reader = ImageReader.newInstance(jpegSize!!.width, jpegSize!!.height, ImageFormat.JPEG, 12).apply {
                setOnImageAvailableListener({ r -> onJpegAvailable(r) }, handler)
            }
            jpegReader = reader
            surfaces += reader.surface
        }

        rawSize = chooseRawSize(caps.rawSizes)
        if (caps.rawSupported && rawSize != null) {
            val reader = ImageReader.newInstance(rawSize!!.width, rawSize!!.height, ImageFormat.RAW_SENSOR, 8).apply {
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
        synchronized(matcherLock) {
            pendingImages.values.forEach { runCatching { it.image.close() } }
            pendingImages.clear()
            pendingResults.clear()
        }
    }

    fun beginScan(sessionId: String) {
        currentSessionId = sessionId
        setPhase(STATE_WAIT_3A)
        stats.burstRequested = 0
        stats.burstCompleted = 0
        stats.burstDropped = 0
        stats.captureCycles = 0
        stats.burstSkipCount = 0L
        stats.imageResultMatched = 0L
        stats.imageResultLate = 0L
        stats.imageResultExpired = 0L
        stats.imageDeferredMatched = 0L
        stats.rawSkipped = 0L
        stats.rawDngSaved = 0L
        stats.jpegSaved = 0L
        stats.fusedImageSaved = 0L
        stats.referenceJpeg = ""
        stats.lastCaptureRejectReason = ""
        stats.triggerRejectReason = ""
        stats.lockFailReason = ""
        stats.lockAttempts = 0
        stats.lockVerifyStreak = 0
        nextBurstAtNs = System.nanoTime() + 1_000_000_000L
        burstInFlight = false
        haveLastBurstPose = false
        exposureRecoveryAttempts = 0
        lastExposureRecoveryNs = 0L
        synchronized(imuLock) { imuSamples.clear() }
        synchronized(poseLock) { poseHistory.clear() }
        synchronized(matcherLock) {
            pendingResults.clear()
            pendingImages.values.forEach { runCatching { it.image.close() } }
            pendingImages.clear()
        }
        burstExpectedFrames = 0
        fusionScheduled = false
        cleanupBurstTmp()
    }

    fun endScan() {
        setPhase(STATE_IDLE)
        burstInFlight = false
        fusionScheduled = false
    }

    fun close() {
        detachSurfaces()
        cleanupBurstTmp()
    }

    // ---------------------------------------------------------------- metadata

    fun onCaptureResult(result: TotalCaptureResult) {
        stats.aeState = result.get(CaptureResult.CONTROL_AE_STATE)
        stats.awbState = result.get(CaptureResult.CONTROL_AWB_STATE)
        stats.afState = result.get(CaptureResult.CONTROL_AF_STATE)
        stats.aeLocked = result.get(CaptureResult.CONTROL_AE_LOCK) ?: false
        stats.awbLocked = result.get(CaptureResult.CONTROL_AWB_LOCK) ?: false
        stats.exposureNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
        stats.iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: 0
        stats.focusDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f

        val aeOk = verifyAe(result)
        val awbOk = verifyAwb(result)
        val afOk = verifyAf(result)
        stats.aeVerified = aeOk
        stats.awbVerified = awbOk
        stats.afVerified = afOk
        stats.afLocked = afOk

        // 两套判据必须分开，否则 WAIT_3A 阶段永远等不到「已锁定」而死锁：
        //  - readyStreak  : 3A 已收敛（AE CONVERGED / AF PASSIVE_FOCUSED 以上）
        //  - lockedStreak : CaptureResult 真正确认锁定（AE/AWB LOCKED + AF FOCUSED_LOCKED）
        val converged = aeReady(result) && awbReady(result) && afReady(result)
        readyStreak = if (converged) readyStreak + 1 else 0
        lockedStreak = if (aeOk && awbOk && afOk) lockedStreak + 1 else 0
        stats.lockVerifyStreak = lockedStreak

        advanceStateMachine(System.nanoTime())

        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        var held: PendingImage? = null
        synchronized(matcherLock) {
            pendingResults[ts] = result
            while (pendingResults.size > 256) {
                pendingResults.remove(pendingResults.entries.iterator().next().key)
            }
            // 双向配对：Image 若先到，这里立刻把它消费掉
            held = pendingImages.remove(ts)
        }
        val pending = held
        if (pending != null) {
            stats.imageDeferredMatched++
            consumeImage(pending, ts, result)
        }
    }

    /** AE 是否已收敛（WAIT_3A 用；SEARCHING 不算） */
    private fun aeReady(result: TotalCaptureResult): Boolean {
        if (result.get(CaptureResult.CONTROL_AE_MODE) == CaptureRequest.CONTROL_AE_MODE_OFF) {
            return true
        }
        return when (result.get(CaptureResult.CONTROL_AE_STATE)) {
            CaptureResult.CONTROL_AE_STATE_CONVERGED,
            CaptureResult.CONTROL_AE_STATE_LOCKED -> true
            else -> false
        }
    }

    /** AF 是否已合焦（PASSIVE_SCAN 不算） */
    private fun afReady(result: TotalCaptureResult): Boolean {
        if (result.get(CaptureResult.CONTROL_AF_MODE) == CaptureRequest.CONTROL_AF_MODE_OFF) {
            return true
        }
        return when (result.get(CaptureResult.CONTROL_AF_STATE)) {
            CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED,
            CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> true
            else -> false
        }
    }

    private fun awbReady(result: TotalCaptureResult): Boolean {
        return when (result.get(CaptureResult.CONTROL_AWB_STATE)) {
            CaptureResult.CONTROL_AWB_STATE_CONVERGED,
            CaptureResult.CONTROL_AWB_STATE_LOCKED -> true
            else -> false
        }
    }

    /** AE 是否已被 CaptureResult 确认为「已锁定」 */
    private fun verifyAe(result: TotalCaptureResult): Boolean {
        if (result.get(CaptureResult.CONTROL_AE_MODE) == CaptureRequest.CONTROL_AE_MODE_OFF) {
            // 手动曝光：以曝光时间/ISO 的稳定性代替 AE_STATE
            val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
            val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: 0
            val expStable = lastManualExpNs <= 0L || abs(exp - lastManualExpNs) <= max(1L, lastManualExpNs / 100L)
            val isoStable = lastManualIso <= 0 || abs(iso - lastManualIso) <= max(1, lastManualIso / 100)
            lastManualExpNs = exp
            lastManualIso = iso
            return exp > 0L && iso > 0 && expStable && isoStable
        }
        val state = result.get(CaptureResult.CONTROL_AE_STATE)
        val stateOk = state == CaptureResult.CONTROL_AE_STATE_LOCKED ||
            (!caps.aeLock && state == CaptureResult.CONTROL_AE_STATE_CONVERGED)
        val lockOk = !caps.aeLock || (result.get(CaptureResult.CONTROL_AE_LOCK) ?: false)
        return stateOk && lockOk
    }

    private fun verifyAwb(result: TotalCaptureResult): Boolean {
        val state = result.get(CaptureResult.CONTROL_AWB_STATE)
        return if (caps.awbLock) {
            state == CaptureResult.CONTROL_AWB_STATE_LOCKED &&
                (result.get(CaptureResult.CONTROL_AWB_LOCK) ?: false)
        } else {
            state == CaptureResult.CONTROL_AWB_STATE_CONVERGED ||
                state == CaptureResult.CONTROL_AWB_STATE_LOCKED
        }
    }

    private fun verifyAf(result: TotalCaptureResult): Boolean {
        val mode = result.get(CaptureResult.CONTROL_AF_MODE)
        if (mode == CaptureRequest.CONTROL_AF_MODE_OFF) {
            if (!caps.manualFocus) {
                // 定焦设备：没有可锁定的对焦马达，视为已确定
                return true
            }
            val d = result.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f
            return d > 0f
        }
        val state = result.get(CaptureResult.CONTROL_AF_STATE)
        if (state == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED) {
            return true
        }
        // 连续对焦模式下若拿不到可锁定的焦距（LENS_FOCUS_DISTANCE 不可用），
        // 只接受「已合焦」PASSIVE_FOCUSED；仍在扫描的 PASSIVE_SCAN 一律不接受。
        return stats.focusDiopters <= 0f &&
            state == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED
    }

    // ------------------------------------------------------------- state machine

    private fun setPhase(phase: String) {
        if (currentPhase == phase) {
            return
        }
        currentPhase = phase
        stats.captureState = phase
        stats.lockRequested = phase == STATE_APPLY_LOCK ||
            phase == STATE_VERIFY_LOCK ||
            phase == STATE_SCAN_LOCKED ||
            phase == STATE_CAPTURE
        if (phase == STATE_WAIT_3A) {
            lockedStreak = 0
            stats.lockVerifyStreak = 0
            lastManualExpNs = 0L
            lastManualIso = 0
        }
    }

    /**
     * WAIT_3A -> APPLY_LOCK -> VERIFY_LOCK -> SCAN_LOCKED -> CAPTURE
     * 只有 CaptureResult 真正确认锁定（AE/AWB/AF 全部达到锁定态）才允许 HQ burst。
     */
    private fun advanceStateMachine(nowNs: Long) {
        when (currentPhase) {
            STATE_WAIT_3A -> {
                // 注意：这里用「已收敛」判据。AE=SEARCHING(1)、AF=PASSIVE_SCAN(1)
                // 都不算就绪，只有真正收敛/合焦才允许下发锁定。
                if (readyStreak >= READY_STREAK) {
                    stats.lockAttempts++
                    lockApplyStartNs = nowNs
                    lockedStreak = 0
                    setPhase(STATE_APPLY_LOCK)
                }
            }

            STATE_APPLY_LOCK -> {
                // 锁定请求由 MainActivity 依据 stats.lockRequested 下发，
                // 这里只需等下一批结果回来做校验
                setPhase(STATE_VERIFY_LOCK)
            }

            STATE_VERIFY_LOCK -> {
                if (lockedStreak >= LOCK_VERIFY_STREAK) {
                    stats.lockFailReason = ""
                    setPhase(STATE_SCAN_LOCKED)
                    onHint("3A 已确认锁定，开始采集高质量关键帧")
                } else if (nowNs - lockApplyStartNs > APPLY_LOCK_TIMEOUT_NS) {
                    stats.lockFailReason =
                        "3A 锁定超时 AE=${stats.aeState} AF=${stats.afState} AWB=${stats.awbState}"
                    stats.lockAttempts++
                    lockApplyStartNs = nowNs
                    setPhase(STATE_WAIT_3A)
                    onHint("3A 锁定未被确认，重新收敛")
                }
            }

            STATE_SCAN_LOCKED, STATE_CAPTURE -> {
                // 锁定后续若漂移，退回重新收敛
                if (lockedStreak == 0 && stats.aeState == CaptureResult.CONTROL_AE_STATE_SEARCHING) {
                    stats.lockFailReason = "AE 漂移，重新锁定"
                    setPhase(STATE_WAIT_3A)
                    return
                }
                // 高光严重过曝：不要锁死一个过曝的曝光，主动解锁重收敛
                if (stats.previewClippedPercent > OVEREXPOSURE_RECOVER_PERCENT &&
                    exposureRecoveryAttempts < MAX_EXPOSURE_RECOVERY_ATTEMPTS &&
                    nowNs - lastExposureRecoveryNs > EXPOSURE_RECOVERY_COOLDOWN_NS
                ) {
                    exposureRecoveryAttempts++
                    lastExposureRecoveryNs = nowNs
                    stats.triggerRejectReason =
                        "高光过曝(${"%.1f".format(stats.previewClippedPercent)}%)，解锁重收敛"
                    onHint("高光过曝，重新收敛曝光")
                    setPhase(STATE_WAIT_3A)
                }
            }
        }
    }

    // ------------------------------------------------------------------ preview

    /** 粗采样 Y 平面算曝光直方图，用于拍摄前门控（约 1/64 像素，开销可忽略） */
    fun onPreviewLuma(y: ByteArray, width: Int, height: Int, rowStride: Int) {
        if (width <= 0 || height <= 0 || rowStride <= 0 || y.isEmpty()) {
            return
        }
        var clipped = 0
        var under = 0
        var total = 0
        val step = 8
        var row = 0
        while (row < height) {
            val base = row * rowStride
            var col = 0
            while (col < width) {
                val idx = base + col
                if (idx < y.size) {
                    val v = y[idx].toInt() and 0xFF
                    if (v >= 250) clipped++ else if (v <= 20) under++
                    total++
                }
                col += step
            }
            row += step
        }
        if (total <= 0) {
            return
        }
        val clipPct = 100f * clipped / total
        val underPct = 100f * under / total
        synchronized(lumaLock) {
            if (previewLumaInit) {
                previewClipEma = previewClipEma * 0.7f + clipPct * 0.3f
                previewUnderEma = previewUnderEma * 0.7f + underPct * 0.3f
            } else {
                previewClipEma = clipPct
                previewUnderEma = underPct
                previewLumaInit = true
            }
            stats.previewClippedPercent = previewClipEma
            stats.previewUnderexposedPercent = previewUnderEma
        }
        stats.previewLumaSamples = total
    }

    // ---------------------------------------------------------------- frame tick

    fun onFrameTick(
        nowNs: Long,
        scanning: Boolean,
        camera: CameraDevice?,
        session: CameraCaptureSession?
    ) {
        stats.gyroRms = computeMotionRms(nowNs)
        stats.vinsDisplacement200ms = computeTranslationDisplacement200ms(nowNs)

        if (!scanning) {
            if (currentPhase != STATE_IDLE) {
                setPhase(STATE_IDLE)
            }
            sweepPendingImages(nowNs)
            return
        }

        sweepPendingImages(nowNs)
        if (currentPhase == STATE_IDLE) {
            setPhase(STATE_WAIT_3A)
        }

        if (burstInFlight) {
            return
        }
        if (currentPhase != STATE_SCAN_LOCKED) {
            return
        }
        if (camera == null || session == null || !active) {
            return
        }

        // 1) 曝光预检：直方图不合格就不发 still capture
        if (stats.previewClippedPercent > MAX_PREVIEW_CLIP_PERCENT) {
            stats.triggerRejectReason =
                "overexposed(预检 clipping=${"%.1f".format(stats.previewClippedPercent)}%)"
            stats.burstSkipCount++
            nextBurstAtNs = nowNs + RETRY_COOLDOWN_NS
            return
        }

        // 2) 冷却（0.8~1.2s）
        if (nowNs < nextBurstAtNs) {
            stats.triggerRejectReason = "cooldown"
            return
        }

        // 3) 手持稳定：陀螺 RMS + VINS 200ms 位移
        if (stats.gyroRms > MAX_GYRO_RMS) {
            stats.triggerRejectReason = "gyroRms=${"%.3f".format(stats.gyroRms)}"
            stats.burstSkipCount++
            return
        }
        if (stats.vinsDisplacement200ms > MAX_VINS_DISP_200MS) {
            stats.triggerRejectReason = "vinsDisp200ms=${"%.3f".format(stats.vinsDisplacement200ms)}"
            stats.burstSkipCount++
            return
        }

        // 4) 新视角是否值得拍
        if (!novelViewpoint()) {
            stats.triggerRejectReason =
                "viewpoint 未变化(dT=${"%.4f".format(stats.lastBurstPoseDelta)}, dR=${"%.1f".format(stats.lastBurstAngleDeltaDeg)}°)"
            stats.burstSkipCount++
            return
        }

        stats.triggerRejectReason = ""
        startBurst(camera, session)
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

    // --------------------------------------------------------------- burst 拍摄

    private fun startBurst(camera: CameraDevice, session: CameraCaptureSession) {
        val count = if (caps.rawSupported && rawReader != null) 5 else 6
        stats.burstRequested += count
        stats.captureCycles++
        setPhase(STATE_CAPTURE)
        synchronized(currentBurstJpegs) { currentBurstJpegs.clear() }
        currentBurstToken = System.nanoTime()
        burstExpectedFrames = count
        burstCompletedThisRound = 0
        burstFailedThisRound = 0
        burstRawSavedThisRound = 0
        fusionScheduled = false

        // 记录本次拍摄的位姿，用于「新视角」判定
        val pose = FloatArray(12)
        if (NativeBridge.nativeGetRenderPose(pose)) {
            System.arraycopy(pose, 0, lastBurstPose, 0, 12)
            haveLastBurstPose = true
        }

        try {
            val requests = (0 until count).map { _ ->
                camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                    jpegReader?.surface?.let { addTarget(it) }
                    rawReader?.surface?.let { addTarget(it) }
                    set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                    // 使用已通过 CaptureResult 验证的曝光/ISO，而不是可能过期的值
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
                    set(
                        CaptureRequest.JPEG_ORIENTATION,
                        characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
                    )
                }.build()
            }
            burstInFlight = true
            session.captureBurst(requests, burstCallback, handler)
        } catch (e: Exception) {
            burstInFlight = false
            stats.burstDropped += count
            setPhase(STATE_SCAN_LOCKED)
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
                finishBurstRound()
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
                finishBurstRound()
            }
        }
    }

    private fun finishBurstRound() {
        burstInFlight = false
        setPhase(STATE_SCAN_LOCKED)
        // 已经不再需要那么频繁，给下一次 burst 留出 0.8~1.2s 的冷却
        val span = (BURST_COOLDOWN_MAX_NS - BURST_COOLDOWN_MIN_NS).toDouble()
        val cd = (BURST_COOLDOWN_MIN_NS + (Math.random() * span).toLong())
        stats.burstCooldownMs = cd / 1_000_000L
        nextBurstAtNs = System.nanoTime() + cd
        if (!fusionScheduled) {
            fusionScheduled = true
            // 等 Image 也到齐（最多 700ms）再做融合
            handler.postDelayed({ fuseCurrentBurst() }, 700L)
        }
    }

    // ------------------------------------------------- Image / CaptureResult 配对

    private fun acquireNext(reader: ImageReader): Image? {
        return try {
            reader.acquireNextImage()
        } catch (e: IllegalStateException) {
            null
        } catch (e: Exception) {
            null
        }
    }

    private fun onJpegAvailable(reader: ImageReader) {
        val image = acquireNext(reader) ?: return
        var deferred = false
        try {
            if (!active) {
                stats.imageResultExpired++
                return
            }
            val ts = image.timestamp
            var result: TotalCaptureResult? = null
            synchronized(matcherLock) {
                result = pendingResults.remove(ts)
                if (result == null) {
                    // Result 还没到：先挂起 Image，等 Result 到达后再配对
                    pendingImages[ts] = PendingImage(image, isRaw = false, arrivedNs = System.nanoTime())
                    deferred = true
                    stats.imageResultLate++
                }
            }
            if (!deferred) {
                stats.imageResultMatched++
                saveJpeg(image, ts)
            }
        } catch (e: Exception) {
            stats.imageResultLate++
            stats.lastCaptureRejectReason = "jpeg: ${e.message}"
        } finally {
            if (!deferred) {
                runCatching { image.close() }
            }
        }
    }

    private fun onRawAvailable(reader: ImageReader) {
        val image = acquireNext(reader) ?: return
        var deferred = false
        try {
            if (!active) {
                stats.imageResultExpired++
                return
            }
            val ts = image.timestamp
            var result: TotalCaptureResult? = null
            synchronized(matcherLock) {
                result = pendingResults.remove(ts)
                if (result == null) {
                    pendingImages[ts] = PendingImage(image, isRaw = true, arrivedNs = System.nanoTime())
                    deferred = true
                    stats.imageResultLate++
                }
            }
            if (!deferred) {
                stats.imageResultMatched++
                saveDng(image, ts, result)
            }
        } catch (e: Exception) {
            stats.imageResultExpired++
            stats.lastCaptureRejectReason = "raw: ${e.message}"
        } finally {
            if (!deferred) {
                runCatching { image.close() }
            }
        }
    }

    private fun consumeImage(pending: PendingImage, ts: Long, result: TotalCaptureResult?) {
        try {
            if (pending.isRaw) {
                saveDng(pending.image, ts, result)
            } else {
                saveJpeg(pending.image, ts)
            }
        } catch (e: Exception) {
            stats.lastCaptureRejectReason =
                (if (pending.isRaw) "raw: " else "jpeg: ") + e.message
        } finally {
            runCatching { pending.image.close() }
        }
    }

    /** 超时兜底：Result 始终没来的 Image 不能一直占着缓冲 */
    private fun sweepPendingImages(nowNs: Long) {
        if (pendingImages.isEmpty()) {
            return
        }
        val stale = mutableListOf<PendingImage>()
        synchronized(matcherLock) {
            val it = pendingImages.entries.iterator()
            while (it.hasNext()) {
                val entry = it.next()
                if (nowNs - entry.value.arrivedNs > IMAGE_RESULT_HOLD_NS) {
                    stale.add(entry.value)
                    it.remove()
                }
            }
        }
        for (p in stale) {
            stats.imageResultExpired++
            if (p.isRaw) {
                // RAW 没有 TotalCaptureResult 就无法生成合法 DNG
                runCatching { p.image.close() }
            } else {
                consumeImage(p, p.image.timestamp, null)
            }
        }
    }

    private fun saveJpeg(image: Image, ts: Long) {
        val bytes = ByteArray(image.planes[0].buffer.remaining())
        image.planes[0].buffer.get(bytes)
        // burst 成员先落在临时目录，融合结束后只保留「参考帧 + 融合结果」
        val file = File(burstTmpDir(currentSessionId), "burst_${ts}.jpg")
        FileOutputStream(file).use { it.write(bytes) }
        stats.jpegSaved++
        synchronized(currentBurstJpegs) {
            currentBurstJpegs.add(file.absolutePath)
        }
    }

    private fun saveDng(image: Image, ts: Long, result: TotalCaptureResult?) {
        if (result == null) {
            stats.imageResultExpired++
            return
        }
        if (burstRawSavedThisRound > 0) {
            // 正常模式只保留一张参考 RAW，其余不落盘
            stats.rawSkipped++
            return
        }
        burstRawSavedThisRound++
        val dir = captureDir(currentSessionId)
        val file = File(dir, "reference_${ts}.dng")
        val dng = DngCreator(characteristics, result)
        try {
            dng.setDescription("MobileScan3D HQ RAW keyframe")
            FileOutputStream(file).use { out -> dng.writeImage(out, image) }
            stats.rawDngSaved++
        } finally {
            dng.close()
        }
    }

    // ------------------------------------------------------------------- fusion

    private fun fuseCurrentBurst() {
        val paths = synchronized(currentBurstJpegs) { currentBurstJpegs.toList() }
        if (paths.size < 2) {
            stats.lastCaptureRejectReason = "burst images < 2"
            fusionScheduled = false
            cleanupBurstTmp()
            scheduleRetake()
            return
        }

        val outDir = captureDir(currentSessionId)
        // 只保留一张参考 JPEG 作为交付样本
        val referenceSrc = File(paths.first())
        if (referenceSrc.exists()) {
            val referenceDst = File(outDir, "reference_${currentBurstToken}.jpg")
            runCatching {
                referenceSrc.copyTo(referenceDst, overwrite = true)
                stats.referenceJpeg = referenceDst.absolutePath
            }
        }

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

        // 严格对齐门限：宁可少融合，也不要把重影糊边的帧塞进去
        val reject = when {
            !ok -> "fusion failed"
            stats.underexposedPercent > 20f -> "underexposed"
            stats.clippedPercent > 5f -> "overexposed"
            stats.alignmentInliers < MIN_ALIGN_INLIERS ->
                "alignment inliers=${stats.alignmentInliers}(<$MIN_ALIGN_INLIERS)"
            stats.alignmentRmsePx > MAX_ALIGN_RMSE_PX ->
                "alignment rmse=${"%.2f".format(stats.alignmentRmsePx)}px(>$MAX_ALIGN_RMSE_PX)"
            stats.eccScore < MIN_ALIGN_ECC ->
                "ecc=${"%.3f".format(stats.eccScore)}(<$MIN_ALIGN_ECC)"
            stats.acceptedFrames < 3 -> "too few accepted frames(${stats.acceptedFrames})"
            stats.confidenceMean < 0.52f -> "low confidence"
            else -> ""
        }

        if (ok && out.exists()) {
            if (reject.isEmpty()) {
                stats.fusedImageSaved++
            } else {
                // 质量不合格的融合结果直接丢弃，避免污染交付目录
                out.delete()
            }
        }

        if (reject.isNotEmpty()) {
            stats.lastCaptureRejectReason = reject
            scheduleRetake()
        } else {
            stats.lastCaptureRejectReason = ""
        }

        fusionScheduled = false
        cleanupBurstTmp()
    }

    private fun cleanupBurstTmp() {
        runCatching {
            val dir = burstTmpDir(currentSessionId)
            dir.listFiles()?.forEach { it.delete() }
        }
    }

    private fun scheduleRetake() {
        stats.burstCooldownMs = RETRY_COOLDOWN_NS / 1_000_000L
        nextBurstAtNs = System.nanoTime() + RETRY_COOLDOWN_NS
        val msg = when {
            stats.lastCaptureRejectReason.startsWith("underexposed") -> "光线不足，请增加照明后保持稳定"
            stats.lastCaptureRejectReason.startsWith("overexposed") -> "高光过曝，请调整拍摄角度"
            stats.lastCaptureRejectReason.startsWith("alignment") -> "关键帧对齐失败，请更慢地移动手机"
            stats.lastCaptureRejectReason.startsWith("ecc") -> "关键帧配准不足，请更慢地移动手机"
            stats.lastCaptureRejectReason.startsWith("low confidence") -> "关键帧质量不足，请保持稳定后自动补拍"
            else -> "关键帧补拍：请保持手机稳定"
        }
        onHint(msg)
    }

    // ------------------------------------------------------------------ motion

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

    /** VINS 位姿在最近 200ms 内的平移量（不是速度，避免采样间隔影响） */
    private fun computeTranslationDisplacement200ms(nowNs: Long): Float {
        val pose = FloatArray(12)
        if (!NativeBridge.nativeGetRenderPose(pose)) return 999f
        synchronized(poseLock) {
            poseHistory.addLast(PoseSample(nowNs, pose.copyOf()))
            while (poseHistory.size > 120) poseHistory.removeFirst()
            while (poseHistory.size > 2 && nowNs - poseHistory.first.timestampNs > 400_000_000L) {
                poseHistory.removeFirst()
            }
            val cutoff = nowNs - 200_000_000L
            val ref = poseHistory.firstOrNull { it.timestampNs >= cutoff } ?: return 0f
            val dx = pose[9] - ref.pose[9]
            val dy = pose[10] - ref.pose[10]
            val dz = pose[11] - ref.pose[11]
            return hypot(hypot(dx, dy), dz)
        }
    }

    private fun novelViewpoint(): Boolean {
        if (!haveLastBurstPose) return true
        val pose = FloatArray(12)
        if (!NativeBridge.nativeGetRenderPose(pose)) return false
        val dx = pose[9] - lastBurstPose[9]
        val dy = pose[10] - lastBurstPose[10]
        val dz = pose[11] - lastBurstPose[11]
        stats.lastBurstPoseDelta = hypot(hypot(dx, dy), dz)
        stats.lastBurstAngleDeltaDeg = rotationAngleDeg(lastBurstPose, pose)
        return stats.lastBurstPoseDelta >= MIN_NEW_VIEW_TRANSLATION ||
            stats.lastBurstAngleDeltaDeg >= MIN_NEW_VIEW_ANGLE_DEG
    }

    /** trace(R_a^T · R_b) -> 旋转角 */
    private fun rotationAngleDeg(a: FloatArray, b: FloatArray): Float {
        var tr = 0f
        for (k in 0 until 9) {
            tr += a[k] * b[k]
        }
        val c = ((tr - 1f) / 2f).coerceIn(-1f, 1f)
        return Math.toDegrees(acos(c).toDouble()).toFloat()
    }

    // ------------------------------------------------------------------ report

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
        sb.appendLine("lockRequested=${stats.lockRequested}")
        sb.appendLine("lockVerifyStreak=${stats.lockVerifyStreak}")
        sb.appendLine("lockAttempts=${stats.lockAttempts}")
        sb.appendLine("lockFailReason=${stats.lockFailReason}")
        sb.appendLine("AE state=${stats.aeState}")
        sb.appendLine("AE locked=${stats.aeLocked}")
        sb.appendLine("AE verified=${stats.aeVerified}")
        sb.appendLine("AWB state=${stats.awbState}")
        sb.appendLine("AWB locked=${stats.awbLocked}")
        sb.appendLine("AWB verified=${stats.awbVerified}")
        sb.appendLine("AF state=${stats.afState}")
        sb.appendLine("AF locked=${stats.afLocked}")
        sb.appendLine("AF verified=${stats.afVerified}")
        sb.appendLine("exposureNs=${stats.exposureNs}")
        sb.appendLine("ISO=${stats.iso}")
        sb.appendLine("focusDiopters=${stats.focusDiopters}")
        sb.appendLine("captureCycles=${stats.captureCycles}")
        sb.appendLine("burstRequested=${stats.burstRequested}")
        sb.appendLine("burstCompleted=${stats.burstCompleted}")
        sb.appendLine("burstDropped=${stats.burstDropped}")
        sb.appendLine("burstSkipCount=${stats.burstSkipCount}")
        sb.appendLine("burstCooldownMs=${stats.burstCooldownMs}")
        sb.appendLine("triggerRejectReason=${stats.triggerRejectReason}")
        sb.appendLine("imageResultMatched=${stats.imageResultMatched}")
        sb.appendLine("imageResultLate=${stats.imageResultLate}")
        sb.appendLine("imageDeferredMatched=${stats.imageDeferredMatched}")
        sb.appendLine("imageResultExpired=${stats.imageResultExpired}")
        sb.appendLine("rawSkipped=${stats.rawSkipped}")
        sb.appendLine("gyroRms=${stats.gyroRms}")
        sb.appendLine("vinsDisplacement200ms=${stats.vinsDisplacement200ms}")
        sb.appendLine("translationDuringBurst=${stats.translationDuringBurst}")
        sb.appendLine("lastBurstPoseDelta=${stats.lastBurstPoseDelta}")
        sb.appendLine("lastBurstAngleDeltaDeg=${stats.lastBurstAngleDeltaDeg}")
        sb.appendLine("previewClippedPercent=${stats.previewClippedPercent}")
        sb.appendLine("previewUnderexposedPercent=${stats.previewUnderexposedPercent}")
        sb.appendLine("previewLumaSamples=${stats.previewLumaSamples}")
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
        sb.appendLine("referenceJpeg=${stats.referenceJpeg}")
        sb.appendLine("lastCaptureRejectReason=${stats.lastCaptureRejectReason}")
        sb.appendLine("hotPixelMapSupported=${caps.hotPixelMapSupported}")
        sb.appendLine("lensShadingMapSupported=${caps.lensShadingMapSupported}")
        sb.appendLine("rawSizes=${caps.rawSizes.joinToString(",") { "${it.width}x${it.height}" }}")
        sb.appendLine("jpegSizes=${caps.jpegSizes.take(8).joinToString(",") { "${it.width}x${it.height}" }}")
        sb.appendLine("yuvSizes=${caps.yuvSizes.take(8).joinToString(",") { "${it.width}x${it.height}" }}")
    }

    // ------------------------------------------------------------------ helpers

    private fun captureDir(sessionId: String): File {
        val dir = File(context.filesDir, "hq_capture/$sessionId")
        dir.mkdirs()
        return dir
    }

    private fun burstTmpDir(sessionId: String): File {
        val dir = File(captureDir(sessionId), BURST_TMP_DIR)
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
