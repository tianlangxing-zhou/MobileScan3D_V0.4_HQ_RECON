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
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.RggbChannelVector
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
    var threeAReady: Boolean = false,
    var exposureNs: Long = 0L,
    var iso: Int = 0,
    var focusDiopters: Float = 0f,
    // 本次 burst 真正下发的手动冻结参数。
    // 评审要求：不要再让报告去解读含糊的「AE locked=true/false」，
    // 直接说明这一次 burst 用的是哪套固定参数、值是多少。
    var burstExposureMode: String = "NONE",
    var burstExposureNs: Long = 0L,
    var burstIso: Int = 0,
    var burstFocusDiopters: Float = 0f,
    var burstAwbMode: String = "NONE",
    var burstRequested: Int = 0,
    var burstCompleted: Int = 0,
    var burstDropped: Int = 0,
    var burstStarted: Long = 0L,
    var captureCycles: Int = 0,
    // 调度门控：每次尝试触发都必然归入其中一个拒绝计数器，
    // 否则「burstStarted 远低于预期」时无法判断是卡在哪一道门上。
    var schedulerCandidates: Long = 0L,
    var schedulerAccepted: Long = 0L,
    var reject3A: Long = 0L,
    var rejectCooldown: Long = 0L,
    var rejectExposure: Long = 0L,
    var rejectMotion: Long = 0L,
    var burstSkipCount: Long = 0L,
    var burstCooldownMs: Long = 0L,
    var triggerRejectReason: String = "",
    // Image <-> CaptureResult 配对：JPEG 与 RAW 分开统计，因为 RAW 没有
    // TotalCaptureResult 就无法生成 DNG，超时后果与 JPEG 不同。
    var jpegReceived: Long = 0L,
    var jpegMatched: Long = 0L,
    var jpegExpired: Long = 0L,
    var rawReceived: Long = 0L,
    var rawMatched: Long = 0L,
    var rawExpired: Long = 0L,
    var pairDeferredMatched: Long = 0L,
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
    var alignmentRmseMean: Float = 0f,
    var alignmentRmseMax: Float = 0f,
    var eccMean: Float = 0f,
    var rejectedAlignment: Int = 0,
    var rejectedEcc: Int = 0,
    var rejectedDecode: Int = 0,
    /** burst 内最清晰帧的下标与清晰度（降级交付时选的就是它） */
    var sharpestIndex: Int = -1,
    var sharpestScore: Float = 0f,
    var inputFrames: Int = 0,
    var acceptedFrames: Int = 0,
    var rejectedFrames: Int = 0,
    var sharpness: Float = 0f,
    var clippedPercent: Float = 0f,
    var underexposedPercent: Float = 0f,
    var noiseSigma: Float = 0f,
    var confidenceMean: Float = 0f,
    var confidenceLowPercent: Float = 0f,
    // 交付结果：fusionSuccess = 真融合；fusionFallbackSingle = 对齐帧数不足，
    // 降级为「最清晰单帧」。两者都算成功交付了一个关键帧。
    var fusionSuccess: Long = 0L,
    var fusionFallbackSingle: Long = 0L,
    var fallbackSingleJpeg: String = "",
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

/**
 * 一次 burst 使用的「冻结 3A」参数快照。
 *
 * 为什么不让 burst 直接沿用 AE/AWB/AF 的 lock：
 * 实机上出现过 `captureState=SCAN_LOCKED` 但 `AE/AWB/AF locked=false` 的矛盾状态 ——
 * 触发锁的请求发出后，硬件未必真的进入 LOCKED，于是「锁定」这件事既不可信也不可复现。
 *
 * 评审推荐的做法是：让自动 3A 先收敛，在收敛那一刻把参数抓下来，
 * 之后 still burst 一律用这套固定手动参数下发。这样每张 burst 帧的曝光、
 * 白平衡、对焦在物理上就是同一个值，融合时不会有亮度/色调跳变，
 * 也绕开了「lock 是否真的生效」这个不可控因素。
 */
private data class Locked3A(
    val haveExposure: Boolean,
    val exposureNs: Long,
    val iso: Int,
    val frameDurationNs: Long,
    val haveFocus: Boolean,
    val focusDiopters: Float,
    val haveAwb: Boolean,
    val gains: RggbChannelVector?,
    val transform: ColorSpaceTransform?
)

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
        /** 触发前的曝光预检门限：clipping 与暗部同时达标才允许发 still capture */
        private const val MAX_PREVIEW_CLIP_PERCENT = 5.0f
        private const val MAX_PREVIEW_UNDEREXPOSED_PERCENT = 20.0f
        /** 已锁定时若高光仍严重过曝，解锁重收敛 */
        private const val OVEREXPOSURE_RECOVER_PERCENT = 6.0f
        /**
         * burst 内「最清晰一帧」的清晰度下限（640 长边灰度上的 Laplacian 方差）。
         * 低于它说明整组都没有一张能看的，与其交付糊图不如重拍。
         * 报告里会同时输出实际值，便于实机标定。
         */
        private const val MIN_SHARPEST_SCORE = 20.0f
        private const val MAX_EXPOSURE_RECOVERY_ATTEMPTS = 2
        private const val EXPOSURE_RECOVERY_COOLDOWN_NS = 3_000_000_000L
        /** 新视角判定：平移或转角超过其一才值得再拍 */
        private const val MIN_NEW_VIEW_TRANSLATION = 0.015f
        private const val MIN_NEW_VIEW_ANGLE_DEG = 3.0f
        /** 严格对齐门限 */
        private const val MIN_ALIGN_INLIERS = 30f
        private const val MAX_ALIGN_RMSE_PX = 1.5f
        private const val MIN_ALIGN_ECC = 0.95f
        /** burst 固定帧数：第 0 帧 JPEG+RAW，第 1..4 帧只出 JPEG */
        private const val BURST_FRAMES = 5
        /** 通过严格对齐的帧少于这个数就不融合，降级为最清晰单帧 */
        private const val MIN_FUSION_FRAMES = 3
        /**
         * Image 与 CaptureResult 配对的超时，判据在 SENSOR_TIMESTAMP 时间域上。
         * 两者都来自 sensor 时钟，直接相减即可；混用 System.currentTimeMillis()
         * 会因为时钟域不同而出现「永远配不上」或「瞬间过期」两种极端。
         */
        private const val PAIR_TIMEOUT_NS = 1_500_000_000L
        /** 还没有任何 CaptureResult 到达时的墙钟兜底，防止 Image 永久占着缓冲 */
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
    /** 最近一个「3A 已收敛」的 CaptureResult，冻结参数从这里抓取 */
    private var lastConvergedResult: TotalCaptureResult? = null
    /** 自动 3A 收敛那一刻抓下来的冻结参数，burst 全程复用 */
    private var locked3a: Locked3A? = null
    /** 最新 CaptureResult 的 SENSOR_TIMESTAMP，配对超时用它做时间基准 */
    private var lastResultSensorTs = 0L

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
        stats.burstStarted = 0L
        stats.schedulerCandidates = 0L
        stats.schedulerAccepted = 0L
        stats.reject3A = 0L
        stats.rejectCooldown = 0L
        stats.rejectExposure = 0L
        stats.rejectMotion = 0L
        stats.jpegReceived = 0L
        stats.jpegMatched = 0L
        stats.jpegExpired = 0L
        stats.rawReceived = 0L
        stats.rawMatched = 0L
        stats.rawExpired = 0L
        stats.pairDeferredMatched = 0L
        stats.rawSkipped = 0L
        stats.rawDngSaved = 0L
        stats.jpegSaved = 0L
        stats.fusedImageSaved = 0L
        stats.fusionSuccess = 0L
        stats.fusionFallbackSingle = 0L
        stats.fallbackSingleJpeg = ""
        stats.referenceJpeg = ""
        stats.burstExposureMode = "NONE"
        stats.burstAwbMode = "NONE"
        stats.lastCaptureRejectReason = ""
        stats.triggerRejectReason = ""
        stats.lockFailReason = ""
        stats.lockAttempts = 0
        stats.lockVerifyStreak = 0
        stats.threeAReady = false
        locked3a = null
        lastConvergedResult = null
        lastResultSensorTs = 0L
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
        stats.threeAReady = readyStreak >= READY_STREAK
        if (converged) {
            // 只有「正在收敛」的结果才是冻结参数的可靠来源：这时 AE/AWB/AF 的数值
            // 全部是硬件自己算出来的，而不是我们下发到一半的锁定请求造成的中间态。
            lastConvergedResult = result
        }

        advanceStateMachine(System.nanoTime())

        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        // 配对超时统一在 SENSOR_TIMESTAMP 时间域上判断，基准就是这里维护的最大值
        if (ts > lastResultSensorTs) lastResultSensorTs = ts
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
            stats.pairDeferredMatched++
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

    // ------------------------------------------------------------- 冻结 3A 快照

    /**
     * 从「3A 已收敛」的 CaptureResult 里抓取参数，生成一次 burst 使用的冻结快照。
     *
     * 每一项能否冻结由设备能力决定，拿不到就如实标记为 false，由
     * [buildLockedStillRequest] 回退到 lock/auto，并在报告里写明实际用的是哪套。
     * 不假装冻结 —— 报告里 `burstExposureMode` 必须反映真正下发的东西。
     */
    private fun snapshot3A(): Locked3A {
        val r = lastConvergedResult
        val expNs = r?.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: stats.exposureNs
        val iso = r?.get(CaptureResult.SENSOR_SENSITIVITY) ?: stats.iso
        val focus = r?.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: stats.focusDiopters
        val gains = r?.get(CaptureResult.COLOR_CORRECTION_GAINS)
        val transform = r?.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)

        // 曝光：需要 MANUAL_SENSOR 能力 + 有效的曝光时间与 ISO
        val haveExposure = caps.manualSensor && expNs > 0L && iso > 0
        // 对焦：需要可写 LENS_FOCUS_DISTANCE，且收敛时确实拿到了一个有限焦距
        val haveFocus = caps.manualFocus && focus > 0f
        // 白平衡：TRANSFORM_MATRIX 模式需要 MANUAL_POST_PROCESSING，
        // 且必须有一个合法的颜色矩阵（增益是可选的，缺失时只用矩阵）
        val haveAwb = caps.manualPostProcessing && transform != null

        return Locked3A(
            haveExposure = haveExposure,
            exposureNs = expNs,
            iso = iso.coerceAtLeast(100),
            // 帧时长必须不小于曝光时间，否则请求非法；留 16ms 余量覆盖读出时间
            frameDurationNs = expNs + 16_000_000L,
            haveFocus = haveFocus,
            focusDiopters = focus,
            haveAwb = haveAwb,
            gains = gains,
            transform = transform
        )
    }

    /**
     * 构造一帧 burst 的 still capture 请求。
     *
     * [includeRaw] 只对第 0 帧为 true：JPEG+RAW 各一份，其余帧只出 JPEG。
     * 5 张 RAW 纯属浪费带宽和存储，而 DNG 只需要一张作为色彩/线性参考。
     */
    private fun buildLockedStillRequest(camera: CameraDevice, includeRaw: Boolean): CaptureRequest {
        val frozen = locked3a ?: snapshot3A()
        val b = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
        jpegReader?.surface?.let { b.addTarget(it) }
        if (includeRaw) {
            rawReader?.surface?.let { b.addTarget(it) }
        }
        b.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)

        // --- AE：优先手动冻结 ---
        if (frozen.haveExposure) {
            b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
            b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, frozen.exposureNs)
            b.set(CaptureRequest.SENSOR_SENSITIVITY, frozen.iso)
            b.set(CaptureRequest.SENSOR_FRAME_DURATION, frozen.frameDurationNs)
        } else {
            b.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            if (caps.aeLock) b.set(CaptureRequest.CONTROL_AE_LOCK, true)
        }

        // --- AWB：优先手动冻结 ---
        // CONTROL_AWB_MODE=OFF 时按规范必须用 COLOR_CORRECTION_MODE_TRANSFORM_MATRIX，
        // 否则整帧色调会漂到未定义的状态。
        if (frozen.haveAwb) {
            b.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_OFF)
            b.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX
            )
            frozen.gains?.let { b.set(CaptureRequest.COLOR_CORRECTION_GAINS, it) }
            frozen.transform?.let { b.set(CaptureRequest.COLOR_CORRECTION_TRANSFORM, it) }
        } else if (caps.awbLock) {
            b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        }

        // --- AF：优先手动冻结 ---
        if (frozen.haveFocus) {
            b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
            b.set(CaptureRequest.LENS_FOCUS_DISTANCE, frozen.focusDiopters)
        } else {
            // 拿不到可写焦距时，只能让硬件自己决定。这里显式选一个连续的自动模式，
            // 而不是指望 TEMPLATE_STILL_CAPTURE 的默认值合适。
            val modes = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
                ?: intArrayOf()
            val fallback = when {
                modes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE) ->
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                modes.contains(CaptureRequest.CONTROL_AF_MODE_AUTO) ->
                    CaptureRequest.CONTROL_AF_MODE_AUTO
                else -> CaptureRequest.CONTROL_AF_MODE_OFF
            }
            b.set(CaptureRequest.CONTROL_AF_MODE, fallback)
            if (fallback == CaptureRequest.CONTROL_AF_MODE_OFF && caps.manualFocus) {
                // AF 关闭时按规范必须给出焦距；0 表示对无穷远
                b.set(CaptureRequest.LENS_FOCUS_DISTANCE, 0f)
            }
        }

        b.set(CaptureRequest.JPEG_QUALITY, 100.toByte())
        b.set(
            CaptureRequest.JPEG_ORIENTATION,
            characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        )
        return b.build()
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
            stats.threeAReady = false
            // 退回重新收敛后，之前抓的冻结参数已经不对应当前光照，必须丢弃重抓
            locked3a = null
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

        // 每一个「真正评估过的触发机会」计一次 candidate，被挡住的必定落进某个
        // reject 计数器。这样 burstStarted 远低于预期时，一眼就能看出卡在哪道门，
        // 而不是像之前那样只能看到 burstStarted 和一堆模糊的 skip。
        stats.schedulerCandidates++

        if (currentPhase != STATE_SCAN_LOCKED) {
            stats.reject3A++
            stats.triggerRejectReason = "state=${currentPhase}"
            return
        }
        if (!stats.threeAReady) {
            stats.reject3A++
            stats.triggerRejectReason = "3A 未收敛"
            return
        }
        if (camera == null || session == null || !active) {
            stats.reject3A++
            stats.triggerRejectReason = "capture session 未就绪"
            return
        }

        // 1) 冷却（0.8~1.2s）。放在其它门之前：冷却期内逐帧评估没有意义，
        // 只会把运动/曝光计数灌满噪声。
        if (nowNs < nextBurstAtNs) {
            stats.rejectCooldown++
            stats.triggerRejectReason = "cooldown"
            return
        }

        // 2) 手持稳定：陀螺 RMS + VINS 200ms 位移
        if (stats.gyroRms > MAX_GYRO_RMS) {
            stats.rejectMotion++
            stats.triggerRejectReason = "gyroRms=${"%.3f".format(stats.gyroRms)}"
            stats.burstSkipCount++
            return
        }
        if (stats.vinsDisplacement200ms > MAX_VINS_DISP_200MS) {
            stats.rejectMotion++
            stats.triggerRejectReason = "vinsDisp200ms=${"%.3f".format(stats.vinsDisplacement200ms)}"
            stats.burstSkipCount++
            return
        }

        // 3) 新视角是否值得拍（没有新增视角信息，拍出来也是重冗余）
        if (!novelViewpoint()) {
            stats.rejectMotion++
            stats.triggerRejectReason =
                "viewpoint 未变化(dT=${"%.4f".format(stats.lastBurstPoseDelta)}, dR=${"%.1f".format(stats.lastBurstAngleDeltaDeg)}°)"
            stats.burstSkipCount++
            return
        }

        // 4) 曝光质量前置：这里就把明显过曝/欠曝的场景挡掉，
        // 而不是等拍完 burst 才在报告里评论一句「这张过曝了」。
        if (stats.previewClippedPercent > MAX_PREVIEW_CLIP_PERCENT) {
            stats.rejectExposure++
            stats.triggerRejectReason =
                "overexposed(预检 clipping=${"%.1f".format(stats.previewClippedPercent)}%)"
            stats.burstSkipCount++
            nextBurstAtNs = nowNs + RETRY_COOLDOWN_NS
            return
        }
        if (stats.previewLumaSamples > 0 &&
            stats.previewUnderexposedPercent > MAX_PREVIEW_UNDEREXPOSED_PERCENT
        ) {
            stats.rejectExposure++
            stats.triggerRejectReason =
                "underexposed(预检 dark=${"%.1f".format(stats.previewUnderexposedPercent)}%)"
            stats.burstSkipCount++
            nextBurstAtNs = nowNs + RETRY_COOLDOWN_NS
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
        val count = BURST_FRAMES
        // 冻结参数理论上已在 APPLY_LOCK 时抓好；万一缺失（例如中途丢过结果）就现抓一份
        val frozen = locked3a ?: snapshot3A().also { locked3a = it }

        stats.burstExposureMode = if (frozen.haveExposure) "MANUAL_FROZEN" else if (caps.aeLock) "AE_LOCK" else "AE_AUTO"
        stats.burstExposureNs = if (frozen.haveExposure) frozen.exposureNs else stats.exposureNs
        stats.burstIso = if (frozen.haveExposure) frozen.iso else stats.iso
        stats.burstFocusDiopters = if (frozen.haveFocus) frozen.focusDiopters else 0f
        stats.burstAwbMode = if (frozen.haveAwb) "MANUAL_FROZEN" else if (caps.awbLock) "AWB_LOCK" else "AWB_AUTO"

        stats.burstRequested += count
        stats.burstStarted++
        stats.schedulerAccepted++
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
            // 只有第 0 帧需要 RAW：一张 DNG 足够做线性/色彩参考，
            // 5 张 RAW 会把 burst 的带宽和落盘时间翻好几倍。
            val requests = (0 until count).map { i ->
                buildLockedStillRequest(camera, includeRaw = (i == 0) && rawReader != null)
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
        stats.jpegReceived++
        var deferred = false
        try {
            if (!active) {
                stats.jpegExpired++
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
                }
            }
            if (!deferred) {
                stats.jpegMatched++
                saveJpeg(image, ts)
            }
        } catch (e: Exception) {
            stats.jpegExpired++
            stats.lastCaptureRejectReason = "jpeg: ${e.message}"
        } finally {
            if (!deferred) {
                runCatching { image.close() }
            }
        }
    }

    private fun onRawAvailable(reader: ImageReader) {
        val image = acquireNext(reader) ?: return
        stats.rawReceived++
        var deferred = false
        try {
            if (!active) {
                stats.rawExpired++
                return
            }
            val ts = image.timestamp
            var result: TotalCaptureResult? = null
            synchronized(matcherLock) {
                result = pendingResults.remove(ts)
                if (result == null) {
                    pendingImages[ts] = PendingImage(image, isRaw = true, arrivedNs = System.nanoTime())
                    deferred = true
                }
            }
            if (!deferred) {
                stats.rawMatched++
                saveDng(image, ts, result)
            }
        } catch (e: Exception) {
            stats.rawExpired++
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
                if (result == null) stats.rawExpired++ else stats.rawMatched++
                saveDng(pending.image, ts, result)
            } else {
                if (result == null) stats.jpegExpired++ else stats.jpegMatched++
                saveJpeg(pending.image, ts)
            }
        } catch (e: Exception) {
            stats.lastCaptureRejectReason =
                (if (pending.isRaw) "raw: " else "jpeg: ") + e.message
        } finally {
            runCatching { pending.image.close() }
        }
    }

    /**
     * 超时兜底：Result 始终没来的 Image 不能一直占着缓冲。
     *
     * 判据优先走 SENSOR_TIMESTAMP 时间域 —— 用「当前最新结果的 sensor 时间戳」减去
     * 「这张 Image 自己的 sensor 时间戳」。两块时间来自同一个 sensor 时钟，直接相减
     * 才有意义；混进 System.currentTimeMillis() 会因为时钟域不同而误判。
     * 只有在一条 Result 都还没回来（没有基准）时，才退回墙钟兜底。
     */
    private fun sweepPendingImages(nowNs: Long) {
        if (pendingImages.isEmpty()) {
            return
        }
        val stale = mutableListOf<PendingImage>()
        val sensorNow = lastResultSensorTs
        synchronized(matcherLock) {
            val it = pendingImages.entries.iterator()
            while (it.hasNext()) {
                val entry = it.next()
                val expired = if (sensorNow > 0L && entry.key > 0L && sensorNow >= entry.key) {
                    sensorNow - entry.key > PAIR_TIMEOUT_NS
                } else {
                    nowNs - entry.value.arrivedNs > IMAGE_RESULT_HOLD_NS
                }
                if (expired) {
                    stale.add(entry.value)
                    it.remove()
                }
            }
        }
        for (p in stale) {
            if (p.isRaw) {
                // RAW 没有 TotalCaptureResult 就无法生成合法 DNG
                stats.rawExpired++
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
            // 计数由调用方（consumeImage）统一负责，这里只负责不生成非法 DNG
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
        val out = File(outDir, "fused_${currentBurstToken}.jpg")
        // 参考帧 / 单帧降级的落盘都放到 applyFusionStats 里按结果决定，
        // 避免出现「reference_*.jpg」和「single_*.jpg」两份重复交付物。
        val statsOut = FloatArray(20)
        // 噪声估计用的是冻结后的实际曝光参数，不是可能已经刷新的 stats
        val frozen = locked3a
        val exposureMs = (if (frozen?.haveExposure == true) frozen.exposureNs else stats.exposureNs) / 1_000_000f
        val iso = if (frozen?.haveExposure == true) frozen.iso else stats.iso
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
                applyFusionStats(ok, out, statsOut, paths)
            }
        }.start()
    }

    private fun applyFusionStats(ok: Boolean, out: File, statsOut: FloatArray, paths: List<String>) {
        stats.inputFrames = if (statsOut.size > 9 && statsOut[9] > 0f) statsOut[9].toInt() else paths.size
        stats.sharpness = statsOut[0]
        stats.clippedPercent = statsOut[1]
        stats.underexposedPercent = statsOut[2]
        stats.noiseSigma = statsOut[3]
        stats.confidenceMean = statsOut[4]
        stats.confidenceLowPercent = statsOut[5]
        stats.alignmentInliers = statsOut[6]
        stats.alignmentRmseMean = statsOut[7]
        stats.eccMean = statsOut[8]
        stats.acceptedFrames = statsOut[10].toInt()
        stats.rejectedFrames = statsOut[11].toInt()
        if (statsOut.size > 12) stats.alignmentRmseMax = statsOut[12]
        val sharpestIndex = if (statsOut.size > 13) statsOut[13].toInt() else -1
        val sharpestScore = if (statsOut.size > 14) statsOut[14] else 0f
        val fusedWritten = if (statsOut.size > 15) statsOut[15] > 0.5f else ok
        if (statsOut.size > 16) stats.rejectedAlignment = statsOut[16].toInt()
        if (statsOut.size > 17) stats.rejectedEcc = statsOut[17].toInt()
        if (statsOut.size > 18) stats.rejectedDecode = statsOut[18].toInt()

        // 分三类处理，而不是笼统地「不合格就重拍」：
        //  1) 光照不合格 / 整组全糊  -> 重拍，融合也救不回来
        //  2) 光照没问题但对齐凑不够 -> 不融合，降级交付「最清晰单帧」
        //  3) 全部达标              -> 真融合
        val exposureReason = when {
            stats.underexposedPercent > 20f -> "underexposed"
            stats.clippedPercent > 5f -> "overexposed"
            else -> ""
        }
        // sharpestIndex < 0 表示连参考帧都没解码成功（native 提前返回），
        // 这时 sharpestScore 必然是 0，不能再把它当成「整组都糊」来解释。
        val decodeFailed = sharpestIndex < 0
        val tooBlurry = !decodeFailed && sharpestScore < MIN_SHARPEST_SCORE
        // 帧数是第一判据：评审要求 acceptedFrames < 3 就不融合。
        // 先判帧数，下面那些内点/残差/ECC 只有在帧数够了却依然不达标时才有解释力。
        val alignReject = when {
            stats.acceptedFrames < MIN_FUSION_FRAMES ->
                "too few accepted frames(${stats.acceptedFrames}/${stats.inputFrames})"
            stats.alignmentInliers < MIN_ALIGN_INLIERS ->
                "alignment inliers=${stats.alignmentInliers}(<$MIN_ALIGN_INLIERS)"
            stats.alignmentRmseMean > MAX_ALIGN_RMSE_PX ->
                "alignment rmse=${"%.2f".format(stats.alignmentRmseMean)}px(>$MAX_ALIGN_RMSE_PX)"
            stats.eccMean < MIN_ALIGN_ECC ->
                "ecc=${"%.3f".format(stats.eccMean)}(<$MIN_ALIGN_ECC)"
            stats.confidenceMean < 0.52f -> "low confidence"
            !fusedWritten -> "fusion did not produce output"
            else -> ""
        }

        var retake = false
        when {
            decodeFailed -> {
                if (out.exists()) out.delete()
                stats.lastCaptureRejectReason =
                    "reference frame decode failed(rejectedDecode=${stats.rejectedDecode})"
                retake = true
            }
            exposureReason.isNotEmpty() -> {
                if (out.exists()) out.delete()
                stats.lastCaptureRejectReason = exposureReason
                retake = true
            }
            tooBlurry -> {
                if (out.exists()) out.delete()
                stats.lastCaptureRejectReason =
                    "all frames blurry(sharpest=${"%.1f".format(sharpestScore)}<$MIN_SHARPEST_SCORE)"
                retake = true
            }
            alignReject.isNotEmpty() -> {
                if (out.exists()) out.delete()
                val picked = if (sharpestIndex in paths.indices) sharpestIndex else 0
                if (saveFallbackSingle(paths[picked])) {
                    stats.fusionFallbackSingle++
                    stats.lastCaptureRejectReason = "$alignReject -> 已降级交付最清晰单帧"
                } else {
                    stats.lastCaptureRejectReason = "$alignReject -> 最清晰单帧留取失败"
                    retake = true
                }
            }
            else -> {
                val refSrc = File(paths.first())
                if (refSrc.exists()) {
                    val refDst = File(captureDir(currentSessionId), "reference_${currentBurstToken}.jpg")
                    runCatching {
                        refSrc.copyTo(refDst, overwrite = true)
                        stats.referenceJpeg = refDst.absolutePath
                    }
                }
                stats.fusedImageSaved++
                stats.fusionSuccess++
                stats.lastCaptureRejectReason = ""
            }
        }

        fusionScheduled = false
        cleanupBurstTmp()
        if (retake) {
            scheduleRetake()
        }
    }

    /**
     * 对齐帧数不足以融合时的降级交付：把 burst 里最清晰的一帧直接留作关键帧。
     *
     * 评审要求「acceptedFrames < 3 不要融合，直接选最清晰的一张」——
     * 一张干净的单帧对重建的价值，高于一张重影糊边的伪融合图。
     */
    private fun saveFallbackSingle(srcPath: String): Boolean {
        val src = File(srcPath)
        if (!src.exists()) return false
        val dst = File(captureDir(currentSessionId), "single_${currentBurstToken}.jpg")
        return runCatching {
            src.copyTo(dst, overwrite = true)
            stats.fallbackSingleJpeg = dst.absolutePath
            true
        }.getOrDefault(false)
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
        sb.appendLine("3aReady=${stats.threeAReady}")
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
        // 本次 burst 真正下发的手动冻结参数：比「AE locked=true」这种含糊结论可信得多
        sb.appendLine("burstExposureMode=${stats.burstExposureMode}")
        sb.appendLine("burstExposureNs=${stats.burstExposureNs}")
        sb.appendLine("burstISO=${stats.burstIso}")
        sb.appendLine("burstFocusDiopters=${stats.burstFocusDiopters}")
        sb.appendLine("burstAwbMode=${stats.burstAwbMode}")
        sb.appendLine("captureCycles=${stats.captureCycles}")
        sb.appendLine("burstStarted=${stats.burstStarted}")
        sb.appendLine("burstRequested=${stats.burstRequested}")
        sb.appendLine("burstCompleted=${stats.burstCompleted}")
        sb.appendLine("burstDropped=${stats.burstDropped}")
        sb.appendLine("burstSkipCount=${stats.burstSkipCount}")
        sb.appendLine("burstCooldownMs=${stats.burstCooldownMs}")
        // 调度门控：candidates - accepted 应当等于下面四个 reject 之和
        sb.appendLine("schedulerCandidates=${stats.schedulerCandidates}")
        sb.appendLine("schedulerAccepted=${stats.schedulerAccepted}")
        sb.appendLine("reject3A=${stats.reject3A}")
        sb.appendLine("rejectCooldown=${stats.rejectCooldown}")
        sb.appendLine("rejectExposure=${stats.rejectExposure}")
        sb.appendLine("rejectMotion=${stats.rejectMotion}")
        sb.appendLine("triggerRejectReason=${stats.triggerRejectReason}")
        sb.appendLine("jpegReceived=${stats.jpegReceived}")
        sb.appendLine("jpegMatched=${stats.jpegMatched}")
        sb.appendLine("jpegExpired=${stats.jpegExpired}")
        sb.appendLine("rawReceived=${stats.rawReceived}")
        sb.appendLine("rawMatched=${stats.rawMatched}")
        sb.appendLine("rawExpired=${stats.rawExpired}")
        sb.appendLine("pairDeferredMatched=${stats.pairDeferredMatched}")
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
        sb.appendLine("alignmentRmseMean=${stats.alignmentRmseMean}")
        sb.appendLine("alignmentRmseMax=${stats.alignmentRmseMax}")
        sb.appendLine("eccMean=${stats.eccMean}")
        sb.appendLine("rejectedAlignment=${stats.rejectedAlignment}")
        sb.appendLine("rejectedEcc=${stats.rejectedEcc}")
        sb.appendLine("rejectedDecode=${stats.rejectedDecode}")
        sb.appendLine("inputFrames=${stats.inputFrames}")
        sb.appendLine("acceptedFrames=${stats.acceptedFrames}")
        sb.appendLine("rejectedFrames=${stats.rejectedFrames}")
        sb.appendLine("sharpness=${stats.sharpness}")
        sb.appendLine("sharpestIndex=${stats.sharpestIndex}")
        sb.appendLine("sharpestScore=${stats.sharpestScore}")
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
