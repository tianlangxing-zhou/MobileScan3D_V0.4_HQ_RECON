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
    // 状态机：IDLE / WAIT_3A / READY / CAPTURE
    var captureState: String = "IDLE",
    var sessionCombinationSupported: Boolean = false,
    var previewSize: String = "unknown",
    var analysisYuvSize: String = "unknown",
    var jpegSize: String = "unknown",
    var rawSize: String = "unknown",
    // 下面三项是「硬件实际报了什么」的纯观测值，不再参与任何决策
    var aeState: Int? = null,
    var aeLocked: Boolean = false,
    var awbState: Int? = null,
    var awbLocked: Boolean = false,
    var afState: Int? = null,
    var afLocked: Boolean = false,
    // 3A 收敛判据的逐项连续帧数：卡住时能立刻看出是哪一路没过，而不是只看一个总数
    var readyAeStreak: Int = 0,
    var readyAwbStreak: Int = 0,
    var readyAfStreak: Int = 0,
    var readyBlockReason: String = "",
    // 参数稳定性兜底：状态枚举不合格、但曝光/焦距实际已经稳住了
    var aeParamsStable: Boolean = false,
    var afParamsStable: Boolean = false,
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
    // 陀螺两个指标必须分开：magnitude 是真实角速度幅值（用于稳定门），
    // jitter 是旧算法（减均值后的 RMS，只反映抖动，仅作诊断保留）。
    var gyroMagnitudeRms: Float = 0f,
    var gyroJitterRms: Float = 0f,
    var vinsDisplacement200ms: Float = 0f,
    var translationDuringBurst: Float = 0f,
    // 新视角变化 = 上一次关键帧 -> 当前（不是 burst 内部运动）
    var viewpointDeltaMeters: Float = 0f,
    var viewpointDeltaDeg: Float = 0f,
    // burst 内部运动 = 第 0 帧 -> 最后一帧
    var burstTranslationMeters: Float = 0f,
    var burstRotationDeg: Float = 0f,
    var burstDurationMs: Long = 0L,
    var previewClippedPercent: Float = 0f,
    var previewUnderexposedPercent: Float = 0f,
    var previewLumaSamples: Int = 0,
    // 亮度分位：欠曝判定的主判据。只看「暗像素占比」会把黑色物体/阴影误判成欠曝。
    var lumaP05: Float = 0f,
    var lumaP50: Float = 0f,
    var lumaP95: Float = 0f,
    /** 已连续稳定的帧数（陀螺 + 位移同时达标） */
    var stableFrames: Int = 0,
    /** 分位判据给出的欠曝结论（P50/P95 双低），与暗像素占比无关 */
    var realUnderexposed: Boolean = false,
    var alignmentInliers: Float = 0f,
    var alignmentRmseMean: Float = 0f,
    var alignmentRmseMax: Float = 0f,
    var eccMean: Float = 0f,
    var rejectedAlignment: Int = 0,
    var rejectedEcc: Int = 0,
    var rejectedDecode: Int = 0,
    /** 参考帧（burst 第 0 张）自己解码失败：native 在 imread 之后就返回了 */
    var referenceDecodeFail: Int = 0,
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

/**
 * 一次 burst 的上下文。
 *
 * 每个 burst 有自己独立的 token 目录，融合只删自己的目录。
 * 旧实现所有 burst 共用 `hq_capture/<session>/tmp_burst/`，而 cleanupBurstTmp()
 * 会删掉整个目录 —— 上一组还在读 12MP JPEG 时，下一组已经把新文件写进同一目录，
 * 前一组清理时把后一组的照片一起删了，于是 fusion 的 imread(paths[0]) 返回空图。
 * 这正好解释了实机「10 张 JPEG 都收到，但 fusion 第一张就读不到」。
 */
private class BurstContext(
    val token: Long,
    val sessionId: String,
    val tempDir: File
) {
    /** 本组已落盘的 JPEG 路径，读写都持 [jpegLock] */
    val jpegPaths = mutableListOf<String>()
    val jpegLock = Any()
    /** 本组期望的帧数 */
    var expectedFrames = 0
    /** 正常模式只保留一张参考 RAW */
    var rawSaved = false
    var startPose: FloatArray? = null
    var startNs = 0L
}

/**
 * 一次 still capture 的请求 tag。
 *
 * 用它把 HQ 的 still result 与每秒 30 条 preview result 区分开：
 * preview result 的 request.tag 不是 BurstFrameTag，配对逻辑直接忽略，
 * 不会挤占配对表、也不会造成误配。
 */
private data class BurstFrameTag(
    val ctx: BurstContext,
    val index: Int,
    val expectsRaw: Boolean
)

/**
 * 一次 capture 的「配对桶」。
 *
 * Camera2 规定：同一次 capture 的所有输出 buffer 与它的 CaptureResult
 * 使用同一个 SENSOR_TIMESTAMP。第 0 帧同时输出 JPEG + RAW，于是
 * **JPEG.timestamp == RAW.timestamp == result.timestamp**。
 * 旧实现是「result 从 pendingResults 里 remove 一次就没了」＋
 * 「pendingImages 一个时间戳只能挂一张 Image」，第 0 帧的 RAW 因此永远
 * 拿不到 Result —— 实机 rawReceived=2 而 rawMatched=1 正是这么来的。
 * 改成一个时间戳一个桶，三样东西各自到位后整体消费。
 */
private class PendingCapture {
    var result: TotalCaptureResult? = null
    var jpeg: Image? = null
    var raw: Image? = null
    var ctx: BurstContext? = null
    var expectsRaw = false
    var frameIndex = -1
    val arrivedNs = System.nanoTime()
}

/** 3A 参数历史样本（用于「参数实际稳定」兜底判据） */
private class ThreeASample(val exposureNs: Long, val iso: Int, val focusD: Float)

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
        const val STATE_READY = "READY"
        const val STATE_CAPTURE = "CAPTURE"

        /**
         * 3A「已收敛」需要连续确认的帧数。
         *
         * 注意这里只要求收敛（AE CONVERGED / AWB CONVERGED / AF 已合焦），
         * **不要求硬件进入 LOCKED**。旧状态机额外要求 VERIFY_LOCK 三重 LOCKED，
         * 而锁定请求本身会把 preview 的 AE_MODE/AF_MODE 关成 OFF，
         * 关掉之后 AE_STATE/AF_STATE 只会报 INACTIVE —— 校验条件和它自己的副作用
         * 互相矛盾，于是 1366 次拍摄机会全部超时，一张 HQ 都没拍到。
         */
        private const val READY_STREAK = 2

        /**
         * 参数稳定性兜底的窗口与门限（见 exposureStable / focusStable）。
         * 6 帧 @30fps ≈ 200ms，足够判断「曝光/焦距还在不在动」。
         */
        private const val THREE_A_HISTORY = 6
        /** EV 代理 log2(exposure * iso) 的极差上限：0.12 EV ≈ 肉眼不可见 */
        private const val MAX_EV_SPREAD = 0.12
        /** focus diopter 的极差上限 */
        private const val MAX_FOCUS_SPREAD_DIOPTER = 0.04f
        /** burst 冷却区间 0.8~1.2s */
        private const val BURST_COOLDOWN_MIN_NS = 800_000_000L
        private const val BURST_COOLDOWN_MAX_NS = 1_200_000_000L
        private const val RETRY_COOLDOWN_NS = 1_500_000_000L
        /**
         * 稳定窗口门限：角速度**幅值** ≤ 0.08 rad/s 且 VINS 200ms 位移 ≤ 0.015。
         *
         * 角速度必须用 magnitude，不能用「减均值后的抖动」：手机以 0.3 rad/s
         * 平滑匀速扫过物体时，每个陀螺样本都≈0.3，减均值后≈0，
         * 旧算法会把「一直在匀速转」判成「很稳定」——恰恰是最该拒绝的场景。
         * 位移门限保持 0.015：HQ burst 最怕平移（视差直接让对齐失败）。
         */
        private const val MAX_GYRO_MAGNITUDE_RMS = 0.08f
        private const val MAX_VINS_DISP_200MS = 0.015f
        /**
         * 稳定窗口长度：4 帧 ≈ 130ms @30fps，落在评审建议的 100~200ms 区间。
         * 单帧稳定没有意义 —— 扫过物体时几乎每帧都在动，但只要用户稍微停一下，
         * 连续 4 帧就能满足，于是自动拍一组 HQ；继续移动则不拍。
         */
        private const val STABLE_WINDOW_FRAMES = 4
        /** 触发前的高光预检门限 */
        private const val MAX_PREVIEW_CLIP_PERCENT = 5.0f
        /**
         * 暗部兜底门限（放宽到 35%）。
         *
         * 「画面里 26% 像素很暗」不等于照片欠曝：拍黑色物体或背景有阴影时，
         * 本来就该有大量暗像素。主判据改用亮度分位（见下），这个百分比只做兜底。
         */
        private const val MAX_PREVIEW_UNDEREXPOSED_PERCENT = 35.0f
        /** 真正欠曝的分位判据：整体中位亮度低，且高光也上不去 */
        private const val MIN_LUMA_P50 = 35f
        private const val MIN_LUMA_P95 = 120f
        /** 冻结参数期间若高光仍严重过曝，放弃这套参数重新等待收敛 */
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
        /**
         * 融合兜底延迟。正常路径是「5 张 JPEG 全部落盘」立刻触发融合，
         * 这个定时器只负责在个别 Image 迟迟不到时把已到的部分送去融合，
         * 取代了旧的无条件固定 700ms 等待。
         */
        private const val FUSION_FALLBACK_DELAY_MS = 1500L
    }

    val caps: HqCameraCaps
    val stats = HqCaptureStats()

    private var jpegReader: ImageReader? = null
    private var rawReader: ImageReader? = null
    private var jpegSize: Size? = null
    private var rawSize: Size? = null
    private var active = false

    // HQ still capture 的配对表：一个 SENSOR_TIMESTAMP 一个桶，共用一把锁避免锁序反转。
    // 桶里同时容纳 result / jpeg / raw（三者时间戳相同）。
    private val matcherLock = Any()
    private val pendingCaptures = LinkedHashMap<Long, PendingCapture>()

    // 3A 参数稳定性历史
    private val historyLock = Any()
    private val threeAHistory = ArrayDeque<ThreeASample>()

    private val imuLock = Any()
    private val imuSamples = ArrayDeque<HqImuSample>()
    private val poseLock = Any()
    private val poseHistory = ArrayDeque<PoseSample>()

    // 预览曝光直方图（粗采样），用于触发前门控
    private val lumaLock = Any()
    private var previewClipEma = 0f
    private var previewUnderEma = 0f
    private var previewLumaInit = false
    /** 256 桶直方图，复用以免每帧分配（受 lumaLock 保护） */
    private val lumaHist = IntArray(256)

    // 状态机
    private var currentPhase = STATE_IDLE
    /** 3A 三条判据各自的连续满足帧数（分开是为了卡住时能定位到具体哪一路） */
    private var readyAeStreak = 0
    private var readyAwbStreak = 0
    private var readyAfStreak = 0
    /** 三条判据「同时」满足的连续帧数，达到 READY_STREAK 即认为 3A 已收敛 */
    private var readyStreak = 0
    /** 稳定窗口计数：陀螺与位移同时达标的连续帧数 */
    private var stableFrames = 0
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
    /**
     * 融合正在进行（或正在等最后几张 JPEG 落盘）。
     *
     * onFrameTick 用 `burstInFlight || processingInFlight` 作为触发门：
     * 一次 burst 从「开始拍摄」到「融合结束 + 临时目录清理完」之间不允许再开新的，
     * 否则上一组还在读 12MP JPEG 时下一组已经在改文件系统了。
     */
    @Volatile
    private var processingInFlight = false
    /** 当前这一组的上下文；融合结束并清理后置 null */
    private var activeBurst: BurstContext? = null
    /** 本组融合是否已经启动（保证只启动一次：要么凑齐、要么兜底超时） */
    private var fusionStarted = false
    private var nextBurstAtNs = 0L
    private var currentSessionId = "unknown"
    private var burstExpectedFrames = 0
    private var burstCompletedThisRound = 0
    private var burstFailedThisRound = 0

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
            pendingCaptures.values.forEach { discardCapture(it) }
            pendingCaptures.clear()
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
        stats.readyBlockReason = ""
        stats.threeAReady = false
        readyStreak = 0
        readyAeStreak = 0
        readyAwbStreak = 0
        readyAfStreak = 0
        stats.readyAeStreak = 0
        stats.readyAwbStreak = 0
        stats.readyAfStreak = 0
        stableFrames = 0
        stats.stableFrames = 0
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
            pendingCaptures.values.forEach { discardCapture(it) }
            pendingCaptures.clear()
        }
        synchronized(historyLock) { threeAHistory.clear() }
        burstExpectedFrames = 0
        processingInFlight = false
        activeBurst = null
        fusionStarted = false
        stats.aeParamsStable = false
        stats.afParamsStable = false
        cleanupAllBurstTmp(currentSessionId)
    }

    fun endScan() {
        setPhase(STATE_IDLE)
        burstInFlight = false
        processingInFlight = false
        activeBurst = null
        fusionStarted = false
    }

    fun close() {
        detachSurfaces()
        cleanupAllBurstTmp(currentSessionId)
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

        // 参数稳定性兜底：OnePlus 这类 HAL 会长时间停留在 AE_SEARCHING /
        // AF_PASSIVE_SCAN，但实际曝光/焦距早就没动了。只看状态枚举会把这种
        // 「其实已经稳了」的机会全部拒掉（上一版实机 readyAeStreak=0、
        // readyAfStreak=0，2207 次机会只成功 2 次）。
        // 所以额外维护一份参数历史：「严格状态合格」OR「参数实际稳定」，二者取或。
        synchronized(historyLock) {
            threeAHistory.addLast(
                ThreeASample(stats.exposureNs, stats.iso, stats.focusDiopters)
            )
            while (threeAHistory.size > THREE_A_HISTORY) threeAHistory.removeFirst()
        }
        val aeParamsStable = exposureStable()
        val afParamsStable = focusStable()
        stats.aeParamsStable = aeParamsStable
        stats.afParamsStable = afParamsStable

        // 只判「收敛」，不判「锁定」。锁定请求本身会把 preview 的 AE/AF 关成 OFF，
        // 关掉之后 AE_STATE/AF_STATE 只会报 INACTIVE —— 校验条件和它自己的副作用
        // 互相矛盾，那正是上一版 1366 次机会全部卡死的原因。
        val aeOk = aeReady(result) || aeParamsStable
        val awbOk = awbReady(result)
        val afOk = afReady(result) || afParamsStable
        stats.afLocked = result.get(CaptureResult.CONTROL_AF_STATE) ==
            CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED

        readyAeStreak = if (aeOk) readyAeStreak + 1 else 0
        readyAwbStreak = if (awbOk) readyAwbStreak + 1 else 0
        readyAfStreak = if (afOk) readyAfStreak + 1 else 0
        stats.readyAeStreak = readyAeStreak
        stats.readyAwbStreak = readyAwbStreak
        stats.readyAfStreak = readyAfStreak

        val converged = aeOk && awbOk && afOk
        readyStreak = if (converged) readyStreak + 1 else 0
        stats.threeAReady = readyStreak >= READY_STREAK
        // 卡住时要说清楚是哪一路没过，而不是只给一个总的 false
        stats.readyBlockReason = when {
            converged -> ""
            !aeOk && !awbOk && !afOk -> "AE+AWB+AF 均未收敛"
            !aeOk && !awbOk -> "AE+AWB 未收敛"
            !aeOk && !afOk -> "AE+AF 未收敛"
            !awbOk && !afOk -> "AWB+AF 未收敛"
            !aeOk -> "AE 未收敛(state=${stats.aeState})"
            !awbOk -> "AWB 未收敛(state=${stats.awbState})"
            else -> "AF 未合焦(state=${stats.afState})"
        }
        if (converged) {
            // 收敛时的结果才是冻结参数的可靠来源：这些数值是硬件自己算出来的，
            // 不是我们下发到一半的锁定请求造成的中间态。
            lastConvergedResult = result
        }

        advanceStateMachine(System.nanoTime())

        // 配对超时统一在 SENSOR_TIMESTAMP 时间域上判断，基准是这里维护的最大值。
        // 注意：它必须跟踪**所有** result（包括每秒 30 条的 preview），否则两组
        // burst 之间基准会冻结，超时判据退化成固定比较，凑不齐的桶永远不会过期。
        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP)
        if (ts != null && ts > lastResultSensorTs) lastResultSensorTs = ts

        // ---- 以下只处理 HQ still capture ----
        // preview result 每秒来 30 条，且 JPEG/RAW reader 根本没有对应的 Image。
        // 用 request.tag 把它们挡在配对表之外：旧实现让它们和 still result 抢
        // LinkedHashMap 的 256 条上限，还会造成误配。
        val tag = result.request.tag as? BurstFrameTag ?: return
        if (ts == null) return
        synchronized(matcherLock) {
            val bucket = pendingCaptures.getOrPut(ts) { PendingCapture() }
            bucket.result = result
            bucket.ctx = tag.ctx
            bucket.expectsRaw = tag.expectsRaw
            bucket.frameIndex = tag.index
            while (pendingCaptures.size > 256) {
                val oldest = pendingCaptures.entries.iterator().next()
                discardCapture(oldest.value)
                pendingCaptures.remove(oldest.key)
            }
        }
        tryConsumeCapture(ts)
    }

    /**
     * 曝光是否「实际稳定」：EV 代理 log2(exposure * iso) 在最近 6 帧的极差。
     * 用 log2 是因为曝光量本身是对数量纲，线性域上的差值不可比。
     */
    private fun exposureStable(): Boolean {
        val samples = synchronized(historyLock) { threeAHistory.toList() }
        if (samples.size < THREE_A_HISTORY) return false
        var mn = Double.MAX_VALUE
        var mx = -Double.MAX_VALUE
        for (s in samples) {
            if (s.exposureNs <= 0L || s.iso <= 0) return false
            val ev = Math.log(s.exposureNs.toDouble() * s.iso.toDouble()) / Math.log(2.0)
            if (!ev.isFinite()) return false
            if (ev < mn) mn = ev
            if (ev > mx) mx = ev
        }
        return (mx - mn) < MAX_EV_SPREAD
    }

    /**
     * 对焦是否「实际稳定」：LENS_FOCUS_DISTANCE 在最近 6 帧的极差。
     *
     * 只在设备真的会报焦距时才用它 —— 报不出焦距的机型 caps.manualFocus 为 false，
     * 那种情况下 afReady() 里的 AF_MODE_OFF / 固定对焦分支已经覆盖了。
     */
    private fun focusStable(): Boolean {
        if (!caps.manualFocus) return false
        val samples = synchronized(historyLock) { threeAHistory.toList() }
        if (samples.size < THREE_A_HISTORY) return false
        var mn = Float.MAX_VALUE
        var mx = -Float.MAX_VALUE
        for (s in samples) {
            val f = s.focusD
            if (!f.isFinite()) return false
            if (f < mn) mn = f
            if (f > mx) mx = f
        }
        return (mx - mn) < MAX_FOCUS_SPREAD_DIOPTER
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

    /**
     * AF 是否已合焦。
     *
     * PASSIVE_SCAN（正在搜索）不算就绪。
     * INACTIVE 在「preview 全程自动对焦」的前提下可以接受：它表示 AF 算法当前没有
     * 在移动镜头，而不是「AF 被我们关掉了」—— 后者只会在我们主动写 AF_MODE_OFF
     * 时出现，而那种情况上面那一行已经直接返回 true 了。
     * 真正在追焦/拉风箱时硬件报的是 PASSIVE_SCAN，仍然会被拒。
     */
    private fun afReady(result: TotalCaptureResult): Boolean {
        if (result.get(CaptureResult.CONTROL_AF_MODE) == CaptureRequest.CONTROL_AF_MODE_OFF) {
            return true
        }
        return when (result.get(CaptureResult.CONTROL_AF_STATE)) {
            CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED,
            CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED,
            CaptureResult.CONTROL_AF_STATE_INACTIVE -> true
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
    private fun buildLockedStillRequest(
        camera: CameraDevice,
        includeRaw: Boolean,
        tag: BurstFrameTag
    ): CaptureRequest {
        val frozen = locked3a ?: snapshot3A()
        val b = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
        // 打 tag：只有带 BurstFrameTag 的结果才进 HQ 配对表，
        // preview result 会被 onCaptureResult 直接忽略。
        b.setTag(tag)
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
        // 注意：这里不再有任何 lockRequested 概念。
        // preview 全程自动 3A，控制器只负责「什么时候可以拍」，不负责去锁预览。
        if (phase == STATE_WAIT_3A) {
            readyAeStreak = 0
            readyAwbStreak = 0
            readyAfStreak = 0
            readyStreak = 0
            stats.readyAeStreak = 0
            stats.readyAwbStreak = 0
            stats.readyAfStreak = 0
            stats.threeAReady = false
            stableFrames = 0
            // 退回重新收敛后，之前抓的冻结参数已经不对应当前光照，必须丢弃重抓
            locked3a = null
        }
    }

    /**
     * IDLE -> WAIT_3A -> READY -> CAPTURE
     *
     * 只要求 3A「收敛」，不要求它「锁定」。
     * 真正的手动冻结只发生在 [buildLockedStillRequest]：拍 burst 那一刻把最近一次
     * 收敛结果写进 still request。preview 全程保持自动 3A，随时适应新角度。
     *
     * 职责因此变得很清楚：
     *   preview  —— 永远自动适应场景
     *   收敛     —— 触发 WAIT_3A -> READY，并在跃迁处抓一份 Locked3A 快照
     *   burst    —— 5 帧使用完全相同的手动参数
     *   burst 后 —— 回到 READY；若场景已变，自动回到 WAIT_3A 重新收敛
     */
    private fun advanceStateMachine(nowNs: Long) {
        when (currentPhase) {
            STATE_WAIT_3A -> {
                // 需要「新鲜」的收敛结果 + 连续 readyStreak 帧。
                // AE=SEARCHING、AF=PASSIVE_SCAN 都不算收敛。
                if (readyStreak >= READY_STREAK && lastConvergedResult != null) {
                    // 3A 刚收敛，趁现在把参数抓下来冻结。
                    // 之后的 burst 全部用这套值下发，不再依赖 AE/AWB/AF 的 lock 是否真的生效。
                    locked3a = snapshot3A()
                    stats.readyBlockReason = ""
                    stats.threeAReady = true
                    stableFrames = 0
                    setPhase(STATE_READY)
                    onHint("3A 已收敛，开始采集高质量关键帧")
                }
            }

            STATE_READY, STATE_CAPTURE -> {
                // 不再要求 AE/AWB/AF LOCKED。
                // 若自动曝光重新开始搜索，说明场景变了（换了角度/光照），
                // 下一次 burst 前重新等收敛。CAPTURE 期间不打断，让在飞的 burst 跑完。
                if (!stats.threeAReady && currentPhase == STATE_READY) {
                    setPhase(STATE_WAIT_3A)
                    return
                }
                // 高光严重过曝：不要拿一套过曝的曝光去拍，丢掉快照重新收敛
                if (stats.previewClippedPercent > OVEREXPOSURE_RECOVER_PERCENT &&
                    exposureRecoveryAttempts < MAX_EXPOSURE_RECOVERY_ATTEMPTS &&
                    nowNs - lastExposureRecoveryNs > EXPOSURE_RECOVERY_COOLDOWN_NS
                ) {
                    exposureRecoveryAttempts++
                    lastExposureRecoveryNs = nowNs
                    stats.triggerRejectReason =
                        "高光过曝(${"%.1f".format(stats.previewClippedPercent)}%)，重新收敛"
                    onHint("高光过曝，重新收敛曝光")
                    setPhase(STATE_WAIT_3A)
                }
            }
        }
    }

    // ------------------------------------------------------------------ preview

    /**
     * 粗采样 Y 平面算亮度分布，用于拍摄前门控（约 1/64 像素，开销可忽略）。
     *
     * 除了暗/亮像素占比，还给出 P05/P50/P95 分位 —— 只有百分比是不够的：
     * 拍黑色物体、背景有阴影时，画面里本来就该有大量暗像素，
     * 「26% 像素很暗」并不等于照片欠曝。所以欠曝判定的主判据用分位，
     * 百分比只作为放宽后的兜底。
     */
    fun onPreviewLuma(y: ByteArray, width: Int, height: Int, rowStride: Int) {
        if (width <= 0 || height <= 0 || rowStride <= 0 || y.isEmpty()) {
            return
        }
        var clipped = 0
        var under = 0
        var total = 0
        val hist = lumaHist
        java.util.Arrays.fill(hist, 0)
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
                    hist[v]++
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
        // 分位直接发布原始值：单帧已有上万个采样点，本身就很稳，
        // 再叠 EMA 只会让「用户把镜头转向亮墙」这件事响应变慢。
        val p05 = percentileFromHist(hist, total, 0.05f)
        val p50 = percentileFromHist(hist, total, 0.50f)
        val p95 = percentileFromHist(hist, total, 0.95f)
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
        stats.lumaP05 = p05
        stats.lumaP50 = p50
        stats.lumaP95 = p95
        stats.previewLumaSamples = total
    }

    /** 从 256 桶直方图取分位，q ∈ [0,1] */
    private fun percentileFromHist(hist: IntArray, total: Int, q: Float): Float {
        if (total <= 0) return 0f
        val target = (q * total).toInt().coerceAtLeast(1)
        var acc = 0
        for (v in 0 until 256) {
            acc += hist[v]
            if (acc >= target) return v.toFloat()
        }
        return 255f
    }

    /**
     * 是否真的欠曝。
     *
     * 主判据：整体中位亮度低，且高光也上不去（说明不是「暗物体 + 亮背景」而是整幅偏暗）。
     * 兜底：暗像素占比超过放宽后的门限（35%）。分位还没采到样时只用兜底。
     */
    private fun reallyUnderexposed(): Boolean {
        val hasLuma = stats.previewLumaSamples > 0
        val byPercentile = hasLuma &&
            stats.lumaP50 < MIN_LUMA_P50 &&
            stats.lumaP95 < MIN_LUMA_P95
        val byDarkPercent =
            stats.previewUnderexposedPercent > MAX_PREVIEW_UNDEREXPOSED_PERCENT
        val result = byPercentile || byDarkPercent
        stats.realUnderexposed = result
        return result
    }

    // ---------------------------------------------------------------- frame tick

    fun onFrameTick(
        nowNs: Long,
        scanning: Boolean,
        camera: CameraDevice?,
        session: CameraCaptureSession?
    ) {
        stats.gyroMagnitudeRms = computeGyroMagnitudeRms(nowNs)
        stats.gyroJitterRms = computeGyroJitterRms(nowNs)
        stats.vinsDisplacement200ms = computeTranslationDisplacement200ms(nowNs)

        // 稳定窗口每帧都要更新，包括冷却期：用户停稳的动作不能被冷却期吞掉，
        // 否则冷却一结束还得再等 4 帧。
        updateStableWindow()

        if (!scanning) {
            if (currentPhase != STATE_IDLE) {
                setPhase(STATE_IDLE)
            }
            sweepPendingCaptures(nowNs)
            return
        }

        sweepPendingCaptures(nowNs)
        if (currentPhase == STATE_IDLE) {
            setPhase(STATE_WAIT_3A)
        }

        // 一次 burst 从拍摄到「融合结束 + 临时目录清理完」之间不允许再开新的：
        // 否则上一组还在读 12MP JPEG，下一组已经开始改文件系统了。
        if (burstInFlight || processingInFlight) {
            return
        }

        // 每一个「真正评估过的触发机会」计一次 candidate，被挡住的必定落进某个
        // reject 计数器。这样 burstStarted 远低于预期时，一眼就能看出卡在哪道门，
        // 而不是像之前那样只能看到 burstStarted 和一堆模糊的 skip。
        stats.schedulerCandidates++

        if (currentPhase != STATE_READY || !stats.threeAReady) {
            stats.reject3A++
            stats.triggerRejectReason =
                if (currentPhase != STATE_READY) "state=$currentPhase"
                else stats.readyBlockReason.ifEmpty { "3A 未收敛" }
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

        // 2) 稳定窗口：要求连续 STABLE_WINDOW_FRAMES 帧陀螺与位移同时达标。
        // 单帧稳定没有意义 —— 扫过物体时几乎每帧都在动，
        // 但只要用户停稳 100~200ms 就会满足，于是自动拍一组 HQ。
        if (stableFrames < STABLE_WINDOW_FRAMES) {
            stats.rejectMotion++
            stats.triggerRejectReason =
                "stable=${stableFrames}/$STABLE_WINDOW_FRAMES " +
                    "(gyroAbs=${"%.3f".format(stats.gyroMagnitudeRms)}, " +
                    "gyroJit=${"%.3f".format(stats.gyroJitterRms)}, " +
                    "disp=${"%.4f".format(stats.vinsDisplacement200ms)})"
            stats.burstSkipCount++
            return
        }

        // 3) 新视角是否值得拍（没有新增视角信息，拍出来也是重冗余）
        if (!novelViewpoint()) {
            stats.rejectMotion++
            stats.triggerRejectReason =
                "viewpoint 未变化(dT=${"%.4f".format(stats.viewpointDeltaMeters)}, dR=${"%.1f".format(stats.viewpointDeltaDeg)}°)"
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
        if (reallyUnderexposed()) {
            stats.rejectExposure++
            stats.triggerRejectReason =
                "underexposed(预检 P50=${"%.0f".format(stats.lumaP50)}, " +
                    "P95=${"%.0f".format(stats.lumaP95)}, " +
                    "dark=${"%.1f".format(stats.previewUnderexposedPercent)}%)"
            stats.burstSkipCount++
            nextBurstAtNs = nowNs + RETRY_COOLDOWN_NS
            return
        }

        stats.triggerRejectReason = ""
        startBurst(camera, session)
    }

    /**
     * 更新稳定窗口计数。
     *
     * 「连续」是关键：只要有一帧不达标就归零重数，避免把「抖一下又稳一下」
     * 误判成稳定。门限见 MAX_GYRO_MAGNITUDE_RMS / MAX_VINS_DISP_200MS。
     */
    private fun updateStableWindow() {
        // 用角速度幅值，不是抖动：匀速扫视时抖动≈0，但手机其实一直在转。
        val stable = stats.gyroMagnitudeRms <= MAX_GYRO_MAGNITUDE_RMS &&
            stats.vinsDisplacement200ms <= MAX_VINS_DISP_200MS
        stableFrames = if (stable) stableFrames + 1 else 0
        stats.stableFrames = stableFrames
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

        // 每一组 burst 一份独立上下文 + 独立临时目录。
        // 绝不能再共用 hq_capture/<session>/tmp_burst/ —— 上一组还在读 12MP JPEG 时
        // 下一组已经往里写新文件，任意一方清理时就会把对方的照片删掉，
        // fusion 的 imread(paths[0]) 于是返回空图。
        val token = System.nanoTime()
        val ctx = BurstContext(
            token = token,
            sessionId = currentSessionId,
            tempDir = File(burstTmpDir(currentSessionId), token.toString()).apply { mkdirs() }
        )
        ctx.expectedFrames = count
        ctx.startNs = token
        activeBurst = ctx
        fusionStarted = false
        // 从这里到融合结束、临时目录清理完，都不允许再开新的 burst
        processingInFlight = true
        burstExpectedFrames = count
        burstCompletedThisRound = 0
        burstFailedThisRound = 0

        // 记录本次拍摄的位姿：既用于「新视角」判定，也用于 burst 内部运动统计
        val pose = FloatArray(12)
        if (NativeBridge.nativeGetRenderPose(pose)) {
            System.arraycopy(pose, 0, lastBurstPose, 0, 12)
            haveLastBurstPose = true
            ctx.startPose = pose.copyOf()
        }

        try {
            // 只有第 0 帧需要 RAW：一张 DNG 足够做线性/色彩参考，
            // 5 张 RAW 会把 burst 的带宽和落盘时间翻好几倍。
            val requests = (0 until count).map { i ->
                val expectsRaw = (i == 0) && rawReader != null
                buildLockedStillRequest(
                    camera,
                    includeRaw = expectsRaw,
                    tag = BurstFrameTag(ctx, i, expectsRaw)
                )
            }
            burstInFlight = true
            session.captureBurst(requests, burstCallback, handler)
        } catch (e: Exception) {
            burstInFlight = false
            stats.burstDropped += count
            // 下发失败必须放掉这一组，否则 processingInFlight 会把后续 burst 永久卡死
            processingInFlight = false
            if (activeBurst === ctx) activeBurst = null
            cleanupBurstDir(ctx)
            setPhase(STATE_READY)
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
        // 采完一组就回到 READY：3A 仍保持自动收敛，只是重新攒稳定窗口
        setPhase(STATE_READY)
        // 给下一次 burst 留出 0.8~1.2s 的冷却
        val span = (BURST_COOLDOWN_MAX_NS - BURST_COOLDOWN_MIN_NS).toDouble()
        val cd = (BURST_COOLDOWN_MIN_NS + (Math.random() * span).toLong())
        stats.burstCooldownMs = cd / 1_000_000L
        nextBurstAtNs = System.nanoTime() + cd

        val ctx = activeBurst
        if (ctx == null) {
            processingInFlight = false
            return
        }

        // burst 内部运动：第 0 帧 -> 最后一帧。
        // 旧的 lastBurstPoseDelta/lastBurstAngleDeltaDeg 其实是「上一关键帧 -> 当前」
        // 的新视角变化，和 burst 内部运动不是一回事，所以拆开单独统计。
        val endPose = FloatArray(12)
        val startPose = ctx.startPose
        if (startPose != null && NativeBridge.nativeGetRenderPose(endPose)) {
            val dx = endPose[9] - startPose[9]
            val dy = endPose[10] - startPose[10]
            val dz = endPose[11] - startPose[11]
            stats.burstTranslationMeters = hypot(hypot(dx, dy), dz)
            stats.burstRotationDeg = rotationAngleDeg(startPose, endPose)
        }
        stats.burstDurationMs = (System.nanoTime() - ctx.startNs) / 1_000_000L

        // 正常路径：第 N 张 JPEG 全部落盘后由 tryConsumeCapture 立刻启动融合。
        // 这里只挂一个兜底定时器，防止个别 Image 迟到导致永远不融合，
        // 取代了旧的无条件固定 700ms 等待。
        handler.postDelayed({
            if (!fusionStarted && processingInFlight && activeBurst === ctx) {
                fusionStarted = true
                startFusion(ctx)
            }
        }, FUSION_FALLBACK_DELAY_MS)
        maybeStartFusion(ctx)
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

    /**
     * 桶里 result / jpeg / raw 都到位后整体消费并移除。
     *
     * 判定条件：result 已到、jpeg 已到、只有 expectsRaw 时才额外等 raw。
     * 这样一个 Result 可以同时服务 JPEG、RAW 和 DNG metadata 三件事。
     */
    private fun tryConsumeCapture(ts: Long) {
        var bucket: PendingCapture? = null
        synchronized(matcherLock) {
            val b = pendingCaptures[ts]
            val ready = b != null && b.result != null && b.jpeg != null &&
                (!b.expectsRaw || b.raw != null)
            if (ready) {
                bucket = b
                pendingCaptures.remove(ts)
            }
        }
        val b = bucket ?: return
        val result = b.result
        val ctx = b.ctx
        val jpeg = b.jpeg
        if (jpeg != null) {
            stats.jpegMatched++
            if (ctx != null) runCatching { saveJpeg(jpeg, ts, ctx) }
            runCatching { jpeg.close() }
        }
        val raw = b.raw
        if (raw != null) {
            if (result != null && ctx != null) {
                stats.rawMatched++
                runCatching { saveDng(raw, ts, result, ctx) }
            } else {
                // RAW 没有 TotalCaptureResult 就无法生成合法 DNG
                stats.rawExpired++
            }
            runCatching { raw.close() }
        }
        b.jpeg = null
        b.raw = null
        // JPEG 落盘后立刻检查能不能融合（不再死等固定延迟）
        if (ctx != null) maybeStartFusion(ctx)
    }

    /** 关掉桶里还没消费的 Image，不落盘不计数（整场重置 / 配对表裁剪用） */
    private fun discardCapture(b: PendingCapture) {
        runCatching { b.jpeg?.close() }
        runCatching { b.raw?.close() }
        b.jpeg = null
        b.raw = null
    }

    /** 桶超时：JPEG 本身是好的，能落盘就落盘；RAW 没有 Result 只能作废 */
    private fun expireCapture(b: PendingCapture) {
        val jpeg = b.jpeg
        val raw = b.raw
        val ctx = b.ctx
        if (jpeg != null) {
            val ts = jpeg.timestamp
            if (ctx != null) {
                stats.jpegMatched++
                runCatching { saveJpeg(jpeg, ts, ctx) }
            } else {
                stats.jpegExpired++
            }
            runCatching { jpeg.close() }
        }
        if (raw != null) {
            stats.rawExpired++
            runCatching { raw.close() }
        }
        b.jpeg = null
        b.raw = null
    }

    private fun onJpegAvailable(reader: ImageReader) {
        val image = acquireNext(reader) ?: return
        stats.jpegReceived++
        val ts = image.timestamp
        var retained = false
        try {
            if (!active) {
                stats.jpegExpired++
                return
            }
            synchronized(matcherLock) {
                val b = pendingCaptures.getOrPut(ts) { PendingCapture() }
                // Image 先于 Result 到达：记一笔「挂起后配对成功」的统计
                if (b.result == null) stats.pairDeferredMatched++
                val old = b.jpeg
                if (old != null) runCatching { old.close() }
                b.jpeg = image
                retained = true
            }
        } catch (e: Exception) {
            stats.jpegExpired++
            stats.lastCaptureRejectReason = "jpeg: ${e.message}"
        } finally {
            if (!retained) runCatching { image.close() }
        }
        if (retained) tryConsumeCapture(ts)
    }

    private fun onRawAvailable(reader: ImageReader) {
        val image = acquireNext(reader) ?: return
        stats.rawReceived++
        val ts = image.timestamp
        var retained = false
        try {
            if (!active) {
                stats.rawExpired++
                return
            }
            synchronized(matcherLock) {
                val b = pendingCaptures.getOrPut(ts) { PendingCapture() }
                if (b.result == null) stats.pairDeferredMatched++
                val old = b.raw
                if (old != null) runCatching { old.close() }
                b.raw = image
                retained = true
            }
        } catch (e: Exception) {
            stats.rawExpired++
            stats.lastCaptureRejectReason = "raw: ${e.message}"
        } finally {
            if (!retained) runCatching { image.close() }
        }
        if (retained) tryConsumeCapture(ts)
    }

    /**
     * 超时兜底：迟迟凑不齐（或根本没有 HQ 请求对应）的桶不能一直占着 ImageReader 缓冲。
     *
     * 判据优先走 SENSOR_TIMESTAMP 时间域 —— 用「已知的最新 result sensor 时间戳」减去
     * 「这张 Image 自己的 sensor 时间戳」。两个数来自同一个 sensor 时钟，直接相减才有意义；
     * 掺进 System.currentTimeMillis() 会因为时钟域不同而误判。
     * 只有在一条 HQ result 都还没回来（没有基准）时，才退回墙钟兜底。
     */
    private fun sweepPendingCaptures(nowNs: Long) {
        val expired = mutableListOf<PendingCapture>()
        synchronized(matcherLock) {
            val sensorNow = lastResultSensorTs
            val it = pendingCaptures.entries.iterator()
            while (it.hasNext()) {
                val entry = it.next()
                val b = entry.value
                val timedOut = if (sensorNow > 0L && entry.key > 0L && sensorNow >= entry.key) {
                    sensorNow - entry.key > PAIR_TIMEOUT_NS
                } else {
                    nowNs - b.arrivedNs > IMAGE_RESULT_HOLD_NS
                }
                if (timedOut) {
                    expired.add(b)
                    it.remove()
                }
            }
        }
        for (b in expired) expireCapture(b)
    }

    private fun saveJpeg(image: Image, ts: Long, ctx: BurstContext) {
        val bytes = ByteArray(image.planes[0].buffer.remaining())
        image.planes[0].buffer.get(bytes)
        // burst 成员先落在「本组自己的」临时目录，融合结束后只保留参考帧 + 融合结果
        val file = File(ctx.tempDir, "burst_${ts}.jpg")
        FileOutputStream(file).use { it.write(bytes) }
        stats.jpegSaved++
        synchronized(ctx.jpegLock) {
            ctx.jpegPaths.add(file.absolutePath)
        }
    }

    private fun saveDng(image: Image, ts: Long, result: TotalCaptureResult?, ctx: BurstContext) {
        if (result == null) {
            // 计数由调用方（tryConsumeCapture）统一负责，这里只负责不生成非法 DNG
            return
        }
        if (ctx.rawSaved) {
            // 正常模式只保留一张参考 RAW，其余不落盘
            stats.rawSkipped++
            return
        }
        ctx.rawSaved = true
        val dir = captureDir(ctx.sessionId)
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

    /** 到齐就立刻融合；没到齐就等兜底定时器。fusionStarted 保证只启动一次。 */
    private fun maybeStartFusion(ctx: BurstContext) {
        if (fusionStarted || !processingInFlight) return
        if (activeBurst !== ctx) return
        val n = synchronized(ctx.jpegLock) { ctx.jpegPaths.size }
        if (n < ctx.expectedFrames) return
        fusionStarted = true
        startFusion(ctx)
    }

    private fun startFusion(ctx: BurstContext) {
        // 融合读的是不可变快照：之后谁再往 ctx 里加/删都不影响这一趟
        val paths = synchronized(ctx.jpegLock) { ctx.jpegPaths.toList() }
        if (paths.size < 2) {
            stats.lastCaptureRejectReason = "burst images < 2"
            finishFusion(ctx)
            scheduleRetake()
            return
        }

        val outDir = captureDir(ctx.sessionId)
        val out = File(outDir, "fused_${ctx.token}.jpg")
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
                applyFusionStats(ctx, ok, out, statsOut, paths)
                finishFusion(ctx)
            }
        }.start()
    }

    /**
     * 融合收尾：只删自己这一组的临时目录，绝不碰别的 burst。
     * processingInFlight 在这里才清零 —— 它同时是「下一组 burst 的准入条件」。
     */
    private fun finishFusion(ctx: BurstContext) {
        cleanupBurstDir(ctx)
        synchronized(ctx.jpegLock) { ctx.jpegPaths.clear() }
        if (activeBurst === ctx) activeBurst = null
        fusionStarted = false
        processingInFlight = false
    }

    private fun applyFusionStats(
        ctx: BurstContext,
        ok: Boolean,
        out: File,
        statsOut: FloatArray,
        paths: List<String>
    ) {
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
        // 专门的「参考帧解码失败」计数：旧报告里错误信息说解码失败、
        // 但 rejectedDecode 却是 0，两处自相矛盾。现在 native 也会为
        // 参考帧失败记一次 rejectedDecode，这里再单独暴露出来。
        stats.referenceDecodeFail = if (decodeFailed) stats.rejectedDecode.coerceAtLeast(1) else 0
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
                if (saveFallbackSingle(ctx, paths[picked])) {
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
                    val refDst = File(captureDir(ctx.sessionId), "reference_${ctx.token}.jpg")
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
    private fun saveFallbackSingle(ctx: BurstContext, srcPath: String): Boolean {
        val src = File(srcPath)
        if (!src.exists()) return false
        val dst = File(captureDir(ctx.sessionId), "single_${ctx.token}.jpg")
        return runCatching {
            src.copyTo(dst, overwrite = true)
            stats.fallbackSingleJpeg = dst.absolutePath
            true
        }.getOrDefault(false)
    }

    /**
     * 只删某一组自己的临时目录。
     * 旧实现 cleanupBurstTmp() 会删掉整个共享 tmp_burst —— 那正是
     * 「A 组融合时把 B 组的照片删掉」的根因，所以它被彻底去掉了。
     */
    private fun cleanupBurstDir(ctx: BurstContext) {
        runCatching { ctx.tempDir.deleteRecursively() }
    }

    /** 整场扫描开始/结束时的兜底清理：此时不可能有 burst 在飞 */
    private fun cleanupAllBurstTmp(sessionId: String) {
        runCatching { burstTmpDir(sessionId).deleteRecursively() }
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

    /** 最近 250ms 的 IMU 样本；不足 6 个说明数据还不够，返回 null */
    private fun recentImu(nowNs: Long): List<HqImuSample>? {
        val cutoff = nowNs - 250_000_000L
        val recent = synchronized(imuLock) { imuSamples.filter { it.timestampNs >= cutoff } }
        return if (recent.size < 6) null else recent
    }

    /**
     * 真实角速度幅值：sqrt(mean(gx² + gy² + gz²))。**稳定门用这个。**
     *
     * 旧实现 computeMotionRms() 算的是「陀螺减均值后的 RMS」，那其实是抖动/方差：
     * 手机以 0.3 rad/s 平滑匀速旋转时，每个样本都≈0.3，减均值后≈0，
     * 于是「一直在匀速转」会被判成「很稳定」—— 恰恰是最该拒绝的场景。
     */
    private fun computeGyroMagnitudeRms(nowNs: Long): Float {
        val recent = recentImu(nowNs) ?: return 999f
        var sum = 0.0
        for (s in recent) {
            sum += (s.gyroX * s.gyroX + s.gyroY * s.gyroY + s.gyroZ * s.gyroZ).toDouble()
        }
        return sqrt((sum / recent.size).toFloat())
    }

    /** 旧算法原样保留，重命名为 jitter，只作为诊断输出（不再参与稳定门） */
    private fun computeGyroJitterRms(nowNs: Long): Float {
        val recent = recentImu(nowNs) ?: return 999f
        val gx = recent.map { it.gyroX }.average().toFloat()
        val gy = recent.map { it.gyroY }.average().toFloat()
        val gz = recent.map { it.gyroZ }.average().toFloat()
        val g = sqrt(
            recent.map {
                val dx = it.gyroX - gx
                val dy = it.gyroY - gy
                val dz = it.gyroZ - gz
                dx * dx + dy * dy + dz * dz
            }.average().toFloat()
        )
        val ax = recent.map { it.accelX }.average().toFloat()
        val ay = recent.map { it.accelY }.average().toFloat()
        val az = recent.map { it.accelZ }.average().toFloat()
        val a = sqrt(
            recent.map {
                val dx = it.accelX - ax
                val dy = it.accelY - ay
                val dz = it.accelZ - az
                dx * dx + dy * dy + dz * dz
            }.average().toFloat()
        )
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
        stats.viewpointDeltaMeters = hypot(hypot(dx, dy), dz)
        stats.viewpointDeltaDeg = rotationAngleDeg(lastBurstPose, pose)
        return stats.viewpointDeltaMeters >= MIN_NEW_VIEW_TRANSLATION ||
            stats.viewpointDeltaDeg >= MIN_NEW_VIEW_ANGLE_DEG
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
        sb.appendLine("3aReady=${stats.threeAReady}")
        // 收敛 vs 锁定：只要求收敛，不再要求硬件报 LOCKED
        sb.appendLine("readyAeStreak=${stats.readyAeStreak}")
        sb.appendLine("readyAwbStreak=${stats.readyAwbStreak}")
        sb.appendLine("readyAfStreak=${stats.readyAfStreak}")
        sb.appendLine("readyBlockReason=${stats.readyBlockReason}")
        sb.appendLine("aeParamsStable=${stats.aeParamsStable}")
        sb.appendLine("afParamsStable=${stats.afParamsStable}")
        sb.appendLine("AE state=${stats.aeState}")
        sb.appendLine("AE locked=${stats.aeLocked}")
        sb.appendLine("AWB state=${stats.awbState}")
        sb.appendLine("AWB locked=${stats.awbLocked}")
        sb.appendLine("AF state=${stats.afState}")
        sb.appendLine("AF locked=${stats.afLocked}")
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
        // 稳定门的输入是 gyroMagnitudeRms（真实角速度幅值）；
        // gyroJitterRms 是旧算法，只留作对比，方便下一轮实机标定门限。
        sb.appendLine("gyroMagnitudeRms=${stats.gyroMagnitudeRms}")
        sb.appendLine("gyroJitterRms=${stats.gyroJitterRms}")
        sb.appendLine("vinsDisplacement200ms=${stats.vinsDisplacement200ms}")
        sb.appendLine("stableFrames=${stats.stableFrames}")
        sb.appendLine("realUnderexposed=${stats.realUnderexposed}")
        sb.appendLine("translationDuringBurst=${stats.translationDuringBurst}")
        sb.appendLine("viewpointDeltaMeters=${stats.viewpointDeltaMeters}")
        sb.appendLine("viewpointDeltaDeg=${stats.viewpointDeltaDeg}")
        sb.appendLine("burstTranslationMeters=${stats.burstTranslationMeters}")
        sb.appendLine("burstRotationDeg=${stats.burstRotationDeg}")
        sb.appendLine("burstDurationMs=${stats.burstDurationMs}")
        sb.appendLine("previewClippedPercent=${stats.previewClippedPercent}")
        sb.appendLine("previewUnderexposedPercent=${stats.previewUnderexposedPercent}")
        sb.appendLine("previewLumaP05=${stats.lumaP05}")
        sb.appendLine("previewLumaP50=${stats.lumaP50}")
        sb.appendLine("previewLumaP95=${stats.lumaP95}")
        sb.appendLine("previewLumaSamples=${stats.previewLumaSamples}")
        sb.appendLine("alignmentInliers=${stats.alignmentInliers}")
        sb.appendLine("alignmentRmseMean=${stats.alignmentRmseMean}")
        sb.appendLine("alignmentRmseMax=${stats.alignmentRmseMax}")
        sb.appendLine("eccMean=${stats.eccMean}")
        sb.appendLine("rejectedAlignment=${stats.rejectedAlignment}")
        sb.appendLine("rejectedEcc=${stats.rejectedEcc}")
        sb.appendLine("rejectedDecode=${stats.rejectedDecode}")
        sb.appendLine("referenceDecodeFail=${stats.referenceDecodeFail}")
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
