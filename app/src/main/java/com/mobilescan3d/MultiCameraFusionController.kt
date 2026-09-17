package com.mobilescan3d

import android.annotation.TargetApi
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executor
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.max
import kotlin.math.sqrt

/**
 * V0.10 OnePlus physical multi-camera helper with empirical relative-pose calibration.
 *
 * The existing primary-camera VIO/depth/TSDF path remains unchanged. This helper
 * explicitly binds two YUV outputs to two physical cameras of one logical camera,
 * runs sparse stereo validation, and registers occasional telephoto frames into
 * the existing HQ texture baker.
 *
 * It deliberately does NOT inject metric stereo depth into TSDF yet. The current
 * project calibrates monocular depth into VINS-world scale, so real-device scale
 * consistency must be verified first.
 */
class MultiCameraFusionController(
    private val context: Context,
    private val cameraManager: CameraManager,
    private val logicalCameraId: String,
    private val logicalCharacteristics: CameraCharacteristics,
    private val cameraHandler: Handler,
    private val onHint: (String) -> Unit
) {
    companion object {
        private const val TAG = "MultiCamV11"
        private const val PAIR_PERIOD_NS = 160_000_000L
        private const val TELE_TEXTURE_PERIOD_NS = 1_100_000_000L
        private const val MAX_TELE_TEXTURE_KEYFRAMES = 24
        private const val CALIBRATED_PAIR_TOLERANCE_NS = 4_000_000L
        private const val APPROX_PAIR_TOLERANCE_NS = 12_000_000L
        private const val STEREO_ANCHOR_STRIDE = 6
        private const val MAX_STEREO_ANCHORS = 96
    }

    private data class CameraModel(
        val id: String,
        val focalMm: Float,
        val horizontalFovDeg: Float,
        val factoryK: FloatArray,
        val k: FloatArray,
        val poseRotation: FloatArray?,
        val poseTranslation: FloatArray?,
        val poseReference: Int
    )

    private data class GrayFrame(
        val timestampNs: Long,
        val gray: ByteArray
    )

    private data class GeometryCandidate(
        val model: CameraModel,
        val baselineMeters: Float,
        val overlap: Float,
        val score: Float
    )

    private data class PhysicalCaptureStats(
        var frames: Long = 0L,
        var timestampCompared: Long = 0L,
        var timestampAbsDeltaSumNs: Double = 0.0,
        var timestampAbsDeltaMaxNs: Long = 0L,
        var lastSensorTimestampNs: Long = 0L,
        var lastExposureNs: Long = 0L,
        var lastFrameDurationNs: Long = 0L,
        var lastRollingShutterSkewNs: Long = 0L,
        var lastIso: Int = -1,
        var lastFocalMm: Float = 0f,
        var lastFocusDistance: Float = 0f,
        var lastCrop: String = "n/a",
        var lastDistortionMode: Int = -1,
        var lastZoomRatio: Float = 1f,
        var lastCropRect: android.graphics.Rect? = null,
        var cropStableFrames: Int = 0
    )

    @Volatile
    var active: Boolean = false
        private set

    @Volatile
    var geometryReady: Boolean = false
        private set

    var primaryReader: ImageReader? = null
        private set

    private var secondaryReader: ImageReader? = null

    val secondarySurface: Surface?
        get() = secondaryReader?.surface

    private var configuredSize: Size? = null
    private var primaryModel: CameraModel? = null
    private var secondaryModel: CameraModel? = null
    private var relativeRPrimaryFromSecondary: FloatArray? = null
    private var relativeTPrimaryFromSecondary: FloatArray? = null

    private var syncType: Int? = null
    private var calibratedSync = false
    private var pairToleranceNs = APPROX_PAIR_TOLERANCE_NS
    private var baselineMeters = 0f
    private var focalRatio = 1f

    /**
     * PLK110 feedback showed vendor pose rotations that cannot be trusted blindly
     * for triangulation. Factory translation is used only as baseline magnitude.
     */
    private var factoryGeometryTrusted = false
    private var pairSelectionReason = "not selected"

    /** V0.11: runtime crop-aware intrinsics epoch. */
    @Volatile
    private var runtimeIntrinsicEpoch = 0

    @Volatile
    private var runtimeCropApplied = false

    private var runtimePrimaryK: FloatArray? = null
    private var runtimeSecondaryK: FloatArray? = null
    private var runtimeCropReason = "not evaluated"

    @Volatile
    private var status = "not configured"

    @Volatile
    private var sessionMode = "not-created"

    private val physicalResultLock = Any()
    private val physicalCaptureStats = LinkedHashMap<String, PhysicalCaptureStats>()

    @Volatile
    private var physicalResultCallbacks = 0L

    @Volatile
    private var physicalResultEntries = 0L

    private val fusionThread = HandlerThread("MultiCamFusion").apply { start() }
    private val fusionHandler = Handler(fusionThread.looper)

    private var latestPrimary: GrayFrame? = null
    private var latestSecondary: GrayFrame? = null
    private var lastPrimarySampleNs = 0L
    private var lastSecondarySampleNs = 0L

    @Volatile
    private var scanEnabled = false

    @Volatile
    private var teleTextureGate = false

    @Volatile
    private var currentSessionId = "unknown"

    private var lastTeleTextureQueuedNs = 0L

    @Volatile
    private var teleTextureRegistered = 0

    @Volatile
    private var teleTexturePoseMiss = 0

    @Volatile
    private var teleTextureWriteFail = 0

    @Volatile
    private var pairDispatchCount = 0L

    @Volatile
    private var pairDroppedForTimestamp = 0L

    private val stereoAnchorBuffer =
        FloatArray(MAX_STEREO_ANCHORS * STEREO_ANCHOR_STRIDE)

    @Volatile
    private var anchorSubmitCalls = 0L

    @Volatile
    private var anchorSubmitAnchors = 0L

    @Volatile
    private var anchorSubmitFailures = 0L

    fun ownsPrimaryReader(candidate: ImageReader?): Boolean =
        candidate != null && candidate === primaryReader

    /**
     * Creates two same-size physical YUV readers. MainActivity continues to use
     * [primaryReader] exactly where it previously used its single logical reader.
     */
    @TargetApi(Build.VERSION_CODES.P)
    fun configure(
        size: Size,
        onPrimaryImage: (Image) -> Unit
    ): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            status = "API < 28"
            return false
        }

        val caps = logicalCharacteristics.get(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES
        ) ?: intArrayOf()
        if (!caps.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA
            )
        ) {
            status = "logical multi-camera capability not exposed"
            return false
        }

        val physicalIds = logicalCharacteristics.physicalCameraIds.toList()
        if (physicalIds.size < 2) {
            status = "logical camera exposes <2 physical ids"
            return false
        }

        val models = physicalIds.mapNotNull { buildCameraModel(it, size) }
        if (models.size < 2) {
            status = "physical camera characteristics unavailable"
            return false
        }

        // Prefer the factory PRIMARY_CAMERA origin as the VIO/main stream.
        // The PLK110 feedback identified physical 2 this way.
        val primary = models
            .filter {
                it.poseReference == CameraCharacteristics.LENS_POSE_REFERENCE_PRIMARY_CAMERA &&
                    it.poseTranslation != null
            }
            .minByOrNull { vectorNorm(it.poseTranslation!!) }
            ?: models
                .filter { it.horizontalFovDeg > 1f }
                .minByOrNull { abs(it.horizontalFovDeg - 72f) }
            ?: return false

        // V0.10 pair selection is geometry-first, not focal-length-first.
        // Useful stereo needs a plausible physical baseline AND overlap.
        val geometryCandidates = models
            .filter { it.id != primary.id }
            .mapNotNull { candidate ->
                val b = factoryBaselineMeters(primary, candidate)
                    ?: return@mapNotNull null
                if (b < 0.004f || b > 0.080f) {
                    return@mapNotNull null
                }
                val a = primary.horizontalFovDeg
                val c = candidate.horizontalFovDeg
                val overlap =
                    if (a > 1f && c > 1f) {
                        (minOf(a, c) / maxOf(a, c)).coerceIn(0.05f, 1f)
                    } else {
                        0.35f
                    }
                val coverageBoost =
                    if (c >= a * 0.95f) 1.25f else 1.0f
                val score = (b * 1000f) * overlap * coverageBoost
                GeometryCandidate(candidate, b, overlap, score)
            }

        val chosen = geometryCandidates.maxByOrNull { it.score }
        if (chosen == null) {
            status = "no physical pair has plausible 4-80mm factory baseline"
            return false
        }

        val secondary = chosen.model
        primaryModel = primary
        secondaryModel = secondary
        configuredSize = size
        baselineMeters = chosen.baselineMeters
        focalRatio =
            if (primary.k[0] > 1e-3f) secondary.k[0] / primary.k[0] else 1f
        pairSelectionReason = buildString {
            append("geometry-score picked ")
            append(primary.id).append("->").append(secondary.id)
            append(" baselineMm=").append("%.2f".format(baselineMeters * 1000f))
            append(" overlap=").append("%.3f".format(chosen.overlap))
            append(" score=").append("%.3f".format(chosen.score))
            if (geometryCandidates.isNotEmpty()) {
                append(" candidates=")
                append(
                    geometryCandidates.joinToString(";") {
                        "${it.model.id}:${"%.2f".format(it.baselineMeters * 1000f)}mm/" +
                            "${"%.2f".format(it.overlap)}/${"%.2f".format(it.score)}"
                    }
                )
            }
        }

        syncType = logicalCharacteristics.get(
            CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE
        )
        calibratedSync =
            syncType == CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED
        pairToleranceNs = if (calibratedSync) {
            CALIBRATED_PAIR_TOLERANCE_NS
        } else {
            APPROX_PAIR_TOLERANCE_NS
        }

        // Do NOT use vendor LENS_POSE_ROTATION directly for geometry on PLK110.
        // Native V0.10 recovers relative R and translation direction from actual
        // synchronized images. Camera2 translation supplies baseline norm only.
        relativeRPrimaryFromSecondary = null
        relativeTPrimaryFromSecondary = null
        factoryGeometryTrusted = false
        geometryReady = false

        val nativeConfigured = runCatching {
            NativeBridge.nativeMultiCamConfigure(
                primary.k,
                secondary.k,
                primary.poseRotation ?: floatArrayOf(0f, 0f, 0f, 1f),
                primary.poseTranslation ?: floatArrayOf(0f, 0f, 0f),
                secondary.poseRotation ?: floatArrayOf(0f, 0f, 0f, 1f),
                secondary.poseTranslation ?: floatArrayOf(0f, 0f, 0f),
                size.width,
                size.height,
                calibratedSync
            )
        }.getOrDefault(false)

        if (!nativeConfigured) {
            status = "empirical multi-camera bootstrap rejected baseline/intrinsics"
            return false
        }

        primaryReader = ImageReader.newInstance(
            size.width,
            size.height,
            ImageFormat.YUV_420_888,
            4
        ).apply {
            setOnImageAvailableListener({ r ->
                val image = r.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    maybeQueuePrimaryPair(image)
                    // Existing processImage() owns/closes the image.
                    onPrimaryImage(image)
                } catch (t: Throwable) {
                    runCatching { image.close() }
                    Log.w(TAG, "primary frame failed", t)
                }
            }, cameraHandler)
        }

        secondaryReader = ImageReader.newInstance(
            size.width,
            size.height,
            ImageFormat.YUV_420_888,
            4
        ).apply {
            setOnImageAvailableListener({ r ->
                val image = r.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    maybeQueueSecondaryPair(image)
                    maybeQueueTeleTexture(image)
                } catch (t: Throwable) {
                    Log.w(TAG, "secondary frame failed", t)
                } finally {
                    image.close()
                }
            }, cameraHandler)
        }

        active = true
        status = buildString {
            append("dual physical YUV self-calibration ready ")
            append(primary.id).append(" -> ").append(secondary.id)
            append(", baselineSeed=").append("%.1f".format(baselineMeters * 1000f)).append("mm")
            append(", sync=").append(if (calibratedSync) "CALIBRATED" else "APPROXIMATE")
            append(", waiting empirical R/t")
        }
        onHint("V0.10 多镜头自标定：physical ${primary.id} + ${secondary.id}")
        return true
    }

    fun noteSessionMode(mode: String) {
        sessionMode = mode
    }

    /**
     * Preview and HQ JPEG/RAW stay logical; the two YUV outputs get physical IDs.
     */
    @TargetApi(Build.VERSION_CODES.P)
    fun createPhysicalSession(
        camera: CameraDevice,
        logicalSurfaces: List<Surface>,
        callback: CameraCaptureSession.StateCallback
    ): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P || !active) return false

        val primary = primaryModel ?: return false
        val secondary = secondaryModel ?: return false
        val primarySurface = primaryReader?.surface ?: return false
        val secondarySurface = secondaryReader?.surface ?: return false

        return try {
            val outputs = ArrayList<OutputConfiguration>(logicalSurfaces.size + 2)
            logicalSurfaces.forEach { outputs.add(OutputConfiguration(it)) }
            outputs.add(
                OutputConfiguration(primarySurface).apply {
                    setPhysicalCameraId(primary.id)
                }
            )
            outputs.add(
                OutputConfiguration(secondarySurface).apply {
                    setPhysicalCameraId(secondary.id)
                }
            )

            val executor = Executor { command ->
                if (!cameraHandler.post(command)) {
                    command.run()
                }
            }
            camera.createCaptureSession(
                SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    outputs,
                    executor,
                    callback
                )
            )
            true
        } catch (t: Throwable) {
            status = "physical session creation threw ${t.javaClass.simpleName}: ${t.message}"
            Log.w(TAG, status, t)
            false
        }
    }

    /**
     * Preserve primaryReader so MainActivity can immediately rebuild the original
     * single logical-camera session on the same Surface.
     */
    fun fallbackToLogical(reason: String) {
        active = false
        geometryReady = false
        scanEnabled = false
        teleTextureGate = false
        runCatching { secondaryReader?.close() }
        secondaryReader = null
        runCatching { NativeBridge.nativeMultiCamReset() }
        runCatching { NativeBridge.nativeResetStereoAnchors() }
        sessionMode = "logical-fallback"
        status = "fallback to logical camera: $reason"
        Log.w(TAG, status)
        onHint("多镜头会话不被 HAL 接受，已自动回退单摄")
    }

    /**
     * Capture-result telemetry for the actual physical streams.
     *
     * Static camera characteristics are not enough: OEM HALs may crop/zoom a
     * non-active physical stream. These values let the feedback report prove
     * whether the K used by stereo still matches the delivered YUV images.
     */
    fun onCaptureResult(result: TotalCaptureResult) {
        if (!active) return
        physicalResultCallbacks++

        val logicalTimestamp =
            result.get(CaptureResult.SENSOR_TIMESTAMP)

        fun record(id: String, physical: CaptureResult) {
            val sensorTs =
                physical.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L
            val exposure =
                physical.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
            val frameDuration =
                physical.get(CaptureResult.SENSOR_FRAME_DURATION) ?: 0L
            val skew =
                physical.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW) ?: 0L
            val iso =
                physical.get(CaptureResult.SENSOR_SENSITIVITY) ?: -1
            val focal =
                physical.get(CaptureResult.LENS_FOCAL_LENGTH) ?: 0f
            val focus =
                physical.get(CaptureResult.LENS_FOCUS_DISTANCE) ?: 0f
            val crop =
                physical.get(CaptureResult.SCALER_CROP_REGION)
            val distortion =
                physical.get(CaptureResult.DISTORTION_CORRECTION_MODE) ?: -1
            val zoom =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    physical.get(CaptureResult.CONTROL_ZOOM_RATIO) ?: 1f
                } else {
                    1f
                }

            synchronized(physicalResultLock) {
                val st = physicalCaptureStats.getOrPut(id) {
                    PhysicalCaptureStats()
                }
                st.frames++
                st.lastSensorTimestampNs = sensorTs
                st.lastExposureNs = exposure
                st.lastFrameDurationNs = frameDuration
                st.lastRollingShutterSkewNs = skew
                st.lastIso = iso
                st.lastFocalMm = focal
                st.lastFocusDistance = focus
                st.lastCrop = crop?.let {
                    "${it.left},${it.top},${it.right},${it.bottom}"
                } ?: "n/a"
                val cropCopy = crop?.let { android.graphics.Rect(it) }
                if (cropCopy != null && cropCopy == st.lastCropRect) {
                    st.cropStableFrames++
                } else {
                    st.lastCropRect = cropCopy
                    st.cropStableFrames = if (cropCopy != null) 1 else 0
                }
                st.lastDistortionMode = distortion
                st.lastZoomRatio = zoom

                if (logicalTimestamp != null &&
                    logicalTimestamp > 0L &&
                    sensorTs > 0L
                ) {
                    val delta = abs(sensorTs - logicalTimestamp)
                    st.timestampCompared++
                    st.timestampAbsDeltaSumNs += delta.toDouble()
                    st.timestampAbsDeltaMaxNs =
                        maxOf(st.timestampAbsDeltaMaxNs, delta)
                }
                physicalResultEntries++
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            result.physicalCameraTotalResults.forEach { (id, physical) ->
                record(id, physical)
            }
        } else {
            @Suppress("DEPRECATION")
            result.physicalCameraResults.forEach { (id, physical) ->
                record(id, physical)
            }
        }

        // The PLK110 V0.10 run proved that physical[3] is delivered with a
        // centered 112,84-3984,2988 crop while static K was scaled from the
        // full 4096x3072 active array. Wait until both crops are stable, then
        // start a new empirical-geometry epoch with crop-aware K.
        maybeApplyRuntimeCropIntrinsics()
    }

    fun updateScanState(
        enabled: Boolean,
        sessionId: String,
        allowTeleTexture: Boolean
    ) {
        if (sessionId != currentSessionId) {
            currentSessionId = sessionId
            lastTeleTextureQueuedNs = 0L
            teleTextureRegistered = 0
            teleTexturePoseMiss = 0
            teleTextureWriteFail = 0
            anchorSubmitCalls = 0L
            anchorSubmitAnchors = 0L
            anchorSubmitFailures = 0L
            physicalResultCallbacks = 0L
            physicalResultEntries = 0L
            runtimeIntrinsicEpoch = 0
            runtimeCropApplied = false
            runtimePrimaryK = primaryModel?.factoryK?.copyOf()
            runtimeSecondaryK = secondaryModel?.factoryK?.copyOf()
            runtimeCropReason = "waiting stable physical crop"
            synchronized(physicalResultLock) {
                physicalCaptureStats.clear()
            }
            runCatching { NativeBridge.nativeResetStereoAnchors() }
        }
        scanEnabled = enabled && active
        // V0.10 geometry pair on PLK110 is expected to be main+ultrawide.
        // Do not turn an empirical stereo pose into a texture-camera world pose.
        teleTextureGate =
            scanEnabled &&
                allowTeleTexture &&
                factoryGeometryTrusted &&
                geometryReady &&
                teleTextureRegistered < MAX_TELE_TEXTURE_KEYFRAMES
    }

    fun hudSummary(): String {
        val p = primaryModel
        val s = secondaryModel
        if (p == null || s == null) return "MultiCam: $status"

        val stats = FloatArray(NativeBridge.MULTICAM_STATS_SLOTS)
        val have = runCatching {
            NativeBridge.nativeGetMultiCamStats(stats)
        }.getOrDefault(false)

        return if (active && have) {
            val stereoStats = FloatArray(NativeBridge.STEREO_ANCHOR_STATS_SLOTS)
            val stereoCount = runCatching {
                NativeBridge.nativeGetStereoAnchorStats(stereoStats)
            }.getOrDefault(0)
            val scaleText =
                if (stereoCount > 14 && stereoStats[14] > 0.5f) {
                    "scale ${"%.3f".format(stereoStats[9])}✓"
                } else {
                    "scale warmup"
                }
            "MultiCam ${p.id}→${s.id} · " +
                "Δ${"%.2f".format(stats[9])}ms · " +
                "inlier ${stats[3].toInt()} · " +
                "anchor ${stats[17].toInt()} · $scaleText · " +
                "teleTex $teleTextureRegistered"
        } else if (active) {
            "MultiCam ${p.id}→${s.id} · geometry=${if (geometryReady) "ready" else "n/a"}"
        } else {
            "MultiCam fallback · $status"
        }
    }

    fun report(sb: StringBuilder) {
        sb.appendLine("[MULTICAM V0.11]")
        sb.appendLine("logicalCameraId=$logicalCameraId")
        sb.appendLine("active=$active")
        sb.appendLine("geometryReady=$geometryReady")
        sb.appendLine("status=$status")
        sb.appendLine("sessionMode=$sessionMode")
        sb.appendLine("pairSelectionReason=$pairSelectionReason")
        sb.appendLine("factoryGeometryTrusted=$factoryGeometryTrusted")
        sb.appendLine("allPhysicalIds=${logicalCharacteristics.physicalCameraIds.joinToString(",")}")
        sb.appendLine("primaryPhysicalId=${primaryModel?.id ?: "n/a"}")
        sb.appendLine("secondaryPhysicalId=${secondaryModel?.id ?: "n/a"}")
        sb.appendLine("primaryFocalMm=${primaryModel?.focalMm ?: -1f}")
        sb.appendLine("primaryHorizontalFovDeg=${primaryModel?.horizontalFovDeg ?: -1f}")
        sb.appendLine("secondaryFocalMm=${secondaryModel?.focalMm ?: -1f}")
        sb.appendLine("secondaryHorizontalFovDeg=${secondaryModel?.horizontalFovDeg ?: -1f}")
        sb.appendLine("focalRatioFx=$focalRatio")
        sb.appendLine("baselineMm=${baselineMeters * 1000f}")
        sb.appendLine(
            "sensorSync=" +
                when (syncType) {
                    CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_CALIBRATED ->
                        "CALIBRATED"
                    CameraCharacteristics.LOGICAL_MULTI_CAMERA_SENSOR_SYNC_TYPE_APPROXIMATE ->
                        "APPROXIMATE"
                    else -> "UNKNOWN"
                }
        )
        sb.appendLine("pairToleranceMs=${pairToleranceNs / 1_000_000f}")
        sb.appendLine("analysisSize=${configuredSize?.width ?: -1}x${configuredSize?.height ?: -1}")
        sb.appendLine("primaryFactoryK=${primaryModel?.factoryK?.joinToString(",") ?: "n/a"}")
        sb.appendLine("secondaryFactoryK=${secondaryModel?.factoryK?.joinToString(",") ?: "n/a"}")
        sb.appendLine("primaryK=${primaryModel?.k?.joinToString(",") ?: "n/a"}")
        sb.appendLine("secondaryK=${secondaryModel?.k?.joinToString(",") ?: "n/a"}")
        sb.appendLine("runtimeIntrinsicEpoch=$runtimeIntrinsicEpoch")
        sb.appendLine("runtimeCropApplied=$runtimeCropApplied")
        sb.appendLine("runtimeCropReason=$runtimeCropReason")
        sb.appendLine("runtimePrimaryK=${runtimePrimaryK?.joinToString(",") ?: "n/a"}")
        sb.appendLine("runtimeSecondaryK=${runtimeSecondaryK?.joinToString(",") ?: "n/a"}")
        sb.appendLine("primaryPoseReference=${primaryModel?.poseReference ?: -1}")
        sb.appendLine("secondaryPoseReference=${secondaryModel?.poseReference ?: -1}")
        sb.appendLine("primaryPoseRotation=${primaryModel?.poseRotation?.joinToString(",") ?: "n/a"}")
        sb.appendLine("primaryPoseTranslationM=${primaryModel?.poseTranslation?.joinToString(",") ?: "n/a"}")
        sb.appendLine("secondaryPoseRotation=${secondaryModel?.poseRotation?.joinToString(",") ?: "n/a"}")
        sb.appendLine("secondaryPoseTranslationM=${secondaryModel?.poseTranslation?.joinToString(",") ?: "n/a"}")
        appendPhysicalCameraInventory(sb)
        appendPhysicalCaptureResults(sb)
        sb.appendLine("pairDispatchCount=$pairDispatchCount")
        sb.appendLine("pairDroppedForTimestamp=$pairDroppedForTimestamp")
        sb.appendLine("anchorSubmitCalls=$anchorSubmitCalls")
        sb.appendLine("anchorSubmitAnchors=$anchorSubmitAnchors")
        sb.appendLine("anchorSubmitFailures=$anchorSubmitFailures")
        sb.appendLine("teleTextureRegistered=$teleTextureRegistered")
        sb.appendLine("teleTexturePoseMiss=$teleTexturePoseMiss")
        sb.appendLine("teleTextureWriteFail=$teleTextureWriteFail")

        val stats = FloatArray(NativeBridge.MULTICAM_STATS_SLOTS)
        if (runCatching { NativeBridge.nativeGetMultiCamStats(stats) }.getOrDefault(false)) {
            sb.appendLine("nativeConfigured=${stats[0] > 0.5f}")
            sb.appendLine("nativePairs=${stats[1].toLong()}")
            sb.appendLine("lastMatches=${stats[2].toInt()}")
            sb.appendLine("lastEpipolarInliers=${stats[3].toInt()}")
            sb.appendLine("lastTriangulated=${stats[4].toInt()}")
            sb.appendLine("nativeBaselineMm=${stats[5]}")
            sb.appendLine("medianStereoDepthM=${stats[6]}")
            sb.appendLine("stereoDepthP10M=${stats[7]}")
            sb.appendLine("stereoDepthP90M=${stats[8]}")
            sb.appendLine("lastPairDeltaMs=${stats[9]}")
            sb.appendLine("nativeFocalRatio=${stats[10]}")
            sb.appendLine("lastStereoProcessingMs=${stats[11]}")
            sb.appendLine("goodStereoPairs=${stats[12].toLong()}")
            sb.appendLine("medianReprojectionPx=${stats[14]}")
            sb.appendLine("nativeCalibratedSync=${stats[15] > 0.5f}")
            sb.appendLine("anchorCandidatesTotal=${stats[16].toLong()}")
            sb.appendLine("lastAnchorCount=${stats[17].toInt()}")
            sb.appendLine("lastMedianParallaxDeg=${stats[18]}")
            sb.appendLine("lastMedianAnchorConfidence=${stats[19]}")
            sb.appendLine("pairsWithAnchors=${stats[20].toLong()}")
            sb.appendLine("anchorsExportedTotal=${stats[21].toLong()}")
            sb.appendLine("anchorsRejectedQuality=${stats[22].toLong()}")
            sb.appendLine("lastPairAnchorSufficient=${stats[23] > 0.5f}")
            if (stats.size >= 32) {
                sb.appendLine("empiricalGeometryReady=${stats[24] > 0.5f}")
                sb.appendLine("empiricalAttempts=${stats[25].toLong()}")
                sb.appendLine("empiricalAccepted=${stats[26].toLong()}")
                sb.appendLine("empiricalGoodStreak=${stats[27].toInt()}")
                sb.appendLine("empiricalBadStreak=${stats[28].toInt()}")
                sb.appendLine("lastEmpiricalInliers=${stats[29].toInt()}")
                sb.appendLine("lastEmpiricalRotationDeltaDeg=${stats[30]}")
                sb.appendLine("lastEmpiricalTranslationDot=${stats[31]}")
            }
            if (stats.size >= 34) {
                sb.appendLine("nativeIntrinsicEpoch=${stats[32].toLong()}")
                sb.appendLine("nativeSecondaryFx=${stats[33]}")
            }
        }

        val stereoStats = FloatArray(NativeBridge.STEREO_ANCHOR_STATS_SLOTS)
        val stereoCount = runCatching {
            NativeBridge.nativeGetStereoAnchorStats(stereoStats)
        }.getOrDefault(0)
        if (stereoCount >= 41) {
            sb.appendLine()
            sb.appendLine("[STEREO METRIC ANCHORS V0.11]")
            sb.appendLine("submittedBatches=${stereoStats[0].toLong()}")
            sb.appendLine("submittedAnchors=${stereoStats[1].toLong()}")
            sb.appendLine("inputRejected=${stereoStats[2].toLong()}")
            sb.appendLine("pendingDropped=${stereoStats[3].toLong()}")
            sb.appendLine("depthFramesSeen=${stereoStats[4].toLong()}")
            sb.appendLine("depthMatchedBatches=${stereoStats[5].toLong()}")
            sb.appendLine("depthMatchedAnchors=${stereoStats[6].toLong()}")
            sb.appendLine("depthMatchMisses=${stereoStats[7].toLong()}")
            sb.appendLine("lastDepthMatchDeltaMs=${stereoStats[8]}")
            sb.appendLine("worldPerMeter=${stereoStats[9]}")
            sb.appendLine("lastFrameWorldPerMeter=${stereoStats[10]}")
            sb.appendLine("lastScaleMedianRel=${stereoStats[11]}")
            sb.appendLine("lastDenseStereoMedianRel=${stereoStats[12]}")
            sb.appendLine("lastScaleSamples=${stereoStats[13].toInt()}")
            sb.appendLine("worldScaleStable=${stereoStats[14] > 0.5f}")
            sb.appendLine("scaleAcceptedFrames=${stereoStats[15].toLong()}")
            sb.appendLine("scaleRejectedFrames=${stereoStats[16].toLong()}")
            sb.appendLine("scaleRejectedNoBaseCalib=${stereoStats[17].toLong()}")
            sb.appendLine("scaleRejectedFewSamples=${stereoStats[18].toLong()}")
            sb.appendLine("scaleRejectedResidual=${stereoStats[19].toLong()}")
            sb.appendLine("scaleRejectedJump=${stereoStats[20].toLong()}")
            sb.appendLine("calibratorStereoFrames=${stereoStats[21].toLong()}")
            sb.appendLine("calibratorStereoAccepted=${stereoStats[22].toLong()}")
            sb.appendLine("calibratorStereoRejected=${stereoStats[23].toLong()}")
            sb.appendLine("calibratorUniqueStereoSamples=${stereoStats[24].toLong()}")
            sb.appendLine("calibratorWeightedCopies=${stereoStats[25].toLong()}")
            sb.appendLine("geometryBatchesPopped=${stereoStats[26].toLong()}")
            sb.appendLine("geometryStaleBatches=${stereoStats[27].toLong()}")
            sb.appendLine("geometryDropNoPose=${stereoStats[28].toLong()}")
            sb.appendLine("geometryDropNoScale=${stereoStats[29].toLong()}")
            sb.appendLine("geometryDropResidual=${stereoStats[30].toLong()}")
            sb.appendLine("sceneTsdfBatches=${stereoStats[31].toLong()}")
            sb.appendLine("sceneTsdfInputAnchors=${stereoStats[32].toLong()}")
            sb.appendLine("sceneTsdfAcceptedAnchors=${stereoStats[33].toLong()}")
            sb.appendLine("sceneTsdfPasses=${stereoStats[34].toLong()}")
            sb.appendLine("targetTsdfBatches=${stereoStats[35].toLong()}")
            sb.appendLine("targetTsdfInputAnchors=${stereoStats[36].toLong()}")
            sb.appendLine("targetTsdfAcceptedAnchors=${stereoStats[37].toLong()}")
            sb.appendLine("targetTsdfPasses=${stereoStats[38].toLong()}")
            sb.appendLine("lastAnchorMedianMetricM=${stereoStats[39]}")
            sb.appendLine("lastAnchorMedianQuality=${stereoStats[40]}")
            if (stereoCount >= 48) {
                sb.appendLine("pendingAnchorQueue=${stereoStats[41].toInt()}")
                sb.appendLine("recentDepthQueue=${stereoStats[42].toInt()}")
                sb.appendLine("readyCalibrationQueue=${stereoStats[43].toInt()}")
                sb.appendLine("geometryQueue=${stereoStats[44].toInt()}")
                sb.appendLine("exactDepthMatchToleranceMs=${stereoStats[45]}")
                sb.appendLine("scaleGoodStreak=${stereoStats[46].toInt()}")
                sb.appendLine("scaleBadStreak=${stereoStats[47].toInt()}")
                sb.appendLine("scaleStableFramesRequired=3")
            }
            if (stereoCount >= 56) {
                sb.appendLine("depthExactMatchedBatches=${stereoStats[48].toLong()}")
                sb.appendLine("depthNearMatchedBatches=${stereoStats[49].toLong()}")
                sb.appendLine("depthAvgMatchDeltaMs=${stereoStats[50]}")
                sb.appendLine("depthMaxMatchDeltaMs=${stereoStats[51]}")
                sb.appendLine("lastTemporalPairScore=${stereoStats[52]}")
                sb.appendLine("stereoResetSerial=${stereoStats[53].toLong()}")
                sb.appendLine("depthNearestMatchToleranceMs=${stereoStats[54]}")
                sb.appendLine("stereoAnchorSchemaVersion=${stereoStats[55]}")
            } else {
                sb.appendLine("stereoAnchorSchemaVersion=0.9")
            }

            val diagnosis = when {
                !active -> "DISABLED_OR_FALLBACK"
                !geometryReady -> "EMPIRICAL_GEOMETRY_WARMUP_OR_REJECTED"
                stereoStats[1] < 1f -> "NO_STEREO_ANCHORS_SUBMITTED"
                stereoStats[5] < 1f -> "STEREO_DEPTH_TIMESTAMP_NOT_MATCHED"
                stereoStats[14] < 0.5f -> "STEREO_TO_VINS_SCALE_WARMUP_OR_REJECTED"
                stereoStats[33] < 1f -> "SCALE_OK_BUT_NO_SCENE_TSDF_ANCHOR_ACCEPTED"
                else -> "STEREO_METRIC_TSDF_ACTIVE"
            }
            sb.appendLine("v11Diagnosis=$diagnosis")
        }
        appendRuntimeDiagnostics(sb)
        sb.appendLine()
    }

    private fun maybeQueuePrimaryPair(image: Image) {
        if (!active) return
        val ts = image.timestamp
        if (lastPrimarySampleNs != 0L && ts - lastPrimarySampleNs < PAIR_PERIOD_NS) return
        lastPrimarySampleNs = ts
        latestPrimary = GrayFrame(ts, copyLuma(image))
        tryDispatchPair()
    }

    private fun maybeQueueSecondaryPair(image: Image) {
        if (!active) return
        val ts = image.timestamp
        if (lastSecondarySampleNs != 0L && ts - lastSecondarySampleNs < PAIR_PERIOD_NS) return
        lastSecondarySampleNs = ts
        latestSecondary = GrayFrame(ts, copyLuma(image))
        tryDispatchPair()
    }

    private fun tryDispatchPair() {
        val p = latestPrimary ?: return
        val s = latestSecondary ?: return
        val delta = p.timestampNs - s.timestampNs

        if (abs(delta) <= pairToleranceNs) {
            latestPrimary = null
            latestSecondary = null
            pairDispatchCount++
            val size = configuredSize ?: return
            fusionHandler.post {
                runCatching {
                    val goodPair = NativeBridge.nativeMultiCamOnPair(
                        p.gray,
                        s.gray,
                        size.width,
                        size.height,
                        p.timestampNs,
                        s.timestampNs
                    )

                    // Native empirical calibration owns geometryReady and needs
                    // multiple consistent synchronized pairs before anchors flow.
                    val mcStats = FloatArray(NativeBridge.MULTICAM_STATS_SLOTS)
                    if (NativeBridge.nativeGetMultiCamStats(mcStats) &&
                        mcStats.size > 24
                    ) {
                        geometryReady = mcStats[24] > 0.5f
                    }

                    // V0.10: submit only after empirical geometry is stable.
                    // The anchor timestamp is the PRIMARY physical camera timestamp,
                    // which is also the timestamp used by processImage()/DepthProvider.
                    if (goodPair && geometryReady && scanEnabled) {
                        val count = NativeBridge.nativeMultiCamGetAnchors(
                            stereoAnchorBuffer
                        ).coerceIn(0, MAX_STEREO_ANCHORS)

                        if (count > 0) {
                            anchorSubmitCalls++
                            val submitted =
                                NativeBridge.nativeSubmitStereoAnchors(
                                    stereoAnchorBuffer,
                                    count,
                                    p.timestampNs,
                                    size.width,
                                    size.height
                                )
                            if (submitted) {
                                anchorSubmitAnchors += count.toLong()
                            } else {
                                anchorSubmitFailures++
                            }
                        }
                    }
                }.onFailure {
                    Log.w(TAG, "native stereo metric-anchor pair failed", it)
                    anchorSubmitFailures++
                }
            }
        } else {
            pairDroppedForTimestamp++
            if (delta < 0L) {
                latestPrimary = null
            } else {
                latestSecondary = null
            }
        }
    }

    private fun maybeQueueTeleTexture(image: Image) {
        if (!active || !factoryGeometryTrusted || !geometryReady || !scanEnabled || !teleTextureGate) return
        if (teleTextureRegistered >= MAX_TELE_TEXTURE_KEYFRAMES) return

        val ts = image.timestamp
        if (lastTeleTextureQueuedNs != 0L &&
            ts - lastTeleTextureQueuedNs < TELE_TEXTURE_PERIOD_NS
        ) {
            return
        }
        lastTeleTextureQueuedNs = ts

        val size = configuredSize ?: return
        val sessionSnapshot = currentSessionId
        val nv21 = imageToNv21(image)
        val sharpness = lumaSharpness(nv21, size.width, size.height)

        fusionHandler.post {
            saveTeleTextureKeyframe(
                nv21,
                size.width,
                size.height,
                ts,
                sharpness,
                sessionSnapshot
            )
        }
    }

    private fun saveTeleTextureKeyframe(
        nv21: ByteArray,
        width: Int,
        height: Int,
        timestampNs: Long,
        sharpness: Float,
        sessionSnapshot: String
    ) {
        if (!scanEnabled || sessionSnapshot != currentSessionId) return

        val secondary = secondaryModel ?: return
        val rPrimaryFromSecondary = relativeRPrimaryFromSecondary ?: return
        val tPrimaryFromSecondary = relativeTPrimaryFromSecondary ?: return

        val primaryPose = FloatArray(NativeBridge.RENDER_POSE_SLOTS)
        val poseOk = runCatching {
            NativeBridge.nativeGetRenderPoseAt(timestampNs, primaryPose)
        }.getOrDefault(false)
        if (!poseOk) {
            teleTexturePoseMiss++
            return
        }

        val secondaryPose = transformCameraToWorldPose(
            primaryPose,
            rPrimaryFromSecondary,
            tPrimaryFromSecondary
        )

        val dir = File(context.filesDir, "hq_capture/$sessionSnapshot/multicam_tele")
        if (!dir.exists() && !dir.mkdirs()) {
            teleTextureWriteFail++
            return
        }
        val file = File(dir, "tele_${timestampNs}.jpg")

        val wrote = runCatching {
            FileOutputStream(file).use { output ->
                YuvImage(
                    nv21,
                    ImageFormat.NV21,
                    width,
                    height,
                    null
                ).compressToJpeg(
                    Rect(0, 0, width, height),
                    92,
                    output
                )
            }
        }.getOrDefault(false)

        if (!wrote || !file.exists() || file.length() <= 0L) {
            teleTextureWriteFail++
            runCatching { file.delete() }
            return
        }

        // Existing HQ stills reach 12MP and should remain preferred globally.
        // The tele frame still wins locally where its projected texel density is higher.
        val quality = (sharpness / 100f).coerceIn(0.30f, 1.80f)
        val registered = runCatching {
            NativeBridge.nativeRegisterTextureKeyframe(
                file.absolutePath,
                width,
                height,
                secondary.k[0],
                secondary.k[1],
                secondary.k[2],
                secondary.k[3],
                secondaryPose,
                quality,
                timestampNs
            )
        }.getOrDefault(false)

        if (registered) {
            teleTextureRegistered++
        } else {
            runCatching { file.delete() }
        }
    }

    private fun appendPhysicalCaptureResults(sb: StringBuilder) {
        sb.appendLine()
        sb.appendLine("[PHYSICAL CAPTURE RESULT V0.11]")
        sb.appendLine("physicalResultCallbacks=$physicalResultCallbacks")
        sb.appendLine("physicalResultEntries=$physicalResultEntries")

        synchronized(physicalResultLock) {
            if (physicalCaptureStats.isEmpty()) {
                sb.appendLine("physicalResults=EMPTY")
                return
            }

            physicalCaptureStats.toSortedMap().forEach { (id, st) ->
                val avgDeltaMs =
                    if (st.timestampCompared > 0L) {
                        st.timestampAbsDeltaSumNs /
                            st.timestampCompared.toDouble() /
                            1_000_000.0
                    } else {
                        -1.0
                    }

                sb.appendLine("physicalResult[$id].frames=${st.frames}")
                sb.appendLine(
                    "physicalResult[$id].lastSensorTimestampNs=" +
                        st.lastSensorTimestampNs
                )
                sb.appendLine(
                    "physicalResult[$id].timestampCompared=" +
                        st.timestampCompared
                )
                sb.appendLine(
                    "physicalResult[$id].avgAbsTimestampDeltaToLogicalMs=" +
                        avgDeltaMs
                )
                sb.appendLine(
                    "physicalResult[$id].maxAbsTimestampDeltaToLogicalMs=" +
                        st.timestampAbsDeltaMaxNs / 1_000_000.0
                )
                sb.appendLine(
                    "physicalResult[$id].lastExposureNs=" +
                        st.lastExposureNs
                )
                sb.appendLine(
                    "physicalResult[$id].lastFrameDurationNs=" +
                        st.lastFrameDurationNs
                )
                sb.appendLine(
                    "physicalResult[$id].lastRollingShutterSkewNs=" +
                        st.lastRollingShutterSkewNs
                )
                sb.appendLine(
                    "physicalResult[$id].lastIso=" +
                        st.lastIso
                )
                sb.appendLine(
                    "physicalResult[$id].lastFocalMm=" +
                        st.lastFocalMm
                )
                sb.appendLine(
                    "physicalResult[$id].lastFocusDistanceDiopters=" +
                        st.lastFocusDistance
                )
                sb.appendLine(
                    "physicalResult[$id].lastCropLTRB=" +
                        st.lastCrop
                )
                sb.appendLine(
                    "physicalResult[$id].cropStableFrames=" +
                        st.cropStableFrames
                )
                sb.appendLine(
                    "physicalResult[$id].lastDistortionMode=" +
                        st.lastDistortionMode
                )
                sb.appendLine(
                    "physicalResult[$id].lastZoomRatio=" +
                        st.lastZoomRatio
                )
            }
        }
    }

    private fun appendPhysicalCameraInventory(sb: StringBuilder) {
        sb.appendLine()
        sb.appendLine("[PHYSICAL CAMERA INVENTORY V0.11]")
        sb.appendLine(
            "logicalHardwareLevel=" +
                (logicalCharacteristics.get(
                    CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL
                ) ?: -1)
        )
        // 公开 SDK 没有 REQUEST_MAX_NUM_OUTPUT_STREAMS（那是 native metadata 的旧 tag
        // ANDROID_REQUEST_MAX_NUM_OUTPUT_STREAMS）。V0.9 修过一次、V0.10 与 V0.11 的
        // overlay 又各带回一次 → 这里第三次修，并已固化进 _patch/p_v11_*.py 便于重放。
        // Java 侧把「最大输出流数」拆成 processed / processed-stalling / raw 三个 key。
        // 纯诊断字段，不参与任何会话装配决策。
        val maxProcStreams = logicalCharacteristics.get(
            CameraCharacteristics.REQUEST_MAX_NUM_OUTPUT_PROC
        ) ?: 0
        val maxProcStallingStreams = logicalCharacteristics.get(
            CameraCharacteristics.REQUEST_MAX_NUM_OUTPUT_PROC_STALLING
        ) ?: 0
        val maxRawStreams = logicalCharacteristics.get(
            CameraCharacteristics.REQUEST_MAX_NUM_OUTPUT_RAW
        ) ?: 0
        sb.appendLine(
            "logicalMaxOutputStreams=" +
                (maxProcStreams + maxProcStallingStreams) +
                " (raw=" + maxRawStreams + ")"
        )
        sb.appendLine(
            "logicalMaxInputStreams=" +
                (logicalCharacteristics.get(
                    CameraCharacteristics.REQUEST_MAX_NUM_INPUT_STREAMS
                ) ?: -1)
        )
        sb.appendLine(
            "logicalTimestampSource=" +
                (logicalCharacteristics.get(
                    CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE
                ) ?: -1)
        )

        for (id in logicalCharacteristics.physicalCameraIds.sorted()) {
            val c = runCatching {
                cameraManager.getCameraCharacteristics(id)
            }.getOrNull()

            if (c == null) {
                sb.appendLine("physical[$id].readFailed=true")
                continue
            }

            val map = c.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            )
            val yuv = runCatching {
                map?.getOutputSizes(ImageFormat.YUV_420_888)
                    ?.sortedWith(
                        compareByDescending<Size> { it.width * it.height }
                            .thenByDescending { it.width }
                    )
            }.getOrNull()

            sb.appendLine("physical[$id].hardwareLevel=" +
                (c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: -1))
            sb.appendLine("physical[$id].focalLengthsMm=" +
                (c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].apertures=" +
                (c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].pixelArray=" +
                (c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
                    ?.let { "${it.width}x${it.height}" } ?: "n/a"))
            sb.appendLine("physical[$id].sensorSizeMm=" +
                (c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    ?.let { "${it.width}x${it.height}" } ?: "n/a"))
            sb.appendLine("physical[$id].activeArray=" +
                (c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                    ?.let { "${it.width()}x${it.height()}" } ?: "n/a"))
            sb.appendLine("physical[$id].preCorrectionArray=" +
                (c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
                    ?.let { "${it.width()}x${it.height()}" } ?: "n/a"))
            sb.appendLine("physical[$id].intrinsic=" +
                (c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].distortion=" +
                (c.get(CameraCharacteristics.LENS_DISTORTION)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].distortionCorrectionModes=" +
                (c.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].poseReference=" +
                (c.get(CameraCharacteristics.LENS_POSE_REFERENCE) ?: -1))
            sb.appendLine("physical[$id].poseRotation=" +
                (c.get(CameraCharacteristics.LENS_POSE_ROTATION)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].poseTranslationM=" +
                (c.get(CameraCharacteristics.LENS_POSE_TRANSLATION)
                    ?.joinToString(",") ?: "n/a"))
            sb.appendLine("physical[$id].timestampSource=" +
                (c.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ?: -1))
            sb.appendLine("physical[$id].yuv420Sizes=" +
                (yuv?.joinToString(",") { "${it.width}x${it.height}" } ?: "n/a"))
            sb.appendLine(
                "physical[$id].supportsAnalysisSize=" +
                    (configuredSize?.let { wanted ->
                        yuv?.any {
                            it.width == wanted.width &&
                                it.height == wanted.height
                        } == true
                    } ?: false)
            )
            val minDurationNs = configuredSize?.let { wanted ->
                runCatching {
                    map?.getOutputMinFrameDuration(
                        ImageFormat.YUV_420_888,
                        wanted
                    )
                }.getOrNull()
            }
            sb.appendLine(
                "physical[$id].analysisMinFrameDurationNs=" +
                    (minDurationNs ?: -1L)
            )
            sb.appendLine(
                "physical[$id].analysisTheoreticalMaxFps=" +
                    if (minDurationNs != null && minDurationNs > 0L) {
                        1_000_000_000.0 / minDurationNs.toDouble()
                    } else {
                        -1.0
                    }
            )
        }
    }

    private fun appendRuntimeDiagnostics(sb: StringBuilder) {
        sb.appendLine()
        sb.appendLine("[RUNTIME V0.11]")
        sb.appendLine("sdk=${Build.VERSION.SDK_INT}")
        sb.appendLine("manufacturer=${Build.MANUFACTURER}")
        sb.appendLine("model=${Build.MODEL}")
        sb.appendLine("device=${Build.DEVICE}")
        sb.appendLine("hardware=${Build.HARDWARE}")
        sb.appendLine("displayBuild=${Build.DISPLAY}")
        sb.appendLine("fingerprint=${Build.FINGERPRINT}")
        sb.appendLine("availableProcessors=${Runtime.getRuntime().availableProcessors()}")
        val rt = Runtime.getRuntime()
        sb.appendLine("javaHeapUsedMb=${(rt.totalMemory() - rt.freeMemory()) / 1048576f}")
        sb.appendLine("javaHeapMaxMb=${rt.maxMemory() / 1048576f}")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val pm = context.getSystemService(Context.POWER_SERVICE) as? android.os.PowerManager
            sb.appendLine("thermalStatus=${pm?.currentThermalStatus ?: -1}")
        } else {
            sb.appendLine("thermalStatus=-1")
        }
    }

    private fun copyLuma(image: Image): ByteArray {
        val w = image.width
        val h = image.height
        val p = image.planes[0]
        val b = p.buffer.duplicate()
        val out = ByteArray(w * h)

        var dst = 0
        for (row in 0 until h) {
            val rowBase = row * p.rowStride
            if (p.pixelStride == 1) {
                b.position(rowBase)
                b.get(out, dst, w)
                dst += w
            } else {
                for (col in 0 until w) {
                    out[dst++] = b.get(rowBase + col * p.pixelStride)
                }
            }
        }
        return out
    }

    /** Robust YUV_420_888 -> NV21 conversion, including row/pixel padding. */
    private fun imageToNv21(image: Image): ByteArray {
        val w = image.width
        val h = image.height
        val out = ByteArray(w * h + w * h / 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer.duplicate()
        val uBuf = uPlane.buffer.duplicate()
        val vBuf = vPlane.buffer.duplicate()

        var dst = 0
        for (row in 0 until h) {
            val base = row * yPlane.rowStride
            for (col in 0 until w) {
                out[dst++] = yBuf.get(base + col * yPlane.pixelStride)
            }
        }

        for (row in 0 until h / 2) {
            val uBase = row * uPlane.rowStride
            val vBase = row * vPlane.rowStride
            for (col in 0 until w / 2) {
                out[dst++] = vBuf.get(vBase + col * vPlane.pixelStride)
                out[dst++] = uBuf.get(uBase + col * uPlane.pixelStride)
            }
        }
        return out
    }

    /** Sparse-grid Laplacian variance; cheap enough for the ~1 Hz tele capture gate. */
    private fun lumaSharpness(nv21: ByteArray, w: Int, h: Int): Float {
        if (w < 8 || h < 8 || nv21.size < w * h) return 0f

        var sum = 0.0
        var sum2 = 0.0
        var n = 0
        val step = 4

        fun luma(x: Int, y: Int): Int = nv21[y * w + x].toInt() and 0xff

        var y = step
        while (y < h - step) {
            var x = step
            while (x < w - step) {
                val c = luma(x, y)
                val lap = 4 * c -
                    luma(x - step, y) -
                    luma(x + step, y) -
                    luma(x, y - step) -
                    luma(x, y + step)

                val v = lap.toDouble()
                sum += v
                sum2 += v * v
                n++
                x += step
            }
            y += step
        }

        if (n <= 1) return 0f
        val mean = sum / n
        return max(0.0, sum2 / n - mean * mean).toFloat()
    }

    private fun buildCameraModel(id: String, output: Size): CameraModel? {
        val c = runCatching {
            cameraManager.getCameraCharacteristics(id)
        }.getOrNull() ?: return null

        val rect = c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
            ?: c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: return null

        val arrW = rect.width().toFloat()
        val arrH = rect.height().toFloat()
        if (arrW <= 0f || arrH <= 0f) return null

        val calibration = c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
        val focalMm = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            ?.firstOrNull() ?: 0f
        val physicalSize = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

        val fx0 = calibration?.getOrNull(0)
            ?: if (focalMm > 0f && physicalSize != null && physicalSize.width > 0f) {
                focalMm / physicalSize.width * arrW
            } else {
                arrW * 0.9f
            }

        val fy0 = calibration?.getOrNull(1)
            ?: if (focalMm > 0f && physicalSize != null && physicalSize.height > 0f) {
                focalMm / physicalSize.height * arrH
            } else {
                arrW * 0.9f
            }

        val cx0 = calibration?.getOrNull(2) ?: (arrW * 0.5f)
        val cy0 = calibration?.getOrNull(3) ?: (arrH * 0.5f)

        // Same center aspect-crop convention used by HqCaptureController.
        val sensorAspect = arrW / arrH
        val outputAspect = output.width.toFloat() / output.height.toFloat()
        var cropLeft = 0f
        var cropTop = 0f
        var cropW = arrW
        var cropH = arrH

        if (sensorAspect > outputAspect) {
            cropW = arrH * outputAspect
            cropLeft = (arrW - cropW) * 0.5f
        } else if (sensorAspect < outputAspect) {
            cropH = arrW / outputAspect
            cropTop = (arrH - cropH) * 0.5f
        }

        val sx = output.width.toFloat() / cropW
        val sy = output.height.toFloat() / cropH

        val k = floatArrayOf(
            fx0 * sx,
            fy0 * sy,
            (cx0 - cropLeft) * sx,
            (cy0 - cropTop) * sy
        )

        val horizontalFovDeg =
            if (focalMm > 0f && physicalSize != null && physicalSize.width > 0f) {
                (2.0 * atan(
                    physicalSize.width.toDouble() / (2.0 * focalMm.toDouble())
                ) * 180.0 / PI).toFloat()
            } else {
                0f
            }

        return CameraModel(
            id = id,
            focalMm = focalMm,
            horizontalFovDeg = horizontalFovDeg,
            factoryK = k.copyOf(),
            k = k,
            poseRotation = c.get(CameraCharacteristics.LENS_POSE_ROTATION)?.copyOf(),
            poseTranslation = c.get(CameraCharacteristics.LENS_POSE_TRANSLATION)?.copyOf(),
            poseReference = c.get(CameraCharacteristics.LENS_POSE_REFERENCE)
                ?: CameraCharacteristics.LENS_POSE_REFERENCE_UNDEFINED
        )
    }

    /**
     * Convert Camera2's actual SCALER_CROP_REGION into the intrinsics of the
     * delivered analysis YUV. This is only applied when active/pre-correction
     * arrays share the same coordinate system; otherwise empirical calibration
     * stays on factory K and the report says why.
     */
    private fun cropAwareK(
        model: CameraModel,
        crop: android.graphics.Rect,
        output: Size
    ): FloatArray? {
        val c = runCatching {
            cameraManager.getCameraCharacteristics(model.id)
        }.getOrNull() ?: return null

        val active =
            c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                ?: return null
        val pre =
            c.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
                ?: active

        if (active.left != pre.left ||
            active.top != pre.top ||
            active.right != pre.right ||
            active.bottom != pre.bottom
        ) {
            runtimeCropReason =
                "active/preCorrection coordinates differ; crop-aware K skipped"
            return null
        }

        if (crop.width() <= 0 || crop.height() <= 0 ||
            crop.left < active.left || crop.top < active.top ||
            crop.right > active.right || crop.bottom > active.bottom
        ) {
            return null
        }

        val intrinsic =
            c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                ?: return null
        if (intrinsic.size < 4) return null

        val sx = output.width.toFloat() / crop.width().toFloat()
        val sy = output.height.toFloat() / crop.height().toFloat()

        return floatArrayOf(
            intrinsic[0] * sx,
            intrinsic[1] * sy,
            (intrinsic[2] - crop.left.toFloat()) * sx,
            (intrinsic[3] - crop.top.toFloat()) * sy
        )
    }

    private fun kRelativeChange(a: FloatArray, b: FloatArray): Float {
        if (a.size < 4 || b.size < 4) return Float.POSITIVE_INFINITY
        var m = 0f
        for (i in 0 until 4) {
            val denom = maxOf(1f, abs(a[i]))
            m = maxOf(m, abs(b[i] - a[i]) / denom)
        }
        return m
    }

    private fun maybeApplyRuntimeCropIntrinsics() {
        val primary = primaryModel ?: return
        val secondary = secondaryModel ?: return
        val output = configuredSize ?: return

        val pState: PhysicalCaptureStats
        val sState: PhysicalCaptureStats
        synchronized(physicalResultLock) {
            pState = physicalCaptureStats[primary.id]?.copy() ?: return
            sState = physicalCaptureStats[secondary.id]?.copy() ?: return
        }

        // 15 consecutive identical results is enough to avoid reacting to a
        // transitional crop during session startup.
        if (pState.cropStableFrames < 15 || sState.cropStableFrames < 15) {
            runtimeCropReason =
                "waiting stable crops p=${pState.cropStableFrames} s=${sState.cropStableFrames}"
            return
        }

        val pCrop = pState.lastCropRect ?: return
        val sCrop = sState.lastCropRect ?: return
        val pk = cropAwareK(primary, pCrop, output) ?: return
        val sk = cropAwareK(secondary, sCrop, output) ?: return

        val pChange = kRelativeChange(primary.k, pk)
        val sChange = kRelativeChange(secondary.k, sk)
        if (maxOf(pChange, sChange) < 0.0025f) {
            runtimePrimaryK = pk
            runtimeSecondaryK = sk
            runtimeCropApplied = true
            runtimeCropReason =
                "stable crop K already current"
            return
        }

        val updated = runCatching {
            NativeBridge.nativeMultiCamUpdateIntrinsics(pk, sk)
        }.getOrDefault(false)

        if (!updated) {
            runtimeCropReason = "native runtime-K update rejected"
            return
        }

        pk.copyInto(primary.k)
        sk.copyInto(secondary.k)
        runtimePrimaryK = pk
        runtimeSecondaryK = sk
        runtimeIntrinsicEpoch++
        runtimeCropApplied = true
        geometryReady = false
        focalRatio =
            if (pk[0] > 1e-3f) sk[0] / pk[0] else 1f

        // Any metric anchor created with the previous K belongs to the previous
        // calibration epoch and must not be mixed with the new one.
        runCatching { NativeBridge.nativeResetStereoAnchors() }

        runtimeCropReason = buildString {
            append("crop-aware K epoch ").append(runtimeIntrinsicEpoch)
            append(" pChange=").append("%.4f".format(pChange))
            append(" sChange=").append("%.4f".format(sChange))
            append(" pCrop=").append(pCrop)
            append(" sCrop=").append(sCrop)
        }
        Log.i(TAG, runtimeCropReason)
    }

    private fun factoryBaselineMeters(
        a: CameraModel,
        b: CameraModel
    ): Float? {
        if (a.poseReference == CameraCharacteristics.LENS_POSE_REFERENCE_UNDEFINED ||
            b.poseReference == CameraCharacteristics.LENS_POSE_REFERENCE_UNDEFINED ||
            a.poseReference != b.poseReference
        ) {
            return null
        }
        val ca = a.poseTranslation ?: return null
        val cb = b.poseTranslation ?: return null
        if (ca.size < 3 || cb.size < 3) return null
        return vectorNorm(
            floatArrayOf(
                cb[0] - ca[0],
                cb[1] - ca[1],
                cb[2] - ca[2]
            )
        )
    }

    /**
     * Camera2: R_i maps common sensor coordinates -> camera_i, C_i is optical
     * center in common coordinates.
     *
     * secondary -> primary:
     *   R_ps = R_p * R_s^T
     *   t_ps = R_p * (C_s - C_p)
     */
    private fun computePrimaryFromSecondary(
        primary: CameraModel,
        secondary: CameraModel
    ): Triple<FloatArray, FloatArray, Float>? {
        if (primary.poseReference == CameraCharacteristics.LENS_POSE_REFERENCE_UNDEFINED ||
            secondary.poseReference == CameraCharacteristics.LENS_POSE_REFERENCE_UNDEFINED ||
            primary.poseReference != secondary.poseReference
        ) {
            return null
        }

        val qp = primary.poseRotation
        val cp = primary.poseTranslation
        val qs = secondary.poseRotation
        val cs = secondary.poseTranslation

        if (qp == null || cp == null || qs == null || cs == null ||
            qp.size < 4 || qs.size < 4 || cp.size < 3 || cs.size < 3
        ) {
            return null
        }

        val rp = quaternionToMatrix(qp)
        val rs = quaternionToMatrix(qs)
        val rps = multiply3(rp, transpose3(rs))
        val deltaC = floatArrayOf(
            cs[0] - cp[0],
            cs[1] - cp[1],
            cs[2] - cp[2]
        )
        val tps = multiplyMatVec3(rp, deltaC)
        return Triple(rps, tps, vectorNorm(deltaC))
    }

    /**
     * T_w_secondary = T_w_primary * T_primary_secondary.
     */
    private fun transformCameraToWorldPose(
        primaryPose12: FloatArray,
        rPrimaryFromSecondary: FloatArray,
        tPrimaryFromSecondary: FloatArray
    ): FloatArray {
        val rWp = primaryPose12.copyOfRange(0, 9)
        val tWp = floatArrayOf(
            primaryPose12[9],
            primaryPose12[10],
            primaryPose12[11]
        )
        val rWs = multiply3(rWp, rPrimaryFromSecondary)
        val rotatedOffset = multiplyMatVec3(rWp, tPrimaryFromSecondary)

        return FloatArray(12).also { out ->
            for (i in 0 until 9) out[i] = rWs[i]
            out[9] = tWp[0] + rotatedOffset[0]
            out[10] = tWp[1] + rotatedOffset[1]
            out[11] = tWp[2] + rotatedOffset[2]
        }
    }

    private fun quaternionToMatrix(q: FloatArray): FloatArray {
        var x = q[0]
        var y = q[1]
        var z = q[2]
        var w = q[3]

        val n = sqrt(x * x + y * y + z * z + w * w)
        if (n > 1e-8f) {
            x /= n
            y /= n
            z /= n
            w /= n
        }

        return floatArrayOf(
            1f - 2f * (y * y + z * z),
            2f * (x * y - z * w),
            2f * (x * z + y * w),

            2f * (x * y + z * w),
            1f - 2f * (x * x + z * z),
            2f * (y * z - x * w),

            2f * (x * z - y * w),
            2f * (y * z + x * w),
            1f - 2f * (x * x + y * y)
        )
    }

    private fun transpose3(a: FloatArray): FloatArray = floatArrayOf(
        a[0], a[3], a[6],
        a[1], a[4], a[7],
        a[2], a[5], a[8]
    )

    private fun multiply3(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(9)
        for (row in 0..2) {
            for (col in 0..2) {
                var v = 0f
                for (k in 0..2) {
                    v += a[row * 3 + k] * b[k * 3 + col]
                }
                out[row * 3 + col] = v
            }
        }
        return out
    }

    private fun multiplyMatVec3(a: FloatArray, v: FloatArray): FloatArray =
        floatArrayOf(
            a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
            a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
            a[6] * v[0] + a[7] * v[1] + a[8] * v[2]
        )

    private fun vectorNorm(v: FloatArray): Float =
        sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    fun close() {
        active = false
        geometryReady = false
        scanEnabled = false
        teleTextureGate = false

        runCatching { primaryReader?.close() }
        runCatching { secondaryReader?.close() }
        primaryReader = null
        secondaryReader = null

        latestPrimary = null
        latestSecondary = null
        runCatching { NativeBridge.nativeMultiCamReset() }
        runCatching { NativeBridge.nativeResetStereoAnchors() }
        fusionThread.quitSafely()
    }
}
