package com.mobilescan3d

import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.MeteringRectangle
import android.media.Image
import android.media.ImageReader
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.PowerManager
import android.os.HandlerThread
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.TextView
import com.mobilescan3d.depth.DepthProvider
import com.mobilescan3d.depth.DepthProviderFactory
import com.mobilescan3d.export.ExportManager
import com.mobilescan3d.persistence.ScanPackageManager
import com.mobilescan3d.render.TexturedArAssetLoader
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity(), SensorEventListener {

    private var lastModelFile: java.io.File? = null
    private var pendingModelCopy: java.io.File? = null
    private val saveModelLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("model/gltf-binary")
    ) { uri ->
        val source = pendingModelCopy
        pendingModelCopy = null
        if (uri != null && source != null) {
            val resolver = applicationContext.contentResolver
            kotlin.concurrent.thread(name = "SaveGlb", isDaemon = true) {
                val message = try {
                    source.inputStream().use { input ->
                        val output = resolver.openOutputStream(uri, "wt")
                            ?: error("无法打开目标文件")
                        output.use { input.copyTo(it) }
                    }
                    "GLB 模型已保存到所选位置"
                } catch (t: Throwable) {
                    "保存失败：${t.message}"
                } finally {
                    source.delete()
                }
                runOnUiThread { if (!isDestroyed) toast(message) }
            }
        } else {
            source?.delete()
        }
    }

    private lateinit var cameraManager: CameraManager
    private lateinit var sensorManager: SensorManager
    private lateinit var texture: TextureView
    private lateinit var glView: GLSurfaceView
    // SurfaceView 的首帧合成在部分 OnePlus HAL 上会在权限 Activity 返回时留下黑色缓冲。
    // 先让 TextureView 独占显示，收到真实预览帧后再挂载透明 GL 叠加层。
    @Volatile private var previewFrameSeen = false
    private lateinit var renderer: PointCloudRenderer

    private lateinit var primaryButton: android.widget.Button
    private lateinit var headerStatus: TextView
    private lateinit var warningBanner: TextView
    private lateinit var statusBarText: TextView
    private lateinit var hudText: TextView

    // ---- V0.13.31：替换包 UI 状态卡（100% 设计还原，结构化单项）----
    // headerStatus/hudText/statusBarText 影子化后，数据拆解写入以下包设计 id。
    private lateinit var tvFps: TextView
    private lateinit var tvDistance: TextView
    private lateinit var tvAutoLight: TextView
    private lateinit var tvCameraMode: TextView
    private lateinit var tvDepth: TextView
    private lateinit var tvMapCount: TextView
    private lateinit var tvDrawCount: TextView
    private lateinit var tvUnit: TextView
    private lateinit var hudCompact: TextView
    private lateinit var pauseButton: android.widget.Button
    private lateinit var scanReticleView: com.mobilescan3d.ui.ScanReticleView
    private lateinit var scanGuidanceOverlay: com.mobilescan3d.ui.guidance.ScanGuidanceOverlay
    private lateinit var tvGuidanceStep: TextView
    private val scanGuidanceController = com.mobilescan3d.scan.guidance.ScanGuidanceController()
    @Volatile private var guidedScanEnabled = true
    private lateinit var scanProgressView: android.widget.ProgressBar
    private lateinit var tvScanProgressPercent: TextView
    private lateinit var tvMotionState: TextView
    private lateinit var tvDistanceState: TextView
    private lateinit var tvTrackingState: TextView
    private lateinit var tvViewpointSummary: TextView
    private lateinit var tvScanStateBadge: TextView
    private lateinit var mainScanUi: android.view.View
    private lateinit var viewerBackdrop: android.view.View
    private lateinit var modelViewerOverlay: android.view.View
    private lateinit var tvViewerStats: TextView
    private lateinit var viewerOptimizeButton: android.view.View
    private lateinit var viewerOptimizeLabel: TextView
    @Volatile private var viewerMeshOptimized = false
    private lateinit var toolAutoLabel: TextView
    private lateinit var toolModeLabel: TextView
    private lateinit var toolLightLabel: TextView
    private lateinit var toolCameraLabel: TextView
    private lateinit var toolLockLabel: TextView
    private lateinit var viewerModeTexture: TextView
    private lateinit var viewerModeSolid: TextView
    private lateinit var viewerModeWire: TextView
    private lateinit var viewerModeInspect: TextView
    private lateinit var viewerModeSurface: TextView
    private lateinit var reviewProgressSurface: android.widget.ProgressBar
    private lateinit var tvReviewSurface: TextView
    private var lastSurfaceCoveragePercent = 0
    private var lastSurfaceRobustPercent = 0
    private var viewerBaseMesh: com.mobilescan3d.export.ExportManager.MeshData? = null
    private var viewerSurfaceVertices: FloatArray? = null
    @Volatile private var recoveryCheckpointBusy = false
    private var recoveryLastCheckpointMs = 0L
    private var recoveryPromptConsumed = false
    private lateinit var tvViewerDimensions: TextView
    private lateinit var tvViewerTopology: TextView
    @Volatile private var viewerAnalysisToken = 0
    private lateinit var viewpointCoverageView: com.mobilescan3d.ui.ViewpointCoverageView
    private lateinit var scanReviewOverlay: android.view.View
    private lateinit var reviewContinueButton: android.widget.Button
    private lateinit var reviewGenerateButton: android.widget.Button
    private lateinit var reviewDiscardButton: android.widget.Button
    private lateinit var tvReviewTopHint: TextView
    private lateinit var tvReviewScore: TextView
    private lateinit var tvReviewRecommendation: TextView
    private lateinit var tvReviewMissing: TextView
    private lateinit var reviewProgressCapture: android.widget.ProgressBar
    private lateinit var reviewProgressViewpoint: android.widget.ProgressBar
    private lateinit var reviewProgressGeometry: android.widget.ProgressBar
    private lateinit var reviewProgressTexture: android.widget.ProgressBar
    private lateinit var tvReviewCapture: TextView
    private lateinit var tvReviewViewpoint: TextView
    private lateinit var tvReviewGeometry: TextView
    private lateinit var tvReviewTexture: TextView

    // Round 3: scan-review + spatial viewpoint-coverage state.
    @Volatile private var scanReviewActive = false
    private var reviewInterrupted = false
    private val viewpointCoverage = FloatArray(14) // 12 orbit + top + bottom
    private val coveragePose = FloatArray(NativeBridge.RENDER_POSE_SLOTS)
    private val coverageAnchorWorld = FloatArray(3)
    private var coverageAnchorValid = false
    private var coverageCurrentSector = -1
    private var coverageCurrentElevationDeg = 0f
    private var coverageLastSampleMs = 0L
    private var lastViewpointCoveragePercent = 0
    private var lastGeometryQualityPercent = 0
    private var lastTextureQualityPercent = 0
    private var lastCoverageGuidance = ""

    // vc183.2: review -> targeted re-scan -> automatic review loop.
    private var reviewRecoveryPlan =
        com.mobilescan3d.scan.guidance.ScanRecoveryPlanner.Plan(emptyList(), "")
    private var recoveryRound = 0
    private var recoveryActive = false
    private var recoveryStartedMs = 0L
    private var recoveryReadySinceMs = 0L

    // Round 2 scanner UX state. These are presentation-layer signals only;
    // they do not alter reconstruction math or calibration.
    @Volatile private var scanPaused = false
    @Volatile private var scanAngularSpeedDps = 0f
    @Volatile private var latestScanDistanceMeters = 0f
    @Volatile private var latestScanDistanceMs = 0L
    @Volatile private var targetMedianDepthMeters = 0f
    private var lastScanSufficiency = 0
    private val scanUiMetrics = FloatArray(NativeBridge.SCAN_UI_METRICS_SLOTS)

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    /** V0.8: logical CameraDevice + two explicitly bound physical YUV streams. */
    private var multiCam: MultiCameraFusionController? = null
    private val cameraRangePolicy = CameraRangePolicy() // camera thread only
    private val cameraDistanceBuffer = FloatArray(4) // depth thread only
    @Volatile private var cameraRangeStatus = "自动 · 等待有效深度"
    @Volatile private var cameraRangeSwitching = false
    private val cameraRangeSerial = java.util.concurrent.atomic.AtomicLong(0)
    private var rangeLogicalSurfaces = emptyList<Surface>()
    // vc161：镜头调配链的可机读证据。改状态立即打，稳定态 5s 心跳；硬下限 1s 防刷屏。
    private var cameraRangeLogLast = ""
    private var cameraRangeLogMinGapMs = 0L
    private var cameraRangeLogHeartbeatMs = 0L
    @Volatile private var cameraRangeSwitchStartedMs = 0L
    private var rangeHqActive = false
    private var rangeTimeout: Runnable? = null
    private var lastDepthScaleSampleNs = 0L
    @Volatile private var latestCameraTimestampNs = 0L
    private var lastRangeDepthTimestampNs = 0L

    private lateinit var hqCapture: HqCaptureController
    private var cameraThread: HandlerThread? = null
    private var sensorThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private val started = AtomicBoolean(false)
    private var lastSensorNs = 0L
    private var lastImu = FloatArray(6)
    private var hasAcc = false
    private var hasGyr = false
    private var lastAccNs = 0L
    private var lastGyrNs = 0L
    private var lastImuOutNs = 0L
    private lateinit var depthProvider: DepthProvider
    private var depthThread: HandlerThread? = null
    private var depthHandler: Handler? = null

    /**
     * 深度来源的能力探测结论。
     * 硬件 DEPTH16 只做探测并进诊断报告 —— 打开它是**另一颗相机**，
     * 与 RGB 主摄不同步且外参未标定，把它设成默认值等于在一条已经
     * 能工作的链路上引入未经验证的变量。详见 DepthProviderFactory。
     */
    private var depthProbe = DepthProviderFactory.Probe.UNKNOWN

    /** PLY / GLB 导出与网格构建（构建在后台线程，绝不占 UI 线程）。 */
    private lateinit var exportManager: ExportManager
    /** Round 8 cumulative multi-session mesh base. */
    private lateinit var segmentStore: com.mobilescan3d.multiscan.ScanSegmentStore
    @Volatile private var multiSegmentMergedActive = false
    @Volatile private var segmentMergeBusy = false
    private var latestQualityReport: java.io.File? = null

    // Main-thread guard spans reconstruction, texture baking, and package saving.
    private var modelOperationBusy = false

    /** 最近一次网格构建结论（HUD / 报告）。 */
    @Volatile private var lastMeshSummary = "n/a"
    /** 最近一次导出的 GLB —— 新的最终产物。 */
    @Volatile private var lastGlbFilename: String? = null
    @Volatile private var lastGlbTriangles: Int = 0
    @Volatile private var lastGlbBytes: Long = 0L

    /** 深度标定读取缓冲，复用避免每帧分配。 */
    private val calibBuf = FloatArray(NativeBridge.DEPTH_CALIBRATION_SLOTS)

    private val depthBusy = AtomicBoolean(false)
    @Volatile private var depthCompletedMs = 0L
    @Volatile private var depthAttemptCompletedMs = 0L
    @Volatile private var depthErrors = 0L
    @Volatile private var depthLastError = ""
    @Volatile private var scanMaxDistanceMeters = 1f
    @Volatile private var autoFillLight = true
    @Volatile private var torchRequested = false
    @Volatile private var torchAvailable = false
    @Volatile private var torchFailed = false
    private val autoLightPolicy = AutoLightPolicy()
    // V0.5：会话世代号。停止/重开扫描时自增，用来丢弃「晚到达」的深度推理
    // 结果 —— 否则上一轮会话的深度会喂进已经被 reset 的 TSDF。
    @Volatile private var depthGeneration = 0L

    /**
     * vc159 ThermalGuard：系统热状态驱动的**自适应降载**。
     *
     * 实拍数据（vc157，OnePlus 15，08:43–09:11）：
     *  - 相机 30fps + VINS(OpenCV) + GPU 渲染持续满载；
     *  - 单目深度模型 `depth_model.tflite` **66MB**，XNNPACK **CPU** 推理
     *    单帧 p50 ≈ 265–386 ms（且随拍摄单调恶化），串行跑 ≈ 常驻占满约一个核；
     *  - `nativeBuildMesh` 的 `rawTris` 无上限增长到 82 万，构建 5–11 s。
     *  结果是 20 分钟就把 8 个核顶到 ~100 ℃、GPU 75–79 ℃、电池 47.7 ℃，
     *  系统落到 **Thermal Status 3(SEVERE)** 并开始降频。
     *
     * 本机制**只在设备已经变热时**下调本 App 自己的重活频率（深度推理间隔、
     * 网格重建间隔），凉下来自动恢复 —— 冷机时不回调、零影响。
     */
    @Volatile private var thermalStatus: Int = PowerManager.THERMAL_STATUS_NONE
    private var thermalListener: PowerManager.OnThermalStatusChangedListener? = null

    private var captureSize: android.util.Size? = null
    private var previewSize: android.util.Size? = null
    private var viewWidth = 0
    private var viewHeight = 0
    private var sensorOrientation = 90
    private var fixedFocus = false
    private var stabilization = false
    private var deviceModel = ""
    private var oisSupported = false
    private var afModes = intArrayOf()
    private var videoStabModes = intArrayOf()
    private var activeArray = Rect()
    private var focusRegion: MeteringRectangle? = null
    private var focusTriggered = false
    private var downX = 0f
    private var downY = 0f
    private var pendingReport: String? = null
    private val createReportLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain")
    ) { uri ->
        val text = pendingReport
        pendingReport = null
        if (uri != null && text != null) {
            try {
                contentResolver.openOutputStream(uri)?.use { out ->
                    out.write(text.toByteArray())
                }
                android.widget.Toast.makeText(this, "报告已保存", android.widget.Toast.LENGTH_LONG).show()
            } catch (t: Throwable) {
                android.util.Log.e("FeedbackReport", "write report failed", t)
                android.widget.Toast.makeText(
                    this,
                    "报告写入失败：${t.javaClass.simpleName}: ${t.message ?: "unknown"}",
                    android.widget.Toast.LENGTH_LONG
                ).show()
            }
        }
    }
    private var previewSurface: Surface? = null
    @Volatile private var scanning = false
    // V0.5：native 会话（nativeCreate 及体素/标定配置）就绪后置 true。
    // nativeCreate 之前就把帧送进 native 会写进尚未初始化的状态。
    @Volatile private var scanNativeReady = false
    // V0.5：停扫后进入「AR 查看」态 —— 只继续跑 VINS 拿位姿（模型才能钉在
    // 真实世界里），深度推理与 TSDF 融合都停掉。
    @Volatile private var arMeshViewing = false
private var lastRelocPollMs = 0L
    private var lastRelocWasLocalized = false
    @Volatile private var relocalizationBusy = false
    @Volatile private var sessionCreated = false
    @Volatile private var openingCamera = false
    @Volatile private var abandonedOpen = false
    // 切镜头时若上一次 openCamera 仍在途（openingCamera=true），openCamera 会直接
    // return，新相机永远打不开（预览黑屏）。此标志让在途回调结束后自动补开。
    @Volatile private var pendingReopen = false
    private var nativeW = 0
    private var nativeH = 0
    private var nativeFx = 0f
    private var nativeFy = 0f
    private var nativeCx = 0f
    private var nativeCy = 0f
    private var supportedSizes = ""
    private var selectedSizeText = ""
    private var selectedPreviewSizeText = ""
    private var screenW = 0
    private var screenH = 0
    private var previewScale = 0f
    private var previewRot = 0
    // 诊断探针：SurfaceTexture 每帧的真实纹理变换矩阵（含 HAL 旋转标志/裁剪）
    @Volatile private var stMatrixText = "n/a"
    @Volatile private var viewMatrixText = "n/a"
    private var lastStMatrixLogNs = 0L
    private val stMatrixFloats = FloatArray(16)
    // 实测纹理变换的 2D 仿射部分 [f0 f4 f12 f1 f5 f13]，供自适应校正矩阵使用
    @Volatile private var stAffine = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
    @Volatile private var hasStAffine = false
    /**
     * camera-normalized UV -> view-normalized 的 2x3 仿射是否已就绪。
     * 它是点云 Renderer 与 Camera Preview 共用同一套屏幕坐标系的唯一凭据，
     * 报告里的 displayRotationApplied 现在反映的就是它。
     */
    @Volatile private var cameraToViewReady = false
    /**
     * camera-normalized UV -> view-normalized 的 2x3 仿射 (a,b,c, d,e,f)。
     *
     * 点云 Renderer 用它做世界点云投影；绿色目标框也**必须共用这一套**。
     * 旧实现画框时直接 `s.x0 * width`，等于假设「相机归一化坐标 == 竖屏
     * 视图归一化坐标」，而实际中间隔着 sensorOrientation 90° +
     * SurfaceTexture transform + TextureView center crop —— 于是 native
     * 明明跟住了目标，屏幕上的框却画在错误位置，看起来就像「追踪不准」。
     */
    @Volatile private var cameraToView = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
    /** Preview 当前帧的时间戳（SurfaceTexture.getTimestamp），AR 用它反查那一刻的 pose */
    @Volatile private var previewTimestampNs = 0L
    /** AR 图层切换按钮 */
    private var arLayerButton: android.widget.TextView? = null
    /** AR 绘制模式，见 PointCloudRenderer.DRAW_* */
    @Volatile private var arDrawMode = PointCloudRenderer.DRAW_LIVE

    // ---- V0.13.3 模型查看器 ----
    /** 查看模式是否激活（激活时 glView 接管触摸手势） */
    @Volatile private var modelViewerActive = false
    /** 查看器按钮（进入/退出共用，文字随状态切换） */
    private var modelViewerButton: android.widget.Button? = null
    /** 进入查看模式前的 AR 图层，退出时恢复 */
    private var modelViewerPrevMode = PointCloudRenderer.DRAW_LIVE
    /** 手势状态：按下的位置/时刻，及当前手势是「旋转」还是「长按平移」 */
    private var viewerLastX = 0f
    private var viewerLastY = 0f
    private var viewerDownX = 0f
    private var viewerDownY = 0f
    private var viewerPanMode = false
    private var viewerMoved = false
    private var viewerMultiTouch = false
    private var viewerLastMidX = 0f
    private var viewerLastMidY = 0f
    private val viewerScaleDetector by lazy {
        android.view.ScaleGestureDetector(this,
            object : android.view.ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScaleBegin(detector: android.view.ScaleGestureDetector): Boolean {
                    cancelViewerLongPress()
                    viewerPanMode = false
                    return modelViewerActive
                }
                override fun onScale(detector: android.view.ScaleGestureDetector): Boolean {
                    val factor = detector.scaleFactor
                    glView.queueEvent { renderer.zoomViewer(factor) }
                    glView.requestRender()
                    return true
                }
            }).apply { isQuickScaleEnabled = false }
    }

    /** 长按判定：按下后 300ms 内没动过就切到「平移」手势 */
    private val viewerLongPressMs = 300L
    private val viewerHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var viewerLongPressRunnable: Runnable? = null
    /** 旋转灵敏度：弧度/像素。一屏宽拖过去 ≈ 转 300°，接近「转一圈」的直觉。 */
    private val viewerRotRadPerPx = 0.005f
    /** V0.12: 实时网格快照是否正在后台重建（防止同一时刻排队两次）。 */
    @Volatile private var liveMeshBuildBusy = false
    /** V0.12: 上一次提交网格快照的时刻（SystemClock.elapsedRealtime）。 */
    @Volatile private var lastLiveMeshRefreshMs = 0L
    /** 已经因为「epoch 未激活」清过一次网格，避免逐帧重复清。 */
    private var liveMeshBlockedNotice = false

    // ---- V0.13.4 P1：自由摆放 AR ----
    /**
     * 是否处于「自由摆放」模式。false（默认）= **原位恢复**：模型顶点在保存
     * 时的世界坐标里，靠持久地图回到原物理位置。true = 模型被搬到当前世界
     * 坐标下的锚点，不依赖旧地图，换地点也能放。
     *
     * 这两件事必须分开：混在一起会出现「换了个房间，模型却死死钉在旧坐标、
     * 看起来像不存在」或者「想看原位却跟着手机跑」这两种互不兼容的抱怨。
     */
    @Volatile private var placeModeActive = false
    /** 摆放模式按钮（进入/退出共用，文字随状态切换） */
    private var placeButton: android.widget.TextView? = null
    /** 模型包围盒中心（模型坐标系），摆放时用它把模型「中心对齐锚点」。 */
    private val placeCenter = FloatArray(3)
    /** 锚点：当前世界坐标下模型中心要放到的位置。 */
    private val placeAnchor = FloatArray(3)
    /** 绕世界 Y 轴的偏航（弧度）与均匀缩放。 */
    @Volatile private var placeYaw = 0f
    @Volatile private var placeScale = 1f
    /** 摆放手势状态 */
    private var placeLastX = 0f
    private var placeLastY = 0f

    /**
     * V0.13.4 P0-C：扫描会话令牌。
     *
     * 后台任务（最终重建、GLB 导出、扫描包恢复、后台建网格）都是异步回跳的，
     * 而开新扫描会**重置 native TSDF 与 sessionId**。只靠 `modelOperationBusy`
     * 这类「忙标志」挡不住所有交错（忙标志只覆盖「提交」那一刻），所以每个
     * 异步任务都要在提交时捕获令牌、在**发布结果前**再校验一次；令牌变了就
     * 丢弃自己的结果 —— 旧结果一旦发布到新会话，用户看到的是「上一轮的模型
     * 叠在这一轮的空场景上」，比什么都不显示更难排查。
     */
    @Volatile private var scanSessionToken = 0L
    private val depthSessionLock = Any()
    /**
     * V0.12: 扫描期网格快照周期。
     *
     * 网格**刻意不追 30FPS**：逐帧跑一次 TSDF -> Marching Tetrahedra 会把
     * CPU 吃光，反而拖慢 VIO 让模型更差。5 秒一次已足够让用户看出
     * 「模型在长」——而点云是每帧刷新的，画面并不会显得僵。
     */
    private val liveMeshRefreshPeriodMs = 5_000L
    /**
     * V0.13.20：实时网格重建的自适应退避上限。
     *
     * 真机数据（vc156 实拍 3 分钟）：`nativeBuildMesh` 在 native 侧持全局锁完成
     * 整段构建，耗时随体素场覆盖增长从 86ms 涨到 1186ms，且每 5s 触发一次；
     * 持锁期内相机帧回调被阻塞（实测 `nativeOnCameraFrame` 峰值 1165.7ms，
     * 与构建耗时逐一对应），VINS 位姿历史出现空洞 → AR 卡顿 + HQ 纹理关键帧被丢。
     *
     * 因此按「上一次构建越慢、下一次间隔越长」退避，把固定 5s 节拍改成
     * `clamp(上次构建耗时 × 15, 5s, 30s)`：构建 1186ms 时退到约 17.8s，
     * 便宜的构建（<333ms）仍保持 5s。这是治标——根治需要把全局锁只覆盖
     * 「Marching Tetrahedra 提取」这一段，把清理/QEM 挪到锁外。
     */
    private val liveMeshRefreshPeriodMaxMs = 30_000L
    /** 上一次实时网格构建在 native 侧的实际耗时（ms），用于自适应退避。 */
    @Volatile private var liveMeshLastBuildMs = 0L
    /** Fusion Epoch 诊断槽缓冲（HUD / 报告用，走 UI 线程）。 */
    private val fusionEpochBuf = FloatArray(NativeBridge.FUSION_EPOCH_STATS_SLOTS)
    /**
     * Fusion Epoch 探测缓冲（**只**给相机回调线程用）。
     *
     * 和 [fusionEpochBuf] 分开不是洁癖：网格节拍在相机线程上跑，而 HUD / 报告
     * 在 UI 线程上跑，共用一个数组会让报告里出现半新半旧的混合读数。
     */
    private val epochProbeBuf = FloatArray(NativeBridge.FUSION_EPOCH_STATS_SLOTS)
    /**
     * V0.13.4：UI 线程专用的 epoch 探测缓冲。
     * 不复用 [epochProbeBuf] —— 那个是相机回调线程的，1Hz 的 UI 轮询
     * 同时写同一数组会读到另一线程写的槽位。native 侧已持 gStateMutex，
     * 所以多线程查询本身是安全的，争的只是这个 Kotlin 数组。
     */
    private val epochUiProbeBuf = FloatArray(NativeBridge.FUSION_EPOCH_STATS_SLOTS)
    // V0.12 扫描期 AR 材质：青绿色半透明网格。
    // 0.30–0.40 是「看得清形状、又不把真实画面糊掉」的折中。
    private val scanMeshAlpha = 0.34f
    /** 停扫 / 查看期网格不透明度：留一点透明才看得出它和真实场景贴不贴。 */
    private val viewMeshAlpha = 0.78f
    private val scanMeshTintR = 0.15f
    private val scanMeshTintG = 0.95f
    private val scanMeshTintB = 1.00f
    /**
     * V0.12 ⑧：选目标后**一次性**检查初始框离画面边缘的距离。
     * 值是一个 deadline（System.nanoTime），过期自动失效 —— 否则用户点完很久
     * 之后框才出现，会莫名其妙弹一句「请把目标放入画面中间」。
     */
    private var edgeMarginCheckUntilNs = 0L
    /** 判定「贴到边缘」的归一化门限（文档给的 3–5% 取中位 4%）。 */
    private val targetEdgeMarginWarn = 0.04f
    private var cameraIds: List<String> = emptyList()
    private var currentCameraId = ""
    private var cameraIndex = 0
    private var focalLengths = floatArrayOf()
    private var apertures = floatArrayOf()
    private var lensFacing = 0
    private var activePhysicalCameraId: String? = null
    private var rollingShutterSkewNs: Long? = null
    private var exposureTimeNs: Long? = null
    private var lastCropRegion: android.graphics.Rect? = null
    private var lastDistortionCorrectionMode: Int? = null
    private var lensDistortion: FloatArray? = null
    private var lastAfState: Int? = null
    private var lastLensFocusDistance: Float? = null
    private var targetFocusDistance: Float? = null
    private var targetFocusLocked = false
    private var focusRelockCount = 0
    private var targetTapX = 0f
    private var targetTapY = 0f
    private var targetRelockFrames = 0
    private var targetState = 0
    private var targetConfidence = 0f
    private var targetTrackedPoints = 0
    private var targetVisibleFraction = 1f
    private var targetCenterX = 0.5f
    private var targetCenterY = 0.5f
    // 「TRACKING -> 离开画面」边沿检测，用于只震一次
    private var lastTargetUiState = 0
    // 最近一次 PresenceGate 结论（协议槽 14）：目标是否真的还在画面里
    @Volatile private var targetPresenceValid = true
    private var vibrator: android.os.Vibrator? = null
    private val targetAfLockEnabled = false
    private var aeLockAvailable = false
    private var awbLockAvailable = false
    private var manualFocusAvailable = false
    private var minFocusDistance = 0f
    private var noiseModes = intArrayOf()
    private var edgeModes = intArrayOf()
    private var aeLock = false
    private var awbLock = false
    // HQ 状态机只做展示用镜像；preview 的 3A 永远保持自动，不再跟随它切换锁定。
    private var captureState = "IDLE"
    // 深度尺度：只统计 raw↔VINS 的比例失配，绝不自动施加修正（见 DepthScaleEstimator）
    private val depthScaleEstimator = DepthScaleEstimator()
    private val depthScaleTargetBuf = FloatArray(NativeBridge.TARGET_STATE_SLOTS)
    // 槽位与 native_engine.cpp 的 kDepthDiagSlots（= NativeBridge.DEPTH_DIAGNOSTIC_SLOTS）一致：
    //   0 P10  1 median  2 P90  3 vinsMedian
    //   4 validPixels  5 sampleCount  6 roiArea
    private val depthScaleDepthBuf = FloatArray(NativeBridge.DEPTH_DIAGNOSTIC_SLOTS)
    private var lastAeState: Int? = null
    private var lastAwbState: Int? = null
    private var fps = 0f
    private var fpsFrames = 0
    private var fpsLastNs = 0L
    private var hudExpanded = false
    private var hudShownPoints = 0f
    private var scanVoxelProfile = 0
    private var systemInitialized = false
    private var imuRegistered = false
    private var objectLockEnabled = false
    private lateinit var targetOverlay: TargetLockOverlay
    /** 目标接近边缘 / 已离开画面的**持续**提示（不是 Toast，Toast 一闪就没了） */
    private lateinit var targetWarningText: TextView
    // 拖框选择状态（view 像素）
    private var dragSelecting = false
    private var dragStartX = 0f
    private var dragStartY = 0f
    /**
     * 目标 Overlay 的刷新周期：33ms ≈ 30Hz。
     *
     * 旧实现把 updateTargetOverlay() 挂在 updateHeader() 里，而后者只在
     * `dt >= 1.0` 时调用 —— 于是 native tracker 每帧都在跑，肉眼看到的
     * 绿框却 1 秒才跳一次，直接产生「追踪卡、追不上物体」的观感。
     */
    private val targetUiPeriodNs = 33_000_000L
    private var targetUiLastNs = 0L
    @Volatile private var resumed = false
    private var sessionId = "unknown"
    private var sessionStartTs = 0L
    private var lastPlyFilename: String? = null
    private var lastPlyVertexCount: Int? = null
    private var lastPlyFileBytes: Long? = null
    private var lastPlyExportTs: Long? = null
    private var lastPlySessionId: String? = null
    private val buildGitSha = BuildConfig.GIT_COMMIT

    private data class TargetUiState(
        val visible: Boolean = false,
        val state: Int = 0,
        val x0: Float = 0f,
        val y0: Float = 0f,
        val x1: Float = 0f,
        val y1: Float = 0f,
        val confidence: Float = 0f,
        val medianDepth: Float = 0f,
        // 新增（协议 10->14）。visibleFraction < 0.40 画黄框；
        // centerX/centerY 未裁剪，目标完全出界时仍能指示找回方向。
        val visibleFraction: Float = 1f,
        val centerX: Float = 0.5f,
        val centerY: Float = 0.5f,
        /**
         * PresenceGate 结论（协议槽 14）。
         *
         * **box 还在画面里 != 物体还在**：真实目标离开后，tracker 常常在背景
         * 纹理上继续输出一个「看起来正常」的框（OpenCV tracking #619 /
         * NanoTrack issue 均有记录）。所以绿框的可见性必须由这一位决定，
         * 而不是 bbox 的几何位置。
         */
        val presenceValid: Boolean = true
    )

    private data class FrameMeta(
        val exposureNs: Long,
        val skewNs: Long,
        val crop: android.graphics.Rect?,
        val physicalId: String?
    )

    private val frameMetaLock = Any()
    private val frameMeta = LinkedHashMap<Long, FrameMeta>()
    private var frameMetaHit = 0L
    private var frameMetaMiss = 0L
    private var frameMetaLateHit = 0L
    private var frameMetaExpired = 0L
    private val pendingFrameMeta = LinkedHashSet<Long>()

    private val captureResultCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: android.hardware.camera2.TotalCaptureResult
        ) {
            if (session !== captureSession || cameraRangeSwitching) return
            activePhysicalCameraId = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                result.get(android.hardware.camera2.CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID) else null
            rollingShutterSkewNs =
                result.get(android.hardware.camera2.CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW)
            exposureTimeNs =
                result.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME)
            lastCropRegion =
                result.get(android.hardware.camera2.CaptureResult.SCALER_CROP_REGION)
            lastDistortionCorrectionMode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
                result.get(android.hardware.camera2.CaptureResult.DISTORTION_CORRECTION_MODE) else null
            lastAfState =
                result.get(android.hardware.camera2.CaptureResult.CONTROL_AF_STATE)
            lastLensFocusDistance =
                result.get(android.hardware.camera2.CaptureResult.LENS_FOCUS_DISTANCE)
            lastAeState =
                result.get(android.hardware.camera2.CaptureResult.CONTROL_AE_STATE)
            lastAwbState =
                result.get(android.hardware.camera2.CaptureResult.CONTROL_AWB_STATE)
            multiCam?.onCaptureResult(result)
            if (scanning && scanNativeReady && multiCam?.primaryProjectionMismatch == true) {
                scanNativeReady = false // no more frames may enter fusion with changed primary K
                cameraRangeStatus = "主摄裁剪变化 · 已保护模型并停止融合"
                runOnUiThread {
                    if (!resumed || multiCam?.primaryProjectionMismatch != true) return@runOnUiThread
                    if (scanning) stopScan()
                    closeCamera()
                    if (texture.isAvailable) openCamera()
                    toast("HAL 改变了主摄裁剪，已保留模型并停止扫描；请导出后重新开始")
                }
            }
            if (::hqCapture.isInitialized) {
                hqCapture.onCaptureResult(result)
            }

            val sensorTs = result.get(android.hardware.camera2.CaptureResult.SENSOR_TIMESTAMP)
            if (sensorTs != null) {
                val exposure = exposureTimeNs ?: 0L
                val skew = rollingShutterSkewNs ?: 0L
                val crop = lastCropRegion
                val physicalId = activePhysicalCameraId
                synchronized(frameMetaLock) {
                    // ImageReader may deliver the image before the capture result.
                    if (pendingFrameMeta.remove(sensorTs)) {
                        frameMetaLateHit++
                        frameMetaHit++
                    } else {
                        frameMeta[sensorTs] =
                            FrameMeta(exposure, skew, crop, physicalId)
                    }
                    while (frameMeta.size > 96) {
                        val key = frameMeta.entries.first().key
                        frameMeta.remove(key)
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        androidx.core.view.WindowCompat.setDecorFitsSystemWindows(window, false)
        val scanPrefs = getSharedPreferences("scan_settings", Context.MODE_PRIVATE)
        val savedDistance = scanPrefs.getFloat("max_distance_m", 1f)
        scanMaxDistanceMeters = if (savedDistance.isFinite()) savedDistance.coerceIn(.2f, 5f) else 1f
        autoFillLight = scanPrefs.getBoolean("auto_fill_light", true)
        scanVoxelProfile = scanPrefs.getInt("voxel_profile", 3).coerceIn(0, 3)
        guidedScanEnabled = scanPrefs.getBoolean("guided_scan_enabled", true)

        // —— 换皮：使用 UI 替换包提供的 XML 布局，保留全部扫描/融合/导出核心逻辑 ——
        setContentView(R.layout.activity_mobile_scan)
        val container = findViewById<android.widget.FrameLayout>(R.id.cameraPreviewContainer)
        val density = resources.displayMetrics.density
        fun dp(value: Int) = (value * density + .5f).toInt()

        texture = TextureView(this).apply {
            // Keep the camera buffer explicitly translucent so the sibling AR SurfaceView
            // cannot force a black fallback buffer on the first permission-return frame.
            setOpaque(false)
        }
        container.addView(texture, ViewGroup.LayoutParams(-1, -1))
        targetOverlay = TargetLockOverlay(this)
        targetOverlay.isClickable = false
        targetOverlay.isFocusable = false
        container.addView(targetOverlay, ViewGroup.LayoutParams(-1, -1))
        texture.addOnLayoutChangeListener { _, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
            if (right - left != oldRight - oldLeft || bottom - top != oldBottom - oldTop) {
                configureTransform(right - left, bottom - top)
            }
        }

        texture.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    downX = event.x
                    downY = event.y
                    if (objectLockEnabled) {
                        dragSelecting = true
                        dragStartX = event.x
                        dragStartY = event.y
                    }
                    true
                }
                android.view.MotionEvent.ACTION_MOVE -> {
                    if (dragSelecting) {
                        val dx = event.x - dragStartX
                        val dy = event.y - dragStartY
                        if (dx * dx + dy * dy >=
                            dragSelectMinPx() * dragSelectMinPx()) {
                            targetOverlay.dragRect = android.graphics.RectF(
                                minOf(dragStartX, event.x), minOf(dragStartY, event.y),
                                maxOf(dragStartX, event.x), maxOf(dragStartY, event.y)
                            )
                            targetOverlay.invalidate()
                        }
                    }
                    true
                }
                android.view.MotionEvent.ACTION_UP,
                android.view.MotionEvent.ACTION_CANCEL -> {
                    val dx = event.x - downX
                    val dy = event.y - downY
                    val wasDrag = dragSelecting
                    dragSelecting = false
                    targetOverlay.dragRect = null
                    if (event.actionMasked == android.view.MotionEvent.ACTION_UP &&
                        wasDrag &&
                        dx * dx + dy * dy >= dragSelectMinPx() * dragSelectMinPx()) {
                        selectTargetRect(downX, downY, event.x, event.y)
                    } else if (event.actionMasked == android.view.MotionEvent.ACTION_UP &&
                        dx * dx + dy * dy < dragSelectMinPx() * dragSelectMinPx()) {
                        if (objectLockEnabled) {
                            selectTarget(event.x, event.y)
                        } else {
                            focusAt(event.x, event.y)
                        }
                    }
                    targetOverlay.invalidate()
                    true
                }
                else -> false
            }
        }

        glView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            holder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)
            setEGLConfigChooser(8, 8, 8, 8, 16, 0)
            renderer = PointCloudRenderer()
            setRenderer(renderer)
            renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
            // Do not place the separate GL Surface above TextureView until Camera2 has
            // delivered at least one real SurfaceTexture frame. On some OnePlus builds,
            // the permission Activity return recreates the GL surface first; its initial
            // transparent buffer is composed as opaque black and hides the camera.
            visibility = android.view.View.INVISIBLE
            // Keep the transparent GL surface above the camera preview, but below normal
            // Android controls so the 3D viewer top bar and "退出" action stay visible.
            // SurfaceView is opaque by default on some Android/HAL combinations: it then
            // paints a black rectangle over TextureView even though the GL clear alpha is 0.
            // Explicitly opt into alpha compositing for the AR overlay before the first frame.
            // On OnePlus 15 the media-overlay path can keep an opaque black SurfaceView
            // buffer after the runtime permission activity returns, even while TextureView
            // reports valid camera frames. A top translucent SurfaceView is composited
            // correctly on this HAL and preserves the transparent AR pass.
            setZOrderOnTop(true)
            holder.setFormat(android.graphics.PixelFormat.TRANSLUCENT)
            isClickable = false
            isFocusable = false
        }
        glView.setOnTouchListener { _, event ->
            if (modelViewerActive) {
                handleModelViewerTouch(event)
                return@setOnTouchListener true
            }
            if (placeModeActive) {
                handlePlacementTouch(event)
                return@setOnTouchListener true
            }
            false
        }
        container.addView(glView, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        // 目标离屏提示（原 hudText 区域保留：常驻文案而非 Toast）
        targetWarningText = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 12f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setBackgroundColor(android.graphics.Color.argb(225, 210, 40, 40))
            setPadding((16 * density).toInt(), (10 * density).toInt(),
                       (16 * density).toInt(), (10 * density).toInt())
            gravity = Gravity.CENTER
            visibility = android.view.View.GONE
        }
        findViewById<android.widget.LinearLayout>(R.id.scanNotices).addView(
            targetWarningText, 0,
            android.widget.LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(8) }
        )

        // Focus UI: visible scan essentials plus on-demand telemetry.
        // 包设计把状态拆为结构化单项：tvFps/tvDistance/tvAutoLight/tvCameraMode +
        // tvDepth/tvMapCount/tvDrawCount/tvUnit；中央提示位 tvScanHint 承载动态提示。
        // 旧复合文本字段（headerStatus/hudText/statusBarText）影子化：逻辑保留、
        // 不入视图树，避免原有多处 setText 写点回归（updateHeader 末尾做数据拆解）。
        primaryButton = findViewById(R.id.btnStartScan)
        modelViewerButton = findViewById(R.id.btnViewModel)
        placeButton = findViewById(R.id.toolPlacementLabel)
        arLayerButton = findViewById(R.id.toolRealtimeLabel)
        headerStatus = android.widget.TextView(this)
        hudText = android.widget.TextView(this)
        statusBarText = android.widget.TextView(this)
        warningBanner = findViewById(R.id.tvScanHint)
        tvFps = TextView(this) // On-demand details; no permanent viewport overlay.
        tvDistance = findViewById(R.id.tvDistance)
        tvAutoLight = TextView(this) // On-demand details; no permanent viewport overlay.
        tvCameraMode = TextView(this) // On-demand details; no permanent viewport overlay.
        tvDepth = findViewById(R.id.tvDepth)
        tvMapCount = TextView(this) // On-demand details; no permanent viewport overlay.
        tvDrawCount = TextView(this) // On-demand details; no permanent viewport overlay.
        tvUnit = findViewById(R.id.tvUnit)
        hudCompact = TextView(this) // On-demand details; no permanent viewport overlay.
        pauseButton = findViewById(R.id.btnPauseScan)
        scanReticleView = findViewById(R.id.scanReticle)
        scanGuidanceOverlay = findViewById(R.id.scanGuidanceOverlay)
        tvGuidanceStep = findViewById(R.id.tvGuidanceStep)
        scanProgressView = findViewById(R.id.scanProgress)
        tvScanProgressPercent = findViewById(R.id.tvScanProgressPercent)
        tvMotionState = TextView(this) // On-demand details; no permanent viewport overlay.
        tvDistanceState = TextView(this) // On-demand details; no permanent viewport overlay.
        tvTrackingState = TextView(this) // On-demand details; no permanent viewport overlay.
        tvViewpointSummary = findViewById(R.id.tvViewpointSummary)
        tvScanStateBadge = findViewById(R.id.tvScanStateBadge)
        mainScanUi = findViewById(R.id.mainScanUi)
        viewpointCoverageView = com.mobilescan3d.ui.ViewpointCoverageView(this)
        scanReviewOverlay = findViewById(R.id.scanReviewOverlay)
        reviewContinueButton = findViewById(R.id.btnReviewContinue)
        reviewGenerateButton = findViewById(R.id.btnReviewGenerate)
        reviewDiscardButton = findViewById(R.id.btnReviewDiscard)
        tvReviewTopHint = findViewById(R.id.tvReviewTopHint)
        tvReviewScore = findViewById(R.id.tvReviewScore)
        tvReviewRecommendation = findViewById(R.id.tvReviewRecommendation)
        tvReviewMissing = findViewById(R.id.tvReviewMissing)
        reviewProgressCapture = findViewById(R.id.reviewProgressCapture)
        reviewProgressViewpoint = findViewById(R.id.reviewProgressViewpoint)
        reviewProgressGeometry = findViewById(R.id.reviewProgressGeometry)
        reviewProgressTexture = findViewById(R.id.reviewProgressTexture)
        tvReviewCapture = findViewById(R.id.tvReviewCapture)
        tvReviewViewpoint = findViewById(R.id.tvReviewViewpoint)
        tvReviewGeometry = findViewById(R.id.tvReviewGeometry)
        tvReviewTexture = findViewById(R.id.tvReviewTexture)
        viewerBackdrop = findViewById(R.id.viewerBackdrop)
        modelViewerOverlay = findViewById(R.id.modelViewerOverlay)
        tvViewerStats = findViewById(R.id.tvViewerStats)
        toolAutoLabel = TextView(this) // On-demand details; no permanent viewport overlay.
        toolModeLabel = TextView(this) // On-demand details; no permanent viewport overlay.
        toolLightLabel = findViewById(R.id.toolLightLabel)
        toolCameraLabel = TextView(this) // On-demand details; no permanent viewport overlay.
        toolLockLabel = findViewById(R.id.toolLockLabel)
        viewerModeTexture = findViewById(R.id.viewerModeTexture)
        viewerModeSolid = findViewById(R.id.viewerModeSolid)
        viewerModeWire = findViewById(R.id.viewerModeWire)
        viewerModeInspect = findViewById(R.id.viewerModeInspect)
        viewerModeSurface = findViewById(R.id.viewerModeSurface)
        reviewProgressSurface = findViewById(R.id.reviewProgressSurface)
        tvReviewSurface = findViewById(R.id.tvReviewSurface)
        tvViewerDimensions = findViewById(R.id.tvViewerDimensions)
        tvViewerTopology = findViewById(R.id.tvViewerTopology)
        viewerOptimizeButton = findViewById(R.id.btnViewerOptimize)
        viewerOptimizeLabel = findViewById(R.id.tvViewerOptimizeLabel)
        findViewById<android.view.View>(R.id.btnStartScan).setOnClickListener { handlePrimaryScanAction() }
        findViewById<android.view.View>(R.id.btnPauseScan).setOnClickListener { togglePauseScan() }
        reviewContinueButton.setOnClickListener { resumeFromScanReview() }
        reviewGenerateButton.setOnClickListener { finalizeFromScanReview() }
        reviewDiscardButton.setOnClickListener { confirmDiscardScan() }
        findViewById<android.view.View>(R.id.btnViewModel).setOnClickListener { toggleModelViewer() }
        findViewById<android.view.View>(R.id.btnSettings).setOnClickListener { showSettingsMenu() }
        findViewById<android.view.View>(R.id.statusCard).setOnClickListener { showScanDetails() }
        findViewById<android.view.View>(R.id.toolLock).setOnClickListener { toggleObjectTracking() }
        findViewById<android.view.View>(R.id.toolLight).setOnClickListener { toggleAutoFillLightFromToolbar() }
        findViewById<android.view.View>(R.id.btnViewerClose).setOnClickListener { exitModelViewer() }
        findViewById<android.view.View>(R.id.btnViewerReset).setOnClickListener {
            glView.queueEvent { renderer.resetViewerView() }
            glView.requestRender()
        }
        findViewById<android.view.View>(R.id.btnViewerExport).setOnClickListener { showExportDrawer() }
        findViewById<android.view.View>(R.id.btnViewerReport).setOnClickListener { showScanQualityReport() }
        viewerOptimizeButton.setOnClickListener { toggleModelOptimization() }
        findViewById<android.view.View>(R.id.btnViewerSegment).setOnClickListener { showSegmentMergeDialog() }
        findViewById<android.view.View>(R.id.btnViewerHelp).setOnClickListener { showViewerHelp() }
        viewerModeTexture.setOnClickListener { setViewerPresentation(PointCloudRenderer.VIEWER_STYLE_TEXTURE) }
        viewerModeSolid.setOnClickListener { setViewerPresentation(PointCloudRenderer.VIEWER_STYLE_SOLID) }
        viewerModeWire.setOnClickListener { setViewerPresentation(PointCloudRenderer.VIEWER_STYLE_WIREFRAME) }
        viewerModeInspect.setOnClickListener { setViewerPresentation(PointCloudRenderer.VIEWER_STYLE_INSPECT) }
        viewerModeSurface.setOnClickListener { setViewerPresentation(PointCloudRenderer.VIEWER_STYLE_SURFACE) }
        findViewById<android.view.View>(R.id.toolPlacement).setOnClickListener { togglePlacement() }
        findViewById<android.view.View>(R.id.toolRealtime).setOnClickListener { cycleArLayer() }
        findViewById<android.view.View>(R.id.toolRestoreAr).setOnClickListener { restoreLatestPersistentAr() }

        // Round 4 / OnePlus 15:
        // Keep the camera/GL full-bleed, but inset every touch/control surface
        // using the *real* system bar + display cutout geometry. The official
        // spec gives screen size/resolution, not a guaranteed punch-hole radius,
        // so no physical cutout diameter is hard-coded here.
        val rootView = findViewById<android.view.View>(R.id.root)
        window.statusBarColor = android.graphics.Color.TRANSPARENT
        window.navigationBarColor = android.graphics.Color.TRANSPARENT
        androidx.core.view.WindowInsetsControllerCompat(window, rootView).apply {
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = false
        }
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(rootView) { _, insets ->
            val safe = insets.getInsets(
                androidx.core.view.WindowInsetsCompat.Type.systemBars() or
                    androidx.core.view.WindowInsetsCompat.Type.displayCutout()
            )
            mainScanUi.setPadding(
                safe.left + dp(12),
                safe.top + dp(6),
                safe.right + dp(12),
                safe.bottom + dp(5)
            )
            scanReviewOverlay.setPadding(
                safe.left, safe.top, safe.right, safe.bottom
            )
            modelViewerOverlay.setPadding(
                safe.left, safe.top, safe.right, safe.bottom
            )
            insets
        }
        androidx.core.view.ViewCompat.requestApplyInsets(rootView)

        // vc159 ThermalGuard：订阅系统热状态，用于在设备发烫时自动降载。
        // OnThermalStatusChangedListener / currentThermalStatus 需要 API 29，
        // 低版本静默跳过（thermalStatus 维持 NONE，等价于「不做任何降载」）。
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                val pm = getSystemService(Context.POWER_SERVICE) as? PowerManager
                if (pm != null) {
                    thermalStatus = pm.currentThermalStatus
                    android.util.Log.i(
                        "ThermalGuard",
                        "initial thermal status = $thermalStatus " +
                            "(depthMin=${thermalDepthMinIntervalMs()}ms meshScale=${thermalMeshIntervalScale()})"
                    )
                    val listener = PowerManager.OnThermalStatusChangedListener { st ->
                        thermalStatus = st
                        android.util.Log.i(
                            "ThermalGuard",
                            "thermal status -> $st (depthMin=${thermalDepthMinIntervalMs()}ms)"
                        )
                    }
                    thermalListener = listener
                    pm.addThermalStatusListener(mainExecutor, listener)
                }
            } catch (t: Throwable) {
                android.util.Log.w("ThermalGuard", "thermal listener unavailable", t)
            }
        }

        // V0.12: 默认进「实时」图层 —— 半透明网格 + 累计 surfel(hits>=1)
        // + 当前帧 target depth 点。native 的逐帧取点开关必须跟着打开，
        // 否则 LIVE 就少了「当前帧点」这一层，用户看不出新数据进没进来。
        try {
            NativeBridge.nativeSetTargetDebugEnabled(
                arDrawMode == PointCloudRenderer.DRAW_TARGET_DEBUG ||
                    arDrawMode == PointCloudRenderer.DRAW_BOTH
            )
        } catch (_: Throwable) {
        }
        renderer.drawMode = arDrawMode
        renderer.setMeshAlpha(scanMeshAlpha)
        // 点大小按屏幕密度缩放：固定 3px 在高 DPI 屏上细得几乎看不见。
        //
        // V0.13：绿色「当前帧 target depth」调试点从 6f 降到 3f（560dpi 下
        // 21px -> 约 10px）。原因不是审美 —— debug 层的取点数上限是 2000，
        // 在 228x357 的 ROI 上 step 会走到 7，也就是**相邻点只隔 7px 而点径
        // 有 21px**，整片区域会被画成一坨连成片的绿色马赛克，看上去像「模型
        // 在这里长歪了」。点径略小于点间距，形状才能被读出来。
        // 累计层保持 4f：两层不仅要颜色不同，尺寸上也要能一眼分开。
        renderer.setPointSizes(4f * density, 3f * density)

        // 目标离开画面时震一下。取不到（无马达 / 无权限）就静默降级，
        // 绝不因为震动失败影响追踪。
        vibrator = try {
            getSystemService(Context.VIBRATOR_SERVICE) as? android.os.Vibrator
        } catch (_: Throwable) {
            null
        }

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestCameraAccess()
        } else {
            startSystem()
        }
    }

    private fun requestCameraAccess() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            if (!systemInitialized) startSystem()
            return
        }

        val permissionPrefs = getSharedPreferences("privacy_settings", Context.MODE_PRIVATE)
        val requestedBefore = permissionPrefs.getBoolean("camera_permission_requested", false)
        val systemWantsRationale = shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)

        // After a permanent denial Android may stop showing the system prompt. Send the user
        // directly to app settings instead of creating a dead-end "Authorize" loop.
        if (requestedBefore && !systemWantsRationale) {
            android.app.AlertDialog.Builder(this)
                .setTitle("相机权限已关闭")
                .setMessage("请在系统设置中允许相机权限，然后返回 MobileScan3D 继续扫描。")
                .setPositiveButton("打开系统设置") { _, _ ->
                    startActivity(
                        android.content.Intent(
                            android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                            android.net.Uri.parse("package:$packageName")
                        )
                    )
                }
                .setNegativeButton("取消", null)
                .show()
            return
        }

        val shouldExplain =
            !permissionPrefs.getBoolean("camera_intro_seen", false) || systemWantsRationale

        fun launchSystemPermission() {
            permissionPrefs.edit()
                .putBoolean("camera_intro_seen", true)
                .putBoolean("camera_permission_requested", true)
                .apply()
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
        }

        if (!shouldExplain) {
            launchSystemPermission()
            return
        }

        android.app.AlertDialog.Builder(this)
            .setTitle(getString(R.string.camera_permission_title))
            .setMessage(getString(R.string.camera_permission_message))
            .setPositiveButton(getString(R.string.camera_permission_continue)) { _, _ ->
                launchSystemPermission()
            }
            .setNegativeButton(getString(R.string.camera_permission_not_now), null)
            .show()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                primaryButton.text = "开始扫描"
                warningBanner.visibility = android.view.View.GONE
                // The permission dialog pauses/resumes the activity on several Android builds.
                // Reassert alpha compositing after the callback so a newly-created TextureView
                // is not covered by a stale opaque GL buffer.
                // Keep the GL Surface hidden until the first Camera2 frame. Toggling
                // Z-order here races the permission Activity return on OnePlus devices.
                previewFrameSeen = false
                glView.visibility = android.view.View.INVISIBLE
                texture.invalidate()
                startSystem()
            } else {
                primaryButton.text = "启用相机"
                headerStatus.text = "需要相机权限；点击启用相机后可扫描"
                warningBanner.text = "需要相机权限；点击启用相机后可扫描"
                warningBanner.visibility = android.view.View.VISIBLE
            }
        }
    }

    private fun ensureSystemReady(): Boolean {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            android.app.AlertDialog.Builder(this).setTitle("需要相机权限")
                .setMessage("扫描需要相机画面。可重新授权，或在系统设置中开启相机权限。")
                .setPositiveButton("授权") { _, _ ->
                    requestCameraAccess()
                }
                .setNeutralButton("系统设置") { _, _ ->
                    startActivity(android.content.Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                        android.net.Uri.parse("package:$packageName")))
                }
                .setNegativeButton("取消", null).show()
            return false
        }
        if (!systemInitialized) startSystem()
        return systemInitialized
    }

    private fun startSystem() {
        if (systemInitialized || isDestroyed) return
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val allCameraIds = cameraManager.cameraIdList.toList()
        val rearCameraIds = allCameraIds.filter { cameraId ->
            cameraManager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }
        cameraIds = rearCameraIds.ifEmpty { allCameraIds }
        if (cameraIds.isEmpty()) {
            headerStatus.text = "没有可用摄像头"
            warningBanner.text = "没有可用摄像头"
            warningBanner.visibility = android.view.View.VISIBLE
            toast("没有可用摄像头，无法扫描")
            return
        }
        currentCameraId = cameraIds.firstOrNull { cameraId ->
            val ch = cameraManager.getCameraCharacteristics(cameraId)
            ch.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK &&
                ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                    ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA) == true
        } ?: cameraIds.first()
        cameraIndex = cameraIds.indexOf(currentCameraId).coerceAtLeast(0)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        cameraThread = HandlerThread("CameraCapture").also { it.start() }
        sensorThread = HandlerThread("IMU").also { it.start() }
        cameraHandler = Handler(cameraThread!!.looper)
        // 深度来源：硬件优先、模型兜底；本版**默认走模型**（实机验收过的
        // 路径），硬件 DEPTH16 只做能力探测并进报告。
        depthProbe = DepthProviderFactory.probe(this)
        android.util.Log.i("DepthProvider", "probe: ${depthProbe.note}")
        depthProvider = DepthProviderFactory.create(
            this, assets, preferHardware = false, probe = depthProbe
        )
        exportManager = ExportManager(this)
        segmentStore = com.mobilescan3d.multiscan.ScanSegmentStore(this)
        depthThread = HandlerThread("DepthInference").also { it.start() }
        depthHandler = Handler(depthThread!!.looper)
        systemInitialized = true
        if (resumed) registerImu()

        texture.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: android.graphics.SurfaceTexture, w: Int, h: Int) {
                viewWidth = w
                viewHeight = h
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(st: android.graphics.SurfaceTexture, w: Int, h: Int) {
                viewWidth = w
                viewHeight = h
                configureTransform(w, h)
            }

            override fun onSurfaceTextureDestroyed(st: android.graphics.SurfaceTexture): Boolean {
                previewFrameSeen = false
                if (::glView.isInitialized) glView.visibility = android.view.View.INVISIBLE
                closeCamera()
                return true
            }

            override fun onSurfaceTextureUpdated(st: android.graphics.SurfaceTexture) {
                // 采样 SurfaceTexture 的真实纹理变换矩阵（HAL 旋转标志/裁剪的地面真值）
                try {
                    st.getTransformMatrix(stMatrixFloats)
                    // Preview 帧自己的 SENSOR_TIMESTAMP。AR 点云必须用它反查
                    // 「屏幕上这一帧画面拍摄时相机在哪」，而不是用此刻最新的 pose
                    // —— 两者不同时刻，手机一转点云就会漂/甩。
                    previewTimestampNs = st.timestamp
                    renderer.setPreviewTimestamp(previewTimestampNs)
                    // V0.12: 相机每来一帧就请求一次 AR 重绘。
                    //
                    // 之前 overlay 只在 depth result 到达时才 requestRender ——
                    // 相机约 30fps，而 mask/depth 只有约 3fps，于是模型看起来
                    // 「滞后、跳动、追着画面跑」。渲染模式仍是 RENDERMODE_WHEN_DIRTY，
                    // 只是把驱动信号从「深度帧」换成「相机帧」。
                    // requestRender() 本身线程安全，这里是相机线程。
                    if (!previewFrameSeen) {
                        previewFrameSeen = true
                        android.util.Log.i("CameraPreview", "first frame received; GL overlay enabled")
                    }
                    // 实时 3D 叠加层（点云/AR）由 glView 承载，首帧前隐藏以规避 OnePlus
                    // 权限返回时的黑屏。但一次性门控 + post 内 resumed/isAvailable 条件在
                    // 「首次授权」时序下可能失败，使 glView 永久停留 INVISIBLE、实时点云
                    // 永远不显示。改为每帧纠正：只要相机在出帧且 glView 尚不可见就显示它。
                    if (::glView.isInitialized && glView.visibility != android.view.View.VISIBLE) {
                        texture.post {
                            if (!isDestroyed && texture.isAvailable) {
                                glView.visibility = android.view.View.VISIBLE
                                glView.bringToFront()
                                glView.requestRender()
                            }
                        }
                    } else if (::glView.isInitialized && glView.visibility == android.view.View.VISIBLE) {
                        glView.requestRender()
                    }
                    val f = stMatrixFloats
                    // TextureView uses top-left UV; SurfaceTexture expects bottom-left GL UV.
                    // Both touch selection and AR projection consume this corrected affine.
                    val na = ScanCoordinates.surfaceTextureToCamera(f)
                    if (!hasStAffine || !na.contentEquals(stAffine)) {
                        val first = !hasStAffine
                        stAffine = na
                        hasStAffine = true
                        // A 变化（首帧/会话重建）后立即用实测值重算校正矩阵
                        texture.post { configureTransform(texture.width, texture.height) }
                        if (first) {
                            android.util.Log.d("CameraPreview", "ST affine captured")
                        }
                    }
                } catch (_: Throwable) {}
                // 每秒落一次诊断日志
                val now = System.nanoTime()
                if (now - lastStMatrixLogNs < 1_000_000_000L) return
                lastStMatrixLogNs = now
                try {
                    val a = stAffine
                    stMatrixText = String.format(
                        java.util.Locale.US, "[%.3f %.3f %.3f; %.3f %.3f %.3f]",
                        a[0], a[1], a[2], a[3], a[4], a[5]
                    )
                    val tv = android.graphics.Matrix()
                    texture.getTransform(tv)
                    val vals = FloatArray(9)
                    tv.getValues(vals)
                    viewMatrixText = String.format(
                        java.util.Locale.US, "[%.3f %.3f %.3f; %.3f %.3f %.3f]",
                        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
                    )
                    android.util.Log.d(
                        "CameraPreview",
                        "ST=$stMatrixText view=${texture.width}x${texture.height} viewM=$viewMatrixText"
                    )
                } catch (_: Throwable) {}
            }
        }
        if (texture.isAvailable) {
            viewWidth = texture.width
            viewHeight = texture.height
            openCamera()
        }
    }

    private fun registerImu() {
        if (imuRegistered || !systemInitialized) return
        imuRegistered = true
        hasAcc = false
        hasGyr = false
        lastImuOutNs = 0L
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val sensorHandler = Handler(sensorThread!!.looper)
        val periodUs = 5_000
        gyro?.let { sensorManager.registerListener(this, it, periodUs, 0, sensorHandler) }
        acc?.let { sensorManager.registerListener(this, it, periodUs, 0, sensorHandler) }
    }

    private fun openCamera() {
        if (!resumed || isDestroyed || !systemInitialized) return
        if (cameraDevice != null) return
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        if (openingCamera) {
            // 有在途的打开请求（可能是别的镜头），标记待重开，由该请求的回调结束后补开
            pendingReopen = true
            return
        }
        openingCamera = true
        try {
            val id = currentCameraId
            abandonedOpen = false
            val chars = cameraManager.getCameraCharacteristics(id)
            torchAvailable = chars.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            torchFailed = false
            sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            deviceModel = "${Build.MANUFACTURER} ${Build.MODEL}"
            lensFacing = chars.get(CameraCharacteristics.LENS_FACING) ?: 0
            focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS) ?: floatArrayOf()
            apertures = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES) ?: floatArrayOf()
            activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                ?: Rect(0, 0, 1920, 1440)
            oisSupported = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
                ?.contains(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON) == true
            afModes = chars.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
            videoStabModes = chars.get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)
                ?: intArrayOf()
            aeLockAvailable = chars.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE) ?: false
            awbLockAvailable = chars.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE) ?: false
            minFocusDistance = chars.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
            manualFocusAvailable = minFocusDistance > 0f
            lensDistortion = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
                chars.get(CameraCharacteristics.LENS_DISTORTION) else null
            noiseModes = chars.get(CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES) ?: intArrayOf()
            edgeModes = chars.get(CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES) ?: intArrayOf()
            val streamConfig = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            val outputSizes = streamConfig.getOutputSizes(ImageFormat.YUV_420_888)
            val previewSizes = streamConfig.getOutputSizes(android.graphics.SurfaceTexture::class.java)
            val dm = resources.displayMetrics
            val size = chooseOptimalSize(outputSizes)
            val pSize = choosePreviewSize(previewSizes, size)
            supportedSizes = outputSizes.joinToString(", ") { "${it.width}x${it.height}" }
            screenW = dm.widthPixels
            screenH = dm.heightPixels
            selectedSizeText = "${size.width}x${size.height}"
            selectedPreviewSizeText = "${pSize.width}x${pSize.height}"
            captureSize = size
            previewSize = pSize
            val focal = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
            val arrW = activeArray.width().toFloat()
            val arrH = activeArray.height().toFloat()
            // 相机驱动通常按输出宽高比做中心裁剪，再统一缩放。像素是方的，fx/fy 必须用同一缩放系数。
            val scale = size.width.toFloat() / arrW
            val cropH = arrW * size.height.toFloat() / size.width.toFloat()
            val cropTop = (arrH - cropH) / 2f
            val fx0 = focal?.getOrNull(0) ?: (arrW * 0.9f)
            val fy0 = focal?.getOrNull(1) ?: (arrW * 0.9f)
            val cx0 = focal?.getOrNull(2) ?: (arrW * 0.5f)
            val cy0 = focal?.getOrNull(3) ?: (arrH * 0.5f)
            nativeW = size.width
            nativeH = size.height
            nativeFx = fx0 * scale
            nativeFy = fy0 * scale
            nativeCx = cx0 * scale
            nativeCy = (cy0 - cropTop) * scale
            renderer.setCameraModel(nativeFx, nativeFy, nativeCx, nativeCy, nativeW, nativeH)
            // 重开相机（onResume）不应重置重建状态——nativeCreate 会清空点云/TSDF，
            // 息屏回来一次就把已积累的扫描全部丢掉。只在首次建会话时创建。
            if (!sessionCreated) {
                applyDeviceCalibrationProfileToNative()
                check(NativeBridge.nativeCreate(nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy)) {
                    "原生重建引擎初始化失败"
                }
                sessionCreated = true
            }

            // V0.8: first try a real physical-camera pair. If the HAL does
            // not expose/accept it, preserve the original logical YUV path.
            multiCam?.close()
            multiCam = MultiCameraFusionController(
                this,
                cameraManager,
                id,
                chars,
                cameraHandler!!
            ) { msg ->
                android.util.Log.i("MultiCamV08", msg)
                toast(msg)
            }
            val multiReady = multiCam?.configure(size) { image ->
                processImage(image)
            } == true
            reader = if (multiReady) {
                multiCam!!.primaryReader
            } else {
                ImageReader.newInstance(
                    size.width,
                    size.height,
                    ImageFormat.YUV_420_888,
                    4
                ).apply {
                    setOnImageAvailableListener(
                        { r -> r.acquireLatestImage()?.let { processImage(it) } },
                        cameraHandler
                    )
                }
            }
            if (::hqCapture.isInitialized) {
                hqCapture.close()
            }
            hqCapture = HqCaptureController(this, chars, cameraHandler!!) { msg ->
                toast(msg)
            }

            cameraManager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    openingCamera = false
                    if (abandonedOpen || id != currentCameraId || !resumed) {
                        camera.close()
                        reopenIfPending()
                        return
                    }
                    pendingReopen = false
                    try {
                        cameraDevice = camera
                        val st = texture.surfaceTexture
                        if (st == null) {
                            camera.close()
                            cameraDevice = null
                            return
                        }
                        val pSize = previewSize ?: size
                        // 必须先 setDefaultBufferSize 再创建 Surface：
                        // Surface 创建时会锁定 SurfaceTexture 当前的默认缓冲尺寸，
                        // 先建 Surface 会让 HAL 按旧的（竖屏）缓冲协商交付内容，
                        // 与 configureTransform 假设的横屏缓冲错位（画面错乱/半屏黑）。
                        st.setDefaultBufferSize(pSize.width, pSize.height)
                        // 重开相机前释放旧 Surface，避免泄漏与旧缓冲尺寸干扰
                        try { previewSurface?.release() } catch (_: Exception) {}
                        previewSurface = Surface(st)
                        texture.post { configureTransform(texture.width, texture.height) }
                        val baseSurfaces = listOf(previewSurface!!, reader!!.surface)
                        val hqSurfaces = hqCapture.prepareSurfaces(pSize, size)
                        val primarySurfaces = baseSurfaces + hqSurfaces

                        fun finishSession(
                            session: CameraCaptureSession,
                            hqActive: Boolean
                        ) {
                            if (abandonedOpen || cameraDevice !== camera || !resumed) {
                                session.close()
                                return
                            }
                            hqCapture.onSessionConfigured(hqActive)
                            captureSession = session
                            rangeHqActive = hqActive
                            rangeLogicalSurfaces = listOf(previewSurface!!) + if (hqActive) hqSurfaces else emptyList()
                            multiCam?.resumePairs()
                            cameraRangePolicy.reset()
                            cameraRangePolicy.committed(CameraRangePolicy.Range.MID, android.os.SystemClock.elapsedRealtime())
                            // Some HALs reset SurfaceTexture transform around first frame.
                            texture.post {
                                configureTransform(texture.width, texture.height)
                            }
                            texture.postDelayed(
                                { configureTransform(texture.width, texture.height) },
                                300L
                            )
                            applyCaptureSettings()
                            started.set(true)
                        }

                        fun startLegacySessionWithHq() {
                            if (abandonedOpen || cameraDevice !== camera || !resumed) return
                            camera.createCaptureSession(
                                primarySurfaces,
                                object : CameraCaptureSession.StateCallback() {
                                    override fun onConfigured(
                                        session: CameraCaptureSession
                                    ) {
                                        finishSession(session, true)
                                    }

                                    override fun onConfigureFailed(
                                        session: CameraCaptureSession
                                    ) {
                                        runCatching { session.close() }
                                        if (abandonedOpen || cameraDevice !== camera || !resumed) return
                                        hqCapture.detachSurfaces()
                                        camera.createCaptureSession(
                                            baseSurfaces,
                                            object : CameraCaptureSession.StateCallback() {
                                                override fun onConfigured(
                                                    fallback: CameraCaptureSession
                                                ) {
                                                    finishSession(fallback, false)
                                                }

                                                override fun onConfigureFailed(
                                                    fallback: CameraCaptureSession
                                                ) {
                                                    toast("Camera session failed")
                                                }
                                            },
                                            cameraHandler
                                        )
                                    }
                                },
                                cameraHandler
                            )
                        }

                        val mc = multiCam
                        if (mc != null && mc.active) {
                            fun startDualWithoutHq() {
                                if (abandonedOpen || cameraDevice !== camera || !resumed) return
                                val startedNoHq = mc.createPhysicalSession(
                                    camera,
                                    listOf(previewSurface!!),
                                    object : CameraCaptureSession.StateCallback() {
                                        override fun onConfigured(
                                            session: CameraCaptureSession
                                        ) {
                                            mc.noteSessionMode("dual-physical-noHQ")
                                            finishSession(session, false)
                                        }

                                        override fun onConfigureFailed(
                                            session: CameraCaptureSession
                                        ) {
                                            runCatching { session.close() }
                                            if (abandonedOpen || cameraDevice !== camera || !resumed) return
                                            mc.fallbackToLogical(
                                                "HAL rejected dual physical without HQ"
                                            )
                                            startLegacySessionWithHq()
                                        }
                                    }
                                )
                                if (!startedNoHq) {
                                    mc.fallbackToLogical(
                                        "dual physical no-HQ createCaptureSession failed"
                                    )
                                    startLegacySessionWithHq()
                                }
                            }

                            // Tier 1: preserve V0.9 HQ path when HAL accepts all streams.
                            // Tier 2: if stream-count/bandwidth is too high, keep BOTH
                            // physical analysis streams and sacrifice HQ still surfaces.
                            val logicalSurfaces = listOf(previewSurface!!) + hqSurfaces
                            val dualStarted = mc.createPhysicalSession(
                                camera,
                                logicalSurfaces,
                                object : CameraCaptureSession.StateCallback() {
                                    override fun onConfigured(
                                        session: CameraCaptureSession
                                    ) {
                                        mc.noteSessionMode("dual-physical+HQ")
                                        finishSession(session, true)
                                    }

                                    override fun onConfigureFailed(
                                        session: CameraCaptureSession
                                    ) {
                                        runCatching { session.close() }
                                        startDualWithoutHq()
                                    }
                                }
                            )

                            if (!dualStarted) {
                                startDualWithoutHq()
                            }
                        } else {
                            startLegacySessionWithHq()
                        }
                    } catch (e: Exception) {
                        camera.close()
                        cameraDevice = null
                        toast("打开相机失败：${e.message}")
                    }
                }

                override fun onDisconnected(camera: CameraDevice) {
                    openingCamera = false
                    camera.close()
                    if (cameraDevice === camera) cameraDevice = null
                    reopenIfPending()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    openingCamera = false
                    camera.close()
                    if (cameraDevice === camera) cameraDevice = null
                    toast("Camera error: $error")
                    reopenIfPending()
                }
            }, cameraHandler)
        } catch (e: Exception) {
            openingCamera = false
            toast("相机初始化失败：${e.message}")
        }
    }

    /** 在途的打开请求被放弃（onPause/切镜头）后，若期间又有新的 openCamera 请求，补开。 */
    private fun reopenIfPending() {
        if (!pendingReopen) return
        pendingReopen = false
        if (!resumed) return // onPause 之后不得在后台重开相机
        if (cameraDevice == null && !openingCamera && texture.isAvailable) {
            runOnUiThread { openCamera() }
        }
    }

    private fun sizeFallback(chars: CameraCharacteristics): android.util.Size =
        chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            .getOutputSizes(ImageFormat.YUV_420_888).first()

    private fun choosePreviewSize(
        choices: Array<android.util.Size>,
        processingSize: android.util.Size
    ): android.util.Size {
        if (choices.isEmpty()) return processingSize
        val targetRatio = processingSize.width.toFloat() / processingSize.height.toFloat()
        val targetArea = processingSize.width.toLong() * processingSize.height
        val candidates = choices.filter {
            it.width >= it.height && it.width <= 1920 && it.height <= 1440
        }.ifEmpty { choices.toList() }
        return candidates.sortedWith(
            compareBy<android.util.Size> {
                kotlin.math.abs(it.width.toFloat() / it.height - targetRatio)
            }.thenBy {
                kotlin.math.abs(it.width.toLong() * it.height - targetArea)
            }
        ).first()
    }

    private fun chooseOptimalSize(
        choices: Array<android.util.Size>
    ): android.util.Size {
        if (choices.isEmpty()) return android.util.Size(1280, 960)
        val targetRatio = 4f / 3f
        val targetArea = 1280L * 960L
        val candidates = choices
            .filter { it.width >= it.height && it.width <= 1920 && it.height <= 1440 }
            .ifEmpty { choices.toList() }
        return candidates.sortedWith(
            compareBy<android.util.Size> {
                kotlin.math.abs(it.width.toFloat() / it.height - targetRatio)
            }.thenBy {
                kotlin.math.abs(it.width.toLong() * it.height - targetArea)
            }
        ).first()
    }

    private class CompareSizesByArea : java.util.Comparator<android.util.Size> {
        override fun compare(lhs: android.util.Size, rhs: android.util.Size): Int =
            java.lang.Long.signum(lhs.width.toLong() * lhs.height - rhs.width.toLong() * rhs.height)
    }

    private fun configureTransform(viewW: Int, viewH: Int) {
        texture.surfaceTexture ?: return
        val size = previewSize ?: captureSize ?: return
        if (viewW == 0 || viewH == 0) return

        val surfaceRotationDegrees = displayRotationDegrees()
        val rot = previewRotationDegrees()
        val rotationRequired = rot % 180 != 0
        val viewWf = viewW.toFloat()
        val viewHf = viewH.toFloat()
        var scaleX: Float
        var scaleY: Float
        if (sensorOrientation == 0) {
            scaleX = if (!rotationRequired) viewWf / size.height else viewWf / size.width
            scaleY = if (!rotationRequired) viewHf / size.width else viewHf / size.height
        } else {
            scaleX = if (rotationRequired) viewWf / size.height else viewWf / size.width
            scaleY = if (rotationRequired) viewHf / size.width else viewHf / size.height
        }
        val scale = maxOf(scaleX, scaleY)
        val halfW = viewWf / 2f
        val halfH = viewHf / 2f
        val matrix = android.graphics.Matrix()
        if (rotationRequired) {
            matrix.setScale(scale / scaleX, scale / scaleY, halfW, halfH)
        } else {
            matrix.setScale(
                viewHf / viewWf / scaleY * scale,
                viewWf / viewHf / scaleX * scale,
                halfW,
                halfH
            )
        }
        matrix.postRotate(-surfaceRotationDegrees.toFloat(), halfW, halfH)
        if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            matrix.postScale(-1f, 1f, halfW, halfH)
        }

        texture.setTransform(matrix)
        previewScale = scale
        previewRot = rot

        try {
            val tv = android.graphics.Matrix()
            texture.getTransform(tv)
            val vals = FloatArray(9)
            tv.getValues(vals)
            viewMatrixText = String.format(
                java.util.Locale.US, "[%.3f %.3f %.3f; %.3f %.3f %.3f]",
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
            )
        } catch (_: Throwable) {}

        android.util.Log.d(
            "CameraPreview",
            "sensor=$sensorOrientation facing=$lensFacing rot=$rot cover=$scale hasA=$hasStAffine " +
                "size=${size.width}x${size.height} view=${viewW}x${viewH}"
        )

        // texture.setTransform() 之后必须立刻把「camera -> view」重算一遍，
        // 否则 Renderer 还在用上一套变换，点云会整体偏一段。
        updateCameraToViewTransform()
    }

    /**
     * 求 camera-normalized UV -> view-normalized 的 2x3 仿射，喂给点云 Renderer。
     *
     * 这里刻意复用「点选目标」那条**已经验证正确**的链，只是反过来走。
     *
     * viewToCameraNorm 用的方向是：
     *   viewPixel --(texture transform)^-1--> texturePixel --/size--> textureUV
     *             --(stAffine)--> cameraUV
     *
     * 现在要求反方向：
     *   cameraUV --(stAffine)^-1--> textureUV --*size--> texturePixel
     *            --(texture transform)--> viewPixel --/size--> viewUV
     *
     * 做法是把 cameraUV 空间的一组基（原点 + 两个单位方向）按上面这条链映射到
     * viewUV，三点就唯一确定这个 2x3 仿射。
     *
     * 这样 Renderer 完全不需要知道「90° 竖屏 / 中心裁剪 / 前摄镜像」这些细节，
     * 也就不会再出现「点云和 Preview 各走一条坐标系」的问题 —— 那正是截图上
     * 点云像撒了一屏、且不随镜头产生正确视差的根因。
     */
    private fun updateCameraToViewTransform() {
        try {
            if (!hasStAffine) {
                cameraToViewReady = false
                return
            }
            if (texture.width <= 0 || texture.height <= 0) {
                cameraToViewReady = false
                return
            }
            val a = stAffine
            val st = android.graphics.Matrix().apply {
                setValues(
                    floatArrayOf(
                        a[0], a[1], a[2],
                        a[3], a[4], a[5],
                        0f, 0f, 1f
                    )
                )
            }
            val invSt = android.graphics.Matrix()
            if (!st.invert(invSt)) {
                cameraToViewReady = false
                return
            }

            // cameraUV 空间的三点：(0,0) (1,0) (0,1)
            val p = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f)
            // cameraUV -> textureUV
            invSt.mapPoints(p)

            val w = texture.width.toFloat()
            val h = texture.height.toFloat()
            for (i in 0..2) {
                p[i * 2] *= w
                p[i * 2 + 1] *= h
            }

            // texturePixel -> viewPixel
            val tv = android.graphics.Matrix()
            texture.getTransform(tv)
            tv.mapPoints(p)

            for (i in 0..2) {
                p[i * 2] /= w
                p[i * 2 + 1] /= h
            }

            val x0 = p[0]
            val y0 = p[1]
            val m = floatArrayOf(
                p[2] - x0, p[4] - x0, x0,
                p[3] - y0, p[5] - y0, y0
            )
            renderer.setCameraToView(m)
            // 目标框也要用同一套变换，所以在这里一并留存。
            // 只有一个真源，就不会出现「点云对了、绿框还是错的」。
            cameraToView = m
            cameraToViewReady = true
        } catch (t: Throwable) {
            cameraToViewReady = false
            android.util.Log.e("CameraPreview", "updateCameraToViewTransform failed", t)
        }
    }

    @Suppress("DEPRECATION")
    private fun displayRotationDegrees(): Int {
        val displayRotation = if (Build.VERSION.SDK_INT >= 30) {
            display?.rotation ?: Surface.ROTATION_0
        } else {
            windowManager.defaultDisplay.rotation
        }
        return when (displayRotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
    }
    private fun previewRotationDegrees(): Int {
        val surfaceRotationDegrees = displayRotationDegrees()
        val sign = if (lensFacing == CameraCharacteristics.LENS_FACING_FRONT) 1 else -1
        return (sensorOrientation - surfaceRotationDegrees * sign + 360) % 360
    }

    /**
     * HQ 状态机（IDLE → WAIT_3A → READY → CAPTURE）只做「当前在干什么」的镜像。
     *
     * 这里**不再**跟随控制器去锁 preview 的 3A。旧实现会在 lockRequested 上升沿把
     * AE_LOCK / AWB_LOCK 与手动曝光/焦距写回 repeating preview，结果 preview 的
     * AE_MODE 被关成 OFF，而 AE_MODE=OFF 时 AE_STATE 按规范只会报 INACTIVE——
     * 于是控制器自己要求的 VERIFY_LOCK 三重 LOCKED 永远不可能满足，
     * 1366 次拍摄机会全部卡死，一张 HQ 都没拍到。
     *
     * 现在：preview 全程自动 3A（只负责「收敛」），真正的 3A 冻结只发生在
     * HqCaptureController.buildLockedStillRequest() 的 still request 上。
     */
    private fun syncCaptureStateFromController() {
        if (!::hqCapture.isInitialized) return
        val st = hqCapture.stats
        if (st.captureState == captureState) return
        captureState = st.captureState
    }

    private fun applyCaptureSettings() {
        val handler = cameraHandler ?: return
        if (android.os.Looper.myLooper() != handler.looper) {
            handler.post { applyCaptureSettings() }
            return
        }
        val session = captureSession ?: return
        val camera = cameraDevice ?: return
        val readerSurface = reader?.surface ?: return
        val ps = previewSurface ?: return
        try {
            val req = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                addTarget(ps)
                addTarget(readerSurface)
                if (multiCam?.active == true) {
                    multiCam?.secondarySurface?.let { addTarget(it) }
                }
                // Preview 的 3A 全程保持自动。
                // HQ 采集只要求「收敛」，不要求「锁定」；真正把参数冻住的地方是
                // HqCaptureController.buildLockedStillRequest()（still request）。
                // 这里绝不能再跟随控制器写 AE_MODE_OFF / 手动曝光 / 手动焦距——
                // 那会把 AE_STATE 打成 INACTIVE，让控制器自己的收敛判定永远不成立。
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                set(CaptureRequest.FLASH_MODE, if (torchRequested && torchAvailable)
                    CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF)
                if (aeLockAvailable) set(CaptureRequest.CONTROL_AE_LOCK, aeLock)
                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                if (awbLockAvailable) set(CaptureRequest.CONTROL_AWB_LOCK, awbLock)

                val afMode = if (targetAfLockEnabled && targetFocusLocked && targetFocusDistance != null && manualFocusAvailable) {
                    CaptureRequest.CONTROL_AF_MODE_OFF
                } else when {
                    fixedFocus -> CaptureRequest.CONTROL_AF_MODE_OFF
                    focusTriggered -> CaptureRequest.CONTROL_AF_MODE_AUTO
                    else -> bestContinuousAfMode()
                }
                set(CaptureRequest.CONTROL_AF_MODE, afMode)
                if (targetAfLockEnabled && targetFocusLocked && targetFocusDistance != null && manualFocusAvailable) {
                    set(CaptureRequest.LENS_FOCUS_DISTANCE, targetFocusDistance!!)
                }
                focusRegion?.let { region ->
                    set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(region))
                    set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(region))
                }

                if (noiseModes.contains(CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL)) {
                    set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL)
                }
                if (edgeModes.contains(CaptureRequest.EDGE_MODE_OFF)) {
                    set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_OFF)
                }

                if (stabilization && videoStabModes.contains(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON)) {
                    set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON)
                } else {
                    set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
                }
                if (oisSupported) {
                    set(
                        CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                        if (stabilization) CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                        else CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                    )
                }
            }.build()
            if (focusTriggered) {
                // AF_TRIGGER 是一次性动作：先发一帧带 TRIGGER_START 的单次请求，
                // 再回到不带 trigger 的 repeating（原来放在 repeating 里等于每帧触发）
                val trigger = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                    addTarget(ps)
                    addTarget(readerSurface)
                    if (multiCam?.active == true) {
                        multiCam?.secondarySurface?.let { addTarget(it) }
                    }
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                    focusRegion?.let { region ->
                        set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(region))
                        set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(region))
                    }
                    set(CaptureRequest.FLASH_MODE, if (torchRequested && torchAvailable)
                        CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF)
                    set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                }.build()
                session.capture(trigger, null, cameraHandler)
            }
            session.setRepeatingRequest(req, captureResultCallback, cameraHandler)
        } catch (e: Exception) {
            if (torchRequested) {
                torchRequested = false
                torchFailed = true
                if (::hqCapture.isInitialized) hqCapture.setFillLight(false)
                android.util.Log.w("AutoFillLight", "Torch request rejected; restoring preview", e)
                applyCaptureSettings() // bounded retry: torchRequested is now false
            } else {
                toast("应用相机参数失败：${e.message}")
            }
        }
    }

    /**
     * Preview 的连续自动对焦模式。
     *
     * 优先 CONTINUOUS_PICTURE：本 app 的交付物是 HQ 静帧，画面构图不变的前提下
     * 对焦精度比「录像模式的平滑度」更重要；拿到更准的 LENS_FOCUS_DISTANCE
     * 才能让 burst 的冻结焦距更可信。不支持时才退回 CONTINUOUS_VIDEO。
     */
    private fun bestContinuousAfMode(): Int = when {
        afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE) ->
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        afModes.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO) ->
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO
        else -> CaptureRequest.CONTROL_AF_MODE_AUTO
    }

    private fun viewToCameraNorm(x: Float, y: Float): android.graphics.PointF? {
        if (texture.width <= 0 || texture.height <= 0) return null
        val viewMatrix = android.graphics.Matrix()
        texture.getTransform(viewMatrix)
        val inverse = android.graphics.Matrix()
        if (!viewMatrix.invert(inverse)) return null
        val p = floatArrayOf(x, y)
        inverse.mapPoints(p)
        var u = (p[0] / texture.width.toFloat()).coerceIn(0f, 1f)
        var v = (p[1] / texture.height.toFloat()).coerceIn(0f, 1f)
        if (hasStAffine) {
            val a = stAffine
            val uu = a[0] * u + a[1] * v + a[2]
            val vv = a[3] * u + a[4] * v + a[5]
            u = uu
            v = vv
        }
        return android.graphics.PointF(u.coerceIn(0f, 1f), v.coerceIn(0f, 1f))
    }

    private fun selectTarget(x: Float, y: Float) {
        val norm = viewToCameraNorm(x, y) ?: return
        targetTapX = x
        targetTapY = y
        val ok = try {
            NativeBridge.nativeSelectTarget(norm.x, norm.y)
        } catch (t: Throwable) {
            false
        }
        if (!ok) {
            targetOverlay.state = TargetUiState(visible = true, state = 1)
            toast("目标暂时无法锁定，请重新点击")
            return
        }
        targetOverlay.state = TargetUiState(visible = true, state = 2)
        // 崩溃隔离测试阶段暂时禁用
        // focusAt(x, y)
        targetFocusLocked = false
        targetFocusDistance = null
        // V0.12 ⑧：锁定后检查一次初始框离画面边缘的距离（见 updateTargetOverlay）。
        edgeMarginCheckUntilNs = System.nanoTime() + 3_000_000_000L
        updateTargetOverlay()
    }

    /** 拖框的最小边长（view 像素）。低于它就当成点按。 */
    private fun dragSelectMinPx(): Float = 24f * resources.displayMetrics.density

    /**
     * camera-normalized UV -> view-normalized UV 的**正**映射。
     *
     * 与 [viewToCameraNorm] 严格互逆，和点云 Renderer 共用同一个 [cameraToView]。
     * 这里刻意不写任何「90° / 裁剪 / 镜像」逻辑 —— 那些细节已经被
     * updateCameraToViewTransform() 吸收进这个 2x3 矩阵里了。
     */
    private fun cameraNormToView(u: Float, v: Float): android.graphics.PointF {
        val a = cameraToView
        return android.graphics.PointF(
            a[0] * u + a[1] * v + a[2],
            a[3] * u + a[4] * v + a[5]
        )
    }

    /**
     * 手指拖框选择目标。
     *
     * 解决的是「固定 240x240 ROI 把背景一起锁进去」：单点只能给一个与目标
     * 形状无关的方块，点一个细长瓶子时 KLT 实际追的是
     * 「物体 + 墙 + 桌子」的混合纹理，框当然贴不住目标。
     * 拖框让用户直接把 bbox 告诉 tracker，几乎没有额外算力。
     */
    private fun selectTargetRect(vx0: Float, vy0: Float, vx1: Float, vy1: Float) {
        val a = viewToCameraNorm(vx0, vy0)
        val b = viewToCameraNorm(vx1, vy1)
        if (a == null || b == null) {
            toast("无法换算框选坐标")
            return
        }
        // 允许任意方向拖动；相机坐标系下取轴对齐包围盒
        val lx = minOf(a.x, b.x)
        val ly = minOf(a.y, b.y)
        val hx = maxOf(a.x, b.x)
        val hy = maxOf(a.y, b.y)
        val ok = try {
            NativeBridge.nativeSelectTargetRect(lx, ly, hx, hy)
        } catch (t: Throwable) {
            false
        }
        if (!ok) {
            targetOverlay.state = TargetUiState(
                visible = true, state = NativeBridge.TARGET_STATE_ARMED
            )
            toast("框选无效，请重新框选（框太小或未开启物体追踪）")
            return
        }
        targetTapX = (vx0 + vx1) * 0.5f
        targetTapY = (vy0 + vy1) * 0.5f
        targetOverlay.state = TargetUiState(
            visible = true, state = NativeBridge.TARGET_STATE_ACQUIRING
        )
        targetFocusLocked = false
        targetFocusDistance = null
        // V0.12 ⑧：拖框时可以直接用用户给的框判，比等 native 回框更快。
        val margin = minOf(lx, ly, 1f - hx, 1f - hy)
        edgeMarginCheckUntilNs = 0L
        if (margin < targetEdgeMarginWarn) {
            toast("请把目标完整放入画面中间再锁定（当前框贴到画面边缘）")
        }
        updateTargetOverlay()
    }

    private fun updateTargetOverlay() {
        val out = FloatArray(NativeBridge.TARGET_STATE_SLOTS)
        val state = try {
            NativeBridge.nativeGetTargetState(out)
        } catch (t: Throwable) {
            0
        }
        targetState = state
        targetConfidence = out.getOrElse(5) { 0f }
        targetTrackedPoints = out.getOrElse(8) { 0f }.toInt()
        val targetMedian = out.getOrElse(6) { 0f }
        if (targetMedian.isFinite() && targetMedian > 0f) {
            targetMedianDepthMeters = targetMedian
        }
        val visibleFraction =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_VISIBLE_FRACTION) { 1f }
        val centerX = out.getOrElse(NativeBridge.TARGET_STATE_INDEX_CENTER_X) { 0.5f }
        val centerY = out.getOrElse(NativeBridge.TARGET_STATE_INDEX_CENTER_Y) { 0.5f }
        targetVisibleFraction = visibleFraction
        targetCenterX = centerX
        targetCenterY = centerY
        val presenceValid =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_PRESENCE_VALID) { 1f } > 0.5f
        targetPresenceValid = presenceValid

        // 绿框可见性：
        //   TRACKING  —— 还必须 PresenceGate 也认可（外观 + Mask + 运动 + 中心一致），
        //                否则「物体已经走了、框还挂在墙上」；
        //   ARM/ACQUIRING —— 用户刚点/刚框完，还没有判定依据，照常显示；
        //   LOST/REACQUIRING —— bbox 已经退化成残片/空矩形，画出来只会误导，
        //                统一交给常驻的 targetWarningText 提示。
        val showBox = when (state) {
            NativeBridge.TARGET_STATE_OFF -> false
            NativeBridge.TARGET_STATE_TRACKING -> presenceValid
            NativeBridge.TARGET_STATE_ARMED,
            NativeBridge.TARGET_STATE_ACQUIRING -> true
            else -> false
        }
        targetOverlay.state = TargetUiState(
            visible = showBox,
            state = state,
            x0 = out.getOrElse(1) { 0f },
            y0 = out.getOrElse(2) { 0f },
            x1 = out.getOrElse(3) { 0f },
            y1 = out.getOrElse(4) { 0f },
            confidence = out.getOrElse(5) { 0f },
            medianDepth = out.getOrElse(6) { 0f },
            visibleFraction = visibleFraction,
            centerX = centerX,
            centerY = centerY,
            presenceValid = presenceValid
        )
        targetOverlay.invalidate()

        // V0.12 ⑧：点选（单点）走的是 native 的固定 ROI，只有等 native 把 bbox
        // 回上来才知道框有没有贴边；所以这里做一次性检查，3s 内没拿到就作废。
        if (edgeMarginCheckUntilNs != 0L) {
            if (System.nanoTime() > edgeMarginCheckUntilNs) {
                edgeMarginCheckUntilNs = 0L
            } else {
                val bx0 = out.getOrElse(1) { 0f }
                val by0 = out.getOrElse(2) { 0f }
                val bx1 = out.getOrElse(3) { 0f }
                val by1 = out.getOrElse(4) { 0f }
                if (bx1 > bx0 && by1 > by0) {
                    edgeMarginCheckUntilNs = 0L
                    val margin = minOf(bx0, by0, 1f - bx1, 1f - by1)
                    if (margin < targetEdgeMarginWarn) {
                        toast("请把目标完整放入画面中间再锁定（当前框贴到画面边缘）")
                    }
                }
            }
        }

        val appearanceOkDiag = out.getOrElse(NativeBridge.TARGET_STATE_INDEX_APPEARANCE_OK) { 0f } > 0.5f
        val identityScoreDiag = out.getOrElse(NativeBridge.TARGET_STATE_INDEX_IDENTITY_SCORE) { -1f }
        if (state == NativeBridge.TARGET_STATE_REACQUIRING || state == NativeBridge.TARGET_STATE_LOST) {
            android.util.Log.w("TargetWarn", String.format(
                "DIAG state=%d vf=%.3f present=%b appearanceOk=%b identity=%.2f",
                state, visibleFraction, presenceValid, appearanceOkDiag, identityScoreDiag))
        }
        updateTargetWarning(state, visibleFraction, centerX, centerY, presenceValid)
    }

    /**
     * 目标 Overlay 的 30Hz 节拍。
     *
     * **刻意不从 updateHeader() 里调**：updateHeader 只在 `dt >= 1.0` 时
     * 触发，那样绿框一秒才跳一次。native tracker 其实每帧都在跑，
     * 只是 UI 没把结果画出来。
     */
    private fun updateTargetUiTick(ts: Long) {
        if (!objectLockEnabled) return
        if (targetUiLastNs != 0L && ts - targetUiLastNs < targetUiPeriodNs) return
        targetUiLastNs = ts
        runOnUiThread { updateTargetOverlay() }
    }

    /**
     * 目标接近边缘 / 已离开画面的**持续**提示。
     *
     * native 早就有 `visibleFraction < 0.40 -> near frame edge` 和
     * `< 0.12 x 2 帧 -> 出界`（V0.12 由 3 帧改为 2 帧），但旧的 UI 只把
     * state==4 画成红框 —— 而目标完全出界时 bbox 是 0x0，红框也画不出来，
     * 用户看到的就是「什么都没发生」。
     *
     * V0.12 改成**两级 camera-rate**：本函数由 updateTargetOverlay() 以 30Hz
     * 调用，所以直接拿本帧的 visibleFraction 判就有了「第一时间」的时效，
     * 不再依赖 native 的 slow state 转换。
     */
    private fun updateTargetWarning(state: Int, visibleFraction: Float, cx: Float, cy: Float,
                                    presenceValid: Boolean) {
        val prev = lastTargetUiState
        lastTargetUiState = state
        // 只在「跟得好好的 -> 目标离开画面」这个边沿震一次。
        // 每帧震会变成持续嗡嗡声，比不震还烦。
        if (prev == NativeBridge.TARGET_STATE_TRACKING &&
            (state == NativeBridge.TARGET_STATE_REACQUIRING ||
                state == NativeBridge.TARGET_STATE_LOST)) {
            vibrateOnce()
        }

        val text = when {
            state == NativeBridge.TARGET_STATE_LOST ->
                // 数据实证：物体几何完全在画面(vf≈1.0)、外观匹配、存在性正常时
                // 仍会被判 LOST（KLT 主链仿射/特征失败），此时说「离开画面」是
                // 语义错配，会误导用户以为物体出界。仅当 vf 真的低(出界)才报离开。
                if (visibleFraction >= 0.5f)
                    "⚠ 跟踪暂时丢失，正在自动找回…" + edgeHint(cx, cy)
                else
                    "目标已离开画面" + edgeHint(cx, cy)

            state == NativeBridge.TARGET_STATE_REACQUIRING ->
                "目标暂时丢失，正在自动找回…" + edgeHint(cx, cy)

            // PresenceGate 在本帧就否掉了「目标还在」：bbox 可能仍然好好地
            // 停在画面中间（tracker 失配到背景纹理上了），但语义上目标已经不在。
            state == NativeBridge.TARGET_STATE_TRACKING && !presenceValid ->
                "⛔ 目标外观/深度不一致，正在自动找回…"

            // V0.12 两级预警：**不等 native state**，直接用本帧（30Hz camera-rate）
            // 的 visibleFraction 判。旧的单一 0.40 阈值太晚，用户感受就是
            // 「物体都快出画了才警告」。慢的那一层（PresenceGate / 几何出界）
            // 保留做语义确认，不再承担「第一时间提醒」的职责。
            state == NativeBridge.TARGET_STATE_TRACKING && visibleFraction < 0.30f ->
                "⛔ 目标即将离开画面" + edgeHint(cx, cy)

            state == NativeBridge.TARGET_STATE_TRACKING && visibleFraction < 0.65f ->
                "⚠ 目标接近边缘，请减速" + edgeHint(cx, cy)

            else -> null
        }

        if (text != null) {
            android.util.Log.w(
                "TargetWarn",
                String.format(
                    "WARN state=%d vf=%.3f cx=%.2f cy=%.2f present=%b : %s",
                    state, visibleFraction, cx, cy, presenceValid, text
                )
            )
        }

        if (text == null) {
            targetWarningText.visibility = android.view.View.GONE
        } else {
            targetWarningText.text = text
            // V0.12: 两级配色 —— 「接近边缘」琥珀底，「即将/已经离开」红底。
            // 底色比文字更快传达严重程度（余光就能看见）。
            val severe = text.startsWith("⛔")
            targetWarningText.setBackgroundColor(
                if (severe) android.graphics.Color.argb(225, 210, 40, 40)
                else android.graphics.Color.argb(215, 190, 120, 0)
            )
            targetWarningText.visibility = android.view.View.VISIBLE
        }
    }

    /**
     * 根据目标中心（相机归一化，可能已被夹到 [0,1] 之外）给出找回方向。
     * 返回空串表示中心还在画面中部，不需要指方向。
     */
    private fun edgeHint(cx: Float, cy: Float): String = when {
        cx < 0.20f -> "\n← 向左移动手机找回目标"
        cx > 0.80f -> "\n→ 向右移动手机找回目标"
        cy < 0.20f -> "\n↑ 向上移动手机找回目标"
        cy > 0.80f -> "\n↓ 向下移动手机找回目标"
        else -> ""
    }

    private fun vibrateOnce() {
        try {
            val v = vibrator ?: return
            if (!v.hasVibrator()) return
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                v.vibrate(
                    android.os.VibrationEffect.createOneShot(
                        100, android.os.VibrationEffect.DEFAULT_AMPLITUDE
                    )
                )
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(100)
            }
        } catch (_: Throwable) {
        }
    }

    private fun maybeRelockTargetFocus() {
        if (!targetAfLockEnabled || !objectLockEnabled || !manualFocusAvailable) return
        if (targetState != 3 || targetConfidence <= 0.60f || targetTrackedPoints < 20) return
        if (!targetFocusLocked) {
            targetFocusDistance = lastLensFocusDistance
            if (targetFocusDistance != null && targetFocusDistance!! > 0.01f) {
                targetFocusLocked = true
                applyCaptureSettings()
            }
            return
        }
        val af = lastAfState ?: return
        val focused = af == android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
            af == android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED
        if (focused) {
            targetRelockFrames = 0
            return
        }
        targetRelockFrames++
        if (targetRelockFrames >= 5) {
            targetRelockFrames = 0
            focusRelockCount++
            focusAt(targetTapX, targetTapY)
        }
    }

    private fun focusAt(x: Float, y: Float) {
        if (fixedFocus) return
        val rect = activeArray
        if (rect.isEmpty) return
        val norm = viewToCameraNorm(x, y) ?: return
        var nx = norm.x
        var ny = norm.y
        val rot = previewRotationDegrees()
        if (rot == 90 || rot == 270) {
            val t = nx
            nx = if (rot == 90) 1f - ny else ny
            ny = t
        }
        val cx = (nx * rect.width()).toInt()
        val cy = (ny * rect.height()).toInt()
        val halfW = (rect.width() / 8).coerceAtLeast(1)
        val halfH = (rect.height() / 8).coerceAtLeast(1)
        focusRegion = MeteringRectangle(
            (cx - halfW).coerceIn(0, rect.width() - 1),
            (cy - halfH).coerceIn(0, rect.height() - 1),
            (halfW * 2).coerceIn(1, rect.width()),
            (halfH * 2).coerceIn(1, rect.height()),
            1000
        )
        focusTriggered = true
        applyCaptureSettings()
        // 连续点按时先移除旧回调，避免多个恢复任务竞争改回 AF 模式
        cameraHandler?.removeCallbacks(resetFocusRunnable)
        cameraHandler?.postDelayed(resetFocusRunnable, 900L)
    }

    private val resetFocusRunnable = Runnable {
        focusTriggered = false
        applyCaptureSettings()
    }

    /**
     * 只负责把报告拼成字符串，**不碰系统文件选择器**。
     *
     * 与 [exportFeedbackSafely] 拆开之后，失败原因能区分成两类：
     *   - 这里抛异常   -> 「报告生成失败」（诊断字段/格式问题）
     *   - launcher 抛  -> 「系统保存界面启动失败」（权限/URI 问题）
     * 合在一起时只能看到一个笼统的 "保存失败"。
     */
    private fun buildFeedbackReport(): String {
        val sb = StringBuilder()
        sb.appendLine("Scan settings: maxDistanceMeters=$scanMaxDistanceMeters autoFillLight=$autoFillLight torchRequested=$torchRequested torchAvailable=$torchAvailable torchFailed=$torchFailed")
        sb.appendLine("Depth worker: busy=${depthBusy.get()} errors=$depthErrors lastError=$depthLastError lastCompletedElapsedMs=$depthCompletedMs")
        sb.appendLine("ThermalGuard: thermalStatus=$thermalStatus depthMinIntervalMs=${thermalDepthMinIntervalMs()} meshIntervalScale=${thermalMeshIntervalScale()} liveMeshLastBuildMs=$liveMeshLastBuildMs")
        sb.appendLine("CameraRange: status=$cameraRangeStatus mode=${cameraRangePolicy.mode.label} range=${cameraRangePolicy.range.label} aux=${multiCam?.secondaryId} selectableAux=${multiCam?.selectableAuxCount ?: -1}")
        sb.appendLine("CameraRange caps: ${multiCam?.capabilitySnapshot() ?: "unavailable"}")
        sb.appendLine("MobileScan3D 配置反馈报告")
        sb.appendLine("时间戳: ${System.currentTimeMillis()}")
        sb.appendLine()
        sb.appendLine("[SESSION]")
        sb.appendLine("gitCommit=$buildGitSha")
        sb.appendLine("buildTime=${BuildConfig.BUILD_TIME_MS}")
        sb.appendLine("sessionId=$sessionId")
        sb.appendLine("scanStartTimestamp=$sessionStartTs")
        sb.appendLine("reportTimestamp=${System.currentTimeMillis()}")
        sb.appendLine("lastPlyFilename=${lastPlyFilename ?: "unknown"}")
        sb.appendLine("plyVertexCount=${lastPlyVertexCount ?: "unknown"}")
        sb.appendLine("plyFileBytes=${lastPlyFileBytes ?: "unknown"}")
        sb.appendLine("plyExportTimestamp=${lastPlyExportTs ?: "unknown"}")
        sb.appendLine("plySessionId=${lastPlySessionId ?: "unknown"}")
        sb.appendLine("plySessionMatch=${lastPlySessionId != null && lastPlySessionId == sessionId}")
        sb.appendLine()
        sb.appendLine("[SELF-CHECK SUMMARY]")
        val warns = mutableListOf<String>()
        if (NativeBridge.nativeVinsInitialized()) sb.appendLine("PASS VINS") else warns.add("VINS not initialized")
        if (targetState == 4) warns.add("TARGET_LOST")
        if (targetTrackedPoints in 1..19) warns.add("TARGET_LOW_FEATURES")
        if (!cameraToViewReady) warns.add("DISPLAY_TRANSFORM_NOT_READY")
        if (lastPlyVertexCount == 0) warns.add("PLY_EMPTY")
        if (lastPlySessionId != null && lastPlySessionId != sessionId) warns.add("PLY_SESSION_MISMATCH")
        if (frameMetaMiss > frameMetaHit / 20L && frameMetaHit > 0) warns.add("CAMERA_METADATA_LOSS")
        if (::hqCapture.isInitialized && hqCapture.stats.lastCaptureRejectReason.isNotEmpty()) {
            warns.add("HQ_CAPTURE_REJECT")
        }
        if (warns.isEmpty()) {
            sb.appendLine("overall=PASS")
        } else {
            sb.appendLine("overall=WARNING")
            warns.forEach { sb.appendLine("WARN $it") }
        }
        val metaTotal = frameMetaHit + frameMetaMiss
        if (metaTotal > 0) {
            sb.appendLine("metadataMatchRate=${"%.2f".format(frameMetaHit * 100.0 / metaTotal)}%")
            sb.appendLine("metadataMissRate=${"%.2f".format(frameMetaMiss * 100.0 / metaTotal)}%")
        }
        sb.appendLine()
        sb.appendLine("[0] 设备型号: $deviceModel")
        sb.appendLine("    预览方向 sensorOrientation=$sensorOrientation actualPreviewRot=$previewRot")
        sb.appendLine("[1] 相机对焦模式: ${if (fixedFocus) "固定对焦(AF关闭)" else "自动对焦(连续)"}")
        sb.appendLine("    支持的AF模式: ${afModes.joinToString(",")}")
        sb.appendLine()
        sb.appendLine("[2] IMU 标定 YAML:")
        sb.appendLine("    acc_n: 0.1")
        sb.appendLine("    acc_w: 0.001")
        sb.appendLine("    gyr_n: 0.001")
        sb.appendLine("    gyr_w: 0.0001")
        sb.appendLine("    image_width: $nativeW")
        sb.appendLine("    image_height: $nativeH")
        sb.appendLine("    fx: $nativeFx")
        sb.appendLine("    fy: $nativeFy")
        sb.appendLine("    cx: $nativeCx")
        sb.appendLine("    cy: $nativeCy")
        sb.appendLine("    ric: [1,0,0,0,1,0,0,0,1]  # 相机-IMU旋转(待标定)")
        sb.appendLine("    tic: [0,0,0]              # 相机-IMU平移(待标定)")
        sb.appendLine("    lastAccel: (${lastImu[0]}, ${lastImu[1]}, ${lastImu[2]})")
        sb.appendLine("    lastGyro: (${lastImu[3]}, ${lastImu[4]}, ${lastImu[5]})")
        sb.appendLine()
        sb.appendLine("[3] 点云采集计算数据:")
        sb.appendLine("CameraRange: $cameraRangeStatus switching=$cameraRangeSwitching")
        sb.appendLine(NativeBridge.nativeGetStats())
        sb.appendLine()
        sb.appendLine("[4] 相机防抖(EIS): ${if (stabilization) "开" else "关"}，设备支持: ${videoStabModes.contains(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON)}")
        sb.appendLine("    光学防抖(OIS): ${if (oisSupported) "支持(已${if (stabilization) "开" else "关"})" else "不支持"}")
        sb.appendLine("    对焦开关: fixedFocus=$fixedFocus  防抖开关: stabilization=$stabilization")
        sb.appendLine("    曝光锁定支持: $aeLockAvailable  当前: $aeLock")
        sb.appendLine("    白平衡锁定支持: $awbLockAvailable  当前: $awbLock")
        sb.appendLine("    手动对焦支持: $manualFocusAvailable  minFocusDistance: $minFocusDistance")
        sb.appendLine("    输出尺寸与标定缩放: activeArray=${activeArray.width()}x${activeArray.height()}  captureSize=${captureSize?.let { "${it.width}x${it.height}" }}")
        sb.appendLine("    降噪/锐化: 降噪最小化=${noiseModes.contains(CaptureRequest.NOISE_REDUCTION_MODE_MINIMAL)}  锐化关闭=${edgeModes.contains(CaptureRequest.EDGE_MODE_OFF)}")
        sb.appendLine()
        if (::hqCapture.isInitialized) {
            hqCapture.report(sb)
            sb.appendLine()
        }
        sb.appendLine("[5] 预览与坐标一致性:")
        sb.appendLine("    屏幕尺寸: ${screenW}x${screenH}")
        sb.appendLine("    支持的预览尺寸: $supportedSizes")
        sb.appendLine("    最终选中 YUV 尺寸: $selectedSizeText")
        sb.appendLine("    最终选中预览尺寸: $selectedPreviewSizeText")
        sb.appendLine("    旋转角度: $previewRot°  (sensorOrientation=$sensorOrientation, lensFacing=$lensFacing)")
        sb.appendLine("    预览填充缩放: scaleX=$previewScale, scaleY=$previewScale")
        sb.appendLine("    SurfaceTexture 纹理矩阵: $stMatrixText")
        sb.appendLine("    视图变换矩阵: $viewMatrixText")
        sb.appendLine("    视图实测尺寸: ${texture.width}x${texture.height}")
        sb.appendLine("    点云/相机一致性: 见上方 [3] 的 Cam pose t 与 PointCloud vs Cam offset；offset 距离越小说明点云质心越靠近相机位姿。")
        sb.appendLine()
        sb.appendLine("[6] 摄像头/屏幕硬件匹配:")
        sb.appendLine("    机型: $deviceModel  Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        sb.appendLine("    屏幕: ${screenW}x${screenH}  densityDpi=${resources.displayMetrics.densityDpi}  density=${resources.displayMetrics.density}")
        sb.appendLine("    当前镜头 ID: $currentCameraId  lensFacing=$lensFacing  sensorOrientation=$sensorOrientation")
        try {
            val ch = cameraManager.getCameraCharacteristics(currentCameraId)
            val physical = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val pixelArray = ch.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            val capabilities = ch.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            sb.appendLine("    像素阵列: ${pixelArray?.let { "${it.width}x${it.height}" } ?: "n/a"}")
            sb.appendLine("    物理传感器尺寸: ${physical?.let { "${it.width}x${it.height} mm" } ?: "n/a"}")
            sb.appendLine("    焦距候选: ${focalLengths.joinToString(", ") { "%.3f".format(it) }}")
            sb.appendLine("    光圈候选: ${apertures.joinToString(", ") { "%.3f".format(it) }}")
            sb.appendLine("    LOGICAL_MULTI_CAMERA: ${capabilities?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA) == true}")
            val timestampSource = ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
            sb.appendLine(
                "    Camera timestamp source: " +
                    when (timestampSource) {
                        CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME -> "REALTIME"
                        CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN -> "UNKNOWN"
                        else -> "UNKNOWN_VALUE($timestampSource)"
                    }
            )
            sb.appendLine("    Active physical camera: " + (activePhysicalCameraId ?: "unknown"))
            sb.appendLine(
                "    Rolling shutter skew: " +
                    (rollingShutterSkewNs?.let { "${it / 1_000_000.0} ms" } ?: "unknown")
            )
            sb.appendLine(
                "    Exposure time: " +
                    (exposureTimeNs?.let { "${it / 1_000_000.0} ms" } ?: "unknown")
            )
            sb.appendLine("    Crop region: " + (lastCropRegion ?: "unknown"))
            sb.appendLine("    Distortion correction mode: " + (lastDistortionCorrectionMode ?: "unknown"))
            sb.appendLine("    Lens distortion: " + (lensDistortion?.joinToString(", ") ?: "unknown"))
            sb.appendLine("    Frame metadata matched: $frameMetaHit")
            sb.appendLine("    Frame metadata missed: $frameMetaMiss")
            sb.appendLine("    Frame metadata lateMatched: $frameMetaLateHit")
            sb.appendLine("    Frame metadata expired: $frameMetaExpired")
            sb.appendLine(
                "    Frame metadata pending: " +
                    synchronized(frameMetaLock) { pendingFrameMeta.size }
            )
            val poseRef = ch.get(CameraCharacteristics.LENS_POSE_REFERENCE)
            val poseRot = ch.get(CameraCharacteristics.LENS_POSE_ROTATION)
            val poseTrans = ch.get(CameraCharacteristics.LENS_POSE_TRANSLATION)
            sb.appendLine("    Lens pose reference: $poseRef")
            sb.appendLine("    Lens pose rotation: " + (poseRot?.joinToString(", ") ?: "null"))
            sb.appendLine("    Lens pose translation: " + (poseTrans?.joinToString(", ") ?: "null"))
        } catch (_: Throwable) {
            sb.appendLine("    摄像头硬件参数读取失败")
        }
        for (cid in cameraIds) {
            try {
                val ch = cameraManager.getCameraCharacteristics(cid)
                val facing = ch.get(CameraCharacteristics.LENS_FACING) ?: 0
                val orient = ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
                val focal = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
                val ap = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.firstOrNull() ?: 0f
                val face = if (facing == CameraCharacteristics.LENS_FACING_FRONT) "前置" else "后置"
                sb.appendLine("    镜头[$cid] $face ${orient}° f=${"%.3f".format(focal)}mm f/${"%.3f".format(ap)}" + if (cid == currentCameraId) "  ✓当前" else "")
            } catch (_: Throwable) {}
        }
        sb.appendLine()
        sb.appendLine("[7] Object Lock:")
        sb.appendLine("    enabled=$objectLockEnabled")
        val out = FloatArray(NativeBridge.TARGET_STATE_SLOTS)
        val state = NativeBridge.nativeGetTargetState(out)
        sb.appendLine("    state=$state")
        sb.appendLine("    bboxCamera=(${out.getOrElse(1) { 0f }}, ${out.getOrElse(2) { 0f }}, ${out.getOrElse(3) { 0f }}, ${out.getOrElse(4) { 0f }})")
        sb.appendLine("    confidence=${out.getOrElse(5) { 0f }}")
        sb.appendLine("    medianDepth=${out.getOrElse(6) { 0f }}")
        sb.appendLine("    roiSharpness=${out.getOrElse(7) { 0f }}")
        sb.appendLine("    trackedPoints=${out.getOrElse(8) { 0f }.toInt()}")
        sb.appendLine("    inlierRatio=${out.getOrElse(9) { 0f }}")
        sb.appendLine("    visibleFraction=${out.getOrElse(NativeBridge.TARGET_STATE_INDEX_VISIBLE_FRACTION) { 1f }}")
        sb.appendLine("    centerNorm=(${out.getOrElse(NativeBridge.TARGET_STATE_INDEX_CENTER_X) { 0.5f }}, ${out.getOrElse(NativeBridge.TARGET_STATE_INDEX_CENTER_Y) { 0.5f }})")
        sb.appendLine("    edgeLostFrames=${out.getOrElse(NativeBridge.TARGET_STATE_INDEX_EDGE_LOST_FRAMES) { 0f }.toInt()}")
        val reportPresence =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_PRESENCE_VALID) { 1f } > 0.5f
        val reportAppearance =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_APPEARANCE_OK) { 0f } > 0.5f
        sb.appendLine("    presenceValid=$reportPresence (box in frame != object present)")
        sb.appendLine("    appearanceBackendOk=$reportAppearance")
        // V0.13：身份 / 膨胀守卫读数。**必须和 nanoScore 放在一起读** ——
        // 「nanoScore 0.94 而 identityScore 0.12」正是 V0.12 那个故障的
        // 指纹：两个 tracker 彼此确认得很漂亮，但都已经不在目标上了。
        val identityScore = out.getOrElse(NativeBridge.TARGET_STATE_INDEX_IDENTITY_SCORE) { -1f }
        val identityRejects =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_IDENTITY_REJECTS) { 0f }.toLong()
        val bboxScaleFromInitial =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_BBOX_SCALE) { 1f }
        val identityAnchorReady =
            out.getOrElse(NativeBridge.TARGET_STATE_INDEX_IDENTITY_ANCHOR_READY) { 0f } > 0.5f
        sb.appendLine("    identityAnchorReady=$identityAnchorReady")
        sb.appendLine("    identityScore=$identityScore (-1 = 无法判定，不参与门控)")
        sb.appendLine("    identityRejects=$identityRejects")
        sb.appendLine("    bboxScaleFromInitial=$bboxScaleFromInitial (上限 2.0，超了直接拒采纳)")
        sb.appendLine("    overlayPeriodNs=$targetUiPeriodNs (30Hz)")
        sb.appendLine("    centerTransform=(fx ${cameraToViewReady} via cameraToView)")
        sb.appendLine("    AF state=$lastAfState lensFocusDistance=$lastLensFocusDistance focusLocked=$targetFocusLocked relockCount=$focusRelockCount")
        sb.appendLine(NativeBridge.nativeGetTargetDiagnostics())
        sb.appendLine()
        // ---- 深度尺度诊断：只上报可观测量，不自动乘 scale ----
        // 缓冲区长度必须用常量，绝不手写数字。这里曾经写死 FloatArray(4)，
        // 而 native 协议已经扩到 7 槽：native 发现长度不足会安全返回 0，
        // Kotlin 却继续访问 dd[4]，于是点「导出反馈报告」直接
        // ArrayIndexOutOfBoundsException 退出。另外 JNI 调用本身也包一层，
        // 任何 native 侧异常都不应该让反馈流程把整个 APP 带走。
        val dd = FloatArray(NativeBridge.DEPTH_DIAGNOSTIC_SLOTS)
        val depthDiagOk = try {
            NativeBridge.nativeGetDepthDiagnostics(dd) != 0
        } catch (t: Throwable) {
            android.util.Log.e("FeedbackReport", "nativeGetDepthDiagnostics failed", t)
            false
        }
        // 一律走 getOrElse：即使 depthDiagOk 意外为 true，越界也不会崩。
        val targetDepthP10 = if (depthDiagOk) dd.getOrElse(0) { 0f } else 0f
        val targetDepthMedian = if (depthDiagOk && dd.getOrElse(1) { 0f } > 0f) {
            dd.getOrElse(1) { 0f }
        } else {
            out.getOrElse(6) { 0f }
        }
        val targetDepthP90 = if (depthDiagOk) dd.getOrElse(2) { 0f } else 0f
        val vinsTriangulatedDepthMedian =
            if (depthDiagOk) dd.getOrElse(3) { 0f } else 0f
        val targetDepthValidPixels =
            if (depthDiagOk) dd.getOrElse(4) { 0f }.toInt() else 0
        val targetDepthSampleCount =
            if (depthDiagOk) dd.getOrElse(5) { 0f }.toInt() else 0
        val targetDepthRoiArea =
            if (depthDiagOk) dd.getOrElse(6) { 0f }.toInt() else 0
        val focusDiopters = if (::hqCapture.isInitialized) {
            hqCapture.stats.focusDiopters
        } else {
            lastLensFocusDistance ?: 0f
        }
        val focusApproxMeters = if (focusDiopters > 0.01f) 1f / focusDiopters else 0f
        // 命名必须自带方向。raw/vins 与 vins/raw 互为倒数，历史上就因为把
        // depthScaleCandidateVins(= raw/vins) 和 estimator 的
        // depthScaleMedian(= vins/raw) 并列在同一份报告里，很容易把修正方向读反。
        val depthOverFocusRatio = if (focusApproxMeters > 0.01f && targetDepthMedian > 0f) {
            targetDepthMedian / focusApproxMeters
        } else {
            0f
        }
        val depthOverVinsRatio = if (vinsTriangulatedDepthMedian > 0.01f && targetDepthMedian > 0f) {
            targetDepthMedian / vinsTriangulatedDepthMedian
        } else {
            0f
        }
        // 真正会乘到 raw depth 上、把它拉到 VINS 尺度的那个系数方向
        val depthCorrectionScaleVins =
            if (depthOverVinsRatio > 0f) 1f / depthOverVinsRatio else 0f
        sb.appendLine("[7.1] 深度尺度诊断（暂不自动施加修正）:")
        sb.appendLine("    targetDepthP10=$targetDepthP10")
        sb.appendLine("    targetDepthMedian=$targetDepthMedian")
        sb.appendLine("    targetDepthP90=$targetDepthP90")
        // 这三个是「上面的中位数到底可不可信」的前提。P10=P50=P90 且
        // validPixels 只有个位数时，说明 ROI 里几乎没有有效深度样本。
        sb.appendLine("    targetDepthValidPixels=$targetDepthValidPixels")
        sb.appendLine("    targetDepthSampleCount=$targetDepthSampleCount")
        sb.appendLine("    targetDepthRoiArea=$targetDepthRoiArea")
        // 这一行必须留在报告里：下次再出现「报告里深度字段全是 0」时，
        // 一眼就能区分是 native 没给数据，还是调用本身失败了。
        sb.appendLine("    depthDiagnosticsOk=$depthDiagOk")
        sb.appendLine("    focusApproxMeters=$focusApproxMeters")
        sb.appendLine("    vinsTriangulatedDepthMedian=$vinsTriangulatedDepthMedian")
        // 三个字段名字自带方向，不会再读反：
        //   depthOverFocusRatio / depthOverVinsRatio  —— raw / 参考量（>1 说明 raw 偏大）
        //   depthCorrectionScaleVins                  —— vins / raw（<1，乘到 raw 上才变米制）
        sb.appendLine("    depthOverFocusRatio=$depthOverFocusRatio")
        sb.appendLine("    depthOverVinsRatio=$depthOverVinsRatio")
        sb.appendLine("    depthCorrectionScaleVins=$depthCorrectionScaleVins")
        // [7.2] DepthScaleEstimator 统计的就是 depthCorrectionScaleVins 方向的 scale
        // （vins / raw），只统计不施加：samples ≥ 30 且 MAD/median < 0.12 且秒级窗口
        // 无漂移才算 stable；这一轮绝不把它乘回深度/点云。
        sb.appendLine("[7.2] DepthScaleEstimator (scale 方向 = vins/raw = depthCorrectionScaleVins，只记录):")
        depthScaleEstimator.report(sb, "    ")
        sb.appendLine()
        sb.appendLine()
        sb.appendLine("[8] World-Locked Render:")
        val renderPose = FloatArray(NativeBridge.RENDER_POSE_SLOTS)
        val renderPoseValid = NativeBridge.nativeGetRenderPose(renderPose)
        sb.appendLine("    enabled=true")
        sb.appendLine("    poseSource=accepted_vins")
        sb.appendLine("    renderPoseValid=$renderPoseValid")
        sb.appendLine("    cameraT=(${renderPose[9]}, ${renderPose[10]}, ${renderPose[11]})")
        val r00 = renderPose[0]
        val r01 = renderPose[1]
        val r02 = renderPose[2]
        val r10 = renderPose[3]
        val r11 = renderPose[4]
        val r12 = renderPose[5]
        val r20 = renderPose[6]
        val r21 = renderPose[7]
        val r22 = renderPose[8]
        sb.appendLine("    cameraRight=($r00, $r10, $r20)")
        sb.appendLine("    cameraDown=($r01, $r11, $r21)")
        sb.appendLine("    cameraForward=($r02, $r12, $r22)")
        // displayRotationApplied 过去是硬编码 false —— 而那时点云确实走的是
        // 「通用透视相机」，和 Preview 的屏幕坐标系互不相干，报 false 是诚实的。
        // 现在 Renderer 直接复现 Preview 的投影链，这个字段才真正有意义：
        // 它反映 camera->view 仿射有没有算出来。
        sb.appendLine("    displayRotationApplied=${cameraToViewReady}")
        sb.appendLine("    projectionMode=intrinsics+previewAffine")
        sb.appendLine("    arDrawMode=${arDrawModeLabel()}")
        sb.appendLine("    arRenderTrigger=preview_frame_when_dirty")
        sb.appendLine("    liveMeshRefreshPeriodMs=$liveMeshRefreshPeriodMs")
        sb.appendLine("    arAccumMinHits=${renderer.accumulatedMinHits}")
        sb.appendLine("    arLiveMinHits=${NativeBridge.AR_MIN_HITS_RAW}")
        sb.appendLine("    arDrawnAccumulated=${renderer.drawnAccumulated}")
        sb.appendLine("    arDrawnTargetDebug=${renderer.drawnDebug}")
        // 点云用的是哪一时刻的 pose：true=按 Preview 时间戳查到的，
        // false=历史里没有，回退成了「此刻最新」。快速转动时这两者差别很大。
        sb.appendLine("    arPoseFromTimestamp=${renderer.poseFromTimestamp}")
        sb.appendLine("    arPreviewTimestampNs=$previewTimestampNs")
        sb.appendLine("    actualPreviewRot=$previewRot")
        sb.appendLine("    autoYaw=false")
        sb.appendLine("    touchOrbit=false")
        sb.appendLine()
        // ---- 真 TSDF / Mesh / GLB / 深度鲁棒标定 ----
        sb.appendLine("[9] 真 TSDF / Mesh / GLB / 深度标定:")
        sb.appendLine("    depthBackend=${if (::depthProvider.isInitialized) depthProvider.backendName else "n/a"}")
        sb.appendLine("    depthProviderAvailable=${if (::depthProvider.isInitialized) depthProvider.available else false}")
        sb.appendLine("    hardwareDepthProbe=${depthProbe.note}")
        multiCam?.report(sb)
        sb.appendLine("    " + depthCalibrationSummary())
        sb.appendLine("    " + fusionEpochSummary())
        sb.appendLine("    " + depthNormSummary())
        appendFusionEpochReport(sb)
        sb.appendLine("    " + meshStatsSummary("meshStats"))
        sb.appendLine("    meshSummary=$lastMeshSummary")
        sb.appendLine("    meshGpuVertices=${renderer.meshUploadedVertices}")
        sb.appendLine("    meshGpuTriangles=${renderer.meshUploadedTriangles}")
        sb.appendLine("    meshLastError=${renderer.meshLastError}")
        sb.appendLine("    arDrawnMeshTriangles=${renderer.drawnMeshTriangles}")
        sb.appendLine("    glbFile=${lastGlbFilename ?: "unknown"}")
        sb.appendLine("    glbTriangles=$lastGlbTriangles")
        sb.appendLine("    glbBytes=$lastGlbBytes")
        sb.appendLine("    voxelPlan=scene:0.020 target:0.008 truncation:0.100 (VINS-world units)")
        sb.appendLine("    meshPipeline=marching_tetrahedra+component_filter+taubin+qem")
        sb.appendLine("    glbFormat=glTF2 binary, POSITION+NORMAL+COLOR_0, uint32 indices")
        sb.appendLine()
        sb.appendLine("CameraRange: $cameraRangeStatus switching=$cameraRangeSwitching")
        sb.appendLine(NativeBridge.nativeGetStats())

        return sb.toString()
    }

    /**
     * 导出反馈报告的**安全入口**。
     *
     * 报告流程过去没有任何保护，任何一处写错（例如访问了长度不足的诊断数组）
     * 都会让用户一点「导出反馈报告」就整个 APP 退出 —— 那正是我们需要
     * 反馈报告的时刻，结果反而拿不到报告。
     *
     * 注意：`catch (Throwable)` 对 native 的 SIGSEGV 无效，但能挡住
     * ArrayIndexOutOfBounds / 空状态 / 格式化异常这类 Java/Kotlin 错误。
     */
    private fun exportFeedbackSafely() {
        val text = try {
            buildFeedbackReport()
        } catch (t: Throwable) {
            android.util.Log.e("FeedbackReport", "buildFeedbackReport crashed", t)
            pendingReport = null
            toast("反馈报告生成失败：${t.javaClass.simpleName}: ${t.message ?: "unknown"}")
            return
        }
        try {
            pendingReport = text
            createReportLauncher.launch("config_${sessionId}.txt")
        } catch (t: Throwable) {
            android.util.Log.e("FeedbackReport", "create document launcher failed", t)
            pendingReport = null
            toast("系统保存界面启动失败：${t.javaClass.simpleName}: ${t.message ?: "unknown"}")
        }
    }

    private fun processImage(image: Image) {
        if (!resumed) { image.close(); return }
        val ts = image.timestamp
        latestCameraTimestampNs = ts
        try {
            if (!((scanning && scanNativeReady) || arMeshViewing)) {
                onFrameTick(ts)
                return // finally closes the image, including the idle preview path
            }
            val p = image.planes
            val y = extractPlane(p[0])
            val u = extractPlane(p[1])
            val v = extractPlane(p[2])
            // V0.5：相机管线有两种状态 ——
            //   1) 扫描中：VINS + 目标追踪 + 相对深度推理 + TSDF 融合
            //   2) 停扫后的 AR 查看：**只**继续喂 VINS 拿位姿（否则 cameraToView
            //      不再更新，生成的网格不会跟随手机），深度与 TSDF 全部停下。
            val nativeFrameActive = (scanning && scanNativeReady) || arMeshViewing

            if (nativeFrameActive) {
                if (scanning && scanNativeReady) {
                    // 曝光直方图预检：clipping 过高时不要发 still capture
                    if (::hqCapture.isInitialized) {
                        hqCapture.onPreviewLuma(y, image.width, image.height, p[0].rowStride)
                        updateAutoFillLight()
                    }
                    val meta = synchronized(frameMetaLock) {
                        val ready = frameMeta.remove(ts)
                        if (ready == null) {
                            pendingFrameMeta.add(ts)
                            while (pendingFrameMeta.size > 96) {
                                val oldest = pendingFrameMeta.first()
                                pendingFrameMeta.remove(oldest)
                                frameMetaMiss++
                                frameMetaExpired++
                            }
                        }
                        ready
                    }
                    if (meta != null) {
                        frameMetaHit++
                    }
                }
                val vinsTs = ts
                val t0Frame = System.nanoTime()
                NativeBridge.nativeOnCameraFrame(y, u, v, image.width, image.height, p[0].rowStride, p[1].rowStride, p[1].pixelStride, ts, vinsTs)
                val frameDtMs = (System.nanoTime() - t0Frame) / 1_000_000.0
                if (frameDtMs > 16.0) {
                    android.util.Log.w("FrameProbe", String.format("nativeOnCameraFrame %.1f ms (trackedPts=%d)", frameDtMs, targetTrackedPoints))
                }
                if (scanning && scanNativeReady) {
                    scheduleDepth(y, u, v, image.width, image.height, p[0].rowStride, p[1].rowStride, p[1].pixelStride, ts)
                }
                glView.requestRender()
            }
        } catch (error: Throwable) {
            android.util.Log.e("ScanFrame", "Camera frame rejected", error)
        } finally {
            image.close()
        }
        onFrameTick(ts)
        if (!scanning && arMeshViewing) {
            pollPersistentRelocalization()
        }
    }

    /**
     * 深度推断（在专用线程上）。
     *
     * 走 [DepthProvider] 接口的「推一帧 + 取最新结果」形态，而不是直接
     * 调具体实现：硬件深度是**异步**到达的（ImageReader 回调），根本没法
     * 同步返回，包成同一形态后上层只有一条代码路径。
     *
     * `t` 必须是**这一帧相机帧的时间戳**，不是推理完成时刻 —— native 侧
     * 用它按 SENSOR_TIMESTAMP 找对应的 VINS pose 快照，用错时间戳会让
     * 深度与位姿差一帧，表现为「边走边扫时模型层层错位」。
     */
    private fun scheduleDepth(y: ByteArray, u: ByteArray, v: ByteArray, w: Int, h: Int, rowStride: Int, uRowStride: Int, uPixelStride: Int, t: Long) {
        val handler = depthHandler ?: return
        // vc159 ThermalGuard：设备发烫时拉开两次深度推理之间的最小间隔，把
        // 66MB 模型的 CPU 占空比从 ~100% 降到 30~40%。冷机时 minInterval=0，
        // 完全不影响原有节奏。（这里先于 CAS 判定，避免占用 depthBusy。）
        val minInterval = thermalDepthMinIntervalMs()
        if (!depthProvider.available) return
        if (minInterval > 0L && depthAttemptCompletedMs != 0L &&
            android.os.SystemClock.elapsedRealtime() - depthAttemptCompletedMs < minInterval) return
        if (!depthBusy.compareAndSet(false, true)) return
        val generation = depthGeneration
        try {
            if (!NativeBridge.nativeRetainDepthFrame(t)) {
                // No source pose => no possible fusion. Retry on the next camera
                // exposure without occupying the expensive inference worker.
                depthBusy.set(false)
                return
            }
            val accepted = handler.post {
                try {
                    // Queued work may belong to a scan that has already stopped.
                    if (generation != depthGeneration || !scanning || !scanNativeReady || !resumed) return@post
                    val provider = depthProvider
                    if (provider.available && provider.submitFrame(
                            y, u, v, w, h, rowStride, uRowStride, uPixelStride, t)) {
                        val res = provider.latest()
                        if (res != null && res.depth.isNotEmpty()) synchronized(depthSessionLock) {
                            if (generation == depthGeneration && scanning && scanNativeReady && resumed && !isDestroyed) {
                                NativeBridge.nativeOnDepthMapWeighted(res.depth, res.width, res.height,
                                    .5f, res.timestampNs, res.representation.nativeCode, res.confidence)
                                depthCompletedMs = android.os.SystemClock.elapsedRealtime()
                                val distanceCount = NativeBridge.nativeGetCameraDistance(res.timestampNs, cameraDistanceBuffer)
                                val distance = if (distanceCount == 4) cameraDistanceBuffer[0] else 0f
                                val source = if (distanceCount == 4) cameraDistanceBuffer[3].toInt() else 0
                                val completed = depthCompletedMs
                                cameraHandler?.post {
                                    if (generation == depthGeneration && scanning && resumed &&
                                        android.os.SystemClock.elapsedRealtime() - completed < 1_500L &&
                                        res.timestampNs > lastRangeDepthTimestampNs &&
                                        latestCameraTimestampNs - res.timestampNs in 0L..1_500_000_000L) {
                                        lastRangeDepthTimestampNs = res.timestampNs
                                        updateCameraRange(distance, source)
                                    }
                                }
                                depthLastError = ""
                                glView.requestRender()
                            }
                        }
                    }
                } catch (error: Throwable) {
                    depthErrors++
                    depthLastError = "${error.javaClass.simpleName}: ${error.message}"
                    android.util.Log.e("ScanDepth", "Depth frame failed; next frame can retry", error)
                } finally {
                    depthAttemptCompletedMs = android.os.SystemClock.elapsedRealtime()
                    depthBusy.set(false)
                }
            }
            if (!accepted) {
                depthBusy.set(false)
                depthErrors++
                depthLastError = "深度线程拒绝任务"
            }
        } catch (error: Throwable) {
            depthBusy.set(false)
            depthErrors++
            depthLastError = "${error.javaClass.simpleName}: ${error.message}"
            android.util.Log.e("ScanDepth", "Unable to schedule depth frame", error)
        }
    }

    private fun updateAutoFillLight() {
        if (!scanning || !resumed || !autoFillLight || !torchAvailable || torchFailed || torchRequested) return
        if (!::hqCapture.isInitialized || !hqCapture.canChangeFillLight()) return
        val st = hqCapture.stats
        if (autoLightPolicy.update(android.os.SystemClock.elapsedRealtime(),
                st.lumaP50, st.lumaP95, st.iso, st.exposureNs)) {
            torchRequested = true
            hqCapture.setFillLight(true)
            applyCaptureSettings()
        }
    }

    private fun resetFillLight() {
        torchRequested = false
        autoLightPolicy.reset()
        if (::hqCapture.isInitialized) hqCapture.setFillLight(false)
    }

    private fun extractPlane(plane: Image.Plane): ByteArray {
        val b = plane.buffer
        val out = ByteArray(b.remaining())
        b.get(out)
        return out
    }

    private fun toggleScan() {
        handlePrimaryScanAction()
    }

    /**
     * The primary action means "start" while idle and "finish" while a scan
     * session exists. Pause/resume is a separate reversible action.
     */
    private fun handlePrimaryScanAction() {
        if (scanning || scanPaused) enterScanReview() else runScanPreflight()
    }

    /**
     * Round 5 preflight: fail early on conditions that would make a 3D scan
     * unusable instead of letting the user discover them halfway around an object.
     */
    private fun runScanPreflight() {
        if (!ensureSystemReady()) return

        if (!recoveryPromptConsumed) {
            val recovery = com.mobilescan3d.persistence.ScanRecoveryManager.latest(applicationContext)
            if (recovery != null) {
                recoveryPromptConsumed = true
                val ageMin = ((System.currentTimeMillis() - recovery.createdAtMs).coerceAtLeast(0L) / 60000L)
                android.app.AlertDialog.Builder(this)
                    .setTitle("发现未完成扫描")
                    .setMessage(
                        "检测到约 ${ageMin} 分钟前的自动断点。\n\n" +
                            "可以恢复已融合的几何并进入 3D 检查/导出。为避免把新的 VINS " +
                            "坐标系错误融合进旧模型，恢复后不会直接继续扫描。"
                    )
                    .setNegativeButton("忽略") { _, _ -> runScanPreflight() }
                    .setNeutralButton("删除断点") { _, _ ->
                        com.mobilescan3d.persistence.ScanRecoveryManager.clear(applicationContext)
                        runScanPreflight()
                    }
                    .setPositiveButton("恢复模型") { _, _ -> restoreRecoveryCheckpoint(recovery) }
                    .show()
                return
            }
        }

        val critical = ArrayList<String>()
        val warnings = ArrayList<String>()

        if (cameraDevice == null) {
            critical.add("相机尚未就绪")
        }
        if (!::depthProvider.isInitialized || !depthProvider.available) {
            critical.add("深度链当前不可用")
        }

        val sm = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        if (sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) == null) {
            critical.add("缺少陀螺仪，无法稳定定位")
        }
        if (sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) == null) {
            critical.add("缺少加速度计，无法稳定定位")
        }

        val freeBytes = filesDir.usableSpace
        val freeMb = freeBytes / (1024L * 1024L)
        when {
            freeMb < 256L -> critical.add("可用存储仅约 ${freeMb}MB，至少需要 256MB 才能开始扫描")
            freeMb < 1024L -> warnings.add("可用存储仅约 ${freeMb}MB，长时间/HQ 扫描建议至少预留 1GB")
        }
        if (guidedScanEnabled && (!objectLockEnabled || targetConfidence < 0.35f)) {
            warnings.add("标准扫描会先引导锁定主体；开始后按右侧“追踪”并点选或拖框选择物体")
        }
        if (scanVoxelProfile == 3) {
            val p = resolvedScanVoxelProfile()
            warnings.add(
                "智能重建将使用：" +
                    when (p) {
                        0 -> "精细物体（4mm 目标体素）"
                        1 -> "快速物体（8mm 目标体素）"
                        else -> "房间/大场景（20mm 场景体素）"
                    }
            )
        }

        if (critical.isEmpty() && warnings.isEmpty()) {
            startScan()
            return
        }

        val message = buildString {
            if (critical.isNotEmpty()) {
                append("需要处理：\n")
                critical.forEach { append("• ").append(it).append('\n') }
            }
            if (warnings.isNotEmpty()) {
                if (isNotEmpty()) append('\n')
                append("建议：\n")
                warnings.forEach { append("• ").append(it).append('\n') }
            }
        }.trim()

        val builder = android.app.AlertDialog.Builder(this)
            .setTitle(if (critical.isEmpty()) "扫描前检查" else "暂时不能开始扫描")
            .setMessage(message)
            .setNegativeButton("返回", null)

        if (critical.isEmpty()) {
            builder.setPositiveButton("继续扫描") { _, _ -> startScan() }
        } else {
            builder.setPositiveButton("重新检查") { _, _ -> runScanPreflight() }
        }
        builder.show()
    }

    private fun togglePauseScan() {
        when {
            scanning -> pauseScan()
            scanPaused -> resumeScan()
        }
    }

    private fun pauseScan() {
        if (!scanning || !sessionCreated) return
        scanning = false
        scanPaused = true
        scanNativeReady = false

        // Reject any depth job already in flight. Keep VINS alive through the
        // AR-view path so the world pose does not freeze while the user pauses.
        synchronized(depthSessionLock) { depthGeneration++ }
        arMeshViewing = true
        cameraHandler?.post { resetFillLight(); applyCaptureSettings() }
        multiCam?.updateScanState(false, sessionId, false)

        pauseButton.text = "继续"
        primaryButton.text = "完成扫描"
        warningBanner.text = "扫描已暂停 · 调整位置后点“继续”，或直接完成本次扫描"
        warningBanner.visibility = android.view.View.VISIBLE
        updateHeader()
    }

    private fun resumeScan() {
        if (!scanPaused || !sessionCreated || !resumed) return
        scanPaused = false
        scanning = true
        synchronized(depthSessionLock) { depthGeneration++ }
        depthCompletedMs = android.os.SystemClock.elapsedRealtime()
        scanNativeReady = true
        arMeshViewing = false
        cameraHandler?.post { resetFillLight(); applyCaptureSettings() }
        applyScanArLayer(true)

        pauseButton.text = "暂停"
        primaryButton.text = "完成扫描"
        updateHeader()
    }

    private fun startScan() {
        if (!ensureSystemReady()) return
        if (modelOperationBusy) {
            toast("模型正在生成或恢复，请稍候")
            return
        }
        if (cameraDevice == null || captureSession == null || nativeW <= 0 || nativeH <= 0) {
            toast("请等待相机预览就绪再开始扫描")
            return
        }
        scanning = true
        scanPaused = false
        lastScanSufficiency = 0
        latestScanDistanceMeters = 0f
        latestScanDistanceMs = 0L
        targetMedianDepthMeters = 0f
        scanAngularSpeedDps = 0f
        scanReviewActive = false
        reviewInterrupted = false
        viewpointCoverage.fill(0f)
        coverageAnchorValid = false
        coverageCurrentSector = -1
        coverageCurrentElevationDeg = 0f
        coverageLastSampleMs = 0L
        lastViewpointCoveragePercent = 0
        lastGeometryQualityPercent = 0
        lastTextureQualityPercent = 0
        lastSurfaceCoveragePercent = 0
        lastSurfaceRobustPercent = 0
        lastCoverageGuidance = ""
        recoveryRound = 0
        recoveryActive = false
        recoveryStartedMs = 0L
        recoveryReadySinceMs = 0L
        reviewRecoveryPlan =
            com.mobilescan3d.scan.guidance.ScanRecoveryPlanner.Plan(emptyList(), "")
        scanGuidanceController.reset()
        if (::scanGuidanceOverlay.isInitialized) scanGuidanceOverlay.resetGuidance()
        if (guidedScanEnabled && !objectLockEnabled) {
            objectLockEnabled = true
            NativeBridge.nativeSetObjectLockEnabled(true)
            if (::targetOverlay.isInitialized) targetOverlay.state = TargetUiState(visible = true, state = 1)
        }
        if (::viewpointCoverageView.isInitialized) viewpointCoverageView.clearCoverage()
        showScanReview(false)
        depthCompletedMs = android.os.SystemClock.elapsedRealtime()
        depthErrors = 0
        depthLastError = ""
        cameraHandler?.post { resetFillLight(); applyCaptureSettings() }
        // V0.13.4 P0-C：会话令牌必须先推进 —— 从这一刻起，所有仍在飞的
        // 后台任务（上一轮的最终导出 / 恢复 / 后台建网格）回跳时都会发现
        // 令牌变了，从而丢弃自己的结果而不是污染这一轮。
        scanSessionToken++
        // V0.13.3：开新扫描必须退出模型查看器 —— 查看器的不透明背景会
        // 盖住相机预览，用户会以為相机坏了；且新扫描会清网格，查看器
        // 立刻变成空转。
        if (modelViewerActive) exitModelViewer()
        // V0.5：新一轮扫描 —— 在 native 会话就绪前不喂帧，并进入世代。
        scanNativeReady = false
        arMeshViewing = false
        depthGeneration++
        // Serialize with inference: session min/max must not leak from the previous scan.
        depthHandler?.post {
            if (::depthProvider.isInitialized) depthProvider.reset()
        }
        sessionStartTs = System.currentTimeMillis()
        // A deliberate new scan supersedes an older unfinished checkpoint.
        com.mobilescan3d.persistence.ScanRecoveryManager.clear(applicationContext)
        recoveryLastCheckpointMs = android.os.SystemClock.elapsedRealtime()
        val formatter = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
        sessionId = formatter.format(java.util.Date()) + "_" + (sessionStartTs % 100000L)
        // V0.10: preserve previous PLY metadata. plySessionMatch tells us
        // whether it belongs to this scan instead of replacing diagnostics with unknown.
        lastGlbFilename = null
        lastModelFile = null
        lastGlbTriangles = 0
        lastGlbBytes = 0L
        lastMeshSummary = "n/a"
        renderer.clearMesh()
        renderer.clearTexturedMesh()
        multiSegmentMergedActive = false
        viewerBaseMesh = null
        viewerSurfaceVertices = null
        // V0.6：新一轮扫描要清掉上一轮的 HQ 纹理关键帧登记表，否则会把
        // 上一场拍的照片烘到这一场的网格上（native 侧 nativeCreate 另有兜底）。
        try {
            NativeBridge.nativeClearTextureKeyframes()
        } catch (_: Throwable) {
        }
        if (::exportManager.isInitialized) {
            exportManager.resetMesh()
        }
        stabilization = false
        aeLock = false
        awbLock = false
        captureState = "IDLE"
        depthScaleEstimator.reset()
        lastDepthScaleSampleNs = 0L
        cameraHandler?.post {
            cameraRangePolicy.reset()
            val rangeCam = multiCam
            logCameraRange(
                "scan-start capabilities: ${rangeCam?.capabilitySnapshot() ?: "multiCam=null"}",
                force = true
            )
            if (rangeCam != null && rangeCam.active && rangeCam.selectableAuxCount < 2) {
                logCameraRange(
                    "note: only ${rangeCam.selectableAuxCount} selectable auxiliary lens -> " +
                        "distance-driven switching cannot change the aux id on this device",
                    force = true
                )
            }
        }
        if (::hqCapture.isInitialized) {
            hqCapture.beginScan(sessionId)
            // 状态机由 HqCaptureController 持有，这里只做镜像
            captureState = hqCapture.stats.captureState
        }
        applyCaptureSettings()
        // 与相机帧处理线程串行化，避免 reset 期间相机线程正在遍历这些容器（native 崩溃）
        val startToken = scanSessionToken
        cameraHandler?.post {
            synchronized(depthSessionLock) {
            if (isDestroyed || !scanning || startToken != scanSessionToken) return@post
            NativeBridge.nativeDestroy()
            applyDeviceCalibrationProfileToNative()
            if (!NativeBridge.nativeCreate(nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy)) {
                sessionCreated = false
                runOnUiThread {
                    if (startToken == scanSessionToken) {
                        scanning = false
                        scanPaused = false
                        primaryButton.text = "开始扫描"
                        pauseButton.visibility = android.view.View.GONE
                        if (::hqCapture.isInitialized) hqCapture.endScan()
                        toast("重建引擎初始化失败，请重新开始")
                    }
                }
                return@post
            }
            // 体素边长与块预算：评审 P1-2 改为按档位切换。物体扫描默认
            // OBJECT_HQ（目标 4mm / 场景 8mm），比 V0.13.3 的 0.012/0.004
            // 更细（场景 ×1.5 分辨率，面数约 ×2.25）；块预算保持一致
            // （scene 64MB + target 32MB，PLK110 可承受）。
            // 截断距离 = voxel×4（tsdf_engine 固定系数），目标侧 16mm，
            // 立体深度噪声典型 5~15mm，尚在容忍带内。
            // 注意：nativeCreate 已 reset TSDF，这里在复位后设置档位才有效。
            NativeBridge.nativeSetVoxelProfile(resolvedScanVoxelProfile())
            NativeBridge.nativeSetScanMaxDistance(scanMaxDistanceMeters)
            NativeBridge.nativeSetDepthCalibrationEnabled(true)
            if (objectLockEnabled) {
                NativeBridge.nativeSetObjectLockEnabled(true)
            }
            val backbone = materializeAsset("nanotrack_backbone_sim.onnx")
            val head = materializeAsset("nanotrack_head_sim.onnx")
            NativeBridge.nativeConfigureTrackerModels(backbone, head)
            sessionCreated = true
            // V0.5：native 侧全部配置完成，从这里起相机帧才允许进入 native。
            try {
                NativeBridge.nativeClearTexturedArAsset()
                NativeBridge.nativeClearPersistentRelocalization()
                NativeBridge.nativeSetPersistentMapCaptureEnabled(true)
            } catch (_: Throwable) {
            }
            scanNativeReady = scanning && resumed && startToken == scanSessionToken
            }
        }
        // V0.12: 扫描期默认进 LIVE 图层 —— 半透明网格 + 累计 surfel(hits>=1)
        // + 当前帧 target depth 点，三者同时显示，用户立刻能看到模型「长出来」。
        // 网格快照的节流状态必须在这里清零，否则上一场留下的
        // lastLiveMeshRefreshMs 会让新一场开头 5 秒不出网格。
        lastLiveMeshRefreshMs = 0L
        liveMeshBuildBusy = false
        liveMeshBlockedNotice = false
        applyScanArLayer(true)
        primaryButton.text = "完成扫描"
        pauseButton.visibility = android.view.View.VISIBLE
        pauseButton.text = "暂停"
        updateHeader()
    }

    /**
     * V0.12：扫描期 / 查看期的 AR 图层 + 材质切换。
     *
     * 扫描期（LIVE）要三样同时可见：半透明网格、累计 surfel（hits>=1）、
     * 当前帧 target depth 点。查看期退回「网格」档 —— 不透明 + 真实顶点色，
     * 那时要看的是几何质量，一次性点和诊断色只会干扰判断。
     */
    private fun maybeSaveRecoveryCheckpoint() {
        if (!scanning || !sessionCreated || recoveryCheckpointBusy) return
        val now = android.os.SystemClock.elapsedRealtime()
        if (recoveryLastCheckpointMs != 0L && now - recoveryLastCheckpointMs < 90_000L) return
        saveRecoveryCheckpoint(force = false)
    }

    private fun saveRecoveryCheckpoint(force: Boolean) {
        if (!sessionCreated || recoveryCheckpointBusy) return
        val handler = depthHandler ?: return
        recoveryCheckpointBusy = true
        val token = scanSessionToken
        val savedSession = sessionId
        val profile = resolvedScanVoxelProfile()
        val hadTarget = objectLockEnabled || targetConfidence > 0f
        handler.post {
            val ok = synchronized(depthSessionLock) {
                if (token != scanSessionToken || !sessionCreated) {
                    false
                } else {
                    com.mobilescan3d.persistence.ScanRecoveryManager.save(
                        applicationContext,
                        savedSession,
                        profile,
                        hadTarget
                    )
                }
            }
            recoveryLastCheckpointMs = android.os.SystemClock.elapsedRealtime()
            recoveryCheckpointBusy = false
            if (force && !ok) {
                android.util.Log.w("ScanRecovery", "checkpoint save failed")
            }
        }
    }

    /**
     * Restore an unfinished TSDF into the native mesh pipeline.
     *
     * This is intentionally a salvage path: VINS estimator state is not serialized,
     * so restored geometry opens in the independent 3D viewer for inspection/export.
     * It is NOT fused with newly initialized camera poses.
     */
    private fun restoreRecoveryCheckpoint(
        info: com.mobilescan3d.persistence.ScanRecoveryManager.Info
    ) {
        if (modelOperationBusy || scanning) return
        if (cameraDevice == null || nativeW <= 0 || nativeH <= 0) {
            toast("请等待相机初始化完成后再恢复")
            return
        }
        modelOperationBusy = true
        scanSessionToken++
        val token = scanSessionToken
        scanNativeReady = false
        scanning = false
        scanPaused = false
        renderer.clearMesh()
        renderer.clearTexturedMesh()

        cameraHandler?.post {
            val loaded = synchronized(depthSessionLock) {
                NativeBridge.nativeDestroy()
                applyDeviceCalibrationProfileToNative()
                val created = NativeBridge.nativeCreate(
                    nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy
                )
                if (!created) {
                    false
                } else {
                    NativeBridge.nativeSetVoxelProfile(info.voxelProfile)
                    com.mobilescan3d.persistence.ScanRecoveryManager.loadIntoNative(info)
                }
            }
            if (!loaded) {
                runOnUiThread {
                    modelOperationBusy = false
                    toast("未完成扫描恢复失败")
                }
                return@post
            }
            sessionCreated = true
            sessionId = info.sessionId + "_recovered"
            runOnUiThread {
                if (isDestroyed || token != scanSessionToken) {
                    modelOperationBusy = false
                    return@runOnUiThread
                }
                exportManager.buildMeshAsync(NativeBridge.MESH_QUALITY_PREVIEW) { mesh ->
                    if (isDestroyed || token != scanSessionToken) {
                        modelOperationBusy = false
                        return@buildMeshAsync
                    }
                    modelOperationBusy = false
                    if (mesh == null || mesh.triangleCount <= 0) {
                        toast("断点中没有足够几何可恢复")
                        return@buildMeshAsync
                    }
                    // Keep the checkpoint until the user explicitly starts a new scan.
                    enterRecoveredModelViewer(mesh)
                }
            }
        }
    }

    private fun enterRecoveredModelViewer(
        mesh: com.mobilescan3d.export.ExportManager.MeshData
    ) {
        renderer.clearTexturedMesh()
        renderer.setMesh(mesh.vertices, mesh.indices)
        viewerBaseMesh = mesh
        renderer.setViewerStyle(PointCloudRenderer.VIEWER_STYLE_SOLID)
        renderer.setViewerBoundaryEdges(null)
        updateViewerModeUi(PointCloudRenderer.VIEWER_STYLE_SOLID)
        analyzeViewerMesh(mesh)

        // Compute bounds for the same automatic viewer camera used by normal model view.
        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        var i = 0
        while (i + 2 < mesh.vertices.size) {
            val x = mesh.vertices[i]
            val y = mesh.vertices[i + 1]
            val z = mesh.vertices[i + 2]
            if (x.isFinite() && y.isFinite() && z.isFinite()) {
                minX = kotlin.math.min(minX, x); maxX = kotlin.math.max(maxX, x)
                minY = kotlin.math.min(minY, y); maxY = kotlin.math.max(maxY, y)
                minZ = kotlin.math.min(minZ, z); maxZ = kotlin.math.max(maxZ, z)
            }
            i += NativeBridge.MESH_VERTEX_FLOATS
        }
        if (minX <= maxX) {
            val cx = (minX + maxX) * 0.5f
            val cy = (minY + maxY) * 0.5f
            val cz = (minZ + maxZ) * 0.5f
            val radius = kotlin.math.max(
                0.03f,
                0.5f * kotlin.math.sqrt(
                    (maxX - minX) * (maxX - minX) +
                        (maxY - minY) * (maxY - minY) +
                        (maxZ - minZ) * (maxZ - minZ)
                )
            )
            glView.queueEvent { renderer.setViewerFrame(cx, cy, cz, radius) }
        }
        renderer.drawMode = PointCloudRenderer.DRAW_MODEL_VIEWER
        modelViewerActive = true
        glView.isClickable = true
        tvViewerStats.text = "${mesh.triangleCount} 面 · 断点恢复几何 · 可检查/导出"
        setScanOverlayVisible(false)
        // 与 enterModelViewerWith 一致：查看器模式 GL 画不透明深背景，必须切回非 on-top
        // 否则 setZOrderOnTop(true) 会把 modelViewerOverlay 的普通 View 按钮盖住（黑屏只见模型）。
        if (::glView.isInitialized) glView.setZOrderOnTop(false)
        glView.requestRender()
        toast("已恢复未完成几何；可检查后导出，或返回开始新扫描")
    }

    private fun applyScanArLayer(scanNow: Boolean) {
        if (scanNow) {
            arDrawMode = PointCloudRenderer.DRAW_LIVE
            renderer.drawMode = arDrawMode
            renderer.accumulatedMinHits = NativeBridge.AR_MIN_HITS_RAW
            renderer.setMeshAlpha(scanMeshAlpha)
            renderer.setMeshTint(
                scanMeshTintR, scanMeshTintG, scanMeshTintB, true
            )
            try {
                NativeBridge.nativeSetTargetDebugEnabled(true)
            } catch (_: Throwable) {
            }
        } else {
            arDrawMode = PointCloudRenderer.DRAW_MESH
            renderer.drawMode = arDrawMode
            renderer.accumulatedMinHits = NativeBridge.AR_MIN_HITS_CONFIRMED
            renderer.setMeshAlpha(viewMeshAlpha)
            renderer.setMeshTint(0f, 0f, 0f, false)
        }
        arLayerButton?.text = arDrawModeLabel()
        if (::glView.isInitialized) glView.requestRender()
    }

    /**
     * Round 3 review gate.
     *
     * "完成扫描" no longer immediately commits/export the session. It freezes
     * fusion, keeps the native reconstruction in memory, builds a preview mesh,
     * and lets the user either supplement missing views or commit the model.
     */
    private fun enterScanReview() {
        if (!sessionCreated) {
            toast("扫描仍在初始化，请稍候再检查")
            return
        }
        if (!(scanning || scanPaused)) return

        scanning = false
        scanPaused = false
        scanReviewActive = true
        scanNativeReady = false
        reviewInterrupted = false
        recoveryActive = false
        recoveryReadySinceMs = 0L
        scanGuidanceController.clearRecoveryTargets()
        synchronized(depthSessionLock) { depthGeneration++ }
        arMeshViewing = true
        cameraHandler?.post { resetFillLight(); applyCaptureSettings() }
        multiCam?.updateScanState(false, sessionId, false)

        // Review should show the reconstructed geometry, not the live translucent
        // accumulation layer. The native session itself remains intact.
        applyScanArLayer(false)
        showScanReview(true)
        updateReviewUi()
        saveRecoveryCheckpoint(force = true)
        buildReviewPreviewMesh()
    }

    private fun resumeFromScanReview() {
        if (!scanReviewActive || !sessionCreated) return
        if (reviewInterrupted) {
            toast("应用曾离开前台，定位连续性已中断；请生成当前模型或放弃本次扫描")
            return
        }

        val canUseDirectedRecovery =
            guidedScanEnabled && reviewRecoveryPlan.targets.isNotEmpty() && recoveryRound < MAX_RECOVERY_ROUNDS
        if (canUseDirectedRecovery) {
            recoveryRound++
            recoveryActive = true
            recoveryStartedMs = android.os.SystemClock.elapsedRealtime()
            recoveryReadySinceMs = 0L
            scanGuidanceController.setRecoveryTargets(reviewRecoveryPlan.targets)
        } else {
            recoveryActive = false
            recoveryStartedMs = 0L
            recoveryReadySinceMs = 0L
            scanGuidanceController.clearRecoveryTargets()
        }

        scanReviewActive = false
        scanning = true
        scanPaused = false
        synchronized(depthSessionLock) { depthGeneration++ }
        depthCompletedMs = android.os.SystemClock.elapsedRealtime()
        scanNativeReady = resumed
        arMeshViewing = false
        showScanReview(false)
        applyScanArLayer(true)
        cameraHandler?.post { resetFillLight(); applyCaptureSettings() }

        primaryButton.text = "完成扫描"
        pauseButton.visibility = android.view.View.VISIBLE
        pauseButton.text = "暂停"
        updateHeader()
        if (canUseDirectedRecovery) {
            val first = reviewRecoveryPlan.targets.first()
            toast("第 $recoveryRound 轮定向补扫：先补${first.label}")
        } else if (reviewRecoveryPlan.targets.isNotEmpty() && recoveryRound >= MAX_RECOVERY_ROUNDS) {
            toast("已完成 $MAX_RECOVERY_ROUNDS 轮定向补扫，本轮按自由扫描继续")
        }
    }

    private fun finalizeFromScanReview() {
        if (!scanReviewActive || !sessionCreated) return
        reviewContinueButton.isEnabled = false
        reviewGenerateButton.isEnabled = false
        reviewDiscardButton.isEnabled = false
        tvReviewTopHint.text = "正在生成最终模型…"
        stopScan()
    }

    private fun confirmDiscardScan() {
        if (!scanReviewActive) return
        android.app.AlertDialog.Builder(this)
            .setTitle("放弃本次扫描？")
            .setMessage("当前未生成的扫描数据会被清空。")
            .setNegativeButton("取消", null)
            .setPositiveButton("放弃") { _, _ -> discardReviewedScan() }
            .show()
    }

    private fun discardReviewedScan() {
        scanSessionToken++
        scanning = false
        scanPaused = false
        scanReviewActive = false
        scanNativeReady = false
        arMeshViewing = false
        sessionCreated = false
        synchronized(depthSessionLock) { depthGeneration++ }
        multiCam?.updateScanState(false, sessionId, false)
        if (::hqCapture.isInitialized) hqCapture.endScan()
        try {
            NativeBridge.nativeSetPersistentMapCaptureEnabled(false)
        } catch (_: Throwable) {
        }
        cameraHandler?.post {
            synchronized(depthSessionLock) {
                try {
                    NativeBridge.nativeDestroy()
                } catch (_: Throwable) {
                }
            }
        }
        if (::exportManager.isInitialized) exportManager.resetMesh()
        renderer.clearMesh()
        renderer.clearTexturedMesh()
        glView.requestRender()
        showScanReview(false)
        lastScanSufficiency = 0
        lastViewpointCoveragePercent = 0
        viewpointCoverage.fill(0f)
        viewpointCoverageView.clearCoverage()
        recoveryRound = 0
        recoveryActive = false
        recoveryStartedMs = 0L
        recoveryReadySinceMs = 0L
        reviewRecoveryPlan =
            com.mobilescan3d.scan.guidance.ScanRecoveryPlanner.Plan(emptyList(), "")
        scanGuidanceController.reset()
        if (::scanGuidanceOverlay.isInitialized) {
            scanGuidanceOverlay.resetGuidance()
            scanGuidanceOverlay.visibility = android.view.View.GONE
        }
        if (::tvGuidanceStep.isInitialized) tvGuidanceStep.visibility = android.view.View.GONE
        primaryButton.text = "开始扫描"
        pauseButton.visibility = android.view.View.GONE
        tvScanStateBadge.text = "待扫描"
        toast("已放弃本次扫描")
    }

    private fun showScanReview(show: Boolean) {
        if (!::scanReviewOverlay.isInitialized) return
        if (show && modelViewerActive) {
            exitModelViewer()
        }
        scanReviewOverlay.visibility =
            if (show) android.view.View.VISIBLE else android.view.View.GONE
        if (::mainScanUi.isInitialized) {
            mainScanUi.visibility =
                if (show) android.view.View.GONE else android.view.View.VISIBLE
        }
        if (::scanReticleView.isInitialized) {
            scanReticleView.visibility =
                if (show) android.view.View.GONE else android.view.View.VISIBLE
        }
        if (::scanGuidanceOverlay.isInitialized) {
            scanGuidanceOverlay.visibility = if (show || !guidedScanEnabled || !(scanning || scanPaused)) {
                android.view.View.GONE
            } else android.view.View.VISIBLE
        }
        if (::tvGuidanceStep.isInitialized) {
            tvGuidanceStep.visibility = if (show || !guidedScanEnabled || !(scanning || scanPaused)) {
                android.view.View.GONE
            } else android.view.View.VISIBLE
        }
        if (::targetOverlay.isInitialized) {
            targetOverlay.visibility =
                if (show) android.view.View.GONE else android.view.View.VISIBLE
        }
        if (::targetWarningText.isInitialized) {
            targetWarningText.visibility = android.view.View.GONE
        }
        if (show) {
            reviewContinueButton.isEnabled = !reviewInterrupted
            reviewGenerateButton.isEnabled = true
            reviewDiscardButton.isEnabled = true
        }
    }

    private fun buildReviewPreviewMesh() {
        if (!::exportManager.isInitialized) return
        val token = scanSessionToken
        tvReviewTopHint.text = "正在生成检查用网格…"
        try {
            exportManager.buildMeshAsync(NativeBridge.MESH_QUALITY_PREVIEW) { mesh ->
                if (isDestroyed || token != scanSessionToken || !scanReviewActive) return@buildMeshAsync
                if (mesh != null && mesh.triangleCount > 0) {
                    renderer.clearTexturedMesh()
                    val weights = exportManager.meshObservationWeights(mesh.vertexCount)
                    val heat = buildSurfaceObservationHeatmap(mesh.vertices, weights)
                    renderer.setMesh(heat ?: mesh.vertices, mesh.indices)
                    updateSurfaceObservationStats(weights)
                    arMeshViewing = true
                    arDrawMode = PointCloudRenderer.DRAW_MESH
                    renderer.drawMode = arDrawMode
                    renderer.setMeshAlpha(viewMeshAlpha)
                    glView.requestRender()
                    tvReviewTopHint.text = "表面热力已开启：优先补扫红/黄区域，再决定生成"
                } else {
                    tvReviewTopHint.text = "网格信息不足，建议继续补扫"
                }
                updateReviewUi()
            }
        } catch (t: Throwable) {
            tvReviewTopHint.text = "预览网格生成失败，可继续补扫后重试"
        }
    }

    private fun updateReviewUi() {
        if (!::scanReviewOverlay.isInitialized) return
        val capture = lastScanSufficiency.coerceIn(0, 100)
        val viewpoint = lastViewpointCoveragePercent.coerceIn(0, 100)
        val geometry = lastGeometryQualityPercent.coerceIn(0, 100)
        val textureQ = lastTextureQualityPercent.coerceIn(0, 100)
        val surface = lastSurfaceCoveragePercent.coerceIn(0, 100)
        val overall = (
            capture * 0.20f +
                viewpoint * 0.15f +
                surface * 0.35f +
                geometry * 0.20f +
                textureQ * 0.10f
            ).toInt().coerceIn(0, 100)

        tvReviewScore.text = "$overall"
        tvReviewScore.setTextColor(
            getColor(
                when {
                    overall >= 80 -> R.color.scan_success
                    overall >= 58 -> R.color.scan_warning
                    else -> R.color.scan_danger
                }
            )
        )
        val recommendation = when {
            overall >= 80 && viewpoint >= 70 -> "可生成"
            overall >= 58 -> "建议补扫"
            else -> "继续扫描"
        }
        tvReviewRecommendation.text = recommendation
        tvReviewRecommendation.setTextColor(
            getColor(
                when (recommendation) {
                    "可生成" -> R.color.scan_success
                    "建议补扫" -> R.color.scan_warning
                    else -> R.color.scan_danger
                }
            )
        )

        reviewProgressCapture.progress = capture
        reviewProgressViewpoint.progress = viewpoint
        reviewProgressSurface.progress = surface
        reviewProgressGeometry.progress = geometry
        reviewProgressTexture.progress = textureQ
        tvReviewCapture.text = "$capture%"
        tvReviewViewpoint.text = "$viewpoint%"
        tvReviewSurface.text = "$surface%"
        tvReviewGeometry.text = "$geometry%"
        tvReviewTexture.text = "$textureQ%"

        reviewRecoveryPlan = com.mobilescan3d.scan.guidance.ScanRecoveryPlanner.build(
            coverage = viewpointCoverage,
            currentSector = coverageCurrentSector,
            captureSufficiency = capture,
            surfaceCoverage = surface,
            geometryQuality = geometry,
            maxTargets = 3
        )
        val recoveryNote = when {
            reviewRecoveryPlan.targets.isEmpty() ->
                "视角补扫：没有明确短板；请重点观察红/黄表面热力区域。"
            recoveryRound >= MAX_RECOVERY_ROUNDS ->
                "${reviewRecoveryPlan.summary} · 已达到自动定向补扫建议轮数，可自由补扫或直接生成。"
            else ->
                "${reviewRecoveryPlan.summary} · 点“继续补扫”后会直接给出方向箭头。"
        }
        tvReviewMissing.text = reviewMissingViewsText() + "\n" + recoveryNote

        reviewContinueButton.text = when {
            reviewRecoveryPlan.targets.isEmpty() -> "继续补扫"
            recoveryRound >= MAX_RECOVERY_ROUNDS -> "自由补扫"
            else -> "继续补扫 · ${reviewRecoveryPlan.targets.size}处"
        }
        tvReviewTopHint.text = when {
            reviewRecoveryPlan.targets.isEmpty() ->
                "红/黄区域表示真实表面观测偏弱；未发现明确的视角补扫方向"
            recoveryRound >= MAX_RECOVERY_ROUNDS ->
                "红/黄区域表示真实表面观测偏弱；可自由补扫或生成当前模型"
            else ->
                "红/黄=真实低观测表面 · ${reviewRecoveryPlan.summary}"
        }

        reviewContinueButton.isEnabled = !reviewInterrupted
        if (reviewInterrupted) {
            reviewContinueButton.alpha = 0.45f
            tvReviewMissing.text = tvReviewMissing.text.toString() +
                "\n定位连续性已中断：不能继续补扫，但可以生成当前模型。"
        } else {
            reviewContinueButton.alpha = 1f
        }
    }

    private fun stopScan() {
        val hadReadyScan = scanNativeReady || (scanPaused && sessionCreated) ||
            (scanReviewActive && sessionCreated)
        scanning = false
        scanPaused = false
        scanReviewActive = false
        showScanReview(false)
        com.mobilescan3d.persistence.ScanRecoveryManager.clear(applicationContext)
        cameraHandler?.post { resetFillLight(); applyCaptureSettings() }
        multiCam?.updateScanState(false, sessionId, false)
        // V0.5：停扫后不再派发深度推理 / TSDF 融合（见 processImage 的
        // nativeFrameActive），但 **VINS 必须继续跑** —— cameraToView 靠它更新，
        // 网格才会钉在真实世界里。所以这里先把 arMeshViewing 打开，网格若构建
        // 失败会在回调里关回去。
        scanNativeReady = false
        try {
            NativeBridge.nativeSetPersistentMapCaptureEnabled(false)
        } catch (_: Throwable) {
        }
        synchronized(depthSessionLock) { depthGeneration++ }
        arMeshViewing = hadReadyScan
        aeLock = false
        awbLock = false
        captureState = "IDLE"
        if (::hqCapture.isInitialized) {
            hqCapture.endScan()
        }
        applyCaptureSettings()
        // V0.12: 停扫后回到「网格」档（不透明 + 真实顶点色）做几何验收。
        applyScanArLayer(false)
        primaryButton.text = "开始扫描"
        pauseButton.visibility = android.view.View.GONE
        pauseButton.text = "暂停"
        if (!hadReadyScan) {
            toast("扫描尚未就绪，本次未导出模型")
            return
        }
        exportModel()
        // 会话结束顺手产出真正的 AR 模型：带索引三角面 + 逐顶点法线 + 逐顶点
        // 颜色的 GLB。构建在 ExportManager 的后台线程上，构建期间 native 会
        // 短暂持锁（预览顿一下），这是为保证构建时体素场不被并发修改 ——
        // TsdfEngine 的块哈希表插入时会 rehash，并发读会踩空。
        buildMeshAndExport(sessionId, NativeBridge.MESH_QUALITY_NORMAL)
    }

    /**
     * 导出点云 PLY。**它不再是最终产物** ——
     *
     * 旧链路的终点就是这里，而 PLY 里只有 `element vertex`，没有
     * `element face`。在 Blender / Unity / AR 里那只是一堆孤立的点，
     * 不是模型。真正的最终产物走 [buildMeshAndExport]（GLB）。
     * PLY 保留，是因为它 still 是「点云级」的可观测中间产物。
     */
    private fun exportModel() {
        if (!::exportManager.isInitialized) return
        if (multiSegmentMergedActive) {
            toast("累计分段模型不能导出为当前单会话 PLY；请导出 GLB")
            return
        }
        val exportSession = sessionId
        val token = scanSessionToken
        exportManager.exportPlyAsync(exportSession) { r ->
            if (isDestroyed || token != scanSessionToken) return@exportPlyAsync
            if (r.ok) {
                lastPlyFilename = r.file.name
                lastPlySessionId = exportSession
                lastPlyVertexCount = r.vertexCount.toInt()
                lastPlyFileBytes = r.fileBytes
                lastPlyExportTs = System.currentTimeMillis()
            }
            toast(r.message)
        }
    }

    /**
     * 重建网格并导出 GLB（glTF 2.0，vertex color）。**新的最终产物链路**。
     *
     * 构建（Marching Tetrahedra + 去小分量 + 孤立面 + Taubin + QEM 简化）
     * 可能几百毫秒到数秒，所以整条链在后台线程，回调再回主线程挂到 Renderer。
     */
    private fun buildMeshAndExport(sessionId: String, quality: Int, shape: Int = 0) {
        if (!::exportManager.isInitialized) return
        if (scanning || modelOperationBusy || segmentMergeBusy) {
            toast("请先停止扫描，并等待当前模型操作完成")
            return
        }

        // Round 8 cumulative meshes already live in meshEngine. Rebuilding from the
        // current TSDF here would silently discard every earlier segment. Export the
        // merged native mesh as-is instead. Cross-session photo texture projection is
        // intentionally not fabricated because older keyframes are not in this VINS frame.
        if (multiSegmentMergedActive) {
            if (shape != 0) {
                toast("分段拼接模型暂不支持长方体/正方体拟合；请先导出累计 GLB")
                return
            }
            modelOperationBusy = true
            toast("正在导出累计分段 GLB…")
            try {
                exportManager.exportCurrentMeshGlbAsync(sessionId + "_segments") { r ->
                    modelOperationBusy = false
                    if (isDestroyed) return@exportCurrentMeshGlbAsync
                    val mesh = exportManager.lastMesh
                    if (r.ok && mesh != null) {
                        renderer.clearTexturedMesh()
                        renderer.setMesh(mesh.vertices, mesh.indices)
                        viewerBaseMesh = mesh
                        lastModelFile = r.file
                        lastGlbFilename = r.file.name
                        lastGlbTriangles = r.triangles
                        lastGlbBytes = r.fileBytes
                        lastMeshSummary =
                            "${r.triangles} 面 / ${r.vertices} 顶点 · ${r.fileBytes / 1024} KB · 多段累计(vertex color)"
                        generateQualityReportSilently()
                        glView.requestRender()
                    }
                    r.persistenceMessage?.let { toast(it) }
                    toast(r.message)
                }
            } catch (t: Throwable) {
                modelOperationBusy = false
                toast("累计模型导出异常：${t.javaClass.simpleName}")
            }
            return
        }

        modelOperationBusy = true
        val label = meshQualityLabel(quality)
        toast("正在重建网格（$label），请稍候…")
        try {
            exportManager.buildAndExportGlb(sessionId, quality, shape) { r ->
                modelOperationBusy = false
                if (isDestroyed || this.sessionId != sessionId) return@buildAndExportGlb
                renderer.clearTexturedMesh()
                val mesh = exportManager.lastMesh
                if (r.ok && mesh != null) {
                    renderer.setMesh(mesh.vertices, mesh.indices)

                    if (r.textured) {
                        val texturedAr = TexturedArAssetLoader.loadInto(renderer)
                        if (!texturedAr.ok) toast(texturedAr.message)
                    }
                    r.persistenceMessage?.let { toast(it) }
                    lastModelFile = r.file
                    lastGlbFilename = r.file.name
                    lastGlbTriangles = r.triangles
                    lastGlbBytes = r.fileBytes
                    lastMeshSummary =
                        "${r.triangles} 面 / ${r.vertices} 顶点 · ${r.fileBytes / 1024} KB · $label" +
                            textureNote(r.textured)
                    generateQualityReportSilently()
                    // V0.5：网格一出来就自动切到「网格」图层 —— 用户刚拍完就直接
                    // 看到结果，不必自己去翻图层按钮。
                    arMeshViewing = true
                    arDrawMode = PointCloudRenderer.DRAW_MESH
                    renderer.drawMode = arDrawMode
                    renderer.setMeshAlpha(viewMeshAlpha)
                    arLayerButton?.text = arDrawModeLabel()
                    try {
                        NativeBridge.nativeSetTargetDebugEnabled(false)
                    } catch (_: Throwable) {
                    }
                } else {
                    renderer.clearMesh()
                    // 没有网格可看，就不要再让 VINS 空转。
                    arMeshViewing = false
                    lastMeshSummary = "失败：${r.message}"
                }
                glView.requestRender()
                toast(r.message)
            }
        } catch (t: Throwable) {
            modelOperationBusy = false
            toast("网格导出异常：${t.javaClass.simpleName}")
        }
    }

    /** 只重建网格并挂到 AR 预览上，不落盘。 */
    private fun buildMeshForPreview() {
        if (!::exportManager.isInitialized || modelOperationBusy) return
        toast("正在生成网格（预览质量）…")
        val token = scanSessionToken
        try {
            exportManager.buildMeshAsync(NativeBridge.MESH_QUALITY_PREVIEW) { mesh ->
                if (isDestroyed || token != scanSessionToken) return@buildMeshAsync
                if (mesh == null || mesh.triangleCount <= 0) {
                    renderer.clearMesh()
                    lastMeshSummary = "失败：体素场不足以提取表面"
                    toast("网格构建失败（体素场不足以提取表面）")
                } else {
                    renderer.setMesh(mesh.vertices, mesh.indices)
                    lastMeshSummary = "${mesh.triangleCount} 面 / ${mesh.vertexCount} 顶点（预览）"
                    toast("网格已生成：${lastMeshSummary}")
                }
                glView.requestRender()
            }
        } catch (t: Throwable) {
            toast("网格生成异常：${t.javaClass.simpleName}")
        }
    }

    private fun meshQualityLabel(q: Int): String = when (q) {
        NativeBridge.MESH_QUALITY_PREVIEW -> "预览"
        NativeBridge.MESH_QUALITY_HQ -> "HQ"
        else -> "常规"
    }

    // ==================================================================
    // V0.13.3 模型查看器
    // ==================================================================

    /** 「查看模型」按钮入口：切换进出查看模式。 */
    private fun toggleModelViewer() {
        if (!ensureSystemReady()) return
        if (modelViewerActive) {
            exitModelViewer()
            return
        }
        val mesh = if (::exportManager.isInitialized) exportManager.lastMesh else null
        if (mesh != null && mesh.triangleCount > 0) {
            enterModelViewerWith(mesh)
            return
        }
        // 没有现成网格：先按预览质量构建一份，成功后再进入。
        if (!::exportManager.isInitialized) {
            toast("还没有可查看的模型，请先扫描")
            return
        }
        toast("正在生成模型，请稍候…")
        val token = scanSessionToken
        try {
            exportManager.buildMeshAsync(NativeBridge.MESH_QUALITY_NORMAL) { built ->
                runOnUiThread {
                    when {
                        isDestroyed || token != scanSessionToken -> toast("已开始新的扫描，取消查看")
                        built != null && built.triangleCount > 0 -> enterModelViewerWith(built)
                        else -> toast("暂无可查看的模型（体素场不足以提取表面），请先扫描")
                    }
                }
            }
        } catch (t: Throwable) {
            toast("模型构建异常：${t.javaClass.simpleName}")
        }
    }

    /** 用一份网格数据进入查看模式：算包围球 -> 自动构图 -> 切图层。 */
    private fun enterModelViewerWith(mesh: com.mobilescan3d.export.ExportManager.MeshData) {
        // 包围球：顶点交错布局 x,y,z 每 9 个 float 一组。几十万顶点的
        // 线性扫描在 UI 线程 ~10ms 量级，一次性成本可接受。
        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        val v = mesh.vertices
        var i = 0
        val n = v.size - 8
        while (i < n) {
            val x = v[i]; val y = v[i + 1]; val z = v[i + 2]
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) { i += 9; continue }
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
            i += 9
        }
        if (minX > maxX || minY > maxY || minZ > maxZ) {
            toast("模型没有有效顶点，无法查看")
            return
        }
        renderer.setMesh(mesh.vertices, mesh.indices)
        viewerBaseMesh = mesh
        viewerSurfaceVertices = null
        syncViewerOptimizationState()
        renderer.setViewerStyle(PointCloudRenderer.VIEWER_STYLE_TEXTURE)
        renderer.setViewerBoundaryEdges(null)
        updateViewerModeUi(PointCloudRenderer.VIEWER_STYLE_TEXTURE)
        analyzeViewerMesh(mesh)
        val cx = (minX + maxX) * 0.5f
        val cy = (minY + maxY) * 0.5f
        val cz = (minZ + maxZ) * 0.5f
        val radius = 0.5f * kotlin.math.sqrt(
            (maxX - minX) * (maxX - minX) +
                (maxY - minY) * (maxY - minY) +
                (maxZ - minZ) * (maxZ - minZ)
        )
        // 查看器用的是合成轨道相机，摆放变换会把模型从包围球中心挪开 ——
        // 必须清掉，否则一点「查看模型」模型就跑到画面外。
        if (placeModeActive) {
            placeModeActive = false
            placeButton?.text = "摆放"
            renderer.setModelMatrix(null)
        }
        modelViewerPrevMode = arDrawMode
        glView.queueEvent { renderer.setViewerFrame(cx, cy, cz, radius) }
        renderer.drawMode = PointCloudRenderer.DRAW_MODEL_VIEWER
        modelViewerActive = true
        glView.isClickable = true
        modelViewerButton?.text = "退出查看"
        if (::tvViewerStats.isInitialized) {
            tvViewerStats.text = "${mesh.triangleCount} 面 · ${mesh.vertexCount} 顶点 · 可旋转检查"
        }
        setScanOverlayVisible(false)
        // 确保承载 3D 的 glView 可见：glView 初始 INVISIBLE、仅相机首帧后被显示；
        // 若进入查看器前 SurfaceTexture 曾被销毁重建（切后台/权限弹窗），glView 可能
        // 仍处于 INVISIBLE，导致 3D 模型查看器空白。这里显式置为可见。
        glView.visibility = android.view.View.VISIBLE
        // 查看器模式：把 GL 层从「永远置顶」切回普通层叠。
        //
        // glView 创建时用了 setZOrderOnTop(true)，GL surface 会被合成分在所有普通 View
        // 之上。实时 AR 需要这样（透明叠加在相机上）。但查看器里 GL 画的是**不透明深色
        // 背景**（onDrawFrame 的 glClearColor alpha=1），一旦 glView 保持 VISIBLE 且置顶，
        // modelViewerOverlay 的「退出/模式/统计」按钮这些普通 View 就永远被 GL 盖住 ——
        // 表现为「只有模型、全黑、UI 看不见但功能还在」，且第 1 次进能看见、第 2/3/4 次
        // 看不见（首次靠首帧揭示的 INVISIBLE→VISIBLE 窗口期侥幸露出，相机持续出帧后
        // glView 保持置顶，之后每次进入都被盖住）。
        // 解决办法：查看器进入时切到非 on-top，让普通 View overlay 浮到 GL 之上；
        // 退出查看器再切回 setZOrderOnTop(true) 恢复 AR 的透明叠加。
        glView.setZOrderOnTop(false)
        glView.requestRender()
        toast("单指旋转 · 双指缩放/平移")
    }

    private fun syncViewerOptimizationState() {
        val stats = if (::exportManager.isInitialized) exportManager.meshOptimizationStats() else null
        viewerMeshOptimized = stats?.optimized == true
        if (::viewerOptimizeLabel.isInitialized) {
            viewerOptimizeLabel.text = if (viewerMeshOptimized) "撤销" else "优化"
            viewerOptimizeLabel.setTextColor(
                getColor(if (viewerMeshOptimized) R.color.scan_warning else R.color.scan_primary)
            )
        }
        if (::viewerOptimizeButton.isInitialized) {
            viewerOptimizeButton.setBackgroundResource(
                if (viewerMeshOptimized) R.drawable.bg_tool_item else R.drawable.bg_tool_active
            )
            viewerOptimizeButton.contentDescription =
                if (viewerMeshOptimized) "撤销模型优化" else "一键优化模型"
        }
    }

    private fun toggleModelOptimization() {
        if (!modelViewerActive || modelOperationBusy || segmentMergeBusy) {
            toast("请先进入稳定的 3D 模型查看状态")
            return
        }
        if (viewerMeshOptimized) {
            undoModelOptimization()
        } else {
            runModelOptimization()
        }
    }

    private fun runModelOptimization() {
        val current = viewerBaseMesh ?: exportManager.lastMesh
        if (current == null || current.triangleCount <= 0) {
            toast("当前没有可优化的网格")
            return
        }

        modelOperationBusy = true
        viewerOptimizeButton.isEnabled = false
        tvViewerStats.text = "正在优化 · 去浮点 / 去孤岛 / 去桌面 / 补小孔 / 简化…"

        exportManager.optimizeCurrentMeshAsync(
            profile = NativeBridge.MESH_OPT_PROFILE_STANDARD,
            removeSupportPlane = true
        ) { result ->
            modelOperationBusy = false
            viewerOptimizeButton.isEnabled = true
            if (!modelViewerActive) return@optimizeCurrentMeshAsync

            val stats = result.stats
            if (!result.ok || result.mesh == null) {
                syncViewerOptimizationState()
                val message = when (stats?.status) {
                    NativeBridge.MESH_OPT_STATUS_HARD_SURFACE_SKIPPED ->
                        "当前是平面/立方体等硬表面拟合模型。自动焊接和简化可能破坏硬边，因此本次未修改模型。"
                    NativeBridge.MESH_OPT_STATUS_GUARD_REJECTED ->
                        "安全门判断优化结果删除了过多几何，因此已放弃结果，原模型保持不变。"
                    NativeBridge.MESH_OPT_STATUS_ALREADY_OPTIMIZED ->
                        "当前模型已经优化，可先点“撤销”恢复原模型。"
                    else ->
                        "优化未完成，原模型保持不变。"
                }
                tvViewerStats.text = "${current.triangleCount} 面 · 原模型未修改"
                android.app.AlertDialog.Builder(this)
                    .setTitle("模型优化未应用")
                    .setMessage(message)
                    .setPositiveButton("关闭", null)
                    .show()
                return@optimizeCurrentMeshAsync
            }

            renderer.clearTexturedMesh()
            viewerMeshOptimized = true
            enterModelViewerWith(result.mesh)
            viewerMeshOptimized = true
            syncViewerOptimizationState()

            val s = stats
            if (s != null) {
                val tableLine = if (s.supportPlaneDetected) {
                    "去桌面/支撑平面：移除 ${s.supportPlaneTrianglesRemoved} 面"
                } else {
                    "去桌面/支撑平面：未检测到可安全删除的区域"
                }
                val reduction = if (s.inputTriangles > 0) {
                    ((s.inputTriangles - s.outputTriangles).coerceAtLeast(0) * 100 / s.inputTriangles)
                } else 0
                android.app.AlertDialog.Builder(this)
                    .setTitle("模型优化完成")
                    .setMessage(
                        "三角面：${s.inputTriangles} → ${s.outputTriangles}（减少 ${reduction}%）\n" +
                            "去孤岛：${s.removedComponents} 个 / ${s.removedComponentTriangles} 面\n" +
                            "$tableLine\n" +
                            "小孔修复：${s.filledHoles} 个（新增 ${s.addedHoleTriangles} 面）\n" +
                            "网格简化：塌缩 ${s.qemCollapsedEdges} 条边\n" +
                            "耗时：${s.elapsedMs} ms\n\n" +
                            "原模型仍保留，可随时点右侧“撤销”。"
                    )
                    .setPositiveButton("完成", null)
                    .show()
            } else {
                toast("模型优化完成 · 可随时撤销")
            }
        }
    }

    private fun undoModelOptimization() {
        modelOperationBusy = true
        viewerOptimizeButton.isEnabled = false
        tvViewerStats.text = "正在恢复优化前模型…"
        exportManager.undoMeshOptimizationAsync { result ->
            modelOperationBusy = false
            viewerOptimizeButton.isEnabled = true
            if (!modelViewerActive) return@undoMeshOptimizationAsync

            if (!result.ok || result.mesh == null) {
                syncViewerOptimizationState()
                toast("没有可恢复的优化前模型")
                return@undoMeshOptimizationAsync
            }

            renderer.clearTexturedMesh()
            viewerMeshOptimized = false
            enterModelViewerWith(result.mesh)
            viewerMeshOptimized = false
            syncViewerOptimizationState()
            toast("已恢复优化前的原始模型")
        }
    }

    private fun setViewerPresentation(style: Int) {
        if (!modelViewerActive) return
        if (style == PointCloudRenderer.VIEWER_STYLE_SURFACE && multiSegmentMergedActive) {
            toast("累计拼接模型来自多个独立 TSDF；观测热力只对单会话网格有定义")
            return
        }
        val base = viewerBaseMesh
        if (base != null) {
            if (style == PointCloudRenderer.VIEWER_STYLE_SURFACE && viewerSurfaceVertices != null) {
                renderer.setMesh(viewerSurfaceVertices, base.indices)
            } else {
                renderer.setMesh(base.vertices, base.indices)
            }
        }
        renderer.setViewerStyle(style)
        updateViewerModeUi(style)
        glView.requestRender()
        when (style) {
            PointCloudRenderer.VIEWER_STYLE_INSPECT ->
                toast("红色边线 = 网格真实开放边界；不一定都是错误，也可能是有意保留的开口")
            PointCloudRenderer.VIEWER_STYLE_SURFACE ->
                toast("表面观测热力来自 TSDF 真实融合权重：红弱、黄可用、绿稳定")
        }
    }

    private fun updateViewerModeUi(style: Int) {
        if (!::viewerModeTexture.isInitialized) return
        val items = listOf(
            viewerModeTexture to PointCloudRenderer.VIEWER_STYLE_TEXTURE,
            viewerModeSolid to PointCloudRenderer.VIEWER_STYLE_SOLID,
            viewerModeWire to PointCloudRenderer.VIEWER_STYLE_WIREFRAME,
            viewerModeInspect to PointCloudRenderer.VIEWER_STYLE_INSPECT,
            viewerModeSurface to PointCloudRenderer.VIEWER_STYLE_SURFACE
        )
        items.forEach { (view, mode) ->
            if (mode == style) {
                view.setBackgroundResource(R.drawable.bg_viewer_mode_active)
                view.setTextColor(getColor(R.color.scan_text_primary))
                view.setTypeface(null, android.graphics.Typeface.BOLD)
            } else {
                view.background = null
                view.setTextColor(getColor(R.color.scan_text_secondary))
                view.setTypeface(null, android.graphics.Typeface.NORMAL)
            }
        }
    }

    private fun analyzeViewerMesh(mesh: com.mobilescan3d.export.ExportManager.MeshData) {
        val token = ++viewerAnalysisToken
        tvViewerDimensions.text = "尺寸：分析中"
        tvViewerTopology.text = "拓扑：分析中"

        Thread {
            val result = try {
                com.mobilescan3d.mesh.MeshInspectionAnalyzer.analyze(
                    mesh.vertices, mesh.indices
                )
            } catch (_: Throwable) {
                null
            }
            val observationWeights = if (multiSegmentMergedActive) {
                // Earlier segments no longer share the current TSDF volume, so a
                // per-vertex TSDF heatmap would be physically misleading.
                FloatArray(0)
            } else try {
                exportManager.meshObservationWeights(mesh.vertexCount)
            } catch (_: Throwable) {
                FloatArray(0)
            }
            val observationHeat = buildSurfaceObservationHeatmap(
                mesh.vertices, observationWeights
            )
            runOnUiThread {
                if (isDestroyed || token != viewerAnalysisToken || !modelViewerActive) return@runOnUiThread
                if (result == null) {
                    tvViewerDimensions.text = "尺寸：分析失败"
                    tvViewerTopology.text = "拓扑：暂不可用"
                    return@runOnUiThread
                }

                fun fmtMeters(v: Float): String =
                    if (v < 1f) "${(v * 100f).toInt()}cm" else "%.2fm".format(v)

                tvViewerDimensions.text =
                    "尺寸：${fmtMeters(result.width)} × ${fmtMeters(result.height)} × ${fmtMeters(result.depth)}"

                tvViewerTopology.text = when {
                    !result.topologyComplete ->
                        "拓扑：模型过大，跳过开放边精确分析"
                    result.boundaryEdgeCount == 0 && result.nonManifoldEdgeCount == 0 ->
                        "拓扑：未检测到开放边"
                    else ->
                        "拓扑：${result.boundaryEdgeCount} 条开放边 · ${result.nonManifoldEdgeCount} 处非流形"
                }

                renderer.setViewerBoundaryEdges(
                    if (result.topologyComplete) result.boundaryEdges else null
                )
                viewerSurfaceVertices = observationHeat
                updateSurfaceObservationStats(observationWeights)
                if (observationWeights.isNotEmpty()) {
                    tvViewerTopology.text = tvViewerTopology.text.toString() +
                        " · 表面稳定 ${lastSurfaceCoveragePercent}%"
                }
                glView.requestRender()
            }
        }.start()
    }

    /** 退出查看模式：恢复 AR 图层与触摸穿透。 */
    private fun exitModelViewer() {
        viewerAnalysisToken++
        renderer.setViewerBoundaryEdges(null)
        viewerBaseMesh = null
        viewerSurfaceVertices = null
        cancelViewerLongPress()
        viewerPanMode = false
        viewerMoved = false
        modelViewerActive = false
        glView.isClickable = false
        renderer.drawMode = arDrawMode
        modelViewerButton?.text = "查看模型"
        setScanOverlayVisible(true)
        // 退出查看器恢复 AR 层叠：实时 AR 的 GL 透明叠加需要 setZOrderOnTop(true)
        // 才能盖在 TextureView 相机画面之上（进入查看器时被我们切回了 false）。
        glView.setZOrderOnTop(true)
        glView.requestRender()
    }

    /**
     * Dedicated viewer composition:
     * - scan controls disappear as one unit
     * - camera texture is hidden
     * - neutral 3D grid backdrop is revealed
     * - GL remains interactive above it
     * - viewer controls are the only active UI
     */
    private fun setScanOverlayVisible(visible: Boolean) {
        if (::mainScanUi.isInitialized) {
            mainScanUi.visibility =
                if (visible) android.view.View.VISIBLE else android.view.View.GONE
        }
        if (::modelViewerOverlay.isInitialized) {
            modelViewerOverlay.visibility =
                if (visible) android.view.View.GONE else android.view.View.VISIBLE
            if (!visible) {
                // GLSurfaceView owns a separate Surface. Reassert the Android control layer
                // whenever the viewer opens so its exit action cannot vanish behind GL.
                modelViewerOverlay.bringToFront()
            }
        }
        if (::viewerBackdrop.isInitialized) {
            viewerBackdrop.visibility =
                if (visible) android.view.View.GONE else android.view.View.VISIBLE
        }
        if (::texture.isInitialized) {
            texture.visibility =
                if (visible) android.view.View.VISIBLE else android.view.View.INVISIBLE
        }
        if (::scanReticleView.isInitialized) {
            scanReticleView.visibility =
                if (visible) android.view.View.VISIBLE else android.view.View.GONE
        }
        if (::targetOverlay.isInitialized) {
            targetOverlay.visibility =
                if (visible) android.view.View.VISIBLE else android.view.View.GONE
        }
        if (::targetWarningText.isInitialized && !visible) {
            targetWarningText.visibility = android.view.View.GONE
        }
    }

    /** VINS world uses +Z up: yaw must preserve height, with the center at the anchor. */
    private fun computePlacementMatrix(): FloatArray =
        ScanCoordinates.placementMatrix(placeYaw, placeScale, placeCenter, placeAnchor)

    private fun applyPlacementMatrix() {
        renderer.setModelMatrix(if (placeModeActive) computePlacementMatrix() else null)
        glView.requestRender()
    }

    /** 「摆放」按钮：把模型放到当前相机前方（新地点也能放）。 */
    private fun togglePlacement() {
        if (modelViewerActive) exitModelViewer()
        if (placeModeActive) {
            // 回到原位恢复：清掉 model-to-world，模型回到保存时的世界坐标。
            placeModeActive = false
            placeButton?.text = "摆放"
            renderer.setModelMatrix(null)
            glView.isClickable = false
            glView.requestRender()
            toast("已回到原位恢复模式（模型回到原扫描位置）")
            return
        }
        val mesh = if (::exportManager.isInitialized) exportManager.lastMesh else null
        if (mesh == null || mesh.triangleCount <= 0) {
            toast("还没有模型可摆放，请先扫描或恢复扫描包")
            return
        }
        // 锚点 = 当前相机前方 1.2m、略低于视线（桌面高度感）。
        val pose = FloatArray(NativeBridge.RENDER_POSE_SLOTS)
        val ok = try {
            NativeBridge.nativeGetRenderPose(pose)
        } catch (_: Throwable) {
            false
        }
        if (!ok) {
            toast("还没有定位位姿，请缓慢移动手机建立定位后再摆放")
            return
        }
        // 模型中心（顶点交错 9 floats：x,y,z,nx,ny,nz,r,g,b）
        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        val v = mesh.vertices
        var i = 0
        val n = v.size - 8
        while (i < n) {
            val x = v[i]; val y = v[i + 1]; val z = v[i + 2]
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) { i += 9; continue }
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
            i += 9
        }
        placeCenter[0] = (minX + maxX) * 0.5f
        placeCenter[1] = (minY + maxY) * 0.5f
        placeCenter[2] = (minZ + maxZ) * 0.5f
        // pose: [R(9) 行主序, t(3)]，前向是 R 的第三列 (pose[2],pose[5],pose[8])
        ScanCoordinates.placementAnchor(pose, 1.2f, 0.20f, placeAnchor)
        placeYaw = 0f
        placeScale = 1f
        placeModeActive = true
        placeButton?.text = "原位"
        glView.isClickable = true
        applyPlacementMatrix()
        toast("自由摆放：左右拖动=旋转，上下拖动=缩放 · 「原位」回到原扫描位置")
    }

    private fun cancelViewerLongPress() {
        viewerLongPressRunnable?.let { viewerHandler.removeCallbacks(it) }
        viewerLongPressRunnable = null
    }

    /**
     * 查看模式触摸手势：
     *  - 按下即拖 -> 旋转
     *  - 按住 300ms 未动 -> 震动一下进入平移，之后拖动 = 物体跟手平移
     */
    private fun handleModelViewerTouch(event: android.view.MotionEvent) {
        viewerScaleDetector.onTouchEvent(event)

        fun midpointX(): Float =
            if (event.pointerCount >= 2) (event.getX(0) + event.getX(1)) * 0.5f else event.x
        fun midpointY(): Float =
            if (event.pointerCount >= 2) (event.getY(0) + event.getY(1)) * 0.5f else event.y

        when (event.actionMasked) {
            android.view.MotionEvent.ACTION_DOWN -> {
                viewerMultiTouch = false
                viewerLastX = event.x
                viewerLastY = event.y
                viewerDownX = event.x
                viewerDownY = event.y
                viewerMoved = false
                cancelViewerLongPress()
            }

            android.view.MotionEvent.ACTION_POINTER_DOWN -> {
                if (event.pointerCount >= 2) {
                    viewerMultiTouch = true
                    viewerLastMidX = midpointX()
                    viewerLastMidY = midpointY()
                    cancelViewerLongPress()
                }
            }

            android.view.MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount >= 2 || viewerMultiTouch) {
                    if (event.pointerCount >= 2) {
                        val mx = midpointX()
                        val my = midpointY()
                        val dx = mx - viewerLastMidX
                        val dy = my - viewerLastMidY
                        viewerLastMidX = mx
                        viewerLastMidY = my
                        // ScaleGestureDetector handles pinch distance; midpoint delta
                        // handles the standard two-finger pan gesture.
                        if (dx != 0f || dy != 0f) {
                            glView.queueEvent { renderer.panViewer(dx, dy) }
                            glView.requestRender()
                        }
                    }
                    return
                }

                if (!viewerScaleDetector.isInProgress) {
                    val dx = event.x - viewerLastX
                    val dy = event.y - viewerLastY
                    viewerLastX = event.x
                    viewerLastY = event.y
                    if (dx != 0f || dy != 0f) {
                        glView.queueEvent {
                            renderer.rotateViewer(
                                dx * viewerRotRadPerPx,
                                dy * viewerRotRadPerPx
                            )
                        }
                        glView.requestRender()
                    }
                }
            }

            android.view.MotionEvent.ACTION_POINTER_UP -> {
                // Keep multi-touch latched until all fingers lift. This avoids the
                // classic pointer-index jump from unexpectedly rotating the model.
                viewerMultiTouch = true
            }

            android.view.MotionEvent.ACTION_UP,
            android.view.MotionEvent.ACTION_CANCEL -> {
                viewerMultiTouch = false
                viewerMoved = false
                cancelViewerLongPress()
            }
        }
    }

    /**
     * V0.13.4 P1：自由摆放模式手势 —— 单指左右拖 = 绕 Y 轴旋转，
     * 单指上下拖 = 均匀缩放（0.3x ~ 4x）。
     *
     * 刻意不做「双指缩放」：这套 AR 画面上单指已经被点选目标占用，
     * 摆放模式下接管单指最省事，也不会和相机对焦抢事件。
     */
    private fun handlePlacementTouch(event: android.view.MotionEvent) {
        when (event.actionMasked) {
            android.view.MotionEvent.ACTION_DOWN -> {
                placeLastX = event.x
                placeLastY = event.y
            }
            android.view.MotionEvent.ACTION_MOVE -> {
                val dx = event.x - placeLastX
                val dy = event.y - placeLastY
                placeLastX = event.x
                placeLastY = event.y
                if (dx != 0f || dy != 0f) {
                    placeYaw += dx * 0.006f
                    // 向上拖放大：屏幕上 dy 为负，取反号。
                    placeScale = (placeScale * (1f - dy * 0.004f)).coerceIn(0.3f, 4f)
                    applyPlacementMatrix()
                }
            }
        }
    }

    /**
     * 深度标定一行摘要（HUD / 报告共用）。
     *
     * **`usable` 才是「可以当米制用」的判据**，`valid` 只是「这一帧拟合成功」。
     * 把两者混为一谈会让人以为标定已经在生效，而实际上尺度并没有被施加。
     */
    /**
     * 折叠 HUD 的尺度标签。旧实现是一句硬编码「非米制」——不管标定是否
     * 生效永远显示「非米制」，和「定位失锁」是同一类静态文案冒充动态状态。
     * 真实判据与 depthCalibrationSummary 一致：usable > 0.5 才算米制。
     */
    private fun metricLabel(): String {
        return try {
            val n = NativeBridge.nativeGetDepthCalibration(calibBuf)
            if (n >= NativeBridge.DEPTH_CALIBRATION_SLOTS &&
                calibBuf[NativeBridge.CALIB_INDEX_USABLE] > 0.5f
            ) "米制" else "非米制"
        } catch (t: Throwable) {
            "非米制"
        }
    }

    private fun depthCalibrationSummary(): String {
        val n = try {
            NativeBridge.nativeGetDepthCalibration(calibBuf)
        } catch (t: Throwable) {
            0
        }
        if (n < NativeBridge.DEPTH_CALIBRATION_SLOTS) return "深度标定 n/a"
        val usable = calibBuf[NativeBridge.CALIB_INDEX_USABLE] > 0.5f
        val model = if (calibBuf[NativeBridge.CALIB_INDEX_INVERSE_MODEL] > 0.5f) "逆深度" else "线性"
        fun f(i: Int, fmt: String): String =
            String.format(java.util.Locale.US, fmt, calibBuf[i])
        return "深度标定${if (usable) "已生效" else "未生效"} · $model" +
            " scale=${f(NativeBridge.CALIB_INDEX_SCALE, "%.4f")}" +
            " shift=${f(NativeBridge.CALIB_INDEX_SHIFT, "%.4f")}" +
            " conf=${f(NativeBridge.CALIB_INDEX_CONFIDENCE, "%.2f")}" +
            " 样本=${calibBuf[NativeBridge.CALIB_INDEX_SAMPLES].toInt()}" +
            " 时序=${f(NativeBridge.CALIB_INDEX_TEMPORAL_RATIO, "%.2f")}" +
            " 拒绝=${calibBuf[NativeBridge.CALIB_INDEX_REJECTED_FRAMES].toInt()}"
    }

    /**
     * V0.12: Fusion Epoch 一行摘要（HUD / 报告共用）。
     *
     *   ACTIVE  = scale/shift/inverse 已冻结，几何正在按它累计
     *   WARMUP  = 还在等连续若干好帧，此时**禁止**累计任何几何
     *
     * 「WARMUP 期间屏幕上是空的」是**预期行为**：宁可先什么都不画，也不能
     * 拿一套还没稳的尺度往体素场里灌 —— 那正是「覆盖大半屏的黑壳」与
     * 「双层 mesh」的来源。
     */
    private fun fusionEpochSummary(): String {
        val n = try {
            NativeBridge.nativeGetFusionEpochStats(fusionEpochBuf)
        } catch (_: Throwable) {
            0
        }
        if (n < NativeBridge.FUSION_EPOCH_STATS_SLOTS) return "融合Epoch n/a"
        val active = fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_ACTIVE] > 0.5f
        // V0.13：PAUSED 是**新状态**，必须和 WARMUP 区分开 ——
        // WARMUP 是「还没选好尺度，先不建」，PAUSED 是「尺度已经冻结、
        // 只是当前几帧的在线标定对不上，所以先停一停」。两者的处置完全
        // 不同（前者等，后者等它自己恢复），合成一个词就再也排查不出来了。
        val suspended = fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SUSPENDED] > 0.5f
        val epochLabel = when {
            suspended -> "PAUSED"
            active -> "ACTIVE"
            else -> "WARMUP"
        }
        return "融合Epoch $epochLabel" +
            " #${fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SERIAL].toInt()}" +
            " good=${fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_GOOD_STREAK].toInt()}" +
            " bad=${fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_BAD_STREAK].toInt()}" +
            " skip=${fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_WARMUP_SKIPPED].toLong()}" +
            " restart=${fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_RESTARTS].toLong()}" +
            " drift=" + String.format(
                java.util.Locale.US, "%.3f",
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_LAST_DRIFT_REL]
            )
    }

    /**
     * V0.12: 把 Fusion Epoch 的完整状态写进 feedback 报告。
     *
     * 排查「为什么屏幕一直空着」时先看这里：
     *   active=false 且 goodStreak 涨不上去  -> 标定 / 深度链没通；
     *   active=true  但 fusedFrames=0        -> epoch 开了却没有深度进来；
     *   restarts 反复增长                    -> 尺度在漂（lastDriftRel 给幅度）。
     */
    private fun appendFusionEpochReport(sb: StringBuilder) {
        val n = try {
            NativeBridge.nativeGetFusionEpochStats(fusionEpochBuf)
        } catch (_: Throwable) {
            0
        }
        sb.appendLine()
        sb.appendLine("[FUSION EPOCH V0.12]")
        if (n < NativeBridge.FUSION_EPOCH_STATS_SLOTS) {
            sb.appendLine("available=false")
            return
        }
        sb.appendLine("available=true")
        sb.appendLine(
            "active=" + (fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_ACTIVE] > 0.5f)
        )
        sb.appendLine(
            "serial=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SERIAL].toLong()
        )
        sb.appendLine(
            "goodStreak=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_GOOD_STREAK].toInt()
        )
        sb.appendLine(
            "badStreak=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_BAD_STREAK].toInt()
        )
        sb.appendLine(
            "warmupSkippedFrames=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_WARMUP_SKIPPED].toLong()
        )
        sb.appendLine(
            "fusedFrames=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_FUSED_FRAMES].toLong()
        )
        sb.appendLine(
            "restarts=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_RESTARTS].toLong()
        )
        sb.appendLine(
            "driftRejects=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_DRIFT_REJECTS].toLong()
        )
        sb.appendLine(
            "lastDriftRel=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_LAST_DRIFT_REL]
        )
        sb.appendLine(
            "frozenScale=" + fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SCALE]
        )
        sb.appendLine(
            "frozenShift=" + fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SHIFT]
        )
        sb.appendLine(
            "frozenConfidence=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_CONFIDENCE]
        )
        sb.appendLine(
            "frozenSamples=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SAMPLES].toInt()
        )
        sb.appendLine(
            "frozenInverseModel=" +
                (fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_INVERSE] > 0.5f)
        )
        sb.appendLine(
            "currentCalibrationUsable=" +
                (fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_CURRENT_USABLE] > 0.5f)
        )
        sb.appendLine(
            "startGoodFramesRequired=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_START_FRAMES].toInt()
        )
        // ---- V0.13 Sticky Fusion Epoch ----
        // 读法：suspended=true 但 restarts 不再增长 = sticky 在正常工作
        // （暂停融合而不是清几何）。catastrophicRebuilds 长期为 0 才是健康。
        sb.appendLine(
            "stickyMode=true (drift suspends fusion, never clears geometry)"
        )
        sb.appendLine(
            "suspended=" +
                (fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SUSPENDED] > 0.5f)
        )
        sb.appendLine(
            "suspendEvents=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SUSPEND_EVENTS].toLong()
        )
        sb.appendLine(
            "suspendedFrames=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_SUSPENDED_FRAMES].toLong()
        )
        sb.appendLine(
            "catastrophicRebuilds=" +
                fusionEpochBuf[NativeBridge.FUSION_EPOCH_INDEX_CATASTROPHIC_REBUILDS].toLong()
        )
    }

    /** Fusion Epoch 是否已激活（网格快照的门控；只从相机回调线程调用）。 */
    private fun fusionEpochActive(): Boolean {
        val n = try {
            NativeBridge.nativeGetFusionEpochStats(epochProbeBuf)
        } catch (_: Throwable) {
            0
        }
        return n >= NativeBridge.FUSION_EPOCH_STATS_SLOTS &&
            epochProbeBuf[NativeBridge.FUSION_EPOCH_INDEX_ACTIVE] > 0.5f
    }

    /**
     * V0.13.4：深度数值域（归一化映射）一行摘要。
     *
     * 「尺度漂移」排查的第一步就是看这几个数：重参数化次数在涨说明数值域
     * 一直在变（这是允许的，只要标定跟着重参数化了）；而
     * 「冻结期间仍在变」> 0 说明 epoch 锁定后 provider 没冻住，需要查冻结
     * 链路。版本号用于把任何一次尺度变化追溯到具体映射版本。
     */
    private fun depthNormSummary(): String {
        val buf = FloatArray(NativeBridge.DEPTH_NORM_STATS_SLOTS)
        val n = try {
            NativeBridge.nativeGetDepthNormStats(buf)
        } catch (_: Throwable) {
            0
        }
        if (n < NativeBridge.DEPTH_NORM_STATS_SLOTS) return "depthNorm n/a"
        val rep = if (::depthProvider.isInitialized &&
            depthProvider is com.mobilescan3d.depth.MonoDepthProvider
        ) {
            "INVERSE_DEPTH"
        } else {
            "n/a"
        }
        return "depthNorm rep=$rep A=${"%.3f".format(buf[0])} " +
            "B=${"%.3f".format(buf[1])} reparam=${buf[3].toInt()} " +
            "chgWhileEpoch=${buf[4].toInt()}"
    }

    /** V0.13.4：UI 线程版 epoch 激活查询（用独立缓冲，避开相机线程争用）。 */
    private fun fusionEpochActiveUi(): Boolean {
        val n = try {
            NativeBridge.nativeGetFusionEpochStats(epochUiProbeBuf)
        } catch (_: Throwable) {
            0
        }
        return n >= NativeBridge.FUSION_EPOCH_STATS_SLOTS &&
            epochUiProbeBuf[NativeBridge.FUSION_EPOCH_INDEX_ACTIVE] > 0.5f
    }

    /**
     * V0.12: 扫描期的「实时网格」快照。
     *
     * 只在 LIVE 图层 + 正在扫描 + epoch 已激活时做；否则**清掉**网格 ——
     * 宁可空着，也不能拿上一套尺度的旧网格冒充「现在的形状」。
     *
     * `buildMeshAsync` 在 ExportManager 的后台线程上构建、回调回到主线程，
     * 所以本函数自身绝不阻塞相机帧回调。**绝不能**改成按相机帧率重建。
     */
    private fun maybeRefreshLiveMesh() {
        if (!scanning ||
            !scanNativeReady ||
            arDrawMode != PointCloudRenderer.DRAW_LIVE ||
            liveMeshBuildBusy ||
            !::exportManager.isInitialized ||
            !::glView.isInitialized
        ) return

        // V0.13.20：采集在途时让出重建。构建会持 native 全局锁（实测 86ms→1186ms），
        // 落在采集窗口内会丢相机帧 → VINS 位姿历史空洞 → burst 帧取不到位姿，
        // 纹理关键帧被静默丢弃。宁可这一段少几次网格刷新，也不能丢纹理。
        if (::hqCapture.isInitialized && hqCapture.captureBusy()) return

        val now = android.os.SystemClock.elapsedRealtime()
        // V0.13.20 自适应退避：上一次构建越慢，下一次间隔越长（5s..30s）。
        // 固定的 5s 节拍在大体素场下会变成「每 5s 冻结相机 ~1s」。
        // vc159 ThermalGuard：发烫时再乘一个热状态系数（×1/×2/×4），封顶 60s，
        // 减少「全核 QEM 满载 + 冻结相机」的次数。
        val interval = ((liveMeshLastBuildMs * 15L)
            .coerceIn(liveMeshRefreshPeriodMs, liveMeshRefreshPeriodMaxMs) *
            thermalMeshIntervalScale()).coerceAtMost(60_000L)
        if (now - lastLiveMeshRefreshMs < interval) return

        // 不要按相机帧率去碰 native 状态，只在网格节拍上问一次。
        if (!fusionEpochActive()) {
            if (!liveMeshBlockedNotice) {
                liveMeshBlockedNotice = true
                renderer.clearMesh()
                lastMeshSummary = "等待 Fusion Epoch"
                glView.requestRender()
            }
            return
        }
        liveMeshBlockedNotice = false

        lastLiveMeshRefreshMs = now
        liveMeshBuildBusy = true
        val token = scanSessionToken
        try {
            exportManager.buildMeshAsync(NativeBridge.MESH_QUALITY_PREVIEW) { mesh ->
                if (isDestroyed || token != scanSessionToken) return@buildMeshAsync
                try {
                    if (scanning && arDrawMode == PointCloudRenderer.DRAW_LIVE) {
                        if (mesh != null && mesh.triangleCount > 0) {
                            renderer.setMesh(mesh.vertices, mesh.indices)
                            lastMeshSummary =
                                "${mesh.triangleCount} 面 / ${mesh.vertexCount} 顶点（LIVE）"
                        } else {
                            renderer.clearMesh()
                            lastMeshSummary = "Fusion Epoch 已激活，网格尚不足"
                        }
                        glView.requestRender()
                    }
                } finally {
                    // 回填 native 实际构建耗时（meshStats 槽 12 = totalMs），
                    // 供下一次自适应退避使用。
                    liveMeshLastBuildMs = try {
                        exportManager.meshStats()
                            .getOrNull(NativeBridge.MESH_STATS_INDEX_TOTAL_MS)
                            ?.toLong()?.coerceAtLeast(0L) ?: liveMeshLastBuildMs
                    } catch (_: Throwable) {
                        liveMeshLastBuildMs
                    }
                    liveMeshBuildBusy = false
                }
            }
        } catch (_: Throwable) {
            liveMeshBuildBusy = false
        }
    }

    /** 网格统计一行摘要（含「提取了多少原始面、去掉了多少小分量」）。 */
    private fun meshStatsSummary(prefix: String = "网格"): String {
        if (!::exportManager.isInitialized) return "$prefix n/a"
        val s = try {
            exportManager.meshStats()
        } catch (t: Throwable) {
            IntArray(0)
        }
        if (s.size < NativeBridge.MESH_STATS_SLOTS) return "$prefix n/a"
        return "$prefix ${s[NativeBridge.MESH_STATS_INDEX_TRIANGLES]}面/" +
            "${s[NativeBridge.MESH_STATS_INDEX_VERTICES]}顶点" +
            " 原始${s[NativeBridge.MESH_STATS_INDEX_RAW_TRIANGLES]}面" +
            " 去分量${s[NativeBridge.MESH_STATS_INDEX_COMPONENTS_REMOVED]}" +
            " 块${s[NativeBridge.MESH_STATS_INDEX_BLOCKS_SCANNED]}" +
            " 体素${s[NativeBridge.MESH_STATS_INDEX_VOXEL_SIZE_UM] / 1000f}mm" +
            " 简化${if (s[NativeBridge.MESH_STATS_INDEX_DECIMATED] != 0) 1 else 0}" +
            " 质量${s[NativeBridge.MESH_STATS_INDEX_QUALITY]}" +
            " ${s[NativeBridge.MESH_STATS_INDEX_TOTAL_MS]}ms" +
            cleanupSummary()
    }

    /**
     * V0.6：GLB 是否带 HQ 纹理，以及烘焙统计一行摘要。
     *
     * 槽位含义见 NativeBridge.TEXTURE_STATS_* ：
     *   2 = 实际使用视角数 / 0 = 已登记视角数 / 3,4 = atlas 宽高 / 7 = 覆盖率×10
     */
    private fun textureNote(textured: Boolean): String {
        if (!textured) {
            // V0.13 修复：真机实测（OnePlus PLK110）发现 VINS 未初始化成功时
            // keyframes=0，nativeBakeTexturedGlb 直接 skip，用户拿到的 GLB
            // 无纹理却毫无提示。这里把「为什么没有纹理」讲清楚，并给出
            // 下一步动作指引（VINS 需要视差才能初始化——手机必须缓慢移动）。
            val kf = try {
                val t = exportManager.textureStats()
                if (t.size >= NativeBridge.TEXTURE_STATS_SLOTS)
                    t[NativeBridge.TEXTURE_STATS_INDEX_REGISTERED_KEYFRAMES]
                else -1
            } catch (_: Throwable) {
                -1
            }
            return if (kf == 0) " · vertex color（未采集到关键帧：扫描时请缓慢移动手机让追踪初始化）"
            else " · vertex color（已登记 $kf 个关键帧但纹理烘焙未生效）"
        }
        val t = try {
            exportManager.textureStats()
        } catch (_: Throwable) {
            IntArray(0)
        }
        if (t.size < NativeBridge.TEXTURE_STATS_SLOTS) return " · HQ 纹理"
        val cov = String.format(
            java.util.Locale.US,
            "%.0f",
            t[NativeBridge.TEXTURE_STATS_INDEX_COVERAGE_X10] / 10f
        )
        return " · HQ 纹理 ${t[NativeBridge.TEXTURE_STATS_INDEX_ATLAS_W]}² " +
            "${t[NativeBridge.TEXTURE_STATS_INDEX_USED_KEYFRAMES]}/" +
            "${t[NativeBridge.TEXTURE_STATS_INDEX_REGISTERED_KEYFRAMES]}视角 覆盖$cov%"
    }

    /**
     * V0.6：导出前几何清理一行摘要（weld / 去漂浮分量 / 补洞 / QEM 塌缩边）。
     *
     * **这份统计只对导出资产成立** —— 屏幕上的 AR overlay 走的是 MeshEngine
     * 自己那条已验收的清理链，两者刻意分开。
     */
    private fun cleanupSummary(): String {
        if (!::exportManager.isInitialized) return ""
        val c = try {
            exportManager.meshCleanupStats()
        } catch (_: Throwable) {
            IntArray(0)
        }
        if (c.size < NativeBridge.MESH_CLEANUP_STATS_SLOTS ||
            c[NativeBridge.MESH_CLEANUP_INDEX_INPUT_TRIANGLES] <= 0
        ) {
            return ""
        }
        return " 清理${c[NativeBridge.MESH_CLEANUP_INDEX_INPUT_TRIANGLES]}" +
            "→${c[NativeBridge.MESH_CLEANUP_INDEX_OUTPUT_TRIANGLES]}" +
            " 去分量${c[NativeBridge.MESH_CLEANUP_INDEX_REMOVED_COMPONENTS]}" +
            " 补洞${c[NativeBridge.MESH_CLEANUP_INDEX_FILLED_HOLES]}" +
            " 塌缩${c[NativeBridge.MESH_CLEANUP_INDEX_QEM_COLLAPSED_EDGES]}"
    }

// --------------------------------------------------------- V0.7 persistent AR

    private fun restoreLatestPersistentAr() {
        if (!ensureSystemReady()) return
        if (scanning || modelOperationBusy) {
            toast("请先停止扫描，并等待当前模型操作完成")
            return
        }
        modelOperationBusy = true
        if (modelViewerActive) exitModelViewer()
        depthGeneration++
        scanNativeReady = false
        arMeshViewing = true
        lastRelocWasLocalized = false
        lastRelocPollMs = 0L
        NativeBridge.nativeSetPersistentMapCaptureEnabled(false)
        renderer.clearMesh()
        renderer.clearTexturedMesh()

        // latest() hashes every package: never call it on the UI thread.
        // V0.13.4 P0-C：提交时捕获令牌，回跳时校验。
        val token = scanSessionToken
        kotlin.concurrent.thread(name = "ScanPackageRestore", isDaemon = true) {
            val result = try {
                val info = ScanPackageManager.latest(applicationContext)
                if (info == null) {
                    ScanPackageManager.Result(false, "没有可恢复的完整扫描包")
                } else {
                    ScanPackageManager.restore(applicationContext, info)
                }
            } catch (t: Throwable) {
                ScanPackageManager.Result(false, "恢复失败：${t.message}")
            }
            runOnUiThread {
                try {
                    // 恢复期间开了新扫描：这次恢复的结果属于旧会话，丢弃。
                    if (!isDestroyed && token == scanSessionToken) {
                        val asset = if (result.ok) TexturedArAssetLoader.loadInto(renderer) else null
                        arMeshViewing = result.ok && asset?.ok == true
                        if (arMeshViewing) {
                            lastModelFile = result.packageInfo?.model
                            applyScanArLayer(false)
                            toast(result.message)
                        } else {
                            renderer.clearTexturedMesh()
                            toast(asset?.message ?: result.message)
                        }
                        glView.requestRender()
                    }
                } finally {
                    modelOperationBusy = false
                }
            }
        }
    }

    private fun pollPersistentRelocalization() {
        if (
            scanning ||
            !arMeshViewing
        ) {
            return
        }

        val now =
            android.os.SystemClock.elapsedRealtime()

        if (
            now - lastRelocPollMs <
            500L
        ) {
            return
        }

        lastRelocPollMs = now

        val handler = depthHandler
        if (
            !relocalizationBusy &&
            handler != null
        ) {
            relocalizationBusy = true
            handler.post {
                try {
                    NativeBridge.nativeTryPersistentRelocalization()
                } catch (_: Throwable) {
                } finally {
                    relocalizationBusy = false
                }
            }
        }

        val stats =
            FloatArray(
                NativeBridge.RELOCALIZATION_STATS_SLOTS
            )

        val ok =
            try {
                NativeBridge.nativeGetRelocalizationStats(
                    stats
                )
            } catch (_: Throwable) {
                false
            }

        if (!ok) return

        val state =
            stats[
                NativeBridge.RELOC_INDEX_STATE
            ].toInt()

        val localized =
            stats[
                NativeBridge.RELOC_INDEX_LOCALIZED
            ] > 0.5f

        val matches =
            stats[
                NativeBridge.RELOC_INDEX_MATCHES
            ].toInt()

        val inliers =
            stats[
                NativeBridge.RELOC_INDEX_INLIERS
            ].toInt()

        val ratio =
            stats[
                NativeBridge.RELOC_INDEX_INLIER_RATIO
            ]

        val reproj =
            stats[
                NativeBridge.RELOC_INDEX_MEDIAN_REPROJ
            ]

        if (localized) {
            lastMeshSummary =
                "AR重定位成功 · " +
                    "$inliers/$matches inliers · " +
                    "${"%.1f".format(java.util.Locale.US, reproj)} px"

            if (!lastRelocWasLocalized) {
                lastRelocWasLocalized = true
                toast(
                    "AR 重定位成功 · " +
                        "$inliers/$matches inliers"
                )
                glView.requestRender()
            }
        } else {
            lastRelocWasLocalized = false

            lastMeshSummary =
                when (state) {
                    1 ->
                        "AR地图已加载 · 等待 VINS 初始化"
                    2 ->
                        "AR重定位中 · " +
                            "$inliers/$matches inliers · " +
                            "ratio=${"%.2f".format(java.util.Locale.US, ratio)}"
                    else ->
                        "AR重定位未启动"
                }
        }
    }

    private fun materializeAsset(assetName: String): String {
        val dir = java.io.File(filesDir, "tracker_models")
        dir.mkdirs()
        val out = java.io.File(dir, assetName)
        if (!out.exists()) {
            assets.open("models/$assetName").use { input ->
                out.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return out.absolutePath
    }

    /**
     * vc159 ThermalGuard：按当前系统热状态给出「两次深度推理之间的最小间隔」。
     *
     * 66MB 单目模型在 XNNPACK CPU 上单帧约 265–386ms，连续跑等于常驻占满约一个核。
     *  - NONE / LIGHT：0ms，不干预（保持原生节奏）；
     *  - MODERATE：450ms，占空比约 40%；
     *  - SEVERE 及以上：800ms，占空比约 30%，优先给 SoC 降温。
     * 深度帧被跳过是安全的：相机会继续送帧，下一帧到达时照常提交。
     */
    private fun thermalDepthMinIntervalMs(): Long = when {
        thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> 800L
        thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE -> 450L
        else -> 0L
    }

    /**
     * vc159 ThermalGuard：热状态下把实时网格重建间隔进一步拉长。
     *
     * `nativeBuildMesh` 的 rawTris 随 TSDF 覆盖无上限增长（实拍见到 82 万），
     * 构建本身是全核 QEM 简化、持全局锁 5–11s。发烫时把重建节拍再放宽，
     * 减少「全核满载 + 冻结相机」的次数。
     */
    private fun thermalMeshIntervalScale(): Long = when {
        thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> 4L
        thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE -> 2L
        else -> 1L
    }

    /**
     * 统一的 Toast 出口（vc159 起带限流）。
     *
     * 实拍日志里系统侧出现过 **386 次** `Toast already killed. pkg=com.mobilescan3d`，
     * 说明有调用点在高频路径上反复弹同一句提示，既刷屏又白耗电。这里做两档限流：
     *  - 相同内容 1.5s 内只弹一次；
     *  - 任意内容 400ms 内最多弹一次。
     */
    private fun toast(msg: String) {
        val now = android.os.SystemClock.elapsedRealtime()
        synchronized(toastLock) {
            if (msg == lastToastMsg && now - lastToastMs < 1_500L) return
            if (now - lastToastMs < 400L) return
            lastToastMsg = msg
            lastToastMs = now
        }
        runOnUiThread {
            android.widget.Toast.makeText(this, msg, android.widget.Toast.LENGTH_SHORT).show()
        }
    }

    @Volatile private var lastToastMs = 0L
    @Volatile private var lastToastMsg = ""
    private val toastLock = Any()

    /**
     * 深度尺度采样：raw 目标深度 ↔ VINS 三角化深度的比例失配。
     *
     * **只统计，不修正**。真正的判定与统计在 [DepthScaleEstimator] 里；
     * 这里的职责只是把两个可观测深度按节流频率喂进去。
     * `nativeGetTargetState` / `nativeGetDepthDiagnostics` 内部都各自持锁，
     * 与本函数所在的相机线程没有嵌套持锁关系（nativeOnCameraFrame 早已返回）。
     */
    private fun sampleDepthScale(nowNs: Long) {
        if (!objectLockEnabled) return
        if (lastDepthScaleSampleNs != 0L && nowNs >= lastDepthScaleSampleNs &&
            nowNs - lastDepthScaleSampleNs < DepthScaleEstimator.SAMPLE_INTERVAL_NS) return
        lastDepthScaleSampleNs = nowNs
        val state = NativeBridge.nativeGetTargetState(depthScaleTargetBuf)
        NativeBridge.nativeGetDepthDiagnostics(depthScaleDepthBuf)
        depthScaleEstimator.maybeSample(
            nowNs = nowNs,
            stateCode = state,
            confidence = depthScaleTargetBuf[5],
            trackedPoints = depthScaleTargetBuf[8].toInt(),
            inlierRatio = depthScaleTargetBuf[9],
            targetRawDepthMedian = depthScaleDepthBuf[1],
            vinsDepthMedian = depthScaleDepthBuf[3],
            // ROI 里真正有深度的像素数。目标出界时它会掉到个位数，
            // 此时 P10/P50/P90 会退化成同一个值（实机 P10=P50=P90=5.14128），
            // 这种样本一进统计就会把 median 往完全错误的方向拽。
            targetDepthValidPixels = depthScaleDepthBuf[4].toInt()
        )
    }

    private fun onFrameTick(ts: Long) {
        // 目标框自成一个 30Hz 节拍，与 Header 的 1Hz 解耦。
        updateTargetUiTick(ts)
        // V0.12: 网格快照走自己的 5s 节拍（函数内部自带节流与忙碌门），
        // 这里绝不做任何按相机帧率的 native 重操作。
        maybeRefreshLiveMesh()
        maybeSaveRecoveryCheckpoint()
        if (::hqCapture.isInitialized && !cameraRangeSwitching) {
            hqCapture.onFrameTick(ts, scanning, cameraDevice, captureSession)
            syncCaptureStateFromController()
        }
        multiCam?.updateScanState(
            enabled = scanning && scanNativeReady,
            sessionId = sessionId,
            allowTeleTexture = scanning &&
                scanNativeReady &&
                ::hqCapture.isInitialized &&
                hqCapture.stats.threeAReady &&
                hqCapture.stats.stableFrames >= 5
        )
        if (scanning) {
            sampleDepthScale(ts)
        }
        fpsFrames++
        if (fpsLastNs == 0L) fpsLastNs = ts
        val dt = (ts - fpsLastNs) / 1_000_000_000.0
        if (dt >= 1.0) {
            val f = (fpsFrames / dt).toFloat()
            fpsFrames = 0
            fpsLastNs = ts
            if (f > 0f) android.util.Log.d(
                "FpsProbe",
                String.format("fps=%.1f%s", f, if (f < 25f) " LOW" else "")
            )
            runOnUiThread {
                fps = f
                updateHeader()
                updateStatusBar()
            }
        }
    }

    private fun updateHeader() {
        val vinsOk = NativeBridge.nativeVinsInitialized()
        val lightStatus = when {
            torchFailed -> "补光不可用"
            torchRequested -> "补光已开启"
            !torchAvailable -> "无闪光灯"
            autoFillLight -> "自动补光"
            else -> "补光关闭"
        }
        headerStatus.text = "FPS %.1f · %s · ≤%.1fm · %s\n镜头：%s".format(
            fps, if (scanning) "扫描中" else "空闲", scanMaxDistanceMeters, lightStatus, cameraRangeStatus)
        if (objectLockEnabled) {
            // Overlay 的刷新已移到 30Hz 的 updateTargetUiTick()，这里只保留
            // AF 重锁 —— 而且**必须**留在这条 1Hz 路径上。
            //
            // maybeRelockTargetFocus() 内部有「连续 5 帧未合焦就重新 focusAt」
            // 的逻辑：1Hz 下是 5 秒的重试间隔，一旦跟着 overlay 提到 30Hz
            // 就变成 0.17 秒，会把相机 AF 打成持续抖动。
            maybeRelockTargetFocus()
        }
        warningBanner.visibility =
            if (scanning && !vinsOk) android.view.View.VISIBLE else android.view.View.GONE
        if (scanning && !vinsOk) {
            // V0.13.1：旧的常量文案「定位失锁：位姿不可用于拼接」在**从未初始化**
            // 时也会显示 —— 用户刚点开始、还没移动出视差，就被判「失锁」，既不准
            // 确也让人以为设备坏了。这两种状态的处理动作完全不同：还没初始化要
            // 平移手机积累视差，初始化后跟丢则要回到已扫描区域。
            val everInit = try {
                NativeBridge.nativeVinsEverInitialized()
            } catch (t: Throwable) {
                false
            }
            warningBanner.text = if (everInit) {
                "定位不可用：已保留模型 · 回到已扫区域；若持续失锁，请先导出再重新扫描"
            } else {
                "初始化中：请手持手机缓慢平移，建立定位后开始拼接"
            }
        }
        if (scanning && vinsOk) {
            val epoch = FloatArray(NativeBridge.FUSION_EPOCH_STATS_SLOTS)
            val count = runCatching { NativeBridge.nativeGetFusionEpochStats(epoch) }.getOrDefault(0)
            val notice = when {
                !::depthProvider.isInitialized || !depthProvider.available ->
                    "深度模型不可用，请检查模型文件后重新开始"
                depthLastError.isNotEmpty() -> "深度帧处理失败，正在重试；详情见反馈报告"
                android.os.SystemClock.elapsedRealtime() - depthCompletedMs > 4000L ->
                    "深度处理延迟，等待当前帧完成；请放慢移动"
                count < epoch.size || epoch[NativeBridge.FUSION_EPOCH_INDEX_ACTIVE] < 0.5f ->
                    "正在标定深度：请缓慢平移，让目标和周围纹理保持清晰"
                epoch[NativeBridge.FUSION_EPOCH_INDEX_SUSPENDED] > 0.5f ->
                    "已保留扫描结果；深度暂不稳定，请放慢移动并回到已扫区域"
                else -> ""
            }
            warningBanner.text = notice
            warningBanner.visibility = if (notice.isEmpty()) android.view.View.GONE else android.view.View.VISIBLE
        }
        val target = try {
            NativeBridge.nativeGetPointCount().toFloat()
        } catch (t: Throwable) {
            0f
        }
        hudShownPoints += (target - hudShownPoints) * 0.35f
        val shown = hudShownPoints.toInt()
        hudText.text = if (hudExpanded) {
            // 只报「总点数」没有诊断价值：实机出现过 600000 总点里 stable 只有 28，
            // 那种情况下屏幕几乎全是一次性噪声，而总数看起来却很壮观。
            // 所以把「地图 / 确认 / 稳定 / 绘制」四个数都列出来。
            val metrics = try {
                NativeBridge.nativeGetHudMetrics()
            } catch (t: Throwable) {
                "点云 $shown 点"
            }
            metrics +
                "\n图层 ${arDrawModeLabel()}" +
                " · 累计绘制 ${renderer.drawnAccumulated}" +
                " 当前帧 ${renderer.drawnDebug}" +
                (if (renderer.drawnMeshTriangles > 0) " 网格 ${renderer.drawnMeshTriangles}" else "") +
                "\n" + depthCalibrationSummary() +
                "\n" + fusionEpochSummary() +
                "\n" + (multiCam?.hudSummary() ?: "MultiCam: n/a") +
                "\n网格 " + (if (lastMeshSummary == "n/a") "未生成" else lastMeshSummary) +
                "\n" + if (renderer.poseFromTimestamp) {
                "AR 按帧时间戳取 pose"
            } else {
                "AR 回退：用最新 pose（时间戳没查到）"
            }
        } else {
            "地图 $shown · 绘制 ${renderer.drawnAccumulated}" +
                (if (renderer.drawnDebug > 0) " + ${renderer.drawnDebug}" else "") +
                " · " + metricLabel()
        }
        // Update compact status and the on-demand detail fields.
        tvFps.text = "%.1f".format(fps)
        val uiNow = android.os.SystemClock.elapsedRealtime()
        val freshDistance = if (latestScanDistanceMs > 0L &&
            uiNow - latestScanDistanceMs < 2_500L &&
            latestScanDistanceMeters.isFinite() &&
            latestScanDistanceMeters > 0f) latestScanDistanceMeters else 0f
        tvDistance.text = if ((scanning || scanPaused) && freshDistance > 0f) {
            "%.2fm".format(freshDistance)
        } else {
            "≤%.1fm".format(scanMaxDistanceMeters)
        }
        tvAutoLight.text = when {
            torchFailed -> "故障"
            torchRequested || autoFillLight -> "开"
            !torchAvailable -> "无"
            else -> "关"
        }
        tvCameraMode.text = cameraRangeStatus.substringBefore("·").trim()
        if (::toolCameraLabel.isInitialized) {
            toolCameraLabel.text = tvCameraMode.text
        }
        if (::toolLightLabel.isInitialized) {
            toolLightLabel.text = when {
                torchFailed -> "故障"
                torchRequested -> "已开启"
                !torchAvailable -> "不可用"
                autoFillLight -> "自动"
                else -> "关闭"
            }
            toolLightLabel.setTextColor(
                getColor(
                    when {
                        torchFailed -> R.color.scan_danger
                        torchRequested || autoFillLight -> R.color.scan_primary
                        else -> R.color.scan_text_muted
                    }
                )
            )
        }
        if (::toolLockLabel.isInitialized) {
            toolLockLabel.text = when {
                !objectLockEnabled -> "关闭"
                targetConfidence >= 0.55f -> "追踪中"
                else -> "选目标"
            }
            toolLockLabel.setTextColor(
                getColor(
                    when {
                        targetConfidence >= 0.55f && objectLockEnabled -> R.color.scan_success
                        objectLockEnabled -> R.color.scan_warning
                        else -> R.color.scan_text_muted
                    }
                )
            )
        }
        if (::toolModeLabel.isInitialized) {
            toolModeLabel.text = when (scanVoxelProfile) {
                0 -> "精细物体"
                1 -> "快速物体"
                2 -> "房间"
                else -> "智能推荐"
            }
        }
        if (::toolAutoLabel.isInitialized) {
            toolAutoLabel.text = if (autoFillLight) "自动" else "自定义"
        }
        val depthOk = ::depthProvider.isInitialized && depthProvider.available && depthLastError.isEmpty()
        tvDepth.text = if (depthOk) "深度: 有效" else "深度: 等待"
        tvDepth.setTextColor(getColor(if (depthOk) R.color.scan_success else R.color.scan_warning))
        findViewById<android.view.View>(R.id.toolLight).contentDescription =
            "补光：${toolLightLabel.text}，点按切换自动补光"
        findViewById<android.view.View>(R.id.toolRealtime).contentDescription =
            "预览图层：${arDrawModeLabel()}，点按切换"
        tvMapCount.text = "地图: $shown"
        tvDrawCount.text = "绘制: ${renderer.drawnAccumulated}" +
            (if (renderer.drawnDebug > 0) " +${renderer.drawnDebug}" else "")
        tvUnit.text = metricLabel()
        // Keep the compact telemetry available to existing reporting code.
        hudCompact.text = "地图 $shown · 绘制 ${renderer.drawnAccumulated}" +
            (if (renderer.drawnDebug > 0) " +${renderer.drawnDebug}" else "") +
            " · " + metricLabel()

        updateScanExperienceUi(vinsOk)

        // A healthy scan should still tell the user what to do next. Warnings
        // stay higher priority; native keyframe guidance fills the quiet state.
        if (scanPaused) {
            warningBanner.text = "扫描已暂停 · 调整位置后点“继续”，或直接完成本次扫描"
            warningBanner.visibility = android.view.View.VISIBLE
        } else if (scanning && warningBanner.visibility == android.view.View.GONE) {
            val nativeGuidance = runCatching { NativeBridge.nativeGetGuidance() }.getOrDefault("")
            val guidance = when {
                nativeGuidance.contains("模糊") ||
                    nativeGuidance.contains("曝光") ||
                    nativeGuidance.contains("放慢") ||
                    nativeGuidance.contains("跟踪") -> nativeGuidance
                lastCoverageGuidance.isNotBlank() -> lastCoverageGuidance
                else -> nativeGuidance
            }
            if (guidance.isNotBlank()) {
                warningBanner.text = guidance
                warningBanner.visibility = android.view.View.VISIBLE
            }
        } else if (!scanning && !scanPaused && warningBanner.visibility == android.view.View.GONE) {
            warningBanner.text = getString(R.string.scan_hint)
            warningBanner.visibility = android.view.View.VISIBLE
        }
        updateScanHintAppearance(vinsOk)
    }

    /**
     * "采集充分度" is a monotonic capture-quality proxy, not a claim that the
     * object's geometric surface coverage has been measured exactly.
     *
     * It combines accepted viewpoints (keyframes), repeatedly observed geometry
     * (confirmed/stable points), and current image quality. The signal is used
     * only for UX guidance and never for reconstruction decisions.
     */
    private fun updateScanExperienceUi(vinsOk: Boolean) {
        val metricCount = runCatching {
            NativeBridge.nativeGetScanUiMetrics(scanUiMetrics)
        }.getOrDefault(0)

        if ((scanning || scanPaused) && metricCount == scanUiMetrics.size) {
            val keyframes = scanUiMetrics[NativeBridge.SCAN_UI_KEYFRAMES].coerceAtLeast(0f)
            val sharpness = scanUiMetrics[NativeBridge.SCAN_UI_SHARPNESS].coerceIn(0f, 1f)
            val exposure = scanUiMetrics[NativeBridge.SCAN_UI_EXPOSURE].coerceIn(0f, 1f)
            val confirmed = scanUiMetrics[NativeBridge.SCAN_UI_CONFIRMED].coerceAtLeast(0f)
            val stable = scanUiMetrics[NativeBridge.SCAN_UI_STABLE].coerceAtLeast(0f)

            val keyframeScore = ((keyframes - 2f) / 26f).coerceIn(0f, 1f)
            val pointScore = (
                kotlin.math.ln(1.0 + confirmed.toDouble()) /
                    kotlin.math.ln(1.0 + 6000.0)
                ).toFloat().coerceIn(0f, 1f)
            val stableRatio = if (confirmed > 0f) (stable / confirmed).coerceIn(0f, 1f) else 0f
            val imageQuality = (sharpness * 0.58f + exposure * 0.42f).coerceIn(0f, 1f)
            lastGeometryQualityPercent = (
                (stableRatio * 0.72f + pointScore * 0.28f) * 100f
                ).toInt().coerceIn(0, 100)
            lastTextureQualityPercent = (imageQuality * 100f).toInt().coerceIn(0, 100)
            updateViewpointCoverage(vinsOk, imageQuality)

            val score = (
                keyframeScore * 0.52f +
                    pointScore * 0.28f +
                    stableRatio * 0.12f +
                    imageQuality * 0.08f
                )
            val percent = (score * 100f).toInt().coerceIn(0, 96)
            lastScanSufficiency = kotlin.math.max(lastScanSufficiency, percent)
        }

        if (metricCount != scanUiMetrics.size) {
            updateViewpointCoverage(vinsOk, 0.55f)
        }

        scanProgressView.progress = lastScanSufficiency
        tvScanProgressPercent.text = "采集 $lastScanSufficiency%"
        tvViewpointSummary.text = "视角 $lastViewpointCoveragePercent%"
        tvScanProgressPercent.setTextColor(
            getColor(if (lastScanSufficiency >= 82) R.color.scan_success else R.color.scan_primary)
        )

        val guidance = if (scanning) {
            runCatching { NativeBridge.nativeGetGuidance() }.getOrDefault("")
        } else ""

        val motionTooFast =
            scanAngularSpeedDps > 72f ||
                guidance.contains("放慢") ||
                guidance.contains("模糊")
        val motionTooSlow =
            scanning && guidance.contains("变化太小")

        tvMotionState.text = when {
            scanPaused -> "移动：已暂停"
            !scanning -> "移动：待开始"
            motionTooFast -> "移动：过快 · 请放慢"
            motionTooSlow -> "移动：视角变化不足"
            else -> "移动：合适"
        }
        tvMotionState.setTextColor(
            getColor(
                when {
                    motionTooFast -> R.color.scan_warning
                    motionTooSlow -> R.color.scan_warning
                    scanning -> R.color.scan_success
                    else -> R.color.scan_text_secondary
                }
            )
        )

        val now = android.os.SystemClock.elapsedRealtime()
        val distance = when {
            objectLockEnabled && targetMedianDepthMeters.isFinite() && targetMedianDepthMeters > 0f ->
                targetMedianDepthMeters
            latestScanDistanceMs > 0L && now - latestScanDistanceMs < 2_500L ->
                latestScanDistanceMeters
            else -> 0f
        }
        val distanceTooFar = distance > 0f && distance > scanMaxDistanceMeters
        val distanceTooNear = distance > 0f && distance < 0.18f
        tvDistanceState.text = when {
            scanPaused && distance > 0f -> "距离：%.2fm · 已暂停".format(distance)
            distanceTooFar -> "距离：%.2fm · 过远".format(distance)
            distanceTooNear -> "距离：%.2fm · 过近".format(distance)
            distance > 0f -> "距离：%.2fm · 合适".format(distance)
            else -> "距离：≤%.1fm · 等待测距".format(scanMaxDistanceMeters)
        }
        tvDistanceState.setTextColor(
            getColor(
                when {
                    distanceTooFar || distanceTooNear -> R.color.scan_warning
                    distance > 0f && (scanning || scanPaused) -> R.color.scan_success
                    else -> R.color.scan_text_secondary
                }
            )
        )

        val everInit = runCatching { NativeBridge.nativeVinsEverInitialized() }.getOrDefault(false)
        val trackingWarning = (scanning || scanPaused) && !vinsOk && everInit
        tvTrackingState.text = when {
            scanPaused && vinsOk -> "定位：保持"
            !(scanning || scanPaused) -> "定位：待开始"
            vinsOk -> "定位：稳定"
            everInit -> "定位：失锁 · 回到已扫区域"
            else -> "定位：初始化中"
        }
        tvTrackingState.setTextColor(
            getColor(
                when {
                    trackingWarning -> R.color.scan_danger
                    vinsOk && (scanning || scanPaused) -> R.color.scan_success
                    else -> R.color.scan_text_secondary
                }
            )
        )

        tvScanStateBadge.text = when {
            modelOperationBusy -> "模型处理中"
            scanReviewActive -> "扫描检查"
            scanPaused -> "已暂停"
            scanning && !vinsOk -> "定位中"
            scanning -> "扫描中"
            lastScanSufficiency > 0 && sessionCreated -> "已完成"
            else -> "待扫描"
        }
        tvScanStateBadge.setTextColor(
            getColor(
                when {
                    scanReviewActive -> R.color.scan_primary
                    scanPaused -> R.color.scan_text_secondary
                    scanning && !vinsOk -> R.color.scan_warning
                    scanning -> R.color.scan_success
                    lastScanSufficiency > 0 && sessionCreated -> R.color.scan_primary
                    else -> R.color.scan_text_secondary
                }
            )
        )

        val reticleWarning = trackingWarning || motionTooFast || distanceTooFar || distanceTooNear
        scanReticleView.setScanUiState(
            lastScanSufficiency / 100f,
            scanning,
            scanPaused,
            reticleWarning
        )

        updateStandardScanGuidance(
            vinsOk = vinsOk,
            vinsEverInitialized = everInit,
            distanceMeters = distance,
            nativeGuidance = guidance
        )
    }

    private fun updateStandardScanGuidance(
        vinsOk: Boolean,
        vinsEverInitialized: Boolean,
        distanceMeters: Float,
        nativeGuidance: String
    ) {
        if (!guidedScanEnabled) {
            lastCoverageGuidance = if (scanning) computeCoverageGuidance() else ""
            if (::scanGuidanceOverlay.isInitialized) {
                scanGuidanceOverlay.resetGuidance()
                scanGuidanceOverlay.visibility = android.view.View.GONE
            }
            if (::tvGuidanceStep.isInitialized) tvGuidanceStep.visibility = android.view.View.GONE
            return
        }

        val active = scanning || scanPaused
        val output = scanGuidanceController.evaluate(
            com.mobilescan3d.scan.guidance.ScanGuidanceInput(
                active = active,
                paused = scanPaused,
                guidedMode = true,
                objectTrackingEnabled = objectLockEnabled,
                targetConfidence = targetConfidence,
                vinsInitialized = vinsOk,
                vinsEverInitialized = vinsEverInitialized,
                distanceMeters = distanceMeters,
                maxDistanceMeters = scanMaxDistanceMeters,
                angularSpeedDps = scanAngularSpeedDps,
                coverage = viewpointCoverage,
                currentSector = coverageCurrentSector,
                currentElevationDeg = coverageCurrentElevationDeg,
                captureSufficiency = lastScanSufficiency,
                geometryQuality = lastGeometryQualityPercent,
                surfaceCoverage = lastSurfaceCoveragePercent,
                nativeGuidance = nativeGuidance,
                nowMs = android.os.SystemClock.elapsedRealtime()
            )
        )

        lastCoverageGuidance = output.primaryInstruction

        if (recoveryActive && output.recoveryMode && output.recoveryRemaining == 0 && scanning && !scanPaused) {
            val now = android.os.SystemClock.elapsedRealtime()
            if (recoveryReadySinceMs == 0L) recoveryReadySinceMs = now
            if (now - recoveryReadySinceMs >= RECOVERY_AUTO_REVIEW_HOLD_MS &&
                now - recoveryStartedMs >= RECOVERY_MIN_SCAN_MS &&
                scanNativeReady
            ) {
                recoveryReadySinceMs = 0L
                recoveryActive = false
                toast("定向补扫完成，重新检查模型")
                enterScanReview()
                return
            }
        } else if (recoveryActive) {
            recoveryReadySinceMs = 0L
        }

        if (::scanGuidanceOverlay.isInitialized) {
            scanGuidanceOverlay.visibility = if (active) android.view.View.VISIBLE else android.view.View.GONE
            scanGuidanceOverlay.setGuidance(viewpointCoverage, coverageCurrentSector, output)
        }
        if (::tvGuidanceStep.isInitialized) {
            tvGuidanceStep.visibility = if (active) android.view.View.VISIBLE else android.view.View.GONE
            tvGuidanceStep.text = output.stepLabel
            tvGuidanceStep.setTextColor(
                getColor(
                    when (output.severity) {
                        com.mobilescan3d.scan.guidance.GuidanceSeverity.SUCCESS -> R.color.scan_success
                        com.mobilescan3d.scan.guidance.GuidanceSeverity.WARNING,
                        com.mobilescan3d.scan.guidance.GuidanceSeverity.DANGER -> R.color.scan_warning
                        else -> R.color.scan_primary
                    }
                )
            )
        }
    }

    /**
     * Builds an approximate orbit heat map from the existing VINS render pose.
     *
     * Anchor initialization:
     *   camera position + camera forward * measured target/range distance.
     * Once initialized, the anchor stays fixed for this scan so the orbit bins
     * remain stable even if range estimates fluctuate.
     *
     * This is a viewpoint-coverage signal. It does not claim exact surface-area
     * coverage, which would require per-surface visibility/observation counts.
     */
    private fun updateViewpointCoverage(vinsOk: Boolean, sampleQuality: Float) {
        if (!::viewpointCoverageView.isInitialized) return
        if (!scanning || !vinsOk) {
            viewpointCoverageView.setCoverage(
                viewpointCoverage, coverageCurrentSector, lastViewpointCoveragePercent
            )
            return
        }

        val now = android.os.SystemClock.elapsedRealtime()
        if (coverageLastSampleMs != 0L && now - coverageLastSampleMs < 450L) return
        coverageLastSampleMs = now

        val poseOk = try {
            NativeBridge.nativeGetRenderPose(coveragePose)
        } catch (_: Throwable) {
            false
        }
        if (!poseOk) return

        val camX = coveragePose[9]
        val camY = coveragePose[10]
        val camZ = coveragePose[11]
        if (!camX.isFinite() || !camY.isFinite() || !camZ.isFinite()) return

        if (!coverageAnchorValid) {
            val freshRange = if (
                latestScanDistanceMs > 0L &&
                now - latestScanDistanceMs < 2_500L &&
                latestScanDistanceMeters.isFinite() &&
                latestScanDistanceMeters > 0.12f
            ) latestScanDistanceMeters else 0f
            val d = when {
                objectLockEnabled &&
                    targetMedianDepthMeters.isFinite() &&
                    targetMedianDepthMeters > 0.12f -> targetMedianDepthMeters
                freshRange > 0f -> freshRange
                else -> (scanMaxDistanceMeters * 0.72f).coerceIn(0.35f, 1.20f)
            }

            // Renderer pose convention already used by placement code:
            // forward is the third column of R.
            val fx = coveragePose[2]
            val fy = coveragePose[5]
            val fz = coveragePose[8]
            coverageAnchorWorld[0] = camX + fx * d
            coverageAnchorWorld[1] = camY + fy * d
            coverageAnchorWorld[2] = camZ + fz * d
            coverageAnchorValid = true
        }

        val dx = camX - coverageAnchorWorld[0]
        val dy = camY - coverageAnchorWorld[1]
        val dz = camZ - coverageAnchorWorld[2]
        val horizontal = kotlin.math.sqrt(dx * dx + dy * dy)
        val radius = kotlin.math.sqrt(horizontal * horizontal + dz * dz)
        if (!radius.isFinite() || radius < 0.12f) return

        var azimuth = kotlin.math.atan2(dy, dx)
        if (azimuth < 0f) azimuth += (Math.PI * 2.0).toFloat()
        val sector = (
            azimuth / ((Math.PI * 2.0 / 12.0).toFloat())
            ).toInt().coerceIn(0, 11)
        val elevationDeg = Math.toDegrees(kotlin.math.atan2(dz, horizontal).toDouble()).toFloat()
        coverageCurrentSector = sector
        coverageCurrentElevationDeg = elevationDeg

        // Quality-aware accumulation: one weak glance should not turn a sector green.
        val quality = sampleQuality.coerceIn(0f, 1f)
        val gain = (0.11f + 0.18f * quality).coerceIn(0.10f, 0.30f)
        viewpointCoverage[sector] = (viewpointCoverage[sector] + gain).coerceAtMost(1f)
        if (elevationDeg > 18f) {
            viewpointCoverage[12] = (viewpointCoverage[12] + gain * 0.86f).coerceAtMost(1f)
        }
        if (elevationDeg < -14f) {
            viewpointCoverage[13] = (viewpointCoverage[13] + gain * 0.86f).coerceAtMost(1f)
        }

        val sideAverage = viewpointCoverage.sliceArray(0 until 12).average().toFloat()
        val weighted = sideAverage * 0.82f +
            viewpointCoverage[12] * 0.10f +
            viewpointCoverage[13] * 0.08f
        lastViewpointCoveragePercent =
            (weighted * 100f).toInt().coerceIn(0, 100)
        if (!guidedScanEnabled) lastCoverageGuidance = computeCoverageGuidance()
        viewpointCoverageView.setCoverage(
            viewpointCoverage, coverageCurrentSector, lastViewpointCoveragePercent
        )
    }

    private fun updateSurfaceObservationStats(weights: FloatArray) {
        if (weights.isEmpty()) {
            lastSurfaceCoveragePercent = 0
            lastSurfaceRobustPercent = 0
            return
        }
        var usable = 0
        var robust = 0
        var valid = 0
        for (w in weights) {
            if (!w.isFinite() || w <= 0f) continue
            valid++
            if (w >= 4f) usable++
            if (w >= 8f) robust++
        }
        if (valid <= 0) {
            lastSurfaceCoveragePercent = 0
            lastSurfaceRobustPercent = 0
        } else {
            lastSurfaceCoveragePercent = (usable * 100 / valid).coerceIn(0, 100)
            lastSurfaceRobustPercent = (robust * 100 / valid).coerceIn(0, 100)
        }
    }

    /**
     * Replaces only RGB fields of the 9-float mesh vertex layout.
     * Geometry/normals stay byte-for-byte unchanged.
     *
     * Heat semantics:
     *   weak < 2 TSDF weight  -> red
     *   2..6                  -> red/yellow -> yellow/green
     *   >= 8                  -> green
     */
    private fun buildSurfaceObservationHeatmap(
        base: FloatArray,
        weights: FloatArray
    ): FloatArray? {
        val vertexCount = base.size / NativeBridge.MESH_VERTEX_FLOATS
        if (vertexCount <= 0 || weights.size < vertexCount) return null
        val out = base.copyOf()
        for (i in 0 until vertexCount) {
            val w = weights[i].coerceIn(0f, 10f)
            val (r, g, b) = when {
                w < 2f -> Triple(1.0f, 0.20f + 0.18f * w, 0.12f)
                w < 6f -> {
                    val t = (w - 2f) / 4f
                    Triple(1.0f - 0.35f * t, 0.56f + 0.38f * t, 0.10f)
                }
                else -> {
                    val t = ((w - 6f) / 4f).coerceIn(0f, 1f)
                    Triple(0.30f - 0.12f * t, 0.92f + 0.06f * t, 0.38f + 0.24f * t)
                }
            }
            val o = i * NativeBridge.MESH_VERTEX_FLOATS
            out[o + 6] = r
            out[o + 7] = g
            out[o + 8] = b
        }
        return out
    }

    private fun computeCoverageGuidance(): String {
        val sideMissing = (0 until 12).count { viewpointCoverage[it] < 0.38f }
        val topMissing = viewpointCoverage[12] < 0.34f
        val bottomMissing = viewpointCoverage[13] < 0.30f
        return when {
            lastViewpointCoveragePercent >= 82 ->
                "视角覆盖已经较完整，可点“完成扫描”检查模型"
            sideMissing <= 3 && topMissing ->
                "抬高手机，补扫物体顶部视角"
            sideMissing <= 2 && bottomMissing ->
                "降低手机，补扫底部和遮挡边缘"
            sideMissing > 0 ->
                "继续环绕物体 · 还有 $sideMissing 个侧面视角不足"
            topMissing ->
                "抬高手机，补扫顶部视角"
            bottomMissing ->
                "降低手机，补扫底部视角"
            else ->
                "覆盖良好，保持稳定完成最后一圈"
        }
    }

    private fun reviewMissingViewsText(): String {
        val sideMissing = (0 until 12).count { viewpointCoverage[it] < 0.38f }
        val notes = ArrayList<String>()
        if (sideMissing > 0) notes.add("侧面仍有 $sideMissing 个视角采集不足")
        if (viewpointCoverage[12] < 0.34f) notes.add("顶部视角不足")
        if (viewpointCoverage[13] < 0.30f && lastScanSufficiency < 82) notes.add("下半部/遮挡边缘视角偏少（桌面接触底面无需强制补扫）")
        if (lastGeometryQualityPercent < 55) notes.add("几何重复观测偏少")
        if (lastSurfaceCoveragePercent in 1..64) {
            notes.add("真实表面稳定观测不足（${lastSurfaceCoveragePercent}%）")
        }
        if (lastTextureQualityPercent < 55) notes.add("纹理画质偏弱")
        return if (notes.isEmpty()) {
            "未发现明显的视角短板；仍建议观察预览网格是否存在真实缺口。"
        } else {
            notes.joinToString(" · ")
        }
    }

    private fun updateScanHintAppearance(vinsOk: Boolean) {
        val text = warningBanner.text?.toString().orEmpty()
        val severe =
            text.contains("失败") ||
                text.contains("不可用") ||
                text.contains("失锁") ||
                text.contains("延迟") ||
                text.contains("不稳定") ||
                text.contains("过快")
        when {
            scanPaused -> {
                warningBanner.setBackgroundResource(R.drawable.bg_hint_paused)
                warningBanner.setTextColor(getColor(R.color.scan_text_primary))
            }
            scanning && (!vinsOk || severe) -> {
                warningBanner.setBackgroundResource(R.drawable.bg_hint_warning)
                warningBanner.setTextColor(getColor(R.color.scan_warning))
            }
            scanning && (text.contains("覆盖良好") || text.contains("可以完成") || text.contains("扫描完整")) -> {
                warningBanner.setBackgroundResource(R.drawable.bg_hint_success)
                warningBanner.setTextColor(getColor(R.color.scan_success))
            }
            else -> {
                warningBanner.setBackgroundResource(R.drawable.bg_hint)
                warningBanner.setTextColor(getColor(R.color.scan_text_primary))
            }
        }
    }

    private fun updateStatusBar() {
        // V0.13.4：epoch 冻结标定之后，把深度归一化映射一起冻住。
        //
        // 重参数化能保证「数值域变了标定也跟着变」，但每次变化都会给 epoch
        // 的漂移监控带来一次扰动。epoch 一旦锁定，(a,b) 就是定死的，此时
        // 让输入映射也定死，整个 epoch 内 d 的语义完全恒定 —— 这才是
        // 「尺度不漂」的强保证：评审 P0-1 之后深度表示固定为 RELATIVE_DEPTH，
        // 不再需要运行期冻结/重参数化；native 侧标定在 epoch 内收敛即可。
        val time = java.text.SimpleDateFormat("HH:mm", java.util.Locale.getDefault())
            .format(java.util.Date())
        val bat = try {
            val bm = getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
            bm.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
        } catch (e: Exception) {
            -1
        }
        val net = try {
            val tm = getSystemService(Context.TELEPHONY_SERVICE) as android.telephony.TelephonyManager
            when (tm.dataNetworkType) {
                android.telephony.TelephonyManager.NETWORK_TYPE_NR -> "5G"
                android.telephony.TelephonyManager.NETWORK_TYPE_LTE -> "4G"
                else -> "网"
            }
        } catch (e: Exception) {
            "网"
        }
        statusBarText.text = "$time · $net · ${if (bat >= 0) "$bat%" else "--"}"
    }

    private fun toggleHud() {
        hudExpanded = !hudExpanded
        updateHeader()
    }

    private fun arDrawModeLabel(): String = when (arDrawMode) {
        PointCloudRenderer.DRAW_ACCUMULATED -> "累计"
        PointCloudRenderer.DRAW_BOTH -> "两者"
        PointCloudRenderer.DRAW_MESH -> "网格"
        PointCloudRenderer.DRAW_LIVE -> "实时"
        else -> "当前帧"
    }

    /**
     * 循环切换 AR 图层：当前帧调试层 -> 累计点云 -> 两者。
     *
     * 顺序刻意如此：先只用「当前一帧 depth + 当前 ROI + 该帧 pose」验证坐标链，
     * 确认无误后再打开累计点云。否则一旦点云发散，无法判断是 Renderer 的
     * 投影错了，还是 depth scale / 精度 / fusion 的问题。
     */
    private fun cycleArLayer() {
        arDrawMode = (arDrawMode + 1) % 5
        renderer.drawMode = arDrawMode
        // 材质跟着图层走：LIVE 半透明（要看见真实场景），其它档回到偏
        // 实心的观感，方便验收几何。
        renderer.setMeshAlpha(
            if (arDrawMode == PointCloudRenderer.DRAW_LIVE) {
                scanMeshAlpha
            } else {
                viewMeshAlpha
            }
        )
        val wantDebug = arDrawMode == PointCloudRenderer.DRAW_TARGET_DEBUG ||
            arDrawMode == PointCloudRenderer.DRAW_BOTH
        try {
            NativeBridge.nativeSetTargetDebugEnabled(wantDebug)
        } catch (_: Throwable) {
        }
        // 切到网格图层时，如果 GPU 上还没有网格就先弄一个出来
        // —— 否则用户会看到一个「图层是网格但屏幕上什么都没有」的状态。
        if ((arDrawMode == PointCloudRenderer.DRAW_MESH ||
                arDrawMode == PointCloudRenderer.DRAW_LIVE) &&
            (renderer.meshUploadedTriangles <= 0 && renderer.texturedMeshUploadedTriangles <= 0)) {
            val cached = if (::exportManager.isInitialized) exportManager.lastMesh else null
            if (cached != null) {
                renderer.setMesh(cached.vertices, cached.indices)
            } else {
                buildMeshForPreview()
            }
        }
        arLayerButton?.text = arDrawModeLabel()
        glView.requestRender()
        toast(
            when (arDrawMode) {
                PointCloudRenderer.DRAW_ACCUMULATED ->
                    "只画累计点云（hits>=${renderer.accumulatedMinHits}，已剔除一次性点）"
                PointCloudRenderer.DRAW_BOTH -> "两层都画：绿色 = 当前帧调试层"
                PointCloudRenderer.DRAW_MESH ->
                    "只画重建网格（TSDF 零交叉面 -> Marching Tetrahedra）"
                PointCloudRenderer.DRAW_LIVE ->
                    "实时预览：世界坐标累计点云 + 连续融合网格"
                else -> "只画当前帧 target depth 调试层（验证 AR 坐标链）"
            }
        )
        updateHeader()
    }

    private fun showCameraRangeDialog() {
        val items = arrayOf("自动：按深度调配辅助摄像头", "近距：优先广角辅助", "中距：优先融合几何",
            "远距：优先长焦辅助", "切换主摄（停止扫描后，重建新会话）")
        android.app.AlertDialog.Builder(this)
            .setTitle("摄像头调配 · 保持定位主摄")
            .setItems(items) { _, which ->
                if (which == 4) { switchCamera(); return@setItems }
                val mode = CameraRangePolicy.Mode.entries[which]
                cameraHandler?.post {
                    cameraRangePolicy.setMode(mode)
                    updateCameraRange(Float.NaN, 0)
                }
            }.show()
    }

    /**
     * vc161：镜头调配链的日志出口。
     *
     * 上一轮实拍（10-01 11:36–11:52）里 `updateCameraRange` 只写 HUD，结果整场没有任何
     * 一行能证明「距离是否有效 / 档位是否被提交 / 每颗镜头是否被尝试过 / 切换耗时」。
     * 这里按「状态变化立即打 + 稳定态 5s 心跳 + 硬下限 1s」的节流输出，绝不逐帧打。
     */
    private fun logCameraRange(message: String, force: Boolean = false) {
        val now = android.os.SystemClock.elapsedRealtime()
        val changed = message != cameraRangeLogLast
        if (!force && !(changed && now >= cameraRangeLogMinGapMs) && now < cameraRangeLogHeartbeatMs) return
        cameraRangeLogLast = message
        cameraRangeLogMinGapMs = now + 1_000L
        cameraRangeLogHeartbeatMs = now + 5_000L
        android.util.Log.i("CameraRange", message)
    }

    private fun updateCameraRange(distance: Float, source: Int) {
        if (!resumed || cameraRangeSwitching) return
        val mc = multiCam
        if (mc?.active != true) {
            cameraRangeStatus = "单摄回退 · 无可调配物理镜头"
            logCameraRange("fallback: multi-camera inactive, no reconfigurable lens")
            return
        }
        if (mc.primaryProjectionMismatch) {
            logCameraRange("hold: primary projection mismatch")
            return
        }
        val now = android.os.SystemClock.elapsedRealtime()
        if (source > 0 && distance.isFinite() && distance > 0f) {
            latestScanDistanceMeters = distance
            latestScanDistanceMs = now
        }
        val desired = cameraRangePolicy.update(distance, source > 0, now)
        val observation = if (source > 0 && distance.isFinite() && distance > 0f)
            "%.2fm%s".format(distance, if (source == 2) "" else "（估计）") else "等待有效深度"
        cameraRangeStatus = "${cameraRangePolicy.mode.label}/${cameraRangePolicy.range.label} · $observation · 辅摄${mc.secondaryId}"
        if (desired == null || modelOperationBusy || openingCamera || modelViewerActive) {
            logCameraRange(
                "hold ${cameraRangePolicy.mode.label}/${cameraRangePolicy.range.label} $observation " +
                    "aux=${mc.secondaryId} reason=" +
                    (if (desired == null)
                        "policy-hold pending=${cameraRangePolicy.pendingRange?.label ?: "none"}" +
                            "/${cameraRangePolicy.pendingSampleCount}"
                    else "gate(modelBusy=$modelOperationBusy open=$openingCamera viewer=$modelViewerActive)")
            )
            return
        }
        if (::hqCapture.isInitialized && !hqCapture.canChangeFillLight()) {
            logCameraRange("hold ${desired.label} $observation reason=fill-light-busy")
            return
        }
        val nextId = mc.candidateForRange(desired, distance)
        if (nextId == null) {
            cameraRangeStatus = "${desired.label}距无可用辅助镜头 · 保留当前摄像头"
            cameraRangePolicy.failed(now)
            logCameraRange("no-lens ${desired.label} $observation: keep aux=${mc.secondaryId}")
            return
        }
        if (nextId == mc.secondaryId) {
            cameraRangePolicy.committed(desired, now)
            cameraRangeStatus = "${cameraRangePolicy.mode.label}/${desired.label} · $observation · 辅摄$nextId（共用）"
            logCameraRange(
                "steady ${cameraRangePolicy.mode.label}/${desired.label} $observation aux=$nextId(shared)"
            )
            return
        }
        logCameraRange("switch aux ${mc.secondaryId} -> $nextId for ${desired.label} $observation")
        reconfigureAuxiliaryCamera(mc, nextId, desired)
    }

    /** Rebuild physical outputs with the SAME primary reader, device and VIO world. */
    private fun reconfigureAuxiliaryCamera(mc: MultiCameraFusionController, nextId: String,
                                           desired: CameraRangePolicy.Range) {
        val camera = cameraDevice ?: return
        val previousSession = captureSession ?: return
        val handler = cameraHandler ?: return
        val previousId = mc.secondaryId ?: return
        val logical = rangeLogicalSurfaces.toList()
        if (logical.isEmpty()) return
        cameraRangeSwitching = true
        cameraRangeStatus = "切换${desired.label}距辅助镜头…"
        cameraRangeSwitchStartedMs = android.os.SystemClock.elapsedRealtime()
        mc.pausePairs()
        runCatching { previousSession.stopRepeating() }
        runCatching { previousSession.abortCaptures() }
        previousSession.close()
        captureSession = null
        var attempt = 0L
        fun start(id: String, rollback: Boolean) {
            val serial = cameraRangeSerial.incrementAndGet()
            attempt = serial
            fun current() = resumed && multiCam === mc && cameraDevice === camera &&
                serial == cameraRangeSerial.get() && serial == attempt
            fun failed() {
                if (!current()) return
                rangeTimeout?.let { handler.removeCallbacks(it) }
                cameraRangePolicy.failed(android.os.SystemClock.elapsedRealtime())
                if (!rollback) {
                    logCameraRange(
                        "switch ACCEPT-FAIL id=$id -> rollback aux=$previousId after " +
                            "${android.os.SystemClock.elapsedRealtime() - cameraRangeSwitchStartedMs}ms",
                        force = true
                    )
                    start(previousId, true)
                    return
                }
                cameraRangeSwitching = false
                cameraRangeStatus = "切换失败，已保留模型；正在恢复相机"
                logCameraRange(
                    "switch FAILED id=$id rollback-failed after " +
                        "${android.os.SystemClock.elapsedRealtime() - cameraRangeSwitchStartedMs}ms",
                    force = true
                )
                runOnUiThread {
                    if (!resumed || multiCam !== mc || cameraDevice !== camera) return@runOnUiThread
                    if (scanning) stopScan()
                    closeCamera()
                    if (texture.isAvailable) openCamera()
                    toast("辅助镜头切换失败，已停止扫描并保留模型，请重新检查定位")
                }
            }
            if (!mc.selectSecondary(id)) { failed(); return }
            val callback = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    if (!current()) { session.close(); return }
                    rangeTimeout?.let { handler.removeCallbacks(it) }
                    rangeTimeout = null
                    captureSession = session
                    cameraRangeSwitching = false
                    mc.resumePairs()
                    mc.noteSessionMode(if (rangeHqActive) "range-physical+HQ" else "range-physical")
                    if (!rollback) cameraRangePolicy.committed(desired, android.os.SystemClock.elapsedRealtime())
                    cameraRangeStatus = if (rollback) "HAL 拒绝切换 · 已恢复辅摄$previousId"
                        else "${cameraRangePolicy.mode.label}/${desired.label} · 辅摄$id · 重新标定双目"
                    logCameraRange(
                        (if (rollback) "switch rolled-back to aux=$id" else "switch OK aux=$id for ${desired.label}") +
                            " in ${android.os.SystemClock.elapsedRealtime() - cameraRangeSwitchStartedMs}ms",
                        force = true
                    )
                    applyCaptureSettings()
                    if (!rollback && cameraRangePolicy.mode != CameraRangePolicy.Mode.AUTO) {
                        handler.post { if (current()) updateCameraRange(Float.NaN, 0) }
                    }
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    session.close(); failed()
                }
            }
            rangeTimeout = Runnable { failed() }.also { handler.postDelayed(it, 4_000L) }
            if (!mc.createPhysicalSession(camera, logical, callback)) failed()
        }
        start(nextId, false)
    }

    private fun switchCamera() {
        if (scanning || modelOperationBusy || cameraRangeSwitching) {
            toast("请在扫描及模型保存完成后切换镜头")
            return
        }
        if (cameraIds.isEmpty()) return
        cameraIndex = (cameraIndex + 1) % cameraIds.size
        currentCameraId = cameraIds[cameraIndex]
        closeCamera()
        cameraHandler?.post {
            synchronized(depthSessionLock) {
                depthGeneration++
                NativeBridge.nativeDestroy()
                sessionCreated = false
            }
            runOnUiThread { if (resumed && texture.isAvailable) openCamera() }
        }
    }

    private fun showFocusStabDialog() {
        val items = arrayOf(
            "固定对焦：${if (fixedFocus) "开" else "关"}",
            "防抖：${if (stabilization) "开" else "关"}",
            "曝光锁定：${if (aeLock) "开" else "关"}${if (!aeLockAvailable) "（不支持）" else ""}",
            "白平衡锁定：${if (awbLock) "开" else "关"}${if (!awbLockAvailable) "（不支持）" else ""}"
        )
        android.app.AlertDialog.Builder(this)
            .setTitle("对焦 / 防抖 / 锁定")
            .setItems(items) { _, which ->
                when (which) {
                    0 -> fixedFocus = !fixedFocus
                    1 -> {
                        if (scanning) {
                            stabilization = false
                            toast("VIO 扫描中强制关闭 EIS/OIS")
                        } else {
                            stabilization = !stabilization
                        }
                    }
                    2 -> aeLock = aeLockAvailable && !aeLock
                    3 -> awbLock = awbLockAvailable && !awbLock
                }
                applyCaptureSettings()
                updateHeader()
            }
            .show()
    }

    private fun showCameraParams() {
        if (!ensureSystemReady()) return
        val container = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(16, 8, 16, 8)
        }
        fun row(text: String, highlight: Boolean) {
            val tv = TextView(this).apply {
                this.text = text
                setTextColor(if (highlight) android.graphics.Color.WHITE else android.graphics.Color.BLACK)
                setBackgroundColor(
                    if (highlight) android.graphics.Color.argb(255, 0, 122, 255)
                    else android.graphics.Color.TRANSPARENT
                )
                textSize = 13f
                setPadding(10, 8, 10, 8)
            }
            container.addView(tv, android.widget.LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 4 })
        }
        for (cid in cameraIds) {
            try {
                val ch = cameraManager.getCameraCharacteristics(cid)
                val facing = ch.get(CameraCharacteristics.LENS_FACING) ?: 0
                val orient = ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
                val foc = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: 0f
                val ap = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.firstOrNull() ?: 0f
                val face = if (facing == CameraCharacteristics.LENS_FACING_FRONT) "前置" else "后置"
                row("镜头$cid · $face · ${orient}° · 焦距 ${"%.2f".format(foc)}mm · f/${"%.2f".format(ap)}" +
                    if (cid == currentCameraId) "  ✓当前" else "", cid == currentCameraId)
            } catch (_: Exception) {
            }
        }
        android.app.AlertDialog.Builder(this)
            .setTitle("相机参数")
            .setView(android.widget.ScrollView(this).apply { addView(container) })
            .setPositiveButton("关闭", null)
            .show()
    }

    private fun showScanUiHelp() {
        val message =
            "取景优先的扫描界面：\n\n" +
                "• 顶部：深度 / 距离 / 尺度，点按查看完整状态\n" +
                "• 右侧：物体追踪 / 自动补光 / 预览图层，点按切换\n" +
                "• 标准扫描：按 1/4–4/4 步骤跟随轨迹环与方向箭头移动\n" +
                "• 更多：可切换标准引导 / 自由扫描，以及镜头 / 对焦 / 重建档位 / 导出\n" +
                "• 底部：开始或完成扫描 / 暂停继续 / 模型\n\n" +
                "方向箭头表示下一建议视角；保持镜头朝向物体并围绕物体移动，" +
                "不要站在原地只旋转手机。采集百分比表示采集充分度；视角百分比是估算的环绕覆盖，" +
                "均不代表真实表面积的完整度。完成扫描后仍会进入检查页，" +
                "可继续补扫或生成模型。\n\n" +
                "控件会自动避让系统栏与屏幕挖孔；中央画面保留点按对焦和物体框选。"
        android.app.AlertDialog.Builder(this)
            .setTitle("扫描界面")
            .setMessage(message)
            .setPositiveButton("知道了", null)
            .show()
    }

    private fun showViewerHelp() {
        android.app.AlertDialog.Builder(this)
            .setTitle("3D 模型查看")
            .setMessage(
                "单指拖动：旋转模型\n" +
                    "双指捏合：缩放\n" +
                    "双指一起拖动：平移\n\n" +
                    "顶部可切换纹理 / 实体 / 线框 / 缺口 / 表面观测。\n" +
                    "“缺口”模式中的红线是真实拓扑开放边，不是估算热力图。\n" +
                    "“观测”来自单次扫描 TSDF 权重；分段累计模型会自动禁用该热力。\n\n" +
                    "右侧“诊断”生成结构化质量报告；“分段”可建立基准并自动 ICP 拼接。\n" +
                    "右侧“重置”可回到自动构图；“AR摆放”会退出独立查看器，" +
                    "把模型放回真实相机画面。"
            )
            .setPositiveButton("知道了", null)
            .show()
    }

    private fun toggleAutoFillLightFromToolbar() {
        if (!torchAvailable) {
            toast("当前镜头没有可用闪光灯")
            return
        }
        autoFillLight = !autoFillLight
        getSharedPreferences("scan_settings", Context.MODE_PRIVATE).edit()
            .putBoolean("auto_fill_light", autoFillLight)
            .apply()
        if (!autoFillLight) {
            cameraHandler?.post {
                resetFillLight()
                applyCaptureSettings()
            }
        }
        toast(if (autoFillLight) "已开启自动补光" else "已关闭自动补光")
        updateHeader()
    }

    private fun applyDeviceCalibrationProfileToNative(): Boolean {
        return com.mobilescan3d.calibration.DeviceCalibrationManager.applyToNative(applicationContext)
    }

    private fun currentVinsHealth(): FloatArray? {
        val out = FloatArray(NativeBridge.VINS_HEALTH_SLOTS)
        val n = runCatching { NativeBridge.nativeGetVinsHealth(out) }.getOrDefault(0)
        return if (n == out.size) out else null
    }

    private fun showCalibrationCenter() {
        if (scanning || modelOperationBusy) {
            toast("请先停止扫描并等待模型操作完成")
            return
        }
        val manager = com.mobilescan3d.calibration.DeviceCalibrationManager
        val profile = manager.load(this)
        val health = currentVinsHealth()
        val healthText = if (health == null) {
            "当前 VINS：暂无可读状态（开始过一次定位后可采纳时延/外参）"
        } else {
            val init = health[NativeBridge.VINS_HEALTH_INITIALIZED] >= 0.5f
            val tdMs = health[NativeBridge.VINS_HEALTH_TIME_OFFSET] * 1000f
            val features = health[NativeBridge.VINS_HEALTH_FEATURES].toInt()
            val gb = health[NativeBridge.VINS_HEALTH_GYRO_BIAS]
            "当前 VINS：${if (init) "已初始化" else "未初始化"} · 特征 $features · " +
                "td ${"%.2f".format(tdMs)}ms · gyroBias ${"%.4f".format(gb)}"
        }
        val message = manager.pretty(profile) + "\n\n" + healthText +
            "\n\n说明：外参实验模式会让 VINS 在下一次会话中自行估计相机-IMU旋转；" +
            "只有完成充分旋转激励并稳定初始化后，才建议采纳当前 VINS 外参。"

        android.app.AlertDialog.Builder(this)
            .setTitle("设备标定中心 · Round 7")
            .setMessage(message)
            .setItems(
                arrayOf(
                    "编辑 / 导入标定 JSON",
                    "采纳当前 VINS 时间偏移",
                    "采纳当前 VINS 外参 + 时间偏移",
                    if (profile.estimateExtrinsic) "关闭实验外参自标定" else "开启实验外参自标定",
                    if (profile.estimateTimeOffset) "冻结当前时间偏移" else "开启在线时间偏移估计",
                    "重置为默认（RIC=I / TIC=0）"
                )
            ) { _, which ->
                when (which) {
                    0 -> showCalibrationJsonEditor()
                    1 -> {
                        val h = currentVinsHealth()
                        if (h == null) toast("当前没有 VINS 健康数据")
                        else manager.withCurrentTimeOffset(this, h)
                            .onSuccess {
                                applyDeviceCalibrationProfileToNative()
                                toast("已保存当前时延；下次重建会话生效")
                            }
                            .onFailure { toast("保存失败：${it.message}") }
                    }
                    2 -> {
                        val h = currentVinsHealth()
                        if (h == null) toast("当前没有 VINS 健康数据")
                        else manager.withCurrentVinsEstimate(this, h)
                            .onSuccess {
                                applyDeviceCalibrationProfileToNative()
                                toast("已采纳当前 VINS 外参/时延；下次重建会话生效")
                            }
                            .onFailure { toast("不能采纳：${it.message}") }
                    }
                    3 -> {
                        manager.save(this, profile.copy(
                            estimateExtrinsic = !profile.estimateExtrinsic,
                            source = if (!profile.estimateExtrinsic) "experimental-online-extrinsic" else profile.source
                        )).onSuccess {
                            applyDeviceCalibrationProfileToNative()
                            toast(if (it.estimateExtrinsic) "已开启：下一会话请做充分旋转激励" else "已关闭实验外参估计")
                        }.onFailure { toast("保存失败：${it.message}") }
                    }
                    4 -> {
                        manager.save(this, profile.copy(
                            estimateTimeOffset = !profile.estimateTimeOffset,
                            source = profile.source
                        )).onSuccess {
                            applyDeviceCalibrationProfileToNative()
                            toast(if (it.estimateTimeOffset) "已开启在线时延估计" else "已冻结时延种子")
                        }.onFailure { toast("保存失败：${it.message}") }
                    }
                    5 -> {
                        manager.reset(this)
                        applyDeviceCalibrationProfileToNative()
                        toast("已恢复默认标定；下次重建会话生效")
                    }
                }
            }
            .setPositiveButton("关闭", null)
            .show()
    }

    private fun showCalibrationJsonEditor() {
        val manager = com.mobilescan3d.calibration.DeviceCalibrationManager
        val edit = android.widget.EditText(this).apply {
            setText(manager.load(this@MainActivity).toJson().toString(2))
            setTextIsSelectable(true)
            minLines = 12
            maxLines = 20
            setHorizontallyScrolling(false)
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                android.text.InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            // 注意：onCreate 里的局部 fun dp() 在此作用域不可见，直接按 density 换算
            val editorDensity = resources.displayMetrics.density
            setPadding(
                (12 * editorDensity + .5f).toInt(), (10 * editorDensity + .5f).toInt(),
                (12 * editorDensity + .5f).toInt(), (10 * editorDensity + .5f).toInt()
            )
        }
        android.app.AlertDialog.Builder(this)
            .setTitle("标定 JSON")
            .setMessage("支持从 Kalibr/离线标定流程换算后粘贴 RIC/TIC/时延。单位：TIC=米，timeOffsetSeconds=秒。")
            .setView(android.widget.ScrollView(this).apply { addView(edit) })
            .setNegativeButton("取消", null)
            .setPositiveButton("验证并保存") { _, _ ->
                manager.parse(edit.text.toString())
                    .fold(
                        onSuccess = { parsed ->
                            manager.save(this, parsed.copy(source = "imported-json"))
                                .onSuccess {
                                    applyDeviceCalibrationProfileToNative()
                                    toast("标定 JSON 已保存；下次重建会话生效")
                                }
                                .onFailure { toast("保存失败：${it.message}") }
                        },
                        onFailure = { toast("标定 JSON 无效：${it.message}") }
                    )
            }
            .show()
    }

    private fun buildQualitySnapshot(): com.mobilescan3d.diagnostics.ScanQualityReporter.UiSnapshot =
        com.mobilescan3d.diagnostics.ScanQualityReporter.UiSnapshot(
            captureSufficiency = lastScanSufficiency,
            viewpointCoverage = lastViewpointCoveragePercent,
            surfaceCoverage = if (multiSegmentMergedActive) 0 else lastSurfaceCoveragePercent,
            surfaceRobust = if (multiSegmentMergedActive) 0 else lastSurfaceRobustPercent,
            geometryQuality = lastGeometryQualityPercent,
            textureQuality = lastTextureQualityPercent,
            angularSpeedDps = scanAngularSpeedDps,
            latestDistanceMeters = latestScanDistanceMeters,
            cumulativeMesh = multiSegmentMergedActive,
            cumulativeSegments = segmentStore.info()?.segmentCount ?: 1
        )

    /** Round 7: every successful model build gets a silent machine-readable report. */
    private fun generateQualityReportSilently() {
        if (!::exportManager.isInitialized || scanning) return
        val sid = sessionId.ifBlank { "scan" }
        val snapshot = buildQualitySnapshot()
        kotlin.concurrent.thread(name = "ScanQualityAuto", isDaemon = true) {
            runCatching {
                com.mobilescan3d.diagnostics.ScanQualityReporter.generate(
                    applicationContext, exportManager, sid, snapshot
                )
            }.onSuccess { latestQualityReport = it.json }
        }
    }

    private fun showScanQualityReport() {
        if (!::exportManager.isInitialized) return
        if (scanning || modelOperationBusy || segmentMergeBusy) {
            toast("请先停止扫描并等待当前模型操作完成")
            return
        }
        modelOperationBusy = true
        val sid = sessionId.ifBlank { "scan" }
        val snapshot = buildQualitySnapshot()
        kotlin.concurrent.thread(name = "ScanQualityReport", isDaemon = true) {
            val result = runCatching {
                com.mobilescan3d.diagnostics.ScanQualityReporter.generate(
                    applicationContext, exportManager, sid, snapshot
                )
            }
            runOnUiThread {
                modelOperationBusy = false
                result.onSuccess { r ->
                    latestQualityReport = r.json
                    android.app.AlertDialog.Builder(this)
                        .setTitle("扫描诊断 · ${r.summary}")
                        .setMessage(
                            "已生成结构化质量证据：\n${r.json.absolutePath}\n\n" +
                                "同时生成易读文本：\n${r.text.absolutePath}\n\n" +
                                "报告会把 VINS、深度标定、融合 epoch、真实表面观测、" +
                                "Mesh 拓扑和纹理统计放在同一时间点，便于对比不同扫描。"
                        )
                        .setPositiveButton("关闭", null)
                        .show()
                }.onFailure { toast("质量报告生成失败：${it.message}") }
            }
        }
    }

    private fun saveSegmentBaseAsync(
        mesh: com.mobilescan3d.export.ExportManager.MeshData,
        sid: String,
        count: Int,
        successMessage: String
    ) {
        if (segmentMergeBusy || modelOperationBusy) return
        segmentMergeBusy = true
        modelOperationBusy = true
        kotlin.concurrent.thread(name = "SegmentBaseSave", isDaemon = true) {
            val ok = segmentStore.save(mesh, sid, count)
            runOnUiThread {
                segmentMergeBusy = false
                modelOperationBusy = false
                toast(if (ok) successMessage else "保存拼接基准失败")
            }
        }
    }

    private fun showSegmentMergeDialog() {
        if (!modelViewerActive || segmentMergeBusy || modelOperationBusy) {
            toast("请先进入稳定的 3D 模型查看状态")
            return
        }
        val current = viewerBaseMesh ?: exportManager.lastMesh
        if (current == null || current.triangleCount <= 0) {
            toast("当前没有可拼接的网格")
            return
        }
        val info = segmentStore.info()
        if (info == null) {
            android.app.AlertDialog.Builder(this)
                .setTitle("分段扫描 · 建立基准")
                .setMessage(
                    "把当前 ${current.triangleCount} 面模型保存为第 1 段。\n\n" +
                        "之后开始新的扫描会话，在 3D 查看器再次点“分段拼接”，" +
                        "系统会围绕重力轴搜索初始方向并用 ICP 对齐。"
                )
                .setNegativeButton("取消", null)
                .setPositiveButton("保存为基准") { _, _ ->
                    saveSegmentBaseAsync(current, sessionId, 1, "已保存第 1 段拼接基准")
                }
                .show()
            return
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("分段拼接 · 已有 ${info.segmentCount} 段")
            .setMessage(
                "基准：${info.triangleCount} 面 / ${info.vertexCount} 顶点\n" +
                    "当前：${current.triangleCount} 面 / ${current.vertexCount} 顶点\n\n" +
                    "自动拼接会拒绝低重叠或高残差结果，不会为了“看起来连上”而强行合并。"
            )
            .setItems(arrayOf(
                "自动对齐并合并当前段",
                "手动方向辅助 + ICP…",
                "用当前模型覆盖拼接基准",
                "清除拼接基准"
            )) { _, which ->
                when (which) {
                    0 -> mergeCurrentSegmentIntoBase()
                    1 -> showManualSegmentYawDialog()
                    2 -> {
                        multiSegmentMergedActive = false
                        saveSegmentBaseAsync(current, sessionId, 1, "已用当前模型覆盖基准")
                    }
                    3 -> {
                        segmentStore.clear()
                        multiSegmentMergedActive = false
                        toast("已清除分段拼接基准")
                    }
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun showManualSegmentYawDialog() {
        val yaws = intArrayOf(0, 45, 90, 135, 180, 225, 270, 315)
        val labels = yaws.map { "约 ${it}°" }.toTypedArray()
        android.app.AlertDialog.Builder(this)
            .setTitle("手动方向辅助")
            .setMessage(
                "选择“当前段相对基准”的大致重力轴旋转。这里只提供初值，" +
                    "随后仍会用完整刚体 ICP 自动优化旋转和平移。"
            )
            .setItems(labels) { _, which -> mergeCurrentSegmentIntoBase(yaws[which]) }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun mergeCurrentSegmentIntoBase(preferredYawDeg: Int = -1) {
        val base = segmentStore.load() ?: run {
            toast("拼接基准读取失败")
            return
        }
        if (segmentMergeBusy || modelOperationBusy) return
        segmentMergeBusy = true
        modelOperationBusy = true
        tvViewerStats.text = if (preferredYawDeg >= 0) {
            "ICP 对齐 · 手动方向约 ${preferredYawDeg}°…"
        } else {
            "正在做多段 ICP 自动对齐…"
        }
        val stats = FloatArray(10)
        val transform = FloatArray(16)
        kotlin.concurrent.thread(name = "MultiScanICP", isDaemon = true) {
            val ok = runCatching {
                NativeBridge.nativeAlignCurrentMeshToReference(
                    base.mesh.vertices, base.mesh.vertexCount,
                    base.mesh.indices, base.mesh.indexCount,
                    preferredYawDeg,
                    stats, transform
                )
            }.getOrDefault(false)
            val merged = if (ok) exportManager.refreshCurrentMesh() else null
            val nextCount = base.info.segmentCount + 1
            val persisted = if (merged != null) {
                segmentStore.save(merged, "${base.info.sessionId}+${sessionId}", nextCount)
            } else false
            runOnUiThread {
                segmentMergeBusy = false
                modelOperationBusy = false
                if (!ok || merged == null) {
                    val rmseMm = stats.getOrElse(1) { 0f } * 1000f
                    val overlap = stats.getOrElse(2) { 0f } * 100f
                    tvViewerStats.text = "拼接未通过质量门"
                    android.app.AlertDialog.Builder(this)
                        .setTitle("自动拼接未通过")
                        .setMessage(
                            "重叠率：${"%.1f".format(overlap)}%\n" +
                                "RMSE：${"%.1f".format(rmseMm)} mm\n\n" +
                                "建议下一段至少保留 30% 左右的共同表面，并从上一段已扫描区域开始。"
                        )
                        .setPositiveButton("知道了", null)
                        .show()
                    return@runOnUiThread
                }

                multiSegmentMergedActive = true
                viewerBaseMesh = merged
                viewerSurfaceVertices = null
                viewerMeshOptimized = false
                syncViewerOptimizationState()
                renderer.clearTexturedMesh()
                renderer.setMesh(merged.vertices, merged.indices)
                renderer.setViewerBoundaryEdges(null)
                renderer.setViewerStyle(PointCloudRenderer.VIEWER_STYLE_SOLID)
                updateViewerModeUi(PointCloudRenderer.VIEWER_STYLE_SOLID)
                analyzeViewerMesh(merged)
                glView.requestRender()

                val rmseMm = stats[1] * 1000f
                val overlap = stats[2] * 100f
                val yaw = stats[5].toInt()
                tvViewerStats.text = "$nextCount 段累计 · ${merged.triangleCount} 面 · ICP ${"%.1f".format(rmseMm)}mm"
                android.app.AlertDialog.Builder(this)
                    .setTitle("分段拼接完成")
                    .setMessage(
                        "累计段数：$nextCount\n" +
                            "有效重叠：${"%.1f".format(overlap)}%\n" +
                            "对齐 RMSE：${"%.1f".format(rmseMm)} mm\n" +
                            "初始重力轴角度假设：${yaw}°\n\n" +
                            if (persisted) {
                                "已把结果更新为新的累计基准，可继续扫描下一段。"
                            } else {
                                "本次内存合并成功，但累计基准写盘失败；请先导出当前 GLB。"
                            }
                    )
                    .setPositiveButton("关闭", null)
                    .show()
            }
        }
    }

    private fun toggleObjectTracking() {
        objectLockEnabled = !objectLockEnabled
        NativeBridge.nativeSetObjectLockEnabled(objectLockEnabled)
        if (objectLockEnabled) {
            targetOverlay.state = TargetUiState(visible = true, state = 1)
        } else {
            NativeBridge.nativeClearTarget()
            targetOverlay.state = TargetUiState()
            targetOverlay.dragRect = null
            dragSelecting = false
            lastTargetUiState = NativeBridge.TARGET_STATE_OFF
            targetWarningText.visibility = android.view.View.GONE
        }
        toast(
            if (objectLockEnabled) "物体追踪已开启 · 点击物体或拖框选择目标"
            else "物体追踪已关闭"
        )
        updateHeader()
    }

    /** Less-used controls stay reachable without permanently covering the object. */
    private fun showSettingsMenu() {
        android.widget.PopupMenu(this, findViewById(R.id.btnSettings)!!).apply {
            menu.add(0, 14, 0, if (guidedScanEnabled) "扫描方式：标准引导" else "扫描方式：自由扫描")
            menu.add(0, 2, 1, "镜头与距离")
            menu.add(0, 3, 2, "对焦 / 防抖")
            menu.add(0, 4, 3, "重建档位")
            menu.add(0, 5, 4, "扫描距离与补光")
            menu.add(0, 6, 5, "扫描状态详情")
            menu.add(0, 7, 6, "扫描检查")
            menu.add(0, 8, 7, "导出 / 模型处理")
            val advanced = menu.addSubMenu("高级与帮助")
            advanced.add(0, 9, 0, "相机参数")
            advanced.add(0, 10, 1, "设备标定中心")
            advanced.add(0, 11, 2, "扫描质量诊断")
            advanced.add(0, 12, 3, "使用帮助")
            advanced.add(0, 13, 4, this@MainActivity.getString(R.string.privacy_data_menu))
            setOnMenuItemClickListener { item ->
                when (item.itemId) {
                    14 -> toggleGuidedScanMode()
                    2 -> showCameraRangeDialog()
                    3 -> showFocusStabDialog()
                    4 -> showModeDialog()
                    5 -> showScanSettings()
                    6 -> showScanDetails()
                    7 -> if (scanning || scanPaused) enterScanReview() else toggleModelViewer()
                    8 -> showExportDrawer()
                    9 -> showCameraParams()
                    10 -> showCalibrationCenter()
                    11 -> showScanQualityReport()
                    12 -> showScanUiHelp()
                    13 -> showPrivacyAndDataInfo()
                    else -> return@setOnMenuItemClickListener false
                }
                true
            }
            show()
        }
    }

    private fun toggleGuidedScanMode() {
        if (scanning || scanPaused) {
            toast("请结束当前扫描后再切换扫描方式")
            return
        }
        guidedScanEnabled = !guidedScanEnabled
        getSharedPreferences("scan_settings", Context.MODE_PRIVATE).edit()
            .putBoolean("guided_scan_enabled", guidedScanEnabled)
            .apply()
        scanGuidanceController.reset()
        if (::scanGuidanceOverlay.isInitialized) {
            scanGuidanceOverlay.resetGuidance()
            scanGuidanceOverlay.visibility = android.view.View.GONE
        }
        if (::tvGuidanceStep.isInitialized) tvGuidanceStep.visibility = android.view.View.GONE
        toast(if (guidedScanEnabled) "已切换为标准扫描引导" else "已切换为自由扫描")
        updateHeader()
    }

    private fun showPrivacyAndDataInfo() {
        android.app.AlertDialog.Builder(this)
            .setTitle(getString(R.string.privacy_data_title))
            .setMessage(getString(R.string.privacy_data_message))
            .setPositiveButton("关闭", null)
            .show()
    }

    private fun showScanDetails() {
        // Snapshot is captured when opened; the native scan continues normally.
        updateHeader()
        // 注意：onCreate 里的局部 fun dp() 在此作用域不可见，用局部 px() 换算
        val uiDensity = resources.displayMetrics.density
        fun px(value: Int) = (value * uiDensity + .5f).toInt()
        val panel = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(px(20), px(12), px(20), px(16))
        }
        val details = TextView(this).apply {
            textSize = 14f
            text = listOf(
                "${tvDepth.text} · ${tvUnit.text}",
                "FPS ${tvFps.text} · 距离 ${tvDistance.text}",
                "镜头 ${tvCameraMode.text} · 补光 ${toolLightLabel.text}",
                "${tvMapCount.text} · ${tvDrawCount.text}",
                tvMotionState.text.toString(),
                tvDistanceState.text.toString(),
                tvTrackingState.text.toString(),
                "物体追踪：${toolLockLabel.text}",
                "扫描方式：${if (guidedScanEnabled) "标准引导" else "自由扫描"}",
                "补扫轮次：$recoveryRound / $MAX_RECOVERY_ROUNDS",
                "重建档位：${toolModeLabel.text}",
                "采集充分度 $lastScanSufficiency% · 视角覆盖 $lastViewpointCoveragePercent%",
                "打开时的状态快照；视角覆盖为估算值。"
            ).joinToString("\n")
        }
        panel.addView(details)
        val coverage = com.mobilescan3d.ui.ViewpointCoverageView(this).apply {
            contentDescription = "估算视角覆盖 $lastViewpointCoveragePercent%"
            setCoverage(viewpointCoverage, coverageCurrentSector, lastViewpointCoveragePercent)
        }
        panel.addView(coverage, android.widget.LinearLayout.LayoutParams(px(160), px(160)).apply {
            gravity = Gravity.CENTER_HORIZONTAL
            topMargin = px(16)
        })
        android.app.AlertDialog.Builder(this)
            .setTitle("扫描状态详情")
            .setView(android.widget.ScrollView(this).apply { addView(panel) })
            .setPositiveButton("关闭", null)
            .show()
    }

    private fun showScanSettings() {
        if (scanning) {
            toast("请先停止扫描再调整距离或补光；新设置用于下一次扫描")
            return
        }
        val panel = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(36, 20, 36, 12)
        }
        val label = TextView(this)
        val slider = android.widget.SeekBar(this).apply {
            max = 48
            progress = ((scanMaxDistanceMeters - .2f) * 10f).toInt().coerceIn(0, 48)
        }
        fun updateLabel() { label.text = "最大扫描距离：%.1f 米（估计值）".format(.2f + slider.progress / 10f) }
        slider.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(bar: android.widget.SeekBar?, value: Int, fromUser: Boolean) = updateLabel()
            override fun onStartTrackingTouch(bar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(bar: android.widget.SeekBar?) {}
        })
        updateLabel()
        val light = android.widget.CheckBox(this).apply {
            text = "暗光时自动持续补光"
            isChecked = autoFillLight
        }
        panel.addView(label)
        panel.addView(slider)
        panel.addView(light)
        panel.addView(TextView(this).apply {
            text = "仅在设定距离内采集点云和模型，默认 1 米。单目距离依赖定位和深度标定，并非测距仪。补光开启后保持至停扫或退到后台，避免反复闪烁；无闪光灯的镜头无法补光。"
        })
        android.app.AlertDialog.Builder(this).setTitle("扫描距离与补光").setView(android.widget.ScrollView(this).apply { addView(panel) })
            .setNegativeButton("取消", null)
            .setPositiveButton("保存") { _, _ ->
                scanMaxDistanceMeters = .2f + slider.progress / 10f
                autoFillLight = light.isChecked
                getSharedPreferences("scan_settings", Context.MODE_PRIVATE).edit()
                    .putFloat("max_distance_m", scanMaxDistanceMeters)
                    .putBoolean("auto_fill_light", autoFillLight).apply()
            }.show()
    }

    private fun saveLatestGlb() {
        if (!::exportManager.isInitialized || modelOperationBusy || pendingModelCopy != null) {
            toast("请等待当前模型操作完成")
            return
        }
        val source = lastModelFile
        if (source == null || !source.isFile || source.length() == 0L) {
            toast("请先停止扫描并生成 GLB 模型")
            return
        }
        modelOperationBusy = true
        // Snapshot before showing the system picker so a later export cannot change it.
        kotlin.concurrent.thread(name = "PrepareGlbCopy", isDaemon = true) {
            val copy = java.io.File(cacheDir, "share_${java.util.UUID.randomUUID()}.glb")
            val error = try {
                source.copyTo(copy, overwrite = false)
                null
            } catch (t: Throwable) {
                copy.delete()
                t.message ?: "无法准备模型文件"
            }
            runOnUiThread {
                modelOperationBusy = false
                if (isDestroyed) {
                    copy.delete()
                } else if (error != null) {
                    toast(error)
                } else {
                    pendingModelCopy = copy
                    try {
                        saveModelLauncher.launch(source.name)
                    } catch (t: Throwable) {
                        pendingModelCopy = null
                        copy.delete()
                        toast("无法打开文件选择器：${t.message}")
                    }
                }
            }
        }
    }

    private fun showExportDrawer() {
        if (!ensureSystemReady()) return
        android.app.AlertDialog.Builder(this)
            .setTitle("导出")
            .setItems(
                arrayOf(
                    "导出反馈报告",
                    "导出实验 PLY（只有顶点，非最终模型）",
                    "生成网格 + 导出 GLB（常规）",
                    "生成网格 + 导出 GLB（HQ）",
                    "只生成网格（AR 预览）",
                    "保存 GLB 到文件…",
                    "硬表面规整 + 照片贴图 GLB",
                    "长方体拟合 + 照片贴图 GLB（可能补面）",
                    "正方体拟合 + 照片贴图 GLB（可能补面）"
                )
            ) { _, which ->
                when (which) {
                    0 -> exportFeedbackSafely()
                    1 -> exportModel()
                    2 -> buildMeshAndExport(sessionId, NativeBridge.MESH_QUALITY_NORMAL)
                    3 -> buildMeshAndExport(sessionId, NativeBridge.MESH_QUALITY_HQ)
                    5 -> saveLatestGlb()
                    6 -> buildMeshAndExport(sessionId, NativeBridge.MESH_QUALITY_HQ, 1)
                    7 -> buildMeshAndExport(sessionId, NativeBridge.MESH_QUALITY_HQ, 2)
                    8 -> buildMeshAndExport(sessionId, NativeBridge.MESH_QUALITY_HQ, 3)
                    else -> {
                        // 预览网格不需要每次都重建：缓存命中就直接挂上去。
                        val cached = if (::exportManager.isInitialized) exportManager.lastMesh else null
                        if (cached != null) {
                            renderer.setMesh(cached.vertices, cached.indices)
                            toast("已复用网格：${cached.triangleCount} 面")
                            glView.requestRender()
                        } else {
                            buildMeshForPreview()
                        }
                    }
                }
            }
            .show()
    }

    private fun showModeDialog() {
        if (scanning || modelOperationBusy) {
            toast("请先停止扫描并等待模型操作完成；档位用于下一次扫描")
            return
        }
        val labels = arrayOf(
            "物体精细 · 4mm 目标体素",
            "物体快速 · 8mm 目标体素",
            "房间/大场景 · 20mm 场景体素",
            "智能推荐 · 按距离与目标锁定自动选择"
        )
        android.app.AlertDialog.Builder(this)
            .setTitle("重建档位（体素设置，非测量精度）")
            .setSingleChoiceItems(labels, scanVoxelProfile) { dialog, which ->
                scanVoxelProfile = which
                getSharedPreferences("scan_settings", Context.MODE_PRIVATE).edit()
                    .putInt("voxel_profile", which).apply()
                toast("下次扫描使用：${labels[which]}")
                updateHeader()
                dialog.dismiss()
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun resolvedScanVoxelProfile(): Int {
        if (scanVoxelProfile in 0..2) return scanVoxelProfile
        return when {
            objectLockEnabled || scanMaxDistanceMeters <= 1.25f -> 0
            scanMaxDistanceMeters <= 2.4f -> 1
            else -> 2
        }
    }


    private fun remapDeviceToCamera(x: Float, y: Float, z: Float, out: FloatArray, offset: Int) {
        if (lensFacing != CameraCharacteristics.LENS_FACING_BACK) {
            out[offset] = x; out[offset + 1] = y; out[offset + 2] = z
            return
        }
        when ((sensorOrientation % 360 + 360) % 360) {
            0 -> { out[offset] = x; out[offset + 1] = -y; out[offset + 2] = -z }
            90 -> { out[offset] = -y; out[offset + 1] = -x; out[offset + 2] = -z }
            180 -> { out[offset] = -x; out[offset + 1] = y; out[offset + 2] = -z }
            270 -> { out[offset] = y; out[offset + 1] = x; out[offset + 2] = -z }
            else -> {
                val r = Math.toRadians(sensorOrientation.toDouble())
                val c = kotlin.math.cos(r).toFloat()
                val s = kotlin.math.sin(r).toFloat()
                out[offset] = c * x - s * y
                out[offset + 1] = -(s * x + c * y)
                out[offset + 2] = -z
            }
        }
    }

    override fun onSensorChanged(e: SensorEvent) {
        if (!resumed || !((scanning && scanNativeReady) || arMeshViewing)) return
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                remapDeviceToCamera(e.values[0], e.values[1], e.values[2], lastImu, 0)
                lastAccNs = e.timestamp
                hasAcc = true
                return
            }
            Sensor.TYPE_GYROSCOPE -> {
                val omegaDps = kotlin.math.sqrt(
                    e.values[0] * e.values[0] +
                        e.values[1] * e.values[1] +
                        e.values[2] * e.values[2]
                ) * 57.29578f
                scanAngularSpeedDps += (omegaDps - scanAngularSpeedDps) * 0.18f
                remapDeviceToCamera(e.values[0], e.values[1], e.values[2], lastImu, 3)
                lastGyrNs = e.timestamp
                hasGyr = true
                if (::hqCapture.isInitialized) {
                    hqCapture.onImuSample(
                        e.timestamp,
                        lastImu[3], lastImu[4], lastImu[5],
                        lastImu[0], lastImu[1], lastImu[2]
                    )
                }
            }
            else -> return
        }
        if (!hasAcc || !hasGyr) return
        val pairedNs = lastGyrNs
        if (kotlin.math.abs(lastAccNs - pairedNs) > 20_000_000L) return
        if (lastImuOutNs == 0L || pairedNs - lastImuOutNs >= 5_000_000L) {
            NativeBridge.nativeOnImu(pairedNs, lastImu[0], lastImu[1], lastImu[2], lastImu[3], lastImu[4], lastImu[5])
            lastImuOutNs = pairedNs
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun closeCamera() {
        cameraRangeSerial.incrementAndGet()
        cameraRangeSwitching = false
        rangeTimeout?.let { cameraHandler?.removeCallbacks(it) }
        rangeTimeout = null
        rangeLogicalSurfaces = emptyList()
        abandonedOpen = true
        // Closing the owning CameraDevice extinguishes its torch; serialize policy reset.
        torchRequested = false
        cameraHandler?.post { resetFillLight() }
        cameraHandler?.removeCallbacks(resetFocusRunnable)
        captureSession?.close(); captureSession = null
        cameraDevice?.close(); cameraDevice = null
        try { previewSurface?.release() } catch (_: Exception) {}
        previewSurface = null
        val mc = multiCam
        if (mc != null && mc.ownsPrimaryReader(reader)) {
            mc.close()
            reader = null
        } else {
            reader?.close()
            reader = null
            mc?.close()
        }
        multiCam = null
        if (::hqCapture.isInitialized) {
            hqCapture.close()
        }
    }

    override fun onPause() {
        // A closed camera is a discontinuity, not a pause in a continuous VIO scan.
        // Finalize an active scan. A review session may remain reviewable, but
        // supplement scanning is disabled after this discontinuity.
        if (scanning) stopScan()
        if (scanReviewActive) {
            reviewInterrupted = true
            if (::reviewContinueButton.isInitialized) reviewContinueButton.isEnabled = false
        }
        resumed = false
        if (::sensorManager.isInitialized && imuRegistered) {
            sensorManager.unregisterListener(this)
            imuRegistered = false
        }
        cancelViewerLongPress()
        viewerMultiTouch = false
        synchronized(depthSessionLock) { depthGeneration++ }
        closeCamera()
        glView.onPause()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        glView.onResume()
        if (scanReviewActive) {
            showScanReview(true)
            updateReviewUi()
        }
        if (!systemInitialized && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startSystem()
        }
        if (systemInitialized) registerImu()
        if (::cameraManager.isInitialized && texture.isAvailable && cameraDevice == null && !openingCamera) {
            openCamera()
        }
    }

    private val MAX_RECOVERY_ROUNDS = 3
    private val RECOVERY_AUTO_REVIEW_HOLD_MS = 1_600L
    private val RECOVERY_MIN_SCAN_MS = 3_500L

    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        if (modelViewerActive) {
            exitModelViewer()
            return
        }
        super.onBackPressed()
    }

    override fun onDestroy() {
        started.set(false)
        scanning = false
        scanNativeReady = false
        scanSessionToken++
        synchronized(depthSessionLock) { depthGeneration++ }
        closeCamera()
        // 权限被拒路径下 startSystem 未执行，sensorManager 可能未初始化，直接访问会崩溃
        if (::sensorManager.isInitialized) {
            sensorManager.unregisterListener(this)
        }
        // vc159 ThermalGuard：注销热状态监听。
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                thermalListener?.let { l ->
                    (getSystemService(Context.POWER_SERVICE) as? PowerManager)
                        ?.removeThermalStatusListener(l)
                }
            } catch (_: Throwable) {
            }
            thermalListener = null
        }
        synchronized(depthSessionLock) { NativeBridge.nativeDestroy() }
        sessionCreated = false
        // 权限被拒路径下 startSystem 未执行，lateinit 字段还没初始化。
        if (::depthProvider.isInitialized) {
            val provider = depthProvider
            depthHandler?.post { runCatching { provider.close() } }
        }
        if (::exportManager.isInitialized) {
            exportManager.shutdown()
        }
        cameraThread?.quitSafely()
        sensorThread?.quitSafely()
        depthThread?.quitSafely()
        super.onDestroy()
    }

    private inner class TargetLockOverlay(context: android.content.Context) : android.view.View(context) {
        var state: TargetUiState = TargetUiState()

        /** 手指拖框时的实时选框（view 像素）。null = 不显示（点按也会清掉）。 */
        var dragRect: android.graphics.RectF? = null

        private val density = resources.displayMetrics.density
        private val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 2.4f * density
            strokeCap = android.graphics.Paint.Cap.ROUND
            color = android.graphics.Color.GREEN
        }

        private val shadePaint = android.graphics.Paint().apply {
            style = android.graphics.Paint.Style.FILL
            color = android.graphics.Color.argb(88, 0, 4, 10)
        }

        private val dragPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 2f * density
            color = android.graphics.Color.YELLOW
        }

        private val labelBgPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.FILL
            color = android.graphics.Color.argb(220, 8, 18, 27)
        }

        private val labelTextPaint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = android.graphics.Color.WHITE
            textSize = 11f * resources.displayMetrics.scaledDensity
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }

        override fun onDraw(canvas: android.graphics.Canvas) {
            super.onDraw(canvas)
            if (width <= 0 || height <= 0) return

            val s = state
            if (s.visible) {
                paint.color = when {
                    s.state == NativeBridge.TARGET_STATE_TRACKING &&
                        (!s.presenceValid || s.visibleFraction < 0.30f) ->
                        android.graphics.Color.rgb(255, 93, 115)
                    s.state == NativeBridge.TARGET_STATE_TRACKING &&
                        s.visibleFraction < 0.65f ->
                        android.graphics.Color.rgb(255, 184, 77)
                    s.confidence < 0.5f ->
                        android.graphics.Color.rgb(255, 184, 77)
                    s.state == NativeBridge.TARGET_STATE_ARMED ||
                        s.state == NativeBridge.TARGET_STATE_ACQUIRING ->
                        android.graphics.Color.rgb(42, 168, 255)
                    else ->
                        android.graphics.Color.rgb(35, 232, 162)
                }

                val p0 = cameraNormToView(s.x0, s.y0)
                val p1 = cameraNormToView(s.x1, s.y0)
                val p2 = cameraNormToView(s.x1, s.y1)
                val p3 = cameraNormToView(s.x0, s.y1)
                val w = width.toFloat()
                val h = height.toFloat()
                val left = minOf(p0.x, p1.x, p2.x, p3.x) * w
                val right = maxOf(p0.x, p1.x, p2.x, p3.x) * w
                val top = minOf(p0.y, p1.y, p2.y, p3.y) * h
                val bottom = maxOf(p0.y, p1.y, p2.y, p3.y) * h

                if (right - left >= 2f && bottom - top >= 2f) {
                    // Target isolation: the subject remains visually clear while
                    // surrounding clutter is de-emphasized.
                    canvas.drawRect(0f, 0f, w, top.coerceAtLeast(0f), shadePaint)
                    canvas.drawRect(0f, bottom.coerceAtMost(h), w, h, shadePaint)
                    canvas.drawRect(0f, top.coerceAtLeast(0f), left.coerceAtLeast(0f), bottom.coerceAtMost(h), shadePaint)
                    canvas.drawRect(right.coerceAtMost(w), top.coerceAtLeast(0f), w, bottom.coerceAtMost(h), shadePaint)

                    drawBracket(canvas, left, top, right, bottom, paint)

                    val stateLabel = when (s.state) {
                        NativeBridge.TARGET_STATE_TRACKING -> "目标锁定"
                        NativeBridge.TARGET_STATE_ACQUIRING -> "正在识别"
                        else -> "选择目标"
                    }
                    val conf = (s.confidence.coerceIn(0f, 1f) * 100f).toInt()
                    val depth = if (s.medianDepth.isFinite() && s.medianDepth > 0f)
                        " · %.2fm".format(s.medianDepth) else ""
                    val label = if (s.state == NativeBridge.TARGET_STATE_TRACKING)
                        "$stateLabel · $conf%$depth" else stateLabel
                    val padX = 8f * density
                    val padY = 6f * density
                    val textW = labelTextPaint.measureText(label)
                    val labelH = labelTextPaint.textSize + padY * 1.4f
                    val labelLeft = left.coerceIn(6f * density, (w - textW - padX * 2f - 6f * density).coerceAtLeast(6f * density))
                    val labelTop = (top - labelH - 7f * density).coerceAtLeast(8f * density)
                    val rect = android.graphics.RectF(
                        labelLeft, labelTop,
                        labelLeft + textW + padX * 2f,
                        labelTop + labelH
                    )
                    canvas.drawRoundRect(rect, 9f * density, 9f * density, labelBgPaint)
                    labelTextPaint.color = paint.color
                    canvas.drawText(
                        label,
                        labelLeft + padX,
                        labelTop + labelH - padY * 0.75f,
                        labelTextPaint
                    )
                }
            }

            dragRect?.let { r ->
                if (r.width() >= 2f && r.height() >= 2f) {
                    canvas.drawRect(r, dragPaint)
                }
            }
        }

        private fun drawBracket(
            canvas: android.graphics.Canvas,
            left: Float,
            top: Float,
            right: Float,
            bottom: Float,
            p: android.graphics.Paint
        ) {
            val maxLen = 28f * density
            val len = minOf(maxLen, (right - left) * 0.24f, (bottom - top) * 0.24f)
            canvas.drawLine(left, top, left + len, top, p)
            canvas.drawLine(left, top, left, top + len, p)
            canvas.drawLine(right - len, top, right, top, p)
            canvas.drawLine(right, top, right, top + len, p)
            canvas.drawLine(left, bottom - len, left, bottom, p)
            canvas.drawLine(left, bottom, left + len, bottom, p)
            canvas.drawLine(right, bottom - len, right, bottom, p)
            canvas.drawLine(right - len, bottom, right, bottom, p)
        }
    }
}
