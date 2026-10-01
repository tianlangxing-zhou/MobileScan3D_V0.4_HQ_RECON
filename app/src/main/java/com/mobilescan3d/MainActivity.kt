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
    private lateinit var renderer: PointCloudRenderer

    private lateinit var primaryButton: android.widget.Button
    private lateinit var headerTitle: TextView
    private lateinit var headerStatus: TextView
    private lateinit var warningBanner: TextView
    private lateinit var statusBarText: TextView
    private lateinit var hudText: TextView
    private lateinit var settingsButton: android.widget.Button

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    /** V0.8: logical CameraDevice + two explicitly bound physical YUV streams. */
    private var multiCam: MultiCameraFusionController? = null
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
    private var arLayerButton: android.widget.Button? = null
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
    private var placeButton: android.widget.Button? = null
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
    private var modeLabel = "连续单帧点云"
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
            activePhysicalCameraId =
                result.get(android.hardware.camera2.CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID)
            rollingShutterSkewNs =
                result.get(android.hardware.camera2.CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW)
            exposureTimeNs =
                result.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME)
            lastCropRegion =
                result.get(android.hardware.camera2.CaptureResult.SCALER_CROP_REGION)
            lastDistortionCorrectionMode =
                result.get(android.hardware.camera2.CaptureResult.DISTORTION_CORRECTION_MODE)
            lastAfState =
                result.get(android.hardware.camera2.CaptureResult.CONTROL_AF_STATE)
            lastLensFocusDistance =
                result.get(android.hardware.camera2.CaptureResult.LENS_FOCUS_DISTANCE)
            lastAeState =
                result.get(android.hardware.camera2.CaptureResult.CONTROL_AE_STATE)
            lastAwbState =
                result.get(android.hardware.camera2.CaptureResult.CONTROL_AWB_STATE)
            multiCam?.onCaptureResult(result)
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
        val scanPrefs = getSharedPreferences("scan_settings", Context.MODE_PRIVATE)
        val savedDistance = scanPrefs.getFloat("max_distance_m", 1f)
        scanMaxDistanceMeters = if (savedDistance.isFinite()) savedDistance.coerceIn(.2f, 5f) else 1f
        autoFillLight = scanPrefs.getBoolean("auto_fill_light", true)

        val root = android.widget.FrameLayout(this)
        val density = resources.displayMetrics.density

        texture = TextureView(this)
        root.addView(texture, ViewGroup.LayoutParams(-1, -1))
        targetOverlay = TargetLockOverlay(this)
        targetOverlay.isClickable = false
        targetOverlay.isFocusable = false
        root.addView(targetOverlay, ViewGroup.LayoutParams(-1, -1))
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
                    // 物体锁定打开时才进入「可能拖框」状态；否则维持原来的点按对焦。
                    if (objectLockEnabled) {
                        dragSelecting = true
                        dragStartX = event.x
                        dragStartY = event.y
                    }
                    true
                }
                android.view.MotionEvent.ACTION_MOVE -> {
                    if (dragSelecting) {
                        // 实时反馈：让用户看到自己框的是哪一块。
                        // 只在超过最小拖拽距离后才显示，避免点按时闪一下黄框。
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
                        // 拖动 -> 精确框选
                        selectTargetRect(downX, downY, event.x, event.y)
                    } else if (dx * dx + dy * dy < 48f * 48f) {
                        // 轻点 -> 快速自动框
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
            setZOrderOnTop(true)
            isClickable = false
            isFocusable = false
        }
        // V0.13.3 模型查看器手势。glView 平时不吃触摸（isClickable=false，
        // 事件穿透到 TextureView 走点选目标/对焦）；进入查看模式时置
        // clickable，手势在这里处理：
        //   - 按下后立刻拖动 -> 旋转（yaw/pitch）
        //   - 按住不动 300ms 后再拖 -> 平移（物体跟手）
        glView.setOnTouchListener { _, event ->
            if (modelViewerActive) {
                handleModelViewerTouch(event)
                return@setOnTouchListener true
            }
            // V0.13.4 P1：自由摆放模式的手势（左右=旋转，上下=缩放）。
            if (placeModeActive) {
                handlePlacementTouch(event)
                return@setOnTouchListener true
            }
            false
        }
        root.addView(glView, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        hudText = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 11f
            setBackgroundColor(android.graphics.Color.argb(120, 0, 0, 0))
            setPadding(10, 6, 10, 6)
            text = "点云 0 点 · 非米制"
            setOnClickListener { toggleHud() }
        }
        root.addView(hudText, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START
        ).apply { leftMargin = 12; topMargin = (135 * density).toInt() })

        // 目标离屏提示。刻意做成**常驻**文案而不是 Toast：
        // 目标整块滑出画面时 bbox 退化成空矩形，连红框都画不出来，
        // 用 Toast 一闪而过就等于「什么都没发生」。
        targetWarningText = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 15f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setBackgroundColor(android.graphics.Color.argb(225, 210, 40, 40))
            setPadding((16 * density).toInt(), (10 * density).toInt(),
                       (16 * density).toInt(), (10 * density).toInt())
            gravity = Gravity.CENTER
            visibility = android.view.View.GONE
        }
        root.addView(targetWarningText, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER
        ))

        statusBarText = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 12f
            setBackgroundColor(android.graphics.Color.argb(110, 0, 0, 0))
            setPadding(16, 6, 16, 6)
            text = "--:-- · -- · --%"
        }
        root.addView(statusBarText, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP
        ))

        settingsButton = android.widget.Button(this).apply {
            text = "⚙️"
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(android.graphics.Color.argb(80, 0, 0, 0))
            setOnClickListener { showSettingsMenu() }
        }
        root.addView(settingsButton, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.END
        ).apply { topMargin = (28 * density).toInt(); rightMargin = 8 })

        val header = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setBackgroundColor(android.graphics.Color.argb(120, 0, 0, 0))
            setPadding(16, 8, 16, 8)
        }
        headerTitle = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 14f
            setTypeface(null, android.graphics.Typeface.BOLD)
            text = "MobileScan3D ${BuildConfig.VERSION_NAME} · $buildGitSha"
        }
        headerStatus = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 13f
            text = "FPS -- · 空闲"
        }
        warningBanner = TextView(this).apply {
            setTextColor(android.graphics.Color.WHITE)
            textSize = 13f
            setBackgroundColor(android.graphics.Color.argb(220, 200, 30, 30))
            setPadding(12, 8, 12, 8)
            text = "定位失锁：位姿不可用于拼接"
            visibility = android.view.View.GONE
        }
        header.addView(headerTitle)
        header.addView(headerStatus)
        header.addView(warningBanner)
        root.addView(header, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP
        ).apply { topMargin = (28 * density).toInt() })

        val bottom = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(android.graphics.Color.argb(160, 0, 0, 0))
            setPadding(12, 10, 12, 14)
        }
        primaryButton = android.widget.Button(this).apply {
            text = "实验扫描（非测量）"
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(android.graphics.Color.argb(230, 0, 122, 255))
            setPadding(20, 12, 20, 12)
            setOnClickListener { toggleScan() }
        }
        val exportButton = android.widget.Button(this).apply {
            text = "导出"
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(android.graphics.Color.argb(120, 255, 255, 255))
            setOnClickListener { showExportDrawer() }
        }
        modelViewerButton = android.widget.Button(this).apply {
            text = "查看模型"
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(android.graphics.Color.argb(200, 88, 86, 214))
            setPadding(20, 12, 20, 12)
            setOnClickListener { toggleModelViewer() }
        }
        bottom.addView(exportButton, android.widget.LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = 8 })
        bottom.addView(modelViewerButton, android.widget.LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = 8 })
        bottom.addView(primaryButton, android.widget.LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 2f))
        root.addView(bottom, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        ))

        val toolbar = android.widget.LinearLayout(this).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(android.graphics.Color.argb(90, 0, 0, 0))
            setPadding(6, 8, 6, 8)
        }
        fun toolButton(text: String, onClick: () -> Unit): android.widget.Button =
            android.widget.Button(this).apply {
                this.text = text
                setTextColor(android.graphics.Color.WHITE)
                setBackgroundColor(android.graphics.Color.argb(80, 255, 255, 255))
                textSize = 12f
                setPadding(8, 6, 8, 6)
                setOnClickListener { onClick() }
            }
        toolbar.addView(toolButton("镜头") { switchCamera() },
            android.widget.LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 6 })
        toolbar.addView(toolButton("物体锁定") {
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
            toast(if (objectLockEnabled) "点击需要扫描的物体" else "已退出物体锁定")
        },
            android.widget.LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 6 })
        toolbar.addView(toolButton("对焦/防抖") { showFocusStabDialog() },
            android.widget.LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 6 })
        // AR 图层切换：默认只画「当前帧 target depth 调试层」。
        // 先用它证明 VINS pose / 光轴 / 内参 / 竖屏旋转 / TextureView 裁剪 /
        // 时间戳同步 这一整条 AR 坐标链是对的，再切到累计点云；
        // 那时若累计点仍然散，问题就只剩 depth scale / 精度 / fusion。
        placeButton = toolButton("摆放") { togglePlacement() }
        toolbar.addView(placeButton,
            android.widget.LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 6 })
        arLayerButton = toolButton(arDrawModeLabel()) { cycleArLayer() }
        toolbar.addView(arLayerButton,
            android.widget.LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 6 })
        toolbar.addView(
            toolButton("恢复AR") {
                restoreLatestPersistentAr()
            },
            android.widget.LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )
        root.addView(toolbar, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER_VERTICAL or Gravity.END
        ).apply { rightMargin = 8 })

        settingsButton.bringToFront()
        setContentView(root)

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
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 100)
        } else {
            startSystem()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startSystem()
        }
    }

    private fun startSystem() {
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val allCameraIds = cameraManager.cameraIdList.toList()
        val rearCameraIds = allCameraIds.filter { cameraId ->
            cameraManager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
        }
        cameraIds = rearCameraIds.ifEmpty { allCameraIds }
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
        depthThread = HandlerThread("DepthInference").also { it.start() }
        depthHandler = Handler(depthThread!!.looper)
        registerImu()

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
                    if (::glView.isInitialized) glView.requestRender()
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
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val sensorHandler = Handler(sensorThread!!.looper)
        val periodUs = 5_000
        gyro?.let { sensorManager.registerListener(this, it, periodUs, 0, sensorHandler) }
        acc?.let { sensorManager.registerListener(this, it, periodUs, 0, sensorHandler) }
    }

    private fun openCamera() {
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
            lensDistortion = chars.get(CameraCharacteristics.LENS_DISTORTION)
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
                NativeBridge.nativeCreate(nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy)
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
                    if (abandonedOpen) {
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
                            if (abandonedOpen) {
                                session.close()
                                return
                            }
                            hqCapture.onSessionConfigured(hqActive)
                            captureSession = session
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
            toast("框选无效，请重新框选（框太小或未开启物体锁定）")
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
        try {
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
        if (minInterval > 0L && depthCompletedMs != 0L &&
            android.os.SystemClock.elapsedRealtime() - depthCompletedMs < minInterval) return
        if (!depthBusy.compareAndSet(false, true)) return
        val generation = depthGeneration
        try {
            NativeBridge.nativeRetainDepthFrame(t)
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
        if (scanning) stopScan() else startScan()
    }

    private fun startScan() {
        if (modelOperationBusy) {
            toast("模型正在生成或恢复，请稍候")
            return
        }
        if (cameraDevice == null || captureSession == null || nativeW <= 0 || nativeH <= 0) {
            toast("请等待相机预览就绪再开始扫描")
            return
        }
        scanning = true
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
            NativeBridge.nativeCreate(nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy)
            // 体素边长与块预算：评审 P1-2 改为按档位切换。物体扫描默认
            // OBJECT_HQ（目标 4mm / 场景 8mm），比 V0.13.3 的 0.012/0.004
            // 更细（场景 ×1.5 分辨率，面数约 ×2.25）；块预算保持一致
            // （scene 64MB + target 32MB，PLK110 可承受）。
            // 截断距离 = voxel×4（tsdf_engine 固定系数），目标侧 16mm，
            // 立体深度噪声典型 5~15mm，尚在容忍带内。
            // 注意：nativeCreate 已 reset TSDF，这里在复位后设置档位才有效。
            NativeBridge.nativeSetVoxelProfile(0)
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
            scanNativeReady = true
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
        primaryButton.text = "停止实验扫描"
        updateHeader()
    }

    /**
     * V0.12：扫描期 / 查看期的 AR 图层 + 材质切换。
     *
     * 扫描期（LIVE）要三样同时可见：半透明网格、累计 surfel（hits>=1）、
     * 当前帧 target depth 点。查看期退回「网格」档 —— 不透明 + 真实顶点色，
     * 那时要看的是几何质量，一次性点和诊断色只会干扰判断。
     */
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

    private fun stopScan() {
        scanning = false
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
        arMeshViewing = true
        aeLock = false
        awbLock = false
        captureState = "IDLE"
        if (::hqCapture.isInitialized) {
            hqCapture.endScan()
        }
        applyCaptureSettings()
        // V0.12: 停扫后回到「网格」档（不透明 + 真实顶点色）做几何验收。
        applyScanArLayer(false)
        primaryButton.text = "实验扫描（非测量）"
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
        if (scanning || modelOperationBusy) {
            toast("请先停止扫描，并等待当前模型操作完成")
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
        renderer.setMesh(mesh.vertices, mesh.indices)
        // 包围球：顶点交错布局 x,y,z 每 9 个 float 一组。几十万顶点的
        // 线性扫描在 UI 线程 ~10ms 量级，一次性成本可接受。
        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE; var minZ = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE
        val v = mesh.vertices
        var i = 0
        val n = v.size - 8
        while (i < n) {
            val x = v[i]; val y = v[i + 1]; val z = v[i + 2]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (y < minY) minY = y
            if (y > maxY) maxY = y
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
            i += 9
        }
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
        renderer.setViewerFrame(cx, cy, cz, radius)
        renderer.drawMode = PointCloudRenderer.DRAW_MODEL_VIEWER
        modelViewerActive = true
        glView.isClickable = true
        modelViewerButton?.text = "退出查看"
        glView.requestRender()
        toast("拖动=旋转 · 按住不动再拖=移动 · 「退出查看」返回相机")
    }

    /** 退出查看模式：恢复 AR 图层与触摸穿透。 */
    private fun exitModelViewer() {
        cancelViewerLongPress()
        viewerPanMode = false
        viewerMoved = false
        modelViewerActive = false
        glView.isClickable = false
        renderer.drawMode = arDrawMode
        modelViewerButton?.text = "查看模型"
        glView.requestRender()
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
        when (event.actionMasked) {
            android.view.MotionEvent.ACTION_DOWN -> {
                viewerDownX = event.x
                viewerDownY = event.y
                viewerLastX = event.x
                viewerLastY = event.y
                viewerPanMode = false
                viewerMoved = false
                val r = Runnable {
                    viewerLongPressRunnable = null
                    if (!viewerMoved) {
                        viewerPanMode = true
                        // 触觉反馈：告诉用户手势已从「旋转」切到「平移」
                        try {
                            val hh = getSystemService(android.content.Context.VIBRATOR_SERVICE)
                                    as? android.os.Vibrator
                            @Suppress("DEPRECATION")
                            hh?.vibrate(20L)
                        } catch (_: Throwable) {
                        }
                    }
                }
                viewerLongPressRunnable = r
                viewerHandler.postDelayed(r, viewerLongPressMs)
            }
            android.view.MotionEvent.ACTION_MOVE -> {
                val dx = event.x - viewerLastX
                val dy = event.y - viewerLastY
                viewerLastX = event.x
                viewerLastY = event.y
                if (!viewerMoved) {
                    val tx = event.x - viewerDownX
                    val ty = event.y - viewerDownY
                    if (tx * tx + ty * ty > 20f * 20f) {
                        viewerMoved = true
                        // 300ms 内就开始拖 = 旋转手势，长按判定作废
                        if (!viewerPanMode) cancelViewerLongPress()
                    } else if (!viewerPanMode) {
                        // 还没确定手势类型也没挪出 slop，先不动作
                        return
                    }
                }
                if (viewerPanMode) {
                    renderer.panViewer(dx, dy)
                } else {
                    renderer.rotateViewer(dx * viewerRotRadPerPx, dy * viewerRotRadPerPx)
                }
                glView.requestRender()
            }
            android.view.MotionEvent.ACTION_UP,
            android.view.MotionEvent.ACTION_CANCEL -> {
                cancelViewerLongPress()
                viewerPanMode = false
                viewerMoved = false
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
        if (::hqCapture.isInitialized) {
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
        headerStatus.text = "FPS %.1f · %s · ≤%.1fm · %s".format(
            fps, if (scanning) "扫描中" else "空闲", scanMaxDistanceMeters, lightStatus)
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

    private fun switchCamera() {
        if (scanning || modelOperationBusy) {
            toast("请在扫描及模型保存完成后切换镜头")
            return
        }
        if (cameraIds.isEmpty()) return
        cameraIndex = (cameraIndex + 1) % cameraIds.size
        currentCameraId = cameraIds[cameraIndex]
        closeCamera()
        if (texture.isAvailable) {
            // openCamera 会同步读取新镜头的 activeArray/内参标定并更新 nativeW/Fx 等，
            // 之后再投递 native 重建（与相机帧处理线程串行化，避免 reset 竞争）。
            openCamera()
            cameraHandler?.post {
                NativeBridge.nativeDestroy()
                NativeBridge.nativeCreate(nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy)
            }
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
            .setView(container)
            .setPositiveButton("关闭", null)
            .show()
    }

    private fun showSettingsMenu() {
        android.widget.PopupMenu(this, settingsButton).apply {
            menu.add("相机参数")
            menu.add("模式选择")
            menu.add("扫描距离与补光")
            setOnMenuItemClickListener { item ->
                when (item.title.toString()) {
                    "相机参数" -> showCameraParams()
                    "模式选择" -> showModeDialog()
                    "扫描距离与补光" -> showScanSettings()
                }
                true
            }
            show()
        }
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
        android.app.AlertDialog.Builder(this).setTitle("扫描距离与补光").setView(panel)
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
        val desc = "连续单帧点云：非米制，深度为相对尺度，点云仅做实时预览，不进行多帧拼接。\n\n" +
            "旧版多帧拼接：尝试按位姿累积多帧，实验性，可能产生漂移。"
        android.app.AlertDialog.Builder(this)
            .setTitle("模式选择")
            .setMessage(desc)
            .setPositiveButton("连续单帧点云") { _, _ ->
                modeLabel = "连续单帧点云"
                NativeBridge.nativeSetMode(0)
            }
            .setNegativeButton("旧版多帧拼接") { _, _ ->
                modeLabel = "旧版多帧拼接"
                NativeBridge.nativeSetMode(1)
            }
            .setNeutralButton("取消", null)
            .show()
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
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                remapDeviceToCamera(e.values[0], e.values[1], e.values[2], lastImu, 0)
                lastAccNs = e.timestamp
                hasAcc = true
                return
            }
            Sensor.TYPE_GYROSCOPE -> {
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
        resumed = false
        synchronized(depthSessionLock) { depthGeneration++ }
        closeCamera()
        glView.onPause()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        glView.onResume()
        if (::cameraManager.isInitialized && texture.isAvailable && cameraDevice == null && !openingCamera) {
            openCamera()
        }
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

        private val paint = android.graphics.Paint().apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 3f * resources.displayMetrics.density
            color = android.graphics.Color.GREEN
        }

        private val dragPaint = android.graphics.Paint().apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 2f * resources.displayMetrics.density
            color = android.graphics.Color.YELLOW
        }

        override fun onDraw(canvas: android.graphics.Canvas) {
            super.onDraw(canvas)
            if (width <= 0 || height <= 0) return

            val s = state
            if (s.visible) {
                paint.color = when {
                    s.state == NativeBridge.TARGET_STATE_LOST ->
                        android.graphics.Color.RED
                    s.state == NativeBridge.TARGET_STATE_REACQUIRING ->
                        android.graphics.Color.rgb(255, 160, 0)
                    // V0.12: 与常驻提示的两级阈值对齐（<0.30 红 / <0.65 黄）。
                    // 框的颜色是最快被余光捕捉到的信号，不能和文字说两套话。
                    s.state == NativeBridge.TARGET_STATE_TRACKING &&
                        s.visibleFraction < 0.30f ->
                        android.graphics.Color.RED
                    s.state == NativeBridge.TARGET_STATE_TRACKING &&
                        s.visibleFraction < 0.65f ->
                        android.graphics.Color.YELLOW
                    s.confidence < 0.5f ->
                        android.graphics.Color.YELLOW
                    else ->
                        android.graphics.Color.GREEN
                }

                // bbox 是**相机归一化**坐标，canvas 是 **view** 坐标系。
                // 两者之间隔着 sensorOrientation 90° + SurfaceTexture transform
                // + TextureView center crop，直接 `x0 * width` 是错的。
                // 这里复用 Renderer 那套已验证的 cameraToView：四个角都映射，
                // 再取轴对齐包围盒 —— 旋转/翻转/裁剪全部被这套变换吸收，
                // 画框处不需要知道任何屏幕方向细节。
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
                // 目标整块出界时 bbox 退化成 0x0，此时不画框，
                // 由常驻的 targetWarningText 负责提示。
                if (right - left >= 2f && bottom - top >= 2f) {
                    canvas.drawRect(left, top, right, bottom, paint)
                }
            }

            dragRect?.let { r ->
                if (r.width() >= 2f && r.height() >= 2f) {
                    canvas.drawRect(r, dragPaint)
                }
            }
        }
    }
}
