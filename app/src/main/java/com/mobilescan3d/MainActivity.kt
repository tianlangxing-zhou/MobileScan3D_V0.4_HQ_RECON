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
import android.os.HandlerThread
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.TextView
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity(), SensorEventListener {

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

    @Volatile
    private var depthBusy = false

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
            } catch (e: Exception) {
                android.widget.Toast.makeText(this, "保存失败：${e.message}", android.widget.Toast.LENGTH_LONG).show()
            }
        }
    }
    private var previewSurface: Surface? = null
    private var scanning = false
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
    private val depthScaleTargetBuf = FloatArray(10)
    // 槽位与 native_engine.cpp 的 kDepthDiagSlots 一致：
    //   0 P10  1 median  2 P90  3 vinsMedian
    //   4 validPixels  5 sampleCount  6 roiArea
    private val depthScaleDepthBuf = FloatArray(7)
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
        val medianDepth: Float = 0f
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
                    frameMeta[sensorTs] = FrameMeta(exposure, skew, crop, physicalId)
                    while (frameMeta.size > 64) {
                        val key = frameMeta.entries.first().key
                        frameMeta.remove(key)
                    }
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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
                    true
                }
                android.view.MotionEvent.ACTION_UP -> {
                    val dx = event.x - downX
                    val dy = event.y - downY
                    if (dx * dx + dy * dy < 48f * 48f) {
                        if (objectLockEnabled) {
                            selectTarget(event.x, event.y)
                        } else {
                            focusAt(event.x, event.y)
                        }
                    }
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
            text = "MobileScan3D 0.5.1 · $buildGitSha"
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
        bottom.addView(exportButton, android.widget.LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = 8 })
        bottom.addView(primaryButton, android.widget.LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 3f))
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
            }
            toast(if (objectLockEnabled) "点击需要扫描的物体" else "已退出物体锁定")
        },
            android.widget.LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 6 })
        toolbar.addView(toolButton("对焦/防抖") { showFocusStabDialog() },
            android.widget.LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(toolbar, android.widget.FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER_VERTICAL or Gravity.END
        ).apply { rightMargin = 8 })

        setContentView(root)

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
        depthProvider = DepthProvider(assets)
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
                    val f = stMatrixFloats
                    // 4x4 中作用于 UV 的 2D 仿射部分：
                    // u' = m0·u + m4·v + m12 ; v' = m1·u + m5·v + m13
                    val na = floatArrayOf(f[0], f[4], f[12], f[1], f[5], f[13])
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

            reader = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 4).apply {
                setOnImageAvailableListener({ r -> r.acquireLatestImage()?.let { processImage(it) } }, cameraHandler)
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

                        fun finishSession(session: CameraCaptureSession, hqActive: Boolean) {
                            if (abandonedOpen) {
                                session.close()
                                return
                            }
                            hqCapture.onSessionConfigured(hqActive)
                            captureSession = session
                            // 会话配置完成后再应用一次变换：首个帧到达时
                            // 部分 HAL 会重置 SurfaceTexture 的 transform
                            texture.post { configureTransform(texture.width, texture.height) }
                            texture.postDelayed({ configureTransform(texture.width, texture.height) }, 300L)
                            applyCaptureSettings()
                            started.set(true)
                        }

                        camera.createCaptureSession(primarySurfaces, object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(session: CameraCaptureSession) {
                                finishSession(session, true)
                            }

                            override fun onConfigureFailed(session: CameraCaptureSession) {
                                hqCapture.detachSurfaces()
                                camera.createCaptureSession(baseSurfaces, object : CameraCaptureSession.StateCallback() {
                                    override fun onConfigured(fallback: CameraCaptureSession) {
                                        finishSession(fallback, false)
                                    }

                                    override fun onConfigureFailed(fallback: CameraCaptureSession) {
                                        toast("Camera session failed")
                                    }
                                }, cameraHandler)
                            }
                        }, cameraHandler)
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
        val session = captureSession ?: return
        val camera = cameraDevice ?: return
        val readerSurface = reader?.surface ?: return
        val ps = previewSurface ?: return
        try {
            val req = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                addTarget(ps)
                addTarget(readerSurface)
                // Preview 的 3A 全程保持自动。
                // HQ 采集只要求「收敛」，不要求「锁定」；真正把参数冻住的地方是
                // HqCaptureController.buildLockedStillRequest()（still request）。
                // 这里绝不能再跟随控制器写 AE_MODE_OFF / 手动曝光 / 手动焦距——
                // 那会把 AE_STATE 打成 INACTIVE，让控制器自己的收敛判定永远不成立。
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
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
                    set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                    focusRegion?.let { region ->
                        set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(region))
                        set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(region))
                    }
                    set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)
                }.build()
                session.capture(trigger, null, cameraHandler)
            }
            session.setRepeatingRequest(req, captureResultCallback, cameraHandler)
        } catch (e: Exception) {
            toast("应用相机参数失败：${e.message}")
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
        updateTargetOverlay()
    }

    private fun updateTargetOverlay() {
        val out = FloatArray(10)
        val state = NativeBridge.nativeGetTargetState(out)
        targetState = state
        targetConfidence = out.getOrElse(5) { 0f }
        targetTrackedPoints = out.getOrElse(8) { 0f }.toInt()
        targetOverlay.state = TargetUiState(
            visible = state != 0,
            state = state,
            x0 = out.getOrElse(1) { 0f },
            y0 = out.getOrElse(2) { 0f },
            x1 = out.getOrElse(3) { 0f },
            y1 = out.getOrElse(4) { 0f },
            confidence = out.getOrElse(5) { 0f },
            medianDepth = out.getOrElse(6) { 0f }
        )
        targetOverlay.invalidate()
        maybeRelockTargetFocus()
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

    private fun generateReport() {
        val sb = StringBuilder()
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
        if (previewRot != 0) warns.add("DISPLAY_ROTATION_NOT_APPLIED")
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
        val out = FloatArray(10)
        val state = NativeBridge.nativeGetTargetState(out)
        sb.appendLine("    state=$state")
        sb.appendLine("    bboxCamera=(${out.getOrElse(1) { 0f }}, ${out.getOrElse(2) { 0f }}, ${out.getOrElse(3) { 0f }}, ${out.getOrElse(4) { 0f }})")
        sb.appendLine("    confidence=${out.getOrElse(5) { 0f }}")
        sb.appendLine("    medianDepth=${out.getOrElse(6) { 0f }}")
        sb.appendLine("    roiSharpness=${out.getOrElse(7) { 0f }}")
        sb.appendLine("    trackedPoints=${out.getOrElse(8) { 0f }.toInt()}")
        sb.appendLine("    inlierRatio=${out.getOrElse(9) { 0f }}")
        sb.appendLine("    AF state=$lastAfState lensFocusDistance=$lastLensFocusDistance focusLocked=$targetFocusLocked relockCount=$focusRelockCount")
        sb.appendLine(NativeBridge.nativeGetTargetDiagnostics())
        sb.appendLine()
        // ---- 深度尺度诊断：只上报可观测量，不自动乘 scale ----
        val dd = FloatArray(4)
        NativeBridge.nativeGetDepthDiagnostics(dd)
        val targetDepthP10 = dd[0]
        val targetDepthMedian = if (dd[1] > 0f) dd[1] else out.getOrElse(6) { 0f }
        val targetDepthP90 = dd[2]
        val vinsTriangulatedDepthMedian = dd[3]
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
        sb.appendLine("    targetDepthValidPixels=${dd[4].toInt()}")
        sb.appendLine("    targetDepthSampleCount=${dd[5].toInt()}")
        sb.appendLine("    targetDepthRoiArea=${dd[6].toInt()}")
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
        val renderPose = FloatArray(12)
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
        sb.appendLine("    displayRotationApplied=false")
        sb.appendLine("    actualPreviewRot=$previewRot")
        sb.appendLine("    autoYaw=false")
        sb.appendLine("    touchOrbit=false")

        pendingReport = sb.toString()
        createReportLauncher.launch("config_${sessionId}.txt")
    }

    private fun processImage(image: Image) {
        val ts = image.timestamp
        try {
            val p = image.planes
            val y = extractPlane(p[0])
            val u = extractPlane(p[1])
            val v = extractPlane(p[2])
            if (scanning) {
                // 曝光直方图预检：clipping 过高时不要发 still capture
                if (::hqCapture.isInitialized) {
                    hqCapture.onPreviewLuma(y, image.width, image.height, p[0].rowStride)
                }
                val meta = synchronized(frameMetaLock) { frameMeta.remove(ts) }
                if (meta != null) frameMetaHit++ else frameMetaMiss++
                val vinsTs = ts
                NativeBridge.nativeOnCameraFrame(y, u, v, image.width, image.height, p[0].rowStride, p[1].rowStride, p[1].pixelStride, ts, vinsTs)
                scheduleDepth(y, u, v, image.width, image.height, p[0].rowStride, p[1].rowStride, p[1].pixelStride, ts)
                glView.requestRender()
            }
        } catch (_: Throwable) {
        } finally {
            image.close()
        }
        onFrameTick(ts)
    }

    private fun scheduleDepth(y: ByteArray, u: ByteArray, v: ByteArray, w: Int, h: Int, rowStride: Int, uRowStride: Int, uPixelStride: Int, t: Long) {
        if (depthBusy) return
        depthBusy = true
        depthHandler?.post {
            try {
                val depth = FloatArray(w * h)
                depthProvider.estimate(y, u, v, w, h, rowStride, uRowStride, uPixelStride, depth)
                NativeBridge.nativeOnDepthMap(depth, w, h, 0.5f, t)
                glView.requestRender()
            } catch (_: Throwable) {
            } finally {
                depthBusy = false
            }
        }
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
        scanning = true
        sessionStartTs = System.currentTimeMillis()
        val formatter = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
        sessionId = formatter.format(java.util.Date()) + "_" + (sessionStartTs % 100000L)
        lastPlyFilename = null
        lastPlyVertexCount = null
        lastPlyFileBytes = null
        lastPlyExportTs = null
        lastPlySessionId = null
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
        cameraHandler?.post {
            NativeBridge.nativeDestroy()
            NativeBridge.nativeCreate(nativeW, nativeH, nativeFx, nativeFy, nativeCx, nativeCy)
            if (objectLockEnabled) {
                NativeBridge.nativeSetObjectLockEnabled(true)
            }
            val backbone = materializeAsset("nanotrack_backbone_sim.onnx")
            val head = materializeAsset("nanotrack_head_sim.onnx")
            NativeBridge.nativeConfigureTrackerModels(backbone, head)
            sessionCreated = true
        }
        primaryButton.text = "停止实验扫描"
        updateHeader()
    }

    private fun stopScan() {
        scanning = false
        aeLock = false
        awbLock = false
        captureState = "IDLE"
        if (::hqCapture.isInitialized) {
            hqCapture.endScan()
        }
        applyCaptureSettings()
        primaryButton.text = "实验扫描（非测量）"
        exportModel()
    }

    private fun exportModel() {
        val dir = getExternalFilesDir(null) ?: filesDir
        val filename = "scan_${sessionId}.ply"
        val path = "${dir.absolutePath}/$filename"
        val ok = NativeBridge.nativeExportPly(path)
        if (ok) {
            val plyFile = java.io.File(path)
            lastPlyFilename = filename
            lastPlySessionId = sessionId
            lastPlyVertexCount = readPlyVertexCount(plyFile)
            lastPlyFileBytes = plyFile.length()
            lastPlyExportTs = System.currentTimeMillis()
        }
        android.widget.Toast.makeText(
            this,
            if (ok) "模型已导出：$path" else "导出失败（模型数据不足）",
            android.widget.Toast.LENGTH_LONG
        ).show()
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

    private fun readPlyVertexCount(file: java.io.File): Int? {
        if (!file.exists()) return null
        file.bufferedReader().use { br ->
            repeat(32) {
                val line = br.readLine() ?: return null
                if (line.startsWith("element vertex ")) {
                    return line.substringAfter("element vertex ").trim().toIntOrNull()
                }
                if (line.trim() == "end_header") return null
            }
        }
        return null
    }

    private fun toast(msg: String) {
        runOnUiThread {
            android.widget.Toast.makeText(this, msg, android.widget.Toast.LENGTH_SHORT).show()
        }
    }

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
        if (::hqCapture.isInitialized) {
            hqCapture.onFrameTick(ts, scanning, cameraDevice, captureSession)
            syncCaptureStateFromController()
        }
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
            runOnUiThread {
                fps = f
                updateHeader()
                updateStatusBar()
            }
        }
    }

    private fun updateHeader() {
        val vinsOk = NativeBridge.nativeVinsInitialized()
        headerStatus.text = "FPS %.1f · %s".format(fps, if (scanning) "扫描中" else "空闲")
        if (objectLockEnabled) {
            updateTargetOverlay()
        }
        warningBanner.visibility =
            if (scanning && !vinsOk) android.view.View.VISIBLE else android.view.View.GONE
        val target = NativeBridge.nativeGetPointCount().toFloat()
        hudShownPoints += (target - hudShownPoints) * 0.35f
        val shown = hudShownPoints.toInt()
        hudText.text = if (hudExpanded) {
            "点云 $shown 点 · 世界锁定\n视角跟随 VINS 相机"
        } else {
            "点云 $shown 点 · 非米制"
        }
    }

    private fun updateStatusBar() {
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

    private fun switchCamera() {
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
            setOnMenuItemClickListener { item ->
                when (item.title.toString()) {
                    "相机参数" -> showCameraParams()
                    "模式选择" -> showModeDialog()
                }
                true
            }
            show()
        }
    }

    private fun showExportDrawer() {
        android.app.AlertDialog.Builder(this)
            .setTitle("导出")
            .setItems(arrayOf("导出反馈报告", "导出实验 PLY")) { _, which ->
                if (which == 0) generateReport() else exportModel()
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
        cameraHandler?.removeCallbacks(resetFocusRunnable)
        captureSession?.close(); captureSession = null
        cameraDevice?.close(); cameraDevice = null
        try { previewSurface?.release() } catch (_: Exception) {}
        previewSurface = null
        reader?.close(); reader = null
        if (::hqCapture.isInitialized) {
            hqCapture.close()
        }
    }

    override fun onPause() {
        resumed = false
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
        // 权限被拒路径下 startSystem 未执行，sensorManager 可能未初始化，直接访问会崩溃
        if (::sensorManager.isInitialized) {
            sensorManager.unregisterListener(this)
        }
        NativeBridge.nativeDestroy()
        sessionCreated = false
        cameraThread?.quitSafely()
        sensorThread?.quitSafely()
        depthThread?.quitSafely()
        super.onDestroy()
    }

    private inner class TargetLockOverlay(context: android.content.Context) : android.view.View(context) {
        var state: TargetUiState = TargetUiState()
        private val paint = android.graphics.Paint().apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 3f * resources.displayMetrics.density
            color = android.graphics.Color.GREEN
        }

        override fun onDraw(canvas: android.graphics.Canvas) {
            super.onDraw(canvas)
            val s = state
            if (!s.visible || width <= 0 || height <= 0) return

            paint.color = when {
                s.state == 3 -> android.graphics.Color.GREEN
                s.state == 4 -> android.graphics.Color.RED
                s.confidence < 0.5f -> android.graphics.Color.YELLOW
                else -> android.graphics.Color.GREEN
            }

            val left = s.x0 * width
            val top = s.y0 * height
            val right = s.x1 * width
            val bottom = s.y1 * height
            canvas.drawRect(left, top, right, bottom, paint)
        }
    }
}
