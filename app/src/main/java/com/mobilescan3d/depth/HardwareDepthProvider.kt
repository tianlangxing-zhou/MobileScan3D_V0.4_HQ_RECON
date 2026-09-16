package com.mobilescan3d.depth

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface

/**
 * 硬件深度（ToF / 双目）：设备自己吐 DEPTH16。
 *
 * ## 为什么它只是「第二选择」
 *
 * 评审明确要求：**不能把硬件深度当成唯一方案**。
 *   - 绝大多数手机根本没有 `REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT`，
 *     只有少数带 ToF / 主动双目（iPad Pro、部分三星 / 华为旗舰）才有；
 *   - 有硬件深度的设备，其深度相机往往**不是** RGB 主摄那个 id，
 *     内参与外参都需要额外标定才能和 RGB 对齐，直接在 AR 里用会错位。
 *
 * 所以本类被设计成「独立会话 + 独立 ImageReader」：
 * 它**不去碰主 RGB 的 CaptureSession**（那条链是实机验收过的 HQ 采集流程，
 * 动它是纯风险），而是自己打开深度相机 id、自己收图、自己解码成米制。
 * 代价是与 RGB 不同步 —— 而不同步这一件事，native 侧的深度标定
 * （MAD + Huber IRLS + EMA）与时序一致性检查本来就要处理，不是新增问题。
 *
 * ## DEPTH16 的数值约定
 *
 * 16 位无符号。按 Android 的通用约定按
 *     `meters = value / 65535 * depthRangeMm / 1000`
 * 换算。当 `depthRangeMm = 65535`（绝大多数设备，即「1 单位 = 1mm」）时
 * 公式退化成 `value / 1000f`，就是常见的毫米直读。
 * 保留 `depthRangeMm` 参数是为了覆盖少数量程自定义的设备 ——
 * **这一项需要实机核对**，取错了会让深度整体差一个常数因子，
 * 而那恰好是 native 侧鲁棒标定能兜住的误差。
 *
 * 线程约定：与 [DepthProvider] 接口一致，所有方法都由同一个专用线程调用。
 * 内部的 ImageReader 回调跑在自己起的 HandlerThread 上，只做「解码 + 存 latest」。
 */
class HardwareDepthProvider(
    context: Context,
    private val cameraId: String,
    private val depthWidth: Int = 320,
    private val depthHeight: Int = 240,
    /** DEPTH16 满量程对应的毫米数；65535 即「1 单位 = 1mm」。 */
    private val depthRangeMm: Float = 65535f
) : DepthProvider {

    private val appContext = context.applicationContext

    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var thread: HandlerThread? = null
    private var handler: Handler? = null

    @Volatile
    private var running = false

    @Volatile
    private var latestResult: DepthProvider.Result? = null

    /** 复用缓冲，避免每帧 new（深度帧率不高，但没必要制造 GC 压力）。 */
    private var buffer = FloatArray(0)
    private var bufferW = 0
    private var bufferH = 0

    @Volatile
    private var frameCount = 0L

    @Volatile
    private var lastError: String = ""

    override val backendName: String = "hardware-depth16-cam$cameraId"

    override val available: Boolean
        get() = running && frameCount > 0L

    /** 诊断用：最近一次失败原因。 */
    val errorText: String get() = lastError

    /**
     * 打开深度相机并开始收图。幂等；失败时返回 false 并把原因写进 [errorText]。
     */
    fun start(): Boolean {
        if (running) return true
        val manager = appContext.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
        if (manager == null) {
            lastError = "CameraManager 不可用"
            return false
        }
        return try {
            val chars = manager.getCameraCharacteristics(cameraId)
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            if (caps == null ||
                !caps.contains(
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT
                )
            ) {
                lastError = "camera $cameraId 不支持 DEPTH_OUTPUT"
                return false
            }
            // 用设备真正支持的尺寸，而不是硬编码。取不到就退回请求尺寸，
            // 让 createCaptureSession 自己报错，比静默用错尺寸好。
            var size = Size(depthWidth, depthHeight)
            chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?.getOutputSizes(ImageFormat.DEPTH16)
                ?.let { sizes ->
                    if (sizes.isNotEmpty()) {
                        size = sizes.minByOrNull { it.width * it.height } ?: sizes[0]
                    }
                }

            val ht = HandlerThread("HardwareDepth").apply { start() }
            thread = ht
            val h = Handler(ht.looper)
            handler = h

            val ir = ImageReader.newInstance(size.width, size.height, ImageFormat.DEPTH16, 3)
            ir.setOnImageAvailableListener({ r -> onDepthImage(r) }, h)
            reader = ir
            running = true

            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    camera = device
                    try {
                        val surface = ir.surface as Surface
                        val req = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(surface)
                        }
                        device.createCaptureSession(
                            listOf(surface),
                            object : CameraCaptureSession.StateCallback() {
                                override fun onConfigured(s: CameraCaptureSession) {
                                    session = s
                                    try {
                                        s.setRepeatingRequest(req.build(), null, h)
                                    } catch (t: Throwable) {
                                        running = false
                                        lastError = "setRepeatingRequest 失败: ${t.message}"
                                        Log.e(TAG, lastError, t)
                                    }
                                }

                                override fun onConfigureFailed(s: CameraCaptureSession) {
                                    running = false
                                    lastError = "深度会话配置失败"
                                    Log.e(TAG, lastError)
                                }
                            },
                            h
                        )
                    } catch (t: Throwable) {
                        running = false
                        lastError = "创建深度会话失败: ${t.message}"
                        Log.e(TAG, lastError, t)
                    }
                }

                override fun onDisconnected(device: CameraDevice) {
                    device.close()
                    camera = null
                    running = false
                    lastError = "深度相机断开"
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                    camera = null
                    running = false
                    lastError = "深度相机错误码 $error"
                }
            }, h)
            true
        } catch (t: Throwable) {
            running = false
            lastError = "打开深度相机异常: ${t.javaClass.simpleName}: ${t.message}"
            Log.e(TAG, lastError, t)
            false
        }
    }

    private fun onDepthImage(r: ImageReader) {
        val image = try {
            r.acquireLatestImage()
        } catch (t: Throwable) {
            null
        } ?: return
        try {
            submitDepthImage(image, image.timestamp)
        } catch (t: Throwable) {
            lastError = "解码 DEPTH16 失败: ${t.message}"
        } finally {
            image.close()
        }
    }

    /**
     * 本实现不吃 YUV 帧 —— RGB 主链的帧与深度相机的帧本来就不是同一路。
     */
    override fun submitFrame(
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int,
        rowStride: Int, uRowStride: Int, uPixelStride: Int,
        timestampNs: Long
    ): Boolean = false

    /**
     * 把一帧 DEPTH16 解码成米制深度。
     *
     * `timestampNs` 用**调用方给的相机帧时间戳**，而不是 `image.timestamp`：
     * 深度相机与 RGB 相机是两个独立的 sensor clock，直接拿深度的时间戳去
     * native 侧查 VINS pose 是查不到的（native 按 SENSOR_TIMESTAMP 索引）。
     */
    override fun submitDepthImage(image: Image, timestampNs: Long): Boolean {
        if (image.format != ImageFormat.DEPTH16) return false
        val w = image.width
        val h = image.height
        if (w <= 0 || h <= 0) return false

        val plane = image.planes[0]
        val rowStrideBytes = plane.rowStride
        if (rowStrideBytes < w * 2) return false
        val rowStrideShorts = rowStrideBytes / 2
        val shorts = plane.buffer.order(java.nio.ByteOrder.nativeOrder()).asShortBuffer()

        if (bufferW != w || bufferH != h) {
            buffer = FloatArray(w * h)
            bufferW = w
            bufferH = h
        }
        val rangeMm = if (depthRangeMm > 1f) depthRangeMm else 65535f
        val unitToMeters = rangeMm / 65535f / 1000f

        // DEPTH16 的第二个 plane（如果有）是 8 位置信度图。
        // 只有标定为 DEPTH_CALIBRATION 的设备才有，取不到就置 null。
        var conf: FloatArray? = null
        try {
            if (image.planes.size > 1) {
                val cp = image.planes[1]
                val cb = cp.buffer
                if (cb.remaining() >= w * h) {
                    val arr = FloatArray(w * h)
                    val base = cb.position()
                    val cRowStride = cp.rowStride
                    val cPixelStride = if (cp.pixelStride > 0) cp.pixelStride else 1
                    var oy = 0
                    while (oy < h) {
                        val ro = base + oy * cRowStride
                        var ox = 0
                        while (ox < w) {
                            val idx = ro + ox * cPixelStride
                            if (idx >= base + cb.limit()) break
                            arr[oy * w + ox] = (cb.get(idx).toInt() and 0xFF) / 255f
                            ++ox
                        }
                        ++oy
                    }
                    conf = arr
                }
            }
        } catch (t: Throwable) {
            conf = null
        }

        var valid = 0
        var y = 0
        while (y < h) {
            val ro = y * rowStrideShorts
            val base = y * w
            var x = 0
            while (x < w) {
                val raw = shorts.get(ro + x).toInt() and 0xFFFF
                if (raw == 0) {
                    // 0 表示该像素无有效深度（Android 约定）
                    buffer[base + x] = 0f
                } else {
                    buffer[base + x] = raw * unitToMeters
                    ++valid
                }
                ++x
            }
            ++y
        }
        if (valid <= 0) return false

        ++frameCount
        latestResult = DepthProvider.Result(
            depth = buffer,
            width = w,
            height = h,
            confidence = conf,
            timestampNs = timestampNs,
            metric = true,
            backend = backendName
        )
        return true
    }

    override fun latest(): DepthProvider.Result? = latestResult

    override fun reset() {
        latestResult = null
        frameCount = 0L
    }

    override fun close() {
        running = false
        latestResult = null
        try {
            session?.close()
        } catch (_: Throwable) {
        }
        try {
            camera?.close()
        } catch (_: Throwable) {
        }
        try {
            reader?.close()
        } catch (_: Throwable) {
        }
        session = null
        camera = null
        reader = null
        handler = null
        thread?.quitSafely()
        thread = null
    }

    companion object {
        private const val TAG = "HardwareDepth"
    }
}
