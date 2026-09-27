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
 * ## DEPTH16
 * 低 13 位为毫米深度，高 3 位为置信度编码，见 Android ImageFormat.DEPTH16。
 * 每次发布独立数组，避免相机回调改写读者正在使用的深度。
 *
 * 线程约定：与 [DepthProvider] 接口一致，所有方法都由同一个专用线程调用。
 * 内部的 ImageReader 回调跑在自己起的 HandlerThread 上，只做「解码 + 存 latest」。
 */
class HardwareDepthProvider(
    context: Context,
    private val cameraId: String,
    private val depthWidth: Int = 320,
    private val depthHeight: Int = 240
) : DepthProvider {

    private val appContext = context.applicationContext

    private val lifecycle = java.util.concurrent.atomic.AtomicLong(0)
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var thread: HandlerThread? = null
    private var handler: Handler? = null

    @Volatile
    private var running = false

    @Volatile
    private var latestResult: DepthProvider.Result? = null

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
        close()
        val generation = lifecycle.incrementAndGet()
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
                    if (!running || generation != lifecycle.get()) { device.close(); return }
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
                                    if (!running || generation != lifecycle.get()) { s.close(); return }
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
                                    s.close()
                                    if (generation != lifecycle.get()) return
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
                    if (generation != lifecycle.get()) return
                    camera = null
                    running = false
                    lastError = "深度相机断开"
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                    if (generation != lifecycle.get()) return
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
            if (running && r === reader) submitDepthImage(image, image.timestamp)
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
     * 保留传感器时间戳。接入 RGB 融合前必须完成时钟、内参与外参对齐；
     * 不得把异步深度帧伪装成当前 RGB 帧。默认工厂仍使用单目模型。
     */
    @Synchronized
    override fun submitDepthImage(image: Image, timestampNs: Long): Boolean {
        if (!running) return false
        if (image.format != ImageFormat.DEPTH16) return false
        val w = image.width
        val h = image.height
        if (w <= 0 || h <= 0) return false

        val plane = image.planes.firstOrNull() ?: return false
        val decoded = Depth16Decoder.decode(plane.buffer, w, h, plane.rowStride, plane.pixelStride)
            ?: return false
        if (decoded.depth.none { it > 0f }) {
            latestResult = null
            return false
        }

        ++frameCount
        latestResult = DepthProvider.Result(
            depth = decoded.depth,
            width = w,
            height = h,
            confidence = decoded.confidence,
            timestampNs = timestampNs,
            metric = true,
            backend = backendName
        )
        return true
    }

    override fun latest(): DepthProvider.Result? = latestResult

    @Synchronized
    override fun reset() {
        latestResult = null
        frameCount = 0L
    }

    @Synchronized
    override fun close() {
        lifecycle.incrementAndGet()
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
