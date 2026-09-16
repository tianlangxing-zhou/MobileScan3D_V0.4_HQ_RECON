package com.mobilescan3d.depth

import android.content.res.AssetManager
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * 单目深度：assets/depth_model.tflite（256x256 输入）。
 *
 * **保持与改造前逐位一致的行为**：会话级 min/max 归一化 + 线性映射到
 * [NEAR, FAR]。这是刻意保留的 —— 第八轮刚在实机上验收过基于这张深度图的
 * 目标 mask 与 presence gate，这里一旦改了数值分布，那部分等于没验证过。
 *
 * 它输出的是 **相对深度（metric = false）**：会话 min/max 会把「模型原始输出」
 * 压到一个固定的米制区间里，看着像米制其实不是。真正的米制关系由 native 侧
 * 用 VINS 稀疏三角化深度做鲁棒标定（MAD + Huber IRLS + EMA）来建立。
 */
class MonoDepthProvider(
    assets: AssetManager,
    private val modelAsset: String = "depth_model.tflite",
    val inputSize: Int = 256
) : DepthProvider {

    private val interpreter: Interpreter?

    private val pre = DepthPreprocessor(inputSize)
    private val output = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(1) } } }
    private val depthSmall = FloatArray(inputSize * inputSize)

    private var globalMin = Float.MAX_VALUE
    private var globalMax = -Float.MAX_VALUE

    private var buffer: FloatArray = FloatArray(0)
    private var bufferW = 0
    private var bufferH = 0
    private var latestResult: DepthProvider.Result? = null

    override val backendName: String = "litert-$modelAsset-$inputSize"
    override val available: Boolean get() = interpreter != null

    init {
        interpreter = try {
            Interpreter(loadModel(assets), Interpreter.Options().apply { setNumThreads(4) })
        } catch (t: Throwable) {
            Log.e(TAG, "depth model load failed: ${modelAsset}", t)
            null
        }
    }

    override fun submitFrame(
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int,
        rowStride: Int, uRowStride: Int, uPixelStride: Int,
        timestampNs: Long
    ): Boolean {
        val itp = interpreter ?: return false
        if (width <= 0 || height <= 0) return false

        pre.fill(y, u, v, width, height, rowStride, uRowStride, uPixelStride)
        itp.run(pre.tensor, output)

        // 会话级 min/max：逐帧 min/max 会随画面内容漂移，直接钳到 [0,1]
        // 又会把输出压成常数；会话级是这两者之间唯一稳定的选择。
        var mn = Float.MAX_VALUE
        var mx = -Float.MAX_VALUE
        for (oy in 0 until inputSize) {
            for (ox in 0 until inputSize) {
                val z = output[0][oy][ox][0]
                if (z < mn) mn = z
                if (z > mx) mx = z
            }
        }
        if (mn < globalMin) globalMin = mn
        if (mx > globalMax) globalMax = mx
        val lo = globalMin
        val hi = globalMax
        val range = (hi - lo).coerceAtLeast(1e-4f)
        for (oy in 0 until inputSize) {
            for (ox in 0 until inputSize) {
                val norm = ((output[0][oy][ox][0] - lo) / range).coerceIn(0f, 1f)
                depthSmall[oy * inputSize + ox] = FAR - norm * (FAR - NEAR)
            }
        }

        if (bufferW != width || bufferH != height) {
            buffer = FloatArray(width * height)
            bufferW = width
            bufferH = height
        }
        for (dy in 0 until height) {
            val sy = dy * inputSize / height
            val base = dy * width
            for (dx in 0 until width) {
                val sx = dx * inputSize / width
                buffer[base + dx] = depthSmall[sy * inputSize + sx]
            }
        }

        latestResult = DepthProvider.Result(
            depth = buffer,
            width = width,
            height = height,
            confidence = null,
            timestampNs = timestampNs,
            metric = false,
            backend = backendName
        )
        return true
    }

    override fun latest(): DepthProvider.Result? = latestResult

    override fun reset() {
        globalMin = Float.MAX_VALUE
        globalMax = -Float.MAX_VALUE
        latestResult = null
    }

    override fun close() {
        try {
            interpreter?.close()
        } catch (_: Throwable) {
        }
    }

    private fun loadModel(assets: AssetManager): ByteBuffer {
        val fd = assets.openFd(modelAsset)
        FileInputStream(fd.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    }

    companion object {
        private const val TAG = "MonoDepthProvider"
        private const val NEAR = 0.4f
        private const val FAR = 6f
    }
}
