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

    // ---- V0.13.4：归一化映射元数据（d = normA * q + normB）----
    /**
     * 当前映射的斜率/截距。由 [globalMin] / [globalMax] 推出：
     *   `d = FAR - norm*(FAR-NEAR)`，`norm = (q - lo)/(hi - lo)`
     *   => `A = -(FAR-NEAR)/(hi-lo)`，`B = FAR + (FAR-NEAR)*lo/(hi-lo)`
     * 注意 A 恒为负：网络输出越大（越远）d 越小，这是这套映射的定义。
     */
    @Volatile private var normA = 0f
    @Volatile private var normB = 0f
    /** 映射每次变化 +1；0 表示「还没有有效映射」。 */
    @Volatile private var normVersion = 0L
    /** 上一帧实际使用的 lo/hi，用来判断这一帧的映射有没有变。 */
    private var appliedLo = 0f
    private var appliedHi = 0f
    /** 冻结标志：true 时 min/max 不再扩张（epoch 锁定后使用）。 */
    @Volatile private var normalizedFrozen = false

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
        // 冻结后不再扩张范围：epoch 锁定期间 d 的语义必须恒定。
        if (!normalizedFrozen) {
            if (mn < globalMin) globalMin = mn
            if (mx > globalMax) globalMax = mx
        }
        val lo = globalMin
        val hi = globalMax
        // V0.13.4：范围退化保护。首帧或画面极平坦时 hi-lo 接近 0，此时
        // 归一化会把整帧压成常数 NEAR 或 FAR（A 趋于无穷），后面一旦范围
        // 扩张就会触发一次幅度极大的重参数化。这里直接沿用上一次映射。
        val rawSpan = hi - lo
        if (rawSpan < MIN_RAW_SPAN && appliedHi > appliedLo) {
            // 维持既有映射，不更新版本号
        } else {
            val range = rawSpan.coerceAtLeast(MIN_RAW_SPAN)
            normA = -(FAR - NEAR) / range
            normB = FAR + (FAR - NEAR) * lo / range
            if (lo != appliedLo || hi != appliedHi) {
                appliedLo = lo
                appliedHi = hi
                ++normVersion
            }
        }
        val a = normA
        val b = normB
        for (oy in 0 until inputSize) {
            for (ox in 0 until inputSize) {
                // 与上报的元数据严格一致：d = a*q + b（q 为网络原始输出）。
                // 因为 globalMin/Max 单调覆盖所有历史输出，q 恒在 [lo,hi] 内，
                // 结果自然落在 [NEAR, FAR]；这里仍钳一次只为防浮点边界。
                depthSmall[oy * inputSize + ox] =
                    (a * output[0][oy][ox][0] + b).coerceIn(NEAR, FAR)
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
            backend = backendName,
            normA = normA,
            normB = normB,
            normVersion = normVersion
        )
        return true
    }

    override fun latest(): DepthProvider.Result? = latestResult

    override fun reset() {
        globalMin = Float.MAX_VALUE
        globalMax = -Float.MAX_VALUE
        normA = 0f
        normB = 0f
        appliedLo = 0f
        appliedHi = 0f
        normalizedFrozen = false
        // 版本号必须推进：新会话的范围与上一会话无关，native 侧要把自己的
        // 归一化基线一并清掉，不能拿上一会话的 A/B 去重参数化。
        ++normVersion
        latestResult = null
    }

    /**
     * V0.13.4：冻结当前归一化映射（之后不再随 min/max 扩张）。
     *
     * 重参数化已经能保证「映射变了标定也跟着变」，但每次变化都会让
     * epoch 的漂移监控多一次扰动。epoch 冻结标定之后，最稳妥的做法是
     * 连输入映射一起冻住 —— 这样整个 epoch 内 d 的语义完全恒定。
     */
    fun freezeNormalization() {
        normalizedFrozen = true
    }

    /** 解除冻结（新会话/重新标定时用）。 */
    fun unfreezeNormalization() {
        normalizedFrozen = false
    }

    /** 当前映射是否已冻结。诊断用。 */
    val normalizationFrozen: Boolean get() = normalizedFrozen

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
        /**
         * V0.13.4：原始输出跨度下限。低于此值认为这一帧的 min/max 退化
         * （平坦画面 / 首帧），不据此改写映射，避免一次极端重参数化。
         */
        private const val MIN_RAW_SPAN = 1e-3f
    }
}
