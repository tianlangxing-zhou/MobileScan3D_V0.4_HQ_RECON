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
 * **评审 P0-1 行为（2026-09-27 改）**：不再做会话级 min/max 动态映射，也不
 * 上采样到相机分辨率。模型原始输出 `q` 以固定表示 [INVERSE_DEPTH] 原样交
 * native，**同一个 q 永远对应同一个米制深度** —— 这是根除尺度漂移的关键。
 * 真正的米制关系由 native 侧用 VINS 稀疏三角化 + stereo anchors 拟合
 * `1/metric = scale*q + shift`（MAD + Huber IRLS + EMA）建立。
 *
 * 注意：下游「目标 mask / presence gate」依赖的是**相对深度的相对排序与分布**，
 * 而相对排序在去掉 min/max 映射后反而更稳定（之前会话级 min/max 扩张会让
 * 同一物体在不同扫描阶段映射到不同的相对深度，反而伤害 mask 一致性）。
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

    // 评审 P0-1：不再做会话级 min/max 动态映射，也不上采样到相机分辨率。
    // 模型原始输出 q 以固定表示（INVERSE_DEPTH）原样交 native，由 native 侧
    // 用 VINS 稀疏三角化 + stereo anchors 拟合 scale/shift 得到米制。
    // 这样同一 q 永远对应同一个米制深度，彻底消除尺度漂移。

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

        // 评审 P0-1 + P0-2：直接取原始模型输出 q，固定 INVERSE_DEPTH 语义，
        // 不做会话级 min/max、不做 [0.4,6] 映射、不上采样到相机分辨率。
        // 分辨率保持 inputSize×inputSize (256×256)，native 侧按深度/相机比
        // 降采样内参 K 完成融合，几何信息基本无损。
        for (oy in 0 until inputSize) {
            val base = oy * inputSize
            for (ox in 0 until inputSize) {
                depthSmall[base + ox] = output[0][oy][ox][0]
            }
        }

        latestResult = DepthProvider.Result(
            depth = depthSmall,
            width = inputSize,
            height = inputSize,
            confidence = null,
            timestampNs = timestampNs,
            metric = false,
            backend = backendName,
            representation = DepthRepresentation.INVERSE_DEPTH
        )
        return true
    }

    override fun latest(): DepthProvider.Result? = latestResult

    override fun reset() {
        latestResult = null
    }

    override fun close() {
        try {
            interpreter?.close()
        } catch (_: Throwable) {
        }
    }

    private fun loadModel(assets: AssetManager): ByteBuffer {
        assets.openFd(modelAsset).use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                return stream.channel.map(
                    FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
                )
            }
        }
    }

    companion object {
        private const val TAG = "MonoDepthProvider"
    }
}
