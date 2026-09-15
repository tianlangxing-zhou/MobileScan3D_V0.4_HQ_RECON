package com.mobilescan3d

import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class DepthProvider(assetManager: AssetManager) {

    private val interpreter: Interpreter

    private val input = Array(1) { Array(INPUT) { Array(INPUT) { FloatArray(3) } } }
    private val output = Array(1) { Array(INPUT) { Array(INPUT) { FloatArray(1) } } }
    private val depth256 = FloatArray(INPUT * INPUT)
    private var globalMin = Float.MAX_VALUE
    private var globalMax = -Float.MAX_VALUE

    init {
        interpreter = Interpreter(loadModel(assetManager), Interpreter.Options().apply {
            setNumThreads(4)
        })
    }

    fun close() {
        interpreter.close()
    }

    // Fills depthOut (size w*h) with metric depth in meters.
    fun estimate(
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        w: Int,
        h: Int,
        rowStride: Int,
        uRowStride: Int,
        uPixelStride: Int,
        depthOut: FloatArray
    ) {
        for (oy in 0 until INPUT) {
            val sy = oy * h / INPUT
            for (ox in 0 until INPUT) {
                val sx = ox * w / INPUT
                val yi = sy * rowStride + sx
                val uvIdx = (sy / 2) * uRowStride + (sx / 2) * uPixelStride

                val yy = y[yi].toInt() and 0xFF
                val uu = u[uvIdx].toInt() and 0xFF
                val vv = v[uvIdx].toInt() and 0xFF

                val c = yy - 16
                val d = uu - 128
                val e = vv - 128
                val r = ((298 * c + 409 * e + 128) shr 8).coerceIn(0, 255)
                val g = ((298 * c - 100 * d - 208 * e + 128) shr 8).coerceIn(0, 255)
                val b = ((298 * c + 516 * d + 128) shr 8).coerceIn(0, 255)

                input[0][oy][ox][0] = (r - MEAN[0]) / STD[0]
                input[0][oy][ox][1] = (g - MEAN[1]) / STD[1]
                input[0][oy][ox][2] = (b - MEAN[2]) / STD[2]
            }
        }

        interpreter.run(input, output)

        // 使用会话级 min/max 做稳定尺度：
        // - 不像逐帧 min/max 那样随画面内容漂移
        // - 也不像直接钳到 [0,1] 那样把输出变成常数
        var mn = Float.MAX_VALUE
        var mx = -Float.MAX_VALUE
        for (oy in 0 until INPUT) {
            for (ox in 0 until INPUT) {
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
        for (oy in 0 until INPUT) {
            for (ox in 0 until INPUT) {
                val norm = ((output[0][oy][ox][0] - lo) / range).coerceIn(0f, 1f)
                depth256[oy * INPUT + ox] = FAR - norm * (FAR - NEAR)
            }
        }

        for (dy in 0 until h) {
            val sy = dy * INPUT / h
            var base = dy * w
            for (dx in 0 until w) {
                val sx = dx * INPUT / w
                depthOut[base + dx] = depth256[sy * INPUT + sx]
            }
        }
    }

    private fun loadModel(assets: AssetManager): ByteBuffer {
        val fd = assets.openFd(MODEL)
        FileInputStream(fd.fileDescriptor).use { stream ->
            return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    companion object {
        private const val MODEL = "depth_model.tflite"
        private const val INPUT = 256
        private const val NEAR = 0.4f
        private const val FAR = 6f
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }
}
