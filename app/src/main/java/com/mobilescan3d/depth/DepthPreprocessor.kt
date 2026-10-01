package com.mobilescan3d.depth

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** YUV420 -> fixed ImageNet normalization; nearest sampling and NHWC layout
 * remain identical to VC160, while contiguous storage removes nested JNI copies.
 * U/V use Camera2's shared chroma row/pixel stride; callers copy whole planes.
 */
class DepthPreprocessor(val inputSize: Int = 256) {

    init { require(inputSize in 1..2048) }
    /** Contiguous native-order FLOAT32 NHWC input, reused by the depth worker. */
    val tensor: ByteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
        .order(ByteOrder.nativeOrder())
    private val sourceX = IntArray(inputSize)
    private var sourceWidth = 0

    fun fill(
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        width: Int,
        height: Int,
        rowStride: Int,
        uRowStride: Int,
        uPixelStride: Int
    ) {
        require(width > 0 && height > 0 && rowStride >= width && uRowStride > 0 && uPixelStride > 0)
        val yLast = (height - 1L) * rowStride + width - 1L
        val uvLast = ((height - 1L) / 2) * uRowStride + ((width - 1L) / 2) * uPixelStride
        require(yLast < y.size && uvLast < u.size && uvLast < v.size) { "Truncated YUV planes" }
        val n = inputSize
        if (sourceWidth != width) {
            for (x in 0 until n) sourceX[x] = (x.toLong() * width / n).toInt()
            sourceWidth = width
        }
        tensor.clear()
        for (oy in 0 until n) {
            val sy = oy * height / n
            for (ox in 0 until n) {
                val sx = sourceX[ox]
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

                tensor.putFloat((r - MEAN[0]) / STD[0])
                tensor.putFloat((g - MEAN[1]) / STD[1])
                tensor.putFloat((b - MEAN[2]) / STD[2])
            }
        }
        tensor.rewind()
    }

    companion object {
        // Depth Anything 系列用的就是这套 ImageNet 统计量
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }
}
