package com.mobilescan3d.depth

/**
 * YUV_420_888 -> 模型输入张量的预处理。
 *
 * 从原来的 DepthProvider 里原样抽出来，**算法一字未改**（YUV->RGB 的定点系数、
 * ImageNet 均值方差、最近邻重采样），只是不再和 TFLite 解释器耦合在一起：
 * 换模型尺度、加硬件深度、或者在标定里复用它，都不需要动推理代码。
 *
 * 注意 u/v 平面是 **整段 stride 拷贝**（见 MainActivity.extractPlane），所以必须
 * 用 uRowStride / uPixelStride 定位，不能假设 uv 是紧凑排列的。
 */
class DepthPreprocessor(val inputSize: Int = 256) {

    /** 形状 [1][inputSize][inputSize][3]，直接喂给 TFLite。 */
    val tensor: Array<Array<Array<FloatArray>>> =
        Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(3) } } }

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
        val n = inputSize
        for (oy in 0 until n) {
            val sy = oy * height / n
            for (ox in 0 until n) {
                val sx = ox * width / n
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

                tensor[0][oy][ox][0] = (r - MEAN[0]) / STD[0]
                tensor[0][oy][ox][1] = (g - MEAN[1]) / STD[1]
                tensor[0][oy][ox][2] = (b - MEAN[2]) / STD[2]
            }
        }
    }

    companion object {
        // Depth Anything 系列用的就是这套 ImageNet 统计量
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }
}
