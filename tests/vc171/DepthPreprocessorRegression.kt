package com.mobilescan3d.depth
import java.util.Random
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.system.measureNanoTime

fun same(a: ByteBuffer, b: ByteBuffer) {
    check(a.capacity() == b.capacity())
    for (i in 0 until a.capacity() step 4) check(a.getInt(i) == b.getInt(i)) { "Tensor mismatch at $i" }
}
fun main() {
    val rng=Random(171)
    val old=BaselinePreprocessor(256); val now=DepthPreprocessor(256)
    // Odd dimensions, padding, pixel-stride changes with unchanged width,
    // tightly truncated last row, and changing camera sizes.
    for ((w,h,ps) in listOf(Triple(641,479,2),Triple(641,479,1),Triple(640,480,2),Triple(3,5,3))) {
        val ys=w+11; val us=((w+1)/2)*ps+7
        val y=ByteArray((h-1)*ys+w).also(rng::nextBytes)
        val u=ByteArray(((h-1)/2)*us+((w-1)/2)*ps+1).also(rng::nextBytes)
        val v=ByteArray(u.size).also(rng::nextBytes)
        old.fill(y,u,v,w,h,ys,us,ps); now.fill(y,u,v,w,h,ys,us,ps); same(old.tensor,now.tensor)
        check(runCatching { now.fill(y.copyOf(y.size-1),u,v,w,h,ys,us,ps) }.isFailure)
    }
    val w=640; val h=480;val y=ByteArray(w*h).also(rng::nextBytes)
    val u=ByteArray(w*h/4).also(rng::nextBytes);val v=ByteArray(u.size).also(rng::nextBytes)
    repeat(300){old.fill(y,u,v,w,h,w,w/2,1);now.fill(y,u,v,w,h,w,w/2,1)}
    val a=mutableListOf<Long>(); val b=mutableListOf<Long>()
    repeat(7) { round ->
        fun baseline()=measureNanoTime{ repeat(150){old.fill(y,u,v,w,h,w,w/2,1)} }
        fun modified()=measureNanoTime{ repeat(150){now.fill(y,u,v,w,h,w,w/2,1)} }
        if(round%2==0){a+=baseline();b+=modified()}else{b+=modified();a+=baseline()}
    }
    same(old.tensor,now.tensor)
    println("PASS tensor bit-exact: padding, odd sizes, stride changes, truncation, reuse")
    println("Desktop JVM median ms/frame baseline=${a.sorted()[3]/150e6} modified=${b.sorted()[3]/150e6}")
}

// VC170 reference for regression/benchmark only.
private class BaselinePreprocessor(val inputSize: Int = 256) {

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
