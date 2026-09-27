package com.mobilescan3d.depth

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Android DEPTH16: low 13 bits are millimetres, high 3 bits encode confidence. */
object Depth16Decoder {
    data class Decoded(val depth: FloatArray, val confidence: FloatArray)

    fun decode(source: ByteBuffer, width: Int, height: Int, rowStride: Int, pixelStride: Int): Decoded? {
        if (width <= 0 || height <= 0 || width > 4096 || height > 4096 || pixelStride < 2) return null
        val rowBytes = (width - 1L) * pixelStride + 2
        if (rowStride < rowBytes) return null
        val required = (height - 1L) * rowStride + rowBytes
        if (required > source.remaining()) return null
        val bytes = source.duplicate().order(ByteOrder.nativeOrder())
        val base = bytes.position()
        val depth = FloatArray(width * height)
        val confidence = FloatArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val raw = bytes.getShort(base + y * rowStride + x * pixelStride).toInt() and 0xffff
                val mm = raw and 0x1fff
                val code = raw ushr 13
                val c = if (code == 0) 1f else (code - 1) / 7f
                val i = y * width + x
                if (mm > 0 && c > 0f) {
                    depth[i] = mm / 1000f
                    confidence[i] = c
                }
            }
        }
        return Decoded(depth, confidence)
    }
}
