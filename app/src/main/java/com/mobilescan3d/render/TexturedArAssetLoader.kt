package com.mobilescan3d.render

import com.mobilescan3d.NativeBridge
import com.mobilescan3d.PointCloudRenderer

/**
 * Pull the native V0.6.1 AR cache into the GLES renderer.
 *
 * Kept outside MainActivity so both "just baked" and "restored package"
 * paths use exactly the same array-size validation.
 */
object TexturedArAssetLoader {

    data class Result(
        val ok: Boolean,
        val message: String,
        val vertices: Int = 0,
        val triangles: Int = 0
    )

    fun loadInto(
        renderer: PointCloudRenderer
    ): Result {
        return try {
            val stats =
                NativeBridge.nativeGetTexturedArAssetStats()

            if (
                stats.size < 3 ||
                stats[0] <= 0 ||
                stats[1] <= 0 ||
                stats[2] <= 0
            ) {
                return Result(
                    false,
                    "native 中没有 HQ 纹理 AR 资产"
                )
            }

            val vertexCount = stats[0]
            val indexCount = stats[1]
            val jpegBytes = stats[2]

            if (
                vertexCount > NativeBridge.AR_TEXTURED_MAX_VERTICES ||
                indexCount > NativeBridge.AR_TEXTURED_MAX_INDICES ||
                jpegBytes > NativeBridge.AR_TEXTURED_MAX_JPEG_BYTES ||
                indexCount % 3 != 0
            ) {
                return Result(
                    false,
                    "HQ AR 资产尺寸异常"
                )
            }

            val vertices =
                FloatArray(
                    vertexCount *
                        NativeBridge.AR_TEXTURED_VERTEX_FLOATS
                )

            val indices =
                IntArray(indexCount)

            val gotVertices =
                NativeBridge.nativeGetTexturedArVertices(
                    vertices
                )

            val gotIndices =
                NativeBridge.nativeGetTexturedArIndices(
                    indices
                )

            val jpeg =
                NativeBridge.nativeGetTexturedArAtlasJpeg()

            if (
                gotVertices != vertexCount ||
                gotIndices != indexCount ||
                jpeg == null ||
                jpeg.size != jpegBytes
            ) {
                return Result(
                    false,
                    "HQ AR 资产 JNI 拷贝不完整"
                )
            }

            renderer.setTexturedMesh(
                vertices,
                indices,
                jpeg
            )

            Result(
                true,
                "HQ 纹理 AR 已加载",
                vertexCount,
                indexCount / 3
            )
        } catch (t: Throwable) {
            Result(
                false,
                "HQ AR 加载失败：${t.message ?: t.javaClass.simpleName}"
            )
        }
    }
}
