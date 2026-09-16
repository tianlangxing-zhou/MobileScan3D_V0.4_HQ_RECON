package com.mobilescan3d.render

import android.graphics.BitmapFactory
import android.opengl.GLES20
import android.opengl.GLUtils
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * V0.6.1 textured AR pass.
 *
 * Input native asset:
 *   vertex: position3 + normal3 + UV2  (8 floats)
 *   index : triangle uint32
 *   atlas : JPEG
 *
 * GLES2 portability:
 * Instead of depending on OES_element_index_uint, indices are expanded once
 * on the GL thread into position3 + uv2 triangle vertices, then rendered with
 * glDrawArrays(GL_TRIANGLES). With the default 80k-triangle QEM target this is
 * still only ~4.8 MB of vertex data and works on old GLES2 drivers too.
 *
 * Projection is deliberately identical to MeshRenderer / PointCloudRenderer:
 *
 *   Pc = Rwc^T(Pw - twc)
 *   Camera2 intrinsics
 *   cameraToView
 *   NDC
 *
 * Therefore texture rendering cannot introduce a second AR camera model.
 */
class TexturedMeshRenderer {

    @Volatile
    var alpha: Float = 0.96f

    @Volatile
    var lastError: String = ""
        private set

    @Volatile
    var uploadedTriangleCount: Int = 0
        private set

    @Volatile
    var uploadedVertexCount: Int = 0
        private set

    val hasAsset: Boolean
        get() = uploadedTriangleCount > 0 || pendingVertices != null

    private var program = 0
    private var vbo = 0
    private var textureId = 0

    private var aPosition = -1
    private var aUv = -1

    private var uCameraT = -1
    private var uCameraRight = -1
    private var uCameraDown = -1
    private var uCameraForward = -1

    private var uFx = -1
    private var uFy = -1
    private var uCx = -1
    private var uCy = -1
    private var uImageW = -1
    private var uImageH = -1

    private var uCameraToView0 = -1
    private var uCameraToView1 = -1
    private var uNear = -1
    private var uFar = -1
    private var uAtlas = -1
    private var uAlpha = -1

    @Volatile
    private var pendingVertices: FloatArray? = null

    @Volatile
    private var pendingIndices: IntArray? = null

    @Volatile
    private var pendingAtlasJpeg: ByteArray? = null

    @Volatile
    private var clearPending = false

    fun init() {
        releaseGpu()

        program = link(VERT, FRAG)
        if (program == 0) {
            lastError = "TexturedMesh shader link failed"
            return
        }

        aPosition = GLES20.glGetAttribLocation(program, "aPosition")
        aUv = GLES20.glGetAttribLocation(program, "aUv")

        uCameraT = GLES20.glGetUniformLocation(program, "uCameraT")
        uCameraRight = GLES20.glGetUniformLocation(program, "uCameraRight")
        uCameraDown = GLES20.glGetUniformLocation(program, "uCameraDown")
        uCameraForward = GLES20.glGetUniformLocation(program, "uCameraForward")

        uFx = GLES20.glGetUniformLocation(program, "uFx")
        uFy = GLES20.glGetUniformLocation(program, "uFy")
        uCx = GLES20.glGetUniformLocation(program, "uCx")
        uCy = GLES20.glGetUniformLocation(program, "uCy")
        uImageW = GLES20.glGetUniformLocation(program, "uImageWidth")
        uImageH = GLES20.glGetUniformLocation(program, "uImageHeight")

        uCameraToView0 =
            GLES20.glGetUniformLocation(program, "uCameraToView0")
        uCameraToView1 =
            GLES20.glGetUniformLocation(program, "uCameraToView1")

        uNear = GLES20.glGetUniformLocation(program, "uNear")
        uFar = GLES20.glGetUniformLocation(program, "uFar")
        uAtlas = GLES20.glGetUniformLocation(program, "uAtlas")
        uAlpha = GLES20.glGetUniformLocation(program, "uAlpha")

        val ids = IntArray(1)
        GLES20.glGenBuffers(1, ids, 0)
        vbo = ids[0]

        lastError = ""
    }

    /**
     * Thread-safe submission. GPU upload happens in draw() on the GL thread.
     */
    fun setAsset(
        vertices8: FloatArray?,
        indices: IntArray?,
        atlasJpeg: ByteArray?
    ) {
        if (
            vertices8 == null ||
            indices == null ||
            atlasJpeg == null ||
            vertices8.size < 8 * 3 ||
            vertices8.size % 8 != 0 ||
            indices.size < 3 ||
            indices.size % 3 != 0 ||
            atlasJpeg.size < 4
        ) {
            clearAsset()
            return
        }

        // Ownership is transferred from TexturedArAssetLoader; these arrays
        // are not mutated afterwards. Avoid a second 5–30 MB Java heap copy.
        pendingVertices = vertices8
        pendingIndices = indices
        pendingAtlasJpeg = atlasJpeg
        clearPending = false
    }

    fun clearAsset() {
        pendingVertices = null
        pendingIndices = null
        pendingAtlasJpeg = null
        clearPending = true
    }

    fun draw(
        pose12: FloatArray,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        imageWidth: Int,
        imageHeight: Int,
        cameraToView: FloatArray,
        near: Float,
        far: Float
    ): Int {
        applyPending()

        if (
            program == 0 ||
            vbo == 0 ||
            textureId == 0 ||
            uploadedTriangleCount <= 0 ||
            pose12.size < 12 ||
            cameraToView.size < 6
        ) {
            return 0
        }

        GLES20.glUseProgram(program)

        GLES20.glUniform3f(
            uCameraT,
            pose12[9],
            pose12[10],
            pose12[11]
        )

        // Row-major Rwc columns = camera axes in world.
        GLES20.glUniform3f(
            uCameraRight,
            pose12[0],
            pose12[3],
            pose12[6]
        )
        GLES20.glUniform3f(
            uCameraDown,
            pose12[1],
            pose12[4],
            pose12[7]
        )
        GLES20.glUniform3f(
            uCameraForward,
            pose12[2],
            pose12[5],
            pose12[8]
        )

        GLES20.glUniform1f(uFx, fx.coerceAtLeast(1f))
        GLES20.glUniform1f(uFy, fy.coerceAtLeast(1f))
        GLES20.glUniform1f(uCx, cx)
        GLES20.glUniform1f(uCy, cy)

        GLES20.glUniform1f(
            uImageW,
            imageWidth.coerceAtLeast(1).toFloat()
        )
        GLES20.glUniform1f(
            uImageH,
            imageHeight.coerceAtLeast(1).toFloat()
        )

        GLES20.glUniform3f(
            uCameraToView0,
            cameraToView[0],
            cameraToView[1],
            cameraToView[2]
        )
        GLES20.glUniform3f(
            uCameraToView1,
            cameraToView[3],
            cameraToView[4],
            cameraToView[5]
        )

        GLES20.glUniform1f(uNear, near)
        GLES20.glUniform1f(uFar, far)
        GLES20.glUniform1f(uAlpha, alpha.coerceIn(0f, 1f))

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(
            GLES20.GL_TEXTURE_2D,
            textureId
        )
        GLES20.glUniform1i(uAtlas, 0)

        GLES20.glBindBuffer(
            GLES20.GL_ARRAY_BUFFER,
            vbo
        )

        val stride = EXPANDED_FLOATS * 4

        GLES20.glEnableVertexAttribArray(aPosition)
        GLES20.glVertexAttribPointer(
            aPosition,
            3,
            GLES20.GL_FLOAT,
            false,
            stride,
            0
        )

        GLES20.glEnableVertexAttribArray(aUv)
        GLES20.glVertexAttribPointer(
            aUv,
            2,
            GLES20.GL_FLOAT,
            false,
            stride,
            3 * 4
        )

        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthFunc(GLES20.GL_LEQUAL)

        if (alpha < 0.999f) {
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(
                GLES20.GL_SRC_ALPHA,
                GLES20.GL_ONE_MINUS_SRC_ALPHA
            )
            // Semi-transparent overlay should test against depth but not
            // permanently occlude later camera-overlay passes.
            GLES20.glDepthMask(false)
        }

        GLES20.glDrawArrays(
            GLES20.GL_TRIANGLES,
            0,
            uploadedVertexCount
        )

        if (alpha < 0.999f) {
            GLES20.glDepthMask(true)
            GLES20.glDisable(GLES20.GL_BLEND)
        }

        GLES20.glDisableVertexAttribArray(aPosition)
        GLES20.glDisableVertexAttribArray(aUv)

        GLES20.glBindBuffer(
            GLES20.GL_ARRAY_BUFFER,
            0
        )
        GLES20.glBindTexture(
            GLES20.GL_TEXTURE_2D,
            0
        )

        return uploadedTriangleCount
    }

    private fun applyPending() {
        if (clearPending) {
            clearPending = false
            releaseTexture()
            uploadedTriangleCount = 0
            uploadedVertexCount = 0
        }

        val vertices = pendingVertices ?: return
        val indices = pendingIndices ?: return
        val jpeg = pendingAtlasJpeg ?: return

        pendingVertices = null
        pendingIndices = null
        pendingAtlasJpeg = null

        try {
            val sourceVertexCount =
                vertices.size / SOURCE_FLOATS

            val expanded =
                FloatArray(indices.size * EXPANDED_FLOATS)

            var dst = 0
            for (index in indices) {
                if (index < 0 || index >= sourceVertexCount) {
                    throw IllegalArgumentException(
                        "mesh index $index / $sourceVertexCount"
                    )
                }

                val src = index * SOURCE_FLOATS

                expanded[dst++] = vertices[src + 0]
                expanded[dst++] = vertices[src + 1]
                expanded[dst++] = vertices[src + 2]

                expanded[dst++] = vertices[src + 6]
                expanded[dst++] = vertices[src + 7]
            }

            val buffer: FloatBuffer =
                ByteBuffer
                    .allocateDirect(expanded.size * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()

            buffer.put(expanded)
            buffer.flip()

            GLES20.glBindBuffer(
                GLES20.GL_ARRAY_BUFFER,
                vbo
            )
            GLES20.glBufferData(
                GLES20.GL_ARRAY_BUFFER,
                expanded.size * 4,
                buffer,
                GLES20.GL_STATIC_DRAW
            )
            GLES20.glBindBuffer(
                GLES20.GL_ARRAY_BUFFER,
                0
            )

            val bitmap =
                BitmapFactory.decodeByteArray(
                    jpeg,
                    0,
                    jpeg.size
                ) ?: throw IllegalArgumentException(
                    "atlas JPEG decode failed"
                )

            releaseTexture()

            val tex = IntArray(1)
            GLES20.glGenTextures(1, tex, 0)
            textureId = tex[0]

            GLES20.glBindTexture(
                GLES20.GL_TEXTURE_2D,
                textureId
            )

            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MIN_FILTER,
                GLES20.GL_LINEAR
            )
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MAG_FILTER,
                GLES20.GL_LINEAR
            )
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_S,
                GLES20.GL_CLAMP_TO_EDGE
            )
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_T,
                GLES20.GL_CLAMP_TO_EDGE
            )

            GLUtils.texImage2D(
                GLES20.GL_TEXTURE_2D,
                0,
                bitmap,
                0
            )

            bitmap.recycle()

            GLES20.glBindTexture(
                GLES20.GL_TEXTURE_2D,
                0
            )

            uploadedVertexCount = indices.size
            uploadedTriangleCount = indices.size / 3
            lastError = ""
        } catch (t: Throwable) {
            lastError =
                t.message ?: t.javaClass.simpleName
            Log.e(
                TAG,
                "textured mesh upload failed",
                t
            )
            releaseTexture()
            uploadedVertexCount = 0
            uploadedTriangleCount = 0
        }
    }

    private fun releaseTexture() {
        if (textureId != 0) {
            GLES20.glDeleteTextures(
                1,
                intArrayOf(textureId),
                0
            )
            textureId = 0
        }
    }

    private fun releaseGpu() {
        releaseTexture()

        if (vbo != 0) {
            GLES20.glDeleteBuffers(
                1,
                intArrayOf(vbo),
                0
            )
            vbo = 0
        }

        if (program != 0) {
            GLES20.glDeleteProgram(program)
            program = 0
        }

        uploadedTriangleCount = 0
        uploadedVertexCount = 0
    }

    private fun compile(
        type: Int,
        source: String
    ): Int {
        val shader =
            GLES20.glCreateShader(type)

        GLES20.glShaderSource(
            shader,
            source
        )
        GLES20.glCompileShader(shader)

        val status = IntArray(1)
        GLES20.glGetShaderiv(
            shader,
            GLES20.GL_COMPILE_STATUS,
            status,
            0
        )

        if (status[0] == 0) {
            val log =
                GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw IllegalStateException(
                "shader compile: $log"
            )
        }

        return shader
    }

    private fun link(
        vertexSource: String,
        fragmentSource: String
    ): Int {
        return try {
            val vs =
                compile(
                    GLES20.GL_VERTEX_SHADER,
                    vertexSource
                )
            val fs =
                compile(
                    GLES20.GL_FRAGMENT_SHADER,
                    fragmentSource
                )

            val p =
                GLES20.glCreateProgram()

            GLES20.glAttachShader(p, vs)
            GLES20.glAttachShader(p, fs)
            GLES20.glLinkProgram(p)

            val status = IntArray(1)
            GLES20.glGetProgramiv(
                p,
                GLES20.GL_LINK_STATUS,
                status,
                0
            )

            GLES20.glDeleteShader(vs)
            GLES20.glDeleteShader(fs)

            if (status[0] == 0) {
                val log =
                    GLES20.glGetProgramInfoLog(p)
                GLES20.glDeleteProgram(p)
                lastError =
                    "program link: $log"
                0
            } else {
                p
            }
        } catch (t: Throwable) {
            lastError =
                t.message ?: "shader error"
            Log.e(TAG, lastError, t)
            0
        }
    }

    companion object {
        private const val TAG =
            "TexturedMeshRenderer"

        private const val SOURCE_FLOATS = 8
        private const val EXPANDED_FLOATS = 5

        private val VERT = """
            #version 100

            attribute vec3 aPosition;
            attribute vec2 aUv;

            uniform vec3 uCameraT;
            uniform vec3 uCameraRight;
            uniform vec3 uCameraDown;
            uniform vec3 uCameraForward;

            uniform float uFx;
            uniform float uFy;
            uniform float uCx;
            uniform float uCy;
            uniform float uImageWidth;
            uniform float uImageHeight;

            uniform vec3 uCameraToView0;
            uniform vec3 uCameraToView1;

            uniform float uNear;
            uniform float uFar;

            varying vec2 vUv;

            void main() {
                vec3 dw =
                    aPosition - uCameraT;

                vec3 pc = vec3(
                    dot(dw, uCameraRight),
                    dot(dw, uCameraDown),
                    dot(dw, uCameraForward)
                );

                if (pc.z <= 0.05) {
                    gl_Position =
                        vec4(2.0, 2.0, 1.0, 1.0);
                    vUv = vec2(0.0);
                    return;
                }

                float cameraU =
                    (uFx * pc.x / pc.z + uCx) /
                    uImageWidth;

                float cameraV =
                    (uFy * pc.y / pc.z + uCy) /
                    uImageHeight;

                vec3 uv =
                    vec3(
                        cameraU,
                        cameraV,
                        1.0
                    );

                float viewU =
                    dot(
                        uCameraToView0,
                        uv
                    );

                float viewV =
                    dot(
                        uCameraToView1,
                        uv
                    );

                float ndcX =
                    2.0 * viewU - 1.0;
                float ndcY =
                    1.0 - 2.0 * viewV;

                float depth01 =
                    clamp(
                        (pc.z - uNear) /
                        (uFar - uNear),
                        0.0,
                        1.0
                    );

                gl_Position =
                    vec4(
                        ndcX,
                        ndcY,
                        depth01 * 2.0 - 1.0,
                        1.0
                    );

                // V0.6 atlas uses glTF's upper-left UV origin. GLUtils
                // forwards Bitmap pixels directly to glTexImage2D; the first
                // bitmap row therefore occupies v=0 in the uploaded texture.
                // Keep the baked UV unchanged (do not add a second Y flip).
                vUv = aUv;
            }
        """.trimIndent()

        private val FRAG = """
            #version 100

            precision mediump float;

            uniform sampler2D uAtlas;
            uniform float uAlpha;

            varying vec2 vUv;

            void main() {
                vec4 c =
                    texture2D(
                        uAtlas,
                        vUv
                    );

                gl_FragColor =
                    vec4(
                        c.rgb,
                        c.a * uAlpha
                    );
            }
        """.trimIndent()
    }
}
