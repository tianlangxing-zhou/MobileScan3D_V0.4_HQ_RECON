package com.mobilescan3d.render

import android.opengl.GLES20
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * AR 网格渲染 pass：VBO + IBO + `glDrawElements(GL_TRIANGLES)`。
 *
 * ## 为什么是「pass」而不是一个独立的 GLSurfaceView.Renderer
 *
 * 评审给的方向是 `PointCloudRenderer → MeshRenderer`。真正要保住的语义是
 * **网格必须走和点云完全一样的相机模型投影链**，而不是「必须再开一个
 * 独立 GLSurfaceView」。理由是那条链是实机验证过的：
 *
 * ```
 * Pw -> Pc = Rwc^T (Pw - twc)
 *    -> cameraUV = (fx*X/Z + cx, fy*Y/Z + cy) / imageSize
 *    -> viewUV   = cameraToView * cameraUV      // SurfaceTexture + TextureView + crop
 *    -> NDC
 * ```
 *
 * 如果这里另起一个 `GLSurfaceView`，就得把 `cameraToView`、相机内参、
 * VINS pose 时间戳查询全部再复制一份 —— 而 `cameraToView` 是
 * `MainActivity.updateCameraToViewTransform()` 用「点选目标」那条已验证的
 * 逆映射反求出来的，复制一份等于埋一个必然不一致的坑。
 *
 * 所以本类是**独立的 GL 资源与绘制逻辑**（自己的 program / VBO / IBO /
 * uniform），但由 [PointCloudRenderer] 在同一个 GL 上下文里调用。
 *
 * ## 关于 `setEGLContextClientVersion`
 *
 * 评审建议从 2 升到 3。`glDrawElements` 需要 32 位索引（HQ 网格顶点数
 * 轻松超过 65535）时，GLES2 上要靠 `GL_OES_element_index_uint`。本类
 * **不用**升 context 版本，而是在 [init] 时查这个扩展：
 *   - 有扩展  -> `GL_UNSIGNED_INT`
 *   - 没有    -> `GL_UNSIGNED_SHORT`（此时拒绝超过 65535 顶点的网格并报错）
 *
 * 这样既拿到了 32 位索引，又不会在极少数只支持 GLES2 的设备上因为强制
 * 请求 ES3 context 失败而直接黑屏 —— 黑屏是这一类改动里代价最高的失败模式。
 *
 * 顶点布局与 native 的 `nativeGetMeshVertices` 一致：**9 个 float**
 * `x,y,z, nx,ny,nz, r,g,b`。
 */
class MeshRenderer {

    private var program = 0
    private var aPosition = -1
    private var aNormal = -1
    private var aColor = -1
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
    private var uAlpha = -1
    private var uUseTint = -1
    private var uTint = -1

    private var vbo = 0
    private var ibo = 0

    private var glReady = false
    private var supportsUintIndex = false

    /** 已上传到 GPU 的网格规模。 */
    @Volatile
    var uploadedVertexCount = 0
        private set

    @Volatile
    var uploadedTriangleCount = 0
        private set

    /** 渲染不透明度的可调项（AR 预览默认半透明，方便看到相机画面）。 */
    @Volatile
    var alpha = 0.85f

    /**
     * V0.12：扫描预览的诊断色。
     *
     * `useTint = true` 时**忽略逐顶点颜色**，整片网格统一用 (tintR, tintG, tintB)
     * 着色。实机上原始顶点色在同一面墙上几乎没有对比度，「模型长到哪儿了」
     * 完全看不出来；改成青绿诊断色 + 低 alpha 之后一眼可辨。
     * 停扫验收几何时关掉，恢复真实颜色。
     */
    @Volatile
    var useTint = false
    @Volatile
    var tintR = 0.15f
    @Volatile
    var tintG = 0.95f
    @Volatile
    var tintB = 1.00f

    // ---- 待上传数据。setMesh() 可能从任意线程调用，真正的上传推迟到 GL 线程 ----
    @Volatile private var pendingVertices: FloatArray? = null
    @Volatile private var pendingIndices: IntArray? = null
    @Volatile private var pendingVersion = 0L
    private var uploadedVersion = -1L

    /** 最近一次上传失败的原因（HUD / 报告用）。 */
    @Volatile
    var lastError: String = ""
        private set

    // ------------------------------------------------------------------ GL 生命周期

    /** 必须在 GL 线程调用（`onSurfaceCreated`）。 */
    fun init(): Boolean {
        if (glReady) return true
        program = link(VERT, FRAG)
        if (program == 0) {
            lastError = "网格着色器编译失败"
            return false
        }
        aPosition = GLES20.glGetAttribLocation(program, "aPosition")
        aNormal = GLES20.glGetAttribLocation(program, "aNormal")
        aColor = GLES20.glGetAttribLocation(program, "aColor")
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
        uCameraToView0 = GLES20.glGetUniformLocation(program, "uCameraToView0")
        uCameraToView1 = GLES20.glGetUniformLocation(program, "uCameraToView1")
        uNear = GLES20.glGetUniformLocation(program, "uNear")
        uFar = GLES20.glGetUniformLocation(program, "uFar")
        uAlpha = GLES20.glGetUniformLocation(program, "uAlpha")
        uUseTint = GLES20.glGetUniformLocation(program, "uUseTint")
        uTint = GLES20.glGetUniformLocation(program, "uTint")

        val exts = GLES20.glGetString(GLES20.GL_EXTENSIONS) ?: ""
        supportsUintIndex = exts.contains("GL_OES_element_index_uint")

        val bufs = IntArray(2)
        GLES20.glGenBuffers(2, bufs, 0)
        vbo = bufs[0]
        ibo = bufs[1]

        glReady = true
        Log.i(TAG, "mesh program ready, uintIndex=$supportsUintIndex")
        return true
    }

    /** GL 上下文销毁前调用（`onSurfaceCreated` 里重建即可，无需显式释放）。 */
    fun release() {
        if (vbo != 0 || ibo != 0) {
            GLES20.glDeleteBuffers(2, intArrayOf(vbo, ibo), 0)
        }
        if (program != 0) {
            GLES20.glDeleteProgram(program)
        }
        vbo = 0
        ibo = 0
        program = 0
        glReady = false
        uploadedVersion = -1L
        uploadedVertexCount = 0
        uploadedTriangleCount = 0
    }

    // ------------------------------------------------------------------ 数据

    /**
     * 提交一份新网格。可在任意线程调用，实际上传发生在下一次 [draw]。
     *
     * @param vertices 交错顶点，长度必须是 9 的倍数（`x,y,z,nx,ny,nz,r,g,b`）
     * @param indices  三角形索引，长度必须是 3 的倍数
     */
    fun setMesh(vertices: FloatArray?, indices: IntArray?) {
        pendingVertices = vertices
        pendingIndices = indices
        pendingVersion++
    }

    fun clearMesh() {
        pendingVertices = null
        pendingIndices = null
        pendingVersion++
    }

    val hasPendingMesh: Boolean get() = pendingVertices != null && pendingIndices != null
    val hasGpuMesh: Boolean get() = uploadedVertexCount > 0 && uploadedTriangleCount > 0

    // ------------------------------------------------------------------ 绘制

    /**
     * 在已绑定好的帧缓冲上画网格。必须在 GL 线程调用。
     *
     * @param pose    `nativeGetRenderPoseAt` 输出：[R(9) 行主序, t(3)]
     * @param cameraToView 6 个 float 的 2x3 仿射，含义见 [PointCloudRenderer]
     * @return 实际画出的三角形数（0 表示没有可画的东西）
     */
    fun draw(
        pose: FloatArray,
        fx: Float, fy: Float, cx: Float, cy: Float,
        imageWidth: Int, imageHeight: Int,
        cameraToView: FloatArray,
        nearPlane: Float,
        farPlane: Float
    ): Int {
        if (!glReady) return 0
        if (pose.size < 12 || cameraToView.size < 6) return 0
        uploadIfNeeded()
        if (uploadedTriangleCount <= 0) return 0

        GLES20.glUseProgram(program)

        // Pc = Rwc^T (Pw - twc)。pose 是行主序 R，所以 Rwc 的三列是
        // (R[0],R[3],R[6]) / (R[1],R[4],R[7]) / (R[2],R[5],R[8])。
        GLES20.glUniform3f(uCameraT, pose[9], pose[10], pose[11])
        GLES20.glUniform3f(uCameraRight, pose[0], pose[3], pose[6])
        GLES20.glUniform3f(uCameraDown, pose[1], pose[4], pose[7])
        GLES20.glUniform3f(uCameraForward, pose[2], pose[5], pose[8])

        GLES20.glUniform1f(uFx, if (fx > 1f) fx else 1f)
        GLES20.glUniform1f(uFy, if (fy > 1f) fy else 1f)
        GLES20.glUniform1f(uCx, cx)
        GLES20.glUniform1f(uCy, cy)
        GLES20.glUniform1f(uImageW, if (imageWidth > 0) imageWidth.toFloat() else 1f)
        GLES20.glUniform1f(uImageH, if (imageHeight > 0) imageHeight.toFloat() else 1f)
        GLES20.glUniform3f(uCameraToView0, cameraToView[0], cameraToView[1], cameraToView[2])
        GLES20.glUniform3f(uCameraToView1, cameraToView[3], cameraToView[4], cameraToView[5])
        GLES20.glUniform1f(uNear, nearPlane)
        GLES20.glUniform1f(uFar, farPlane)
        val a = alpha.coerceIn(0.05f, 1f)
        GLES20.glUniform1f(uAlpha, a)
        GLES20.glUniform1f(uUseTint, if (useTint) 1f else 0f)
        GLES20.glUniform3f(uTint, tintR, tintG, tintB)

        val blended = a < 0.999f
        if (blended) {
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        }

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo)
        val stride = MESH_VERTEX_FLOATS * 4
        GLES20.glEnableVertexAttribArray(aPosition)
        GLES20.glVertexAttribPointer(aPosition, 3, GLES20.GL_FLOAT, false, stride, 0)
        if (aNormal >= 0) {
            GLES20.glEnableVertexAttribArray(aNormal)
            GLES20.glVertexAttribPointer(aNormal, 3, GLES20.GL_FLOAT, false, stride, 3 * 4)
        }
        if (aColor >= 0) {
            GLES20.glEnableVertexAttribArray(aColor)
            GLES20.glVertexAttribPointer(aColor, 3, GLES20.GL_FLOAT, false, stride, 6 * 4)
        }

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, ibo)
        val type = if (supportsUintIndex) GLES20.GL_UNSIGNED_INT else GLES20.GL_UNSIGNED_SHORT
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, uploadedTriangleCount * 3, type, 0)

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        GLES20.glDisableVertexAttribArray(aPosition)
        if (aNormal >= 0) GLES20.glDisableVertexAttribArray(aNormal)
        if (aColor >= 0) GLES20.glDisableVertexAttribArray(aColor)
        if (blended) GLES20.glDisable(GLES20.GL_BLEND)

        return uploadedTriangleCount
    }

    private fun uploadIfNeeded() {
        if (pendingVersion == uploadedVersion) return
        val verts = pendingVertices
        val idx = pendingIndices
        if (verts == null || idx == null) {
            uploadedVersion = pendingVersion
            uploadedVertexCount = 0
            uploadedTriangleCount = 0
            return
        }
        if (verts.size < MESH_VERTEX_FLOATS * 3 || idx.size < 3) {
            lastError = "网格数据过小"
            uploadedVersion = pendingVersion
            uploadedVertexCount = 0
            uploadedTriangleCount = 0
            return
        }
        val vcount = verts.size / MESH_VERTEX_FLOATS
        if (!supportsUintIndex && vcount > 0xFFFF) {
            // 32 位索引不可用：宁可明确报错，也不要用 16 位索引去画一个
            // 顶点数溢出 65535 的网格 —— 那会画出完全错乱的三角形。
            lastError = "设备不支持 32 位索引，网格 $vcount 顶点超出 65535，请用「预览」质量"
            Log.w(TAG, lastError)
            uploadedVersion = pendingVersion
            uploadedVertexCount = 0
            uploadedTriangleCount = 0
            return
        }

        val vbuf = ByteBuffer.allocateDirect(verts.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        vbuf.put(verts)
        vbuf.flip()
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo)
        GLES20.glBufferData(
            GLES20.GL_ARRAY_BUFFER, verts.size * 4, vbuf, GLES20.GL_STATIC_DRAW
        )

        val triCount = idx.size / 3
        val idxBytes: Int
        if (supportsUintIndex) {
            val ibuf = ByteBuffer.allocateDirect(triCount * 3 * 4)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer()
            ibuf.put(idx, 0, triCount * 3)
            ibuf.flip()
            GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, ibo)
            GLES20.glBufferData(
                GLES20.GL_ELEMENT_ARRAY_BUFFER, triCount * 3 * 4, ibuf, GLES20.GL_STATIC_DRAW
            )
            idxBytes = triCount * 3 * 4
        } else {
            val ibuf = ByteBuffer.allocateDirect(triCount * 3 * 2)
                .order(ByteOrder.nativeOrder())
                .asShortBuffer()
            for (i in 0 until triCount * 3) ibuf.put(idx[i].toShort())
            ibuf.flip()
            GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, ibo)
            GLES20.glBufferData(
                GLES20.GL_ELEMENT_ARRAY_BUFFER, triCount * 3 * 2, ibuf, GLES20.GL_STATIC_DRAW
            )
            idxBytes = triCount * 3 * 2
        }
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)

        uploadedVersion = pendingVersion
        uploadedVertexCount = vcount
        uploadedTriangleCount = triCount
        lastError = ""
        Log.i(TAG, "mesh uploaded: verts=$vcount tris=$triCount idxBytes=$idxBytes")
    }

    private fun compile(type: Int, src: String): Int {
        val s = GLES20.glCreateShader(type)
        GLES20.glShaderSource(s, src)
        GLES20.glCompileShader(s)
        val ok = IntArray(1)
        GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            Log.e(TAG, "shader compile failed: ${GLES20.glGetShaderInfoLog(s)}")
            GLES20.glDeleteShader(s)
            return 0
        }
        return s
    }

    private fun link(vs: String, fs: String): Int {
        val v = compile(GLES20.GL_VERTEX_SHADER, vs)
        val f = compile(GLES20.GL_FRAGMENT_SHADER, fs)
        if (v == 0 || f == 0) return 0
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, v)
        GLES20.glAttachShader(p, f)
        GLES20.glLinkProgram(p)
        val ok = IntArray(1)
        GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, ok, 0)
        GLES20.glDeleteShader(v)
        GLES20.glDeleteShader(f)
        if (ok[0] == 0) {
            Log.e(TAG, "program link failed: ${GLES20.glGetProgramInfoLog(p)}")
            GLES20.glDeleteProgram(p)
            return 0
        }
        return p
    }

    companion object {
        private const val TAG = "MeshRenderer"

        /** 与 native `MESH_VERTEX_FLOATS` / `nativeGetMeshVertices` 一致。 */
        const val MESH_VERTEX_FLOATS = 9

        /**
         * 顶点着色器：与点云**共用同一条相机模型投影链**。
         *
         * 两个刻意的地方：
         *  1. 法线只用 pose 的旋转部分做同样的 `Rwc^T` 变换（不减去 twc），
         *     这才是方向向量的正确变换；
         *  2. 明暗用 `abs(dot(n, viewDir))` —— 网格法线来自 TSDF 梯度，
         *     朝向已经是对的，但 Taubin 平滑 / QEM 之后个别三角形仍可能翻面，
         *     用 abs 不会因为一个面朝向错误就出现纯黑碎片。
         */
        private val VERT = """
            #version 100
            attribute vec3 aPosition;
            attribute vec3 aNormal;
            attribute vec3 aColor;
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
            varying vec3 vColor;
            varying float vShade;
            void main() {
                vColor = aColor;
                vec3 dw = aPosition - uCameraT;
                vec3 pc = vec3(
                    dot(dw, uCameraRight),
                    dot(dw, uCameraDown),
                    dot(dw, uCameraForward)
                );
                vec3 nw = aNormal;
                vec3 nc = vec3(
                    dot(nw, uCameraRight),
                    dot(nw, uCameraDown),
                    dot(nw, uCameraForward)
                );
                float nl = length(nc);
                if (nl > 1e-5) {
                    nc = nc / nl;
                } else {
                    nc = vec3(0.0, 0.0, 1.0);
                }
                vec3 vd = vec3(0.0, 0.0, 1.0);
                if (length(pc) > 1e-5) {
                    vd = normalize(-pc);
                }
                vShade = 0.35 + 0.65 * abs(dot(nc, vd));
                if (pc.z <= 0.05) {
                    gl_Position = vec4(2.0, 2.0, 1.0, 1.0);
                    return;
                }
                float cameraU = (uFx * pc.x / pc.z + uCx) / uImageWidth;
                float cameraV = (uFy * pc.y / pc.z + uCy) / uImageHeight;
                vec3 uv = vec3(cameraU, cameraV, 1.0);
                float viewU = dot(uCameraToView0, uv);
                float viewV = dot(uCameraToView1, uv);
                float ndcX = 2.0 * viewU - 1.0;
                float ndcY = 1.0 - 2.0 * viewV;
                float depth01 = clamp((pc.z - uNear) / (uFar - uNear), 0.0, 1.0);
                gl_Position = vec4(ndcX, ndcY, depth01 * 2.0 - 1.0, 1.0);
            }
        """.trimIndent()

        private val FRAG = """
            #version 100
            precision mediump float;
            varying vec3 vColor;
            varying float vShade;
            uniform float uAlpha;
            uniform float uUseTint;
            uniform vec3 uTint;
            void main() {
                // V0.12：用 mix 而不是向量三元 —— GLSL ES 1.00 下各驱动对
                // 向量三元的支持程度不一，mix 是核心函数，没有兼容风险。
                vec3 base = mix(vColor, uTint, step(0.5, uUseTint));
                gl_FragColor = vec4(base * vShade, uAlpha);
            }
        """.trimIndent()
    }
}
