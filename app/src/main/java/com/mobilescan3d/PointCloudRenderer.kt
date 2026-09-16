package com.mobilescan3d

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * AR 点云渲染器。
 *
 * ## 为什么不再用 lookAt + perspectiveM
 *
 * 旧实现是这样画点云的：
 * ```
 * World Point -> VINS pose -> Matrix.setLookAtM + Matrix.perspectiveM -> Portrait GLSurfaceView
 * ```
 * 也就是说，**它把点云当成一个独立的 3D viewer，用一个通用透视相机去拍**。
 * 而屏幕上真正显示画面的 Camera Preview 走的是完全另一条链：
 * ```
 * Preview -> SurfaceTexture transform -> 90° portrait -> center crop
 * ```
 * 两条链的屏幕坐标系互不相干，于是：
 *   - 点云和真实画面没有正确的 AR 视差关系（像贴了一层不动的贴纸）；
 *   - `fx / cx / cy` 收了却没用，只拿 `fy` 反推了个 FOV。
 *
 * 现在改成直接复现 Camera Preview 的投影过程：
 * ```
 * World Point Pw
 *      -> Pc = Rwc^T (Pw - twc)                相机坐标系
 *      -> cameraUV = (fx*X/Z + cx, fy*Y/Z + cy) / imageSize
 *      -> viewUV = cameraToView * cameraUV      SurfaceTexture affine + TextureView transform + crop
 *      -> NDC
 * ```
 * `cameraToView` 由 MainActivity 用**已经验证正确**的那条逆映射
 * （点选目标用的 viewToCameraNorm）反求出来，所以不需要去猜 90°、也不需要
 * 在 shader 里硬编码任何屏幕方向。
 */
class PointCloudRenderer : GLSurfaceView.Renderer {

    private var program = 0
    private var posLoc = 0
    private var colorLoc = 0
    private var uCameraTLoc = 0
    private var uCamRightLoc = 0
    private var uCamDownLoc = 0
    private var uCamForwardLoc = 0
    private var uFxLoc = 0
    private var uFyLoc = 0
    private var uCxLoc = 0
    private var uCyLoc = 0
    private var uImageWLoc = 0
    private var uImageHLoc = 0
    private var uCameraToView0Loc = 0
    private var uCameraToView1Loc = 0
    private var uNearLoc = 0
    private var uFarLoc = 0
    private var uPointSizeLoc = 0
    private var uPointSpaceLoc = 0
    private var viewportW = 1
    private var viewportH = 1

    // ---- 相机内参（nativeCreate 时下发的就是这一套，必须完全一致）----
    @Volatile private var cameraFx = 1100f
    @Volatile private var cameraFy = 901f
    @Volatile private var cameraCx = 640f
    @Volatile private var cameraCy = 480f
    @Volatile private var cameraImageWidth = 1280
    @Volatile private var cameraImageHeight = 960
    @Volatile private var haveCameraModel = false

    /**
     * camera-normalized UV -> view-normalized 的 2x3 仿射：
     *   viewU = m[0]*cameraU + m[1]*cameraV + m[2]
     *   viewV = m[3]*cameraU + m[4]*cameraV + m[5]
     *
     * 由 MainActivity.updateCameraToViewTransform() 算出。
     */
    @Volatile private var cameraToView = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f)
    @Volatile private var haveCameraToView = false

    /** Preview 的时间戳（SurfaceTexture.getTimestamp），用来查那一时刻的 VINS pose */
    @Volatile private var previewTimestampNs = 0L
    /** 上一次绘制有没有成功按 Preview 时间戳取到 pose（false 表示走的是回退） */
    @Volatile var poseFromTimestamp = false
        private set

    /** 绘制模式：见 DRAW_* 常量 */
    @Volatile var drawMode = DRAW_TARGET_DEBUG
    /** 累计点云的 hits 过滤门限 */
    @Volatile var accumulatedMinHits = NativeBridge.AR_MIN_HITS_CONFIRMED
    /** 最近一次实际画出的点数（HUD / 报告用） */
    @Volatile var drawnAccumulated = 0
        private set
    @Volatile var drawnDebug = 0
        private set

    @Volatile private var accumPointSize = 3f
    @Volatile private var debugPointSize = 5f

    private val poseBuf = FloatArray(NativeBridge.RENDER_POSE_SLOTS)
    private val accumData = FloatArray(NativeBridge.AR_MAX_POINTS * NativeBridge.POINT_SLOTS)
    private val debugData =
        FloatArray(NativeBridge.AR_TARGET_DEBUG_MAX_POINTS * NativeBridge.POINT_SLOTS)

    private val accumBuffer: FloatBuffer =
        ByteBuffer.allocateDirect(NativeBridge.AR_MAX_POINTS * NativeBridge.POINT_SLOTS * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    private val debugBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(NativeBridge.AR_TARGET_DEBUG_MAX_POINTS * NativeBridge.POINT_SLOTS * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    // ------------------------------------------------------------------ setters

    /** Preview 帧的时间戳，必须在 onSurfaceTextureUpdated() 里更新 */
    fun setPreviewTimestamp(ts: Long) {
        previewTimestampNs = ts
    }

    /**
     * 设置 camera-normalized UV -> view-normalized 的 2x3 仿射。
     * 由 MainActivity 在 configureTransform / stAffine 变化后调用。
     */
    fun setCameraToView(m: FloatArray) {
        if (m.size < 6) return
        cameraToView = floatArrayOf(m[0], m[1], m[2], m[3], m[4], m[5])
        haveCameraToView = true
    }

    fun setCameraModel(fx: Float, fy: Float, cx: Float, cy: Float, imageWidth: Int, imageHeight: Int) {
        if (fx > 1f && fy > 1f && imageWidth > 0 && imageHeight > 0) {
            cameraFx = fx
            cameraFy = fy
            cameraCx = cx
            cameraCy = cy
            cameraImageWidth = imageWidth
            cameraImageHeight = imageHeight
            haveCameraModel = true
        }
    }

    /** 兼容旧调用点：只有 fy 与高度时按同一缩放系数补齐 */
    fun setCameraIntrinsics(fy: Float, imageHeight: Int) {
        if (fy > 1f && imageHeight > 0) {
            setCameraModel(cameraFx, fy, cameraImageWidth / 2f, imageHeight / 2f,
                cameraImageWidth, imageHeight)
        }
    }

    fun setPointSizes(accumulated: Float, debug: Float) {
        if (accumulated > 0f) accumPointSize = accumulated
        if (debug > 0f) debugPointSize = debug
    }

    // ------------------------------------------------------------------ GL 回调

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        program = link(VERT, FRAG)
        posLoc = GLES20.glGetAttribLocation(program, "aPosition")
        colorLoc = GLES20.glGetAttribLocation(program, "aColor")
        uCameraTLoc = GLES20.glGetUniformLocation(program, "uCameraT")
        uCamRightLoc = GLES20.glGetUniformLocation(program, "uCameraRight")
        uCamDownLoc = GLES20.glGetUniformLocation(program, "uCameraDown")
        uCamForwardLoc = GLES20.glGetUniformLocation(program, "uCameraForward")
        uFxLoc = GLES20.glGetUniformLocation(program, "uFx")
        uFyLoc = GLES20.glGetUniformLocation(program, "uFy")
        uCxLoc = GLES20.glGetUniformLocation(program, "uCx")
        uCyLoc = GLES20.glGetUniformLocation(program, "uCy")
        uImageWLoc = GLES20.glGetUniformLocation(program, "uImageWidth")
        uImageHLoc = GLES20.glGetUniformLocation(program, "uImageHeight")
        uCameraToView0Loc = GLES20.glGetUniformLocation(program, "uCameraToView0")
        uCameraToView1Loc = GLES20.glGetUniformLocation(program, "uCameraToView1")
        uNearLoc = GLES20.glGetUniformLocation(program, "uNear")
        uFarLoc = GLES20.glGetUniformLocation(program, "uFar")
        uPointSizeLoc = GLES20.glGetUniformLocation(program, "uPointSize")
        uPointSpaceLoc = GLES20.glGetUniformLocation(program, "uPointSpace")
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthFunc(GLES20.GL_LEQUAL)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportW = if (width > 0) width else 1
        viewportH = if (height > 0) height else 1
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        drawnAccumulated = 0
        drawnDebug = 0

        // 坐标链没就绪时宁可不画：画出来的点位置一定是错的，
        // 那种「看起来有点云」比「什么都没有」更难排查。
        if (!haveCameraModel || !haveCameraToView) {
            poseFromTimestamp = false
            return
        }

        // ---- 位姿：优先按 Preview 的时间戳查历史 ----
        // 屏幕上这一帧画面有它自己的 SENSOR_TIMESTAMP；拿「此刻最新」的 pose
        // 去画它，手机一转动点云就会漂。只有查不到时才回退。
        val ts = previewTimestampNs
        var ok = false
        if (ts > 0L) {
            ok = try {
                NativeBridge.nativeGetRenderPoseAt(ts, poseBuf)
            } catch (t: Throwable) {
                false
            }
        }
        poseFromTimestamp = ok
        if (!ok) {
            ok = try {
                NativeBridge.nativeGetRenderPose(poseBuf)
            } catch (t: Throwable) {
                false
            }
        }
        if (!ok) return

        val wantAccum = drawMode != DRAW_TARGET_DEBUG
        val wantDebug = drawMode != DRAW_ACCUMULATED

        var accumCount = 0
        if (wantAccum) {
            accumCount = try {
                NativeBridge.nativeGetGaussians(
                    accumData, NativeBridge.AR_MAX_POINTS, accumulatedMinHits
                )
            } catch (t: Throwable) {
                0
            }
        }
        var debugCount = 0
        if (wantDebug) {
            debugCount = try {
                NativeBridge.nativeGetTargetDepthDebug(
                    debugData, NativeBridge.AR_TARGET_DEBUG_MAX_POINTS
                )
            } catch (t: Throwable) {
                0
            }
        }
        if (accumCount <= 0 && debugCount <= 0) return

        GLES20.glUseProgram(program)

        // 世界点 -> 相机：Pc = Rwc^T (Pw - twc)。
        // pose 布局是 [R(9) 行主序, t(3)]，所以 Rwc 的三列就是
        // (R[0],R[3],R[6]) / (R[1],R[4],R[7]) / (R[2],R[5],R[8])。
        GLES20.glUniform3f(uCameraTLoc, poseBuf[9], poseBuf[10], poseBuf[11])
        GLES20.glUniform3f(uCamRightLoc, poseBuf[0], poseBuf[3], poseBuf[6])
        GLES20.glUniform3f(uCamDownLoc, poseBuf[1], poseBuf[4], poseBuf[7])
        GLES20.glUniform3f(uCamForwardLoc, poseBuf[2], poseBuf[5], poseBuf[8])

        GLES20.glUniform1f(uFxLoc, cameraFx.coerceAtLeast(1f))
        GLES20.glUniform1f(uFyLoc, cameraFy.coerceAtLeast(1f))
        GLES20.glUniform1f(uCxLoc, cameraCx)
        GLES20.glUniform1f(uCyLoc, cameraCy)
        GLES20.glUniform1f(uImageWLoc, cameraImageWidth.coerceAtLeast(1).toFloat())
        GLES20.glUniform1f(uImageHLoc, cameraImageHeight.coerceAtLeast(1).toFloat())

        val m = cameraToView
        GLES20.glUniform3f(uCameraToView0Loc, m[0], m[1], m[2])
        GLES20.glUniform3f(uCameraToView1Loc, m[3], m[4], m[5])

        GLES20.glUniform1f(uNearLoc, NEAR_PLANE)
        GLES20.glUniform1f(uFarLoc, FAR_PLANE)

        if (accumCount > 0) {
            accumBuffer.clear()
            accumBuffer.put(accumData, 0, accumCount * NativeBridge.POINT_SLOTS)
            accumBuffer.flip()
            // 累计模型是 world 空间：需要 Rwc^T (Pw - twc) 换到相机系
            drawPoints(accumBuffer, accumCount, accumPointSize, POINT_SPACE_WORLD)
            drawnAccumulated = accumCount
        }
        if (debugCount > 0) {
            debugBuffer.clear()
            debugBuffer.put(debugData, 0, debugCount * NativeBridge.POINT_SLOTS)
            debugBuffer.flip()
            // 当前帧目标 live 点是**相机坐标**：直接投影，完全不经过 VINS 位姿
            drawPoints(debugBuffer, debugCount, debugPointSize, POINT_SPACE_CAMERA)
            drawnDebug = debugCount
        }

        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(colorLoc)
    }

    private fun drawPoints(buffer: FloatBuffer, count: Int, size: Float, pointSpace: Float) {
        GLES20.glUniform1f(uPointSizeLoc, size)
        GLES20.glUniform1f(uPointSpaceLoc, pointSpace)
        buffer.position(0)
        GLES20.glVertexAttribPointer(
            posLoc, 3, GLES20.GL_FLOAT, false, NativeBridge.POINT_SLOTS * 4, buffer
        )
        GLES20.glEnableVertexAttribArray(posLoc)
        buffer.position(3)
        GLES20.glVertexAttribPointer(
            colorLoc, 3, GLES20.GL_FLOAT, false, NativeBridge.POINT_SLOTS * 4, buffer
        )
        GLES20.glEnableVertexAttribArray(colorLoc)
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count)
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        return shader
    }

    private fun link(vertexSource: String, fragmentSource: String): Int {
        val vs = compile(GLES20.GL_VERTEX_SHADER, vertexSource)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, vs)
        GLES20.glAttachShader(p, fs)
        GLES20.glLinkProgram(p)
        GLES20.glDeleteShader(vs)
        GLES20.glDeleteShader(fs)
        return p
    }

    companion object {
        /** 只画当前帧 target depth 调试层（验证 AR 坐标链用，默认） */
        const val DRAW_TARGET_DEBUG = 0
        /** 只画累计世界点云 */
        const val DRAW_ACCUMULATED = 1
        /** 两者都画 */
        const val DRAW_BOTH = 2

        /** 视锥裁剪用的近/远平面。只影响点云自身的深度排序，不影响屏幕位置。 */
        private const val NEAR_PLANE = 0.05f
        private const val FAR_PLANE = 20f

        /**
         * uPointSpace 取值：点的坐标到底在哪个空间里。
         *
         *   POINT_SPACE_CAMERA (0) —— aPosition 已经是**相机坐标** (Xc, Yc, Zc)。
         *       当前帧的目标 live 点属于这一种。它和这一帧的 depth 同一个坐标系，
         *       所以直接投影就行，**完全不需要 VINS 位姿**，也就不受
         *       `arPoseFromTimestamp=false` 的影响。
         *
         *   POINT_SPACE_WORLD (1) —— aPosition 是世界坐标，需要
         *       `pc = Rwc^T (Pw - twc)` 换到相机系。累计目标模型属于这一种。
         *
         * 这两条链以前混在一起（当前帧的点也先转 world 再用另一个时刻的 pose
         * 投回去），所以「Mask 对不对」和「pose 链对不对」根本分不开。
         */
        private const val POINT_SPACE_CAMERA = 0f
        private const val POINT_SPACE_WORLD = 1f

        /**
         * 顶点着色器 —— 直接实现 Camera2 的相机模型投影。
         *
         * 关键点：**不要在这里猜屏幕旋转**。cameraU/cameraV 已经是
         * 「相机归一化坐标」，剩下的 SurfaceTexture 仿射、TextureView 变换、
         * center crop 全部由 uCameraToView 一次性表达，而那套矩阵来自
         * 点选目标时已经验证过的逆映射链。
         */
        private val VERT = """
            #version 100
            attribute vec3 aPosition;
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
            uniform float uPointSize;
            uniform float uPointSpace;
            varying vec3 vColor;
            void main() {
                vColor = aColor;
                vec3 pc;
                if (uPointSpace < 0.5) {
                    // 相机空间：坐标本来就是 (Xc, Yc, Zc)，直接用
                    pc = aPosition;
                } else {
                    // 世界空间：Pc = Rwc^T (Pw - twc)
                    vec3 dw = aPosition - uCameraT;
                    pc = vec3(
                        dot(dw, uCameraRight),
                        dot(dw, uCameraDown),
                        dot(dw, uCameraForward)
                    );
                }
                if (pc.z <= 0.05) {
                    // 相机背后的点：推出裁剪空间并让点尺寸归零
                    gl_Position = vec4(2.0, 2.0, 1.0, 1.0);
                    gl_PointSize = 0.0;
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
                gl_PointSize = uPointSize;
            }
        """.trimIndent()

        private val FRAG = """
            #version 100
            precision mediump float;
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0);
            }
        """.trimIndent()
    }
}
