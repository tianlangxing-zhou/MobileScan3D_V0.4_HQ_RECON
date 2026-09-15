package com.mobilescan3d

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class PointCloudRenderer : GLSurfaceView.Renderer {

    private var program = 0
    private var posLoc = 0
    private var colorLoc = 0
    private var mvpLoc = 0
    private var aspect = 1f
    private var startMs = System.currentTimeMillis()
    @Volatile var rotX = 0f
    @Volatile var rotY = 0f
    @Volatile var zoom = 1f
    private val vertexData = FloatArray(MAX_POINTS * 6)
    private var vertexBuffer: FloatBuffer = ByteBuffer.allocateDirect(MAX_POINTS * 6 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        program = link(VERT, FRAG)
        posLoc = GLES20.glGetAttribLocation(program, "aPosition")
        colorLoc = GLES20.glGetAttribLocation(program, "aColor")
        mvpLoc = GLES20.glGetUniformLocation(program, "uMvp")
        GLES20.glClearColor(0f, 0f, 0f, 0f)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        aspect = if (height == 0) 1f else width.toFloat() / height.toFloat()
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val count = NativeBridge.nativeGetGaussians(vertexData, MAX_POINTS)
        if (count <= 0) {
            return
        }

        vertexBuffer.clear()
        vertexBuffer.put(vertexData, 0, count * 6)
        vertexBuffer.flip()

        val autoYaw = ((System.currentTimeMillis() - startMs) / 1000f) * 35f
        val model = FloatArray(16)
        Matrix.setIdentityM(model, 0)
        Matrix.scaleM(model, 0, 0.35f * zoom, 0.35f * zoom, 0.35f * zoom)
        Matrix.rotateM(model, 0, rotX, 1f, 0f, 0f)
        Matrix.rotateM(model, 0, rotY + autoYaw, 0f, 1f, 0f)

        val view = FloatArray(16)
        Matrix.setLookAtM(view, 0, 0f, 0.4f, 3f, 0f, 0.2f, 0f, 0f, 1f, 0f)
        val proj = FloatArray(16)
        Matrix.perspectiveM(proj, 0, 60f, aspect, 0.1f, 20f)
        val vp = FloatArray(16)
        Matrix.multiplyMM(vp, 0, proj, 0, view, 0)
        val mvp = FloatArray(16)
        Matrix.multiplyMM(mvp, 0, vp, 0, model, 0)

        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(mvpLoc, 1, false, mvp, 0)

        vertexBuffer.position(0)
        GLES20.glVertexAttribPointer(posLoc, 3, GLES20.GL_FLOAT, false, 6 * 4, vertexBuffer)
        GLES20.glEnableVertexAttribArray(posLoc)
        vertexBuffer.position(3)
        GLES20.glVertexAttribPointer(colorLoc, 3, GLES20.GL_FLOAT, false, 6 * 4, vertexBuffer)
        GLES20.glEnableVertexAttribArray(colorLoc)

        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count)

        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(colorLoc)
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
        private const val MAX_POINTS = 80000

        private val VERT = """
            #version 100
            attribute vec3 aPosition;
            attribute vec3 aColor;
            uniform mat4 uMvp;
            varying vec3 vColor;
            void main() {
                gl_Position = uMvp * vec4(aPosition, 1.0);
                gl_PointSize = 3.0;
                vColor = aColor;
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
