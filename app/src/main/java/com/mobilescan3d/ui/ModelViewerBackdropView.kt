package com.mobilescan3d.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * Neutral background for the dedicated 3D model viewer.
 *
 * The camera preview is intentionally hidden in viewer mode. A subtle horizon
 * and perspective floor grid make depth/scale easier to read without mixing the
 * reconstructed model with the live camera scene.
 */
class ModelViewerBackdropView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x24496784
        strokeWidth = dp(0.75f)
    }
    private val majorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x3D4B7EA5
        strokeWidth = dp(1.0f)
    }
    private val horizonPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x4C2AA8FF
        strokeWidth = dp(1.1f)
    }
    private val centerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x7A2AA8FF
        strokeWidth = dp(1.15f)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        bgPaint.shader = LinearGradient(
            0f, 0f, 0f, h,
            intArrayOf(0xFF07101A.toInt(), 0xFF050A10.toInt(), 0xFF020509.toInt()),
            floatArrayOf(0f, 0.56f, 1f),
            Shader.TileMode.CLAMP
        )
        canvas.drawRect(0f, 0f, w, h, bgPaint)

        val horizon = h * 0.61f
        canvas.drawLine(0f, horizon, w, horizon, horizonPaint)

        // Perspective rays from a vanishing point slightly above the center.
        val vx = w * 0.5f
        val vy = horizon
        val bottom = h
        val rayCount = 12
        for (i in -rayCount..rayCount) {
            val x = vx + i * (w / 8f)
            val p = if (i % 3 == 0) majorPaint else gridPaint
            canvas.drawLine(vx, vy, x, bottom, p)
        }

        // Horizontal floor lines get denser toward the horizon.
        val floorH = max(1f, bottom - horizon)
        for (i in 1..12) {
            val t = i / 12f
            val eased = t * t
            val y = horizon + floorH * eased
            canvas.drawLine(0f, y, w, y, if (i % 3 == 0) majorPaint else gridPaint)
        }

        // Small center locator helps judge panning without competing with model.
        val c = dp(8f)
        canvas.drawLine(vx - c, horizon, vx + c, horizon, centerPaint)
        canvas.drawLine(vx, horizon - c, vx, horizon + c, centerPaint)
    }

    private fun dp(v: Float): Float = v * resources.displayMetrics.density
}
