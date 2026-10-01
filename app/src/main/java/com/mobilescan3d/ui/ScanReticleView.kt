package com.mobilescan3d.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.mobilescan3d.R

class ScanReticleView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var scanProgress = 0f
    private var scanning = false
    private var paused = false
    private var warning = false

    private val primary = ContextCompat.getColor(context, R.color.scan_primary)
    private val success = ContextCompat.getColor(context, R.color.scan_success)
    private val warn = ContextCompat.getColor(context, R.color.scan_warning)
    private val muted = ContextCompat.getColor(context, R.color.scan_text_muted)

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x183B5C78
        strokeWidth = dp(0.8f)
    }

    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = dp(2.6f)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val frameGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = dp(7f)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val centerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xCC9DB0C4.toInt()
        strokeWidth = dp(1.5f)
        style = Paint.Style.STROKE
    }

    private val centerDotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = primary
        style = Paint.Style.FILL
    }

    private val sideTickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x554E718F
        strokeWidth = dp(1f)
        strokeCap = Paint.Cap.ROUND
    }

    private val progressTrackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x443A5168
        strokeWidth = dp(3f)
        strokeCap = Paint.Cap.ROUND
    }

    private val progressPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = primary
        strokeWidth = dp(3f)
        strokeCap = Paint.Cap.ROUND
    }

    fun setScanUiState(progress: Float, active: Boolean, isPaused: Boolean, hasWarning: Boolean) {
        scanProgress = progress.coerceIn(0f, 1f)
        scanning = active
        paused = isPaused
        warning = hasWarning
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        val centerX = w * 0.46f
        val centerY = h * 0.50f
        val frameW = w * 0.50f
        val frameH = frameW * 1.12f
        val left = centerX - frameW / 2f
        val top = centerY - frameH / 2f
        val right = centerX + frameW / 2f
        val bottom = centerY + frameH / 2f

        canvas.save()
        canvas.clipRect(left, top, right, bottom)
        for (i in 1..2) {
            val x = left + frameW * i / 3f
            val y = top + frameH * i / 3f
            canvas.drawLine(x, top, x, bottom, gridPaint)
            canvas.drawLine(left, y, right, y, gridPaint)
        }
        canvas.restore()

        val stateColor = when {
            warning -> warn
            paused -> muted
            scanning -> primary
            else -> primary
        }
        framePaint.color = stateColor
        frameGlowPaint.color = when {
            warning -> 0x45FFB84D
            paused -> 0x33728096
            else -> 0x402AA8FF
        }
        progressPaint.color = when {
            warning -> warn
            paused -> muted
            scanProgress >= 0.82f -> success
            else -> primary
        }
        centerDotPaint.color = stateColor

        val corner = dp(25f)
        val radius = dp(10f)
        drawCorners(canvas, left, top, right, bottom, corner, radius, frameGlowPaint)
        drawCorners(canvas, left, top, right, bottom, corner, radius, framePaint)

        canvas.drawCircle(centerX, centerY, dp(8f), centerPaint)
        canvas.drawCircle(centerX, centerY, dp(2.3f), centerDotPaint)

        val tick = dp(8f)
        canvas.drawLine(left - tick, centerY, left - dp(2f), centerY, sideTickPaint)
        canvas.drawLine(right + dp(2f), centerY, right + tick, centerY, sideTickPaint)
        canvas.drawLine(centerX, top - tick, centerX, top - dp(2f), sideTickPaint)
        canvas.drawLine(centerX, bottom + dp(2f), centerX, bottom + tick, sideTickPaint)

        // A subtle progress rail is part of the reticle itself, so users do not
        // need to look away from the object to understand cumulative capture.
        val progressY = bottom + dp(14f)
        canvas.drawLine(left + dp(8f), progressY, right - dp(8f), progressY, progressTrackPaint)
        val usable = frameW - dp(16f)
        canvas.drawLine(
            left + dp(8f),
            progressY,
            left + dp(8f) + usable * scanProgress,
            progressY,
            progressPaint
        )
    }

    private fun drawCorners(
        canvas: Canvas,
        left: Float,
        top: Float,
        right: Float,
        bottom: Float,
        len: Float,
        r: Float,
        paint: Paint
    ) {
        canvas.drawArc(RectF(left, top, left + r * 2f, top + r * 2f), 180f, 90f, false, paint)
        canvas.drawLine(left + r, top, left + len, top, paint)
        canvas.drawLine(left, top + r, left, top + len, paint)

        canvas.drawArc(RectF(right - r * 2f, top, right, top + r * 2f), 270f, 90f, false, paint)
        canvas.drawLine(right - len, top, right - r, top, paint)
        canvas.drawLine(right, top + r, right, top + len, paint)

        canvas.drawArc(RectF(left, bottom - r * 2f, left + r * 2f, bottom), 90f, 90f, false, paint)
        canvas.drawLine(left, bottom - len, left, bottom - r, paint)
        canvas.drawLine(left + r, bottom, left + len, bottom, paint)

        canvas.drawArc(RectF(right - r * 2f, bottom - r * 2f, right, bottom), 0f, 90f, false, paint)
        canvas.drawLine(right, bottom - len, right, bottom - r, paint)
        canvas.drawLine(right - len, bottom, right - r, bottom, paint)
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
