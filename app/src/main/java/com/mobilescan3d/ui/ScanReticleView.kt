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

    private var scanning = false
    private var paused = false
    private var warning = false
    private val primary = ContextCompat.getColor(context, R.color.scan_primary)
    private val warn = ContextCompat.getColor(context, R.color.scan_warning)
    private val muted = ContextCompat.getColor(context, R.color.scan_text_muted)
    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = dp(1.5f)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    @Suppress("UNUSED_PARAMETER") // Keep the existing UI update contract.
    fun setScanUiState(progress: Float, active: Boolean, isPaused: Boolean, hasWarning: Boolean) {
        if (scanning == active && paused == isPaused && warning == hasWarning) return
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

        // Size against the actual free workspace, after cutout and controls.
        // Keep the upper corners below the two shortcuts, including large text.
        val shortcutHeight = (parent as? View)?.findViewById<View>(R.id.toolRail)?.height ?: 0
        val topInset = maxOf(dp(136f), shortcutHeight.toFloat() + dp(8f))
        val usableHeight = h - topInset - dp(16f)
        val frameW = w * 0.72f
        val frameH = minOf(frameW * 1.22f, usableHeight)
        if (frameH < dp(48f)) return
        val centerX = w * 0.5f
        val centerY = topInset + usableHeight * 0.5f
        val left = centerX - frameW / 2f
        val right = centerX + frameW / 2f
        val top = centerY - frameH / 2f
        val bottom = centerY + frameH / 2f

        framePaint.color = when {
            warning -> warn
            paused -> muted
            else -> primary
        }
        framePaint.alpha = if (warning) 220 else if (scanning) 145 else 110
        // Four quiet corners only: no grid, center dot, glow or duplicate progress.
        val corner = minOf(dp(18f), frameH * 0.18f)
        val radius = minOf(dp(6f), corner * 0.4f)
        drawCorners(canvas, left, top, right, bottom, corner, radius, framePaint)
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
