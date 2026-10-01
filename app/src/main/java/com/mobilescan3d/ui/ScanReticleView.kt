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

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x223C4B5E
        strokeWidth = dp(1f)
    }

    private val bracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.scan_primary)
        strokeWidth = dp(3f)
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val crossPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF667386.toInt()
        strokeWidth = dp(2f)
        strokeCap = Paint.Cap.ROUND
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // subtle 3x3 grid
        val w = width.toFloat()
        val h = height.toFloat()
        canvas.drawLine(w / 3f, 0f, w / 3f, h, gridPaint)
        canvas.drawLine(w * 2f / 3f, 0f, w * 2f / 3f, h, gridPaint)
        canvas.drawLine(0f, h / 3f, w, h / 3f, gridPaint)
        canvas.drawLine(0f, h * 2f / 3f, w, h * 2f / 3f, gridPaint)

        // scan frame
        val frameW = w * 0.36f
        val frameH = h * 0.22f
        val left = (w - frameW) / 2f
        val top = (h - frameH) / 2f - dp(20f)
        val right = left + frameW
        val bottom = top + frameH
        val len = dp(26f)
        val r = dp(10f)

        // corners
        canvas.drawArc(RectF(left, top, left + r*2, top + r*2), 180f, 90f, false, bracketPaint)
        canvas.drawLine(left + r, top, left + len, top, bracketPaint)
        canvas.drawLine(left, top + r, left, top + len, bracketPaint)

        canvas.drawArc(RectF(right - r*2, top, right, top + r*2), 270f, 90f, false, bracketPaint)
        canvas.drawLine(right - len, top, right - r, top, bracketPaint)
        canvas.drawLine(right, top + r, right, top + len, bracketPaint)

        canvas.drawArc(RectF(left, bottom - r*2, left + r*2, bottom), 90f, 90f, false, bracketPaint)
        canvas.drawLine(left, bottom - len, left, bottom - r, bracketPaint)
        canvas.drawLine(left + r, bottom, left + len, bottom, bracketPaint)

        canvas.drawArc(RectF(right - r*2, bottom - r*2, right, bottom), 0f, 90f, false, bracketPaint)
        canvas.drawLine(right, bottom - len, right, bottom - r, bracketPaint)
        canvas.drawLine(right - len, bottom, right - r, bottom, bracketPaint)

        // center +
        val cx = w / 2f
        val cy = top + frameH / 2f
        val cross = dp(14f)
        canvas.drawLine(cx - cross, cy, cx + cross, cy, crossPaint)
        canvas.drawLine(cx, cy - cross, cx, cy + cross, crossPaint)
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
}
