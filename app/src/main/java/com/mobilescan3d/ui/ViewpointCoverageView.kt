package com.mobilescan3d.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.mobilescan3d.R
import kotlin.math.roundToInt

/**
 * Scanner viewpoint-coverage heat ring.
 *
 * Slots 0..11 are horizontal orbit sectors. Slot 12 is elevated/top coverage,
 * slot 13 is low/bottom coverage. Values are [0,1].
 *
 * Important: this visualizes captured viewpoints, not exact geometric
 * surface-area coverage. That distinction is intentional: a viewpoint signal
 * can be derived robustly from the existing VINS pose without pretending that
 * an unseen surface has already been measured.
 */
class ViewpointCoverageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val coverage = FloatArray(14)
    private var currentSector = -1
    private var percent = 0

    private val trackColor = 0x3A526A82
    private val danger = ContextCompat.getColor(context, R.color.scan_danger)
    private val warning = ContextCompat.getColor(context, R.color.scan_warning)
    private val success = ContextCompat.getColor(context, R.color.scan_success)
    private val primary = ContextCompat.getColor(context, R.color.scan_primary)
    private val textPrimary = ContextCompat.getColor(context, R.color.scan_text_primary)
    private val textSecondary = ContextCompat.getColor(context, R.color.scan_text_secondary)

    private val arcPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(7f)
        strokeCap = Paint.Cap.BUTT
    }

    private val currentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        strokeCap = Paint.Cap.ROUND
        color = primary
    }

    private val capTrackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = trackColor
    }

    private val capPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val percentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = textPrimary
        textAlign = Paint.Align.CENTER
        textSize = sp(13f)
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = textSecondary
        textAlign = Paint.Align.CENTER
        textSize = sp(7.5f)
    }

    fun setCoverage(values: FloatArray, activeSector: Int, shownPercent: Int) {
        for (i in coverage.indices) {
            coverage[i] = values.getOrElse(i) { 0f }.coerceIn(0f, 1f)
        }
        currentSector = activeSector
        percent = shownPercent.coerceIn(0, 100)
        invalidate()
    }

    fun clearCoverage() {
        coverage.fill(0f)
        currentSector = -1
        percent = 0
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width * 0.5f
        val cy = height * 0.50f
        val radius = minOf(width, height) * 0.34f
        val oval = RectF(cx - radius, cy - radius, cx + radius, cy + radius)

        val gap = 4f
        val sweep = 360f / 12f
        for (i in 0 until 12) {
            val start = -90f + i * sweep + gap * 0.5f
            arcPaint.color = trackColor
            canvas.drawArc(oval, start, sweep - gap, false, arcPaint)

            val value = coverage[i]
            if (value > 0.03f) {
                arcPaint.color = heatColor(value)
                canvas.drawArc(oval, start, (sweep - gap) * value, false, arcPaint)
            }

            if (i == currentSector) {
                val markerOval = RectF(
                    oval.left - dp(5f), oval.top - dp(5f),
                    oval.right + dp(5f), oval.bottom + dp(5f)
                )
                canvas.drawArc(markerOval, start + 3f, sweep - gap - 6f, false, currentPaint)
            }
        }

        // Top/bottom viewpoint caps.
        val capR = dp(4.5f)
        val capX = cx + radius + dp(11f)
        val topY = cy - dp(12f)
        val bottomY = cy + dp(12f)
        canvas.drawCircle(capX, topY, capR + dp(1.5f), capTrackPaint)
        canvas.drawCircle(capX, bottomY, capR + dp(1.5f), capTrackPaint)
        capPaint.color = heatColor(coverage[12])
        canvas.drawCircle(capX, topY, capR * coverage[12].coerceAtLeast(0.16f), capPaint)
        capPaint.color = heatColor(coverage[13])
        canvas.drawCircle(capX, bottomY, capR * coverage[13].coerceAtLeast(0.16f), capPaint)

        canvas.drawText("$percent%", cx, cy + dp(2f), percentPaint)
        canvas.drawText("视角覆盖", cx, cy + dp(14f), labelPaint)
        canvas.drawText("上", capX, topY - dp(8f), labelPaint)
        canvas.drawText("下", capX, bottomY + dp(11f), labelPaint)
    }

    private fun heatColor(value: Float): Int = when {
        value >= 0.72f -> success
        value >= 0.34f -> warning
        else -> danger
    }

    private fun dp(value: Float): Float = value * resources.displayMetrics.density
    private fun sp(value: Float): Float = value * resources.displayMetrics.scaledDensity
}
