package com.mobilescan3d.ui.guidance

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.mobilescan3d.R
import com.mobilescan3d.scan.guidance.GuidanceOrbitDirection
import com.mobilescan3d.scan.guidance.GuidanceStage
import com.mobilescan3d.scan.guidance.GuidanceVertical
import com.mobilescan3d.scan.guidance.ScanGuidanceOutput
import kotlin.math.cos
import kotlin.math.sin

/**
 * Consumer scan path overlay. It deliberately draws only light-weight guidance
 * around the subject and never covers the center of the object.
 */
class ScanGuidanceOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val coverage = FloatArray(14)
    private var currentWorldSector = -1
    private var originWorldSector = -1
    private var output = ScanGuidanceOutput(
        GuidanceStage.IDLE, "", "", com.mobilescan3d.scan.guidance.GuidanceSeverity.NEUTRAL
    )

    private val primary = ContextCompat.getColor(context, R.color.scan_primary)
    private val success = ContextCompat.getColor(context, R.color.scan_success)
    private val warning = ContextCompat.getColor(context, R.color.scan_warning)
    private val muted = ContextCompat.getColor(context, R.color.scan_text_muted)

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(4.0f)
        strokeCap = Paint.Cap.ROUND
    }
    private val accentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(6.0f)
        strokeCap = Paint.Cap.ROUND
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val guideRect = RectF()
    private val arrowPath = Path()

    fun setGuidance(values: FloatArray, currentSector: Int, state: ScanGuidanceOutput) {
        for (i in coverage.indices) coverage[i] = values.getOrElse(i) { 0f }.coerceIn(0f, 1f)
        currentWorldSector = currentSector
        if (originWorldSector < 0 && currentSector in 0..11) originWorldSector = currentSector
        output = state
        contentDescription = listOf(state.stepLabel, state.primaryInstruction)
            .filter { it.isNotBlank() }.joinToString("，")
        invalidate()
    }

    fun resetGuidance() {
        coverage.fill(0f)
        currentWorldSector = -1
        originWorldSector = -1
        output = ScanGuidanceOutput(
            GuidanceStage.IDLE, "", "", com.mobilescan3d.scan.guidance.GuidanceSeverity.NEUTRAL
        )
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (output.stage == GuidanceStage.IDLE || output.stage == GuidanceStage.ACQUIRE_TARGET ||
            width <= 0 || height <= 0 || originWorldSector < 0) return

        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w * 0.47f
        val cy = h * 0.54f
        val rx = minOf(w * 0.34f, dp(155f))
        val ry = minOf(h * 0.24f, rx * 0.62f)
        guideRect.set(cx - rx, cy - ry, cx + rx, cy + ry)

        val sweep = 360f / 12f
        val gap = 7f
        for (displaySector in 0 until 12) {
            val worldSector = worldSectorForDisplay(displaySector)
            val value = coverage[worldSector]
            val start = -90f + displaySector * sweep + gap * 0.5f

            trackPaint.color = when {
                value >= 0.72f -> withAlpha(success, 105)
                value >= 0.38f -> withAlpha(warning, 105)
                else -> withAlpha(muted, 55)
            }
            canvas.drawArc(guideRect, start, sweep - gap, false, trackPaint)
        }

        val currentDisplay = displaySectorForWorld(currentWorldSector)
        if (currentDisplay in 0..11) {
            val mid = -90f + currentDisplay * sweep + sweep * 0.5f
            drawPositionDot(canvas, cx, cy, rx, ry, mid)
        }

        val recommended = output.recommendedSector
        if (recommended in 0..11) {
            val display = displaySectorForWorld(recommended)
            val start = -90f + display * sweep + gap * 0.5f
            accentPaint.color = primary
            canvas.drawArc(guideRect, start, sweep - gap, false, accentPaint)
            val mid = -90f + display * sweep + sweep * 0.5f
            drawOrbitArrow(canvas, cx, cy, rx, ry, mid, output.orbitDirection)
        }

        when (output.vertical) {
            GuidanceVertical.UP -> drawVerticalArrow(canvas, cx, cy - ry - dp(14f), true)
            GuidanceVertical.DOWN -> drawVerticalArrow(canvas, cx, cy + ry + dp(14f), false)
            GuidanceVertical.NONE -> Unit
        }

        if (output.stage == GuidanceStage.READY) {
            accentPaint.color = success
            accentPaint.strokeWidth = dp(3f)
            canvas.drawOval(guideRect, accentPaint)
            accentPaint.strokeWidth = dp(6f)
        }
    }

    private fun drawPositionDot(canvas: Canvas, cx: Float, cy: Float, rx: Float, ry: Float, angleDeg: Float) {
        val r = Math.toRadians(angleDeg.toDouble())
        val x = cx + cos(r).toFloat() * rx
        val y = cy + sin(r).toFloat() * ry
        fillPaint.color = 0xFFFFFFFF.toInt()
        canvas.drawCircle(x, y, dp(4.2f), fillPaint)
        fillPaint.color = primary
        canvas.drawCircle(x, y, dp(2.4f), fillPaint)
    }

    private fun drawOrbitArrow(
        canvas: Canvas,
        cx: Float,
        cy: Float,
        rx: Float,
        ry: Float,
        angleDeg: Float,
        direction: GuidanceOrbitDirection
    ) {
        if (direction == GuidanceOrbitDirection.NONE) return
        val r = Math.toRadians(angleDeg.toDouble())
        val x = cx + cos(r).toFloat() * rx
        val y = cy + sin(r).toFloat() * ry
        var tx = -sin(r).toFloat()
        var ty = cos(r).toFloat()
        if (direction == GuidanceOrbitDirection.COUNTER_CLOCKWISE) {
            tx = -tx; ty = -ty
        }
        val nx = -ty
        val ny = tx
        val len = dp(11f)
        val half = dp(6f)
        val tipX = x + tx * len
        val tipY = y + ty * len
        arrowPath.reset()
        arrowPath.moveTo(tipX, tipY)
        arrowPath.lineTo(x - tx * dp(3f) + nx * half, y - ty * dp(3f) + ny * half)
        arrowPath.lineTo(x - tx * dp(3f) - nx * half, y - ty * dp(3f) - ny * half)
        arrowPath.close()
        fillPaint.color = primary
        canvas.drawPath(arrowPath, fillPaint)
    }

    private fun drawVerticalArrow(canvas: Canvas, x: Float, y: Float, up: Boolean) {
        val dir = if (up) -1f else 1f
        accentPaint.color = primary
        accentPaint.strokeWidth = dp(3.5f)
        canvas.drawLine(x, y - dir * dp(5f), x, y + dir * dp(10f), accentPaint)
        arrowPath.reset()
        arrowPath.moveTo(x, y + dir * dp(14f))
        arrowPath.lineTo(x - dp(7f), y + dir * dp(5f))
        arrowPath.lineTo(x + dp(7f), y + dir * dp(5f))
        arrowPath.close()
        fillPaint.color = primary
        canvas.drawPath(arrowPath, fillPaint)
    }

    private fun displaySectorForWorld(world: Int): Int {
        if (world !in 0..11 || originWorldSector !in 0..11) return -1
        return mod12(world - originWorldSector + 6)
    }

    private fun worldSectorForDisplay(display: Int): Int =
        mod12(display - 6 + originWorldSector)

    private fun mod12(value: Int): Int = ((value % 12) + 12) % 12

    private fun withAlpha(color: Int, alpha: Int): Int =
        (color and 0x00FFFFFF) or ((alpha.coerceIn(0, 255)) shl 24)

    private fun dp(v: Float): Float = v * resources.displayMetrics.density
}
