package com.mobilescan3d.depth

import android.util.Log

/** Thread-confined rolling timings; fixed storage; percentile sorting occurs only once per 30 frames. */
internal class DepthPerformance(private val tag: String, private vararg val stages: String) {
    private val values = Array(stages.size) { FloatArray(120) }
    private var head = 0
    private var count = 0
    private var frames = 0L

    fun record(vararg milliseconds: Float) {
        if (milliseconds.size != stages.size) return
        for (i in values.indices) values[i][head] = milliseconds[i]
        head = (head + 1) % 120
        count = (count + 1).coerceAtMost(120)
        frames++
        if (frames % 30L != 0L) return
        val report = values.indices.joinToString(" | ") { i ->
            val sorted = values[i].copyOf(count).apply { sort() }
            val p50 = sorted[((count - 1) * .50f + .5f).toInt()]
            val p95 = sorted[((count - 1) * .95f + .5f).toInt()]
            "${stages[i]} p50=${p50}ms p95=${p95}ms"
        }
        Log.i(tag, "frames=$frames $report")
    }

    fun reset() { head = 0; count = 0; frames = 0 }
}
