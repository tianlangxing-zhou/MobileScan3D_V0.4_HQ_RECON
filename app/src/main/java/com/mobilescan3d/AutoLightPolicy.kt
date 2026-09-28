package com.mobilescan3d

/** Camera-thread policy. A scan latches fill light to avoid torch/AE feedback flicker. */
class AutoLightPolicy {
    private var darkSinceMs = -1L
    private var lastSampleMs = -1L
    var enabled = false
        private set

    fun reset() { darkSinceMs = -1L; lastSampleMs = -1L; enabled = false }

    fun update(nowMs: Long, p50: Float, p95: Float, iso: Int, exposureNs: Long): Boolean {
        if (enabled) return true
        if (!p50.isFinite() || !p95.isFinite() || p50 < 0 || p95 < p50) return false
        if (lastSampleMs >= 0 && (nowMs < lastSampleMs || nowMs - lastSampleMs > 1500)) darkSinceMs = -1
        lastSampleMs = nowMs
        val dark = (p50 < 55f && p95 < 150f) || (iso >= 800 && exposureNs >= 20_000_000L)
        if (!dark) { darkSinceMs = -1L; return false }
        if (darkSinceMs < 0) darkSinceMs = nowMs
        if (nowMs - darkSinceMs >= 900L) enabled = true
        return enabled
    }
}
