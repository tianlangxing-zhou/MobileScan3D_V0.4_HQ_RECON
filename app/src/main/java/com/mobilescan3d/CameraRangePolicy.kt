package com.mobilescan3d

/** Camera-thread owned. A range is committed only after the HAL accepts the session. */
class CameraRangePolicy {
    enum class Range(val label: String) { NEAR("近"), MID("中"), FAR("远") }
    enum class Mode(val label: String) { AUTO("自动"), NEAR("近"), MID("中"), FAR("远") }
    var mode = Mode.AUTO
        private set
    var range = Range.MID
        private set
    private var pending: Range? = null
    private var pendingSince = 0L
    private var pendingSamples = 0
    private var lastObservation = -1L
    private var lastChange = -10_000L
    private var retryAfter = 0L
    var meters = Float.NaN
        private set

    /**
     * vc161：把「候选档位是否还在等采样」暴露给日志。
     * update() 返回 null 有两种完全不同的含义——「已是目标档位/无有效距离」（稳态）
     * 与「候选未满 3 次采样或 900ms」（等待中）。只看返回值无法区分，实拍时会把
     * 稳态误读成「策略卡住」。2026-10-01 12:16 自检即出现该歧义。
     */
    val pendingRange: Range? get() = pending
    val pendingSampleCount: Int get() = pendingSamples

    fun setMode(value: Mode) { mode = value; pending = null; retryAfter = 0L }
    fun reset() { pending = null; lastObservation = -1L; meters = Float.NaN }
    fun committed(value: Range, nowMs: Long) {
        range = value; lastChange = nowMs; pending = null
    }
    fun failed(nowMs: Long) { retryAfter = nowMs + 10_000L; pending = null }

    fun update(distanceMeters: Float, valid: Boolean, nowMs: Long): Range? {
        val fresh = valid && distanceMeters.isFinite() && distanceMeters in 0.08f..20f
        if (lastObservation >= 0 && (nowMs < lastObservation || nowMs - lastObservation > 2_000L)) {
            pending = null
        }
        lastObservation = nowMs
        meters = if (fresh) distanceMeters else Float.NaN
        val desired = when (mode) {
            Mode.NEAR -> Range.NEAR
            Mode.MID -> Range.MID
            Mode.FAR -> Range.FAR
            Mode.AUTO -> {
                if (!fresh) { pending = null; return null }
                // 0.45/1.50m nominal boundaries, with 0.10/0.25m hysteresis.
                when (range) {
                    Range.NEAR -> if (distanceMeters > 1.75f) Range.FAR else if (distanceMeters > .55f) Range.MID else Range.NEAR
                    Range.MID -> if (distanceMeters < .35f) Range.NEAR else if (distanceMeters > 1.75f) Range.FAR else Range.MID
                    Range.FAR -> if (distanceMeters < .35f) Range.NEAR else if (distanceMeters < 1.25f) Range.MID else Range.FAR
                }
            }
        }
        if (desired == range) { pending = null; return null }
        if (desired != pending) { pending = desired; pendingSince = nowMs; pendingSamples = 0 }
        pendingSamples++
        if (nowMs < retryAfter || (mode == Mode.AUTO && nowMs - lastChange < 3_000L)) return null
        if (mode == Mode.AUTO && (nowMs - pendingSince < 900L || pendingSamples < 3)) return null
        return desired
    }
}
