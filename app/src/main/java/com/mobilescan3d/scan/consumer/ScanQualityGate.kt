package com.mobilescan3d.scan.consumer

object ScanQualityGate {
    enum class Recommendation { GENERATE, RESCAN, CONTINUE }

    data class Result(
        val score: Int,
        val recommendation: Recommendation,
        val reasons: List<String>
    ) {
        val shouldWarnBeforeGenerate: Boolean
            get() = recommendation != Recommendation.GENERATE
    }

    fun evaluate(
        capture: Int,
        viewpoint: Int,
        surface: Int,
        geometry: Int,
        texture: Int
    ): Result {
        val c = capture.coerceIn(0, 100)
        val v = viewpoint.coerceIn(0, 100)
        val s = surface.coerceIn(0, 100)
        val g = geometry.coerceIn(0, 100)
        val t = texture.coerceIn(0, 100)
        val score = (
            c * 0.20f +
                v * 0.15f +
                s * 0.35f +
                g * 0.20f +
                t * 0.10f
            ).toInt().coerceIn(0, 100)

        val reasons = ArrayList<String>(4)
        if (v < 65) reasons += "环绕视角仍不足（$v%）"
        if (s < 65) reasons += "真实表面观测仍偏弱（$s%）"
        if (g < 58) reasons += "几何稳定度偏低（$g%）"
        if (t < 48) reasons += "纹理/画面质量偏低（$t%）"
        if (c < 58) reasons += "有效采集量仍偏少（$c%）"

        val recommendation = when {
            score >= 80 && v >= 70 && s >= 68 -> Recommendation.GENERATE
            score >= 58 -> Recommendation.RESCAN
            else -> Recommendation.CONTINUE
        }
        return Result(score, recommendation, reasons.take(3))
    }
}
