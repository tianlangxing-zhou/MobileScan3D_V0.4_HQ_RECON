package com.mobilescan3d.scan.adaptive

class AdaptiveScanAdvisor {
    data class Input(
        val active: Boolean,
        val targetSelected: Boolean,
        val targetConfidence: Float,
        val sharpness: Float,
        val exposure: Float,
        val featureCount: Float,
        val stableRatio: Float,
        val nowMs: Long
    )
    data class Advice(val text: String, val severity: Int)

    private var candidate = ""
    private var candidateSince = 0L
    private var published = ""

    fun reset() { candidate = ""; candidateSince = 0L; published = "" }

    fun evaluate(i: Input): Advice {
        if (!i.active) { reset(); return Advice("", 0) }
        val next = when {
            i.exposure < 0.20f -> "光线不足，建议增加柔和照明后继续"
            i.exposure > 0.94f -> "局部过曝明显，建议避开直射强光"
            i.sharpness < 0.20f -> "画面清晰度偏低，请稳定手机并放慢移动"
            i.targetSelected && i.sharpness > 0.42f && i.exposure in 0.30f..0.88f &&
                i.featureCount < 42f ->
                "物体纹理较少，换对比更明显的背景或增加侧向光"
            i.targetSelected && i.featureCount >= 70f && i.stableRatio < 0.16f &&
                i.targetConfidence < 0.48f ->
                "表面可重建性偏低，可能受反光/透明/遮挡影响；换角度或柔和侧光"
            else -> ""
        }
        if (next.isBlank()) { reset(); return Advice("", 0) }
        if (next != candidate) { candidate = next; candidateSince = i.nowMs; return Advice(published, if (published.isBlank()) 0 else 2) }
        if (published != next && i.nowMs - candidateSince >= 1_800L) published = next
        return Advice(published, if (published.isBlank()) 0 else 2)
    }
}
