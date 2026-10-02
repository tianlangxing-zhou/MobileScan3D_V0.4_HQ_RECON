package com.mobilescan3d.scan.guidance

/**
 * Creates a small, actionable recovery plan from the existing viewpoint coverage.
 *
 * Review mesh heat-map colors still come from real TSDF observation weights.
 * This planner only answers "from which view should the user re-observe next?"
 */
object ScanRecoveryPlanner {
    data class Plan(
        val targets: List<GuidanceRecoveryTarget>,
        val summary: String
    ) {
        val isEmpty: Boolean get() = targets.isEmpty()
    }

    fun build(
        coverage: FloatArray,
        currentSector: Int,
        captureSufficiency: Int,
        surfaceCoverage: Int,
        geometryQuality: Int,
        maxTargets: Int = 3
    ): Plan {
        val c = FloatArray(14) { coverage.getOrElse(it) { 0f }.coerceIn(0f, 1f) }
        val candidates = ArrayList<ScoredTarget>()

        // Choose the weakest side sectors, but suppress immediate neighbours so
        // one broad missing region does not consume every recommendation slot.
        val sideOrder = (0 until 12).sortedBy { c[it] }
        val chosenSides = ArrayList<Int>()
        for (sector in sideOrder) {
            if (c[sector] >= SIDE_PLAN_THRESHOLD) break
            if (chosenSides.any { circularDistance(it, sector) <= 1 }) continue
            val deficit = (SIDE_TARGET - c[sector]).coerceAtLeast(0f)
            candidates += ScoredTarget(
                score = deficit * 1.20f + qualityBoost(surfaceCoverage, geometryQuality),
                target = GuidanceRecoveryTarget(
                    GuidanceRecoveryKind.SIDE,
                    sector,
                    relativeSectorLabel(currentSector, sector)
                )
            )
            chosenSides += sector
        }

        if (c[12] < TOP_PLAN_THRESHOLD) {
            candidates += ScoredTarget(
                score = (TOP_TARGET - c[12]).coerceAtLeast(0f) * 1.35f + 0.10f,
                target = GuidanceRecoveryTarget(GuidanceRecoveryKind.TOP, 12, "顶部")
            )
        }

        // Lower views are optional and are only recommended when the rest of the
        // scan still lacks capture sufficiency. The physical contact bottom is not
        // treated as a mandatory missing surface.
        if (c[13] < LOWER_PLAN_THRESHOLD && captureSufficiency < 82) {
            candidates += ScoredTarget(
                score = (LOWER_TARGET - c[13]).coerceAtLeast(0f) * 0.78f,
                target = GuidanceRecoveryTarget(GuidanceRecoveryKind.LOWER, 13, "下半部")
            )
        }

        val result = candidates.sortedByDescending { it.score }
            .map { it.target }
            .take(maxTargets.coerceIn(1, 4))

        val summary = if (result.isEmpty()) {
            "当前没有明确的定向补扫目标"
        } else {
            "优先补扫：" + result.joinToString(" · ") { it.label }
        }
        return Plan(result, summary)
    }

    private fun qualityBoost(surfaceCoverage: Int, geometryQuality: Int): Float {
        val surfaceDeficit =
            if (surfaceCoverage in 1..64) (65 - surfaceCoverage) / 100f else 0f
        val geometryDeficit =
            if (geometryQuality in 1..54) (55 - geometryQuality) / 120f else 0f
        return (surfaceDeficit + geometryDeficit).coerceAtMost(0.35f)
    }

    private fun relativeSectorLabel(current: Int, target: Int): String {
        if (current !in 0..11) return "侧面 ${target + 1}/12"
        var d = target - current
        if (d > 6) d -= 12
        if (d < -6) d += 12
        return when {
            d == 0 -> "当前位置侧面"
            d == 6 || d == -6 -> "背面"
            d in 1..2 -> "右侧"
            d in 3..5 -> "右后侧"
            d in -2..-1 -> "左侧"
            else -> "左后侧"
        }
    }

    private fun circularDistance(a: Int, b: Int): Int {
        val d = kotlin.math.abs(a - b)
        return minOf(d, 12 - d)
    }

    private data class ScoredTarget(
        val score: Float,
        val target: GuidanceRecoveryTarget
    )

    private const val SIDE_PLAN_THRESHOLD = 0.40f
    private const val TOP_PLAN_THRESHOLD = 0.36f
    private const val LOWER_PLAN_THRESHOLD = 0.28f

    internal const val SIDE_TARGET = 0.52f
    internal const val TOP_TARGET = 0.46f
    internal const val LOWER_TARGET = 0.38f
}
