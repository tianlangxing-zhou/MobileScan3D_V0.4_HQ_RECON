package com.mobilescan3d.scan.guidance

import kotlin.math.abs

/**
 * Consumer-facing scan guidance. This layer only interprets existing tracking,
 * pose/coverage and quality signals; it never changes reconstruction math.
 */
class ScanGuidanceController {
    private var preferredStep = 1
    private var lastObservedSector = -1
    private var lastSectorChangeMs = 0L
    private var lastRecommendedSector = -1
    private var lastRecommendationMs = 0L
    private val recoveryTargets = ArrayList<GuidanceRecoveryTarget>()
    private var recoverySessionArmed = false
    private var recoverySessionComplete = false

    fun setRecoveryTargets(targets: List<GuidanceRecoveryTarget>) {
        recoveryTargets.clear()
        recoveryTargets.addAll(targets.take(4))
        recoverySessionArmed = recoveryTargets.isNotEmpty()
        recoverySessionComplete = false
        lastRecommendedSector = -1
        lastRecommendationMs = 0L
    }

    fun clearRecoveryTargets() {
        recoveryTargets.clear()
        recoverySessionArmed = false
        recoverySessionComplete = false
    }

    fun reset() {
        preferredStep = 1
        lastObservedSector = -1
        lastSectorChangeMs = 0L
        lastRecommendedSector = -1
        lastRecommendationMs = 0L
        recoveryTargets.clear()
        recoverySessionArmed = false
        recoverySessionComplete = false
    }

    fun evaluate(input: ScanGuidanceInput): ScanGuidanceOutput {
        if (!input.guidedMode || !input.active) {
            return ScanGuidanceOutput(
                GuidanceStage.IDLE, "", "", GuidanceSeverity.NEUTRAL
            )
        }

        observeOrbitDirection(input.currentSector, input.nowMs)
        val coverage = normalizedCoverage(input.coverage)
        val sideMissing = (0 until 12).count { coverage[it] < SIDE_GOOD }
        val topMissing = coverage[12] < TOP_GOOD
        val lowerMissing = coverage[13] < LOWER_GOOD
        val stepLabel = progressStepLabel(
            input.objectTrackingEnabled,
            input.targetConfidence,
            sideMissing,
            topMissing,
            lowerMissing,
            input.captureSufficiency
        )
        val candidate = chooseRecommendedSector(coverage, input.currentSector, input.nowMs)
        val orbitDirection = if (candidate < 0 || input.currentSector < 0) {
            GuidanceOrbitDirection.NONE
        } else if (signedSectorDelta(input.currentSector, candidate) >= 0) {
            GuidanceOrbitDirection.CLOCKWISE
        } else {
            GuidanceOrbitDirection.COUNTER_CLOCKWISE
        }
        val completion = completionConfidence(input, coverage)

        if (input.paused) {
            return ScanGuidanceOutput(
                GuidanceStage.PAUSED,
                stepLabel,
                "扫描已暂停 · 调整位置后可继续",
                GuidanceSeverity.INFO,
                candidate,
                orbitDirection,
                GuidanceVertical.NONE,
                completion
            )
        }

        if (!input.objectTrackingEnabled) {
            return ScanGuidanceOutput(
                GuidanceStage.ACQUIRE_TARGET,
                "1/4 锁定物体",
                "开启右侧“追踪”，选择要扫描的物体",
                GuidanceSeverity.WARNING,
                completionConfidence = completion
            )
        }
        if (input.targetConfidence < 0.35f) {
            return ScanGuidanceOutput(
                GuidanceStage.ACQUIRE_TARGET,
                "1/4 锁定物体",
                "点击物体或拖框选择目标，并保持完整入框",
                GuidanceSeverity.WARNING,
                completionConfidence = completion
            )
        }

        if (!input.vinsInitialized) {
            val text = if (input.vinsEverInitialized) {
                "定位暂时丢失 · 回到已扫描区域，保持目标在画面内"
            } else {
                "定位初始化中 · 保持镜头朝向物体，缓慢平移手机"
            }
            return ScanGuidanceOutput(
                if (input.vinsEverInitialized) GuidanceStage.RECOVER_TRACKING else GuidanceStage.INITIALIZE_TRACKING,
                stepLabel,
                text,
                if (input.vinsEverInitialized) GuidanceSeverity.DANGER else GuidanceSeverity.INFO,
                candidate,
                orbitDirection,
                GuidanceVertical.NONE,
                completion
            )
        }

        val d = input.distanceMeters
        if (d.isFinite() && d > 0f) {
            if (d < MIN_DISTANCE_METERS) {
                return ScanGuidanceOutput(
                    GuidanceStage.ADJUST_DISTANCE, stepLabel,
                    "稍微后退，让物体完整留在扫描框内",
                    GuidanceSeverity.WARNING, candidate, orbitDirection,
                    GuidanceVertical.NONE, completion
                )
            }
            if (d > input.maxDistanceMeters.coerceAtLeast(MIN_DISTANCE_METERS + 0.05f)) {
                return ScanGuidanceOutput(
                    GuidanceStage.ADJUST_DISTANCE, stepLabel,
                    "靠近一点，保持目标细节清晰",
                    GuidanceSeverity.WARNING, candidate, orbitDirection,
                    GuidanceVertical.NONE, completion
                )
            }
        }

        val native = input.nativeGuidance
        if (input.angularSpeedDps > FAST_ROTATION_DPS ||
            native.contains("放慢") || native.contains("模糊") || native.contains("曝光")) {
            return ScanGuidanceOutput(
                GuidanceStage.MOTION_WARNING, stepLabel,
                when {
                    native.contains("曝光") -> native
                    native.contains("模糊") -> "移动慢一点，减少画面模糊"
                    else -> "移动慢一点，保持手机稳定"
                },
                GuidanceSeverity.WARNING, candidate, orbitDirection,
                GuidanceVertical.NONE, completion
            )
        }

        if (native.contains("纹理较少") ||
            native.contains("表面可重建性") ||
            native.contains("光线不足") ||
            native.contains("过曝明显")) {
            return ScanGuidanceOutput(
                GuidanceStage.MOTION_WARNING, stepLabel, native,
                GuidanceSeverity.WARNING, candidate, orbitDirection,
                GuidanceVertical.NONE, completion
            )
        }

        // High angular motion while the camera position remains in the same orbit
        // sector is a practical sign of "turning the phone" instead of walking around
        // the object. The user-facing copy explicitly corrects that common mistake.
        if (input.currentSector >= 0 &&
            input.angularSpeedDps > PURE_ROTATION_DPS &&
            lastSectorChangeMs > 0L && input.nowMs - lastSectorChangeMs > PURE_ROTATION_HOLD_MS &&
            sideMissing >= 5) {
            return ScanGuidanceOutput(
                GuidanceStage.MOTION_WARNING, stepLabel,
                "请围绕物体移动，不要只在原地转动手机",
                GuidanceSeverity.WARNING, candidate, orbitDirection,
                GuidanceVertical.NONE, completion
            )
        }

        if (recoverySessionComplete) {
            return ScanGuidanceOutput(
                GuidanceStage.READY,
                "补扫完成",
                "本轮缺口已经补齐，正在返回模型检查",
                GuidanceSeverity.SUCCESS,
                completionConfidence = completion,
                recoveryMode = true,
                recoveryRemaining = 0
            )
        }

        if (recoverySessionArmed) {
            while (recoveryTargets.isNotEmpty() && isRecoveryResolved(recoveryTargets.first(), coverage)) {
                recoveryTargets.removeAt(0)
            }
            if (recoveryTargets.isEmpty()) {
                recoverySessionArmed = false
                recoverySessionComplete = true
                return ScanGuidanceOutput(
                    GuidanceStage.READY,
                    "补扫完成",
                    "本轮缺口已经补齐，正在返回模型检查",
                    GuidanceSeverity.SUCCESS,
                    completionConfidence = completion,
                    recoveryMode = true,
                    recoveryRemaining = 0
                )
            }

            val target = recoveryTargets.first()
            val recommended = if (target.kind == GuidanceRecoveryKind.SIDE) target.sector else -1
            val recoveryDirection = if (recommended < 0 || input.currentSector < 0) {
                GuidanceOrbitDirection.NONE
            } else if (signedSectorDelta(input.currentSector, recommended) >= 0) {
                GuidanceOrbitDirection.CLOCKWISE
            } else {
                GuidanceOrbitDirection.COUNTER_CLOCKWISE
            }
            val instruction = when (target.kind) {
                GuidanceRecoveryKind.SIDE ->
                    "沿箭头方向补扫${target.label}，保持镜头朝向物体"
                GuidanceRecoveryKind.TOP ->
                    "略微抬高手机，补扫检查页标出的顶部弱区域"
                GuidanceRecoveryKind.LOWER ->
                    "降低手机补扫下半部，无需扫描与桌面接触的底面"
            }
            return ScanGuidanceOutput(
                GuidanceStage.RECOVER_MISSING,
                "定向补扫 ${recoveryTargets.size}",
                instruction,
                GuidanceSeverity.INFO,
                recommendedSector = recommended,
                orbitDirection = recoveryDirection,
                vertical = when (target.kind) {
                    GuidanceRecoveryKind.TOP -> GuidanceVertical.UP
                    GuidanceRecoveryKind.LOWER -> GuidanceVertical.DOWN
                    else -> GuidanceVertical.NONE
                },
                completionConfidence = completion,
                recoveryMode = true,
                recoveryRemaining = recoveryTargets.size,
                recoveryTargetLabel = target.label
            )
        }

        if (isReady(input, coverage, sideMissing, topMissing)) {
            return ScanGuidanceOutput(
                GuidanceStage.READY,
                "扫描完整",
                "扫描信息已经足够，可以完成并检查模型",
                GuidanceSeverity.SUCCESS,
                completionConfidence = completion
            )
        }

        if (sideMissing <= 3 && topMissing) {
            return ScanGuidanceOutput(
                GuidanceStage.TOP_COVERAGE,
                "3/4 上下补扫",
                "略微抬高手机，补扫物体顶部",
                GuidanceSeverity.INFO,
                vertical = GuidanceVertical.UP,
                completionConfidence = completion
            )
        }

        // Bottom is intentionally optional for completion: tabletop objects often have
        // an unobservable contact surface. We only ask for a lower viewing angle when
        // the rest of the scan is already strong and capture sufficiency is still low.
        if (sideMissing <= 2 && !topMissing && lowerMissing && input.captureSufficiency < 82) {
            return ScanGuidanceOutput(
                GuidanceStage.LOWER_COVERAGE,
                "3/4 上下补扫",
                "降低手机，补扫物体下半部（无需扫描底面）",
                GuidanceSeverity.INFO,
                vertical = GuidanceVertical.DOWN,
                completionConfidence = completion
            )
        }

        if (sideMissing <= 4) {
            return ScanGuidanceOutput(
                GuidanceStage.RECOVER_MISSING,
                "4/4 补齐缺口",
                if (candidate >= 0) "沿箭头方向补扫缺失侧面" else "继续补扫未覆盖侧面",
                GuidanceSeverity.INFO,
                candidate,
                orbitDirection,
                GuidanceVertical.NONE,
                completion
            )
        }

        return ScanGuidanceOutput(
            GuidanceStage.ORBIT_HORIZONTAL,
            "2/4 环绕扫描",
            if (candidate >= 0) "保持镜头朝向物体，沿箭头方向缓慢环绕" else "保持镜头朝向物体，缓慢环绕一圈",
            GuidanceSeverity.INFO,
            candidate,
            orbitDirection,
            GuidanceVertical.NONE,
            completion
        )
    }

    private fun observeOrbitDirection(currentSector: Int, nowMs: Long) {
        if (currentSector !in 0..11) return
        if (lastObservedSector < 0) {
            lastObservedSector = currentSector
            lastSectorChangeMs = nowMs
            return
        }
        if (currentSector == lastObservedSector) return
        val delta = signedSectorDelta(lastObservedSector, currentSector)
        if (abs(delta) in 1..3) preferredStep = if (delta > 0) 1 else -1
        lastObservedSector = currentSector
        lastSectorChangeMs = nowMs
    }

    private fun chooseRecommendedSector(coverage: FloatArray, currentSector: Int, nowMs: Long): Int {
        fun missing(i: Int) = coverage[i] < SIDE_GOOD
        if ((0 until 12).none(::missing)) {
            lastRecommendedSector = -1
            return -1
        }

        if (lastRecommendedSector in 0..11 &&
            nowMs - lastRecommendationMs < RECOMMENDATION_HOLD_MS &&
            missing(lastRecommendedSector)) {
            return lastRecommendedSector
        }

        val candidate = if (currentSector in 0..11) {
            var found = -1
            for (distance in 1..6) {
                val preferred = mod12(currentSector + preferredStep * distance)
                if (missing(preferred)) { found = preferred; break }
                val opposite = mod12(currentSector - preferredStep * distance)
                if (missing(opposite)) { found = opposite; break }
            }
            if (found >= 0) found else if (missing(currentSector)) currentSector else -1
        } else {
            (0 until 12).minByOrNull { coverage[it] } ?: -1
        }

        if (candidate != lastRecommendedSector) {
            lastRecommendedSector = candidate
            lastRecommendationMs = nowMs
        }
        return candidate
    }

    private fun progressStepLabel(
        trackingEnabled: Boolean,
        targetConfidence: Float,
        sideMissing: Int,
        topMissing: Boolean,
        lowerMissing: Boolean,
        capture: Int
    ): String = when {
        !trackingEnabled || targetConfidence < 0.35f -> "1/4 锁定物体"
        sideMissing > 4 -> "2/4 环绕扫描"
        topMissing || (lowerMissing && capture < 82) -> "3/4 上下补扫"
        sideMissing > 0 -> "4/4 补齐缺口"
        else -> "扫描完整"
    }

    private fun completionConfidence(input: ScanGuidanceInput, coverage: FloatArray): Int {
        val side = coverage.sliceArray(0 until 12).average().toFloat().coerceIn(0f, 1f)
        val top = coverage[12].coerceIn(0f, 1f)
        // Bottom/contact-surface coverage carries a deliberately small weight.
        val lower = coverage[13].coerceIn(0f, 1f)
        val capture = (input.captureSufficiency / 100f).coerceIn(0f, 1f)
        val geometry = if (input.geometryQuality > 0) (input.geometryQuality / 100f).coerceIn(0f, 1f) else capture
        val surface = if (input.surfaceCoverage > 0) (input.surfaceCoverage / 100f).coerceIn(0f, 1f) else geometry
        val score = side * 0.46f + top * 0.14f + lower * 0.05f + capture * 0.20f + geometry * 0.10f + surface * 0.05f
        return (score * 100f).toInt().coerceIn(0, 100)
    }

    private fun isReady(
        input: ScanGuidanceInput,
        coverage: FloatArray,
        sideMissing: Int,
        topMissing: Boolean
    ): Boolean {
        val sideAverage = coverage.sliceArray(0 until 12).average().toFloat()
        val geometryOkay = input.geometryQuality <= 0 || input.geometryQuality >= 48
        return sideMissing <= 1 && !topMissing && sideAverage >= 0.58f &&
            input.captureSufficiency >= 78 && geometryOkay
    }

    private fun isRecoveryResolved(
        target: GuidanceRecoveryTarget,
        coverage: FloatArray
    ): Boolean = when (target.kind) {
        GuidanceRecoveryKind.SIDE ->
            target.sector in 0..11 && coverage[target.sector] >= ScanRecoveryPlanner.SIDE_TARGET
        GuidanceRecoveryKind.TOP ->
            coverage[12] >= ScanRecoveryPlanner.TOP_TARGET
        GuidanceRecoveryKind.LOWER ->
            coverage[13] >= ScanRecoveryPlanner.LOWER_TARGET
    }

    private fun normalizedCoverage(source: FloatArray): FloatArray =
        FloatArray(14) { source.getOrElse(it) { 0f }.coerceIn(0f, 1f) }

    private fun signedSectorDelta(from: Int, to: Int): Int {
        var d = mod12(to) - mod12(from)
        if (d > 6) d -= 12
        if (d < -6) d += 12
        return d
    }

    private fun mod12(value: Int): Int = ((value % 12) + 12) % 12

    companion object {
        private const val SIDE_GOOD = 0.38f
        private const val TOP_GOOD = 0.34f
        private const val LOWER_GOOD = 0.30f
        private const val MIN_DISTANCE_METERS = 0.18f
        private const val FAST_ROTATION_DPS = 72f
        private const val PURE_ROTATION_DPS = 28f
        private const val PURE_ROTATION_HOLD_MS = 1800L
        private const val RECOMMENDATION_HOLD_MS = 1200L
    }
}
