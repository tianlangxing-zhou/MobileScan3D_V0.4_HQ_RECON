package com.mobilescan3d.scan.guidance

enum class GuidanceStage {
    IDLE,
    ACQUIRE_TARGET,
    INITIALIZE_TRACKING,
    RECOVER_TRACKING,
    ADJUST_DISTANCE,
    MOTION_WARNING,
    ORBIT_HORIZONTAL,
    TOP_COVERAGE,
    LOWER_COVERAGE,
    RECOVER_MISSING,
    READY,
    PAUSED
}

enum class GuidanceSeverity { NEUTRAL, INFO, WARNING, DANGER, SUCCESS }
enum class GuidanceOrbitDirection { NONE, CLOCKWISE, COUNTER_CLOCKWISE }
enum class GuidanceVertical { NONE, UP, DOWN }

enum class GuidanceRecoveryKind { SIDE, TOP, LOWER }

data class GuidanceRecoveryTarget(
    val kind: GuidanceRecoveryKind,
    val sector: Int = -1,
    val label: String
)

data class ScanGuidanceInput(
    val active: Boolean,
    val paused: Boolean,
    val guidedMode: Boolean,
    val objectTrackingEnabled: Boolean,
    val targetConfidence: Float,
    val vinsInitialized: Boolean,
    val vinsEverInitialized: Boolean,
    val distanceMeters: Float,
    val maxDistanceMeters: Float,
    val angularSpeedDps: Float,
    val coverage: FloatArray,
    val currentSector: Int,
    val currentElevationDeg: Float,
    val captureSufficiency: Int,
    val geometryQuality: Int,
    val surfaceCoverage: Int,
    val nativeGuidance: String,
    val nowMs: Long
)

data class ScanGuidanceOutput(
    val stage: GuidanceStage,
    val stepLabel: String,
    val primaryInstruction: String,
    val severity: GuidanceSeverity,
    val recommendedSector: Int = -1,
    val orbitDirection: GuidanceOrbitDirection = GuidanceOrbitDirection.NONE,
    val vertical: GuidanceVertical = GuidanceVertical.NONE,
    val completionConfidence: Int = 0,
    val recoveryMode: Boolean = false,
    val recoveryRemaining: Int = 0,
    val recoveryTargetLabel: String = ""
)
