package com.mobilescan3d.scan.consumer

import com.mobilescan3d.scan.guidance.GuidanceStage

class ScanMilestoneTracker {
    enum class Event {
        TARGET_LOCKED,
        READY_TO_ORBIT,
        COVERAGE_25,
        COVERAGE_50,
        COVERAGE_75,
        COVERAGE_90,
        READY_TO_BUILD
    }

    private var targetLockedSent = false
    private var orbitReadySent = false
    private var readySent = false
    private var coverageMask = 0

    fun reset() {
        targetLockedSent = false
        orbitReadySent = false
        readySent = false
        coverageMask = 0
    }

    fun update(
        active: Boolean,
        stage: GuidanceStage,
        targetConfidence: Float,
        coveragePercent: Int
    ): List<Event> {
        if (!active) return emptyList()
        val out = ArrayList<Event>(2)

        if (!targetLockedSent && targetConfidence >= 0.55f &&
            stage != GuidanceStage.ACQUIRE_TARGET
        ) {
            targetLockedSent = true
            out += Event.TARGET_LOCKED
        }

        if (!orbitReadySent &&
            (stage == GuidanceStage.ORBIT_HORIZONTAL ||
                stage == GuidanceStage.TOP_COVERAGE ||
                stage == GuidanceStage.LOWER_COVERAGE)
        ) {
            orbitReadySent = true
            out += Event.READY_TO_ORBIT
        }

        fun cross(bit: Int, threshold: Int, event: Event) {
            if (coveragePercent >= threshold && coverageMask and bit == 0) {
                coverageMask = coverageMask or bit
                out += event
            }
        }
        cross(1, 25, Event.COVERAGE_25)
        cross(2, 50, Event.COVERAGE_50)
        cross(4, 75, Event.COVERAGE_75)
        cross(8, 90, Event.COVERAGE_90)

        if (!readySent && stage == GuidanceStage.READY) {
            readySent = true
            out += Event.READY_TO_BUILD
        }
        return out
    }
}
