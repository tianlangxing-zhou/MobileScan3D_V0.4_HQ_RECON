package com.mobilescan3d.scan.camera

class CameraPreviewRecoveryPolicy {
    enum class Action {
        RETRY_CURRENT,
        ENABLE_COMPAT_AND_RETRY,
        ASK_USER
    }

    private var consecutiveTimeouts = 0

    fun onPreviewTimeout(compatModeAlreadyEnabled: Boolean): Action {
        consecutiveTimeouts++
        return when {
            consecutiveTimeouts == 1 -> Action.RETRY_CURRENT
            !compatModeAlreadyEnabled -> Action.ENABLE_COMPAT_AND_RETRY
            else -> Action.ASK_USER
        }
    }

    fun onFrameReceived() {
        consecutiveTimeouts = 0
    }

    fun reset() {
        consecutiveTimeouts = 0
    }
}
