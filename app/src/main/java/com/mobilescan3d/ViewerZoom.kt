package com.mobilescan3d

/** Pure orbit-distance math, shared by the renderer and host regressions. */
object ViewerZoom {
    fun distance(current: Float, radius: Float, scaleFactor: Float): Float {
        if (!current.isFinite() || !radius.isFinite() || radius <= 0f ||
            !scaleFactor.isFinite() || scaleFactor <= 0f) return current
        // Keep the eye outside the bounding sphere and the 5cm shader clip plane.
        val near = (radius + .06f).coerceAtLeast(.10f)
        val far = (radius * 40f).coerceAtLeast(3f).coerceAtLeast(near)
        return (current / scaleFactor.coerceIn(.5f, 2f)).coerceIn(near, far)
    }
}
