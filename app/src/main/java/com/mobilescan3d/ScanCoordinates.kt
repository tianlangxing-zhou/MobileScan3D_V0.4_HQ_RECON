package com.mobilescan3d

import kotlin.math.cos
import kotlin.math.sin

/** Coordinate boundaries only. Reconstruction and saved AR data stay in VINS +Z-up world. */
object ScanCoordinates {
    /**
     * SurfaceTexture consumes GL UV (bottom-left origin), while TextureView/touch
     * coordinates have a top-left origin. Compose M_ST * flipY on the INPUT side.
     * Its output already addresses top-left camera buffer memory: do not flip it again.
     * Preserve producer rotation, mirroring and crop in M_ST.
     */
    fun surfaceTextureToCamera(m: FloatArray): FloatArray {
        require(m.size >= 16)
        return floatArrayOf(
            m[0], -m[4], m[12] + m[4],
            m[1], -m[5], m[13] + m[5]
        )
    }

    /** Row-major Rwc + translation; camera columns are right/down/forward (det R = +1). */
    fun viewerPose(yaw: Float, pitch: Float, distance: Float,
                   cx: Float, cy: Float, cz: Float, out: FloatArray) {
        require(out.size >= 12)
        val p = pitch.coerceIn(-1.4f, 1.4f)
        val cp = cos(p); val sp = sin(p)
        val cyaw = cos(yaw); val syaw = sin(yaw)
        // Orbit around world +Z. At yaw=0, eye is on -Y looking towards +Y.
        val fx = -cp * syaw; val fy = cp * cyaw; val fz = -sp
        // right = normalize(forward x worldUp), down = forward x right.
        val rx = cyaw; val ry = syaw
        val dx = -fz * ry; val dy = fz * rx; val dz = fx * ry - fy * rx
        out[0] = rx; out[1] = dx; out[2] = fx
        out[3] = ry; out[4] = dy; out[5] = fy
        out[6] = 0f; out[7] = dz; out[8] = fz
        out[9] = cx - fx * distance
        out[10] = cy - fy * distance
        out[11] = cz - fz * distance
    }

    /** Column-major model-to-world: anchor + Rz(yaw) * scale * (point - center). */
    fun placementMatrix(yaw: Float, scale: Float, center: FloatArray,
                        anchor: FloatArray): FloatArray {
        require(center.size >= 3 && anchor.size >= 3)
        val c = cos(yaw); val s = sin(yaw)
        val rx = c * center[0] - s * center[1]
        val ry = s * center[0] + c * center[1]
        return floatArrayOf(
            scale * c, scale * s, 0f, 0f,
            -scale * s, scale * c, 0f, 0f,
            0f, 0f, scale, 0f,
            anchor[0] - scale * rx,
            anchor[1] - scale * ry,
            anchor[2] - scale * center[2], 1f
        )
    }

    fun placementAnchor(pose: FloatArray, distance: Float, drop: Float, out: FloatArray) {
        require(pose.size >= 12 && out.size >= 3)
        out[0] = pose[9] + pose[2] * distance
        out[1] = pose[10] + pose[5] * distance
        out[2] = pose[11] + pose[8] * distance - drop
    }
}
