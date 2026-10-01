package com.mobilescan3d.persistence

import android.content.Context
import com.mobilescan3d.NativeBridge
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * Round 6 unfinished-scan recovery.
 *
 * The checkpoint stores the exact sparse scene/target TSDF fields. VINS estimator
 * state is intentionally NOT serialized. After an app/process interruption the
 * checkpoint can therefore recover the geometry for inspection/export, but it is
 * not silently fused with a freshly initialized VINS world.
 */
object ScanRecoveryManager {
    private const val FORMAT_VERSION = 1
    private const val DIR = "unfinished_scan"
    private const val MANIFEST = "manifest.json"
    private const val SCENE = "scene.tsdf"
    private const val TARGET = "target.tsdf"

    data class Info(
        val dir: File,
        val scene: File,
        val target: File,
        val sessionId: String,
        val createdAtMs: Long,
        val voxelProfile: Int,
        val hadTarget: Boolean
    )

    private fun dir(context: Context): File =
        File(context.filesDir, DIR)

    @Synchronized
    fun save(
        context: Context,
        sessionId: String,
        voxelProfile: Int,
        hadTarget: Boolean
    ): Boolean {
        if (sessionId.isBlank()) return false
        val finalDir = dir(context)
        val tmp = File(context.filesDir, ".$DIR.tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()

        val scene = File(tmp, SCENE)
        val target = File(tmp, TARGET)

        return try {
            if (!NativeBridge.nativeSaveTsdfCheckpoint(
                    scene.absolutePath,
                    target.absolutePath
                )) {
                tmp.deleteRecursively()
                return false
            }
            if (scene.length() <= 0L || target.length() <= 0L) {
                tmp.deleteRecursively()
                return false
            }

            val json = JSONObject()
                .put("formatVersion", FORMAT_VERSION)
                .put("sessionId", sessionId)
                .put("createdAtMs", System.currentTimeMillis())
                .put("voxelProfile", voxelProfile)
                .put("hadTarget", hadTarget)
                .put("sceneSha256", sha256(scene))
                .put("targetSha256", sha256(target))

            File(tmp, MANIFEST).writeText(json.toString(2), Charsets.UTF_8)

            finalDir.deleteRecursively()
            if (!tmp.renameTo(finalDir)) {
                tmp.deleteRecursively()
                return false
            }
            latest(context) != null
        } catch (_: Throwable) {
            tmp.deleteRecursively()
            false
        }
    }

    fun latest(context: Context): Info? {
        val d = dir(context)
        val manifest = File(d, MANIFEST)
        if (!manifest.isFile) return null
        return try {
            val json = JSONObject(manifest.readText(Charsets.UTF_8))
            if (json.optInt("formatVersion", -1) != FORMAT_VERSION) return null
            val scene = File(d, SCENE)
            val target = File(d, TARGET)
            if (!scene.isFile || !target.isFile) return null
            if (sha256(scene) != json.optString("sceneSha256", "")) return null
            if (sha256(target) != json.optString("targetSha256", "")) return null
            Info(
                dir = d,
                scene = scene,
                target = target,
                sessionId = json.optString("sessionId", "recovered"),
                createdAtMs = json.optLong("createdAtMs", d.lastModified()),
                voxelProfile = json.optInt("voxelProfile", 0).coerceIn(0, 2),
                hadTarget = json.optBoolean("hadTarget", false)
            )
        } catch (_: Throwable) {
            null
        }
    }

    fun loadIntoNative(info: Info): Boolean = try {
        NativeBridge.nativeLoadTsdfCheckpoint(
            info.scene.absolutePath,
            info.target.absolutePath
        )
    } catch (_: Throwable) {
        false
    }

    fun clear(context: Context) {
        dir(context).deleteRecursively()
        File(context.filesDir, ".$DIR.tmp").deleteRecursively()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n <= 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
