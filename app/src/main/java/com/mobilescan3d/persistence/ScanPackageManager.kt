package com.mobilescan3d.persistence

import android.content.Context
import com.mobilescan3d.NativeBridge
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * V0.7 cross-session scan package.
 *
 * scan_packages/<sessionId>/
 *   manifest.json
 *   model.glb       standard external asset
 *   ar_asset.msar   exact UV mesh + embedded atlas used by AR overlay
 *   vins_map.vmap   VINS-world 3D points + ORB descriptors
 *
 * A package is published only when BOTH relocalization map and AR asset are
 * valid. This avoids offering a "restore AR" entry that can show geometry but
 * has no way to place it back in the physical world.
 */
object ScanPackageManager {

    private const val FORMAT_VERSION = 7
    private const val ROOT_DIR = "scan_packages"

    private const val MANIFEST = "manifest.json"
    private const val MODEL = "model.glb"
    private const val AR_ASSET = "ar_asset.msar"
    private const val VINS_MAP = "vins_map.vmap"

    data class PackageInfo(
        val sessionId: String,
        val dir: File,
        val model: File,
        val arAsset: File,
        val vinsMap: File,
        val createdAtMs: Long,
        val mapPoints: Int = 0
    )

    data class Result(
        val ok: Boolean,
        val message: String,
        val packageInfo: PackageInfo? = null
    )

    private fun root(context: Context): File {
        val base =
            context.getExternalFilesDir(null)
                ?: context.filesDir

        return File(base, ROOT_DIR).apply {
            mkdirs()
        }
    }

    fun saveCurrent(
        context: Context,
        sessionId: String,
        sourceGlb: File
    ): Result {
        if (
            sessionId.isBlank() ||
            !sourceGlb.exists() ||
            sourceGlb.length() <= 0L
        ) {
            return Result(
                false,
                "GLB 不存在，无法保存跨会话 AR"
            )
        }

        val safeId =
            sessionId.replace(
                Regex("[^A-Za-z0-9_.-]"),
                "_"
            )

        val root = root(context)
        val tmp =
            File(
                root,
                ".${safeId}.tmp"
            )

        val finalDir =
            File(
                root,
                safeId
            )

        tmp.deleteRecursively()
        tmp.mkdirs()

        val model =
            File(tmp, MODEL)
        val arAsset =
            File(tmp, AR_ASSET)
        val vinsMap =
            File(tmp, VINS_MAP)

        return try {
            sourceGlb.inputStream().use { input ->
                model.outputStream().use { output ->
                    input.copyTo(output)
                }
            }

            val mapOk =
                NativeBridge.nativeSavePersistentMap(
                    vinsMap.absolutePath
                )

            if (!mapOk || vinsMap.length() <= 0L) {
                tmp.deleteRecursively()
                return Result(
                    false,
                    "VINS 持久地图不足：请增加带纹理区域的绕拍角度"
                )
            }

            val assetOk =
                NativeBridge.nativeSaveTexturedArAsset(
                    arAsset.absolutePath
                )

            if (!assetOk || arAsset.length() <= 0L) {
                tmp.deleteRecursively()
                return Result(
                    false,
                    "HQ 纹理 AR 资产未生成，无法保存跨会话包"
                )
            }

            val stats =
                FloatArray(
                    NativeBridge.RELOCALIZATION_STATS_SLOTS
                )

            try {
                NativeBridge.nativeGetRelocalizationStats(
                    stats
                )
            } catch (_: Throwable) {
            }

            val created =
                System.currentTimeMillis()

            val manifest =
                JSONObject()
                    .put(
                        "formatVersion",
                        FORMAT_VERSION
                    )
                    .put(
                        "sessionId",
                        safeId
                    )
                    .put(
                        "createdAtMs",
                        created
                    )
                    .put(
                        "model",
                        MODEL
                    )
                    .put(
                        "arAsset",
                        AR_ASSET
                    )
                    .put(
                        "vinsMap",
                        VINS_MAP
                    )
                    .put(
                        "mapPoints",
                        stats
                            .getOrElse(
                                NativeBridge.RELOC_INDEX_MAP_POINTS
                            ) { 0f }
                            .toInt()
                    )
                    .put(
                        "modelSha256",
                        sha256(model)
                    )
                    .put(
                        "arAssetSha256",
                        sha256(arAsset)
                    )
                    .put(
                        "vinsMapSha256",
                        sha256(vinsMap)
                    )

            File(
                tmp,
                MANIFEST
            ).writeText(
                manifest.toString(2),
                Charsets.UTF_8
            )

            if (finalDir.exists()) {
                finalDir.deleteRecursively()
            }

            // Same filesystem under app external files: rename is atomic on
            // Android's normal ext4/f2fs storage.
            if (!tmp.renameTo(finalDir)) {
                finalDir.mkdirs()
                tmp.copyRecursively(
                    finalDir,
                    overwrite = true
                )
                tmp.deleteRecursively()
            }

            val info =
                readPackage(finalDir)
                    ?: return Result(
                        false,
                        "扫描包写入后校验失败"
                    )

            Result(
                true,
                "跨会话 AR 已保存：${info.sessionId}",
                info
            )
        } catch (t: Throwable) {
            tmp.deleteRecursively()

            Result(
                false,
                "保存扫描包失败：${t.message ?: t.javaClass.simpleName}"
            )
        }
    }

    fun latest(
        context: Context
    ): PackageInfo? =
        list(context)
            .maxByOrNull {
                it.createdAtMs
            }

    fun list(
        context: Context
    ): List<PackageInfo> {
        return root(context)
            .listFiles()
            ?.asSequence()
            ?.filter {
                it.isDirectory &&
                    !it.name.startsWith(".")
            }
            ?.mapNotNull {
                readPackage(it)
            }
            ?.sortedByDescending {
                it.createdAtMs
            }
            ?.toList()
            ?: emptyList()
    }

    fun restore(
        context: Context,
        packageInfo: PackageInfo
    ): Result {
        val verified =
            readPackage(
                packageInfo.dir
            )
                ?: return Result(
                    false,
                    "扫描包损坏或校验失败"
                )

        return try {
            // Asset can be loaded first, but renderer must not draw it until
            // nativeGetRenderPose* succeeds after visual relocalization.
            val assetOk =
                NativeBridge.nativeLoadTexturedArAsset(
                    verified.arAsset.absolutePath
                )

            if (!assetOk) {
                return Result(
                    false,
                    "AR 纹理资产加载失败"
                )
            }

            val mapOk =
                NativeBridge.nativeLoadPersistentMap(
                    verified.vinsMap.absolutePath
                )

            if (!mapOk) {
                return Result(
                    false,
                    "VINS 持久地图加载失败"
                )
            }

            Result(
                true,
                "地图已加载，请缓慢移动手机寻找原扫描区域",
                verified
            )
        } catch (t: Throwable) {
            Result(
                false,
                "恢复失败：${t.message ?: t.javaClass.simpleName}"
            )
        }
    }

    fun delete(
        packageInfo: PackageInfo
    ): Boolean =
        packageInfo.dir.deleteRecursively()

    private fun readPackage(
        dir: File
    ): PackageInfo? {
        return try {
            val manifestFile =
                File(dir, MANIFEST)

            if (!manifestFile.exists()) {
                return null
            }

            val json =
                JSONObject(
                    manifestFile.readText(
                        Charsets.UTF_8
                    )
                )

            if (
                json.optInt(
                    "formatVersion",
                    -1
                ) != FORMAT_VERSION
            ) {
                return null
            }

            val session =
                json.optString(
                    "sessionId",
                    ""
                )

            if (session.isBlank()) {
                return null
            }

            val model =
                File(
                    dir,
                    json.optString(
                        "model",
                        MODEL
                    )
                )

            val asset =
                File(
                    dir,
                    json.optString(
                        "arAsset",
                        AR_ASSET
                    )
                )

            val map =
                File(
                    dir,
                    json.optString(
                        "vinsMap",
                        VINS_MAP
                    )
                )

            if (
                !model.exists() ||
                !asset.exists() ||
                !map.exists()
            ) {
                return null
            }

            if (
                sha256(model) !=
                    json.optString(
                        "modelSha256",
                        ""
                    ) ||
                sha256(asset) !=
                    json.optString(
                        "arAssetSha256",
                        ""
                    ) ||
                sha256(map) !=
                    json.optString(
                        "vinsMapSha256",
                        ""
                    )
            ) {
                return null
            }

            PackageInfo(
                sessionId = session,
                dir = dir,
                model = model,
                arAsset = asset,
                vinsMap = map,
                createdAtMs =
                    json.optLong(
                        "createdAtMs",
                        dir.lastModified()
                    ),
                mapPoints =
                    json.optInt(
                        "mapPoints",
                        0
                    )
            )
        } catch (_: Throwable) {
            null
        }
    }

    private fun sha256(
        file: File
    ): String {
        val digest =
            MessageDigest.getInstance(
                "SHA-256"
            )

        file.inputStream().use { input ->
            val buffer =
                ByteArray(64 * 1024)

            while (true) {
                val n =
                    input.read(buffer)

                if (n <= 0) break

                digest.update(
                    buffer,
                    0,
                    n
                )
            }
        }

        return digest
            .digest()
            .joinToString("") {
                "%02x".format(it.toInt() and 0xFF)
            }
    }
}
