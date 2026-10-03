package com.mobilescan3d.persistence

import android.content.Context
import com.mobilescan3d.NativeBridge
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

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
    private const val MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1_500L * 1024L * 1024L

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
    data class ArchiveResult(
        val ok: Boolean,
        val message: String,
        val file: File? = null,
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

    @Synchronized
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

        // Immutable publication: a failed re-export must not delete the previous package.
        val safeId = sessionId.replace(Regex("[^A-Za-z0-9_-]"), "_").take(96) +
            "_" + java.util.UUID.randomUUID().toString()

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

            if (readPackage(tmp) == null) {
                tmp.deleteRecursively()
                return Result(false, "扫描包发布前校验失败")
            }
            // Do not fall back to copying into a visible directory: readers could
            // observe a partial package. Keep all publication on one filesystem.
            if (finalDir.exists() || !tmp.renameTo(finalDir)) {
                tmp.deleteRecursively()
                return Result(false, "扫描包原子发布失败，旧扫描包已保留")
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

    @Synchronized
    fun exportArchive(
        context: Context,
        packageInfo: PackageInfo,
        outputDir: File
    ): ArchiveResult {
        val verified = readPackage(packageInfo.dir)
            ?: return ArchiveResult(false, "扫描包损坏或校验失败")
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            return ArchiveResult(false, "无法创建工程包导出目录")
        }

        val safe = verified.sessionId
            .replace(Regex("[^A-Za-z0-9_-]"), "_")
            .take(72)
        val out = File(outputDir, "MobileScan3D_${safe}.ms3d.zip")
        val pending = File(outputDir, out.name + ".pending")
        pending.delete()

        return try {
            ZipOutputStream(pending.outputStream().buffered()).use { zip ->
                fun add(file: File, name: String) {
                    zip.putNextEntry(ZipEntry(name))
                    file.inputStream().buffered().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
                add(File(verified.dir, MANIFEST), MANIFEST)
                add(verified.model, MODEL)
                add(verified.arAsset, AR_ASSET)
                add(verified.vinsMap, VINS_MAP)

                val metadata = JSONObject()
                    .put("archiveFormat", 1)
                    .put("appPackage", context.packageName)
                    .put("exportedAtMs", System.currentTimeMillis())
                    .put("sessionId", verified.sessionId)
                    .toString(2)
                    .toByteArray(Charsets.UTF_8)
                zip.putNextEntry(ZipEntry("archive_meta.json"))
                zip.write(metadata)
                zip.closeEntry()
            }

            if (!pending.isFile || pending.length() <= 0L ||
                (out.exists() && !out.delete()) ||
                !pending.renameTo(out)
            ) {
                pending.delete()
                ArchiveResult(false, "工程包原子发布失败")
            } else {
                ArchiveResult(true, "工程包已生成：${out.name}", out, verified)
            }
        } catch (t: Throwable) {
            pending.delete()
            ArchiveResult(false, "工程包导出失败：${t.message ?: t.javaClass.simpleName}")
        }
    }

    @Synchronized
    fun importArchive(
        context: Context,
        archive: File
    ): ArchiveResult {
        if (!archive.isFile || archive.length() <= 0L) {
            return ArchiveResult(false, "工程包文件不存在")
        }

        val importRoot = root(context)
        val token = java.util.UUID.randomUUID().toString()
        val tmp = File(importRoot, ".import_$token.tmp")
        tmp.deleteRecursively()
        tmp.mkdirs()

        val allowed = setOf(MANIFEST, MODEL, AR_ASSET, VINS_MAP, "archive_meta.json")
        val required = setOf(MANIFEST, MODEL, AR_ASSET, VINS_MAP)
        val seen = HashSet<String>()
        var totalBytes = 0L

        return try {
            ZipInputStream(archive.inputStream().buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    val name = entry.name
                    if (entry.isDirectory) {
                        zip.closeEntry()
                        continue
                    }
                    if (name !in allowed || "/" in name || "\\" in name || name.startsWith(".")) {
                        tmp.deleteRecursively()
                        return ArchiveResult(false, "工程包包含不支持的文件：$name")
                    }
                    if (!seen.add(name)) {
                        tmp.deleteRecursively()
                        return ArchiveResult(false, "工程包包含重复文件：$name")
                    }

                    val target = File(tmp, name)
                    target.outputStream().buffered().use { output ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            val n = zip.read(buffer)
                            if (n <= 0) break
                            totalBytes += n
                            if (totalBytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES) {
                                tmp.deleteRecursively()
                                return ArchiveResult(false, "工程包解压后过大，已拒绝导入")
                            }
                            output.write(buffer, 0, n)
                        }
                    }
                    zip.closeEntry()
                }
            }

            if (!seen.containsAll(required)) {
                tmp.deleteRecursively()
                return ArchiveResult(false, "工程包缺少必要扫描资产")
            }

            val verified = readPackage(tmp)
            if (verified == null) {
                tmp.deleteRecursively()
                return ArchiveResult(false, "工程包校验失败：文件损坏或哈希不匹配")
            }

            val safe = verified.sessionId
                .replace(Regex("[^A-Za-z0-9_-]"), "_")
                .take(72)
            val finalDir = File(importRoot, "${safe}_import_${token.take(8)}")
            if (finalDir.exists() || !tmp.renameTo(finalDir)) {
                tmp.deleteRecursively()
                return ArchiveResult(false, "工程包原子导入失败")
            }

            val published = readPackage(finalDir)
            if (published == null) {
                finalDir.deleteRecursively()
                ArchiveResult(false, "导入后复核失败")
            } else {
                ArchiveResult(
                    true,
                    "工程包已导入：${published.sessionId}",
                    packageInfo = published
                )
            }
        } catch (t: Throwable) {
            tmp.deleteRecursively()
            ArchiveResult(false, "工程包导入失败：${t.message ?: t.javaClass.simpleName}")
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

            // Version 7 has fixed filenames; never follow paths supplied by a manifest.
            if (json.optString("model", MODEL) != MODEL ||
                json.optString("arAsset", AR_ASSET) != AR_ASSET ||
                json.optString("vinsMap", VINS_MAP) != VINS_MAP) return null
            val model = File(dir, MODEL)
            val asset = File(dir, AR_ASSET)
            val map = File(dir, VINS_MAP)
            if (listOf(model, asset, map).any {
                    !it.isFile || it.length() <= 0L || it.canonicalFile.parentFile != dir.canonicalFile
                }) return null

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
