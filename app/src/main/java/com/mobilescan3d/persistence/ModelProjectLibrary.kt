package com.mobilescan3d.persistence

import android.content.Context
import java.io.File

object ModelProjectLibrary {
    enum class Kind { COMPLETED_PACKAGE, EXPORTED_MODEL, UNFINISHED_SCAN }

    data class Entry(
        val id: String,
        val kind: Kind,
        val displayName: String,
        val createdAtMs: Long,
        val bytes: Long,
        val modelFile: File? = null,
        val packageInfo: ScanPackageManager.PackageInfo? = null,
        val recoveryInfo: ScanRecoveryManager.Info? = null
    )

    private const val PREFS = "model_project_library_names"

    fun list(context: Context, exportDir: File): List<Entry> {
        val entries = ArrayList<Entry>()
        val packageModels = HashSet<String>()

        ScanPackageManager.list(context).forEach { info ->
            val canonical = runCatching { info.model.canonicalPath }.getOrNull()
            if (canonical != null) packageModels += canonical
            val id = "pkg:${info.sessionId}"
            entries += Entry(
                id = id,
                kind = Kind.COMPLETED_PACKAGE,
                displayName = name(context, id, completedName(info.createdAtMs)),
                createdAtMs = info.createdAtMs,
                bytes = info.model.length(),
                modelFile = info.model,
                packageInfo = info
            )
        }

        exportDir.listFiles()
            ?.asSequence()
            ?.filter { it.isFile }
            ?.filter {
                val n = it.name.lowercase()
                (n.endsWith(".glb") || n.endsWith(".obj")) && !n.contains(".pending")
            }
            ?.forEach { file ->
                val canonical = runCatching { file.canonicalPath }.getOrNull()
                if (canonical != null && canonical in packageModels) return@forEach
                val id = "file:${canonical ?: file.absolutePath}"
                entries += Entry(
                    id = id,
                    kind = Kind.EXPORTED_MODEL,
                    displayName = name(context, id, file.nameWithoutExtension),
                    createdAtMs = file.lastModified().takeIf { it > 0 } ?: System.currentTimeMillis(),
                    bytes = file.length(),
                    modelFile = file
                )
            }

        ScanRecoveryManager.latest(context)?.let { recovery ->
            val id = "recovery:${recovery.sessionId}"
            entries += Entry(
                id = id,
                kind = Kind.UNFINISHED_SCAN,
                displayName = name(context, id, "未完成扫描"),
                createdAtMs = recovery.createdAtMs,
                bytes = recovery.scene.length() + recovery.target.length(),
                recoveryInfo = recovery
            )
        }
        return entries.sortedByDescending { it.createdAtMs }
    }

    fun rename(context: Context, entry: Entry, newName: String): Boolean {
        val clean = newName.trim().replace(Regex("[\\r\\n\\t]"), " ").take(48)
        if (clean.isBlank()) return false
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putString(entry.id, clean).apply()
        return true
    }

    fun delete(context: Context, entry: Entry): Boolean {
        val ok = when (entry.kind) {
            Kind.COMPLETED_PACKAGE -> entry.packageInfo?.let { ScanPackageManager.delete(it) } ?: false
            Kind.EXPORTED_MODEL -> entry.modelFile?.let { !it.exists() || it.delete() } ?: false
            Kind.UNFINISHED_SCAN -> { ScanRecoveryManager.clear(context); true }
        }
        if (ok) context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(entry.id).apply()
        return ok
    }

    private fun name(context: Context, id: String, fallback: String): String =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(id, fallback)?.takeIf { it.isNotBlank() } ?: fallback

    private fun completedName(ts: Long): String {
        val f = java.text.SimpleDateFormat("MM-dd HH:mm", java.util.Locale.getDefault())
        return "模型 ${f.format(java.util.Date(ts))}"
    }
}
