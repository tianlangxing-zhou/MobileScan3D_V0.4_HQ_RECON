package com.mobilescan3d.ui.library

import android.app.AlertDialog
import android.widget.EditText
import androidx.activity.ComponentActivity
import com.mobilescan3d.persistence.ModelProjectLibrary
import com.mobilescan3d.persistence.ScanPackageManager
import com.mobilescan3d.persistence.ScanRecoveryManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ProjectLibraryDialog {
    data class Callbacks(
        val restorePackage: (ScanPackageManager.PackageInfo) -> Unit,
        val restoreRecovery: (ScanRecoveryManager.Info) -> Unit,
        val shareFile: (File) -> Unit
    )

    fun show(activity: ComponentActivity, exportDir: File, callbacks: Callbacks) {
        val loading = AlertDialog.Builder(activity)
            .setTitle("我的模型")
            .setMessage("正在校验本地模型…")
            .setCancelable(false)
            .create()
        loading.show()

        kotlin.concurrent.thread(name = "ProjectLibraryIndex", isDaemon = true) {
            val entries = runCatching {
                ModelProjectLibrary.list(activity.applicationContext, exportDir)
            }.getOrDefault(emptyList())
            activity.runOnUiThread {
                if (activity.isDestroyed) return@runOnUiThread
                runCatching { loading.dismiss() }
                showEntries(activity, entries, exportDir, callbacks)
            }
        }
    }

    private fun showEntries(
        activity: ComponentActivity,
        entries: List<ModelProjectLibrary.Entry>,
        exportDir: File,
        callbacks: Callbacks
    ) {
        if (entries.isEmpty()) {
            AlertDialog.Builder(activity)
                .setTitle("我的模型")
                .setMessage("还没有保存的模型或未完成扫描。")
                .setPositiveButton("关闭", null).show()
            return
        }
        val date = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
        val labels = entries.map {
            val kind = when (it.kind) {
                ModelProjectLibrary.Kind.COMPLETED_PACKAGE -> "已完成"
                ModelProjectLibrary.Kind.EXPORTED_MODEL -> "已导出"
                ModelProjectLibrary.Kind.UNFINISHED_SCAN -> "未完成"
            }
            "${it.displayName}\n$kind · ${date.format(Date(it.createdAtMs))} · ${formatBytes(it.bytes)}"
        }.toTypedArray()

        AlertDialog.Builder(activity)
            .setTitle("我的模型 · ${entries.size}")
            .setItems(labels) { _, i -> showEntry(activity, entries[i], exportDir, callbacks) }
            .setNegativeButton("关闭", null).show()
    }

    private fun showEntry(
        activity: ComponentActivity,
        entry: ModelProjectLibrary.Entry,
        exportDir: File,
        callbacks: Callbacks
    ) {
        val actions = when (entry.kind) {
            ModelProjectLibrary.Kind.COMPLETED_PACKAGE -> arrayOf("恢复 AR", "分享模型", "重命名", "删除")
            ModelProjectLibrary.Kind.EXPORTED_MODEL -> arrayOf("分享模型", "重命名", "删除")
            ModelProjectLibrary.Kind.UNFINISHED_SCAN -> arrayOf("恢复模型", "重命名", "删除断点")
        }
        AlertDialog.Builder(activity)
            .setTitle(entry.displayName)
            .setItems(actions) { _, which ->
                when (entry.kind) {
                    ModelProjectLibrary.Kind.COMPLETED_PACKAGE -> when (which) {
                        0 -> entry.packageInfo?.let(callbacks.restorePackage)
                        1 -> entry.modelFile?.let(callbacks.shareFile)
                        2 -> rename(activity, entry, exportDir, callbacks)
                        3 -> confirmDelete(activity, entry, exportDir, callbacks)
                    }
                    ModelProjectLibrary.Kind.EXPORTED_MODEL -> when (which) {
                        0 -> entry.modelFile?.let(callbacks.shareFile)
                        1 -> rename(activity, entry, exportDir, callbacks)
                        2 -> confirmDelete(activity, entry, exportDir, callbacks)
                    }
                    ModelProjectLibrary.Kind.UNFINISHED_SCAN -> when (which) {
                        0 -> entry.recoveryInfo?.let(callbacks.restoreRecovery)
                        1 -> rename(activity, entry, exportDir, callbacks)
                        2 -> confirmDelete(activity, entry, exportDir, callbacks)
                    }
                }
            }
            .setNegativeButton("返回", null).show()
    }

    private fun rename(
        activity: ComponentActivity,
        entry: ModelProjectLibrary.Entry,
        exportDir: File,
        callbacks: Callbacks
    ) {
        val edit = EditText(activity).apply {
            setText(entry.displayName); setSelection(text.length); setSingleLine(true)
        }
        AlertDialog.Builder(activity)
            .setTitle("重命名").setView(edit)
            .setNegativeButton("取消", null)
            .setPositiveButton("保存") { _, _ ->
                ModelProjectLibrary.rename(activity.applicationContext, entry, edit.text.toString())
                show(activity, exportDir, callbacks)
            }.show()
    }

    private fun confirmDelete(
        activity: ComponentActivity,
        entry: ModelProjectLibrary.Entry,
        exportDir: File,
        callbacks: Callbacks
    ) {
        AlertDialog.Builder(activity)
            .setTitle("删除“${entry.displayName}”？")
            .setMessage("操作不可撤销。")
            .setNegativeButton("取消", null)
            .setPositiveButton("删除") { _, _ ->
                ModelProjectLibrary.delete(activity.applicationContext, entry)
                show(activity, exportDir, callbacks)
            }.show()
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1024L * 1024L -> "%.1f MB".format(bytes / (1024.0 * 1024.0))
        bytes >= 1024L -> "%.0f KB".format(bytes / 1024.0)
        else -> "$bytes B"
    }
}
