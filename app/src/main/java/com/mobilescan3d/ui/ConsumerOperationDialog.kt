package com.mobilescan3d.ui

import android.app.AlertDialog
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView

class ConsumerOperationDialog(private val context: Context) {
    private val handler = Handler(Looper.getMainLooper())
    private var dialog: AlertDialog? = null
    private var detailView: TextView? = null
    private var startedMs = 0L
    private var baseDetail = ""

    private val tick = object : Runnable {
        override fun run() {
            val d = dialog ?: return
            if (!d.isShowing) return
            val seconds = ((SystemClock.elapsedRealtime() - startedMs) / 1000L).coerceAtLeast(0L)
            detailView?.text = buildString {
                append(baseDetail)
                append("\n已处理 ").append(seconds).append(" 秒")
                if (seconds >= 20L) {
                    append("\n大模型或高温设备会更慢；当前任务仍在本机继续处理。")
                }
            }
            handler.postDelayed(this, 1_000L)
        }
    }

    fun show(title: String, detail: String) {
        dismiss()
        startedMs = SystemClock.elapsedRealtime()
        baseDetail = detail
        val density = context.resources.displayMetrics.density
        val box = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            val pad = (24f * density).toInt()
            setPadding(pad, pad, pad, (18f * density).toInt())
        }
        box.addView(
            ProgressBar(context),
            LinearLayout.LayoutParams((44f * density).toInt(), (44f * density).toInt())
        )
        detailView = TextView(context).apply {
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(0, (16f * density).toInt(), 0, 0)
            text = detail
        }
        box.addView(
            detailView,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        )
        dialog = AlertDialog.Builder(context)
            .setTitle(title)
            .setView(box)
            .setNegativeButton("隐藏进度") { _, _ ->
                dialog = null
                detailView = null
                handler.removeCallbacks(tick)
            }
            .setCancelable(false)
            .create()
        dialog?.show()
        handler.post(tick)
    }

    fun dismiss() {
        handler.removeCallbacks(tick)
        runCatching { dialog?.dismiss() }
        dialog = null
        detailView = null
    }
}
