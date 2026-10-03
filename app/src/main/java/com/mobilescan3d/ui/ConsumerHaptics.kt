package com.mobilescan3d.ui

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

class ConsumerHaptics(context: Context) {
    enum class Event {
        TARGET_LOCKED,
        TARGET_LOST,
        READY_TO_ORBIT,
        COVERAGE_MILESTONE,
        READY_TO_BUILD,
        WARNING
    }

    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val vibrator: Vibrator? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)
                ?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            appContext.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    } catch (_: Throwable) {
        null
    }

    private var lastPulseMs = 0L

    fun isEnabled(): Boolean = prefs.getBoolean(KEY_ENABLED, true)

    fun setEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_ENABLED, enabled).apply()
    }

    fun emit(event: Event) {
        if (!isEnabled()) return
        val v = vibrator ?: return
        if (!v.hasVibrator()) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastPulseMs < 260L) return
        lastPulseMs = now

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val effect = when (event) {
                    Event.TARGET_LOCKED ->
                        VibrationEffect.createOneShot(28L, 90)
                    Event.TARGET_LOST, Event.WARNING ->
                        VibrationEffect.createWaveform(
                            longArrayOf(0L, 55L, 55L, 55L),
                            intArrayOf(0, 150, 0, 150),
                            -1
                        )
                    Event.READY_TO_ORBIT ->
                        VibrationEffect.createWaveform(
                            longArrayOf(0L, 24L, 36L, 34L),
                            intArrayOf(0, 90, 0, 120),
                            -1
                        )
                    Event.COVERAGE_MILESTONE ->
                        VibrationEffect.createOneShot(22L, 70)
                    Event.READY_TO_BUILD ->
                        VibrationEffect.createWaveform(
                            longArrayOf(0L, 28L, 32L, 28L, 32L, 42L),
                            intArrayOf(0, 95, 0, 110, 0, 150),
                            -1
                        )
                }
                v.vibrate(effect)
            } else {
                @Suppress("DEPRECATION")
                v.vibrate(if (event == Event.TARGET_LOST || event == Event.WARNING) 100L else 35L)
            }
        } catch (_: Throwable) {
            // Haptics must never affect scanning.
        }
    }

    companion object {
        private const val PREFS = "scan_settings"
        private const val KEY_ENABLED = "consumer_haptics_enabled"
    }
}
