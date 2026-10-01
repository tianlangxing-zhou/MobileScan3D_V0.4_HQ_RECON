package com.mobilescan3d.calibration

import android.content.Context
import android.os.Build
import com.mobilescan3d.NativeBridge
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Persistent camera<->IMU calibration profile for one physical Android device.
 *
 * The app intentionally does not invent an extrinsic calibration from UI motion.
 * Users can import a profile produced by a proper calibration tool, while the
 * online VINS time-offset estimate can be adopted as a seed for future sessions.
 */
object DeviceCalibrationManager {
    private const val PREF = "device_calibration_r7"
    private const val KEY = "profile_json"
    private const val VERSION = 1

    data class Profile(
        val deviceKey: String,
        val ric: FloatArray,
        val tic: FloatArray,
        val timeOffsetSeconds: Float,
        val estimateExtrinsic: Boolean,
        val estimateTimeOffset: Boolean,
        val updatedAtMs: Long,
        val source: String
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("version", VERSION)
            .put("deviceKey", deviceKey)
            .put("ric", JSONArray(ric.toList()))
            .put("tic", JSONArray(tic.toList()))
            .put("timeOffsetSeconds", timeOffsetSeconds.toDouble())
            .put("estimateExtrinsic", estimateExtrinsic)
            .put("estimateTimeOffset", estimateTimeOffset)
            .put("updatedAtMs", updatedAtMs)
            .put("source", source)
    }

    fun deviceKey(): String = listOf(
        Build.MANUFACTURER,
        Build.MODEL,
        Build.DEVICE,
        Build.HARDWARE
    ).joinToString("|") { it ?: "unknown" }

    fun identity(): Profile = Profile(
        deviceKey = deviceKey(),
        ric = floatArrayOf(1f,0f,0f, 0f,1f,0f, 0f,0f,1f),
        tic = floatArrayOf(0f,0f,0f),
        timeOffsetSeconds = 0f,
        estimateExtrinsic = false,
        estimateTimeOffset = true,
        updatedAtMs = 0L,
        source = "factory-default(identity)"
    )

    fun load(context: Context): Profile {
        val raw = context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
            .getString(KEY, null) ?: return identity()
        return parse(raw).getOrElse { identity() }
    }

    fun save(context: Context, profile: Profile): Result<Profile> = runCatching {
        validate(profile).getOrThrow()
        val normalized = profile.copy(
            deviceKey = deviceKey(),
            updatedAtMs = System.currentTimeMillis()
        )
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
            .edit().putString(KEY, normalized.toJson().toString()).apply()
        normalized
    }

    fun reset(context: Context) {
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE).edit().remove(KEY).apply()
    }

    fun parse(text: String): Result<Profile> = runCatching {
        val j = JSONObject(text)
        val rj = j.getJSONArray("ric")
        val tj = j.getJSONArray("tic")
        require(rj.length() == 9) { "ric 必须有 9 个数" }
        require(tj.length() == 3) { "tic 必须有 3 个数" }
        val r = FloatArray(9) { rj.getDouble(it).toFloat() }
        val t = FloatArray(3) { tj.getDouble(it).toFloat() }
        val p = Profile(
            deviceKey = j.optString("deviceKey", deviceKey()),
            ric = r,
            tic = t,
            timeOffsetSeconds = j.optDouble("timeOffsetSeconds", 0.0).toFloat(),
            estimateExtrinsic = j.optBoolean("estimateExtrinsic", false),
            estimateTimeOffset = j.optBoolean("estimateTimeOffset", true),
            updatedAtMs = j.optLong("updatedAtMs", 0L),
            source = j.optString("source", "imported-json")
        )
        validate(p).getOrThrow()
        p
    }

    fun validate(profile: Profile): Result<Unit> = runCatching {
        require(profile.ric.size == 9) { "ric 长度必须为 9" }
        require(profile.tic.size == 3) { "tic 长度必须为 3" }
        require(profile.ric.all { it.isFinite() }) { "ric 含非有限值" }
        require(profile.tic.all { it.isFinite() }) { "tic 含非有限值" }
        require(profile.timeOffsetSeconds.isFinite() && abs(profile.timeOffsetSeconds) <= 0.150f) {
            "timeOffset 必须在 ±150ms 内"
        }
        fun dotRow(a: Int, b: Int): Float =
            profile.ric[a*3] * profile.ric[b*3] +
                profile.ric[a*3+1] * profile.ric[b*3+1] +
                profile.ric[a*3+2] * profile.ric[b*3+2]
        for (i in 0..2) require(abs(dotRow(i,i) - 1f) < 0.15f) { "ric 行向量未归一化" }
        require(abs(dotRow(0,1)) < 0.15f && abs(dotRow(0,2)) < 0.15f && abs(dotRow(1,2)) < 0.15f) {
            "ric 不是近似正交矩阵"
        }
        val r = profile.ric
        val det = r[0]*(r[4]*r[8]-r[5]*r[7]) - r[1]*(r[3]*r[8]-r[5]*r[6]) + r[2]*(r[3]*r[7]-r[4]*r[6])
        require(det in 0.80f..1.20f) { "ric 行列式异常：$det" }
        val tn = sqrt(profile.tic.sumOf { (it * it).toDouble() }).toFloat()
        require(tn <= 0.50f) { "tic 长度超过 0.5m，疑似单位错误" }
    }

    fun applyToNative(context: Context): Boolean {
        val p = load(context)
        return try {
            NativeBridge.nativeSetVinsCalibrationProfile(
                p.ric,
                p.tic,
                p.timeOffsetSeconds,
                p.estimateExtrinsic,
                p.estimateTimeOffset
            )
        } catch (_: Throwable) {
            false
        }
    }

    fun withCurrentVinsEstimate(context: Context, health: FloatArray): Result<Profile> {
        require(health.size >= NativeBridge.VINS_HEALTH_SLOTS) { "VINS health 数据不足" }
        require(health[NativeBridge.VINS_HEALTH_INITIALIZED] >= 0.5f) { "VINS 尚未稳定初始化" }
        val ric = FloatArray(9) { health[NativeBridge.VINS_HEALTH_RIC_START + it] }
        val tic = FloatArray(3) { health[NativeBridge.VINS_HEALTH_TIC_START + it] }
        val td = health[NativeBridge.VINS_HEALTH_TIME_OFFSET]
        val current = load(context)
        return save(
            context,
            current.copy(
                ric = ric,
                tic = tic,
                timeOffsetSeconds = td,
                // After adopting a converged profile, freeze extrinsic in normal scans.
                estimateExtrinsic = false,
                estimateTimeOffset = true,
                source = "adopted-vins-extrinsic-and-time-offset"
            )
        )
    }

    fun withCurrentTimeOffset(context: Context, health: FloatArray): Result<Profile> {
        require(health.size >= NativeBridge.VINS_HEALTH_SLOTS) { "VINS health 数据不足" }
        val td = health[NativeBridge.VINS_HEALTH_TIME_OFFSET]
        require(td.isFinite() && abs(td) <= 0.150f) { "当前时延估计无效" }
        val current = load(context)
        return save(
            context,
            current.copy(
                timeOffsetSeconds = td,
                source = "adopted-vins-time-offset"
            )
        )
    }

    fun pretty(profile: Profile): String {
        fun f(v: Float) = "%.6f".format(v)
        val r = profile.ric
        val t = profile.tic
        return buildString {
            appendLine("设备：${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("来源：${profile.source}")
            appendLine("RIC:")
            appendLine("  [${f(r[0])}, ${f(r[1])}, ${f(r[2])}]")
            appendLine("  [${f(r[3])}, ${f(r[4])}, ${f(r[5])}]")
            appendLine("  [${f(r[6])}, ${f(r[7])}, ${f(r[8])}]")
            appendLine("TIC(m): [${f(t[0])}, ${f(t[1])}, ${f(t[2])}]")
            appendLine("timeOffset: ${"%.3f".format(profile.timeOffsetSeconds * 1000f)} ms")
            appendLine("外参在线估计：${if (profile.estimateExtrinsic) "实验模式" else "关闭"}")
            append("时延在线估计：${if (profile.estimateTimeOffset) "开启" else "冻结"}")
        }
    }
}
