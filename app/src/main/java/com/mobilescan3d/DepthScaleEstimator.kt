package com.mobilescan3d

import kotlin.math.abs

/**
 * 深度尺度估计器 —— **只统计 + shadow 验证，绝不施加**。
 *
 * 实机证据（2026-09-16 报告）：
 * ```
 * AI target depth (raw)   = 4.563 m
 * Focus approximate       = 0.272 m
 * VINS triangulated depth = 0.527 m
 * AI / focus  ≈ 16.79×      AI / VINS ≈ 8.65×
 * 纠正乘数 vins/raw ≈ 0.527 / 4.563 ≈ 0.116
 * ```
 * 也就是说 raw depth 与 VINS 三角化深度之间存在一个稳定的比例失配，但
 * **一次测试场景不能拿来永久标定**：可能随场景尺度、目标距离、纹理条件而变。
 *
 * 因此这里只做两件事：
 *  1. 在 Object Lock 可靠、且 raw / VINS 两条深度都可观测时，持续收集
 *     `scale = vinsTriangulatedDepthMedian / targetRawDepthMedian`，
 *     输出 median / MAD / min / max / 样本数 / 漂移 / 是否稳定。
 *  2. **shadow correction 验证**（本轮新增）：用当前窗口 median 算
 *     `shadowMetricDepth = rawDepth × median`，并记录它与 VINS 深度的偏差。
 *     这一步只写报告，**不参与任何重建**。
 *
 * **任何地方都不允许据此自动修改深度或点云**；等样本足够多、稳定判据在
 * 连续 3～5 次不同距离的扫描里都成立之后，再由人决定是纯 scale 还是
 * 拟合 `metric = a × raw + b`。
 */
class DepthScaleEstimator {

    companion object {
        /** Object Lock 必须处于 TRACKING（与 cpp/object_tracker.h 的 TargetState 对齐） */
        const val STATE_TRACKING = 3

        /**
         * 采样间隔。
         *
         * 特意不做「每帧」：`vinsFeatureDepthMedian()` 每次都会临时分配一个特征向量
         * 再做 nth_element，30fps 全采是纯浪费；而且 100ms 一个样本能让 60 个样本覆盖
         * 约 6s 的时间窗，「最近几秒是否漂移」这个判断本来就该用秒级窗口做。
         */
        const val SAMPLE_INTERVAL_NS = 100_000_000L

        /** 保留的样本数：60 × 100ms ≈ 6s 窗口 */
        const val MAX_SAMPLES = 60

        /** 判定「稳定」所需的最少样本数（≈3s） */
        const val MIN_SAMPLES = 30

        /** 离散度门限：MAD / median */
        const val MAX_RELATIVE_MAD = 0.12f

        /** 漂移门限：后半窗 median 相对前半窗 median 的变化幅度 */
        const val MAX_DRIFT_RATIO = 0.10f

        /** 漂移判定要求前半窗至少有这么多样本 */
        const val MIN_DRIFT_HALF = 8

        /** Object Lock 必须满足的可靠性门限 */
        const val MIN_CONFIDENCE = 0.7f
        const val MIN_TRACKED_POINTS = 50
        const val MIN_INLIER_RATIO = 0.7f

        /**
         * ROI 内必须有多少个「真的有深度」的像素，这个中位数才算观测。
         *
         * 目标出界时 ROI 会被裁到只剩几行，有效像素掉到个位数，此时
         * P10/P50/P90 会退化成同一个值（实机出现过 `P10=P50=P90=5.14128`）。
         * 这种样本一旦进入统计，会把 median 往完全错误的方向拽。
         */
        const val MIN_VALID_DEPTH_PIXELS = 100
    }

    /**
     * 定长环形缓冲，插入序（最旧 → 最新）可读。
     * 用 FloatArray 而不是 ArrayDeque<Float> 是为了避免装箱。
     */
    private val buffer = FloatArray(MAX_SAMPLES)
    /** 下一个写入位置 */
    private var head = 0
    /** 当前样本数，上限 MAX_SAMPLES */
    private var size = 0

    // shadow correction 的只读验证缓冲：与 buffer 同步 push，索引一一对应
    private val shadowDepthBuf = FloatArray(MAX_SAMPLES)
    private val shadowVinsBuf = FloatArray(MAX_SAMPLES)
    private var shadowHead = 0
    private var shadowSize = 0

    private var lastSampleNs = 0L

    /** 当前窗口内的样本数 */
    var sampleCount: Int = 0
        private set
    var median: Float = 0f
        private set
    var mad: Float = 0f
        private set
    var minValue: Float = 0f
        private set
    var maxValue: Float = 0f
        private set
    var driftRatio: Float = 0f
        private set
    var stable: Boolean = false
        private set
    /** 最近一次拒绝采样的原因（空串表示上次采样成功） */
    var lastRejectReason: String = ""
        private set
    var rejectedSamples: Long = 0L
        private set

    // ---- shadow correction（只记录，绝不施加）----
    var shadowSampleCount: Int = 0
        private set
    var shadowDepthMedian: Float = 0f
        private set
    var shadowVsVinsErrorMeters: Float = 0f
        private set
    var shadowVsVinsErrorPercent: Float = 0f
        private set
    /** 最近一次采样时 ROI 里真正有深度的像素数，用于判断样本可信度 */
    var lastValidDepthPixels: Int = 0
        private set

    fun reset() {
        head = 0
        size = 0
        shadowHead = 0
        shadowSize = 0
        lastSampleNs = 0L
        sampleCount = 0
        median = 0f
        mad = 0f
        minValue = 0f
        maxValue = 0f
        driftRatio = 0f
        stable = false
        lastRejectReason = ""
        rejectedSamples = 0L
        shadowSampleCount = 0
        shadowDepthMedian = 0f
        shadowVsVinsErrorMeters = 0f
        shadowVsVinsErrorPercent = 0f
        lastValidDepthPixels = 0
    }

    /** 第 k 个样本（k=0 最旧） */
    private fun at(k: Int): Float {
        val oldest = if (size < MAX_SAMPLES) 0 else head
        return buffer[(oldest + k) % MAX_SAMPLES]
    }

    private fun push(v: Float) {
        buffer[head] = v
        head = (head + 1) % MAX_SAMPLES
        if (size < MAX_SAMPLES) size++
    }

    private fun shadowAt(buf: FloatArray, k: Int): Float {
        val oldest = if (shadowSize < MAX_SAMPLES) 0 else shadowHead
        return buf[(oldest + k) % MAX_SAMPLES]
    }

    private fun pushShadow(shadowDepth: Float, vinsDepth: Float) {
        shadowDepthBuf[shadowHead] = shadowDepth
        shadowVinsBuf[shadowHead] = vinsDepth
        shadowHead = (shadowHead + 1) % MAX_SAMPLES
        if (shadowSize < MAX_SAMPLES) shadowSize++
        shadowSampleCount = shadowSize
    }

    /**
     * 满足门限时收集一个 scale 样本，返回是否真的采样了。
     *
     * 门限不满足时**不**清空已有样本 —— 短暂的 track 抖动不应该把几秒的统计历史抹掉，
     * 只记录拒绝原因和累计次数，方便在报告里区分「从来没采到」和「间歇性采到」。
     */
    fun maybeSample(
        nowNs: Long,
        stateCode: Int,
        confidence: Float,
        trackedPoints: Int,
        inlierRatio: Float,
        targetRawDepthMedian: Float,
        vinsDepthMedian: Float,
        targetDepthValidPixels: Int
    ): Boolean {
        lastValidDepthPixels = targetDepthValidPixels
        if (lastSampleNs != 0L && nowNs - lastSampleNs < SAMPLE_INTERVAL_NS) return false
        lastSampleNs = nowNs

        val reason = when {
            stateCode != STATE_TRACKING ->
                "state=$stateCode(非 TRACKING)"
            confidence <= MIN_CONFIDENCE ->
                "confidence=${"%.2f".format(confidence)}≤$MIN_CONFIDENCE"
            trackedPoints <= MIN_TRACKED_POINTS ->
                "trackedPoints=$trackedPoints≤$MIN_TRACKED_POINTS"
            inlierRatio <= MIN_INLIER_RATIO ->
                "inlierRatio=${"%.2f".format(inlierRatio)}≤$MIN_INLIER_RATIO"
            targetDepthValidPixels < MIN_VALID_DEPTH_PIXELS ->
                "validDepthPixels=$targetDepthValidPixels<$MIN_VALID_DEPTH_PIXELS(目标可能出界)"
            targetRawDepthMedian <= 0f ->
                "targetRawDepthMedian=${"%.3f".format(targetRawDepthMedian)}≤0"
            vinsDepthMedian <= 0f ->
                "vinsDepthMedian=${"%.3f".format(vinsDepthMedian)}≤0"
            else -> ""
        }
        if (reason.isNotEmpty()) {
            rejectedSamples++
            lastRejectReason = reason
            return false
        }

        val scale = vinsDepthMedian / targetRawDepthMedian
        if (!scale.isFinite() || scale <= 0f) {
            rejectedSamples++
            lastRejectReason = "scale=${"%.4f".format(scale)}(非法)"
            return false
        }

        push(scale)
        recompute()
        // shadow 值用「本样本进来之后」的窗口 median，才是可复现的确定性结果；
        // 它只写报告，绝不参与重建。
        if (stable || sampleCount >= MIN_SAMPLES) {
            pushShadow(targetRawDepthMedian * median, vinsDepthMedian)
            recomputeShadow()
        }
        lastRejectReason = ""
        return true
    }

    private fun recompute() {
        val n = size
        sampleCount = n
        if (n == 0) {
            median = 0f
            mad = 0f
            minValue = 0f
            maxValue = 0f
            driftRatio = 0f
            stable = false
            return
        }

        // 按插入序展开（最旧 → 最新）：漂移判定要用它，排序会破坏顺序。
        val ordered = FloatArray(n)
        for (k in 0 until n) ordered[k] = at(k)

        val sorted = ordered.copyOf()
        sorted.sort()

        minValue = sorted[0]
        maxValue = sorted[n - 1]
        median = medianOfSorted(sorted)

        // MAD = 各样本到 median 的绝对偏差的中位数
        val dev = FloatArray(n)
        for (k in 0 until n) dev[k] = abs(ordered[k] - median)
        dev.sort()
        mad = medianOfSorted(dev)
        val relativeMad = if (median > 0f) mad / median else Float.MAX_VALUE

        // 漂移：后半窗 median 相对前半窗 median 的变化幅度。
        // 只在前半窗样本足够时才有意义，否则噪声太大。
        val half = n / 2
        driftRatio = if (half >= MIN_DRIFT_HALF) {
            val older = medianOfRange(ordered, 0, half)
            val newer = medianOfRange(ordered, half, n)
            if (older > 0f) abs(newer - older) / older else 0f
        } else {
            0f
        }

        stable = n >= MIN_SAMPLES &&
            relativeMad < MAX_RELATIVE_MAD &&
            driftRatio < MAX_DRIFT_RATIO
    }

    /**
     * shadow 误差：把 raw 深度按窗口 median 乘成「假装是米制」的深度，
     * 再与同一时刻的 VINS 三角化深度比。这一轮只看它是否稳定在 0 附近，
     * **不把它接回 TSDF / 点云**。
     */
    private fun recomputeShadow() {
        val n = shadowSize
        shadowSampleCount = n
        if (n == 0) {
            shadowDepthMedian = 0f
            shadowVsVinsErrorMeters = 0f
            shadowVsVinsErrorPercent = 0f
            return
        }
        val depths = FloatArray(n)
        val errMeters = FloatArray(n)
        val errPercent = FloatArray(n)
        for (k in 0 until n) {
            val sh = shadowAt(shadowDepthBuf, k)
            val vv = shadowAt(shadowVinsBuf, k)
            depths[k] = sh
            errMeters[k] = sh - vv
            errPercent[k] = if (vv > 1e-3f) 100f * (sh - vv) / vv else 0f
        }
        depths.sort()
        errMeters.sort()
        errPercent.sort()
        shadowDepthMedian = medianOfSorted(depths)
        shadowVsVinsErrorMeters = medianOfSorted(errMeters)
        shadowVsVinsErrorPercent = medianOfSorted(errPercent)
    }

    /** 已排序数组的中位数 */
    private fun medianOfSorted(sorted: FloatArray): Float {
        val n = sorted.size
        if (n == 0) return 0f
        return if (n % 2 == 1) sorted[n / 2]
        else 0.5f * (sorted[n / 2 - 1] + sorted[n / 2])
    }

    /** 未排序数组 [from, to) 区间的中位数 */
    private fun medianOfRange(arr: FloatArray, from: Int, to: Int): Float {
        val len = to - from
        if (len <= 0) return 0f
        val seg = FloatArray(len)
        System.arraycopy(arr, from, seg, 0, len)
        seg.sort()
        return medianOfSorted(seg)
    }

    /**
     * 写进诊断报告。字段名与评审要求一致：
     * `depthScaleSamples` / `depthScaleMedian` / `depthScaleMAD` / `depthScaleStable`
     * / `shadowDepthMedian` / `shadowVsVinsErrorMeters` / `shadowVsVinsErrorPercent`。
     */
    fun report(sb: StringBuilder, indent: String = "") {
        sb.appendLine("${indent}depthScaleDirection=vins/raw (=depthCorrectionScaleVins)")
        sb.appendLine("${indent}depthScaleSamples=$sampleCount")
        sb.appendLine("${indent}depthScaleMedian=$median")
        sb.appendLine("${indent}depthScaleMAD=$mad")
        sb.appendLine("${indent}depthScaleMin=$minValue")
        sb.appendLine("${indent}depthScaleMax=$maxValue")
        sb.appendLine("${indent}depthScaleDriftRatio=$driftRatio")
        sb.appendLine("${indent}depthScaleStable=$stable")
        sb.appendLine("${indent}depthScaleRejected=$rejectedSamples")
        sb.appendLine("${indent}depthScaleMinValidDepthPixels=$MIN_VALID_DEPTH_PIXELS")
        sb.appendLine("${indent}depthScaleLastValidDepthPixels=$lastValidDepthPixels")
        if (lastRejectReason.isNotEmpty()) {
            sb.appendLine("${indent}depthScaleRejectReason=$lastRejectReason")
        }
        // shadow correction 验证：只记录，绝不写回 TSDF / 点云。
        // shadowCorrectionApplied 恒为 false —— 这一轮明确不施加修正。
        sb.appendLine("${indent}shadowCorrectionMode=true")
        sb.appendLine("${indent}shadowCorrectionApplied=false")
        sb.appendLine("${indent}shadowSamples=$shadowSampleCount")
        sb.appendLine("${indent}shadowDepthMedian=$shadowDepthMedian")
        sb.appendLine("${indent}shadowVsVinsErrorMeters=$shadowVsVinsErrorMeters")
        sb.appendLine("${indent}shadowVsVinsErrorPercent=$shadowVsVinsErrorPercent")
    }
}
