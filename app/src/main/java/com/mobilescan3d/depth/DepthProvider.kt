package com.mobilescan3d.depth

import android.media.Image

/**
 * 深度来源的统一接口。
 *
 * 为什么要把深度抽象出来（评审 P0-1）：
 * 之前在 Activity 里直接 `depthProvider.estimate(...)`，深度来源既换不了、
 * 也没法把「硬件深度优先、模型兜底」这条策略表达出来。现在把
 *   - 设备真的暴露 DEPTH16（ToF / 双目）时的 [HardwareDepthProvider]
 *   - 没有硬件时用单目模型推理的 [MonoDepthProvider]
 * 收敛到同一个接口，Activity 只跟接口打交道。
 *
 * 形态刻意用「推一帧 + 取最新结果」而不是「同步返回」：
 *   - 硬件深度是**异步**到达的（ImageReader 回调），根本没法同步返回；
 *   - 单目推理是同步的，但把它也包进同一形态后，上层只有一条代码路径。
 *
 * **线程约定**：`submitFrame` / `submitDepthImage` / `latest` / `close`
 * 都要求由**同一个专用线程**调用（本工程是 "DepthInference" HandlerThread）。
 * 不满足时调用方要自己加同步。
 */
interface DepthProvider {

    /**
     * 一帧深度结果。
     *
     * @param depth       长度 width*height 的深度缓冲。**由 provider 复用**，
     *                    调用方必须在下一帧之前取走需要的数据，不要长期持有引用。
     * @param metric      true  = 已经是米制；
     *                    false = 模型原始 / 相对尺度（由 native 侧用 VINS
     *                            稀疏三角化深度做鲁棒标定后再用）。
     * @param timestampNs 对应的相机帧时间戳（不是推理完成时刻！）。
     *
     * @param normA / normB / normVersion
     *        V0.13.4 尺度漂移修复：**这一帧深度所用归一化映射的元数据**。
     *
     *        非米制 provider 输出的 d 是网络原始值 q 的仿射：
     *            `d = normA * q + normB`
     *        而 `normA/normB` 由会话级 min/max 决定，**会随扫描推进变化**
     *        （看到更近/更远的表面时范围扩张）。native 侧的标定拟合的是
     *        `z = a*d + b`，一旦 d 的含义悄悄变了而 (a,b) 还冻结着，几何
     *        就会整体膨胀或收缩 —— 这就是「同一物体扫到后面越来越大」的
     *        根因。
     *
     *        所以每一帧都必须把它所用的映射带出去，`normVersion` 每次映射
     *        变化 +1；调用方发现版本号变了，要在**喂这一帧之前**用
     *        `nativeSetDepthNormMapping` 让 native 显式重参数化标定，
     *        而不是让标定继续用旧映射去解释新数值。
     *        米制 provider（硬件深度）保持 A=1、B=0 不变即可。
     */
    data class Result(
        val depth: FloatArray,
        val width: Int,
        val height: Int,
        val confidence: FloatArray?,
        val timestampNs: Long,
        val metric: Boolean,
        val backend: String,
        val normA: Float = 1f,
        val normB: Float = 0f,
        val normVersion: Long = 0L
    )

    /** 后端名，进诊断报告与 HUD。 */
    val backendName: String

    /** 当前是否真的能出深度（模型缺失 / 设备不支持时为 false）。 */
    val available: Boolean

    /**
     * 提交一帧 YUV_420_888 相机帧。
     * @return 本帧是否产出了可用深度（产出了就用 [latest] 取）。
     */
    fun submitFrame(
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        width: Int,
        height: Int,
        rowStride: Int,
        uRowStride: Int,
        uPixelStride: Int,
        timestampNs: Long
    ): Boolean

    /**
     * 提交一帧硬件深度（DEPTH16）。只有 [HardwareDepthProvider] 会用到；
     * 单目实现直接返回 false。
     */
    fun submitDepthImage(image: Image, timestampNs: Long): Boolean = false

    /** 取最近一帧结果；没有则 null。 */
    fun latest(): Result?

    /** 清空历史（新会话）。 */
    fun reset() {
    }

    fun close()
}
