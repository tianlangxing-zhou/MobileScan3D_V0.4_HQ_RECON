package com.mobilescan3d.depth

import android.media.Image

/**
 * 深度表示的固定语义（评审 P0-1）。
 *
 * 关键：**一旦模型确定，表示就固定，运行时不要再猜 linear 还是 inverse**。
 * - [RELATIVE_DEPTH]：模型输出 raw `q` 与“真实远近”单调递增（大模型越大=越近
 *   或越大=越远，由标定 affine 吸收符号），native 侧用 VINS 稀疏三角化 + stereo
 *   anchors 拟合 `metric = scale*q + shift` 得到米制。
 * - [INVERSE_DEPTH]：模型输出 raw `q` 是视差/逆深度，native 用 `metric =
 *   scale/q + shift`。
 *
 * 选择哪种由 `depth_model.tflite` 的真实输出定义**固定一次**；不要像旧实现那样
 * 用会话级 min/max 把 q 重映射到 [0.4,6]——那会让同一 q 在不同扫描阶段对应不同
 * 米制，正是尺度漂移的根因。
 */
enum class DepthRepresentation {
    RELATIVE_DEPTH,
    INVERSE_DEPTH
}

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
     * @param representation 深度固定表示语义（RELATIVE_DEPTH / INVERSE_DEPTH）。
     *                    **固定一次**，运行时绝不根据单帧内容重新猜测 linear
     *                    还是 inverse。native 侧用 VINS 稀疏三角化 +
     *                    stereo anchors 拟合 `metric = scale*q + shift`
     *                    （逆深度则是 `scale/q + shift`）得到米制。
     */
    data class Result(
        val depth: FloatArray,
        val width: Int,
        val height: Int,
        val confidence: FloatArray?,
        val timestampNs: Long,
        val metric: Boolean,
        val backend: String,
        val representation: DepthRepresentation = DepthRepresentation.RELATIVE_DEPTH
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
