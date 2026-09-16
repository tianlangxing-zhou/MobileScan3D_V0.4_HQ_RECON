package com.mobilescan3d.depth

import android.content.Context
import android.content.res.AssetManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log

/**
 * 深度来源工厂：**硬件优先、模型兜底**。
 *
 * ## 为什么默认仍然走模型
 *
 * 评审把「接入真实 DepthProvider」列为 P0，但实机排查的结论是：
 * 深度其实**一直有**在产生并送进 native（`assets/depth_model.tflite` 是真
 * FlatBuffer，每帧走 `nativeOnDepthMap`），真正的缺陷是
 *   (a) 深度是「会话 min/max 映射到假米制」的相对尺度 —— 已由 native 侧
 *       的鲁棒标定（MAD + Huber IRLS + EMA + 时序一致性）修掉；
 *   (b) 深度来源没有抽象 —— 已由 [DepthProvider] 修掉。
 *
 * 硬件深度（[HardwareDepthProvider]）打开的是**另一颗相机**，与 RGB 主摄
 * 不同步、且外参未标定。把它设成默认值等于在一个已经能工作的链路上引入
 * 一个未经验证的变量。所以这里的策略是：
 *   - [probe] 永远可查，结果进诊断报告，用户能知道自己的设备有没有；
 *   - [create] 只有在调用方显式要求（`preferHardware = true`）时才用硬件。
 *
 * 这是一个**有意的保守默认**，不是没接。
 */
object DepthProviderFactory {

    private const val TAG = "DepthProviderFactory"

    /**
     * 设备深度能力探测结果。
     *
     * @param supported       是否存在任何一颗支持 `DEPTH_OUTPUT` 的相机
     * @param cameraId        与请求 lensFacing 匹配的那颗（可能是 null）
     * @param note            人类可读结论，直接进报告
     */
    data class Probe(
        val supported: Boolean,
        val cameraId: String?,
        val note: String
    ) {
        companion object {
            val UNKNOWN = Probe(false, null, "未探测")
        }
    }

    /**
     * 查 `REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT`。
     *
     * 注意「有 DEPTH_OUTPUT」和「这颗相机就是 RGB 主摄」是两件事：
     * 多数设备的深度输出挂在**单独的物理相机 id** 上，所以这里单独返回
     * `cameraId`，而不是假设它等于主摄 id。
     */
    fun probe(
        context: Context,
        lensFacing: Int = CameraCharacteristics.LENS_FACING_BACK
    ): Probe {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as? CameraManager
            ?: return Probe(false, null, "CameraManager 不可用")

        return try {
            var anyDepth = false
            var picked: String? = null
            var depthIds = StringBuilder()
            for (id in manager.cameraIdList) {
                val chars = try {
                    manager.getCameraCharacteristics(id)
                } catch (t: Throwable) {
                    continue
                }
                val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
                    ?: continue
                if (!caps.contains(
                        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT
                    )
                ) {
                    continue
                }
                anyDepth = true
                if (depthIds.isNotEmpty()) depthIds.append(',')
                depthIds.append(id)
                if (picked == null && chars.get(CameraCharacteristics.LENS_FACING) == lensFacing) {
                    picked = id
                }
            }
            when {
                !anyDepth ->
                    Probe(false, null, "本机没有 DEPTH_OUTPUT 相机（走单目模型）")
                picked == null ->
                    Probe(true, null, "有深度相机($depthIds) 但朝向与主摄不一致（走单目模型）")
                else ->
                    Probe(true, picked, "深度相机 id=$picked（可用，本版未启用）")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "probe failed", t)
            Probe(false, null, "探测异常: ${t.javaClass.simpleName}")
        }
    }

    /**
     * 建一个可用的深度来源。
     *
     * @param preferHardware 只有明确传 true 且设备真的支持时才会走硬件；
     *                       硬件路径启动失败会**自动回退**到单目模型，不会
     *                       把一个 `available == false` 的 provider 交出去。
     */
    fun create(
        context: Context,
        assets: AssetManager,
        preferHardware: Boolean = false,
        probe: Probe = Probe.UNKNOWN
    ): DepthProvider {
        if (preferHardware && probe.supported && probe.cameraId != null) {
            val hw = HardwareDepthProvider(context, probe.cameraId)
            if (hw.start()) {
                Log.i(TAG, "using ${hw.backendName}")
                return hw
            }
            Log.w(TAG, "hardware depth start failed: ${hw.errorText}; falling back to mono")
            hw.close()
        }
        val mono = MonoDepthProvider(assets)
        Log.i(TAG, "using ${mono.backendName} available=${mono.available}")
        return mono
    }
}
