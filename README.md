# MobileScan3D V0.4 HQ Reconstruction

无 ARCore、纯 Android 本地 3D 重建工程继续版。

## V0.4 核心

Camera2 + IMU → VIO → Keyframes → AI/MVS Depth → Depth Fusion → Sparse TSDF → Gaussian → PLY

### 重点
- 不依赖 ARCore。
- 支持外部端侧深度模型通过 JNI 注入米制 depth。
- 首次加入真正的 sparse voxel hash TSDF，并可导出 PLY。
- 保留 Gaussian 增量表示。
- 为 LiteRT/NPU/GPU 深度模型和 Vulkan compute 做好接口。

### 重要诚实说明
V0.4 不是“已经训练好的手机 AI 扫描器”。模型权重尚未捆绑；如果没有真实 AI/MVS depth，程序仍使用保守 fallback 仅验证完整数据链路。不要把 fallback 点云当成最终精度。

## 构建

Android Studio + NDK + CMake。Android 官方文档建议 Vulkan 项目运行时检查实际 Vulkan 版本/驱动能力；NDK 25+ 含 Vulkan 1.3 headers。

## 设备方向

首要目标仍是 ARM64 旗舰 Android。建议优先 Vulkan 1.3 + hardware compute；随后再扩大兼容范围。
