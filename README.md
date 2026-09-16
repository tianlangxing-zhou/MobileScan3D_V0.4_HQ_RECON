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

## V0.6.1 + V0.7 Persistent AR（当前版本）

```text
versionCode = 70
versionName = 0.7.0-persistent-ar
```

- **V0.6.1**：APP 内真正渲染 HQ 纹理 AR（GLES2 + baked UV，纹理不可用时退回 vertex-color Mesh）。
- **V0.7**：跨会话视觉重定位 —— 保存 `saved VINS world XYZ + ORB descriptor`，
  重开 APP 后用 ORB + PnP-RANSAC 恢复 `savedWorld ← liveVinsWorld`，把模型钉回原扫描位置。

设计说明：`docs/V0.6.1_V0.7_PERSISTENT_AR.md`
落地记录（含编译修复与验证结果）：`docs/V0.7_IMPLEMENTATION.md`

### 构建

```bash
./gradlew :app:assembleDebug
```

产物：`app/build/outputs/apk/debug/app-debug.apk`（arm64-v8a debug，约 105 MB）。

> 注意：`ar_textured_asset.h` 必须使用 `#include "v06/uv_unwrap.h"`。
> 若写成不带路径的 `"uv_unwrap.h"`，NDK 编译会因 include 路径不含 `v06/` 而失败。

### 实机流程

1. 首次扫描 → 停止导出 → 自动生成 HQ textured GLB + AR cache + 持久地图（提示「跨会话 AR 已保存」）。
2. 重开 APP，回到原环境，点击「恢复AR」→ 缓慢平移找原扫描区域 → 提示「AR 重定位成功」后模型出现。
