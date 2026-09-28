# MobileScan3D V0.13.15 近距离扫描覆盖包

适用基线：`tianlangxing-zhou/MobileScan3D_V0.4_HQ_RECON`，提交 `98e15aecc835648422b54366f4b28fb1ef8f0ade`。
版本：`0.13.15-near-scan`，versionCode `144`。这是源码增量覆盖包，不是完整工程，也不含 APK。

## 替换方法

1. 备份当前工程。把 ZIP 解压到临时目录。
2. 将解压后的 `app`、`tools`、`docs` **合并复制到工程根目录**，同名文件覆盖。根目录是含 `settings.gradle.kts` 的目录。不要删除整个原目录再替换。
3. 同时复制 `README_REPLACE_ZH.md`、`OVERLAY_MANIFEST.json` 和 `review_01315.patch`。patch 仅供审查；覆盖文件后不要再次应用 patch。
4. 保留原来的 `local.properties`、SDK/NDK/Ceres/OpenCV 配置、签名、模型与第三方库。若你的工程有本地改动或已超出上述基线，先对照 patch 合并。
5. 在工程根目录运行 `python tools/review_tests/verify_overlay.py`，应全部通过。
6. Android Studio 执行 Clean/Rebuild；或 Windows 执行 `gradlew.bat clean :app:assembleDebug`。Kotlin 与 JNI 接口同时更新，不能复用旧版 `.so`。

## 新功能入口

右上角齿轮 → **扫描距离与补光**：

- 最大扫描距离：**0.2–5.0 米**，步进 0.1 米，默认 **1.0 米**。这是相机到表面的估计直线距离，不是到世界坐标原点的距离。
- 暗光自动持续补光：默认开启。暗光持续约 0.9 秒后开灯；开启后保持到停止扫描或退出前台，避免灯自身造成反复开关。HQ 拍摄也保持相同补光，切换后等待约 1.2 秒和 3A 重新稳定。
- 设置保存后用于下一次扫描；正在扫描时不会改变范围。已保存的旧模型不会被新距离设置裁剪。

范围过滤在写入场景/目标 TSDF、累计点云、当前帧目标深度层与双目深度入口执行，导出 GLB/PLY 使用过滤后的几何。单目米制距离依赖 VINS 和深度标定，界面明确标为估计值；不能当作测距仪的精确边界。没有闪光灯的镜头会显示“无闪光灯”。

## 自检情况

已通过宿主端回归、C++ 语法检查、Kotlin 宿主类型检查及覆盖文件校验。**未完成完整 Android APK 构建和真机验证，不能据此承诺软件完全无问题或任意情况下永不停顿。** 跟踪不可信、目标离屏、尺度确实异常时仍会暂停融合并提示，避免把错误数据写入模型。

完整问题清单、测试范围、真机验收步骤见 `docs/review_01315/REVIEW_ZH.md`。
