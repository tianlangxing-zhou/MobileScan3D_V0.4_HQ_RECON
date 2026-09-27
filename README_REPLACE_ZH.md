# MobileScan3D V0.13.6-review 覆盖包

适用基线：GitHub main `2cafb5382e21eec4795255309c413d14efadae5b`（V0.13.5）。

这是增量覆盖包，不是完整工程，不含 APK。完整修改清单和测试边界见 `docs/review_0136/REVIEW_ZH.md`。

1. 先备份你当前的工程。
2. 解压 ZIP，把其中的 `app`、`docs`、`tools` **合并复制**到原工程根目录（该目录包含 `settings.gradle.kts`），同名文件选择覆盖。不要删除原目录再复制；未包含的模型、第三方库和原文件需要保留。
3. 同时复制根目录的 `OVERLAY_MANIFEST.json` 和本说明。`review_0136.patch` 是审查用差异，已覆盖文件后不要重复应用 patch。
4. 在工程根目录运行 `python tools/review_tests/verify_overlay.py`，确认交付文件的 SHA-256 都匹配。
5. 保留你原来的 `local.properties`、签名、模型和 Ceres/OpenCV/SDK/NDK 路径配置。在 Android Studio 中 Clean/Rebuild，或在 Windows 运行 `gradlew.bat clean :app:assembleDebug`。
6. 请先做一次短扫描和停扫导出验证，再进行重要采集。若当前工程已经超出上述基线或有自己的改动，请先对比 `review_0136.patch`，避免直接覆盖你的修改。

已执行的宿主回归与语法检查通过；**完整 Android 构建和真机效果尚未验证**。包内没有把这些未验证项目记为通过。
