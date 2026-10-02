# vc18402 缺失功能收口增量包

基线：
- versionCode 18401
- versionName 0.15.4-model-cleanup-camera-fix

应用：直接在工程根目录解压并覆盖同名文件。

结果：
- versionCode 18402
- versionName 0.15.4-consumer-gap-fix

本包只补缺失/未收口功能，不重复打包 vc183.1–vc183.3。

验证：
- Android XML 全量解析通过
- MainActivity / Renderer / ExportManager / FileProvider 静态接线通过
- 未在当前快照环境执行完整 APK/AAB 构建
