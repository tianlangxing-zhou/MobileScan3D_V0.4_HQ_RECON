# MobileScan3D 项目长期记忆

## 构建 / 调试环境
- **构建必须用本地 Gradle**：`E:/AndroidDev/gradle-8.13/bin/gradle` + `JAVA_HOME=E:/AndroidDev/jdk-17.0.20.1+1`
  + `ANDROID_SDK_ROOT=E:/AndroidDev/sdk` + `GRADLE_USER_HOME=E:/AndroidDev/gradle-home`。
  用 `./gradlew` 会去 services.gradle.org 下载分发包，直连必超时。
- adb：`E:/AndroidDev/sdk/platform-tools/adb.exe`（不在 PATH），真机 serial `3B15AP02BZD00000`
- 推 GitHub：~~旧环境代理 `https_proxy=http://127.0.0.1:52098`~~ **已于 2026-09-27 失效**（端口无监听）。当前**直连 github.com 返回 200，可直接 `git push origin main`，无需代理**。`mirror` 远程 `ghfast.top` 仅 fetch 可达、push 需 auth 不可用。若将来代理恢复再切回。
- 构建需 `ANDROID_NDK_HOME`（如 `E:/AndroidDev/Sdk/ndk/26.1.10909125`）+ `dangerouslyDisableSandbox`，否则 CMake/NDK 步骤拿不到工具链。
- 真机脚本在 `E:\MobileScan3D\devtest\`（device_driver.py + run*.py）

## 测试方法的硬约束
VINS **必须靠视差运动**才能初始化。脚本自动跑时手机静止 → VINS 永不初始化，
「地图 0 · 绘制 0」是测试条件的产物，不是崩溃 bug。
要验证「真能重建出 3D 场景」，必须**手持手机缓慢平移**复测。
判定重建链路是否真活的依据：HUD 的地图/绘制计数从 0 开始增长 + banner 不再显示初始化中。

## 已确立的架构约定
- 日志洪水已治理：ROS_DEBUG 空实现 + Ceres `-DMAX_LOG_LEVEL=-1`，
  正常量级约 1 万行/2min；若再暴涨说明有新的热循环打印。
- `native create` 每轮出现 2 次属预期：冷启动 1 次 + `startScan()` 主动重建 1 次。
  真正要防的是 Activity 重建导致的额外 create（已用 `android:configChanges` 解决）。
- 场景点云 `g` 与 TSDF 是两套容器：TSDF 出网格 ≠ 场景点云非空。
  `nativeGetPointCount()` 在锁定过目标后只报 `targetG.count()`。

## 协作偏好
- 用户偏好：先严谨分析数据定位根因，再改代码；不接受靠外部网页 AI 转述。
  浏览器自动化（gptcat）方案已弃用。
- 每个修复都要真机回归验证并在 commit message 里写清「验证数据」。
