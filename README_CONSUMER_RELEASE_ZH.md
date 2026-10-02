# MobileScan3D 消费级发布审查（vc182 / 0.14.2-consumer-ux2）

本文件记录基于 `vc179 / 0.13.39-ui-focus` 源码快照完成的消费级发布收敛。目标是降低首次使用失败率、明确本地数据处理方式，并把不适合直接上架的研发态问题显式化。


## vc182 本轮 UI 合并

- **物体追踪提升为一级常用功能**：主扫描画面右侧工具栏直接提供“追踪”，一键开启/关闭，并显示“关闭 / 选目标 / 追踪中”；“更多”菜单不再重复放置该功能。
- **3D 模型查看退出修复**：顶栏左侧改为明确的“退出”文字按钮；GLSurfaceView 不再使用 `setZOrderOnTop(true)`，避免独立 Surface 把 Android 控件盖住。
- **双重退出路径**：3D 模型查看同时支持顶栏“退出”和 Android 返回键/返回手势。
- **帮助文案同步**：界面帮助已更新为“物体追踪位于右侧一级工具栏”，避免文案仍提示去二级菜单。
- 本轮继续保留 vc180 的首次权限说明、隐私入口、设备过滤、存储预检、release hardening、CI 路径支持等全部修改。

## 本次已落地

1. **首次权限说明**：首次请求相机前先说明用途、本地处理方式和撤回权限路径；拒绝后仍保留“重新授权 / 系统设置”入口。
2. **隐私入口**：`更多 → 高级与帮助 → 隐私与数据` 可随时查看当前数据处理说明。
3. **设备兼容过滤**：Manifest 明确要求加速度计和陀螺仪。当前 VIO 预检本来就把缺失任一传感器判为不可扫描，安装阶段过滤比安装后失败更符合消费级体验。
4. **存储预检**：低于 256MB 阻止新扫描；低于 1GB 给出长时间/HQ 扫描警告。
5. **发布配置收敛**：应用名统一为 `MobileScan3D`；禁止明文网络流量；release 显式关闭 Java/JNI 调试；release lint 失败会阻断构建。
6. **可复现构建入口**：Ceres/OpenCV 路径同时支持 `local.properties` 与环境变量 `CERES_SOURCE_DIR`、`CERES_BUILD_DIR`、`OPENCV_ANDROID_SDK`，便于 CI/发布机使用。
7. **仓库卫生**：移除过期的 `MainActivity.kt.bak_20260914`，补充 `.gitignore`，避免日志、采集数据、APK/AAB、本机配置和备份文件进入发布源码包。

## P0：正式商业发布前必须确认

### 1. VINS-Mono / CamOdoCal 许可证

当前 native 构建直接编译 `app/src/main/cpp/vins/`，结构和文件名与 VINS-Mono 一致；CMake 还直接编译 `third_party/camodocal/src/gpl/gpl.cc`。上游 VINS-Mono README 标注 GPLv3，并明确商业使用需联系作者；因此在闭源商业发行前，必须由法务/负责人确认：

- 这些文件的准确来源、版本和许可证；
- 你的发行方式是否满足对应开源义务；或
- 是否已获得适合商业发行的额外许可；或
- 是否需要换成许可证兼容的 VIO/相机模型实现。

**不要仅靠删掉许可证文件或改目录名解决。** 许可证义务取决于实际代码来源和使用方式。此处是工程发布风险提示，不是法律意见。

### 2. 完整发布包验证

本 ZIP 是“核心算法 + UI 源码快照”，原快照明确不包含 `third_party`、约 65MB 模型 assets、JNI 运行库、Gradle wrapper 和 `local.properties`，因此不能在本环境做完整 APK/AAB 编译和真机回归。正式发布必须回到完整仓库完成：

- `assembleRelease` / `bundleRelease`；
- Android Lint release；
- ARM64 真机冷启动、授权拒绝/再次授权、后台恢复；
- 至少 3 类设备：旗舰 Snapdragon、旗舰 Dimensity、无硬件深度机型；
- 5/10/20 分钟连续扫描，检查内存、热降载、TSDF checkpoint、GLB 导出；
- 断电/杀进程恢复；
- 不同厂商相机 HAL 下的 Camera2 / multi-camera 退化路径。

## P1：建议下一轮继续做

- 把 `MainActivity.kt`（约 300KB）拆分为权限/相机/扫描状态机/模型查看/设置等模块，降低回归风险。
- 把布局与 Kotlin 中大量硬编码中文文案迁到资源文件，为国际化、A/B 文案和无障碍做准备。
- 增加最小化的匿名本地质量事件结构（不联网也可先落 JSON），例如：扫描开始失败原因、模型生成失败阶段、设备档位、耗时、热状态；用户主动导出诊断时再分享。
- 在完整仓库接入 CI：release 编译、lint、C++ 单测、关键 Kotlin 纯逻辑单测、APK/AAB 体积门禁。
- 正式上架前准备隐私政策、第三方开源声明、用户支持入口、版本更新说明与兼容设备说明。

## 构建路径配置

本次改动后可使用任一方式：

```properties
# local.properties
ceres.sourceDir=/absolute/path/to/ceres-solver
ceres.buildDir=/absolute/path/to/ceres-android-build
opencv.sdkDir=/absolute/path/to/OpenCV-android-sdk   # 可选
```

或 CI 环境变量：

```bash
export CERES_SOURCE_DIR=/absolute/path/to/ceres-solver
export CERES_BUILD_DIR=/absolute/path/to/ceres-android-build
export OPENCV_ANDROID_SDK=/absolute/path/to/OpenCV-android-sdk  # 可选
```

如果完整仓库保留 vendored OpenCV 4.12，`OPENCV_ANDROID_SDK` 可以不设置。
