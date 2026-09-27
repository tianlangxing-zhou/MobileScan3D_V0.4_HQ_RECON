# MobileScan3D 检查结果与 AR 模型实施方案

检查基线：`53bb36d5714a35760af0f568f716996820adc88b`，分支 `main`。
实际 Gradle 版本：`0.13.0-target-identity-sticky-fusion` / versionCode 130。
范围：本次重点审查扫描结束、深度标定、网格/纹理导出、AR 包保存恢复路径；未审计全部第三方源码。修改在本地，未推送 GitHub。

## 1. 结论

保留 Android Camera2 + IMU + VINS 的无 ARCore 主线。工程已经有 signed TSDF、Marching Tetrahedra、网格清理、UV 展开、多视角纹理、GLB、GLES AR 渲染和 ORB/PnP 跨会话重定位，无需重新堆一套扫描架构。

当下应先做稳定闭环，再改善输入深度和相机几何。提高体素密度、面数或纹理分辨率不会自动提高测量精度。当前源码标记“实验扫描（非测量）”是合理的。

现有闭环：

Camera2/IMU → VINS 位姿 + 相对深度/立体锚点 → 标定与 Fusion Epoch → 目标 TSDF → 三角网格 → UV/纹理 → GLB → App 内 AR。

跨会话需要额外保存 `ar_asset.msar` 和 `vins_map.vmap`。GLB 是通用模型；单独复制 GLB 不会包含恢复到原物理位置所需的环境地图。

## 2. 本次补丁已实现

| 文件/路径 | 问题 | 改动 |
|---|---|---|
| `cpp/export/gltf_exporter.cpp` | 有颜色、无法线时 COLOR_0 固定引用 accessor 2，实际上是索引 accessor | 按实际属性组合生成颜色 accessor 编号 |
| 两个 GLB exporter | 越界索引被改成 0 或直接写入；NaN/Inf 和残缺三角形未可靠拒绝 | 写文件前验证索引、属性长度及有限数值，拒绝坏网格 |
| 两个 GLB exporter | 包围盒序列化精度不足；全局 locale 可影响 JSON | 使用 classic locale 与 float max_digits10；写盘后 flush 检查 |
| `cpp/depth_calib.cpp` | 拟合按 min(d,z) 读取，但评分按 d.size 读取，长度不一致可越界 | 统一有效长度，并过滤非法深度样本 |
| `cpp/native_engine.cpp` | 烘焙快照之外读取共享 meshQuality/voxelSize | 在同一锁内复制配置，与网格快照配套 |
| `ExportManager.kt` | 纹理失败可能遗留上次 native 纹理；直接覆盖正式 GLB | 导出前清理旧资产；先写 pending 文件，成功后重命名发布 |
| `MainActivity.kt` | 旧缓存仍可能被当成本次 HQ 纹理显示 | 先清渲染纹理，仅在本次 textured=true 时加载 |
| `MainActivity.kt` | PLY 导出同步执行；恢复时在主线程扫描/哈希所有包 | PLY 放到导出队列；包查找/校验/加载移到后台 |
| 导出与主界面 | 重建/保存期间可以启动新扫描或切镜头 | 添加操作门控，包保存完成后才释放；扫描时禁止最终导出及切镜头 |
| `ScanPackageManager.kt` | 同名包先删除旧包，rename 失败后复制到可见目录 | 每次保存独立 UUID 包，发布前校验；只用目录 rename 发布，失败保留旧包 |
| `ScanPackageManager.kt` | manifest 文件名可指向包外位置 | 固定 v7 文件名，并验证规范路径和非空文件 |
| `MainActivity.kt` | depth provider 会话 min/max 缺少新扫描 reset 调用 | 在深度处理队列中重置，避免上次扫描范围污染新扫描 |
| 导出菜单 | 模型只存 App 私有外部目录，不便拿走 | 新增“保存 GLB 到文件…”，使用系统文件选择器、后台复制；支持当前生成和恢复后的模型 |

注意：上述 UI 操作门控不等于完整的 Activity 生命周期隔离。进程退出、销毁、原生长任务取消与新 Activity 初始化之间的竞态仍需要下一阶段统一任务管理。此补丁也没有改变单目深度模型的输出语义，没有承诺解决所有漂移。

## 3. 应用方法

在你的工程根目录执行，先保留未提交工作。补丁按上述基线生成，若你已有新修改，应审查冲突，勿强制覆盖。

```bash
git switch -c fix/scan-ar-export
git apply --check /path/to/mobilescan3d-ar-fixes.patch
git apply /path/to/mobilescan3d-ar-fixes.patch
```

Windows 的 `/path/to/` 替换为实际路径。ZIP 同时包含变更后的源码和测试，正常应用只需 patch。

`local.properties` 需要保持你原本可编译的路径：

```properties
sdk.dir=/your/android-sdk
ceres.sourceDir=/your/ceres-source
ceres.buildDir=/your/ceres-android-arm64-build
opencv.sdkDir=/your/OpenCV-android-sdk
```

Gradle 读取 `ceres.buildDir/lib/libceres.a`；OpenCV 路径应包含 `sdk/native/jni/include` 和 `sdk/native/libs/arm64-v8a`。本工程配置 JDK 17、compileSdk 36、NDK 27.1.12297006。

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Windows 用 `gradlew.bat :app:assembleDebug`。本次未提升 versionCode，也未生成 APK。

## 4. 手机上的验收流程

1. 固定同一主摄，选有纹理、静止、非透明反光物体；保留有纹理背景。进行有平移的缓慢移动，使 VINS 初始化。
2. 开始扫描，锁定目标，绕拍正面、两侧、背面和斜上方。先验证一个物体，不同时更改多项算法参数。
3. 停止扫描。等待后台网格、纹理、GLB 和持久化结果；失败时区分“没有网格”“仅顶点颜色”“地图不足”。模型导出成功不等于跨会话定位成功。
4. 在“导出 → 保存 GLB 到文件…”选择目录；在独立 glTF 查看器或 Blender 打开，检查三角面、贴图、法线、尺寸和方向。此次修复不自动转换 VINS 世界坐标到标准摆放坐标。
5. 退出再打开 App，回到同一场景点击“恢复AR”；未定位时模型应隐藏，定位成功后才出现。换场景不能把恢复原位和自由摆放混为一谈。
6. 故意测试快速重复点击、保存中再次开始、保存中切镜头、取消文件选择、磁盘不足、纹理帧不足、坏扫描包。旧模型文件/旧扫描包不应因失败被删除。
7. 对进后台、销毁重建、系统回收单独验收；这些场景未在本环境实测。

现有重定位代码的基本接受条件为至少 16 内点、内点率至少 0.35、中位重投影误差不超过 3 px，以及连续两次一致解。它们是算法门槛，不等同于物理定位误差保证。

## 5. 下一轮必须优先做的优化

### P0-A：统一深度数值域与 Fusion Epoch

具体冲突：`MonoDepthProvider.kt` 会持续更新 globalMin/globalMax，把网络输出映射到 0.4–6；`native_engine.cpp` 的 epochCalib 却冻结。归一化范围改变后，同一个网络输出变成不同的 d，冻结的标定映射不再等价。重置 provider 只解决跨扫描污染，不能解决扫描内部变化。

建议实施：

- 给深度帧增加明确语义：RELATIVE_DEPTH / RELATIVE_INVERSE_DEPTH / METRIC_Z，以及模型版本、归一化参数、时间戳。
- 首选保留网络原始输出，按模型契约选择 `z=a*q+b` 或 `1/z=a*q+b`；同时修改所有假定 raw>0 的输入门控。不能只删掉 min/max 两行。
- 如果短期保持正数归一化，定义 `d=A*q+B`；在 epoch 中一起冻结 A、B，后续输出继续使用该映射。范围扩展不得静默更换映射后继续融合。
- 若必须更换 A、B，就携带元数据显式重参数化。旧标定 `f=a_old*d_old+b_old`，新输入 `d_new=A_new*q+B_new` 时，令 `a_new=a_old*A_old/A_new`、`b_new=b_old+a_old*B_old-a_new*B_new`；线性深度和逆深度两种 f 都适用。保持有效数值检查。
- 对固定物体/固定姿态录制序列回放，改变背景近远和亮暗，检查输入域变化是否造成几何膨胀。

完成标准：相同录制数据可重复得到相近包围盒；每次尺度变化可追溯到标定版本。真实尺寸误差需要实物标尺验证，不从“4mm 体素”推断“4mm 精度”。

### P0-B：相机、IMU、深度与纹理严格同步

检查 `Image.timestamp`、IMU timestamp、CaptureResult 和双摄 timestampSource。Android 相机 REALTIME 与 elapsedRealtimeNanos 同一基准；UNKNOWN 不保证可直接比较，必须明确标定/拒绝未验证组合。

记录每个融合帧的图像时间、位姿时间差、深度时间差、镜头 ID、内参版本、裁剪/旋转、焦距。RGB、深度和 JPEG 必须使用对应分辨率内参；双摄深度需真实基线与外参。用离线录制测重投影误差后再调整阈值。

### P0-C：把扫描会话从 Activity 中分离

新增 `ScanSessionController`，所有任务携带不可复用的 sessionToken。把扫描、最终重建、保存、恢复、销毁排入受控任务体系。native 开始任务时捕获 token，发布 mesh/AR cache 前再次校验，旧任务只销毁局部结果。停止扫描要等待已提交融合帧及 HQ JPEG 注册完成后再制作最终快照。

不能只用 `shutdownNow()`：它不会可靠中断执行中的 JNI/C++ 烘焙。进程级 native 会话的 destroy/create 必须与后台任务生命周期配套。

### P1-A：建立模型质量分级

把成功拆成 geometryReady、textureReady、persistentArReady 三项。已有 GLB 成功但地图不足时保留可分享模型。纹理不能以“编码出 JPEG”作为 HQ 的唯一标准；结合 `hqTexels / paintedTexels`、有真实纹理支持的三角面占比、有效视角数、视角覆盖判断。不要直接用整张 atlas 的覆盖率做门槛，UV 留白会误导。

保存 JSON 质量报告：三角面数、连通分量、边界洞、包围盒、标定残差、纹理支持比例、VINS 状态、重定位点数。补洞产生的表面标为推断，避免当成实测。

### P1-B：减少停扫和 AR 卡顿

`nativeBuildMesh` 仍持 gStateMutex 做耗时重建，后台线程不能消除对相机/渲染的锁阻塞。下一步在受控时刻复制 TSDF 必需块和元数据，锁外做提面/QEM；测量复制内存峰值再决定增量方案。2K atlas 保持默认，按设备内存预算选面数，先测 P50/P95 耗时和峰值内存再启用 4K。

### P1-C：统一外部模型坐标

glTF 使用米制、右手坐标、+Y 向上。当前 exporter 原样输出内部坐标，没有独立 model-to-world 变换。验证 VINS/目标模型坐标后，在导出副本中做 Y-up 转换、底部落地、水平居中，并在 manifest 保存双向变换。内部 saved-world AR cache 和视觉地图保持同一坐标，避免为修复外部朝向而破坏原位恢复。

## 6. 扩展功能顺序

| 顺序 | 功能 | 实现方向 | 验收 |
|---|---|---|---|
| 1 | 模型列表与质量报告 | 扫描包缩略图、名称、尺寸、三项质量状态、保存/删除 | 不再只恢复最新包；不在 UI 全量哈希 |
| 2 | 自由放置 AR | 增加独立 model-to-world 变换、平面检测/射线拾取、旋转/缩放；保留“原位恢复”模式 | 新地点也可放模型，且不修改旧环境地图 |
| 3 | 扫描覆盖提示 | 基于相机视角、可见三角面及纹理质量显示未覆盖区域 | 指导补拍，不只是计数照片 |
| 4 | 桌面/云端高质量重建 | 保存 RGB、内参、时间戳、位姿、可选深度和 mask，异步重建并返回模型 | 同一输入与手机结果可比较；不阻塞扫描 |
| 5 | Web/跨平台分享 | 标准 GLB + 可选 USDZ、网页预览；外部 Android AR 可另接 Scene Viewer | 支持设备进入 AR，其余设备退回 3D 预览 |

无 ARCore 是当前工程选择。Google Scene Viewer 的 AR 模式有其设备/服务要求，不能承诺给所有无 ARCore 手机提供通用 AR；继续使用自研 VINS/GLES 才能保持原路线。暂缓多地图合并、3D Gaussian Splatting、全套 Vulkan 重写，直到数据质量和导出稳定。

建议将工作分为三次交付：第一批合入本补丁并真机验收；第二批做深度域、时间同步、会话生命周期；第三批做模型列表、坐标规范化和自由摆放。第二批完成前不要宣传测量精度。

## 7. 验证记录与限制

已运行：`python3 tools/review_tests/run.py`。

通过：无属性、仅法线、仅颜色、法线+颜色、纹理五类 GLB；独立 Python 解析 GLB header/chunks、accessor 类型、包围盒精度、索引、内嵌 JPEG；坏索引、残缺面、NaN/Inf、残缺颜色拒绝；非英语小数分隔符 locale；深度数组长度不等。

C++ 测试启用 AddressSanitizer 和 UndefinedBehaviorSanitizer，测试通过。沙箱不能读取 /proc 线程列表，关闭了 LeakSanitizer；未进行泄漏验证。没有运行 Khronos 官方 validator，这些测试不能替代完整规范合规性检查。

尝试 `:app:assembleDebug --offline`，Gradle wrapper 仍需下载 8.13，因网络不可达失败，未进入 Android/Kotlin 编译。环境也没有工程所需的 local.properties/Ceres/OpenCV 配置。无 APK、无手机、无扫描数据，因此不能验证实际模型精度、纹理覆盖、AR 漂移或整包性能。代码补丁必须在你的原构建环境和手机上按第 4 节验收。

官方参考：

- glTF 2.0：https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- CameraCharacteristics / timestamp source：https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics
- Scene Viewer：https://developers.google.com/ar/develop/scene-viewer
