# V0.13.17 方向问题审查与验证

基线：`abaa641616e6b5398d91426510182805ae0623c7`。
本次读取了最新仓库源码及仓库内历史扫描日志；没有读取上一轮对话中的 ZIP。
仓库日志不是本次修复后的装机证据。用户所述新测试没有附新的截图/日志，
因此以下结论区分“源码中已确认的缺陷”与“仍需真机确认的整体效果”。

## 已确认的缺陷

### 1. 预览坐标少一次输入原点转换

`MainActivity.onSurfaceTextureUpdated()` 原样抽取 SurfaceTexture 矩阵的 2D 部分，
用于 `viewToCameraNorm()`，再求逆给点云/网格渲染器。
但 View/touch 的归一化坐标从左上开始，SurfaceTexture 矩阵接受的是 GL 左下原点坐标。
旧算法将输入 `(u,v)` 直接送入 M；修复为 `M * (u,1-v,0,1)`。

对于矩阵元素 f（列主序），修正的 2x3 仿射为：

```
[ f0, -f4, f12+f4 ]
[ f1, -f5, f13+f5 ]
```

转换放在输入端，保留 HAL 旋转、镜像、裁剪；不在 shader 输出端盲目取负。
该仿射同时供点选、拖框、目标框投影与所有 AR 图层使用。
无旋转、无裁剪时，标准 ST 的纵向翻转因此被抵消，屏幕顶端正确对应图像顶端。

依据：AOSP `GLConsumer::computeTransformMatrix` 明确在最终矩阵左侧添加 FlipV，
将 GL 坐标转为 buffer 内存左上原点；TextureView 的布局/触摸仍是左上原点。

- https://developer.android.com/reference/android/graphics/SurfaceTexture#getTransformMatrix(float[])
- https://android.googlesource.com/platform/frameworks/native/+/refs/heads/main/libs/gui/GLConsumerUtils.cpp
- 实际核对源码镜像：https://github.com/LineageOS/android_frameworks_native/blob/lineage-23.0/libs/gui/GLConsumerUtils.cpp
- TextureView 绘制参考：https://github.com/aosp-mirror/platform_frameworks_base/blob/master/libs/hwui/pipeline/skia/LayerDrawable.cpp

### 2. 查看器世界轴和相机手性不一致

本工程 VINS 初始化 `Utility::g2R()` 将 g 对齐 +Z；`processIMU()` 使用
`R * (acc - bias) - g`，静止的加速度计比力方向代表物理向上，因此世界 +Z 向上。
相机仍使用原始图像 X 右、Y 下、Z 前。旧注释把 +Z 当成物理向下，这是错误的。

原查看器却绕 Y 轴旋转，并使用 `right = up × forward; down = right × forward`。
其基向量行列式为 -1，是反射而非合法相机旋转。
现在绕 +Z 轴构造相机，使用 `right = forward × up; down = forward × right`，
保持行列式 +1。普通与纹理模型复用这一查看位姿。

### 3. 自由摆放用了 Y-up 约定

原 `Ry` 会改变模型的物理高度方向；锚点沿 -Y 下移实际不是向下。
修复为 `Rz`、沿 -Z 下移，旋转仍以模型包围球中心为中心。

### 4. 外部 GLB 缺少坐标约定转换

两种导出器直接输出 VINS Z-up 网格，没有 glTF Y-up 转换。
现在二者共享一个根节点矩阵：`(x,y,z) -> (x,z,-y)`，即绕 X 轴 -90°。
这是行列式 +1 的旋转，法线随节点正确变换，不改变三角形绕序或 UV。

- glTF 坐标规范：https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#coordinate-system-and-units

导出器只写节点矩阵，不改重建网格内存和 `ArTexturedAsset` 二进制格式。
应用内 AR 恢复读取原始网格/专用缓存，不从外部 GLB 重新解析顶点，因此不会重复旋转。

## 实际通过的验证

| 检查 | 结果 |
|---|---|
| Kotlin `ScanCoordinates` 实际编译运行 | 通过（宿主 Kotlin 2.0.21 / Java 17；工程声明 Kotlin 2.2.20） |
| 4 个旋转方向 × 镜像开关 × 非对称裁剪开关 × 9 个点 | 144 组正反映射通过 |
| 完整 360° 轨道、5 个俯仰角 | 180 组正交、行列式 +1、顶部朝上、右侧无镜像检查通过 |
| 同组摆放与锚点检查 | 中心落在锚点、顶部保持竖直、向下偏移正确 |
| 4 种顶点属性 GLB + 1 种纹理 GLB | 文件结构、实际节点变换、顶部/法线朝上、右手坐标通过 |
| 纹理 AR 资产保存/恢复 | 顶点、法线、UV、索引、JPEG 一致，通过 |
| 原有异常输入/locale/深度数组回归 | 通过 |
| 原近距离 TSDF 策略回归 | 径向距离、单位缩放、无效输入、冻结证据、近处网格通过 |
| C++ AddressSanitizer / UndefinedBehaviorSanitizer | 未报告内存/未定义行为错误；容器限制下关闭 LeakSanitizer |

执行命令：

```
python3 tools/review_tests/run_orientation.py
python3 tools/review_tests/run.py
python3 tools/review_tests/run_scan_policy.py
```

依赖：Python 3、Pillow、g++；方向测试需要 Java 和 kotlinc，可通过 JAVA/KOTLINC 环境变量指定。
这些测试不等同于 Android APK 构建或 GPU 截图验证。

## 验证边界

本环境未配置 Android SDK/NDK/Ceres 构建链，也没有连接手机。
不能将 APK 构建、特定机型 HAL 映射、真实物体重建效果记为通过。
姿态时间差、跟踪质量与本次方向修复属于不同问题；没有用额外模型翻转去掩盖它们。
请按根目录说明进行一次新的非对称物体扫描验收。
