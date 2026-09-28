# MobileScan3D V0.13.17 模型方向修复覆盖包

适用基线：GitHub 仓库 `tianlangxing-zhou/MobileScan3D_V0.4_HQ_RECON`，提交
`abaa641616e6b5398d91426510182805ae0623c7`（V0.13.16 vendored OpenCV）。
安装后版本号：`0.13.17-orientation-fix`，versionCode 145。

## 替换方式

1. 备份当前工程；如果本地在上述基线后另有修改，先对照包内补丁合并。
2. 解压 ZIP，将其中的 `app/`、`tools/`、`docs/` **合并复制到工程根目录，覆盖同名文件**。
   同时复制根目录的说明、`ORIENTATION_OVERLAY_MANIFEST.json` 和 `orientation_fix.patch`。
   不要删除原有目录后再复制：这是覆盖包，不是完整工程。
3. 必须复制两个新增源码：`ScanCoordinates.kt` 和 `export/coordinate_conventions.h`。
   不需要修改 CMake 源文件清单。
4. 在 Android Studio 执行 Sync、Clean Project、Rebuild Project，然后安装新 APK。
   保留现有 `local.properties`、Ceres、模型文件及仓库内 OpenCV 配置。
5. 如有 Python，可在工程根目录运行：
   `python tools/review_tests/verify_orientation_overlay.py`。
   它检查本覆盖包文件的 SHA-256，不会修改工程。

也可在干净基线上使用 `git apply --check orientation_fix.patch`，再执行
`git apply orientation_fix.patch`；与直接复制二选一，不能先覆盖后重复打补丁。

## 修复范围

- 预览：补齐 SurfaceTexture 的 GL 原点到 TextureView 屏幕原点转换，点选/拖框与点云、AR 网格共用修正映射。
- 查看模型：使用 VINS 的世界 +Z 向上，修正相机基向量的镜像问题。
- 自由摆放：绕世界 Z 轴旋转，向下偏移沿负 Z，防止拖动旋转时模型翻倒。
- GLB：普通顶点色和 HQ 纹理导出都增加 Z-up → Y-up 根节点旋转；AR 缓存继续使用原世界坐标。
- 修正“世界 +Z 向下”和“单个仿射系数为负就一定倒置”的误导性诊断注释。

上一轮距离过滤、自动补光及连续扫描修复随当前仓库基线保留。

## 自检结果与限制

- 已通过：144 组预览/点选正反映射、180 组查看/摆放姿态、5 种 GLB 输出、AR 资产保存恢复、近距离 TSDF 回归。
- 已通过：新增 Kotlin 坐标模块实际编译执行；C++ 导出和近距离测试在 ASan/UBSan 下运行。
- 尚未验证：完整 Android APK 构建、手机 GPU/相机 HAL 实际显示、真机扫描稳定性。
  本包是源码修复包，不包含已装机验收的 APK。详见 `docs/review_01317/REVIEW_ZH.md`。

## 建议的一次装机验收

用一个上下明显不同、左右也不对称的物体（顶部贴红色箭头，右侧贴蓝点）：

1. 启动全新扫描，分别点选/拖框物体上部与下部，确认目标框落点一致。
2. 检查绿色当前帧点、累计点、AR 网格是否与实物同向；缓慢移动手机观察。
3. 停止扫描，进入“查看模型”：箭头朝上，蓝点没有镜像；旋转查看和长按平移正常。
4. 使用“摆放”，左右拖动应绕竖直轴旋转，顶部持续朝上；切回“原位”后对齐。
5. 导出普通 GLB 和 HQ 纹理 GLB，用支持 glTF 节点变换的查看器打开，对比几何和纹理方向。

先测试新扫描。旧 GLB 文件不会自动改写；已有错误目标选择/融合产生的几何也不会因更新自动修复。
旧扫描包的 AR 缓存仍可读取，但外部旧 GLB 要重新导出才能获得新坐标变换。
