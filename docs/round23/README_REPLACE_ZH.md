# 第 2＋3 轮替换包

基线：本次上传的 `MobileScan3D_core_src_vc153_perpixel_20261001.zip`。
这是增量源码覆盖包，仅包含本次修改或新增文件；不包含模型、SDK、编译缓存或 APK。

## 如何替换

1. 备份本地工程。本包不适用于其他分支或已经另行修改过的同名文件直接盲覆盖。
2. 解压 ZIP，将包内 `MobileScan3D/` 的内容合并到本地工程根目录，即包含 `app/`、`gradlew` 的目录。不要多套一层 MobileScan3D。
3. 可先运行包内 `tools/review_tests/verify_round23_overlay.py --before <本地工程根目录>`，检查将被覆盖的文件是否仍是上传基线或本补丁版本。新增文件缺失是正常情况；遇到其他哈希应先合并本地修改。
4. 覆盖后运行 `python tools/review_tests/verify_round23_overlay.py <本地工程根目录>` 检查替换完整性。
5. Android Studio 重新 Sync／Rebuild。沿用完整本地工程已有的 Ceres、Eigen、OpenCV、NDK 和模型配置，无新增第三方运行库。所有 Kotlin／JNI／CMake 文件必须一起覆盖。

## 第 2 轮完成内容

附件已有初版逐像素处理；本次在其基础上修正并补齐：

- 从 DepthProvider.Result.confidence 经新 JNI 接口传入逐像素来源置信度。单目模型未提供置信度时，采用时序与深度边缘启发式；原有 0.5 整帧系数仍为保守基准，不冒充网络预测的不确定度。
- 时序重投影使用最近表面 z-buffer，同一当前像素只统计一次，避免遍历顺序决定冲突。原来的整帧比例遍历合并到掩码遍历。
- 冲突像素停止写入，边缘适度降权；未知区域与遮挡／显露候选保持中性权重。遮挡判断是深度轮廓邻域启发式，并非完整多视图可见性求解。
- 掩码有效性限定于当前帧；参考深度需尺寸、内参、标定状态及时间间隔匹配。避免换分辨率、恢复扫描或参考不可用时复用旧掩码。
- 复用深度、标定深度、epoch 深度、投影深度、权重及范围过滤缓冲。每帧清理有效长度，保留容量；原生回调互斥保护复用缓冲，状态仍由 gStateMutex 保护。
- 原生记录 calib／temporal／weight／fuse／surfel／stereo／lock／total 的最近 120 次 P50/P95，每 30 次回调输出；提前退出也计入总耗时，区分 entered 与 done。日志标签 `PerfStage`。
- 单目推理记录 preprocess／inference／copy／total，日志标签 `DepthInferencePerf`。

这次没有改成无锁融合或体素投影融合；性能数字需要目标手机同一路径回放测量，不承诺固定提速百分比。减少融合写入也不会自动减少神经网络推理耗时。

## 第 3 轮完成内容

停止扫描后，打开“导出”菜单可选：

- **硬表面规整＋照片贴图 GLB**：面积均匀采样、RANSAC 平面检测、协方差最小特征向量拟合，规整近似平行／垂直平面；保留原有网格边界，不主动补面。若会造成面翻转或塌缩则退回原网格。
- **长方体拟合＋照片贴图 GLB**：从三个正交方向的平面证据估计姿态和尺寸，生成 24 个独立面顶点、12 个三角形的硬边箱体。
- **正方体拟合＋照片贴图 GLB**：在箱体约束上增加三轴等长约束，尺寸不匹配会拒绝，不强制把长方体拉成正方体。

箱体要求至少三个方向有充分面覆盖；表面支持率、残差、法线一致性、薄片退化和分散面覆盖均参与判定。曲面、凹凸明显或数据不足时返回扫描网格，并在结果信息中说明。它是单物体规则形状拟合，不是任意 CAD 特征识别；小于容差的细节仍可能被简化。

拟合成功后跳过后续 weld／平滑／小分量清理，防止直角被磨圆或只有两片三角形的箱面被删除。复用现有 xatlas UV 和多视角照片烘焙；没有可用照片时沿用顶点色降级。未充分观测的面会标记为补面，其尺寸和外观属于模型推断；无纹理覆盖的区域可能为回退色。

另存为 `scan_<session>_planar.glb`、`_cuboid.glb` 或 `_cube.glb`，对应诊断文件记录 shapeMode 和 shapeReport。不会改写 TSDF 或普通 `scan_<session>.glb`；可随时重新导出原始扫描网格。同一模式再次导出会覆盖该模式上一次结果。

## 算法来源及依赖

本包是轻量 C++ 原创实现，借鉴通用 RANSAC、最小二乘平面与正交约束机制。没有直接链接、复制或移植 CGAL／PolyFit，不宣称实现完整 PolyFit 多面体优化。工程原有第三方许可证保持不变。

机制资料：
- CGAL Shape Regularization 官方手册：https://doc.cgal.org/latest/Shape_regularization/index.html
- PolyFit 官方项目：https://github.com/LiangliangNan/PolyFit
- PolyFit 论文：https://openaccess.thecvf.com/content_ICCV_2017/papers/Nan_PolyFit_Polygonal_Surface_ICCV_2017_paper.pdf

## 验证范围

已运行主机 C++ 回归，覆盖：
- 时序一致／冲突／显露、重投影碰撞、异常大投影、来源置信度、缓冲复用及尺寸输入检查；零权重不分配 TSDF 体素。
- 旋转箱体、含噪箱体、正方体、三个相邻面缺面扫描；球体、薄片、仅两面、分离物体、明显凹陷及非法网格拒绝；拒绝时输出不被覆盖。
- 硬边法线、三角形朝向、xatlas UV、箱体 GLB 顶点／索引结构。
- 附件原有全部六个 host native suites：GLB、geometry、scan policy、optimization、continuity、robustness。
- ASan／UBSan／float-cast-overflow 检查；沙盒不支持 /proc 的泄漏枚举，LeakSanitizer 关闭。日志中的 executable name 警告来自同一环境限制。

运行新增测试：`python tools/review_tests/run_round23.py`。
完整测试：`python tools/review_tests/run_native.py`（原有 continuity 测试需要完整工程中的 Eigen）。

**未完成 Android APK 编译、Kotlin 编译及手机实测。** 此附件不含完整的 Android／OpenCV／Ceres 依赖环境；本机只对可独立编译的原生算法与导出模块做了验证，并核对 Kotlin/JNI/CMake 接口。真实照片纹理烘焙、UI 导出操作、线程生命周期与设备端内存／P95 耗时需在完整本地工程和目标手机验收。

建议用同一纸盒、带凹槽盒和球体分别导出普通／规整模型，对比尺寸、棱边、孔洞、补面提示和照片贴图。第 4 轮讨论的自适应采样／稳定曲面压缩不在本次第 2＋3 轮范围内。
