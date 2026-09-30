# vc153 运行效率与数值精度补丁

## 覆盖方式

本 ZIP 是增量覆盖包，只包含 4 个生产源码文件、测试入口/新增测试和本文。
解压后，将 `MobileScan3D/` 内的内容合并到原工程根目录，覆盖同名文件。
不要删除原工程或用这个增量包单独构建。无需改动 Gradle、JNI 接口、资源、模型、版本号或外部依赖配置。
建议覆盖前备份原工程，覆盖后在已有 Android 构建环境运行：

```bash
./gradlew clean :app:assembleDebug
```

基线是本次上传的 `MobileScan3D_core_src_vc153_20260930.zip`，
其中 `versionCode=153`，`versionName=0.13.19.7-app-icon`。
GitHub 页面在本次环境中无法获取，未核对远端最新提交；请只对该附件对应的工程覆盖。

## 修改内容

- `app/src/main/cpp/tsdf_engine.cpp`：同块采样复用块指针，减少哈希查找；已存在体素不再重复维护边界。
  缓存仅存在于单次调用内，兼容重置、哈希扩容、负坐标，以及内存配额用满后继续融合已有块。
  未降低像素采样密度、体素分辨率或权重精度，未修改 TSDF 数据布局。
- `app/src/main/cpp/depth_calib.cpp`：用中心化双精度加权最小二乘替代易发生消减的原始矩公式；
  IRLS 使用有符号残差的 MAD，并围绕残差中位数剔除异常点，避免初始偏移导致正常样本全被排除；
  复用迭代缓冲区、就地求中位数，减少分配和拷贝；修正样本抽稀步长的整除边界及配对样本统计。
- `app/src/main/cpp/depth_calib.h`：标定求值、输入域重参数化使用双精度中间量，降低乘加消减误差，保持公开参数类型兼容。
- `app/src/main/cpp/surfel_engine.cpp`：点云质心采用双精度累加；修正 `copyPoints` 中过大 `minHits` 转为 uint16 时回绕的问题。
- `tools/review_tests/optimization_regression.cpp`、`run_optimization.py`：新增精度回归、融合一致性检查和可复现 A/B 微基准。
- `tools/review_tests/run_native.py`：将新增测试接入原有原生测试入口。

## 已完成验证

主机 g++ / C++20 测试通过：

- `run_geometry.py`：深度投影、逆深度标定、无效样本、符号反转、TSDF 内存配额、平面网格。
- `run.py`：5 种 GLB 导出、坐标朝向、AR 资产、畸形输入与不等长深度数组。
- `run_robustness.py`：数值异常、单射线权重、网格哈希冲突、资产读写。
- `run_scan_policy.py`：扫描范围、单位、无效输入、冻结证据与近场网格。
- `run_optimization.py`：大偏移标定、常量退化、含噪声/离群点的正/逆深度拟合、采样预算、质心、命中阈值和融合重置。

上述测试使用 ASan/UBSan；新增优化测试与 robustness 测试还启用了 float-cast-overflow。
未报告内存/未定义行为错误。受环境限制，LeakSanitizer 未启用。

### A/B 结果（仅为本次主机合成数据测试）

| 项目 | 原版 | 修改版 |
| --- | ---: | ---: |
| TSDF 每帧耗时中位数 | 5.29119 ms | 4.86241 ms |
| 大偏移线性标定有效性 | 误判为无效 | 有效 |
| 线性标定最大相对误差（噪声+10%离群点） | 23.3758% | 0.0274532% |
| 逆深度标定最大相对误差（噪声+10%离群点） | 0.0367544% | 0.0324612% |
| 1024 对样本、上限 512 时实际采样数 | 342 | 512 |
| 10 万点质心最大坐标误差 | 0.122375 m | 2.5779e-8 m |

TSDF 耗时下降约 **8.10%**。测量为同机 `-O3`，7 轮、每轮 16 帧，128×128 深度，
12 mm 体素、128 块配额，包含旋转/平移、深度空洞、颜色与满配额情形；不包含相机、推理、VIO 或导出耗时。
两版均产生 29,432 个已观测体素、128 块，所有体素 TSDF/几何权重/颜色/颜色权重的确定性摘要一致：
`10245960488155071710`。摘要按块无序聚合，不依赖 unordered_map 的遍历顺序。
质心场景特意使用 x≈1000 m 的世界坐标以暴露 float 长序列累加误差；这不是实际扫描精度测量。
大偏移标定测试输入为 1,000,000 附近的小幅变化；修改版在该测试中输出误差为零。

复现新回归（工程根目录）：

```bash
python3 tools/review_tests/run_optimization.py
```

与备份的原版做 A/B：

```bash
python3 tools/review_tests/run_optimization.py --baseline /path/to/original/MobileScan3D
```

完整依赖齐备时，可继续运行原有 `python3 tools/review_tests/run_native.py`。

## 验证边界

本环境与核心源码附件缺少完整 Android SDK/NDK、Ceres、OpenCV、Eigen 等构建依赖，
未生成 APK，也未执行需要 Eigen 的 `run_continuity.py`，未做 Android 真机帧率、功耗、长时间扫描与真实物体误差测试。
上述改动改善的是已验证的计算路径与数值问题，不保证整机提速比例或绝对重建精度；
实际效果仍取决于深度模型、相机/IMU 标定、运动和拍摄条件。
