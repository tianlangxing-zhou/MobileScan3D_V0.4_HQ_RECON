# 四项修复交付说明

基线提交：985738c2c84dbb1444ac974f01f8020b54de41e5。
本包包含已实现的修复，不是待实施方案；未推送 GitHub。

1. TSDF 坐标转整数越界：grid_index.h 在转换前检查有限性和整数边界，预留邻域访问余量，TSDF 和聚类简化共用检查。
2. 同一射线重复计权：记录上一个命中体素，跳过重复写入和重复哈希；后续帧仍能正常更新权重。
3. 网格简化哈希碰撞：采用完整三维坐标键判等，哈希碰撞不再错误合并不同格子或删掉正常三角形。
4. AR 资产校验及写入失败：验证索引、三角形完整性和有限属性；读取前检查文件实际长度；全部验证后交换缓存；显式关闭文件并检查写入结果。

## 使用方法

- 完整项目包：MobileScan3D_TSDF_AR_Fixed_Full.zip，所有修复已包含，直接解压使用。
- 小型覆盖包：MobileScan3D_TSDF_AR_Fixes_Overlay.zip，不是独立工程。解压到原项目根目录，让 app/、tools/ 与原项目同名目录合并，替换对应文件。
- 覆盖包中的 app/src/main/cpp/grid_index.h 是新增文件，必须保留，不可只复制三个 cpp 文件。
- 若原项目已经有其他修改，优先使用 docs/four_fixes/four_fixes_only.patch；先 git apply --check，再 git apply。已覆盖文件或使用完整包时，不要重复应用补丁。
- 适用基线为上述提交。没有针对更新的远端提交执行合并验证。

## 验证

本轮已实际运行 python3 tools/review_tests/run_native.py，五套主机原生回归全部通过：导出、几何、扫描策略、连续重建、四项边界回归。
原始输出：docs/four_fixes/validation.log。
检测使用 ASan/UBSan；新增边界回归包含 float-cast-overflow。因环境限制关闭 LeakSanitizer。
Android APK 编译、安装和真机扫描未验证。完整项目仍需按 SETUP.md 配置 Android SDK、NDK、Ceres 和本机路径。

所有交付文件的摘要见 FOUR_FIXES_SHA256SUMS.txt（覆盖包）或 REVIEW_SHA256SUMS.txt（完整项目包）。
