# MobileScan3D 第 2＋3＋4 轮合并状态替换包（vc154 基线）

适用基线：本次上传的 `MobileScan3D_core_src_vc154_round23_20261001.zip`。
该基线已经包含第 2、3 轮。本包在其上实现第 4 轮，**仅附相对该基线新增／修改的文件**；重叠的源文件已保留第 2、3 轮逻辑，不需要再次叠加旧补丁。

1. 备份本地工程。
2. 可在解压的补丁目录运行 `python3 MobileScan3D/tools/review_tests/verify_round234_overlay.py --before <本地工程根目录>` 检查基线。自行修改过的文件会报告 DIFFERENT，需要先合并。
3. 将 ZIP 内 `MobileScan3D/` 的内容合并覆盖到本地工程根目录（内含 `app/`、`settings.gradle.kts` 的目录）。不要用整个目录替换／删除原工程；保留本地依赖、模型和签名配置。
4. 覆盖后在工程根目录运行 `python3 tools/review_tests/verify_round234_overlay.py`，随后在完整 Android 环境重新编译。

新增：几何自适应采样、深度／遮挡／孔洞边界保护、稳定区域降低重复融合频率、共面预览点压缩与矛盾观测重新激活。场景和目标两条路径均已接入，默认开启；稀疏 stereo 约束保留完整更新。

本包通过主机原生回归和合成数据验证；未完成 Android 编译、手机扫描与真实照片烘焙验收。具体收益、参数、边界及测试记录见 `docs/round4/README_REPLACE_ZH.md`。
