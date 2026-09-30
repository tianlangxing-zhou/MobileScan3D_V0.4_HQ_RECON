# MobileScan3D 第 2＋3 轮增量替换包

将本包 MobileScan3D 目录下的内容合并到本地工程根目录，保留原有依赖和模型。
基线必须是本次上传的 `MobileScan3D_core_src_vc153_perpixel_20261001.zip`。

已包含：逐像素置信度、时序掩码修正、缓冲复用、分阶段耗时统计，以及硬表面规整／长方体／正方体拟合与原有照片贴图导出流程接入。

详细操作、限制、测试命令和算法资料见 `docs/round23/README_REPLACE_ZH.md`。
主机原生回归测试已通过；未完成 Android／Kotlin 编译和手机真实照片烘焙验收。此包是源码补丁，不是 APK。

可先使用 `tools/review_tests/verify_round23_overlay.py --before <本地工程目录>` 检查覆盖基线；合并后去掉 --before 检查覆盖完整性。
