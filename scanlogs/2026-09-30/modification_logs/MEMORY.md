# MobileScan3D_V0.4_HQ_RECON 长期记忆

> 流程在 skill `mobilescan3d-build-verify-commit`；按日细节见 `memory/YYYY-MM-DD.md`。

## 一、硬规则
1. 改完必 commit + push `origin/main`，核 HEAD == `ls-remote origin refs/heads/main`。**外部 AI 也推同仓库 → 动手前先 ls-remote。**
2. APK → `D:\WorkBuddy_3DGS\APK输出\MobileScan3D_V0.4_<sha8>.apk`。顺序：改完 → commit → `assembleDebug` → 拷 APK（否则 GIT_COMMIT 带 `-dirty`）→ md5 比对 → `verify_apk.py <apk> <sha8>`。
3. 大改动前后留 `_backup_YYYYMMDD\` + 尽快 commit（曾整棵 `app/src` 丢过）。**预览左右镜像：不修/不改/不问**。
4. **禁区**：VINS 本体；KLT/NanoTrack 门限；`TARGET_DEPTH_FILTER_ENABLED`；`setLookAtM+perspectiveM`；`nativeGetRenderPose` 不可当某帧 pose。外部补丁**先审计再落地**。

## 二、环境
- Bash 逐字列全 `.../PortableGit/versions/1.2.0/` 的 `cmd`+`mingw64/bin`+`usr/bin`（花括号展开不生效）+ `export HOME=/c/Users/Administrator`；未设时连 `ls`/`grep` 都没有 → 走 Glob/Grep/Read。
- Python `C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe`（Windows 原生、路径 `D:/`、**无 numpy**）。`JAVA_HOME=POSIX /d/WorkBuddy_3DGS/.toolchain/jdk/jdk-17.0.20.1+1`；`.toolchain/gradle/gradle-8.13/bin/gradle --offline`；**别用 `cmd //c gradle.bat`**（假成功）。
- **并行 Edit 会静默丢改动** → >1 处改动一律 Python 补丁脚本，全锚点校验后**一次性写盘**；行尾每轮现场探测。假失败（exit 127 + 空 stdout）只看 `git rev-parse HEAD`/`status --porcelain`。
- adb 在 O+Connect：`C:/Program Files/O+Connect/daemon/bin/adb.exe`（本机无 SDK）。`MSYS_NO_PATHCONV=1`；install 的本地 APK 路径**必须 Windows 原生单引号 `'D:\...\x.apk'`**。签名不一致 → `uninstall` 再装（**清数据，先问大叔**）。
- OpenCV 4.12 已 vendoring，链接/运行同一份 `jniLibs/libopencv_java4.so`，**别换该 .so**。校验外部包落地**必须归一化 `\r\n`→`\n` 再比 SHA256**。
- logcat 环缓冲会冲掉稀疏诊断；`debug.ply` 是 **ASCII**。

## 三、反直觉结论（都踩过）
- ★**两 tracker 共谋**：`adoptNanoBox()` 用 Nano 框重播种 KLT 且写死 `inlierRatio=1.0f` →「IoU 0.783/inlier 1.0」不构成证据。
- ★**Sticky Epoch**：冻结后不一致**只暂停融合，绝不清几何**；★**latch 型状态位必须有恢复出口**。
- ★`std::mutex` 不可重入（`...Locked()` 命名）；★相似度门 `-1` **必须放行**；★「功能名存在」≠「功能被实现」。
- `DepthScaleEstimator` 只统计不施加，米制对齐唯一入口是 native `depthCalibrator`；**AR 投影只认 `cameraToView`**，绝不 `x0*width`。
- HQ 捕获绝不让「锁定」参与触发（`AE_MODE=OFF` → AE_STATE 只能 INACTIVE → 死锁）；luma 按 **BGR**；CMake 不加 `-ffast-math`。
- **尺度混叠**：改 scale 只重置 epoch 不清几何 → 双层/撕裂。

## 四、版本索引
- **现行 `623371f1` `0.13.19.6-tsdf-ar-fixes`（vc 152）** ← 2026-09-30 外部 TSDF/AR 修复包：新增 `grid_index.h::checkedGridIndex()` / tsdf 跳过重复射线 / mesh_engine 用 `CellKey` 三维键判等 / ar_textured_asset 全量校验后才换缓存。已装机。
- `1e03f27e`(151 atlas 2K→4K) **实测生效**：atlas 4814×4826、每三角形 290px（旧 776×771 仅 79px）。`985738c2`(150 放宽 epoch 挂起门控) **实测生效**：`epochSusp=0 fused=34 gated=25`（旧 `epochSusp=1 fused=23 gated=36`）。
- 更早版本见 `git log`。**坏包**：`MobileScan3D_Optimized_20260930.zip` 截断（源码全丢、build.gradle 为旧 150）已否决，审计见 `_patch/opt_audit/AUDIT_REPORT.md`。

## 五、下一步
- **17:42 扫描已验收：融合链路健康；覆盖率 37.7% 判定为取景问题、非回归**（详见 `2026-09-30.md`）。待大叔定：A（零改动，推荐）重扫退到 0.5~0.8m + 绕行/大角度转动；B 提高可用关键帧数（现 `BURST_FRAMES=4`，池仅 11 帧，动 HQ 流程有风险）；C 降网格上限（rawTris=105490 被 decimate 到 100000）。
- **遗留盲区**：`TargetDiag` 连续两次扫描 0 条（`haveTi=false`）；epoch 漂移数值只写在依赖 `haveTi` 的 TargetDiag 里 → 不锁目标时看不到 drift。长期：PLK110 Camera-IMU 外参标定（现 `ric=identity`、`tic=0`）。
