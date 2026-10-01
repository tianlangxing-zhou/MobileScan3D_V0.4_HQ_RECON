# 2026-10-01 MobileScan3D 实机诊断与采集数据

## 迭代背景
- 设备：OnePlus 15（PLK110，arm64-v8a）
- 版本线：vc168（epoch 开启回归修复）→ vc169（世界尺度建立）→ vc170（epoch 漂移重锚）

## 今日采集数据（见各子目录）

| 目录 | 装机版本 | 关键内容 | 用途 |
|---|---|---|---|
| `analyze_vc168/` | vc168 | `scan_20261001_145835_15281` GLB/PLY + `bakeDiag.json` + `vc168.log` | 世界尺度=0 对照组（全场景） |
| `analyze_151608/` | vc168 | 比例尺参考物 26×8cm 快扫 | 尺真值拟合（扁平物） |
| `analyze_dettol/` | vc169 | 滴露瓶 `165218` 近景 | 世界尺度复核（全场景被桌面主导） |
| `analyze_dettol2/` | vc169 | 滴露瓶 `170518` 重扫（纹理 90%覆盖） | 干净孤立瓶体、近景真值复核 |
| `analyze_vc169/` | vc169 | `vc169.log` 探针 | 机制解锁但未闩锁的证据 |

## 关键诊断结论

### 1. 世界尺度（worldScale）弃治
- 读 `native_engine.cpp` 确认 `scanWorldPerMeter` **不乘到导出网格顶点**（仅用于扫描范围过滤 / 相机距离估计 / 立体锚点注入）；GLB 顶点恒为 VINS 世界坐标（`mesh_engine` 用 `voxel=0.02m` 生成，node 矩阵纯旋转无缩放）。
- vc169 真机验证：
  - 15:41:23 首次 `worldScaleUsable=1`、`worldPerMeter=0.6632`、双目 `matched` ~0.2%→25%、`calibStereo uniq` 0→387（vc169 的 2px 深度膨胀采样确实修好了 anchor 像素深度非有限被丢的根因）→ **机制解锁**；
  - 但下一帧即回退 0（`scaleGoodStreak` 被 `scaleRej jump` 清零）；
  - 170518 又短暂闩锁 `worldPerMeter=1.5054` 后回退，估计在 1.0↔1.7 乱跳 → 噪声大、未闩锁。
- 滴露瓶真值复核（277×197×283mm）：RANSAC 桌面平面 + 3D 占据聚类干净孤立瓶体 → 模型比真值大 ~10–15%（k≈0.85–0.90），**绝非 3× 压缩**。"压缩 3×"是 vc162 扁平深度（shift 过大、fusionDepth 仅 2.4cm）的陈旧假象，vc165/166 修深度后早已消失。
- **结论**：VINS 世界本即米制，vc169/vc170 的 worldScale 对本设备无收益；若被闩锁反而有害（1.5054 会把世界缩 ~1.5×、方向错）。worldScale 特性弃治，不继续做 vc170-worldScale。

### 2. epoch 漂移挂起（susp=1）根因 + vc170 修复
- **根因**：`native_engine.cpp:2574` 把**冻结 `epochCalib`** 与**瞬时 live 标定器**在同一参考工作点 `epochRefRaw` 上比漂移；两条恢复路径都因 live 标定器噪声而失效：
  - 路径①（漂移回门限）：比 frozen vs 瞬时 live，live scale `0.0033↔0.0062` 乱跳 → `lastDriftRel` 永远 0.57–0.89 > 0.20 → 永不回弹；
  - 路径②（`validatesFrozen` vs VINS 真值，`scan_policy.h:27`）：要求 frozen 对 VINS 在 8% 内吻合 ≥80%，但 frozen 本身已偏 ~10–15% → `good` 永远 <80% → `frozenRecoveries=0`。
  - ⇒ `susp=1` 一旦触发（`badStreak` 21→68 持续涨）就永久锁死，期间 `epochFusionDepth=nullptr` → 新几何停止写入。
- **vc170 修复**（`native_engine.cpp` epoch 状态机，未动 Java/JNI/CMake）：
  - 漂移检测改用 **live 标定器 EMA**（`kEpochLiveEmaNew=0.15`）消除单帧噪声误判；
  - 加 **稳定性门**（`kEpochLiveVarSuspendRel=0.05`：live 相对方差 <5% 才认作“真实漂移”）；
  - 加 **重锚 escape**：`susp=1` 且持续背离且 live 稳定 → `epochCalib` 采用平滑 live、`epochRefRaw` 重取、清 streak、`susp=0`、`++epochReanchors`。**彻底解除永久 susp=1**；
  - `EpochProbe` 增 `liveEma scale/shift/var/relVar/stable/reanchors` 诊断字段。
  - 版本 `169→170`，`versionName=0.13.30-epoch-reanchor`；本地 Gradle `:app:assembleDebug` **BUILD SUCCESSFUL**；APK md5 `31e4164d73e5c4750ee5a77feb3d07d3`。

### 3. 风险
- vc170 重锚瞬间 `epochCalib` 改变 → 新几何用新尺度、旧几何留旧尺度，**可能有 ~10–15% 尺度接缝**（远好于完全不生长）。若复测发现重锚过频、尺度抖，把 `kEpochLiveVarSuspendRel` 调小（更严稳定性门）或 `kEpochLiveEmaNew` 调小（更慢跟随）即可。

## 待办（依赖真机复连）
- vc170 装机 PLK110 真机验证：`EpochProbe` 看 `reanchors>0`、`susp` 回 0、几何持续生长、无回归；验证通过后再追加 commit（本 commit 已含源码与数据，验证若改码再补 commit）。
