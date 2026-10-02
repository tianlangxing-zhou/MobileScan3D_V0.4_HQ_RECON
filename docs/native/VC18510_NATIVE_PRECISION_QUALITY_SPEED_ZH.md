# vc18510 Native 精度 / 模型质量 / 速度强化

## 基线

- GitHub commit: `f9bfbdb`
- versionCode: `18501`
- versionName: `0.15.6-overlap-icp-anchor`
- 本包结果:
  - versionCode `18510`
  - versionName `0.16.0-native-precision-speed`

本轮只修改 Native/算法相关路径和版本号，不覆盖 Kotlin/UI。

---

## 一、扫描精度

### 1. Frame-to-model ICP：从单纯 point-to-point 升级为鲁棒混合求解

旧实现的问题：

- 名称/注释提到 trimmed ICP，但对应点并没有真正按残差裁剪。
- 只做 point-to-point Kabsch。
- ICP `maxCorr = 6cm`，但 surfel 最近邻哈希只查 2cm cell 的固定 3×3×3 邻域，
  4–6cm 的合法匹配经常数据结构层面根本找不到。
- 共面/弱几何下 Kabsch 容易欠约束。

新实现：

1. 最近邻搜索半径按 `maxDist` 动态换算为 2cm hash shell。
2. 先查近邻，必要时逐 shell 扩展，避免每个点总是做最坏情况搜索。
3. ICP 前两轮 6cm、随后 4.5cm、最后 3cm，coarse-to-fine 收紧。
4. 每轮真正保留约 85% 低残差对应点。
5. 目标 surfel 返回法向和 hits。
6. 优先执行 Huber IRLS 风格 point-to-plane Gauss-Newton。
7. 6×6 法方程做可观测性/rank 检查。
8. 平面/线性等退化情况自动退回 trimmed point-to-point Kabsch。
9. 单次修正仍然不回馈 VINS。
10. 保留逐迭代和总平移/旋转上限。

这套结构参考成熟 RGB-D / reconstruction 系统常见的：
- point-to-plane model tracking；
- coarse-to-fine correspondence；
- robust loss / IRLS；
- degeneracy/nullspace protection。

### 2. 深度预处理：confidence-aware 小核双边式滤波

原有 3×3 range-aware filter 已经比简单均值好，但没有利用输入 confidence。

改为：

- 3×3；
- 紧支撑 Tukey-like range kernel；
- 邻域 confidence 加权；
- 中心点保持较强权重；
- 不补洞；
- 不做 temporal blending，因此不会人为制造跨帧 ghost；
- 深度突变两侧不互相平均。

目的：
- 平面噪声更低；
- 轮廓/薄结构更少被抹平；
- 弱 confidence 像素不会和高可信像素同权。

### 3. TSDF 融合增加轻量 incidence weighting

在 adaptive geometry 已经计算可靠法向时：

`weight *= 0.55 + 0.45 * |n · ray|`

作用：

- 掠射角深度对表面位置更敏感，降低其权重有助于减少 silhouette 变厚。
- floor 保持 0.55，不会把侧面直接饿死。
- protected detail 不使用该降权，避免尖锐细节被过度削弱。

没有加入激进 free-space carving。
原因：当前是手机 VIO + learned/estimated depth 链，pose/depth drift 下 aggressive carving 可能把真实几何删掉。

---

## 二、模型质量

### 1. Surfel coarse normal 不再沿用第一个点

coarse surfel 原先平均位置和颜色，但法向沿用组内第一个 surfel。

现在同步平均并归一化法向。

这对 point-to-plane ICP 很关键，否则一个 patch 的代表点位置是平均值、法向却来自任意首点，会产生不一致约束。

### 2. Surfel 压缩 / 再激活增加滞回

原来的组合：

- 压缩：`dot > .98 && planeError < 5mm`
- 再访问粗点：`dot < .996 || error > 2mm` 就拆回细点

两者会在普通传感器噪声下反复“压缩→拆开”。

改为：

- 压缩条件保持原 vc185 建锚策略；
- 再激活约 `dot < .985 || error > 4mm`；
- protected detail 仍立即恢复细结构。

目标：稳定 ICP 锚，减少 churn。

### 3. 基础 MeshEngine QEM 修正

旧 QEM 的主要问题：

- 每条 collapse 固定取边中点；
- 没有 collapse 前相邻三角形翻面/退化检查。

新实现：

- 求解局部 quadric 最优点；
- 3×3 系统病态时比较 midpoint/endpoints；
- 最优点离边过远时拒绝该解；
- collapse 前检测相邻三角形退化；
- 限制法线翻转，超过约 70° 的 collapse 拒绝。

这与后续 `v06/mesh_postprocess` 的保护思想保持一致，避免基础/实时网格先被破坏。

---

## 三、速度

### 1. ICP 不再每轮都走完整 point-to-point 数据复制

法向可用且系统可观测时直接求 6×6 point-to-plane；
只有退化时才构造 src/dst 做 Kabsch fallback。

### 2. Surfel 最近邻分层搜索

常见近匹配仍优先走 3×3×3；
只有找不到足够近点时扩 shell。

因此不会把“支持 6cm”简单实现成每个 sample 都无条件扫描大立方邻域。

### 3. 哈希表 reserve

- surfel rebuild/compression hash map 根据当前规模预留；
- TSDF sparse block map 只 warm-reserve 最多 262,144 bucket 对应规模。

特别注意：没有按理论 `maxBlocks` 一次性 reserve。
这是为了避免“速度优化换成启动时额外吃十几 MB bucket memory”。

---

## 四、明确摒弃/没有采用的方案

### 不恢复全局 `-ffast-math`

当前工程已经去掉全局 fast-math，这是正确方向。

TSDF/VINS/depth/guard 中有大量 NaN/finite/range 检查，重新打开全局 fast-math 可能让编译器破坏这些安全假设。

### 不重新启用 xatlas 多线程

工程已有注释记录 vendor xatlas 并行路径存在 race/deadlock 风险。
没有为了 benchmark 数字重新打开。

### 不在本轮上 BundleFusion 式全局 GPU 重积分

成熟桌面方案可以用全局 pose graph / reintegration 提高闭环一致性，但对当前 Android 本地实时链属于大架构改造，GPU/内存/调试成本高。

建议未来作为“离线 HQ 二次优化”单独做，不混进当前稳定实时融合路径。

### 不激进改写 VINS

vc185 的根问题是让 VINS drift 后的 frame-to-model correction 能建立锚。
本轮继续强化这个 correction，而不是直接改 VINS 内部优化器参数，避免同时改变视觉惯性定位和重建两个系统后无法归因。

---

## 五、回归结果

### Robust ICP synthetic 非对称曲面

同一组 synthetic pose：

旧 `f9bfbdb`：
- applied=1
- rmse ≈ 4.63mm
- 最终 translation residual ≈ **34.34mm**

vc18510：
- point-to-plane iterations = 3
- fallback = 0
- rmse ≈ `7.5e-8 m`
- 最终 translation residual 接近 **0**

注意：这是确定性的合成回归，不是真机精度宣传。

### Surfel hash-radius

约 5cm 偏离的查询，在 6cm maxCorr 下能找到 coarse anchor；
验证动态 shell 修复了“配置允许但哈希搜不到”的问题。

### Depth filter

测试：
- noisy center `1.012m` → `1.004m`
- 邻接 `1.20m` 深度突变保持 `1.20m`

### QEM

wavy grid：
- 882 triangles → 441 triangles
- 回归检查没有产生退化面或翻面。

---

## 六、真机必须继续做的验收

同一个物体、同一距离、同一手机，对比 f9bfbdb 和 vc18510：

1. `icpAppliedCount / attempted`
2. ICP RMSE
3. fusion accepted ratio
4. guard reject ratio
5. surfel coarse count / reactivated count
6. TSDF blocks / peak RSS
7. Mesh open edges
8. 重叠壳层厚度
9. 尺寸误差
10. build mesh ms / textured GLB ms
11. 5/10/20 分钟 thermal 与内存
12. 弱纹理 / 反光 / 小物体 / 尖角 / 薄片分别测试

只有真机 A/B 才能决定是否继续调：
- Huber delta
- ICP maxCorr
- incidence floor
- QEM normal-flip 阈值
- surfel hysteresis

不要仅凭 synthetic case 继续放宽参数。
