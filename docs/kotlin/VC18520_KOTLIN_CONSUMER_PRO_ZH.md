# vc18520 Kotlin 消费级强化说明

## 版本
- 建议基线：GitHub `f9bfbdb`
- 建议先应用：`vc18510 / 0.16.0-native-precision-speed`
- 本轮结果：
  - versionCode `18520`
  - versionName `0.16.1-kotlin-consumer-pro`

## 1. 首次扫描三步教练
第一次点击扫描时只强调：
1. 选中主体，主体保持静止；
2. 手机绕物体缓慢移动；
3. 扫描检查后再决定补扫/生成。

避免长教程造成信息过载。

## 2. 关键阶段触觉反馈
新增 `ConsumerHaptics` + `ScanMilestoneTracker`：
- 目标锁定；
- 可以开始环绕；
- 覆盖 25/50/75/90%；
- 可生成；
- 目标丢失。

所有事件有节流，不会持续震动。
高级菜单可关闭触觉。

## 3. 生成前质量 Gate
扫描检查页统一用 `ScanQualityGate` 计算：
- capture 20%
- viewpoint 15%
- surface observation 35%
- geometry 20%
- texture 10%

弱模型不会被强制阻止生成，但会显示最主要 3 个短板，并把“继续补扫”作为推荐动作。
定位已经中断时则允许直接生成当前模型。

## 4. 后台中断改为保留到 Review
旧行为：
- 扫描中收到电话 / Home / App 切后台
- `onPause()` 直接 `stopScan()`
- 可能意外启动最终网格导出

新行为：
- 停止继续融合；
- 强制保存 recovery checkpoint；
- 标记定位连续性已中断；
- 回到前台进入扫描检查；
- 用户自己决定生成还是放弃。

不会假装可以跨 Camera/VIO 中断继续融合。

## 5. Camera2 首帧恢复
新增：
- 4.5 秒无首帧 → 自动重试；
- 第二次 → 自动单主摄 compatibility mode；
- 再失败 → 给用户明确重试入口；
- 镜头菜单可手工切 compatibility mode。

## 6. 长处理真实反馈
新增 `ConsumerOperationDialog`：
- review mesh；
- model build / GLB；
- HQ texture；
- model optimization / undo；
- project archive import/export。

只显示真实 elapsed seconds，不伪造百分比。
用户可隐藏进度，任务继续在 App 内执行。

## 7. 完整工程迁移
ScanPackageManager 新增：
- `.ms3d.zip` export；
- `.ms3d.zip` import；
- 固定文件白名单；
- duplicate entry 拒绝；
- path traversal 拒绝；
- 最大解压 1.5GB；
- 原 manifest SHA-256 复核；
- 原子发布。

Project Library 增加：
- 分享工程包；
- 导入工程包。

## 8. Kotlin 热路径小优化
`updateViewpointCoverage()` 不再每次调用：
`sliceArray(0 until 12).average()`
避免周期性 FloatArray 分配，改为直接 12-bin 累加。

## 9. Viewer 退出
已经存在的 PopupWindow 退出兜底真正接到 Viewer 生命周期；
进入独立 Viewer 时将 GL Surface 从 on-top 切下，退出恢复 AR 所需层级。

## 不修改
- TSDF
- ICP
- VINS
- Depth fusion
- Mesh math
- vc18510 Native 文件
