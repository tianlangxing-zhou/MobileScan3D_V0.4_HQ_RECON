# MobileScan3D vc183 下一版本完整规划
建议版本：vc183 / 0.15.0-guided-scan

## 一、版本主题

**Guided Scan / 标准扫描引导**

把 vc182 已经具备的：

- 物体追踪
- VIO
- 深度
- TSDF
- 视角覆盖
- 3D 查看
- 扫描检查

整合成一个普通消费者真正会用的标准扫描流程。

版本核心价值：

> 用户无需理解 3D 重建技术，也能按照实时方向提示完成一个更完整、更稳定的模型。

---

## 二、版本目标

### P0 目标

1. 标准扫描模式可完整跑通；
2. 有目标锁定；
3. 有水平环绕方向引导；
4. 有顶部/下半部补扫；
5. 有缺口定向补扫；
6. 有明确完成判断；
7. 引导不能明显降低帧率；
8. 引导不能影响现有重建结果；
9. 引导丢失时必须可降级到普通扫描。

### P1 目标

1. 自由扫描模式；
2. 更细的下一视角评分；
3. 更友好的首次教程；
4. 方向提示动画；
5. 完成质量分级；
6. 3D 查看页高亮缺失区域。

---

## 三、建议代码结构

当前项目的 `MainActivity.kt` 已经承担过多职责。vc183 不建议继续直接追加大量扫描引导逻辑。

建议新增：

```text
ui/guidance/
├── ScanGuidanceOverlay.kt
├── ScanGuidanceController.kt
├── ScanGuidanceState.kt
├── ScanGuidanceRenderer.kt
└── GuidanceInstruction.kt

scan/coverage/
├── ViewpointCoverageTracker.kt
├── CoverageBin.kt
├── SurfaceCoverageSummary.kt
└── NextBestViewSelector.kt
```

MainActivity 只做：

```text
采集底层状态
→ 传给 ScanGuidanceController
→ 订阅 guidance output
→ 更新 ScanGuidanceOverlay
```

---

## 四、开发阶段

### Phase 1 — 状态抽象

目标：

把当前散落在 UI/算法里的状态统一成 Guidance Input。

实现：

- 目标追踪置信度
- 相机相对目标方位角
- 俯仰角
- 当前距离
- 运动速度
- 深度有效率
- 视角覆盖桶
- 重建表面置信度

交付：

`ScanGuidanceInput`

---

### Phase 2 — 水平视角覆盖

实现方位角离散桶。

建议先 12 桶：

```text
每 30° 一个桶
```

每桶记录：

- 有效帧数量
- 深度质量
- 追踪质量
- TSDF 贡献

输出：

```text
UNKNOWN
WEAK
GOOD
COMPLETE
```

---

### Phase 3 — UI 轨迹环

新增轻量 Overlay：

- 轨迹环
- 当前方向箭头
- 完成区域
- 推荐区域

性能要求：

- 不创建大量临时对象
- 不影响 Camera2/GL 帧率
- 不在每帧做复杂 Path 重建
- 目标 >= 30 FPS UI 刷新体验

---

### Phase 4 — Guidance State Machine

加入状态：

```text
Acquire
Lock
Distance
HorizontalOrbit
Top
Lower
RecoverMissing
Ready
```

要求：

- 状态转换有迟滞
- 不抖动
- 不在两种指令之间频繁切换

---

### Phase 5 — Next Best View

第一版使用规则算法。

候选：

- 左
- 右
- 左上
- 右上
- 左下
- 右下
- 背面方向

根据缺失程度和当前位置选择一个推荐方向。

---

### Phase 6 — 完成判断

建立 Completion Evaluator。

不应只看一个指标。

建议：

```text
Completion =
    horizontal coverage
  + vertical coverage
  + surface confidence
  + geometry stability
  + tracking quality
```

输出：

- Continue
- RecommendedComplete
- StrongComplete

---

### Phase 7 — 3D Review 联动

扫描结束后：

如果模型有缺口：

- 在 3D 模型中高亮
- 用户选择“继续补扫”
- 回到扫描页面
- Guidance 自动指向对应缺失区域

这将形成真正闭环：

```text
扫描
→ 检查
→ 找缺口
→ 自动指引补扫
→ 生成
```

---

## 五、UI 改造规划

### 主扫描页

保留 vc182：

- 顶部状态
- 右侧一级追踪
- 补光
- 预览
- 底部扫描按钮

新增：

- 中央扫描轨迹环
- 当前推荐方向箭头
- 动作提示
- 完成状态提示

不增加更多永久数字。

---

## 六、首次使用体验

第一次进入“标准扫描”时，可以有 3 屏以内的极简说明：

### 屏 1
`将物体完整放入画面`

### 屏 2
`保持镜头朝向物体，围绕它缓慢移动`

### 屏 3
`跟随屏幕方向提示补齐顶部和背面`

然后进入实际扫描。

不要做 7–10 页长教程。

---

## 七、性能预算

Guidance 逻辑必须轻量。

建议预算：

- Guidance Controller：<= 3 ms / update
- UI Overlay：不阻塞 Camera / GL thread
- Coverage 更新：5–10 Hz 足够
- 文案/箭头状态：5 Hz 以内
- 原始传感器仍按算法所需频率工作

Next Best View 不需要每帧重算。

---

## 八、失败与降级策略

如果：

- 目标追踪置信度下降
- 深度不可用
- IMU 异常
- 覆盖模型未初始化

则 Guidance 自动降级：

```text
标准扫描
↓
基础扫描模式
```

基础模式仍保留：

- 追踪
- 距离
- 扫描状态
- 完成/生成

不能因为 Guidance 模块故障导致整次扫描无法使用。

---

## 九、建议的本地事件

当前应用无网络权限，可先做本地事件。

建议记录：

```text
guidance_state_enter
guidance_direction_change
tracking_lost
distance_warning
scan_completed
scan_abandoned
review_continue_scan
```

只写本地诊断文件。

以后如果产品决定增加遥测，再另行做隐私评估和授权。

---

## 十、版本节奏

### vc183
标准扫描核心闭环

### vc184
优化缺口引导 + 3D Review 联动

### vc185
不同物体类型的策略自适应

例如：

- 小物体
- 中型摆件
- 家具
- 人体/半身

不建议 vc183 一次做完。

---

## 十一、vc183 Definition of Done

只有满足以下条件才建议发版本：

- 标准扫描从锁定到生成可完整走通
- 方向提示不会来回抖动
- 用户逆向移动不会导致错误提示风暴
- 水平 360° 覆盖判断稳定
- 顶部引导可正确触发/跳过
- 桌面物体不会因为底面不可见永远无法完成
- 完成状态和实际模型质量基本一致
- Android 返回/后台恢复不破坏 Guidance 状态
- Guidance 关闭后与 vc182 重建结果无明显退化
- 至少 3 档 Android 设备验证
