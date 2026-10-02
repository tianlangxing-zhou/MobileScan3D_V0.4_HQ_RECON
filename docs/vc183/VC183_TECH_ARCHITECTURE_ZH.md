# MobileScan3D vc183 技术架构建议

## 1. 目标

避免继续把功能堆进 `MainActivity.kt`。

建议 vc183 开始做模块化拆分。

---

## 2. 推荐模块

```text
app/src/main/java/com/mobilescan3d/

scan/
├── ScanSession.kt
├── ScanSessionManager.kt
├── ScanCheckpointManager.kt
├── ScanQualityEvaluator.kt
└── ScanMode.kt

scan/guidance/
├── ScanGuidanceController.kt
├── ScanGuidanceState.kt
├── GuidanceInstruction.kt
├── NextBestViewSelector.kt
└── ViewpointCoverageTracker.kt

scan/quality/
├── MotionQualityMonitor.kt
├── DepthQualityMonitor.kt
├── LightingQualityMonitor.kt
├── MaterialRiskDetector.kt
└── CompletionEvaluator.kt

model/
├── ModelCleanupPipeline.kt
├── ModelProject.kt
├── ModelRepository.kt
├── ModelMeasurementService.kt
└── ModelExportService.kt

ui/guidance/
├── ScanGuidanceOverlay.kt
└── ScanGuidanceRenderer.kt

ui/library/
├── ProjectLibraryActivity.kt
└── ProjectCardAdapter.kt

ui/viewer/
├── ModelViewerController.kt
├── ModelReviewOverlay.kt
└── MeasurementOverlay.kt
```

---

## 3. Guidance 输入输出

输入：

```text
trackingConfidence
targetBoundingBox
targetCenter3D
cameraPose
linearVelocity
angularVelocity
distanceToTarget
depthValidRatio
viewpointCoverage
surfaceConfidence
thermalState
storageState
```

输出：

```text
guidanceState
primaryInstruction
secondaryInstruction
recommendedAzimuth
recommendedElevation
severity
completionConfidence
```

---

## 4. Viewpoint Coverage

建议第一版：
- 水平方位 12 桶
- 上下各增加 elevation band

例如：

```text
LOW
MID
HIGH
```

总计形成轻量球面覆盖表。

每个 bin：
- frameCount
- validDepthRatio
- trackingScore
- tsdfContribution
- lastSeenTimestamp

---

## 5. Next Best View

第一阶段使用规则评分：

```text
score =
  missingWeight
+ lowSurfaceConfidence
+ geometryNeed
+ textureNeed
- movementCost
- trackingRisk
- occlusionRisk
```

要求：
- 5–10Hz 足够
- 0.8–1.5s 滞回
- 避免左右反复跳

---

## 6. 模型清理 Pipeline

推荐顺序：

```text
Remove obvious outliers
→ Connected components
→ Keep target component
→ Plane/background removal
→ Small hole repair
→ Normal repair
→ Mesh simplification
→ Texture cleanup
```

每步必须：
- 可取消
- 有进度
- 失败不破坏原始模型
- 保存原始版本

---

## 7. 项目数据模型

建议：

```text
ModelProject
- id
- name
- createdAt
- updatedAt
- status
- thumbnailPath
- rawModelPath
- optimizedModelPath
- checkpointPath
- qualitySummary
- measurementSummary
- exportHistory
```

---

## 8. Checkpoint

必须版本化：

```text
checkpointVersion
scanVersion
deviceInfo
cameraConfig
targetState
poseHistory
coverageState
tsdfStateRef
guidanceState
```

旧 checkpoint 不兼容时必须明确拒绝，而不是强行读取导致崩溃。

---

## 9. 本地存储策略

建议目录：

```text
files/
└── projects/
    └── <project-id>/
        ├── metadata.json
        ├── thumbnail.webp
        ├── checkpoint/
        ├── raw/
        ├── optimized/
        └── exports/
```

---

## 10. 性能预算

Guidance：
- <= 3ms / update
- 5–10Hz

UI：
- 避免每帧创建 Path/Object
- 不阻塞 GL Thread
- 不阻塞 Camera Thread

Cleanup：
- 后台 worker/thread
- 有进度
- 可取消

Library：
- 缩略图懒加载
- 不在主线程解析模型

---

## 11. 降级原则

任何消费增强模块失败，都不能让核心扫描不可用。

例如：
- Guidance 失败 → 基础扫描
- 材质检测失败 → 不提示
- Cleanup 失败 → 保留原模型
- 测量失败 → 不显示尺寸
- Library metadata 损坏 → 尝试恢复 raw model

---

## 12. 安全边界

在 vc183 内保持：
- 默认本地处理
- 不新增无必要网络权限
- 用户主动导出才离开 App 私有目录
- 日志不写入原始图像内容，除非用户主动生成诊断包
