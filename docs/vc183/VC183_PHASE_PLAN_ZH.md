# MobileScan3D vc183 阶段实施计划

## vc183.1 / 0.15.1-guided-scan

### 必做
- Guidance Controller
- ScanGuidanceOverlay
- 水平 12-bin 覆盖
- 距离提示
- 运动速度提示
- 纯旋转提示
- 顶部补扫
- 下半部补扫
- 完成判断

### Gate
新用户可以跟提示完成扫描。

---

## vc183.2 / 0.15.2-recovery-loop

### 必做
- 3D Review 缺失高亮
- 缺失方向映射
- 继续补扫
- session 保持
- 补扫后重新 review
- 最多 3 轮补扫

### Gate
明显背面/顶部缺口可被引导补齐。

---

## vc183.3 / 0.15.3-model-cleanup

### 必做
- Connected components
- 去孤岛
- 去浮点
- 平面/桌面去除
- 小孔修复
- 法线修复
- 网格简化
- “优化模型”入口

### Gate
优化失败不破坏原始模型。

---

## vc183.4 / 0.15.4-project-library

### 必做
- 我的模型
- metadata
- thumbnail
- project status
- 重命名
- 删除
- 打开
- 继续补扫
- checkpoint
- 异常恢复

### Gate
App 被杀后能恢复未完成项目。

---

## vc183.5 / 0.15.5-adaptive-scan

### 必做
- 自动目标尺度
- 自动推荐距离
- 光照不足检测
- 高反光风险
- 低纹理风险
- 背景相似风险
- 自动性能档位
- 热状态降级

### Gate
默认模式下用户不需要调整底层重建参数。

---

## vc183.6 / 0.15.6-measure-share

### 必做
- 高宽深
- 两点测量
- 单位切换
- AR 摆放
- GLB/OBJ/PLY
- 快速/标准/高质量导出
- Android 系统分享

### Gate
模型生成后能够被真正使用和分享。

---

## vc183 GA

### 必须
- 六阶段回归
- 数据迁移
- checkpoint 兼容检查
- UI 统一
- 文案统一
- 性能回归
- 设备兼容
- 商店发布素材
