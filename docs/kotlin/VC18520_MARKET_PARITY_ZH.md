# vc18520 Kotlin 消费级体验 / 竞品对照

审查日期：2026-10-02  
基线：GitHub `f9bfbdb` + 建议先叠加 vc18510 Native overlay。

## 市场侧反复出现的需求

### Polycam
公开 Google Play 页面强调：
- 3D capture 后可裁剪、旋转、缩放；
- 多格式导出与分享；
- Android 设备覆盖面。

同时公开评价里能看到 Android/HyperOS 打不开、长时间使用闪退等兼容问题。

vc18520 对应：
- Camera 首帧 4.5 秒 watchdog；
- 第一次自动重试；
- 第二次自动进入单主摄兼容模式；
- 用户也可手工切换兼容模式；
- Viewer 退出兜底接线。

来源：
https://play.google.com/store/apps/details?id=ai.polycam

### KIRI Engine
近期评价中有用户明确反馈：
- 看过教程后仍多次扫描失败；
- 大项目处理接近完成时长时间无反馈或失败；
- 正确、稳定的采集方式对结果影响非常大。

vc18520 对应：
- 首次扫描只讲 3 件关键动作，不做长教程；
- 扫描阶段/覆盖里程碑用轻触觉确认；
- 扫描检查页生成前做质量 Gate；
- 长模型处理显示真实已处理秒数，不伪造百分比。

来源：
https://play.google.com/store/apps/details?id=com.kiriengine.app

### Scaniverse
近期评价中：
- 用户认可 Classic 模式简单易用；
- 有用户希望把原始/工程数据带到 PC；
- 历史公开评价里存在 GLB/OBJ 导出失败与处理崩溃。

vc18520 对应：
- 不引入账号和云依赖；
- 保留“标准引导 / 自由扫描”的简单入口；
- 完整扫描包增加 `.ms3d.zip` 导出/导入；
- 工程包保留模型、AR 资产、VINS 持久地图；
- 导入继续使用原 SHA-256 manifest 验证。

来源：
https://play.google.com/store/apps/details?id=com.nianticlabs.scaniverse

### RealityScan
Google Play 当前产品说明强调：
- photo coverage 颜色预览用于质量检查；
- Project Library 管理项目；
- capture/model/processing 持续改善。

1.9 更新还专门修复：
- 大字体导致 capture mode 无法选择；
- crop box / model view / 离开 capture mode 等体验问题。

公开评价也能看到：
- 前景/背景目标选择不清晰；
- 登录门槛；
- 换设备/工程延续诉求。

MobileScan3D 当前已有真实 TSDF observation heatmap 和一级物体追踪；
vc18520 进一步补：
- 完整工程迁移；
- 低质量生成前明确短板；
- 后台中断后回检查页，而不是误触发重建；
- 无账号、纯本地继续保持。

来源：
https://play.google.com/store/apps/details?id=com.epicgames.realityscan

## 本轮没有照搬的东西

- 不加账号体系。
- 不依赖云处理。
- 不为了“完成感”伪造 99% 进度。
- 不把触觉做成每帧震动，只在关键里程碑提示。
- 不因为用户切后台就直接重建/导出最终模型。
- 不在 Kotlin 里改 Native 重建数学。
