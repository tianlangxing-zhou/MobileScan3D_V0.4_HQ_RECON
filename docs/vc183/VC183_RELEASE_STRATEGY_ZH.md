# MobileScan3D vc183 发布策略

## 1. 版本管理

统一大版本：
`vc183`

内部：
- 0.15.1
- 0.15.2
- 0.15.3
- 0.15.4
- 0.15.5
- 0.15.6

最终稳定版：
`0.15.x`

---

## 2. 灰度顺序

### Alpha
内部开发设备
- 功能 correctness
- 崩溃
- checkpoint
- model cleanup

### Beta
真实用户小范围
重点：
- 是否看懂方向箭头
- 是否知道围绕物体移动
- 是否会漏顶部
- 是否会误以为“旋转手机”

### RC
设备兼容与发布验证

---

## 3. 用户测试

至少 10–20 个无 3D 扫描经验用户。

任务：
`扫描一个桌面物体，并完成模型导出。`

记录：
- 首次完成率
- 扫描时长
- 补扫次数
- 中途放弃率
- 模型缺口率
- cleanup 使用率
- 导出成功率

---

## 4. 推荐消费级 KPI

不要只看算法精度。

更应看：
- First Scan Success Rate
- Scan Completion Rate
- Average Rescan Count
- Review-to-Fix Success
- Model Export Success
- Session Recovery Success
- Project Reopen Success

---

## 5. 不建议在 vc183 同时推进

- 云账号
- 云同步
- 社交社区
- 云端 AI 重建
- AI 材质生成
- 商城
- 用户关注/评论

优先保证本地扫描体验完整。
