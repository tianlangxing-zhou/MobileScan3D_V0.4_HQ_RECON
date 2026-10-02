# MobileScan3D vc183 Acceptance Gates

## Gate A — 扫描成功率
- 首次用户能完成扫描
- 顶部/背面缺失率下降
- 追踪丢失后可恢复
- Guidance 不抖动

## Gate B — 扫描闭环
- Review 能发现明显缺失
- Continue Scan 返回正确方向
- session 不丢失
- 多轮补扫稳定

## Gate C — 模型清理
- 不误删主体
- 原始模型永久保留
- cleanup 可取消
- cleanup 失败可恢复
- 网格简化不严重破坏外形

## Gate D — 项目库
- 100+ 项目仍可打开
- 缩略图加载不卡 UI
- 删除有确认
- 异常项目可隔离
- checkpoint 可恢复

## Gate E — 自适应扫描
- 推荐距离稳定
- 光照提示不过度频繁
- 反光提示不误报到不可用程度
- 热状态下降级而非直接失败

## Gate F — 测量与导出
- 尺寸有明确单位
- 尺度不可信时显示“估算”
- GLB/OBJ/PLY 可被第三方工具打开
- 导出失败可重试

## 全局 Gate
- release build 可构建
- lint 0 error
- 无明显 ANR
- 10 分钟扫描无持续内存增长
- 后台/锁屏/来电恢复正常
- 低存储处理正确
- 3D viewer 退出按钮始终可见
- Android 返回手势可退出 viewer
