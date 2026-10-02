# vc18402 缺失功能收口

基线：vc18401 / 0.15.4-model-cleanup-camera-fix

本补丁只补之前核对为缺失/部分缺失/有回归风险的项目，不重做 vc183.1–vc183.3：
- Viewer SurfaceView 独立窗口退出兜底（保留 OnePlus `setZOrderOnTop(true)`）
- 我的模型
- 两点测量
- OBJ + Android 系统分享
- 低光/过曝/低纹理/低可重建性风险提示
- 发布 checklist / 许可证 Gate / .gitignore

不修改 TSDF、VINS、深度网络或 vc183.3 模型优化数学。
