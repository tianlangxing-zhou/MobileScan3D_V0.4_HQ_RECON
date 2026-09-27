# scanlogs — 扫描日志与修改日志归档

每次完成一次真机数据采集后，将**当天**的全部扫描日志数据与修改日志，
按日期分目录上传到本仓库并推送到 GitHub，便于跨会话追溯与复盘。

## 目录约定（分日期序列）
```
scanlogs/
  2026-09-27/          # 每天一个目录，目录名 = 日期 (YYYY-MM-DD)
    README.md          # 当日索引：文件清单 + 背景 + 排除说明
    scan_sessions/     # 手持扫描产出：GLB 重建 + 调试点云 PLY
    device_logs/       # 设备日志：adb logcat 快照 / 流式采集
    build_logs/        # 当日 APK 构建日志
    modification_logs/ # 修改日志：.workbuddy/memory 当日日志 + MEMORY.md
  2026-09-28/          # 次日自动新建……
    ...
```

## 上传内容（每次采集后）
- **扫描日志数据**：`devtest/exports_v*/`（GLB/PLY）、`devtest/*.logcat`、`devtest/session_*.log`、构建日志
- **修改日志**：`E:/MobileScan3D/.workbuddy/memory/YYYY-MM-DD.md` 与 `MEMORY.md`

## 排除原则
- 单文件 > 50 MB 或昨日数据不纳入（GitHub 单文件限制 100 MB，且非当日扫描）
- 自动测试轮的海量截图 / slice 日志（R01–R129 等）视为诊断噪声，单独按需追加
- 始终保留源文件，本目录为**副本归档**，不删除 `devtest/` 与 `.workbuddy/memory/` 原始数据

## 操作命令（在 repo 根目录）
```bash
# 1. 建当日目录并复制（见各轮实际执行）
# 2. 提交推送
git add scanlogs/
git commit -m "docs(scanlogs): 归档 YYYY-MM-DD 扫描日志与修改日志"
git push origin main
```
