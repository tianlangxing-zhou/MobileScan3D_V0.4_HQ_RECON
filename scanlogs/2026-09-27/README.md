# 扫描日志 · 2026-09-27

本目录归档 **2026-09-27** 当天的全部扫描日志数据与修改日志，按类型分子目录。

## 背景
当日完成第十四轮迭代（V0.13.8，`787bbcb`）：基于您 19:03 / 19:05 两轮手持短扫的取证数据，
优化纹理覆盖率（关键帧吞吐 + fallback 原因诊断）。详见 `modification_logs/2026-09-27.md`。

## 子目录说明

### `scan_sessions/` — 手持扫描产出（GLB 重建结果 + 调试点云）
| 文件 | 大小 | 说明 |
|---|---|---|
| `scan_20260927_190525_25883.glb` | 2.96 MB | 19:05 会话，8470 三角面，手办 0.12m 正确尺寸，带 JPEG atlas |
| `scan_20260927_190525_25883_debug.ply` | 30 KB | 同上会话调试点云 |
| `scan_20260927_190333_13646.glb` | 3.71 MB | 19:03 会话，21435 三角面，0.40m 跨度（连同周边场景） |
| `scan_20260927_190333_13646_debug.ply` | 106 KB | 同上会话调试点云 |
| `scan_20260927_185456_96509_debug.ply` | 160 B | 18:54 空跑会话，点云为空（无 GLB） |

### `device_logs/` — 设备日志（adb logcat）
| 文件 | 大小 | 说明 |
|---|---|---|
| `postshot.logcat` | 1.96 MB | 19:01 全量 logcat 快照 |
| `postshot2.logcat` | 1.85 MB | 19:07 全量 logcat 快照 |
| `postshot_main.log` | 1.04 MB | main 缓冲区快照 |
| `postshot_crash.log` | 0 B | crash 缓冲区（空 = 无崩溃） |
| `session_19_live.logcat` | 2.27 KB | 19:04 流式采集（pid 3114） |
| `session_stream.log` | 381 B | 19:18 流式采集（tag 过滤，缓冲冲刷极快） |
| `session_1903.log` | 0 B | 19:03 会话提取（缓冲已被冲刷，空） |
| `session_1903_app.log` | 0 B | 同上 app 标签提取（空） |

> 注：main 缓冲区被相机 HAL 噪音冲刷极快，`-d` 快照常拿不到目标会话行；
> 后续改用 `logcat -s <tag>:V '*:S'` 流式采集（`session_stream.log`）以保住烘焙/导出诊断。

### `build_logs/` — 当日构建日志
| 文件 | 大小 | 说明 |
|---|---|---|
| `build_v0133.log` | 4.34 KB | V0.13.3 构建 |
| `build_v0134.log` | 6.59 KB | V0.13.4 构建 |
| `monitor_live.log` | 152 KB | 实时监控采集 |

### `modification_logs/` — 修改日志（workspace memory）
| 文件 | 大小 | 说明 |
|---|---|---|
| `2026-09-27.md` | 28.7 KB | 当日逐轮工作日志（含 V0.13.8 修改详情与验证数据） |
| `MEMORY.md` | 2.25 KB | 项目长期记忆（构建/调试硬约束、协作偏好） |

## 已排除（不纳入本目录）
- `devtest/run*.logcat`（09-26，36/36/20 MB）—— 昨日、超大
- `devtest/live.logcat`（97 MB）、`monitor/bugreport_1115.zip`（39 MB）—— 超 GitHub 单文件限制、非扫描专用
- `devtest/monitor/` 下 R01–R129 截图与 slice 日志（09:00–14:24 自动测试轮，海量噪声）—— 非用户手持扫描数据
- `devtest/_viz/`、`deepseek*`、`gptcat*` 等辅助产物

如需补录上述任何一类，告知即可单独追加。
