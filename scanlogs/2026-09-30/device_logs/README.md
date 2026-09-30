# device_logs（2026-09-30）

## 本目录内容

| 文件 | 说明 |
|---|---|
| `bakeDiag_20260930.log` | 从当日 logcat 提取的**全部 `bakeDiag` 行**（46 行），含 17:12（vc150，对照）与 17:45（vc152，验收）两次烘焙的汇总、逐关键帧拒绝统计、逐关键帧位姿/内参。 |

`bakeDiag_20260930.log` 结构：
- 1 行汇总：`texTris / fbTris / painted / hq / kf`
- 11 行拒绝统计：`project / border / depth / facing / area / pass`
- 11 行关键帧：`4096x3072 fx/fy / t / fwd / q / verts / inFront / inFrame / bbox`

## 排除说明

当日原始 logcat 快照为 **`D:\WorkBuddy_3DGS\_scanlog9.txt`（185,303,396 B）**，
超过本仓库归档原则的「单文件 50 MB」上限（且 GitHub 单文件限制 100 MB），**未纳入**。

原始文件保留在本机，需要回溯时直接在本机检索。快速提取方式（Python 流式，避免整读 185MB）：

```python
with open(r'D:\WorkBuddy_3DGS\_scanlog9.txt', encoding='utf-8', errors='replace') as f:
    for line in f:
        if 'bakeDiag' in line or 'FusionDiag' in line:
            print(line.rstrip())
```

> 注意：logcat 环缓冲（170 MB+）会冲掉稀疏诊断。本次 `TargetDiag` 全 0 条（`haveTi=false`），
> 无法从日志确认是「从未打印」还是「被缓冲冲掉」——按代码路径判定为前者（未锁目标）。
