# V0.13.15 审查与修改结果

## 基线与范围

仓库：https://github.com/tianlangxing-zhou/MobileScan3D_V0.4_HQ_RECON

基线提交：`98e15aecc835648422b54366f4b28fb1ef8f0ade`。本轮检查近期跟踪恢复、AR 位姿容差、xatlas UV、纹理关键帧改动，并沿 Camera2 → 深度推理 → 标定/目标掩膜 → 场景和目标融合 → 实时显示/导出检查。修改记录基于源码分析；仓库内历史设备日志属于旧版本证据，不作为本次真机测试。

## 已修改的问题

| 问题 | 修改与结果 |
|---|---|
| 没有扫描距离设置，远景可进入场景与目标模型 | 加入持久化的 0.2–5 米最大距离设置，默认 1 米。按相机坐标射线计算欧氏距离，在 TSDF 和 surfel 写入前过滤，并覆盖实时深度显示。 |
| 双目稀疏融合可能绕过普通深度过滤 | 使用双目锚点自身的物理米制深度过滤，不把 VINS-world 数值直接当米。目标双目入口同时执行掩膜时序门控。 |
| 单目与物理距离尺度不同 | 稳定双目 world-per-meter 可用时执行尺度换算；否则使用 VINS 米制估计，界面明确标注估计值。距离估计误差不能由软件阈值消除。 |
| 暗光只拒绝 HQ 拍摄，没有主动补光 | 预览亮度分位数与曝光/ISO 联合检测。持续暗光约 0.9 秒后启用 Camera2 TORCH。灯在本次扫描中保持开启，防止亮度反馈使灯反复闪烁。 |
| 预览亮灯而 still 请求关灯会导致照片暗、曝光错配 | Preview、对焦单次请求、HQ JPEG/RAW 请求使用同一补光状态。补光切换后清除旧曝光稳定历史，等待约 1.2 秒与重新收敛。 |
| 相机不支持补光、请求失败 | 检查当前镜头 FLASH_INFO_AVAILABLE；补光请求抛异常时关闭补光重发预览参数，保留可继续扫描的相机请求，界面和反馈报告记录状态。 |
| 深度任务入队失败后 busy 无法释放 | 使用 AtomicBoolean 获取单任务资格，检查 Handler.post 返回值；异常、任务拒绝、正常完成均释放资格。错误不再静默吞掉。 |
| 慢推理时源帧可能被 60 帧环形缓存淘汰 | 派发深度任务前固定保存这一帧的 RGB/位姿，返回后优先按精确时间戳匹配。最多额外保留一帧，约 225 KiB RGB，不积压推理任务。 |
| 暂停/重开时旧深度可能迟到 | 保留并加强会话世代检查；后台退出时推进世代，前台状态不满足时不处理或提交新深度，避免旧结果进入当前模型。 |
| 活跃标定短暂不可用时目标深度可能再次缩放 | 标定开启时目标深度始终视为已经完成尺度换算，不再因当前拟合失效而乘上目标局部缩放。 |
| 目标 mask 用动态标定，TSDF 用冻结标定 | epoch 建立后，目标掩膜、presence 与融合使用同一冻结深度映射，减少掩膜随标定漂移而误拒绝。 |
| 中等标定漂移可能令 epoch 长期暂停 | 增加独立证据恢复：至少 20 个 VINS 有效样本，至少 20 个且 80% 样本与冻结映射相差不超过 8%，连续 6 帧后恢复原冻结映射。不会把新尺度直接混进旧几何，也不会仅凭等待时间强制恢复。 |
| 关闭深度标定后可能把合成假深度/原始逆深度写入几何 | 删除基于特征数编造的深度兜底；标定关闭时不允许把 inverse-depth 原值当作米制融合。 |
| 最新 REACQUIRING 分支每帧把计数重置为 45 | 删除该赋值，保留单调计数，让 270 帧扩展期限真正可达。当前重试周期为 1 帧，所以此修复主要解决超时失效，并非当前周期性检测缺失。 |
| presence=false 后只给一帧找回机会 | 恢复正常的 45 帧找回窗口，窗口内继续 NanoTrack 检测，短暂遮挡不立即变成永久 LOST。 |
| last-good box 找回可能把身份“不可判断”当成匹配 | 无 Nano 分数的 last-good 分支要求实际 identity score 达标，再采用候选框；不能用 -1 作为恢复证据。 |
| presence 不成立时仍显示绿色当前帧点云 | 显示也要求 presence 和掩膜时序有效，减少“已生成目标”的误导。 |
| 连拍先失败再成功时，成功回调只计成功数，可能永不收尾 | 改为按帧编号去重，统一统计成功与失败的终态总数；顺序无关，重复回调不提前完成。 |
| HAL 丢失终态回调或序列被取消，HQ 忙状态可能永久保持 | 增加取消回调处理与 5 秒终态超时，然后进入已有图像配对/补拍路径。不会生成缺帧的虚假融合结果。 |
| 旧扫描的 HQ 融合回调可能污染新扫描、释放新扫描 busy | 发布前检查当前 burst 身份与 sessionId；旧结果丢弃。收尾只清理自己的状态；发布失败也通过 finally 清理。关闭相机会清除中断的 burst 状态。 |
| 设置入口被后添加的顶部信息栏遮挡 | 把设置按钮置于上层，确保距离设置可点击。 |
| 相机参数在 UI/相机线程交错提交 | 统一调度到相机 Handler。新增深度工作状态、范围和补光诊断。 |

范围只影响新的几何观测，不限制用于定位的远景特征。远景特征可帮助定位，但不会因此直接变成新模型表面。已有模型和历史扫描包保持原有内容。

## 近期改动的审查结论与边界

- 保留 xatlas 按索引构建有效面、跳过未 chart 面的修复，未恢复遇到孤儿顶点就整批 fallback 的旧行为；该文件通过宿主语法检查。
- 保留 RGB 快照分辨率与纹理关键帧改进，GLB 导出回归仍通过。
- 500ms AR 位姿查询是最近邻近似，并非精确同步；纠正了源码中“不会拿错时刻”的过度保证。快速运动时的位姿时间差仍需设备测量。
- 上一版本的“AR 上下方向”修改是诊断探针。仓库没有足以确定该问题根因的本轮真机轴向数据，本轮未随意翻转纹理或世界坐标，也不能声称真机方向问题已被验证修复。
- VINS 失锁、几何不可信、目标身份不匹配时仍保留安全门控。目标确实离开且超时后可重新框选；没有以强行融合背景来维持点数增长。
- 有限内存预算仍存在：满预算时已有体素可以继续更新，但不能无限添加全新区域。点数不增长不等于停止处理。

## 已执行自检

| 检查 | 结果与实际范围 |
|---|---|
| 新距离/恢复策略 C++ 回归 | 通过：近/远阈值、边界、斜射线欧氏距离、world-to-meter 换算、NaN/Inf/非法输入、冻结标定独立证据通过/拒绝。 |
| 近距离 TSDF → 网格 | 通过：同一合成帧左面约 0.503m、右面 2m，最大距离 1m；提取网格只保留近面，顶点深度误差 <2mm。这是合成几何测试，不是手机精度承诺。 |
| 自动补光 Kotlin/JVM 回归 | 通过：持续暗光延时、亮光打断、开灯保持、重置、帧间隔中断、AE 高增益场景、非法亮度。 |
| 连拍终态 Kotlin/JVM 回归 | 通过：5 帧全部 120 种到达顺序 × 32 种成功/失败组合 = 3,840 组；包含重复终态与无效索引。 |
| 既有 GLB 回归 | 通过：5 种 GLB、索引/边界/缓冲对齐、嵌入 JPEG、非法网格与逗号 locale。 |
| 既有几何回归 | 通过：投影内参、固定逆深度标定、错长/无效样本、斜率翻转拒绝、世界坐标、TSDF 满预算更新、平面网格。 |
| 既有 DEPTH16 JVM 回归 | 通过：位域、置信度、行/像素 padding、buffer 起点/边界、独立帧数组。 |
| C++ 修改文件编译器语法检查 | native_engine.cpp、object_tracker.cpp，以及复核的 uv_unwrap.cpp 通过 g++ -fsyntax-only；使用 OpenCV 4.12/JNI/Vulkan 头文件、项目 Eigen 与 Android log 声明桩。不等同 Android ABI 链接。 |
| Kotlin 宿主类型检查 | 全部生产 Kotlin 源码通过 Kotlin 2.0.21 编译器检查，使用 Android API 36 jar，AndroidX Activity、TFLite、BuildConfig 使用最小声明桩；仅有既有弃用警告。不等同真实依赖的 Gradle APK 构建。 |
| ASan/UBSan | C++ 回归通过。宿主 /proc 限制使 LeakSanitizer 不可用，关闭泄漏检测；不声明内存泄漏检测通过。 |
| 交付检查 | git diff --check、覆盖后 SHA-256 校验、ZIP CRC 和 patch 可应用性检查。 |

没有 Android SDK/NDK 与项目所需 Ceres/OpenCV 本地构建配置，也没有连接手机。本轮没有产出或验证 APK，没有把真机帧率、发热、光照效果、AR 方向与长时稳定性记为通过。

## 本地复验

在工程根目录执行：

```bash
python tools/review_tests/run.py
python tools/review_tests/run_geometry.py
python tools/review_tests/run_scan_policy.py
kotlinc app/src/main/java/com/mobilescan3d/AutoLightPolicy.kt tools/review_tests/AutoLightRegression.kt -include-runtime -d auto-light-test.jar
java -jar auto-light-test.jar
kotlinc app/src/main/java/com/mobilescan3d/BurstCompletion.kt tools/review_tests/BurstCompletionRegression.kt -include-runtime -d burst-test.jar
java -jar burst-test.jar
kotlinc app/src/main/java/com/mobilescan3d/depth/Depth16Decoder.kt tools/review_tests/Depth16Regression.kt -include-runtime -d depth16-test.jar
java -jar depth16-test.jar
python tools/review_tests/verify_overlay.py
```

C++ 回归需要 g++，GLB 检查还需 Python Pillow；JVM 检查需 Kotlin/JDK。新增回归源码和运行脚本均包含在覆盖包里。

## 真机验收步骤

1. Clean/Rebuild 后安装，确认标题版本 `0.13.15-near-scan`。设置最大距离 1 米，前景约 0.4–0.8 米，背景至少 2 米；新扫描并检查点云、GLB/PLY 中没有背景表面。再用 0.5/1.5 米分别复验，注意标定误差。
2. 暗处开始扫描，检查约 0.9 秒后补光开启、HQ 照片曝光重新稳定；停扫/切后台关灯，返回继续扫描后重新检测。检查没有灯的镜头和关闭自动补光设置。
3. 连续扫描 2–3 分钟，短暂遮挡后移回；查看点云是否恢复、定位不可信时是否明确提示。对已扫面点数可以不增，但融合帧/合并计数应继续变化。
4. HQ 连拍中切后台再返回、停止后立即重开、生成模型后再次扫描；确认无永久 busy、无旧模型纹理混入。
5. 检查 AR 方向、手机移动时模型跟随、GLB 外部查看器中的朝向与纹理。若仍有方向异常，保留 ArAxesProbe、ArPoseProbe 与完整反馈报告定位。

## API 依据

Camera2 FLASH_MODE_TORCH 配合 AE_MODE_ON/OFF，可用于预览与 still capture，需 FLASH_INFO_AVAILABLE：
https://developer.android.com/reference/android/hardware/camera2/CaptureRequest#FLASH_MODE

该依据仅说明接口行为，不代替当前设备 HAL 实测。
