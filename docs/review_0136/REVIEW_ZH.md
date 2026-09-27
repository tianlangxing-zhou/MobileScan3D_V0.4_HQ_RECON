# V0.13.6 审查修复说明

基线仓库：https://github.com/tianlangxing-zhou/MobileScan3D_V0.4_HQ_RECON

基线：`2cafb5382e21eec4795255309c413d14efadae5b`（V0.13.5）。审查日期：2026-09-27。交付前重新查询 main，仍为此提交。修改仅在交付包中，未推送 GitHub。

本轮重点核对 V0.13.4、V0.13.5 的变更，并沿相机→深度→标定→分割→TSDF→网格→显示→导出路径追查。下面列出已实际修改的内容；这不是“所有环境绝对无缺陷”的保证。

## 已修复的问题

| # | 问题与影响 | 修复 |
|---|---|---|
| 1 | 深度融合用 160×90 RGB 缓存尺寸缩放原相机内参，造成几何拉伸 | 从原始相机输入尺寸计算深度内参，全场景和目标融合统一使用 |
| 2 | 当前目标调试点仍使用原相机分辨率内参 | 同步缩放到深度栅格，修复画面叠加错位 |
| 3 | 时序一致性重投影仍使用未缩放的内参 | 与深度融合使用同一套深度内参 |
| 4 | 256×256 深度仍按步长 4 积分，有效采样约为 64×64 | 默认每深度像素采样；保留现有可配置步长接口 |
| 5 | Surfel 只沿 160×90 RGB 缓存取点，丢掉部分深度细节 | 遍历深度栅格，独立查 RGB 缓存颜色 |
| 6 | 体素块配额满后整帧返回，已有表面也停止细化 | 只跳过无法分配的新块，继续更新已有块，配额不增长 |
| 7 | TSDF 值在射线落点计算，却在体素格点提取，造成表面位置偏移 | 按实际存储格点在相机系的 Z 计算有符号距离 |
| 8 | 四面体对角边与轴向边缓存键冲突，错误合并顶点 | 用完整方向位掩码区分 7 种边，同步修正颜色与法线取样端点 |
| 9 | 未观测体素被当成正 TSDF，生成额外背壳 | 仅对四个顶点均已观测的四面体提面，未知空间不再伪造为表面外部 |
| 10 | Surfel 把世界 Z 当相机深度，删除负 Z 或较大世界 Z 的合法点 | 相机距离由调用方判断，Surfel 校验有限世界坐标与置信度 |
| 11 | 深度表示枚举没有送进 JNI，标定仍在两种模型间逐帧切换 | Kotlin/JNI/C++ 传递固定表示，支持固定线性或固定逆深度拟合 |
| 12 | 当前 MiDaS 模型被声明为线性深度，文档逆深度公式也不一致 | 根据当前模型输出层与 MiDaS 约定设为 INVERSE_DEPTH，统一为 z=1/(a·q+b) |
| 13 | 稀疏标定使用最新观测，可能与异步深度帧错配 | 在 VINS 窗口按深度传感器时间戳找观测；35ms 内无匹配则不凑配对 |
| 14 | 标定状态在主状态锁外更新，可与会话复位、诊断读取竞争 | 先独立获取 VINS 样本，再在 gStateMutex 下访问标定与融合状态 |
| 15 | 标定只检查 raw 数组长度，少量有效样本也可能越过门槛 | 同时检查配对长度、过滤后有效样本数、最大采样配置 |
| 16 | 物理无效的拟合预测被直接跳过，评分可能过于乐观 | 无效预测计入惩罚，不再伪装成好内点；拒绝平滑跨越斜率正负翻转 |
| 17 | Epoch 预热比较了每帧不同的场景深度中位数，移动容易影响稳定判断 | 预热阶段使用固定 raw 参考点比较映射 |
| 18 | 原始模型 q 直接进入按距离设阈值的目标分割 | 分割改用标定深度；预热期间保留外观跟踪，不用未标定 mask 判目标丢失 |
| 19 | mask 多档收紧受固定下限限制；闭运算可补回背景；过扩张结果未阻止融合 | 下限随档位收紧，结果与候选深度带相交，过扩张结果不进入 presence/fusion |
| 20 | mask 失效后目标调试点可能继续显示上一帧 | 未满足显示条件时清空旧调试点 |
| 21 | 纹理遮挡深度在屏幕空间线性插值 Z，倾斜面遮挡判断错误 | 改为插值 1/Z 后取倒数 |
| 22 | mat3 法线矩阵使用 glUniformMatrix4fv 上传 | 改为 glUniformMatrix3fv |
| 23 | 网格顶点、索引、atlas 分别发布，GL 线程可能混读不同版本；完成旧上传可能吞掉新版本 | 两种 Renderer 使用一次性发布的数据快照，上传只确认所取快照 |
| 24 | GL 上下文重建后旧句柄仍被使用，纹理 CPU 数据也已丢失 | 重建句柄并保留待重新上传的模型快照；不删除新上下文中可能已复用的旧数值句柄 |
| 25 | 预览、LIVE 网格、PLY 回调和 ExportManager 缓存可能回写新会话 | 加会话令牌与缓存世代校验，旧任务不得发布结果 |
| 26 | 新会话 nativeCreate 与已经完成推理的深度提交可能竞争；TFLite 在 UI 线程关闭 | 用短临界区串行 native 深度提交/会话重置，销毁使旧任务失效，provider 在深度线程关闭 |
| 27 | DEPTH16 将高 3 位置信度当成距离，并假设另有置信度 plane | 独立解码低 13 位毫米距离和高 3 位置信度，零置信度置无效 |
| 28 | DEPTH16 缓冲区复用、stride/剩余长度处理不足，旧相机回调可污染新生命周期 | 独立发布帧数组，支持 buffer position 和行/像素 padding，校验边界并拒绝旧回调 |
| 29 | 模型资产文件描述符未完整关闭；版本标记停留 0.13.0；无 Git 时错误文本可能进入构建标记 | 关闭 AssetFileDescriptor，版本改为 0.13.6-review，Git 失败返回 unknown |
| 30 | 用户在深度标定或融合暂停时看到空白却没有操作指引 | 增加标定等待、模型不可用和深度不稳定时的中文提示 |

## 自检结果

| 检查 | 结果与范围 |
|---|---|
| GLB 回归 | 通过：无/有法线与颜色的 4 种组合、内嵌 JPEG 纹理、索引、JSON/二进制对齐、浮点边界、非法网格拒绝、逗号 locale |
| 几何回归 | 通过：非正方形原相机到 256² 的投影一致性、固定逆深度拟合、无效样本/错长数组/斜率翻转拒绝 |
| TSDF/网格回归 | 通过：格点 SDF、满预算仍能更新、无效置信度拒绝、负世界坐标；输入平面 1.007m，提取结果全部顶点误差小于 2mm，内部边具有两个邻接三角面 |
| 内存/未定义行为检查 | 上述 C++ 回归使用 ASan+UBSan，通过；受执行环境限制关闭 LeakSanitizer，因此没有完成泄漏检测 |
| DEPTH16 JVM 回归 | 通过：真实 Kotlin 解码器编译运行，检查位域、置信度、非零 buffer 起点、行/像素 padding、截断缓冲区、独立帧数组 |
| C++ 编译器语法检查 | native_engine.cpp、vins_bridge.cpp、target_mask_engine.cpp、texture_baker.cpp 通过 g++ -fsyntax-only；使用 OpenCV 4.12/Ceres 2.2/Vulkan/JNI 头，Android 日志和生成配置采用宿主声明桩；不等同 Android ABI 链接 |
| Kotlin 语法检查 | 全部 17 个生产 Kotlin 源文件及 app/build.gradle.kts 通过语法解析；DEPTH16 部分另经 Kotlin 2.0.21 编译执行；其余 Android Kotlin 未完整类型检查 |
| 差异与覆盖包 | git diff --check、ZIP CRC、文件 SHA-256 及覆盖后校验脚本检查 |

## 验证边界与取舍

- 没有完整 Android SDK/NDK/Ceres/OpenCV 工程配置，没有 APK 构建或设备运行。本次不能承诺“完全无问题”，也没有声称真机精度或帧率已提升到某个数值。
- 去除伪背壳后，未扫描面可能更明显地保持开放，这是正确表示未知区域；应补拍更多视角，而不是用无观测数据制造封闭外壳。
- 更密集积分和保留可恢复的 CPU 模型快照会增加运算/内存开销；OBJECT_HQ 的持续帧率、发热与内存峰值需真机确认。现有 OBJECT_FAST/ROOM 原生档位仍可使用。
- DEPTH16 解码修复不代表独立硬件深度相机已经完成 RGB 内外参、时间同步和外参配准；默认仍使用单目深度，不自动启用未对齐硬件数据。
- 当前模型输出张量名为 `midas_net_custom/sequential/re_lu_9/Relu`，形状 `[1,256,256,1]`。据此及 MiDaS 官方语义固定为逆深度；若自行更换模型，应同时调整表示定义和预处理。
- VINS 尺度对齐并非测量认证。新提示不替代实际质量验收。

## 本地复验

在工程根目录：

```bash
python tools/review_tests/run.py
python tools/review_tests/run_geometry.py
kotlinc app/src/main/java/com/mobilescan3d/depth/Depth16Decoder.kt tools/review_tests/Depth16Regression.kt -include-runtime -d depth16-test.jar
java -jar depth16-test.jar
python tools/review_tests/verify_overlay.py
```

前两项需要 g++；GLB 检查还需要 Python Pillow；第三项需要 Kotlin/JDK。覆盖校验不依赖编译工具。

在原 Android 构建环境执行 Clean/Rebuild（或 `gradlew.bat clean :app:assembleDebug`）。Kotlin/JNI 接口本轮一起修改，不能只替换 Kotlin 或仅复用旧 .so。

真机重点：同一物体绕拍、返回原位、快速停扫重开、预览构建中重开、熄屏/后台后恢复、导出 GLB 在查看器检查纹理、满体素预算继续扫已有面。记录帧率、发热、内存和 VINS/标定诊断，确认错误日志后再用于重要采集。

## 规范依据

- MiDaS 官方 PyTorch Hub：相对逆深度输出。https://pytorch.org/hub/intelisl_midas_v2/
- Android ImageFormat.DEPTH16：13 位距离和 3 位置信度。https://developer.android.com/reference/android/graphics/ImageFormat#DEPTH16

以上规范只支持接口/模型语义判断，不能替代当前设备实测。
