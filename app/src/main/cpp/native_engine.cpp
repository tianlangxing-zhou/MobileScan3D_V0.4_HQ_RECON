#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>
#include <sstream>
#include <string>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <vector>
#include <deque>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <utility>

#include "vins/vins_bridge.h"
#include "vins/pose_prediction.h"
#include "vins/estimator/feature_manager.h"
#include "surfel_engine.h"
#include "vio_engine.h"
#include "depth_fusion.h"
#include "ai_quality.h"
#include "tsdf_engine.h"
#include "ai_backend.h"
#include "keyframe_engine.h"
#include "object_tracker.h"
#include "target_mask_engine.h"
#include "depth_calib.h"
#include "camera_distance.h"
#include "depth_confidence.h"
#include "depth_refinement.h"
#include "color_contours.h"
#include "fusion_guard.h"
#include "frame_icp.h"
#include "depth_geometry.h"
#include "scan_policy.h"
#include "fusion_evidence.h"
#include "target_mask_temporal.h"
#include "stereo_anchor_bridge.h"
#include "mesh/mesh_engine.h"
#include "mesh/mesh_registration.h"
#include "mesh/hard_surface.h"
#include "export/gltf_exporter.h"
#include "v06/mesh_postprocess.h"
#include "v06/uv_unwrap.h"
#include "v06/texture_baker.h"
#include "v06/textured_glb_exporter.h"
#include "ar_textured_asset.h"
#include "persistent_relocalizer.h"
#include <fstream>
#include <opencv2/imgcodecs.hpp>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MobileScan3D", __VA_ARGS__)
#define TARGET_DEPTH_FILTER_ENABLED 0

// 全局状态被 4 个线程并发访问：IMU 线程（nativeOnImu）、相机线程
// （nativeOnCameraFrame）、深度线程（nativeOnDepthMap）、GL 渲染/UI 线程
// （nativeGetGaussians / nativeGetStats）。原代码完全无锁，snaps 的 deque
// 迭代器失效和 g_ 的 vector 扩容都是未定义行为（崩溃/花屏根因）。
static std::mutex gStateMutex;
// Creation/destruction must also exclude the camera's unlocked VINS stage.
// Lock order is camera callback -> state; never acquire these in reverse.
static std::mutex gCameraCallbackMutex;
static uint64_t gNativeGeneration = 0; // guarded by gStateMutex
static float scanMaxDistanceMeters = 1.f;
static float scanWorldPerMeter = 1.f; // VINS estimate until stereo scale is validated.
static uint64_t rangeRejectedPixels = 0;
static uint64_t frozenEvidenceRecoveries = 0;
static int frozenEvidenceStreak = 0;
static scan_policy::FusionEvidence frozenEvidenceWindow;

static NullAiBackend aiBackend;
static SurfelEngine g;
static VioEngine vio;
static DepthFusion df;
static AiQualityEngine ai;
static TsdfEngine tsdf;
// ---------------------------------------------------------------- 目标专用模型
// 与全场景的 g / tsdf **物理隔离**：天花板、墙、地面根本没有进入这两个容器的
// 代码路径。比「给 g 加一个过滤标志」强得多 —— 后者只要有一个调用点漏掉，
// 背景就又会漏进来，而且很难在实机上发现。
// Renderer 只读 targetG，PLY 只导 targetTsdf。
static SurfelEngine targetG;
static TsdfEngine targetTsdf;
static TargetMaskEngine targetMaskEngine;

// ---------------------------------------------------------------- mesh 管线
// 评审 P0-3：旧导出只有顶点、没有三角面，那不是 AR 模型。这里补一条真正的
//   TSDF -> Marching Tetrahedra -> 去小分量/平滑/简化 -> GLB
// 链路。构建是一次性操作（用户点「生成网格」或停止扫描时），不是每帧。
static MeshEngine meshEngine;
static bool meshHasHardEdges = false;
static HardSurfaceStats hardSurfaceStats;
static MeshBuildStats meshStats;
static int meshQuality = 1;            // 0 = preview / 1 = normal / 2 = hq
static bool meshDirty = true;          // 体素场变了置脏，避免重复构建
static uint64_t meshBuilds = 0;
static uint64_t meshDirtyMarks = 0;
/** 顶点交错布局：x,y,z, nx,ny,nz, r,g,b —— 9 个 float 一个顶点（与 Kotlin 契约一致）。 */
static constexpr int MESH_VERTEX_FLOATS = 9;

// ---------------------------------------------------------------- V0.6 纹理管线
// 与 mesh 管线分开：这些东西只在「停扫导出」时用一次（几百毫秒到数秒），
// 每帧路径完全不碰。HQ 关键帧由 HqCaptureController 在 HQ 合成 / 降级单帧
// 成功后注册进来，和网格一样按会话清理。
//
// 锁：**统一用 gStateMutex，不再引入第二把 gTextureMutex**。resetMeshPipeline()
// 是在 gStateMutex 下被调用的，再加一把纹理锁就形成 gStateMutex -> gTextureMutex
// 的嵌套；而烘焙路径若反向取锁就是死锁。共用一把锁的代价可以忽略 ——
// 纹理状态只在会话开始/结束与停扫导出时变化，不在每帧路径上。
static MeshPostProcessStats meshCleanupStats;
static std::vector<TextureKeyframe> gTextureKeyframes;
static TextureBakeStats textureBakeStats;
static UvUnwrapStats uvUnwrapStats;
static std::mutex gArAssetMutex;
static ArTexturedAsset gArTexturedAsset;
static PersistentRelocalizer gPersistentRelocalizer;
static std::mutex gRelocFrameMutex;
static std::vector<std::uint8_t> gRelocGray;
static std::uint64_t gRelocFrameTimestampNs = 0u;
static std::uint64_t gRelocConsumedTimestampNs = 0u;
static float gRelocFrameRwc[9] = {1,0,0, 0,1,0, 0,0,1};
static float gRelocFrameTwc[3] = {0,0,0};
static int gV07InputW = 1;
static int gV07InputH = 1;

// --------------------------------------------------------------- 深度标定
// 评审 P0-4：把「单一 median ratio」升级成 MAD 剔除 + Huber IRLS + EMA 的
// 鲁棒回归，并同时支持线性 / 逆深度两种模型。
//
// 标定目标刻意选 **VINS 自身尺度**而不是米制：TSDF 融合用的位姿（s.R / s.t）
// 就是 VINS world 下的。深度与位姿必须处在同一尺度，否则点云会整体膨胀或
// 缩小 —— 这正是评审说的「模型膨胀/缩小」。旧代码里深度是「按会话 min/max
// 映射到 [0.4, 6]m 的假米制」，与 VINS 位姿根本不是一个尺度。
static DepthCalibrator depthCalibrator;
static bool depthCalibrationEnabled = true;
static float lastCalibScale = 1.f;
static float lastCalibShift = 0.f;
static float lastCalibConfidence = 0.f;
static int lastCalibSamples = 0;
static bool lastCalibValid = false;
static bool lastCalibInverse = false;
// V0.13.25: 标定健康度（参考深度相对纵深 / 标定输出 1/z 分辨力），用于
// 判定「模型发平」是否源于拟合样本纵深不足。
static float lastCalibRefSpanRel = 0.f;
static float lastCalibOutInvZSpan = 0.f;
// V0.13.35: 标定输出米制深度相对跨度（是否把场景压平的直接度量）。
static float lastCalibOutZSpanRel = 0.f;
static uint64_t calibFrames = 0;
static uint64_t calibRejectFrames = 0;
// V0.10: exact depth field entering TSDF after calibration.
static bool lastFusionDepthCalibrated = false;
static camera_distance::Estimate cameraDistance;
static uint64_t cameraDistanceTs = 0;
static int cameraDistanceSource = 0; // 1: VINS-scale estimate, 2: stereo-verified metres
static float lastFusionDepthMin = 0.f;
static float lastFusionDepthMax = 0.f;
static float lastFusionDepthMean = 0.f;
static uint64_t lastFusionDepthValid = 0;
static uint64_t lastFusionDepthConversionFallback = 0;
static constexpr int kMaxCalibSamples = 256;

// Fusion calibration is immutable within a scan. Drift suspends new writes;
// only explicit new-scan / new-target actions may erase geometry.
// Independent sparse evidence or recovery of the fit can resume the same epoch.
static constexpr int kEpochStableFrames = 8;
static constexpr int kEpochCalibLossFrames = 5;
static constexpr float kEpochRebuildRatio = 0.08f;
static constexpr float kEpochDriftSuspendRatio = 0.20f;  // V0.13.19.4：暂停新融合的漂移门限（独立于开启稳定性门槛 8%）
static constexpr float kEpochDriftResumeRatio = 0.10f;   // V0.13.32：恢复门限（滞回，低于 suspend 的 20%，避免边界抖动）
/** V0.13：连续多少帧漂移超标 -> 暂停新融合（几何保持不动）。 */
static constexpr int kEpochDriftSuspendFrames = 10;
/** V0.13：漂移连续恢复正常多少帧 -> 解除暂停。 */
static constexpr int kEpochResumeFrames = 6;
/** V0.13：极端映射差诊断阈值；不再触发几何清空。 */
static constexpr float kEpochCatastrophicRatio = 0.75f;
/** V0.13：极端失效判据 2（必须持续这么多帧才算真的塌了，不是抖动）。 */
static constexpr int kEpochCatastrophicFrames = 30;
/**
 * V0.13.34：灾难暂停的限时自动恢复（10s）。
 * 实机证据（PLK110 vc172 扫描）：live 标定是场景相关噪声（spanRel 0.35~0.74、
 * scale 0.0006↔0.0023 乱跳），用它对比冻结映射产生的「漂移」绝大多数是假漂移；
 * 旧实现的 20% 挂起 / 10% 恢复滞回死区让 susp 一旦置位就再也解不开
 * （FusionDiag depthFrames=60 fused=2 gated=57）。灾难门保留三重条件
 * （>75% 且连续 10 帧且无冻结证据），误报率低；即便如此，暂停超过 10s
 * 仍强制恢复 —— 冻结链路自洽，停写只会得到半截扫描。
 */
static constexpr int64_t kEpochCatastrophicMaxSuspendNs = 10'000'000'000LL;
// V0.13.30：live 标定器 EMA 平滑（漂移检测的鲁棒参考）+ 稳定性门。
// VC171: only fresh accepted fits update this diagnostic. Recovery also needs
// current geometric evidence; EMA convergence cannot authorize a scale change.
static constexpr float kEpochLiveEmaNew = 0.15f;        // EMA 新观测权重
static constexpr float kEpochLiveVarSuspendRel = 0.05f; // 相对方差门限（stddev<~22% 才认作稳定）
static bool epochActive = false;
static DepthCalibration epochCalib{};
static float epochRefRaw = 0.f;
static float epochRefZ = 0.f;
// V0.13.26：epoch 冻结那一刻的标定成熟度。用于判定「冻结过早」——
// 若冻结时 samples/confidence 很低而之后 live 标定大幅改善，frozen/live 会
// 长期背离（实测 frozen shift=0.7619 -> live shift=2.7906，driftRel=0.7230），
// epoch 被漂移门反复挂起。
static int epochFrozenSamples = 0;
static float epochFrozenConfidence = 0.f;
static int epochStableStreak = 0;
// 连续「坏」帧：漂移超标 **或** 标定不可用。两类坏帧共用同一个 streak，
// 与外部补丁包的 fusionEpochBadStreak 语义一致。
static int epochBadStreak = 0;
static uint64_t epochDriftRejects = 0;
static float epochLastDriftRel = 0.f;
static bool haveEpochPrevZ = false;
static float epochPrevZ = 0.f;
// V0.12: 诊断用 —— 换目标时**保留**标定的次数 vs 真正整会话重置的次数。
// 「点云一直是 0」的排查里，这两个数字能一眼区分
// 「标定被锁目标清掉了」和「标定本来就没收敛」。
static uint64_t depthCalibKeptAcrossTarget = 0;
static uint64_t depthCalibFullResets = 0;
static uint64_t epochIndex = 0;
static uint64_t epochOpens = 0;
static uint64_t epochRebuilds = 0;
static uint64_t fusionFusedFrames = 0;
static uint64_t fusionGatedFrames = 0;
// V0.13 Sticky 状态：冻结之后「暂停融合 / 恢复融合」而不是清几何。
static bool epochSuspended = false;
static int epochSuspendStreak = 0;
static int epochResumeStreak = 0;
static int epochCatastrophicStreak = 0;
static uint64_t epochSuspendEvents = 0;
static uint64_t epochCatastrophicRebuilds = 0;
static uint64_t fusionSuspendedFrames = 0;
// V0.13.34：灾难暂停的计时锚点与限时恢复计数（见 kEpochCatastrophicMaxSuspendNs）。
static int64_t epochSuspendedSinceNs = 0;
static uint64_t epochCatastrophicTimeoutResumes = 0;
// V0.13.30：epoch 漂移检测用的 live 标定器 EMA（消除单帧噪声误判永久 susp）。
static float epochLiveEmaScale = 1.f;
static float epochLiveEmaShift = 0.f;
static bool  epochLiveEmaInvDepth = false;
static bool  epochLiveEmaInit = false;
static float epochLiveEmaVar = 0.f;   // (instant - ema)^2 的 EMA，用于稳定性判据
static uint64_t epochReanchors = 0;   // V0.13.36 复用：灾难超时恢复时对健康 live 的受控重锚（VC171 曾无条件重锚导致不安全而禁用；现要求 usable + 正 scale + 工作点分歧 >75% 三重条件，见恢复分支）。

// ============================================================================
//  V0.13.4 深度数值域（归一化映射）追踪
// ============================================================================
//
// 非米制深度 provider 的输出是网络原始值 q 的仿射 `d = A*q + B`，A/B 由
// 会话级 min/max 决定 —— 扫描推进时看到更近/更远的表面，范围会扩张，
// 于是同一个 q 被映射成不同的 d。
//
// 而 Fusion Epoch 冻结的是 (a,b)：`z = a*d + b`。d 的语义悄悄变了而 (a,b)
// 不变，几何就整体膨胀/收缩 —— 这是「扫到后面物体越来越大」的根因。
//
// 修法不是「不许范围扩张」（那会把远处表面压成常数），而是**每次扩张都
// 显式重参数化标定**：见 DepthCalibration::reparameterizeLinearInput。
// 下面是这套机制的基线状态与统计。
static float gDepthNormA = 0.f;
static float gDepthNormB = 0.f;
static uint64_t gDepthNormVersion = 0;
static bool gHaveDepthNorm = false;
/** 因数值域变化而重参数化标定的次数（诊断：漂移是否来自数值域变化）。 */
static uint64_t gDepthNormReparams = 0;
/** 上一次重参数化时 a 的相对变化量（诊断：单次跳变幅度）。 */
static float gDepthNormLastScaleRel = 0.f;
/** epoch 已冻结但数值域仍在变化的次数（诊断：冻结是否真的生效）。 */
static uint64_t gDepthNormChangesWhileEpoch = 0;

static void resetFusionEpoch() {
    frozenEvidenceWindow.reset();
    frozenEvidenceStreak = 0;
    epochActive = false;
    epochCalib = DepthCalibration{};
    epochRefRaw = 0.f;
    epochRefZ = 0.f;
    epochFrozenSamples = 0;
    epochFrozenConfidence = 0.f;
    epochStableStreak = 0;
    haveEpochPrevZ = false;
    epochPrevZ = 0.f;
    epochBadStreak = 0;
    epochDriftRejects = 0;
    epochLastDriftRel = 0.f;
    epochIndex = 0;
    epochOpens = 0;
    epochRebuilds = 0;
    fusionFusedFrames = 0;
    fusionGatedFrames = 0;
    epochSuspended = false;
    epochSuspendStreak = 0;
    epochResumeStreak = 0;
    epochCatastrophicStreak = 0;
    epochSuspendEvents = 0;
    epochCatastrophicRebuilds = 0;
    fusionSuspendedFrames = 0;
    epochSuspendedSinceNs = 0;
    epochCatastrophicTimeoutResumes = 0;
    epochLiveEmaScale = 1.f;
    epochLiveEmaShift = 0.f;
    epochLiveEmaInvDepth = false;
    epochLiveEmaInit = false;
    epochLiveEmaVar = 0.f;
    epochReanchors = 0;
}

/**
 * raw 深度的稀疏采样中位数。**只**用来给 epoch 挑一个参考工作点：
 * 必须是一个「与场景无关、只与映射有关」的输入值，才能用
 * `toMetric(raw)` 的前后差判断标定是否漂了。
 */
static float rawDepthSampleMedian(const float* d, int w, int h, bool inverse) {
    if (!d || w < 4 || h < 4) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<float> s;
    s.reserve(static_cast<size_t>(w / 4 + 1) * static_cast<size_t>(h / 4 + 1));
    for (int y = 2; y < h; y += 4) {
        for (int x = 2; x < w; x += 4) {
            const float v = d[static_cast<size_t>(y) * w + x];
            if (depth_refinement::valid(v, inverse)) {
                s.push_back(v);
            }
        }
    }
    if (s.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return medianOf(s);
}

// 时序一致性：上一帧深度按相对位姿重投影到当前帧，比较逐像素一致性，
// 不一致就压低这一帧的融合权重（消「毛刺 / 浮点 / 重影」）。
static FusionGuard fusionGuard;
static ColorContours colorContours;
static uint64_t lastConsumedDepthTs=0, duplicateDepthRejects=0;
static std::vector<float> lastFusedDepth;
static float lastFuseR[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
static float lastFuseT[3] = {0.f, 0.f, 0.f};
static bool haveLastFusePose = false;
static bool lastFuseCalibrated = false;
static int lastFuseW = 0, lastFuseH = 0;
static uint64_t lastFuseTs = 0;
static float lastFuseK[4]{};
static float lastTemporalRatio = 1.f;
static uint64_t temporalChecks = 0;
static uint64_t temporalRejects = 0;

// V0.13.22 死锁修复：参考帧时效。
// 时序比对拿 lastFusedDepth 当参考，而 lastFuseTs 只在**融合成功**时前进。
// 于是融合一旦连续失败，参考就越来越旧、重投影一致性必然崩塌（agree→0）、
// 再触发整帧拒绝 —— 「越拒越旧、越旧越拒」的正反馈死锁。
// 真机实测：gated=105/120、tmask agree=0、网格冻结在 verts=4257 长达 2min21s，
// 用户移动摄像头完全不出新几何。参考超过时效即视为「中性」不参与比对，让当前帧
// 凭逐像素掩码自由融合，重新推动 lastFuseTs 前进，从死锁里爬出来。
// V0.13.22：阈值必须大于「深度帧间隔」，否则时序比对永远不执行。
// 真机实测（SEVERE 热档）帧间隔约 0.8–1.16s > 0.7s，导致 stale=42/60，
// 时序掩码从未生成、深度未经一致性检验就全量融合 —— 这正是「模型发平、
// 与实物对不上」的直接推手。放到 3s 覆盖正常帧间隔；参考再旧才跳过。
static constexpr uint64_t kTemporalRefMaxAgeNs = 3000000000ULL;  // 3.0 s
static uint64_t temporalRefStale = 0;    // 因参考过旧而跳过比对的帧数
static float lastTemporalRefAgeMs = -1.f; // 上一帧参考的年龄（诊断用）

// V0.13.39 过期参考加固（重叠/错位修复 B 案）：
// 「参考过旧 → 中性放行」是 vc162 死锁的逃生门，但放行期间帧**全权重、零比对**
// 落盘 —— 融合一旦中断 >3s（epoch 切换/标定拒绝/取景离开），恢复后的第一批帧
// 携带的位姿/标定与中断前可能已系统性地错开（真机实测 0.37m 硬跳变复制、
// meanSignedDiff=-0.11），这正是「同一表面出现两份」的主通道。
// 加固后：过期参考仍放行（死锁逃生门保留），但 (a) 全帧降权 conf×0.45，错位
// 壳在 TSDF 里竞争不过旧表面；(b) 相对旧参考位移/转角超限的帧直接拒写，连续
// 拒写上限 4 帧后必须放一帧锚定帧（conf×0.35）重新推动参考链，杜绝死锁复发。
static constexpr float kTemporalStaleConfFactor = 0.45f;
static constexpr float kTemporalStaleMaxTrans = 0.10f;   // m，相对旧参考
static constexpr float kTemporalStaleMaxRot = 0.21f;     // rad ≈ 12°
static constexpr int kTemporalStaleMaxConsecRejects = 4;
static uint64_t temporalStaleWrites = 0;        // 过期参考下降权写入的帧数
static uint64_t temporalStaleMotionRejects = 0; // 过期参考+大位移拒写帧数
static int temporalStaleConsecRejects = 0;      // 连续拒写计数（有界放行）

// V0.13.41（vc184 / Option A）帧到模型 ICP 位姿精修：
// vc183 实测暴露的矛盾 —— 三道门全按「位姿重投影一致性」判据，VINS 渐变漂移
// 让重投影残差超容差 → guard checks=20 rejected=20、fused 13/60（不重影的
// 代价是几乎不融合：开放边 1.2~1.5 万、纹理烘焙 0 关键帧、质量分 39~48）。
// 治法：每帧先对齐 surfel 稳定模型求刚体修正量（frame_icp.h），修正后的
// 位姿进三道门与 TSDF/surfel 写入。修正每帧独立（不回馈 VINS、不跨帧累计）、
// 有界（>8cm 或 >6° 放弃）、欠约束放弃（行为退回旧版）。漂移被就地抵消，
// 重投影判据重新成立 → 门放行、几何对齐写入。
// 模型就绪门槛：粗索引（2cm 稳定代表点）达到该数量才尝试 —— 模型没建立时
// 没有「对齐目标」，修正无从谈起。
static constexpr size_t kIcpMinModelPoints = 2000;
static uint64_t icpAttempts = 0;       // 进入 ICP 主流程的帧数（模型就绪+样本足）
static uint64_t icpAppliedCount = 0;   // 修正被采纳的帧数
static uint64_t icpRejectSamples = 0;  // 帧内有效深度样本不足
static uint64_t icpRejectOverlap = 0;  // 与模型无重叠（新区域/模型未建立）
static uint64_t icpRejectBound = 0;    // 修正超界（防 ICP 拉飞）
static uint64_t icpRejectDegenerate = 0; // 刚体拟合退化（共面病态）
static frame_icp::Diag lastIcpDiag;    // 上一帧 ICP 结果（诊断输出用）

// V0.13.39：epoch 开启/重锚会更换深度->米制映射（epochCalib）。lastFusedDepth
// 是**旧映射域**的融合深度（见 temporalConsistencyMask 处的域一致性注释），
// 跨域比较会把新 epoch 的每一帧都判成「全面冲突」（真机实测 ratio=0.001、
// meanSignedDiff=-0.11m）—— 时序门控全零、融合冻住，直到参考过期 3s 后进入
// 「中性放行」无门控窗口，跳变几何由此成批落盘。epoch 变更时把参考一并作废，
// 下一帧以新域重新建立参考：代价只是 1 帧无比对，换来域内自洽。
static void resetTemporalReferenceForEpochChange() {
    lastFusedDepth.clear();
    haveLastFusePose = false;
    lastFuseCalibrated = false;
    lastFuseW = lastFuseH = 0;
    lastFuseTs = 0;
    temporalStaleConsecRejects = 0;
}
// FusionDiag 打在 2137 的逐帧重置**之前**的时机上，lastTemporal* 会被清零，
// 所以另存一份「上一帧真实结果」专供诊断，否则日志里 ratio 永远是 1。
static float lastDiagTemporalRatio = 1.f;
static int lastDiagTemporalTested = 0;
static int lastDiagTemporalAgree = 0;

// ---- 逐像素时序掩码 + 融合权重（复用缓冲） ----
// 深度是 256² = 65536 个像素，每帧重新分配这几块纯属浪费。nativeOnDepthMapImpl
// 全程持 gStateMutex，且 Kotlin 侧 depthBusy 的 CAS 保证同一时刻只有一帧在走
// 这条路径，所以这里不需要额外加锁。
static std::vector<float> gProjectedDepth;
static std::vector<uint8_t> gTemporalMask;  // 未测试 / 一致 / 冲突 / 遮挡 / 显露
static std::vector<float> gPixelWeight;     // 逐像素融合权重 [0,1]
static std::vector<float> gRangedFusion;    // filterScanRange 后的待融合深度
static std::vector<float> gRangedSurfel;    // 可视化点云路径单独过滤时的深度
static int lastTemporalTested = 0;
static int lastTemporalAgree = 0;
static int lastTemporalDisagree = 0;
static float lastLockWaitMs = 0.f;
static cv::Mat targetMaskMat;        // 与 depth 同尺寸的 CV_8U，目标内 255
static TargetMaskStats targetMaskStats;

// ---- V0.13 Mask temporal gate ----
// Adaptive Mask 只管住了「一帧之内的面积 / 边界」，两帧之间还可能整块跳变
// （目标被遮挡、mask 从瓶子跳到桌面）。这种帧进 TSDF 会在体素场里留下一个
// 位置错误的面，所以宁可这一帧不融合。
static constexpr float kMaskTemporalMinIou = 0.30f;
static constexpr float kMaskAreaJumpRatio = 0.60f;
static constexpr float kMaskCenterJumpFrac = 0.15f;
static cv::Mat prevTargetMask;
static cv::Rect2f prevTargetBox;
static cv::Mat alignedTargetMask;
static float maskTemporalIou = 0.f;
static uint64_t maskTemporalRejects = 0;
static bool maskTemporalOk = true;

// ---- V0.13 Strict target-only LIVE ----
// 一旦本会话**锁定过目标**，累计层就只允许读目标模型。旧行为是
// `(targetG.count() > 0) ? targetG : g`，于是「目标点云还是 0」时屏幕
// 上会静默出现**全场景点云** —— 用户以为自己看到的实时目标几何，
// 其实是墙和桌子。锁定过就必须诚实：0 就是 0。
static bool targetLockEverArmed = false;
// PresenceGate 连续失败帧数（只在 gStateMutex 下访问）
static int presenceFailStreak = 0;
// 目标专属深度尺度：scaleTarget = vinsRoiMedian / rawDepthMedian，EMA 平滑。
// 全局「全场 VINS median ÷ 目标 raw median」会把墙/地面/天花板的深度混进来，
// 只有目标 ROI 内的 VINS 三角化特征才是这个物体自己的尺度。
static float targetDepthScaleEma = 0.f;
static int targetDepthScaleSamples = 0;
static float lastTargetVinsMedian = 0.f;
static float lastTargetRawMedian = 0.f;
static int lastTargetVinsFeatureCount = 0;
static uint64_t targetMaskFrames = 0;
static uint64_t targetFuseFrames = 0;
static uint64_t presenceLostFrames = 0;
static KeyframeEngine kf;
static std::shared_ptr<ObjectTracker> objectTracker;
static bool vk = false;
static int mode = 0;
static uint64_t frames = 0;
static uint32_t lastKF = 0;
static float gFx = 1000.f, gFy = 1000.f, gCx = 0, gCy = 0;
static bool haveExternalDepth = false;
static float vinsT[3] = {0, 0, 0};
static float vinsQ[4] = {0, 0, 0, 1};
static bool vinsPoseOk = false;
static bool haveLastGoodVinsPose = false;
static float lastGoodVinsT[3] = {0.f, 0.f, 0.f};
static bool vinsLostAfterInit = false;
// V0.7.0.1: current loss state, no longer a permanent latch.
static uint64_t vinsRecoveryCount = 0;
static uint64_t vinsRejectStreak = 0;
static uint64_t depthSnapMatches = 0;
static uint64_t depthSnapMisses = 0;
// VC160+：深度回调早退归因。entered/done 长期只有 ~22%，但旧日志只能看出
// 「早退了」看不出「为什么」。实机 10:23–10:32 的 PerfStage 显示 entered=270
// done=61，而每帧单目深度推理要 262–424ms CPU —— 白烧掉的算力直接变成发热，
// 所以必须能一眼看出主导原因，否则只能靠猜。
static uint64_t depthEarlyWorldDiscont = 0;  // VINS 世界跳变，保模型
static uint64_t depthEarlyNoSnap = 0;        // 12ms 窗口里没有对应相机帧快照
static uint64_t depthEarlyStaleGen = 0;      // 快照属于上一代目标
static int64_t lastDepthSnapDiffNs = -1;
static constexpr int64_t kDepthSnapMaxDiffNs = 12'000'000LL;
static float rawVinsT[3] = {0.f, 0.f, 0.f};
static float rawVinsQ[4] = {0.f, 0.f, 0.f, 1.f};
static float acceptedVinsT[3] = {0.f, 0.f, 0.f};
static float acceptedVinsQ[4] = {0.f, 0.f, 0.f, 1.f};
static float lastVinsStep = 0.f;
static uint64_t vinsRejectCount = 0;
static const char* vinsRejectReason = "";
static bool haveFirstReject = false;
static float firstRejectStep = 0.f;
static float firstRejectT[3] = {0.f, 0.f, 0.f};
static float firstRejectVelocity = 0.f;
static float firstRejectAccBias = 0.f;
static float firstRejectGyroBias = 0.f;
static double firstRejectTd = 0.0;
static uint64_t vinsFrames = 0;
static uint64_t depthFrames = 0;
static double lastVinsMs = 0.0;

// ---------------------------------------------------------------- Round 7 device calibration profile
// These values are configured by Kotlin before nativeCreate(). They persist across
// nativeDestroy() so a camera/session restart can reuse the same device profile.
static float gConfiguredRic[9] = {1.f,0.f,0.f, 0.f,1.f,0.f, 0.f,0.f,1.f};
static float gConfiguredTic[3] = {0.f,0.f,0.f};
static double gConfiguredTd = 0.0;
static int gConfiguredEstimateExtrinsicMode = 0;
static int gConfiguredEstimateTd = 1;
static uint64_t gCalibrationProfileSerial = 0;

static uint64_t targetGeneration = 0;

struct FrameSnap {
    uint64_t targetGeneration = 0;
    TargetTrackInfo target;
    uint64_t ts = 0;
    float t[3] = {0, 0, 0};
    float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<uint8_t> rgb;
    int w = 0;
    int h = 0;
};

static std::deque<FrameSnap> snaps;
// One in-flight inference owns its source RGB + pose independently of the ring.
static FrameSnap retainedDepthSnap;
static bool haveRetainedDepthSnap = false;
// VC160+：钉快照失败归因（见 nativeRetainDepthFrame 注释）。
static uint64_t retainOk = 0;
static uint64_t retainMissNoPose = 0;
static uint64_t retainMissMismatch = 0;
static uint64_t retainMissBadTs = 0;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeRetainDepthFrame(JNIEnv*, jobject, jlong ts) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    haveRetainedDepthSnap = false;
    // VC160+：0.13.x 这里只看 snaps.back()。若该帧没有位姿（storeSnap 只在
    // vinsPoseOk 时执行），钉不住；而 nativeOnDepthMapImpl 里那个 12ms 窗口在
    // 推理结束（300–400ms 后）早已滑走，结果整帧深度被丢 —— 白烧一次单目推理。
    // 实机 entered/done 长期只有 ~22%，到底是「这一帧本来就没位姿」还是
    // 「快照在、只是时间戳对不上」，用两个计数器分开，避免继续靠猜。
    const uint64_t want = ts > 0 ? static_cast<uint64_t>(ts) : 0u;
    if (want == 0) {
        ++retainMissBadTs;
        return JNI_FALSE;
    }
    // 从后往前找同时间戳的那一帧，而不是只看最新一帧。
    for (auto it = snaps.rbegin(); it != snaps.rend(); ++it) {
        if (it->ts != want) continue;
        try {
            retainedDepthSnap = *it;
            haveRetainedDepthSnap = true;
            ++retainOk;
        } catch (const std::exception& error) {
            LOGI("Unable to pin depth snapshot: %s", error.what());
            ++retainMissMismatch;
        }
        return haveRetainedDepthSnap ? JNI_TRUE : JNI_FALSE;
    }
    if (!vinsPoseOk) {
        ++retainMissNoPose;   // 该帧没有可用位姿，storeSnap 根本没存
    } else {
        ++retainMissMismatch; // 有位姿，但快照已滑出 60 帧窗口 / 时间戳对不上
    }
    return JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetScanMaxDistance(JNIEnv*, jobject, jfloat meters) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (std::isfinite(meters)) scanMaxDistanceMeters = std::clamp(meters, .2f, 5.f);
}

static bool scanRangeAccepts(float z, float x, float y, float fx, float fy, float cx, float cy,
                             float unitsPerMeter = 0.f) {
    return scan_policy::inRange(z, (x - cx) / fx, (y - cy) / fy,
        scanMaxDistanceMeters, unitsPerMeter > 0.f ? unitsPerMeter : scanWorldPerMeter);
}

static void filterScanRange(std::vector<float>& depth, int w, int h,
                            float fx, float fy, float cx, float cy) {
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        float& z = depth[static_cast<size_t>(y) * w + x];
        if (!scanRangeAccepts(z, x, y, fx, fy, cx, cy)) {
            if (z > 0.f && std::isfinite(z)) ++rangeRejectedPixels;
            z = 0.f;
        }
    }
}
// V0.13.7：原本 160x120 的快照是「逐顶点颜色 GLB」的唯一色彩来源，
// 分辨率太低 -> 顶点颜色糊成一团，模型看上去「和被扫描物体差别很大、模糊不清楚」。
// 提到 320x240 后顶点颜色采样密度翻 4 倍，纹理烘焙失败退回 vertex color 时也清晰可辨。
// 60 帧历史上限下内存约 13.8MB（60 * 320*240*3），可接受。
static const int SNAP_W = 320;
static const int SNAP_H = 240;
static int gVinsW = 640;
static int gVinsH = 480;

// NanoTrack 真实彩色输入的节流计数：每 5 帧生成一张 BGR 缩略图
static uint64_t gTrackerColorFrameCounter = 0;

// tracker 的灰度输入缓冲（640 长边）。nativeOnCameraFrame 只由相机线程串行调用，
// 所以这个缓冲不需要额外加锁。
static std::vector<uint8_t> gTrackerGray;

// ---------------------------------------------------------------- AR pose 历史
/**
 * 一帧「被采纳的 VINS 位姿」样本，带它自己的 SENSOR_TIMESTAMP。
 *
 * 为什么必须带时间戳：Renderer 过去直接拿 nativeGetRenderPose()，
 * 也就是「回调发生这一刻」的最新 pose。但屏幕上正在显示的 Camera frame
 * 有自己的 SENSOR_TIMESTAMP，二者不是同一时刻。手机静止时看不出来，
 * 一转起来点云就会漂/甩/跟不上 —— 因为 AR 拿「现在」的 pose 去画
 * 「过去某一帧」的画面。
 *
 * 正确做法：拿 Preview 的时间戳回来查表，取那一时刻的 pose。
 */
using RenderPoseSample = scan_pose::CameraSample;

static std::deque<RenderPoseSample> renderPoseHistory;
static constexpr size_t kRenderPoseHistoryMax = 120;
// Bound exposure mismatch; never substitute a half-second-old view.
static constexpr int64_t kRenderPoseMaxAgeNs = 80'000'000LL; // 80ms
// V0.13.34：strict 查询 miss 后「最新样本兜底」的次数（诊断，日志节流回显）。
static uint64_t renderPoseBoundedFallbacks = 0;

// ------------------------------------------------- 当前帧 target depth 调试层
/**
 * AR 调试层：**只取当前一帧 depth + 当前 target ROI + 该帧 pose**，不掺历史积累。
 *
 * 意义在于把「AR 坐标链」和「点云融合质量」这两件容易互相甩锅的事拆开：
 * 如果这层绿色点能贴住现实物体、手机平移有正确视差、转动后仍停在原处，
 * 就证明 VINS pose / 相机光轴 / fx fy cx cy / 竖屏旋转 / TextureView 裁剪 /
 * 时间戳同步 整条链是对的；此后累计点云若仍然散，问题就只剩
 * depth scale / depth 精度 / fusion，而不必再回头怀疑 Renderer。
 */
static constexpr int kTargetDebugMaxPoints = 2000;
static bool targetDebugEnabled = false;
static std::vector<float> targetDebugPoints(kTargetDebugMaxPoints * 6, 0.f);
static int targetDebugPointCount = 0;
static int targetDebugRoiPixels = 0;
static int targetDebugValidPixels = 0;
static int targetDebugRoiX0 = 0;
static int targetDebugRoiY0 = 0;
static int targetDebugRoiX1 = 0;
static int targetDebugRoiY1 = 0;
static uint64_t targetDebugTs = 0;
static bool targetDebugPoseOk = false;
static uint64_t targetDebugBuilds = 0;
static uint64_t targetDebugQueries = 0;

// AR 渲染用的过滤门限与计数器（由 Java 侧下发，报告里回显）
static int gArMinHits = 2;
static size_t gArDrawnPoints = 0;

static void quatToR(float qx, float qy, float qz, float qw, float R[9]) {
    R[0] = 1 - 2 * (qy * qy + qz * qz);
    R[1] = 2 * (qx * qy - qz * qw);
    R[2] = 2 * (qx * qz + qy * qw);
    R[3] = 2 * (qx * qy + qz * qw);
    R[4] = 1 - 2 * (qx * qx + qz * qz);
    R[5] = 2 * (qy * qz - qx * qw);
    R[6] = 2 * (qx * qz - qy * qw);
    R[7] = 2 * (qy * qz + qx * qw);
    R[8] = 1 - 2 * (qx * qx + qy * qy);
}

static void eulerToR(float yaw, float pitch, float roll, float R[9]) {
    float cy = cosf(yaw), sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll), sr = sinf(roll);
    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;
}

static void makePose(float R[9], float t[3]) {
    if (vinsPoseOk) {
        quatToR(vinsQ[0], vinsQ[1], vinsQ[2], vinsQ[3], R);
        t[0] = vinsT[0];
        t[1] = vinsT[1];
        t[2] = vinsT[2];
    } else {
        eulerToR(vio.yaw(), vio.pitch(), vio.roll(), R);
        t[0] = vio.tx();
        t[1] = vio.ty();
        t[2] = vio.tz();
    }
}

static void rotatePoint(const float R[9], const float t[3],
                        float Xc, float Yc, float Zc,
                        float& Xw, float& Yw, float& Zw) {
    Xw = R[0] * Xc + R[1] * Yc + R[2] * Zc + t[0];
    Yw = R[3] * Xc + R[4] * Yc + R[5] * Zc + t[1];
    Zw = R[6] * Xc + R[7] * Yc + R[8] * Zc + t[2];
}

static void storeSnap(uint64_t ts, int w, int h, int rs, int urs, int ups,
                      const uint8_t* y, const uint8_t* u, const uint8_t* v,
                      const float R[9], const float t[3]) {
    FrameSnap s;
    s.ts = ts;
    s.targetGeneration = targetGeneration;
    if (objectTracker) s.target = objectTracker->info();
    for (int i = 0; i < 3; i++) {
        s.t[i] = t[i];
    }
    for (int i = 0; i < 9; i++) {
        s.R[i] = R[i];
    }
    s.w = SNAP_W;
    s.h = SNAP_H;
    s.rgb.resize((size_t)SNAP_W * SNAP_H * 3);
    for (int yy = 0; yy < SNAP_H; yy++) {
        int sy = std::min(h - 1, (yy * h) / SNAP_H);
        for (int xx = 0; xx < SNAP_W; xx++) {
            int sx = std::min(w - 1, (xx * w) / SNAP_W);
            int yi = sy * rs + sx;
            int ui = (sy / 2) * urs + (sx / 2) * ups;
            int Y = y[yi];
            int U = u[ui] - 128;
            int V = v[ui] - 128;
            int C = Y - 16;
            int Rr = std::clamp((298 * C + 409 * V + 128) >> 8, 0, 255);
            int Gg = std::clamp((298 * C - 100 * U - 208 * V + 128) >> 8, 0, 255);
            int Bb = std::clamp((298 * C + 516 * U + 128) >> 8, 0, 255);
            size_t o = ((size_t)yy * SNAP_W + xx) * 3;
            s.rgb[o] = (uint8_t)Rr;
            s.rgb[o + 1] = (uint8_t)Gg;
            s.rgb[o + 2] = (uint8_t)Bb;
        }
    }
    snaps.push_back(std::move(s));
    // V0.11: delayed stereo/depth association needs a longer RGB/pose history.
    while (snaps.size() > 60) {
        snaps.pop_front();
    }
}

static void downscaleGray(const uint8_t* src, int w, int h, int stride,
                          uint8_t* dst, int dw, int dh) {
    if (!src || !dst || w <= 0 || h <= 0 || dw <= 0 || dh <= 0) {
        return;
    }
    for (int y = 0; y < dh; y++) {
        int sy = std::min(h - 1, y * h / dh);
        const uint8_t* row = src + (size_t)sy * stride;
        uint8_t* out = dst + (size_t)y * dw;
        for (int x = 0; x < dw; x++) {
            out[x] = row[std::min(w - 1, x * w / dw)];
        }
    }
}

// YUV_420_888 -> 缩小后的 BGR（供 NanoTrack 做真实彩色外观输入）。
// y/u/v 三个平面可能带行填充，uv 的 pixelStride 在 semi-planar 设备上为 2。
static void downscaleYuvToBgr(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                              int w, int h, int rs, int urs, int ups,
                              size_t uSize, size_t vSize,
                              uint8_t* dst, int dw, int dh) {
    if (!y || !u || !v || !dst || w <= 0 || h <= 0 || dw <= 0 || dh <= 0) {
        return;
    }
    if (ups <= 0) {
        ups = 1;
    }
    if (urs <= 0) {
        urs = 1;
    }
    const int cw = std::max(1, w / 2);

    for (int dy = 0; dy < dh; dy++) {
        const int sy = std::min(h - 1, dy * h / dh);
        const uint8_t* yRow = y + (size_t)sy * rs;
        const size_t cRowOff = std::min((size_t)(sy / 2) * urs,
                                        uSize > 0 ? uSize - 1 : (size_t)0);
        uint8_t* out = dst + (size_t)dy * dw * 3;
        for (int dx = 0; dx < dw; dx++) {
            const int sx = std::min(w - 1, dx * w / dw);
            const int cx = std::min(cw - 1, sx / 2);
            const size_t cIdx = cRowOff + (size_t)cx * ups;

            const uint8_t uu = (uSize > 0 && cIdx < uSize) ? u[cIdx] : 128;
            const uint8_t vv = (vSize > 0 && cIdx < vSize) ? v[cIdx] : 128;

            const int Y = std::max(0, (int)yRow[sx] - 16);
            const int U = (int)uu - 128;
            const int V = (int)vv - 128;

            const int r = (298 * Y + 409 * V + 128) >> 8;
            const int g = (298 * Y - 100 * U - 208 * V + 128) >> 8;
            const int b = (298 * Y + 516 * U + 128) >> 8;

            out[dx * 3 + 0] = (uint8_t)std::clamp(b, 0, 255);
            out[dx * 3 + 1] = (uint8_t)std::clamp(g, 0, 255);
            out[dx * 3 + 2] = (uint8_t)std::clamp(r, 0, 255);
        }
    }
}

// 点云生成规律：从屏幕中心向外、由被追踪物体向外。
// 中心置信度高、边缘低；被追踪物体中心附近高、远处低 —— 让实时点云
// 在屏幕中心与物体周围最密，向边缘/远处自然变稀（"从中心向外生长"的观感）。
static constexpr float kScreenEdgeFalloff = 0.85f; // 0=无中心加权, 1=边缘最强压低
static constexpr float kScreenMin = 0.20f;          // 屏幕边缘点最低保留权重
static constexpr float kObjFalloff = 2.2f;          // 物体中心向外衰减速率
static constexpr float kObjMin = 0.30f;             // 远离物体点最低保留权重

/**
 * 喂场景点云（可视化 surfel）。
 *
 * **前置条件**：`depth` 必须已由调用方做过 filterScanRange() —— 与 fuseDepth()
 * 共用同一份过滤结果，不再各复制各过滤一遍。
 *
 * @param pixelWeight 可选逐像素可信度，与 TSDF 用的是同一张图，保证「点云看到的」
 *        与「体素写进去的」可信度一致。
 */
static void feedSceneSurfels(const float* depth, int w, int h, const FrameSnap& s,
                             float confidence, bool haveObj, float objCx, float objCy,
                             const float* pixelWeight = nullptr, const uint8_t* contourPriority = nullptr,
                             const float* poseR = nullptr, const float* poseT = nullptr) {
    if (!depth || w < 4 || h < 4 || s.rgb.empty() || s.w <= 0 || s.h <= 0) {
        return;
    }
    // V0.13.41：poseR/poseT 非空时替代 s.R/s.t（ICP 修正位姿），其余不变。
    const float* poseRot = poseR ? poseR : s.R;
    const float* poseTrans = poseT ? poseT : s.t;
    // 与 fuseDepth 同一套内参降采样（深度 256² -> 相机分辨率）。
    const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
    const float dFx = dk.fx, dFy = dk.fy, dCx = dk.cx, dCy = dk.cy;
    g.beginFrame();
    const float diag = std::sqrt(static_cast<float>(w * w + h * h));
    for(int pass=0;pass<(contourPriority?2:1);++pass)
    for (int y = 0; y < h; ++y) {
        const int yy = std::min(s.h - 1, y * s.h / h);
        for (int x = 0; x < w; ++x) {
            if(contourPriority && bool(contourPriority[size_t(y)*w+x])!=(pass==0))continue;
            const int xx = std::min(s.w - 1, x * s.w / w);
            float z = depth[(size_t)y * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            // ---- 中心加权：屏幕中心 + 被追踪物体中心 ----
            const float u = (x + 0.5f) / w;
            const float v = (y + 0.5f) / h;
            const float dc = std::sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f));
            float wScreen = 1.f - std::min(1.f, dc * 1.41421356f) * kScreenEdgeFalloff;
            wScreen = std::max(kScreenMin, wScreen);
            float wObj = 1.f;
            if (haveObj) {
                const float dx = static_cast<float>(x) - objCx;
                const float dy = static_cast<float>(y) - objCy;
                const float dobj = std::sqrt(dx * dx + dy * dy) / diag;
                wObj = 1.f - std::min(1.f, dobj * kObjFalloff);
                wObj = std::max(kObjMin, wObj);
            }
            // 逐像素可信度（TSDF 用同一张图，二者口径一致）
            float wP = 1.f;
            if (pixelWeight) {
                const float pw = pixelWeight[(size_t)y * w + x];
                if (!std::isfinite(pw) || pw <= 0.f) continue;
                wP = std::clamp(pw, 0.f, 1.f);
            }
            const float wC = wScreen * wObj * wP;
            if (wC < 0.02f) {
                continue;
            }
            const float c = confidence * wC;
            if (c < 0.05f) {
                continue; // 与 SurfelEngine::ingestPoint 丢弃阈值一致
            }
            float Xc = (x - dCx) * z / dFx;
            float Yc = (y - dCy) * z / dFy;
            float Xw, Yw, Zw;
            rotatePoint(poseRot, poseTrans, Xc, Yc, z, Xw, Yw, Zw);
            size_t o = ((size_t)yy * s.w + xx) * 3;
            auto geometry = adaptive::geometry(depth,w,h,x,y,dFx,dFy,dCx,dCy,poseRot);
            if(contourPriority && contourPriority[size_t(y)*w+x]) {
                geometry.protectedDetail=true;geometry.colorBoundary=true;
            }
            g.ingestAdaptivePoint(Xw, Yw, Zw, s.rgb[o], s.rgb[o + 1], s.rgb[o + 2], c, geometry, x, y);
        }
    }
    g.endFrame();
}

/**
 * 融合一帧深度进场景 TSDF。
 *
 * **前置条件**：`depth` 必须已经由调用方做过 filterScanRange()。原来这里自己
 * 复制一份再过滤，而 feedSceneSurfels() 对同一份深度又复制过滤了一遍 —— 同一
 * 帧被复制两次、过滤两次。现在上移到调用方做一次、两个消费者共用。
 *
 * @param pixelWeight 可选逐像素可信度（长度 w*h，[0,1]），nullptr 表示整帧同权。
 */
static void fuseDepth(const float* depth, int w, int h, const FrameSnap& s, float confidence,
                      const float* pixelWeight = nullptr, const uint8_t* contourPriority = nullptr,
                      const float* poseR = nullptr, const float* poseT = nullptr) {
    if (!depth || w < 4 || h < 4 || s.rgb.empty() || s.w <= 0 || s.h <= 0) {
        return;
    }
    // V0.13.41：poseR/poseT 非空时替代 s.R/s.t（ICP 修正位姿），其余不变。
    const float* poseRot = poseR ? poseR : s.R;
    const float* poseTrans = poseT ? poseT : s.t;
    // 评审 P0-2：深度现在以 256² 到达（不再上采样到相机分辨率），内参必须按
    // 深度/相机分辨率比降采样，否则几何会被拉伸到相机分辨率量级。
    const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
    const float dFx = dk.fx, dFy = dk.fy, dCx = dk.cx, dCy = dk.cy;
    tsdf.integrateDepth(depth, w, h, s.rgb.data(), s.w, s.h,
                        dFx, dFy, dCx, dCy, poseRot, poseTrans, confidence,
                        1.f, 0.f, pixelWeight, true, contourPriority);
    // 注意：场景 surfel（g）的喂入已**解耦**到 feedSceneSurfels()，
    // 与 TSDF 共用融合门控；暂停时保留已有几何，不再喂不可信观测。
}

// ------------------------------------------------------------- 分阶段耗时统计
//
// 真机上「这帧慢」只是个总数，定位不到慢在哪个阶段。而后面两件想做的事
// （缩短 gStateMutex 占用、TSDF 改成「活跃块筛选 + 体素投影」两阶段）都该
// 先有基线数据再决定投入 —— 否则既判断不了收益，也发现不了回退。
//
// 每个阶段留一个环形缓冲，周期输出 P50/P95（不是均值：均值会被少数极端帧
// 拉平，而卡顿恰恰是由尾部的那几帧决定的）。
static constexpr int kPerfWindow = 120;
// 深度回调在真机上只有几 fps（TFLite 单目 256² 的推理是瓶颈），60 帧要十几秒
// 才出一条，排查时太稀疏；30 帧在 3fps 下约 10 秒一条，够用又不至于刷屏。
static constexpr int kPerfLogInterval = 30;

class PerfRing {
public:
    void push(float v) {
        buf_[head_] = v;
        head_ = (head_ + 1) % kPerfWindow;
        if (n_ < kPerfWindow) ++n_;
    }
    /** 只在周期输出时调用（每 kPerfLogInterval 帧一次），不在热路径上分配。 */
    float percentile(float q) const {
        if (n_ == 0) return 0.f;
        float v[kPerfWindow];
        std::copy(buf_, buf_ + n_, v);
        std::sort(v, v + n_);
        int idx = std::clamp(int(q * (n_ - 1) + 0.5f), 0, n_ - 1);
        return v[idx];
    }
    void reset() { n_ = 0; head_ = 0; }
private:
    float buf_[kPerfWindow]{};
    int n_ = 0;
    int head_ = 0;
};

enum PerfStage : int {
    kPerfCalib = 0,
    kPerfTemporal,
    kPerfWeight,
    kPerfFuse,
    kPerfSurfel,
    kPerfStereo,
    kPerfLock,
    kPerfTotal,
    kPerfStageCount,
};

static PerfRing gPerf[kPerfStageCount];
static const char* const kPerfStageNames[kPerfStageCount] = {
    "calib", "temporal", "weight", "fuse", "surfel", "stereo", "lock", "total"};
// entered 与 completed 分开计数：深度回调里有几条 early return（位姿快照找不到、
// VINS world 不连续、target 换代）。只统计走到末尾的帧会让人误以为「深度没在跑」，
// 而实际可能是每帧都提前退出了 —— 这是两种完全不同的故障，必须能分开看。
static uint64_t gPerfFrames = 0;      // 进入回调的帧数（入口处递增）
static uint64_t gPerfCompleted = 0;   // 走到函数末尾的帧数
// VC160+：宽容位姿查询（HQ burst 用）失败归因。实机 10:23–10:32 有 21/46 个
// burst 被「pose gate」丢弃，其中 87 次宽容查询是成功的（平均偏差 379ms、
// 最大 499ms，已经贴着 500ms 上限），说明失败不是"历史没有位姿"这么简单。
// 把三种失败原因分开计数，下一轮才能确切知道该放宽容差还是修时间轴。
static uint64_t poseTolMissWorld = 0;   // 世界跳变，拒绝
static uint64_t poseTolMissEmpty = 0;   // 位姿历史为空（VINS 从未初始化）
static uint64_t poseTolMissOver = 0;    // 历史非空，但最近样本超出容差
static uint64_t poseTolHit = 0;         // 成功命中（含宽松回退）

static inline float perfMsSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

static void perfLogIfDue() {
    if (gPerfFrames % kPerfLogInterval != 0) {
        return;
    }
    std::string s;
    for (int i = 0; i < kPerfStageCount; ++i) {
        char b[72];
        snprintf(b, sizeof(b), "%s p50=%.1f p95=%.1f",
                 kPerfStageNames[i],
                 gPerf[i].percentile(0.5f),
                 gPerf[i].percentile(0.95f));
        if (i) s += " | ";
        s += b;
    }
    // tmask 三项用来观察逐像素掩码到底判定出了多少东西：tested 太少说明重投影
    // 几乎没有重叠（位姿动得太快或深度缺失），disagree 长期偏高说明深度本身不稳。
    // lockWait 是锁等待 —— 只统计持锁时长会漏掉它，而它正是"渲染线程被卡住"的来源。
    __android_log_print(ANDROID_LOG_INFO, "PerfStage",
                        "entered=%llu done=%llu %s | tmask tested=%d agree=%d disagree=%d | lockWait=%.1f",
                        static_cast<unsigned long long>(gPerfFrames),
                        static_cast<unsigned long long>(gPerfCompleted), s.c_str(),
                        lastTemporalTested, lastTemporalAgree, lastTemporalDisagree,
                        static_cast<double>(lastLockWaitMs));
    // 早退/失败归因单独一行，保持 PerfStage 主行格式不变（外部脚本在解析它）。
    __android_log_print(ANDROID_LOG_INFO, "PerfGate",
                        "depthEarly world=%llu noSnap=%llu staleGen=%llu dup=%llu | "
                        "retain ok=%llu noPose=%llu mismatch=%llu badTs=%llu | "
                        "poseTol hit=%llu missWorld=%llu missEmpty=%llu missOver=%llu",
                        static_cast<unsigned long long>(depthEarlyWorldDiscont),
                        static_cast<unsigned long long>(depthEarlyNoSnap),
                        static_cast<unsigned long long>(depthEarlyStaleGen),
                        static_cast<unsigned long long>(duplicateDepthRejects),
                        static_cast<unsigned long long>(retainOk),
                        static_cast<unsigned long long>(retainMissNoPose),
                        static_cast<unsigned long long>(retainMissMismatch),
                        static_cast<unsigned long long>(retainMissBadTs),
                        static_cast<unsigned long long>(poseTolHit),
                        static_cast<unsigned long long>(poseTolMissWorld),
                        static_cast<unsigned long long>(poseTolMissEmpty),
                        static_cast<unsigned long long>(poseTolMissOver));
}

// ------------------------------------------------------------- V0.9 stereo anchors
static constexpr int64_t kStereoAnchorPoseMaxDiffNs = 12'000'000LL;
static constexpr uint64_t kStereoAnchorMaxAgeNs = 1'200'000'000ULL;
static constexpr int64_t kStereoTargetMaskMaxDiffNs = 45'000'000LL;

static const FrameSnap* findStereoAnchorSnap(
        uint64_t timestampNs,
        int64_t* outDiffNs) {
    const FrameSnap* best = nullptr;
    int64_t bestDiff = kStereoAnchorPoseMaxDiffNs;
    for (const auto& snap : snaps) {
        int64_t d =
            static_cast<int64_t>(snap.ts) -
            static_cast<int64_t>(timestampNs);
        if (d < 0) d = -d;
        if (d < bestDiff) {
            bestDiff = d;
            best = &snap;
        }
    }
    if (outDiffNs) {
        *outDiffNs = best ? bestDiff : -1;
    }
    return best;
}

static int buildStereoSparseDepth(
        const mobilescan3d::stereo_anchor::WorldAnchorBatch& batch,
        int outW,
        int outH,
        int pixelStep,
        float minWeight,
        const cv::Mat* normalizedMask,
        std::vector<float>* outDepth) {
    if (!outDepth || outW < 4 || outH < 4) return 0;

    outDepth->assign(
        static_cast<size_t>(outW) * outH,
        0.f);

    const int step = std::clamp(pixelStep, 1, 4);
    const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, outW, outH);
    int accepted = 0;

    for (const auto& a : batch.anchors) {
        if (a.weight + 1e-4f < minWeight ||
            !(a.zWorld > 0.05f) ||
            !std::isfinite(a.zWorld)) {
            continue;
        }

        // Stereo already carries physical meters: never compare zWorld directly to meters.
        if (!scanRangeAccepts(a.zMetric, a.u * (outW - 1), a.v * (outH - 1),
                              dk.fx, dk.fy, dk.cx, dk.cy, 1.f)) continue;

        if (normalizedMask &&
            !normalizedMask->empty() &&
            normalizedMask->type() == CV_8U) {
            const int mx = std::clamp(
                static_cast<int>(std::lround(
                    a.u * static_cast<float>(normalizedMask->cols - 1))),
                0,
                normalizedMask->cols - 1);
            const int my = std::clamp(
                static_cast<int>(std::lround(
                    a.v * static_cast<float>(normalizedMask->rows - 1))),
                0,
                normalizedMask->rows - 1);
            if (!normalizedMask->at<uint8_t>(my, mx)) {
                continue;
            }
        }

        int x = std::clamp(
            static_cast<int>(std::lround(
                a.u * static_cast<float>(outW - 1))),
            0,
            outW - 1);
        int y = std::clamp(
            static_cast<int>(std::lround(
                a.v * static_cast<float>(outH - 1))),
            0,
            outH - 1);

        // TsdfEngine visits only the pixelStep lattice.
        x = (x / step) * step;
        y = (y / step) * step;

        const size_t idx =
            static_cast<size_t>(y) * outW + x;

        // MultiCam exports strongest-first; preserve the first collision.
        if ((*outDepth)[idx] > 0.f) {
            continue;
        }
        (*outDepth)[idx] = a.zWorld;
        accepted++;
    }

    return accepted;
}

static int integrateStereoAnchorBatch(
        TsdfEngine& dst,
        const mobilescan3d::stereo_anchor::WorldAnchorBatch& batch,
        const FrameSnap& snap,
        const cv::Mat* normalizedMask,
        int* outPasses) {
    if (outPasses) *outPasses = 0;
    if (batch.anchors.empty() ||
        batch.imageWidth < 4 ||
        batch.imageHeight < 4 ||
        snap.rgb.empty()) {
        return 0;
    }

    // Sparse constraints do not need a full 1280x960 zero-filled depth map.
    // Keep the same rays/intrinsics but cap the lattice at a 640px long edge.
    const int sourceMax =
        std::max(batch.imageWidth, batch.imageHeight);
    const float sparseScale =
        sourceMax > 640
            ? 640.f / static_cast<float>(sourceMax)
            : 1.f;
    const int bw =
        std::max(
            4,
            static_cast<int>(
                std::lround(batch.imageWidth * sparseScale)));
    const int bh =
        std::max(
            4,
            static_cast<int>(
                std::lround(batch.imageHeight * sparseScale)));

    const float sx =
        static_cast<float>(bw) /
        static_cast<float>(std::max(1, gV07InputW));
    const float sy =
        static_cast<float>(bh) /
        static_cast<float>(std::max(1, gV07InputH));

    const float fx = gFx * sx;
    const float fy = gFy * sy;
    const float cx = gCx * sx;
    const float cy = gCy * sy;

    int uniqueAccepted = 0;
    int passes = 0;
    std::vector<float> sparse;

    // TsdfEngine clamps a single observation confidence to <=1. Weight therefore
    // becomes repeated sparse observations: all anchors once, strong anchors 2-3x.
    for (int pass = 1; pass <= 3; ++pass) {
        const int n = buildStereoSparseDepth(
            batch,
            bw,
            bh,
            dst.pixelStep(),
            static_cast<float>(pass),
            normalizedMask,
            &sparse);
        if (pass == 1) {
            uniqueAccepted = n;
        }
        if (n <= 0) {
            continue;
        }

        dst.integrateDepth(
            sparse.data(),
            bw,
            bh,
            snap.rgb.data(),
            snap.w,
            snap.h,
            fx,
            fy,
            cx,
            cy,
            snap.R,
            snap.t,
            1.0f);
        passes++;
    }

    if (outPasses) *outPasses = passes;
    return uniqueAccepted;
}

// ------------------------------------------------------------- 目标专用融合
/**
 * 清空目标专用模型与它的全部诊断状态。
 * 调用点：新建会话 / 销毁 / 选中新目标 / 清除目标。
 * 必须在 gStateMutex 下调用。
 */
/**
 * 重置深度标定（含 V0.12 的 Fusion Calibration Epoch 与 V0.9 的 stereo world scale）。
 *
 * V0.12 起**调用点只有两个**：新会话（nativeCreate）/ 销毁会话（nativeDestroy）。
 * 「换目标」不再清标定 —— 标定描述的是「深度模型输出 -> VINS world 尺度」，
 * 与当前锁的是哪个目标无关；而 V0.12 的融合门控要求标定可用，
 * 把清标定挂在换目标上会直接变成「锁定目标后点云一直是 0」。
 *
 * 必须在 gStateMutex 下调用。
 */
static void resetDepthCalibration() {
    cameraDistance = {}; cameraDistanceTs = 0; cameraDistanceSource = 0;
    ++depthCalibFullResets;
    depthCalibrator.reset();
    // V0.9: invalidate stereo world-scale whenever depth calibration resets.
    mobilescan3d::stereo_anchor::reset();
    lastCalibScale = 1.f;
    lastCalibShift = 0.f;
    lastCalibConfidence = 0.f;
    lastCalibSamples = 0;
    lastCalibValid = false;
    lastCalibInverse = false;
    lastCalibRefSpanRel = 0.f;
    lastCalibOutInvZSpan = 0.f;
    lastCalibOutZSpanRel = 0.f;
    calibFrames = 0;
    calibRejectFrames = 0;
    lastFusionDepthCalibrated = false;
    lastFusionDepthMin = 0.f;
    lastFusionDepthMax = 0.f;
    lastFusionDepthMean = 0.f;
    lastFusionDepthValid = 0;
    lastFusionDepthConversionFallback = 0;
    fusionGuard.reset(); colorContours.reset();
    lastConsumedDepthTs=duplicateDepthRejects=0;
    depthEarlyWorldDiscont=depthEarlyNoSnap=depthEarlyStaleGen=0;
    retainOk=retainMissNoPose=retainMissMismatch=retainMissBadTs=0;
    lastFusedDepth.clear();
    haveLastFusePose = false;
    lastFuseCalibrated = false;
    lastTemporalRatio = 1.f;
    temporalChecks = 0;
    temporalRejects = 0;
    temporalStaleWrites = 0;
    temporalStaleMotionRejects = 0;
    temporalStaleConsecRejects = 0;
    lastTemporalTested = 0;
    lastTemporalAgree = 0;
    lastTemporalDisagree = 0;
    gProjectedDepth.clear();
    lastFuseW = lastFuseH = 0; lastFuseTs = 0;
    gPerfFrames = gPerfCompleted = 0;
    lastLockWaitMs = 0.f;
    gTemporalMask.clear();
    gPixelWeight.clear();
    gRangedFusion.clear();
    gRangedSurfel.clear();
    for (int i = 0; i < kPerfStageCount; ++i) {
        gPerf[i].reset();
    }
    resetFusionEpoch();
}

static void resetMeshPipeline() {
    meshEngine = MeshEngine();
    meshHasHardEdges = false;
    hardSurfaceStats = {};
    meshStats = MeshBuildStats{};
    meshCleanupStats = MeshPostProcessStats{};
    gTextureKeyframes.clear();
    textureBakeStats = TextureBakeStats{};
    uvUnwrapStats = UvUnwrapStats{};
    meshDirty = true;
    // V0.12: 这里**不再**重置深度标定。
    //
    // 旧代码把 depthCalibrator.reset() 放在这里，而 resetTargetModel() 又会调
    // resetMeshPipeline() —— 于是「用户每锁一次目标就把深度标定清零」。
    // 标定描述的是「深度模型输出 -> VINS world 尺度」，与当前锁的是哪个目标
    // 毫无关系；而 V0.12 起「标定未就绪不融合」，于是这个旧行为会直接变成
    // 「锁定目标后点云一直是 0」。
    //
    // 需要清标定的场合只有一个：新会话 / 销毁会话 —— 见 resetDepthCalibration()。
    ++depthCalibKeptAcrossTarget;
}

static void resetTargetModel() {
    ++targetGeneration; // In-flight depth cannot be assigned to a newly selected object.
    targetG.reset();
    targetTsdf.reset();
    targetMaskMat.release();
    targetMaskStats = TargetMaskStats{};
    // V0.13：换目标/新会话时 mask 时序历史必须一起作废，否则第一帧就会
    // 拿新目标的 mask 去和上一个目标的 mask 求 IoU，白白触发一次时序拒绝。
    prevTargetMask.release();
    prevTargetBox = {};
    alignedTargetMask.release();
    maskTemporalIou = 0.f;
    maskTemporalOk = true;
    presenceFailStreak = 0;
    targetDepthScaleEma = 0.f;
    targetDepthScaleSamples = 0;
    lastTargetVinsMedian = 0.f;
    lastTargetRawMedian = 0.f;
    lastTargetVinsFeatureCount = 0;
    // 换了目标（或新会话）之后，旧网格是对旧目标/旧体素场提的，必须作废。
    resetMeshPipeline();
}

/** 目标专属深度尺度的最小样本数：样本太少就先用原始深度，不猜。 */
static constexpr int kTargetScaleMinSamples = 10;
/** 目标专属尺度的合理区间，挡住 vins/raw 同时偏小时产生的野值。 */
static constexpr float kTargetScaleMin = 0.01f;
static constexpr float kTargetScaleMax = 20.f;

/** 当前生效的目标深度尺度（vins/raw）。样本不足时返回 1（即不修正）。 */
static float currentTargetDepthScale() {
    // 全局深度标定已经把深度对齐到 VINS 尺度时，不再叠加「目标专属尺度」——
    // 两者目标完全相同，叠加就变成双重缩放。
    if (depthCalibrationEnabled) {
        // Frozen epoch depth is already calibrated even during live-fit dropout.
        return 1.f;
    }
    if (targetDepthScaleSamples < kTargetScaleMinSamples) {
        return 1.f;
    }
    if (!std::isfinite(targetDepthScaleEma) || !(targetDepthScaleEma > 0.f)) {
        return 1.f;
    }
    return std::clamp(targetDepthScaleEma, kTargetScaleMin, kTargetScaleMax);
}

/**
 * 只把 **mask 内** 的深度融进目标专用模型。
 *
 * 与 fuseDepth() 的两点区别：
 *   1. mask 外的 depth 一律置 0 —— TSDF 与 Gaussian 都看不到背景，
 *      这是「墙/天花板/桌子物理上不可能进入目标点云」的落地点；
 *   2. mask 内深度先乘**目标专属尺度**（vins/raw）再转 3D。累计模型是米制的，
 *      必须用目标自己的尺度；当前帧 live 层不需要（它直接用相机坐标投影，
 *      完全不依赖 depth 的绝对准确性）。
 */
static void fuseTargetDepth(const float* depth, int w, int h, const FrameSnap& s,
                            const cv::Mat& maskIn, float confidence,
                            const float* pixelWeight = nullptr, const uint8_t* contourPriority = nullptr,
                            const float* poseR = nullptr, const float* poseT = nullptr) {
    if (!depth || w < 4 || h < 4 || s.rgb.empty() || maskIn.empty()) {
        return;
    }
    // V0.13.41：poseR/poseT 非空时替代 s.R/s.t（ICP 修正位姿），其余不变。
    const float* poseRot = poseR ? poseR : s.R;
    const float* poseTrans = poseT ? poseT : s.t;
    if (maskIn.cols != w || maskIn.rows != h || maskIn.type() != CV_8U) {
        return;
    }
    cv::Mat mask = maskIn; // 浅拷贝，只为拿到非 const 的 ptr()

    const float scale = currentTargetDepthScale();

    // 复用缓冲：目标融合每帧都走这里，256² = 65536 float 的分配完全没必要每帧做。
    // （nativeOnDepthMapImpl 全程持 gStateMutex，本函数只在该上下文被调用。）
    static std::vector<float> masked;
    masked.assign(static_cast<size_t>(w) * h, 0.f);
    for (int y = 0; y < h; y++) {
        const float* srcRow = depth + static_cast<size_t>(y) * w;
        const uint8_t* mRow = mask.ptr<uint8_t>(y);
        float* dstRow = masked.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x++) {
            if (!mRow[x]) {
                continue;
            }
            const float z = srcRow[x];
            if (!std::isfinite(z) || z <= 0.f) {
                continue;
            }
            dstRow[x] = z * scale;
        }
    }

    // 评审 P0-2：深度以 256² 到达，内参按深度/相机比降采样。
    const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
    const float dFx = dk.fx, dFy = dk.fy, dCx = dk.cx, dCy = dk.cy;
    filterScanRange(masked, w, h, dFx, dFy, dCx, dCy);
    targetTsdf.integrateDepth(masked.data(), w, h, s.rgb.data(), s.w, s.h,
                              dFx, dFy, dCx, dCy, poseRot, poseTrans, confidence,
                              1.f, 0.f, pixelWeight, true, contourPriority);

    targetG.beginFrame();
    for(int pass=0;pass<(contourPriority?2:1);++pass)
    for (int y = 0; y < h; ++y) {
        const int yy = std::min(s.h - 1, y * s.h / h);
        const uint8_t* mRow = mask.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            if(contourPriority && bool(contourPriority[size_t(y)*w+x])!=(pass==0))continue;
            const int xx = std::min(s.w - 1, x * s.w / w);
            if (!mRow[x]) {
                continue;
            }
            const float z = masked[static_cast<size_t>(y) * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            // 与场景侧同口径：目标点云也吃逐像素可信度，避免「体素里被降权的
            // 像素，点云里还是满权重」这种两套容器不一致。
            float wP = 1.f;
            if (pixelWeight) {
                const float pw = pixelWeight[static_cast<size_t>(y) * w + x];
                if (!std::isfinite(pw) || pw <= 0.f) continue;
                wP = std::clamp(pw, 0.f, 1.f);
            }
            const float Xc = (x - dCx) * z / dFx;
            const float Yc = (y - dCy) * z / dFy;
            float Xw, Yw, Zw;
            rotatePoint(poseRot, poseTrans, Xc, Yc, z, Xw, Yw, Zw);
            const size_t o = (static_cast<size_t>(yy) * s.w + xx) * 3;
            auto geometry = adaptive::geometry(masked.data(),w,h,x,y,dFx,dFy,dCx,dCy,poseRot);
            if(contourPriority && contourPriority[size_t(y)*w+x]) {
                geometry.protectedDetail=true;geometry.colorBoundary=true;
            }
            targetG.ingestAdaptivePoint(Xw, Yw, Zw, s.rgb[o], s.rgb[o + 1], s.rgb[o + 2],
                                       confidence * wP, geometry, x, y);
        }
    }
    targetG.endFrame();
    targetFuseFrames++;
}

// ---------------------------------------------------------------- PresenceGate
/** 外观分数下限（NanoTrack score）。 */
static constexpr float kPresenceAppearanceScore = 0.55f;
/** 没有外观后端时的替代下限：用 KLT 侧的 confidence，而不是直接判 false。 */
static constexpr float kPresenceConfidenceFallback = 0.35f;
/** mask 面积下限，与 TargetMaskEngine::kMinMaskArea 一致。 */
static constexpr int kPresenceMinMaskArea = 100;
/** KLT 运动判据：内点率或跟踪点数任一达标即可。 */
static constexpr float kPresenceMinInlierRatio = 0.45f;
static constexpr int kPresenceMinPoints = 25;
/** tracker 中心与 mask 质心的不一致上限（按 bbox 对角线归一化）。 */
static constexpr float kPresenceMaxCenterOffset = 0.20f;
/** 连续多少帧判定失败才认为「目标不在」。1 帧太敏感、3 帧太慢 —— 取 2。 */
static constexpr int kPresenceFailFrames = 2;

struct PresenceDecision {
    bool valid = false;
    bool appearanceOk = false;
    bool motionOk = false;
    bool maskOk = false;
    bool centerOk = false;
    float centerOffsetRatio = 0.f;
};

/**
 * 「目标真的还在吗？」
 *
 * 核心认知转变：**box 还在画面里 != 物体还在**。
 * 真实目标离开后，tracker 往往会在背景纹理上找到一个「看起来正常」的框并
 * 继续输出（OpenCV tracking #619 / NanoTrack issue 都有记录）。所以判定依据
 * 必须是外观 + Mask + 运动 + 中心一致四项合成，而不是 bbox 的几何位置。
 *
 * presenceOk = appearanceOk && maskOk && (motionOk || centerOk)
 */
static PresenceDecision evaluatePresence(const TargetTrackInfo& ti,
                                        const TargetMaskStats& ms,
                                        int depthW, int depthH) {
    PresenceDecision d;
    // 外观：有 Nano 就用它的分数；模型没加载时退回 KLT 置信度，
    // 否则没有 NanoTrack 的设备会永远被判「目标不在」。
    d.appearanceOk = ti.nanoLoaded
        ? (ti.nanoScore >= kPresenceAppearanceScore)
        : (ti.confidence >= kPresenceConfidenceFallback);
    d.motionOk = (ti.inlierRatio >= kPresenceMinInlierRatio) ||
                 (ti.trackedPoints >= kPresenceMinPoints);
    d.maskOk = ms.valid && !ms.overExpanded && ms.area >= kPresenceMinMaskArea;
    if (d.maskOk) {
        const float bw = (ti.x1 - ti.x0) * static_cast<float>(depthW);
        const float bh = (ti.y1 - ti.y0) * static_cast<float>(depthH);
        const float diag = std::sqrt(bw * bw + bh * bh);
        const float tcx = 0.5f * (ti.x0 + ti.x1) * static_cast<float>(depthW);
        const float tcy = 0.5f * (ti.y0 + ti.y1) * static_cast<float>(depthH);
        const float dx = ms.centerX - tcx;
        const float dy = ms.centerY - tcy;
        const float dist = std::sqrt(dx * dx + dy * dy);
        d.centerOffsetRatio = diag > 1.f ? dist / diag : 0.f;
        d.centerOk = d.centerOffsetRatio < kPresenceMaxCenterOffset;
    }
    // V0.13.19.2 PresenceGate 修正：绕物扫描时目标外观/视角/光照必然变化，
    // NanoTrack 外观分数会规律性掉到 kPresenceAppearanceScore 以下 —— 这是预期
    // 现象，不该据此判「目标不在」（否则每帧弹「外观/深度不一致」且目标专属几何
    // 被饿死）。PresenceGate 的本意是「bbox 在画面 != 物体还在」（防 tracker 锁
    // 背景纹理），真正的判据应是几何一致性：mask 在 +（运动或中心一致），且只有当
    // 运动与中心**同时**失配时才需要外观来兜底确认。外观既不能单独判亡，也不该
    // 一票否决一个几何强一致的跟踪框。
    const bool geomOk = d.motionOk || d.centerOk;
    const bool geomStrong = d.motionOk && d.centerOk;
    d.valid = d.maskOk && geomOk && (geomStrong || d.appearanceOk);
    return d;
}

/** 当前帧没有可用 mask（目标出界 / 无目标）时清掉时序历史。 */
static void resetMaskTemporalGate() {
    prevTargetMask.release();
    prevTargetBox = {};
    alignedTargetMask.release();
    maskTemporalIou = 0.f;
    maskTemporalOk = true;
}

/**
 * V0.13 Mask temporal gate。
 *
 * Adaptive Mask（多档深度容差 + 面积比/边界门）解决的是「**一帧之内** mask
 * 吞掉整个搜索框」；但连续两帧之间仍可能整块跳变：目标被手挡住时 mask 会
 * 短暂消失，或者 mask 从瓶子跳到后面的桌面。这类帧一旦进 TSDF，就会在体素
 * 场里留下一个**位置错误的面**，而且越扫越多。
 *
 * 判据刻意用三个都容易被理解的量：IoU、面积相对变化、质心位移（按 bbox
 * 对角线归一化）。任一超标就这一帧不融合 —— 代价只是少一帧数据，
 * 而放进错误几何的代价是整个模型残掉。
 */
static void updateMaskTemporalGate(const TargetTrackInfo& ti, int depthW, int depthH) {
    maskTemporalOk = true;
    if (targetMaskMat.empty() || !targetMaskStats.valid ||
        targetMaskStats.area < kPresenceMinMaskArea) {
        resetMaskTemporalGate();
        return;
    }
    const cv::Rect2f box(ti.x0 * depthW, ti.y0 * depthH,
                         (ti.x1-ti.x0) * depthW, (ti.y1-ti.y0) * depthH);
    const auto comparison = compareTargetMasks(prevTargetMask, targetMaskMat,
                                               prevTargetBox, box, alignedTargetMask);
    maskTemporalIou = comparison.comparable ? comparison.iou : 0.f;
    if (comparison.comparable && (comparison.iou < kMaskTemporalMinIou ||
        comparison.areaJump > kMaskAreaJumpRatio || comparison.centerJump > kMaskCenterJumpFrac)) {
        maskTemporalOk = false;
        ++maskTemporalRejects;
    }
    // Reuse storage; history is source-exposure aligned, just like ti.
    targetMaskMat.copyTo(prevTargetMask);
    prevTargetBox = box;
}

static jboolean nativeCreateImpl(jint w, jint h, jfloat fx, jfloat fy, jfloat cx, jfloat cy) {
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192 ||
        !std::isfinite(fx) || !std::isfinite(fy) || fx <= 0.f || fy <= 0.f ||
        !std::isfinite(cx) || !std::isfinite(cy)) return JNI_FALSE;
    std::lock_guard<std::mutex> cameraLock(gCameraCallbackMutex);
    gPersistentRelocalizer.resetLiveAlignment();
    {
        std::lock_guard<std::mutex> rlk(gRelocFrameMutex);
        gRelocGray.clear();
        gRelocFrameTimestampNs = 0u;
        gRelocConsumedTimestampNs = 0u;
    }
    std::lock_guard<std::mutex> lk(gStateMutex);
    ++gNativeGeneration;
    gV07InputW = w;
    gV07InputH = h;
    gFx = fx;
    gFy = fy;
    gCx = cx;
    gCy = cy;
    g.reset();
    vio.reset();
    df.reset();
    ai.reset();
    tsdf.reset();
    // 体素分辨率：全场景沿用 20mm（与历史行为一致，避免内存暴涨）；
    // 目标专属模型用 8mm —— 它只为单个物体建网格，体素数可控，
    // 而 mesh 的质量直接取决于体素分辨率。
    tsdf.setVoxelSize(0.020f);
    targetTsdf.setVoxelSize(0.008f);
    resetMeshPipeline();
    // V0.12: 新会话 = 新相机/新深度源，深度标定必须从头来。
    resetDepthCalibration();
    // 目标专用模型与诊断：新会话必须从头开始，否则会残留上一轮的目标点云
    targetG.reset();
    targetTsdf.reset();
    targetMaskMat.release();
    targetMaskStats = TargetMaskStats{};
    targetMaskEngine = TargetMaskEngine();
    resetMaskTemporalGate();
    // V0.13：新会话 = 还没锁定过目标，累计层可以（也应该）先显示全场景，
    // 免得开屏就是全黑。一旦用户框过目标，这条就永久关掉。
    targetLockEverArmed = false;
    maskTemporalRejects = 0;
    presenceFailStreak = 0;
    targetDepthScaleEma = 0.f;
    targetDepthScaleSamples = 0;
    lastTargetVinsMedian = 0.f;
    lastTargetRawMedian = 0.f;
    lastTargetVinsFeatureCount = 0;
    targetMaskFrames = 0;
    targetFuseFrames = 0;
    presenceLostFrames = 0;
    kf.reset();
    objectTracker = std::make_shared<ObjectTracker>();
    aiBackend.initialize(w, h);
    frames = 0;
    rangeRejectedPixels = frozenEvidenceRecoveries = 0;
    frozenEvidenceStreak = 0;
    scanWorldPerMeter = 1.f;
    haveRetainedDepthSnap = false;
    retainedDepthSnap = FrameSnap{};
    lastKF = 0;
    haveExternalDepth = false;
    vinsPoseOk = false;
    haveLastGoodVinsPose = false;
    vinsLostAfterInit = false;
    lastGoodVinsT[0] = 0.f;
    lastGoodVinsT[1] = 0.f;
    lastGoodVinsT[2] = 0.f;
    rawVinsT[0] = rawVinsT[1] = rawVinsT[2] = 0.f;
    rawVinsQ[0] = rawVinsQ[1] = rawVinsQ[2] = 0.f;
    rawVinsQ[3] = 1.f;
    acceptedVinsT[0] = acceptedVinsT[1] = acceptedVinsT[2] = 0.f;
    acceptedVinsQ[0] = acceptedVinsQ[1] = acceptedVinsQ[2] = 0.f;
    acceptedVinsQ[3] = 1.f;
    lastVinsStep = 0.f;
    vinsRejectCount = 0;
    vinsRejectReason = "";
    haveFirstReject = false;
    firstRejectStep = 0.f;
    firstRejectT[0] = firstRejectT[1] = firstRejectT[2] = 0.f;
    firstRejectVelocity = 0.f;
    firstRejectAccBias = 0.f;
    firstRejectGyroBias = 0.f;
    firstRejectTd = 0.0;
    vinsT[0] = vinsT[1] = vinsT[2] = 0;
    vinsQ[0] = vinsQ[1] = vinsQ[2] = 0;
    vinsQ[3] = 1;
    snaps.clear();
    mobilescan3d::stereo_anchor::reset();
    vinsResetInitLatch();
    vinsFrames = depthFrames = 0;
    lastVinsMs = 0.0;

    uint32_t v = 0;
    auto pfn = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(
        VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    VkResult r = pfn ? pfn(&v) : VK_ERROR_INITIALIZATION_FAILED;
    vk = (r == VK_SUCCESS || r == VK_INCOMPLETE);
    LOGI("V0.4 native create Vulkan=%d", vk);

    float ric[9];
    float tic[3];
    std::copy(gConfiguredRic, gConfiguredRic + 9, ric);
    std::copy(gConfiguredTic, gConfiguredTic + 3, tic);
    const double initialTd = gConfiguredTd;
    const int estimateExtrinsicMode = gConfiguredEstimateExtrinsicMode;
    const int estimateTd = gConfiguredEstimateTd;
    const float vScale = 640.f / (float)std::max(w, h);
    gVinsW = std::max(1, (int)(w * vScale));
    gVinsH = std::max(1, (int)(h * vScale));
    const float vFx = fx * vScale;
    const float vFy = fy * vScale;
    const float vCx = cx * vScale;
    const float vCy = cy * vScale;
    vinsInit(
        vFx, vFy, vCx, vCy, gVinsW, gVinsH, ric, tic,
        0.1f, 0.001f, 0.001f, 0.0001f,
        initialTd, estimateExtrinsicMode, estimateTd);
    return JNI_TRUE;
}

/** Round 7: configure camera<->IMU calibration used by the NEXT nativeCreate(). */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetVinsCalibrationProfile(
        JNIEnv* env, jobject,
        jfloatArray ricArray, jfloatArray ticArray,
        jfloat tdSeconds, jboolean estimateExtrinsic, jboolean estimateTd) {
    if (!ricArray || !ticArray ||
        env->GetArrayLength(ricArray) < 9 || env->GetArrayLength(ticArray) < 3 ||
        !std::isfinite(tdSeconds) || std::abs(tdSeconds) > 0.150f) {
        return JNI_FALSE;
    }
    float r[9]{};
    float t[3]{};
    env->GetFloatArrayRegion(ricArray, 0, 9, r);
    env->GetFloatArrayRegion(ticArray, 0, 3, t);
    if (env->ExceptionCheck()) return JNI_FALSE;

    // Reject obviously invalid matrices. A small non-orthogonality is tolerated
    // because profiles imported from calibration tools are rounded text values.
    auto rowDot = [&](int a, int b) {
        return r[a*3+0]*r[b*3+0] + r[a*3+1]*r[b*3+1] + r[a*3+2]*r[b*3+2];
    };
    const float n0 = rowDot(0,0), n1 = rowDot(1,1), n2 = rowDot(2,2);
    const float det =
        r[0]*(r[4]*r[8]-r[5]*r[7]) -
        r[1]*(r[3]*r[8]-r[5]*r[6]) +
        r[2]*(r[3]*r[7]-r[4]*r[6]);
    if (!std::isfinite(det) || det < 0.75f || det > 1.25f ||
        std::abs(n0-1.f) > 0.20f || std::abs(n1-1.f) > 0.20f ||
        std::abs(n2-1.f) > 0.20f ||
        std::abs(rowDot(0,1)) > 0.20f ||
        std::abs(rowDot(0,2)) > 0.20f ||
        std::abs(rowDot(1,2)) > 0.20f) {
        return JNI_FALSE;
    }
    const float tNorm = std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
    if (!std::isfinite(tNorm) || tNorm > 0.50f) return JNI_FALSE;

    std::lock_guard<std::mutex> lk(gStateMutex);
    std::copy(r, r + 9, gConfiguredRic);
    std::copy(t, t + 3, gConfiguredTic);
    gConfiguredTd = static_cast<double>(tdSeconds);
    gConfiguredEstimateExtrinsicMode = estimateExtrinsic == JNI_TRUE ? 2 : 0;
    gConfiguredEstimateTd = estimateTd == JNI_TRUE ? 1 : 0;
    ++gCalibrationProfileSerial;
    return JNI_TRUE;
}

/**
 * Round 7 fixed-width VINS health protocol.
 * 0 init, 1 velocity, 2 accBias, 3 gyroBias, 4 gravity, 5 trackedFeatures,
 * 6 lastImuDtSec, 7 estimatedTdSec, 8..16 RIC, 17..19 TIC.
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetVinsHealth(
        JNIEnv* env, jobject, jfloatArray out) {
    if (!out || env->GetArrayLength(out) < 20) return 0;
    VinsHealth h;
    if (!vinsGetHealth(&h)) return 0;
    jfloat v[20]{};
    v[0] = h.initialized ? 1.f : 0.f;
    v[1] = h.velocity;
    v[2] = h.accBias;
    v[3] = h.gyroBias;
    v[4] = h.gravity;
    v[5] = static_cast<float>(h.trackedFeatures);
    v[6] = static_cast<float>(h.lastImuDt);
    v[7] = static_cast<float>(h.timeOffset);
    for (int i = 0; i < 9; ++i) v[8+i] = h.ric[i];
    for (int i = 0; i < 3; ++i) v[17+i] = h.tic[i];
    env->SetFloatArrayRegion(out, 0, 20, v);
    return 20;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeCreate(JNIEnv*, jobject, jint w, jint h,
                                               jfloat fx, jfloat fy, jfloat cx, jfloat cy) {
    try { return nativeCreateImpl(w, h, fx, fy, cx, cy); }
    catch (const std::exception& ex) { LOGI("nativeCreate failed: %s", ex.what()); }
    catch (...) { LOGI("nativeCreate failed: unknown exception"); }
    return JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeDestroy(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> cameraLock(gCameraCallbackMutex);
    std::lock_guard<std::mutex> lk(gStateMutex);
    ++gNativeGeneration;
    g.reset();
    vio.reset();
    df.reset();
    ai.reset();
    tsdf.reset();
    targetG.reset();
    targetTsdf.reset();
    targetMaskMat.release();
    targetMaskStats = TargetMaskStats{};
    presenceFailStreak = 0;
    targetDepthScaleSamples = 0;
    targetDepthScaleEma = 0.f;
    resetMeshPipeline();
    // V0.12: 销毁会话 —— 标定随会话一起作废。
    resetDepthCalibration();
    kf.reset();
    objectTracker.reset();
    snaps.clear();
    haveRetainedDepthSnap = false;
    retainedDepthSnap = FrameSnap{};
    vinsPoseOk = false;
    mobilescan3d::stereo_anchor::reset();
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnImu(JNIEnv*, jobject, jlong t,
                                               jfloat ax, jfloat ay, jfloat az,
                                               jfloat gx, jfloat gy, jfloat gz) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    vio.imu((uint64_t)t, ax, ay, az, gx, gy, gz);
    vinsInputImu((double)t * 1e-9, ax, ay, az, gx, gy, gz);
}

/** nativeOnCameraFrame 的实现体；JNI 包装负责兜住所有 C++ 异常。 */
static void nativeOnCameraFrameImpl(
        JNIEnv* e,
        jbyteArray y, jbyteArray u, jbyteArray v,
        jint w, jint h, jint rs, jint urs, jint ups, jlong frameTs, jlong vinsTs) {
    std::lock_guard<std::mutex> cameraCallbackLock(gCameraCallbackMutex);
    if (!y || !u || !v || w <= 0 || h <= 0 || w > 8192 || h > 8192 ||
        rs < w || urs <= 0 || ups <= 0 || frameTs <= 0 || vinsTs <= 0) return;
    const int64_t yRequired = int64_t(h-1)*rs + w;
    const int64_t uvRequired = int64_t((h-1)/2)*urs + int64_t((w-1)/2)*ups + 1;
    if (yRequired > e->GetArrayLength(y) || uvRequired > e->GetArrayLength(u) ||
        uvRequired > e->GetArrayLength(v)) return;
    jsize un = e->GetArrayLength(u);
    std::vector<uint8_t> ubuf((size_t)un);
    e->GetByteArrayRegion(u, 0, un, reinterpret_cast<jbyte*>(ubuf.data()));
    if (e->ExceptionCheck()) return;

    jsize vn = e->GetArrayLength(v);
    std::vector<uint8_t> vbuf((size_t)vn);
    e->GetByteArrayRegion(v, 0, vn, reinterpret_cast<jbyte*>(vbuf.data()));
    if (e->ExceptionCheck()) return;

    jsize n = e->GetArrayLength(y);
    std::vector<uint8_t> ybuf((size_t)n);
    e->GetByteArrayRegion(y, 0, n, reinterpret_cast<jbyte*>(ybuf.data()));
    if (e->ExceptionCheck()) return;

    const uint8_t* yy = ybuf.data();
    const uint8_t* uu = ubuf.data();
    const uint8_t* vv = vbuf.data();

    {
        // 轻量的统计/质量分析（df/ai/vio 为共享状态，需要加锁）
        std::lock_guard<std::mutex> lk(gStateMutex);
        df.ingestLuma(yy, w, h, rs);
        ai.analyze(yy, w, h, rs);
        vio.frame((uint64_t)frameTs, yy, w, h, rs);
    }

    // VINS 可能耗时数十至上百毫秒，持有 gStateMutex 会拖死渲染/UI 线程，
    // 因此放在锁外执行（vins_bridge 内部有自己的互斥锁）。
    std::vector<uint8_t> vinsGray((size_t)gVinsW * gVinsH);
    downscaleGray(yy, w, h, rs, vinsGray.data(), gVinsW, gVinsH);
    const auto vinsStart = std::chrono::steady_clock::now();
    vinsInputImage((double)vinsTs * 1e-9, vinsGray.data(), gVinsW, gVinsH, gVinsW);

// ------------------------------------------------------- V0.7 persistent map
    //
    // Camera thread responsibilities are intentionally light:
    //   1) capture ORB descriptors only on sparse map keyframes;
    //   2) cache the exact VINS-processed 640x480 gray frame + same-time pose.
    //
    // Full-frame ORB detection / BF matching / PnP is NOT done here. A
    // background JNI call consumes the cached frame so relocalization cannot
    // stall Camera2/VINS.
    float mapCameraRwc[9];
    float mapCameraTwc[3];

    if (vinsGetCameraPoseMatrix(
            mapCameraRwc,
            mapCameraTwc)) {
        const std::uint64_t processedTs =
            vinsLastProcessedImageTimestampNs();

        const std::uint64_t currentVinsTs =
            vinsTs > 0
                ? static_cast<std::uint64_t>(vinsTs)
                : 0u;

        const std::uint64_t dt =
            processedTs > currentVinsTs
                ? processedTs - currentVinsTs
                : currentVinsTs - processedTs;

        // Only use the gray frame whose timestamp exactly corresponds to the
        // estimator state/feature observations (2ms tolerance for rounding).
        if (processedTs > 0u &&
            dt <= 2'000'000ULL) {
            if (gPersistentRelocalizer.captureEnabled()) {
                constexpr int kMaxMapObs = 512;

                VinsWorldFeature features[kMaxMapObs];

                const int count =
                    vinsGetCurrentWorldFeatures(
                        features,
                        kMaxMapObs);

                if (count > 0) {
                    PersistentWorldObservation obs[kMaxMapObs];

                    for (int i = 0; i < count; ++i) {
                        obs[i].u = features[i].u;
                        obs[i].v = features[i].v;
                        obs[i].x = features[i].x;
                        obs[i].y = features[i].y;
                        obs[i].z = features[i].z;
                        obs[i].featureId =
                            features[i].featureId;
                    }

                    gPersistentRelocalizer.captureFrame(
                        vinsGray.data(),
                        gVinsW,
                        gVinsH,
                        gVinsW,
                        currentVinsTs,
                        mapCameraRwc,
                        mapCameraTwc,
                        obs,
                        count);
                }
            }

            if (gPersistentRelocalizer.hasLoadedMap()) {
                std::lock_guard<std::mutex> rlk(
                    gRelocFrameMutex);

                gRelocGray = vinsGray;
                gRelocFrameTimestampNs =
                    currentVinsTs;

                std::copy(
                    mapCameraRwc,
                    mapCameraRwc + 9,
                    gRelocFrameRwc);

                std::copy(
                    mapCameraTwc,
                    mapCameraTwc + 3,
                    gRelocFrameTwc);
            }
        }
    }
    const auto vinsEnd = std::chrono::steady_clock::now();
    const double vinsMs = std::chrono::duration<double, std::milli>(vinsEnd - vinsStart).count();

    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (tracker) {
        try {
            // ---- tracking 主分辨率降到 640 长边 ----
            // 原来 KLT 直接在 1280x960 的 Y 平面上跑（1,228,800 px）；降到 640 长边
            // 之后只剩约 307,200 px（约四分之一），LK 金字塔的搜索压力降一个量级。
            //
            // 注意：这里**不需要**把 tracker 输出的 bbox 再乘以缩放系数 ——
            // ObjectTracker 的 x0/y0/x1/y1 全部是按图像尺寸归一化后输出的，
            // 归一化坐标天然与分辨率无关（depth 侧也是用同一套归一化乘它的 w/h）。
            constexpr int kTrackMaxDim = 640;
            const double tScale = std::min(
                1.0, static_cast<double>(kTrackMaxDim) /
                         static_cast<double>(std::max(1, std::max(w, h))));
            const int tw = std::max(16, static_cast<int>(std::lround(w * tScale)));
            const int th = std::max(16, static_cast<int>(std::lround(h * tScale)));
            gTrackerGray.resize(static_cast<size_t>(tw) * static_cast<size_t>(th));
            downscaleGray(yy, w, h, rs, gTrackerGray.data(), tw, th);
            // Current-exposure color, before Nano initialization/update. Reuse
            // storage and cap at 320 px; stale color must never correct fresh KLT.
            constexpr int kColorMaxDim = 320;
            const double cScale = std::min(1.0, double(kColorMaxDim) / std::max(w, h));
            const int cw = std::max(16, int(std::lround(w * cScale)));
            const int ch = std::max(16, int(std::lround(h * cScale)));
            static std::vector<uint8_t> color;
            color.resize(size_t(cw) * ch * 3);
            downscaleYuvToBgr(yy, uu, vv, w, h, rs, urs, ups,
                              ubuf.size(), vbuf.size(), color.data(), cw, ch);
            tracker->setColorFrame(color.data(), cw, ch, cw * 3,
                                   static_cast<uint64_t>(frameTs));

            tracker->updateFrame(gTrackerGray.data(), tw, th, tw,
                                 static_cast<uint64_t>(frameTs));
            tracker->track(gTrackerGray.data(), tw, th, tw,
                           static_cast<uint64_t>(frameTs));

            // V0.13.11 取证：量化 Nano(TFLite) 推理耗时占 camera 帧的比例。
            // 与 Kotlin 侧 FrameProbe(测 nativeOnCameraFrame 总耗时) 对照，
            // 即可判断相机线程掉帧是否由 Nano 同步推理尖峰导致，避免盲改。
            if (tracker->nanoUpdateCalls() > 0 &&
                tracker->nanoUpdateCalls() % 30 == 0) {
                __android_log_print(ANDROID_LOG_INFO, "NanoProbe",
                                    "nanoUpdate %.1f ms calls=%d",
                                    tracker->nanoLastMs(),
                                    static_cast<int>(tracker->nanoUpdateCalls()));
            }

        } catch (const cv::Exception& ex) {
            __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Target", "OpenCV tracker exception: %s", ex.what());
        } catch (const std::exception& ex) {
            __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Target", "Tracker exception: %s", ex.what());
        } catch (...) {
            __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Target", "Unknown tracker exception");
        }
    }

    float vp[7] = {};
    const bool poseOk = vinsGetCameraPoseAt(static_cast<uint64_t>(vinsTs), vp);
    if (poseOk) {
        rawVinsT[0] = vp[0];
        rawVinsT[1] = vp[1];
        rawVinsT[2] = vp[2];
        rawVinsQ[0] = vp[3];
        rawVinsQ[1] = vp[4];
        rawVinsQ[2] = vp[5];
        rawVinsQ[3] = vp[6];
    }

    float R[9], T[3];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        lastVinsMs = vinsMs;
        vinsFrames++;
        if (poseOk &&
            std::isfinite(vp[0]) &&
            std::isfinite(vp[1]) &&
            std::isfinite(vp[2]) &&
            std::isfinite(vp[3]) &&
            std::isfinite(vp[4]) &&
            std::isfinite(vp[5]) &&
            std::isfinite(vp[6])) {
            bool sane = true;
            if (haveLastGoodVinsPose) {
                const float dx = vp[0] - lastGoodVinsT[0];
                const float dy = vp[1] - lastGoodVinsT[1];
                const float dz = vp[2] - lastGoodVinsT[2];
                const float step = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (step > 0.5f) {
                    sane = false;
                }
            }

            // V0.7.0.1: a transient bad pose must not permanently kill the session.
            if (sane) {
                if (vinsLostAfterInit) {
                    ++vinsRecoveryCount;
                }
                vinsLostAfterInit = false;
                vinsRejectStreak = 0;
                vinsRejectReason = "";
                vinsT[0] = vp[0];
                vinsT[1] = vp[1];
                vinsT[2] = vp[2];
                vinsQ[0] = vp[3];
                vinsQ[1] = vp[4];
                vinsQ[2] = vp[5];
                vinsQ[3] = vp[6];

                acceptedVinsT[0] = vp[0];
                acceptedVinsT[1] = vp[1];
                acceptedVinsT[2] = vp[2];
                acceptedVinsQ[0] = vp[3];
                acceptedVinsQ[1] = vp[4];
                acceptedVinsQ[2] = vp[5];
                acceptedVinsQ[3] = vp[6];

                if (haveLastGoodVinsPose) {
                    const float dx = vp[0] - lastGoodVinsT[0];
                    const float dy = vp[1] - lastGoodVinsT[1];
                    const float dz = vp[2] - lastGoodVinsT[2];
                    lastVinsStep = std::sqrt(dx * dx + dy * dy + dz * dz);
                }

                lastGoodVinsT[0] = vp[0];
                lastGoodVinsT[1] = vp[1];
                lastGoodVinsT[2] = vp[2];

                // 记录「带时间戳的位姿」：AR 要用 Preview 的 SENSOR_TIMESTAMP
                // 反查这一时刻的相机 pose，而不是拿此刻最新的 pose 硬套。
                {
                    RenderPoseSample rp;
                    rp.ts = static_cast<uint64_t>(frameTs);
                    quatToR(vp[3], vp[4], vp[5], vp[6], rp.R);
                    rp.t[0] = vp[0];
                    rp.t[1] = vp[1];
                    rp.t[2] = vp[2];
                    renderPoseHistory.push_back(rp);
                    while (renderPoseHistory.size() > kRenderPoseHistoryMax) {
                        renderPoseHistory.pop_front();
                    }
                }

                haveLastGoodVinsPose = true;
                vinsPoseOk = true;
            } else {
                vinsRejectCount++;
                ++vinsRejectStreak;
                vinsRejectReason = "step";
                vinsPoseOk = false;
                if (haveLastGoodVinsPose) {
                    if (!haveFirstReject) {
                        const float dx = vp[0] - lastGoodVinsT[0];
                        const float dy = vp[1] - lastGoodVinsT[1];
                        const float dz = vp[2] - lastGoodVinsT[2];
                        firstRejectStep = std::sqrt(dx * dx + dy * dy + dz * dz);
                        firstRejectT[0] = vp[0];
                        firstRejectT[1] = vp[1];
                        firstRejectT[2] = vp[2];
                        VinsHealth health;
                        if (vinsGetHealth(&health)) {
                            firstRejectVelocity = health.velocity;
                            firstRejectAccBias = health.accBias;
                            firstRejectGyroBias = health.gyroBias;
                            firstRejectTd = health.timeOffset;
                        }
                        haveFirstReject = true;
                    }
                    vinsLostAfterInit = true;
                    snaps.clear();
                    renderPoseHistory.clear();
                    haveRetainedDepthSnap = false;
                }
            }
        } else {
            if (poseOk) {
                vinsRejectCount++;
                ++vinsRejectStreak;
                vinsRejectReason = "non_finite";
            }
            if (haveLastGoodVinsPose) {
                if (!haveFirstReject) {
                    firstRejectStep = 0.f;
                    firstRejectT[0] = vp[0];
                    firstRejectT[1] = vp[1];
                    firstRejectT[2] = vp[2];
                    VinsHealth health;
                    if (vinsGetHealth(&health)) {
                        firstRejectVelocity = health.velocity;
                        firstRejectAccBias = health.accBias;
                        firstRejectGyroBias = health.gyroBias;
                        firstRejectTd = health.timeOffset;
                    }
                    haveFirstReject = true;
                }
                vinsLostAfterInit = true;
                // A missing CURRENT pose does not invalidate an earlier accepted
                // exposure still in inference. A real VINS world reset does.
                if (poseOk || vinsWorldDiscontinuous()) {
                    snaps.clear();
                    renderPoseHistory.clear();
                    haveRetainedDepthSnap = false;
                }
            }
            vinsPoseOk = false;
            vinsRejectReason = vinsWorldDiscontinuous() ? "world_reset_restart_required" : "no_time_aligned_pose";
        }

        if (vinsPoseOk) {
            makePose(R, T);

            bool accept = kf.consider((uint64_t)frameTs, ai.q().sharpness, ai.q().exposure,
                                      vio.visualNovelty(), vio.features(),
                                      T[0], T[1], T[2]);
            if (accept) {
                lastKF++;
            }

            storeSnap((uint64_t)frameTs, w, h, rs, urs, ups, yy, uu, vv, R, T);
        }
        frames++;
    }

    // No synthetic depth fallback: tracking features are not measured geometry.

}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnCameraFrame(
        JNIEnv* e, jobject,
        jbyteArray y, jbyteArray u, jbyteArray v,
        jint w, jint h, jint rs, jint urs, jint ups, jlong frameTs, jlong vinsTs) {
    // 与 nativeOnDepthMap 同样的 JNI 边界防线。
    //
    // 原函数里只有 ObjectTracker 那一小段有 try/catch；主体（VINS 前端
    // goodFeaturesToTrack / findFundamentalMat、重定位 ORB、Ceres 后端、
    // 色彩空间分支里的 OpenCV 调用）全是裸奔的。这条路径是 **30Hz 相机帧**，
    // 也是 Ceres 求解真正发生的地方（VINS 在 vinsInputImage 里跑 BA），
    // 任何一个 cv::Exception 逃出去就是 std::terminate。
    try {
        nativeOnCameraFrameImpl(e, y, u, v, w, h, rs, urs, ups, frameTs, vinsTs);
    } catch (const cv::Exception& ex) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Frame",
                            "nativeOnCameraFrame OpenCV exception: %s", ex.what());
    } catch (const std::exception& ex) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Frame",
                            "nativeOnCameraFrame exception: %s", ex.what());
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Frame",
                            "nativeOnCameraFrame unknown exception");
    }
}

/**
 * 当前帧 live 目标点：**只取本次 depth + 本次目标 Mask**，直接存相机坐标。
 *
 * ## 为什么去掉了 world 变换
 *
 * 旧实现走的是
 * `Xc,Yc,Zc -> rotatePoint(R,T) -> Xw,Yw,Zw -> 再用另一个时刻的相机 pose 投回屏幕`。
 * 而实机上 `arPoseFromTimestamp=false`（按 Preview 时间戳压根查不到对应时刻的
 * pose），于是本该贴着当前目标的点，被另一个时刻的 pose 投到了屏幕外面。
 *
 * 而当前帧的 AR **根本没必要经过 world**：这一帧的 depth 就是这台相机在
 * 这一刻拍的，点和像素本来就在同一个相机坐标系里。直接存 (Xc, Yc, Z) 之后，
 * Renderer 用 `pc = aPosition` 就画完了，**完全不依赖 VINS 位姿与时间戳**。
 *
 * 这一步的诊断价值也最大：只要 Depth Mask 是对的，绿色点就一定落在绿色目标
 * 区域里；如果点到别处去了，那问题一定在 Mask 而不在 pose 链上。
 *
 * ROI 用 ObjectTracker 的归一化 bbox（和 depth 帧共用同一套归一化坐标），
 * 再叠加本次的 Mask 过滤：**Mask 外的像素一个都不取**。
 */
static void buildTargetDebugLayer(const float* d, int w, int h,
                                 const TargetTrackInfo& ti,
                                 const cv::Mat& mask) {
    targetDebugPointCount = 0;
    targetDebugRoiPixels = 0;
    targetDebugValidPixels = 0;
    // live 层不再依赖位姿，恒为「可用」；报告里的字段名同步改成 liveSpaceOk
    targetDebugPoseOk = true;
    targetDebugTs = 0;
    if (!targetDebugEnabled) {
        return;
    }
    if (d == nullptr || w < 4 || h < 4) {
        return;
    }
    if (ti.state != TargetState::TRACKING && ti.state != TargetState::ACQUIRING) {
        return;
    }

    const int x0 = std::max(0, static_cast<int>(ti.x0 * w));
    const int y0 = std::max(0, static_cast<int>(ti.y0 * h));
    const int x1 = std::min(w - 1, static_cast<int>(ti.x1 * w));
    const int y1 = std::min(h - 1, static_cast<int>(ti.y1 * h));
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    targetDebugRoiX0 = x0;
    targetDebugRoiY0 = y0;
    targetDebugRoiX1 = x1;
    targetDebugRoiY1 = y1;

    const int roiW = x1 - x0 + 1;
    const int roiH = y1 - y0 + 1;
    targetDebugRoiPixels = roiW * roiH;

    const bool useMask = !mask.empty() && mask.cols == w && mask.rows == h &&
                         mask.type() == CV_8U;

    const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
    int step = 1;
    while ((roiW / step) * (roiH / step) > kTargetDebugMaxPoints) {
        step++;
    }

    for (int y = y0; y <= y1; y += step) {
        for (int x = x0; x <= x1; x += step) {
            if (useMask && !mask.at<uint8_t>(y, x)) {
                continue;
            }
            const float z = d[static_cast<size_t>(y) * w + x];
            if (!(z > 0.08f && z < 8.f)) {
                continue;
            }
            if (!scanRangeAccepts(z, x, y, dk.fx, dk.fy, dk.cx, dk.cy)) continue;
            targetDebugValidPixels++;
            if (targetDebugPointCount >= kTargetDebugMaxPoints) {
                continue;
            }
            const float Xc = (x - dk.cx) * z / dk.fx;
            const float Yc = (y - dk.cy) * z / dk.fy;
            float* p = targetDebugPoints.data() +
                       static_cast<size_t>(targetDebugPointCount) * 6;
            // **相机坐标**：Renderer 侧用 uPointSpace=0 时 pc = aPosition。
            p[0] = Xc;
            p[1] = Yc;
            p[2] = z;
            // 固定绿色：这一层的用途是「和真实画面比对」，颜色必须一眼可辨，
            // 不能和累计点云的真实颜色混在一起。
            p[3] = 0.20f;
            p[4] = 1.00f;
            p[5] = 0.35f;
            targetDebugPointCount++;
        }
    }
    targetDebugBuilds++;
}

/** nativeOnDepthMap 的实现体；JNI 包装负责兜住所有 C++ 异常。 */
static void nativeOnDepthMapImpl(
        JNIEnv* e, jfloatArray depth, jint w, jint h,
        jfloat confidence, jlong t, jint representation, jfloatArray sourceConfidence = nullptr) {
    // Serialize JNI callers before reusable storage; reset only takes gStateMutex.
    static std::mutex depthCallbackMutex;
    std::lock_guard<std::mutex> depthCallbackLock(depthCallbackMutex);
    if (!depth || w <= 0 || h <= 0 || w > 4096 || h > 4096 ||
        !std::isfinite(confidence) || confidence <= 0.f || representation < 0 || representation > 1) {
        return;
    }
    jsize n = e->GetArrayLength(depth);
    if (n < (jsize)(w * h)) {
        return;
    }
    const auto tFrameStart = std::chrono::steady_clock::now();
    // 复用缓冲：深度回调每帧都要一块 256² 的 float，没必要每帧分配一次。
    // 由原生 depthCallbackMutex 串行保护，不依赖 Kotlin 调用约定。
    static std::vector<float> d;
    d.resize(static_cast<size_t>(w) * h);
    e->GetFloatArrayRegion(depth, 0, w * h, d.data());
    if (e->ExceptionCheck()) return;
    static std::vector<float> sourceWeights;
    sourceWeights.clear();
    if (sourceConfidence) {
        if (e->GetArrayLength(sourceConfidence) != w*h) return;
        sourceWeights.resize(static_cast<size_t>(w)*h);
        e->GetFloatArrayRegion(sourceConfidence, 0, w*h, sourceWeights.data());
        if (e->ExceptionCheck()) return;
        depth_refinement::applySourceConfidence(d.data(), sourceWeights.data(), d.size());
    }

    static std::vector<float> refinedDepth;
    depth_refinement::spatial(d.data(), w, h, representation == 1, refinedDepth);
    d.swap(refinedDepth);

    // Fetch timestamp-aligned sparse observations before taking gStateMutex.
    // Every calibrator read/write below is protected against reset/UI diagnostics.
    float frameSamples[kMaxCalibSamples * 3];
    const int ns = vinsFeatureSamples(frameSamples, kMaxCalibSamples, static_cast<uint64_t>(t));
    // 锁等待也算进总耗时：锁本身竞争是卡顿的来源之一，只统计持锁时长会漏掉它。
    const auto tLockWait = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(gStateMutex);
    const float lockWaitMs = perfMsSince(tLockWait);
    ++gPerfFrames;
    lastLockWaitMs = lockWaitMs;
    gPerf[kPerfLock].push(lockWaitMs);
    struct FrameTiming {
        std::chrono::steady_clock::time_point start;
        ~FrameTiming() noexcept {
            try { gPerf[kPerfTotal].push(perfMsSince(start)); perfLogIfDue(); }
            catch (...) { /* Logging must never terminate an unwinding JNI call. */ }
        }
    } frameTiming{tFrameStart};
    lastTemporalRatio = 1.f;
    lastTemporalTested = lastTemporalAgree = lastTemporalDisagree = 0;
    if (vinsWorldDiscontinuous()) { ++depthEarlyWorldDiscont; return; } // Preserve the old-world model.
    const FrameSnap* match = nullptr;
    int64_t bestSnapDiffNs = kDepthSnapMaxDiffNs;
    if (haveRetainedDepthSnap && retainedDepthSnap.ts == static_cast<uint64_t>(t)) {
        match = &retainedDepthSnap;
        bestSnapDiffNs = 0;
    }
    for (const auto& snap : snaps) {
        int64_t diff =
            static_cast<int64_t>(snap.ts) -
            static_cast<int64_t>(t);
        if (diff < 0) diff = -diff;
        if (diff < bestSnapDiffNs) {
            bestSnapDiffNs = diff;
            match = &snap;
        }
    }
    if (match != nullptr) {
        ++depthSnapMatches;
        lastDepthSnapDiffNs = bestSnapDiffNs;
    } else {
        ++depthSnapMisses;
        lastDepthSnapDiffNs = -1;
    }

    if (!match) { ++depthEarlyNoSnap; return; }
    if (match->targetGeneration != targetGeneration) { ++depthEarlyStaleGen; return; }
    if(t<=0 || static_cast<uint64_t>(t)<=lastConsumedDepthTs){++duplicateDepthRejects;return;}
    lastConsumedDepthTs=static_cast<uint64_t>(t);

    // V0.9: raw model depth retained briefly for exact-timestamp stereo pairing.
    mobilescan3d::stereo_anchor::onDepthFrame(
        static_cast<uint64_t>(t),
        d.data(),
        w,
        h);

    auto config = depthCalibrator.config();
    config.allowInverseDepth = representation == 1;
    config.forceInverseDepth = representation == 1;
    depthCalibrator.setConfig(config);
    if (objectTracker && objectTracker->info().lastFrameTs == static_cast<uint64_t>(t)) {
        objectTracker->updateFromDepth(d.data(), w, h, static_cast<uint64_t>(t));
    }

    bool frozenEvidenceGood = false;
    bool frozenEvidenceContradicted = false;
    bool calibrationUpdatedThisFrame = false;
    // ---- V0.9 深度标定：VINS 稀疏深度 + stereo metric anchors ----
    //
    // VINS is still the pose/world scale. Stereo first passes a robust physical
    // meter -> VINS-world scale gate. Until that scale is stable, stereo is
    // diagnostics-only. Once stable, clean stereo samples enter the existing
    // MAD + Huber IRLS + EMA calibrator at 2-3x sample weight.
    const auto tCalib = std::chrono::steady_clock::now();
    if (depthCalibrationEnabled) {
        static std::vector<float> dPairs, zPairs;
        dPairs.clear(); zPairs.clear();
        dPairs.reserve(
            static_cast<size_t>(std::max(0, ns)) + 288u);
        zPairs.reserve(
            static_cast<size_t>(std::max(0, ns)) + 288u);

        for (int i = 0; i < ns; ++i) {
            const float nu =
                frameSamples[i * 3 + 0];
            const float nv =
                frameSamples[i * 3 + 1];
            const float vz =
                frameSamples[i * 3 + 2];

            if (!(nu >= 0.f && nu <= 1.f &&
                  nv >= 0.f && nv <= 1.f)) {
                continue;
            }
            if (!std::isfinite(vz) ||
                !(vz > 0.05f)) {
                continue;
            }

            const int px = std::min(
                w - 1,
                std::max(
                    0,
                    static_cast<int>(
                        nu * static_cast<float>(w))));
            const int py = std::min(
                h - 1,
                std::max(
                    0,
                    static_cast<int>(
                        nv * static_cast<float>(h))));

            float dv = 0.f;
            if (!depth_refinement::calibrationSample(d.data(), w, h, px, py,
                                                     representation == 1, dv)) continue;

            dPairs.push_back(dv);
            zPairs.push_back(vz);
        }

        int stereoUniqueSamples = 0;
        int stereoWeightedCopies = 0;

        // Count distinct VINS observations before weighted stereo duplicates.
        frozenEvidenceGood = epochActive && scan_policy::validatesFrozen(epochCalib, dPairs, zPairs);
        frozenEvidenceContradicted = epochActive && dPairs.size() >= 20 && !frozenEvidenceGood;
        mobilescan3d::stereo_anchor::CalibrationBatch stereoBatch;
        while (
            mobilescan3d::stereo_anchor::takeCalibrationBatch(
                &stereoBatch)) {

            const DepthCalibration base =
                depthCalibrator.calibration();

            mobilescan3d::stereo_anchor::updateWorldScale(
                base,
                stereoBatch);

            if (!mobilescan3d::stereo_anchor::worldScaleUsable()) {
                // Calibration warmup is not a geometry drop: recent geometry
                // batches remain queued until scale becomes stable.
                continue;
            }

            const float stereoToWorld =
                mobilescan3d::stereo_anchor::worldPerMeter();

            const size_t stereoN = std::min({
                stereoBatch.rawDepth.size(),
                stereoBatch.metricDepth.size(),
                stereoBatch.quality.size()});

            for (size_t i = 0; i < stereoN; ++i) {
                const float dv =
                    stereoBatch.rawDepth[i];
                const float zm =
                    stereoBatch.metricDepth[i];
                const float q =
                    stereoBatch.quality[i];

                if (!std::isfinite(dv) ||
                    !std::isfinite(zm) ||
                    !(zm > 0.12f) ||
                    q < 0.45f) {
                    continue;
                }

                const float zWorld =
                    zm * stereoToWorld;
                if (!std::isfinite(zWorld) ||
                    !(zWorld > 0.05f) ||
                    zWorld > 12.f) {
                    continue;
                }

                const int copies =
                    q >= 0.80f ? 3 : 2;

                for (int c = 0; c < copies; ++c) {
                    if (dPairs.size() >= 768u) {
                        break;
                    }
                    dPairs.push_back(dv);
                    zPairs.push_back(zWorld);
                    stereoWeightedCopies++;
                }
                stereoUniqueSamples++;
                if (dPairs.size() >= 768u) {
                    break;
                }
            }
        }

        bool calibAccepted = false;
        if (dPairs.size() >=
            static_cast<size_t>(
                depthCalibrator.config().minSamples)) {
            calibAccepted =
                depthCalibrator.update(
                    dPairs,
                    zPairs);
            calibrationUpdatedThisFrame = calibAccepted;
            if (calibAccepted) {
                calibFrames++;
            } else {
                calibRejectFrames++;
            }
        }

        // VC164: 标定样本分布探针。一次区分三种退化：
        //   (a) d 平坦  -> 单目网络输出无结构（近距/无纹理场景失效）
        //   (b) invz 平坦 -> 参考深度(VINS 稀疏)无结构，拟合必然得出 scale~0（深度被压平）
        //   (c) 两者都有结构但 scale~0 -> 拟合算法本身的问题
        if ((depthFrames % 30) == 0 && dPairs.size() >= 8) {
            std::vector<float> probeD(dPairs);
            std::vector<float> probeZ(zPairs);
            std::vector<float> probeIz;
            probeIz.reserve(zPairs.size());
            for (float zv : zPairs) {
                if (zv > 1e-4f && std::isfinite(zv)) probeIz.push_back(1.0f / zv);
            }
            std::sort(probeD.begin(), probeD.end());
            std::sort(probeZ.begin(), probeZ.end());
            std::sort(probeIz.begin(), probeIz.end());
            auto pq = [](const std::vector<float>& v, double q) -> float {
                if (v.empty()) return 0.f;
                size_t i = static_cast<size_t>(q * static_cast<double>(v.size() - 1));
                if (i >= v.size()) i = v.size() - 1;
                return v[i];
            };
            LOGI("CalibProbe ns=%d vinsN=%zu stereoUniq=%d total=%zu | d p10=%.1f p50=%.1f p90=%.1f | z p10=%.3f p50=%.3f p90=%.3f | invz p10=%.3f p50=%.3f p90=%.3f",
                 ns, static_cast<size_t>(std::max(0, ns)), stereoUniqueSamples,
                 dPairs.size(),
                 pq(probeD, 0.10), pq(probeD, 0.50), pq(probeD, 0.90),
                 pq(probeZ, 0.10), pq(probeZ, 0.50), pq(probeZ, 0.90),
                 pq(probeIz, 0.10), pq(probeIz, 0.50), pq(probeIz, 0.90));
        }

        if (stereoUniqueSamples > 0) {
            mobilescan3d::stereo_anchor::noteCalibratorContribution(
                stereoUniqueSamples,
                stereoWeightedCopies,
                calibAccepted);
        }

        lastCalibValid =
            depthCalibrator.usable();
        const DepthCalibration& c =
            depthCalibrator.calibration();
        lastCalibScale = c.scale;
        lastCalibShift = c.shift;
        lastCalibConfidence = c.confidence;
        lastCalibSamples = c.samples;
        lastCalibInverse =
            c.inverseDepthModel;
        lastCalibRefSpanRel = c.refDepthSpanRel;
        lastCalibOutInvZSpan = c.outputInvZSpan;
        lastCalibOutZSpanRel = c.outputDepthSpanRel;
    }
    // Calibrated depth is also the input to the metric-threshold target mask.
    // Before calibration, retain appearance tracking but do not classify presence
    // from uncalibrated model values.
    scanWorldPerMeter = mobilescan3d::stereo_anchor::worldScaleUsable()
        ? mobilescan3d::stereo_anchor::worldPerMeter() : 1.f;
    const bool calibEnabledEff = depthCalibrationEnabled;
    const bool calibratedNow = calibEnabledEff && depthCalibrator.usable();
    static std::vector<float> zCal;
    zCal.clear();
    const float* depthForFusion = d.data();
    if (calibratedNow) {
        const DepthCalibration& c = depthCalibrator.calibration();
        zCal.assign((size_t)w * h, 0.f);
        for (size_t i = 0; i < zCal.size(); ++i) {
            const float dv = d[i];
            if (!std::isfinite(dv)) {
                continue;
            }
            const float zz = c.toMetric(dv, 0.f);
            if (zz > 0.f && std::isfinite(zz)) {
                zCal[i] = zz;
            }
        }
        depthForFusion = zCal.data();
    }

    // V0.10: measure the exact depth field about to enter TSDF.
    float fusionDepthMinFrame = 1e30f;
    float fusionDepthMaxFrame = 0.f;
    double fusionDepthSumFrame = 0.0;
    uint64_t fusionDepthValidFrame = 0;
    uint64_t fusionDepthFallbackFrame = 0;
    const size_t fusionCount = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < fusionCount; ++i) {
        const float zf = depthForFusion[i];
        if (zf > 0.f && std::isfinite(zf)) {
            fusionDepthMinFrame = std::min(fusionDepthMinFrame, zf);
            fusionDepthMaxFrame = std::max(fusionDepthMaxFrame, zf);
            fusionDepthSumFrame += static_cast<double>(zf);
            fusionDepthValidFrame++;
        } else if (calibratedNow && std::isfinite(d[i])) {
            fusionDepthFallbackFrame++;
        }
    }
    if (fusionDepthValidFrame == 0) {
        fusionDepthMinFrame = 0.f;
    }

    lastFusionDepthCalibrated = calibratedNow && fusionDepthValidFrame > 0;
    lastFusionDepthMin = fusionDepthMinFrame;
    lastFusionDepthMax = fusionDepthMaxFrame;
    lastFusionDepthMean =
        fusionDepthValidFrame > 0
            ? static_cast<float>(
                fusionDepthSumFrame /
                static_cast<double>(fusionDepthValidFrame))
            : 0.f;
    lastFusionDepthValid = fusionDepthValidFrame;
    lastFusionDepthConversionFallback = fusionDepthFallbackFrame;

    // ---- V0.12 Fusion Calibration Epoch 状态机 ----
    // 判据一律在**同一个 raw 工作点**上比较，这样场景远近变化不会污染结论。
    {
        // The frozen reference is immutable; no per-frame median allocation.
        // Relative inverse q has no physical [0.05,60] range or sign constraint.
        const float rawNow = epochActive ? epochRefRaw :
            rawDepthSampleMedian(d.data(), w, h, representation == 1);
        const DepthCalibration liveCal = depthCalibrator.calibration();
        if (!epochActive && !haveEpochPrevZ && std::isfinite(rawNow)) epochRefRaw = rawNow;
        const float zLiveNow = calibratedNow ? liveCal.toMetric(epochRefRaw, 0.f) : 0.f;
        const bool liveOk = std::isfinite(rawNow) && std::isfinite(zLiveNow) && zLiveNow > 0.f;

        // VC171: update EMA only on a newly accepted fit, never a cached replay.
        // Maintain live calibration EMA as a drift diagnostic.
        // 瞬时 live 因低接受率（~28%）+ 场景相关拟合而噪声大，直接比 frozen
        // 会被误判为永久漂移 -> susp=1 再也解不开。EMA 平滑后对比才稳。
        if (calibrationUpdatedThisFrame && calibratedNow && liveCal.valid &&
            std::isfinite(liveCal.scale) && std::isfinite(liveCal.shift)) {
            if (!epochLiveEmaInit || epochLiveEmaInvDepth != liveCal.inverseDepthModel) {
                epochLiveEmaScale = liveCal.scale;
                epochLiveEmaShift = liveCal.shift;
                epochLiveEmaInvDepth = liveCal.inverseDepthModel;
                epochLiveEmaVar = 0.f;
                epochLiveEmaInit = true;
            } else {
                epochLiveEmaScale += kEpochLiveEmaNew * (liveCal.scale - epochLiveEmaScale);
                epochLiveEmaShift += kEpochLiveEmaNew * (liveCal.shift - epochLiveEmaShift);
                epochLiveEmaInvDepth = liveCal.inverseDepthModel;
                const float dev = liveCal.scale - epochLiveEmaScale;
                epochLiveEmaVar = 0.9f * epochLiveEmaVar + 0.1f * dev * dev;
            }
        }

        if (!calibEnabledEff) {
            // 退路：整条标定链被关掉时，退回「融合 raw depth」的旧行为。
            // 否则「关掉标定」会静默变成「永远不融合」—— 比标定不准更糟。
            // 同时把 active 标记成 1，让诊断里的 active 与实际融合行为一致。
            if (!epochActive) {
                epochActive = true;
                epochStableStreak = kEpochStableFrames;
                epochBadStreak = 0;
                ++epochIndex;
                ++epochOpens;
                resetTemporalReferenceForEpochChange();
            }
        } else if (epochActive && frozenEvidenceGood) {
            // Current sparse geometry supports the mapping that built this map.
            // A disagreeing live fit cannot veto this independent observation.
            epochBadStreak = epochSuspendStreak = epochCatastrophicStreak = 0;
            epochResumeStreak = 0;
        } else if (!calibratedNow || !liveOk) {
            // V0.13.34：标定失效**只作诊断计数，不再暂停写几何**。
            //
            // 实机证据（PLK110 vc172 扫描 20:46~20:53）：depthCalibrator.usable()=1
            // 但 susp=1、badStreak=9~41、driftRejects=0，FusionDiag depthFrames=60
            // fused=2 gated=57。本分支由 liveOk（对固定 epochRefRaw 求值）触发 ——
            // live 拟合换工作域后对该 raw 点求不出正值，这是 live 的属性，不是
            // epoch 的属性。而 epoch 融合深度走 epochCalib 冻结参数（见下方
            // epochFusionDepth），与 live 标定器的可用性**完全无关**；用 live 状态
            // 去停冻结链路既无必要，又因恢复滞回（10%~20%）死区永远解不开 ——
            // 点云停长、旧模型跟着位姿转，用户看到「模型与物体错位/穿帮」。
            //
            // 保留 badStreak 计数作为「live 标定失效持续帧数」诊断量
            // （EpochProbe 回显）；真正的写入门只剩下方灾难级条件。
            ++epochBadStreak;
            epochResumeStreak = 0;
            if (!epochActive) {
                epochStableStreak = 0;
                haveEpochPrevZ = false;
            }
        } else if (!epochActive) {
            // 还没开 epoch：要求连续 kEpochStableFrames 帧稳定（相邻帧之间
            // 同一工作点算出的 z 变化不超过 kEpochRebuildRatio）。
            const bool stableStep =
                haveEpochPrevZ && epochPrevZ > 0.f &&
                std::fabs(zLiveNow - epochPrevZ) / epochPrevZ <= kEpochRebuildRatio;
            epochStableStreak = stableStep ? (epochStableStreak + 1) : 1;
            epochPrevZ = zLiveNow;
            haveEpochPrevZ = true;
            epochBadStreak = 0;
            if (epochStableStreak >= kEpochStableFrames) {
                epochCalib = liveCal;
                epochCalib.valid = true;
                epochRefZ = zLiveNow;
                epochActive = true;
                // V0.13.26 诊断：记下冻结那一刻的标定成熟度。
                epochFrozenSamples = liveCal.samples;
                epochFrozenConfidence = liveCal.confidence;
                epochStableStreak = 0;
                epochBadStreak = 0;
                epochLastDriftRel = 0.f;
                // V0.13：刚冻结的 epoch 一定处于「未暂停」状态。
                epochSuspended = false;
                epochSuspendStreak = 0;
                epochResumeStreak = 0;
                epochCatastrophicStreak = 0;
                ++epochIndex;
                ++epochOpens;
                resetTemporalReferenceForEpochChange();
            }
        } else {
            // Monitor the frozen/live mapping disagreement at the SAME raw value.
            // V0.13.34：分歧本身**不再暂停写几何**（降级为诊断），只有灾难级
            // 分歧（>75% 且连续 kEpochDriftSuspendFrames 帧 且稀疏证据不支持冻结）
            // 才暂停 —— 那是「标定模型翻转 / 真尺度崩坏」的形态，冻结参数自身
            // 不可信，暂停是合理的。普通的 20%~75% 分歧来自 live 拟合的场景
            // 相关性（实测 scale 0.0006↔0.0023、z 中位数 0.36~0.53m），冻结
            // 几何自洽，继续用冻结尺度写不会产生尺度接缝。
            const float zFrozen = epochCalib.toMetric(epochRefRaw, 0.f);
            DepthCalibration emaLive;
            emaLive.scale = epochLiveEmaScale;
            emaLive.shift = epochLiveEmaShift;
            emaLive.inverseDepthModel = epochLiveEmaInvDepth;
            emaLive.valid = epochLiveEmaInit;
            const float zLiveAtRef = emaLive.toMetric(epochRefRaw, 0.f);
            const bool bothOk = std::isfinite(zFrozen) && zFrozen > 0.f &&
                                std::isfinite(zLiveAtRef) && zLiveAtRef > 0.f;
            if (bothOk) {
                epochLastDriftRel = std::fabs(zLiveAtRef - zFrozen) / zFrozen;
            }
            if (bothOk && epochLastDriftRel > kEpochDriftSuspendRatio) {
                // 诊断计数：保留原「漂移超标帧数」语义（EpochProbe 回显），
                // 但不再累加 suspendStreak / 不再置 susp。
                ++epochDriftRejects;
            }
            const bool catastrophic =
                bothOk && epochLastDriftRel > kEpochCatastrophicRatio && !frozenEvidenceGood;
            if (catastrophic) {
                ++epochBadStreak;
                ++epochSuspendStreak;
                epochResumeStreak = 0;
                if (!epochSuspended && epochSuspendStreak >= kEpochDriftSuspendFrames) {
                    epochSuspended = true;
                    epochSuspendedSinceNs = static_cast<int64_t>(t);
                    ++epochSuspendEvents;
                }
                // Even prolonged disagreement cannot justify erasing a scan.
                epochCatastrophicStreak = std::min(epochCatastrophicStreak + 1,
                                                   kEpochCatastrophicFrames);
            } else if (bothOk && epochLastDriftRel <= kEpochDriftResumeRatio) {
                // 漂移回落到恢复门限内：清计数、按 kEpochResumeFrames 解除暂停。
                // 恢复后 epochCalib 不变，新几何仍用冻结尺度，不会产生尺度接缝。
                epochBadStreak = 0;
                epochSuspendStreak = 0;
                epochCatastrophicStreak = 0;
                if (epochSuspended) {
                    ++epochResumeStreak;
                    if (epochResumeStreak >= kEpochResumeFrames) {
                        epochSuspended = false;
                        epochResumeStreak = 0;
                    }
                }
            } else {
                // 10%~20% 恢复带与 20%~75% 分歧带：既往不咎 —— 既不累加也不暂停。
                // V0.13.32 的死区滞回正是 susp 永久挂起（点云停长）的根因。
                epochBadStreak = 0;
                epochSuspendStreak = 0;
                epochResumeStreak = 0;
            }
            // 灾难暂停的限时自动恢复：见 kEpochCatastrophicMaxSuspendNs 注释。
            // 冻结链路自洽，停写只会得到半截扫描；三重条件的灾难门误报率低，
            // 但万一它被一个持续 40%+ 的噪声带顶住，10s 后强制恢复。
            if (epochSuspended && epochSuspendedSinceNs > 0 &&
                static_cast<int64_t>(t) - epochSuspendedSinceNs >
                    kEpochCatastrophicMaxSuspendNs) {
                epochSuspended = false;
                epochResumeStreak = 0;
                epochBadStreak = epochSuspendStreak = epochCatastrophicStreak = 0;
                ++epochCatastrophicTimeoutResumes;
                // ---- V0.13.36 (vc176)：超时恢复时用健康 live 重锚 ----
                // 实机证据（PLK110 23:03 会话）：近处健康冻结（scale=+0.0026/
                // shift=4.97, refZ=0.17）后用户后退扫全屋，live 拟合健康自适应
                // （liveEma stable=1, shift 4.97→0.79, fusionDepth 0.14→0.46m），
                // frozen 在工作点失真 105% → 灾难挂起。旧的纯恢复会继续用失真
                // frozen 写几何 → 立刻再次触发灾难 → 「停写 10s → 恢复 → 再挂」
                // 循环，扫描永远停在冻结时刻的场景尺度上。
                // 重锚接受尺度接缝（旧几何保留，不再追加），换取后续几何正确：
                // 对拉丝薄壳来说接缝无意义，对新区域这是唯一能写对的路径。
                // 防线：仅当 live 标定可用、工作点映射有限为正、且分歧仍是
                // 灾难级（>75%）时才重锚 —— 分歧已回落说明 frozen 仍可信，
                // 保持不动。epochIndex 不动（重锚不是新 epoch 的稳定链路）。
                if (depthCalibrator.usable()) {
                    const DepthCalibration liveAnchor =
                        depthCalibrator.calibration();
                    const float zLiveAnchor =
                        liveAnchor.toMetric(epochRefRaw, 0.f);
                    const float zFrozenAnchor =
                        epochCalib.toMetric(epochRefRaw, 0.f);
                    if (liveAnchor.scale > 0.f &&
                        std::isfinite(zLiveAnchor) && zLiveAnchor > 0.f &&
                        std::isfinite(zFrozenAnchor) && zFrozenAnchor > 0.f &&
                        std::fabs(zLiveAnchor - zFrozenAnchor) / zFrozenAnchor >
                            kEpochCatastrophicRatio) {
                        epochCalib = liveAnchor;
                        epochCalib.valid = true;
                        epochRefZ = zLiveAnchor;
                        epochLastDriftRel = 0.f;
                        ++epochReanchors;
                        // 重锚换了映射域，旧参考跨域比对只会全面冲突，作废。
                        resetTemporalReferenceForEpochChange();
                        LOGI("EpochReanchor: reanchored to live calib "
                             "scale=%.5f shift=%.4f at refRaw=%.2f "
                             "(zFrozen=%.4f zLive=%.4f) total=%llu",
                             static_cast<double>(liveAnchor.scale),
                             static_cast<double>(liveAnchor.shift),
                             static_cast<double>(epochRefRaw),
                             static_cast<double>(zFrozenAnchor),
                             static_cast<double>(zLiveAnchor),
                             (unsigned long long)epochReanchors);
                    }
                }
            }
            // Do not change epochCalib while old geometry is retained. A stable
            // new fit alone cannot align already fused points to the new mapping.
            // Frozen sparse evidence below can safely resume the original scale.

        }
    }

    // Require six distinct supporting exposures; tolerate short sample gaps,
    // but expire stale support and cancel on observed geometric contradiction.
    const bool frozenConfirmed = frozenEvidenceWindow.observe(
        static_cast<uint64_t>(t), epochActive && frozenEvidenceGood,
        frozenEvidenceContradicted, kEpochResumeFrames);
    frozenEvidenceStreak = frozenEvidenceWindow.count();
    if (epochActive && epochSuspended && frozenConfirmed) {
        epochSuspended = false;
        epochBadStreak = epochSuspendStreak = epochResumeStreak = epochCatastrophicStreak = 0;
        ++frozenEvidenceRecoveries;
    }
    gPerf[kPerfCalib].push(perfMsSince(tCalib));

    // epoch 生效时才做「冻结参数」的第二份换算 —— 这一份才是真正进 TSDF 的。
    // 复用容量但每帧 clear，保留 !zEpoch.empty() 的逐帧有效性语义。
    static std::vector<float> zEpoch;
    zEpoch.clear();
    const float* epochFusionDepth = nullptr;
    // Masks use the same frozen mapping even while integration is suspended.
    // Only epochFusionDepth grants permission to write new geometry.
    if (epochActive && calibEnabledEff) {
        zEpoch.assign((size_t)w * h, 0.f);
        for (size_t i = 0; i < zEpoch.size(); ++i) {
            const float dv = d[i];
            if (!std::isfinite(dv)) {
                continue;
            }
            const float zz = epochCalib.toMetric(dv, 0.f);
            if (zz > 0.f && std::isfinite(zz)) {
                zEpoch[i] = zz;
            }
        }
        depthForFusion = zEpoch.data();
        if (!epochSuspended) epochFusionDepth = zEpoch.data();
    }

    df.ingestExternalDepth(d.data(), w, h, confidence, (uint64_t)t);
    haveExternalDepth = true;
    depthFrames++;

    if ((depthFrames % 60) == 0) {
        // 保持原字段前缀不变（外部脚本在解析），追加时序/护罩归因。
        // ratio/tested/agree 来自上一帧真实结果；stale 是「参考过旧而放行」的帧数，
        // 它一涨就说明死锁修复生效；guard 三项说明重复壳护罩到底拦了多少。
        LOGI("FusionDiag depthFrames=%llu usable=%d epochActive=%d epochSusp=%d fused=%llu gated=%llu "
             "| temporal checks=%llu rejects=%llu ratio=%.3f tested=%d agree=%d stale=%llu refAgeMs=%.0f "
             "staleW=%llu staleRej=%llu "
             "| guard refs=%lld checks=%llu rejected=%llu tested=%d ratio=%.3f",
             (unsigned long long)depthFrames, depthCalibrator.usable() ? 1 : 0,
             epochActive ? 1 : 0, epochSuspended ? 1 : 0,
             (unsigned long long)fusionFusedFrames, (unsigned long long)fusionGatedFrames,
             (unsigned long long)temporalChecks, (unsigned long long)temporalRejects,
             static_cast<double>(lastDiagTemporalRatio), lastDiagTemporalTested,
             lastDiagTemporalAgree, (unsigned long long)temporalRefStale,
             static_cast<double>(lastTemporalRefAgeMs),
             (unsigned long long)temporalStaleWrites,
             (unsigned long long)temporalStaleMotionRejects,
             static_cast<long long>(fusionGuard.references()),
             (unsigned long long)fusionGuard.checks,
             (unsigned long long)fusionGuard.rejected,
             fusionGuard.lastTested,
             static_cast<double>(fusionGuard.lastRatio));
        LOGI("FrameICP attempts=%llu applied=%llu rejectSamples=%llu rejectOverlap=%llu "
             "rejectBound=%llu rejectDegenerate=%llu lastApplied=%d lastReason=%d "
             "lastInliers=%d lastRmse=%.4f lastTrans=%.4f lastRotDeg=%.3f",
             (unsigned long long)icpAttempts, (unsigned long long)icpAppliedCount,
             (unsigned long long)icpRejectSamples, (unsigned long long)icpRejectOverlap,
             (unsigned long long)icpRejectBound, (unsigned long long)icpRejectDegenerate,
             lastIcpDiag.applied ? 1 : 0, lastIcpDiag.reason, lastIcpDiag.inliers,
             static_cast<double>(lastIcpDiag.rmseM), static_cast<double>(lastIcpDiag.transM),
             static_cast<double>(lastIcpDiag.rotDeg));
    }

    // V0.13.22 深度取证专用行（每 30 帧）：定位「模型发平、与实物对不上位置」。
    //  - calib scale/shift：米制换算是否合理（raw 实机常 ~4.4）
    //  - fusionDepth min/max/mean：真正要进 TSDF 的深度值域（米）
    //  - scanMax/worldPerMeter：扫描范围与 VINS 世界尺度是否自洽
    if ((depthFrames % 30) == 0) {
        LOGI("DepthProbe calib valid=%d scale=%.4f shift=%.4f conf=%.3f samples=%lld "
             "spanRel=%.3f invZSpan=%.3f zSpanRel=%.3f acc=%llu rej=%llu reason=%s | "
             "fusionDepth min=%.3f max=%.3f mean=%.3f valid=%llu fallback=%llu | "
             "scanMax=%.2f worldPerMeter=%.4f worldScaleUsable=%d | epochA=%d susp=%d",
             lastCalibValid ? 1 : 0,
             static_cast<double>(lastCalibScale),
             static_cast<double>(lastCalibShift),
             static_cast<double>(lastCalibConfidence),
             static_cast<long long>(lastCalibSamples),
             static_cast<double>(lastCalibRefSpanRel),
             static_cast<double>(lastCalibOutInvZSpan),
             static_cast<double>(lastCalibOutZSpanRel),
             (unsigned long long)calibFrames,
             (unsigned long long)calibRejectFrames,
             depthCalibrator.lastRejectReason() ? depthCalibrator.lastRejectReason() : "",
             static_cast<double>(lastFusionDepthMin),
             static_cast<double>(lastFusionDepthMax),
             static_cast<double>(lastFusionDepthMean),
             (unsigned long long)lastFusionDepthValid,
             (unsigned long long)lastFusionDepthConversionFallback,
             static_cast<double>(scanMaxDistanceMeters),
             static_cast<double>(scanWorldPerMeter),
             mobilescan3d::stereo_anchor::worldScaleUsable() ? 1 : 0,
             epochActive ? 1 : 0, epochSuspended ? 1 : 0);
    }

    // V0.13.23 取证（每 30 帧，纯只读）：定位「模型发平 / 与实物对不上位置」的
    // 两条候选根因 —— (A) 世界尺度 worldPerMeter 从未建立（停留在默认 1.0），
    // (B) epoch 被漂移暂停后再也恢复不了（epochFusionDepth 恒空 => 深度全被 gated）。
    // 一次性把两边的计数器全打出来，避免再来回猜。
    if ((depthFrames % 30) == 0) {
        LOGI("EpochProbe active=%d susp=%d opens=%llu index=%llu suspendEvents=%llu suspendedFrames=%llu "
             "driftRejects=%llu lastDriftRel=%.4f frozenRecoveries=%llu stableStreak=%d badStreak=%d suspendStreak=%d "
             "| frozen scale=%.4f shift=%.4f inv=%d refRaw=%.4f refZ=%.4f frozenN=%d frozenConf=%.3f "
             "| live scale=%.4f shift=%.4f conf=%.3f samples=%d inv=%d "
             "| norm A=%.5f B=%.5f reparms=%llu lastScaleRel=%.4f changesWhileEpoch=%llu "
             "| liveEma scale=%.4f shift=%.4f var=%.6f relVar=%.4f stable=%d reanchors=%llu "
             "| catTimeoutResumes=%llu (V0.13.34: drift over 20pc only counts diagnostics; suspend only on catastrophic)",
             (int)epochActive, (int)epochSuspended,
             (unsigned long long)epochOpens, (unsigned long long)epochIndex,
             (unsigned long long)epochSuspendEvents, (unsigned long long)fusionSuspendedFrames,
             (unsigned long long)epochDriftRejects, static_cast<double>(epochLastDriftRel),
             (unsigned long long)frozenEvidenceRecoveries,
             epochStableStreak, epochBadStreak, epochSuspendStreak,
             static_cast<double>(epochCalib.scale), static_cast<double>(epochCalib.shift),
             (int)epochCalib.inverseDepthModel,
             static_cast<double>(epochRefRaw), static_cast<double>(epochRefZ),
             epochFrozenSamples, static_cast<double>(epochFrozenConfidence),
             static_cast<double>(lastCalibScale), static_cast<double>(lastCalibShift),
             static_cast<double>(lastCalibConfidence), lastCalibSamples, (int)lastCalibInverse,
             static_cast<double>(gDepthNormA), static_cast<double>(gDepthNormB),
             (unsigned long long)gDepthNormReparams,
             static_cast<double>(gDepthNormLastScaleRel),
             (unsigned long long)gDepthNormChangesWhileEpoch,
             static_cast<double>(epochLiveEmaScale), static_cast<double>(epochLiveEmaShift),
             static_cast<double>(epochLiveEmaVar),
             static_cast<double>((epochLiveEmaScale * epochLiveEmaScale > 1e-12f)
                                     ? epochLiveEmaVar / (epochLiveEmaScale * epochLiveEmaScale)
                                     : 1e9f),
             (int)(epochLiveEmaInit && (epochLiveEmaScale * epochLiveEmaScale > 1e-12f)
                       ? epochLiveEmaVar / (epochLiveEmaScale * epochLiveEmaScale) < kEpochLiveVarSuspendRel
                       : 0),
             (unsigned long long)epochReanchors,
             (unsigned long long)epochCatastrophicTimeoutResumes);

        float st[mobilescan3d::stereo_anchor::kStatsSlots];
        mobilescan3d::stereo_anchor::fillStats(st);
        LOGI("StereoProbe worldPerMeter=%.4f stable=%.0f frameScale=%.4f medRel=%.3f denseRel=%.3f samples=%.0f "
             "| scaleAcc=%.0f scaleRej=%.0f(noBase=%.0f few=%.0f resid=%.0f jump=%.0f) "
             "| calibStereo acc=%.0f rej=%.0f uniq=%.0f copies=%.0f "
             "| geom batches=%.0f stale=%.0f dropNoPose=%.0f dropNoScale=%.0f dropResid=%.0f "
             "| submitted=%.0f anchors=%.0f matched=%.0f inputRej=%.0f seen=%.0f miss=%.0f",
             static_cast<double>(st[9]), static_cast<double>(st[14]),
             static_cast<double>(st[10]), static_cast<double>(st[11]), static_cast<double>(st[12]),
             static_cast<double>(st[13]),
             static_cast<double>(st[15]), static_cast<double>(st[16]),
             static_cast<double>(st[17]), static_cast<double>(st[18]),
             static_cast<double>(st[19]), static_cast<double>(st[20]),
             static_cast<double>(st[22]), static_cast<double>(st[23]),
             static_cast<double>(st[24]), static_cast<double>(st[25]),
             static_cast<double>(st[26]), static_cast<double>(st[27]),
             static_cast<double>(st[28]), static_cast<double>(st[29]),
             static_cast<double>(st[30]),
             static_cast<double>(st[0]), static_cast<double>(st[1]),
             static_cast<double>(st[5]), static_cast<double>(st[2]),
             static_cast<double>(st[4]), static_cast<double>(st[7]));
    }

    // 先抓一份当前 target 状态：mask / presence / 调试层都要用。
    //
    // 注意：本函数在这一段之前**已经持有 gStateMutex**（见上面的 lock_guard lk），
    // 所以这里只能直接读全局 objectTracker，绝不能再 lock 一次 ——
    // std::mutex 不可重入，再锁一次就是自锁死。ObjectTracker 内部有自己的
    // mutex_，isTracking()/info() 都会自己去拿，不会和外层冲突。
    // Inference completes later: use the source exposure's target rectangle,
    // not the rectangle from the camera's newer viewpoint.
    const TargetTrackInfo ti = match->target;
    const bool haveTi = objectTracker && ti.state == TargetState::TRACKING;
    const bool targetStateIsCurrent = objectTracker &&
        objectTracker->info().lastFrameTs == static_cast<uint64_t>(t);

    // VC160: sample the LIVE calibrated field, before scan-range clipping. Raw
    // monocular values and the diagnostic shadow scale must never drive lenses.
    cameraDistance = {};
    cameraDistanceTs = static_cast<uint64_t>(t);
    cameraDistanceSource = 0;
    const bool reliableRangeTarget = haveTi && ti.presenceValid &&
        std::isfinite(ti.confidence) && ti.confidence >= .7f;
    if (calibratedNow && !epochSuspended && lastCalibConfidence >= .5f && !zCal.empty() &&
        (ti.state == TargetState::OFF || ti.state == TargetState::ARMED || reliableRangeTarget)) {
        const bool reliableTarget = reliableRangeTarget;
        const bool stereoMetric = mobilescan3d::stereo_anchor::worldScaleUsable();
        cameraDistance = camera_distance::estimate(zCal.data(), w, h,
            stereoMetric ? scanWorldPerMeter : 1.f,
            reliableTarget ? ti.x0 : .3f, reliableTarget ? ti.y0 : .3f,
            reliableTarget ? ti.x1 : .7f, reliableTarget ? ti.y1 : .7f,
            sourceWeights.empty() ? nullptr : sourceWeights.data());
        if (cameraDistance.meters > 0) cameraDistanceSource = stereoMetric ? 2 : 1;
    }

    // ---- 目标分割 Mask + PresenceGate ----
    bool presenceOk = false;
    bool maskOk = false;
    if (haveTi && (calibratedNow || (epochActive && calibEnabledEff))) {
        const int bx0 = std::max(0, static_cast<int>(ti.x0 * w));
        const int by0 = std::max(0, static_cast<int>(ti.y0 * h));
        const int bx1 = std::min(w - 1, static_cast<int>(ti.x1 * w));
        const int by1 = std::min(h - 1, static_cast<int>(ti.y1 * h));
        if (bx1 > bx0 && by1 > by0) {
            const cv::Rect searchBox(bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1);
            const cv::Point seed((bx0 + bx1) / 2, (by0 + by1) / 2);
            targetMaskEngine.build(depthForFusion, w, h, searchBox, seed,
                                   targetMaskMat, targetMaskStats);
            targetMaskFrames++;
            maskOk = targetMaskStats.valid && !targetMaskStats.overExpanded &&
                     targetMaskStats.area >= kPresenceMinMaskArea;
            // V0.13 Mask temporal gate：Adaptive Mask 管住了「一帧之内的
            // 面积/边界」，这里再管住「两帧之间的跳变」。
            updateMaskTemporalGate(ti, w, h);
        } else {
            // 目标整块出界：searchBox 退化成空矩形，无从建 mask
            targetMaskStats = TargetMaskStats{};
            targetMaskStats.rejectReason = "target bbox off screen";
            targetMaskMat.release();
            resetMaskTemporalGate();
        }

        const PresenceDecision pd = evaluatePresence(ti, targetMaskStats, w, h);
        presenceOk = pd.valid;
        if ((depthFrames % 60) == 0 && !presenceOk) {
            LOGI("PresenceDiag nano=%d nanoScore=%.3f conf=%.3f inlier=%.3f pts=%d "
                 "maskV=%d over=%d area=%d centerOff=%.3f "
                 "appearanceOk=%d motionOk=%d maskOk=%d centerOk=%d",
                 (int)ti.nanoLoaded, ti.nanoScore, ti.confidence, ti.inlierRatio,
                 ti.trackedPoints, (int)pd.maskOk, (int)targetMaskStats.overExpanded,
                 targetMaskStats.area, pd.centerOffsetRatio,
                 (int)pd.appearanceOk, (int)pd.motionOk, (int)pd.maskOk, (int)pd.centerOk);
        }
        if ((depthFrames % 60) == 0) {
            // V0.13.19.3 实时目标诊断：identity/mask/epoch-drift/depthscale 周期打印，
            // 用于定位「目标漂移 / 锁不住 / 粗糙」根因。纯只读打印，不改行为。
            LOGI("TargetDiag anchor=%d idScore=%.3f idRej=%llu bboxScale=%.3f "
                 "maskArea=%.0f areaRatio=%.3f borderTouch=%.3f over=%d "
                 "epochSusp=%d driftRej=%llu lastDriftRel=%.4f frozenScale=%.4f frozenShift=%.4f "
                 "depthApplied=%.4f depthEma=%.4f depthSamp=%d "
                 "targetG=%llu targetVox=%llu targetFuse=%llu",
                 (int)ti.identityAnchorReady,
                 ti.identityScore, (unsigned long long)ti.identityRejects,
                 ti.bboxScaleFromInitial,
                 (float)targetMaskStats.area, targetMaskStats.areaRatio,
                 targetMaskStats.borderTouch, (int)targetMaskStats.overExpanded,
                 (int)epochSuspended, (unsigned long long)epochDriftRejects,
                 epochLastDriftRel, epochCalib.scale, epochCalib.shift,
                 currentTargetDepthScale(), targetDepthScaleEma, (int)targetDepthScaleSamples,
                 (unsigned long long)targetG.count(),
                 (unsigned long long)targetTsdf.voxels(),
                 (unsigned long long)targetFuseFrames);
        }
        if (presenceOk) {
            presenceFailStreak = 0;
        } else {
            presenceFailStreak++;
            if (targetStateIsCurrent && presenceFailStreak >= kPresenceFailFrames) {
                // 连续两帧判「目标不在」：立刻退回 REACQUIRING 并清 confidence。
                // UI 同一帧就会把绿框收掉 —— 绝不能等 bbox 滑出画面才反应，
                // 因为「物体走了但 tracker 停在墙上」时 bbox 永远不会出界。
                objectTracker->markPresenceLost();
                presenceLostFrames++;
            }
        }
        // 只更新诊断位（不参与状态机）：连续 1 帧失败就已经不该再画绿框了。
        if (targetStateIsCurrent) objectTracker->setPresenceValid(presenceFailStreak == 0);
    } else {
        targetMaskStats = TargetMaskStats{};
        targetMaskMat.release();
        resetMaskTemporalGate();
        presenceFailStreak = 0;
    }

    // V0.7.0.1 current-frame target debug:
    // camera-space only; it must not depend on a VINS pose snapshot.
    //
    // V0.12：标定还没 usable 时**宁可暂时不画**。此时 depthForFusion 还是 raw
    // 量级（实机 ~4~5），画出来就是一个「看起来已经扫出来了」的错误大壳 ——
    // 用户会以为模型长歪了，实际只是标定没到位。空白比错误的大壳诚实。
    if (haveTi && maskOk && presenceOk && maskTemporalOk &&
        calibEnabledEff && (calibratedNow || epochActive)) {
        buildTargetDebugLayer(
            depthForFusion, w, h, ti, targetMaskMat);
        targetDebugTs = static_cast<uint64_t>(t);
    } else {
        targetDebugPointCount = 0;
        targetDebugValidPixels = 0;
    }

    if (match != nullptr) {
        // ---- 时序一致性：上一帧深度按相对位姿重投影到当前帧 ----
        // 单目深度网络在细节上本来就会抖，动得快时更明显。这里把「同一表面在
        // 两帧里深度不一致」的像素识别出来，用一致性比例给融合降权。
        float conf = confidence;
        bool temporalAccepted = true;
        bool temporalMaskValid = false;
        // V0.13.39：过期参考帧是否因大位移被拒写（在下方 fusionDepth 选择后生效）。
        bool staleMotionBlocked = false;
        const auto temporalK = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
        const bool sameK = temporalK.fx == lastFuseK[0] && temporalK.fy == lastFuseK[1] &&
                           temporalK.cx == lastFuseK[2] && temporalK.cy == lastFuseK[3];
        const bool metricDepth = epochActive && calibEnabledEff;
        // ---- V0.13.41 帧到模型 ICP 位姿精修（Option A）----
        // effR/effT 是本帧后续**所有**位姿消费者的统一入口：时序比对、
        // stale-motion 检查、fusionGuard、TSDF/surfel/target 写入、参考链
        // 更新。ICP 采纳时为修正位姿，否则恒等于 VINS 原始位姿（旧版行为）。
        const float* effR = match->R;
        const float* effT = match->t;
        float icpOutR[9];
        float icpOutT[3];
        if (metricDepth && !zEpoch.empty() &&
            g.compressedCount() >= kIcpMinModelPoints) {
            ++icpAttempts;
            lastIcpDiag = frame_icp::frameToModelIcp(
                zEpoch.data(), w, h, temporalK.fx, temporalK.fy,
                temporalK.cx, temporalK.cy, match->R, match->t,
                [](float px, float py, float pz, float maxD,
                   float* ox, float* oy, float* oz) {
                    return g.nearestStableSurfel(px, py, pz, maxD, ox, oy, oz);
                },
                icpOutR, icpOutT);
            switch (lastIcpDiag.reason) {
                case frame_icp::kIcpRejectSamples: ++icpRejectSamples; break;
                case frame_icp::kIcpRejectOverlap: ++icpRejectOverlap; break;
                case frame_icp::kIcpRejectBound: ++icpRejectBound; break;
                case frame_icp::kIcpRejectDegenerate: ++icpRejectDegenerate; break;
                default: break;
            }
            if (lastIcpDiag.applied) {
                effR = icpOutR;
                effT = icpOutT;
                ++icpAppliedCount;
            }
        }
        // V0.13.22：参考必须「存在 + 时间戳前进 + 未超时效」才参与比对。
        // 参考过旧时本帧按中性放行（temporalMaskValid 保持 false，逐像素退化为边缘因子），
        // 这是从「融合失败 -> 参考冻结 -> 一致性崩塌 -> 再拒」死锁里脱身的关键。
        const uint64_t nowTs = static_cast<uint64_t>(t);
        const bool refNewer = nowTs > lastFuseTs;
        const uint64_t refAgeNs = refNewer ? (nowTs - lastFuseTs) : 0;
        const bool refFresh = haveLastFusePose && refNewer &&
                              refAgeNs <= kTemporalRefMaxAgeNs;
        if (haveLastFusePose) {
            lastTemporalRefAgeMs = refNewer ? static_cast<float>(refAgeNs) / 1e6f : -1.f;
            if (refNewer && !refFresh) {
                ++temporalRefStale;
                // V0.13.39 过期参考加固：量出当前帧相对旧参考的相机运动。
                // 位移小 -> 帧内容大概率仍与旧几何对得上，降权放行即可；
                // 位移/转角超限 -> 位姿或标定在断档期间系统性漂移，这种帧
                // 全权重落盘就是「同表面两份」的来源，先拒；连续拒到上限后
                // 放一帧弱锚定（conf×0.35）推动参考链前进，死锁不会复发。
                float relT[3] = {effT[0] - lastFuseT[0],
                                 effT[1] - lastFuseT[1],
                                 effT[2] - lastFuseT[2]};
                const float transNorm = std::sqrt(relT[0] * relT[0] +
                                                  relT[1] * relT[1] +
                                                  relT[2] * relT[2]);
                float relR[9];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        float acc = 0.f;
                        for (int k = 0; k < 3; ++k) {
                            acc += effR[k * 3 + i] * lastFuseR[k * 3 + j];
                        }
                        relR[i * 3 + j] = acc;
                    }
                }
                const float cosTheta =
                    std::clamp((relR[0] + relR[4] + relR[8] - 1.f) * 0.5f, -1.f, 1.f);
                const float rotAngle = std::acos(cosTheta);
                const bool motionLarge = transNorm > kTemporalStaleMaxTrans ||
                                         rotAngle > kTemporalStaleMaxRot;
                if (motionLarge &&
                    temporalStaleConsecRejects < kTemporalStaleMaxConsecRejects) {
                    staleMotionBlocked = true;
                    ++temporalStaleMotionRejects;
                    ++temporalStaleConsecRejects;
                } else {
                    // 弱锚定帧：降权写入并成为新参考，参考链由此恢复新鲜。
                    conf = std::min(conf, confidence * (motionLarge
                        ? 0.35f : kTemporalStaleConfFactor));
                    ++temporalStaleWrites;
                    temporalStaleConsecRejects = 0;
                }
            }
        }
        if (refFresh && lastFuseCalibrated == metricDepth &&
            lastFusedDepth.size() == (size_t)w * h && lastFuseW == w && lastFuseH == h && sameK) {
            // Pc_cur = Rcur^T * Rprev * Pc_prev + Rcur^T * (tprev - tcur)
            // （Rcur/tcur = ICP 修正后的 effR/effT：相对位姿在漂移被抵消后
            // 重新真实，重投影一致性判据才能按设计容差工作。）
            float Rrel[9];
            float trel[3];
            const float dv[3] = {lastFuseT[0] - effT[0],
                                 lastFuseT[1] - effT[1],
                                 lastFuseT[2] - effT[2]};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float acc = 0.f;
                    for (int k = 0; k < 3; ++k) {
                        acc += effR[k * 3 + i] * lastFuseR[k * 3 + j];
                    }
                    Rrel[i * 3 + j] = acc;
                }
                float acc = 0.f;
                for (int k = 0; k < 3; ++k) {
                    acc += effR[k * 3 + i] * dv[k];
                }
                trel[i] = acc;
            }
            const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
            const auto tTemporal = std::chrono::steady_clock::now();
            gTemporalMask.resize(static_cast<size_t>(w) * h);
            int mTested = 0, mAgree = 0;
            // V0.13.26：两端必须与实际进 TSDF 的深度**同域**。
            //
            // lastFusedDepth 存的是 epochFusionDepth（冻结标定）经扫描范围过滤后的副本，
            // 而这里原先拿 depthForFusion（live 标定）当当前帧 —— 一旦 live 与冻结标定
            // 分道扬镳，比较就变成「12% 尺度差 vs 1.5cm/2.5% 容差」，逐像素全部判冲突：
            //   13:28:25 PerfStage tmask tested=14450 agree=0 disagree=14450
            // （当时 frozen shift=1.3452 / live shift=1.5140，同一 raw 处 z 差 11.7%）。
            // 掩码被全量打成 disagree -> buildPixelWeight 把权重归零 -> 该帧白跑。
            // 换成 epoch 域后两端同源，残差只剩真实运动造成的重投影差。
            const float* curForTemporal =
                (metricDepth && !zEpoch.empty()) ? zEpoch.data() : depthForFusion;
            temporalConsistencyMask(lastFusedDepth.data(), curForTemporal, w, h,
                                    dk.fx, dk.fy, dk.cx, dk.cy, Rrel, trel,
                                    gTemporalMask.data(), /*step=*/1,
                                    0.015f, 0.025f, &mTested, &mAgree, &gProjectedDepth);
            temporalMaskValid = true;
            lastTemporalRatio = mTested < 16 ? 1.f : float(mAgree)/mTested;
            ++temporalChecks;
            if (lastTemporalRatio < 0.60f && mTested >= std::max(32,w*h/64)) {
                ++temporalRejects;
                temporalAccepted = false;
            }
            lastTemporalTested = mTested;
            lastTemporalAgree = mAgree;
            lastTemporalDisagree = mTested - mAgree;
            lastDiagTemporalRatio = lastTemporalRatio;
            lastDiagTemporalTested = mTested;
            lastDiagTemporalAgree = mAgree;
            gPerf[kPerfTemporal].push(perfMsSince(tTemporal));

            // V0.13.22 取证：一致率极低时把「重投影深度 A vs 当前深度 B」的实测差
            // 打出来。系统性大偏移 => 尺度/位姿问题；随机小噪声 => 深度网络抖动。
            // 这条日志直接决定「模型发平、对不上位置」往哪个方向修。
            if (mTested >= 512 && mAgree * 10 < mTested) {
                int n = 0;
                double sumAbs = 0.0, sumSigned = 0.0;
                float sA = 0.f, sB = 0.f;
                for (size_t i = 0; i < gProjectedDepth.size(); i += 1024) {
                    const float a = gProjectedDepth[i];
                    const float b = curForTemporal[i];
                    if (!(a > 0.05f) || !(b > 0.05f) || !std::isfinite(b)) continue;
                    if (n == 0) { sA = a; sB = b; }
                    sumAbs += std::fabs(a - b);
                    sumSigned += (a - b);
                    ++n;
                }
                if (n > 0) {
                    LOGI("TemporalProbe tested=%d agree=%d ratio=%.3f n=%d "
                         "A=%.4f B=%.4f meanAbsDiff=%.4f meanSignedDiff=%.4f refAgeMs=%.0f",
                         mTested, mAgree, static_cast<double>(lastTemporalRatio), n,
                         static_cast<double>(sA), static_cast<double>(sB),
                         sumAbs / n, sumSigned / n,
                         static_cast<double>(lastTemporalRefAgeMs));
                }
            }

            // Strong disagreement in the overlapping area rejects dense writes.
            // Sparse/no overlap is neutral (temporalConsistencyRatio returns 1).
            //
            // 有逐像素掩码后，整帧因子的下限从 0.25 提到 0.5：局部该降多少由
            // 掩码精确决定，整帧再乘一个 0.25 会把「掩码已经判一致的像素」也
            // 一起拖下去 —— 那是双重惩罚，会让表面整体偏薄。整帧因子现在只当
            // 粗粒度安全网（整帧明显不对时整体收敛慢一点），细粒度交给掩码。
            conf = confidence * std::clamp(lastTemporalRatio, 0.5f, 1.f);
        }

        // V0.12 融合门控：只有「标定已连续稳定、尺度已冻结」的 epoch 才允许写
        // TSDF 与累计 surfel。标定没到位时深度还是 raw 量级（实机 ~4.4），
        // 而 TSDF 的体素尺寸是按 VINS world 尺度选的 —— 灌进去只会得到一堆
        // 位置错误的体素。宁可这一帧不建，也不要把垃圾建进去。
        //
        // 注意：这条门控**只管 dense depth 融合**。V0.9 的稀疏 stereo anchor
        // 自带米制尺度（走 stereo_anchor_bridge 的 world scale），不受影响。
        // 退路：`nativeSetDepthCalibrationEnabled(false)` 时退回旧行为 ——
        // 直接融合 `depthForFusion`（未标定的 raw 尺度）。否则「关掉标定」
        // 会静默变成「永远不融合」，那是比标定不准更糟的失败形态。
        // V0.13.22：不再「时序不一致就整帧丢弃」。
        //
        // 逐像素掩码（gTemporalMask -> buildPixelWeight，kTemporalWeightDisagree=0）
        // 已经能精确地把冲突像素权重归零，整帧否决是**冗余**的，而它带来的代价是
        // 死锁：真机实测 agree=0 -> temporalAccepted=false -> fusionDepth=nullptr
        // -> fused 停在 14/120、网格冻结在 verts=4257 达 2min21s，用户移动摄像头
        // 完全长不出新几何、画面也看不到点云。
        //
        // 移除后的三层行为才是移动扫描该有的：
        //   重叠且一致 -> 权重 1.0   （写入，正常融合）
        //   重叠且冲突 -> 权重 0.0   （不写，防重复壳依然有效）
        //   无重叠新区域 -> 权重 0.85（写入，移动时照常长出新几何）
        // 深度整体崩坏时逐像素掩码会把全部权重打到 0，安全网仍然在。
        const float* fusionDepth = calibEnabledEff
            ? epochFusionDepth
            : (representation == 0 ? depthForFusion : nullptr);
        // 整帧一致性只做「额外降权」，不再做「否决」。
        if (!temporalAccepted && fusionDepth != nullptr) {
            conf = std::min(conf, confidence * 0.35f);
        }
        // V0.13.39：过期参考 + 相对旧参考大位移 -> 本帧不进 dense 融合。
        // 只挡 dense（TSDF/surfel/目标）；稀疏 stereo anchor 自带米制尺度
        // 与残差门，不受影响。此否决有界（连续 ≤4 帧必放弱锚定帧），不会
        // 重演 vc162 的「参考冻结 → 永拒」死锁。
        if (staleMotionBlocked) {
            fusionDepth = nullptr;
        }

        // ---- 逐像素融合权重 ----
        // 两个消费者（TSDF 融合、可视化点云）用同一张图，保证「点云看到的」与
        // 「体素写进去的」可信度口径一致。首帧没有上一帧可比对时退化为纯边缘因子。
        const float* pixelWeight = nullptr;
        if (fusionDepth != nullptr || (epochActive && calibEnabledEff && !zEpoch.empty())) {
            const float* weightDepth = fusionDepth ? fusionDepth : zEpoch.data();
            const auto tWeight = std::chrono::steady_clock::now();
            buildPixelWeight(weightDepth, w, h,
                             temporalMaskValid ? gTemporalMask.data() : nullptr,
                             gPixelWeight, sourceWeights.empty() ? nullptr : sourceWeights.data());
            gPerf[kPerfWeight].push(perfMsSince(tWeight));
            pixelWeight = gPixelWeight.data();
        }

        // 过滤：一帧只做一次。原来 fuseDepth() 和 feedSceneSurfels() 各自复制一份
        // 再 filterScanRange 一遍，同一帧被复制两次、过滤两次。
        const float* fusionRanged = nullptr;
        if (fusionDepth != nullptr) {
            const auto dk = depthIntrinsics(gFx, gFy, gCx, gCy, gV07InputW, gV07InputH, w, h);
            gRangedFusion.assign(fusionDepth, fusionDepth + static_cast<size_t>(w) * h);
            filterScanRange(gRangedFusion, w, h, dk.fx, dk.fy, dk.cx, dk.cy);
            fusionRanged = gRangedFusion.data();
        }

        // Reference checks do not expire after a pause. Only accepted, actually
        // integrated samples are retained; reject a second shell on return visits.
        const float guardK[4]={temporalK.fx,temporalK.fy,temporalK.cx,temporalK.cy};
        bool modelAccepted=true;
        if(fusionRanged)modelAccepted=fusionGuard.accept(fusionRanged,w,h,guardK,effR,effT,
                                                        static_cast<uint64_t>(t));
        if(!modelAccepted) {
            fusionDepth=nullptr;fusionRanged=nullptr;
        }
        colorContours.reset();
        if(fusionRanged)colorContours.build(match->rgb.data(),match->w,match->h,
                                            fusionRanged,w,h,pixelWeight);
        const uint8_t* contourPriority=colorContours.priorityPixels ? colorContours.band.data() : nullptr;

        if (fusionRanged != nullptr) {
            const auto tFuse = std::chrono::steady_clock::now();
            // 全场景地图
            fuseDepth(fusionRanged, w, h, *match, conf, pixelWeight, contourPriority, effR, effT);

            // 目标专用模型：**只有 presenceOk、mask 有效、且 mask 时序稳定
            // 才允许融合**。这是「墙/天花板/桌子进不了目标点云」的落地点；
            // V0.13 又加了一道 —— mask 单帧整块跳变的那一帧也不进。
            if (haveTi && presenceOk && maskOk && maskTemporalOk) {
                fuseTargetDepth(fusionRanged, w, h, *match, targetMaskMat, conf,
                                pixelWeight, contourPriority, effR, effT);
            }
            gPerf[kPerfFuse].push(perfMsSince(tFuse));
            fusionFusedFrames++;
        } else {
            fusionGatedFrames++;
            // V0.13：把「因为 epoch 被暂停而不融合」和「因为还没到稳定期」
            // 分开计数 —— 前者是主动选择，后者是等待，排查时含义完全不同。
            if (epochActive && epochSuspended) {
                fusionSuspendedFrames++;
            }
        }

        // Accumulated preview obeys exactly the same acceptance as the mesh.
        // The camera-space target debug layer above remains the live-only view.
        if(fusionRanged) {
            const auto tSurfel=std::chrono::steady_clock::now();
            feedSceneSurfels(fusionRanged,w,h,*match,conf,
                            (haveTi && maskOk),targetMaskStats.centerX,targetMaskStats.centerY,
                            pixelWeight,contourPriority,effR,effT);
            gPerf[kPerfSurfel].push(perfMsSince(tSurfel));
        }

        // ---- V0.9 sparse stereo high-confidence TSDF constraints ----
        const auto tStereo = std::chrono::steady_clock::now();
        auto stereoWorldBatches =
            mobilescan3d::stereo_anchor::takeWorldAnchorBatches(
                static_cast<uint64_t>(t),
                kStereoAnchorMaxAgeNs,
                2);

        bool stereoGeometryChanged=false;
        for (const auto& batch : stereoWorldBatches) {
            // V0.13.22：同步解除 temporalAccepted 对稀疏 stereo anchor 的整帧否决
            // （理由见 dense 融合处）。保守壳护罩 modelAccepted 仍然保留。
            if(!modelAccepted)continue;
            int64_t stereoPoseDiffNs = -1;
            const FrameSnap* stereoSnap =
                findStereoAnchorSnap(
                    batch.timestampNs,
                    &stereoPoseDiffNs);

            if (!stereoSnap) {
                mobilescan3d::stereo_anchor::noteGeometryDropNoPose(
                    static_cast<int>(batch.anchors.size()));
                continue;
            }

            int scenePasses = 0;
            const int sceneAccepted =
                integrateStereoAnchorBatch(
                    tsdf,
                    batch,
                    *stereoSnap,
                    nullptr,
                    &scenePasses);

            stereoGeometryChanged=stereoGeometryChanged || sceneAccepted>0;
            mobilescan3d::stereo_anchor::noteSceneTsdf(
                static_cast<int>(batch.anchors.size()),
                sceneAccepted,
                scenePasses,
                true);

            int64_t targetDt =
                static_cast<int64_t>(batch.timestampNs) -
                static_cast<int64_t>(t);
            if (targetDt < 0) targetDt = -targetDt;

            const bool targetMaskUsable =
                haveTi &&
                presenceOk &&
                maskOk && maskTemporalOk &&
                targetDt <= kStereoTargetMaskMaxDiffNs;

            if (targetMaskUsable) {
                int targetPasses = 0;
                const int targetAccepted =
                    integrateStereoAnchorBatch(
                        targetTsdf,
                        batch,
                        *stereoSnap,
                        &targetMaskMat,
                        &targetPasses);

                mobilescan3d::stereo_anchor::noteTargetTsdf(
                    static_cast<int>(batch.anchors.size()),
                    targetAccepted,
                    targetPasses,
                    true);
            } else {
                mobilescan3d::stereo_anchor::noteTargetTsdf(
                    static_cast<int>(batch.anchors.size()),
                    0,
                    0,
                    false);
            }
        }
        gPerf[kPerfStereo].push(perfMsSince(tStereo));

        // Retain an accepted reference, never an uncalibrated/rejected observation.
        if (fusionDepth != nullptr) {
            lastFusedDepth.assign(fusionRanged, fusionRanged + (size_t)w * h);
            for(size_t i=0;i<lastFusedDepth.size();++i)
                if(pixelWeight && pixelWeight[i]<=0)lastFusedDepth[i]=0;
            fusionGuard.commit(fusionRanged,w,h,guardK,effR,effT,
                               static_cast<uint64_t>(t),pixelWeight);
            for (int i = 0; i < 9; ++i) {
                lastFuseR[i] = effR[i];
            }
            for (int i = 0; i < 3; ++i) {
                lastFuseT[i] = effT[i];
            }
            haveLastFusePose = true;
            lastFuseCalibrated = metricDepth;
            lastFuseW = w; lastFuseH = h; lastFuseTs = static_cast<uint64_t>(t);
            lastFuseK[0] = temporalK.fx; lastFuseK[1] = temporalK.fy;
            lastFuseK[2] = temporalK.cx; lastFuseK[3] = temporalK.cy;
        }

        // 体素场变了 -> 网格变脏
        if(fusionRanged || stereoGeometryChanged){meshDirty = true; ++meshDirtyMarks;}
    }

    if (!targetDebugEnabled) {
        // 关掉时把上一帧的残留清空，否则切换开关后屏幕上会留着旧点
        targetDebugPointCount = 0;
    }

    ++gPerfCompleted;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnDepthMap(
        JNIEnv* e, jobject, jfloatArray depth, jint w, jint h,
        jfloat confidence, jlong t, jint representation) {
    // JNI 边界是所有 native 异常的最后一道防线。
    //
    // C++ 异常一旦越过 JNI 回到 ART，Kotlin 侧任何 try/catch 都抓不到
    // （JNI 没有异常传播机制），只会走到 std::terminate 直接杀进程 ——
    // 表现就是「点一下目标立刻闪退」，而且 logcat 里往往只有一句
    // "terminating with uncaught exception of type cv::Exception"。
    //
    // 已知实例：target_mask_engine.cpp 里 centroids_.at<cv::Vec2d>()
    // 在 Debug 构建下命中 elemSize()==sizeof(_Tp) 断言抛 cv::Exception，
    // 就是从这条深度线程路径逃出去的。修复了根因之后，这里仍然保留兜底：
    // 30Hz 的深度回调里任何一个 OpenCV/STL 异常都不该带走整个 APP。
    try {
        nativeOnDepthMapImpl(e, depth, w, h, confidence, t, representation);
    } catch (const cv::Exception& ex) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Depth",
                            "nativeOnDepthMap OpenCV exception: %s", ex.what());
    } catch (const std::exception& ex) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Depth",
                            "nativeOnDepthMap exception: %s", ex.what());
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Depth",
                            "nativeOnDepthMap unknown exception");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeOnDepthMapWeighted(
        JNIEnv* e, jobject, jfloatArray depth, jint w, jint h,
        jfloat confidence, jlong t, jint representation, jfloatArray sourceConfidence) {
    try {
        nativeOnDepthMapImpl(e, depth, w, h, confidence, t, representation, sourceConfidence);
    } catch (const std::exception& ex) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Depth", "weighted depth: %s", ex.what());
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "MobileScan3D-Depth", "weighted depth: unknown exception");
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeExportPly(JNIEnv* e, jobject, jstring path) {
    const char* p = e->GetStringUTFChars(path, nullptr);
    std::lock_guard<std::mutex> lk(gStateMutex);
    // 有目标模型时**只导目标**（验收要求：PLY 只含目标，不再有墙、天花板、桌子）。
    // 还没锁定过目标时退回全场景 TSDF，否则导出会是一个空文件。
    bool ok = targetTsdf.voxels() > 0 ? targetTsdf.exportPly(p) : tsdf.exportPly(p);
    e->ReleaseStringUTFChars(path, p);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetPersistentMapCaptureEnabled(
        JNIEnv*,
        jobject,
        jboolean enabled) {
    gPersistentRelocalizer.setCaptureEnabled(
        enabled == JNI_TRUE);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSavePersistentMap(
        JNIEnv* env,
        jobject,
        jstring path) {
    if (!path) return JNI_FALSE;

    const char* p =
        env->GetStringUTFChars(
            path,
            nullptr);

    if (!p) return JNI_FALSE;

    const bool ok =
        gPersistentRelocalizer.saveMap(p);

    env->ReleaseStringUTFChars(
        path,
        p);

    return ok
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeLoadPersistentMap(
        JNIEnv* env,
        jobject,
        jstring path) {
    if (!path) return JNI_FALSE;

    const char* p =
        env->GetStringUTFChars(
            path,
            nullptr);

    if (!p) return JNI_FALSE;

    const bool ok =
        gPersistentRelocalizer.loadMap(p);

    env->ReleaseStringUTFChars(
        path,
        p);

    if (ok) {
        // Old history belongs to a different live VINS world. Until the new
        // session is visually aligned to the saved world, no historical pose
        // is allowed to leak to the renderer.
        {
            std::lock_guard<std::mutex> lk(
                gStateMutex);
            renderPoseHistory.clear();
        }
        {
            std::lock_guard<std::mutex> rlk(
                gRelocFrameMutex);
            gRelocGray.clear();
            gRelocFrameTimestampNs = 0u;
            gRelocConsumedTimestampNs = 0u;
        }
    }

    return ok
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeClearPersistentRelocalization(
        JNIEnv*,
        jobject) {
    gPersistentRelocalizer.resetAll();

    {
        std::lock_guard<std::mutex> lk(
            gStateMutex);
        renderPoseHistory.clear();
    }
    {
        std::lock_guard<std::mutex> rlk(
            gRelocFrameMutex);
        gRelocGray.clear();
        gRelocFrameTimestampNs = 0u;
        gRelocConsumedTimestampNs = 0u;
    }
}


extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeTryPersistentRelocalization(
        JNIEnv*,
        jobject) {
    std::vector<std::uint8_t> gray;
    std::uint64_t timestampNs = 0;
    float Rwc[9];
    float twc[3];

    {
        std::lock_guard<std::mutex> lk(
            gRelocFrameMutex);

        if (gRelocFrameTimestampNs == 0u ||
            gRelocFrameTimestampNs ==
                gRelocConsumedTimestampNs ||
            gRelocGray.empty()) {
            return gPersistentRelocalizer.localized()
                ? JNI_TRUE
                : JNI_FALSE;
        }

        gray = gRelocGray;
        timestampNs =
            gRelocFrameTimestampNs;

        std::copy(
            gRelocFrameRwc,
            gRelocFrameRwc + 9,
            Rwc);

        std::copy(
            gRelocFrameTwc,
            gRelocFrameTwc + 3,
            twc);

        gRelocConsumedTimestampNs =
            timestampNs;
    }

    const float sx =
        static_cast<float>(gVinsW) /
        static_cast<float>(
            std::max(1, gV07InputW));

    const float sy =
        static_cast<float>(gVinsH) /
        static_cast<float>(
            std::max(1, gV07InputH));

    const bool ok =
        gPersistentRelocalizer.tryRelocalize(
            gray.data(),
            gVinsW,
            gVinsH,
            gVinsW,
            timestampNs,
            gFx * sx,
            gFy * sy,
            gCx * sx,
            gCy * sy,
            Rwc,
            twc);

    return ok
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRelocalizationStats(
        JNIEnv* env,
        jobject,
        jfloatArray out) {
    constexpr int kSlots = 13;

    if (!out ||
        env->GetArrayLength(out) < kSlots) {
        return JNI_FALSE;
    }

    const auto s =
        gPersistentRelocalizer.stats();

    jfloat values[kSlots] = {
        static_cast<jfloat>(s.state),
        static_cast<jfloat>(s.mapPoints),
        static_cast<jfloat>(s.capturedKeyframes),
        static_cast<jfloat>(s.capturedPoints),
        static_cast<jfloat>(s.lastDetected),
        static_cast<jfloat>(s.lastMatches),
        static_cast<jfloat>(s.lastInliers),
        static_cast<jfloat>(s.lastInlierRatio),
        static_cast<jfloat>(s.lastMedianReprojectionPx),
        static_cast<jfloat>(s.attempts),
        static_cast<jfloat>(s.successes),
        s.mapLoaded ? 1.0f : 0.0f,
        s.localized ? 1.0f : 0.0f
    };

    env->SetFloatArrayRegion(
        out,
        0,
        kSlots,
        values);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTexturedArAssetStats(
        JNIEnv* env,
        jobject) {
    jint values[3] = {0, 0, 0};

    {
        std::lock_guard<std::mutex> lk(
            gArAssetMutex);

        values[0] =
            static_cast<jint>(
                std::min<std::size_t>(
                    gArTexturedAsset.vertexCount(),
                    static_cast<std::size_t>(
                        std::numeric_limits<jint>::max())));

        values[1] =
            static_cast<jint>(
                std::min<std::size_t>(
                    gArTexturedAsset.indexCount(),
                    static_cast<std::size_t>(
                        std::numeric_limits<jint>::max())));

        values[2] =
            static_cast<jint>(
                std::min<std::size_t>(
                    gArTexturedAsset.jpeg().size(),
                    static_cast<std::size_t>(
                        std::numeric_limits<jint>::max())));
    }

    jintArray out =
        env->NewIntArray(3);

    if (!out) return nullptr;

    env->SetIntArrayRegion(
        out,
        0,
        3,
        values);

    return out;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTexturedArVertices(
        JNIEnv* env,
        jobject,
        jfloatArray out) {
    if (!out) return 0;

    std::lock_guard<std::mutex> lk(
        gArAssetMutex);

    if (!gArTexturedAsset.ready()) {
        return 0;
    }

    const auto& src =
        gArTexturedAsset.vertices();

    const jsize cap =
        env->GetArrayLength(out);

    if (cap <
        static_cast<jsize>(
            src.size())) {
        return 0;
    }

    env->SetFloatArrayRegion(
        out,
        0,
        static_cast<jsize>(
            src.size()),
        src.data());

    return static_cast<jint>(
        gArTexturedAsset.vertexCount());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTexturedArIndices(
        JNIEnv* env,
        jobject,
        jintArray out) {
    if (!out) return 0;

    std::lock_guard<std::mutex> lk(
        gArAssetMutex);

    if (!gArTexturedAsset.ready()) {
        return 0;
    }

    const auto& src =
        gArTexturedAsset.indices();

    const jsize cap =
        env->GetArrayLength(out);

    if (cap <
        static_cast<jsize>(
            src.size())) {
        return 0;
    }

    std::vector<jint> tmp(
        src.size());

    for (std::size_t i = 0;
         i < src.size();
         ++i) {
        tmp[i] =
            static_cast<jint>(
                src[i]);
    }

    env->SetIntArrayRegion(
        out,
        0,
        static_cast<jsize>(
            tmp.size()),
        tmp.data());

    return static_cast<jint>(
        src.size());
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTexturedArAtlasJpeg(
        JNIEnv* env,
        jobject) {
    std::lock_guard<std::mutex> lk(
        gArAssetMutex);

    if (!gArTexturedAsset.ready()) {
        return nullptr;
    }

    const auto& src =
        gArTexturedAsset.jpeg();

    jbyteArray out =
        env->NewByteArray(
            static_cast<jsize>(
                src.size()));

    if (!out) return nullptr;

    env->SetByteArrayRegion(
        out,
        0,
        static_cast<jsize>(
            src.size()),
        reinterpret_cast<const jbyte*>(
            src.data()));

    return out;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSaveTexturedArAsset(
        JNIEnv* env,
        jobject,
        jstring path) {
    if (!path) return JNI_FALSE;

    const char* p =
        env->GetStringUTFChars(
            path,
            nullptr);

    if (!p) return JNI_FALSE;

    bool ok = false;

    {
        std::lock_guard<std::mutex> lk(
            gArAssetMutex);

        ok =
            gArTexturedAsset.save(p);
    }

    env->ReleaseStringUTFChars(
        path,
        p);

    return ok
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeLoadTexturedArAsset(
        JNIEnv* env,
        jobject,
        jstring path) {
    if (!path) return JNI_FALSE;

    const char* p =
        env->GetStringUTFChars(
            path,
            nullptr);

    if (!p) return JNI_FALSE;

    bool ok = false;

    {
        std::lock_guard<std::mutex> lk(
            gArAssetMutex);

        ok =
            gArTexturedAsset.load(p);
    }

    env->ReleaseStringUTFChars(
        path,
        p);

    return ok
        ? JNI_TRUE
        : JNI_FALSE;
}
extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeClearTexturedArAsset(
        JNIEnv*,
        jobject) {
    std::lock_guard<std::mutex> lk(
        gArAssetMutex);
    gArTexturedAsset.clear();
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetMode(JNIEnv*, jobject, jint m) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    mode = m;
}

// V0.13：把 ObjectTracker 的身份诊断镜像到 native 报告里。
struct TargetIdentityMirror {
    bool anchorReady = false;
    float score = -1.f;
    uint64_t rejects = 0;
    float bboxScale = 1.f;
};
static TargetIdentityMirror gTargetIdentityMirror;

/**
 * 刷新身份诊断镜像。
 *
 * ★ 必须在**已持有 gStateMutex** 的上下文中调用 —— std::mutex 不可重入，
 * 这里再去 lock 一次 gStateMutex 就是自锁死（本项目曾经因为这一点把 APP
 * 整卡住过）。所以本函数只读全局 objectTracker 指针本身，不再取锁；
 * `ObjectTracker::info()` 内部有自己的 mutex_，与外层不冲突
 * （nativeOnDepthMap 里在持 gStateMutex 的情况下也是这么读的）。
 */
static void refreshTargetIdentityMirrorLocked() {
    if (!objectTracker) {
        return;
    }
    const TargetTrackInfo i = objectTracker->info();
    gTargetIdentityMirror.anchorReady = i.identityAnchorReady;
    gTargetIdentityMirror.score = i.identityScore;
    gTargetIdentityMirror.rejects = i.identityRejects;
    gTargetIdentityMirror.bboxScale = i.bboxScaleFromInitial;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetStats(JNIEnv* e, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    float R[9], T[3];
    makePose(R, T);

    float pminX, pminY, pminZ, pmaxX, pmaxY, pmaxZ;
    float pcx, pcy, pcz;
    g.boundingBox(&pminX, &pminY, &pminZ, &pmaxX, &pmaxY, &pmaxZ);
    g.centroid(&pcx, &pcy, &pcz);

    const float offX = pcx - T[0];
    const float offY = pcy - T[1];
    const float offZ = pcz - T[2];
    const float offDist = std::sqrt(offX * offX + offY * offY + offZ * offZ);

    float dmin = 0.f, dmax = 0.f, dmean = 0.f;
    int dvalid = 0;
    if (!df.depth().empty()) {
        dmin = std::numeric_limits<float>::max();
        dmax = -std::numeric_limits<float>::max();
        double sum = 0.0;
        int cnt = 0;
        for (float z : df.depth()) {
            if (std::isfinite(z) && z > 0.f) {
                dmin = std::min(dmin, z);
                dmax = std::max(dmax, z);
                sum += z;
                cnt++;
            }
        }
        dvalid = cnt;
        dmean = cnt > 0 ? (float)(sum / cnt) : 0.f;
    }

    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(3);
    s << "V0.5.1 HQ Reconstruction\n"
      << "Tracking frames: " << vio.frames() << "  features: " << vio.features() << "\n"
      << "VINS: " << (vinsInitialized() ? "nonlinear initialized" : "initializing")
      << "  poseSource: " << (vinsPoseOk ? "camera_at_exposure" : "unavailable") << "\n"
      << "Fusion guard: refs=" << fusionGuard.references() << " checks=" << fusionGuard.checks
      << " rejected=" << fusionGuard.rejected << " tested=" << fusionGuard.lastTested
      << " ratio=" << fusionGuard.lastRatio << " duplicateDepth=" << duplicateDepthRejects << "\n"
      << "Color contours: linePixels=" << colorContours.contourPixels
      << " priorityPixels=" << colorContours.priorityPixels << "\n"
      << "World continuity: " << (vinsWorldDiscontinuous() ? "RESET: export then restart scan" : "ok") << "\n"
      << "VINS raw q: (" << vinsQ[0] << ", " << vinsQ[1] << ", " << vinsQ[2] << ", " << vinsQ[3] << ")\n"
      << "VINS raw t: (" << vinsT[0] << ", " << vinsT[1] << ", " << vinsT[2] << ")\n"
      << "VINS estimator raw t: (" << rawVinsT[0] << ", " << rawVinsT[1] << ", " << rawVinsT[2] << ")\n"
      << "VINS accepted t: (" << acceptedVinsT[0] << ", " << acceptedVinsT[1] << ", " << acceptedVinsT[2] << ")\n"
      << "VINS last step: " << lastVinsStep << "\n"
      << "VINS reject count: " << vinsRejectCount << "\n"
      << "VINS reject reason: " << vinsRejectReason << "\n"
      << "ScanRange: maxMeters=" << scanMaxDistanceMeters
      << " worldPerMeter=" << scanWorldPerMeter
      << " rejectedPixels=" << rangeRejectedPixels
      << " frozenEvidenceRecoveries=" << frozenEvidenceRecoveries << "\n"
      << "VINS lost latch: " << (vinsLostAfterInit ? "true" : "false") << "\n"
      << "VINS recoveries: " << vinsRecoveryCount
      << " rejectStreak: " << vinsRejectStreak << "\n"
      << "DepthPoseSnap: matches=" << depthSnapMatches
      << " misses=" << depthSnapMisses
      << " lastDiffMs="
      << (lastDepthSnapDiffNs >= 0
              ? (static_cast<double>(lastDepthSnapDiffNs) / 1e6)
              : -1.0)
      << " maxDiffMs=" << (static_cast<double>(kDepthSnapMaxDiffNs) / 1e6)
      << "\n"
      << "VINS invalid feature depth resets: " << featureManagerInvalidDepthResetCount() << "\n";

    VinsHealth health;
    if (vinsGetHealth(&health)) {
        s << "VINS health: velocity=" << health.velocity
          << " accBias=" << health.accBias
          << " gyroBias=" << health.gyroBias
          << " gravity=" << health.gravity
          << " trackedFeatures=" << health.trackedFeatures
          << " lastImuDt=" << health.lastImuDt
          << " estimatedTD=" << (health.timeOffset * 1000.0) << "ms\n";
        s << "VINS estimated RIC: ["
          << health.ric[0] << " " << health.ric[1] << " " << health.ric[2] << "; "
          << health.ric[3] << " " << health.ric[4] << " " << health.ric[5] << "; "
          << health.ric[6] << " " << health.ric[7] << " " << health.ric[8] << "]\n";
        s << "VINS estimated TIC: ("
          << health.tic[0] << ", " << health.tic[1] << ", " << health.tic[2] << ")\n";
    }

    if (haveFirstReject) {
        s << "First VINS reject: step=" << firstRejectStep
          << " t=(" << firstRejectT[0] << ", " << firstRejectT[1] << ", " << firstRejectT[2] << ")"
          << " velocity=" << firstRejectVelocity
          << " accBias=" << firstRejectAccBias
          << " gyroBias=" << firstRejectGyroBias
          << " td=" << (firstRejectTd * 1000.0) << "ms\n";
    }

    s
      << "Cam pose t: (" << T[0] << ", " << T[1] << ", " << T[2] << ")\n"
      << "Cam pose R: ["
      << R[0] << ", " << R[1] << ", " << R[2] << "; "
      << R[3] << ", " << R[4] << ", " << R[5] << "; "
      << R[6] << ", " << R[7] << ", " << R[8] << "]\n"
      << "VIO yaw/pitch/roll: " << vio.yaw() << " / " << vio.pitch() << " / " << vio.roll() << "\n"
      << "VIO t: (" << vio.tx() << ", " << vio.ty() << ", " << vio.tz() << ")"
      << "  novelty: " << vio.visualNovelty() << "\n"
      << "Depth: " << df.width() << "x" << df.height() << " ts=" << df.timestamp()
      << " samples=" << df.stats().samples << " valid=" << df.stats().valid
      << " refined=" << df.stats().refined << " conf=" << df.stats().confidence << "\n"
      << "Depth range: min=" << dmin << " max=" << dmax << " mean=" << dmean
      << " validPix=" << dvalid << "\n"
      << "AI quality: sharp=" << ai.q().sharpness << " exposure=" << ai.q().exposure
      << " motion=" << ai.q().motion << " geometry=" << ai.q().geometry
      << " completion=" << ai.q().completion << "\n"
      << "Keyframes: " << kf.size() << "  accepted: " << lastKF << "\n";
    if (kf.last()) {
        const Keyframe* k = kf.last();
        s << "  lastKF ts=" << k->ts << " score=" << k->score
          << " sharp=" << k->sharp << " exp=" << k->exposure
          << " novel=" << k->novelty << " feats=" << k->features
          << " t=(" << k->tx << ", " << k->ty << ", " << k->tz << ")\n";
    }
    refreshTargetIdentityMirrorLocked();
    s << "Gaussian: " << g.count() << " stable=" << g.stableCount()
      << " merged=" << g.mergedCount()
      << " confirmed2=" << g.confirmedCount(2)
      << " confirmed3=" << g.confirmedCount(3) << "\n"
      << "AR: minHits=" << gArMinHits << " drawn=" << gArDrawnPoints
      << " renderPoseHistory=" << renderPoseHistory.size() << "\n"
      << "TargetDepthDebug: enabled=" << (targetDebugEnabled ? 1 : 0)
      << " points=" << targetDebugPointCount
      << " roi=" << targetDebugRoiX0 << "," << targetDebugRoiY0 << "-"
      << targetDebugRoiX1 << "," << targetDebugRoiY1
      << " roiPixels=" << targetDebugRoiPixels
      << " validPixels=" << targetDebugValidPixels
      << " ts=" << targetDebugTs
      // live 层现在直接存相机坐标，不再经过 world pose，所以这里恒为 1；
      // 字段名同步改成 liveSpaceOk，免得以后被误读成「pose 有效」
      << " liveSpaceOk=" << (targetDebugPoseOk ? 1 : 0)
      << " space=camera"
      << " builds=" << targetDebugBuilds
      << " queries=" << targetDebugQueries << "\n"
      << "TargetMask: frames=" << targetMaskFrames
      << " valid=" << (targetMaskStats.valid ? 1 : 0)
      << " area=" << targetMaskStats.area
      << " candidatePixels=" << targetMaskStats.candidatePixels
      << " searchPixels=" << targetMaskStats.searchPixels
      << " seedDepth=" << targetMaskStats.seedDepth
      << " tolerance=" << targetMaskStats.tolerance
      << " center=(" << targetMaskStats.centerX << ", " << targetMaskStats.centerY << ")"
      // V0.13 Adaptive Mask：面积比是「mask 有没有吞掉整个搜索框」的直接读数，
      // 边界接触比例是「连通域是不是被框边截断（本来会延伸更远）」的读数。
      // 这两个值加起来才能判断「mask 忠实地圈住了背景」这件事。
      << " areaRatio=" << targetMaskStats.areaRatio
      << " borderTouch=" << targetMaskStats.borderTouch
      << " tolStep=" << targetMaskStats.toleranceStep
      << " overExpanded=" << (targetMaskStats.overExpanded ? 1 : 0)
      << " reason=" << targetMaskStats.rejectReason << "\n"
      << "MaskTemporal: iou=" << maskTemporalIou
      << " rejects=" << maskTemporalRejects
      << " ok=" << (maskTemporalOk ? 1 : 0) << "\n"
      << "TargetIdentity: anchor=" << (gTargetIdentityMirror.anchorReady ? 1 : 0)
      << " score=" << gTargetIdentityMirror.score
      << " rejects=" << gTargetIdentityMirror.rejects
      << " bboxScale=" << gTargetIdentityMirror.bboxScale << "\n"
      << "TargetModel: gaussians=" << targetG.count()
      << " confirmed2=" << targetG.confirmedCount(2)
      << " stable=" << targetG.confirmedCount(3)
      << " tsdfVoxels=" << targetTsdf.voxels()
      << " fuseFrames=" << targetFuseFrames
      << " liveSource=" << (targetLockEverArmed ? "target-only" : "auto") << "\n"
      << "TargetPresence: failStreak=" << presenceFailStreak
      << " lostFrames=" << presenceLostFrames << "\n"
      << "TargetDepthScale: samples=" << targetDepthScaleSamples
      << " ema=" << targetDepthScaleEma
      << " applied=" << currentTargetDepthScale()
      << " vinsRoiMedian=" << lastTargetVinsMedian
      << " rawMedian=" << lastTargetRawMedian
      << " vinsFeatures=" << lastTargetVinsFeatureCount << "\n"
      << "PointCloud bbox min=(" << pminX << ", " << pminY << ", " << pminZ << ")"
      << " max=(" << pmaxX << ", " << pmaxY << ", " << pmaxZ << ")\n"
      << "PointCloud centroid=(" << pcx << ", " << pcy << ", " << pcz << ")\n"
      << "PointCloud vs Cam offset=(" << offX << ", " << offY << ", " << offZ
      << ")  dist=" << offDist << "\n"
      << "AdaptiveScene candidates=" << tsdf.adaptiveStats().candidates
      << " selected=" << tsdf.adaptiveStats().selected << " skipped=" << tsdf.adaptiveStats().skipped
      << " protected=" << tsdf.adaptiveStats().protectedSamples << " reactivated=" << tsdf.adaptiveStats().reactivated
      << " cells=" << tsdf.adaptiveCells() << "\n"
      << "AdaptiveTarget candidates=" << targetTsdf.adaptiveStats().candidates
      << " selected=" << targetTsdf.adaptiveStats().selected << " skipped=" << targetTsdf.adaptiveStats().skipped
      << " protected=" << targetTsdf.adaptiveStats().protectedSamples << " reactivated=" << targetTsdf.adaptiveStats().reactivated
      << " cells=" << targetTsdf.adaptiveCells() << "\n"
      << "PreviewCompact scenePatches=" << g.compressedCount() << " reclaimed=" << g.reclaimedCount()
      << " reactivated=" << g.reactivatedCount() << " storageEstimate=" << g.storageBytes()
      << " targetPatches=" << targetG.compressedCount() << " targetReclaimed=" << targetG.reclaimedCount()
      << " targetReactivated=" << targetG.reactivatedCount() << " targetStorageEstimate=" << targetG.storageBytes() << "\n"
      << "TSDF: voxels=" << tsdf.voxels()
      << " blocks=" << tsdf.blocks()
      << " voxelSize=" << tsdf.voxelSize()
      << " memMB=" << (tsdf.memoryBytes() / (1024 * 1024))
      << " coloredVoxels=" << tsdf.coloredVoxels()
      << " targetVoxels=" << targetTsdf.voxels()
      << " targetColoredVoxels=" << targetTsdf.coloredVoxels()
      << " targetVoxelSize=" << targetTsdf.voxelSize() << "\n"
      << "DepthCalib: enabled=" << (depthCalibrationEnabled ? 1 : 0)
      << " valid=" << (lastCalibValid ? 1 : 0)
      << " model=" << (lastCalibInverse ? "inverse" : "linear")
      << " scale=" << lastCalibScale
      << " shift=" << lastCalibShift
      << " confidence=" << lastCalibConfidence
      << " samples=" << lastCalibSamples
      << " accepted=" << calibFrames
      << " rejected=" << calibRejectFrames << "\n"
      << "DepthFusionApplied: calibrated=" << (lastFusionDepthCalibrated ? 1 : 0)
      << " min=" << lastFusionDepthMin
      << " max=" << lastFusionDepthMax
      << " mean=" << lastFusionDepthMean
      << " valid=" << lastFusionDepthValid
      << " conversionFallback=" << lastFusionDepthConversionFallback << "\n"
      << "CorePatch: vc171-continuity-identity reanchor=disabled" << "\n"
      << "V13FusionEpoch: active=" << (epochActive ? 1 : 0)
      // V0.13：sticky=1 表示这一代 epoch **冻结后不再因为漂移清几何**，
      // suspended=1 表示当前只是暂停了新融合（几何还在）。
      << " sticky=1"
      << " evidenceWindow=" << frozenEvidenceStreak
      << " suspended=" << (epochSuspended ? 1 : 0)
      << " suspendEvents=" << epochSuspendEvents
      << " suspendedFrames=" << fusionSuspendedFrames
      << " catastrophicRebuilds=" << epochCatastrophicRebuilds
      << " catastrophicRatio=" << kEpochCatastrophicRatio
      << " catastrophicFrames=" << kEpochCatastrophicFrames
      << " index=" << epochIndex
      << " opens=" << epochOpens
      << " rebuilds=" << epochRebuilds
      << " stableFrames=" << kEpochStableFrames
      << " stableStreak=" << epochStableStreak
      << " rebuildRatio=" << kEpochRebuildRatio
      << " driftSuspendFrames=" << kEpochDriftSuspendFrames
      << " driftSuspendRatio=" << kEpochDriftSuspendRatio
      << " calibLossFrames=" << kEpochCalibLossFrames
      << " badStreak=" << epochBadStreak
      << " driftRejects=" << epochDriftRejects
      << " lastDriftRel=" << epochLastDriftRel
      << " frozenScale=" << epochCalib.scale
      << " frozenShift=" << epochCalib.shift
      << " frozenInverse=" << (epochCalib.inverseDepthModel ? 1 : 0)
      << " refRaw=" << epochRefRaw
      << " refZ=" << epochRefZ
      << " fusedFrames=" << fusionFusedFrames
      << " gatedFrames=" << fusionGatedFrames << "\n"
      << "V13DepthCalibReset: meshPipelineResets=" << depthCalibKeptAcrossTarget
      << " fullResets=" << depthCalibFullResets << "\n"
      << "Temporal: ratio=" << lastTemporalRatio
      << " checks=" << temporalChecks
      << " rejects=" << temporalRejects << "\n"
      << "Mesh: dirty=" << (meshDirty ? 1 : 0)
      << " builds=" << meshBuilds
      << " quality=" << meshQuality
      << " vertices=" << meshStats.outVertices
      << " triangles=" << meshStats.outTriangles
      << " rawTriangles=" << meshStats.rawTriangles
      << " componentsRemoved=" << meshStats.componentsRemoved
      << " decimated=" << (meshStats.decimated ? 1 : 0)
      << " buildMs=" << meshStats.totalMs
      << " note=" << meshStats.note << "\n"
      << "Vulkan: " << (vk ? "available" : "unavailable") << "\n"
      << mobilescan3d::stereo_anchor::summary() << "\n"
      << "Next view: " << kf.guidance();
    return e->NewStringUTF(s.str().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetGuidance(JNIEnv* e, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return e->NewStringUTF(kf.guidance().c_str());
}


// Round 2: compact scanner telemetry for the live UI.
// Keep this protocol intentionally small; nativeGetStats() is a diagnostic report,
// not something the UI should parse every second.
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetScanUiMetrics(
        JNIEnv* e, jobject, jfloatArray out) {
    static constexpr int kSlots = 8;
    if (!out || e->GetArrayLength(out) < kSlots) return 0;

    jfloat values[kSlots] = {};
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const AiQuality q = ai.q();
        const bool targetOnly = targetLockEverArmed;
        const size_t confirmed = targetOnly ? targetG.confirmedCount(2) : g.confirmedCount(2);
        const size_t stable = targetOnly ? targetG.confirmedCount(3) : g.confirmedCount(3);

        values[0] = static_cast<jfloat>(kf.size());
        values[1] = static_cast<jfloat>(kf.overlap());
        values[2] = q.sharpness;
        values[3] = q.exposure;
        values[4] = static_cast<jfloat>(vio.features());
        values[5] = static_cast<jfloat>(confirmed);
        values[6] = static_cast<jfloat>(stable);
        values[7] = targetOnly ? 1.0f : 0.0f;
    }
    e->SetFloatArrayRegion(out, 0, kSlots, values);
    return kSlots;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetGaussians(JNIEnv* e, jobject,
                                                      jfloatArray out, jint maxPoints,
                                                      jint minHits) {
    // Validate before acquiring JNI elements. Division avoids signed overflow
    // when an invalid caller supplies INT_MAX for maxPoints.
    if (!out || maxPoints <= 0 || maxPoints > e->GetArrayLength(out) / 6) return 0;
    jfloat* dst = e->GetFloatArrayElements(out, nullptr);
    if (!dst) {
        return 0;
    }
    // 长度契约：native 写 out + written*6，written <= maxPoints。
    // 调用方必须给够 maxPoints*6 个槽，这里再兜一次底，避免越界写 Java 数组。
    size_t n;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const int need = minHits > 0 ? minHits : 1;
        gArMinHits = need;
        // Renderer 只读**目标模型**：背景点云根本没有进入 targetG 的代码路径。
        // 还没有目标（用户没锁定过）时退回全场景 g，避免一上来屏幕全空。
        //
        // V0.13 Strict target-only：**一旦本会话锁定过目标就彻底禁止回退**。
        // 旧行为下 `targetG.count()==0` 会静默退回全场景 g —— 用户以为看到的
        // 是「实时目标几何」，其实是墙和桌子（实机截图里底部那一片累计点）。
        // 锁定过就必须诚实：目标点云是 0，那就返回 0 个点。
        if (targetLockEverArmed) {
            if (targetG.count() > 0) {
                n = targetG.copyPoints(dst, (size_t)maxPoints, need);
            } else {
                n = 0;
            }
        } else {
            SurfelEngine& src = (targetG.count() > 0) ? targetG : g;
            n = src.copyPoints(dst, (size_t)maxPoints, need);
        }
        gArDrawnPoints = n;
    }
    e->ReleaseFloatArrayElements(out, dst, 0);
    return (jint)n;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsInitialized(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return (vinsPoseOk && !vinsWorldDiscontinuous()) ? JNI_TRUE : JNI_FALSE;
}

// V0.13.1：UI 靠它区分「还没初始化（该提示用户移动）」与「初始化后跟丢
// （该提示回到已扫区域）」。两者用同一句「定位失锁」会把用户带偏。
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeVinsEverInitialized(JNIEnv*, jobject) {
    return vinsEverInitialized() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetHudMetrics(JNIEnv* e, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(1);
    s << "VINS " << vinsFrames << " 帧" << (vinsInitialized() ? "" : ", 初始化中")
      << " / " << lastVinsMs << " ms"
      << " · 特征 " << vio.features()
      << " · 深度 " << depthFrames << " 帧";
    // 只报一个「总量」没有意义：实机出现过 600000 总点里 stable 只有 28，
    // 那种情况下屏幕上几乎全是一次性噪声，而总数看起来却很壮观。
    if (targetLockEverArmed) {
        // V0.13 Strict target-only：锁定过目标就只报目标模型，哪怕它是 0。
        // 这里绝不能落回 g.count() —— 那会让 HUD 和屏幕上的实际内容对不上。
        s << " · 目标 " << targetG.count()
          << " 确认 " << targetG.confirmedCount(2)
          << " 绘制 " << gArDrawnPoints
          << " (target-only)";
    } else if (targetG.count() > 0) {
        s << " · 目标 " << targetG.count()
          << " 确认 " << targetG.confirmedCount(2)
          << " 绘制 " << gArDrawnPoints;
    } else {
        s << " · 地图 " << g.count()
          << " 确认 " << g.confirmedCount(2)
          << " 稳定 " << g.confirmedCount(3)
          << " 绘制 " << gArDrawnPoints;
    }
    return e->NewStringUTF(s.str().c_str());
}

// nativeGetDepthDiagnostics 的槽数。
// **必须与 Kotlin 侧 NativeBridge.DEPTH_DIAGNOSTIC_SLOTS 保持一致。**
// 这里做了长度检查、长度不足就安全返回 0；但如果调用方按旧槽数建数组、
// 又按新槽数索引，越界会发生在 Kotlin 侧（「导出反馈报告」闪退那次就是这个形态）。
static constexpr int kDepthDiagSlots = 7;

// nativeGetTargetState 的槽数。
// **必须与 Kotlin 侧 NativeBridge.TARGET_STATE_SLOTS 保持一致。**
// 0..9 是原有字段，10..13 是本轮新增：
//   10 visibleFraction  目标可见比例（<0.12 连续 3 帧 -> REACQUIRING）
//   11 centerXNorm      目标中心 X（相机归一化，未裁剪，可越界）
//   12 centerYNorm      目标中心 Y
//   13 edgeLostFrames   连续「可见比例 < 12%」的帧数
//   14 presenceValid    PresenceGate 结论（0/1）：UI 据此决定绿框画不画
//   15 appearanceOk     外观后端是否可用（0/1，仅诊断）
// UI 靠 10..13 才能给出「目标接近边缘 / 已离开画面」的持续提示和方向箭头；
// 旧协议只给裁剪后的 bbox，目标完全出界时 bbox 退化成空矩形，UI 只能干等。
// 14/15 是这一轮新增：**box 还在画面里 != 物体还在**，绿框的可见性必须由
// presenceValid 决定，而不是 bbox 的几何位置。
// V0.13 起 16..19 新增：
//   16 identityScore        候选框与初始外观锚点的 NCC（-1 = 无法判定）
//   17 identityRejects      因身份不通过而拒绝采纳 Nano 的累计次数
//   18 bboxScaleFromInitial 当前候选框相对初始框的膨胀倍数
//   19 identityAnchorReady  身份锚点是否可用（0/1）
// 这四个值的作用是让「Nano 分数很高、但框已经不在目标上」这件事故
// 在 UI 上可读：旧协议里只能看到 Nano score 与 KLT inlier，两者会
// 一起漂走、互相确认，从数字上完全看不出问题。
static constexpr int kTargetStateSlots = 20;

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetPointCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    // 目标模型优先：UI 上的「点数」应该反映将要导出的东西
    // V0.13：锁定过目标之后不再把全场景点数顶上来，否则 HUD 会显示一个
    // 和目标无关的大数字，和屏幕上真正画出来的 0 个点互相矛盾。
    if (targetLockEverArmed) {
        return (jint)targetG.count();
    }
    return (jint)((targetG.count() > 0) ? targetG.count() : g.count());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSelectTarget(JNIEnv*, jobject, jfloat u, jfloat v) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (!objectTracker) return JNI_FALSE;
    try {
        const bool ok = objectTracker->requestTarget(u, v);
        if (ok) {
            // 新目标：目标专用模型必须清零重建，否则新旧目标会混在同一个点云里
            resetTargetModel();
            // V0.13：从此累计层只读目标模型（见 nativeGetGaussians）。
            targetLockEverArmed = true;
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

// 用户手指拖出的矩形（相机归一化坐标，允许任意方向）。
// 走 gStateMutex 只是为了安全拿到 shared_ptr；真正的工作在 ObjectTracker
// 自己的 mutex_ 里完成，所以这里持锁时间很短，不会挡住相机/深度线程。
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSelectTargetRect(
        JNIEnv*, jobject, jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) return JNI_FALSE;
    try {
        const bool ok = tracker->requestTargetRect(x0, y0, x1, y1);
        if (ok) {
            // 新目标：目标专用模型清零重建
            std::lock_guard<std::mutex> lk(gStateMutex);
            resetTargetModel();
            // V0.13：从此累计层只读目标模型（见 nativeGetGaussians）。
            targetLockEverArmed = true;
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeClearTarget(JNIEnv*, jobject) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (tracker) tracker->clearTarget();
    if (tracker) {
        // 清目标 = 清目标模型：否则下一次锁定会继承上一轮的点云
        std::lock_guard<std::mutex> lk(gStateMutex);
        resetTargetModel();
        // V0.13：用户主动解除锁定 -> 回到「自动」显示模式（全场景）。
        targetLockEverArmed = false;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetObjectLockEnabled(JNIEnv*, jobject, jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (objectTracker) objectTracker->setEnabled(enabled);
}

// UI 侧的绿色目标框现在以 30Hz 拉取本接口（原来是 1Hz，肉眼看起来
// 「追踪很慢」其实就是刷新率问题，不是 tracker 算得慢）。
// 所以这里**不能**像以前那样整段持有 gStateMutex：
// gStateMutex 同时保护 VINS / Depth / Camera 主链，长期占用会直接拖慢它们。
// 正确做法是只在取 shared_ptr 的瞬间持锁，info() 靠 ObjectTracker 自己的 mutex_。
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetState(JNIEnv* env, jobject, jfloatArray out) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) {
        return 0;
    }
    if (out == nullptr || env->GetArrayLength(out) < kTargetStateSlots) {
        return static_cast<jint>(tracker->info().state);
    }

    const TargetTrackInfo info = tracker->info();
    jfloat* dst = env->GetFloatArrayElements(out, nullptr);
    if (dst != nullptr) {
        dst[0] = static_cast<float>(static_cast<int>(info.state));
        dst[1] = info.x0;
        dst[2] = info.y0;
        dst[3] = info.x1;
        dst[4] = info.y1;
        dst[5] = info.confidence;
        dst[6] = info.medianDepth;
        dst[7] = info.roiSharpness;
        dst[8] = static_cast<float>(info.trackedPoints);
        dst[9] = info.inlierRatio;
        dst[10] = info.visibleFraction;
        dst[11] = info.centerXNorm;
        dst[12] = info.centerYNorm;
        dst[13] = static_cast<float>(info.edgeLostFrames);
        dst[14] = info.presenceValid ? 1.f : 0.f;
        dst[15] = info.appearanceAvailable ? 1.f : 0.f;
        // V0.13 身份/几何诊断。UI 才能把「Nano 说 0.94 但身份只有 0.12」
        // 这种「两个 tracker 一起漂走」的形态显示出来。
        dst[16] = info.identityScore;
        dst[17] = static_cast<float>(info.identityRejects);
        dst[18] = info.bboxScaleFromInitial;
        dst[19] = info.identityAnchorReady ? 1.f : 0.f;
        env->ReleaseFloatArrayElements(out, dst, 0);
    }
    return static_cast<jint>(info.state);
}

// Exact source timestamp prevents rejected frames from reusing a stale distance.
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetCameraDistance(JNIEnv* env, jobject,
                                                         jlong timestampNs, jfloatArray out) {
    if (!out || env->GetArrayLength(out) < 4) return 0;
    std::lock_guard<std::mutex> lock(gStateMutex);
    float values[4] = {};
    if (timestampNs > 0 && cameraDistanceTs == static_cast<uint64_t>(timestampNs)) {
        values[0] = cameraDistance.meters;
        values[1] = cameraDistance.coverage;
        values[2] = cameraDistance.spread;
        values[3] = static_cast<float>(cameraDistanceSource);
    }
    env->SetFloatArrayRegion(out, 0, 4, values);
    return env->ExceptionCheck() ? 0 : 4;
}

// 深度尺度诊断：只上报可观测量，不自动施加任何 scale 修正。
// out[0]=targetDepthP10 out[1]=targetDepthMedian out[2]=targetDepthP90
// out[3]=vinsTriangulatedDepthMedian
// out[4]=targetDepthValidPixels out[5]=targetDepthSampleCount out[6]=targetDepthRoiArea
//
// 后三个是「这个中位数到底可不可信」的前提：目标出界时 ROI 会被裁到只剩几行，
// 有效像素掉到个位数，P10/P50/P90 就会退化成同一个值（实机 P10=P50=P90=5.14128）。
// 上层据此拒绝这类样本，而不是把它算进 depth scale 的统计里。
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetDepthDiagnostics(JNIEnv* env, jobject, jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < kDepthDiagSlots) {
        return 0;
    }

    float vals[kDepthDiagSlots] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (tracker) {
        const TargetTrackInfo info = tracker->info();
        vals[0] = info.depthP10;
        vals[1] = info.medianDepth;
        vals[2] = info.depthP90;
        vals[4] = static_cast<float>(info.depthValidPixels);
        vals[5] = static_cast<float>(info.depthSampleCount);
        vals[6] = static_cast<float>(info.depthRoiArea);
    }

    // vinsFeatureDepthMedian 内部持有自己的 vins 锁，必须在 gStateMutex 之外调用
    vals[3] = vinsFeatureDepthMedian();

    env->SetFloatArrayRegion(out, 0, kDepthDiagSlots, vals);
    return 1;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetDiagnostics(JNIEnv* env, jobject) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) {
        return env->NewStringUTF("tracker=null");
    }

    const TargetTrackInfo i = tracker->info();
    std::ostringstream s;
    s << "haveCameraFrame=" << (i.haveCameraFrame ? "true" : "false") << "\n"
      << "frame=" << i.frameWidth << "x" << i.frameHeight << "\n"
      << "lastFrameTs=" << i.lastFrameTs << "\n"
      << "cameraUpdateCalls=" << i.cameraUpdateCalls << "\n"
      << "trackerUpdateCalls=" << i.trackerUpdateCalls << "\n"
      << "targetRequestCalls=" << i.targetRequestCalls << "\n"
      << "targetRequestAccepted=" << i.targetRequestAccepted << "\n"
      << "targetRequestRejected=" << i.targetRequestRejected << "\n"
      << "acquireCalls=" << i.acquireCalls << "\n"
      << "acquireSuccess=" << i.acquireSuccess << "\n"
      << "acquireFail=" << i.acquireFail << "\n"
      << "trackSuccess=" << i.trackSuccess << "\n"
      << "trackLost=" << i.trackLost << "\n"
      << "nanoLoaded=" << (i.nanoLoaded ? "true" : "false") << "\n"
      << "nanoInitCalls=" << i.nanoInitCalls << "\n"
      << "nanoUpdateCalls=" << i.nanoUpdateCalls << "\n"
      << "nanoFailures=" << i.nanoFailures << "\n"
      << "nanoRecoveries=" << i.nanoRecoveries << "\n"
      << "nanoScore=" << i.nanoScore << "\n"
      << "nanoKltIou=" << i.nanoKltIou << "\n"
      << "nanoKltRejects=" << i.nanoKltRejects << "\n"
      << "nanoLastMs=" << i.nanoLastMs << "\n"
      << "weakKltFrames=" << i.weakKltFrames << "\n"
      << "kltGoodPoints=" << i.kltGoodPoints << "\n"
      << "nanoForcedUpdates=" << i.nanoForcedUpdates << "\n"
      << "nanoWeakAttempts=" << i.nanoWeakAttempts << "\n"
      << "nanoWeakRecoveries=" << i.nanoWeakRecoveries << "\n"
      << "nanoUsedRealColor=" << (i.nanoUsedRealColor ? "true" : "false") << "\n"
      << "colorFrameValid=" << (i.colorFrameValid ? "true" : "false") << "\n"
      << "colorFrameSize=" << i.colorFrameWidth << "x" << i.colorFrameHeight << "\n"
      << "colorFrameCalls=" << i.colorFrameCalls << "\n"
      << "globalSearchReady=" << i.globalSearchReady << "\n"
      << "globalSearches=" << i.globalSearches << "\n"
      << "globalRecoveries=" << i.globalRecoveries << "\n"
      << "globalInliers=" << i.globalInliers << "\n"
      << "depthP10=" << i.depthP10 << "\n"
      << "depthMedian=" << i.medianDepth << "\n"
      << "depthP90=" << i.depthP90 << "\n"
      << "depthValidPixels=" << i.depthValidPixels << "\n"
      << "depthSampleCount=" << i.depthSampleCount << "\n"
      << "depthRoiArea=" << i.depthRoiArea << "\n"
      << "templateAllocated=" << (i.targetTemplateAllocated ? "true" : "false") << "\n"
      << "templateSize=" << i.templateWidth << "x" << i.templateHeight << "\n"
      << "prevGrayValid=" << (i.prevGrayValid ? "true" : "false") << "\n"
      << "prevGraySize=" << i.prevGrayWidth << "x" << i.prevGrayHeight << "\n"
      << "prevPointCount=" << i.prevPointCount << "\n"
      << "bboxWidthPx=" << i.bboxWidthPx << "\n"
      << "bboxHeightPx=" << i.bboxHeightPx << "\n"
      << "fullBBox=" << i.fullBBoxWidthPx << "x" << i.fullBBoxHeightPx << "\n"
      << "visibleBBox=" << i.visibleBBoxWidthPx << "x" << i.visibleBBoxHeightPx << "\n"
      << "visibleFraction=" << i.visibleFraction << "\n"
      << "lastAffineScale=" << i.lastAffineScale << "\n"
      << "affineScaleEMA=" << i.affineScaleEMA << "\n"
      << "reseedCount=" << i.reseedCount << "\n"
      << "lastEvent=" << i.lastEvent << "\n"
      << "depthFilterCalls=" << i.depthFilterCalls << "\n"
      << "depthFilterSkipped=" << i.depthFilterSkipped << "\n"
      << "presenceValid=" << (i.presenceValid ? "true" : "false") << "\n"
      << "presenceLostCount=" << i.presenceLostCount << "\n"
      << "appearanceAvailable=" << (i.appearanceAvailable ? "true" : "false") << "\n"
      << "appearanceBackend=" << i.appearanceBackend << "\n"
      << "lastError=" << i.lastError;
    return env->NewStringUTF(s.str().c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRenderPose(JNIEnv* env, jobject, jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 12) {
        return JNI_FALSE;
    }

    float pose[12];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        if (!vinsPoseOk || !haveLastGoodVinsPose) {
            return JNI_FALSE;
        }
        quatToR(acceptedVinsQ[0], acceptedVinsQ[1], acceptedVinsQ[2], acceptedVinsQ[3], pose);
        pose[9] = acceptedVinsT[0];
        pose[10] = acceptedVinsT[1];
        pose[11] = acceptedVinsT[2];
    }

    // V0.7 saved-world transform (latest).
    float savedR[9];
    float savedT[3];
    if (!gPersistentRelocalizer.transformPose(
            pose, pose + 9, savedR, savedT)) {
        return JNI_FALSE;
    }
    for (int i = 0; i < 9; ++i) pose[i] = savedR[i];
    pose[9] = savedT[0];
    pose[10] = savedT[1];
    pose[11] = savedT[2];
    env->SetFloatArrayRegion(out, 0, 12, pose);
    return JNI_TRUE;
}

// 当前 depth 帧 + 当前 target ROI 的调试层点数（AR 坐标链验证用）
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTargetDepthDebug(JNIEnv* e, jobject,
                                                             jfloatArray out,
                                                             jint maxPoints) {
    if (out == nullptr || maxPoints <= 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(gStateMutex);
    targetDebugQueries++;
    const int n = std::min(static_cast<int>(maxPoints), targetDebugPointCount);
    if (n <= 0) {
        return 0;
    }
    if (e->GetArrayLength(out) < static_cast<jsize>(n) * 6) {
        return 0;
    }
    e->SetFloatArrayRegion(out, 0, static_cast<jsize>(n) * 6,
                           targetDebugPoints.data());
    return (jint)n;
}

// 打开/关闭调试层的构建。关掉时不再做任何像素级的提取与转换。
extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetTargetDebugEnabled(JNIEnv*, jobject,
                                                               jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    targetDebugEnabled = (enabled == JNI_TRUE);
    if (!targetDebugEnabled) {
        targetDebugPointCount = 0;
    }
}

/**
 * 按 SENSOR_TIMESTAMP 查历史位姿（AR 正确跟随的前提）。
 *
 * 使用相邻有效曝光位姿的 translation lerp + quaternion slerp。
 * 历史不足时只允许 80ms 内的最近样本，拒绝跨越跟踪空洞。
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRenderPoseAt(JNIEnv* env, jobject,
                                                         jlong timestampNs,
                                                         jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 12) {
        return JNI_FALSE;
    }
    float pose[12];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        if (!vinsPoseOk || vinsWorldDiscontinuous() || renderPoseHistory.empty()) {
            return JNI_FALSE;
        }
        if (timestampNs <= 0 || !scan_pose::lookup(renderPoseHistory,
                static_cast<uint64_t>(timestampNs), kRenderPoseMaxAgeNs, pose)) {
            return JNI_FALSE;
        }
    }
    // V0.7 saved-world transform (timestamp).
    float savedR[9];
    float savedT[3];
    if (!gPersistentRelocalizer.transformPose(
            pose, pose + 9, savedR, savedT)) {
        return JNI_FALSE;
    }
    for (int i = 0; i < 9; ++i) pose[i] = savedR[i];
    pose[9] = savedT[0];
    pose[10] = savedT[1];
    pose[11] = savedT[2];
    env->SetFloatArrayRegion(out, 0, 12, pose);
    return JNI_TRUE;
}

/**
 * V0.13.34：AR 渲染专用「带兜底」位姿查询。
 *
 * 严格窗口（80ms 内插值查询）失败、但历史里最新样本足够新（|newest.ts - ts|
 * ≤ newestAgeNs，传 0 用默认 150ms）时，退回最新样本而不是整帧不画。
 *
 * 实机（PLK110 vc172 扫描）：热身后仍有 ~12.6% 帧查不到（ArPoseProbe
 * latest=12.4%~13%），快速转动时恰恰是这些帧最需要画 —— 整帧不画表现为
 * 「点云/网格闪烁、不连续」。150ms 内的位姿滞后在常规扫描运动（<60°/s）
 * 下误差 ≤9°，比「模型消失」更可接受。世界跳变 / 历史为空仍然拒绝
 * （与 strict 版一致）；strict 命中时与 nativeGetRenderPoseAt 结果完全相同。
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRenderPoseAtBounded(JNIEnv* env, jobject,
                                                                jlong timestampNs,
                                                                jlong newestAgeNs,
                                                                jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 12) {
        return JNI_FALSE;
    }
    if (newestAgeNs <= 0) {
        newestAgeNs = 150'000'000LL;
    }
    float pose[12];
    bool bounded = false;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        if (vinsWorldDiscontinuous() || renderPoseHistory.empty()) {
            return JNI_FALSE;
        }
        const bool strictHit = timestampNs > 0 && vinsPoseOk &&
            scan_pose::lookup(renderPoseHistory,
                              static_cast<uint64_t>(timestampNs),
                              kRenderPoseMaxAgeNs, pose);
        if (!strictHit) {
            // 兜底：不要求 vinsPoseOk（step 门拒绝帧正是最需要兜底的帧），
            // 只要求最新样本离请求时间戳足够近。
            const RenderPoseSample& newest = renderPoseHistory.back();
            int64_t age = static_cast<int64_t>(newest.ts) - timestampNs;
            if (age < 0) age = -age;
            if (timestampNs <= 0 || age > newestAgeNs) {
                return JNI_FALSE;
            }
            for (int i = 0; i < 9; ++i) pose[i] = newest.R[i];
            pose[9] = newest.t[0];
            pose[10] = newest.t[1];
            pose[11] = newest.t[2];
            bounded = true;
        }
    }
    // V0.7 saved-world transform（与 strict 版一致）。
    float savedR[9];
    float savedT[3];
    if (!gPersistentRelocalizer.transformPose(
            pose, pose + 9, savedR, savedT)) {
        return JNI_FALSE;
    }
    for (int i = 0; i < 9; ++i) pose[i] = savedR[i];
    pose[9] = savedT[0];
    pose[10] = savedT[1];
    pose[11] = savedT[2];
    env->SetFloatArrayRegion(out, 0, 12, pose);
    if (bounded) {
        ++renderPoseBoundedFallbacks;
        if ((renderPoseBoundedFallbacks % 180) == 1) {
            LOGI("RenderPoseBounded fallbacks=%llu (strict miss -> newest within %lldms)",
                 (unsigned long long)renderPoseBoundedFallbacks,
                 (long long)(newestAgeNs / 1'000'000LL));
        }
    }
    return JNI_TRUE;
}

/**
 * V0.13.2：nativeGetRenderPoseAt 的关键帧专用宽松版。
 *
 * 12MP still burst 期间预览管线停摆，VINS 位姿历史（renderPoseHistory，
 * 只在预览帧的 VINS pose 被接受时追加）在 burst 时间戳处出现空洞 ——
 * strict 80ms 窗口永远查不到，实机日志「poseValid=false poseAtOk=false」
 * 由此而来，所有 HQ 纹理关键帧被静默丢弃，导出 GLB 永远 plain。
 *
 * burst 的触发前提就是「用户已稳定持机」（3A 稳定门 + 运动门控），所以
 * 「burst 开始前最后一个位姿」对 burst 内任何一帧都是可接受的近似。
 * 容差由调用方传入（关键帧路径用 500ms）；AR 渲染仍走 strict 80ms 版。
 */
// V0.13.14 ArAxesProbe 节流时间戳（按位姿样本 ts，1s 一条）。
static int64_t sAxesProbeLastLogNs = 0;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetRenderPoseAtTol(JNIEnv* env, jobject,
                                                            jlong timestampNs,
                                                            jlong maxAgeNs,
                                                            jfloatArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 12) {
        return JNI_FALSE;
    }
    const int64_t tol = maxAgeNs > 0
        ? static_cast<int64_t>(maxAgeNs)
        : kRenderPoseMaxAgeNs;
    float pose[12];
    // V0.13.14 ArAxesProbe：锁外可见的样本信息捕获。
    int64_t probeSampleTs = 0;
    int64_t probeDeltaMs = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        // VC160+：宽容路径不再因为「最新一帧 VINS 被 step 门拒绝」而整体放弃。
        // 本函数存在的理由恰恰是「burst 期间预览管线停摆、没有新位姿」，所以瞬时
        // vinsPoseOk=false 正是它应当容忍的场景，而不是判废依据。实机 10:23–10:32
        // 有 21/46 个 HQ burst 拿到 poseValid=false poseAtOk=false 被静默丢弃，而
        // 同窗口宽容查询成功 87 次（平均偏差 379ms）—— 位姿历史是好的，只是被这个
        // 瞬时标志挡住了。世界跳变仍拒绝（那才是真的不能沿用旧位姿）；历史为空也
        // 拒绝（VINS 从未初始化，没有可用近似）。容差仍由调用方给（burst 用 500ms），
        // AR 渲染走 strict 80ms 版，本改动不影响它。
        if (vinsWorldDiscontinuous()) {
            ++poseTolMissWorld;
            return JNI_FALSE;
        }
        if (renderPoseHistory.empty()) {
            ++poseTolMissEmpty;
            return JNI_FALSE;
        }
        const int64_t want = static_cast<int64_t>(timestampNs);
        const RenderPoseSample* best = nullptr;
        int64_t bestDelta = 0;
        for (const RenderPoseSample& rp : renderPoseHistory) {
            int64_t d = static_cast<int64_t>(rp.ts) - want;
            if (d < 0) {
                d = -d;
            }
            if (best == nullptr || d < bestDelta) {
                best = &rp;
                bestDelta = d;
            }
        }
        if (best == nullptr || bestDelta > tol) {
            ++poseTolMissOver;
            return JNI_FALSE;
        }
        ++poseTolHit;
        // V0.13.2 诊断：cov 异常低时需要区分「宽容窗口拿到了过期位姿（扫描中
        // 手机已转动）」与「旋转约定错误」。年龄大 => 过期；年龄小但投影仍偏 =>
        // 约定错误。只在宽容查询路径打日志（AR 用的 strict 版不打，避免刷屏）。
        LOGI("renderPoseAtTol match: deltaMs=%.0f newestAgeMs=%.0f hist=%zu",
             bestDelta / 1e6,
             (want - static_cast<int64_t>(renderPoseHistory.back().ts)) / 1e6,
             renderPoseHistory.size());
        for (int i = 0; i < 9; ++i) {
            pose[i] = best->R[i];
        }
        pose[9] = best->t[0];
        pose[10] = best->t[1];
        pose[11] = best->t[2];
        probeSampleTs = static_cast<int64_t>(best->ts);
        probeDeltaMs = bestDelta / 1'000'000LL;
    }
    // Axis diagnostics only: VINS subtracts g from world-frame accelerometer
    // specific force, so initialization aligns physical UP with world +Z.
    // Camera columns refer to raw sensor pixels, not portrait screen axes.
    // Neither the sign of raw camera-down.z nor a single preview-affine
    // coefficient diagnoses inversion; sensor orientation must also be applied.
    // 1s 节流（按位姿样本时间戳），每秒至多 1 条。
    const bool axesProbeDue =
        (probeSampleTs - sAxesProbeLastLogNs) > 1'000'000'000LL;
    float rawDownFwd[6] = {
        pose[1], pose[4], pose[7],   // raw col1 (down) 世界分量
        pose[2], pose[5], pose[8]    // raw col2 (forward) 世界分量
    };
    if (axesProbeDue) {
        sAxesProbeLastLogNs = probeSampleTs;
    }
    float savedR[9];
    float savedT[3];
    if (!gPersistentRelocalizer.transformPose(
            pose, pose + 9, savedR, savedT)) {
        return JNI_FALSE;
    }
    for (int i = 0; i < 9; ++i) pose[i] = savedR[i];
    pose[9] = savedT[0];
    pose[10] = savedT[1];
    pose[11] = savedT[2];
    if (axesProbeDue) {
        LOGI("ArAxesProbe raw down=(%.3f,%.3f,%.3f) fwd=(%.3f,%.3f,%.3f) "
             "post down=(%.3f,%.3f,%.3f) fwd=(%.3f,%.3f,%.3f) deltaMs=%.0f",
             rawDownFwd[0], rawDownFwd[1], rawDownFwd[2],
             rawDownFwd[3], rawDownFwd[4], rawDownFwd[5],
             pose[1], pose[4], pose[7],
             pose[2], pose[5], pose[8],
             static_cast<double>(probeDeltaMs));
    }
    env->SetFloatArrayRegion(out, 0, 12, pose);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeConfigureTrackerModels(JNIEnv* env, jobject, jstring backbone, jstring head) {
    std::shared_ptr<ObjectTracker> tracker;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tracker = objectTracker;
    }
    if (!tracker) return JNI_FALSE;

    const char* b = env->GetStringUTFChars(backbone, nullptr);
    const char* h = env->GetStringUTFChars(head, nullptr);
    if (!b || !h) {
        if (b) env->ReleaseStringUTFChars(backbone, b);
        if (h) env->ReleaseStringUTFChars(head, h);
        return JNI_FALSE;
    }

    const bool ok = tracker->configureNano(b, h);
    // 外观后端接缝：工厂当前只会返回 nanotrack（LightTrack-ncnn 需要 ncnn 运行库，
    // 本工程没有链接），这里如实配置并把后端名写进诊断报告。
    tracker->configureAppearance(std::string(), std::string(), b, h);
    env->ReleaseStringUTFChars(backbone, b);
    env->ReleaseStringUTFChars(head, h);
    return ok ? JNI_TRUE : JNI_FALSE;
}


// ============================================================================
//  Mesh / GLB / 深度标定 JNI
// ============================================================================

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetVoxelSizes(JNIEnv*, jobject,
                                                       jfloat sceneMeters,
                                                       jfloat targetMeters) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (sceneMeters > 0.f) {
        tsdf.setVoxelSize(sceneMeters);
    }
    if (targetMeters > 0.f) {
        targetTsdf.setVoxelSize(targetMeters);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetVoxelBudget(JNIEnv*, jobject,
                                                        jint sceneBlocks,
                                                        jint targetBlocks) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (sceneBlocks > 0) {
        tsdf.setMaxBlocks((size_t)sceneBlocks);
    }
    if (targetBlocks > 0) {
        targetTsdf.setMaxBlocks((size_t)targetBlocks);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetVoxelProfile(JNIEnv*, jobject,
                                                         jint profile) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    // 评审 P1-2：按扫描档位切换 TSDF 体素分辨率 + 块预算。
    // 字段在 nativeCreate 里被重置为默认（场景 20mm / 目标 8mm），
    // 因此本函数必须在 startScan 之后、喂帧之前调用。
    //   0 = OBJECT_HQ : 目标 4mm / 场景 8mm，小物体细节
    //   1 = OBJECT_FAST: 目标 8mm / 场景 12mm，实时更顺
    //   2 = ROOM      : 目标 8mm / 场景 20mm，大空间覆盖
    // 高分辨率档位给更多块预算，避免「半边模型」截断。
    float targetVox = 0.008f, sceneVox = 0.020f;
    size_t targetBlocks = 8192, sceneBlocks = 8192;
    switch (profile) {
        case 0:  // OBJECT_HQ
            targetVox = 0.004f;
            sceneVox  = 0.008f;
            targetBlocks = 8192;
            sceneBlocks  = 16384;
            break;
        case 1:  // OBJECT_FAST
            targetVox = 0.008f;
            sceneVox  = 0.012f;
            targetBlocks = 8192;
            sceneBlocks  = 12288;
            break;
        case 2:  // ROOM
            targetVox = 0.008f;
            sceneVox  = 0.020f;
            targetBlocks = 8192;
            sceneBlocks  = 8192;
            break;
        default:
            return;  // 未知档位：保持 nativeCreate 设好的默认
    }
    targetTsdf.setVoxelSize(targetVox);
    targetTsdf.setMaxBlocks(targetBlocks);
    tsdf.setVoxelSize(sceneVox);
    tsdf.setMaxBlocks(sceneBlocks);
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetDepthCalibrationEnabled(JNIEnv*, jobject,
                                                                   jboolean enabled) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    depthCalibrationEnabled = (enabled == JNI_TRUE);
}

/**
 * V0.13.4：上报当前深度帧所用的归一化映射 `d = aNorm*q + bNorm`。
 *
 * Kotlin 侧（MonoDepthProvider）每帧都会带上这三个数；只要 `version` 变了，
 * 就必须在**喂这一帧之前**调用本函数，让标定跟着换数值域。
 *
 * 谁会被重参数化：
 *   - `depthCalibrator` 当前已收敛的标定（下一帧拟合的 EMA 基线）
 *   - `epochCalib`（冻结中的 epoch 参数）
 *   - `epochRefRaw`（漂移监控的工作点，它是旧数值域下的 d，必须换算）
 *     —— 漏了它，漂移监控会拿「新 d」去比「旧 d 算出的 z」，永远误报漂移。
 */
extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSetDepthNormMapping(JNIEnv*, jobject,
                                                             jfloat aNorm,
                                                             jfloat bNorm,
                                                             jlong version) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    if (!std::isfinite(aNorm) || !std::isfinite(bNorm)) return;
    if (std::fabs(aNorm) < 1e-12f) return;
    if (version <= 0) return;

    const bool first = !gHaveDepthNorm;
    const bool changed = gHaveDepthNorm && (uint64_t)version != gDepthNormVersion;
    if (!first && !changed) return;

    if (first) {
        gDepthNormA = aNorm;
        gDepthNormB = bNorm;
        gDepthNormVersion = (uint64_t)version;
        gHaveDepthNorm = true;
        return;
    }

    const float aOld = gDepthNormA;
    const float bOld = gDepthNormB;
    const float aNew = aNorm;
    const float bNew = bNorm;
    gDepthNormA = aNew;
    gDepthNormB = bNew;
    gDepthNormVersion = (uint64_t)version;
    ++gDepthNormReparams;
    // 诊断幅度：斜率相对变化。A 变大 -> d 的刻度被压缩 -> 标定要放大回去。
    gDepthNormLastScaleRel = (aOld != 0.f)
        ? std::fabs((aNew - aOld) / aOld) : 0.f;

    depthCalibrator.reparameterizeInput(aOld, bOld, aNew, bNew);
    epochCalib.reparameterizeLinearInput(aOld, bOld, aNew, bNew);
    epochLiveEmaInit = false;
    epochLiveEmaVar = 0.f;
    epochResumeStreak = 0;
    frozenEvidenceWindow.reset();
    frozenEvidenceStreak = 0;
    // epoch 参考工作点：旧域 -> 新域
    if (epochActive && std::isfinite(epochRefRaw)) {
        epochRefRaw = DepthCalibration::remapRaw(epochRefRaw, aOld, bOld, aNew, bNew);
    }
    // 冻结期间数值域还变，说明 provider 没被冻结（或映射来自其它 provider）；
    // 如实记一笔，便于在报告里区分「重参数化生效」与「数值域一直在动」。
    if (epochActive) ++gDepthNormChangesWhileEpoch;
}

/**
 * V0.13.4 深度数值域诊断（[DEPTH_NORM_STATS_SLOTS] = 5 槽）：
 *   0 当前 A（斜率）  1 当前 B（截距）  2 版本号  3 重参数化次数
 *   4 「已冻结 epoch 期间仍在变化」的次数
 * 版本号用于把任何一次尺度变化追溯到具体的归一化版本。
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetDepthNormStats(JNIEnv* e, jobject,
                                                           jfloatArray out) {
    if (out == nullptr) return 0;
    const jsize cap = e->GetArrayLength(out);
    if (cap < 5) return 0;
    jfloat v[5];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        v[0] = gDepthNormA;
        v[1] = gDepthNormB;
        v[2] = static_cast<jfloat>(gDepthNormVersion);
        v[3] = static_cast<jfloat>(gDepthNormReparams);
        v[4] = static_cast<jfloat>(gDepthNormChangesWhileEpoch);
    }
    e->SetFloatArrayRegion(out, 0, 5, v);
    return 5;
}

/**
 * 深度标定状态（12 槽）。
 *   0 scale   1 shift   2 confidence   3 samples       4 valid
 *   5 inverseModel  6 enabled  7 acceptedFrames  8 rejectedFrames
 *   9 temporalRatio 10 temporalRejects 11 usable
 */
/**
 * V0.12 Fusion Epoch 状态（16 槽）—— 给 Kotlin 侧 HUD / 报告用。
 *
 *   0 active        1 serial        2 goodStreak     3 badStreak
 *   4 warmupSkipped 5 fusedFrames   6 restarts       7 driftRejects
 *   8 lastDriftRel  9 scale        10 shift         11 confidence
 *  12 samples      13 inverseModel 14 currentUsable 15 startGoodFrames
 *
 * 槽位定义必须与 Kotlin 侧 NativeBridge.FUSION_EPOCH_INDEX_* 严格一致。
 * warmupSkipped 复用 fusionGatedFrames：二者语义相同（epoch 未生效、融合被
 * 门控拦下的帧数）。
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetFusionEpochStats(
        JNIEnv* e, jobject, jfloatArray out) {
    if (out == nullptr) {
        return 0;
    }
    const jsize cap = e->GetArrayLength(out);
    if (cap < 20) {
        return 0;
    }
    jfloat v[20];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const DepthCalibration& current = depthCalibrator.calibration();
        v[0] = epochActive ? 1.f : 0.f;
        v[1] = static_cast<jfloat>(epochIndex);
        v[2] = static_cast<jfloat>(epochStableStreak);
        v[3] = static_cast<jfloat>(epochBadStreak);
        v[4] = static_cast<jfloat>(fusionGatedFrames);
        v[5] = static_cast<jfloat>(fusionFusedFrames);
        v[6] = static_cast<jfloat>(epochRebuilds);
        v[7] = static_cast<jfloat>(epochDriftRejects);
        v[8] = epochLastDriftRel;
        v[9] = epochCalib.scale;
        v[10] = epochCalib.shift;
        v[11] = epochCalib.confidence;
        v[12] = static_cast<jfloat>(epochCalib.samples);
        v[13] = epochCalib.inverseDepthModel ? 1.f : 0.f;
        v[14] = (depthCalibrator.usable() && current.valid) ? 1.f : 0.f;
        v[15] = static_cast<jfloat>(kEpochStableFrames);
        // V0.13 Sticky Fusion Epoch：16..19 是新增的「暂停/极端重建」统计。
        v[16] = epochSuspended ? 1.f : 0.f;
        v[17] = static_cast<jfloat>(epochSuspendEvents);
        v[18] = static_cast<jfloat>(fusionSuspendedFrames);
        v[19] = static_cast<jfloat>(epochCatastrophicRebuilds);
    }
    e->SetFloatArrayRegion(out, 0, 20, v);
    return e->ExceptionCheck() ? 0 : 20;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetDepthCalibration(JNIEnv* e, jobject,
                                                             jfloatArray out) {
    if (out == nullptr) {
        return 0;
    }
    const jsize cap = e->GetArrayLength(out);
    if (cap < 12) {
        return 0;
    }
    jfloat v[12];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        v[0] = lastCalibScale;
        v[1] = lastCalibShift;
        v[2] = lastCalibConfidence;
        v[3] = (jfloat)lastCalibSamples;
        v[4] = lastCalibValid ? 1.f : 0.f;
        v[5] = lastCalibInverse ? 1.f : 0.f;
        v[6] = depthCalibrationEnabled ? 1.f : 0.f;
        v[7] = (jfloat)calibFrames;
        v[8] = (jfloat)calibRejectFrames;
        v[9] = lastTemporalRatio;
        v[10] = (jfloat)temporalRejects;
        v[11] = depthCalibrator.usable() ? 1.f : 0.f;
    }
    e->SetFloatArrayRegion(out, 0, 12, v);
    return 12;
}

/**
 * 从当前体素场构建三角网格（Marching Tetrahedra + 清理 + 简化）。
 *
 * 有目标模型时优先用 targetTsdf —— 与「有目标就只导目标」的 PLY / 点云口径一致。
 * 这是一次性操作，可能耗时几百毫秒，不要在帧回调里调。
 */
static jboolean buildMeshWithShape(jint quality, jint shape) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    const auto started = std::chrono::steady_clock::now();
    meshQuality = std::clamp((int)quality, 0, 2);

    MeshOptions opt;
    opt.quality = meshQuality;
    meshHasHardEdges = false;
    hardSurfaceStats = {};
    const bool requestedShape = shape >= 1 && shape <= 3;
    if (requestedShape) { opt.enableSmoothing = false; opt.enableDecimation = false; }
    const TsdfEngine& src = (targetTsdf.voxels() > 0) ? targetTsdf : tsdf;

    bool ok = meshEngine.build(src, opt, meshStats);
    if (ok && requestedShape) {
        Mesh fitted;
        if (fitHardSurface(meshEngine.mesh(), static_cast<HardSurfaceMode>(shape),
                           meshStats.voxelSize, fitted, hardSurfaceStats)) {
            meshEngine.replaceExportMesh(std::move(fitted));
            meshStats.outVertices = meshEngine.mesh().vertexCount();
            meshStats.outTriangles = meshEngine.mesh().triangleCount();
            meshHasHardEdges = true;
        } else {
            // Use the original scan quality pipeline on a rejected fit.
            opt.enableSmoothing = true; opt.enableDecimation = true;
            ok = meshEngine.build(src, opt, meshStats);
        }
        meshStats.note += " | " + hardSurfaceStats.message;
    }
    meshStats.totalMs = perfMsSince(started);
    meshBuilds++;
    // 构建成功后体素场与网格一致；失败则保持脏，允许用户重试。
    meshDirty = !ok;

    LOGI("nativeBuildMesh quality=%d ok=%d src=%s verts=%zu tris=%zu rawTris=%zu "
         "compRemoved=%zu/%zu decimated=%d ms=%.1f note=%s",
         meshQuality, ok ? 1 : 0,
         (&src == &targetTsdf) ? "target" : "scene",
         meshStats.outVertices, meshStats.outTriangles, meshStats.rawTriangles,
         meshStats.componentsRemoved, meshStats.componentsBefore,
         meshStats.decimated ? 1 : 0, meshStats.totalMs, meshStats.note.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeBuildMesh(JNIEnv*, jobject, jint quality) {
    try { return buildMeshWithShape(quality, 0); }
    catch (const std::exception& ex) { LOGI("mesh build: %s", ex.what()); return JNI_FALSE; }
    catch (...) { return JNI_FALSE; }
}
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeBuildMeshWithShape(JNIEnv*, jobject, jint quality, jint shape) {
    if (shape < 0 || shape > 3) return JNI_FALSE;
    try { return buildMeshWithShape(quality, shape); }
    catch (const std::exception& ex) { LOGI("shape build: %s", ex.what()); return JNI_FALSE; }
    catch (...) { return JNI_FALSE; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetHardSurfaceReport(JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return env->NewStringUTF(hardSurfaceStats.message.c_str());
}

/** mesh 统计（16 槽，全部为整型）。 */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshStats(JNIEnv* e, jobject) {
    jint v[16];
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        v[0] = (jint)meshStats.outVertices;
        v[1] = (jint)meshStats.outTriangles;
        v[2] = (jint)meshStats.rawVertices;
        v[3] = (jint)meshStats.rawTriangles;
        v[4] = (jint)meshStats.componentsBefore;
        v[5] = (jint)meshStats.componentsRemoved;
        v[6] = (jint)meshStats.trianglesRemovedRaw;
        v[7] = (jint)meshStats.trianglesRemovedComp;
        v[8] = (jint)meshStats.blocksScanned;
        v[9] = meshStats.decimated ? 1 : 0;
        v[10] = meshStats.ok ? 1 : 0;
        v[11] = (jint)meshQuality;
        v[12] = (jint)meshStats.totalMs;
        v[13] = (jint)meshBuilds;
        v[14] = (jint)(meshStats.voxelSize * 1000000.f);  // 微米
        v[15] = meshStats.smoothIterations;
    }
    jintArray out = e->NewIntArray(16);
    if (out == nullptr) {
        return nullptr;
    }
    e->SetIntArrayRegion(out, 0, 16, v);
    return out;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshVertexCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return (jint)meshEngine.mesh().vertexCount();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshIndexCount(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    return (jint)meshEngine.mesh().indices.size();
}

/** 顶点交错布局 x,y,z,nx,ny,nz,r,g,b（9 float/顶点）。返回写入的顶点数。 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshVertices(JNIEnv* e, jobject,
                                                         jfloatArray out,
                                                         jint maxVertices) {
    if (out == nullptr || maxVertices <= 0) {
        return 0;
    }
    std::vector<float> buf;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& m = meshEngine.mesh();
        const size_t nv = m.vertexCount();
        if (nv == 0) {
            return 0;
        }
        const jsize cap = e->GetArrayLength(out) / MESH_VERTEX_FLOATS;
        n = std::min((size_t)(maxVertices > 0 ? maxVertices : 0), (size_t)cap);
        if (n > nv) {
            n = nv;
        }
        buf.resize(n * MESH_VERTEX_FLOATS);
        const bool hasN = m.normals.size() >= nv * 3;
        const bool hasC = m.colors.size() >= nv * 3;
        for (size_t i = 0; i < n; ++i) {
            float* p = buf.data() + i * MESH_VERTEX_FLOATS;
            p[0] = m.positions[i * 3 + 0];
            p[1] = m.positions[i * 3 + 1];
            p[2] = m.positions[i * 3 + 2];
            if (hasN) {
                p[3] = m.normals[i * 3 + 0];
                p[4] = m.normals[i * 3 + 1];
                p[5] = m.normals[i * 3 + 2];
            } else {
                p[3] = 0.f;
                p[4] = 0.f;
                p[5] = 1.f;
            }
            if (hasC) {
                p[6] = m.colors[i * 3 + 0];
                p[7] = m.colors[i * 3 + 1];
                p[8] = m.colors[i * 3 + 2];
            } else {
                p[6] = p[7] = p[8] = 0.6f;
            }
        }
    }
    if (n == 0) {
        return 0;
    }
    e->SetFloatArrayRegion(out, 0, (jsize)(n * MESH_VERTEX_FLOATS), buf.data());
    return (jint)n;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshIndices(JNIEnv* e, jobject,
                                                        jintArray out,
                                                        jint maxIndices) {
    if (out == nullptr || maxIndices <= 0) {
        return 0;
    }
    std::vector<jint> buf;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& m = meshEngine.mesh();
        if (m.indices.empty()) {
            return 0;
        }
        const jsize cap = e->GetArrayLength(out);
        n = std::min((size_t)maxIndices, (size_t)cap);
        if (n > m.indices.size()) {
            n = m.indices.size();
        }
        buf.resize(n);
        for (size_t i = 0; i < n; ++i) {
            buf[i] = (jint)m.indices[i];
        }
    }
    if (n == 0) {
        return 0;
    }
    e->SetIntArrayRegion(out, 0, (jsize)n, buf.data());
    return (jint)n;
}

/**
 * Round 8 cumulative multi-segment registration.
 * referenceVertices use the normal Kotlin interleaved layout (9 float/vertex).
 * Current native mesh is the moving segment. Successful registration replaces
 * meshEngine's export mesh with the cumulative merged mesh.
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeAlignCurrentMeshToReference(
        JNIEnv* env, jobject,
        jfloatArray referenceVertices, jint referenceVertexCount,
        jintArray referenceIndices, jint referenceIndexCount,
        jint preferredYawDeg,
        jfloatArray outStats, jfloatArray outTransform) {
    if (!referenceVertices || !referenceIndices || !outStats || !outTransform ||
        referenceVertexCount <= 0 || referenceIndexCount < 3 ||
        env->GetArrayLength(referenceVertices) < referenceVertexCount * MESH_VERTEX_FLOATS ||
        env->GetArrayLength(referenceIndices) < referenceIndexCount ||
        env->GetArrayLength(outStats) < 10 || env->GetArrayLength(outTransform) < 16) {
        return JNI_FALSE;
    }

    std::vector<jfloat> vr(static_cast<std::size_t>(referenceVertexCount) * MESH_VERTEX_FLOATS);
    std::vector<jint> ir(static_cast<std::size_t>(referenceIndexCount));
    env->GetFloatArrayRegion(referenceVertices, 0, static_cast<jsize>(vr.size()), vr.data());
    env->GetIntArrayRegion(referenceIndices, 0, referenceIndexCount, ir.data());
    if (env->ExceptionCheck()) return JNI_FALSE;

    Mesh reference;
    reference.positions.resize(static_cast<std::size_t>(referenceVertexCount) * 3);
    reference.normals.resize(static_cast<std::size_t>(referenceVertexCount) * 3);
    reference.colors.resize(static_cast<std::size_t>(referenceVertexCount) * 3);
    for (int i = 0; i < referenceVertexCount; ++i) {
        const float* v = vr.data() + static_cast<std::size_t>(i) * MESH_VERTEX_FLOATS;
        for (int k = 0; k < 3; ++k) {
            reference.positions[static_cast<std::size_t>(i)*3+k] = v[k];
            reference.normals[static_cast<std::size_t>(i)*3+k] = v[3+k];
            reference.colors[static_cast<std::size_t>(i)*3+k] = v[6+k];
        }
    }
    reference.indices.resize(static_cast<std::size_t>(referenceIndexCount));
    for (int i = 0; i < referenceIndexCount; ++i) {
        const int idx = ir[static_cast<std::size_t>(i)];
        if (idx < 0 || idx >= referenceVertexCount) return JNI_FALSE;
        reference.indices[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(idx);
    }

    Mesh moving;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        moving = meshEngine.mesh();
    }
    if (moving.triangleCount() < 100) return JNI_FALSE;

    Mesh merged;
    std::array<float,16> T{};
    MeshRegistrationStats stats;
    const bool ok = alignAndMergeMeshes(
        reference, moving, &merged, &T, &stats,
        preferredYawDeg >= 0 ? static_cast<int>(preferredYawDeg) : -1);

    jfloat stat[10] = {
        ok ? 1.f : 0.f,
        stats.rmseMeters,
        stats.overlap,
        static_cast<float>(stats.inliers),
        static_cast<float>(stats.iterations),
        static_cast<float>(stats.yawHypothesisDeg),
        static_cast<float>(stats.referenceVertices),
        static_cast<float>(stats.movingVertices),
        static_cast<float>(stats.outputVertices),
        static_cast<float>(stats.outputTriangles)
    };
    env->SetFloatArrayRegion(outStats, 0, 10, stat);
    env->SetFloatArrayRegion(outTransform, 0, 16, T.data());
    if (!ok) return JNI_FALSE;

    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        meshEngine.replaceExportMesh(std::move(merged));
        meshStats.outVertices = meshEngine.mesh().vertexCount();
        meshStats.outTriangles = meshEngine.mesh().triangleCount();
        meshStats.ok = true;
        meshStats.note = "R8 cumulative ICP merge";
        meshDirty = false;
        ++meshBuilds;
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeResetMesh(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    meshEngine = MeshEngine();
    meshHasHardEdges = false;
    hardSurfaceStats = {};
    meshStats = MeshBuildStats{};
    meshDirty = true;
}

/** 导出 glTF 2.0 二进制 GLB（vertex color）。返回是否成功。 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeExportGlb(JNIEnv* e, jobject, jstring path) {
    if (path == nullptr) {
        return JNI_FALSE;
    }
    const char* p = e->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return JNI_FALSE;
    }
    bool ok = false;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        GlbExportStats gs;
        ok = exportGlb(meshEngine.mesh(), std::string(p), "MobileScan3D_scan", &gs);
        LOGI("nativeExportGlb ok=%d verts=%zu tris=%zu json=%zu bin=%zu bytes=%zu color=%d note=%s",
             ok ? 1 : 0, gs.vertices, gs.triangles, gs.jsonBytes, gs.binBytes,
             gs.fileBytes, gs.hasVertexColor ? 1 : 0, gs.note.c_str());
    }
    e->ReleaseStringUTFChars(path, p);
    return ok ? JNI_TRUE : JNI_FALSE;
}

// ===========================================================================
//  V0.6：HQ 多视角纹理 -> 自包含 textured GLB
// ===========================================================================
//
// 数据流（只在「停扫导出」时跑一次，每帧路径零开销）：
//
//   meshEngine.mesh()（本工程 SoA 布局）
//     -> toAosMesh()                 边界转换（见 v06/aos_mesh.h 的说明）
//     -> MeshPostProcessor::run()    亚毫米 weld / 去漂浮分量 / ear-clipping 补小洞 / QEM
//     -> UvUnwrapper::unwrap()       xatlas（已 vendor）或 triangle-atlas 回退
//     -> TextureBaker::bake()        Z-buffer 可见性 + 视角评分 + 每 texel 前 3 视角混合
//     -> cv::imencode(".jpg")        atlas 压成 JPEG
//     -> TexturedGlbExporter::write()  内嵌 JPEG 的 glTF 2.0 二进制
//
// 任何一步失败都返回 JNI_FALSE，调用方（MainActivity）回退到 V0.5 的
// vertex-color GLB —— 几何已经算好，不因为纹理失败就一起判废。
//
// 注意：**本工程不在 nativeBuildMesh 里跑 MeshPostProcessor**。nativeBuildMesh
// 用的 MeshEngine 已自带去小分量 / 去孤立面 / Taubin / QEM 是一条已验收的链路，
// 不动它；V0.6 的清理只作用在导出资产这份 AoS 副本上。因此屏幕上 AR overlay
// 显示的仍是 V0.5 的 vertex-color 网格，而 GLB 是清理过、带纹理的资产。
// ===========================================================================

/** 本工程 SoA Mesh -> V0.6 模块使用的 AoS Mesh。 */
static AosMesh toAosMesh(const Mesh& m) {
    AosMesh out;
    const std::size_t n = m.vertexCount();
    if (n == 0 || m.indices.size() < 3) {
        return out;
    }
    out.vertices.resize(n);
    const bool haveNormals = (m.normals.size() >= n * 3);
    const bool haveColors = (m.colors.size() >= n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        AosVertex& v = out.vertices[i];
        v.px = m.positions[i * 3 + 0];
        v.py = m.positions[i * 3 + 1];
        v.pz = m.positions[i * 3 + 2];
        if (haveNormals) {
            v.nx = m.normals[i * 3 + 0];
            v.ny = m.normals[i * 3 + 1];
            v.nz = m.normals[i * 3 + 2];
        }
        if (haveColors) {
            v.r = m.colors[i * 3 + 0];
            v.g = m.colors[i * 3 + 1];
            v.b = m.colors[i * 3 + 2];
        }
    }
    out.indices = m.indices;
    return out;
}

/** 开始新一轮扫描时清空 HQ 关键帧登记表。 */
extern "C" JNIEXPORT void JNICALL
Java_com_mobilescan3d_NativeBridge_nativeClearTextureKeyframes(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(gStateMutex);
    gTextureKeyframes.clear();
    textureBakeStats = TextureBakeStats{};
    uvUnwrapStats = UvUnwrapStats{};
}

/**
 * 登记一张 HQ still 作为纹理候选视角。
 *
 * pose12 = camera->world 的 12 个 float（R 行主序 9 个 + t 3 个），由 Kotlin
 * 侧用 nativeGetRenderPoseAt(frameSensorTs) 拿到 —— 也就是**拍摄那一刻**的
 * VINS 位姿，不是当前帧位姿。
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeRegisterTextureKeyframe(
        JNIEnv* env, jobject,
        jstring path, jint width, jint height,
        jfloat fx, jfloat fy, jfloat cx, jfloat cy,
        jfloatArray pose12, jfloat quality, jlong timestampNs) {
    if (!path || !pose12 || width <= 0 || height <= 0 ||
        fx <= 1.0f || fy <= 1.0f ||
        env->GetArrayLength(pose12) < 12) {
        return JNI_FALSE;
    }

    const char* p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return JNI_FALSE;
    }
    std::string imagePath(p);
    env->ReleaseStringUTFChars(path, p);

    {
        std::ifstream test(imagePath, std::ios::binary);
        if (!test.good()) {
            return JNI_FALSE;
        }
    }

    jfloat values[12];
    env->GetFloatArrayRegion(pose12, 0, 12, values);

    TextureKeyframe k;
    k.imagePath = std::move(imagePath);
    k.width = width;
    k.height = height;
    k.fx = fx;
    k.fy = fy;
    k.cx = cx;
    k.cy = cy;
    for (int i = 0; i < 9; ++i) {
        k.Rwc[i] = values[i];
    }
    k.twc[0] = values[9];
    k.twc[1] = values[10];
    k.twc[2] = values[11];
    k.quality = std::clamp(static_cast<float>(quality), 0.05f, 3.0f);
    k.timestampNs = timestampNs > 0 ? static_cast<std::uint64_t>(timestampNs) : 0u;

    std::lock_guard<std::mutex> lk(gStateMutex);

    // 同一个文件被回调重复投递时去重。
    for (const auto& existing : gTextureKeyframes) {
        if (existing.imagePath == k.imagePath) {
            return JNI_TRUE;
        }
    }
    gTextureKeyframes.push_back(std::move(k));

    // 硬上限：元数据不能无界增长。烘焙时会另按 quality + 视角多样性挑选。
    constexpr std::size_t kRegistryLimit = 96;
    if (gTextureKeyframes.size() > kRegistryLimit) {
        auto worst = std::min_element(
            gTextureKeyframes.begin(), gTextureKeyframes.end(),
            [](const TextureKeyframe& a, const TextureKeyframe& b) {
                return a.quality < b.quality;
            });
        if (worst != gTextureKeyframes.end()) {
            gTextureKeyframes.erase(worst);
        }
    }
    return JNI_TRUE;
}

/** 清理几何 + 展开 UV + 多视角烘焙 + 写出内嵌 JPEG 的 textured GLB。 */
static jboolean nativeBakeTexturedGlbImpl(
        JNIEnv* env, jstring path, jint atlasResolution, jint maxKeyframes) {
    if (path == nullptr) {
        return JNI_FALSE;
    }

    AosMesh meshCopy;
    float sourceVoxelSize = 0.f;
    int sourceQuality = 1;
    bool preserveHardEdges = false;
    std::vector<TextureKeyframe> keyframes;
    uint64_t bakeGeneration = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& src = meshEngine.mesh();
        if (src.empty()) {
            return JNI_FALSE;
        }
        sourceVoxelSize = meshStats.voxelSize;
        sourceQuality = meshQuality;
        preserveHardEdges = meshHasHardEdges;
        meshCopy = toAosMesh(src);
        keyframes = gTextureKeyframes;
        bakeGeneration = gNativeGeneration;
    }
    if (meshCopy.empty() || keyframes.empty()) {
        LOGI("nativeBakeTexturedGlb skip: mesh=%zu keyframes=%zu",
             meshCopy.triangleCount(), keyframes.size());
        return JNI_FALSE;
    }

    const int resolution = std::clamp(static_cast<int>(atlasResolution), 512, 4096);

    // ---- 几何清理：weld / 去漂浮分量 / ear-clipping 补小洞 / QEM ----
    MeshPostProcessOptions cleanup;
    cleanup.weldEpsilon = std::clamp(sourceVoxelSize * 0.08f, 0.00030f, 0.00120f);
    cleanup.minComponentTriangles = sourceQuality >= 2 ? 24 : (sourceQuality <= 0 ? 72 : 48);
    cleanup.minComponentAreaRatio = sourceQuality >= 2 ? 0.0020f : 0.0035f;
    cleanup.maxHoleEdges = sourceQuality >= 2 ? 72 : 56;
    cleanup.maxHoleDiameterMeters = sourceQuality >= 2 ? 0.060f : 0.075f;
    cleanup.maxHoleDiameterBBoxRatio = sourceQuality >= 2 ? 0.10f : 0.12f;
    cleanup.targetTriangles = sourceQuality >= 2 ? 180000 : (sourceQuality <= 0 ? 30000 : 80000);
    cleanup.qemMaxPasses = 10;
    cleanup.qemMaxNormalFlipDeg = sourceQuality >= 2 ? 65.0f : 72.0f;
    cleanup.preserveBoundary = true;

    MeshPostProcessStats cleanupStats;
    // Welding/smoothing a fitted box destroys its independent face normals and may
    // remove its two-triangle components. Keep the fitted geometry exactly as built.
    if (preserveHardEdges) {
        cleanupStats.inputTriangles = cleanupStats.outputTriangles = meshCopy.triangleCount();
    }
    if (!preserveHardEdges && !MeshPostProcessor::run(meshCopy, cleanup, &cleanupStats)) {
        LOGI("nativeBakeTexturedGlb FAILED: mesh cleanup");
        return JNI_FALSE;
    }

    UvMesh uvMesh;
    UvUnwrapStats uvStats;
    if (!UvUnwrapper::unwrap(meshCopy, resolution, 8, uvMesh, &uvStats)) {
        LOGI("nativeBakeTexturedGlb FAILED: uv unwrap");
        return JNI_FALSE;
    }

    TextureBakeOptions options;
    options.atlasResolution = resolution;
    options.maxKeyframes = std::clamp(static_cast<int>(maxKeyframes), 4, 24);
    options.maxBlendFrames = 3;
    options.sourceMaxSide = resolution >= 4096 ? 2200 : (resolution >= 2048 ? 1600 : 1280);
    options.visibilityMaxSide = resolution >= 4096 ? 512 : 384;
    options.gutterDilationPixels = resolution >= 4096 ? 10 : 6;
    options.jpegQuality = resolution >= 4096 ? 94 : 92;

    cv::Mat atlas;
    TextureBakeStats bakeStats;
    if (!TextureBaker::bake(uvMesh, keyframes, options, atlas, &bakeStats)) {
        LOGI("nativeBakeTexturedGlb FAILED: texture bake");
        return JNI_FALSE;
    }

    std::vector<std::uint8_t> jpeg;
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, options.jpegQuality};
    if (!cv::imencode(".jpg", atlas, jpeg, params) || jpeg.empty()) {
        LOGI("nativeBakeTexturedGlb FAILED: jpeg encode");
        return JNI_FALSE;
    }

    const char* p = env->GetStringUTFChars(path, nullptr);
    if (p == nullptr) {
        return JNI_FALSE;
    }
    const bool ok = TexturedGlbExporter::write(p, uvMesh, jpeg);
    env->ReleaseStringUTFChars(path, p);

    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        // A long bake can outlive its Activity/session. Never publish its AR
        // mesh into a newly created scan even if its staging file was written.
        if (bakeGeneration != gNativeGeneration) return JNI_FALSE;
        if (ok) {
            std::lock_guard<std::mutex> alk(gArAssetMutex);
            if (!gArTexturedAsset.set(uvMesh, jpeg)) return JNI_FALSE;
        }
        meshCleanupStats = cleanupStats;
        if (ok) {
            textureBakeStats = bakeStats;
            uvUnwrapStats = uvStats;
        }
    }

    LOGI("nativeBakeTexturedGlb ok=%d cleanTris=%zu->%zu weld=%zu compsRemoved=%zu "
         "holes=%zu(+%zu tris) uv=%s atlas=%dx%d frames=%d/%d cov=%.1f%% jpeg=%zu",
         ok ? 1 : 0,
         cleanupStats.inputTriangles, cleanupStats.outputTriangles,
         cleanupStats.weldedVertices, cleanupStats.removedComponents,
         cleanupStats.filledHoles, cleanupStats.addedHoleTriangles,
         uvStats.usedXatlas ? "xatlas" : "fallback",
         uvStats.atlasWidth, uvStats.atlasHeight,
         bakeStats.usedKeyframes, bakeStats.requestedKeyframes,
         bakeStats.coveragePercent, jpeg.size());
    // V0.13.2 诊断：cov=0.1% 时无法区分「视角本来就该只盖一点」与
    // 「投影系统性失效（位姿错 / 空间不一致）」。把逐三角形命中统计和
    // 每个关键帧的位姿/内参打出来，下一轮日志直接定位。
    LOGI("bakeDiag texTris=%zu fbTris=%zu painted=%zu hq=%zu kf=%d",
         bakeStats.texturedTriangles, bakeStats.fallbackTriangles,
         bakeStats.paintedTexels, bakeStats.hqTexels,
         bakeStats.loadedKeyframes);
    // V0.13.2 诊断：texTris 远低于在帧率预期，逐关卡统计拒绝原因。
    // [0]=project [1]=border [2]=depth [3]=facing [4]=area [5]=pass
    for (std::size_t ri = 0; ri < bakeStats.frameRejects.size(); ++ri) {
        const auto& r = bakeStats.frameRejects[ri];
        LOGI("bakeDiag reject kf=%zu project=%zu border=%zu depth=%zu "
             "facing=%zu area=%zu pass=%zu",
             ri, r[0], r[1], r[2], r[3], r[4], r[5]);
    }
    {
        // Use the captured bake inputs; diagnostics must not block live pose queries.
        for (const auto& kf : keyframes) {
            // V0.13.2 诊断：把网格顶点按烘焙同一套投影公式投进关键帧画幅，
            // 统计在帧率。若在帧率~0 而照片里物体明明可见，则位姿/约定有错。
            int inFront = 0;
            int inFrame = 0;
            const int total = static_cast<int>(meshCopy.vertices.size());
            float bmin[3] = {0,0,0}, bmax[3] = {0,0,0};
            for (int axis = 0; axis < 3; ++axis) { bmin[axis] = 1e9f; bmax[axis] = -1e9f; }
            for (const auto& v : meshCopy.vertices) {
                const float p[3] = {v.px, v.py, v.pz};
                for (int axis = 0; axis < 3; ++axis) {
                    if (p[axis] < bmin[axis]) bmin[axis] = p[axis];
                    if (p[axis] > bmax[axis]) bmax[axis] = p[axis];
                }
                const float d[3] = {v.px - kf.twc[0],
                                    v.py - kf.twc[1],
                                    v.pz - kf.twc[2]};
                const float pcx = kf.Rwc[0]*d[0] + kf.Rwc[3]*d[1] + kf.Rwc[6]*d[2];
                const float pcy = kf.Rwc[1]*d[0] + kf.Rwc[4]*d[1] + kf.Rwc[7]*d[2];
                const float pcz = kf.Rwc[2]*d[0] + kf.Rwc[5]*d[1] + kf.Rwc[8]*d[2];
                if (pcz <= 0.02f) continue;
                ++inFront;
                const float u = kf.fx * pcx / pcz + kf.cx;
                const float vv = kf.fy * pcy / pcz + kf.cy;
                if (u >= 0.f && u < static_cast<float>(kf.width) &&
                    vv >= 0.f && vv < static_cast<float>(kf.height)) {
                    ++inFrame;
                }
            }
            LOGI("bakeDiag kf %dx%d fx=%.0f fy=%.0f t=(%.3f,%.3f,%.3f) "
                 "fwd=(%.3f,%.3f,%.3f) q=%.2f "
                 "verts=%d inFront=%d inFrame=%d "
                 "bbox=[(%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)]",
                 kf.width, kf.height, kf.fx, kf.fy,
                 kf.twc[0], kf.twc[1], kf.twc[2],
                 kf.Rwc[2], kf.Rwc[5], kf.Rwc[8],
                 kf.quality,
                 total, inFront, inFrame,
                 bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]);
        }
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

// OpenCV/allocation failures must return to Kotlin's vertex-colour fallback;
// a Kotlin catch cannot intercept a C++ exception escaping the JNI boundary.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeBakeTexturedGlb(
        JNIEnv* env, jobject, jstring path, jint atlasResolution, jint maxKeyframes) {
    try { return nativeBakeTexturedGlbImpl(env, path, atlasResolution, maxKeyframes); }
    catch (const std::exception& ex) { LOGI("Texture bake failed: %s", ex.what()); }
    catch (...) { LOGI("Texture bake failed: unknown exception"); }
    return JNI_FALSE;
}

/**
 * 纹理统计（9 槽）。
 *   0 已登记关键帧数 / 1 实际加载 / 2 实际使用 / 3 atlasW / 4 atlasH
 *   5 有纹理的三角形 / 6 回退 vertex color 的三角形 / 7 覆盖率×10 / 8 是否 xatlas
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetTextureStats(
        JNIEnv* env, jobject, jintArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 9) {
        return JNI_FALSE;
    }
    jint values[9] = {0};
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        values[0] = static_cast<jint>(std::min<std::size_t>(
            gTextureKeyframes.size(), static_cast<std::size_t>(std::numeric_limits<jint>::max())));
        values[1] = textureBakeStats.loadedKeyframes;
        values[2] = textureBakeStats.usedKeyframes;
        values[3] = uvUnwrapStats.atlasWidth;
        values[4] = uvUnwrapStats.atlasHeight;
        values[5] = static_cast<jint>(std::min<std::size_t>(
            textureBakeStats.texturedTriangles, static_cast<std::size_t>(std::numeric_limits<jint>::max())));
        values[6] = static_cast<jint>(std::min<std::size_t>(
            textureBakeStats.fallbackTriangles, static_cast<std::size_t>(std::numeric_limits<jint>::max())));
        values[7] = static_cast<jint>(std::clamp(textureBakeStats.coveragePercent * 10.0f, 0.0f, 1000.0f));
        values[8] = uvUnwrapStats.usedXatlas ? 1 : 0;
    }
    env->SetIntArrayRegion(out, 0, 9, values);
    return JNI_TRUE;
}

/**
 * 导出前几何清理统计（10 槽）。
 *   0 输入三角形 / 1 输出三角形 / 2 weld 后顶点 / 3 去掉的分量
 *   4 去掉的分量三角形 / 5 边界环 / 6 补的洞 / 7 补洞新增三角形
 *   8 QEM 塌缩边 / 9 输出顶点
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshCleanupStats(
        JNIEnv* env, jobject, jintArray out) {
    if (out == nullptr || env->GetArrayLength(out) < 10) {
        return JNI_FALSE;
    }
    jint values[10] = {0};
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        values[0] = static_cast<jint>(meshCleanupStats.inputTriangles);
        values[1] = static_cast<jint>(meshCleanupStats.outputTriangles);
        values[2] = static_cast<jint>(meshCleanupStats.weldedVertices);
        values[3] = static_cast<jint>(meshCleanupStats.removedComponents);
        values[4] = static_cast<jint>(meshCleanupStats.removedComponentTriangles);
        values[5] = static_cast<jint>(meshCleanupStats.boundaryLoops);
        values[6] = static_cast<jint>(meshCleanupStats.filledHoles);
        values[7] = static_cast<jint>(meshCleanupStats.addedHoleTriangles);
        values[8] = static_cast<jint>(meshCleanupStats.qemCollapsedEdges);
        values[9] = static_cast<jint>(meshCleanupStats.outputVertices);
    }
    env->SetIntArrayRegion(out, 0, 10, values);
    return JNI_TRUE;
}

// ============================================================================
//  Round 6: TSDF checkpoint save/load + per-vertex observation strength
//  (merged from R5_R6 cumulative incremental patch; native_engine.cpp
//   three-way merge — these are pure additions at file end, vc175/vc176
//   calibration/epoch fixes in the middle are preserved)
// ============================================================================

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeSaveTsdfCheckpoint(
        JNIEnv* env, jobject, jstring scenePath, jstring targetPath) {
    if (!scenePath || !targetPath) return JNI_FALSE;
    const char* scene = env->GetStringUTFChars(scenePath, nullptr);
    const char* target = env->GetStringUTFChars(targetPath, nullptr);
    if (!scene || !target) {
        if (scene) env->ReleaseStringUTFChars(scenePath, scene);
        if (target) env->ReleaseStringUTFChars(targetPath, target);
        return JNI_FALSE;
    }
    bool ok = false;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        ok = tsdf.saveCheckpoint(scene) && targetTsdf.saveCheckpoint(target);
    }
    env->ReleaseStringUTFChars(scenePath, scene);
    env->ReleaseStringUTFChars(targetPath, target);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeLoadTsdfCheckpoint(
        JNIEnv* env, jobject, jstring scenePath, jstring targetPath) {
    if (!scenePath || !targetPath) return JNI_FALSE;
    const char* scene = env->GetStringUTFChars(scenePath, nullptr);
    const char* target = env->GetStringUTFChars(targetPath, nullptr);
    if (!scene || !target) {
        if (scene) env->ReleaseStringUTFChars(scenePath, scene);
        if (target) env->ReleaseStringUTFChars(targetPath, target);
        return JNI_FALSE;
    }

    TsdfEngine sceneLoaded;
    TsdfEngine targetLoaded;
    const bool filesOk =
        sceneLoaded.loadCheckpoint(scene) &&
        targetLoaded.loadCheckpoint(target);

    bool ok = false;
    if (filesOk) {
        std::lock_guard<std::mutex> lk(gStateMutex);
        tsdf = std::move(sceneLoaded);
        targetTsdf = std::move(targetLoaded);
        targetLockEverArmed = targetTsdf.voxels() > 0;
        resetMeshPipeline();
        meshDirty = true;
        ok = tsdf.voxels() > 0 || targetTsdf.voxels() > 0;
    }

    env->ReleaseStringUTFChars(scenePath, scene);
    env->ReleaseStringUTFChars(targetPath, target);
    return ok ? JNI_TRUE : JNI_FALSE;
}

/**
 * Round 6: per-mesh-vertex TSDF observation strength.
 *
 * Output is de-quantized TSDF weight (roughly accumulated confidence, 0..64),
 * sampled in a 3x3x3 neighborhood around the final mesh vertex. This is tied to
 * the actual reconstructed surface, unlike viewpoint-orbit coverage.
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_mobilescan3d_NativeBridge_nativeGetMeshObservationWeights(
        JNIEnv* env, jobject, jfloatArray out, jint maxVertices) {
    if (!out || maxVertices <= 0) return 0;
    std::vector<jfloat> values;
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(gStateMutex);
        const Mesh& m = meshEngine.mesh();
        const size_t nv = m.vertexCount();
        if (nv == 0 || m.positions.size() < nv * 3) return 0;

        const jsize cap = env->GetArrayLength(out);
        n = std::min(
            std::min(static_cast<size_t>(maxVertices), nv),
            static_cast<size_t>(std::max<jsize>(0, cap)));
        if (n == 0) return 0;

        const TsdfEngine& src = (targetTsdf.voxels() > 0) ? targetTsdf : tsdf;
        const float voxel = src.voxelSize();
        if (!(voxel > 1e-5f) || !std::isfinite(voxel)) return 0;

        values.resize(n, 0.f);
        for (size_t i = 0; i < n; ++i) {
            const float x = m.positions[i * 3 + 0];
            const float y = m.positions[i * 3 + 1];
            const float z = m.positions[i * 3 + 2];
            const int vx = static_cast<int>(std::lround(x / voxel));
            const int vy = static_cast<int>(std::lround(y / voxel));
            const int vz = static_cast<int>(std::lround(z / voxel));

            uint16_t best = 0;
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        best = std::max(best, src.weightAt(vx + dx, vy + dy, vz + dz));

            values[i] = static_cast<jfloat>(best) / kTsdfWeightScale;
        }
    }
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(n), values.data());
    return static_cast<jint>(n);
}
