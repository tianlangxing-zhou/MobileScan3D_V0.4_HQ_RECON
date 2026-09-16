package com.mobilescan3d
object NativeBridge {
    init { System.loadLibrary("mobilescan3d") }

    /**
     * JNI 输出数组的槽数 —— **必须与 C++ 侧常量一致**。
     *
     * 把数字集中到一处，是因为「两边各写一个槽数」已经真实地造成过一次崩溃：
     * 深度诊断从 4 槽扩到 7 槽时，native 侧 (`kDepthDiagSlots`) 和实时采样
     * 都改了，`generateReport()` 里却还留着 `FloatArray(4)`。native 发现长度
     * 不足会安全返回 0，但 Kotlin 继续访问 `dd[4]`，于是点「导出反馈报告」
     * 直接 ArrayIndexOutOfBoundsException 退出。
     *
     * 新增诊断字段时只改这里 + 对应的 native 常量，调用方一律用这些常量建数组。
     */
    const val DEPTH_DIAGNOSTIC_SLOTS = 7
    /**
     * 14 -> 16：新增 PresenceGate 结论与外观后端状态。
     *
     *   0  state            1  x0        2  y0        3  x1        4  y1
     *   5  confidence       6  medianDepth   7  roiSharpness
     *   8  trackedPoints    9  inlierRatio
     *   10 visibleFraction  11 centerXNorm 12 centerYNorm 13 edgeLostFrames
     *   14 presenceValid    15 appearanceAvailable
     *
     * 为什么必须有 11/12（中心点）：目标整块滑出画面时 1..4 的 bbox 会被
     * 裁剪成空矩形，UI 连「目标往哪个方向去了」都无从判断，只能干等。
     * 中心点即使跑出 [0,1] 也仍然保留方向信息。
     *
     * 为什么必须有 14（presenceValid）：**box 还在画面里 != 物体还在**。
     * 真实目标离开后，tracker 常常在背景纹理上继续输出一个「看起来正常」的框，
     * 所以绿框的可见性必须由 presenceValid 决定，而不是 bbox 的几何位置。
     */
    const val TARGET_STATE_SLOTS = 16
    /** [nativeGetTargetState] 输出的下标，避免调用方再手写数字。 */
    const val TARGET_STATE_INDEX_STATE = 0
    const val TARGET_STATE_INDEX_VISIBLE_FRACTION = 10
    const val TARGET_STATE_INDEX_CENTER_X = 11
    const val TARGET_STATE_INDEX_CENTER_Y = 12
    const val TARGET_STATE_INDEX_EDGE_LOST_FRAMES = 13
    /** PresenceGate 结论：1 = 目标确实还在（绿框才允许画） */
    const val TARGET_STATE_INDEX_PRESENCE_VALID = 14
    /** 外观后端是否可用（0/1，仅诊断） */
    const val TARGET_STATE_INDEX_APPEARANCE_OK = 15

    /**
     * 与 C++ `TargetState` 枚举逐项对应（顺序即数值）。
     * 新增了 REACQUIRING = 5：目标短暂出屏时的中间态，
     * 约 1.5s 内由 NanoTrack 自己找回来，超时才转 LOST。
     */
    const val TARGET_STATE_OFF = 0
    const val TARGET_STATE_ARMED = 1
    const val TARGET_STATE_ACQUIRING = 2
    const val TARGET_STATE_TRACKING = 3
    const val TARGET_STATE_LOST = 4
    const val TARGET_STATE_REACQUIRING = 5

    const val RENDER_POSE_SLOTS = 12

    /** 点云/调试层每个点 6 个浮点：x, y, z, r, g, b */
    const val POINT_SLOTS = 6
    /** AR 累计点云一次最多取多少点（native 侧写 out + written*6，written ≤ maxPoints） */
    const val AR_MAX_POINTS = 80000
    /**
     * AR 当前帧 target depth 调试层最多多少点。
     * **必须 <= native 的 kTargetDebugMaxPoints(=2000)**：Kotlin 按这个容量建数组，
     * native 写 n*6 个浮点、n <= 2000，两边才能对上。
     */
    const val AR_TARGET_DEBUG_MAX_POINTS = 2000

    /**
     * AR 累计点云的 hits 过滤门限。
     *
     *   RAW       = 1（调试用，含大量一次性点）
     *   CONFIRMED = 2（AR 默认：至少被两帧看到过）
     *   STABLE    = 3（已经合并过，最干净）
     * 旧实现从不过滤，hits=1 的一次性点也以实心 3px 画满屏幕。
     */
    const val AR_MIN_HITS_CONFIRMED = 2
    const val AR_MIN_HITS_STABLE = 3

    // ------------------------------------------------------------------
    //  网格 / GLB / 深度标定
    // ------------------------------------------------------------------

    /**
     * 网格顶点交错布局：`x,y,z, nx,ny,nz, r,g,b` —— 9 个 float / 顶点。
     *
     * **必须与 native 的 `MESH_VERTEX_FLOATS` 完全一致**：Kotlin 侧按它
     * 分配 `FloatArray(vertexCount * MESH_VERTEX_FLOATS)`，native 按它写
     * `out[i * 9 + k]`，两边差一个数字就是越界或截断。
     */
    const val MESH_VERTEX_FLOATS = 9

    /**
     * 网格质量档位，与 native `MeshOptions::quality` 对应。
     *
     * 三角形预算：预览 2 万 / 常规 10 万 / HQ 40 万。
     * AR 预览永远用 PREVIEW —— 十多万个三角形在手机上跑 30FPS 没有意义，
     * 反而把 GPU 时间全吃掉。
     */
    const val MESH_QUALITY_PREVIEW = 0
    const val MESH_QUALITY_NORMAL = 1
    const val MESH_QUALITY_HQ = 2

    /**
     * [nativeGetDepthCalibration] 的输出槽数。
     *
     *   0  scale              1  shift            2  confidence
     *   3  samples            4  valid            5  inverseModel
     *   6  enabled            7  acceptedFrames   8  rejectedFrames
     *   9  temporalRatio      10 temporalRejects  11 usable
     *
     * `inverseModel = 1` 时 `z = 1 / (scale * d + shift)`，否则 `z = scale*d + shift`。
     * `usable` 才是「可以拿它当米制用」的判据 —— `valid` 只是「这一帧拟合成功」。
     */
    const val DEPTH_CALIBRATION_SLOTS = 12
    const val CALIB_INDEX_SCALE = 0
    const val CALIB_INDEX_SHIFT = 1
    const val CALIB_INDEX_CONFIDENCE = 2
    const val CALIB_INDEX_SAMPLES = 3
    const val CALIB_INDEX_VALID = 4
    const val CALIB_INDEX_INVERSE_MODEL = 5
    const val CALIB_INDEX_ENABLED = 6
    const val CALIB_INDEX_ACCEPTED_FRAMES = 7
    const val CALIB_INDEX_REJECTED_FRAMES = 8
    const val CALIB_INDEX_TEMPORAL_RATIO = 9
    const val CALIB_INDEX_TEMPORAL_REJECTS = 10
    const val CALIB_INDEX_USABLE = 11

    /**
     * [nativeGetMeshStats] 的输出槽数（全整型）。
     *
     *   0  outVertices      1  outTriangles    2  rawVertices     3  rawTriangles
     *   4  componentsBefore 5  componentsRemoved 6 trianglesRemovedRaw
     *   7  trianglesRemovedComp 8 blocksScanned 9 decimated       10 ok
     *   11 quality          12 totalMs         13 builds          14 voxelSizeUm
     *   15 smoothIterations
     */
    const val MESH_STATS_SLOTS = 16
    const val MESH_STATS_INDEX_VERTICES = 0
    const val MESH_STATS_INDEX_TRIANGLES = 1
    const val MESH_STATS_INDEX_RAW_TRIANGLES = 3
    const val MESH_STATS_INDEX_COMPONENTS_REMOVED = 5
    const val MESH_STATS_INDEX_BLOCKS_SCANNED = 8
    const val MESH_STATS_INDEX_DECIMATED = 9
    const val MESH_STATS_INDEX_OK = 10
    const val MESH_STATS_INDEX_QUALITY = 11
    const val MESH_STATS_INDEX_TOTAL_MS = 12
    const val MESH_STATS_INDEX_BUILDS = 13
    const val MESH_STATS_INDEX_VOXEL_SIZE_UM = 14

    external fun nativeCreate(w:Int,h:Int,fx:Float,fy:Float,cx:Float,cy:Float):Boolean
    external fun nativeDestroy()
    external fun nativeOnImu(t:Long,ax:Float,ay:Float,az:Float,gx:Float,gy:Float,gz:Float)
    external fun nativeOnCameraFrame(y:ByteArray,u:ByteArray,v:ByteArray,w:Int,h:Int,rowStride:Int,uRowStride:Int,uPixelStride:Int,frameTimestampNs:Long,vinsTimestampNs:Long)
    external fun nativeOnDepthMap(depth:FloatArray,w:Int,h:Int,confidence:Float,timestamp:Long)
    external fun nativeExportPly(path:String):Boolean
    external fun nativeSetMode(m:Int)
    external fun nativeGetStats():String
    external fun nativeGetGuidance():String
    external fun nativeGetGaussians(out: FloatArray, maxPoints: Int, minHits: Int): Int
    external fun nativeGetTargetDepthDebug(out: FloatArray, maxPoints: Int): Int
    external fun nativeSetTargetDebugEnabled(enabled: Boolean)
    external fun nativeVinsInitialized(): Boolean
    external fun nativeGetHudMetrics(): String
    external fun nativeGetPointCount(): Int
    external fun nativeSelectTarget(u: Float, v: Float): Boolean
    /**
     * 用户手指拖出的矩形（相机归一化坐标，允许任意方向）。
     *
     * 对比单点 [nativeSelectTarget]：点按只能回退到固定 240x240 的方块，
     * 点一个细长瓶子时会把周围背景一起锁进 ROI，KLT 追的是
     * 「物体 + 墙 + 桌子」的混合纹理，绿框自然贴不住目标。
     */
    external fun nativeSelectTargetRect(
        x0: Float, y0: Float, x1: Float, y1: Float
    ): Boolean
    external fun nativeClearTarget()
    external fun nativeSetObjectLockEnabled(enabled: Boolean)
    external fun nativeGetTargetState(out: FloatArray): Int
    external fun nativeGetDepthDiagnostics(out: FloatArray): Int
    external fun nativeGetTargetDiagnostics(): String
    external fun nativeGetRenderPose(out: FloatArray): Boolean
    /**
     * 按 SENSOR_TIMESTAMP 查那一时刻的 VINS 位姿（AR 跟随用的就是这个）。
     *
     * 旧的 [nativeGetRenderPose] 给的是「此刻最新」的 pose，而屏幕上显示的
     * Camera frame 有自己的时间戳，两者不同时刻 —— 静止时看不出来，
     * 一转起来点云就漂。时间戳超出 80ms 会返回 false，调用方应回退。
     */
    external fun nativeGetRenderPoseAt(timestampNs: Long, out: FloatArray): Boolean
    external fun nativeConfigureTrackerModels(backbonePath: String, headPath: String): Boolean
    external fun nativeFuseBurst(inputPaths: Array<String>, outputPath: String, exposureMs: Float, iso: Int, minAcceptedFrames: Int, stats: FloatArray, frameStats: FloatArray): Boolean
    external fun nativeVinsInit(fx: Float, fy: Float, cx: Float, cy: Float, w: Int, h: Int, ric: FloatArray, tic: FloatArray, accN: Float, accW: Float, gyrN: Float, gyrW: Float)
    external fun nativeVinsImu(t: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float)
    external fun nativeVinsImage(t: Double, gray: ByteArray, w: Int, h: Int, stride: Int)
    external fun nativeVinsGetPose(out: FloatArray): Boolean

    // ------------------------------------------------------------------
    //  真 TSDF / Mesh / GLB / 深度标定
    // ------------------------------------------------------------------

    /**
     * 体素边长（米）。`sceneMeters` 作用于全场景 TSDF，`targetMeters` 作用于
     * 独立的目标模型 TSDF。
     *
     * 注意**这里填的不是真实米制尺度**：融合用的位姿 `Rwc/twc` 是 VINS world，
     * 而深度是按会话 min/max 映射出来的相对尺度，两者相差约一个常数因子。
     * 所以体素大小与截断距离是按「VINS world 的可见表面尺度」选的，
     * 真实的米制对齐由 [nativeGetDepthCalibration] 那套标定负责。
     */
    external fun nativeSetVoxelSizes(sceneMeters: Float, targetMeters: Float)

    /**
     * 体素块预算（稀疏体素块哈希的容量上限）。
     * 一个块 = 8x8x8 体素（3072 字节）。超过预算后新的块不再分配，
     * 融合会整帧停止而不是静默丢一部分，避免出现「半边模型」。
     */
    external fun nativeSetVoxelBudget(sceneBlocks: Int, targetBlocks: Int)

    /**
     * 开关 VINS 稀疏深度鲁棒标定（MAD 剔除 + Huber IRLS + EMA）。
     * 关掉会退回旧的「单个 median 比值」尺度。
     */
    external fun nativeSetDepthCalibrationEnabled(enabled: Boolean)

    /** 读深度标定状态（[DEPTH_CALIBRATION_SLOTS] 槽）。返回写入的槽数。 */
    external fun nativeGetDepthCalibration(out: FloatArray): Int

    /**
     * 从当前体素场构建三角网格。**这是一次可能耗时几百毫秒的同步操作**，
     * 必须在后台线程调用（[com.mobilescan3d.export.ExportManager] 已经这么做了）。
     *
     * 有目标模型时优先用目标体素场（与「有目标就只导目标」的口径一致）。
     */
    external fun nativeBuildMesh(quality: Int): Boolean

    /** 网格统计（[MESH_STATS_SLOTS] 槽）。 */
    external fun nativeGetMeshStats(): IntArray

    external fun nativeGetMeshVertexCount(): Int
    external fun nativeGetMeshIndexCount(): Int

    /**
     * 拉顶点：交错 9 float（[MESH_VERTEX_FLOATS]），`out` 至少要
     * `maxVertices * MESH_VERTEX_FLOATS` 长。返回实际写入的顶点数。
     */
    external fun nativeGetMeshVertices(out: FloatArray, maxVertices: Int): Int

    /** 拉索引（三角形，每 3 个一组）。返回实际写入的索引数。 */
    external fun nativeGetMeshIndices(out: IntArray, maxIndices: Int): Int

    external fun nativeResetMesh()

    /** 导出 glTF 2.0 二进制（GLB），逐顶点颜色。返回是否成功。 */
    external fun nativeExportGlb(path: String): Boolean
}
