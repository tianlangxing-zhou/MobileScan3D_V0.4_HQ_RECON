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
    const val TARGET_STATE_SLOTS = 10
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
}
