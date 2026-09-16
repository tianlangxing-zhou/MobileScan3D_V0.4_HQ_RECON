package com.mobilescan3d
object NativeBridge {
    init { System.loadLibrary("mobilescan3d") }
    external fun nativeCreate(w:Int,h:Int,fx:Float,fy:Float,cx:Float,cy:Float):Boolean
    external fun nativeDestroy()
    external fun nativeOnImu(t:Long,ax:Float,ay:Float,az:Float,gx:Float,gy:Float,gz:Float)
    external fun nativeOnCameraFrame(y:ByteArray,u:ByteArray,v:ByteArray,w:Int,h:Int,rowStride:Int,uRowStride:Int,uPixelStride:Int,frameTimestampNs:Long,vinsTimestampNs:Long)
    external fun nativeOnDepthMap(depth:FloatArray,w:Int,h:Int,confidence:Float,timestamp:Long)
    external fun nativeExportPly(path:String):Boolean
    external fun nativeSetMode(m:Int)
    external fun nativeGetStats():String
    external fun nativeGetGuidance():String
    external fun nativeGetGaussians(out: FloatArray, maxPoints: Int): Int
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
    external fun nativeConfigureTrackerModels(backbonePath: String, headPath: String): Boolean
    external fun nativeFuseBurst(inputPaths: Array<String>, outputPath: String, exposureMs: Float, iso: Int, stats: FloatArray): Boolean
    external fun nativeVinsInit(fx: Float, fy: Float, cx: Float, cy: Float, w: Int, h: Int, ric: FloatArray, tic: FloatArray, accN: Float, accW: Float, gyrN: Float, gyrW: Float)
    external fun nativeVinsImu(t: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float)
    external fun nativeVinsImage(t: Double, gray: ByteArray, w: Int, h: Int, stride: Int)
    external fun nativeVinsGetPose(out: FloatArray): Boolean
}
