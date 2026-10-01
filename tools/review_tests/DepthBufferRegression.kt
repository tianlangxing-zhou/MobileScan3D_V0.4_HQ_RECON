import com.mobilescan3d.depth.DepthPreprocessor
import kotlin.math.abs
fun main() {
    val w=19;val h=13;val ys=24;val us=24;val ps=2;val n=8
    val y=ByteArray((h-1)*ys+w){((it*37)%256).toByte()}
    val u=ByteArray(((h-1)/2)*us+((w-1)/2)*ps+1){((it*13)%256).toByte()}
    val v=ByteArray(u.size){((it*71)%256).toByte()}
    val pre=DepthPreprocessor(n)
    repeat(2) {
        pre.fill(y,u,v,w,h,ys,us,ps)
        check(pre.tensor.position()==0)
        for(oy in 0 until n)for(ox in 0 until n){
            val sx=ox*w/n;val sy=oy*h/n
            val c=(y[sy*ys+sx].toInt() and 255)-16
            val d=(u[sy/2*us+sx/2*ps].toInt() and 255)-128
            val e=(v[sy/2*us+sx/2*ps].toInt() and 255)-128
            val rgb=intArrayOf(((298*c+409*e+128) shr 8).coerceIn(0,255),((298*c-100*d-208*e+128) shr 8).coerceIn(0,255),((298*c+516*d+128) shr 8).coerceIn(0,255))
            val mean=floatArrayOf(123.675f,116.28f,103.53f);val std=floatArrayOf(58.395f,57.12f,57.375f)
            for(i in 0..2)check(abs(pre.tensor.float-(rgb[i]-mean[i])/std[i])<1e-7f)
        }
    }
    check(runCatching{pre.fill(y.copyOf(4),u,v,w,h,ys,us,ps)}.isFailure)
    check(runCatching{DepthPreprocessor(0)}.isFailure)
    println("PASS buffer: VC160 numeric equivalence, odd dimensions, padded strides, pixelStride=2, buffer reuse, truncated-plane rejection")
}
