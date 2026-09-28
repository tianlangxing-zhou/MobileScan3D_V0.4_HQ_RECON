import com.mobilescan3d.ScanCoordinates
import kotlin.math.abs
import kotlin.math.PI

private fun close(a: Float, b: Float) {
    check(abs(a - b) < 2e-5f) { "$a != $b" }
}
private fun map(a: FloatArray, u: Float, v: Float) =
    floatArrayOf(a[0]*u+a[1]*v+a[2], a[3]*u+a[4]*v+a[5])
private fun inverse(a: FloatArray): FloatArray {
    val d = a[0]*a[4]-a[1]*a[3]
    return floatArrayOf(a[4]/d,-a[1]/d,(a[1]*a[5]-a[4]*a[2])/d,
        -a[3]/d,a[0]/d,(a[3]*a[2]-a[0]*a[5])/d)
}
private fun column(p: FloatArray, n: Int) = floatArrayOf(p[n],p[3+n],p[6+n])
private fun dot(a: FloatArray, b: FloatArray) = a.indices.sumOf { (a[it]*b[it]).toDouble() }.toFloat()
private fun cross(a: FloatArray, b: FloatArray) = floatArrayOf(
    a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
private fun project(p: FloatArray, point: FloatArray): FloatArray {
    val d = FloatArray(3) { point[it]-p[9+it] }
    val z = dot(d,column(p,2)); check(z > 0)
    return floatArrayOf(dot(d,column(p,0))/z, dot(d,column(p,1))/z)
}
private fun transform(m: FloatArray, p: FloatArray) = FloatArray(3) { r ->
    m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r]
}

fun main() {
    // Independent AOSP GLConsumer fixtures: M = flipY * cropGL * producerTransform.
    // A display-space clockwise rotation has these inverse top-left image mappings.
    val expected = arrayOf(
        floatArrayOf(1f,0f,0f,0f,1f,0f),
        floatArrayOf(0f,1f,0f,-1f,0f,1f),
        floatArrayOf(-1f,0f,1f,0f,-1f,1f),
        floatArrayOf(0f,-1f,1f,1f,0f,0f)
    )
    val stFixtures = arrayOf(
        floatArrayOf(1f,0f,0f,0f,-1f,1f),
        floatArrayOf(0f,-1f,1f,-1f,0f,1f),
        floatArrayOf(-1f,0f,1f,0f,1f,0f),
        floatArrayOf(0f,1f,0f,1f,0f,0f)
    )
    var mappingChecks = 0
    for (rotation in 0..3) for (mirror in listOf(false,true)) for (crop in listOf(false,true)) {
        val st = stFixtures[rotation].copyOf()
        if (mirror) { st[0] = -st[0]; st[1] = -st[1]; st[2] = 1f-st[2] }
        val sx = if(crop) .74f else 1f; val sy = if(crop) .62f else 1f
        val tx = if(crop) .11f else 0f; val ty = if(crop) .23f else 0f
        for (i in 0..2) st[i] *= sx
        for (i in 3..5) st[i] *= sy
        st[2] += tx; st[5] += ty
        val m = FloatArray(16)
        m[0]=st[0];m[4]=st[1];m[12]=st[2];m[1]=st[3];m[5]=st[4];m[13]=st[5]
        m[10]=1f;m[15]=1f
        val actual = ScanCoordinates.surfaceTextureToCamera(m)
        val reverse = inverse(actual)
        for (u in listOf(.1f,.5f,.9f)) for(v in listOf(.15f,.5f,.85f)) {
            val want=map(expected[rotation],u,v)
            if(mirror) want[0]=1f-want[0]
            want[0]=tx+sx*want[0];want[1]=ty+sy*want[1]
            val got=map(actual,u,v)
            close(got[0],want[0]);close(got[1],want[1])
            val back=map(reverse,got[0],got[1]);close(back[0],u);close(back[1],v)
            mappingChecks++
        }
    }
    // A physical top marker must remain above the center through the complete orbit.
    val center=floatArrayOf(.3f,-.8f,1.2f)
    var poses=0
    for (degrees in 0 until 360 step 10) for (pitch in listOf(-1.4f,-.35f,0f,.35f,1.4f)) {
        val yaw=(degrees*PI/180).toFloat()
        val p=FloatArray(12)
        ScanCoordinates.viewerPose(yaw,pitch,2f,center[0],center[1],center[2],p)
        for(i in 0..2) for(j in 0..2) close(dot(column(p,i),column(p,j)),if(i==j) 1f else 0f)
        close(dot(cross(column(p,0),column(p,1)),column(p,2)),1f)
        val mid=project(p,center);close(mid[0],0f);close(mid[1],0f)
        val top=project(p,floatArrayOf(center[0],center[1],center[2]+.1f))
        check(top[1]<mid[1]) { "physical top is upside down" }
        val right=FloatArray(3) { center[it]+.1f*column(p,0)[it] }
        check(project(p,right)[0]>mid[0]) { "horizontal mirror" }
        val anchor=floatArrayOf(-.4f,1.2f,2.1f)
        val model=ScanCoordinates.placementMatrix(yaw,1.5f,center,anchor)
        val moved=transform(model,center)
        for(i in 0..2) close(moved[i],anchor[i])
        val movedTop=transform(model,floatArrayOf(center[0],center[1],center[2]+.1f))
        close(movedTop[0],anchor[0]);close(movedTop[1],anchor[1]);close(movedTop[2],anchor[2]+.15f)
        val placed=FloatArray(3);ScanCoordinates.placementAnchor(p,1.2f,.2f,placed)
        for(i in 0..2) close(placed[i],p[9+i]+p[2+i*3]*1.2f-if(i==2) .2f else 0f)
        poses++
    }
    println("PASS $mappingChecks preview/touch mappings; $poses upright right-handed viewer/placement poses")
}
