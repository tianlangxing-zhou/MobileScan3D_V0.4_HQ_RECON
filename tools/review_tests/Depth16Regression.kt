import com.mobilescan3d.depth.Depth16Decoder
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun main() {
    val bytes = ByteBuffer.allocate(44).order(ByteOrder.nativeOrder())
    bytes.position(4)
    // Two rows of three pixels: padded rows, pixel stride four, nonzero base position.
    val samples = intArrayOf(1000, (1 shl 13) or 1000, (2 shl 13) or 2000,
                            (7 shl 13) or 8191, 0, (3 shl 13) or 1500)
    for (y in 0..1) for (x in 0..2) bytes.putShort(4+y*16+x*4, samples[y*3+x].toShort())
    bytes.limit(30) // Last row need not include trailing padding.
    val result = checkNotNull(Depth16Decoder.decode(bytes,3,2,16,4))
    check(bytes.position()==4)
    check(result.depth[0]==1f && result.confidence[0]==1f)
    check(result.depth[1]==0f && result.confidence[1]==0f)
    check(result.depth[2]==2f && kotlin.math.abs(result.confidence[2]-1f/7)<1e-6)
    check(kotlin.math.abs(result.depth[3]-8.191f)<1e-6)
    check(result.depth[4]==0f && result.confidence[4]==0f)
    check(result.depth[5]==1.5f)
    val again=checkNotNull(Depth16Decoder.decode(bytes,3,2,16,4))
    again.depth[0]=9f
    check(result.depth[0]==1f) // Published frames never alias reusable buffers.
    bytes.limit(29)
    check(Depth16Decoder.decode(bytes,3,2,16,4)==null)
    check(Depth16Decoder.decode(bytes,3,2,1,4)==null)
    check(Depth16Decoder.decode(bytes,Int.MAX_VALUE,2,16,4)==null)
    println("PASS DEPTH16 bit fields, confidence, padded stride, buffer offset/bounds, immutable frames")
}
