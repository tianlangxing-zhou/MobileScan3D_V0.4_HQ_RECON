import com.mobilescan3d.AutoLightPolicy
fun main() {
    val light = AutoLightPolicy()
    check(!light.update(0, 30f, 90f, 100, 10_000_000))
    check(!light.update(500, 30f, 90f, 100, 10_000_000))
    check(light.update(900, 30f, 90f, 100, 10_000_000))
    check(light.update(1500, 130f, 230f, 100, 10_000_000)) // no self-induced flicker
    light.reset()
    check(!light.enabled)
    check(!light.update(2000, 30f, 90f, 100, 10_000_000))
    check(!light.update(2400, 120f, 230f, 100, 10_000_000))
    check(!light.update(2800, 30f, 90f, 100, 10_000_000)) // bright frame resets evidence
    check(!light.update(3200, 30f, 90f, 100, 10_000_000))
    check(!light.update(9000, 30f, 90f, 100, 10_000_000)) // dropped frames are not continuous evidence
    check(light.update(9900, 30f, 90f, 100, 10_000_000))
    light.reset()
    check(!light.update(0, 120f, 200f, 1600, 30_000_000))
    check(light.update(1000, 120f, 200f, 1600, 30_000_000)) // AE disguises dark scene
    light.reset()
    check(!light.update(0, Float.NaN, 100f, 100, 10_000_000))
    check(!light.update(1000, 120f, 230f, 100, 10_000_000))
    println("PASS auto-light delay, latch, reset, bright interruption, frame gaps, AE high gain, invalid input")
}
