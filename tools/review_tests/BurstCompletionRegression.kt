import com.mobilescan3d.BurstCompletion
fun main() {
    // Every ordering of five frames, and all success/failure combinations.
    fun permutations(xs: List<Int>): List<List<Int>> = if (xs.isEmpty()) listOf(emptyList()) else
        xs.flatMap { v -> permutations(xs - v).map { listOf(v) + it } }
    for (order in permutations((0..4).toList())) for (mask in 0..31) {
        val state = BurstCompletion(5)
        order.forEachIndexed { n, i ->
            check(state.record(i, mask and (1 shl i) != 0))
            check(!state.record(i, true)) // duplicates cannot release a burst early
            check(state.done == (n == 4))
        }
        check(state.completed == Integer.bitCount(mask))
        check(state.completed + state.failed == 5)
        check(!state.record(-1, true) && !state.record(5, false))
    }
    println("PASS 3840 mixed callback orders, duplicate callbacks, invalid frame indices")
}
