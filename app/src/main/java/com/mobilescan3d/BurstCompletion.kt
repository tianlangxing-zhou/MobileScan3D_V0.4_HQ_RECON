package com.mobilescan3d

/** Order-independent Camera2 terminal result accounting; duplicate callbacks are ignored. */
class BurstCompletion(private val expected: Int) {
    init { require(expected > 0) }
    private val seen = BooleanArray(expected)
    var completed = 0
        private set
    var failed = 0
        private set
    val done: Boolean get() = completed + failed == expected
    fun record(index: Int, success: Boolean): Boolean {
        if (index !in seen.indices || seen[index]) return false
        seen[index] = true
        if (success) completed++ else failed++
        return true
    }
}
