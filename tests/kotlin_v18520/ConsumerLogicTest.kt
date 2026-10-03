import com.mobilescan3d.scan.consumer.ScanQualityGate
import com.mobilescan3d.scan.consumer.ScanMilestoneTracker
import com.mobilescan3d.scan.camera.CameraPreviewRecoveryPolicy
import com.mobilescan3d.scan.guidance.GuidanceStage

fun main() {
    check(ScanQualityGate.evaluate(45, 38, 42, 50, 55).shouldWarnBeforeGenerate)
    check(!ScanQualityGate.evaluate(88, 82, 86, 84, 78).shouldWarnBeforeGenerate)

    val milestones = ScanMilestoneTracker()
    val first = milestones.update(true, GuidanceStage.ORBIT_HORIZONTAL, 0.72f, 26)
    check(ScanMilestoneTracker.Event.TARGET_LOCKED in first)
    check(ScanMilestoneTracker.Event.READY_TO_ORBIT in first)
    check(ScanMilestoneTracker.Event.COVERAGE_25 in first)

    val policy = CameraPreviewRecoveryPolicy()
    check(policy.onPreviewTimeout(false) == CameraPreviewRecoveryPolicy.Action.RETRY_CURRENT)
    check(policy.onPreviewTimeout(false) == CameraPreviewRecoveryPolicy.Action.ENABLE_COMPAT_AND_RETRY)
    check(policy.onPreviewTimeout(true) == CameraPreviewRecoveryPolicy.Action.ASK_USER)
    println("vc18520 consumer logic PASS")
}
