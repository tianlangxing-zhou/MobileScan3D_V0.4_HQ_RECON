#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <opencv2/opencv.hpp>

namespace {

// 严格对齐门限：不满足就丢弃该帧，不再为了凑满 5 张强行融合。
// 18px 级的残余误差在 HQ texture 上会造成明显重影和糊边。
constexpr int kMinAlignInliers = 30;
constexpr double kMaxAlignRmsePx = 1.5;
constexpr double kMinAlignEcc = 0.95;

// 参考帧自己都解码不开时算一次 decode 失败
constexpr int kReasonOk = 0;
constexpr int kReasonDecodeFail = 1;
// 特征点太少（goodFeaturesToTrack 没找到，或 LK 前后向校验后活下来的不够）
constexpr int kReasonFeaturesTooFew = 2;
// RANSAC 内点不足
constexpr int kReasonInliers = 3;
// affine 重投影残差超门限
constexpr int kReasonRmse = 4;
// ECC 相关性不足 / 尺寸不匹配
constexpr int kReasonEcc = 5;
// OpenCV 或 std 异常
constexpr int kReasonException = 6;

/**
 * statsOut 的槽位布局（与 HqCaptureController.applyFusionStats 一一对应）
 *   0 sharpness(ref)      1 stillAnyChannelHighlightPercent   2 stillShadowPercent
 *   3 noiseSigma          4 confidenceMean                    5 confidenceLowPercent
 *   6 inliersMean         7 rmseMean                          8 eccMean
 *   9 nInputs            10 accepted                         11 rejected
 *  12 rmseMax            13 sharpestIndex                    14 sharpestScore
 *  15 fusedWritten       16 rejectedAlign                    17 rejectedEcc
 *  18 rejectedDecode     19 alignedCount
 *  20 stillHighlightPercent  21 stillLumaP05  22 stillLumaP50  23 stillLumaP95
 *  24 minAcceptedFrames（回显，方便在报告里核对两侧门限一致）
 */
constexpr int kStatsSlots = 25;

/**
 * frameStatsOut 每帧 8 个浮点，用于回答「那 3 张到底卡在哪一道门」：
 *   0 frameIndex  1 isReference  2 fbGood(LK 存活数)  3 ransacInliers
 *   4 rmse        5 ecc          6 rejectReason      7 sharpness
 * 未尝试的项写 -1（例如参考帧没有 LK/RANSAC/ECC 过程）。
 */
constexpr int kFrameStride = 8;

double medianAbsDeviation(const cv::Mat& a, const cv::Mat& b)
{
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }
    std::vector<double> values;
    values.reserve(static_cast<size_t>(a.rows * a.cols / 4));
    for (int y = 0; y < a.rows; y += 4) {
        for (int x = 0; x < a.cols; x += 4) {
            values.push_back(std::abs(
                static_cast<double>(a.at<uint8_t>(y, x)) -
                static_cast<double>(b.at<uint8_t>(y, x))));
        }
    }
    if (values.empty()) {
        return 0.0;
    }
    const size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    const double med = values[n / 2];
    std::vector<double> dev;
    dev.reserve(n);
    for (double v : values) {
        dev.push_back(std::abs(v - med));
    }
    std::nth_element(dev.begin(), dev.begin() + n / 2, dev.end());
    return dev[n / 2] * 1.4826;
}

cv::Mat downscaleGray(const cv::Mat& gray, int maxDim = 640)
{
    if (gray.empty()) return cv::Mat();
    const double scale = std::min(
        1.0,
        static_cast<double>(maxDim) / std::max(gray.cols, gray.rows));
    if (scale >= 1.0) return gray;
    cv::Mat small;
    cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    return small;
}

/**
 * 统一分辨率下的清晰度评分（Laplacian 方差）。
 *
 * 必须在同一尺度上比较，否则 4096 宽的参考帧和 1080 宽的候选帧算出来的方差
 * 根本不可比，所以先降到 maxDim=640 再算。
 * 上层用它从 burst 里挑「最清晰的一帧」做降级交付。
 */
double sharpnessScore(const cv::Mat& gray)
{
    if (gray.empty()) {
        return 0.0;
    }
    const cv::Mat small = downscaleGray(gray);
    if (small.empty()) {
        return 0.0;
    }
    cv::Mat lap;
    cv::Laplacian(small, lap, CV_32F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

/** 从 256 桶直方图取分位，q ∈ [0,1] */
int percentileFromHist(const int* hist, int total, double q)
{
    if (total <= 0) return 0;
    int target = static_cast<int>(q * total);
    if (target < 1) target = 1;
    int acc = 0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc >= target) return v;
    }
    return 255;
}

/**
 * 单张 still JPEG 的曝光统计。
 *
 * 旧实现有两个明确错误：
 *  (a) 把「任一通道接近黑或白」都算成 clipped —— 一个像素只要某个通道接近黑色
 *      就被记成过曝，于是报告里的 71.85% clipped 根本不能解释成 71.85% 高光过曝。
 *  (b) luma 权重按 RGB 顺序取值。cv::imread() 默认返回 **BGR**，所以 c[0] 是 B、
 *      c[2] 才是 R，权重必须写成 114*B + 587*G + 299*R。
 *
 * 现在拆成三个互不混淆的量 + 三个亮度分位，与 preview 侧同一套语义。
 */
void computeStillExposure(
    const cv::Mat& bgr,
    int& shadowPixels,
    int& highlightPixels,
    int& anyChannelHighlight,
    int& totalPixels,
    int& outP05,
    int& outP50,
    int& outP95)
{
    shadowPixels = 0;
    highlightPixels = 0;
    anyChannelHighlight = 0;
    totalPixels = 0;
    outP05 = outP50 = outP95 = 0;
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        return;
    }

    int hist[256] = {};
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b& c = row[x];
            const int b = c[0];
            const int g = c[1];
            const int r = c[2];
            // 真正的亮度（BT.601，注意通道顺序是 BGR）
            const int luma = (114 * b + 587 * g + 299 * r) / 1000;
            if (luma <= 20) {
                shadowPixels++;
            }
            if (luma >= 250) {
                highlightPixels++;
            }
            // 只有「某个通道已经打到上限」才是真正的通道级饱和
            if (b > 252 || g > 252 || r > 252) {
                anyChannelHighlight++;
            }
            hist[luma]++;
            totalPixels++;
        }
    }
    if (totalPixels <= 0) {
        return;
    }
    outP05 = percentileFromHist(hist, totalPixels, 0.05);
    outP50 = percentileFromHist(hist, totalPixels, 0.50);
    outP95 = percentileFromHist(hist, totalPixels, 0.95);
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeFuseBurst(
    JNIEnv* env,
    jobject,
    jobjectArray inputPaths,
    jstring outputPath,
    jfloat exposureMs,
    jint iso,
    jint minAcceptedFrames,
    jfloatArray statsOut,
    jfloatArray frameStatsOut)
{
    if (!inputPaths || !outputPath || !statsOut ||
        env->GetArrayLength(statsOut) < kStatsSlots) {
        return JNI_FALSE;
    }

    const jsize nInputs = env->GetArrayLength(inputPaths);
    if (nInputs <= 0) {
        return JNI_FALSE;
    }

    // 融合所需最少帧数由 Kotlin 传入，两侧不再各写一个魔数。
    // 旧实现 C++ 用 2、Kotlin 用 MIN_FUSION_FRAMES=3，于是 accepted=2 时
    // native 白做一次合成、写出 JPEG，Kotlin 再把它删掉。
    int minAccepted = static_cast<int>(minAcceptedFrames);
    if (minAccepted < 2) minAccepted = 2;

    std::vector<std::string> paths;
    paths.reserve(static_cast<size_t>(nInputs));
    for (jsize i = 0; i < nInputs; ++i) {
        jstring js = static_cast<jstring>(
            env->GetObjectArrayElement(inputPaths, i));
        if (!js) {
            paths.emplace_back();
            continue;
        }
        const char* p = env->GetStringUTFChars(js, nullptr);
        if (p) {
            paths.emplace_back(p);
            env->ReleaseStringUTFChars(js, p);
        } else {
            paths.emplace_back();
        }
        env->DeleteLocalRef(js);
    }

    const char* out = env->GetStringUTFChars(outputPath, nullptr);
    if (!out) {
        return JNI_FALSE;
    }
    const std::string outPath(out);
    env->ReleaseStringUTFChars(outputPath, out);

    float stats[kStatsSlots] = {};
    stats[9] = static_cast<float>(nInputs);
    stats[13] = -1.f;
    stats[24] = static_cast<float>(minAccepted);

    // 逐帧诊断缓冲：每帧 kFrameStride 个浮点
    std::vector<float> frameStats;
    const bool haveFrameStats =
        frameStatsOut != nullptr &&
        env->GetArrayLength(frameStatsOut) >= static_cast<jsize>(nInputs) * kFrameStride;
    if (haveFrameStats) {
        frameStats.assign(static_cast<size_t>(nInputs) * kFrameStride, -1.f);
        for (jsize i = 0; i < nInputs; ++i) {
            frameStats[static_cast<size_t>(i) * kFrameStride] = static_cast<float>(i);
            frameStats[static_cast<size_t>(i) * kFrameStride + 6] =
                static_cast<float>(kReasonDecodeFail); // 默认：没走到最后就算失败
        }
    }
    const auto pushFrameStats = [&]() {
        if (haveFrameStats) {
            env->SetFloatArrayRegion(
                frameStatsOut, 0, static_cast<jsize>(frameStats.size()),
                frameStats.data());
        }
    };

    cv::Mat reference = cv::imread(paths[0], cv::IMREAD_COLOR);
    if (reference.empty()) {
        // 参考帧自己解不开：这是一次 decode 失败，必须计入 rejectedDecode，
        // 否则报告里会同时出现「reference frame decode failed」和 rejectedDecode=0
        // 这种自相矛盾的组合（旧版本就是这样）。
        stats[18] = 1.f;
        if (haveFrameStats) {
            frameStats[1] = 1.f; // isReference
            frameStats[6] = static_cast<float>(kReasonDecodeFail);
        }
        env->SetFloatArrayRegion(statsOut, 0, kStatsSlots, stats);
        pushFrameStats();
        return JNI_FALSE;
    }

    cv::Mat refGray;
    cv::cvtColor(reference, refGray, cv::COLOR_BGR2GRAY);

    // burst 内最清晰帧的追踪：对齐帧数不足时上层直接取这一帧交付
    int sharpestIndex = 0;
    double sharpestScore = sharpnessScore(refGray);

    // 清晰度评分与 sharpestScore 用同一套尺度（都先降到 640 长边），
    // 否则 stats[0] 与 stats[14] 量纲不同，看报告的人会以为其中一个是错的。
    stats[0] = static_cast<float>(sharpestScore);

    if (haveFrameStats) {
        frameStats[1] = 1.f; // isReference
        frameStats[6] = static_cast<float>(kReasonOk);
        frameStats[7] = static_cast<float>(sharpestScore);
    }

    // ---- 参考帧曝光统计（BGR -> luma，拆 shadow / highlight / anyChannelHighlight）----
    {
        int shadowPixels = 0, highlightPixels = 0, anyChannelHighlight = 0, totalPixels = 0;
        int p05 = 0, p50 = 0, p95 = 0;
        computeStillExposure(
            reference, shadowPixels, highlightPixels, anyChannelHighlight,
            totalPixels, p05, p50, p95);
        if (totalPixels > 0) {
            stats[1] = static_cast<float>(100.0 * anyChannelHighlight / totalPixels);
            stats[2] = static_cast<float>(100.0 * shadowPixels / totalPixels);
            stats[20] = static_cast<float>(100.0 * highlightPixels / totalPixels);
        }
        stats[21] = static_cast<float>(p05);
        stats[22] = static_cast<float>(p50);
        stats[23] = static_cast<float>(p95);
    }

    std::vector<cv::Point2f> refPoints;
    cv::goodFeaturesToTrack(refGray, refPoints, 300, 0.01, 5.0);

    double noiseSigma = std::max(2.0, static_cast<double>(iso) / 20.0);
    const double noise2 = (noiseSigma / 255.0) * (noiseSigma / 255.0) + 1e-6;

    cv::Mat sumB = cv::Mat::zeros(reference.size(), CV_32F);
    cv::Mat sumG = cv::Mat::zeros(reference.size(), CV_32F);
    cv::Mat sumR = cv::Mat::zeros(reference.size(), CV_32F);
    cv::Mat sumWB = cv::Mat::zeros(reference.size(), CV_32F);
    cv::Mat sumWG = cv::Mat::zeros(reference.size(), CV_32F);
    cv::Mat sumWR = cv::Mat::zeros(reference.size(), CV_32F);

    auto accumulateFrame = [&](const cv::Mat& f) {
        cv::Mat ff;
        f.convertTo(ff, CV_32F, 1.0 / 255.0);
        for (int y = 0; y < reference.rows; ++y) {
            const cv::Vec3b* refRow = reference.ptr<cv::Vec3b>(y);
            const float* pRow = ff.ptr<float>(y);
            float* bRow = sumB.ptr<float>(y);
            float* gRow = sumG.ptr<float>(y);
            float* rRow = sumR.ptr<float>(y);
            float* wbRow = sumWB.ptr<float>(y);
            float* wgRow = sumWG.ptr<float>(y);
            float* wrRow = sumWR.ptr<float>(y);
            for (int x = 0; x < reference.cols; ++x) {
                const int idx = x * 3;
                const double rb = refRow[x][0] / 255.0;
                const double rg = refRow[x][1] / 255.0;
                const double rr = refRow[x][2] / 255.0;
                const double db = pRow[idx] - rb;
                const double dg = pRow[idx + 1] - rg;
                const double dr = pRow[idx + 2] - rr;
                const double wb = 1.0 / (noise2 + db * db);
                const double wg = 1.0 / (noise2 + dg * dg);
                const double wr = 1.0 / (noise2 + dr * dr);
                bRow[x] += static_cast<float>(wb * pRow[idx]);
                gRow[x] += static_cast<float>(wg * pRow[idx + 1]);
                rRow[x] += static_cast<float>(wr * pRow[idx + 2]);
                wbRow[x] += static_cast<float>(wb);
                wgRow[x] += static_cast<float>(wg);
                wrRow[x] += static_cast<float>(wr);
            }
        }
    };

    accumulateFrame(reference);

    int accepted = 1; // 参考帧自身
    int rejected = 0;
    int alignedCount = 0;
    double inlierSum = 0.0;
    double rmseSum = 0.0;
    double rmseMax = 0.0;
    double eccSum = 0.0;
    int eccCount = 0;
    // 被哪一道门挡掉的：分开展示才能判断是「点太少」还是「配准不够」
    int rejectedAlign = 0;
    int rejectedEcc = 0;
    int rejectedDecode = 0;
    bool haveNoise = false;

    for (jsize i = 1; i < nInputs; ++i) {
        float* fs = haveFrameStats
            ? frameStats.data() + static_cast<size_t>(i) * kFrameStride
            : nullptr;
        const auto markReason = [fs](int reason) {
            if (fs) fs[6] = static_cast<float>(reason);
        };

        cv::Mat cur = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (cur.empty()) {
            rejected++;
            rejectedDecode++;
            markReason(kReasonDecodeFail);
            continue;
        }

        cv::Mat curGray;
        cv::cvtColor(cur, curGray, cv::COLOR_BGR2GRAY);

        // 每一帧都参与「最清晰帧」评选（同尺度比较），
        // 即使它因对齐失败被拒绝 —— 它仍可能是整组里最好看的一张。
        const double curScore = sharpnessScore(curGray);
        if (fs) fs[7] = static_cast<float>(curScore);
        if (curScore > sharpestScore) {
            sharpestScore = curScore;
            sharpestIndex = static_cast<int>(i);
        }

        cv::Mat warped;
        bool good = false;
        // 1 = 对齐门（含异常 / 特征不足 / RANSAC 不达标），2 = ECC 门
        int rejectKind = 1;
        int reason = kReasonFeaturesTooFew;
        double frameInliers = 0.0;
        double frameRmse = 0.0;
        double frameEcc = 0.0;
        double frameFbGood = 0.0;

        try {
            if (refPoints.size() < 20) {
                // 参考帧本身特征就太少，所有候选帧都会走到这里
                reason = kReasonFeaturesTooFew;
            } else {
                std::vector<cv::Point2f> next;
                std::vector<cv::Point2f> back;
                std::vector<uint8_t> statusF;
                std::vector<uint8_t> statusB;
                std::vector<float> errF;
                std::vector<float> errB;
                cv::calcOpticalFlowPyrLK(
                    refGray, curGray, refPoints, next, statusF, errF,
                    cv::Size(21, 21), 3);
                cv::calcOpticalFlowPyrLK(
                    curGray, refGray, next, back, statusB, errB,
                    cv::Size(21, 21), 3);

                std::vector<cv::Point2f> gPrev, gNext;
                for (size_t k = 0; k < statusF.size() &&
                    k < statusB.size() && k < next.size(); ++k) {
                    if (!statusF[k] || !statusB[k]) continue;
                    const double fb = cv::norm(refPoints[k] - back[k]);
                    if (fb > 1.5 || errF[k] > 20.0) continue;
                    const cv::Point2f& p = next[k];
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
                        p.x < 0 || p.y < 0 ||
                        p.x >= curGray.cols || p.y >= curGray.rows) {
                        continue;
                    }
                    gPrev.push_back(refPoints[k]);
                    gNext.push_back(p);
                }
                frameFbGood = static_cast<double>(gPrev.size());

                if (gPrev.size() >= static_cast<size_t>(kMinAlignInliers)) {
                    cv::Mat inliers;
                    cv::Mat affine = cv::estimateAffinePartial2D(
                        gPrev, gNext, inliers, cv::RANSAC, 3.0);
                    if (!affine.empty()) {
                        // RMSE 必须是「affine 模型拟合之后还剩多少误差」，
                        // 而不是「特征从上一帧移动了多少像素」。
                        // 旧实现算的是后者：手机轻微平移 5px 时，即使模型完全正确，
                        // RMSE 也会是 5px，然后被 1.5px 的门限误杀。这是把好帧
                        // 成片拒掉的一个重要原因。
                        const double m00 = affine.at<double>(0, 0);
                        const double m01 = affine.at<double>(0, 1);
                        const double m02 = affine.at<double>(0, 2);
                        const double m10 = affine.at<double>(1, 0);
                        const double m11 = affine.at<double>(1, 1);
                        const double m12 = affine.at<double>(1, 2);
                        int inlierCount = 0;
                        double sumSq = 0.0;
                        for (int k = 0; k < inliers.rows; ++k) {
                            if (inliers.at<uchar>(k)) {
                                const cv::Point2f& src = gPrev[k];
                                const cv::Point2f& dst = gNext[k];
                                const double px = m00 * src.x + m01 * src.y + m02;
                                const double py = m10 * src.x + m11 * src.y + m12;
                                const double ex = px - dst.x;
                                const double ey = py - dst.y;
                                sumSq += ex * ex + ey * ey;
                                inlierCount++;
                            }
                        }
                        frameInliers = inlierCount;
                        frameRmse = inlierCount > 0
                            ? std::sqrt(sumSq / inlierCount)
                            : 1e9;
                        // 旧 bug：rmseMax 声明后从未更新过，stats[12] 永远是 0，
                        // 报告里 alignmentRmseMax 因此毫无意义。
                        rmseMax = std::max(rmseMax, frameRmse);

                        // 第一道门：RANSAC 内点数量与残差同时达标才继续
                        if (inlierCount < kMinAlignInliers) {
                            reason = kReasonInliers;
                        } else if (frameRmse > kMaxAlignRmsePx) {
                            reason = kReasonRmse;
                        } else {
                            // affine 由 estimateAffinePartial2D(gPrev=ref, gNext=cur)
                            // 解出，方向是 reference -> current；而 warpAffine 的
                            // dst 是 reference、src 是 cur，方向恰好相反。
                            // 必须用 WARP_INVERSE_MAP 告诉 OpenCV「按 dst->src 使用这个矩阵」，
                            // 否则 OpenCV 会再求一次逆，把图像往反方向拉。
                            cv::warpAffine(
                                cur, warped, affine, reference.size(),
                                cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                                cv::BORDER_REPLICATE);
                            good = true;
                            reason = kReasonOk;
                        }
                    } else {
                        reason = kReasonFeaturesTooFew;
                    }
                } else {
                    reason = kReasonFeaturesTooFew;
                }
            }

            if (good) {
                // 过了对齐门，之后任何失败都归到 ECC 门
                rejectKind = 2;
                cv::Mat refSmall = downscaleGray(refGray);
                cv::Mat warpedGray;
                cv::cvtColor(warped, warpedGray, cv::COLOR_BGR2GRAY);
                cv::Mat warpedSmall = downscaleGray(warpedGray);
                if (refSmall.empty() || warpedSmall.empty() ||
                    refSmall.size() != warpedSmall.size()) {
                    good = false;
                    reason = kReasonEcc;
                } else {
                    cv::Mat ecc = cv::Mat::eye(2, 3, CV_32F);
                    cv::TermCriteria criteria(
                        cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                        30, 1e-4);
                    const double eccScore = cv::findTransformECC(
                        refSmall, warpedSmall, ecc, cv::MOTION_EUCLIDEAN,
                        criteria);
                    // 第二道门：ECC 相关性不足同样视为不可融合的帧
                    if (!std::isfinite(eccScore) || eccScore < kMinAlignEcc) {
                        good = false;
                        reason = kReasonEcc;
                    } else {
                        const double scaleX =
                            static_cast<double>(reference.cols) / refSmall.cols;
                        const double scaleY =
                            static_cast<double>(reference.rows) / refSmall.rows;
                        ecc.at<float>(0, 2) =
                            static_cast<float>(ecc.at<float>(0, 2) * scaleX);
                        ecc.at<float>(1, 2) =
                            static_cast<float>(ecc.at<float>(1, 2) * scaleY);
                        cv::Mat refined;
                        // findTransformECC 的矩阵同样是 template -> input 方向
                        // （template=refSmall, input=warpedSmall），把 warped 拉回
                        // reference 坐标系同样需要 WARP_INVERSE_MAP。
                        cv::warpAffine(
                            warped, refined, ecc, reference.size(),
                            cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                            cv::BORDER_REPLICATE);
                        warped = refined;
                        frameEcc = eccScore;
                        reason = kReasonOk;
                    }
                }
            }
        } catch (const cv::Exception&) {
            good = false;
            reason = kReasonException;
            rejectKind = 1;
        } catch (const std::exception&) {
            good = false;
            reason = kReasonException;
            rejectKind = 1;
        }

        if (fs) {
            fs[2] = static_cast<float>(frameFbGood);
            fs[3] = static_cast<float>(frameInliers);
            fs[4] = static_cast<float>(frameRmse);
            fs[5] = static_cast<float>(frameEcc);
        }

        if (!good || warped.empty() || warped.size() != reference.size()) {
            rejected++;
            // rejectedAlign / rejectedEcc 之前声明了却从来没累加过，
            // 所以报告里它们恒为 0 —— 并不代表没有失败，只是没统计。
            // 按失败发生在哪一道门归类，这两个数才有诊断价值。
            if (rejectKind == 2) {
                rejectedEcc++;
            } else {
                rejectedAlign++;
            }
            markReason(reason);
            continue;
        }

        markReason(kReasonOk);

        if (!haveNoise) {
            const double measured = medianAbsDeviation(refGray, curGray);
            if (measured > 0.5 && std::isfinite(measured)) {
                noiseSigma = measured;
                haveNoise = true;
            }
        }

        accumulateFrame(warped);
        accepted++;
        alignedCount++;
        inlierSum += frameInliers;
        rmseSum += frameRmse;
        eccSum += frameEcc;
        eccCount++;
    }

    stats[10] = static_cast<float>(accepted);
    stats[11] = static_cast<float>(rejected);
    stats[12] = static_cast<float>(rmseMax);
    stats[13] = static_cast<float>(sharpestIndex);
    stats[14] = static_cast<float>(sharpestScore);
    stats[16] = static_cast<float>(rejectedAlign);
    stats[17] = static_cast<float>(rejectedEcc);
    stats[18] = static_cast<float>(rejectedDecode);
    stats[19] = static_cast<float>(alignedCount);
    if (alignedCount > 0) {
        stats[6] = static_cast<float>(inlierSum / alignedCount);
        stats[7] = static_cast<float>(rmseSum / alignedCount);
    }
    if (eccCount > 0) {
        stats[8] = static_cast<float>(eccSum / eccCount);
    }
    if (noiseSigma < 0.5 || !std::isfinite(noiseSigma)) {
        noiseSigma = std::max(2.0, static_cast<double>(iso) / 20.0);
    }
    stats[3] = static_cast<float>(noiseSigma);

    if (accepted < minAccepted) {
        // 通过严格对齐门限的帧数不够：如实上报统计与逐帧诊断，
        // 由上层给出具体拒绝原因（acceptedFrames / rejectedAlign / rejectedEcc），
        // 而不是笼统的 "fusion failed"。
        // 也**不再**做无谓的合成 —— 旧实现这里用 accepted<2，与 Kotlin 的
        // MIN_FUSION_FRAMES=3 不一致，accepted=2 时会白算一遍再被上层删掉。
        env->SetFloatArrayRegion(statsOut, 0, kStatsSlots, stats);
        pushFrameStats();
        return JNI_TRUE;
    }

    const double inv3 = 1.0 / (3.0 * noiseSigma / 255.0 + 1e-6);
    double confSum = 0.0;
    int lowConf = 0;
    int pixelCount = 0;

    cv::Mat fused(reference.size(), CV_8UC3);
    for (int y = 0; y < reference.rows; ++y) {
        const float* b = sumB.ptr<float>(y);
        const float* g = sumG.ptr<float>(y);
        const float* r = sumR.ptr<float>(y);
        const float* wb = sumWB.ptr<float>(y);
        const float* wg = sumWG.ptr<float>(y);
        const float* wr = sumWR.ptr<float>(y);
        const cv::Vec3b* refRow = reference.ptr<cv::Vec3b>(y);
        cv::Vec3b* outRow = fused.ptr<cv::Vec3b>(y);
        for (int x = 0; x < reference.cols; ++x) {
            const double fb = b[x] / std::max(wb[x], 1e-6f);
            const double fg = g[x] / std::max(wg[x], 1e-6f);
            const double fr = r[x] / std::max(wr[x], 1e-6f);
            auto sat = [](double v) {
                return static_cast<uint8_t>(
                    std::max(0.0, std::min(1.0, v)) * 255.0 + 0.5);
            };
            outRow[x] = cv::Vec3b(sat(fb), sat(fg), sat(fr));

            const double resid = std::abs(refRow[x][0] / 255.0 - fb) +
                                 std::abs(refRow[x][1] / 255.0 - fg) +
                                 std::abs(refRow[x][2] / 255.0 - fr);
            const double conf = std::exp(-(resid * inv3));
            confSum += conf;
            if (conf < 0.5) lowConf++;
            pixelCount++;
        }
    }

    stats[4] = pixelCount > 0
        ? static_cast<float>(confSum / pixelCount)
        : 0.f;
    stats[5] = pixelCount > 0
        ? static_cast<float>(100.0 * lowConf / pixelCount)
        : 0.f;

    try {
        std::vector<int> params;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(95);
        if (!cv::imwrite(outPath, fused, params)) {
            env->SetFloatArrayRegion(statsOut, 0, kStatsSlots, stats);
            pushFrameStats();
            return JNI_FALSE;
        }
    } catch (const cv::Exception&) {
        env->SetFloatArrayRegion(statsOut, 0, kStatsSlots, stats);
        pushFrameStats();
        return JNI_FALSE;
    }

    // 只有真正写出了融合图才置 1；上层据此判断「有没有东西可交付」
    stats[15] = 1.f;
    env->SetFloatArrayRegion(statsOut, 0, kStatsSlots, stats);
    pushFrameStats();
    return JNI_TRUE;
}
