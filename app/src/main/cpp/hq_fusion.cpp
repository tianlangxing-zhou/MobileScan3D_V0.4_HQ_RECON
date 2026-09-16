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

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mobilescan3d_NativeBridge_nativeFuseBurst(
    JNIEnv* env,
    jobject,
    jobjectArray inputPaths,
    jstring outputPath,
    jfloat exposureMs,
    jint iso,
    jfloatArray statsOut)
{
    // statsOut 共 20 槽，布局见下方 stats[..] 赋值处的注释
    if (!inputPaths || !outputPath || env->GetArrayLength(statsOut) < 20) {
        return JNI_FALSE;
    }

    const jsize nInputs = env->GetArrayLength(inputPaths);
    if (nInputs <= 0) {
        return JNI_FALSE;
    }

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

    // stats 槽位约定（与 HqCaptureController.applyFusionStats 一一对应）：
    //   0 sharpness(ref)      1 clippedPercent     2 underexposedPercent
    //   3 noiseSigma          4 confidenceMean     5 confidenceLowPercent
    //   6 inliersMean         7 rmseMean           8 eccMean
    //   9 nInputs            10 accepted          11 rejected
    //  12 rmseMax            13 sharpestIndex     14 sharpestScore
    //  15 fusedWritten       16 rejectedAlign     17 rejectedEcc
    //  18 rejectedDecode     19 alignedCount
    float stats[20] = {};
    stats[9] = static_cast<float>(nInputs);
    stats[13] = -1.f;

    cv::Mat reference = cv::imread(paths[0], cv::IMREAD_COLOR);
    if (reference.empty()) {
        env->SetFloatArrayRegion(statsOut, 0, 20, stats);
        return JNI_FALSE;
    }

    cv::Mat refGray;
    cv::cvtColor(reference, refGray, cv::COLOR_BGR2GRAY);

    // burst 内最清晰帧的追踪：对齐帧数不足时上层直接取这一帧交付
    int sharpestIndex = 0;
    double sharpestScore = sharpnessScore(refGray);

    cv::Mat lap;
    cv::Laplacian(refGray, lap, CV_32F);
    cv::Scalar lapMean, lapStd;
    cv::meanStdDev(lap, lapMean, lapStd);
    stats[0] = static_cast<float>(lapStd[0] * lapStd[0]);

    int clipped = 0;
    int underexposed = 0;
    const int totalPixels = reference.rows * reference.cols;
    for (int y = 0; y < reference.rows; ++y) {
        const cv::Vec3b* row = reference.ptr<cv::Vec3b>(y);
        for (int x = 0; x < reference.cols; ++x) {
            const cv::Vec3b& c = row[x];
            if (c[0] < 3 || c[1] < 3 || c[2] < 3 ||
                c[0] > 252 || c[1] > 252 || c[2] > 252) {
                clipped++;
            }
            const int luma = (c[0] * 299 + c[1] * 587 + c[2] * 114) / 1000;
            if (luma < 25) {
                underexposed++;
            }
        }
    }
    stats[1] = totalPixels > 0 ? static_cast<float>(100.0 * clipped / totalPixels) : 0.f;
    stats[2] = totalPixels > 0 ? static_cast<float>(100.0 * underexposed / totalPixels) : 0.f;

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
        cv::Mat cur = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (cur.empty()) {
            rejected++;
            rejectedDecode++;
            continue;
        }

        cv::Mat curGray;
        cv::cvtColor(cur, curGray, cv::COLOR_BGR2GRAY);

        // 每一帧都参与「最清晰帧」评选（同尺度比较），
        // 即使它因对齐失败被拒绝 —— 它仍可能是整组里最好看的一张。
        const double curScore = sharpnessScore(curGray);
        if (curScore > sharpestScore) {
            sharpestScore = curScore;
            sharpestIndex = static_cast<int>(i);
        }

        cv::Mat warped;
        bool good = false;
        // 1 = 对齐门（含异常 / 特征不足 / RANSAC 不达标），2 = ECC 门
        int rejectKind = 1;
        double frameInliers = 0.0;
        double frameRmse = 0.0;
        double frameEcc = 0.0;

        try {
            if (refPoints.size() >= 20) {
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

                if (gPrev.size() >= static_cast<size_t>(kMinAlignInliers)) {
                    cv::Mat inliers;
                    cv::Mat affine = cv::estimateAffinePartial2D(
                        gPrev, gNext, inliers, cv::RANSAC, 3.0);
                    if (!affine.empty()) {
                        int inlierCount = 0;
                        double sumSq = 0.0;
                        for (int k = 0; k < inliers.rows; ++k) {
                            if (inliers.at<uchar>(k)) {
                                inlierCount++;
                                const cv::Point2f& a = gPrev[k];
                                const cv::Point2f& b = gNext[k];
                                sumSq += (a.x - b.x) * (a.x - b.x) +
                                         (a.y - b.y) * (a.y - b.y);
                            }
                        }
                        frameInliers = inlierCount;
                        frameRmse = inlierCount > 0
                            ? std::sqrt(sumSq / inlierCount)
                            : 1e9;

                        // 第一道门：RANSAC 内点数量与残差同时达标才继续
                        if (inlierCount >= kMinAlignInliers &&
                            frameRmse <= kMaxAlignRmsePx) {
                            cv::warpAffine(
                                cur, warped, affine, reference.size(),
                                cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                            good = true;
                        }
                    }
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
                        cv::warpAffine(
                            warped, refined, ecc, reference.size(),
                            cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                        warped = refined;
                        frameEcc = eccScore;
                    }
                }
            }
        } catch (const cv::Exception&) {
            good = false;
        } catch (const std::exception&) {
            good = false;
        }

        if (!good || warped.empty() || warped.size() != reference.size()) {
            rejected++;
            continue;
        }

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

    if (accepted < 2) {
        // 没有任何帧通过严格对齐门限：如实上报统计，
        // 由上层给出具体拒绝原因（acceptedFrames 等），而不是笼统的 "fusion failed"。
        env->SetFloatArrayRegion(statsOut, 0, 20, stats);
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
            env->SetFloatArrayRegion(statsOut, 0, 20, stats);
            return JNI_FALSE;
        }
    } catch (const cv::Exception&) {
        env->SetFloatArrayRegion(statsOut, 0, 20, stats);
        return JNI_FALSE;
    }

    // 只有真正写出了融合图才置 1；上层据此判断「有没有东西可交付」
    stats[15] = 1.f;
    env->SetFloatArrayRegion(statsOut, 0, 20, stats);
    return JNI_TRUE;
}
