#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <opencv2/opencv.hpp>

namespace {

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
    if (!inputPaths || !outputPath || env->GetArrayLength(statsOut) < 12) {
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

    float stats[12] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    stats[9] = static_cast<float>(nInputs);

    cv::Mat reference = cv::imread(paths[0], cv::IMREAD_COLOR);
    if (reference.empty()) {
        env->SetFloatArrayRegion(statsOut, 0, 12, stats);
        return JNI_FALSE;
    }

    cv::Mat refGray;
    cv::cvtColor(reference, refGray, cv::COLOR_BGR2GRAY);

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

    int accepted = 1;
    int rejected = static_cast<int>(nInputs) - 1;
    int alignedCount = 0;
    double inlierSum = 0.0;
    double rmseSum = 0.0;
    bool haveNoise = false;

    for (jsize i = 1; i < nInputs; ++i) {
        cv::Mat cur = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (cur.empty()) {
            continue;
        }
        rejected--;

        cv::Mat curGray;
        cv::cvtColor(cur, curGray, cv::COLOR_BGR2GRAY);
        cv::Mat warped;
        bool good = false;
        double frameInliers = 0.0;
        double frameRmse = 0.0;

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

                if (gPrev.size() >= 8) {
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
                            : 0.0;

                        cv::warpAffine(
                            cur, warped, affine, reference.size(),
                            cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                        good = inlierCount >= 8;
                    }
                }
            }

            if (good) {
                cv::Mat refSmall = downscaleGray(refGray);
                cv::Mat warpedGray;
                cv::cvtColor(warped, warpedGray, cv::COLOR_BGR2GRAY);
                cv::Mat warpedSmall = downscaleGray(warpedGray);
                if (!refSmall.empty() && !warpedSmall.empty() &&
                    refSmall.size() == warpedSmall.size()) {
                    cv::Mat ecc = cv::Mat::eye(2, 3, CV_32F);
                    cv::TermCriteria criteria(
                        cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                        30, 1e-4);
                    const double eccScore = cv::findTransformECC(
                        refSmall, warpedSmall, ecc, cv::MOTION_EUCLIDEAN,
                        criteria);
                    if (std::isfinite(eccScore) && eccScore > -2.0) {
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
                        stats[8] = static_cast<float>(eccScore);
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
    }

    stats[10] = static_cast<float>(accepted);
    stats[11] = static_cast<float>(rejected);
    if (alignedCount > 0) {
        stats[6] = static_cast<float>(inlierSum / alignedCount);
        stats[7] = static_cast<float>(rmseSum / alignedCount);
    }
    if (noiseSigma < 0.5 || !std::isfinite(noiseSigma)) {
        noiseSigma = std::max(2.0, static_cast<double>(iso) / 20.0);
    }
    stats[3] = static_cast<float>(noiseSigma);

    if (accepted < 2) {
        env->SetFloatArrayRegion(statsOut, 0, 12, stats);
        return JNI_FALSE;
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
            env->SetFloatArrayRegion(statsOut, 0, 12, stats);
            return JNI_FALSE;
        }
    } catch (const cv::Exception&) {
        env->SetFloatArrayRegion(statsOut, 0, 12, stats);
        return JNI_FALSE;
    }

    env->SetFloatArrayRegion(statsOut, 0, 12, stats);
    return JNI_TRUE;
}
