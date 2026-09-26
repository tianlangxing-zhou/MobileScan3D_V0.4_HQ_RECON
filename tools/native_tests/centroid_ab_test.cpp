// MobileScan3D — target_mask_engine.cpp 质心读取 A/B 对照测试
//
// 背景：TargetMaskEngine::build() 里用
//     centroids_.at<cv::Vec2d>(chosen)
// 读取 connectedComponentsWithStats 的 centroids 输出。
// 但该输出是 CV_64F、nlabels x 2 的**两列单通道**矩阵，
// 每个元素只是一个 double（elemSize()==8），而 sizeof(cv::Vec2d)==16。
//
// Mat::at<T>() 是内联在头文件里的模板，会带着**调用方自己的编译选项**
// 展开 —— Debug 构建没有 NDEBUG，CV_DbgAssert(elemSize() == sizeof(_Tp))
// 生效，于是抛 cv::Exception；该异常穿过 JNI 边界后没有 C++ catch，
// 直接 std::terminate 杀掉进程（Kotlin 侧 try/catch 完全兜不住）。
//
// 本程序在**同一个进程**里跑两种写法，直接对比。
// 编译时不定义 NDEBUG，与 app 的 assembleDebug 原生构建保持一致。

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>

static const char* kSep =
    "------------------------------------------------------------";

/** 对给定的 centroids 矩阵跑 A/B 两种读取方式。 */
static void ab(const char* name, const cv::Mat& centroids, int chosen,
               double expectX, double expectY) {
    printf("[%s]\n", name);
    printf("  centroids: rows=%d cols=%d type=%d (CV_64F=%d) elemSize=%zu"
           " | sizeof(cv::Vec2d)=%zu | chosen=%d\n",
           centroids.rows, centroids.cols, centroids.type(), CV_64F,
           centroids.elemSize(), sizeof(cv::Vec2d), chosen);
    printf("  expected   : (%g, %g)\n", expectX, expectY);

    // ---- 旧写法 ----
    printf("  OLD at<cv::Vec2d>(chosen) : ");
    fflush(stdout);
    try {
        const cv::Vec2d c = centroids.at<cv::Vec2d>(chosen);
        printf("returned (%g, %g)", c[0], c[1]);
        if (c[0] == expectX && c[1] == expectY) {
            printf("  -> values OK\n");
        } else {
            printf("  -> VALUES WRONG\n");
        }
    } catch (const cv::Exception& ex) {
        printf("THREW cv::Exception\n    msg=%s\n", ex.what());
    } catch (const std::exception& ex) {
        printf("THREW std::exception: %s\n", ex.what());
    } catch (...) {
        printf("THREW unknown\n");
    }

    // ---- 新写法 ----
    printf("  NEW at<double>(chosen,0/1): ");
    fflush(stdout);
    try {
        double x = 0.0, y = 0.0;
        if (centroids.type() == CV_64F && centroids.cols == 2 &&
            chosen >= 0 && chosen < centroids.rows) {
            x = centroids.at<double>(chosen, 0);
            y = centroids.at<double>(chosen, 1);
            printf("returned (%g, %g)", x, y);
            if (x == expectX && y == expectY) {
                printf("  -> values OK\n");
            } else {
                printf("  -> VALUES WRONG\n");
            }
        } else {
            printf("guard rejected (degenerate fallback)\n");
        }
    } catch (const cv::Exception& ex) {
        printf("THREW cv::Exception: %s\n", ex.what());
    } catch (...) {
        printf("THREW unknown\n");
    }
    puts(kSep);
}

int main() {
    printf("OpenCV version: %s\n", CV_VERSION);
#ifdef NDEBUG
    printf("NDEBUG        : defined  -> CV_DbgAssert DISABLED (Release)\n");
#else
    printf("NDEBUG        : NOT defined -> CV_DbgAssert ACTIVE (Debug)\n");
#endif
    puts(kSep);

    // ---- 场景 1：真实 connectedComponentsWithStats 输出，两个连通域 ----
    {
        cv::Mat img = cv::Mat::zeros(40, 40, CV_8U);
        img(cv::Rect(3, 3, 10, 8)) = 255;
        img(cv::Rect(25, 20, 7, 9)) = 255;
        cv::Mat labels, stats, cent;
        const int n = cv::connectedComponentsWithStats(img, labels, stats, cent,
                                                       8, CV_32S);
        printf("scenario1: connectedComponentsWithStats -> nlabels=%d\n", n);
        ab("1. real cc output, chosen=1", cent, 1,
           cent.at<double>(1, 0), cent.at<double>(1, 1));
    }

    // ---- 场景 2：只有背景 + 一个连通域（chosen=1 是最后一行）----
    // 这一档最关键：Release 下旧写法读 16 字节，会从最后一行越界 8 字节。
    {
        cv::Mat img = cv::Mat::zeros(32, 32, CV_8U);
        img(cv::Rect(10, 10, 6, 6)) = 255;
        cv::Mat labels, stats, cent;
        const int n = cv::connectedComponentsWithStats(img, labels, stats, cent,
                                                       8, CV_32S);
        printf("scenario2: nlabels=%d (chosen is the LAST row)\n", n);
        ab("2. single component, chosen=last row", cent, n - 1,
           cent.at<double>(n - 1, 0), cent.at<double>(n - 1, 1));
    }

    // ---- 场景 3：kToleranceSteps 循环常见的多档结果（5 个连通域）----
    {
        cv::Mat cent(5, 2, CV_64F);
        for (int i = 0; i < 5; ++i) {
            cent.at<double>(i, 0) = 12.5 + i * 3.0;
            cent.at<double>(i, 1) = 40.25 + i * 7.0;
        }
        ab("3. 5 components, chosen=3", cent, 3, 21.5, 61.25);
    }

    // ---- 场景 4：chosen=0（退化路径，seed 落空时取最大连通域）----
    {
        cv::Mat cent(3, 2, CV_64F);
        cent.at<double>(0, 0) = 0.0;  cent.at<double>(0, 1) = 0.0;
        cent.at<double>(1, 0) = 100.5; cent.at<double>(1, 1) = 200.75;
        cent.at<double>(2, 0) = 300.0; cent.at<double>(2, 1) = 400.0;
        ab("4. chosen=0 (background label)", cent, 0, 0.0, 0.0);
    }

    // ---- 场景 5：防御分支 —— centroids 不是预期形状时不能抛异常 ----
    {
        cv::Mat bad = cv::Mat::zeros(2, 2, CV_32F);  // 类型不对
        ab("5. degenerate centroids (CV_32F) must not throw", bad, 1, 0.0, 0.0);
    }

    printf("DONE\n");
    return 0;
}
